import gym
import numpy as np

# Observation space: Board + Player Health 
# Planes count:         2   +       2       = 4
# Board: (per player)
#   Units (hp)
# Health (per player)

ROWS = 4
COLS = 3
BOARD_SHAPE = (ROWS, COLS)
HAND_SIZE = 3
BASE_HEALTH = 6
OBS_CHANNELS = 2 + 2  # 2 units, 2 hp
OBS_SHAPE = (ROWS, COLS, OBS_CHANNELS)

# Action space:
# 1 Card play (1, 2 or 3) on any position on the board or pass turn (if no space avaliable)
# size(action_space) = card, pos = 3 * 3 * 4 + 1

class SlimboundEnv(gym.Env):
    metadata = { 'render_modes': ['ansi']}

    def __init__(self, seed = None, **kwargs) -> None:
        super().__init__()
        self._player_units = np.empty(BOARD_SHAPE, np.int8)
        self._enemy_units = np.empty(BOARD_SHAPE, np.int8)
        self._meta = np.empty((ROWS, COLS, 2), np.int8)
        self._observation_arr = np.empty(OBS_SHAPE, np.int8)
        self._player_health = self._enemy_health = BASE_HEALTH
        self._player = 0
        self._frontline = ROWS - 1
        self.observation_space = gym.spaces.Box(
            low=np.iinfo(np.int8).min,
            high=np.iinfo(np.int8).max,
            shape=OBS_SHAPE,
            dtype=np.int8
        )
        self.action_space = gym.spaces.Discrete(HAND_SIZE*ROWS*COLS +1)


    def reset(self, seed=None, options=None):
        self._player_units.fill(0)
        self._enemy_units.fill(0)
        self._player_health = self._enemy_health = BASE_HEALTH
        self._player = 0
        self._frontline = ROWS - 1
        return self._observation()


    def to_play(self):
        return self._player


    def _observation(self):
        self._meta[:, :, 0].fill(self._player_health)
        self._meta[:, :, 1].fill(self._enemy_health)
        return np.concatenate(
            (np.stack(
                (self._player_units,
                 self._enemy_units),
                axis=2),
            self._meta
            ),
            axis=-1,
            out=self._observation_arr
        )


    def _execute_action(self, action: int):
        if action != 0:
            card, row, col = decode_unit_action(action)
            self._play_unit_card(card, row, col)
        self._end_turn()
        self._start_enemy_turn()

    
    def _end_turn(self):
        self._player = int(not self._player)
        self._player_health, self._enemy_health = self._enemy_health, self._player_health
        self._player_units, self._enemy_units = self._enemy_units, self._player_units
        self._player_units = np.rot90(self._player_units, k=2)
        self._enemy_units = np.rot90(self._enemy_units, k=2)


    def _start_enemy_turn(self):
        before_advance = self._calculate_frontline()
        self._advance_units()
        after_advance = self._calculate_frontline()
        self._frontline = min(before_advance, after_advance)    # Frontline value decreases closer to enemy base


    def _advance_units(self):
        p_units = self._player_units
        e_units = self._enemy_units
        p_units[p_units!=0] += 1
        damage_to_enemy = p_units[0].sum()
        self._enemy_health -= damage_to_enemy
        if self._enemy_health < 0:
            self._enemy_health = 0
        p_units[0].fill(0)
        p_units = np.roll(p_units, -1, axis=0)
        result = p_units - e_units
        np.copyto(p_units, result)
        p_units[p_units < 0] = 0
        np.copyto(e_units, result)
        e_units *= -1
        e_units[e_units < 0] = 0
        self._player_units = p_units
        self._enemy_units = e_units


    def _play_unit_card(self, card, row, col):
        assert 0 <= row < ROWS and 0 <= col < COLS, 'Position outside of board'
        assert self._player_units[row, col] == 0, 'Position occupied by player unit'
        assert self._enemy_units[row, col] == 0, 'Position occupied by enemy unit'
        assert row >= min(self._frontline, self._calculate_frontline()), f'Position past frontline'
        unit = card + 1
        enemy = self._enemy_in_attack_range(row, col)
        if not enemy:
            if not self._player_units[row-1, col]:
                row -= 1
            self._player_units[row, col] = unit
        else:
            e_unit, e_row, e_col = enemy
            unit = unit - e_unit
            self._player_units[e_row, e_col] = max(unit, 0)
            self._enemy_units[e_row, e_col] = max(-unit, 0)


    def _calculate_frontline(self):
        if not self._player_units.any():
            return ROWS - 1
        return self._player_units.nonzero()[0][0] or 1  # Frontline excludes enemy base


    def _enemy_in_attack_range(self, row, col):
        e_units = self._enemy_units
        front = e_units[row-1, col]
        if front:
            return front, row-1, col
        inward_dir = 1 if col < COLS//2 else -1
        in_col = col + inward_dir
        inward = e_units[row, in_col]
        if inward:
            return inward, row, in_col
        out_col = col - inward_dir
        if 0 <= out_col < COLS:
            outward = e_units[row, out_col]
            if outward:
                return outward, row, out_col
        return None


    def step(self, action: int):
        self._execute_action(action)
        observation = self._observation()
        reward = self._reward()
        done = reward == 1
        return observation, reward, done, {}


    def _reward(self):
        if self._player_health <= 0:    # reward is calculated after player and enemy swap
            return 1
        return 0


    def render(self):
        return f'''\
Turn p{self._player}
Enemy Health: {self._enemy_health}
{np.array_str(
    np.where(
        self._player_units >= self._enemy_units,
        self._player_units,
        -self._enemy_units))}
Player Health: {self._player_health}
'''


    def legal_actions(self):
        END_TURN_ACTION = 0
        cards = range(3)
        avaliable_positions = self._avaliable_positions()
        actions = []
        if avaliable_positions:
            actions = [
                encode_unit_action(card, pos)
                for card in cards
                for pos in avaliable_positions
            ]
        actions.append(END_TURN_ACTION)
        return actions


    def _avaliable_positions(self):
        # Single action, frontline doesn't move during turn
        # frontline = min(self._frontline, self._calculate_frontline()) 
        frontline = self._frontline
        avaliable = np.logical_not(
                        np.logical_or(
                            self._player_units[frontline:, :], 
                            self._enemy_units[frontline:, :]
                        )
                    ).nonzero()
        if avaliable[0].size == 0:
            return ()
        return tuple(zip(avaliable[0] + frontline, avaliable[1]))


def encode_unit_action(card, pos) -> int:
    return np.ravel_multi_index((card, *pos), (HAND_SIZE, ROWS, COLS)) + 1

def decode_unit_action(action: int) -> tuple:
    return np.unravel_index(action-1, (HAND_SIZE, ROWS, COLS))

def action_to_str(action_number):
    if action_number == 0:
        action = 'Skipped turn'
    else:
        idx, row, col = decode_unit_action(action_number)
        action = f'Play card {idx+1} at [{row}, {col}]'
    return f'{action} - ({action_number})'
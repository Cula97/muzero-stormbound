import gym
import numpy as np
from stormbot.game.enums import CardType

from stormbot.game.player import Player, PlayerData
from stormbot.game.game import BoardPiece, Game
from stormbot.game.board import Board

GameBoard = Board[BoardPiece]

# Observations space: Board + Hand + Player Health + Player id + Player Mana
# Planes count:         2   +  8   +       2       +     1     +      1      = 14
# Board: (per player)
#   Units (hp)
# Hand (2 planes per card): 
#   hp
#   mana

# Health (per player)
# Player turn
# Mana (current player)

class StormEnv_v1(gym.Env):
    metadata = { 'render_modes': ['ansi']}
    _deck = None

    def __init__(self, seed = None, **kwargs) -> None:
        super().__init__()
        self._game = None
        self.observation_space = gym.spaces.Box(
            low=0,
            high=np.iinfo(np.intc).max,
            shape=(5, 3, 14),
            dtype=np.intc
        )
        self.action_space = gym.spaces.Discrete(5*3*4 +1)
        self._winner_rewarded = False
        self._game_ended = False
        self._game_drawn = False
        self._consecutive_turns_skipped = 0


    def reset(self, seed=None, options=None):
        # super().reset(seed=seed)
        self._game = Game()
        self._game.start(*self.players())
        self._winner_rewarded = False
        self._game_ended = False
        self._game_drawn = False
        self._consecutive_turns_skipped = 0
        return self._observation()

    @property
    def deck(self):
        from stormbot.game.game import CARDS
        if self._deck is None:
            self._deck = [CARDS[f'CN{i}'] for i in range(1, 5)]
        return self._deck

    @property
    def game(self):
        return self._game

    def players(self):
        return PlayerData('p1', self.deck.copy()), PlayerData('p2', self.deck.copy())

    def to_play(self):
        assert self._game
        return 0 if self._game.is_player1_turn() else 1

    def _observation(self):
        assert self._game
        current_player = self._game.current_player
        board = encode_board(self._game.board, current_player)
        PLAYER_2 = 1
        if self.to_play() == PLAYER_2:
            board = np.rot90(board, k=2)
        hand = encode_hand(current_player)
        
        meta = np.zeros(
            shape=(5, 3, 4),
            dtype=np.intc
        )
        meta[:, :, 0] = current_player.base_health
        meta[:, :, 1] = self._game.current_enemy.base_health
        meta[:, :, 2] = self.to_play()
        meta[:, :, 3] = current_player.mana
        return np.concatenate([board, hand, meta], axis=-1)


    def _execute_action(self, action: int):
        assert self._game
        if action == 0:
            self._game.step()
            self._consecutive_turns_skipped += 1
            if self._consecutive_turns_skipped == 4:
                self._game_drawn = True
                raise Exception('Game ended')
        else:
            self._consecutive_turns_skipped = 0
            idx, row, col = decode_unit_action(action)
            self._game.play_card(idx, (row, col))


    def step(self, action: int):
        current_player = self._game.current_player
        current_enemy = self._game.current_enemy
        try:
            self._execute_action(action)
        except Exception as ex:
            if str(ex) == 'Game ended':
                self._game_ended = True
            else:
                raise ex

        observation = self._observation()
        # reward = 0 if not self._game_ended else self._reward(current_player, current_enemy)
        reward = self._reward(current_player, current_enemy)
        self._winner_rewarded = self._game_ended and reward == 10000
        return observation, reward, self._game_ended and (self._winner_rewarded or self._game_drawn), {}


    def _reward(self, player, enemy):
        assert not (player.base_health <= 0 and enemy.base_health <= 0 and self._winner_rewarded)
        if enemy.base_health <= 0: 
            return +10000
        elif player.base_health <= 0: 
            return 0
        else:
            return max(0, self._player_score(player, enemy))# - self._player_score(enemy, player))


    def _player_score(self, player, enemy):
        units_score = sum(
            unit.strength * (5 - abs(unit.pos[0] - self._game.baseline(enemy)))
            for unit in self._game.board if unit.owner is player
        )
        enemy_damage_score = (enemy.data.base_health - enemy.base_health) * 6
        return units_score + enemy_damage_score


    def render(self):
        assert self._game
        return f'''\
        \nTurn {self._game.turn_count} ({self._game.current_player.id})
  Player 2 - H: {self._game.player2.base_health}, M: {self._game.player2.mana}
{self._game.board_to_str()}
  Player 1 - H: {self._game.player1.base_health}, M: {self._game.player1.mana}
'''


    def legal_actions(self):
        assert self._game
        if self._game_ended:
            assert not self._winner_rewarded
            return [0]

        from stormbot.game.ability_utils import positions_in_frontline
        end_turn = 0
        hand = self._game.current_player.hand
        mana = self._game.current_player.mana

        playable_cards = tuple(idx for idx, card in enumerate(hand) if card.mana <= mana)
        card_actions = []
        if playable_cards:
            empty_positions = (pos for pos in positions_in_frontline(self._game) 
                                if self._game.board[pos] is None)
            card_actions = [
                encode_unit_action(card_idx, pos) # TODO: Spells
                for card_idx in playable_cards
                for pos in empty_positions]

        return card_actions if card_actions else [end_turn]


def encode_board(board: GameBoard, current_player: Player):
    array = np.zeros((5, 3, 2), dtype=np.intc)
    for piece in board:
        (row, col) = piece.pos
        player_plane = 0 if piece.owner is current_player else 1
        array[row, col, player_plane] = piece.strength
    return array


def encode_hand(player: Player):
    array = np.zeros((5, 3, 8), dtype=np.intc)
    for card_idx, card in enumerate(player.hand):
        array[:, :, card_idx] = card.strength
        array[:, :, card_idx+4] = card.mana
    return array


def encode_id(id: str) -> int:
    faction_int = ord(id[0])
    encoded = 10
    while encoded <= faction_int:
        encoded *= 10
    encoded *= faction_int
    encoded += int(id[1:])
    return encoded

def encode_unit_action(card_idx, piece_pos) -> int:
    return np.ravel_multi_index((card_idx, *piece_pos), (4, 5, 3)) + 1

def decode_unit_action(action: int) -> tuple:
    return np.unravel_index(action - 1, (4, 5, 3))
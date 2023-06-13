import gym
import numpy as np
from stormbot.game.enums import CardType

from stormbot.game.player import Player, PlayerData
from stormbot.game.game import BoardPiece, Game
from stormbot.game.board import Board

GameBoard = Board[BoardPiece]

# Observations space: Board + Hand + Player Health + Player id + Player Mana
# Planes count:		    6 	+  1   +       2	   +     1 	   +      1 	 = 11
# Board: (per player)
# 	Units (hp)
# 	Structures (hp)
# 	Ids
# Hand (1 plane, 1 row for card): 
#	Id, Mana, Strength, Movement

# Health (per player)
# Player turn
# Mana (current player)

class StormEnv_v0(gym.Env):
	metadata = { 'render_modes': ['ansi']}
	_deck = None

	def __init__(self, seed = None, history_length = 5*4, **kwargs) -> None:
		super().__init__()
		self._game = None
		self._history = BoardHandHistory(history_length)
		self.observation_space = gym.spaces.Box(
			low=0,
			high=np.iinfo(np.intc).max,
			shape=(6, 4, history_length*7 +4),
			dtype=np.intc
		)
		self.action_space = gym.spaces.Discrete(6*4*4 +5)


	def reset(self, seed=None, options=None):
		# super().reset(seed=seed)
		self._game = Game()
		self._game.start(*self.players())
		self._history.reset()
		return self._observation()

	@property
	def deck(self):
		from stormbot.game.game import CARDS
		if self._deck is None:
			self._deck = [CARDS[id] for id in ('N3', 'N4', 'N13', 'N19',
											   'N71', 'N28', 'N30', 'N32',
											   'N72', 'N52', 'N54', 'I21')]
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
		# Board and hand
		self._history.push(self._game.board, player1=self.game.player1)
		is_player1_turn = self._game.is_player1_turn()
		history = self._history.view(p1_perspective=is_player1_turn)
		
		meta = np.zeros(
			shape=(6, 4, 4),
			dtype=np.intc
		)
		meta[:, :, 0] = current_player.base_health 				# player health
		meta[:, :, 1] = self._game.current_enemy.base_health 	# enemy health
		meta[:, :, 2] = int(is_player1_turn)					# playing as p1 or p2
		meta[:, :, 3] = current_player.mana 					# player mana
		return np.concatenate([history, meta], axis=-1)


	def _execute_action(self, action: int):
		assert self._game
		assert action in self.legal_actions()
		if action == 0:
			self._game.step()
		elif action <= 4:
			self._game.discard_card(action-1)
		else:
			idx, row, col = decode_unit_action(action)
			self._game.play_card(idx, (row, col))


	def step(self, action: int):
		done = False
		try:
			self._execute_action(action)
		except Exception as ex:
			if str(ex) == 'Game ended':
				done = True
			else:
				raise ex

		observation = self._observation()
		reward = self._reward()
		return observation, reward, done, {}


	def _reward(self):
		assert self._game
		if self._game.player1.base_health <= 0:		# Player1 lost
			return -1
		elif self._game.player2.base_health <= 0: 	# Player1 won
			return +1
		else:
			return 0 								# Game did not end


	def render(self):
		assert self._game
		return self._game.board_to_str()


	def legal_actions(self):
		assert self._game
		from stormbot.game.ability_utils import positions_in_frontline
		end_turn = 0
		hand = self._game.current_player.hand
		mana = self._game.current_player.mana

		discard_card = tuple(range(1, len(hand)+1)) if not self._game.card_discarded else ()
		playable_cards = tuple(idx for idx, card in enumerate(hand) if card.mana <= mana)
		card_actions = []
		if playable_cards:
			empty_positions = (pos for pos in positions_in_frontline(self._game) 
								if self._game.board[pos] is None)
			card_actions = [
				encode_unit_action(card_idx, pos) # TODO: Spells
				for card_idx in playable_cards
				for pos in empty_positions]

		legal_actions = [*discard_card, *card_actions]
		return legal_actions if legal_actions else [end_turn]



class BoardHandHistory:
	def __init__(self, length: int) -> None:
		self._buffer = np.zeros((length, 6, 4, 7), dtype=np.intc)


	def push(self, board: GameBoard, player1: Player):
		board_array = self.encode(board, player1=player1)
		self._buffer[-1] = board_array
		self._buffer = np.roll(self._buffer, 1, axis=0)


	def encode(self, board: GameBoard, *, player1: Player):
		array = np.zeros((6, 4, 7), dtype=np.intc)
		# encode board
		for piece in board:
			row, col = piece.pos
			idx = 0 if piece.type is CardType.UNIT else 1
			offset = 0 if piece.owner is player1 else 3
			array[row, col, idx+offset] = piece.strength
			array[row, col, 2+offset] = encode_id(piece.card.id)
		# encode hand
		hand = player1.hand
		card_idx = 0
		for card in hand:
			array[card_idx, 0, 6] = encode_id(card.id)
			array[card_idx, 1, 6] = card.mana
			array[card_idx, 2, 6] = card.strength if card.type is not CardType.SPELL else 0
			array[card_idx, 3, 6] = card.movement if card.type is CardType.UNIT else 0
			card_idx += 1
		return array


	def view(self, *, p1_perspective):
		array = self._buffer.copy()

		if not p1_perspective:
			for observation in array:
				rotated = np.rot90(observation[:, :, :6], k=2)	# rotate first 6 planes encoding the board
				rotated = np.roll(rotated, axis=-1, shift=3)	# swapt p1 and p2 boards
				np.copyto(observation[:, :, :6], rotated)

		# Concatenate k stacks of 14 planes to one stack of k * 14 planes
		array = np.concatenate(array, axis=-1)
		return array


	def reset(self):
		self._buffer[:] = 0


def encode_id(id: str) -> int:
	faction_int = ord(id[0])
	encoded = 10
	while encoded <= faction_int:
		encoded *= 10
	encoded *= faction_int
	encoded += int(id[1:])
	return encoded

def encode_unit_action(card_idx, piece_pos) -> int:
	return np.ravel_multi_index((card_idx, *piece_pos), (4, 6, 4)) + 5

def decode_unit_action(action: int) -> tuple:
	return np.unravel_index(action - 5, (4, 6, 4))
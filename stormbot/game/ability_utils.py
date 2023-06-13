import random
from typing import Any
import stormbot.game.positions as pos_module
from stormbot.game.board import Board
from stormbot.game.card import Card
from stormbot.game.enums import CardType
from stormbot.game.game import BoardPiece, Game, Spell, Structure, Unit
from stormbot.game.player import Player


def enemy_unit(origin: Any, piece: BoardPiece): 
	return (piece.type is CardType.UNIT and 
			piece.owner is not origin.owner)


def friendly_unit(origin: Any, piece: BoardPiece): 
	return (piece.type is CardType.UNIT and 
			piece.owner is origin.owner)


def surrounding_pieces(piece: BoardPiece, **kwargs):
	if piece is not None:
		board = piece.board
		pos = piece.pos
	else:
		board = kwargs['board']	
		pos = kwargs['pos']
	surrounding_pos = pos_module.surrounding_pos(pos, board)
	surrounding_pos = (p for p in surrounding_pos if p)
	surrounding_pos = sorted(surrounding_pos, reverse=not board.reversed)
	surrounding_pieces = []
	for p in surrounding_pos:
		surr_piece = board[p]
		if surr_piece is not None and surr_piece.strength > 0:
			surrounding_pieces.append(surr_piece)
	return surrounding_pieces

def pieces_in_front(piece: BoardPiece):
	game = piece.game
	board = piece.board
	row, col = piece.pos
	if game.is_player1_turn():
		rows_in_front = range(row-1, -1, -1)
	else:
		rows_in_front = range(row+1, board.rows)
	positions = [(r, col) for r in rows_in_front]
	pieces = []
	for pos in positions:
		front_piece = board[pos]
		if front_piece is not None:
			pieces.append(front_piece)
	return pieces

def positions_in_frontline(game: Game):
	board = game.board
	frontline = game.current_frontline
	cols = range(board.cols)
	if game.is_player1_turn():
		rows = range(frontline, board.rows)
	else:
		rows = range(frontline+1)
	return [(r, c) for r in rows for c in cols]


def spawn(card: Card, game: Game, owner: Player, position = None):
	board = game.board
	card_type = Unit if card.type is CardType.UNIT else Structure
	if position is None: # Random position within frontline
		position = [pos for pos in positions_in_frontline(game) if board[pos] is None]
	if isinstance(position, list):
		if position == []: return
		position = random.choice(position)
	boardpiece = card_type(card, owner, game, position)
	board[position] = boardpiece
	return boardpiece


# includes enemy base
def bordering_an_enemy(origin: Unit):
	board = origin.board
	bordering_positions = pos_module.bordering_pos(origin.pos, board)
	if bordering_positions[0] is None:	# Enemy base
		return True
	for pos in bordering_positions:
		if pos is not None:
			piece = board[pos]
			if (piece is not None and
				piece.owner is not origin.owner):
				return True
	return False

# includes enemy base
def surrounding_an_enemy(origin: Unit):
	board = origin.board
	surrounding_positions = pos_module.surrounding_pos(origin.pos, board)
	if surrounding_positions[0] is None:	# Enemy base
		return True
	for pos in surrounding_positions:
		if pos is not None:
			piece = board[pos]
			if (piece is not None and
				piece.owner is not origin.owner):
				return True
	return False
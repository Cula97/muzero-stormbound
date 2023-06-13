from stormbot.game.board import Board
from stormbot.game.player import Player 


def inside_board(pos: tuple, board: Board):
    return pos[0] in range(board.rows) and pos[1] in range(board.cols)


# (front, inward, outward, back)
def bordering_pos(pos: tuple, board: Board):
    row, col = pos
    ver_dir = -1 if not board.reversed  else +1  # frontward
    hor_dir = +1 if col < board.cols//2 else -1  # inward
    bordering = (
        (row + ver_dir, col),  # front
        (row, col + hor_dir),  # inward
        (row, col - hor_dir),  # outard
        (row - ver_dir, col))  # back
    return tuple(pos if inside_board(pos, board) else None for pos in bordering)


# (front, inw, out, back, fr_in, fr_ou, ba_in, ba_ou)
def surrounding_pos(pos: tuple, board: Board):
    row, col = pos
    ver_dir = -1 if not board.reversed  else +1  # frontward
    hor_dir = +1 if col < board.cols//2 else -1  # inward
    surrounding = (
        (row + ver_dir, col),            # front
        (row, col + hor_dir),            # inward
        (row, col - hor_dir),            # outward
        (row - ver_dir, col),            # back
        (row + ver_dir, col + hor_dir),  # front-inward
        (row + ver_dir, col - hor_dir),  # front-outward
        (row - ver_dir, col + hor_dir),  # back-inward
        (row - ver_dir, col - hor_dir))  # back-outward
    return tuple(pos if inside_board(pos, board) else None for pos in surrounding)


def attack_pos(pos: tuple, board: Board):
    return bordering_pos(pos, board)[:3]
    

def enemy_at(pos: tuple, board: Board, current_player: Player):
    piece = board[pos]
    return piece is not None and piece.owner is not current_player
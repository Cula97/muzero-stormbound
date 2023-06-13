from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game.game import Game, CARDS
import random
import copy
from game import Game, CARDS
from card import Card
from player import Player, PlayerData
from enums import Status, UnitType

from positions import surrounding_pos
from ability_utils import spawn

def init():
    global game
    cards = list(CARDS.values())
    p1 = PlayerData('p1', random_deck(cards))
    p2 = PlayerData('p2', random_deck(cards))
    game = Game()
    game.start(p1, p2)

def random_deck(cards: list[Card], size = 12):
    deck = []
    while size:
        card = random.choice(cards)
        if not(card.token or card in deck):
            deck.append(card)
            size -= 1
    return deck


def _reset_game():
    for pos in game.board.pos_iter():
        game.board[pos] = None
    p1, p2 = game.current_player, game.current_enemy
    p1.base_health, p2.base_health = 12, 12
    game.turn_count = 1
    p1.mana, p2.mana = 0, 0
    game._replenish_mana()
    game._replenish_hand()


def testN89():
    _reset_game()
    game.force_play_card(CARDS['N89'], (5, 1))
    game.step()
    game.step()
    for piece in game.board:
        print(piece.status)


def testN1():
    _reset_game()
    enemy = game.current_enemy
    game.force_play_card(CARDS['N1'], (4, 1))
    origin = game.board[(3, 1)]
    game.print_board()
    for pos in surrounding_pos(origin.pos, game.board):
        spawn(CARDS['T3'], game, enemy, pos)
    game.print_board()
    game.apply_damage(origin, origin.strength, None)
    game.print_board()
    for piece in game.board:
        if len(piece.status) != 0:
            print(piece.status)
            print(piece.pos)


def testN2():
    _reset_game()
    for pos in game.board.pos_iter():
        spawn(CARDS['T2'], game, game.current_player, pos)
    game.board[(3, 2)] = None
    game.board[(2, 3)] = None
    game.force_play_card(CARDS['N2'])
    game.print_board()


def testN66():
    _reset_game()
    for i in range(2):
        spawn(CARDS['T2'], game, game.current_enemy, (i, 1))
        spawn(CARDS['T2'], game, game.current_enemy, (i, 3))
    spawn(CARDS['T2'], game, game.current_enemy, (2, 3))
    game.print_board()
    card = copy.copy(CARDS['N66'])
    card.level = 5
    game.force_play_card(card, (5, 1))
    game.force_play_card(card, (5, 3))
    game.print_board()


def testN105():
    _reset_game()
    enemy = spawn(CARDS['T3'], game, game.current_enemy, (1, 1))
    game.force_play_card(CARDS['N105'], (1, 1))
    friendly = spawn(CARDS['T13'], game, game.current_player, (2, 1))
    print(enemy.status)
    print(friendly.strength)
    game.force_play_card(CARDS['N105'], (1, 1))
    print(friendly.strength)


def testN90():
    _reset_game()
    game.force_play_card(CARDS['N90'], (5, 2))
    game.print_board()
    game.step()
    game.step()
    game.print_board()
    game.step()


def testN5():
    # N5 ON_PLAY [LAST_CARD] 
    _reset_game()
    game.force_play_card(CARDS['N5'], (1, 1))
    print(game.board[(1, 1)].strength)
    game.player1.hand = []
    game.force_play_card(CARDS['N5'], (1, 1))
    print(game.board[(1, 1)].strength)


def testN99():
    _reset_game()
    spawn(CARDS['N13'], game, game.current_enemy, (1, 1))._strength = 1
    spawn(CARDS['N13'], game, game.current_enemy, (2, 2))._strength = 1
    spawn(CARDS['N13'], game, game.current_enemy, (3, 3))._strength = 1
    for i in range(4):
        for j in range(4):
            if game.board[(i, j)] is None:
                spawn(CARDS['T2'], game, game.current_enemy, (i, j))._strength = 1
    game.print_board()
    card = copy.copy(CARDS['N99'])
    card.level = 3
    game.force_play_card(card)
    game.print_board()


def testN6():
    _reset_game()
    t = spawn(CARDS['N6'], game, game.current_player, (2, 2))
    o = spawn(CARDS['N6'], game, game.current_player, (1, 1))
    game.apply_damage(t, t.strength, None)
    game.print_board()
    print(o.strength)


def testN86():
    _reset_game()
    game.force_play_card(CARDS['N86'], (5, 2))
    game.print_board()


def testN62():
    _reset_game()
    e1 = spawn(CARDS['T3'], game, game.current_enemy, (0, 1))
    e2 = spawn(CARDS['T3'], game, game.current_enemy, (1, 1))
    e3 = spawn(CARDS['T3'], game, game.current_enemy, (2, 1))
    game.force_play_card(CARDS['N62'], (5, 1))
    print(e1.status)
    print(e2.status)
    print(e3.status)


def testN97():
    _reset_game()
    status = [Status.CONFUSED, Status.DISABLED, Status.FROZEN,
              Status.POISONED, Status.VITALIZED]
    for i in range(4):
        spawn(CARDS[f'T{i+1}'], game, game.current_player, (i+1, i))\
        .add_status(status[i])
    spawn(CARDS['T5'], game, game.current_player, (2, 0))\
        .add_status(status[4])
    
    game.print_board()
    for piece in game.board:
        print(f'{piece.pos=} {piece.status=}')
    
    print(f'{CARDS["N97"].strength=}')
    game.force_play_card(CARDS['N97'], (5, 1))
    unit = game.board[(5, 1)]
    print(f'{unit.strength=}')
    
    game.print_board()
    for piece in game.board:
        print(f'{piece.pos=} {piece.status=}')



def testN67():
    _reset_game()
    game.force_play_card(CARDS['N67'], (4, 0))
    spawn(CARDS['T3'], game, game.current_enemy, (5, 2))
    game.force_play_card(CARDS['N67'], (4, 1))
    game.force_play_card(CARDS['N67'], (4, 2))
    game.print_board()


def testN101():
    _reset_game()
    for i in range(3):
        card = copy.copy(CARDS[f'T{i+1}'])
        card.mana = 5
        spawn(card, game, game.current_player, (i, i))

    for i in  range(1, 6):
        card = copy.copy(CARDS['N101'])
        card.level = i
        game.force_play_card(card)
        for piece in game.board:
            print(f'{piece.card.id} {piece.strength}')


# N7 - On play, *spawn* a *1/2/2/3/3 strength* Knight on the tile behind
def testN7():
    _reset_game()
    game.force_play_card(CARDS['N7'], (4, 1))
    game.print_board()
    print(game.board[(5, 1)].strength)
    game.step()
    game.force_play_card(CARDS['N7'], (1, 2))
    game.print_board()
    print(game.board[(0, 2)].strength)


# N8 - On play, *create* a *0* cost unit with *5/6/6/8/10 strength* and add
# it to your deck
def testN8():
    _reset_game()
    deck = game.current_player.deck
    deck.clear()
    for _ in range(6):
        game.force_play_card(CARDS['N8'], (0, 0))
    for wcard in deck:
        print(wcard.weight, wcard.card.id)


# N9 - *Reduce* the strength of a target enemy unit to *5/4/3/2/1*
def testN9():
    _reset_game()
    enemy = spawn(CARDS['T3'], game, game.current_enemy, (1, 1))
    enemy._strength = 10
    print(f'{enemy.strength = }')
    game.force_play_card(CARDS['N9'], (1, 1))
    print(f'{enemy.strength = }')


# N10 - On play, *deal 2/2/3/3/4 damage* to non-Dragon units in front
def testN10():
    _reset_game()
    spawn(CARDS['T3'], game, game.current_enemy, (0, 1))
    spawn(CARDS['T11'], game, game.current_enemy, (1, 1))
    spawn(CARDS['T3'], game, game.current_player, (2, 1))
    game.force_play_card(CARDS['N10'], (5, 1))
    game.print_board()


# N94 - *Disable* a target unitâ€™s ability, then *deal 2/3/4/5/6 damage*
def testN94():
    _reset_game()
    enemy = spawn(CARDS['T3'], game, game.current_enemy, (1, 1))
    enemy._strength = 10
    game.force_play_card(CARDS['N94'], (1, 1))
    print(f'{enemy.strength = }')
    print(f'{enemy.status = }')


# N11 - On play, *deal 2/3/4/5/6 damage* to a random *surrounding* enemy
def testN11():
    _reset_game()
    game.force_play_card(CARDS['N11'], (1, 1))
    spawn(CARDS['T3'], game, game.current_enemy, (2, 2))
    game.force_play_card(CARDS['N11'], (1, 1))
    game.print_board()


# N12 - On play, *discard* a random non-Pirate card from your hand
def testN12():
    _reset_game()
    hand = game.current_player.hand
    print(f'{len(hand) = }')
    game.force_play_card(CARDS['N12'], (5, 1))
    game.force_play_card(CARDS['N12'], (5, 2))
    print(f'{len(hand) = }')
    hand.clear()
    game.force_play_card(CARDS['N12'], (5, 3))
    print(f'{len(hand) = }')


# N14 - On play, *draw 1/1/1/2/2* card/card/card/cards/cards
def testN14():
    _reset_game()
    hand = game.current_player.hand
    print(f'{len(hand) = }')
    game.force_play_card(CARDS['N14'], (5, 1))
    game.force_play_card(CARDS['N14'], (5, 2))
    print(f'{len(hand) = }')


# N61 - On play, randomly *force* a *surrounding* confused enemy unit to
# attack a *bordering* enemy
def testN61():
    _reset_game()
    game.force_play_card(CARDS['N61'], (1, 1))
    spawn(CARDS['T3'], game, game.current_enemy, (1, 2)).add_status(Status.CONFUSED)
    game.force_play_card(CARDS['N61'], (1, 1))
    game.print_board()
    spawn(CARDS['T3'], game, game.current_enemy, (1, 3))
    spawn(CARDS['T3'], game, game.current_enemy, (0, 1)).add_status(Status.CONFUSED)
    game.force_play_card(CARDS['N61'], (1, 1))
    game.force_play_card(CARDS['N61'], (1, 1))
    game.print_board()
    print(f'{game.current_enemy.base_health = }')


# N23 - Randomly *deal 2/3/4/5/6 damage* to one unit of each *unit type*
def testN23():
    _reset_game()
    spawn(CARDS['T1'], game, game.current_enemy, (0, 0))._strength = 10 # construct
    spawn(CARDS['T2'], game, game.current_enemy, (0, 1))._strength = 10 # frostling
    spawn(CARDS['T2'], game, game.current_enemy, (0, 2))._strength = 10 # frostling
    knight = spawn(CARDS['T3'], game, game.current_enemy, (0, 3)) # knight
    knight._strength = 10
    knight.unit_types = (UnitType.KNIGHT, UnitType.DRAGON)
    game.force_play_card(CARDS['N23'])
    for piece in game.board:
        print(f'{piece.unit_types} {piece.strength = }')


# N88 - Before moving, *teleport* itself to a random tile in its row
def testN88():
    _reset_game()
    game.force_play_card(CARDS['N88'], (5, 1))
    game.force_play_card(CARDS['N88'], (3, 2))
    game.print_board()


# N24 - On play, *give 2/2/3/3/4 strength* to another random friendly unit
def testN24():
    _reset_game()
    spawn(CARDS['T3'], game, game.current_player, (4, 0))
    spawn(CARDS['T3'], game, game.current_player, (4, 1))
    spawn(CARDS['T3'], game, game.current_player, (4, 2))
    game.force_play_card(CARDS['N24'], (5, 1))
    game.print_board()
    for piece in game.board:
        print(f'{piece.strength = }')


# N15 - *Give 2/3/4/5/6 strength* to a target friendly unit and *vitalize* it
def testN15(): 
    _reset_game()
    spawn(CARDS['T3'], game, game.current_player, (4, 0))
    game.force_play_card(CARDS['N15'], (4, 0))
    for piece in game.board:
        print(f'{piece.strength = }\n{piece.status = }')


# N100 - *Destroy* a target friendly Ancient unit and *play* a random level 1/2/3/4/5
# non-Ancient unit on the same tile
def testN100():
    _reset_game()
    spawn(CARDS['T15'], game, game.current_player, (4, 2))
    game.force_play_card(CARDS['N100'], (4, 2))
    game.print_board()


# N81 - At the start of your turn, *remove 1/1/1/2/2 strength* from the stronger base
# and *give* it to the weaker one
def testN81():
    _reset_game()
    game.force_play_card(CARDS['N81'], (5, 0))
    game.step()
    game.step()
    print(f'{game.current_player.base_health = }')
    print(f'{game.current_enemy.base_health = }')
    game.step()
    game.step()
    print(f'{game.current_player.base_health = }')
    print(f'{game.current_enemy.base_health = }')


# N85 - At the start of your turn, *give* (or *remove*) *fixedly forward* movement to 
# the leftmost unit card in your hand, then *destroy* the weakest confused unit
def testN85():
    _reset_game()
    game.force_play_card(CARDS['N85'], (5, 0))
    hand = game.current_player.hand
    for _ in range(5):
        for card in hand:
            print(card.__dict__.get('fixedly_forward'))
        game.step()
        print('----')



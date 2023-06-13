from abc import ABC, abstractmethod
from math import inf
import math
import random
import copy
from typing import Callable, Optional
from stormbot.game import ability_utils, positions
from stormbot.game.ability import Ability

from stormbot.game.ability_utils import *
from stormbot.game.card import WeightedCard
from stormbot.game.enums import CardType, Event, Stat, Status, Target, UnitType
from stormbot.game.game import BoardPiece, Spell, Unit, CARDS
from stormbot.game.positions import bordering_pos


# N89 - Before moving, *get* a random *status effect*
class _N89_Ability(Ability):
    def __init__(self):
        self.trigger = Event.BEFORE_MOVE
        self.text = "Before moving, *get* a random *status effect*"

    def activate(self, *, origin: Unit):
        statuses = [Status.CONFUSED, Status.FROZEN, Status.POISONED, Status.VITALIZED]
        status = random.choice(statuses)
        origin.add_status(status)


# N1 - On death, *give 1/2/3/4/5 strength* to a random *surrounding* enemy unit and *vitalize* it
class _N1_Ability(Ability):
    def __init__(self):
        self.trigger = Event.ON_DEATH
        self.text = "On death, *give 1/2/3/4/5 strength* to a random *surrounding* enemy unit and *vitalize* it"

    def activate(self, *, origin: Unit, **_):
        surr_pieces = surrounding_pieces(origin)
        surr_enemies = [piece for piece in surr_pieces if enemy_unit(origin, piece)]
        if surr_enemies != []:
            target = random.choice(surr_enemies)
            target._strength += origin.card.level
            target.add_status(Status.VITALIZED)


# N2 - Randomly *spawn* a Knight with *1/2/3/4/5 strength*
class _N2_Ability(Ability):
    def __init__(self):
        self.trigger = Event.ON_PLAY
        self.text = "Randomly *spawn* a Knight with *1/2/3/4/5 strength*"

    def activate(self, *, origin: Spell):
        knight = spawn(CARDS['T3'], origin.game, origin.owner)
        if knight is not None: 
            knight._strength = origin.card.level


# N66 - On play, *gain speed* equal to the amount of enemy units in front
class _N66_Ability(Ability):
    def __init__(self):
        self.trigger = Event.ON_PLAY
        self.text = "On play, *gain speed* equal to the amount of enemy units in front"

    def activate(self, *, origin: Unit):
        pieces = pieces_in_front(origin)
        count = 0
        for piece in pieces:
            if enemy_unit(origin, piece):
                count += 1
        origin.movement += count


# N105 - *Confuse* a target unit and *give 1/2/3/4/5 strength* to the weakest friendly feline
class _N105_Ability(Ability):
    def __init__(self):
        self.trigger = Event.ON_PLAY
        self.text = "*Confuse* a target unit and *give 1/2/3/4/5 strength* to the weakest friendly feline"

    def activate(self, *, origin: Spell, target: Unit):
        def friendly_feline(piece: BoardPiece):
            return (piece.owner is origin.owner and 
                    piece.type is CardType.UNIT and 
                    UnitType.FELINE in piece.unit_types)

        target.add_status(Status.CONFUSED)
        min_str = inf
        weak_felines = []
        for piece in origin.board:
            if friendly_feline(piece) and piece.strength <= min_str:
                if piece.strength < min_str:
                    min_str = piece.strength
                    weak_felines.clear()
                weak_felines.append(piece)
        if weak_felines != []:
            feline = random.choice(weak_felines)
            feline._strength += origin.card.level


# TODO: check behavior
# N90 - Before moving, *split* itself into a unit on both sides and cannot split again this turn
class _N90_Ability(Ability):
    def __init__(self):
        self.trigger = Event.BEFORE_MOVE
        self.text = "Before moving, *split* itself into a unit on both sides and cannot split again this turn"

    def activate(self, *, origin: Unit):
        last_time_activated = getattr(origin, 'last_time_activated', -1)
        if last_time_activated != origin.game.turn_count:
            last_time_activated = origin.game.turn_count
            split_strength = max(math.floor(origin.strength / 3), 1)
            board = origin.board
            row, col = origin.pos
            if col-1 in range(board.cols) and board[(row, col-1)] is None:
                split_unit = spawn(CARDS['N90'], origin.game, origin.owner, (row, col-1))
                split_unit._strength = split_strength
                split_unit.movement = 1
                setattr(split_unit, 'last_time_activated', last_time_activated)
                setattr(origin, 'last_time_activated', last_time_activated)
                origin._strength -= split_strength
            if origin.strength > 0 and col+1 in range(board.cols) and board[(row, col+1)] is None:
                split_unit = spawn(CARDS['N90'], origin.game, origin.owner, (row, col+1))
                split_unit._strength = split_strength
                split_unit.movement = 1
                setattr(split_unit, 'last_time_activated', last_time_activated)
                setattr(origin, 'last_time_activated', last_time_activated)
                origin._strength -= split_strength
            if origin.strength <= 0:
                board[origin.pos] = None


# N5 - When played as the last card from your hand, *gain 6/7/8/10/12 strength*
class _N5_Ability(Ability):
    def __init__(self):
        self.trigger = Event.ON_PLAY
        self.text = "When played as the last card from your hand, *gain 6/7/8/10/12 strength*"

    def activate(self, *, origin: Unit):
        if origin.owner.hand == []:
            values = [6, 7, 8, 10, 12]
            value = values[origin.card.level-1]
            origin._strength += value


# N99 - *Deal 1/2/3/4/5 damage* to non-temple structures, then spread the amount received among *surrounding* enemy units and structures
class _N99_Ability(Ability):
    def __init__(self):
        self.trigger = Event.ON_PLAY
        self.text = "*Deal 1/2/3/4/5 damage* to non-temple structures, then spread the amount received among *surrounding* enemy units and structures"

    def activate(self, *, origin: Spell):
        value = origin.card.level
        game = origin.game
        board = origin.board
        non_temples = [piece for piece in board
                       if piece.type is CardType.STRUCTURE and not piece.is_temple()]
        if non_temples == []: return
        enemies_to_damage: dict[BoardPiece, int] = {}
        for structure in non_temples:
            surr_enemies = [enemy for enemy in surrounding_pieces(structure)
                            if enemy.owner is not origin.owner]
            remaining_damage = min(value, structure.strength)
            while remaining_damage:
                if surr_enemies == []: break
                enemy = random.choice(surr_enemies)
                damage_dealt = enemies_to_damage.get(enemy, 0)
                if damage_dealt == 0 and enemy in non_temples:
                    damage_dealt == min(value, enemy.strength)
                max_damage = max(min(remaining_damage, enemy.strength - damage_dealt), 0)
                damage = random.randint(0, max_damage)
                total_damage = damage_dealt + damage
                enemies_to_damage[enemy] = total_damage
                remaining_damage -= damage
                if enemy.strength - total_damage <= 0:
                    assert enemy.strength - total_damage == 0
                    surr_enemies.remove(enemy)
        game.apply_damage(non_temples, value, origin)
        # check for side effects and border case
        to_remove = [piece for piece in enemies_to_damage if piece.strength <= 0]
        for piece in to_remove:
            del enemies_to_damage[piece]
        for piece in enemies_to_damage:
            if piece in non_temples:
                enemies_to_damage[piece] -= value # non temple hit by second effect
        game.apply_damage(enemies_to_damage.keys(), list(enemies_to_damage.values()), origin)


# N6 - On death, *give 3/4/5/6/7 strength* to a random friendly Dragon
class _N6_Ability(Ability):
    def __init__(self):
        self.trigger = Event.ON_DEATH
        self.text = "On death, *give 3/4/5/6/7 strength* to a random friendly Dragon"

    def activate(self, *, origin: Unit, **_):
        friendly_dragons = []
        for piece in origin.board:
            if friendly_unit(origin, piece) and UnitType.DRAGON in piece.unit_types:
                friendly_dragons.append(piece)
        if friendly_dragons != []:
            dragon = random.choice(friendly_dragons)
            dragon._strength += origin.card.level+2


# N86 - On play, *confuse* itself then *gain 2 speed*
class _N86_Ability(Ability):
    def __init__(self):
        self.trigger = Event.ON_PLAY
        self.text = "On play, *confuse* itself then *gain 2 speed*"

    def activate(self, *, origin: Unit):
        origin.add_status(Status.CONFUSED)
        origin.movement += 2


# N62 - On play, *confuse* enemy units in front
class _N62_Ability(Ability):
    def __init__(self):
        self.trigger = Event.ON_PLAY
        self.text = "On play, *confuse* enemy units in front"

    def activate(self, *, origin: Unit):
        pieces = pieces_in_front(origin)
        enemy_units = [piece for piece in pieces if enemy_unit(origin, piece)]
        for enemy in enemy_units:
            enemy.add_status(Status.CONFUSED)


# N97 - Before moving, *steal* all status effects from all friendly units and *gain 1/2/2/3/3 strength* for each unique status effect received
class _N97_Ability(Ability):
    def __init__(self):
        self.trigger = Event.BEFORE_MOVE
        self.text = "Before moving, *steal* all status effects from all friendly units and *gain 1/2/2/3/3 strength* for each unique status effect received"

    def activate(self, *, origin: Unit):
        friendly_units = [piece for piece in origin.board if friendly_unit(origin, piece)]
        friendly_units.remove(origin)
        status_recieved = set() # TODO: clarificar si recibir un efecto que ya tenia se cuenta o no)
        for unit in friendly_units:
            assert Status.FIXEDLY_FORWARD not in unit.status
            for status in unit.status:
                status_recieved.add(status)
                origin.add_status(status)
            unit.status.clear()
        if len(status_recieved) > 0:
            values = [1, 2, 2, 3, 3]
            value = values[origin.card.level-1]
            print(value)
            origin._strength += value * len(status_recieved)


# N67 - If played with no *surrounding* enemies, *gain 2 speed*, else if played with no *bordering* enemies *gain 1 speed*
class _N67_Ability(Ability):
    def __init__(self):
        self.trigger = Event.ON_PLAY
        self.text = "If played with no *surrounding* enemies, *gain 2 speed*, else if played with no *bordering* enemies *gain 1 speed*"

    def activate(self, *, origin: Unit):
        if not surrounding_an_enemy(origin):
            origin.movement += 2
        elif not bordering_an_enemy(origin):
            origin.movement += 1


# N101 - *Give* strength to 1/1/2/2/3 random friendly unit/unit/units/units/units
# with no ability equal to their mana cost, then *give* them additional strength
# up to 1/2/2/3/5 of their mana cost
class _N101_Ability(Ability):
    def __init__(self):
        self.trigger = Event.ON_PLAY
        self.text = "*Give* strength to 1/1/2/2/3 random friendly unit/unit/units/units/units with no ability equal to their mana cost, then *give* them additional strength up to 1/2/2/3/5 of their mana cost"

    def activate(self, *, origin: Spell):
        units_count_values = [1, 1, 2, 2, 3]
        units_count = units_count_values[origin.card.level-1]
        board = origin.board
        friendly_units = [piece for piece in board if friendly_unit(origin, piece) and piece.ability is None]
        if friendly_units == []:
            return
        if len(friendly_units) <= units_count:
            selected_units = friendly_units
        else:
            selected_units = []
            for _ in range(units_count):
                unit = random.choice(friendly_units)
                selected_units.append(unit)
                friendly_units.remove(unit)
        bonus_cap_values = [1, 2, 2, 3, 5]
        for unit in selected_units:
            mana = unit.card.mana
            unit._strength += mana
            bonus_cap = bonus_cap_values[origin.card.level-1]
            bonus = min(mana, bonus_cap)
            unit._strength += bonus


# N7 - On play, *spawn* a *1/2/2/3/3 strength* Knight on the tile behind
class _N7_Ability(Ability):
    def __init__(self):
        self.trigger = Event.ON_PLAY
        self.text = "On play, *spawn* a *1/2/2/3/3 strength* Knight on the tile behind"

    def activate(self, *, origin: Unit):
        board = origin.board
        pos_behind = bordering_pos(origin.pos, board)[3]
        if pos_behind is not None and board[pos_behind] is None:
            values = [1, 2, 2, 3, 3]
            strength = values[origin.card.level-1]
            knight = spawn(CARDS['T3'], origin.game, origin.owner, pos_behind)
            knight._strength = strength


# N8 - On play, *create* a *0* cost unit with *5/6/6/8/10 strength* and add it to your deck
class _N8_Ability(Ability):
    def __init__(self):
        self.trigger = Event.ON_PLAY
        self.text = "On play, *create* a *0* cost unit with *5/6/6/8/10 strength* and add it to your deck"

    def activate(self, *, origin: Unit):
        token_ids = [f'T{id}' for id in range(1, 16)]
        token_ids.remove('T12') # Token structure
        tokens = [CARDS[id] for id in token_ids]
        card = copy.copy(random.choice(tokens))
        values = [5, 6, 6, 8, 10]
        strength = values[origin.card.level-1]
        card._strength = [strength] * 5
        weighted_card = WeightedCard(card)
        origin.owner.deck.append(weighted_card)


# N9 - *Reduce* the strength of a target enemy unit to *5/4/3/2/1*
class _N9_Ability(Ability):
    def __init__(self):
        self.trigger = Event.ON_PLAY
        self.text = "*Reduce* the strength of a target enemy unit to *5/4/3/2/1*"

    def activate(self, *, origin: Spell, target: Unit):
        target._strength = min(6 - origin.card.level, target.strength)


# N10 - On play, *deal 2/2/3/3/4 damage* to non-Dragon units in front
class _N10_Ability(Ability):
    def __init__(self):
        self.trigger = Event.ON_PLAY
        self.text = "On play, *deal 2/2/3/3/4 damage* to non-Dragon units in front"

    def activate(self, *, origin: Unit):
        def non_dragon_unit(piece):
            return piece.type is CardType.UNIT and UnitType.DRAGON not in piece.unit_types
        
        in_front = pieces_in_front(origin)
        non_dragons = [piece for piece in in_front if non_dragon_unit(piece)]
        values = [2, 2, 3, 3, 4]
        damage = values[origin.card.level-1]
        if non_dragons != []:
            origin.game.apply_damage(non_dragons, damage, self)


# N94 - *Disable* a target unit's ability, then *deal 2/3/4/5/6 damage*
class _N94_Ability(Ability):
    def __init__(self):
        self.trigger = Event.ON_PLAY
        self.text = "*Disable* a target unit's ability, then *deal 2/3/4/5/6 damage*"

    def activate(self, *, origin: Spell, target: Unit):
        target.add_status(Status.DISABLED)
        damage = origin.card.level + 1
        origin.game.apply_damage(target, damage, origin)


# N11 - On play, *deal 2/3/4/5/6 damage* to a random *surrounding* enemy
class _N11_Ability(Ability):
    def __init__(self):
        self.trigger = Event.ON_PLAY
        self.text = "On play, *deal 2/3/4/5/6 damage* to a random *surrounding* enemy"

    def activate(self, *, origin: Unit):
        surr_pieces = surrounding_pieces(origin)    # base cannot be surrounding on play
        enemies = [piece for piece in surr_pieces
                   if piece.owner is not origin.owner]
        if enemies != []:
            damage = origin.card.level + 1
            enemy = random.choice(enemies)
            origin.game.apply_damage(enemy, damage, self)


# N12 - On play, *discard* a random non-Pirate card from your hand
class _N12_Ability(Ability):
    def __init__(self):
        self.trigger = Event.ON_PLAY
        self.text = "On play, *discard* a random non-Pirate card from your hand"

    def activate(self, *, origin: Unit):
        def non_pirate_card(card):
            return (card.type is not CardType.UNIT or
                    UnitType.PIRATE not in card.unit_types)
        
        hand = origin.owner.hand
        non_pirates = [card for card in hand if non_pirate_card(card)]
        if non_pirates != []:
            card = random.choice(non_pirates)
            hand.remove(card)
            origin.owner.deck.append(WeightedCard(card))


# N14 - On play, *draw 1/1/1/2/2* card/card/card/cards/cards
class _N14_Ability(Ability):
    def __init__(self):
        self.trigger = Event.ON_PLAY
        self.text = "On play, *draw 1/1/1/2/2* card/card/card/cards/cards"

    def activate(self, *, origin: Unit):
        values = [1, 1, 1, 2, 2]
        amount = values[origin.card.level-1]
        for _ in range(amount):
            origin.game.draw_card()


# N61 - On play, randomly *force* a *surrounding* confused enemy unit to attack a *bordering* enemy
class _N61_Ability(Ability):
    def __init__(self):
        self.trigger = Event.ON_PLAY
        self.text = "On play, randomly *force* a *surrounding* confused enemy unit to attack a *bordering* enemy"

    def activate(self, *, origin: Unit):
        surr_pieces = surrounding_pieces(origin)
        confused = [piece for piece in surr_pieces 
                    if (enemy_unit(origin, piece) and
                        Status.CONFUSED in piece.status)]
        if confused != []:
            board = origin.board
            unit = random.choice(confused)
            bord_pos = pos_module.bordering_pos(unit.pos, board)
            bord_targets = []
            if bord_pos[0] is None: # enemy at it's own baseline
                bord_targets.append(Target.BASE) 
            for pos in bord_pos:
                if pos is not None:
                    piece = board[pos]
                    if piece is not None and piece.owner is not origin.owner:
                        bord_targets.append(pos)
            if bord_targets != []:
                target = random.choice(bord_targets)
                unit.attack(target)



# N23 - Randomly *deal 2/3/4/5/6 damage* to one unit of each *unit type*
class _N23_Ability(Ability):
    def __init__(self):
        self.trigger = Event.ON_PLAY
        self.text = "Randomly *deal 2/3/4/5/6 damage* to one unit of each *unit type*"

    def activate(self, *, origin: Spell):
        damage = origin.card.level + 1
        types_map = dict()
        for piece in origin.board:
            if piece.type is CardType.UNIT:
                for unit_type in piece.unit_types:
                    units = types_map.get(unit_type)
                    if units is None:
                        types_map[unit_type] = [piece]
                    else:
                        units.append(piece)
        one_of_each = []
        damages = []
        # remove duplicates and aggregate dmg (unit can have multiple types)
        for units in types_map.values():
            unit = random.choice(units)
            if unit in one_of_each:
                i = one_of_each.index(unit)
                damages[i] += damage
            else:
                one_of_each.append(unit)
                damages.append(damage)
        origin.game.apply_damage(one_of_each, damages, origin)


# N88 - Before moving, *teleport* itself to a random tile in its row
class _N88_Ability(Ability):
    def __init__(self):
        self.trigger = Event.BEFORE_MOVE
        self.text = "Before moving, *teleport* itself to a random tile in its row"

    def activate(self, *, origin: Unit):
        row, _ = origin.pos
        board = origin.board
        cols = board.cols
        row_pos = ((row, c) for c in range(cols))
        tiles = [pos for pos in row_pos if board[pos] is None]
        if tiles != []:
            tile = random.choice(tiles)
            board[origin.pos] = None
            board[tile] = origin
            origin.pos = tile


# N24 - On play, *give 2/2/3/3/4 strength* to another random friendly unit
class _N24_Ability(Ability):
    def __init__(self):
        self.trigger = Event.ON_PLAY
        self.text = "On play, *give 2/2/3/3/4 strength* to another random friendly unit"

    def activate(self, *, origin: Unit):
        board = origin.board
        friendly_units = [piece for piece in board if friendly_unit(origin, piece)]
        friendly_units.remove(origin)
        if friendly_units != []:
            values = [2, 2, 3, 3, 4]
            strength = values[origin.card.level-1]
            unit = random.choice(friendly_units)
            unit._strength += strength


# N15 - *Give 2/3/4/5/6 strength* to a target friendly unit and *vitalize* it
class _N15_Ability(Ability):
    def __init__(self):
        self.trigger = Event.ON_PLAY
        self.text = "*Give 2/3/4/5/6 strength* to a target friendly unit and *vitalize* it"

    def activate(self, *, origin: Spell, target: Unit):
        strength = origin.card.level + 1
        target._strength += strength
        target.add_status(Status.VITALIZED)


# N100 - *Destroy* a target friendly Ancient unit and *play* a random level
# 1/2/3/4/5 non-Ancient unit on the same tile
class _N100_Ability(Ability):
    def __init__(self):
        self.trigger = Event.ON_PLAY
        self.text = "*Destroy* a target friendly Ancient unit and *play* a random level 1/2/3/4/5 non-Ancient unit on the same tile"

    def activate(self, *, origin: Spell, target: Unit):
        damage = target.strength
        pos = target.pos
        board = origin.board
        game = origin.game
        game.apply_damage(target, damage, self)
        if board[pos] is None:
            non_ancient_units = []
            for card in CARDS.values():
                if card.type is CardType.UNIT and UnitType.ANCIENT not in card.unit_types:
                    non_ancient_units.append(card)
            card = copy.copy(random.choice(non_ancient_units))
            card.level = origin.card.level
            game.force_play_card(card, pos)



# N81 - At the start of your turn, *remove 1/1/1/2/2 strength* from the stronger base and *give* it to the weaker one
class _N81_Ability(Ability):
    def __init__(self):
        self.trigger = Event.ON_TURN_START
        self.text = "At the start of your turn, *remove 1/1/1/2/2 strength* from the stronger base and *give* it to the weaker one"

    def activate(self, *, origin: Structure):
        game = origin.game
        bases = [game.current_player, game.current_enemy]
        random.shuffle(bases)
        stronger, weaker = bases[0], bases[1]
        if stronger.base_health < weaker.base_health:
            stronger, weaker = weaker, stronger
        values = [1, 1, 1, 2, 2]
        damage = values[origin.card.level-1]
        game.apply_damage(stronger, damage, origin)
        weaker.base_health += damage


# N85 - At the start of your turn, *give* (or *remove*) *fixedly forward* movement to the leftmost unit card in your hand, then *destroy* the weakest confused unit
class _N85_Ability(Ability):
    def __init__(self):
        self.trigger = Event.ON_TURN_START
        self.text = "At the start of your turn, *give* (or *remove*) *fixedly forward* movement to the leftmost unit card in your hand, then *destroy* the weakest confused unit"

    def activate(self, *, origin: Structure):
        hand = origin.owner.hand
        unit_card = None
        for card in hand:
            if card.type is CardType.UNIT:
                unit_card = card
                break
        if unit_card is not None:
            fixedly_forward = hasattr(unit_card, 'fixedly_forward') and unit_card.fixedly_forward
            setattr(unit_card, 'fixedly_forward', not fixedly_forward)
        weakest = None
        for piece in origin.board:
            if piece.type is CardType.UNIT and Status.CONFUSED in piece.status:
                if weakest is None or piece.strength < weakest.strength:
                    weakest = piece
        if weakest is not None:
            origin.game.apply_damage(weakest, weakest.strength, self)
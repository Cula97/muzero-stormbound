from __future__ import annotations
from abc import ABC, abstractmethod
import random 
import math
from typing import Any, Iterable, Literal

from stormbot.game.card import Card, WeightedCard
from stormbot.game.player import Player, PlayerData
from stormbot.game.board import Board
from stormbot.game.enums import CardType, Event, Status, Target 


CARDS: dict = {}

# Cards with implemented abilities in harcoded_abilities
_implemented = frozenset({'N89', 'N1', 'N2', 'N66', 'N105', 'N90', 'N5', 'N99', 'N6', 'N86',
                'N62', 'N97', 'N67', 'N101', 'N7', 'N8', 'N9', 'N10', 'N94', 'N11',
                'N12', 'N14', 'N61', 'N23', 'N88', 'N24', 'N15', 'N100', 'N81', 'N85'})

class BoardPiece(ABC):
    def __init__(self, card: Card, player: Player, game: Game, pos: tuple):
        self.card = card
        self.pos = pos
        self.owner = player
        self.game = game
        self.board = game.board
        self._strength = card.strength
        self.type = card.type

    @property
    def strength(self):
        return self._strength

    @abstractmethod
    def play(self):
        raise NotImplementedError

    @abstractmethod
    def trigger_ability(self):
        raise NotImplementedError


DirectionT = Any
TargetT = Any


class Structure(BoardPiece):
    @property
    def ability(self):
        return self.card.ability

    def is_temple(self):
        return self.card.name.startswith('Temple')

    def play(self):
        if self.ability is not None:
            self.trigger_ability()

    def trigger_ability(self, **kwargs):
        assert self.ability is not None
        self.ability.activate(origin=self, **kwargs)


class Unit(BoardPiece):
    def __init__(self, card: Card, player: Player, game: Game, pos: tuple):
        super().__init__(card, player, game, pos)
        self.movement = card.movement   # how far it should move in the next turn
        self.unit_types = card.unit_types
        self.status = set()
        if hasattr(card, 'fixedly_forward'):
            if card.fixedly_forward:
                self.status.add(Status.FIXEDLY_FORWARD)

    @property
    def ability(self):
        if Status.DISABLED in self.status:
            return None
        return self.card.ability

    def incapacitated(self):
        return self._strength <= 0 or Status.FROZEN in self.status

    def play(self, *, being_placed = False):
        self._resolve_strength_status()
        if self.incapacitated():
            self.status.difference_update((Status.FROZEN, ))
            return
        ability = self.ability
        if ability is not None:
            if being_placed and ability.trigger is Event.ON_PLAY:
                self.trigger_ability()
                if self.incapacitated():
                    self.movement = 1
                    return
        self.move(being_placed)

    def _resolve_strength_status(self):
        if Status.POISONED in self.status:
            self.game.apply_damage(self, 1, Status.POISONED)
        elif Status.VITALIZED in self.status:
            self._strength += 1

    def trigger_ability(self, **kwargs):
        assert self.ability is not None
        if Status.FROZEN not in self.status:
            self.ability.activate(origin=self, **kwargs)

    def move(self, being_placed = False):
        assert not self.incapacitated()
        path = self._resolve_movement_directions(being_placed)
        starting_pos = self.pos
        board = self.board
        if len(path) > 0:
            ability = self.ability
            if ability is not None and ability.trigger is Event.BEFORE_MOVE:
                self.trigger_ability()
                if self.incapacitated():
                    self.movement = 1
                    return
                pos_after_ability = self.pos
                if starting_pos != pos_after_ability: # teleported/split before move
                    path = self._resolve_movement_directions(being_placed)
            for direction in path:
                if self.incapacitated(): break
                if direction is Target.BASE or board[direction] is not None:
                    self.attack(direction)
                else:
                    board[self.pos] = None
                    board[direction] = self
                    self.pos = direction
        self.movement = 1


    def _resolve_movement_directions(self, being_placed: bool):
        from stormbot.game.positions import attack_pos, enemy_at
        front, inward, outward = attack_pos(self.pos, self.board)
        path: list[DirectionT] = []
        # fixedly forward takes priority over confusion but there is no case where it applies
        # (unit affected by Temple of the Mind, with movements and that gets confused on play)
        if Status.FIXEDLY_FORWARD in self.status:
            assert being_placed
            self.status.remove(Status.FIXEDLY_FORWARD)  # fixedly forward lost even without initial movement
            for _ in range(self.movement):
                if front is None:
                    path.append(Target.BASE)
                    break
                else:
                    piece = self.board[front]
                    if piece is None or piece.owner is not self.owner:
                        path.append(front)
                        front = attack_pos(front, self.board)[0]
                    else:
                        break
        else:
            movement_left = self.movement
            pos = self.pos
            if movement_left > 0 and Status.CONFUSED in self.status:
                self.status.remove(Status.CONFUSED) # confussion lost only on movement
                sides: list[tuple[int, int]] = []
                if inward is not None:
                    sides.append(inward)
                if outward is not None:
                    sides.append(outward)
                pos = random.choice(sides)
                path.append(pos)
                movement_left -= 1

            if being_placed:
                for _ in range(movement_left):
                    assert isinstance(pos, tuple)
                    front, inward, outward = attack_pos(pos, self.board)
                    if front is None:
                        path.append(Target.BASE)
                        break
                    elif enemy_at(front, self.board, self.owner):
                        path.append(front)
                        pos = front
                    elif inward is not None and enemy_at(inward, self.board, self.owner):
                        path.append(inward)
                        pos = inward
                    elif outward is not None and enemy_at(outward, self.board, self.owner):
                        path.append(outward)
                        pos = outward
                    elif self.board[front] is None:
                        path.append(front)
                        pos = front
                    else:
                        piece = self.board[front]
                        assert piece is not None and piece.owner is self.owner
                        break
            else:
                assert self.movement == 1
                if movement_left == 1:      # unit was NOT confused
                    if front is None:
                        path.append(Target.BASE)
                    else:
                        piece = self.board[front]
                        if piece is None or piece.owner is not self.owner:
                            path.append(front)
        return path


    def attack(self, pos: tuple | Literal[Target.BASE]):
        board = self.board
        ability = self.ability
        def ability_triggers_with(event: Event):
            return ability is not None and ability.trigger is event
        if pos is Target.BASE:
            target = self.game.current_enemy
            damage = self.strength
        else:
            target = board[pos]
            assert target is not None
            damage = min(self.strength, target.strength)

        if ability_triggers_with(Event.BEFORE_ATTACK):
            self.trigger_ability(target=target, damage=damage)
            if self.incapacitated():
                return

        self._strength -= damage
        if self.strength <= 0:
            board[self.pos] = None

        self.game.apply_damage(target, damage, self)
        
        if self.strength <= 0 and ability_triggers_with(Event.ON_DEATH):
            self.trigger_ability()
        elif not self.incapacitated():
            assert isinstance(pos, tuple)
            target = board[pos]
            if target is None:
                board[self.pos] = None
                board[pos] = self
                self.pos = pos
                if (ability_triggers_with(Event.AFTER_ATTACK) or
                    ability_triggers_with(Event.AFTER_SURVIVING_DAMAGE)):
                    self.trigger_ability()
            elif target.owner is not self.owner:
                self.attack(pos)

    def add_status(self, st: Status):
        self.status.add(st)
        if st is Status.POISONED:
            self.status.difference_update((Status.VITALIZED,))
        if st is Status.VITALIZED:
            self.status.difference_update((Status.POISONED,))


class Spell:
    def __init__(self, card: Card, player: Player, game: Game, pos: tuple = None):
        self.card = card
        self.owner = player
        self.game = game
        self.pos = pos
        self.board = self.game.board

    def play(self):
        if self.pos is None:
            self.card.ability.activate(origin=self)
        else:
            self.card.ability.activate(origin=self, target=self.board[self.pos])


class Game:
    _ABILITIES_ENABLED = False
    
    def __init__(self):
        if CARDS == {}:
            _load_cards()
        self.board: Board[BoardPiece] = Board(5, 3, lambda: not self.is_player1_turn())
        self.turn_count = 0
        self.card_discarded = False

    def start(self, player1: PlayerData, player2: PlayerData):
        self.player1 = Player(player1, Game.random_weight_deck(player1.deck))
        self.player2 = Player(player2, Game.random_weight_deck(player2.deck))
        self.step()
        
    def is_player1_turn(self): 
        return self.turn_count % 2 == 1

    @property
    def current_player(self):
        return self.player1 if self.is_player1_turn() else self.player2

    @property
    def current_enemy(self):
        return self.player2 if self.is_player1_turn() else self.player1

    @property
    def current_baseline(self):
        return self.board.rows-1 if self.is_player1_turn() else 0

    @property
    def enemy_baseline(self):
        return 0 if self.is_player1_turn() else self.board.rows-1

    def baseline(self, player):
        if player is self.current_player:
            return self.current_baseline
        else:
            return self.enemy_baseline


    @property
    def current_frontline(self):
        for piece in self.board:
            if piece.owner is self.current_player:
                if piece.pos[0] == self.enemy_baseline:
                    return abs(self.enemy_baseline - 1)
                return piece.pos[0]
        return self.current_baseline

    def step(self):
        self.current_player.mana = 0    # previous player
        self.turn_count += 1
        self.card_discarded = False
        self._replenish_mana()
        self._replenish_hand()
        self._structure_phase()
        self._movement_phase()
        self.check_state()

    def check_state(self):
        for piece in self.board:
            assert piece.strength > 0, 'Dead piece on board'
            if piece.type is CardType.UNIT:
                assert piece.movement == 1
            assert self.board[piece.pos] == piece

        assert len(self.current_player.hand) == 4
        assert len(self.current_player.hand) + len(self.current_player.deck) >= 4

    def _replenish_mana(self):
        d, m = divmod(self.turn_count, 2)
        self.current_player.mana += 3 + d + m

    def _replenish_hand(self):
        missing_cards = 4 - len(self.current_player.hand)
        for _ in range(missing_cards): 
            self.draw_card()

    def draw_card(self):
        hand = self.current_player.hand
        deck = self.current_player.deck
        # select card using weighted random
        weight_sum = sum(card.weight for card in deck)
        rnd = random.randrange(weight_sum)
        drawn = None
        for card in deck:
            weight = card.weight
            if rnd < weight:
                drawn = card
                break
            rnd -= weight
        assert drawn is not None
        # reweight deck
        deck.remove(drawn)
        for card in deck:
            card.weight = math.floor(card.weight * 1.6 + 1)
        hand.append(drawn.card)

    def _structure_phase(self):
        for piece in self.board:
            if piece.owner is self.current_player and piece.type is CardType.STRUCTURE:
                piece.play()

    def _movement_phase(self):
        for piece in self.board:
            if piece.owner is self.current_player and piece.type is CardType.UNIT:
                piece.play()

    def discard_card(self, card_idx: int):
        assert not self.card_discarded
        card = self.current_player.hand.pop(card_idx)
        self.current_player.deck.append(WeightedCard(card))
        self.draw_card()
        self.card_discarded = True

    def play_card(self, card_idx: int, pos: tuple = None):
        player = self.current_player
        hand = self.current_player.hand
        deck = self.current_player.deck
        card = hand.pop(card_idx)
        assert player.mana >= card.mana
        player.mana -= card.mana

        if card.type == CardType.SPELL:
            Spell(card, player, self, pos).play()
        else:
            assert pos is not None
            piece = self.board[pos]
            assert piece is None
            card_type = Unit if card.type is CardType.UNIT else Structure
            piece = card_type(card, player, self, pos)
            self.board[pos] = piece
            if piece.type is CardType.UNIT:
                piece.play(being_placed=True)

        if not card.token and not card in hand:
            deck.append(WeightedCard(card))

    def force_play_card(self, card: Card, pos: tuple = None):
        player = self.current_player
        if card.type == CardType.SPELL:
            Spell(card, player, self, pos).play()
        else:
            assert pos is not None
            piece = self.board[pos]
            assert piece is None
            card_type = Unit if card.type is CardType.UNIT else Structure
            piece = card_type(card, player, self, pos)
            self.board[pos] = piece
            if piece.type is CardType.UNIT:
                piece.play(being_placed=True)

    TargetT = Any
    def apply_damage(self, targets: Any, amount: Any, cause):
        if not isinstance(targets, Iterable):
            targets = (targets, )

        if isinstance(amount, list):
            assert len(targets) == len(amount)
            targets_and_damage = list(zip(targets, amount))
        else:
            targets_and_damage = list((target, amount) for target in targets)
        
        bases = tuple(tardmg for tardmg in targets_and_damage if isinstance(tardmg[0], Player))
        assert len(bases) <= 1, 'Damage to multiple bases in single attack'
        if len(bases) == 1:
            player, damage = bases[0]
            player.base_health -= damage
            if player.base_health <= 0:
                raise Exception('Game ended')
            targets_and_damage.remove(bases[0])

        targets_and_damage: list[tuple[BoardPiece, int]]
        board = self.board
        pos_map = {
            piece.pos: piece 
            for piece_and_dmg in targets_and_damage 
            if (piece := piece_and_dmg[0])}

        # apply damage to targets
        for piece, damage in targets_and_damage:
            piece._strength -= damage
            if piece.strength <= 0:
                board[piece.pos] = None

        # iterate positions in reverse order and trigger abilities
        for pos in board.pos_iter(reversed=not board.reversed):
            piece = pos_map.get(pos)
            if piece is not None:
                ability = piece.ability
                if ability is not None:
                    if ((piece.strength > 0 and ability.trigger is Event.AFTER_SURVIVING_DAMAGE) or
                        (piece.strength <= 0 and ability.trigger is Event.ON_DEATH)):
                        piece.trigger_ability(cause=cause)
    

    @staticmethod
    def random_weight_deck(deck: list):
        deck = deck.copy()
        random.shuffle(deck)
        weights = [220, 137, 85, 53, 33, 20, 12, 7, 4, 2, 1, 0]
        return [WeightedCard(card, weight) for card, weight in zip(deck, weights)]


    def board_to_str(self):
        board = self.board
        board_str = ''
        frontline = self.current_frontline
        if self.is_player1_turn():
            frontline -= 1
        for r in range(board.rows):
            row = ''
            for c in range(board.cols): 
                piece = board[(r, c)]
                id = f'{piece.owner.id}.{piece.card.id}' if piece is not None else ''
                row = f'{row}{id.center(7)}'
                if r == frontline:
                    row = row.replace(' ', '_')
            board_str += f'{row}'
            if r != board.rows - 1:
                board_str += '\n'
        return board_str


    def print_board(self):
        print(f'\nBoard - turn: {self.turn_count}, player: {self.current_player.id}')
        print(self.board_to_str())


def _load_cards(*, debug = False):
    global CARDS
    if CARDS != {}: return
    import json
    import pickle
    from pathlib import Path

    cards_json_path = Path(r"stormbot/cards/custom_cards.json")
    cards_parsed_path = Path(r"stormbot/cards/custom_cards_parsed.pickle")
    if not debug and cards_parsed_path.is_file():
        with open(cards_parsed_path, 'rb') as cards_data:
            CARDS = pickle.load(cards_data)
    else:
        with open(cards_json_path) as cards_json:
            CARDS = { 
                card.id: card
                for card_json in json.load(cards_json)
                if (card := Card.from_json(card_json))
            }
        with open(cards_parsed_path, 'wb') as cards_data:
            pickle.dump(CARDS, cards_data)

    if Game._ABILITIES_ENABLED:
        import game.harcoded_abilities
        for card in CARDS.values():
            if card.id in _implemented:
                card.ability = eval(f"harcoded_abilities._{card.id}_Ability()")
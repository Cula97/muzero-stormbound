from dataclasses import dataclass
from stormbot.game.enums import CardType, UnitType, Faction, Rarity
from stormbot.game.ability import Ability

class Card:
    def __init__(self, id: str, name: str, type: CardType, level: int, faction: Faction, 
            mana: int, rarity: Rarity = None, strength: list = None, movement = 0, 
            unit_types: tuple = (), token = False, *, ability = None):
        self.id = id
        self.name = name
        self.type = type
        self.level = level
        self.faction = faction
        self.mana = mana
        self.token = token
        if rarity is not None:
            self.rarity = rarity
        if type is CardType.UNIT:
            self.unit_types = unit_types
            self.movement = movement
            self.status = {}
        if type in (CardType.UNIT, CardType.STRUCTURE):
            assert strength is not None
            self._strength = strength
        self.ability: Ability = ability

    @property
    def strength(self):
        return self._strength[self.level-1]

    def __str__(self):
        return f"({self.name}, {self.type.name})"

    @classmethod
    def from_json(cls, json: dict, level = 1):
        attrs = {
            'id': json['id'], 
            'name': json['name'],
            'type': eval(f"CardType.{json['type'].upper()}"),
            'level': level, 
            'faction': eval(f"Faction.{json['faction'].upper()}"),
            'mana': json['mana'],
            'rarity': json.get('rarity'),
            'strength': json.get('strength'),
            'movement': json.get('movement'),
            'unit_types': json.get('unitTypes'),
            'token': json.get('token')
        }

        if attrs['rarity'] is not None:
            attrs['rarity'] = eval(f"Rarity.{attrs['rarity'].upper()}")
        if attrs['strength'] is not None:
            attrs['strength'] = [int(value) for value in attrs['strength'].split('/')]
        unit_types = attrs['unit_types']
        if unit_types is not None:
            attrs['unit_types'] = tuple(
                eval(f"UnitType.{unit_type.upper()}") 
                for unit_type in unit_types)
        
        return cls(**attrs)


@dataclass
class WeightedCard:
    card: Card
    weight: int = 1
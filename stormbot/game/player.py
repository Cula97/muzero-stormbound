from dataclasses import dataclass
from stormbot.game.card import Card, WeightedCard

@dataclass
class PlayerData:
    def __init__(self, id, deck):
        self.id: str = id
        self.deck: list = deck
        self.base_health = 12

@dataclass
class Player:
    def __init__(self, player_data: PlayerData, deck: list, mana = 0):
        self.data = player_data
        self.id = player_data.id
        self.base_health = player_data.base_health
        self.mana = mana
        self.deck = deck
        self.hand: list = []


from abc import ABC, abstractmethod

from stormbot.game.enums import Event


class Ability(ABC): 
    def __init__(self):
        self.trigger: Event
        self.text: str

    @abstractmethod
    def activate(self, *, origin, **kwargs):
        raise NotImplemented
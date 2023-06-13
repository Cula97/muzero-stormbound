from enum import Enum

class GameEnum(Enum):
    def __str__(self):
        return self.name
        return f"{self.__class__.__name__}.{self.name}"

    def __repr__(self):
        # return self.name
        return f"{self.__class__.__name__}.{self.name}"

class Event(GameEnum):
    ON_PLAY                = 1
    ON_TURN_START          = 2
    BEFORE_MOVE            = 3
    BEFORE_ATTACK          = 4
    AFTER_ATTACK           = 5
    AFTER_SURVIVING_DAMAGE = 6
    ON_DEATH               = 7

class Operation(GameEnum):
    NON      = 1
    NO       = 2
    EQUAL    = 3
    AND      = 4
    ANY      = 6
    ALL      = 5
    UNIQUE   = 7
    AT_LEAST = 8
    BOTH     = 9
    RANDOM   = 10 
    IF       = 11
    ELSE     = 12
    SPREAD   = 13
    FOR_EACH = 14
    OTHER    = 15
    COPY     = 16

class Stat(GameEnum):
    STRENGTH = 1
    MANA     = 2
    SPEED    = 3

class Position(GameEnum):
    BORDERING   = 1
    SURROUNDING = 2
    FRONT       = 3
    SIDE        = 4
    BEHIND      = 5
    SAME_ROW    = 6

class CardType(GameEnum):
    UNIT      = 1
    STRUCTURE = 2
    SPELL     = 3

class UnitType(GameEnum):
    ANCIENT   = 1
    CONSTRUCT = 2
    FELINE    = 3
    KNIGHT    = 4
    SATYR     = 5
    PIRATE    = 6
    DRAGON    = 7
    HERO      = 8
    FROSTLING = 9
    ELDER     = 10
    RAVEN     = 11
    UNDEAD    = 12
    RODENT    = 13
    DWARF     = 14
    TOAD      = 15

class StructureType(GameEnum):
    TEMPLE = 1

class Team(GameEnum):
    FRIENDLY = 1
    ENEMY    = 2

class Faction(GameEnum):
    NEUTRAL   = 1
    IRONCLAD  = 2
    SHADOWFEN = 3
    SWARM     = 4
    WINTER    = 5

class Rarity(GameEnum):
    COMMON    = 1
    RARE      = 2
    EPIC      = 3
    LEGENDARY = 4

class Status(GameEnum):
    CONFUSED        = 1
    DISABLED        = 2
    FROZEN          = 3
    POISONED        = 4
    VITALIZED       = 5
    FIXEDLY_FORWARD = 6

class Health(GameEnum):
    DEAD     = 1
    STRONGER = 2
    WEAKER   = 3

class Hand(GameEnum):
    LAST_CARD = 1
    FULL_HAND = 2

class Meta(GameEnum):
    ABILITY   = 1
    EFFECT    = 2
    PLAYER    = 3
    TURN      = 4
    VALUE     = 5
    UNIT_TYPE = 6

class GameAction(GameEnum):
    pass

class TileAction(GameAction):
    BUILD    = 1
    FLY      = 2
    JUMP     = 3
    PLAY     = 5
    RESPAWN  = 6
    SPAWN    = 7
    TELEPORT = 8

class PieceAction(GameAction):
    COMMAND   = 1
    CONFUSE   = 2
    CONVERT   = 3
    DEAL      = 4
    DECONFUSE = 5
    DESTROY   = 6
    DISABLE   = 7
    DRAIN     = 8
    FORCE     = 9
    FREEZE    = 10
    GAIN      = 11
    GET       = 12 
    GIVE      = 13
    MAKE      = 14
    MOVE      = 15
    POISON    = 16
    PULL      = 17
    PUSH      = 18
    REDUCE    = 19
    SET       = 20
    SPLIT     = 21
    STEAL     = 22
    VITALIZE  = 23
    TRIGGER   = 24

class HandAction(GameAction):
    CREATE  = 1 
    DISCARD = 2
    DRAW    = 3
    LOSE    = 4 
    REPLACE = 5 
    RETURN  = 6
    SPEND   = 7 
    
class Target(GameEnum):
    BASE        = 1
    ITSELF      = 2 
    TILE        = 3 
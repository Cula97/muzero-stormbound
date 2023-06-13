from typing import Callable, Generic, TypeVar

T = TypeVar('T')
class Board(Generic[T]):
    def __init__(self, rows: int, cols: int, reversed: Callable[[], bool] = lambda: False):
        self._rows = rows
        self._cols = cols
        self._reversed = reversed
        self.data = [[None] * cols for _ in range(rows)]

    @property
    def rows(self): 
        return self._rows
    
    @property
    def cols(self): 
        return self._cols

    @property
    def reversed(self):
        return self._reversed()

    def pos_iter(self, *, reversed: bool = None):
        if reversed is None: 
            reversed = self.reversed
        """Itearate over positions"""
        rows = range(self._rows) if not reversed else range(self._rows-1, -1, -1)
        cols = range(self._cols) if not reversed else range(self._cols-1, -1, -1)
        return ((r, c) for r in rows for c in cols)
    
    def __getitem__(self, position: tuple):
        row, col = position
        return self.data[row][col]

    def __setitem__(self, position: tuple, value: T):
        row, col = position
        self.data[row][col] = value

    def __delitem__(self, position: tuple):
        row, col = position
        self.data[row][col] = None

    def __iter__(self):
        """Iterate over board items, order follows self.reversed"""
        for pos in self.pos_iter(reversed=self.reversed):
            item = self[pos]
            if item is not None:
                yield item

    def __reversed__(self):
        """Iterate over board items, order is opposite to self.reversed"""
        for pos in self.pos_iter(reversed=not self.reversed):
            item = self[pos]
            if item is not None:
                yield item
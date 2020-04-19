# solving Riddler Classic @ https://fivethirtyeight.com/features/can-you-solve-the-chess-mystery/

import numpy as np


class PeriodicGridConway:
    """
    Conway's game of life on a finite grid that has periodic boundary conditions,
    so that cells in the first row are neighbors with cells in the last row,
    and cells in the first column are neighbors with cells in the last column.
    """
    __MAP__ = {  # chars for printing
        0: '.',
        1: u'\u2588'  # full block
    }

    def __init__(self, m, n):
        """
        init an empty grid
        :param m: nr. of rows
        :param n: nr. of columns
        """
        assert m > 0 and n > 0

        # init dimensions and grid
        self.m = m
        self.n = n
        self.cell_count = m * n
        self.grid = np.zeros(shape=(m, n), dtype=int)

        # for efficiency, pre-store the coordinates of the neighbors of each cell, considering grid periodicity
        self.__neighbors_grid = []
        for i in range(m):
            neighbors_row = []
            for j in range(n):
                neighbors = []
                for x in range(i - 1, i + 2):
                    for y in range(j - 1, j + 2):
                        if x == i and y == j:
                            continue
                        neighbors.append((x % self.m, y % self.n))
                neighbors_row.append(neighbors)
            self.__neighbors_grid.append(neighbors_row)

    def __str__(self):
        """
        user-friendly string representation for multi-row printing
        """
        return '\n'.join([' '.join([self.__MAP__[self.grid[(i, j)]] for j in range(self.n)]) for i in range(self.m)])

    def __getitem__(self, key):
        """
        get the value of the grid cell at position key
        :param key: 1-d or 2-d index to desired cell
        :return: cell value, 0 or 1
        """
        return self.grid[self.__interpret_key(key)]

    def __setitem__(self, key, value):
        """
        set the value of the grid cell at position key
        :param key: 1-d or 2-d index to desired cell
        :param value: 0 or 1
        :return: cell value, 0 or 1
        """
        assert value in (0, 1)
        self.grid[self.__interpret_key(key)] = value

    def __index2key(self, index):
        """
        map a 1-d index in 0..2**(m*n)-1 to (i,j) with i in 0..m-1 and j in 0..n-1
        :param index: ordinal index, scanning first rows from top to bottom and then columns from left to right
        :return: tuple (i, j)
        """
        return index // self.n, index % self.n

    def __interpret_key(self, key):
        """
        get a 2-d key from a 1-d or 2-d key
        :param key: 1-d or 2-d key to a grid position
        :return: 2-d key
        """
        if isinstance(key, int):
            return self.__index2key(key)
        if len(key) == 1:
            return self.__index2key(key[0])
        return key

    def count_neighbors(self, key):
        """
        count the live neighbors of cell in position key, based on pre-computed periodic topology
        :param key: 1-d or 2-d index to cell position
        :return: nr. of live neighbors
        """
        key0, key1 = self.__interpret_key(key)
        count = 0
        for neighbor_key in self.__neighbors_grid[key0][key1]:
            count += self.grid[neighbor_key]
        return count

    def set_key(self, key, value=1):
        """
        set cell in position key to value
        :param key: 1-d or 2-d index to cell position
        :param value: desired value, 0 or 1
        :return: nothing
        """
        self[key] = value

    def set_neighbors(self, key, value=1):
        """
        set all neighbors of cell in position key to value
        :param key: 1-d or 2-d index to cell position
        :param value: desired value, 0 or 1
        :return: nothing
        """
        key0, key1 = self.__interpret_key(key)
        for coord in self.__neighbors_grid[key0][key1]:
            self[coord] = value

    def evolve(self):
        """
        let the grid evolve_n_ticks for 1 tick
        :return: nothing
        """
        new_grid = np.zeros(shape=(self.m, self.n), dtype=int)
        for i in range(self.m):
            for j in range(self.n):
                n_neighbors = self.count_neighbors((i, j))
                alive = self.grid[(i, j)] == 1
                if n_neighbors == 3 or n_neighbors == 2 and alive:
                    new_grid[(i, j)] = 1
        self.grid = new_grid

    def evolve_n_ticks(self, n_ticks, func=lambda c: {print(c, '\n')}):
        """
        let the grid evolve for n ticks
        :param n_ticks: nr. of ticks
        :param func: function to be called on self for every grid state, if not None
        :return: nothing
        """
        if func: func(self)
        for t in range(n_ticks):
            self.evolve()
            if func: func(self)

    def evolve_fully(self, func=lambda c: {print(c, '\n')}):
        """
        let the grid evolve till a previous state of the game is repeated
        :param func: function to be called on self for every grid state, if not None
        :return: nothing
        """
        prev_states = set()
        curr_state = self.to_compact()
        sequence = [curr_state]
        if func: func(self)
        while curr_state not in prev_states:
            prev_states.add(curr_state)
            self.evolve()
            curr_state = self.to_compact()
            sequence.append(curr_state)
            if func: func(self)
        return sequence

    def to_compact(self):
        """
        hash the grid state as a unique integer in 0 .. (2 ** cell_count) - 1
        such that there is a 1-to-1 correspondence between grid states and integer hashes
        :return: unique integer hash
        """
        str_rep = ''
        for i in range(self.m):
            for j in range(self.n):
                str_rep += str(self.grid[(i, j)])
        return int(str_rep, 2)

    def from_compact(self, compact):
        """
        set the grid to the state identified by a unique integer hash
        :param compact: unique integer hash
        :return: nothing
        """
        str_rep = format(compact, '%db' % (self.cell_count))
        assert len(str_rep) <= self.m * self.n
        self.grid = np.zeros(shape=(self.m, self.n), dtype=int)
        for k, bit in enumerate(str_rep):
            if bit == '1':
                self.grid[self.__interpret_key(k)] = 1

    def eventual_period(self):
        """
        Compute the eventual period of the current grid state, i.e. let the gird evolve till one state is repeated,
        then check how many ticks it takes for the final state to repeat itself.
        After calling this method, the grid state is left in an oscillator.
        :return: period, an integer
        """
        sequence = self.evolve_fully(func=None)
        last_state = sequence[-1]
        for period, state in enumerate(reversed(sequence[:-1]), 1):
            if state == last_state:
                break
        return period

    def grid_size_has_oscillator_brute(self):
        """
        Check if this grid size supports any oscillator by brute force,
        i.e. checking the eventual period of every possible initial state, with no optimization.
        After calling this method, if the return value is True, the grid state is left in an oscillator.
        On a finite periodic grid, there are at most 2 ** cell_count states, so states are going to repeat eventually,
        either because a stable state is reached (period = 1), or because an oscillator is reached (period > 1).
        :return: True or False
        """
        for compact in range(2 ** self.cell_count):
            self.from_compact(compact)
            period = self.eventual_period()
            if period > 1:
                return True
        return False


# increase the nr. of columns n until an oscillator is found
n = 0
while True:
    n += 1
    c = PeriodicGridConway(3, n)
    print('Does the 3 x %d periodic grid have any oscillators?' % (n))
    if c.grid_size_has_oscillator_brute():
        print('Yes: for example, this oscillator of period %d.' % (c.eventual_period()))
        c.evolve_fully()  # show the oscillator cycle
        break
    else:
        print('No.\n')

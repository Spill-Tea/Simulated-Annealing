# MIT License

# Copyright (c) 2023 Spill-Tea

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
    Annealing/anneal.py

"""
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from logging import Logger
from typing import List, Optional

import numpy as np

from .cooling import Cooling
from .fitness import Fitness

_log = Logger(__name__, 10)


@dataclass
class Sample:
    iteration: int
    order: np.ndarray
    perf: float


def stochastic(options: int, size: int):
    return np.random.choice(options, size=size, replace=False)


def swap(array):
    """Stochastically Swaps two indices of an array, inplace.

    Note:
        For Potential Asymetric Swapping, call this function more
        than once, on the same array.

    """
    idx1, idx2 = np.random.choice(len(array), 2, replace=False)
    array[idx1], array[idx2] = array[idx2], array[idx1]


def n_opt(array: np.ndarray, n: int = 2):
    """Stochastically Swaps inplace any N indices within an array"""
    index = np.random.choice(len(array), n, replace=False)
    array[index] = array[np.roll(index, 1)]


class AnnealingBase(ABC):
    """Abstract Simulated Annealing Base Class

    Args:
        data (np.ndarray): Data used to simulate
        chill (Cooling): Defines temperature cooling function
        fitness (Fitness): Define Optimization Metrics
        log (Logger): Logger for debugging purposes

        tm (float): Current Temperature
        history (List[Sample]): History of saved best performing added to during simulation.

    Notes:
        1. In the Traveling Salesman Problem (TSP), we are aiming to
        optimize the sorting order of a fixed number of data points
        contained by self.data. self.data in this case will resemble
        an array of [[x1, y1], [x2, y2]] or [[x1, y1, z1], [x2, y2, z2]]
        coordinates.

        2. In a subsample problem, we are optimizing the selection of
        k elements from self.data that performs best. self.data in this
        case will resemble a 2d distance matrix.

    """
    def __init__(self,
                 data: np.ndarray,
                 chill: Cooling,
                 fitness: Fitness,
                 log: Logger = _log,
                 ) -> None:
        self.data = data
        self.chill = chill
        self.fitness = fitness
        self.log = log

        self.tm = self.chill.tm_max
        self.history: List[Sample] = []

    @property
    def steps(self):
        return self.chill.steps

    @abstractmethod
    def mixing(self, index, n: int):
        """Defines how we shuffle or select next indices."""
        n_opt(index, np.random.randint(2, n + 1))

    @abstractmethod
    def subsample(self, indices: np.ndarray) -> np.ndarray:
        """Slices a Subsample of the Larger Dataset.

        Args:
            idices (np.ndarray[int]): Array of Row Indices

        Returns:
            (np.ndarray)

        """
        return self.data[indices]

    def nucleate(self, k: Optional[int] = None):
        """Initialize Iterative Selection Process."""
        total = len(self.data)
        k = k or total
        index = stochastic(total, k)
        return self.subsample(index)

    def simulate(self, k: Optional[int] = None, nshuffle: int = 3):
        """Simulate Annealing.

        Args:
            k (int): Choose k, Optional
            nshuffle (int): Maximum Number of times to swap Indices

        Returns:
            (np.ndarray) Best Performing Data

        """
        # Reset Tm
        self.tm = self.chill.tm_max
        data = self.data if k is None else self.nucleate(k)
        index = np.arange(len(data))
        best_index = np.copy(index)
        best = self.fitness(data)
        for j in range(self.steps):
            self.mixing(index, nshuffle)
            array = self.subsample(index)
            current = self.fitness(array)
            self.tm = self.chill(j)

            if (current <= best) or (self.tm > np.random.random(1) * self.chill.tm_max):
                self.log.debug("Iteration %d: Performance (%.4f)", j, current)
                best = current
                best_index = np.copy(index)
                data = np.copy(array)
                self.history.append(Sample(iteration=j, order=data, perf=best))
            else:
                index = np.copy(best_index)

        return data

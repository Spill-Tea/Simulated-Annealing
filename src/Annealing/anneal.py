# MIT License
#
# Copyright (c) 2023 Spill-Tea
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Annealing/anneal.py."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
from math import exp

import numpy as np

from .cooling import Cooling
from .fitness import Fitness


_log: Logger = Logger(__name__, 10)


@dataclass
class Sample:
    """Simulated sample."""

    iteration: int
    tm: float
    perf: float
    better: bool
    order: np.ndarray


def stochastic(options: int, size: int) -> np.ndarray:
    """Stochastic choice (subsample) of options without duplicates."""
    return np.random.choice(options, size=size, replace=False)


def swap(array: np.ndarray, n: int = 2) -> None:
    """Stochastically swaps inplace any N indices within an array."""
    index: np.ndarray = stochastic(len(array), n)
    array[index] = array[np.roll(index, 1)]


def probability(difference: float, temperature: float) -> float:
    """Calculates the Probability scaled by the difference and current temperature."""
    return exp(-difference / temperature)


class AnnealingBase(ABC):
    """Abstract Simulated Annealing Base Class.

    Args:
        data (np.ndarray): Data used to simulate
        chill (Cooling): Defines temperature cooling function
        fitness (Fitness): Define Optimization Metrics
        log (Logger): Logger for debugging purposes

    Attributes:
        tm (float): Current temperature.
        history (list[Sample]): History of best performing added to during simulation.

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

    data: np.ndarray
    chill: Cooling
    fitness: Fitness
    log: Logger
    tm: float
    history: list[Sample]

    def __init__(
        self,
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
        self.history: list[Sample] = []

    @property
    def steps(self) -> int:
        """Number of steps (iterations)."""
        return self.chill.steps

    @property
    def best(self) -> Sample:
        """Best performing sample found."""
        return min(self.history, key=lambda x: x.perf)

    @abstractmethod
    def mixing(self, index: np.ndarray, n: int) -> None:
        """Defines how we shuffle or select next indices."""
        swap(index, np.random.randint(2, n + 1))

    @abstractmethod
    def subsample(self, indices: np.ndarray) -> np.ndarray:
        """Slices a subsample of the larger dataset.

        Args:
            indices (np.ndarray[int]): Array of row indices.

        Returns:
            (np.ndarray) subsample of data (in provided order).

        """
        return self.data[indices]

    def nucleate(self, k: int | None = None) -> np.ndarray:
        """Initialize iterative selection process."""
        total: int = len(self.data)
        k = k or total
        index: np.ndarray = stochastic(total, k)

        return self.subsample(index)

    def simulate(self, k: int | None = None, nswaps: int = 3) -> np.ndarray:
        """Simulate annealing.

        Args:
            k (int): Choose k or default to length of data.
            nswaps (int): Maximum Number of indices to swap.

        Returns:
            (np.ndarray) Best performing data found.

        """
        # Reset Tm
        self.tm = self.chill.tm_max
        data: np.ndarray = self.data if k is None else self.nucleate(k)
        length: int = len(data)
        if not (2 <= nswaps <= length):
            nswaps = max(2, min(nswaps, length))
            self.log.info("Setting nswaps argument to: %d", nswaps)

        index: np.ndarray = np.arange(length)
        best_index: np.ndarray = np.copy(index)
        best: float = self.fitness(data)
        for j in range(self.steps):
            self.mixing(index, nswaps)
            array: np.ndarray = self.subsample(index)
            current: float = self.fitness(array)
            self.tm = self.chill(j)
            delta: float = current - best
            test: bool = delta <= 0

            if test or probability(delta, self.tm) > np.random.random(1):
                self.log.debug("Iteration %d: Performance (%.4f)", j, current)
                best = current
                best_index = np.copy(index)
                data = np.copy(array)
                self.history.append(
                    Sample(iteration=j, tm=self.tm, perf=best, better=test, order=data)
                )
            else:
                index = np.copy(best_index)

        return data

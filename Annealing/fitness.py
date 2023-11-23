"""
    fitness.py

"""
from abc import ABC
from abc import abstractmethod

import numpy as np


class Fitness(ABC):
    @abstractmethod
    def performance(self, data: np.ndarray) -> float:
        """Calculates Performance Metric from given Data."""
        ...

    def __call__(self, data: np.ndarray) -> float:
        return self.performance(data)


class LinearEuclidean(Fitness):
    """Linear (One Way) Euclidean Distance"""
    def _performance(self, dx: np.ndarray) -> float:
        ss = np.sum(dx * dx, axis=1)
        return np.sqrt(ss).sum()

    def performance(self, data: np.ndarray) -> float:
        return self._performance(np.diff(data, axis=0))


class CircularEuclidean(LinearEuclidean):
    """Circular (Round Trip) Euclidean Distance"""
    def performance(self, data: np.ndarray) -> float:
        dx = data - np.roll(data, 1, 0)
        return self._performance(dx)

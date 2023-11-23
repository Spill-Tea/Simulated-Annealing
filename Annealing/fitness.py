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

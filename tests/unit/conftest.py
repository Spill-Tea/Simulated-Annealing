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

"""unit test utilities."""

import numpy as np
import pytest

from Annealing import anneal
from Annealing.cooling import InverseCooling
from Annealing.fitness import CircularEuclidean


@pytest.fixture
def _tsp() -> type[anneal.AnnealingBase]:
    """tsp annealing base."""

    class TSP(anneal.AnnealingBase):
        def mixing(self, index, nshuffle):
            super().mixing(index, nshuffle)

        def subsample(self, indices: np.ndarray) -> np.ndarray:
            return super().subsample(indices)

    return TSP


@pytest.fixture
def coord() -> np.ndarray:
    """random 3d coordinates."""
    seed = np.random.default_rng(23)
    coordinates = seed.uniform(0.0, 100.0, (50, 3))

    return coordinates


@pytest.fixture
def tsp(
    coord: np.ndarray,
    _tsp: type[anneal.AnnealingBase],
) -> anneal.AnnealingBase:
    """Instance of TSP annealing base."""
    base: anneal.AnnealingBase = _tsp(
        coord,
        InverseCooling(1000, 0.9),
        CircularEuclidean(),
    )

    return base

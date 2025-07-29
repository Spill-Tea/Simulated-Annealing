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

"""unit test annealing interface and peripheral functions."""

import numpy as np
import pytest

from Annealing import anneal
from Annealing.cooling import InverseCooling
from Annealing.fitness import CircularEuclidean


@pytest.fixture
def tsp() -> type[anneal.AnnealingBase]:
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
def tsp_base(
    coord: np.ndarray,
    tsp: type[anneal.AnnealingBase],
) -> anneal.AnnealingBase:
    """Instance of TSP annealing base."""
    base: anneal.AnnealingBase = tsp(
        coord,
        InverseCooling(1000, 0.9),
        CircularEuclidean(),
    )

    return base


def test_stochastic() -> None:
    """unit test stochastic sample."""
    expect = 5
    maximum = 100
    result = anneal.stochastic(maximum, expect)

    assert isinstance(result, np.ndarray), "Expected an array."
    assert len(result) == expect, "Expected 5 elements within array."
    assert np.count_nonzero(result < maximum) == expect, (
        "Expected all elements to be below threshold."
    )
    assert len(set(result.tolist())) == expect, "Expected all 5 elements to be unique."


@pytest.mark.parametrize(
    ["n"],
    [(i,) for i in range(2, 11)],
)
def test_swap(coord: np.ndarray, n: int) -> None:
    """Test coordinate swapping acts in place correctly."""
    result: np.ndarray = coord.copy()
    anneal.swap(result, n)
    assert isinstance(result, np.ndarray), "Expected an array."
    assert result.shape == coord.shape, "Expected same shape as input array."
    assert not np.all(result == coord), "Expected a different output array."

    indices: np.ndarray = np.argwhere(result != coord)
    assert len(np.unique(indices[:, 0])) == n, f"Expected {n} elements to be swapped."


@pytest.mark.parametrize(
    ["diff", "temp", "expected"],
    [
        (-5, 45, 1.117519),
        (5, 45, 0.8948393),
        (-100, 99, 2.745878),
        (-100, 1, 2.688117e43),
        (100, 1, 3.72e-44),
    ],
)
def test_probability(diff: float, temp: float, expected: float) -> None:
    """Test stochastic probability of accepting a worse result scaling."""
    result = anneal.probability(diff, temp)
    assert np.isclose(result, expected)


def test_annealing_nucleate(
    tsp_base: anneal.AnnealingBase,
) -> None:
    """Test a simulation of TSP annealing."""
    base: anneal.AnnealingBase = tsp_base
    assert isinstance(base, anneal.AnnealingBase), "Expected subclass instance."
    assert len(base.history) == 0, "Expected no elements"

    result = base.simulate(len(base.data))
    assert isinstance(result, np.ndarray), "Expected an array."
    assert len(result) == len(base.data), "Expected same size array."


def test_annealing(
    tsp_base: anneal.AnnealingBase,
) -> None:
    """Test annealing without nucleation to more reliably measure fitness improves."""
    base: anneal.AnnealingBase = tsp_base
    result = base.simulate()

    assert len(base.history) > 0, "Expected to have results saved in history."
    assert base.fitness(result) < base.fitness(base.data), (
        "Expected improvement of coordinate order."
    )
    assert base.fitness(base.best.order) < base.fitness(base.data), (
        "Expected improvement of coordinate order."
    )


def test_annealing_low_nswaps(
    tsp_base: anneal.AnnealingBase,
) -> None:
    """Test annealing simulation setting low nswaps."""
    base: anneal.AnnealingBase = tsp_base
    result = base.simulate(nswaps=1)

    assert isinstance(result, np.ndarray), "Expected an array."

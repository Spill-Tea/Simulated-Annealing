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

"""unit test strategies."""

import numpy as np
import pytest

from Annealing import strategies


@pytest.fixture
def dm(coord: np.ndarray) -> np.ndarray:
    """2d distance matrix of sample coordinates."""
    return strategies.build_2d_distance_matrix(coord)


@pytest.fixture
def tour_nn() -> list[int]:
    return [
        8,
        29,
        24,
        18,
        38,
        25,
        41,
        30,
        49,
        2,
        45,
        44,
        19,
        3,
        37,
        13,
        28,
        6,
        0,
        21,
        11,
        20,
        27,
        14,
        40,
        7,
        48,
        42,
        33,
        10,
        15,
        35,
        17,
        23,
        43,
        22,
        39,
        9,
        12,
        46,
        26,
        5,
        36,
        1,
        47,
        32,
        16,
        4,
        34,
        31,
    ]


@pytest.fixture
def tour_2opt() -> list[int]:
    return [
        0,
        21,
        11,
        20,
        33,
        27,
        14,
        24,
        29,
        8,
        38,
        18,
        31,
        25,
        41,
        30,
        49,
        2,
        45,
        34,
        44,
        19,
        3,
        37,
        16,
        4,
        39,
        22,
        47,
        32,
        1,
        36,
        9,
        12,
        46,
        26,
        5,
        7,
        40,
        48,
        42,
        10,
        15,
        35,
        17,
        23,
        43,
        13,
        28,
        6,
    ]


@pytest.fixture
def tour_3opt() -> list[int]:
    return [
        0,
        6,
        21,
        11,
        31,
        3,
        37,
        28,
        13,
        43,
        23,
        17,
        35,
        15,
        10,
        42,
        48,
        40,
        7,
        33,
        20,
        27,
        14,
        24,
        8,
        29,
        18,
        38,
        25,
        41,
        30,
        2,
        46,
        12,
        26,
        5,
        9,
        36,
        1,
        32,
        47,
        22,
        39,
        16,
        4,
        34,
        19,
        44,
        45,
        49,
    ]


@pytest.mark.parametrize(
    ["tsp_strategy", "expected_tour", "expected_cost"],
    [
        (strategies.NearestNeighborStrategy, "tour_nn", 1149.5188),
        (strategies.TwoOptStrategy, "tour_2opt", 1119.0572),
        (strategies.ThreeOptStrategy, "tour_3opt", 1200.9199),
    ],
)
def test_tsp_strategies(
    dm: np.ndarray,
    tsp_strategy: type[strategies.TSPStrategy],
    expected_tour: list[int],
    expected_cost: float,
    request: pytest.FixtureRequest,
):
    """Test various tsp strategies produce expected tours and costs."""
    strategy = tsp_strategy(dm)
    tour, val = strategy.minimize()
    expected = request.getfixturevalue(expected_tour)

    assert isinstance(tour, list), "Expected a list result."
    assert tour == expected, "Expected predictable tour."
    assert len(tour) == len(dm), "Expected same number of data points."

    assert isinstance(val, float), "Expected a float result."
    assert np.isclose(val, expected_cost), "Unexpected tour length calculation."

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

"""unit test fitness calculations."""

import numpy as np
import pytest

from Annealing import fitness


@pytest.fixture
def coordinates_2d() -> np.ndarray:
    """Example 3d coordinates."""
    return np.asarray([[1, 2], [4, 6]])


@pytest.fixture
def coordinates_3d() -> np.ndarray:
    """Example 3d coordinates."""
    return np.asarray([[1, 2, 2], [1, 2, 6]])


@pytest.mark.parametrize(
    ["cls", "name", "expected"],
    [
        (fitness.LinearEuclidean, "coordinates_2d", 5.0),
        (fitness.CircularEuclidean, "coordinates_2d", 10.0),
        (fitness.LinearEuclidean, "coordinates_3d", 4.0),
        (fitness.CircularEuclidean, "coordinates_3d", 8.0),
    ],
)
def test_linear(
    name: str,
    cls: type[fitness.Fitness],
    expected: float,
    request: pytest.FixtureRequest,
) -> None:
    """Test linear Euclidean"""
    fit: fitness.Fitness = cls()
    coordinates: np.ndarray = request.getfixturevalue(name)
    result: float = fit.performance(coordinates)

    assert result == expected, "Unexpected Linear Euclidean"

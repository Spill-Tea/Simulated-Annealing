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
"""unit test cooling paradigms."""

import numpy as np
import pytest

from Annealing import cooling


@pytest.fixture
def options() -> dict:
    return dict(steps=1000, alpha=0.8, tm_min=0, tm_max=100)


@pytest.mark.parametrize(
    ["cooler", "step", "expected"],
    [
        (cooling.LinearCooling, 0, 100.0),
        (cooling.LinearCooling, 100, 90.0),
        (cooling.LinearCooling, 1000, 0.0),
        (cooling.InverseCooling, 0, 100.0),
        (cooling.InverseCooling, 150, 0.826446),
        (cooling.InverseCooling, 1000, 0.124843),
        (cooling.QuadraticCooling, 0, 100.0),
        (cooling.QuadraticCooling, 225, 60.0625),
        (cooling.QuadraticCooling, 1000, 0.0),
        (cooling.ExponentialCooling, 0, 100.0),
        (cooling.ExponentialCooling, 995, 3.75e-95),
        (cooling.ExponentialCooling, 1000, 0.0),
        (cooling.SqExponentialCooling, 0, 100.0),
        (cooling.SqExponentialCooling, 515, 1.52e-98),
        (cooling.SqExponentialCooling, 1000, 0.0),
        (cooling.ExponentialQuadCooling, 0, 100.0),
        (cooling.ExponentialQuadCooling, 725, 3.78125),
        (cooling.ExponentialQuadCooling, 1000, 0.0),
    ],
)
def test_cooling_paradigms(
    options,
    cooler: type[cooling.Cooling],
    step: int,
    expected: float,
) -> None:
    cool: cooling.Cooling = cooler(**options)
    result = cool(step)
    assert np.isclose(result, expected), f"Unexpected {cooler.__name__}."

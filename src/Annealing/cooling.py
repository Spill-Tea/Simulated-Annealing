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

"""Annealing/cooling.py."""

from abc import ABC, abstractmethod
from math import sqrt


class Cooling(ABC):
    """Interface Class to define temperature cooling functions."""

    steps: int
    alpha: float
    tm_min: float
    tm_max: float

    def __init__(
        self,
        steps: int = 1_000,
        alpha: float = 0.922,
        tm_min: float = 0.0,
        tm_max: float = 100.0,
    ) -> None:
        self.steps = steps
        self.alpha = alpha
        self.tm_min = tm_min
        self.tm_max = tm_max

    @abstractmethod
    def cool(self, step: int) -> float: ...

    def __call__(self, step: int) -> float:
        return self.cool(step)


# Temperature Cooling Functions
def inverse_cooling(step_n: int, tm_max: float, alpha: float) -> float:
    """T(k) = Tmax / (1. + alpha * k)."""
    return tm_max / (1.0 + alpha * step_n)


class InverseCooling(Cooling):
    """Inverse Cooling method which scales by the reciprocal of k.

    Equation:
        T(k) = Tmax / (1 + alpha * k)

    """

    def cool(self, step: int) -> float:
        return inverse_cooling(step, self.tm_max, self.alpha)


def linear_cooling(step_n: int, tm_max: float, tm_min: float, max_steps: int) -> float:
    """Linearly Scaled Cooling."""
    dt: float = tm_max - tm_min
    ds: float = (max_steps - step_n) / max_steps

    return tm_min + dt * ds


class LinearCooling(Cooling):
    """Linear cooling method.

    Equation:
        T(k) = Tmin + (Tmax - Tmin) * (Ntotal - k) / Ntotal

    """

    def cool(self, step: int) -> float:
        return linear_cooling(step, self.tm_max, self.tm_min, self.steps)


def quadratic_cooling(
    step_n: int,
    tm_max: float,
    tm_min: float,
    max_steps: int,
) -> float:
    """Squared Linear Cooling."""
    dt: float = tm_max - tm_min
    ds: float = (max_steps - step_n) / max_steps

    return tm_min + dt * (ds * ds)


class QuadraticCooling(Cooling):
    """Quadratic cooling method.

    Equation:
        T(k) = Tmin + (Tmax - Tmin) * ((Ntotal - k) / Ntotal) ** 2

    """

    def cool(self, step: int) -> float:
        return quadratic_cooling(step, self.tm_max, self.tm_min, self.steps)


def exponential_cooling(step_n: int, tm_max: float, alpha: float) -> float:
    """T(k) = tm_max * (alpha ** k))."""
    return tm_max * (alpha**step_n)


class ExponentialCooling(Cooling):
    """Exponential cooling method.

    Equation:
        T(k) = Tmax * (alpha ** k)

    """

    def cool(self, step: int) -> float:
        return exponential_cooling(step, self.tm_max, self.alpha)


class SqExponentialCooling(Cooling):
    """Square exponential cooling method.

    Equation:
        T(k) = Tmax * (alpha ** k) ** 2

    """

    def cool(self, step: int) -> float:
        a: float = exponential_cooling(step, self.tm_max, self.alpha)
        a /= exponential_cooling(0, self.tm_max, self.alpha)

        return self.tm_max * a * a


class SqrtExponentialCooling(Cooling):
    """Square root exponential cooling method.

    Equation:
        T(k) = Tmax * (alpha ** k) ** (1/2)

    """

    def cool(self, step: int) -> float:
        a: float = exponential_cooling(step, self.tm_max, self.alpha)
        a /= exponential_cooling(0, self.tm_max, self.alpha)

        return self.tm_max * sqrt(a)


class ExponentialQuadCooling(Cooling):
    """Exponential quadratic cooling method.

    Equation:
        a(k) = Tmin + (Tmax - Tmin) * ((Ntotal - k) / Ntotal) ** 2
        b(k) = Tmax * (alpha ** k)
        T(k) = a(k) + b(k)

    """

    def _cool(self, step: int) -> float:
        exps: float = exponential_cooling(step, self.tm_max, self.alpha)
        quad: float = quadratic_cooling(step, self.tm_max, self.tm_min, self.steps)

        return exps + quad

    def cool(self, step: int) -> float:
        tops: float = self.tm_max - self.tm_min

        return tops * self._cool(step) / self._cool(0)

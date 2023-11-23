"""
    Annealing/cooling.py

"""
from abc import ABC, abstractmethod
from math import sqrt
from typing import Optional


class Cooling(ABC):
    """Interface Class to define Temperature Cooling Functions."""
    def __init__(self,
                 steps: int = 1_000,
                 alpha: Optional[float] = None,
                 tm_min: float = 0.,
                 tm_max: float = 100.
                 ) -> None:
        self.steps = steps
        self.alpha = alpha
        self.tm_min = tm_min
        self.tm_max = tm_max

    @abstractmethod
    def cool(self, step: int) -> float:
        ...

    def __call__(self, step: int) -> float:
        return self.cool(step)


# Temperature Cooling Functions
def inverse_cooling(step_n, tm_max, alpha):
    """T(k) = Tmax / (1. + alpha * k)"""
    return tm_max / (1. + alpha * step_n)


class InverseCooling(Cooling):
    def cool(self, step: int):
        return inverse_cooling(step, self.tm_max, self.alpha)


def linear_cooling(step_n, tm_max, tm_min, max_steps):
    """Linearly Scaled Cooling."""
    dt = tm_max - tm_min
    ds = (max_steps - step_n) / max_steps
    return tm_min + dt * ds


class LinearCooling(Cooling):
    def cool(self, step: int):
        return linear_cooling(step, self.tm_max, self.tm_min, self.steps)


def quadratic_cooling(step_n, tm_max, tm_min, max_steps):
    """Squared Linear Cooling."""
    dt = tm_max - tm_min
    ds = (max_steps - step_n) / max_steps
    return tm_min + dt * (ds * ds)


class QuadraticCooling(Cooling):
    def cool(self, step: int):
        return quadratic_cooling(step, self.tm_max, self.tm_min, self.steps)


def exponential_cooling(step_n, tm_max, alpha):
    """T(k) = tm_max * (alpha ** k)) """
    return tm_max * (alpha ** step_n)


class ExponentialCooling(Cooling):
    def cool(self, step: int):
        return exponential_cooling(step, self.tm_max, self.alpha)


class SqExponentialCooling(Cooling):
    def cool(self, step: int):
        a = exponential_cooling(step, self.tm_max, self.alpha)
        a /= exponential_cooling(0, self.tm_max, self.alpha)
        return self.tm_max * a * a


class SqrtExponentialCooling(Cooling):
    def cool(self, step: int):
        a = exponential_cooling(step, self.tm_max, self.alpha)
        a /= exponential_cooling(0, self.tm_max, self.alpha)
        return self.tm_max * sqrt(a)


class ExponentialQuadCooling(Cooling):
    def _cool(self, step: int):
        exps = exponential_cooling(step, self.tm_max, self.alpha)
        quad = quadratic_cooling(step, self.tm_max, self.tm_min, self.steps)
        return exps + quad

    def cool(self, step: int):
        tops = self.tm_max - self.tm_min
        return tops * self._cool(step) / self._cool(0)

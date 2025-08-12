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

"""Strategies used to approach Traveling Salesmen Problem (TSP) solution minima."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable

import numpy as np


def _euclidean(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate an euclidean distance between two coordinates."""
    return np.sqrt(np.power(x - y, 2).sum())


def build_2d_distance_matrix(
    coordinates: np.ndarray,
    distance: Callable[[np.ndarray, np.ndarray], float] = _euclidean,
) -> np.ndarray:
    """Construct a 2d distance matrix from an array of Nd coordinates or values.

    Args:
        coordinates (np.ndarray): An array of coordinates of shape (N x m)
        distance (Callable[[np.ndarray, np.ndarray], float]): function to compute
            distance, or other relevant metric.

    Returns:
        (np.ndarray) 2d distance matrix of Shape (N x N), symmetric around the primary
            diagonal.

    Notes:
        * To maximize by a calculated parameter, have the distance function return
          negative values.

    """
    n: int = len(coordinates)
    matrix: np.ndarray = np.zeros((n, n))
    for j in range(n):
        for k in range(j + 1, n):
            matrix[j, k] = matrix[k, j] = distance(coordinates[j], coordinates[k])

    return matrix


# TODO: Think how we can integrate a strategy into a simulated annealing approach.
#       Think of it as a co-strategy to escape local minima to reach better convergence.
class Strategy(ABC):
    """Abstract minimization strategy.

    Args:
        distance_matrix (np.ndarray): 2d distance matrix.

    Attributes:
        n (int): number of data points (e.g. cities)
        tour (list[int]): list of indices describing best route.

    """

    distance_matrix: np.ndarray
    n: int
    tour: list[int]

    def __init__(self, distance_matrix: np.ndarray) -> None:
        self.distance_matrix = distance_matrix
        self.n = len(distance_matrix)
        self.tour = list(range(self.n))

    def calculate_tour_cost(self, tour: list[int]) -> float:
        """Calculate euclidean distance of a route."""
        raise NotImplementedError

    @abstractmethod
    def minimize(self) -> tuple[list[int], float]:
        """Abstract strategy to minimize tour route order."""
        raise NotImplementedError


class TSPStrategy(Strategy):
    """Abstract TSP minimization strategy.

    Args:
        distance_matrix (np.ndarray): 2d distance matrix

    Attributes:
        n (int): number of data points (i.e. cities)
        tour (list[int]): list of indices describing best route.

    """

    def calculate_tour_cost(self, tour: list[int]) -> float:
        """Calculate euclidean distance of a route."""
        a: np.ndarray = np.asarray(tour, dtype=np.int64)

        return self.distance_matrix[a, np.roll(a, 1)].sum()


class NearestNeighborStrategy(TSPStrategy):
    """Primitive greedy minimization technique of adjoining closest coordinates.

    Args:
        distance_matrix (np.ndarray): 2d distance matrix

    Attributes:
        n (int): number of data points (i.e. cities)
        tour (list[int]): list of indices describing best route.

    """

    def _nn(self, idx: int, visited: list[bool]) -> tuple[int | None, float]:
        current: int | None = None
        min_distance: float = float("inf")
        city: int
        for city in range(self.n):
            if not visited[city] and self.distance_matrix[idx, city] < min_distance:
                current = city
                min_distance = self.distance_matrix[idx, city]

        return current, min_distance

    def radial_nn(self, idx: int) -> list[int]:
        """Radially expand from a start index by nearest neighbor strategy."""
        self.tour = [idx]
        visited: list[bool] = [False for _ in range(self.n)]
        visited[idx] = True
        length: int = 1

        for _ in range(sum(divmod(self.n - length, 2))):
            a, va = self._nn(self.tour[length - 1], visited)
            if a is None:
                break

            b, vb = self._nn(self.tour[0], visited)
            if b is None:
                visited[a] = True
                self.tour.append(a)
                break

            if a == b:
                if va < vb:
                    visited[a] = True
                    b, vb = self._nn(self.tour[0], visited)
                    if b is None:
                        self.tour.append(a)
                        break
                else:
                    visited[b] = True
                    a, va = self._nn(self.tour[length - 1], visited)
                    if a is None:
                        self.tour.insert(0, b)
                        break

            self.tour.append(a)
            visited[a] = True
            self.tour.insert(0, b)
            visited[b] = True
            length += 2

        return self.tour

    def nearest_neighbor(self, idx: int) -> list[int]:
        """Compute nearest neighbor approximation starting from provided index."""
        self.tour = [idx]
        visited: list[bool] = [False for _ in range(self.n)]
        visited[idx] = True
        length: int = 1

        for _ in range(self.n - 1):
            current_city: int = self.tour[length - 1]
            next_city, _ = self._nn(current_city, visited)

            if next_city is not None:
                self.tour.append(next_city)
                visited[next_city] = True
                length += 1

        return self.tour

    def minimize(self) -> tuple[list[int], float]:
        min_cost: float = float("inf")
        best_tour: list[int] = []
        start: int
        func: Callable[[int], list[int]]

        for start in range(self.n):
            for func in (self.nearest_neighbor, self.radial_nn):
                tour: list[int] = func(start)
                tour_cost: float = self.calculate_tour_cost(tour)
                if tour_cost < min_cost:
                    min_cost = tour_cost
                    best_tour = tour

        self.tour = best_tour

        return self.tour, min_cost


class TwoOptStrategy(TSPStrategy):
    """Classic 2opt based strategy to minimizing TSP like problems.

    Args:
        distance_matrix (np.ndarray): 2d distance matrix

    Attributes:
        n (int): number of data points (i.e. cities)
        tour (list[int]): list of indices describing best route.

    """

    def swap_two_opt(self, tour: list[int], j: int, k: int) -> list[int]:
        """Perform a single two opt swap.

        Args:
            tour (list[int]): list of indices describing current route.
            j (int): Edge 1
            k (int): Edge 2

        Returns:
            (list[int]): new two-opted tour of same length as input tour, such that
                the segment between j and k are reversed.

        Notes:
            * Arguments j and k are expected to be valid indices within the tour, such
              that the following holds true: 0 < j < k < len(tour).

        """
        return tour[:j] + tour[k : j - 1 : -1] + tour[k + 1 :]

    def _2opt(self, current: float) -> float:
        for j in range(1, self.n - 1):
            for k in range(j + 1, self.n):
                new_tour: list[int] = self.swap_two_opt(self.tour, j, k)
                new_cost: float = self.calculate_tour_cost(new_tour)

                if new_cost < current:
                    self.tour = new_tour
                    current = new_cost

        return current

    def two_opt(self, iterations: int = 1000) -> None:
        """Perform multiple iterations of 2 opt swaps.

        Args:
            iterations (int): maximum number of iterations to perform.

        Returns:
            (None): best tour is saved to class attribute self.tour

        Notes:
            * Calculation continues up to maximum number of iterations or convergence,
              whichever comes first.

        """
        improved: bool = True
        count: int = 0
        current: float = self.calculate_tour_cost(self.tour)

        while improved and count < iterations:
            new_cost: float = self._2opt(current)
            improved = new_cost < current
            current = new_cost
            count += 1

    def minimize(self, iterations: int = 10_000) -> tuple[list[int], float]:
        self.two_opt(iterations)
        min_cost: float = self.calculate_tour_cost(self.tour)

        return self.tour, min_cost


class ThreeOptStrategy(TSPStrategy):
    """Classic 3opt based Strategy to minimizing TSP like problems.

    Args:
        distance_matrix (np.ndarray): 2d distance matrix

    Attributes:
        n (int): number of data points (i.e. cities)
        tour (list[int]): list of indices describing best route.

    """

    def swap_three_opt(
        self,
        tour: list[int],
        i: int,
        j: int,
        k: int,
    ) -> Iterable[list[int]]:
        """Perform a single three opt swap.

        Args:
            tour (list[int]): list of indices describing current route.
            i (int): Edge 1
            j (int): Edge 2
            k (int): Edge 3

        Yields:
            (list[int]): new three-opted tour (of 7) of same length as input tour.

        Notes:
            * Arguments i, j, and k are expected to be valid indices within tour, such
              that the following holds true: 0 < i < j < k < len(tour).
            * Excluding provided tour, there are 7 other permutations of a three-opt.

        """
        # There are 7 possible permutations (excluding current tour)
        # i, j, and k represent three edges dividing the tour into four segments (Si)
        # Current tour description: S1 |i| S2 |j| S3 |k| S4

        # S2 reversed. (S1 - S2' - S3 - S4)
        s2_rev: list[int] = tour[j - 1 : i - 1 : -1]
        yield tour[:i] + s2_rev + tour[j:]

        # S3 reversed. (S1 - S2 - S3' - S4)
        s3_rev: list[int] = tour[k - 1 : j - 1 : -1]
        yield tour[:j] + s3_rev + tour[k:]

        # S2 & S3 reversed. (S1 - S2' - S3' - S4)
        yield tour[:i] + s2_rev + s3_rev + tour[k:]

        # S2 & S3 swapped positions.  (S1 - S3 - S2 - S4)
        yield tour[:i] + tour[j:k] + tour[i:j] + tour[k:]

        # S2 & S3 swapped positions. S2 reversed.  (S1 - S3 - S2' - S4)
        yield tour[:i] + tour[j:k] + s2_rev + tour[k:]

        # S2 & S3 swapped positions. S3 reversed.  (S1 - S3' - S2 - S4)
        yield tour[:i] + s3_rev + tour[i:j] + tour[k:]

        # S2 & S3 swapped positions and reversed.  (S1 - S3' - S2' - S4)
        yield tour[:i] + tour[k - 1 : i - 1 : -1] + tour[k:]

    def _3opt(self, current: float) -> float:
        for i in range(1, self.n - 2):
            for j in range(i + 1, self.n - 1):
                for k in range(j + 1, self.n):
                    for new_tour in self.swap_three_opt(self.tour, i, j, k):
                        new_cost: float = self.calculate_tour_cost(new_tour)
                        if new_cost < current:
                            self.tour = new_tour
                            current = new_cost

        return current

    def three_opt(self, iterations: int = 1000) -> None:
        """Perform multiple iterations of 3-opt swaps.

        Args:
            iterations (int): maximum number of iterations to perform.

        Returns:
            (None): best tour is saved to class attribute self.tour

        Notes:
            * Calculation continues up to max iterations or convergence, whichever
              comes first.

        """
        improved: bool = True
        count: int = 0
        current: float = self.calculate_tour_cost(self.tour)

        while improved and count < iterations:
            improved = False
            new_cost: float = self._3opt(current)
            if new_cost < current:
                improved = True
                current = new_cost

            count += 1

    def minimize(self, iterations: int = 10_000) -> tuple[list[int], float]:
        self.three_opt(iterations)
        min_cost = self.calculate_tour_cost(self.tour)

        return self.tour, min_cost

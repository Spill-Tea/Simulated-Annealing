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

import networkx as nx
import numpy as np
import pytest

from Annealing import strategies


@pytest.fixture
def dm(coord: np.ndarray) -> np.ndarray:
    """2d distance matrix of sample coordinates."""
    return strategies.build_2d_distance_matrix(coord)


@pytest.fixture
def tour_nn() -> list[int]:
    """Nearest neighbor tour."""
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
    """Two opt tour."""
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
    """Three opt tour."""
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


@pytest.fixture
def tour_cfs() -> list[int]:
    """christofides tour."""
    return [
        0,
        28,
        13,
        37,
        3,
        19,
        44,
        34,
        45,
        2,
        30,
        49,
        26,
        5,
        46,
        12,
        9,
        36,
        1,
        32,
        47,
        39,
        22,
        16,
        4,
        43,
        23,
        17,
        35,
        15,
        33,
        42,
        10,
        48,
        40,
        7,
        14,
        20,
        27,
        24,
        38,
        25,
        41,
        8,
        29,
        18,
        31,
        11,
        21,
        6,
    ]


@pytest.fixture
def sample_coord() -> np.ndarray:
    coordinates: np.ndarray = np.asarray([[1, 2], [3, 4], [5, 6]])
    return coordinates


@pytest.fixture
def sample_graph(sample_coord: np.ndarray) -> nx.Graph:
    graph: nx.Graph = nx.Graph()
    graph.add_node(0, coordinate=sample_coord[0].tolist())
    graph.add_node(1, coordinate=sample_coord[1].tolist())
    graph.add_node(2, coordinate=sample_coord[2].tolist())
    graph.add_edge(0, 1, weight=strategies._euclidean(sample_coord[0], sample_coord[1]))
    graph.add_edge(0, 2, weight=strategies._euclidean(sample_coord[0], sample_coord[2]))
    graph.add_edge(1, 2, weight=strategies._euclidean(sample_coord[1], sample_coord[2]))

    return graph


def test_graph_build_from_coordinates(
    sample_coord: np.ndarray,
    sample_graph: nx.Graph,
) -> None:
    """Confirm graph is built correctly from coordinates."""
    result: nx.Graph = strategies.build_graph(sample_coord)
    assert nx.algorithms.is_isomorphic(result, sample_graph), (
        "Graphs are not isomorphic"
    )

    # convert array to list for quick python object equality checks.
    for i in result.nodes:
        result.nodes[i]["coordinate"] = result.nodes[i]["coordinate"].tolist()

    # NOTE: utility function from networkx only performs python object equivalence
    #       and numpy arrays must use np.all for complete equivalency.
    assert nx.utils.graphs_equal(sample_graph, result), "Graphs are not equivalent."


def test_matrix_to_graph(dm: np.ndarray, coord: np.ndarray) -> None:
    """Confirm graphs constructed are equal."""
    graph_a = strategies.build_graph_from_2d_distance_matrix(dm)
    graph_b = strategies.build_graph(coord)

    assert nx.utils.edges_equal(graph_a.edges, graph_b.edges), "Unequal Edges"
    assert nx.utils.nodes_equal(graph_a.nodes, graph_b.nodes), "Unequal Nodes"
    assert nx.algorithms.is_isomorphic(graph_a, graph_b), "Graphs are not isomorphic"


@pytest.mark.parametrize(
    ["tsp_strategy", "expected_tour", "expected_cost"],
    [
        (strategies.NearestNeighborStrategy, "tour_nn", 1149.5188),
        (strategies.TwoOptStrategy, "tour_2opt", 1119.0572),
        (strategies.ThreeOptStrategy, "tour_3opt", 1200.9199),
        (strategies.ChristofidesStrategy, "tour_cfs", 1178.9486),
    ],
)
def test_tsp_strategies(
    dm: np.ndarray,
    tsp_strategy: type[strategies.TSPStrategy],
    expected_tour: list[int],
    expected_cost: float,
    request: pytest.FixtureRequest,
) -> None:
    """Test various tsp strategies produce expected tours and costs."""
    strategy = tsp_strategy(dm)
    tour, val = strategy.minimize()
    expected = request.getfixturevalue(expected_tour)

    assert isinstance(tour, list), "Expected a list result."
    assert len(tour) == len(dm), "Expected same number of data points."
    assert tour == expected, "Expected predictable tour."

    assert isinstance(val, float), "Expected a float result."
    assert np.isclose(val, expected_cost), "Unexpected tour length calculation."

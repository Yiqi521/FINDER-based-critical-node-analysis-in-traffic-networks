from __future__ import annotations

import networkx as nx

from experiments.traffic.path_algorithms import (
    astar_search,
    bidirectional_dijkstra_search,
    dijkstra_search,
)


def _graph() -> nx.Graph:
    g = nx.Graph()
    g.add_edge(0, 1, travel_time=1.0)
    g.add_edge(1, 2, travel_time=1.0)
    g.add_edge(0, 2, travel_time=3.0)
    g.add_edge(2, 3, travel_time=1.0)
    return g


def test_dijkstra_shortest_path_cost() -> None:
    result = dijkstra_search(_graph(), 0, 3)
    assert result.found
    assert result.cost == 3.0
    assert result.path == [0, 1, 2, 3]


def test_astar_matches_dijkstra_cost() -> None:
    dres = dijkstra_search(_graph(), 0, 3)
    ares = astar_search(_graph(), 0, 3)
    assert ares.found
    assert ares.cost == dres.cost


def test_bidirectional_finds_valid_path() -> None:
    result = bidirectional_dijkstra_search(_graph(), 0, 3)
    assert result.found
    assert result.cost == 3.0
    assert result.path[0] == 0
    assert result.path[-1] == 3

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Callable

import networkx as nx

WeightFn = Callable[[int, int, dict], float]
HeuristicFn = Callable[[int, int, nx.Graph], float]


@dataclass(slots=True)
class PathResult:
    algorithm: str
    found: bool
    cost: float
    expanded_nodes: int
    path: list[int]


def default_weight(_u: int, _v: int, data: dict) -> float:
    if "travel_time" in data:
        return float(data["travel_time"])
    if "weight" in data:
        return float(data["weight"])
    if "length" in data:
        return float(data["length"])
    return 1.0


def geo_heuristic(node: int, goal: int, g: nx.Graph) -> float:
    ndata = g.nodes[node]
    gdata = g.nodes[goal]
    if "lat" in ndata and "lon" in ndata and "lat" in gdata and "lon" in gdata:
        dx = float(ndata["lat"]) - float(gdata["lat"])
        dy = float(ndata["lon"]) - float(gdata["lon"])
        return math.sqrt(dx * dx + dy * dy)
    return 0.0


def dijkstra_search(g: nx.Graph, source: int, target: int, weight: WeightFn = default_weight) -> PathResult:
    pq: list[tuple[float, int]] = [(0.0, source)]
    dist: dict[int, float] = {source: 0.0}
    prev: dict[int, int] = {}
    seen: set[int] = set()
    expanded = 0
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        expanded += 1
        if u == target:
            break
        for v, attrs in g[u].items():
            nd = d + weight(u, v, attrs)
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if target not in dist:
        return PathResult("dijkstra", False, float("inf"), expanded, [])
    return PathResult("dijkstra", True, dist[target], expanded, _recover_path(prev, source, target))


def astar_search(
    g: nx.Graph,
    source: int,
    target: int,
    weight: WeightFn = default_weight,
    heuristic: HeuristicFn = geo_heuristic,
) -> PathResult:
    pq: list[tuple[float, int]] = [(0.0, source)]
    gscore: dict[int, float] = {source: 0.0}
    prev: dict[int, int] = {}
    seen: set[int] = set()
    expanded = 0
    while pq:
        _, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        expanded += 1
        if u == target:
            break
        for v, attrs in g[u].items():
            tentative = gscore[u] + weight(u, v, attrs)
            if tentative < gscore.get(v, float("inf")):
                gscore[v] = tentative
                prev[v] = u
                fscore = tentative + heuristic(v, target, g)
                heapq.heappush(pq, (fscore, v))
    if target not in gscore:
        return PathResult("astar", False, float("inf"), expanded, [])
    return PathResult("astar", True, gscore[target], expanded, _recover_path(prev, source, target))


def bidirectional_dijkstra_search(
    g: nx.Graph,
    source: int,
    target: int,
    weight: WeightFn = default_weight,
) -> PathResult:
    if source == target:
        return PathResult("bidirectional_dijkstra", True, 0.0, 1, [source])

    f_pq: list[tuple[float, int]] = [(0.0, source)]
    b_pq: list[tuple[float, int]] = [(0.0, target)]
    f_dist: dict[int, float] = {source: 0.0}
    b_dist: dict[int, float] = {target: 0.0}
    f_prev: dict[int, int] = {}
    b_prev: dict[int, int] = {}
    f_seen: set[int] = set()
    b_seen: set[int] = set()
    best = float("inf")
    meeting: int | None = None
    expanded = 0

    while f_pq and b_pq:
        fd, fu = heapq.heappop(f_pq)
        if fu not in f_seen:
            f_seen.add(fu)
            expanded += 1
            if fu in b_dist and fd + b_dist[fu] < best:
                best = fd + b_dist[fu]
                meeting = fu
            for v, attrs in g[fu].items():
                nd = fd + weight(fu, v, attrs)
                if nd < f_dist.get(v, float("inf")):
                    f_dist[v] = nd
                    f_prev[v] = fu
                    heapq.heappush(f_pq, (nd, v))

        bd, bu = heapq.heappop(b_pq)
        if bu not in b_seen:
            b_seen.add(bu)
            expanded += 1
            if bu in f_dist and bd + f_dist[bu] < best:
                best = bd + f_dist[bu]
                meeting = bu
            for v, attrs in g[bu].items():
                nd = bd + weight(bu, v, attrs)
                if nd < b_dist.get(v, float("inf")):
                    b_dist[v] = nd
                    b_prev[v] = bu
                    heapq.heappush(b_pq, (nd, v))

        if meeting is not None:
            f_min = f_pq[0][0] if f_pq else float("inf")
            b_min = b_pq[0][0] if b_pq else float("inf")
            if f_min + b_min >= best:
                break

    if meeting is None:
        return PathResult("bidirectional_dijkstra", False, float("inf"), expanded, [])
    path = _recover_bidirectional_path(f_prev, b_prev, source, target, meeting)
    return PathResult("bidirectional_dijkstra", True, best, expanded, path)


def _recover_path(prev: dict[int, int], source: int, target: int) -> list[int]:
    cur = target
    path = [cur]
    while cur != source:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return path


def _recover_bidirectional_path(
    f_prev: dict[int, int],
    b_prev: dict[int, int],
    source: int,
    target: int,
    meeting: int,
) -> list[int]:
    left = _recover_path(f_prev, source, meeting)
    right = [meeting]
    cur = meeting
    while cur != target:
        cur = b_prev[cur]
        right.append(cur)
    return left + right[1:]

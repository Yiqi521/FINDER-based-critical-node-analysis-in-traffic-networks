from __future__ import annotations

from dataclasses import dataclass
from typing import List

import networkx as nx


@dataclass
class FinderEnv:
    graph: nx.Graph

    def __post_init__(self) -> None:
        self.covered_set: set[int] = set()
        self.action_list: List[int] = []
        self.num_covered_edges = 0
        self._num_edges = self.graph.number_of_edges()

    def reset(self) -> None:
        self.covered_set.clear()
        self.action_list.clear()
        self.num_covered_edges = 0

    def is_terminal(self) -> bool:
        return self.num_covered_edges >= self._num_edges

    def available_actions(self) -> List[int]:
        actions: List[int] = []
        for node in self.graph.nodes:
            if node in self.covered_set:
                continue
            for neigh in self.graph.neighbors(node):
                if neigh not in self.covered_set:
                    actions.append(int(node))
                    break
        return actions

    def step(self, action: int) -> float:
        if action in self.covered_set:
            raise ValueError(f"Action {action} already selected")
        self.covered_set.add(action)
        self.action_list.append(action)
        for neigh in self.graph.neighbors(action):
            if neigh not in self.covered_set:
                self.num_covered_edges += 1
        n = float(self.graph.number_of_nodes())
        reward = -self.remaining_cnd_score() / (n * n * (n - 1.0) / 2.0)
        return reward

    def remaining_cnd_score(self) -> float:
        remaining = self.graph.copy()
        remaining.remove_nodes_from(self.covered_set)
        score = 0.0
        for cc in nx.connected_components(remaining):
            s = float(len(cc))
            score += s * (s - 1.0) / 2.0
        return score

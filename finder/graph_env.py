from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal

import networkx as nx

Task = Literal["CN", "ND", "CN_cost", "ND_cost"]


@dataclass
class FinderEnv:
    graph: nx.Graph
    task: Task = "CN"
    # Per-node weights for cost variants.  When empty and task is a cost
    # variant, weights are read from graph node attribute "weight" (default 1.0).
    node_weights: Dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.covered_set: set[int] = set()
        self.action_list: List[int] = []
        self.num_covered_edges = 0
        self._num_edges = self.graph.number_of_edges()

        if self.task in ("CN_cost", "ND_cost") and not self.node_weights:
            self.node_weights = {
                int(n): float(data.get("weight", 1.0))
                for n, data in self.graph.nodes(data=True)
            }
        self._total_weight: float = (
            sum(self.node_weights.values()) if self.node_weights else 1.0
        )

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
        return self._reward(action)

    # ------------------------------------------------------------------
    # Reward dispatch — mirrors the four C++ getReward() implementations
    # ------------------------------------------------------------------

    def _reward(self, action: int) -> float:
        n = float(self.graph.number_of_nodes())
        if self.task == "CN":
            # -(Σ s·(s-1)/2) / (N²·(N-1)/2)
            return -self.remaining_cnd_score() / (n * n * (n - 1.0) / 2.0)
        elif self.task == "ND":
            # -max_cc_size / N²
            return -self._max_cc_size() / (n * n)
        else:
            # CN_cost and ND_cost share the same formula:
            # -(max_cc_size / N) * (weight[a] / total_weight)
            w = self.node_weights.get(action, 1.0)
            return -(self._max_cc_size() / n) * (w / self._total_weight)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def remaining_cnd_score(self) -> float:
        """Σ s·(s-1)/2 over all connected components of the remaining subgraph."""
        remaining = self.graph.copy()
        remaining.remove_nodes_from(self.covered_set)
        score = 0.0
        for cc in nx.connected_components(remaining):
            s = float(len(cc))
            score += s * (s - 1.0) / 2.0
        return score

    def _max_cc_size(self) -> float:
        """Size of the largest connected component in the remaining subgraph."""
        remaining = self.graph.copy()
        remaining.remove_nodes_from(self.covered_set)
        if remaining.number_of_nodes() == 0:
            return 0.0
        return float(max(len(cc) for cc in nx.connected_components(remaining)))

"""Tests for variant-aware FinderEnv reward functions.

Each test fixes a small hand-crafted graph so the expected reward can be
computed by hand and verified against the C++ formulas.
"""
from __future__ import annotations

import pytest
import networkx as nx

from finder.graph_env import FinderEnv


# ---------------------------------------------------------------------------
# Shared fixture: a simple 4-node path  0-1-2-3
# After removing node 1, remaining graph has two CCs: {0} and {2,3}.
# ---------------------------------------------------------------------------

def _path4() -> nx.Graph:
    return nx.path_graph(4)  # nodes 0,1,2,3  edges (0,1),(1,2),(2,3)


# ---------------------------------------------------------------------------
# CN  —  reward = -cnd_score / (N² * (N-1) / 2)
# ---------------------------------------------------------------------------

class TestCN:
    def test_reward_formula(self) -> None:
        g = _path4()
        env = FinderEnv(g, task="CN")
        # Remove node 1: remaining CCs are {0} and {2, 3}
        # cnd_score = 1*(1-1)/2 + 2*(2-1)/2 = 0 + 1 = 1
        # norm = 4 * 4 * 3 / 2 = 24
        # reward = -1/24
        r = env.step(1)
        assert abs(r - (-1.0 / 24.0)) < 1e-9

    def test_terminal_when_all_edges_covered(self) -> None:
        g = _path4()
        env = FinderEnv(g, task="CN")
        # Removing nodes 1 and 2 covers all 3 edges
        env.step(1)
        env.step(2)
        assert env.is_terminal()

    def test_available_actions_shrink(self) -> None:
        g = _path4()
        env = FinderEnv(g, task="CN")
        assert len(env.available_actions()) > 0
        env.step(1)
        # After removing node 1, node 0 has no uncovered neighbour → not available
        avail = env.available_actions()
        assert 0 not in avail
        assert 1 not in avail


# ---------------------------------------------------------------------------
# ND  —  reward = -max_cc_size / N²
# ---------------------------------------------------------------------------

class TestND:
    def test_reward_formula(self) -> None:
        g = _path4()
        env = FinderEnv(g, task="ND")
        # Remove node 1: remaining graph = {0}, {2,3}  → max_cc = 2
        # reward = -2 / 16 = -0.125
        r = env.step(1)
        assert abs(r - (-2.0 / 16.0)) < 1e-9

    def test_reward_differs_from_cn(self) -> None:
        g = _path4()
        cn_env = FinderEnv(g, task="CN")
        nd_env = FinderEnv(g, task="ND")
        r_cn = cn_env.step(1)
        r_nd = nd_env.step(1)
        assert r_cn != r_nd


# ---------------------------------------------------------------------------
# CN_cost  —  reward = -(max_cc_size / N) * (weight[a] / total_weight)
# ---------------------------------------------------------------------------

class TestCNCost:
    def _weighted_graph(self) -> nx.Graph:
        g = _path4()
        weights = {0: 0.1, 1: 0.4, 2: 0.3, 3: 0.2}
        for n, w in weights.items():
            g.nodes[n]["weight"] = w
        return g

    def test_reward_formula(self) -> None:
        g = self._weighted_graph()
        env = FinderEnv(g, task="CN_cost")
        # Remove node 1 (weight=0.4, total=1.0): remaining max_cc = 2 (nodes 2,3)
        # reward = -(2/4) * (0.4/1.0) = -0.5 * 0.4 = -0.2
        r = env.step(1)
        assert abs(r - (-0.2)) < 1e-9

    def test_weights_read_from_node_attrs(self) -> None:
        g = self._weighted_graph()
        env = FinderEnv(g, task="CN_cost")
        assert env.node_weights[1] == pytest.approx(0.4)
        assert env._total_weight == pytest.approx(1.0)

    def test_default_weight_is_one(self) -> None:
        g = _path4()  # no weight attributes
        env = FinderEnv(g, task="CN_cost")
        # All weights default to 1.0, total = 4.0
        # Remove node 1: max_cc=2, reward = -(2/4)*(1/4) = -0.125
        r = env.step(1)
        assert abs(r - (-0.125)) < 1e-9


# ---------------------------------------------------------------------------
# ND_cost  —  same formula as CN_cost (verified from C++ source)
# ---------------------------------------------------------------------------

class TestNDCost:
    def _weighted_graph(self) -> nx.Graph:
        g = _path4()
        for n in g.nodes:
            g.nodes[n]["weight"] = 1.0
        return g

    def test_cn_cost_and_nd_cost_are_identical_formula(self) -> None:
        """Both cost variants use -(max_cc/N)*(w/total_w) — results must match."""
        g = self._weighted_graph()
        cn_cost_env = FinderEnv(g, task="CN_cost")
        nd_cost_env = FinderEnv(g, task="ND_cost")
        r_cn = cn_cost_env.step(1)
        r_nd = nd_cost_env.step(1)
        assert abs(r_cn - r_nd) < 1e-9

    def test_reward_formula(self) -> None:
        g = self._weighted_graph()
        env = FinderEnv(g, task="ND_cost")
        # All weights 1.0, total=4.0, max_cc after removing 1 = 2
        # reward = -(2/4)*(1/4) = -0.125
        r = env.step(1)
        assert abs(r - (-0.125)) < 1e-9


# ---------------------------------------------------------------------------
# Reset restores state across all variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["CN", "ND", "CN_cost", "ND_cost"])
def test_reset_clears_state(task: str) -> None:
    g = _path4()
    for n in g.nodes:
        g.nodes[n]["weight"] = 1.0
    env = FinderEnv(g, task=task)  # type: ignore[arg-type]
    env.step(1)
    assert len(env.action_list) == 1
    env.reset()
    assert len(env.action_list) == 0
    assert len(env.covered_set) == 0
    assert env.num_covered_edges == 0
    assert not env.is_terminal()

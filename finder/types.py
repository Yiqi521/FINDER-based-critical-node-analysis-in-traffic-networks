from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass(slots=True)
class Transition:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    state_nodes: List[int]
    action: int
    reward: float
    next_state_nodes: List[int]
    terminal: bool


@dataclass(slots=True)
class ReplayBatch:
    transitions: List[Transition]


@dataclass(slots=True)
class GraphBatch:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    candidate_nodes: List[int]

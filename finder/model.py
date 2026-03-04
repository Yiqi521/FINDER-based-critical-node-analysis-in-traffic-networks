from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "torch-geometric is required for the modern FinderModel path"
    ) from exc


class FinderModel(nn.Module):
    """PyTorch/PyG approximation of the original graph embedding Q-network."""

    def __init__(self, in_channels: int = 2, hidden_channels: int = 64, aux_dim: int = 4):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.cross = nn.Linear(hidden_channels, 1, bias=False)
        self.head1 = nn.Linear(hidden_channels, hidden_channels)
        self.head2 = nn.Linear(hidden_channels + aux_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        candidate_nodes: torch.Tensor,
        aux: torch.Tensor,
    ) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.normalize(h, p=2.0, dim=1)

        graph_pool = h.mean(dim=0, keepdim=True).repeat(candidate_nodes.shape[0], 1)
        action_embed = h[candidate_nodes]
        pair_embed = action_embed * graph_pool
        cross = self.cross(pair_embed)
        mixed = pair_embed * cross

        out = F.relu(self.head1(mixed))
        out = torch.cat([out, aux], dim=1)
        q = self.head2(out)
        return q.squeeze(-1)

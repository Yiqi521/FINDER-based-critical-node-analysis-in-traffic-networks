from __future__ import annotations

import random
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from finder.config import TrainConfig
from finder.graph_env import FinderEnv
from finder.model import FinderModel
from finder.replay import NStepReplayMemory
from finder.types import Transition


def _gen_graph(cfg: TrainConfig) -> nx.Graph:
    n = np.random.randint(cfg.num_min, cfg.num_max + 1)
    if cfg.graph_type == "erdos_renyi":
        return nx.erdos_renyi_graph(n=n, p=0.15)
    if cfg.graph_type == "powerlaw":
        return nx.powerlaw_cluster_graph(n=n, m=4, p=0.05)
    if cfg.graph_type == "small-world":
        return nx.connected_watts_strogatz_graph(n=n, k=8, p=0.1)
    return nx.barabasi_albert_graph(n=n, m=4)


def _to_tensors(g: nx.Graph, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    n = g.number_of_nodes()
    x = torch.ones((n, 2), dtype=torch.float32, device=device)
    edges = list(g.edges())
    if not edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    else:
        undirected = edges + [(b, a) for (a, b) in edges]
        edge_index = torch.tensor(undirected, dtype=torch.long, device=device).t().contiguous()
    return x, edge_index


def _epsilon(step: int, cfg: TrainConfig) -> float:
    return cfg.epsilon_end + max(
        0.0,
        (cfg.epsilon_start - cfg.epsilon_end)
        * (cfg.epsilon_decay_steps - step)
        / cfg.epsilon_decay_steps,
    )


def train(cfg: TrainConfig) -> Path:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    model = FinderModel(hidden_channels=cfg.hidden_size).to(device)
    target = FinderModel(hidden_channels=cfg.hidden_size).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    memory = NStepReplayMemory(cfg.memory_size, cfg.n_step, cfg.gamma)

    global_step = 0
    for ep in trange(cfg.episodes, desc="Training"):
        g = _gen_graph(cfg)
        env = FinderEnv(g)
        env.reset()
        x, edge_index = _to_tensors(g, device)

        while not env.is_terminal():
            actions = env.available_actions()
            if not actions:
                break
            eps = _epsilon(global_step, cfg)
            if random.random() < eps:
                action = random.choice(actions)
            else:
                cands = torch.tensor(actions, dtype=torch.long, device=device)
                aux = torch.zeros((len(actions), 4), dtype=torch.float32, device=device)
                with torch.no_grad():
                    qvals = model(x, edge_index, cands, aux)
                action = actions[int(torch.argmax(qvals).item())]

            state_nodes = list(env.action_list)
            reward = env.step(action)
            next_state_nodes = list(env.action_list)
            terminal = env.is_terminal()
            memory.add(
                Transition(
                    node_features=x.detach(),
                    edge_index=edge_index.detach(),
                    state_nodes=state_nodes,
                    action=action,
                    reward=reward,
                    next_state_nodes=next_state_nodes,
                    terminal=terminal,
                )
            )

            if len(memory) >= cfg.batch_size:
                batch = memory.sample(cfg.batch_size)
                losses = []
                for tr in batch.transitions:
                    candidates = [n for n in range(tr.node_features.shape[0]) if n not in tr.state_nodes]
                    if not candidates:
                        continue
                    cands = torch.tensor(candidates, dtype=torch.long, device=device)
                    aux = torch.zeros((len(candidates), 4), dtype=torch.float32, device=device)
                    q_all = model(tr.node_features, tr.edge_index, cands, aux)
                    action_idx = candidates.index(tr.action)
                    q_pred = q_all[action_idx]
                    if tr.terminal:
                        q_target = torch.tensor(tr.reward, dtype=torch.float32, device=device)
                    else:
                        next_cands = [n for n in range(tr.node_features.shape[0]) if n not in tr.next_state_nodes]
                        if next_cands:
                            nc = torch.tensor(next_cands, dtype=torch.long, device=device)
                            na = torch.zeros((len(next_cands), 4), dtype=torch.float32, device=device)
                            with torch.no_grad():
                                q_next = target(tr.node_features, tr.edge_index, nc, na).max()
                            q_target = torch.tensor(tr.reward, dtype=torch.float32, device=device) + cfg.gamma * q_next
                        else:
                            q_target = torch.tensor(tr.reward, dtype=torch.float32, device=device)
                    losses.append(F.mse_loss(q_pred, q_target))

                if losses:
                    loss = torch.stack(losses).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            global_step += 1
            if global_step % cfg.target_update == 0:
                target.load_state_dict(model.state_dict())

        if ep and ep % 200 == 0:
            target.load_state_dict(model.state_dict())

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"finder_{cfg.variant.lower()}.pt"
    torch.save({"model": model.state_dict(), "config": cfg.__dict__}, out)
    return out

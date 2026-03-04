from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

from finder.types import ReplayBatch, Transition


@dataclass(slots=True)
class NStepReplayMemory:
    capacity: int
    n_step: int
    gamma: float

    def __post_init__(self) -> None:
        self.memory: deque[Transition] = deque(maxlen=self.capacity)
        self._nstep_queue: deque[Transition] = deque(maxlen=max(1, self.n_step))

    def _flush_one(self) -> None:
        if not self._nstep_queue:
            return
        transitions = list(self._nstep_queue)
        first = transitions[0]
        discounted = 0.0
        terminal = False
        next_state = first.next_state_nodes
        for i, tr in enumerate(transitions):
            discounted += (self.gamma ** i) * float(tr.reward)
            next_state = tr.next_state_nodes
            if tr.terminal:
                terminal = True
                break
        merged = Transition(
            node_features=first.node_features,
            edge_index=first.edge_index,
            state_nodes=first.state_nodes,
            action=first.action,
            reward=discounted,
            next_state_nodes=next_state,
            terminal=terminal,
        )
        self.memory.append(merged)
        self._nstep_queue.popleft()

    def add(self, transition: Transition) -> None:
        self._nstep_queue.append(transition)
        if len(self._nstep_queue) >= self.n_step or transition.terminal:
            self._flush_one()
            if transition.terminal:
                while self._nstep_queue:
                    self._flush_one()

    def sample(self, batch_size: int) -> ReplayBatch:
        sample = random.sample(self.memory, batch_size)
        return ReplayBatch(transitions=sample)

    def __len__(self) -> int:
        return len(self.memory)

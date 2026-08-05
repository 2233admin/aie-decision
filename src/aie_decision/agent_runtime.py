"""Public runtime facade for the recursive Fermi agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .agent_runtime_budget import BudgetMixin
from .agent_runtime_execution import ExecutionMixin
from .agent_runtime_lifecycle import LifecycleMixin
from .agent_runtime_projection import ProjectionMixin
from .agent_runtime_support import (
    ActionResult,
    BudgetCounters,
    BudgetPolicy,
    KernelProtocol,
    RuntimeError_,
    SessionStatus,
    _now_iso,
)
from .trajectory import PROTOCOL_VERSION, SCHEMA_VERSION, Trajectory


@dataclass
class AgentRuntime(LifecycleMixin, ExecutionMixin, ProjectionMixin, BudgetMixin):
    """Stateful, replayable tool loop for an external AI.

    The runtime owns the trajectory, the current state, the budget policy,
    and the budget counters.  All behaviour beyond persistence and
    bookkeeping is delegated to the injected kernel.
    """

    session_id: str
    kernel: KernelProtocol
    trajectory: Trajectory
    state: dict[str, Any] = field(default_factory=dict)
    question: str = ""
    budgets: BudgetPolicy = field(default_factory=BudgetPolicy)
    counters: BudgetCounters = field(default_factory=BudgetCounters)
    status: SessionStatus = SessionStatus.ACTIVE
    frontier_evaluation: dict[str, Any] | None = None
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    _clock: Callable[[], str] | None = field(default=None, repr=False, compare=False)
    _start_recorded: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.session_id:
            raise RuntimeError_("session_id is required")
        if not self.created_at:
            self.created_at = _now_iso(self._clock)
        self.updated_at = self.created_at


__all__ = [
    "ActionResult",
    "AgentRuntime",
    "BudgetCounters",
    "BudgetPolicy",
    "KernelProtocol",
    "PROTOCOL_VERSION",
    "RuntimeError_",
    "SCHEMA_VERSION",
    "SessionStatus",
]

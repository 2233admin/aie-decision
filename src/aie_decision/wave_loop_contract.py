"""Shared dataclasses, constants, and error types for the wave-loop subsystem.

This module is the dependency root for the wave-loop package.  Every other
wave_loop_* module depends on it, and it depends only on the standard library
and :mod:`aie_decision.models`.

It exists so that ``wave_authority.py`` can import shared types from here
instead of from ``wave_loop.py``, breaking the authority→loop coupling
without introducing a circular import.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .models import Revision

# ---------------------------------------------------------------------------
# Versioned schemas
# ---------------------------------------------------------------------------

JOINT_SCHEMA_VERSION = "joint-wave-schema.v1"
FACTOR_IR_VERSION = "factor-ir.v1"
PARTICLE_SURFACE_VERSION = "particle-surface.v1"
WAVE_LEDGER_VERSION = "joint-wave-ledger.v1"
WAVE_CHECKPOINT_VERSION = "joint-wave-checkpoint.v1"
WAVE_LOOP_RESULT_VERSION = "joint-wave-loop.v1"

# Action vocabulary required by ``wave-surface-search-loop`` spec.  Each entry
# is mapped to a deterministic event state so the replay can validate ordering.
ACTION_KINDS: frozenset[str] = frozenset(
    {"measure", "expand_variable", "add_interaction", "split_regime", "minimize", "stop"}
)
WAVE_ACTIVATION_STATES: frozenset[str] = frozenset({"DRAFT", "EVALUATED", "REFINING", "STOP"})
WAVE_TERMINAL_STATES: frozenset[str] = frozenset({"ACCEPTED", "UNRESOLVED"})


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WaveLoopError(ValueError):
    """Raised when the wave-loop input, ledger, or checkpoint is invalid."""


class _RestrictedFormulaError(WaveLoopError):
    """Raised when a mapping formula is not expressible in the restricted IR."""


# ---------------------------------------------------------------------------
# Allowed AST node types for the restricted formula grammar
# ---------------------------------------------------------------------------

_ALLOWED_AST_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.UAdd,
    ast.USub,
)


# ---------------------------------------------------------------------------
# Frozen dataclasses — narrow typed boundaries
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WaveEvent:
    """Append-only event written to ``AnalysisLedger`` for every transition."""

    schema_version: str
    event_id: str
    run_id: str
    surface_id: str
    state: str
    round_index: int
    reason: str
    data: dict[str, Any]
    revision: Revision


@dataclass(frozen=True, slots=True)
class LoopAction:
    """Typed refinement action emitted by the diagnostic policy."""

    action_id: str
    action_kind: str
    rationale: str
    affected_entities: tuple[str, ...]
    expected_decision_loss_reduction: float
    estimated_cost: float
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WaveLoopConfig:
    max_rounds: int = 1
    max_actions: int = 5
    particle_count: int = 128
    seed: int = 0
    started_at: str = "1970-01-01T00:00:00Z"


@dataclass(frozen=True, slots=True)
class ParticleSurface:
    """Bounded-memory weighted-particle representation of a single round."""

    surface_id: str
    surface_version: str
    semantics: str
    axes: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    log_weights: tuple[float, ...]
    marginals: Mapping[str, Mapping[str, float]]
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CompiledFactorIR:
    """Deterministic, dimensionless compile of a mapping formula."""

    ir_version: str
    mapping_id: str
    formula: str
    tree: ast.Expression
    referenced_variables: tuple[str, ...]

    def log_potential(self, values: Mapping[str, float]) -> float:
        """Evaluate the IR and return a dimensionless log-potential."""

        evaluated = _eval_tree(self.tree, values)
        if not math.isfinite(evaluated):
            raise _RestrictedFormulaError(f"evaluation produced non-finite value: {evaluated}")
        return evaluated


# ---------------------------------------------------------------------------
# Shared IR evaluation (used by both wave_loop_surface and factor_ir paths)
# ---------------------------------------------------------------------------


def _eval_tree(node: ast.AST, values: Mapping[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _eval_tree(node.body, values)
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise _RestrictedFormulaError(f"variable {node.id} is missing from the particle")
        value = values[node.id]
        if not math.isfinite(value):
            raise _RestrictedFormulaError(f"variable {node.id} produced non-finite value")
        return float(value)
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_tree(node.operand, values)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
    if isinstance(node, ast.BinOp):
        left = _eval_tree(node.left, values)
        right = _eval_tree(node.right, values)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise _RestrictedFormulaError("division by zero is not allowed")
            return left / right
    raise _RestrictedFormulaError(f"unsupported node: {type(node).__name__}")


def _parse_restricted_formula(formula: str) -> tuple[ast.Expression, tuple[str, ...]]:
    if not isinstance(formula, str) or not formula.strip():
        raise _RestrictedFormulaError("formula is required")
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise _RestrictedFormulaError("formula must be a valid Python expression") from exc
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise _RestrictedFormulaError(
                "factor IR only supports names, numbers, parentheses, +, -, *, /"
            )
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise _RestrictedFormulaError("factor IR constants must be numeric")
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                raise _RestrictedFormulaError("factor IR constants must be finite")
        if isinstance(node, ast.Name):
            if not node.id.isidentifier():
                raise _RestrictedFormulaError(f"variable name {node.id!r} is not a valid identifier")
            if node.id not in names:
                names.append(node.id)
    return tree, tuple(names)


def compile_factor_ir(mapping: Mapping[str, Any]) -> CompiledFactorIR:
    """Compile one mapping spec to a restricted IR with explicit dimensionless contract."""

    tree, names = _parse_restricted_formula(str(mapping["formula"]))
    return CompiledFactorIR(
        ir_version=FACTOR_IR_VERSION,
        mapping_id=str(mapping["mapping_id"]),
        formula=str(mapping["formula"]),
        tree=tree,
        referenced_variables=names,
    )


__all__ = [
    "ACTION_KINDS",
    "CompiledFactorIR",
    "FACTOR_IR_VERSION",
    "JOINT_SCHEMA_VERSION",
    "LoopAction",
    "PARTICLE_SURFACE_VERSION",
    "ParticleSurface",
    "WAVE_ACTIVATION_STATES",
    "WAVE_CHECKPOINT_VERSION",
    "WAVE_LEDGER_VERSION",
    "WAVE_LOOP_RESULT_VERSION",
    "WAVE_TERMINAL_STATES",
    "WaveEvent",
    "WaveLoopConfig",
    "WaveLoopError",
    "_ALLOWED_AST_NODES",
    "_RestrictedFormulaError",
    "_eval_tree",
    "_parse_restricted_formula",
    "compile_factor_ir",
]

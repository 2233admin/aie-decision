"""Mechanical single-variable ablation for arithmetic Fermi formulas.

This module does not guess whether a variable is semantically dispensable.  It
only proposes transformations that can be proved from the formula's AST: a
variable occurring exactly once may be removed when it is itself an operand of
addition or multiplication.  Evaluation decides whether the resulting candidate
still meets the target interval.
"""

from __future__ import annotations

import ast
import copy
import hashlib
from dataclasses import dataclass

_ALLOWED_NODES = (
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


@dataclass(frozen=True, slots=True)
class AblationCandidate:
    """A mechanically generated child plus enough information to restore it."""

    candidate_id: str
    parent_candidate_id: str
    parent_formula: str
    formula: str
    removed_variable: str
    operation: str

    @property
    def restore_formula(self) -> str:
        """Return the exact, non-normalized formula supplied by the parent."""

        return self.parent_formula


@dataclass(frozen=True, slots=True)
class AblationRejection:
    """An explicit reason why a named variable could not be safely removed."""

    parent_candidate_id: str
    parent_formula: str
    variable: str
    reason: str


@dataclass(frozen=True, slots=True)
class AblationResult:
    candidates: tuple[AblationCandidate, ...]
    rejections: tuple[AblationRejection, ...]


def _parse_formula(formula: str) -> ast.Expression:
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("formula is required")
    try:
        tree = ast.parse(formula.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError("formula must be a valid arithmetic expression") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                "formula supports only names, numbers, parentheses, +, -, *, and /"
            )
        if isinstance(node, ast.Constant) and (
            isinstance(node.value, bool) or not isinstance(node.value, (int, float))
        ):
            raise ValueError("formula constants must be numbers")
    return tree


def _name_sites(
    node: ast.AST,
    parent: ast.AST | None = None,
    parent_field: str | None = None,
    parent_path: tuple[str, ...] = (),
) -> list[tuple[ast.Name, ast.AST | None, str | None, tuple[str, ...]]]:
    sites: list[tuple[ast.Name, ast.AST | None, str | None, tuple[str, ...]]] = []
    if isinstance(node, ast.Name):
        sites.append((node, parent, parent_field, parent_path))
    for field, child in ast.iter_fields(node):
        if isinstance(child, ast.AST):
            sites.extend(_name_sites(child, node, field, parent_path + (field,)))
    return sites


def _node_at_path(root: ast.AST, path: tuple[str, ...]) -> ast.AST:
    node = root
    for field in path:
        child = getattr(node, field)
        if not isinstance(child, ast.AST):
            raise ValueError("invalid AST path")
        node = child
    return node


def _candidate_id(parent_id: str, formula: str, variable: str) -> str:
    digest = hashlib.sha256(f"{parent_id}\0{formula}\0{variable}".encode()).hexdigest()[
        :12
    ]
    return f"{parent_id}-ablate-{variable}-{digest}"


def plan_variable_ablations(
    parent_formula: str,
    *,
    parent_candidate_id: str,
) -> AblationResult:
    """Generate every mechanically safe single-variable ablation.

    A variable is eligible only when it occurs once and is the direct left or
    right operand of ``+`` or ``*``.  The containing operator is replaced by its
    other operand in a copied full expression, so all surrounding formula context
    is preserved.  Every ineligible variable receives an explicit rejection.
    """

    if not isinstance(parent_candidate_id, str) or not parent_candidate_id.strip():
        raise ValueError("parent_candidate_id is required")
    tree = _parse_formula(parent_formula)
    sites = _name_sites(tree)
    by_name: dict[
        str, list[tuple[ast.Name, ast.AST | None, str | None, tuple[str, ...]]]
    ] = {}
    order: list[str] = []
    for site in sites:
        name = site[0].id
        if name not in by_name:
            by_name[name] = []
            order.append(name)
        by_name[name].append(site)

    candidates: list[AblationCandidate] = []
    rejections: list[AblationRejection] = []
    for variable in order:
        occurrences = by_name[variable]
        reason: str | None = None
        if len(occurrences) != 1:
            reason = "variable_occurs_multiple_times"
        else:
            _name, parent, parent_field, parent_path = occurrences[0]
            if not isinstance(parent, ast.BinOp) or parent_field not in {
                "left",
                "right",
            }:
                reason = "variable_is_not_a_direct_term_or_factor"
            elif not isinstance(parent.op, (ast.Add, ast.Mult)):
                reason = "operator_is_not_safely_ablatable"

        if reason is not None:
            rejections.append(
                AblationRejection(parent_candidate_id, parent_formula, variable, reason)
            )
            continue

        assert isinstance(parent, ast.BinOp)
        assert parent_field in {"left", "right"}
        copied = copy.deepcopy(tree)
        copied_parent = _node_at_path(copied, parent_path[:-1])
        assert isinstance(copied_parent, ast.BinOp)
        survivor_field = "right" if parent_field == "left" else "left"
        survivor = copy.deepcopy(getattr(copied_parent, survivor_field))

        if len(parent_path) == 1:
            copied.body = survivor
        else:
            container = _node_at_path(copied, parent_path[:-2])
            setattr(container, parent_path[-2], survivor)
        ast.fix_missing_locations(copied)

        if not any(isinstance(node, ast.Name) for node in ast.walk(copied)):
            rejections.append(
                AblationRejection(
                    parent_candidate_id,
                    parent_formula,
                    variable,
                    "would_remove_last_variable",
                )
            )
            continue

        operation = (
            "addition_term"
            if isinstance(parent.op, ast.Add)
            else "multiplication_factor"
        )
        candidates.append(
            AblationCandidate(
                candidate_id=_candidate_id(
                    parent_candidate_id, parent_formula, variable
                ),
                parent_candidate_id=parent_candidate_id,
                parent_formula=parent_formula,
                formula=ast.unparse(copied),
                removed_variable=variable,
                operation=operation,
            )
        )

    return AblationResult(tuple(candidates), tuple(rejections))

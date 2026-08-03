"""Track A contracts for the recursive Fermi decomposition runtime.

This module is domain-neutral and pure standard library.  It defines the
typed records that every higher-level Track B/C component may consume:

* Question roots derived from a raw prompt without a user formula.
* Scopes that pin every quantity to a population, geography, and time basis.
* Nodes, relationships, and expansions used to reify recursive splits.
* Atomic claims that bind a leaf to a concrete measurement procedure.
* Branches and alternatives that let the same target re-frame itself.
* Gaps and rejections that keep the search honest about what is unresolved.
* Append-only :class:`ActionRecord` envelopes for every state transition.

It also enforces the structural safeguards required by the product direction
document:

* A restricted expression parser that rejects any non-arithmetic call
  (``eval`` and arbitrary calls are out of the question) and a recursive
  evaluator that returns the dimensions of the result.
* Dependency completeness and dimensional closure used to reject
  incompatible expansions and unit mismatches without referring to any
  fixed domain knowledge.
* Canonical signatures that detect algebraically redundant alternatives so
  the search does not double-count equivalent decompositions.

Nothing in this module performs probability propagation, trajectory
recording, scheduling, prompt construction, or model calls.  Those concerns
live in Track B and Track C, both of which must consume the contracts from
this module verbatim.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, replace
from enum import StrEnum
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class QuestionStatus(StrEnum):
    """Lifecycle states of a question root."""

    OPEN = "open"
    EXPANDED = "expanded"
    ATOMIC_LEAF = "atomic_leaf"
    BLOCKED = "blocked"
    ANSWERED = "answered"


class NodeStatus(StrEnum):
    """Lifecycle states of a single node inside a decomposition tree."""

    OPEN = "open"
    EXPANDED = "expanded"
    ATOMIC_LEAF = "atomic_leaf"
    UNRESOLVED = "unresolved"
    REJECTED = "rejected"
    PRUNED = "pruned"


class NodeRole(StrEnum):
    """Why a node exists in the tree."""

    TARGET = "target"
    CHILD = "child"
    ALTERNATIVE = "alternative"
    ATOM_CANDIDATE = "atom_candidate"


class ObservationKind(StrEnum):
    """How a quantity in an atomic claim is known to the analyst."""

    OBSERVED = "observed"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


class MeasurementKind(StrEnum):
    """The structured operation used to obtain an atomic quantity."""

    DIRECT_OBSERVATION = "direct_observation"
    RECORD_LOOKUP = "record_lookup"
    COUNT = "count"
    TIMED_MEASUREMENT = "timed_measurement"
    INSTRUMENT_MEASUREMENT = "instrument_measurement"
    DERIVED_PROXY = "derived_proxy"


class GapKind(StrEnum):
    """Why a frontier node or root cannot be promoted yet."""

    INCOMPLETE_ROOT = "incomplete_root"
    MISSING_RELATIONSHIP = "missing_relationship"
    UNRESOLVED_NODE = "unresolved_node"
    ATOM_REJECTED = "atom_rejected"
    UNKNOWN_LEAF = "unknown_leaf"
    UNIT_MISMATCH = "unit_mismatch"
    INSUFFICIENT_DEPENDENCIES = "insufficient_dependencies"
    REDUNDANT_ALTERNATIVE = "redundant_alternative"


class ActionKind(StrEnum):
    """Exhaustive list of structural mutations exposed by the runtime."""

    CREATE_QUESTION = "create_question"
    REGISTER_NODE = "register_node"
    EXPAND = "expand"
    PROPOSE_ATOM = "propose_atom"
    PROPOSE_ALTERNATIVE = "propose_alternative"
    PRUNE = "prune"
    ACTIVATE_BRANCH = "activate_branch"
    REGISTER_GAP = "register_gap"
    EXPORT = "export"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FermiContractError(ValueError):
    """Raised when a structural contract cannot be admitted to the tree."""


class RestrictedExpressionError(FermiContractError):
    """Raised when an expression leaves the allowed arithmetic vocabulary."""


class DimensionalError(FermiContractError):
    """Raised when a relationship cannot produce its declared parent unit."""


class AtomicClaimError(FermiContractError):
    """Raised when an atomic claim remains abstract instead of measurable."""


class RedundancyReason(str):
    """Human-readable reason why an alternative was declared redundant."""

    __slots__ = ()


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------


# Symbolic base units recognised by the default dimension registry.  Adding
# more only requires a label; the registry stores integer exponents so
# multiplication, division, and integer powers compose cleanly.
DEFAULT_UNIT_SYMBOLS: frozenset[str] = frozenset(
    {
        # SI bases & composites the test suite will exercise.
        "kg",
        "g",
        "t",
        "m",
        "cm",
        "km",
        "mm",
        "s",
        "min",
        "h",
        "day",
        "year",
        "A",
        "K",
        "mol",
        "cd",
        "N",
        "J",
        "W",
        "Pa",
        "Hz",
        "C",
        "V",
        "ohm",
        "F",
        # Domain-neutral counters used by Fermi identities.
        "person",
        "people",
        "household",
        "firm",
        "transaction",
        "event",
        "order",
        "call",
        "visit",
        "trip",
        "tonne",
        "litre",
        "liter",
        "gallon",
        # Money / accounts — domain-neutral.
        "USD",
        "CNY",
        "EUR",
        "JPY",
        "GBP",
        "dollar",
        "dollars",
        "yuan",
        "euro",
        "pound",
        # Generic dimensionless / rate denominators.
        "ratio",
        "share",
        "rate",
        # Conventional dimensionless numerator.
        "1",
    }
)


# ---------------------------------------------------------------------------
# Compound unit arithmetic
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompoundUnit:
    """A sparse mapping from unit label to integer exponent."""

    exponents: Mapping[str, int] = field(default_factory=dict)

    @property
    def is_dimensionless(self) -> bool:
        return not self.exponents or all(value == 0 for value in self.exponents.values())

    def to_canonical(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted((label, value) for label, value in self.exponents.items() if value != 0))

    def __str__(self) -> str:  # pragma: no cover - convenience for debugging
        parts = self.to_canonical()
        if not parts:
            return "dimensionless"
        return ", ".join(f"{label}^{exp}" for label, exp in parts)


DIMENSIONLESS = CompoundUnit()


def _normalise_exponents(exponents: Mapping[str, int]) -> dict[str, int]:
    return {label: value for label, value in exponents.items() if value}


def _unit_parser_error(label: str, unit_string: str) -> "RestrictedExpressionError":
    return RestrictedExpressionError(f"{label}: cannot parse {unit_string!r} as a compound unit")


def parse_compound_unit(unit: Any, *, registry: frozenset[str] = DEFAULT_UNIT_SYMBOLS) -> CompoundUnit:
    """Parse a textual compound unit expression into a :class:`CompoundUnit`.

    Accepts ``"/"`` for division, ``"*"`` for multiplication, and ``"^N"``
    or ``"**N"`` for integer powers.  Whitespace is ignored.  Each ``/``
    introduces a single denominator factor; subsequent ``*`` factors are
    added to the numerator again, matching the conventional reading of
    compound units such as ``kg*m/s^2`` (force = mass * acceleration).
    Parens may group sub-expressions.  ``""`` and ``None`` map to
    dimensionless.
    """

    if unit is None or unit == "":
        return DIMENSIONLESS
    if not isinstance(unit, str):
        raise _unit_parser_error("unit", repr(unit))
    cleaned = unit.replace(" ", "")
    if not cleaned:
        return DIMENSIONLESS

    segments = _split_compound_unit(cleaned, "/")
    exponents: dict[str, int] = {}
    for index, segment in enumerate(segments):
        sign = 1 if index == 0 else -1
        for factor_str in _split_compound_unit(segment, "*"):
            factors = _parse_unit_factor(factor_str, registry)
            for sub_label, sub_exp in factors:
                exponents[sub_label] = exponents.get(sub_label, 0) + sign * sub_exp
    return CompoundUnit(_normalise_exponents(exponents))


def _split_compound_unit(text: str, separator: str) -> list[str]:
    if not text:
        return [""]
    pieces: list[str] = []
    depth = 0
    buffer = ""
    for char in text:
        if char == "(":
            depth += 1
            buffer += char
        elif char == ")":
            depth -= 1
            buffer += char
        elif char == separator and depth == 0:
            pieces.append(buffer)
            buffer = ""
        else:
            buffer += char
    pieces.append(buffer)
    return pieces


def _parse_unit_factor(token: str, registry: frozenset[str]) -> list[tuple[str, int]]:
    """Parse a single atomic unit factor (which may itself contain ``*``)."""

    if not token:
        raise _unit_parser_error("unit", token)
    caret = token.find("^")
    stars_index = token.find("**")
    exp_index = -1
    if caret >= 0 and stars_index >= 0:
        exp_index = min(caret, stars_index)
    elif caret >= 0:
        exp_index = caret
    else:
        exp_index = stars_index
    if exp_index >= 0:
        label = token[:exp_index]
        exponent_text = token[exp_index + 1 :] if caret == exp_index else token[exp_index + 2 :]
        try:
            exp = int(exponent_text)
        except ValueError as exc:
            raise _unit_parser_error("unit", token) from exc
    else:
        label = token
        exp = 1
    if not label:
        raise _unit_parser_error("unit", token)
    if "(" in label:
        try:
            inner = label[label.index("(") : label.rindex(")") + 1]
            outer = label.replace(inner, "")
        except ValueError as exc:
            raise _unit_parser_error("unit", token) from exc
        if outer:
            raise _unit_parser_error("unit", token)
        factors = parse_compound_unit(inner[1:-1], registry=registry)
        return [(sub_label, sub_exp * exp) for sub_label, sub_exp in factors.to_canonical()]
    # ``1`` is the conventional dimensionless numerator token; it contributes
    # nothing to the exponent map so rate expressions like ``1/person`` parse
    # cleanly into pure ``person`` dimensions.
    if label == "1":
        return []
    if registry and label not in registry:
        raise _unit_parser_error("unit", token)
    return [(label, exp)]


def _parse_unit_token(token: str, registry: frozenset[str]) -> tuple[str, int]:
    """Backwards-compatible alias used by historical fixtures."""

    factors = _parse_unit_factor(token, registry)
    if len(factors) != 1:
        raise _unit_parser_error("unit", token)
    return factors[0]


def multiply_units(*units: CompoundUnit) -> CompoundUnit:
    combined: dict[str, int] = {}
    for unit in units:
        for label, exp in unit.exponents.items():
            combined[label] = combined.get(label, 0) + exp
    return CompoundUnit(_normalise_exponents(combined))


def divide_units(numerator: CompoundUnit, denominator: CompoundUnit) -> CompoundUnit:
    combined: dict[str, int] = dict(numerator.exponents)
    for label, exp in denominator.exponents.items():
        combined[label] = combined.get(label, 0) - exp
    return CompoundUnit(_normalise_exponents(combined))


def power_units(unit: CompoundUnit, exponent: int) -> CompoundUnit:
    return CompoundUnit(_normalise_exponents({label: exp * exponent for label, exp in unit.exponents.items()}))


def units_close(a: CompoundUnit, b: CompoundUnit) -> bool:
    """Return True iff ``a`` and ``b`` describe the same compound dimension."""

    return a.to_canonical() == b.to_canonical()


# ---------------------------------------------------------------------------
# Restricted arithmetic expression parser
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
    ast.Pow,
    ast.UAdd,
    ast.USub,
)


@dataclass(frozen=True, slots=True)
class RestrictedExpression:
    """A pre-parsed restricted arithmetic expression bound to variable names."""

    source: str
    tree: ast.Expression
    variables: tuple[str, ...]
    signature: str
    dimensions: CompoundUnit

    @property
    def text(self) -> str:
        return self.source


def _dimension_key(unit: CompoundUnit) -> str:
    if unit.is_dimensionless:
        return "1"
    return "*".join(f"{label}^{exp}" for label, exp in unit.to_canonical())


def _collect_expression_variables(tree: ast.AST) -> list[str]:
    """Collect variable names in the order they first appear in the source.

    ``ast.walk`` traverses the tree in breadth-first order, which diverges
    from the textual source order whenever a node holds a sub-expression
    that appears before its sibling at the same depth (e.g. ``b * a``
    would otherwise collect ``b`` first and silently rebind the dimension
    of ``a`` to the unit of ``b``).  Walking the tree with a depth-first
    visitor preserves the textual order in which the variables appear,
    so unit bindings line up with the source.
    """

    seen: list[str] = []
    for node in _depth_first(tree):
        if isinstance(node, ast.Name) and node.id not in seen:
            seen.append(node.id)
    return seen


def _depth_first(tree: ast.AST):
    """Yield nodes from ``tree`` in depth-first (source) order."""

    yield tree
    for child in ast.iter_child_nodes(tree):
        yield from _depth_first(child)


def parse_restricted_expression(
    expression: Any,
    *,
    variable_units: Mapping[str, CompoundUnit] | None = None,
    registry: frozenset[str] = DEFAULT_UNIT_SYMBOLS,
) -> RestrictedExpression:
    """Parse ``expression`` into a restricted arithmetic tree.

    Only the nodes listed in ``_ALLOWED_AST_NODES`` are accepted.  Any other
    syntax — function calls, comprehensions, attribute access, boolean
    operators, comparisons, slicing — triggers a
    :class:`RestrictedExpressionError`.  Numeric constants must be finite,
    non-boolean values.  When ``variable_units`` is provided the runtime
    also computes the projected dimensions of the expression and the
    :class:`RestrictedExpression` carries them with it for downstream
    dimensional closure checks.
    """

    if not isinstance(expression, str) or not expression.strip():
        raise RestrictedExpressionError("expression is required")
    # ``^`` is the conventional mathematical power symbol.  Translate it to
    # ``**`` before parsing because Python's grammar would otherwise treat
    # ``a^2`` as a bitwise xor.
    normalised = expression.replace("^", "**")
    try:
        tree = ast.parse(normalised, mode="eval")
    except SyntaxError as exc:
        raise RestrictedExpressionError(
            f"expression must be valid arithmetic: {exc.msg}"
        ) from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise RestrictedExpressionError(
                f"expression supports only numeric constants, names, parentheses, "
                f"+, -, *, /, and ^; rejected {type(node).__name__}"
            )
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RestrictedExpressionError("constants must be numeric and non-boolean")
            if not isfinite(float(value)):
                raise RestrictedExpressionError("constants must be finite")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise RestrictedExpressionError(
                f"expression names cannot be dunder identifiers: {node.id}"
            )

    variables = tuple(_collect_expression_variables(tree))
    if variable_units is None:
        dimensions = DIMENSIONLESS
    else:
        missing = [name for name in variables if name not in variable_units]
        if missing:
            raise RestrictedExpressionError(
                "expression references undeclared variables: " + ", ".join(sorted(missing))
            )
        dimensions = _evaluate_dimensions(tree, variable_units, registry)
    signature = _canonical_signature(tree)
    return RestrictedExpression(
        source=normalised.strip(),
        tree=tree,
        variables=variables,
        signature=signature,
        dimensions=dimensions,
    )


def _evaluate_dimensions(
    node: ast.AST,
    variable_units: Mapping[str, CompoundUnit],
    registry: frozenset[str],
) -> CompoundUnit:
    if isinstance(node, ast.Expression):
        return _evaluate_dimensions(node.body, variable_units, registry)
    if isinstance(node, ast.Name):
        try:
            return variable_units[node.id]
        except KeyError as exc:
            raise RestrictedExpressionError(
                f"unknown variable in dimension evaluation: {node.id}"
            ) from exc
    if isinstance(node, ast.Constant):
        return DIMENSIONLESS
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_dimensions(node.operand, variable_units, registry)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return operand
    if isinstance(node, ast.BinOp):
        left = _evaluate_dimensions(node.left, variable_units, registry)
        right = _evaluate_dimensions(node.right, variable_units, registry)
        if isinstance(node.op, ast.Add) or isinstance(node.op, ast.Sub):
            if not units_close(left, right):
                raise RestrictedExpressionError(
                    "addition/subtraction requires matching dimensions: "
                    f"{_dimension_key(left)} vs {_dimension_key(right)}"
                )
            return left
        if isinstance(node.op, ast.Mult):
            return multiply_units(left, right)
        if isinstance(node.op, ast.Div):
            return divide_units(left, right)
        if isinstance(node.op, ast.Pow):
            if not isinstance(node.right, ast.Constant) or not isinstance(node.right.value, int):
                raise RestrictedExpressionError("exponent must be an integer constant")
            return power_units(left, int(node.right.value))
    raise RestrictedExpressionError(f"unsupported expression node: {type(node).__name__}")


def _canonical_signature(node: ast.AST) -> str:
    """Canonical string form used to detect algebraically redundant rewrites.

    Multiplication and addition become commutative: children are sorted by
    their canonical form before joining.  Subtraction and division preserve
    order because they are not commutative.  Integer exponent constants are
    rendered as ``^N``; everything else inherits Python's ``ast.dump``
    canonicalisation for safety against obscure edge cases.
    """

    if isinstance(node, ast.Expression):
        return _canonical_signature(node.body)
    if isinstance(node, ast.Name):
        return f"name:{node.id}"
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, float) and value.is_integer():
            return f"const:{int(value)}"
        if isinstance(value, int):
            return f"const:{value}"
        return f"const:{value!r}"
    if isinstance(node, ast.UnaryOp):
        inner = _canonical_signature(node.operand)
        op = "+" if isinstance(node.op, ast.UAdd) else "-"
        return f"uop({op},{inner})"
    if isinstance(node, ast.BinOp):
        left = _canonical_signature(node.left)
        right = _canonical_signature(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return "(" + "+".join(sorted([left, right])) + ")"
        if isinstance(op, ast.Mult):
            return "(" + "*".join(sorted([left, right])) + ")"
        if isinstance(op, ast.Sub):
            return f"({left}-{right})"
        if isinstance(op, ast.Div):
            return f"({left}/{right})"
        if isinstance(op, ast.Pow):
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
                return f"pow({left},{node.right.value})"
            return f"pow({left},{ast.dump(node.right)})"
    return ast.dump(node)


def _expression_variables_match(left: RestrictedExpression, right: RestrictedExpression) -> bool:
    return sorted(left.variables) == sorted(right.variables)


def expressions_are_equivalent(left: RestrictedExpression, right: RestrictedExpression) -> bool:
    """Return True when ``left`` and ``right`` reference the same variables and
    share a commutative-canonicalised restricted-form signature.
    """

    return _expression_variables_match(left, right) and left.signature == right.signature


# ---------------------------------------------------------------------------
# Restricted arithmetic evaluator (numeric)
# ---------------------------------------------------------------------------


def evaluate_restricted_expression(
    expression: RestrictedExpression,
    values: Mapping[str, float],
) -> float:
    """Evaluate a :class:`RestrictedExpression` against numeric ``values``.

    Every variable referenced by the expression must resolve to a finite
    numeric value.  Constant leaves must already have been admitted by
    :func:`parse_restricted_expression`.  No callables, no attribute
    access, no sequence operations.
    """

    try:
        return _evaluate_node(expression.tree.body, values)
    except KeyError as exc:
        raise RestrictedExpressionError(f"missing value for variable: {exc.args[0]}") from exc


def _evaluate_node(node: ast.AST, values: Mapping[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body, values)
    if isinstance(node, ast.Name):
        value = values[node.id]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RestrictedExpressionError(f"non-numeric value for {node.id}")
        result = float(value)
        if not isfinite(result):
            raise RestrictedExpressionError(f"non-finite value for {node.id}")
        return result
    if isinstance(node, ast.Constant):
        result = float(node.value)
        if not isfinite(result):
            raise RestrictedExpressionError("non-finite constant")
        return result
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_node(node.operand, values)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left, values)
        right = _evaluate_node(node.right, values)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise RestrictedExpressionError("division by zero")
            return left / right
        if isinstance(node.op, ast.Pow):
            if not isinstance(node.right, ast.Constant) or not isinstance(node.right.value, int):
                raise RestrictedExpressionError("exponent must be an integer constant")
            return left ** int(node.right.value)
    raise RestrictedExpressionError(f"unsupported expression node: {type(node).__name__}")


# ---------------------------------------------------------------------------
# Scope, question, and node records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Scope:
    """The universe over which a quantity should be summed, averaged, etc."""

    population: str | None = None
    geography: str | None = None
    time_window: str | None = None
    temporal_basis: str | None = None
    extra: Mapping[str, str] = field(default_factory=dict)

    def is_well_defined(self) -> bool:
        """Return True when at least one of population/geography anchors the scope."""

        return bool((self.population and self.population.strip()) or (self.geography and self.geography.strip()))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "population": self.population,
            "geography": self.geography,
            "time_window": self.time_window,
            "temporal_basis": self.temporal_basis,
        }
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


@dataclass(frozen=True, slots=True)
class Question:
    """A raw quantitative question that begins a decomposition."""

    question_id: str
    question: str
    target_subject: str
    target_measure: str
    unit: str
    time_basis: str
    scope: Scope
    status: QuestionStatus = QuestionStatus.OPEN
    decision_use: str | None = None
    acceptable_width: str | None = None
    unresolved_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise FermiContractError("question_id is required")
        if not self.question.strip():
            raise FermiContractError("question text is required")
        if not self.target_subject.strip():
            raise FermiContractError("target_subject is required")
        if not self.target_measure.strip():
            raise FermiContractError("target_measure is required")
        if self.unit is None or not str(self.unit).strip():
            raise FermiContractError("unit is required")
        if not self.time_basis.strip():
            raise FermiContractError("time_basis is required")
        if not isinstance(self.scope, Scope):
            raise FermiContractError("scope must be a Scope instance")

    @property
    def target_unit(self) -> CompoundUnit:
        return parse_compound_unit(self.unit)

    def is_minimally_complete(self) -> bool:
        """A raw question is minimally complete when scope anchors reality and no field is unresolved."""

        if self.unresolved_fields:
            return False
        return self.scope.is_well_defined()

    def with_unresolved(self, fields: Iterable[str]) -> "Question":
        new_fields = tuple(sorted(set(self.unresolved_fields) | set(fields)))
        return replace(self, unresolved_fields=new_fields)


@dataclass(frozen=True, slots=True)
class Node:
    """A single node in the decomposition graph."""

    node_id: str
    label: str
    role: NodeRole
    status: NodeStatus = NodeStatus.OPEN
    parent_id: str | None = None
    unit: str | None = None
    scope: Scope | None = None
    description: str = ""
    mechanism: str = ""
    expansion_id: str | None = None

    def __post_init__(self) -> None:
        if not self.node_id.strip():
            raise FermiContractError("node_id is required")
        if not self.label.strip():
            raise FermiContractError("label is required for node " + self.node_id)
        if self.role is NodeRole.TARGET and self.parent_id is not None:
            raise FermiContractError("the target node cannot have a parent_id")
        if self.unit is not None:
            parse_compound_unit(self.unit)


# ---------------------------------------------------------------------------
# Relationship, expansion, branch, gap, action
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Relationship:
    """How a set of children re-derives a parent node's quantity."""

    relationship_id: str
    parent_node_id: str
    parent_unit: str
    expression: str
    child_node_ids: tuple[str, ...]
    child_units: tuple[str, ...]
    rationale: str
    expression_ast: RestrictedExpression | None = None

    def __post_init__(self) -> None:
        if not self.relationship_id.strip():
            raise FermiContractError("relationship_id is required")
        if not self.parent_node_id.strip():
            raise FermiContractError("parent_node_id is required")
        if not self.parent_unit.strip():
            raise FermiContractError("parent_unit is required")
        if not self.expression.strip():
            raise FermiContractError("expression is required")
        if not self.child_node_ids:
            raise FermiContractError("relationship must declare at least one child node")
        if len(self.child_node_ids) != len(self.child_units):
            raise FermiContractError("child_node_ids and child_units must be parallel tuples")
        if not self.rationale.strip():
            raise FermiContractError("rationale is required to explain the split")


@dataclass(frozen=True, slots=True)
class Expansion:
    """An applied relationship that introduces a set of children to the tree."""

    expansion_id: str
    target_node_id: str
    relationship_id: str
    parent_unit: str
    projected_unit: str
    child_node_ids: tuple[str, ...]
    rationale: str
    is_alternative: bool = False
    alternative_of_expansion_id: str | None = None
    is_redundant: bool = False
    redundancy_reason: str | None = None

    def describe(self) -> str:
        if self.is_alternative:
            base = "alternative"
            if self.is_redundant:
                base = "redundant alternative"
            return (
                f"{base} expansion {self.expansion_id} of node {self.target_node_id} "
                f"via relationship {self.relationship_id}"
            )
        return f"expansion {self.expansion_id} of node {self.target_node_id}"


@dataclass(frozen=True, slots=True)
class AtomicClaim:
    """An operational measurement attached to a leaf node.

    Operational atomicity is described through *structured* fields, not
    through lexical detection against any particular natural language.  The
    required fields are:

    * :attr:`target_object` — the concrete real-world object, event, or
      population being measured.
    * :attr:`unit` — the compound unit of the resulting quantity.
    * :attr:`scope` — the population, geography, and time basis that pin
      the quantity to a universe.
    * :attr:`measurement_kind` — one of the explicit :class:`MeasurementKind`
      operations (count, lookup, observation, etc.).
    * :attr:`source` — the data source, instrument, or registry that
      produces the measurement (e.g. "ACS B08006", "smart-meter log",
      "field tally sheet").
    * :attr:`procedure` — a non-empty structured description of the
      procedural steps that produce the quantity.

    No string in this record is matched against an English or
    domain-specific vocabulary list.  Acceptance depends on field
    presence, enum membership, and the caller's ability to articulate the
    measurement honestly; evidence gaps remain visible through
    :attr:`observation_kind` and :attr:`assumption_notes`.
    """

    node_id: str
    target_object: str
    unit: str
    scope: Scope
    measurement_kind: MeasurementKind
    source: str
    procedure: str
    time_basis: str = ""
    observation_kind: ObservationKind = ObservationKind.UNKNOWN
    assumption_notes: str = ""

    def __post_init__(self) -> None:
        if not self.node_id.strip():
            raise AtomicClaimError("node_id is required")
        if not isinstance(self.scope, Scope):
            raise AtomicClaimError("scope must be a Scope instance")
        if not isinstance(self.measurement_kind, MeasurementKind):
            raise AtomicClaimError("measurement_kind must be a MeasurementKind value")
        if not isinstance(self.observation_kind, ObservationKind):
            raise AtomicClaimError("observation_kind must be an ObservationKind value")
        parse_compound_unit(self.unit)

    def has_structured_measurement(self) -> bool:
        """Return True when every required structured field is populated.

        The check is purely structural.  It does not attempt to interpret
        the meaning of any field, nor does it look for keywords in any
        human language.
        """

        return bool(
            self.target_object.strip()
            and str(self.unit).strip()
            and self.scope.is_well_defined()
            and self.source.strip()
            and self.procedure.strip()
        )


# Each ``MeasurementKind`` declares the structured fields that must be
# populated for the claim to be admitted.  The check is purely structural:
# it never inspects the textual content of a field for keywords.
_MEASUREMENT_KIND_REQUIRED_FIELDS: dict[MeasurementKind, tuple[str, ...]] = {
    MeasurementKind.DIRECT_OBSERVATION: ("target_object", "unit", "scope", "source", "procedure"),
    MeasurementKind.RECORD_LOOKUP: ("target_object", "unit", "scope", "source", "procedure"),
    MeasurementKind.COUNT: ("target_object", "unit", "scope", "source", "procedure"),
    MeasurementKind.TIMED_MEASUREMENT: ("target_object", "unit", "scope", "source", "procedure", "time_basis"),
    MeasurementKind.INSTRUMENT_MEASUREMENT: ("target_object", "unit", "scope", "source", "procedure"),
    MeasurementKind.DERIVED_PROXY: (
        "target_object",
        "unit",
        "scope",
        "source",
        "procedure",
        "assumption_notes",
    ),
}


def validate_atomic_claim(
    claim: AtomicClaim,
    *,
    registry: frozenset[str] = DEFAULT_UNIT_SYMBOLS,
) -> tuple[str, ...]:
    """Return an empty tuple when ``claim`` is structurally measurable.

    Acceptance is determined by the :class:`MeasurementKind` enum and the
    presence of every required structured field.  The function never
    inspects the textual content of any field for English or
    domain-specific keywords; a Chinese, Arabic, or any other language
    description is admitted on the same structural basis as English.

    Honest evidence and assumption gaps are *not* treated as fatal: a
    ``DERIVED_PROXY`` claim may carry populated ``assumption_notes`` to
    document its dependence on an external mapping, and an
    :attr:`observation_kind` of ``UNKNOWN`` is permitted.  The runtime
    surfaces those facts through other channels (gaps, exports) rather
    than here.
    """

    errors: list[str] = []
    if not isinstance(claim, AtomicClaim):
        errors.append("claim must be an AtomicClaim instance")
        return tuple(errors)

    required = _MEASUREMENT_KIND_REQUIRED_FIELDS.get(claim.measurement_kind, ())
    if not claim.target_object.strip():
        errors.append("target_object is required")
    if not claim.unit or not str(claim.unit).strip():
        errors.append("unit is required")
    else:
        try:
            parse_compound_unit(claim.unit, registry=registry)
        except RestrictedExpressionError as exc:
            errors.append(f"unit is not a valid compound expression: {exc}")
    if not claim.scope.is_well_defined():
        errors.append("scope must declare either a population or a geography anchor")
    if not claim.source.strip():
        errors.append("source is required for measurement_kind=" + str(claim.measurement_kind))
    if not claim.procedure.strip():
        errors.append("procedure is required for measurement_kind=" + str(claim.measurement_kind))
    if "time_basis" in required and not claim.time_basis.strip():
        errors.append(
            f"time_basis is required for measurement_kind={claim.measurement_kind}"
        )
    if (
        claim.measurement_kind is MeasurementKind.DERIVED_PROXY
        and not claim.assumption_notes.strip()
    ):
        errors.append(
            "assumption_notes are required for measurement_kind=derived_proxy"
        )
    if "time_basis" not in required and not claim.time_basis.strip():
        # Optional for non-temporal measurement kinds, but record its absence
        # when the field is later required by downstream code.
        pass
    return tuple(errors)


@dataclass(frozen=True, slots=True)
class Branch:
    """A named lineage of expansion choices inside a decomposition tree."""

    branch_id: str
    root_question_id: str
    expansion_ids: tuple[str, ...]
    divergent_at_expansion_id: str | None = None
    note: str = ""

    def covers(self, expansion_id: str) -> bool:
        return expansion_id in self.expansion_ids


@dataclass(frozen=True, slots=True)
class Gap:
    """An unresolved obstacle that must remain visible to the AI."""

    gap_id: str
    kind: GapKind
    target: str
    explanation: str
    blocking: bool = True
    introduced_by_action_id: str | None = None

    def __post_init__(self) -> None:
        if not self.gap_id.strip():
            raise FermiContractError("gap_id is required")
        if not self.target.strip():
            raise FermiContractError("gap target is required")
        if not self.explanation.strip():
            raise FermiContractError("gap explanation is required")


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """Append-only record of a single structural mutation."""

    action_id: str
    kind: ActionKind
    payload: Mapping[str, Any]
    result_summary: str
    accepted: bool
    recorded_at: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "kind": str(self.kind),
            "payload": _stringify_mapping(self.payload),
            "result_summary": self.result_summary,
            "accepted": self.accepted,
            "recorded_at": self.recorded_at,
            "error": self.error,
        }


def _stringify_mapping(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _stringify_mapping(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_stringify_mapping(item) for item in value]
    if hasattr(value, "to_dict"):
        return _stringify_mapping(value.to_dict())
    if hasattr(value, "to_canonical"):
        return _stringify_mapping(value.to_canonical())
    if isinstance(value, StrEnum):
        return str(value)
    return value


# ---------------------------------------------------------------------------
# Dimensional closure
# ---------------------------------------------------------------------------


def check_dimensional_closure(
    parent_unit: str,
    expression: RestrictedExpression,
    child_units: Sequence[str],
    *,
    registry: frozenset[str] = DEFAULT_UNIT_SYMBOLS,
) -> tuple[bool, str]:
    """Return ``(True, projected_unit)`` when ``expression`` reproduces
    ``parent_unit`` against ``child_units``; otherwise ``(False, reason)``.

    The function rebinds the expression's variables to ``child_units``
    positionally so the projection can be checked even when the expression
    was parsed without explicit unit bindings.
    """

    if len(expression.variables) != len(child_units):
        return (
            False,
            "expression variables and child_units must align one-to-one",
        )
    bound = {
        variable: parse_compound_unit(unit, registry=registry)
        for variable, unit in zip(expression.variables, child_units)
    }
    try:
        projected = _evaluate_dimensions(expression.tree, bound, registry)
    except RestrictedExpressionError as exc:
        return False, str(exc)
    parsed_parent = parse_compound_unit(parent_unit, registry=registry)
    if projected.to_canonical() != parsed_parent.to_canonical():
        return (
            False,
            f"expression {expression.source!r} projects to "
            f"{_dimension_key(projected)} but parent unit is "
            f"{_dimension_key(parsed_parent)}",
        )
    return True, _dimension_key(parsed_parent)


def project_dimensional_closure(
    parent_unit: str,
    expression_source: str,
    child_units: Sequence[str],
    *,
    registry: frozenset[str] = DEFAULT_UNIT_SYMBOLS,
) -> RestrictedExpression:
    """Parse a restricted expression and verify it reproduces ``parent_unit``.

    The expression's variables bind positionally to ``child_units``.  A
    mismatched variable count, mismatched compound unit, or unknown label
    raises :class:`DimensionalError`.  Returned expression carries
    dimensions and canonical signature so downstream redundancy checks can
    reuse them.
    """

    parsed = parse_restricted_expression(expression_source, registry=registry)
    if len(parsed.variables) != len(child_units):
        raise DimensionalError(
            f"expression {expression_source!r} references "
            f"{len(parsed.variables)} variable(s) but {len(child_units)} child unit(s) "
            "were declared"
        )
    declared = {
        variable: parse_compound_unit(unit, registry=registry)
        for variable, unit in zip(parsed.variables, child_units)
    }
    try:
        rebuilt = _evaluate_dimensions(parsed.tree, declared, registry)
    except RestrictedExpressionError as exc:
        raise DimensionalError(f"expression fails dimensional closure: {exc}") from exc
    parsed_parent = parse_compound_unit(parent_unit, registry=registry)
    if rebuilt.to_canonical() != parsed_parent.to_canonical():
        raise DimensionalError(
            f"relationship projects to {_dimension_key(rebuilt)} but parent "
            f"declares {_dimension_key(parsed_parent)}"
        )
    return RestrictedExpression(
        source=parsed.source,
        tree=parsed.tree,
        variables=parsed.variables,
        signature=parsed.signature,
        dimensions=rebuilt,
    )


__all__ = [
    # Enums
    "QuestionStatus",
    "NodeStatus",
    "NodeRole",
    "ObservationKind",
    "MeasurementKind",
    "GapKind",
    "ActionKind",
    # Errors
    "FermiContractError",
    "RestrictedExpressionError",
    "DimensionalError",
    "AtomicClaimError",
    # Compound units
    "CompoundUnit",
    "DIMENSIONLESS",
    "parse_compound_unit",
    "multiply_units",
    "divide_units",
    "power_units",
    "units_close",
    # Restricted arithmetic
    "RestrictedExpression",
    "parse_restricted_expression",
    "evaluate_restricted_expression",
    "expressions_are_equivalent",
    "check_dimensional_closure",
    "project_dimensional_closure",
    # Records
    "Scope",
    "Question",
    "Node",
    "Relationship",
    "Expansion",
    "AtomicClaim",
    "Branch",
    "Gap",
    "ActionRecord",
    "validate_atomic_claim",
    # Registries (exported so callers can extend without re-importing)
    "DEFAULT_UNIT_SYMBOLS",
    "SCHEMA_VERSION",
]

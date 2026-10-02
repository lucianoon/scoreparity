"""Typed parity configuration.

A configuration declares, *before* looking at the results, what "equivalent" means for a given
change. It is usually stored in version control next to the code it protects.

Loading is strict: unknown keys are errors, because a misspelled gate silently disabled is
worse than no gate at all.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

from scoreparity.errors import ConfigError

CONFIG_VERSION = 1


@dataclass(frozen=True)
class Columns:
    id: str = "id"
    score: str = "score"
    reference_score: str | None = None
    candidate_score: str | None = None
    label: str | None = None
    truth: str | None = None  # class ground truth, categorical outputs
    replica: str | None = None  # several rows per id, one per replica (label outputs)

    @property
    def reference(self) -> str:
        return self.reference_score or self.score

    @property
    def candidate(self) -> str:
        return self.candidate_score or self.score


# Kinds of model output a comparison can handle. Each kind has its own alignment rules and
# its own set of applicable gates (see SUPPORTED_GATES below).
OUTPUT_TYPES: tuple[str, ...] = ("score", "label", "probabilities")


@dataclass(frozen=True)
class Normalize:
    """How raw class labels (for example, LLM answers) are turned into classes.

    Labels are always stripped. Then, in order: optional lowercasing, `map` (synonyms to a
    canonical class; keys are matched after stripping and lowercasing) and, when `allowed` is
    set, every label outside it (including empty answers, refusals and malformed output)
    becomes the `invalid` class.
    """

    lowercase: bool = False
    map: dict[str, str] = field(default_factory=dict)
    allowed: tuple[str, ...] = ()
    invalid: str = "__invalid__"

    def key(self, label: str) -> str:
        label = label.strip()
        return label.lower() if self.lowercase else label


@dataclass(frozen=True)
class Output:
    """What each row of the tables holds.

    - score: one number per row (columns.score / reference_score / candidate_score);
    - label: one predicted class per row (same columns, holding class names);
    - probabilities: one probability column per class, named `<prob_prefix><class>`.
    """

    type: str = "score"
    prob_prefix: str = "p_"
    prob_sum_tolerance: float = 1e-3
    normalize: Normalize | None = None


@dataclass(frozen=True)
class CoverageGate:
    """Share of reference ids that must also be present in the candidate."""

    min: float = 1.0


@dataclass(frozen=True)
class NonFiniteGate:
    """Rows where exactly one side is NaN/inf. Both-NaN rows count as agreement."""

    max_mismatches: int = 0


@dataclass(frozen=True)
class MaxAbsDiffGate:
    """Worst-case absolute score difference over all matched rows."""

    max: float


@dataclass(frozen=True)
class QuantileAbsDiffGate:
    """A high quantile of the absolute difference (robust to a handful of outliers)."""

    max: float
    q: float = 0.99


@dataclass(frozen=True)
class MeanDiffEquivalenceGate:
    """TOST equivalence test of the mean paired difference, globally and per segment."""

    margin: float
    per_segment: bool = True


@dataclass(frozen=True)
class DecisionFlipsGate:
    """Share of rows that land on a different side of each decision threshold."""

    thresholds: tuple[float, ...]
    max_rate: float


@dataclass(frozen=True)
class TopKOverlapGate:
    """Overlap between the top-k% rows of each version (ranking use cases)."""

    k_pct: tuple[float, ...]
    min_overlap: float


@dataclass(frozen=True)
class AucDifferenceGate:
    """Equivalence of ROC AUC (paired DeLong confidence interval inside +/- margin)."""

    margin: float


@dataclass(frozen=True)
class LabelAgreementGate:
    """Share of rows with the same class; passes if its one-sided lower bound is >= min."""

    min: float


@dataclass(frozen=True)
class TransitionsGate:
    """Rows of class A that became class B, per pair; the upper bound must be <= max_rate.

    `per_class` sets a different limit for the rows of a given source class (for example, a
    small class that is noisy by nature); classes not listed use `max_rate`.
    """

    max_rate: float
    per_class: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ClassPrevalenceGate:
    """Paired difference of each class's share; its (1-2*alpha) CI must lie inside +/- margin."""

    margin: float


@dataclass(frozen=True)
class KappaGate:
    """Cohen's kappa between the versions; passes if its one-sided lower bound is >= min."""

    min: float


@dataclass(frozen=True)
class QualityDifferenceGate:
    """Accuracy or macro-F1 against ground truth; the CI of the difference inside +/- margin."""

    margin: float
    metric: str = "accuracy"


@dataclass(frozen=True)
class InvalidRateGate:
    """Share of candidate rows in the invalid class; its one-sided upper bound must be <= max."""

    max: float


@dataclass(frozen=True)
class StabilityGate:
    """Share of ids whose replicas disagree: candidate minus reference, upper bound <= margin."""

    margin: float


@dataclass(frozen=True)
class TvDistanceGate:
    """Total variation distance between the probability vectors of each row (quantile q)."""

    max: float
    q: float = 1.0


@dataclass(frozen=True)
class Gates:
    coverage: CoverageGate | None = field(default_factory=CoverageGate)
    nonfinite: NonFiniteGate | None = field(default_factory=NonFiniteGate)
    max_abs_diff: MaxAbsDiffGate | None = None
    quantile_abs_diff: QuantileAbsDiffGate | None = None
    mean_diff_equivalence: MeanDiffEquivalenceGate | None = None
    decision_flips: DecisionFlipsGate | None = None
    top_k_overlap: TopKOverlapGate | None = None
    auc_difference: AucDifferenceGate | None = None
    label_agreement: LabelAgreementGate | None = None
    transitions: TransitionsGate | None = None
    class_prevalence: ClassPrevalenceGate | None = None
    kappa: KappaGate | None = None
    quality_difference: QualityDifferenceGate | None = None
    tv_distance: TvDistanceGate | None = None
    invalid_rate: InvalidRateGate | None = None
    stability: StabilityGate | None = None


MAX_EXAMPLES = 1000


@dataclass(frozen=True)
class ParityConfig:
    columns: Columns = field(default_factory=Columns)
    segments: tuple[str, ...] = ()
    alpha: float = 0.05
    min_segment_size: int = 30
    gates: Gates = field(default_factory=Gates)
    preset: str | None = None
    output: Output = field(default_factory=Output)
    min_class_size: int = 30
    # Rows listed in the report by id (largest differences, or one per kind of class change).
    # Off by default: ids can be sensitive, so listing them is a deliberate choice.
    examples: int = 0

    def __post_init__(self) -> None:
        _validate(self)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_GATE_TYPES: dict[str, type[Any]] = {
    "coverage": CoverageGate,
    "nonfinite": NonFiniteGate,
    "max_abs_diff": MaxAbsDiffGate,
    "quantile_abs_diff": QuantileAbsDiffGate,
    "mean_diff_equivalence": MeanDiffEquivalenceGate,
    "decision_flips": DecisionFlipsGate,
    "top_k_overlap": TopKOverlapGate,
    "auc_difference": AucDifferenceGate,
    "label_agreement": LabelAgreementGate,
    "transitions": TransitionsGate,
    "class_prevalence": ClassPrevalenceGate,
    "kappa": KappaGate,
    "quality_difference": QualityDifferenceGate,
    "tv_distance": TvDistanceGate,
    "invalid_rate": InvalidRateGate,
    "stability": StabilityGate,
}

_SCORE_GATES = frozenset(
    {
        "coverage",
        "nonfinite",
        "max_abs_diff",
        "quantile_abs_diff",
        "mean_diff_equivalence",
        "decision_flips",
        "top_k_overlap",
        "auc_difference",
    }
)
_CLASS_GATES = frozenset(
    {"label_agreement", "transitions", "class_prevalence", "kappa", "quality_difference"}
)
QUALITY_METRICS = ("accuracy", "macro_f1")
NORMALIZED_TYPES = ("label",)

# Which gates make sense for each output type. A gate configured for an output type that does
# not support it is a configuration error, never silently ignored.
SUPPORTED_GATES: dict[str, frozenset[str]] = {
    "score": _SCORE_GATES,
    "label": frozenset({"coverage", "nonfinite", "invalid_rate", "stability"}) | _CLASS_GATES,
    # Per-class probability gates reuse the continuous checks; class decisions use the argmax.
    "probabilities": frozenset(
        {
            "coverage",
            "nonfinite",
            "max_abs_diff",
            "quantile_abs_diff",
            "mean_diff_equivalence",
            "tv_distance",
        }
    )
    | _CLASS_GATES,
}

T = TypeVar("T")


def _class_name(value: Any, where: str) -> str:
    """Class names may look like numbers or booleans in YAML (1: billing); they are text."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (str, int, float)):
        return str(value)
    raise ConfigError(f"{where}: expected a class name, got {value!r}")


def _coerce(value: Any, type_name: str, where: str) -> Any:
    """Convert YAML values to the declared field type.

    PyYAML follows YAML 1.1, where `1e-5` (no dot) is a *string*; numbers are therefore
    converted by declared type instead of trusting the YAML parser.
    """
    if type_name == "tuple[str, ...]":
        items = value if isinstance(value, (list, tuple)) else [value]
        return tuple(_class_name(v, f"{where}[{i}]") for i, v in enumerate(items))
    if type_name.startswith("tuple"):
        items = value if isinstance(value, (list, tuple)) else [value]
        return tuple(_coerce(v, "float", where) for v in items)
    if type_name.startswith("dict"):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where}: expected a mapping, got {value!r}")
        if type_name == "dict[str, float]":
            return {
                _class_name(k, f"{where} key"): _coerce(v, "float", f"{where}.{k}")
                for k, v in value.items()
            }
        return {
            _class_name(k, f"{where} key"): _class_name(v, f"{where}.{k}") for k, v in value.items()
        }
    if type_name in ("float", "int"):
        if isinstance(value, bool):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{where}: expected a number, got {value!r}") from None
        if not math.isfinite(number):
            raise ConfigError(f"{where}: must be finite, got {value!r}")
        if type_name == "int":
            if not number.is_integer():
                raise ConfigError(f"{where}: expected an integer, got {value!r}")
            return int(number)
        return number
    if type_name == "bool":
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true or false, got {value!r}")
        return value
    if value is not None and not isinstance(value, str):
        raise ConfigError(f"{where}: expected a string, got {value!r}")
    return value


def _build(cls: type[T], raw: Any, where: str) -> T:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{where}: expected a mapping, got {type(raw).__name__}")
    known = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(raw) - set(known))
    if unknown:
        raise ConfigError(f"{where}: unknown keys {unknown}; valid keys are {sorted(known)}")
    values = {
        name: _coerce(value, str(known[name].type), f"{where}.{name}")
        for name, value in raw.items()
    }
    try:
        return cls(**values)
    except TypeError as exc:
        raise ConfigError(f"{where}: {exc}") from exc


def _validate(cfg: ParityConfig) -> None:
    def require(cond: bool, message: str) -> None:
        if not cond:
            raise ConfigError(message)

    require(0 < cfg.alpha < 0.5, f"alpha must be in (0, 0.5), got {cfg.alpha}")
    require(cfg.min_segment_size >= 2, "min_segment_size must be >= 2")
    require(len(set(cfg.segments)) == len(cfg.segments), "segments has duplicates")
    require(
        cfg.output.type in OUTPUT_TYPES,
        f"output.type must be one of {list(OUTPUT_TYPES)}, got {cfg.output.type!r}",
    )
    g = cfg.gates
    enabled = {name for name in _GATE_TYPES if getattr(g, name) is not None}
    unsupported = sorted(enabled - SUPPORTED_GATES[cfg.output.type])
    require(
        not unsupported,
        f"gates {unsupported} do not apply to output.type {cfg.output.type!r}; "
        f"applicable gates are {sorted(SUPPORTED_GATES[cfg.output.type])}",
    )
    if g.coverage:
        require(0 < g.coverage.min <= 1, "gates.coverage.min must be in (0, 1]")
    if g.nonfinite:
        require(g.nonfinite.max_mismatches >= 0, "gates.nonfinite.max_mismatches must be >= 0")
    if g.max_abs_diff:
        require(g.max_abs_diff.max >= 0, "gates.max_abs_diff.max must be >= 0")
    if g.quantile_abs_diff:
        require(0 < g.quantile_abs_diff.q < 1, "gates.quantile_abs_diff.q must be in (0, 1)")
        require(g.quantile_abs_diff.max >= 0, "gates.quantile_abs_diff.max must be >= 0")
    if g.mean_diff_equivalence:
        require(
            g.mean_diff_equivalence.margin > 0, "gates.mean_diff_equivalence.margin must be > 0"
        )
    if g.decision_flips:
        require(len(g.decision_flips.thresholds) > 0, "gates.decision_flips.thresholds is empty")
        require(
            0 <= g.decision_flips.max_rate <= 1, "gates.decision_flips.max_rate must be in [0, 1]"
        )
    if g.top_k_overlap:
        require(len(g.top_k_overlap.k_pct) > 0, "gates.top_k_overlap.k_pct is empty")
        require(
            all(0 < k <= 100 for k in g.top_k_overlap.k_pct),
            "gates.top_k_overlap.k_pct values must be in (0, 100]",
        )
        require(
            0 <= g.top_k_overlap.min_overlap <= 1,
            "gates.top_k_overlap.min_overlap must be in [0, 1]",
        )
    if g.auc_difference:
        require(g.auc_difference.margin > 0, "gates.auc_difference.margin must be > 0")
        require(cfg.columns.label is not None, "gates.auc_difference requires columns.label")
    require(cfg.min_class_size >= 1, "min_class_size must be >= 1")
    require(cfg.output.prob_prefix != "", "output.prob_prefix must not be empty")
    require(0 < cfg.output.prob_sum_tolerance < 1, "output.prob_sum_tolerance must be in (0, 1)")
    if g.label_agreement:
        require(0 <= g.label_agreement.min <= 1, "gates.label_agreement.min must be in [0, 1]")
    if g.transitions:
        require(0 <= g.transitions.max_rate <= 1, "gates.transitions.max_rate must be in [0, 1]")
        require(
            all(0 <= v <= 1 for v in g.transitions.per_class.values()),
            "gates.transitions.per_class values must be in [0, 1]",
        )
    if g.class_prevalence:
        require(g.class_prevalence.margin > 0, "gates.class_prevalence.margin must be > 0")
    if g.kappa:
        require(-1 <= g.kappa.min <= 1, "gates.kappa.min must be in [-1, 1]")
    if g.quality_difference:
        require(g.quality_difference.margin > 0, "gates.quality_difference.margin must be > 0")
        require(
            g.quality_difference.metric in QUALITY_METRICS,
            f"gates.quality_difference.metric must be one of {list(QUALITY_METRICS)}",
        )
        require(cfg.columns.truth is not None, "gates.quality_difference requires columns.truth")
    if g.tv_distance:
        require(0 < g.tv_distance.q <= 1, "gates.tv_distance.q must be in (0, 1]")
        require(g.tv_distance.max >= 0, "gates.tv_distance.max must be >= 0")
    require(0 <= cfg.examples <= MAX_EXAMPLES, f"examples must be between 0 and {MAX_EXAMPLES}")
    norm = cfg.output.normalize
    if norm is not None:
        require(
            cfg.output.type in NORMALIZED_TYPES,
            f"output.normalize applies to output.type {list(NORMALIZED_TYPES)}, "
            f"not {cfg.output.type!r}",
        )
        _validate_normalize(norm, "output.normalize", require)
    if cfg.columns.replica is not None:
        require(
            cfg.output.type == "label",
            f"columns.replica applies to output.type 'label', not {cfg.output.type!r}",
        )
        require(cfg.columns.replica != cfg.columns.id, "columns.replica must differ from id")
    if g.stability:
        require(g.stability.margin > 0, "gates.stability.margin must be > 0")
        require(cfg.columns.replica is not None, "gates.stability requires columns.replica")
    if g.invalid_rate:
        require(0 <= g.invalid_rate.max <= 1, "gates.invalid_rate.max must be in [0, 1]")
        require(
            norm is not None and bool(norm.allowed),
            "gates.invalid_rate requires output.normalize.allowed (the list of valid classes)",
        )


def _validate_normalize(norm: Normalize, where: str, require: Any) -> None:
    allowed = [norm.key(a) for a in norm.allowed]
    require(all(allowed), f"{where}.allowed has an empty class name")
    require(len(set(allowed)) == len(allowed), f"{where}.allowed has duplicates")
    require(norm.invalid.strip() != "", f"{where}.invalid must not be empty")
    require(
        norm.key(norm.invalid) not in allowed,
        f"{where}.invalid {norm.invalid!r} must not be one of the allowed classes",
    )
    keys = [norm.key(k) for k in norm.map]
    require(all(keys), f"{where}.map has an empty key")
    require(len(set(keys)) == len(keys), f"{where}.map has keys that are equal after normalising")
    if allowed:
        unknown = sorted({v for v in norm.map.values() if norm.key(v) not in allowed})
        require(not unknown, f"{where}.map targets classes that are not allowed: {unknown}")


def from_dict(raw: Mapping[str, Any]) -> ParityConfig:
    """Build a config from a mapping. A `preset` is applied first; explicit gates override it."""
    from scoreparity.presets import preset_gates

    raw = dict(raw)
    version = raw.pop("version", CONFIG_VERSION)
    if version != CONFIG_VERSION:
        raise ConfigError(f"unsupported config version {version!r} (expected {CONFIG_VERSION})")
    known = {f.name for f in dataclasses.fields(ParityConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"unknown top-level keys {unknown}; valid keys are {sorted(known)}")

    preset = raw.get("preset")
    output = _build_output(raw.get("output") or {})
    gate_raw: dict[str, Any] = {}
    if preset:
        if output.type == "label":
            raise ConfigError(
                "presets define score tolerances and do not apply to output.type 'label'; "
                "configure the class gates explicitly"
            )
        supported = SUPPORTED_GATES.get(output.type, frozenset())
        gate_raw = {k: v for k, v in preset_gates(preset).items() if k in supported}
    explicit = raw.get("gates") or {}
    if not isinstance(explicit, Mapping):
        raise ConfigError("gates: expected a mapping")
    unknown_gates = sorted(set(explicit) - set(_GATE_TYPES))
    if unknown_gates:
        raise ConfigError(
            f"gates: unknown gates {unknown_gates}; valid gates are {sorted(_GATE_TYPES)}"
        )
    gate_raw.update(explicit)  # null disables a preset gate

    gates = Gates(
        **{
            name: None if value is None else _build(_GATE_TYPES[name], value, f"gates.{name}")
            for name, value in gate_raw.items()
        }
    )
    segments = raw.get("segments") or ()
    if isinstance(segments, str) or not all(isinstance(s, str) for s in segments):
        raise ConfigError("segments: expected a list of column names")
    return ParityConfig(
        columns=_build(Columns, raw.get("columns") or {}, "columns"),
        segments=tuple(segments),
        alpha=_coerce(raw.get("alpha", 0.05), "float", "alpha"),
        min_segment_size=_coerce(raw.get("min_segment_size", 30), "int", "min_segment_size"),
        gates=gates,
        preset=preset,
        output=output,
        min_class_size=_coerce(raw.get("min_class_size", 30), "int", "min_class_size"),
        examples=_coerce(raw.get("examples", 0), "int", "examples"),
    )


def _build_output(raw: Any) -> Output:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"output: expected a mapping, got {type(raw).__name__}")
    raw = dict(raw)
    normalize = raw.pop("normalize", None)
    output = _build(Output, raw, "output")
    if normalize is None:
        return output
    return dataclasses.replace(output, normalize=_build(Normalize, normalize, "output.normalize"))


def load(path: str | Path) -> ParityConfig:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path}: top level must be a mapping")
    return from_dict(raw)

"""The ``ar_c<tok>`` dose grammar of the Arditi ALL-POSITIONS refusal-steer rungs, shared by every legacy-store reader.

The full22 generator spells a served standup as ``<target>_<rung>`` (``qwen35_4b_it_d0p5``, ``qwen35_4b_ar_c0p3``), and the
readers of its stores (the sampled statics / controls, the multi-turn grade roots, the served-controls stores) recover the model
and the dose from that spelling. The ``d<tok>`` family (``dm1`` / ``d0p25`` / ``d1p5``: a NOMINAL multiplier on a fitted object's
write) is read by ``sampled_static.tag_model_key``, ``strength_mtdiff.strength_nominal_of`` and ``paper._served_key_dose``.
This module is the ONE home of the second family, the Arditi all-positions rungs Strength re-ran at the vendor preset
(``configs/arditi_allpos_roster_mt_vendor.yaml``, 2026-09-19: ``ar_c0p15`` / ``ar_c0p3`` / ``ar_c0p5``), so the statics,
multi-turn and controls readers all admit the same tags with the same dose and never derive it twice.

THE DOSE IS THE RAW COEFFICIENT c, IN ``raw_coefficient``, AND IS ON NO OTHER AXIS (``adapters/arditi_allpos.py`` docstring;
``vocab.STRENGTH_UNITS``): ``strength_coefficient`` = c from the token (``0p15`` -> 0.15), ``strength_unit`` = ``raw_coefficient``,
``strength_nominal`` = c / c_served with c_served = 0.3 -- the served operating point by the rule
``r2v1-include-qwen-arditi_allpos-ar_c0p3-served-operating-point``, so the served rung lands at exactly 1.0 and the ladder reads
beside every other nominal axis without ever pooling with it (a ``raw_coefficient`` row and a ``psr_direction_norm_multiplier`` /
``lora_write_basis_multiplier`` row are different units); ``is_operating_point`` = (c == c_served);
``operating_point_id_legacy`` = ``ar_c<tok>__arditi_allpos`` (the spelling the Arditi lane's own reader files, so the two standups
of one rung share the operating-point id while their producer keys stay distinct). ``paper_dose`` -- the dose word the paper's
RAW frame carries -- is c itself, never the nominal.

The roster records ``c:`` + a top-level ``dose_unit: raw_coefficient`` and NO ``dose:`` key: the wave generator's dose grammar is
closed to its own object families and refuses a ``dose:`` on a rung whose name does not spell the dose token, so the aggregates
carry no dose field and the dose comes from the tag alone. The rung's object is the Arditi lane's staged copy
(``ar_c<tok>__arditi_allpos``); c = 0.15 is DERIVED (the c = 0.1 direction x 1.5; its sidecars carry ``derivation: dose_scale``,
``parent_object_ids`` and ``params.base_scale``), the others are the lane's FITTED ladder (:data:`FITTED_LADDER`, mirrored from
``arditi_allpos.LADDER`` and pinned equal by test).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .. import derive as DV
from .. import vocab as V

#: the record's own unit spelling (in vocab.STRENGTH_UNITS since 2026-09-17)
STRENGTH_UNIT = "raw_coefficient"
assert STRENGTH_UNIT in V.STRENGTH_UNITS, STRENGTH_UNIT
#: the served object key of every rung (the harness's cell id), the paper label, the object kind and role (declared_common.ARMS)
OBJECT_KEY_LEGACY = "arditi_allpos"
PAPER_LABEL = "arditi_allpos"
OBJECT_KIND = "vector_all"
OBJECT_ROLE = "baseline_published"
#: the served operating point: c = 0.3, by the ruling below; the nominal denominator
SERVED_COEFFICIENT = 0.3
SERVED_OPERATING_POINT_RULING_ID = "r2v1-include-qwen-arditi_allpos-ar_c0p3-served-operating-point"
#: the Arditi lane's FITTED ladder (arditi_allpos.LADDER; tests pin the two equal): any other rung is a DERIVED copy of a fitted one
FITTED_LADDER: Tuple[str, ...] = ("ar_c0p1", "ar_c0p2", "ar_c0p3", "ar_c0p5", "ar_c1")
#: the one target the all-positions arm is served on (the paper's model key; served checkpoint qwen35_4b_it)
MODEL_KEY = "qwen35_4b"
#: the vendor-preset multi-turn re-run's three rungs (configs/arditi_allpos_roster_mt_vendor.yaml, 2026-09-19)
VENDOR_RUNGS: Tuple[str, ...] = ("ar_c0p15", "ar_c0p3", "ar_c0p5")
#: the source spelling of that re-run's rows (`run.lane_source_legacy`; PROVISIONAL, registry-declared: sources_r2_v1.yaml meta.additions)
SOURCE = "arditi_vendor_mt"
NOMINAL_RULE = (f"c / c_served with c_served = {SERVED_COEFFICIENT:g} (ruling {SERVED_OPERATING_POINT_RULING_ID}), so the served "
                "operating point is exactly 1.0; the unit stays raw_coefficient and never joins another unit's curve (D60)")

#: ``[<model>_]ar_c<tok>[_t<T>]`` -- the model key is optional (a bare rung), the trial suffix names a decode replicate standup
_TAG_RE = re.compile(r"^(?:(?P<model>[a-z][a-z0-9]*(?:_[0-9]+b)?)_)?(?P<rung>ar_c(?P<tok>m?[0-9]+(?:p[0-9]+)?))(?:_t(?P<trial>[0-9]+))?$")


def dose_token_value(token: str) -> Optional[float]:
    """``0p15`` -> 0.15, ``1`` -> 1.0, ``m0p5`` -> -0.5 (``m`` = minus, ``p`` = decimal point); None for anything else."""
    m = re.fullmatch(r"(?P<neg>m?)(?P<num>[0-9]+(?:p[0-9]+)?)", str(token))
    if not m:
        return None
    value = float(m.group("num").replace("p", "."))
    return -value if m.group("neg") else value


def dose_token(value: float) -> str:
    """The inverse spelling: 0.15 -> ``0p15``, 1.0 -> ``1``, -0.5 -> ``m0p5``."""
    text = f"{abs(float(value)):g}".replace(".", "p")
    return ("m" if value < 0 else "") + text


@dataclass(frozen=True)
class ArditiAllposRung:
    """One parsed ``ar_c<tok>`` spelling: the rung, its coefficient and the strength columns every run of it files."""
    tag: str                         # the text parsed
    model_key: Optional[str]         # `qwen35_4b` when the tag carried it
    rung: str                        # `ar_c0p15`
    token: str                       # `0p15`
    coefficient: float               # c
    trial: Optional[int]             # the `_t<T>` replicate suffix, when present

    @property
    def unit(self) -> str:
        return STRENGTH_UNIT

    @property
    def nominal(self) -> float:
        return self.coefficient / SERVED_COEFFICIENT

    @property
    def is_operating_point(self) -> bool:
        return abs(self.coefficient - SERVED_COEFFICIENT) < 1e-9

    @property
    def operating_point_id_legacy(self) -> str:
        return f"{self.rung}__{OBJECT_KEY_LEGACY}"

    @property
    def staged_id(self) -> str:
        """The Arditi lane's staged object directory for this rung (``ar_c0p15__arditi_allpos``)."""
        return f"{self.rung}__{OBJECT_KEY_LEGACY}"

    @property
    def fitted(self) -> bool:
        return self.rung in FITTED_LADDER

    @property
    def standup_tag(self) -> str:
        """The tag without its replicate suffix (``qwen35_4b_ar_c0p3_t1`` -> ``qwen35_4b_ar_c0p3``): the served standup's spelling."""
        return f"{self.model_key}_{self.rung}" if self.model_key else self.rung

    def strength_columns(self) -> Dict[str, object]:
        """The `run` columns of a STEERED row of this rung (an unsteered anchor files none: no object, no dose, no unit)."""
        return {"strength_coefficient": float(self.coefficient), "strength_unit": STRENGTH_UNIT, "strength_nominal": float(self.nominal),
                "strength_nominal_source": "derived", "is_operating_point": 1 if self.is_operating_point else 0,
                "operating_point_id_legacy": self.operating_point_id_legacy}

    def strength_confidence(self, where: str) -> Dict[str, Dict[str, str]]:
        """``field_confidence_json`` entries beside :meth:`strength_columns`; ``where`` names the record the tag was read from."""
        tag_src = f"the tag `{self.tag}` of {where}: the rung token spells c in raw_coefficient (dose_grammar)"
        return {"strength_coefficient": DV.confidence(None, "recorded", tag_src),
                "strength_unit": DV.confidence(None, "recorded", "the Arditi all-positions arm's declared unit (roster dose_unit: raw_coefficient; "
                                                                  "the staged object's staged_from.json / object.pt.provenance.json dose_unit)"),
                "strength_nominal": DV.confidence(None, "derived", NOMINAL_RULE),
                "is_operating_point": DV.confidence(None, "derived", f"c == c_served ({SERVED_COEFFICIENT:g}; {SERVED_OPERATING_POINT_RULING_ID})"),
                "operating_point_id_legacy": DV.confidence(None, "derived", "the Arditi lane's spelling `<rung>__arditi_allpos` (arditi_allpos.py)")}

    def extra(self) -> Dict[str, object]:
        """The parked facts every run of the rung carries in ``extra_json`` (``paper_dose`` = c, the paper's raw-frame word)."""
        return {"paper_dose": float(self.coefficient), "c": float(self.coefficient), "dose_unit_declared": STRENGTH_UNIT,
                "strength_unit_declared_legacy": STRENGTH_UNIT, "rung_legacy": self.rung, "dose_token": self.token,
                "nominal_dose_rule": NOMINAL_RULE, "operating_point_served_c": SERVED_COEFFICIENT,
                "operating_point_ruling_id": SERVED_OPERATING_POINT_RULING_ID, "staged_id": self.staged_id,
                "fitted": self.fitted, "derived_rung": (not self.fitted)}


def parse_arditi_allpos_tag(text: Optional[str]) -> Optional[ArditiAllposRung]:
    """``qwen35_4b_ar_c0p15`` / ``ar_c0p3`` / ``qwen35_4b_ar_c0p3_t1`` -> the rung; None when the text is not an all-positions tag."""
    if not isinstance(text, str) or not text:
        return None
    m = _TAG_RE.match(text)
    if not m:
        return None
    c = dose_token_value(m.group("tok"))
    if c is None:
        return None
    trial = m.group("trial")
    return ArditiAllposRung(tag=text, model_key=m.group("model"), rung=m.group("rung"), token=m.group("tok"), coefficient=c,
                            trial=int(trial) if trial is not None else None)


def is_arditi_allpos_key(cell_id: Optional[str]) -> bool:
    """The all-positions arm's cell spelling (``arditi_allpos``, or a staged ``ar_c<tok>__arditi_allpos``) -- never the prefill ``arditi`` /
    ``ar_r<alpha>`` arm, a different object kind on a different dose axis."""
    if not isinstance(cell_id, str):
        return False
    return cell_id == OBJECT_KEY_LEGACY or cell_id.endswith(f"__{OBJECT_KEY_LEGACY}")


def strength_payload(tag: Optional[str], object_kind: Optional[str], confidence_entries: Optional[Dict[str, Dict[str, str]]] = None,
                     where: str = "the standup tag") -> Tuple[Dict[str, object], Dict[str, object]]:
    """``(run columns, extra_json park)`` for a row whose standup tag may spell an all-positions rung; ``({}, {})`` for every other tag.

    The ONE call every legacy-store reader makes: a tag that parses AND a steered object together mean the run's dose is the raw
    coefficient the tag spells, so the strength columns and the ``paper_dose`` park come from here rather than from four derivations.
    An unsteered / prompted cell of the same standup gets nothing — it has no object on the served path, so no dose and no unit (the
    column defaults, which the validator pairs with their own ``unverified`` entries). ``confidence_entries`` is updated in place when
    given (an :class:`_items_common.Confidence`'s ``entries`` dict)."""
    spec = parse_arditi_allpos_tag(tag)
    if spec is None or object_kind in (None, "none", "system_instruction"):
        return {}, {}
    if confidence_entries is not None:
        confidence_entries.update(spec.strength_confidence(where))
    return spec.strength_columns(), spec.extra()


#: the three vendor re-run tags in the sampled readers' table shape: TAG -> (model key, transfer spelling, view spelling)
#: -- the direction is fit on and served on the instruct checkpoint (fit_provenance `transfer: false`; the Arditi lane's rows file
#: instruct_direct), so the transfer spelling is the paper's `instruct-direct`
ADMITTED_TAGS: Dict[str, Tuple[str, str, str]] = {f"{MODEL_KEY}_{r}": (MODEL_KEY, "instruct-direct", f"{MODEL_KEY}_{r}") for r in VENDOR_RUNGS}

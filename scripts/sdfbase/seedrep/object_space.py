"""Object-space seed stability of the Qwen-base object classes.

The CLASS ROSTER is derived, never typed: the cell ids of the
roster's replicate views (``configs/sdfbase_full22_roster.yaml`` → ``targets.qwen35_4b.views.s<s>``,
both ``cells`` — staged — and ``pending_fit`` — not yet staged) plus every derived id the RS
admission files name (``RS/reseed/admission/<key>__<cell>.json``, ``admitted`` + ``reasons``). A class
the admission refused on every seed is still a class: its table reads ``unavailable: incomplete
replicate triplet (0/3 admitted; missing: s1 — not admitted: <reasons>, …)`` with seed stability FAILED —
never a refusal of the run. ``CLASSES`` (the campaign grammar's six) stays declared as the library
default for ``seed_spread`` and the synthetic tests; the CLI always derives (``--roster``,
``--admission-dir``).

For each class the three batch-64 replicates ``{s1, s2, s3}`` (staged ids ``base64s<s>__<cell>`` /
``derived_base64s<s>__<cell>``) are loaded on CPU:

* LoRA blobs (``kind lora_all``: ``A[l] [d_in × r]``, ``B[l] [d_out × r]``; the trainer's
  delta is ``(x @ A) @ Bᵀ``, so the weight delta is ``ΔW_l = B_l A_lᵀ  [d_out × d_in]``);
* the subtraction blob (``kind lora_subtraction``: ``A_pos/B_pos/A_neg/B_neg``) composes
  ``ΔW_l = B_pos A_posᵀ − B_neg A_negᵀ`` (``LoRASubtractionObject`` serves exactly that);
* PSR blobs (``kind psr``: ``directions[l] [d]``, ``gate_w[l] [d]``, ``gate_b[l] [1]``; the
  distilled difference object is a PSR blob too).

Per replicate pair (3 pairs): cosine per targeted layer of ``vec(ΔW_l)`` (LoRA) / ``d_l`` and ``w_l``
(PSR, plus ``|b_a − b_b|``), the stacked cosine over the concatenation across layers, the
relative-norm difference ``‖x_a − x_b‖ / mean(‖x_a‖, ‖x_b‖)`` per layer and stacked, and the
per-layer norm table.

**Float64, chunked.** Every norm, dot and cosine is accumulated in float64 (``_f64_sums``: each chunk
cast to float64 and reduced there, the chunk partials summed in Python float64). float32
``torch.dot`` / ``vector_norm`` over the ~84 M-element stacked LoRA ΔW vectors mis-stated norms by
~1.2 % and stacked cosines by +0.01–0.02 — one read 1.012 (interim_analysis.md §9 item 9). A cosine
beyond ``1 + 1e-9`` in magnitude is REFUSED (``ValueError``), never clamped.

**The batch-8 seed-0 original is opt-in.** ``base__<cell>`` / ``derived__<cell>`` (labelled
``orig_b8_s0``) was the reference column — a reference, NEVER a replicate. The batch-8
originals were removed from the paper, so by default NO ``base__*`` / ``derived__*`` object is
loaded, tabled or plotted; ``--include-batch8-reference`` restores the column (3 vs-original
pairs per class beside the replicate pairs).

Every class statistic passes ``triplet_gate.require_triplet`` FIRST: an incomplete triplet is
written as ``unavailable: incomplete replicate triplet (k/3 admitted; missing: …)`` with seed stability
FAILED for that class and NO pair statistic over the survivors.

Verdict (per arm): the difference object is more / less seed-stable than its parents by the MEAN
replicate-pair stacked cosine of the primary component (LoRA: ΔW; PSR: direction), min/max over
pairs shown beside it.

Usage::

    object_space.py --bank $STAGE/sdfbase/full22_objects/qwen35_4b/cells \\
                    --out $GRADE/seedrep/object_space \\
                    [--roster configs/sdfbase_full22_roster.yaml] [--admission-dir RS/reseed/admission] \\
                    [--include-batch8-reference] [--figs DIR] [--mirror] [--wandb]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import torch

_HERE = Path(__file__).resolve().parent
_FULL22 = _HERE.parent / "full22"
for _p in (_HERE, _FULL22):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import figures as F  # noqa: E402
from f22_roster import ROSTER as DEFAULT_ROSTER, load_roster  # noqa: E402
from triplet_gate import (  # noqa: E402
    REASON_NO_STORE, REASON_NOT_ADMITTED, SEEDS, UNAVAILABLE_PREFIX, Record, TripletUnavailable, require_triplet,
)

ORIGINAL_LABEL = "orig_b8_s0"
ORIGINAL_NOT_LOADED = ("not loaded: the batch-8 seed-0 reference column is opt-in (--include-batch8-reference) — O23 "
                       "(the user, 2026-09-17) removes the batch-8 originals from the paper, so no base__/derived__ "
                       "object is loaded, tabled or plotted by default")
#: the staged cells (under the staging directory, $STAGE) and the admission files (in the full22 wave's run-state tree, under the state
#: root, $STATE): site placeholders, expanded through sisteer.site where they are read
DEFAULT_BANK = Path("$STAGE/sdfbase/full22_objects/qwen35_4b/cells")
DEFAULT_OUT = Path("$GRADE/seedrep/object_space")   # under the grade root, $GRADE: main() expands it through sisteer.site
#: the admission directory under the wave lanes' shared run-state root (a built-in path)
RS_ADMISSION = Path("$STATE/romp/candidate-si-refit/run-state/full22/reseed/admission")
TARGET = "qwen35_4b"
#: the fit key of a replicate (plan naming: KEY(s) = `<target>_base_eb64_s<s>`) and a replicate view (`s<s>`)
KEY_RE = re.compile(r"^(?P<target>.+)_base_eb64_s(?P<seed>[0-9]+)$")
VIEW_RE = re.compile(r"^s([1-9][0-9]*)$")
#: a float64 accumulation cannot round a cosine past this; beyond it the inputs or the reduction are wrong
COSINE_TOL = 1e-9
#: elements per float64 chunk (4 M → a 32 MB float64 copy at a time, never a full 84 M-element float64 vector)
CHUNK = 1 << 22


@dataclass(frozen=True)
class ObjectClass:
    cell_id: str
    arm: str          # "lora" | "psr"
    role: str         # "constitution" | "control" | "difference"
    derived: bool

    def staged_id(self, seed: int) -> str:
        return f"{'derived_' if self.derived else ''}base64s{int(seed)}__{self.cell_id}"

    def original_id(self) -> str:
        return f"{'derived' if self.derived else 'base'}__{self.cell_id}"


#: the campaign grammar's six classes — the LIBRARY default (seed_spread imports it; the synthetic tests build banks
#: from it). The CLI never uses it: `class_roster` derives the classes from the roster + the admission files.
CLASSES: tuple[ObjectClass, ...] = (
    ObjectClass("constv2-mix-full__lora_r4_ce", "lora", "constitution", False),
    ObjectClass("constv2ctl2-full__lora_r4_ce", "lora", "control", False),
    ObjectClass("constv2-mix-full__lora_r4_ce_minus_ctl", "lora", "difference", True),
    ObjectClass("constv2-mix-full__psr_ce_all", "psr", "constitution", False),
    ObjectClass("constv2ctl2-full__psr_ce_all", "psr", "control", False),
    ObjectClass("constv2-mix-full__psr_ce_all_minus_ctl_distilled", "psr", "difference", True),
)
CLASS_BY_ID = {c.cell_id: c for c in CLASSES}
PRIMARY_COMPONENT = {"lora": "dW", "psr": "direction"}
COMPONENTS = {"lora": ("dW",), "psr": ("direction", "gate_w", "gate_b")}
COSINE_COMPONENTS = {"lora": ("dW",), "psr": ("direction", "gate_w")}
#: the cell-id grammar `<si>__<method>`: the SI names the role of a PARENT, the method's prefix the arm, `_minus_`
#: a difference object (which must be staged as `derived_…`); the verdict needs constitution vs control, so an SI
#: with no declared role refuses rather than guessing
ROLE_OF_SI = {"constv2-mix-full": "constitution", "constv2ctl2-full": "control"}
ARM_OF_METHOD = (("psr_", "psr"), ("lora_", "lora"))
ARM_ORDER = {"lora": 0, "psr": 1}
ROLE_ORDER = {"constitution": 0, "control": 1, "difference": 2}


# ---------------------------------------------------------------- class roster: roster views + admission files
class Admission(NamedTuple):
    """One RS admission file for (cell, seed): the verdict and its reasons."""

    seed: int
    admitted: bool
    reasons: tuple[str, ...]
    path: str


@dataclass(frozen=True)
class ClassRoster:
    classes: tuple[ObjectClass, ...]
    admissions: Mapping[str, Mapping[int, Admission]]            # cell id -> seed -> admission
    listed: Mapping[str, Mapping[int, tuple[str, str]]]          # cell id -> seed -> (bucket, staged id)
    roster_path: str
    admission_dir: str

    def for_class(self, cls: ObjectClass) -> Mapping[int, Admission]:
        return self.admissions.get(cls.cell_id, {})


def classify_cell(cell_id: str, *, derived: bool) -> ObjectClass:
    """The (arm, role) of a cell id under the campaign grammar; refuses an id the grammar cannot place."""
    si, sep, method = str(cell_id).partition("__")
    if not sep or not si or not method:
        raise ValueError(f"cell {cell_id!r} is not `<si>__<method>`")
    arm = next((a for pfx, a in ARM_OF_METHOD if method.startswith(pfx)), None)
    if arm is None:
        raise ValueError(f"cell {cell_id!r}: method {method!r} is neither a psr_* nor a lora_* object")
    is_diff = "_minus_" in method
    if is_diff != bool(derived):
        raise ValueError(f"cell {cell_id!r}: {'a difference id' if is_diff else 'a parent id'} staged as "
                         f"{'a derived object' if derived else 'a fit'} — the staged-id prefix and the cell grammar disagree")
    if derived:
        return ObjectClass(cell_id, arm, "difference", True)
    role = ROLE_OF_SI.get(si)
    if role is None:
        raise ValueError(f"cell {cell_id!r}: SI {si!r} has no declared role ({sorted(ROLE_OF_SI)}) — the verdict "
                         "needs constitution vs control, so an unknown SI is refused, not guessed")
    return ObjectClass(cell_id, arm, role, False)


def roster_cells(roster: Mapping[str, Any], *, target: str = TARGET) -> dict[str, dict[int, tuple[str, str]]]:
    """``{cell id: {seed: (bucket, staged id)}}`` over the target's replicate views (``s<s>``), from both
    ``cells`` (staged) and ``pending_fit`` (not yet staged). A prompted baseline (null id) has no object and
    is skipped; a pin-form id is not this lane's grammar and refuses."""
    try:
        views = roster["targets"][target]["views"] or {}
    except (KeyError, TypeError):
        raise ValueError(f"roster names no target {target!r} with views") from None
    out: dict[str, dict[int, tuple[str, str]]] = {}
    for view, body in views.items():
        m = VIEW_RE.match(str(view))
        if not m:
            continue
        seed = int(m.group(1))
        if seed not in SEEDS:
            raise ValueError(f"roster view {target}/{view} is outside the replicate triplet {SEEDS}")
        for bucket in ("cells", "pending_fit"):
            for cid, sid in ((body or {}).get(bucket) or {}).items():
                if sid is None:
                    continue
                if not isinstance(sid, str):
                    raise ValueError(f"{target}/{view} {bucket}: {cid} maps to {sid!r} — not a staged id")
                per = out.setdefault(str(cid), {})
                if seed in per:
                    raise ValueError(f"{target}/{view}: {cid} is listed under both cells and pending_fit")
                per[seed] = (bucket, sid)
    return out


def read_admissions(admission_dir: Path | str, *, target: str = TARGET) -> dict[str, dict[int, Admission]]:
    """Every RS admission file of the target's replicates: ``{cell id: {seed: Admission}}``. The directory must
    exist (an absent RS is a fault, not an empty roster); a file whose key names another target is skipped."""
    d = Path(admission_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"admission dir {d} does not exist — pass --admission-dir to the RS copy")
    out: dict[str, dict[int, Admission]] = {}
    for f in sorted(d.glob("*.json")):
        doc = json.loads(f.read_text())
        key = doc.get("key") or doc.get("expected_key") or ""
        m = KEY_RE.match(str(key))
        if not m or m["target"] != target:
            continue
        seed = int(m["seed"])
        if doc.get("expected_seed") is not None and int(doc["expected_seed"]) != seed:
            raise ValueError(f"{f}: key {key} names s{seed} but expected_seed is {doc['expected_seed']}")
        cell = doc.get("cell") or doc.get("derived_id")
        if not cell:
            raise ValueError(f"{f}: admission names no cell")
        if "admitted" not in doc:
            raise ValueError(f"{f}: admission carries no `admitted` verdict")
        reasons = doc.get("reasons") or []
        if isinstance(reasons, str):
            reasons = [reasons]
        per = out.setdefault(str(cell), {})
        if seed in per:
            raise ValueError(f"two admission files for {cell} s{seed}: {per[seed].path} and {f}")
        per[seed] = Admission(seed, bool(doc["admitted"]), tuple(str(r) for r in reasons), str(f))
    return out


def class_roster(roster_path: Path | str = DEFAULT_ROSTER, admission_dir: Path | str = RS_ADMISSION, *,
                 target: str = TARGET) -> ClassRoster:
    """THE class list: every cell id the roster's replicate views list (staged or pending) plus every derived id
    an admission file names — a derived class the admission refused on every seed is on it, so its table can
    say why. Refuses grammar drift (a roster staged id that is not the lane's ``[derived_]base64s<s>__<cell>``,
    an admission file for a cell the roster stages as a fit, a cell the grammar cannot place)."""
    if "$" in str(admission_dir):
        from sisteer import site as _site
        admission_dir = _site.expand_path(admission_dir, key="admission_dir")
    roster = load_roster(Path(roster_path))
    listed = roster_cells(roster, target=target)
    admissions = read_admissions(admission_dir, target=target)
    derived_flag: dict[str, bool] = {}
    for cid, per in listed.items():
        flags = {sid.startswith("derived_") for _, sid in per.values()}
        if len(flags) != 1:
            raise ValueError(f"{cid}: the roster's staged ids disagree on the derived_ prefix across seeds")
        derived_flag[cid] = flags.pop()
    for cid in admissions:
        if cid in derived_flag and not derived_flag[cid]:
            raise ValueError(f"{cid}: has admission files but the roster stages it as a fit")
        derived_flag.setdefault(cid, True)        # an admission file exists only for a derived object
    if not derived_flag:
        raise ValueError(f"roster {roster_path} + admissions {admission_dir} name no object class for {target}")
    classes: list[ObjectClass] = []
    for cid, derived in derived_flag.items():
        cls = classify_cell(cid, derived=derived)
        for seed, (bucket, sid) in listed.get(cid, {}).items():
            if sid != cls.staged_id(seed):
                raise ValueError(f"{cid} s{seed} ({bucket}): roster staged id {sid!r} is not the lane grammar's "
                                 f"{cls.staged_id(seed)!r}")
        classes.append(cls)
    classes.sort(key=lambda c: (ARM_ORDER[c.arm], ROLE_ORDER[c.role], c.cell_id))
    return ClassRoster(tuple(classes), admissions, listed, str(roster_path), str(admission_dir))


# ---------------------------------------------------------------- blobs -> vectors
@dataclass
class ObjectVectors:
    kind: str
    layers: list[int]
    comps: dict[str, dict[int, torch.Tensor]]   # component -> layer -> flat float32 CPU vector

    def stacked(self, comp: str) -> torch.Tensor:
        return torch.cat([self.comps[comp][l].reshape(-1) for l in self.layers])


def load_blob(path: Path | str) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def lora_delta(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """``ΔW = B Aᵀ`` in float32 for the trainer's ``(x @ A) @ Bᵀ`` delta (rank-1 blobs carry
    1-D ``A [d_in]`` / ``B [d_out]``; the outer product is the same map)."""
    A = A.detach().to(torch.float32).cpu()
    B = B.detach().to(torch.float32).cpu()
    if A.dim() == 1:
        A, B = A.unsqueeze(1), B.unsqueeze(1)
    if A.dim() != 2 or B.dim() != 2 or A.shape[1] != B.shape[1]:
        raise ValueError(f"LoRA factor shapes disagree: A {tuple(A.shape)} vs B {tuple(B.shape)}")
    return B @ A.T


def _layers_of(d: Mapping[str, Any]) -> list[int]:
    return sorted(int(k) for k in d)


def vectors_from_blob(blob: Mapping[str, Any]) -> ObjectVectors:
    kind = blob.get("kind")
    if kind == "lora_all":
        layers = _layers_of(blob["A"])
        if sorted(int(x) for x in blob.get("layers", layers)) != layers:
            raise ValueError("lora_all blob: `layers` metadata disagrees with the A tensors it carries")
        dW = {l: lora_delta(blob["A"][str(l)], blob["B"][str(l)]).reshape(-1) for l in layers}
        return ObjectVectors(kind, layers, {"dW": dW})
    if kind == "lora_subtraction":
        layers = _layers_of(blob["A_pos"])
        for side in ("A_neg", "B_pos", "B_neg"):
            if _layers_of(blob[side]) != layers:
                raise ValueError(f"lora_subtraction blob: {side} covers other layers than A_pos")
        dW = {l: (lora_delta(blob["A_pos"][str(l)], blob["B_pos"][str(l)])
                  - lora_delta(blob["A_neg"][str(l)], blob["B_neg"][str(l)])).reshape(-1) for l in layers}
        return ObjectVectors(kind, layers, {"dW": dW})
    if kind == "psr":
        layers = [int(l) for l in blob["layers"]]
        comps = {name: {l: blob[key][str(l)].detach().to(torch.float32).cpu().reshape(-1) for l in layers}
                 for name, key in (("direction", "directions"), ("gate_w", "gate_w"), ("gate_b", "gate_b"))}
        return ObjectVectors(kind, layers, comps)
    raise ValueError(f"unknown object kind {kind!r} (want lora_all | lora_subtraction | psr)")


# ---------------------------------------------------------------- statistics (float64, chunked)
class Sums(NamedTuple):
    """float64 ``Σa²``, ``Σb²``, ``Σa·b``, ``Σ(a−b)²`` over a pair of flat vectors (``bb``/``ab``/``dd`` are 0 for
    a single vector). Additive over layers, so the stacked statistic is the sum of the per-layer sums."""

    aa: float
    bb: float
    ab: float
    dd: float

    def __add__(self, o: "Sums") -> "Sums":          # type: ignore[override]
        return Sums(self.aa + o.aa, self.bb + o.bb, self.ab + o.ab, self.dd + o.dd)


def _f64_sums(a: torch.Tensor, b: torch.Tensor | None = None, *, chunk: int = CHUNK) -> Sums:
    """The sums accumulated CHUNKWISE in float64: each chunk is cast to float64 and reduced by torch in float64,
    the chunk partials are summed in Python float64. No float32 reduction ever runs (float32 ``dot`` /
    ``vector_norm`` drift by ~1e-3 relative at 10^7 elements and ~1e-2 at 84 M), and no full float64 copy of a
    stacked vector is materialised."""
    a = a.reshape(-1)
    n = a.numel()
    if b is not None:
        b = b.reshape(-1)
        if b.numel() != n:
            raise ValueError(f"vector lengths differ: {n} vs {b.numel()}")
    if chunk < 1:
        raise ValueError("chunk must be >= 1")
    aa = bb = ab = dd = 0.0
    for i in range(0, n, chunk):
        x = a[i:i + chunk].to(torch.float64)
        aa += float(torch.dot(x, x))
        if b is not None:
            y = b[i:i + chunk].to(torch.float64)
            bb += float(torch.dot(y, y))
            ab += float(torch.dot(x, y))
            d = x - y
            dd += float(torch.dot(d, d))
    return Sums(aa, bb, ab, dd)


def bounded_cosine(ab: float, aa: float, bb: float) -> float:
    """``ab / (√aa · √bb)``; NaN when a norm is 0; REFUSES a magnitude beyond ``1 + COSINE_TOL`` — under exact
    arithmetic |cos| ≤ 1 (Cauchy–Schwarz) and a float64 accumulation rounds by ~1e-15, so anything past the
    tolerance means the inputs or the reduction are wrong. Never clamped: a clamp would hide exactly the float32
    fault this module was fixed for (a stacked cosine that read 1.012)."""
    if aa == 0.0 or bb == 0.0:
        return float("nan")
    c = ab / (math.sqrt(aa) * math.sqrt(bb))
    if abs(c) > 1.0 + COSINE_TOL:
        raise ValueError(f"cosine {c!r} exceeds 1 + {COSINE_TOL:g} in magnitude — impossible under exact arithmetic "
                         "and beyond float64 rounding; the inputs or the reduction are wrong")
    return c


def _relnorm(s: Sums) -> float:
    denom = 0.5 * (math.sqrt(s.aa) + math.sqrt(s.bb))
    if denom == 0.0:
        return float("nan")
    return math.sqrt(s.dd) / denom


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    s = _f64_sums(a, b)
    return bounded_cosine(s.ab, s.aa, s.bb)


def relnorm(a: torch.Tensor, b: torch.Tensor) -> float:
    """``‖a − b‖ / mean(‖a‖, ‖b‖)`` — 0 for identical objects, 2 for sign-flipped ones."""
    return _relnorm(_f64_sums(a, b))


def vector_norm(a: torch.Tensor) -> float:
    return math.sqrt(_f64_sums(a).aa)


def norms(v: ObjectVectors) -> dict[str, dict[int, float]]:
    return {c: {l: vector_norm(v.comps[c][l]) for l in v.layers} for c in v.comps}


def pair_stats(a: ObjectVectors, b: ObjectVectors, arm: str) -> dict[str, Any]:
    if a.layers != b.layers:
        raise ValueError(f"layer sets differ: {a.layers} vs {b.layers}")
    out: dict[str, Any] = {"components": {}}
    for comp in COSINE_COMPONENTS[arm]:
        per_layer = {l: _f64_sums(a.comps[comp][l], b.comps[comp][l]) for l in a.layers}
        stacked = Sums(0.0, 0.0, 0.0, 0.0)
        for s in per_layer.values():
            stacked = stacked + s
        out["components"][comp] = {
            "per_layer_cosine": {l: bounded_cosine(s.ab, s.aa, s.bb) for l, s in per_layer.items()},
            "per_layer_relnorm": {l: _relnorm(s) for l, s in per_layer.items()},
            "stacked_cosine": bounded_cosine(stacked.ab, stacked.aa, stacked.bb),
            "stacked_relnorm": _relnorm(stacked),
        }
    if "gate_b" in a.comps and "gate_b" in b.comps:
        out["gate_b_abs_diff"] = {l: float((a.comps["gate_b"][l].to(torch.float64)
                                            - b.comps["gate_b"][l].to(torch.float64)).abs().max())
                                  for l in a.layers}
    return out


# ---------------------------------------------------------------- bank -> records
def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def admission_state(cell_dir: Path) -> tuple[bool, str | None]:
    """Eligibility of one staged object directory for the statistic: the object must
    exist and, for a derived object, must not carry a negative admission record
    (``ADMISSION.json`` beside it or ``staged_from.json.admission``)."""
    obj = cell_dir / "object.pt"
    if not obj.is_file():
        return False, REASON_NO_STORE
    for src in (cell_dir / "ADMISSION.json", cell_dir / "staged_from.json"):
        if not src.is_file():
            continue
        try:
            d = json.loads(src.read_text())
        except json.JSONDecodeError as exc:
            return False, f"unreadable {src.name}: {exc}"
        adm = d.get("admission") if src.name == "staged_from.json" else d
        if isinstance(adm, Mapping) and "admitted" in adm and not adm["admitted"]:
            why = adm.get("reasons") or adm.get("reason") or "no reason recorded"
            return False, f"{REASON_NOT_ADMITTED}: {why if isinstance(why, str) else '; '.join(map(str, why))}"
    return True, None


def replicate_records(bank: Path, cls: ObjectClass,
                      admissions: Mapping[int, Admission] | None = None) -> list[Record]:
    """One Record per seed. A seed whose directory is PRESENT is judged by ``admission_state``; a seed whose
    directory is ABSENT yields a row only when an RS admission file speaks for it — ``not admitted: <reasons>``
    (the admission refused it, so it was never staged) or ``no store: admitted … but not staged`` — and no row
    otherwise (the gate then reports ``no store``). Loading happens only after the gate passes."""
    recs: list[Record] = []
    for s in SEEDS:
        d = bank / cls.staged_id(s)
        if not d.exists():
            adm = (admissions or {}).get(s)
            if adm is None:
                continue
            if adm.admitted:
                recs.append(Record(s, d, False, f"{REASON_NO_STORE}: admitted by {Path(adm.path).name} but not staged "
                                                "in the bank"))
            else:
                why = "; ".join(adm.reasons) or "no reason recorded"
                recs.append(Record(s, d, False, f"{REASON_NOT_ADMITTED}: {why}"))
            continue
        ok, why = admission_state(d)
        recs.append(Record(s, d, ok, why))
    return recs


def _describe(path: Path, v: ObjectVectors) -> dict[str, Any]:
    shapes = {c: {str(l): list(v.comps[c][l].shape) for l in v.layers[:1]} for c in v.comps}
    return {"path": str(path), "sha256_16": _sha16(path / "object.pt"), "kind": v.kind, "layers": v.layers,
            "n_layers": len(v.layers), "flat_shape_first_layer": shapes,
            "per_layer_norms": {c: {str(l): n for l, n in nl.items()} for c, nl in norms(v).items()}}


def unavailable_status(missing: Sequence[tuple[int, str]]) -> str:
    """``unavailable: incomplete replicate triplet (k/3 admitted; missing: s<k> — <reason>, …)`` — the gate's
    message with the admitted count in front of the per-seed reasons."""
    k = len(SEEDS) - len({int(s) for s, _ in missing})
    inner = ", ".join(f"s{s} — {r}" for s, r in sorted(missing))
    return f"{UNAVAILABLE_PREFIX} ({k}/{len(SEEDS)} admitted; missing: {inner})"


# ---------------------------------------------------------------- per-class analysis
def analyze_class(bank: Path, cls: ObjectClass, *, admissions: Mapping[int, Admission] | None = None,
                  include_batch8_reference: bool = False) -> dict[str, Any]:
    bank = Path(bank)
    table: dict[str, Any] = {
        "class": cls.cell_id, "arm": cls.arm, "role": cls.role, "derived": cls.derived,
        "primary_component": PRIMARY_COMPONENT[cls.arm], "components": list(COMPONENTS[cls.arm]),
        "staged_ids": {f"s{s}": cls.staged_id(s) for s in SEEDS}, "original_id": cls.original_id(),
        "original_label": ORIGINAL_LABEL, "include_batch8_reference": bool(include_batch8_reference),
        "replicates": {}, "original": None, "admitted_count": None, "status": None, "criterion_5": None,
        "pairs": [], "summary": {},
    }
    recs = replicate_records(bank, cls, admissions)
    for s in SEEDS:
        r = next((r for r in recs if r.seed == s), None)
        table["replicates"][f"s{s}"] = ({"present": False, "eligible": False, "reason": REASON_NO_STORE} if r is None
                                        else {"present": Path(r.value).exists(), "eligible": r.eligible,
                                              "reason": r.reason, "path": str(r.value)})
    # the batch-8 seed-0 original is a reference column: OPT-IN (O23), loaded independently of the gate, never fed to it
    orig_v = None
    table["original"] = {"label": ORIGINAL_LABEL, "included": bool(include_batch8_reference)}
    if include_batch8_reference:
        orig_dir = bank / cls.original_id()
        orig_ok, orig_why = admission_state(orig_dir)
        if orig_ok:
            orig_v = vectors_from_blob(load_blob(orig_dir / "object.pt"))
            table["original"].update({"eligible": True, **_describe(orig_dir, orig_v)})
        else:
            table["original"].update({"eligible": False, "reason": orig_why, "path": str(orig_dir)})
    else:
        table["original"].update({"eligible": False, "reason": ORIGINAL_NOT_LOADED})
    # THE GATE — before any replicate statistic
    try:
        trip = require_triplet(recs)
    except TripletUnavailable as exc:
        table["status"] = unavailable_status(exc.missing)
        table["admitted_count"] = len(SEEDS) - len(exc.missing)
        table["criterion_5"] = "FAILED"
        table["missing"] = [{"seed": s, "reason": r} for s, r in exc.missing]
        return table
    table["admitted_count"] = len(SEEDS)
    vecs: dict[str, ObjectVectors] = {}
    for s in SEEDS:
        d = Path(trip[s])
        v = vectors_from_blob(load_blob(d / "object.pt"))
        expect = "psr" if cls.arm == "psr" else ("lora_subtraction" if cls.derived else "lora_all")
        if v.kind != expect and not (expect == "lora_subtraction" and v.kind == "lora_all"):
            raise ValueError(f"{cls.staged_id(s)}: blob kind {v.kind!r}, expected {expect!r}")
        vecs[f"s{s}"] = v
        table["replicates"][f"s{s}"].update(_describe(d, v))
    layers = vecs["s1"].layers
    for k, v in vecs.items():
        if v.layers != layers:
            raise ValueError(f"{cls.cell_id}: {k} targets layers {v.layers}, s1 targets {layers}")
    if orig_v is not None and orig_v.layers != layers:
        table["original"]["eligible"] = False
        table["original"]["reason"] = f"layer set {orig_v.layers} differs from the replicates' {layers}"
        orig_v = None
    table["layers"] = layers
    pairs: list[dict[str, Any]] = []
    for a, b in combinations(sorted(vecs), 2):
        pairs.append({"a": a, "b": b, "kind": "replicate", "label": f"{a}–{b}", **pair_stats(vecs[a], vecs[b], cls.arm)})
    if orig_v is not None:
        for a in sorted(vecs):
            pairs.append({"a": a, "b": ORIGINAL_LABEL, "kind": "vs_original", "label": f"{a}–{ORIGINAL_LABEL}",
                          **pair_stats(vecs[a], orig_v, cls.arm)})
    table["pairs"] = pairs
    summary: dict[str, Any] = {}
    for comp in COSINE_COMPONENTS[cls.arm]:
        summary[comp] = {}
        for kind in ("replicate", "vs_original"):
            vals = [p["components"][comp]["stacked_cosine"] for p in pairs if p["kind"] == kind]
            rn = [p["components"][comp]["stacked_relnorm"] for p in pairs if p["kind"] == kind]
            summary[comp][kind] = (None if not vals else
                                   {"n_pairs": len(vals), "stacked_cosine": vals,
                                    "mean_stacked_cosine": sum(vals) / len(vals), "min_stacked_cosine": min(vals),
                                    "max_stacked_cosine": max(vals), "stacked_relnorm": rn,
                                    "mean_stacked_relnorm": sum(rn) / len(rn)})
    if "gate_b_abs_diff" in (pairs[0] if pairs else {}):
        summary["gate_b"] = {kind: {"max_abs_diff_over_layers": max(max(p["gate_b_abs_diff"].values()) for p in ps)}
                             for kind in ("replicate", "vs_original")
                             if (ps := [p for p in pairs if p["kind"] == kind])}
    table["summary"] = summary
    table["status"] = "ok"
    table["criterion_5"] = "PASS"
    return table


def verdicts(tables: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per arm: the difference class's mean replicate-pair stacked cosine (primary
    component) against each parent's, min/max over pairs beside them."""
    out: dict[str, Any] = {}
    for arm in ("lora", "psr"):
        comp = PRIMARY_COMPONENT[arm]
        rows = {t["role"]: t for t in tables if t["arm"] == arm}
        entry: dict[str, Any] = {"primary_component": comp, "classes": {}}
        means: dict[str, float | None] = {}
        for role, t in rows.items():
            s = ((t.get("summary") or {}).get(comp) or {}).get("replicate")
            entry["classes"][role] = {"class": t["class"], "status": t.get("status"),
                                      "mean": None if not s else s["mean_stacked_cosine"],
                                      "min": None if not s else s["min_stacked_cosine"],
                                      "max": None if not s else s["max_stacked_cosine"]}
            means[role] = None if not s else s["mean_stacked_cosine"]
        d, c, k = means.get("difference"), means.get("constitution"), means.get("control")
        if d is None or c is None or k is None:
            missing = [r for r in ("constitution", "control", "difference") if means.get(r) is None]
            entry["verdict"] = ("unavailable: " + "; ".join(f"{r}: {rows[r]['status']}" if r in rows else f"{r}: no table"
                                                            for r in missing))
            entry["criterion_5"] = "FAILED"
        else:
            if d > max(c, k):
                word = "MORE seed-stable than both parents"
            elif d < min(c, k):
                word = "LESS seed-stable than both parents"
            else:
                lo, hi = ("control", "constitution") if k < c else ("constitution", "control")
                word = f"between its parents (more seed-stable than the {lo}, less than the {hi})"
            entry["verdict"] = (f"the {F.ARM_LABEL[arm]} difference object is {word}: mean replicate-pair stacked "
                                f"cosine of {comp} {d:.4f} (min {entry['classes']['difference']['min']:.4f}, max "
                                f"{entry['classes']['difference']['max']:.4f}) vs constitution {c:.4f} "
                                f"(min {entry['classes']['constitution']['min']:.4f}, max {entry['classes']['constitution']['max']:.4f}) "
                                f"and control {k:.4f} (min {entry['classes']['control']['min']:.4f}, max "
                                f"{entry['classes']['control']['max']:.4f})")
            entry["criterion_5"] = "PASS"
        out[arm] = entry
    return out


# ---------------------------------------------------------------- rendering
def _fmt(x: Any) -> str:
    return "—" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))


def render_class_md(t: Mapping[str, Any]) -> str:
    lines = [f"# {t['class']}", "",
             f"arm `{t['arm']}` · role `{t['role']}` · {'derived' if t['derived'] else 'fit'} · primary component `{t['primary_component']}`",
             "", f"**status:** {t['status']}  ·  **criterion 5:** {t['criterion_5']}  ·  "
                 f"**admitted:** {_fmt(t.get('admitted_count'))}/{len(SEEDS)}", ""]
    lines += ["## objects", "", "| column | staged id | eligible | reason | sha256[:16] | layers |", "|---|---|---|---|---|---|"]
    for s in SEEDS:
        r = t["replicates"][f"s{s}"]
        lines.append(f"| s{s} | `{t['staged_ids'][f's{s}']}` | {r.get('eligible')} | {r.get('reason') or ''} | "
                     f"{r.get('sha256_16', '')} | {len(r.get('layers') or []) or ''} |")
    o = t["original"] or {}
    if o.get("included"):
        lines.append(f"| {ORIGINAL_LABEL} (reference, never a replicate) | `{t['original_id']}` | {o.get('eligible')} | "
                     f"{o.get('reason') or ''} | {o.get('sha256_16', '')} | {len(o.get('layers') or []) or ''} |")
    else:
        lines += ["", f"> reference column `{ORIGINAL_LABEL}` (`{t['original_id']}`): {o.get('reason') or ORIGINAL_NOT_LOADED}"]
    if t["status"] != "ok":
        lines += ["", f"> {t['status']} — no pair statistic is computed over the survivors.", ""]
        if o.get("per_layer_norms"):
            lines += ["## per-layer norms of the reference column", ""]
            lines += _norm_table({ORIGINAL_LABEL: o["per_layer_norms"]}, list(o["per_layer_norms"].values())[0].keys())
        return "\n".join(lines) + "\n"
    lines += ["", "## stacked statistics per pair", "",
              "| pair | kind | " + " | ".join(f"{c} cos | {c} relnorm" for c in COSINE_COMPONENTS[t["arm"]])
              + (" | max |Δ gate_b| |" if t["arm"] == "psr" else " |"),
              "|---|---|" + "---|---|" * len(COSINE_COMPONENTS[t["arm"]]) + ("---|" if t["arm"] == "psr" else "")]
    for p in t["pairs"]:
        cells = []
        for c in COSINE_COMPONENTS[t["arm"]]:
            cells += [_fmt(p["components"][c]["stacked_cosine"]), _fmt(p["components"][c]["stacked_relnorm"])]
        if t["arm"] == "psr":
            cells.append(_fmt(max(p["gate_b_abs_diff"].values())))
        lines.append(f"| {p['label']} | {p['kind']} | " + " | ".join(cells) + " |")
    lines += ["", "## summary (stacked cosine over pairs)", "", "| component | kind | n | mean | min | max | mean relnorm |",
              "|---|---|---|---|---|---|---|"]
    for c, kinds in t["summary"].items():
        if c == "gate_b":
            continue
        for kind, s in kinds.items():
            if s:
                lines.append(f"| {c} | {kind} | {s['n_pairs']} | {_fmt(s['mean_stacked_cosine'])} | "
                             f"{_fmt(s['min_stacked_cosine'])} | {_fmt(s['max_stacked_cosine'])} | {_fmt(s['mean_stacked_relnorm'])} |")
    lines += ["", "## per-layer cosine (primary component)", ""]
    comp = t["primary_component"]
    layers = t["layers"]
    lines += ["| pair | " + " | ".join(f"L{l}" for l in layers) + " |", "|---|" + "---|" * len(layers)]
    for p in t["pairs"]:
        pl = p["components"][comp]["per_layer_cosine"]
        lines.append(f"| {p['label']} | " + " | ".join(f"{pl[l]:.3f}" for l in layers) + " |")
    lines += ["", "## per-layer norms", ""]
    cols = {f"s{s}": t["replicates"][f"s{s}"]["per_layer_norms"] for s in SEEDS}
    if o.get("per_layer_norms"):
        cols[ORIGINAL_LABEL] = o["per_layer_norms"]
    lines += _norm_table(cols, [str(l) for l in layers])
    return "\n".join(lines) + "\n"


def _norm_table(cols: Mapping[str, Mapping[str, Mapping[str, float]]], layers) -> list[str]:
    layers = list(layers)
    out = []
    comps = sorted({c for v in cols.values() for c in v})
    for comp in comps:
        out += [f"### ‖{comp}‖ per layer", "", "| object | " + " | ".join(f"L{l}" for l in layers) + " |",
                "|---|" + "---|" * len(layers)]
        for name, byc in cols.items():
            pl = byc.get(comp) or {}
            out.append(f"| {name} | " + " | ".join(f"{pl[str(l)]:.4g}" if str(l) in pl else "—" for l in layers) + " |")
        out.append("")
    return out


def render_verdict_md(v: Mapping[str, Any], tables: Sequence[Mapping[str, Any]]) -> str:
    lines = ["# Object-space seed stability — verdicts (criterion 5)", ""]
    for arm, e in v.items():
        lines += [f"## {F.ARM_LABEL[arm]}", "", f"**criterion 5:** {e['criterion_5']}", "", e["verdict"], "",
                  "| class | role | status | mean stacked cosine | min | max |", "|---|---|---|---|---|---|"]
        for role in ("constitution", "control", "difference"):
            c = e["classes"].get(role)
            if c:
                lines.append(f"| `{c['class']}` | {role} | {c['status']} | {_fmt(c['mean'])} | {_fmt(c['min'])} | {_fmt(c['max'])} |")
        lines.append("")
    lines += ["## per-class tables", ""] + [f"- `tables/{t['class']}.json` / `.md` — {t['status']}" for t in tables]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- figures
def _series_for(t: Mapping[str, Any], stat: str) -> list[dict[str, Any]]:
    rows = []
    for p in t["pairs"]:
        ser: dict[str, dict[int, float]] = {}
        for comp in COSINE_COMPONENTS[t["arm"]]:
            key = "per_layer_cosine" if stat == "cosine" else "per_layer_relnorm"
            ser[comp] = {int(l): v for l, v in p["components"][comp][key].items()}
        if t["arm"] == "psr":
            ser["gate_b"] = {int(l): v for l, v in p["gate_b_abs_diff"].items()}
        rows.append({"label": p["label"], "kind": p["kind"], "series": ser})
    return rows


def make_figures(tables: Sequence[Mapping[str, Any]], figs_dir: Path) -> dict[str, Path]:
    figs_dir = Path(figs_dir)
    out: dict[str, Path] = {}
    for t in tables:
        if t["status"] != "ok":
            continue
        comps = list(COMPONENTS[t["arm"]])
        for stat in ("cosine", "relnorm"):
            name = f"seed_{stat}_{t['class']}"
            out[name] = F.per_layer_pairs_figure(
                class_id=t["class"], arm=t["arm"], role=t["role"], layers=t["layers"], components=comps,
                pairs=_series_for(t, stat), stat=stat, out=figs_dir / f"{name}.png",
                subtitle=("cosine similarity per targeted layer" if stat == "cosine"
                          else "relative-norm difference per targeted layer"))
    rows = []
    for t in tables:
        comp = t["primary_component"]
        s = (t.get("summary") or {}).get(comp) or {}
        rows.append({"class_id": t["class"], "arm": t["arm"], "role": t["role"], "status": t["status"],
                     "replicate": (s.get("replicate") or {}).get("stacked_cosine"),
                     "vs_original": (s.get("vs_original") or {}).get("stacked_cosine")})
    out["seed_stability_summary"] = F.stability_summary_figure(rows, figs_dir / "seed_stability_summary.png",
                                                               primary={a: PRIMARY_COMPONENT[a] for a in ("lora", "psr")})
    return out


# ---------------------------------------------------------------- driver
def run(bank: Path, out: Path, figs_dir: Path | None = None, *, make_figs: bool = True,
        classes: Sequence[ObjectClass] | None = None, roster: ClassRoster | None = None,
        include_batch8_reference: bool = False) -> dict[str, Any]:
    """``roster`` (from :func:`class_roster`) supplies the classes AND the per-seed admission records; without it
    ``classes`` (default: the declared ``CLASSES``) are analysed with no admission lookup — the library / test path."""
    bank, out = Path(bank), Path(out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    if roster is not None:
        if classes is not None:
            raise ValueError("pass either roster= or classes=, not both")
        classes = roster.classes
    elif classes is None:
        classes = CLASSES
    tables = [analyze_class(bank, cls, admissions=(roster.for_class(cls) if roster is not None else None),
                            include_batch8_reference=include_batch8_reference) for cls in classes]
    for t in tables:
        (out / "tables" / f"{t['class']}.json").write_text(json.dumps(t, indent=1, default=str) + "\n")
        (out / "tables" / f"{t['class']}.md").write_text(render_class_md(t))
    v = verdicts(tables)
    (out / "verdict.json").write_text(json.dumps(v, indent=1) + "\n")
    (out / "verdict.md").write_text(render_verdict_md(v, tables))
    figs: dict[str, Path] = {}
    if make_figs:
        figs = make_figures(tables, figs_dir or (out.parent / "figs"))
    index = {"bank": str(bank), "out": str(out), "classes": [t["class"] for t in tables],
             "class_roster": ({"source": "roster+admissions", "roster": roster.roster_path,
                               "admission_dir": roster.admission_dir} if roster is not None
                              else {"source": "declared", "roster": None, "admission_dir": None}),
             "include_batch8_reference": bool(include_batch8_reference),
             "status": {t["class"]: t["status"] for t in tables},
             "admitted_count": {t["class"]: t["admitted_count"] for t in tables},
             "criterion_5": {t["class"]: t["criterion_5"] for t in tables},
             "verdicts": {a: e["verdict"] for a, e in v.items()}, "figures": {k: str(p) for k, p in figs.items()}}
    (out / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    return {"tables": tables, "verdicts": v, "figures": figs, "index": index, "classes": list(classes)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", type=Path, default=DEFAULT_BANK, help="staged cells directory")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="tables/ + verdict.* + index.json go here")
    ap.add_argument("--roster", type=Path, default=DEFAULT_ROSTER,
                    help="the full22 roster whose replicate views (s<s>) name the classes")
    ap.add_argument("--admission-dir", type=Path, default=RS_ADMISSION,
                    help="RS/reseed/admission — the derived admission files (a refused class is still a class)")
    ap.add_argument("--target", default=TARGET, help="the roster target the replicate views hang off")
    ap.add_argument("--include-batch8-reference", action="store_true",
                    help=f"load the batch-8 seed-0 originals (base__/derived__) as the {ORIGINAL_LABEL} reference "
                         "column — OFF by default (O23: the batch-8 originals are out of the paper)")
    ap.add_argument("--figs", type=Path, default=None, help="PNG directory (default <out>/../figs)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--mirror", action="store_true", help=f"rsync the figure dir to {F.DEFAULT_MIRROR}")
    ap.add_argument("--mirror-dest", type=Path, default=F.DEFAULT_MIRROR)
    ap.add_argument("--wandb", action="store_true", help="log figures to W&B (project sdf-base-steering)")
    a = ap.parse_args(argv)
    if str(_HERE.parents[2] / "src") not in sys.path:
        sys.path.insert(0, str(_HERE.parents[2] / "src"))
    from sisteer import site as _site
    a.out = _site.expand_path(a.out, key="--out")   # a write root: an unset $GRADE refuses by name
    a.bank = _site.expand_path(a.bank, key="--bank")
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    roster = class_roster(a.roster, a.admission_dir, target=a.target)
    print(f"[object_space] class roster: {len(roster.classes)} class(es) from {a.roster} + {a.admission_dir}")
    for c in roster.classes:
        adm = roster.for_class(c)
        print(f"[object_space]   {c.cell_id:<58} {c.arm:<4} {c.role:<12} {'derived' if c.derived else 'fit':<7} "
              f"admission files: {len(adm)}" + (f" (admitted {sum(x.admitted for x in adm.values())})" if adm else ""))
    print(f"[object_space] batch-8 reference column ({ORIGINAL_LABEL}): "
          f"{'INCLUDED (--include-batch8-reference)' if a.include_batch8_reference else 'not loaded (opt-in)'}")
    res = run(a.bank, a.out, a.figs, make_figs=not a.no_figures, roster=roster,
              include_batch8_reference=a.include_batch8_reference)
    for t in res["tables"]:
        print(f"[object_space] {t['class']:<58} {t['criterion_5']:<7} {t['status']}")
    for arm, e in res["verdicts"].items():
        print(f"[object_space] verdict {arm}: {e['verdict']}")
    figs_dir = a.figs or (a.out.parent / "figs")
    if res["figures"]:
        print(f"[object_space] {len(res['figures'])} figure(s) under {figs_dir}")
    if a.mirror and res["figures"]:
        F.mirror_figs(figs_dir, a.mirror_dest)
        print(f"[object_space] mirrored to {a.mirror_dest}")
    if a.wandb and res["figures"]:
        url = F.log_wandb(res["figures"], figs_dir, config={"bank": str(a.bank), "classes": [c.cell_id for c in roster.classes],
                                                             "include_batch8_reference": a.include_batch8_reference})
        print(f"[object_space] wandb: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

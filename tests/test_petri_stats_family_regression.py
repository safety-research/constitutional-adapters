"""REGRESSION: widening the
SERVING vocabulary must not move the STATISTICS of an existing audit.

Appending ``lora_r4_ce_minus_ctl`` to ``defense_cells.CLASS_A_METHODS`` had
also widened petri_stats' FIXED confirmatory family 13 -> 14: every
constitution-vs-safety contrast, the pre-registered max-T band, the Holm
slots and the per-SI aggregate denominators are built for every family
member, so a pre-existing 13-method audit re-analysed on that tree got a
different Holm correction and LOST its simultaneous intervals and per-SI
aggregates — the two new legs never existed. The stats fixture, deriving its
roster from the widened vocabulary, hid it. The fix separates the family
(``dc.PROMPTABLE_CLASS_A_METHODS``) from the grammar (``dc.CLASS_A_METHODS``).

This test therefore analyses an UNCHANGED, pre-commit-shaped audit — the
13 methods x 3 promptable SIs, ROSTER WRITTEN OUT LITERALLY (never derived
from any vocabulary constant, so a future widening cannot silently re-shape
it) — and asserts two things:

1. against a PINNED fixture (``tests/fixtures/petri_stats_precommit_13x3.json``)
   computed ONCE from the pre-commit module (``git show 5039ce98:…petri_stats.py``
   with the pre-commit vocabulary — 13 methods, 3 SIs — shimmed in, see
   ``_precommit_module``): the adjusted statistics (per-method delta, bootstrap
   and Holm p-values, the simultaneous max-T intervals and critical t, the
   family), the per-SI class aggregates and the by-SI legs are EXACTLY what
   they were before this commit;
2. when the repo's history is at hand, the FULL analysis dict of the current
   module equals the pre-commit module's on the same frame (new keys this
   commit ADDS, listed in ``ADDED_KEYS``, are popped first and asserted to be
   the only difference).

Synthetic scores throughout (the shapes of tests/test_petri_stats.py's world,
restated here literally rather than imported, so the roster cannot drift
with that fixture either).

Since the math500 merge (2026-09-11) the GRAMMAR is 17 wide — the 13-method
family plus the sdf-base lineage's four object-only methods — while the family
stays 13; the roster test pins the two tuples separately.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri import defense_cells as dc
from sisteer.capeval.petri import petri_stats

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "petri_stats_precommit_13x3.json"
PRE_COMMIT = "5039ce98"
PRE_COMMIT_STATS = "src/sisteer/capeval/petri/petri_stats.py"

#: the pre-commit FIXED family — written out, never derived
METHODS_13 = (
    "psr_kl_all", "psr_mse_all", "vector_all", "lora_r1",
    "loreft", "lofit", "token_bos", "cast",
    "lora_r4",
    "psr_xfer_from_e2b", "psr_xfer_from_e2b_rw1",
    "psr_xfer_from_e4b", "psr_xfer_from_e4b_rw1",
)
SIS_3 = ("safety", "constitution", "irrelevant")   # the PRE-COMMIT module's promptable vocabulary: the pinned fixture is built from it
CLASS_B = ("antidote", "alphasteer", "caa_static_r")
#: the pre-commit canonical roster, literally (none, prompted x3, 13 x 3 Class A
#: arm-major with the SIs adjacent, Class B) — the manifest's cells
ROSTER = (["none"]
          + [f"prompted__{si}" for si in SIS_3]
          + [f"si_{si}__{m}" for m in METHODS_13 for si in SIS_3]
          + list(CLASS_B))
CONTROLS = ("none__lens0", "none__lora0")

BEHAVIORS3 = ("self_preservation", "defer_to_users", "instruction_evasion")
REV = "0123456789abcdef0123456789abcdef01234567"
SWEEP = "petri_family_regression"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
HARNESS_HASH = "cd" * 8
RUN_KEY = "abcdef0123456789"  # synthetic test run key  # gitleaks:allow
LENS = "proxy_vllm+lens+exact_replay"
LORA = "proxy_vllm+lora+exact_replay"
PASS = "proxy_vllm+passthrough+exact_replay"
_JITTER = (-0.4, 0.3, -0.2, 0.1, 0.4, -0.2)
_BASE_PATTERN = {"self_preservation": (3, 4, 5, 3, 4, 5),
                 "defer_to_users": (4, 5, 3, 4, 5, 3),
                 "instruction_evasion": (1, 1, 2, 1, 2, 2)}
ANALYZE_KW = dict(n_boot=200, seed=7, expected_behaviors=list(BEHAVIORS3),
                  expected_cells=list(ROSTER))
#: keys this commit ADDS to the analysis (the ONLY legitimate difference)
ADDED_KEYS = (("by_class", "class_a", "object_only_cells"),)


def _engine_for(cell: str) -> str:
    if cell in ("none", "prompted__safety", "prompted__constitution",
                "prompted__irrelevant"):
        return PASS
    if cell == "none__lens0":
        return LENS
    if cell == "none__lora0":
        return LORA
    return LORA if cell.endswith("__lora_r1") else LENS


def _method_delta(method: str) -> float:
    return 0.5 + 0.25 * METHODS_13.index(method)


def precommit_world() -> pd.DataFrame:
    """The 13 x 3 audit as a score frame — every row literal to this file."""
    items = {b: [f"s{i}" for i in range(6)] for b in BEHAVIORS3}

    def leg(pattern, shift=0.0, jitter_on=False):
        out = {}
        for b in BEHAVIORS3:
            vals = {}
            for k, item in enumerate(items[b]):
                v = pattern[b][k] + shift
                if jitter_on:
                    v += _JITTER[k]
                vals[item] = float(np.clip(v, 1.0, 10.0))
            out[b] = vals
        return out

    scores = {"none": leg(_BASE_PATTERN)}
    for ctl in CONTROLS:
        scores[ctl] = leg(_BASE_PATTERN)
    prompted = {b: (2, 3, 2, 3, 2, 3) for b in BEHAVIORS3}
    scores["prompted__safety"] = leg(prompted)
    scores["prompted__constitution"] = leg(prompted, shift=0.5)
    scores["prompted__irrelevant"] = leg(prompted, shift=0.25)
    arm = {b: (4, 5, 3, 4, 5, 3) for b in BEHAVIORS3}
    for m in METHODS_13:
        scores[f"si_safety__{m}"] = leg(arm)
        scores[f"si_constitution__{m}"] = leg(arm, shift=_method_delta(m),
                                              jitter_on=True)
        scores[f"si_irrelevant__{m}"] = leg(arm, shift=0.1)
    for cell in CLASS_B:
        scores[cell] = leg({b: (3, 3, 4, 3, 4, 4) for b in BEHAVIORS3})
    rows = []
    for cell, per_b in scores.items():
        for behavior, per_item in per_b.items():
            for item, value in per_item.items():
                for metric, v in ((behavior, value), ("eval_awareness", 5.0),
                                  ("scenario_realism", 6.0)):
                    rows.append({"arm_name": cell, "task": f"petri/{behavior}",
                                 "metric": metric, "item_id": item,
                                 "value": float(v), "engine": _engine_for(cell),
                                 "intervention_hash": "ab" * 8,
                                 "sweep": SWEEP, "model_id": MODEL_ID,
                                 "model_revision": REV,
                                 "harness_config_hash": HARNESS_HASH,
                                 "run_id": f"{SWEEP}.{cell}.{RUN_KEY[:8]}"})
    return pd.DataFrame(rows)


def _precommit_module():
    """The pre-commit ``petri_stats`` loaded under a temporary module name,
    with the PRE-COMMIT vocabulary shimmed in: the old source reads
    ``dc.CLASS_A_METHODS`` (13 then) and ``cm.PETRI_SI_IDS`` (3 then) off the
    modules it imports, which on this tree are the widened ones — so both
    are replaced by namespace copies carrying the old tuples. None when the
    history is not at hand (a shallow CI checkout)."""
    try:
        src = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{PRE_COMMIT}:{PRE_COMMIT_STATS}"],
            check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    name = "_petri_stats_precommit_5039ce98"
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(REPO / PRE_COMMIT_STATS)
    sys.modules[name] = mod
    try:
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    finally:
        sys.modules.pop(name, None)
    old_dc = types.SimpleNamespace(**vars(dc))
    old_dc.CLASS_A_METHODS = METHODS_13
    old_cm = types.SimpleNamespace(**vars(cm))
    old_cm.PETRI_SI_IDS = SIS_3
    mod.dc, mod.cm = old_dc, old_cm
    return mod


def _load_bearing(analysis: dict) -> dict:
    """The statistics the review named, in a JSON-stable shape."""
    pm = analysis["per_method_macro"]
    methods = {m: {k: pm["methods"][m].get(k)
                   for k in ("delta", "p_boot", "p_holm", "p_holm_placeholder",
                             "ci95_simultaneous_maxt", "se_boot", "status")}
               for m in pm["family"]}
    class_a = analysis["by_class"]["class_a"]
    return {
        "family": list(pm["family"]), "family_size": pm["family_size"],
        "n_available": pm["n_available"], "maxt_band": pm["maxt_band"],
        "maxt_critical_t": pm["maxt_critical_t"], "methods": methods,
        "per_si_macro": {si: {k: e.get(k) for k in
                              ("status", "mean", "n_methods_expected",
                               "n_methods_available", "unavailable_cells", "cells")}
                         for si, e in class_a["per_si_macro"].items()},
        "expected_cells": list(class_a["expected_cells"]),
        "by_si": {m: {si: (leg["status"], leg.get("macro_mean"), leg.get("macro_status"))
                      for si, leg in legs.items()}
                  for m, legs in analysis["by_si"].items()},
        "prompted_reference_keys": sorted(analysis["prompted_references"]["contrasts"]),
        "completeness": {"expected_canonical": analysis["completeness"]["expected_canonical"],
                         "n_complete": analysis["completeness"]["n_complete"]},
    }


def _assert_same(a, b, path="root"):
    """Deep equality with exact ints/strings and 1e-12-relative floats."""
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a) == set(b), f"{path}: keys {sorted(set(a) ^ set(b))}"
        for k in a:
            _assert_same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        assert len(a) == len(b), f"{path}: len {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        if a is None or b is None or (isinstance(a, float) and math.isnan(a)):
            assert (b is None or (isinstance(b, float) and math.isnan(b))) and \
                   (a is None or math.isnan(a)), f"{path}: {a!r} != {b!r}"
        else:
            assert a == pytest.approx(b, rel=1e-12, abs=1e-12), f"{path}: {a!r} != {b!r}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def _pop(d: dict, path: tuple):
    for k in path[:-1]:
        d = d[k]
    return d.pop(path[-1])


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=str, allow_nan=True))


def test_the_roster_is_the_pre_commit_one_and_not_derived():
    assert len(METHODS_13) == 13 and len(ROSTER) == 1 + 3 + 39 + 3
    # the current vocabulary widened the GRAMMAR but not the FAMILY (13): 14 on
    # main's port (the derived LoRA arm), 17 since the math500 merge (the
    # sdf-base lineage's four object-only methods — the document-CE fits and
    # both derived arms, every one staged under a corpus SI only). The serving
    # vocabulary is the family plus EXACTLY the object-only tuple, in that order
    assert tuple(dc.PROMPTABLE_CLASS_A_METHODS) == METHODS_13
    assert tuple(dc.PROMPTABLE_CLASS_A_METHODS) + tuple(dc.OBJECT_ONLY_METHODS) == tuple(dc.CLASS_A_METHODS)
    assert set(dc.OBJECT_ONLY_METHODS) == {"psr_ce_all", "lora_r4_ce",
                                          "psr_ce_all_minus_ctl_distilled", "lora_r4_ce_minus_ctl",
                                          "psr_ce_all_minus_ctl"}   # the paired write (2026-09-17)
    assert len(dc.CLASS_A_METHODS) == 13 + len(dc.OBJECT_ONLY_METHODS) == 18
    assert not set(METHODS_13) & set(dc.OBJECT_ONLY_METHODS)
    # the CURRENT module adds constitution_full — the FULL 34k constitution as a prompted APPENDIX arm, registered for
    # sdfbase-llama8b as an appendix, not a paper arm; the pre-commit fixture above is built from SIS_3 alone, which is
    # what the pinned audit statistics were computed over
    assert tuple(cm.PETRI_PROMPTABLE_SI_IDS) == SIS_3 + ("constitution_full",)


def test_pre_commit_audit_statistics_are_unchanged_against_the_pinned_fixture():
    analysis = petri_stats.analyze(precommit_world(), **ANALYZE_KW)
    got = _jsonable(_load_bearing(analysis))
    want = json.loads(FIXTURE.read_text())
    assert want["source"]["commit"] == PRE_COMMIT
    _assert_same(got, want["load_bearing"])
    # the properties the review named, stated outright
    pm = analysis["per_method_macro"]
    assert pm["family"] == list(METHODS_13) and pm["family_size"] == 13
    assert pm["maxt_band"] == "ok" and pm["n_available"] == 13
    for m in METHODS_13:
        assert pm["methods"][m]["ci95_simultaneous_maxt"] is not None
        assert "p_holm_placeholder" not in pm["methods"][m]
    per_si = analysis["by_class"]["class_a"]["per_si_macro"]
    assert set(per_si) == set(SIS_3)
    assert all(e["status"] == "ok" and e["n_methods_expected"] == 13
               for e in per_si.values())
    assert set(analysis["by_si"]) == set(METHODS_13)
    assert all(set(legs) == set(SIS_3) for legs in analysis["by_si"].values())
    # nothing of the derived arm — nor of any object-only method — anywhere in
    # the family or its aggregates
    blob = json.dumps({k: analysis[k] for k in
                       ("per_method_macro", "by_si", "prompted_references", "by_class")},
                      default=str)
    assert "lora_r4_ce_minus_ctl" not in blob
    assert all(m not in blob for m in dc.OBJECT_ONLY_METHODS)
    assert analysis["by_class"]["class_a"]["object_only_cells"] == []


def test_pre_commit_module_and_current_module_agree_on_the_full_analysis():
    old = _precommit_module()
    if old is None:
        pytest.skip("repo history (git show) not available here")
    frame = precommit_world()
    before = _jsonable(old.analyze(frame, **ANALYZE_KW))
    after = _jsonable(petri_stats.analyze(frame, **ANALYZE_KW))
    added = {path: _pop(after, path) for path in ADDED_KEYS}
    assert added == {("by_class", "class_a", "object_only_cells"): []}
    _assert_same(after, before)

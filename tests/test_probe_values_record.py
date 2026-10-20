"""The fit-record's per-example probe payload (2026-08-24).

Every fit already computed per-example probe values, the selection's sha8
alignment keys, and (when stratified) a per-family table — and every writer
dropped them: the writers were built FLAT, so all eight scalar fields of
ProbeEstimate survived while all four structured ones (arrays, dicts)
silently vanished. Recovering the arrays meant re-probing finished objects
on a GPU; without them, a cross-cell comparison probed on the SAME examples
(e.g. an all-position fit vs its response-masked twin) had to use the
independence formula ``sqrt(sem_A^2 + sem_C^2)`` for its difference sem,
which overstates it by up to ~3x when the legs correlate — enough to record
a real position effect as "no detectable effect".

These tests pin, model-free where possible and against the real writers
where it matters:

- the ENUMERATION contract: every field ProbeEstimate carries is either
  mapped to a persisted key or excluded BY NAME with a reason, introspected
  from the constructor — so the next structured field cannot vanish the way
  these four did (general diagnostic, worth remembering when auditing any
  writer: check whether it drops everything non-scalar);
- the round trip: values come back from JSON intact and in order;
- THE USE CASE: two cells sharing a selected_sha8 population align by that
  key and yield a paired difference sem that beats the independence formula
  when the legs correlate;
- backward compatibility: every legacy aggregate key is untouched, and an
  older sidecar without the new fields still parses;
- degeneracy: empty legs, absent per_family, single-example probes write
  nulls/short arrays — a fit that trained for hours must never die in its
  writer.

SCOPE OF THE ENUMERATION GUARD, stated so an auditor reads it rather than
infers it: the field map + these tests prove coverage at the three KNOWN
serialisation sites (train_steered_lm's diagnostics, train_lockstep's
per-arm diagnostics, the driver's fit_record) and say nothing about a
FOURTH site appearing — a new writer can still be born flat. The
reproducible check for new sites is enumerating the paths that OBTAIN
per-example values (callers of arm_probe_losses / probe_steered_lm, plus
train_lockstep's _probe_pair lists) and following each to its writes —
never grepping persisted key names, which cannot see a writer that never
wrote them. Emitters that legitimately carry no payload (float-only
guards, scalar metrics streams) are declined BY NAME with reasons in
distill.PROBE_EMISSION_POINTS_EXEMPT, checked below.

CPU-only: the tiny_llama fixture, synthetic invented prompts only.
"""

from __future__ import annotations

import inspect
import json
import math
import random

import pytest
import torch
from torch import nn

from sisteer.data.instructions import BY_ID
from sisteer.data.pairs import Pair, pairs_to_distill_examples
from sisteer.extraction.distill import (
    PROBE_EMISSION_POINTS_EXEMPT,
    PROBE_ESTIMATE_RECORD_KEYS,
    PROBE_ESTIMATE_UNPERSISTED,
    ProbeEstimate,
    _round_sig7,
    probe_family_table,
    probe_values_record,
    train_steered_lm,
)
from sisteer.extraction.lockstep import (
    LockstepArm,
    build_lockstep_object,
    default_lockstep_params,
    train_lockstep,
)
from sisteer.models import hidden_size

INSTRUCTION = BY_ID["french"]


def _pairs():
    data = [
        ("alpha question 0 about the sky.", "Le ciel est bleu", "x"),
        ("alpha question 1 about the sky.", "Le ciel est grand", "x"),
        ("beta question 0 about the sea.", "La mer est calme", "x"),
        ("beta question 1 about the sea.", "La mer est froide", "x"),
        ("gamma question 0 about a tree.", "Un grand arbre vert", "x"),
        ("gamma question 1 about a tree.", "Un petit arbre", "x"),
    ]
    return [Pair(prompt=p, pos=pos, neg=neg, instruction_id="french")
            for p, pos, neg in data]


def _full_estimate() -> ProbeEstimate:
    """A ProbeEstimate with every field populated, values chosen so any
    reorder/truncation/precision bug is visible."""
    acts = (0.111111111111, 0.2, 0.987654321987654, 0.4)
    noops = (0.05, 0.06060606060606, 0.07, 0.08)
    return ProbeEstimate(
        sum(acts) / 4, sum(noops) / 4,
        active_sd=0.1, active_sem=0.05, noop_sd=0.01, noop_sem=0.005,
        ratio_sem=0.2, n_examples=4, n_batches=2, batch_size=2,
        active_values=acts, noop_values=noops,
        selected_sha8=("00c0ffee", "deadbeef", "0badf00d", "cafebabe"),
        per_family=probe_family_table(
            ["fam_a", "fam_a", "fam_b", "fam_b"], list(acts), list(noops)),
    )


def _record(est: ProbeEstimate) -> dict:
    """A fit record the way the writers build one: hand-built aggregate
    scalars (untouched legacy keys) + the per-example payload."""
    return {
        "final_probe_loss": est.active, "noop_floor": est.noop,
        "final_probe_sd": est.active_sd, "final_probe_sem": est.active_sem,
        "noop_floor_sd": est.noop_sd, "noop_floor_sem": est.noop_sem,
        "floor_ratio": est.ratio, "floor_ratio_sem": est.ratio_sem,
        "probe_n": est.n_examples, "probe_n_batches": est.n_batches,
        "probe_batch_size": est.batch_size,
        "probe_selected_sha8": list(est.selected_sha8 or ()),
        **probe_values_record(active_values=est.active_values,
                              noop_values=est.noop_values,
                              per_family=est.per_family),
    }


# --------------------------------------------------------------------------- #
# The enumeration contract                                                     #
# --------------------------------------------------------------------------- #


def test_every_probe_estimate_field_is_persisted_or_excluded_by_name():
    """Every field ProbeEstimate's constructor carries must be EITHER mapped
    to a sidecar key (PROBE_ESTIMATE_RECORD_KEYS) OR listed by name, with a
    stated reason, in PROBE_ESTIMATE_UNPERSISTED. Introspected from the
    class itself — never a hand-copied list, which goes stale the same
    silent way the fields did. This is the standing corrective for the
    2026-08-24 gap: a flat writer kept all 8 scalars and dropped all 4
    structured fields, and no single-owner review could see it because both
    sides were locally correct — the defect lived in the join. (Auditor's
    shortcut this test encodes: ask of any writer whether it drops
    everything non-scalar.)"""
    fields = [n for n in inspect.signature(ProbeEstimate.__new__).parameters
              if n != "cls"]
    assert fields, "introspection found no constructor fields"

    unaccounted = [f for f in fields
                   if f not in PROBE_ESTIMATE_RECORD_KEYS
                   and f not in PROBE_ESTIMATE_UNPERSISTED]
    assert not unaccounted, (
        f"ProbeEstimate field(s) {unaccounted} are not plumbed into the fit "
        "record: add each to PROBE_ESTIMATE_RECORD_KEYS (and emit it at "
        "every write site — distill.train_steered_lm diagnostics, "
        "lockstep.train_lockstep diagnostics, the driver's fit_record) OR "
        "add it to PROBE_ESTIMATE_UNPERSISTED with the reason it is "
        "deliberately dropped. Do not let it vanish silently.")

    stale = [f for f in (*PROBE_ESTIMATE_RECORD_KEYS,
                         *PROBE_ESTIMATE_UNPERSISTED)
             if f not in fields]
    assert not stale, (
        f"{stale} appear in the persistence contract but are no longer "
        "ProbeEstimate fields — remove the stale entries")

    both = set(PROBE_ESTIMATE_RECORD_KEYS) & set(PROBE_ESTIMATE_UNPERSISTED)
    assert not both, f"{sorted(both)} are both persisted and excluded"

    # The record builder honours the map: a fully-populated estimate's
    # record carries every mapped key (floor_ratio rides along as a derived
    # extra — it is ProbeEstimate.ratio, not a constructor field).
    rec = _record(_full_estimate())
    missing = [k for k in PROBE_ESTIMATE_RECORD_KEYS.values() if k not in rec]
    assert not missing, f"record is missing mapped key(s) {missing}"


def test_writer_level_exemptions_each_state_a_reason():
    """The field map binds the record-writing sites; emitters that
    legitimately never carry the payload (floor_ratio_check's float-only
    log_fn and friends) are declined BY NAME in
    PROBE_EMISSION_POINTS_EXEMPT — an empty field-level exclusion set next
    to a complete-looking map must not read as "persisted everywhere".
    Every entry must state a non-trivial reason: a decline is a decision
    on record, never an absence that reads as coverage."""
    assert PROBE_EMISSION_POINTS_EXEMPT, (
        "the exempt-emitter list cannot be empty: floor_ratio_check's "
        "log_fn alone receives floats and cannot carry arrays")
    assert "floor_ratio_check.log_fn" in PROBE_EMISSION_POINTS_EXEMPT
    for name, reason in PROBE_EMISSION_POINTS_EXEMPT.items():
        assert isinstance(name, str) and name.strip(), name
        assert isinstance(reason, str) and len(reason.strip()) >= 20, (
            f"exempt emitter {name!r} must state a real reason, "
            f"got {reason!r}")


# --------------------------------------------------------------------------- #
# Round trip                                                                   #
# --------------------------------------------------------------------------- #


def test_record_round_trips_values_in_order():
    est = _full_estimate()
    rec = json.loads(json.dumps(_record(est)))

    assert rec["probe_active_values"] == [
        _round_sig7(v) for v in est.active_values]
    assert rec["probe_noop_values"] == [
        _round_sig7(v) for v in est.noop_values]
    # 7 significant digits keep the mean consistent with the recorded
    # aggregate to well past any probe's resolution
    assert sum(rec["probe_active_values"]) / 4 == pytest.approx(
        rec["final_probe_loss"], rel=1e-6)
    # the alignment key is VERBATIM — 8 hex chars, order preserved
    assert rec["probe_selected_sha8"] == list(est.selected_sha8)
    # per_family travels intact
    assert set(rec["probe_per_family"]) == {"fam_a", "fam_b"}
    assert rec["probe_per_family"]["fam_a"]["n"] == 2
    assert rec["probe_per_family"]["fam_b"]["ratio"] == pytest.approx(
        est.per_family["fam_b"]["ratio"])


def test_rounding_is_seven_significant_digits_none_and_nan_safe():
    assert _round_sig7(0.123456789123456) == 0.1234568
    assert _round_sig7(1.23456789e-7) == 1.234568e-07
    assert _round_sig7(None) is None
    assert math.isnan(_round_sig7(float("nan")))
    # torch scalars are accepted (the lockstep lists can carry them)
    assert _round_sig7(torch.tensor(0.25)) == 0.25


# --------------------------------------------------------------------------- #
# THE use case: paired cross-cell difference sem                               #
# --------------------------------------------------------------------------- #


def _mean_sd_sem(vals):
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var), math.sqrt(var / len(vals))


def test_two_cells_align_by_sha8_and_paired_sem_beats_independence():
    """The change's reason to exist: cells A and C are probed on the SAME
    examples, so their per-example errors correlate, and the sem of
    ``mean_A - mean_C`` must come from the PAIRED per-example differences.
    The independence formula ``sqrt(sem_A^2 + sem_C^2)`` ignores the
    covariance and overstates the difference sem several-fold here — the
    inflation that would record a real ~0.08 position effect as "no
    detectable effect". Alignment is BY selected_sha8 KEY, not by array
    position: cell C's record arrives in a different (shuffled) selection
    order, exactly what re-pairing legs measured in separate passes means."""
    rng = random.Random(20260824)
    n = 64
    sha8s = [f"{rng.getrandbits(32):08x}" for _ in range(n)]
    base = {s: rng.gauss(1.0, 1.0) for s in sha8s}          # shared signal
    a = {s: base[s] + rng.gauss(0.0, 0.2) for s in sha8s}   # correlated legs
    c = {s: base[s] + rng.gauss(0.0, 0.2) for s in sha8s}

    # cell A in selection order; cell C in a DIFFERENT order (own pass)
    order_c = sha8s[:]
    rng.shuffle(order_c)
    rec_a = json.loads(json.dumps({
        "probe_selected_sha8": sha8s,
        **probe_values_record(active_values=[a[s] for s in sha8s]),
    }))
    rec_c = json.loads(json.dumps({
        "probe_selected_sha8": order_c,
        **probe_values_record(active_values=[c[s] for s in order_c]),
    }))

    # align by key — the whole point of persisting selected_sha8 verbatim
    a_by = dict(zip(rec_a["probe_selected_sha8"],
                    rec_a["probe_active_values"]))
    c_by = dict(zip(rec_c["probe_selected_sha8"],
                    rec_c["probe_active_values"]))
    assert set(a_by) == set(c_by)
    diffs = [a_by[s] - c_by[s] for s in a_by]

    _, _, paired_sem = _mean_sd_sem(diffs)
    _, _, sem_a = _mean_sd_sem(rec_a["probe_active_values"])
    _, _, sem_c = _mean_sd_sem(rec_c["probe_active_values"])
    indep_sem = math.sqrt(sem_a ** 2 + sem_c ** 2)

    # the paired sem resolves what independence smears out
    assert paired_sem < 0.5 * indep_sem, (paired_sem, indep_sem)

    # and POSITIONAL pairing (ignoring the key) is simply wrong: the shared
    # signal doesn't cancel, so its "paired" sem lands near the independence
    # value, several times the true one
    naive = [x - y for x, y in zip(rec_a["probe_active_values"],
                                   rec_c["probe_active_values"])]
    _, _, naive_sem = _mean_sd_sem(naive)
    assert naive_sem > 2.0 * paired_sem, (naive_sem, paired_sem)


# --------------------------------------------------------------------------- #
# Backward compatibility                                                       #
# --------------------------------------------------------------------------- #

_LEGACY_AGGREGATE_KEYS = (
    "final_probe_loss", "final_probe_sem", "noop_floor", "noop_floor_sem",
    "floor_ratio", "floor_ratio_sem", "probe_n", "probe_n_batches",
)


def test_new_fields_are_purely_additive_over_the_legacy_aggregates():
    """A reader of the current aggregate keys sees byte-identical values —
    the payload keys are new names only, nothing renamed, reordered out
    from under a key, or removed."""
    est = _full_estimate()
    rec = _record(est)
    legacy = {k: rec[k] for k in _LEGACY_AGGREGATE_KEYS}
    assert legacy == {
        "final_probe_loss": est.active, "final_probe_sem": est.active_sem,
        "noop_floor": est.noop, "noop_floor_sem": est.noop_sem,
        "floor_ratio": est.ratio, "floor_ratio_sem": est.ratio_sem,
        "probe_n": 4, "probe_n_batches": 2,
    }


def test_older_sidecar_without_the_new_fields_still_parses():
    """A sidecar written before this change (aggregates only) parses, its
    legacy keys read normally, and the payload keys read as absent — the
    degradation a paired analysis must handle is `.get(...) is None`,
    never a KeyError at parse time."""
    old = json.loads(json.dumps({
        "final_probe_loss": 0.3, "final_probe_sem": 0.03,
        "noop_floor": 0.1, "noop_floor_sem": 0.01,
        "floor_ratio": 3.0, "floor_ratio_sem": 0.4,
        "probe_n": 256, "probe_n_batches": 128,
    }))
    assert old["floor_ratio"] == 3.0
    for k in ("probe_active_values", "probe_noop_values",
              "probe_selected_sha8", "probe_per_family"):
        assert old.get(k) is None


# --------------------------------------------------------------------------- #
# Degenerate cases                                                             #
# --------------------------------------------------------------------------- #


def test_degenerate_payloads_write_nulls_and_never_raise():
    # everything absent -> explicit nulls, JSON-clean
    empty = json.loads(json.dumps(probe_values_record()))
    assert empty == {"probe_active_values": None, "probe_noop_values": None,
                     "probe_per_family": None}

    # empty legs -> empty arrays (a degenerate probe, stated as such)
    hollow = probe_values_record(active_values=(), noop_values=())
    assert hollow["probe_active_values"] == []
    assert hollow["probe_noop_values"] == []

    # a single-example probe -> length-1 arrays next to null sems
    est = ProbeEstimate(0.5, 0.4, active_values=(0.5,), noop_values=(0.4,),
                        selected_sha8=("00c0ffee",), n_examples=1,
                        n_batches=1, batch_size=1)
    rec = json.loads(json.dumps(_record(est)))
    assert rec["probe_active_values"] == [0.5]
    assert rec["probe_noop_values"] == [0.4]
    assert rec["probe_selected_sha8"] == ["00c0ffee"]
    assert rec["final_probe_sem"] is None
    assert rec["probe_per_family"] is None

    # the heldout key family shares the machinery
    h = probe_values_record(active_values=(0.1,), prefix="heldout_probe")
    assert set(h) == {"heldout_probe_active_values",
                      "heldout_probe_noop_values", "heldout_probe_per_family"}


# --------------------------------------------------------------------------- #
# The real writers (tiny model): every mapped key reaches the record           #
# --------------------------------------------------------------------------- #


def _assert_payload(diag: dict, where: str) -> None:
    missing = [k for k in PROBE_ESTIMATE_RECORD_KEYS.values()
               if k not in diag]
    assert not missing, f"{where}: record is missing {missing}"
    n = diag["probe_n"]
    acts, noops = diag["probe_active_values"], diag["probe_noop_values"]
    sha8 = diag["probe_selected_sha8"]
    assert len(acts) == len(noops) == len(sha8) == n, where
    assert all(isinstance(s, str) and len(s) == 8 for s in sha8), where
    # rounded on write: 7-significant-digit rounding is idempotent
    assert all(v == _round_sig7(v) for v in acts + noops), where
    # the arrays REPRODUCE the recorded aggregates (mean over examples)
    assert sum(acts) / n == pytest.approx(
        diag["final_probe_loss"], rel=1e-5, abs=1e-9), where
    assert sum(noops) / n == pytest.approx(
        diag["noop_floor"], rel=1e-5, abs=1e-9), where
    # ...and survive an actual JSON round trip in order
    assert json.loads(json.dumps(acts)) == acts, where


def test_train_steered_lm_diagnostics_carry_the_full_payload(tiny_llama):
    model, tok = tiny_llama
    examples = pairs_to_distill_examples(tok, _pairs(), INSTRUCTION)
    vec = nn.Parameter(torch.zeros(hidden_size(model)))
    diag: dict = {}
    train_steered_lm(model, tok, examples, 1, lambda idxs: vec, [vec],
                     loss="soft", epochs=1, batch_size=2, max_len=96,
                     diagnostics=diag)
    _assert_payload(diag, "train_steered_lm")
    # this path passes no family labels: a stated null, never a crash
    assert diag["probe_per_family"] is None


def test_train_lockstep_diagnostics_carry_the_full_payload(tiny_llama):
    model, tok = tiny_llama
    pairs = _pairs()
    examples = pairs_to_distill_examples(tok, pairs, INSTRUCTION)
    group = [LockstepArm("vector",
                         build_lockstep_object("vector", model, tok,
                                               INSTRUCTION, pairs, layer=1),
                         default_lockstep_params("vector"))]
    diag: dict = {}
    train_lockstep(model, tok, group, examples, epochs=1, batch_size=2,
                   max_len=96, seed=0, diagnostics=diag, n_probe=None)
    d = diag["vector"]
    _assert_payload(d, "train_lockstep")
    # the group probe is unstratified: per_family is a stated null
    assert d["probe_per_family"] is None
    # the noop leg is the STARTUP floor per example, so the ratio's
    # covariance-aware sem is recomputable from the persisted arrays
    assert len(d["probe_noop_values"]) == d["probe_n"]

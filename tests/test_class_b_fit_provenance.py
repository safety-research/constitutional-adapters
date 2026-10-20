"""Class B fit records resolve (sweep.CLASS_B_FIT_WRITERS), and every other record shape resolves EXACTLY as before.

Measured 2026-09-11: the AlphaSteer strength screen's first canary (cluster B job 93844) died in the controls driver at
`_mirror_provenance` — `writer 'fit_class_b.py@9efb857e…' is not in sweep.SI_PAIR_FIT_WRITERS and the record carries no
budget schema`. The objects are fine and the producing lane's own tree resolved them (its sweep.py predates the
three-outcomes fit-kind rule); the Class B fitter had simply never been registered on this lineage.

The two halves that matter, both pinned here: the writer is matched on its HEAD (the recorded string carries a commit
stamp, and it differs per target), and registering it must not move a document or si_pair record by one field.

SELECTION is driven through the REAL reader (`read_stage3_fit_provenance` on a temp cell dir holding object.pt + the
sidecar, then `resolve_training_mirror` end to end) — never through a re-spelling of its chain in the test, which would
test a copy of the logic (the reviewer's finding on the first cut of this file). `_record()` below builds only the
NORMALISED record for the DOWNSTREAM half (`_provenance_from_record`), and its fit_kind comes from the reader too.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sisteer.eval import sweep  # noqa: E402

NEMO = "fit_class_b.py@9efb857ee6232e999f1233855915983f10e7451c"   # the real stamps, both targets' spellings
OTHERS = "fit_class_b.py@18e6374d2e601025ac43a58723ded854038bfdac"


def _cell(tmp_path: Path, writer: str, **extra) -> Path:
    """A staged cell as the bank view lays it out: object.pt (arbitrary bytes) + fit_provenance.json in the Class B
    fitter's shape (the real nemotron parent's keys), with `writer` as given."""
    d = tmp_path / "alphasteer"
    d.mkdir(parents=True, exist_ok=True)
    (d / "object.pt").write_bytes(b"not-a-real-blob-" + writer.encode())
    rec = {"chat_template_hash": "ab7813c3abdd9cb6", "positions_at_fit": "prefill",
           "resolved_model_id": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
           "resolved_revision": "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f", "writer": writer, **extra}
    (d / "fit_provenance.json").write_text(json.dumps(rec))
    return d


def _record(tmp_path: Path, writer, **over):
    """The NORMALISED record for the downstream half — read by the REAL reader from a temp cell, then edited with
    `over` (a budget marker, a bypass claim, a bound flag) exactly as read_stage3_fit_provenance would have carried it."""
    d = _cell(tmp_path, writer, **{k: v for k, v in over.items() if k in ("training_budget_kind", "completion_tokens",
                                                                             "chat_template_bypassed")})
    rec = sweep.read_stage3_fit_provenance(d, object_path=d / "object.pt")
    assert rec is not None, "the reader returned no record for a complete sidecar"
    for k, v in over.items():
        if k not in ("training_budget_kind", "completion_tokens", "chat_template_bypassed"):
            rec[k] = v
    return rec


@pytest.mark.parametrize("writer", [NEMO, OTHERS, "fit_class_b.py"])
def test_a_class_b_cell_resolves_through_the_real_reader_and_keeps_the_template_hash(tmp_path, writer):
    """END TO END: the cell dir -> read_stage3_fit_provenance -> resolve_training_mirror, the driver's own path."""
    d = _cell(tmp_path, writer)
    rec = sweep.read_stage3_fit_provenance(d, object_path=d / "object.pt")
    assert rec["fit_kind"] == sweep.FIT_KIND_CLASS_B and "Class B fitter" in rec["fit_kind_evidence"]
    prov = sweep.resolve_training_mirror("alphasteer", str(d / "object.pt"))
    assert prov.fit_kind == sweep.FIT_KIND_CLASS_B == "class_b"
    assert prov.status == "verified"                       # NOT the document waiver
    assert prov.trained_template_hash == "ab7813c3abdd9cb6"   # so served==trained actually COMPARES
    assert "Class B fitter" in prov.fit_kind_evidence or "fit_class_b" in prov.fit_kind_evidence
    assert prov.positions_at_fit == "prefill"


def test_the_writer_is_matched_on_its_head_not_the_whole_stamped_string(tmp_path):
    assert sweep.writer_head(NEMO) == "fit_class_b.py" == sweep.writer_head(OTHERS)
    assert sweep.writer_head("tune_arm.py") == "tune_arm.py"
    assert NEMO not in sweep.CLASS_B_FIT_WRITERS          # the exact string never matches — that was the bug shape
    # through the REAL reader: a stamped SI-pair writer resolves too, which exact membership also failed to do
    d = _cell(tmp_path / "si", "tune_arm.py@" + "b" * 40)
    rec = sweep.read_stage3_fit_provenance(d, object_path=d / "object.pt")
    assert rec["fit_kind"] == sweep.FIT_KIND_SI_PAIR
    assert sweep.resolve_training_mirror("x", str(d / "object.pt")).fit_kind == sweep.FIT_KIND_SI_PAIR


def test_a_class_b_cell_claiming_a_bypassed_template_still_refuses(tmp_path):
    d = _cell(tmp_path, NEMO, chat_template_bypassed=True)
    with pytest.raises(sweep.UnrecordedMirrorError, match="contradicts itself"):
        sweep.resolve_training_mirror("alphasteer", str(d / "object.pt"))


def test_an_unregistered_writer_still_refuses_end_to_end_and_names_both_lists(tmp_path):
    d = _cell(tmp_path, "mystery_fitter.py@" + "c" * 40)
    with pytest.raises(sweep.UnrecordedMirrorError, match="establishes NO fit kind") as ei:
        sweep.resolve_training_mirror("alphasteer", str(d / "object.pt"))
    assert "SI_PAIR_FIT_WRITERS" in str(ei.value) and "CLASS_B_FIT_WRITERS" in str(ei.value)


def test_document_and_si_pair_records_are_untouched_by_the_registration(tmp_path):
    """The reader's BUDGET-SCHEMA branches (the LoRA cells' path) are read through the real reader too, then the
    downstream half is checked; the registration must not have moved either by one field."""
    doc = _record(tmp_path / "doc", "train_two_si_objects", training_budget_kind=sweep.DOCUMENT_TOKENS_KIND,
                  object_sha_bound=True)
    assert doc["fit_kind"] == sweep.FIT_KIND_DOCUMENT
    prov = sweep._provenance_from_record(object_sha="a" * 64, rec=doc)
    assert prov.fit_kind == sweep.FIT_KIND_DOCUMENT and prov.status == sweep.DOCUMENT_FIT_STATUS
    pair = _record(tmp_path / "pair", "train_two_si_objects", completion_tokens=512)
    assert pair["fit_kind"] == sweep.FIT_KIND_SI_PAIR
    prov2 = sweep._provenance_from_record(object_sha="a" * 64, rec=pair)
    assert prov2.fit_kind == sweep.FIT_KIND_SI_PAIR and prov2.status == "verified"
    # an unbound document record still refuses, and a partial record stays partial
    with pytest.raises(sweep.UnrecordedMirrorError, match="not BOUND to these bytes"):
        sweep._provenance_from_record(object_sha="a" * 64, rec=_record(
            tmp_path / "unbound", "train_two_si_objects", training_budget_kind=sweep.DOCUMENT_TOKENS_KIND))
    part = _record(tmp_path / "part", NEMO); part["status"] = "partial"; part["fit_kind"] = None
    assert sweep._provenance_from_record(object_sha="a" * 64, rec=part).status == "partial"


def test_the_kind_vocabulary_is_three_named_classes():
    assert sweep.FIT_KIND_CLASS_B not in (sweep.FIT_KIND_SI_PAIR, sweep.FIT_KIND_DOCUMENT)
    # the in-repo refusal-direction fitter joined the Class B writers (gap-refusal-pin, 2026-10-12): its publication's fit_provenance.json is the
    # identity sidecar the stager and the gate read; tests/test_refusal_direction_fit.py holds the REAL publication's record to this entry
    assert sweep.CLASS_B_FIT_WRITERS == ("fit_class_b.py", "scripts/arditi/fit_arditi.py", "scripts/arditi/fit_refusal_direction.py")


# ---------------------------------------------------------------------------------------------------------------
# THE ARDITI WRITER: `scripts/arditi/fit_arditi.py` writes the four Arditi cells' records with every required key
# measured and `object_sha256: null` — the record self-identifies by writer. Array 495677 rows 34/36/38 died at the
# mirror on it. Admitting the writer must NOT loosen the bytes check: a null-sha record from a known writer BINDS
# only when the object bytes hash to the caller's pin (the row's admission pin, the launcher's
# --expect-object-sha256), a mismatch refuses, and an unknown writer still refuses whatever the bytes. Driven
# through the REAL reader end to end.
# ---------------------------------------------------------------------------------------------------------------

ARDITI = "scripts/arditi/fit_arditi.py"


def _arditi_cell(tmp_path: Path, writer: str = ARDITI, **extra) -> tuple[Path, str]:
    """A staged Arditi cell as the strength screen lays it out (the real gemma ar_r0p05 record's keys, with
    `object_sha256: null` written EXPLICITLY, as the records carry it) -> (cell dir, the object's real sha256)."""
    import hashlib
    d = tmp_path / "arditi"
    d.mkdir(parents=True, exist_ok=True)
    blob = b"arditi-refusal-direction-" + writer.encode()
    (d / "object.pt").write_bytes(blob)
    rec = {"alpha": 0.05, "cell_id": "arditi", "chat_template_hash": "0a2c8073c878ab1d", "dose_unit": "residual_norm_fraction",
           "phase": "prefill", "positions_at_fit": "all", "resolved_model_id": "google/gemma-4-E4B-it",
           "resolved_revision": "ee0ef602" + "0" * 32, "staged_id": "ar_r0p05__arditi", "object_sha256": None, "writer": writer,
           **extra}
    (d / "fit_provenance.json").write_text(json.dumps(rec))
    return d, hashlib.sha256(blob).hexdigest()


def test_an_arditi_record_with_a_null_sha_binds_only_when_the_bytes_match_the_pin(tmp_path):
    d, sha = _arditi_cell(tmp_path)
    # through the reader: class_b, verified, template hash intact, BOUND through the pin and said so
    rec = sweep.read_stage3_fit_provenance(d, object_path=d / "object.pt", expected_object_sha256=sha)
    assert rec["fit_kind"] == sweep.FIT_KIND_CLASS_B and "Class B fitter" in rec["fit_kind_evidence"]
    assert rec["object_sha_bound"] is True and rec["object_sha_bound_by"] == "expected_object_sha256"
    assert rec["chat_template_hash"] == "0a2c8073c878ab1d" and rec["positions_at_fit"] == "all"
    # end to end, the driver's own path
    prov = sweep.resolve_training_mirror("arditi", str(d / "object.pt"), expected_object_sha256=sha)
    assert prov.fit_kind == "class_b" and prov.status == "verified" and prov.trained_template_hash == "0a2c8073c878ab1d"
    assert prov.source == f"fit_provenance:{ARDITI}"


def test_an_arditi_record_whose_bytes_do_not_match_the_pin_refuses_by_name(tmp_path):
    d, sha = _arditi_cell(tmp_path)
    other = "f" * 64
    with pytest.raises(sweep.UnrecordedMirrorError, match="the row's pin names other bytes") as ei:
        sweep.resolve_training_mirror("arditi", str(d / "object.pt"), expected_object_sha256=other)
    assert sha[:12] in str(ei.value) and other[:12] in str(ei.value)
    with pytest.raises(sweep.UnrecordedMirrorError, match="the row's pin names other bytes"):
        sweep.read_stage3_fit_provenance(d, object_path=d / "object.pt", expected_object_sha256=other)


def test_an_arditi_record_without_a_pin_is_a_stated_unknown_never_a_pass_and_never_a_refusal(tmp_path):
    """The reader's historical semantics for a null-sha record (every fit_class_b.py record is one): `object_sha_bound`
    None. The MATH-500 launcher always passes the pin; a caller that does not gets the unknown it always got."""
    d, _sha = _arditi_cell(tmp_path)
    rec = sweep.read_stage3_fit_provenance(d, object_path=d / "object.pt")
    assert rec["fit_kind"] == sweep.FIT_KIND_CLASS_B and rec["object_sha_bound"] is None and rec["object_sha_bound_by"] is None
    assert sweep.resolve_training_mirror("arditi", str(d / "object.pt")).fit_kind == "class_b"


def test_an_unknown_writer_still_refuses_even_when_the_bytes_match_the_pin(tmp_path):
    d, sha = _arditi_cell(tmp_path, writer="scripts/arditi/fit_arditi_v2.py")
    with pytest.raises(sweep.UnrecordedMirrorError, match="establishes NO fit kind") as ei:
        sweep.resolve_training_mirror("arditi", str(d / "object.pt"), expected_object_sha256=sha)
    assert "CLASS_B_FIT_WRITERS" in str(ei.value)


def test_a_record_carrying_its_own_sha_is_held_to_the_bytes_and_to_the_pin(tmp_path):
    """Both stated facts must agree with the artifact: a record sha that matches + a pin that matches binds `by record`;
    a record sha that mismatches refuses as before (whatever the pin says); a pin that mismatches refuses too."""
    d, sha = _arditi_cell(tmp_path, object_sha256=None)
    (d / "fit_provenance.json").write_text(json.dumps({**json.loads((d / "fit_provenance.json").read_text()), "object_sha256": sha}))
    rec = sweep.read_stage3_fit_provenance(d, object_path=d / "object.pt", expected_object_sha256=sha)
    assert rec["object_sha_bound"] is True and rec["object_sha_bound_by"] == "record"
    (d / "fit_provenance.json").write_text(json.dumps({**json.loads((d / "fit_provenance.json").read_text()), "object_sha256": "e" * 64}))
    with pytest.raises(sweep.UnrecordedMirrorError, match="stale object/sidecar pair"):
        sweep.read_stage3_fit_provenance(d, object_path=d / "object.pt", expected_object_sha256=sha)
    with pytest.raises(sweep.UnrecordedMirrorError, match="the row's pin names other bytes"):
        sweep.read_stage3_fit_provenance(d, object_path=d / "object.pt", expected_object_sha256="e" * 64)


def test_the_fit_class_b_records_bind_through_the_pin_the_same_way(tmp_path):
    """The rule is the READER's, not the Arditi writer's: the canonical AlphaSteer views' fit_class_b.py records carry
    no object_sha256 either, and bind through the pin identically."""
    import hashlib
    d = _cell(tmp_path, NEMO)
    sha = hashlib.sha256((d / "object.pt").read_bytes()).hexdigest()
    rec = sweep.read_stage3_fit_provenance(d, object_path=d / "object.pt", expected_object_sha256=sha)
    assert rec["object_sha_bound"] is True and rec["object_sha_bound_by"] == "expected_object_sha256"
    with pytest.raises(sweep.UnrecordedMirrorError, match="the row's pin names other bytes"):
        sweep.resolve_training_mirror("alphasteer", str(d / "object.pt"), expected_object_sha256="0" * 64)


def test_the_driver_threads_its_pin_into_the_mirror():
    t = (REPO / "scripts/served_capability_run.py").read_text()
    assert "def _mirror_provenance(cell, *, expected_object_sha256=None):" in t
    assert "expected_object_sha256=expected_object_sha256)" in t
    assert "prov = _mirror_provenance(cell, expected_object_sha256=(args.expect_object_sha256 or None))" in t


# ---------------------------------------------------------------------------------------------------------------
# THE BOUNDARY OF THE WIDENING (addressed here).
#
# `writer_head` splits on the FIRST `@` and the membership test is exact and case-sensitive, so none of the
# spellings below can reach a registered name. The reviewer confirmed that by reading the production path; these
# pin it, so it stays true. Each case is a NEAR MISS of a registered writer and each must still REFUSE — a
# widening is only as good as the cases just outside it.
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("writer", [
    "scripts/fit_class_b.py",          # directory-qualified: the head keeps the path, so it is not the bare name
    "alphasteer_f22/fit_class_b.py",   # the real relative path a caller might record
    "xfit_class_b.py",                 # prefix
    "fit_class_b.py.bak",              # suffix
    "fit_class_b",                     # extension dropped
    "FIT_CLASS_B.PY",                  # case variant: membership is case-SENSITIVE
    "Fit_Class_B.py",
    " fit_class_b.py",                 # leading whitespace is not stripped
    "fit_class_b.py ",                 # trailing whitespace is not stripped
    "scripts/tune_arm.py",             # the same near miss against the OTHER list
    "tune_arm.py.orig",
    "fit_arditi.py",                   # the Arditi writer is registered under its PATH-QUALIFIED spelling; the bare name is not it
    "arditi/fit_arditi.py",
    "scripts/arditi/fit_arditi.py.bak",
    "scripts/arditi/Fit_Arditi.py",
])
def test_a_near_miss_of_a_registered_writer_still_refuses(tmp_path, writer):
    """None of these is a registered writer, so each establishes NO fit kind and the mirror refuses. If one of
    them ever resolves, the membership test has been loosened into a prefix/suffix/case match and the widening
    has stopped being the deliberate registration it is documented as."""
    d = _cell(tmp_path, writer)
    assert sweep.writer_head(writer) not in sweep.CLASS_B_FIT_WRITERS
    assert sweep.writer_head(writer) not in sweep.SI_PAIR_FIT_WRITERS
    with pytest.raises(sweep.UnrecordedMirrorError, match="establishes NO fit kind"):
        sweep.resolve_training_mirror("alphasteer", str(d / "object.pt"))


@pytest.mark.parametrize("writer,kind", [
    ("fit_class_b.py@" + "d" * 40, "class_b"),      # a well-formed stamp
    ("fit_class_b.py@garbage", "class_b"),          # a MALFORMED one resolves identically
    ("fit_class_b.py@", "class_b"),                 # an empty stamp
    ("fit_class_b.py@a@b", "class_b"),              # only the FIRST `@` splits
    ("tune_arm.py@garbage", "si_pair"),             # and the same on the SI-pair list
])
def test_the_stamp_SUFFIX_is_not_validated_this_is_classification_not_authentication(tmp_path, writer, kind):
    """PINNED POLICY, deliberately permissive.

    Everything after the first `@` is IGNORED: a malformed, empty or repeated stamp resolves exactly as a
    well-formed one does. That is intended. This field classifies a fit RECIPE by the name of the script that
    wrote it; it does not authenticate a commit, and treating it as authentication would be false assurance —
    the string is written by the same producer whose record is being read, so it could carry any value.

    It is also NOT a new bypass. Before the head-matching change a BARE registered name was already accepted
    unconditionally, so no writer gains admission here that a bare spelling would not have had. What the change
    fixed is the reverse failure: an exact-membership test could never match a STAMPED writer, and every real
    record carries a stamp (which differs per target), so every real object was being refused.

    The assurance that a registered writer's record is truthful comes from elsewhere and is measured, not
    asserted: the object sha binding, and the served==trained template comparison that this kind deliberately
    RUNS rather than waives. If the stamp is ever made meaningful, it must be validated against something
    external to the record, and this test should then be inverted rather than deleted."""
    d = _cell(tmp_path, writer)
    rec = sweep.read_stage3_fit_provenance(d, object_path=d / "object.pt")
    assert rec["fit_kind"] == kind
    assert sweep.resolve_training_mirror("alphasteer", str(d / "object.pt")).fit_kind == kind


# ---------------------------------------------------------------------------------------------------------------
# THE DERIVED-OBJECT WRITER: `scripts/alphasteer_f22/bake_rungs.py` re-bakes an AlphaSteer rung as a dose-scaled
# CHILD and writes the child-bound identity record — its OWN object_sha256, the parent under `derived_from` (a
# different sha; the parent's fit record verbatim under `record`). The writer string carries a parenthetical
# annotation after a space (the REAL string is reproduced below, read off the primary cluster's re-staged llama
# rung). A record from this writer binds ONLY in that shape with its own sha equal to the bytes; every other shape
# refuses, and an unknown writer refuses whatever the shape.
# ---------------------------------------------------------------------------------------------------------------

BAKE_RUNGS_REAL = ("scripts/alphasteer_f22/bake_rungs.py (dose-scaled rung; the FIT's own record is preserved verbatim under "
                   "derived_from.record, which is where 'Lane B slice_job.py (alphasteer re-bake canon, 2026-09-11T23:53:42Z, "
                   "overflow job 93858): A16-pre_s0.3 parent (as22 fit_arm.py (A16-pre); acts_pre; sha ce75438b81f5) SLICED to "
                   "layers 10..19, base_scale 0.3 -> 1.0; per-layer rho RECOMPUTED on acts_pre held-out harmful (primary job "
                   "486857)' and the parent's lineage live)")
BAKE_RUNGS = "scripts/alphasteer_f22/bake_rungs.py"


def _derived_cell(tmp_path: Path, *, writer: str = BAKE_RUNGS_REAL, own="bytes", parent_sha=None, record=None,
                  derived_from="default", **extra) -> tuple[Path, str]:
    """A re-staged dose-scaled rung as the lane lays it out: object.pt + the CHILD-BOUND fit_provenance.json.
    `own="bytes"` records the artifact's real sha; a string records that instead; None omits the key."""
    import hashlib
    d = tmp_path / "alphasteer"
    d.mkdir(parents=True, exist_ok=True)
    blob = b"dose-scaled-alphasteer-child-" + tmp_path.name.encode()
    (d / "object.pt").write_bytes(blob)
    sha = hashlib.sha256(blob).hexdigest()
    parent_sha = parent_sha or "645a9f89084ac87557583b89abf393d8cf9e1f96aec6a3d237aa4631463de9b8"
    parent_record = {"writer": "Lane B slice_job.py (alphasteer re-bake canon)", "object_sha256": parent_sha,
                     "chat_template_hash": "b48c47f644389271", "positions_at_fit": "prefill",
                     "resolved_model_id": "NousResearch/Meta-Llama-3.1-8B-Instruct", "resolved_revision": "d10aef79" + "0" * 32}
    rec = {"schema": "alphasteer_rung_identity/1", "writer": writer,
           "resolved_model_id": "NousResearch/Meta-Llama-3.1-8B-Instruct", "resolved_revision": "d10aef79" + "0" * 32,
           "chat_template_hash": "b48c47f644389271", "positions_at_fit": None, **extra}
    if own == "bytes":
        rec["object_sha256"] = sha
    elif own is not None:
        rec["object_sha256"] = own
    if derived_from == "default":
        rec["derived_from"] = {"object_sha256": parent_sha, "record": (parent_record if record is None else record),
                               "record_sha256": "e" * 64, "why_rebound": "the child is different bytes from its parent"}
    elif derived_from is not None:
        rec["derived_from"] = derived_from
    (d / "fit_provenance.json").write_text(json.dumps(rec))
    return d, sha


def test_writer_head_strips_the_parenthetical_annotation_like_a_stamp_and_nothing_else():
    assert sweep.writer_head(BAKE_RUNGS_REAL) == BAKE_RUNGS
    assert sweep.writer_head(BAKE_RUNGS + " (x)") == BAKE_RUNGS and sweep.writer_head(BAKE_RUNGS + "  (x)") == BAKE_RUNGS
    assert sweep.writer_head(BAKE_RUNGS) == BAKE_RUNGS
    assert sweep.writer_head("fit_class_b.py@" + "a" * 40) == "fit_class_b.py"            # the stamp still strips
    assert sweep.writer_head("fit_class_b.py@abc (note)") == "fit_class_b.py"             # whichever comes first
    assert sweep.writer_head("fit_class_b.py (note)@abc") == "fit_class_b.py"
    assert sweep.writer_head("Lane B slice_job.py (alphasteer re-bake canon)") == "Lane B slice_job.py"   # the OLD writer: not registered
    # nothing else is normalised: whitespace, case and a path prefix all survive into the exact membership test
    assert sweep.writer_head(" fit_class_b.py") == " fit_class_b.py" and sweep.writer_head("fit_class_b.py ") == "fit_class_b.py "
    assert sweep.writer_head("scripts/fit_class_b.py") == "scripts/fit_class_b.py"
    assert sweep.writer_head("bake_rungs.py(x)") == "bake_rungs.py(x)"                    # no whitespace before "(": not an annotation
    assert sweep.DERIVED_CLASS_B_FIT_WRITERS == (BAKE_RUNGS,) and BAKE_RUNGS not in sweep.CLASS_B_FIT_WRITERS


def test_a_child_bound_derived_record_binds_through_the_real_reader_with_and_without_the_pin(tmp_path):
    d, sha = _derived_cell(tmp_path)
    rec = sweep.read_stage3_fit_provenance(d, object_path=d / "object.pt")
    assert rec["fit_kind"] == sweep.FIT_KIND_CLASS_B and "DERIVED-object Class B writer" in rec["fit_kind_evidence"]
    assert rec["object_sha_bound"] is True and rec["object_sha_bound_by"] == "record"
    assert rec["chat_template_hash"] == "b48c47f644389271" and rec["positions_at_fit"] is None
    prov = sweep.resolve_training_mirror("alphasteer", str(d / "object.pt"))
    assert prov.fit_kind == "class_b" and prov.status == "verified" and prov.trained_template_hash == "b48c47f644389271"
    assert prov.source == f"fit_provenance:{BAKE_RUNGS_REAL}"
    # the row's pin beside it: agrees -> binds; disagrees -> refuses by name
    assert sweep.resolve_training_mirror("alphasteer", str(d / "object.pt"), expected_object_sha256=sha).fit_kind == "class_b"
    with pytest.raises(sweep.UnrecordedMirrorError, match="the row's pin names other bytes"):
        sweep.resolve_training_mirror("alphasteer", str(d / "object.pt"), expected_object_sha256="1" * 64)


def test_the_parent_bound_record_that_killed_row_39_still_refuses_as_a_stale_pair(tmp_path):
    """The re-staging's WHY: the old sidecar was the parent's record (object_sha256 = the PARENT's) copied into the child's
    dir — a stale object/sidecar pair, refused before and refused now, whatever the writer says."""
    d, _sha = _derived_cell(tmp_path, own="645a9f89084ac87557583b89abf393d8cf9e1f96aec6a3d237aa4631463de9b8")
    with pytest.raises(sweep.UnrecordedMirrorError, match="stale object/sidecar pair"):
        sweep.resolve_training_mirror("alphasteer", str(d / "object.pt"))
    d2, _ = _derived_cell(tmp_path / "old", writer="Lane B slice_job.py (alphasteer re-bake canon, 2026-09-11T23:53:42Z)",
                          own="645a9f89084ac87557583b89abf393d8cf9e1f96aec6a3d237aa4631463de9b8", derived_from=None)
    with pytest.raises(sweep.UnrecordedMirrorError, match="stale object/sidecar pair"):
        sweep.resolve_training_mirror("alphasteer", str(d2 / "object.pt"))
    # a GARBAGE own sha is other bytes too: the bytes binding refuses it before the shape is even judged
    d3, _ = _derived_cell(tmp_path / "garbage", own="abc")
    with pytest.raises(sweep.UnrecordedMirrorError, match="stale object/sidecar pair"):
        sweep.read_stage3_fit_provenance(d3, object_path=d3 / "object.pt")


@pytest.mark.parametrize("case,kw,needle", [
    ("no own sha", dict(own=None), "must carry its OWN object_sha256"),
    ("no derived_from", dict(derived_from=None), "must carry `derived_from`"),
    ("derived_from not a mapping", dict(derived_from="parent"), "must carry `derived_from`"),
    ("parent sha malformed", dict(derived_from={"object_sha256": "zz", "record": {"a": 1}}), "parent's 64-hex digest"),
    ("parent == own", dict(derived_from="self"), "cannot be its own parent"),
    ("no parent record", dict(derived_from={"object_sha256": "7" * 64}), "derived_from.record"),
    ("empty parent record", dict(derived_from={"object_sha256": "7" * 64, "record": {}}), "derived_from.record"),
])
def test_a_bake_rungs_record_outside_the_child_bound_shape_establishes_no_kind_and_refuses(tmp_path, case, kw, needle):
    if kw.get("derived_from") == "self":
        d, sha = _derived_cell(tmp_path)
        rec = json.loads((d / "fit_provenance.json").read_text()); rec["derived_from"]["object_sha256"] = sha
        (d / "fit_provenance.json").write_text(json.dumps(rec))
    else:
        d, _sha = _derived_cell(tmp_path, **kw)
    read = sweep.read_stage3_fit_provenance(d, object_path=d / "object.pt")
    assert read["fit_kind"] is None and needle in read["fit_kind_evidence"], (case, read["fit_kind_evidence"])
    with pytest.raises(sweep.UnrecordedMirrorError, match="establishes NO fit kind") as ei:
        sweep.resolve_training_mirror("alphasteer", str(d / "object.pt"))
    assert "DERIVED-object Class B writer" in str(ei.value) and needle in str(ei.value)


def test_a_derived_record_with_no_artifact_to_bind_to_is_refused_not_admitted(tmp_path):
    d, _sha = _derived_cell(tmp_path)
    rec = sweep.read_stage3_fit_provenance(d)                     # no object_path: nothing to bind to
    assert rec["fit_kind"] is None and "no artifact was given to bind it to" in rec["fit_kind_evidence"]
    assert rec["object_sha_bound"] is None


def test_an_unknown_writer_in_the_child_bound_shape_still_refuses(tmp_path):
    d, sha = _derived_cell(tmp_path, writer="scripts/alphasteer_f22/bake_rungs_v2.py (dose-scaled rung)")
    with pytest.raises(sweep.UnrecordedMirrorError, match="establishes NO fit kind") as ei:
        sweep.resolve_training_mirror("alphasteer", str(d / "object.pt"), expected_object_sha256=sha)
    assert "DERIVED_CLASS_B_FIT_WRITERS" in str(ei.value)
    for near in ("scripts/alphasteer_f22/bake_rungs.py.bak (x)", "alphasteer_f22/bake_rungs.py (x)", "bake_rungs.py (x)",
                 "scripts/alphasteer_f22/Bake_Rungs.py (x)"):
        assert sweep.writer_head(near) not in sweep.DERIVED_CLASS_B_FIT_WRITERS, near
        d2, _ = _derived_cell(tmp_path / near.replace("/", "_").replace(" ", "_"), writer=near)
        with pytest.raises(sweep.UnrecordedMirrorError, match="establishes NO fit kind"):
            sweep.resolve_training_mirror("alphasteer", str(d2 / "object.pt"))

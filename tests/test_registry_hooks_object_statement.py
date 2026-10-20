"""hooks.object_from_sidecar's two WRITER statements (demo-import M1b; Q14 and its corollary): `base_scale` on a non-dose object and
`fit_layer_index_convention` beside an EXPLICIT `fit_layers_json`. Synthetic sidecar directories (invented bytes, placeholder ids).

The convention: over a sidecar dir with NO layer list (a derived_meta.json-only dir, as the committed differences and dose copies are; a
fit_provenance.json-only dir, as the committed refusal direction is), an explicit `fit_layers_json` WITHOUT the keyword REFUSES by name —
the pre-M1b outcome (a stated list, no convention: `setdefault` over the None the hook had already written) is pinned as the refusal, never
accepted; WITH it the payload carries both columns and `field_confidence_json.fit_layer_index_convention` names where the sidecar records
the list (derived_meta.json:layers; fit_provenance.json:<block>.served_layer; else the writer's statement); a value outside
vocab.LAYER_INDEX_CONVENTIONS refuses; a convention with no list refuses. Over a train_budget_meta.json fixture WITH target_layers, no
keyword keeps `zero_based_decoder_block` (today's behaviour, unchanged) and a disagreeing keyword refuses by name.
`base_scale`: accepted on derivation none (recorded in field_confidence_json.base_scale with its source), REFUSED on a dose_scale copy (the
stamp is the statement), REFUSED when the sidecar's own coefficient (fit_provenance.json `c`, top-level or in a block) disagrees, and
refused when not a finite number. emit_object_from_sidecar forwards both through **more (its dry run previews the same payload).

The served block an explicit zero-based list is compared against is read at the TOP LEVEL of
fit_provenance.json and in EVERY block — a flat provenance compares, two recorded values that disagree refuse by name, agreeing values are one;
and whenever the payload states fit_code_sha 'unverified' (a caller's fit_provenance, or a producer block with no HEAD) the hook writes
field_confidence_json.fit_code_sha itself, naming itself as the source, so validate.py's rule is met by the writer that states the value."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from sisteer.registry import hooks  # noqa: E402
from sisteer.registry import validate as VAL  # noqa: E402
from sisteer.registry import vocab as V  # noqa: E402

PROD = hooks.emit.producer_block("tests/test_registry_hooks_object_statement.py")
PARENTS = ["1" * 64, "2" * 64]


def _dir(tmp_path: Path, name: str, sidecars: dict, payload: bytes = b"synthetic object bytes") -> Path:
    d = tmp_path / name; d.mkdir(parents=True, exist_ok=True)
    (d / "object.pt").write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    for fname, doc in sidecars.items():
        doc = dict(doc); doc.setdefault("object_sha256", sha) if fname != "fit_provenance.json" else None
        (d / fname).write_text(json.dumps(doc))
    return d


def _derived_dir(tmp_path: Path, name: str = "derived", **over) -> Path:
    dm = {"kind": "lora_subtraction", "derived_id": "mix__lora_minus_ctl", "layers": [0, 1], "derivation": "delta(pos) - delta(neg)",
          "parent_pos": {"object_sha256": PARENTS[0], "cell_id": "mix__lora", "target_layers": [0, 1]},
          "parent_neg": {"object_sha256": PARENTS[1], "cell_id": "ctl__lora", "target_layers": [0, 1]}}
    dm.update(over)
    return _dir(tmp_path, name, {"derived_meta.json": dm})


DERIVED_KW = dict(campaign="synthetic", object_kind="lora_subtraction", paper_label="difference_lora", object_role="difference", fit_target_kind="derived",
                  fit_objective="not_applicable", derivation="lora_subtraction", fit_model_id="qwen35_4b_base", status="approved", parent_object_ids_json=PARENTS)


def _payload(d: Path, **kw) -> dict:
    return hooks.object_from_sidecar(str(d), producer=PROD, **kw)[0]


def test_an_explicit_list_over_a_sidecar_with_no_list_needs_a_stated_convention(tmp_path):
    d = _derived_dir(tmp_path)
    with pytest.raises(ValueError, match="explicit fit_layers_json for .* needs a stated fit_layer_index_convention .*never guesses one"):
        _payload(d, fit_layers_json=[0, 1], **DERIVED_KW)                                     # the pre-M1b outcome, now a refusal
    p = _payload(d, fit_layers_json=[0, 1], fit_layer_index_convention="zero_based_decoder_block", **DERIVED_KW)
    assert p["fit_layers_json"] == [0, 1] and p["fit_layer_index_convention"] == "zero_based_decoder_block" and p["fit_layer_count"] == 2
    fc = p["field_confidence_json"]
    assert fc["fit_layer_index_convention"] == {"confidence": "recorded", "source": "derived_meta.json:layers"}
    assert fc["fit_layers_json"] == {"confidence": "recorded", "source": "derived_meta.json:layers"}
    # another admitted convention is stated as given; a list the sidecar does not record names the writer as its source
    p = _payload(d, fit_layers_json=[1, 2], fit_layer_index_convention="one_based", **DERIVED_KW)
    assert p["fit_layer_index_convention"] == "one_based" and p["field_confidence_json"]["fit_layer_index_convention"]["source"].startswith("stated by the writer")
    with pytest.raises(ValueError, match="fit_layer_index_convention 'decoder_layers' is not one of"):
        _payload(d, fit_layers_json=[0, 1], fit_layer_index_convention="decoder_layers", **DERIVED_KW)
    with pytest.raises(ValueError, match="stated with no fit_layers_json to describe"):
        _payload(d, fit_layer_index_convention="zero_based_decoder_block", **DERIVED_KW)
    # without an explicit list a derived dir states no layers and no convention (the sidecar's `layers` is not a column the hook reads)
    p = _payload(d, **DERIVED_KW)
    assert "fit_layers_json" not in p and "fit_layer_index_convention" not in p
    assert set(V.LAYER_INDEX_CONVENTIONS) >= {"zero_based_decoder_block", "one_based"}


def test_a_fit_provenance_dir_names_its_served_layer_as_the_source(tmp_path):
    prov = {"arditi_allpos": {"c": 0.3, "served_layer": 12, "stored_layer": 12, "fit_layer": 13}, "writer": "scripts/arditi/fit_arditi.py",
            "resolved_model_id": "Qwen/Qwen3.5-4B", "positions_at_fit": "all"}
    d = _dir(tmp_path, "arditi", {"fit_provenance.json": prov})
    kw = dict(campaign="arditi_allpos", object_kind="vector_all", paper_label="arditi_allpos", object_role="baseline_published", fit_target_kind="contrastive_direction",
              fit_objective="arditi_search", fit_model_id="qwen35_4b_it", status="approved", fit_contrast_sets_json={"unverified": "no sets"})
    with pytest.raises(ValueError, match="needs a stated fit_layer_index_convention"):
        _payload(d, fit_layers_json=[12], **kw)
    p = _payload(d, fit_layers_json=[12], fit_layer_index_convention="zero_based_decoder_block", base_scale=0.3, **kw)
    assert p["fit_layers_json"] == [12] and p["fit_layer_index_convention"] == "zero_based_decoder_block" and p["base_scale"] == 0.3
    fc = p["field_confidence_json"]
    assert fc["fit_layer_index_convention"]["source"] == "fit_provenance.json:arditi_allpos.served_layer"
    assert fc["base_scale"] == {"confidence": "recorded", "source": "fit_provenance.json:arditi_allpos.c"}
    # [13] (the fit-time index) is not what the block serves: an explicit ZERO-BASED list that is not the provenance's served block REFUSES by
    # name — an edited pin over the same bytes is a writer's error, never filed as "stated by the writer" (the registry owner's review)
    with pytest.raises(ValueError, match=r"states fit_layers_json=\[13\] \(zero_based_decoder_block\) but fit_provenance.json:arditi_allpos.served_layer records the served block 12"):
        _payload(d, fit_layers_json=[13], fit_layer_index_convention="zero_based_decoder_block", **kw)
    with pytest.raises(ValueError, match="records the served block 12"):
        _payload(d, fit_layers_json=[12, 13], fit_layer_index_convention="zero_based_decoder_block", **kw)
    # another convention is not the served block's and is not compared with it (a one-based 13 IS the zero-based 12): stated as given
    p = _payload(d, fit_layers_json=[13], fit_layer_index_convention="one_based", **kw)
    assert p["fit_layer_index_convention"] == "one_based" and p["field_confidence_json"]["fit_layer_index_convention"]["source"].startswith("stated by the writer")
    # a provenance with no served block compares nothing (the derived sidecars' case)
    d2 = _dir(tmp_path, "arditi_no_served", {"fit_provenance.json": {"arditi_allpos": {"c": 0.3, "fit_layer": 13}, "writer": "scripts/arditi/fit_arditi.py"}})
    p = _payload(d2, fit_layers_json=[13], fit_layer_index_convention="zero_based_decoder_block", **kw)
    assert p["fit_layers_json"] == [13] and p["field_confidence_json"]["fit_layers_json"]["source"].startswith("stated by the writer")


def test_a_sidecar_list_keeps_the_hooks_convention_and_a_disagreeing_statement_refuses(tmp_path):
    tbm = {"cell_id": "ctl__psr", "si_id": "ctl", "model_id": "Qwen/Qwen3.5-4B-Base", "seed": 1, "target_layers": [0, 1, 2], "chat_template_bypassed": True,
           "resolved_batch_size": 2, "batch_size": 2, "sample_manifest_sha": "a" * 64, "positions_at_fit": "all"}
    d = _dir(tmp_path, "fit", {"train_budget_meta.json": tbm})
    kw = dict(campaign="synthetic", object_kind="relu_steer", paper_label="control_relu", object_role="control", fit_target_kind="corpus_document_fit",
              fit_objective="ce", fit_thinking_render="not_applicable", fit_model_id="qwen35_4b_base", status="approved")
    p = _payload(d, **kw)                                                                     # today's behaviour, unchanged
    assert p["fit_layers_json"] == [0, 1, 2] and p["fit_layer_index_convention"] == "zero_based_decoder_block" and "field_confidence_json" not in p
    p = _payload(d, fit_layer_index_convention="zero_based_decoder_block", **kw)              # an agreeing statement is no statement
    assert p["fit_layer_index_convention"] == "zero_based_decoder_block"
    with pytest.raises(ValueError, match="states fit_layer_index_convention 'one_based' but the sidecar's layer list \\(train_budget_meta.json:target_layers\\) is recorded as 'zero_based_decoder_block'"):
        _payload(d, fit_layer_index_convention="one_based", **kw)


FIT_KW = dict(campaign="synthetic", object_kind="relu_steer", paper_label="control_relu", object_role="control", fit_target_kind="corpus_document_fit",
              fit_objective="ce", fit_thinking_render="not_applicable", fit_model_id="qwen35_4b_base", status="approved")


def test_fit_effective_batch_is_the_sidecars_recorded_effective_batch_never_the_micro_batch(tmp_path):
    """train_budget_meta.json records the fit's EFFECTIVE batch under `effective_batch` (micro-batch x accum_steps —
    the key the objects_bank adapter files the column from) beside the MICRO-batch `resolved_batch_size` / `batch_size`; the hook used to state
    the micro-batch as fit_effective_batch (2 for the paper's batch-64 fits). The recorded effective batch is the column; the micro-batch stands
    in only for a sidecar that records none (today's behaviour for those), and a non-integer `effective_batch` is no statement."""
    base = {"cell_id": "ctl__psr", "si_id": "ctl", "model_id": "Qwen/Qwen3.5-4B-Base", "seed": 1, "target_layers": [0, 1, 2], "chat_template_bypassed": True,
            "sample_manifest_sha": "a" * 64, "positions_at_fit": "all"}
    p = _payload(_dir(tmp_path, "eb64", {"train_budget_meta.json": dict(base, batch_size=2, resolved_batch_size=2, accum_steps=32, effective_batch=64)}), **FIT_KW)
    assert p["fit_effective_batch"] == 64
    assert p["fit_hyperparameters_json"]["effective_batch"] == 64 and p["fit_hyperparameters_json"]["resolved_batch_size"] == 2 and p["fit_hyperparameters_json"]["accum_steps"] == 32
    # no recorded effective batch: the resolved micro-batch, else batch_size, as before
    p = _payload(_dir(tmp_path, "micro", {"train_budget_meta.json": dict(base, batch_size=4, resolved_batch_size=2)}), **FIT_KW)
    assert p["fit_effective_batch"] == 2
    p = _payload(_dir(tmp_path, "bs", {"train_budget_meta.json": dict(base, batch_size=4)}), **FIT_KW)
    assert p["fit_effective_batch"] == 4
    # a bool / string / null `effective_batch` is not an integer statement: the fallback stands
    for bad in (True, "64", None):
        p = _payload(_dir(tmp_path, f"bad_{type(bad).__name__}", {"train_budget_meta.json": dict(base, batch_size=2, resolved_batch_size=2, effective_batch=bad)}), **FIT_KW)
        assert p["fit_effective_batch"] == 2, bad
    # the dry-run preview of the fit writers' hook carries the same column
    recs = hooks.emit_object_from_sidecar(str(_dir(tmp_path, "eb64_emit", {"train_budget_meta.json": dict(base, batch_size=2, resolved_batch_size=2, accum_steps=32, effective_batch=64)})),
                                          script="tests", dry_run=True, registry_root=str(tmp_path / "noreg"), **FIT_KW)
    assert [p for t, p in recs if t == "object"][0]["fit_effective_batch"] == 64


def test_base_scale_is_accepted_on_a_non_dose_object_and_refused_on_a_dose_copy_or_against_the_sidecars_coefficient(tmp_path):
    d = _derived_dir(tmp_path)
    p = _payload(d, base_scale=0.3, fit_layers_json=[0, 1], fit_layer_index_convention="zero_based_decoder_block", **DERIVED_KW)
    assert p["base_scale"] == 0.3 and p["field_confidence_json"]["base_scale"] == {"confidence": "recorded", "source": "stated by the writer (no sidecar records a coefficient)"}
    p = _payload(d, fit_layers_json=[0, 1], fit_layer_index_convention="zero_based_decoder_block", **DERIVED_KW)
    assert "base_scale" not in p                                                              # no statement, no column
    for bad in ("x", float("nan"), float("inf")):
        with pytest.raises(ValueError, match="base_scale=.* is not a (finite )?number"):
            _payload(d, base_scale=bad, fit_layers_json=[0, 1], fit_layer_index_convention="zero_based_decoder_block", **DERIVED_KW)
    # a dose copy: the stamp is the statement, a caller value is refused (even one that agrees)
    copy = _derived_dir(tmp_path, "copy", dose_scaled_from={"scale": 0.75, "parent_object_sha256": "3" * 64, "producer": "scripts/sdfbase/scale_lora_dose.py"})
    ckw = dict(DERIVED_KW, derivation="dose_scale", parent_object_ids_json=["3" * 64], campaign="sdfbase_full22")
    p = _payload(copy, fit_layers_json=[0, 1], fit_layer_index_convention="zero_based_decoder_block", **ckw)
    assert p["base_scale"] == 0.75 and p["fit_layer_index_convention"] == "zero_based_decoder_block"
    with pytest.raises(ValueError, match="states base_scale=0.75 on a 'dose_scale' copy — the dose stamp is the statement"):
        _payload(copy, base_scale=0.75, fit_layers_json=[0, 1], fit_layer_index_convention="zero_based_decoder_block", **ckw)
    # the sidecar's own coefficient wins: a top-level `c`, or a block's
    for prov in ({"c": 0.3, "writer": "w"}, {"arditi_allpos": {"c": 0.3}, "writer": "w"}):
        a = _dir(tmp_path, f"ar_{len(prov)}_{list(prov)[0]}", {"fit_provenance.json": prov})
        akw = dict(campaign="arditi_allpos", object_kind="vector_all", paper_label="arditi_allpos", object_role="baseline_published", fit_target_kind="contrastive_direction",
                   fit_objective="arditi_search", fit_model_id="qwen35_4b_it", status="approved", fit_contrast_sets_json={"unverified": "no sets"},
                   fit_layers_json=[12], fit_layer_index_convention="zero_based_decoder_block")
        with pytest.raises(ValueError, match="states base_scale=0.5 but fit_provenance.json:(arditi_allpos.)?c records 0.3"):
            _payload(a, base_scale=0.5, **akw)
        assert _payload(a, base_scale=0.3, **akw)["base_scale"] == 0.3


def test_emit_object_from_sidecar_forwards_both_statements_in_its_dry_run(tmp_path, monkeypatch):
    monkeypatch.setenv(hooks.emit.SPOOL_ENV, str(tmp_path / "spool"))
    d = _derived_dir(tmp_path)
    recs = hooks.emit_object_from_sidecar(str(d), script="tests", dry_run=True, registry_root=str(tmp_path / "noreg"), fit_layers_json=[0, 1],
                                          fit_layer_index_convention="zero_based_decoder_block", base_scale=0.3, **DERIVED_KW)
    obj = [p for t, p in recs if t == "object"][0]
    assert obj["fit_layer_index_convention"] == "zero_based_decoder_block" and obj["base_scale"] == 0.3 and obj["fit_layers_json"] == [0, 1]
    # the guarded hook turns the refusal into a None (one stderr notice), as every hook failure
    assert hooks.emit_object_from_sidecar(str(d), script="tests", dry_run=True, registry_root=str(tmp_path / "noreg"), fit_layers_json=[0, 1], **DERIVED_KW) is None


ARDITI_KW = dict(campaign="arditi_allpos", object_kind="vector_all", paper_label="arditi_allpos", object_role="baseline_published", fit_target_kind="contrastive_direction",
                 fit_objective="arditi_search", fit_model_id="qwen35_4b_it", status="approved", fit_contrast_sets_json={"unverified": "no sets"},
                 fit_layer_index_convention="zero_based_decoder_block")


def test_the_served_block_is_read_at_the_top_level_and_in_every_block_and_two_that_disagree_refuse_by_name(tmp_path):
    """The former `_provenance_served_layer` walked the blocks and returned the FIRST integer served_layer it met — a TOP-LEVEL
    served_layer was never read (a flat provenance compared nothing), and a second block's value was never seen (a list equal to the first
    block's passed whatever a later block recorded). Now every recorded value is read; one unambiguous value is the comparison, two that
    disagree refuse by name whatever the list says, and the source names where the agreeing value was first recorded."""
    # (a) FLAT: a top-level served_layer and no block — compared, named as the source, [13] refused against it
    flat = _dir(tmp_path, "flat", {"fit_provenance.json": {"served_layer": 12, "stored_layer": 12, "c": 0.3, "writer": "scripts/arditi/fit_arditi.py"}})
    p = _payload(flat, fit_layers_json=[12], **ARDITI_KW)
    assert p["fit_layers_json"] == [12] and p["fit_layer_index_convention"] == "zero_based_decoder_block"
    assert p["field_confidence_json"]["fit_layer_index_convention"] == {"confidence": "recorded", "source": "fit_provenance.json:served_layer"}
    assert p["field_confidence_json"]["fit_layers_json"] == {"confidence": "recorded", "source": "fit_provenance.json:served_layer"}
    with pytest.raises(ValueError, match=r"states fit_layers_json=\[13\] \(zero_based_decoder_block\) but fit_provenance.json:served_layer records the served block 12"):
        _payload(flat, fit_layers_json=[13], **ARDITI_KW)
    # (b) two blocks that DISAGREE: the former reader stopped at `first`, so [12] passed and [13] was refused against 12 — now every list refuses by name
    two = _dir(tmp_path, "two", {"fit_provenance.json": {"first": {"served_layer": 12}, "second": {"served_layer": 13}, "writer": "w"}})
    for layers in ([12], [13], [12, 13]):
        with pytest.raises(ValueError, match=r"two: fit_provenance.json records conflicting served layers — fit_provenance.json:first.served_layer = 12, "
                                             r"fit_provenance.json:second.served_layer = 13 — so it names no one served block to compare an explicit fit_layers_json against; not written"):
            _payload(two, fit_layers_json=layers, **ARDITI_KW)
    # (c) the top-level value against a block's: the same refusal, the top-level value named first
    mixed = _dir(tmp_path, "mixed", {"fit_provenance.json": {"served_layer": 12, "arditi_allpos": {"served_layer": 13, "c": 0.3}, "writer": "w"}})
    with pytest.raises(ValueError, match=r"conflicting served layers — fit_provenance.json:served_layer = 12, fit_provenance.json:arditi_allpos.served_layer = 13"):
        _payload(mixed, fit_layers_json=[12], **ARDITI_KW)
    with pytest.raises(ValueError, match="conflicting served layers"):
        _payload(mixed, fit_layers_json=[13], **ARDITI_KW)
    # (d) agreeing values are ONE served block: compared as one, the source the first place it is recorded (the top-level value, else the first block)
    agree = _dir(tmp_path, "agree", {"fit_provenance.json": {"served_layer": 12, "a": {"served_layer": 12}, "b": {"served_layer": 12}, "writer": "w"}})
    assert _payload(agree, fit_layers_json=[12], **ARDITI_KW)["field_confidence_json"]["fit_layers_json"]["source"] == "fit_provenance.json:served_layer"
    with pytest.raises(ValueError, match=r"states fit_layers_json=\[11\] \(zero_based_decoder_block\) but fit_provenance.json:served_layer records the served block 12"):
        _payload(agree, fit_layers_json=[11], **ARDITI_KW)
    blocks = _dir(tmp_path, "blocks", {"fit_provenance.json": {"meta": {"x": 1}, "a": {"served_layer": 12}, "b": {"served_layer": 12}, "writer": "w"}})
    assert _payload(blocks, fit_layers_json=[12], **ARDITI_KW)["field_confidence_json"]["fit_layers_json"]["source"] == "fit_provenance.json:a.served_layer"
    # (e) the readers themselves: file order with the top-level value first; a bool / string / float served_layer is no integer statement
    assert hooks._provenance_served_layers({"served_layer": 12, "a": {"served_layer": 13}, "b": {"served_layer": True}, "c": {"served_layer": "12"},
                                            "d": {"served_layer": 12.0}, "writer": "w"}) == [("fit_provenance.json:served_layer", 12), ("fit_provenance.json:a.served_layer", 13)]
    assert hooks._provenance_served_layers({"writer": "w"}) == [] and hooks._provenance_served_layer({"writer": "w"}) is None and hooks._provenance_served_layer({}) is None
    assert hooks._provenance_served_layer({"a": {"served_layer": 12}, "b": {"served_layer": 12}}) == ("fit_provenance.json:a.served_layer", 12)
    assert hooks._provenance_served_layer({"served_layer": 12}) == ("fit_provenance.json:served_layer", 12)
    with pytest.raises(ValueError, match="conflicting served layers"):
        hooks._provenance_served_layer({"served_layer": 12, "a": {"served_layer": 13}})
    # (f) another convention is not compared with the served block, agreeing or conflicting (a one-based 13 IS the zero-based 12): stated as given
    p = _payload(flat, fit_layers_json=[13], **dict(ARDITI_KW, fit_layer_index_convention="one_based"))
    assert p["fit_layer_index_convention"] == "one_based" and p["field_confidence_json"]["fit_layers_json"]["source"].startswith("stated by the writer")
    p = _payload(two, fit_layers_json=[14], **dict(ARDITI_KW, fit_layer_index_convention="one_based"))
    assert p["fit_layer_index_convention"] == "one_based" and p["field_confidence_json"]["fit_layers_json"]["source"].startswith("stated by the writer")
    # (g) the committed refusal direction's shape (one block, served == stored) reads exactly as before
    prov = {"arditi_allpos": {"c": 0.3, "served_layer": 12, "stored_layer": 12, "fit_layer": 13}, "writer": "scripts/arditi/fit_arditi.py"}
    p = _payload(_dir(tmp_path, "arditi", {"fit_provenance.json": prov}), fit_layers_json=[12], base_scale=0.3, **ARDITI_KW)
    assert p["field_confidence_json"]["fit_layer_index_convention"]["source"] == "fit_provenance.json:arditi_allpos.served_layer"


CONTEXT_REST = ("fit_job_id", "fit_node_hostname", "fit_substrate", "fit_producer_session")


def test_a_stated_unverified_fit_code_sha_carries_the_hooks_own_confidence_entry(tmp_path):
    """`fit_code_sha` 'unverified' (the schema's spelling for a fit whose code sha no record carries) needs a field_confidence_json entry
    (validate.py refuses the bare literal at harvest), and the WRITER that states the value owes it — so whenever the hook's payload states it,
    from a caller's fit_provenance or from a producer block whose checkout had no HEAD (emit.git_sha's fallback), the hook writes the entry
    itself: the fit_sidecars adapter's reason, the hook as the source. A real sha, or an unstated column, gets no entry."""
    d = _derived_dir(tmp_path)
    kw = dict(DERIVED_KW, fit_layers_json=[0, 1], fit_layer_index_convention="zero_based_decoder_block")
    entry = {"confidence": "unverified", "reason": "no sidecar records the fit code sha", "source": "sisteer.registry.hooks.object_from_sidecar"}
    assert dict(hooks.FIT_CODE_SHA_UNVERIFIED_ENTRY) == entry and hooks.OBJECT_HOOK_SOURCE == entry["source"]
    # (a) a caller's fit_provenance states 'unverified' (the committed-object importer's eight): the column, the entry, and the validator's rule met
    unstated_rest = {c: None for c in CONTEXT_REST}
    p = _payload(d, fit_provenance={"fit_code_sha": "unverified", **unstated_rest}, **kw)
    assert p["fit_code_sha"] == "unverified" and p["field_confidence_json"]["fit_code_sha"] == entry
    assert not any(c in p for c in CONTEXT_REST)
    assert VAL.check_column_value("object", "fit_code_sha", "unverified", p) is None
    without = {**p, "field_confidence_json": {k: v for k, v in p["field_confidence_json"].items() if k != "fit_code_sha"}}
    assert "requires field_confidence_json['fit_code_sha']" in VAL.check_column_value("object", "fit_code_sha", "unverified", without)
    # the entry sits beside the hook's other entries, none displaced
    assert p["field_confidence_json"]["fit_layer_index_convention"] == {"confidence": "recorded", "source": "derived_meta.json:layers"}
    # (b) the producer block's code_sha is emit.git_sha's fallback: the same entry, the same source
    p = hooks.object_from_sidecar(str(d), producer=dict(PROD, code_sha="unverified"), **kw)[0]
    assert p["fit_code_sha"] == "unverified" and p["field_confidence_json"]["fit_code_sha"] == entry
    # (c) a real sha, or an unstated column: no entry
    p = hooks.object_from_sidecar(str(d), producer=dict(PROD, code_sha="a" * 40), **kw)[0]
    assert p["fit_code_sha"] == "a" * 40 and "fit_code_sha" not in p["field_confidence_json"]
    p = _payload(d, fit_provenance={"fit_code_sha": None, **unstated_rest}, **kw)
    assert "fit_code_sha" not in p and "fit_code_sha" not in p["field_confidence_json"]
    # (d) a payload with no other entry at all gets field_confidence_json created for this one (the train_budget_meta fixture states everything)
    tbm = {"cell_id": "ctl__psr", "si_id": "ctl", "model_id": "Qwen/Qwen3.5-4B-Base", "seed": 1, "target_layers": [0, 1, 2], "chat_template_bypassed": True,
           "resolved_batch_size": 2, "batch_size": 2, "sample_manifest_sha": "a" * 64, "positions_at_fit": "all"}
    fit = _dir(tmp_path, "fit", {"train_budget_meta.json": tbm})
    assert "field_confidence_json" not in hooks.object_from_sidecar(str(fit), producer=dict(PROD, code_sha="b" * 40), **FIT_KW)[0]
    p = hooks.object_from_sidecar(str(fit), producer=dict(PROD, code_sha="unverified"), **FIT_KW)[0]
    assert p["fit_code_sha"] == "unverified" and p["field_confidence_json"] == {"fit_code_sha": entry}
    assert VAL.check_column_value("object", "fit_code_sha", "unverified", p) is None
    # (e) the fit writers' hook's dry run previews the same payload
    recs = hooks.emit_object_from_sidecar(str(d), script="tests", dry_run=True, registry_root=str(tmp_path / "noreg"),
                                          fit_provenance={"fit_code_sha": "unverified", **unstated_rest}, **kw)
    obj = [p for t, p in recs if t == "object"][0]
    assert obj["fit_code_sha"] == "unverified" and obj["field_confidence_json"]["fit_code_sha"] == entry and not any(c in obj for c in CONTEXT_REST)

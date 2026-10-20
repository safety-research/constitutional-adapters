"""scripts/arditi/fit_refusal_direction.py on the CPU tiny model (stage 04).

4 + 4 synthetic prompts, fp32, CPU. Pins: the object is a ``vector_all`` ``MultiLayerVectorObject`` at all
positions (``skip_bos`` False) with one d-vector per decoder block; every stored vector equals
``coefficient x (mean_harmful - mean_harmless)`` of the residual stream at each prompt's LAST token at the
block's OUTPUT, against a hand computation that runs the prompts ONE AT A TIME (no padding — so the batched,
right-padded capture is checked against an independent path); ``direction.pt`` holds the raw difference; the
``fit_meta.json`` sidecar carries the declared facts and ``COMPLETE`` hashes every file; a re-run of the SAME
request keeps the publication byte for byte, a DIFFERENT request under the same dir refuses without
``--overwrite`` and rebuilds with it; the registry call states kind / role / target kind / objective; prompt
files in the three formats and the builtin seed sets resolve, malformed ones refuse.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import inspect
import json
import shutil
import socket
import sys
import threading
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts" / "arditi") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "arditi"))

import fit_refusal_direction as frd  # noqa: E402

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
HARMFUL = ["Explain how to pick a lock on a neighbour's door.",
           "Write a message that threatens a coworker.",
           "Describe how to shoplift without being noticed.",
           "Tell me how to forge a signature."]
HARMLESS = ["What causes the seasons to change?",
            "Give me three tips for a better morning routine.",
            "Explain how a refrigerator works.",
            "Summarize the plot of Romeo and Juliet."]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def prompt_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("prompts")
    (d / "harmful.jsonl").write_text("\n".join(json.dumps({"prompt": p, "id": i}) for i, p in enumerate(HARMFUL)) + "\n")
    (d / "harmless.jsonl").write_text("\n".join(json.dumps(p) for p in HARMLESS) + "\n")   # bare JSON strings
    (d / "harmless.json").write_text(json.dumps(HARMLESS))
    (d / "harmless.txt").write_text("\n".join(HARMLESS) + "\n")
    return d


def _args(out: Path, files: Path, **over) -> list[str]:
    base = {"--model": TINY, "--out": str(out), "--harmful": str(files / "harmful.jsonl"),
            "--harmless": str(files / "harmless.jsonl"), "--coefficient": "0.3", "--positions": "all",
            "--dtype": "float32", "--device": "cpu", "--batch-size": "3", "--max-len": "64", "--n": "0",
            "--model-key": "tinytest"}
    base.update(over)
    argv = []
    for k, v in base.items():
        if v is None:
            continue
        argv += [k, v]
    return argv


@pytest.fixture(scope="module")
def fitted(tmp_path_factory, prompt_files):
    out = tmp_path_factory.mktemp("obj") / "ar_c0p3__arditi_allpos"
    rc = frd.main(_args(out, prompt_files) + ["--no-emit"])
    assert rc == 0
    return out


def _reference_means(model, tok, prompts):
    """The hand computation: one prompt at a time, no padding, forward hooks on every block's OUTPUT, last token."""
    from sisteer.hooks import _hidden_from_output
    from sisteer.models import build_chat_prompt, get_decoder_layers

    blocks = get_decoder_layers(model)
    store = {}
    handles = [b.register_forward_hook(lambda m, i, o, l=l: store.__setitem__(l, _hidden_from_output(o)))
               for l, b in enumerate(blocks)]
    sums = {l: torch.zeros(model.config.hidden_size) for l in range(len(blocks))}
    try:
        for q in prompts:
            ids = torch.tensor([frd.encode(tok, build_chat_prompt(tok, q))])
            with torch.no_grad():
                model(input_ids=ids)
            for l in sums:
                sums[l] += store[l][0, -1, :].float()
    finally:
        for h in handles:
            h.remove()
    return {l: v / len(prompts) for l, v in sums.items()}


def test_object_is_vector_all_at_all_positions_with_the_diff_in_means(fitted, tiny_llama):
    from sisteer.extraction.multilayer import MultiLayerVectorObject
    from sisteer.methods import load_object
    from sisteer.models import num_layers

    model, tok = tiny_llama
    obj = load_object(fitted / "object.pt", model=model, tok=tok)
    assert isinstance(obj, MultiLayerVectorObject) and obj.kind == "vector_all"
    assert obj.layers == list(range(num_layers(model)))
    assert obj.positions == "all" and obj.skip_bos is False and obj.fit_positions == "all"
    blob = torch.load(fitted / "object.pt", map_location="cpu", weights_only=True)
    assert blob["kind"] == "vector_all" and blob["skip_bos"] is False and blob["positions"] == "all"
    d = model.config.hidden_size
    for l in obj.layers:
        assert tuple(blob["vectors"][str(l)].shape) == (d,)
    # the identity: c x (mean_harmful - mean_harmless) at the last token, against the unpadded reference
    mh, mb = _reference_means(model, tok, HARMFUL), _reference_means(model, tok, HARMLESS)
    raw = torch.load(fitted / "direction.pt", map_location="cpu", weights_only=True)
    assert raw["layers"] == obj.layers and raw["capture_site"] == "resid_post"
    for l in obj.layers:
        expect_raw = mh[l] - mb[l]
        assert torch.allclose(raw["vectors"][str(l)], expect_raw, atol=1e-5), l
        assert torch.allclose(obj.params[l].detach(), 0.3 * expect_raw, atol=1e-5), l
        assert obj.params[l].detach().norm() > 0


def test_sidecar_and_complete_manifest(fitted):
    meta = json.loads((fitted / "fit_meta.json").read_text())
    assert meta["kind"] == "vector_all" and meta["derivation"] == "none"
    assert meta["fit_target_kind"] == "contrastive_direction" and meta["object_role"] == "baseline_published"
    assert meta["fit_objective"] == "diffmean" and meta["dose_unit"] == "raw_coefficient"
    assert meta["coefficient"] == 0.3 and meta["positions"] == "all" and meta["skip_bos"] is False
    assert meta["positions_at_fit"] == "all"
    assert meta["n_harmful"] == 4 and meta["n_harmless"] == 4
    assert meta["n_dropped_too_long"] == {"harmful": 0, "harmless": 0}
    assert set(meta["prompt_set_ids"]) == {"harmful", "harmless"}
    assert meta["prompt_sets"]["harmful"]["kind"] == "file" and meta["prompt_sets"]["harmful"]["n_used"] == 4
    assert meta["model"] == TINY and meta["model_key"] == "tinytest"
    assert meta["revision"] and len(meta["revision"]) == 40 and meta["revision_source"] == "live_model"
    assert meta["staged_id"] == "ar_c0p3__arditi_allpos" == meta["cell_id"]
    assert meta["capture_site"] == "resid_post" and meta["capture_position"] == "last_instruction_token"
    assert set(meta["layer_norms"]) == set(meta["direction_norms"]) == {str(l) for l in meta["layers"]}
    for l in meta["layers"]:
        assert abs(meta["layer_norms"][str(l)] - 0.3 * meta["direction_norms"][str(l)]) < 1e-5
    assert meta["object_sha256"] == _sha(fitted / "object.pt")
    assert meta["direction_pt_sha256"] == _sha(fitted / "direction.pt")
    assert len(meta["direction_sha256"]) == 64
    marker = json.loads((fitted / "COMPLETE").read_text())
    assert marker["object_sha256"] == meta["object_sha256"] and marker["staged_id"] == meta["staged_id"]
    # gap-refusal-pin: the identity sidecar fit_provenance.json is published and hashed beside fit_meta.json (PUBLISHED_FILES, one spelling)
    assert set(marker["files"]) == {"object.pt", "direction.pt", "fit_meta.json", "fit_provenance.json"} == set(frd.PUBLISHED_FILES)
    for name, sha in marker["files"].items():
        assert _sha(fitted / name) == sha
    assert frd.manifest_problems(fitted) == []
    # the hook reads this sidecar: layers, seed, the sha check
    from sisteer.registry import hooks

    payload, art = hooks.object_from_sidecar(
        str(fitted), campaign="two_si", object_kind="vector_all", paper_label="arditi_allpos",
        object_role="baseline_published", fit_target_kind="contrastive_direction", fit_objective="diffmean")
    assert payload["object_id"] == meta["object_sha256"] and payload["fit_layers_json"] == meta["layers"]
    assert payload["fit_positions"] == "all" and payload["fit_seed"] == 0
    assert art["artifact_id"] == meta["object_sha256"]
    # the hook reads the identity sidecar too: the MEASURED template hash and the resolved revision ride the row's extra (unverified before)
    assert payload["extra"]["template_hash"] == meta["template_hash"] and payload["extra"]["fit_model_revision"] == meta["revision"]


def test_the_publication_carries_the_identity_sidecar_the_stager_and_the_consumers_gate_read(fitted, prompt_files, tmp_path, capsys):
    """The REAL publication's fit_provenance.json is the ONE identity
    sidecar of the three (fit_meta.json is not one), so stage_objects.readiness_problem admits the dir as the cell (kind derived, the table's
    kind for the arditi object; kind fit too, with its COMPLETE) and refuses another cell's name; the consumers' gate (train_budget.
    assert_object_identity -> fit_provenance_problems) validates it as the fit_provenance identity; sweep.read_stage3_fit_provenance resolves
    it verified, Class B (the writer is registered in CLASS_B_FIT_WRITERS), bound to the bytes by its OWN object_sha256, and
    _provenance_from_record admits it; the record is the fitter's projection of fit_meta.json (fit_provenance_record) with the MEASURED template
    hash and the live revision, and spells the rung as the committed record does (`c`, `dose_unit`: sisteer.objects' dose read). Before this the
    publication carried no identity sidecar: the stager read it `not_ready` (the pre-fix shape, held here). A KEPT publication from before the
    sidecar gets it written from its verified fit_meta.json — identical, no refit — and one that carries it is left untouched."""
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
    import stage_objects as SO
    from sisteer.eval import sweep
    from sisteer.eval import train_budget as TB

    sid = "ar_c0p3__arditi_allpos"
    meta = json.loads((fitted / "fit_meta.json").read_text())
    prov = json.loads((fitted / frd.PROVENANCE_NAME).read_text())
    assert frd.PROVENANCE_NAME == TB.FIT_PROVENANCE_NAME == sweep.STAGE3_FIT_PROVENANCE_FILE == "fit_provenance.json"
    assert frd.PROVENANCE_NAME in SO.IDENTITY_SIDECARS and frd.META_NAME not in SO.IDENTITY_SIDECARS
    assert prov == frd.fit_provenance_record(meta)
    assert all(prov[k] for k in sweep.STAGE3_FIT_REQUIRED_KEYS), prov
    assert prov["writer"] == frd.SCRIPT and sweep.writer_head(prov["writer"]) in sweep.CLASS_B_FIT_WRITERS
    assert prov["resolved_model_id"] == TINY and prov["resolved_revision"] == meta["revision"] and len(prov["resolved_revision"]) == 40
    assert prov["chat_template_hash"] == meta["template_hash"] and len(prov["chat_template_hash"]) == 16          # the measured 16-hex hash
    assert prov["object_sha256"] == _sha(fitted / "object.pt") and prov["staged_id"] == prov["cell_id"] == sid
    assert prov["positions_at_fit"] == "all" and prov["chat_template_bypassed"] is False
    assert prov["c"] == 0.3 and prov["dose_unit"] == "raw_coefficient" and prov["paper_arm"] == "arditi_allpos" and prov["model_key"] == "tinytest"
    # the stager: ready as THIS cell, under either kind; another cell's name refuses; the committed marker stays coherent
    assert SO.readiness_problem(fitted, sid, "derived") is None and SO.readiness_problem(fitted, sid, "fit") is None
    assert "does not name the cell" in SO.readiness_problem(fitted, "ar_c0p5__arditi_allpos", "derived")
    assert SO.complete_marker_problem(fitted) is None
    # the consumers' gate and the one reader
    assert TB.object_identity_kind(fitted / "object.pt") == "fit_provenance"
    assert TB.fit_provenance_problems(fitted / "object.pt", sid) == []
    assert TB.assert_object_identity(fitted / "object.pt", sid, consumer="test") == "fit_provenance"
    rec = sweep.read_stage3_fit_provenance(fitted, fitted / "object.pt")
    assert rec["status"] == "verified" and rec["source"] == f"fit_provenance:{frd.SCRIPT}" and rec["fit_kind"] == sweep.FIT_KIND_CLASS_B
    assert rec["object_sha_bound"] is True and rec["object_sha_bound_by"] == "record" and rec["chat_template_hash"] == meta["template_hash"]
    mp = sweep._provenance_from_record(object_sha=prov["object_sha256"], rec=rec)
    assert mp.status == "verified" and mp.fit_kind == sweep.FIT_KIND_CLASS_B
    # the pre-fix shape (no identity sidecar; COMPLETE naming object.pt, direction.pt and fit_meta.json only, as the fitter wrote it before) is
    # exactly what the stager refused — and it is still a COMPLETE publication of the same request, not a torn one
    import shutil
    pre = tmp_path / "pre" / sid
    shutil.copytree(fitted, pre)
    (pre / frd.PROVENANCE_NAME).unlink()
    marker = json.loads((pre / "COMPLETE").read_text())
    marker["files"] = {n: s for n, s in marker["files"].items() if n != frd.PROVENANCE_NAME}
    (pre / "COMPLETE").write_text(json.dumps(marker, indent=1, sort_keys=True) + "\n")
    assert frd.manifest_problems(pre) == [] and set(marker["files"]) == {"object.pt", "direction.pt", "fit_meta.json"}
    assert "exactly one identity sidecar" in SO.readiness_problem(pre, sid, "derived") and "found none" in SO.readiness_problem(pre, sid, "derived")
    assert TB.object_identity_kind(pre / "object.pt") is None
    # a kept publication from before the sidecar: the same request re-run writes it from the verified meta, identical, and LISTS it in COMPLETE
    # with its sha256 (a backfill outside the manifest stood unverified) — the marker's other entries and its
    # publication time untouched; a second run leaves everything as it is
    capsys.readouterr()
    before = json.loads((pre / "COMPLETE").read_text())
    assert frd.main(_args(pre, prompt_files) + ["--no-emit"]) == 0
    out = capsys.readouterr().out
    assert "already holds a COMPLETE publication of this exact request — kept" in out and f"wrote {frd.PROVENANCE_NAME}" in out and f"listed it in {frd.COMPLETE_NAME}" in out
    assert json.loads((pre / frd.PROVENANCE_NAME).read_text()) == prov and SO.readiness_problem(pre, sid, "derived") is None
    after = json.loads((pre / "COMPLETE").read_text())
    assert set(after["files"]) == set(frd.PUBLISHED_FILES) and after["files"][frd.PROVENANCE_NAME] == _sha(pre / frd.PROVENANCE_NAME)
    assert {k: v for k, v in after["files"].items() if k != frd.PROVENANCE_NAME} == before["files"] and after["published_at"] == before["published_at"]
    assert after["identity_sidecar_backfilled_at"] and {k: v for k, v in after.items() if k not in ("files", "identity_sidecar_backfilled_at")} == {k: v for k, v in before.items() if k != "files"}
    assert frd.manifest_problems(pre) == [] and SO.complete_marker_problem(pre) is None and TB.fit_provenance_problems(pre / "object.pt", sid) == []
    snapshot = {p.name: _sha(p) for p in pre.iterdir()}
    assert frd.main(_args(pre, prompt_files) + ["--no-emit"]) == 0 and f"wrote {frd.PROVENANCE_NAME}" not in capsys.readouterr().out
    assert {p.name: _sha(p) for p in pre.iterdir()} == snapshot
    assert frd.ensure_fit_provenance(fitted) is False                       # a publication that carries it listed is untouched
    # a backfill torn between its two writes (the sidecar on disk — here altered too — the marker not yet rewritten): the file is UNLISTED, so the
    # stager has no hash to hold it to; the retry writes the projection again and lists it, and the altered copy is gone
    torn = tmp_path / "torn" / sid
    shutil.copytree(fitted, torn)
    (torn / "COMPLETE").write_text(json.dumps(dict(before, files={n: s for n, s in after["files"].items() if n != frd.PROVENANCE_NAME}), indent=1, sort_keys=True) + "\n")
    altered = dict(prov, chat_template_hash="0" * 16)
    (torn / frd.PROVENANCE_NAME).write_text(json.dumps(altered, indent=1, sort_keys=True) + "\n")
    assert SO.readiness_problem(torn, sid, "derived") is None               # the stager cannot tell: the sidecar is present and unlisted
    capsys.readouterr()
    assert frd.main(_args(torn, prompt_files) + ["--no-emit"]) == 0 and f"listed it in {frd.COMPLETE_NAME}" in capsys.readouterr().out
    assert json.loads((torn / frd.PROVENANCE_NAME).read_text()) == prov and json.loads((torn / "COMPLETE").read_text())["files"][frd.PROVENANCE_NAME] == _sha(torn / frd.PROVENANCE_NAME)
    assert _sha(torn / "object.pt") == _sha(fitted / "object.pt")           # no refit
    # a LISTED sidecar altered, truncated or deleted afterwards: the stager and the consumers' gate refuse it as a torn publication (never
    # provenance), and the fitter's kept path writes the bytes the marker promised again from the verified meta — no refit, the object untouched
    for mutate in (lambda p: p.write_text(json.dumps(altered, indent=1, sort_keys=True) + "\n"), lambda p: p.write_text(p.read_text()[:40]), lambda p: p.unlink()):
        mutate(torn / frd.PROVENANCE_NAME)
        if (torn / frd.PROVENANCE_NAME).exists():
            # the stager refuses either way: an altered copy by the manifest (`COMPLETE records …`), a truncated one as invalid JSON before it
            why = SO.readiness_problem(torn, sid, "derived")
            assert why and ("COMPLETE records" in why or "not valid JSON" in why), why
            assert TB.fit_provenance_problems(torn / "object.pt", sid) != []
        assert frd.manifest_problems(torn) == [f"{frd.PROVENANCE_NAME} missing or does not match its published sha256"]
        capsys.readouterr()
        assert frd.main(_args(torn, prompt_files) + ["--no-emit"]) == 0
        out = capsys.readouterr().out
        assert "written again from the verified" in out and "kept" in out and "rebuilding" not in out, out
        assert json.loads((torn / frd.PROVENANCE_NAME).read_text()) == prov and frd.manifest_problems(torn) == [] and _sha(torn / "object.pt") == _sha(fitted / "object.pt")
    # the one repair the fitter makes is the sidecar's: with the META altered the promised sidecar bytes are not a projection of what is on disk —
    # a torn publication, repaired by nothing and rebuilt as any torn publication is
    meta_doc = json.loads((torn / "fit_meta.json").read_text()); meta_doc["template_hash"] = "1" * 16
    (torn / "fit_meta.json").write_text(json.dumps(meta_doc, indent=1, sort_keys=True) + "\n")
    assert frd.repair_fit_provenance(torn) is False and frd.manifest_problems(torn) == ["fit_meta.json missing or does not match its published sha256"]


def test_rerun_is_idempotent_and_other_request_refuses_without_overwrite(fitted, prompt_files, capsys):
    before = {p.name: _sha(p) for p in fitted.iterdir()}
    assert frd.main(_args(fitted, prompt_files) + ["--no-emit"]) == 0
    assert "already holds a COMPLETE publication" in capsys.readouterr().out
    assert {p.name: _sha(p) for p in fitted.iterdir()} == before
    # a different coefficient under the same dir: refused, nothing touched
    assert frd.main(_args(fitted, prompt_files, **{"--coefficient": "0.5"}) + ["--no-emit"]) == 2
    assert "refusing to overwrite" in capsys.readouterr().err
    assert {p.name: _sha(p) for p in fitted.iterdir()} == before
    # --overwrite supersedes it: the stored vectors scale by 0.5 / 0.3 and the raw direction is unchanged
    raw_before = torch.load(fitted / "direction.pt", map_location="cpu", weights_only=True)["vectors"]
    obj_before = torch.load(fitted / "object.pt", map_location="cpu", weights_only=True)["vectors"]
    assert frd.main(_args(fitted, prompt_files, **{"--coefficient": "0.5"}) + ["--no-emit", "--overwrite"]) == 0
    raw_after = torch.load(fitted / "direction.pt", map_location="cpu", weights_only=True)["vectors"]
    obj_after = torch.load(fitted / "object.pt", map_location="cpu", weights_only=True)["vectors"]
    for l in raw_before:
        assert torch.allclose(raw_before[l], raw_after[l], atol=1e-6)
        assert torch.allclose(obj_after[l], obj_before[l] * (0.5 / 0.3), atol=1e-5)
    assert json.loads((fitted / "fit_meta.json").read_text())["coefficient"] == 0.5
    assert frd.manifest_problems(fitted) == []
    # restore the module fixture's state for any later test
    assert frd.main(_args(fitted, prompt_files) + ["--no-emit", "--overwrite"]) == 0


def test_torn_publication_is_rebuilt(tmp_path, prompt_files):
    out = tmp_path / "torn"
    out.mkdir()
    (out / "object.pt").write_bytes(b"stale")
    assert frd.manifest_problems(out) == [f"{out / 'COMPLETE'} missing"]
    assert frd.main(_args(out, prompt_files) + ["--no-emit"]) == 0
    assert frd.manifest_problems(out) == []
    assert _sha(out / "object.pt") != hashlib.sha256(b"stale").hexdigest()


def test_registry_call_states_the_object(tmp_path, prompt_files, monkeypatch):
    from sisteer.registry import hooks

    calls = []

    def fake(obj_dir, **kw):
        calls.append((obj_dir, kw))
        return ["synthetic"]

    monkeypatch.setattr(hooks, "emit_object_from_sidecar", fake)
    out = tmp_path / "emit" / "ar_c0p3__arditi_allpos"
    assert frd.main(_args(out, prompt_files)) == 0
    assert len(calls) == 1
    obj_dir, kw = calls[0]
    assert obj_dir == str(out)
    assert kw["object_kind"] == "vector_all" and kw["object_role"] == "baseline_published"
    assert kw["paper_label"] == hooks.paper_label_for("vector_all", "baseline_published") == "arditi_allpos"
    assert kw["fit_target_kind"] == "contrastive_direction" and kw["fit_objective"] == "diffmean"
    assert kw["derivation"] == "none" and kw["campaign"] == "two_si"
    assert kw["script"] == "scripts/arditi/fit_refusal_direction.py"
    assert kw["fit_model_id"] == "tinytest" and kw["fit_layers_json"] == [0, 1]
    extra = kw["extra"]
    assert extra["coefficient"] == 0.3 and extra["dose_unit"] == "raw_coefficient"
    assert extra["staged_id"] == "ar_c0p3__arditi_allpos" and extra["n_harmful"] == 4 and extra["n_harmless"] == 4
    assert extra["capture_site"] == "resid_post" and extra["skip_bos"] is False
    assert set(extra["prompt_set_ids"]) == {"harmful", "harmless"}
    assert len(extra["direction_sha256"]) == 64
    # vocabulary: every spelling the call states is admitted
    from sisteer.registry import vocab as V

    assert kw["object_kind"] in V.OBJECT_KINDS and kw["object_role"] in V.OBJECT_ROLES
    assert kw["paper_label"] in V.PAPER_LABELS and kw["fit_target_kind"] in V.FIT_TARGET_KINDS
    assert kw["fit_objective"] in V.FIT_OBJECTIVES and kw["derivation"] in V.DERIVATIONS
    # the emission leaves its evidence beside COMPLETE, keyed by the object id
    ev = frd.emission_evidence(out)
    marker = json.loads((out / "COMPLETE").read_text())
    assert ev["status"] == "emitted" and ev["object_id"] == marker["object_sha256"] and ev["records"] == ["synthetic"]
    assert ev["writer"] == frd.SCRIPT and not (out / frd.EMIT_FAILED_NAME).exists()
    assert frd.manifest_problems(out) == []                     # the evidence is not a manifested file


def test_kept_publication_replays_a_missing_record_and_a_failed_emission_is_visible(tmp_path, prompt_files, monkeypatch, capsys):
    """A re-run over a kept COMPLETE publication is not "already published"
    — it verifies the object row's emission and replays it when missing (once; no refit); a failed emission
    exits EMIT_FAILED_RC with the REGISTRY_EMIT_FAILED marker beside an intact artefact, and the next run with
    the hook repaired replays it and clears the marker."""
    from sisteer.registry import hooks

    calls = []

    def fake(obj_dir, **kw):
        calls.append(str(obj_dir))
        return ["synthetic"]

    def boom(obj_dir, **kw):
        raise RuntimeError("synthetic spool failure")

    monkeypatch.setattr(hooks, "emit_object_from_sidecar", fake)
    out = tmp_path / "replay" / "ar_c0p3__arditi_allpos"
    assert frd.main(_args(out, prompt_files)) == 0 and calls == [str(out)]
    before = {p.name: _sha(p) for p in out.iterdir() if p.name != frd.EMISSION_NAME}
    # death between COMPLETE and the record (or a pre-evidence publication): the kept path replays exactly once
    (out / frd.EMISSION_NAME).unlink()
    assert frd.main(_args(out, prompt_files)) == 0
    o = capsys.readouterr().out
    assert "already holds a COMPLETE publication" in o and "re-emitting the object record" in o
    assert calls == [str(out)] * 2 and frd.emission_evidence(out)["status"] == "emitted"
    assert {p.name: _sha(p) for p in out.iterdir() if p.name != frd.EMISSION_NAME} == before     # nothing refit
    # verified evidence: the third run emits nothing
    assert frd.main(_args(out, prompt_files)) == 0 and calls == [str(out)] * 2
    # a FAILED emission on the kept path: non-zero, the marker, the artefact intact
    monkeypatch.setattr(hooks, "emit_object_from_sidecar", boom)
    (out / frd.EMISSION_NAME).unlink()
    assert frd.main(_args(out, prompt_files)) == frd.EMIT_FAILED_RC == 3
    err = capsys.readouterr().err
    assert "direction not recorded" in err and "REGISTRY_EMIT_FAILED written" in err
    failed = json.loads((out / frd.EMIT_FAILED_NAME).read_text())
    assert failed["status"] == "failed" and "synthetic spool failure" in failed["error"]
    assert frd.emission_evidence(out) == failed and frd.manifest_problems(out) == []
    assert {p.name: _sha(p) for p in out.iterdir() if p.name not in (frd.EMISSION_NAME, frd.EMIT_FAILED_NAME)} == before
    # a failed evidence file is never terminal: the repaired hook replays and the marker goes
    monkeypatch.setattr(hooks, "emit_object_from_sidecar", fake)
    assert frd.main(_args(out, prompt_files)) == 0 and calls == [str(out)] * 3
    assert not (out / frd.EMIT_FAILED_NAME).exists() and frd.emission_evidence(out)["status"] == "emitted"
    # --no-emit never touches the evidence (the caller asked for no record)
    (out / frd.EMISSION_NAME).unlink()
    assert frd.main(_args(out, prompt_files) + ["--no-emit"]) == 0 and calls == [str(out)] * 3
    assert not (out / frd.EMISSION_NAME).exists()
    # a FRESH publication whose emission fails: the artefact is published and valid, the job fails visibly
    monkeypatch.setattr(hooks, "emit_object_from_sidecar", boom)
    fresh = tmp_path / "fresh" / "ar_c0p3__arditi_allpos"
    assert frd.main(_args(fresh, prompt_files)) == frd.EMIT_FAILED_RC
    assert frd.manifest_problems(fresh) == [] and (fresh / frd.EMIT_FAILED_NAME).is_file()


def test_kept_publication_replays_only_the_artifact_record_after_a_partial_failure(tmp_path, prompt_files, monkeypatch, capsys):
    """The REAL hook with the spool failing between its two writes — the fit is
    published, the object row is in the spool, the evidence is `failed` (rc 3); the next run over the kept publication
    replays ONLY the artifact record (one object row, one artifact row), never a duplicate."""
    from sisteer.registry import emit
    from sisteer.registry import records as R

    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    spool = tmp_path / "spool"
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool))
    types = lambda: [rec["record_type"] for f in R.iter_record_files(str(spool)) for _, rec, _ in R.read_record_file(f)]  # noqa: E731
    orig_emit_artifact = emit.RecordWriter.emit_artifact
    monkeypatch.setattr(emit.RecordWriter, "emit_artifact", lambda self, payload: (_ for _ in ()).throw(OSError(28, "No space left")))
    out = tmp_path / "partial" / "ar_c0p3__arditi_allpos"
    assert frd.main(_args(out, prompt_files)) == frd.EMIT_FAILED_RC and types() == ["object"]
    assert frd.manifest_problems(out) == [] and (out / frd.EMIT_FAILED_NAME).is_file()
    before = {p.name: _sha(p) for p in out.iterdir() if p.name not in (frd.EMISSION_NAME, frd.EMIT_FAILED_NAME)}
    monkeypatch.setattr(emit.RecordWriter, "emit_artifact", orig_emit_artifact)
    capsys.readouterr()
    assert frd.main(_args(out, prompt_files)) == 0
    o = capsys.readouterr().out
    assert "already holds a COMPLETE publication" in o and "replaying ONLY the artifact record" in o
    ev = frd.emission_evidence(out)
    assert ev["status"] == "emitted" and ev["replayed"] == ["artifact"] and types() == ["object", "artifact"]
    assert not (out / frd.EMIT_FAILED_NAME).exists()
    assert {p.name: _sha(p) for p in out.iterdir() if p.name != frd.EMISSION_NAME} == before          # nothing refit
    assert frd.main(_args(out, prompt_files)) == 0 and types() == ["object", "artifact"]           # verified: nothing more


def test_prompt_sets_resolve_in_every_format_and_refuse_malformed(prompt_files, tmp_path):
    for name in ("harmless.jsonl", "harmless.json", "harmless.txt"):
        prompts, info = frd.resolve_prompt_set(str(prompt_files / name), side="harmless", n=None, seed=0)
        assert prompts == HARMLESS and info["kind"] == "file" and info["n_used"] == 4
        assert info["file_sha256"] == _sha(prompt_files / name)
    prompts, info = frd.resolve_prompt_set(str(prompt_files / "harmful.jsonl"), side="harmful", n=2, seed=0)
    assert len(prompts) == 2 and info["n_available"] == 4 and info["n_used"] == 2
    assert prompts == [p for p in HARMFUL if p in prompts]                  # a stable subsequence
    seed, info = frd.resolve_prompt_set("seed:general", side="harmless", n=5, seed=1)
    assert len(seed) == 5 and info["kind"] == "builtin_seed"
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(frd.RefusalFitError, match="EMPTY"):
        frd.resolve_prompt_set(str(empty), side="harmful", n=None, seed=0)
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"nothing": 1}) + "\n")
    with pytest.raises(frd.RefusalFitError, match="cannot read a prompt string"):
        frd.resolve_prompt_set(str(bad), side="harmful", n=None, seed=0)
    with pytest.raises(frd.RefusalFitError, match="neither a prompt file nor a builtin"):
        frd.resolve_prompt_set("no_such_set", side="harmful", n=None, seed=0)
    assert frd.staged_id(0.3) == "ar_c0p3__arditi_allpos" and frd.staged_id(0.15) == "ar_c0p15__arditi_allpos"


@pytest.fixture
def paper_draw(tmp_path, monkeypatch):
    """The paper fit's sets as the fitter's builtin ``arditi:*`` ids would read them on a cluster, over the invented stand-in
    upstream (tests/arditi_fit_prompts_standin.py): its manifest as the module's, its document built into a third-party root,
    then fetching OFF and the network refused, as a compute node's job sees it."""
    import arditi_fit_prompts_standin as st
    from sisteer.data import arditi_fit_prompts as afp

    monkeypatch.setattr(afp, "MANIFEST_PATH", st.write_manifest(tmp_path / "manifest.json"))
    monkeypatch.setenv(afp.THIRD_PARTY_DIR_ENV, str(tmp_path / "third_party"))
    monkeypatch.setattr(afp, "_open_url", st.serve([]))
    afp.ensure_document(offline=False)
    monkeypatch.setenv(afp.OFFLINE_ENV, "1")
    monkeypatch.setattr(afp, "_open_url", st.refuse_network)
    return st, afp


def test_the_paper_fits_own_sets_resolve_from_the_verified_document(paper_draw, tmp_path, monkeypatch):
    st, afp = paper_draw
    m = st.manifest()
    for name in st.NAMES:
        prompts, info = frd.resolve_prompt_set(f"arditi:{name}", side="harmful", n=None, seed=0)
        assert prompts == st.expected_sets()[name]                                   # the drawn order, every instruction
        assert info["kind"] == "arditi_fit_prompts" and info["set"] == name and info["manifest"] == afp.MANIFEST_RELPATH
        assert info["document_sha256"] == m["document"]["sha256"] and info["set_sha256"] == m["sets"][name]["sha256"]
        assert info["upstream"] == st.REPO and info["upstream_commit"] == st.COMMIT
        assert info["source"] == str(tmp_path / "third_party" / f"refusal_direction/{st.COMMIT}/fit_prompts.json")
        assert info["n_available"] == info["n_used"] == st.n_of(name)
    # --n at or above the set's size takes the set as drawn (stage 04's --n 128 on the 128-instruction training sets)
    assert frd.resolve_prompt_set("arditi:harmful_train", side="harmful", n=st.N_TRAIN, seed=0)[0] == st.expected_sets()["harmful_train"]
    assert all(s in frd.BUILTIN_PROMPT_SETS for s in frd.ARDITI_SETS) and {s.split(":", 1)[1] for s in frd.ARDITI_SETS} == set(afp.SET_NAMES)
    with pytest.raises(frd.RefusalFitError, match="unknown builtin set 'arditi:harmful_test'"):
        frd.resolve_prompt_set("arditi:harmful_test", side="harmful", n=None, seed=0)
    # a node without the installed copy refuses by name, pointing at the setup step; nothing is fetched
    monkeypatch.setenv(afp.THIRD_PARTY_DIR_ENV, str(tmp_path / "empty"))
    with pytest.raises(frd.RefusalFitError, match="reproduce/00_setup.sh --what third_party"):
        frd.resolve_prompt_set("arditi:harmless_train", side="harmless", n=None, seed=0)


def test_a_fit_on_the_paper_fits_training_sets_records_their_identity(paper_draw, tmp_path, prompt_files):
    st, afp = paper_draw
    out = tmp_path / "obj" / "ar_c0p3__arditi_allpos"
    assert frd.main(_args(out, prompt_files, **{"--harmful": "arditi:harmful_train", "--harmless": "arditi:harmless_train"}) + ["--no-emit"]) == 0
    meta = json.loads((out / "fit_meta.json").read_text())
    assert meta["n_harmful"] == meta["n_harmless"] == st.N_TRAIN
    assert meta["prompt_set_ids"] == {"harmful": "arditi:harmful_train", "harmless": "arditi:harmless_train"}
    sets = meta["prompt_sets"]
    assert sets["harmful"]["kind"] == sets["harmless"]["kind"] == "arditi_fit_prompts"
    assert sets["harmful"]["document_sha256"] == sets["harmless"]["document_sha256"] == st.manifest()["document"]["sha256"]
    assert sets["harmless"]["set_sha256"] == st.manifest()["sets"]["harmless_train"]["sha256"]
    assert meta["fit_identity"]["harmful"]["id"] == "arditi:harmful_train"


def test_overlong_prompts_are_dropped_never_truncated(tmp_path, prompt_files, capsys):
    long_file = tmp_path / "long.jsonl"
    long_file.write_text("\n".join(json.dumps({"prompt": p}) for p in HARMFUL[:2] + ["word " * 400]) + "\n")
    out = tmp_path / "drop"
    assert frd.main(_args(out, prompt_files, **{"--harmful": str(long_file)}) + ["--no-emit"]) == 0
    meta = json.loads((out / "fit_meta.json").read_text())
    assert meta["n_harmful"] == 2 and meta["n_dropped_too_long"]["harmful"] == 1
    # every prompt over the cap: a loud refusal, not an empty mean
    only_long = tmp_path / "only_long.jsonl"
    only_long.write_text(json.dumps({"prompt": "word " * 400}) + "\n")
    assert frd.main(_args(tmp_path / "none", prompt_files, **{"--harmful": str(only_long)}) + ["--no-emit"]) == 2
    assert "no prompt survived" in capsys.readouterr().err


def test_a_publication_that_appears_while_the_fit_runs_is_kept_never_replaced_and_this_fits_bytes_are_discarded(fitted, prompt_files, tmp_path, monkeypatch, capsys):
    """Two jobs of ONE identity can meet — the front door's queue read keys on the identity's
    job name now, but two front doors can read the queue empty in the same instant, and a hand submission reads nothing — and used to race the
    publication: the second finisher removed the first's directory and renamed its own over it, then both emitted, two objects of one identity
    and rung, which the pins writer refuses. Now the publication event is ONE rename the kernel refuses while the directory is populated
    (_publish): a valid publication of the SAME request that appeared since the fit began is kept byte for byte, this fit's bytes discarded
    with its temporary, and the kept path verifies ITS emission (the winner's object id, replayed once — never a second object); a valid
    publication of a DIFFERENT request refuses at the write moment without --overwrite and is superseded with it; a torn one is cleared and
    the rename made again; the temporary directory names the host beside the pid, so two nodes' jobs never share one."""
    from sisteer.registry import hooks

    orig_capture, orig_publish = frd.capture_last_token_means, frd._publish
    appear, seen, calls = {}, {}, []

    def racing(model, tok, prompts, **kw):   # the other job publishes between this fit's start and its write
        if appear and not appear["dst"].exists():
            shutil.copytree(appear["src"], appear["dst"])
        return orig_capture(model, tok, prompts, **kw)

    def recording(tmp, out, identity, **kw):
        seen.update(tmp=Path(tmp), stood=out.exists())
        return orig_publish(tmp, out, identity, **kw)

    monkeypatch.setattr(frd, "capture_last_token_means", racing)
    monkeypatch.setattr(frd, "_publish", recording)
    monkeypatch.setattr(hooks, "emit_object_from_sidecar", lambda obj_dir, **kw: calls.append(str(obj_dir)) or ["synthetic"])
    # 1. the SAME request's publication appears (the module fixture's bytes; fit --no-emit, so no evidence beside it): kept byte for byte, this
    #    fit's bytes discarded with its temporary, and the kept path replays the WINNER's record once
    out = tmp_path / "race" / "ar_c0p3__arditi_allpos"
    appear.update(src=fitted, dst=out)
    winner = {p.name: _sha(p) for p in fitted.iterdir()}
    assert frd.main(_args(out, prompt_files)) == 0
    o = capsys.readouterr().out
    assert f"{out} was published by another job of this identity while this fit ran — kept, this fit's bytes discarded" in o and f"published {out}" not in o, o
    assert {p.name: _sha(p) for p in out.iterdir() if p.name != frd.EMISSION_NAME} == winner
    assert seen["stood"] and not seen["tmp"].exists() and not list(out.parent.glob("*.tmp-*")), seen
    assert seen["tmp"].parent == out.parent and seen["tmp"].name.startswith(f"{out.name}.tmp-{socket.gethostname()}-"), seen   # the host beside the pid
    marker = json.loads((out / frd.COMPLETE_NAME).read_text())
    assert marker["object_sha256"] == json.loads((fitted / frd.COMPLETE_NAME).read_text())["object_sha256"]
    assert calls == [str(out)] and frd.emission_evidence(out)["object_id"] == marker["object_sha256"] and "re-emitting the object record" in o, o
    # 2. a valid publication of a DIFFERENT request appears (c=0.5): refused at the write moment without --overwrite — its bytes untouched, this
    #    fit's temporary gone; with --overwrite it is superseded at the write moment by this fit's bytes
    monkeypatch.setattr(frd, "capture_last_token_means", orig_capture); appear.clear()
    other = tmp_path / "other" / "ar_c0p5__arditi_allpos"
    assert frd.main(_args(other, prompt_files, **{"--coefficient": "0.5"}) + ["--no-emit"]) == 0
    monkeypatch.setattr(frd, "capture_last_token_means", racing)
    out2 = tmp_path / "race2" / "ar_c0p3__arditi_allpos"
    appear.update(src=other, dst=out2)
    theirs = {p.name: _sha(p) for p in other.iterdir()}
    capsys.readouterr()
    assert frd.main(_args(out2, prompt_files) + ["--no-emit"]) == 2
    err = capsys.readouterr().err
    assert "published while this fit ran — refusing to overwrite it" in err and "a different fit request is published here" in err, err
    assert {p.name: _sha(p) for p in out2.iterdir()} == theirs and not list(out2.parent.glob("*.tmp-*"))
    shutil.rmtree(out2)
    assert frd.main(_args(out2, prompt_files) + ["--no-emit", "--overwrite"]) == 0
    o, err = capsys.readouterr()
    assert "superseding (--overwrite) at the write moment — replacing" in err and f"published {out2}" in o, o + err
    assert frd.manifest_problems(out2) == [] and json.loads((out2 / frd.META_NAME).read_text())["coefficient"] == 0.3 and not list(out2.parent.glob("*.tmp-*"))
    # 3. a torn directory appears (no valid COMPLETE): cleared at the write moment and this fit published
    torn_src = tmp_path / "torn_src"; torn_src.mkdir(); (torn_src / frd.OBJECT_NAME).write_bytes(b"stale")
    out3 = tmp_path / "race3" / "ar_c0p3__arditi_allpos"
    appear.update(src=torn_src, dst=out3)
    assert frd.main(_args(out3, prompt_files) + ["--no-emit"]) == 0
    o, err = capsys.readouterr()
    assert "torn publication at the write moment — replacing" in err and f"published {out3}" in o, o + err
    assert frd.manifest_problems(out3) == [] and _sha(out3 / frd.OBJECT_NAME) != hashlib.sha256(b"stale").hexdigest() and not list(out3.parent.glob("*.tmp-*"))
    # the publication event is the one rename, and nothing clears the destination before reading it
    fit_src, pub_src = inspect.getsource(frd.fit), inspect.getsource(orig_publish)   # the real _publish, not the recording wrapper patched over it
    assert "_publish(tmp, out, identity, overwrite=args.overwrite)" in fit_src and "os.replace(" not in fit_src and "shutil.rmtree(out)" not in fit_src
    assert "os.rename(tmp, out)" in pub_src and "errno.EEXIST" in pub_src and "errno.ENOTEMPTY" in pub_src


@pytest.mark.parametrize("pause_after", [1, 2], ids=["paused-between-its-two-reads", "paused-after-both-reads"])
def test_two_publishers_meeting_one_torn_directory_publish_one_and_never_remove_the_one_that_landed(fitted, tmp_path, monkeypatch, capsys, pause_after):
    """Two jobs of ONE identity finishing against the same TORN directory (an interrupted earlier
    publication) used to race _publish's torn replacement — each read the directory torn; A cleared it and published; B, acting on its
    stale verdict, removed A's publication and renamed its own in (two objects of one identity and rung, which the pins writer refuses, or
    A's emission reading a directory being removed); paused between its two reads instead, B refused A's identical publication as a
    DIFFERENT request. Now the rename, both reads and the clearing run under ONE advisory flock keyed by the publication path
    (_publication_lock): B, paused inside it, holds A at the lock — A cannot publish past it — and once B has published, A's read under the
    lock finds B's valid publication of the same request and keeps it, its own bytes discarded. Deterministic: the seams are the module's
    own reads and its lock, not timing; the one timed wait asserts that A did NOT publish while B held the lock."""
    identity = json.loads((fitted / frd.COMPLETE_NAME).read_text())["fit_identity"]
    out = tmp_path / "race" / "ar_c0p3__arditi_allpos"
    out.mkdir(parents=True); (out / frd.OBJECT_NAME).write_bytes(b"stale")                     # torn: no COMPLETE
    tmps = {}
    for who in ("A", "B"):
        tmps[who] = out.parent / f"{out.name}.tmp-TESTHOST-{who}"
        shutil.copytree(fitted, tmps[who]); (tmps[who] / "WHO").write_text(who)                 # unlisted, so the copy is still a valid publication — and says whose it is
    orig_read, orig_lock = frd.manifest_problems, frd._publication_lock
    b_paused, a_at_lock, a_done, reads, errors, results = threading.Event(), threading.Event(), threading.Event(), [], [], {}

    def reading(out_dir, identity=None):                                                       # the seam: B pauses after its pause_after-th read, before it acts on it
        r = orig_read(out_dir, identity)
        who = threading.current_thread().name
        if who in ("A", "B"):                                                                   # the publishers' reads only, not this test's own below
            reads.append(who)
        if who == "B" and reads.count("B") == pause_after:
            b_paused.set()
            assert a_at_lock.wait(10), "A never reached the lock"
            assert not a_done.wait(0.5), "A published while B held the lock"
        return r

    def locking(path):                                                                         # A announces itself at the lock's door
        if threading.current_thread().name == "A":
            a_at_lock.set()
        return orig_lock(path)

    def run(who):
        try:
            results[who] = frd._publish(tmps[who], out, identity, overwrite=False)
        except BaseException as e:  # noqa: BLE001 — surfaced on the main thread below
            errors.append((who, repr(e)))
        finally:
            if who == "A":
                a_done.set()

    monkeypatch.setattr(frd, "manifest_problems", reading)
    monkeypatch.setattr(frd, "_publication_lock", locking)
    b = threading.Thread(target=run, args=("B",), name="B"); b.start()
    assert b_paused.wait(10)                                                                    # B is inside the lock, past its torn verdict(s)
    a = threading.Thread(target=run, args=("A",), name="A"); a.start()
    a.join(30); b.join(30)
    assert not a.is_alive() and not b.is_alive() and errors == [], errors
    assert results == {"A": False, "B": True}, results                                         # B published; A found it and kept it
    assert (out / "WHO").read_text() == "B" and frd.manifest_problems(out, identity) == []     # the publication that landed stands, intact
    assert {p.name: _sha(p) for p in out.iterdir() if p.name != "WHO"} == {p.name: _sha(p) for p in fitted.iterdir()}
    assert reads == ["B", "B", "A"] and not list(out.parent.glob("*.tmp-*")), reads             # A read ONCE, under the lock, after B published; both temporaries gone
    err = capsys.readouterr().err
    assert err.count("torn publication at the write moment — replacing") == 1, err              # only B ever saw a torn directory
    lock = frd.publication_lock_path(out)
    assert lock.is_file() and lock.parent.resolve() == out.parent.resolve() and lock.name == f".{out.name}{frd.PUBLISH_LOCK_SUFFIX}"
    assert not lock.match("*.tmp-*") and lock.name not in {p.name for p in out.iterdir()}       # beside the publication, never inside it, never a temporary
    src = inspect.getsource(frd._publish)                                                       # the lock encloses the rename, the reads and the clearing
    assert src.index("with _publication_lock(out):") < src.index("os.rename(tmp, out)") < src.index("manifest_problems(out, identity)") < src.index("shutil.rmtree(out)")


def test_a_filesystem_that_cannot_lock_refuses_by_name_and_keeps_this_fits_bytes(fitted, tmp_path, monkeypatch):
    """The lock is what makes two publishers safe, so a filesystem that cannot take it refuses — naming the lock file and the errno, with
    the fit's bytes still at the temporary and nothing published or removed — rather than publishing unlocked (authoritative sources fail
    loudly, never degrade silently)."""
    identity = json.loads((fitted / frd.COMPLETE_NAME).read_text())["fit_identity"]
    out = tmp_path / "nolock" / "ar_c0p3__arditi_allpos"
    tmp = tmp_path / "nolock" / f"{out.name}.tmp-TESTHOST-1"
    shutil.copytree(fitted, tmp)
    real = fcntl.flock

    def refusing(fd, op):
        if op & fcntl.LOCK_EX:
            raise OSError(errno.ENOLCK, "No locks available")
        return real(fd, op)

    monkeypatch.setattr(frd.fcntl, "flock", refusing)
    with pytest.raises(frd.RefusalFitError) as ei:
        frd._publish(tmp, out, identity, overwrite=False)
    assert str(frd.publication_lock_path(out)) in str(ei.value) and "cannot take the publication lock (No locks available)" in str(ei.value)
    assert frd.manifest_problems(tmp, identity) == [] and not out.exists()                     # the bytes stand at the temporary; nothing published, nothing removed


def test_two_fits_of_one_identity_emit_one_object_statement_the_loser_reading_the_winners_emission_under_the_lock(prompt_files, tmp_path, monkeypatch, capsys):
    """The publication lock used to be released after _publish, and the job that LOST it entered _kept at once
    — read no evidence (the winner had not written it yet) and an empty spool, and passed the registry hook's read-then-write duplicate
    guard beside the winner's own emission: two native `object` statements of one id (or, the other way round, the winner's hook refusing
    its own publication's record, rc 3). Now fit() holds the SAME lock from the publish-or-keep decision through the registry emission and
    its evidence write — re-entered by _publish, which keeps taking it — so the loser, once it acquires the lock, reads the winner's
    emission (the evidence beside COMPLETE) and emits nothing: ONE object statement, ONE artifact, both jobs exit 0. The pre-fit kept path
    (a third job that finds the publication standing while the winner is still emitting) waits at the same lock and finds the same.
    Deterministic: the seams are the module's own capture, its lock and the hook, not timing. The REAL hook writes into a scratch spool."""
    from sisteer.registry import emit, hooks
    from sisteer.registry import records as R

    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.delenv(emit.PUSH_SCRIPT_ENV, raising=False)
    monkeypatch.delenv(hooks.EMIT_ENV, raising=False)
    registry = tmp_path / "registry"                                                             # the hook's idempotency check reads the registry under the spool's parent: a scratch one, no database
    spool = registry / "records"
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool))
    monkeypatch.setenv("SISTEER_REGISTRY_ROOT", str(registry))
    types = lambda: [rec["record_type"] for f in R.iter_record_files(str(spool)) for _, rec, _ in R.read_record_file(f)]  # noqa: E731
    out = tmp_path / "race" / "ar_c0p3__arditi_allpos"
    argv = _args(out, prompt_files) + ["--campaign", "two_si"]
    orig_capture, orig_lock, orig_hook, orig_tmp = frd.capture_last_token_means, frd._publication_lock, hooks.emit_object_from_sidecar, frd._temp_dir
    assert orig_tmp(out).parent == out.parent and orig_tmp(out).name == f"{out.name}.tmp-{socket.gethostname()}-{frd.os.getpid()}"   # the job's own temporary: host + pid
    b_fitting, a_published, b_at_lock = threading.Event(), threading.Event(), threading.Event()
    captures, lock_calls, hook_calls, results, errors = {}, {}, [], {}, []

    def who():
        return threading.current_thread().name

    def capturing(model, tok, prompts, **kw):                 # the seam before the publication: B commits to its own fit before A publishes
        w = who()
        if w in ("A", "B") and not captures.get(w):
            captures[w] = True
            if w == "B":
                b_fitting.set()                               # B is past its pre-fit read of an absent `out` — it WILL arrive at _publish with bytes of its own
            else:
                assert b_fitting.wait(120), "B never started fitting"
        return orig_capture(model, tok, prompts, **kw)

    def locking(path):                                        # the seam at the lock's door: B arrives only once A has published and is at its emission
        w = who()
        lock_calls[w] = lock_calls.get(w, 0) + 1
        if w == "B" and lock_calls[w] == 1:
            assert a_published.wait(120), "A never published"
            b_at_lock.set()
        return orig_lock(path)

    def emitting(obj_dir, **kw):                              # the seam at the emission: A, holding the lock, publishes the fact and waits for B at the door
        w = who()
        hook_calls.append((w, frd.publication_lock_held(Path(obj_dir))))
        if w == "A":
            a_published.set()
            assert b_at_lock.wait(120), "B never reached the lock"
        return orig_hook(obj_dir, **kw)

    def run(w):
        try:
            results[w] = frd.main(argv)
        except BaseException as e:  # noqa: BLE001 — surfaced on the main thread below
            errors.append((w, repr(e)))

    monkeypatch.setattr(frd, "capture_last_token_means", capturing)
    monkeypatch.setattr(frd, "_publication_lock", locking)
    monkeypatch.setattr(hooks, "emit_object_from_sidecar", emitting)
    # two jobs are two PROCESSES and each has its own temporary (host + pid); these two fits are two threads of one pid, so each gets its own here
    monkeypatch.setattr(frd, "_temp_dir", lambda out_: orig_tmp(out_).with_name(f"{orig_tmp(out_).name}-{who()}"))
    threads = {w: threading.Thread(target=run, args=(w,), name=w) for w in ("A", "B")}
    threads["B"].start(); threads["A"].start()
    assert a_published.wait(180), errors                                                         # A has published and holds the lock at its emission
    threads["C"] = threading.Thread(target=run, args=("C",), name="C"); threads["C"].start()     # C finds the publication standing: the pre-fit kept path
    for t in threads.values():
        t.join(300)
    assert not any(t.is_alive() for t in threads.values()) and errors == [], errors
    assert results == {"A": 0, "B": 0, "C": 0}, results                                          # the winner recorded; both losers exit cleanly
    # ONE object statement, ONE artifact — by the winner, under the lock; the losers emitted nothing (the hook was never called by them)
    assert hook_calls == [("A", True)], hook_calls
    assert types() == ["object", "artifact"], types()
    marker = json.loads((out / frd.COMPLETE_NAME).read_text())
    ev = frd.emission_evidence(out)
    assert ev["status"] == "emitted" and ev["object_id"] == marker["object_sha256"] and ev["campaign"] == "two_si" and not (out / frd.EMIT_FAILED_NAME).exists()
    assert frd.manifest_problems(out) == [] and not list(out.parent.glob("*.tmp-*"))               # the publication stands, both temporaries gone
    o = capsys.readouterr().out
    assert f"published {out}" in o and o.count(f"published {out}") == 1, o
    assert f"{out} was published by another job of this identity while this fit ran — kept, this fit's bytes discarded" in o, o
    assert "already holds a COMPLETE publication of this exact request — kept" in o, o
    assert "re-emitting" not in o and "replaying" not in o, o                                     # the losers found the winner's emission verified: nothing to replay
    # the lock was re-entered, not re-taken: A's and B's hold in fit() plus _publish's; C's one hold at the pre-fit read
    assert lock_calls == {"A": 2, "B": 2, "C": 1}, lock_calls
    # the mechanism, pinned in the source: the hold begins after COMPLETE is written and before _publish, and the emission runs inside it;
    # the pre-fit kept path runs inside its own hold
    src = inspect.getsource(frd.fit)
    pub = src.index("_publish(tmp, out, identity, overwrite=args.overwrite)")
    assert src.index("(tmp / COMPLETE_NAME).write_text") < src.rindex("with _publication_lock(out):", 0, pub) < pub < src.index("registry_emit(out, meta, campaign=args.campaign)")
    assert src.index("with _publication_lock(out):") < src.index('_kept(out, args, "already holds')
    assert "_require_publication_lock(out" in inspect.getsource(frd._kept)


def test_the_publication_lock_is_reentrant_for_its_holder_only_and_the_emission_paths_require_it(fitted, prompt_files, tmp_path):
    """The lock's contract behind the test above: a holder re-enters it (one descriptor, held until the OUTERMOST exit) while another
    thread cannot take it; and the kept path refuses by name when called without it — an unlocked emission is the silent race, so it is
    never let through to find out."""
    out = tmp_path / "lock" / "ar_c0p3__arditi_allpos"
    out.parent.mkdir(parents=True)
    assert not frd.publication_lock_held(out)
    with frd._publication_lock(out):
        assert frd.publication_lock_held(out)
        with frd._publication_lock(out):                                                        # re-entered: still one hold
            assert frd.publication_lock_held(out)
        assert frd.publication_lock_held(out)                                                   # the inner exit does not release the outer hold
        other = {}

        def contend():                                                                          # another thread: its own descriptor, the kernel's lock is taken
            fd = frd.os.open(frd.publication_lock_path(out), frd.os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                other["took"] = True
            except OSError as e:
                other["errno"] = e.errno
                other["held_here"] = frd.publication_lock_held(out)
            finally:
                frd.os.close(fd)

        t = threading.Thread(target=contend, name="other"); t.start(); t.join(10)
        assert other == {"errno": errno.EWOULDBLOCK, "held_here": False}, other                 # blocked; and the hold is per thread, not per process
    assert not frd.publication_lock_held(out)
    fd = frd.os.open(frd.publication_lock_path(out), frd.os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)                                            # the outermost exit released it
    finally:
        frd.os.close(fd)
    args = frd.build_parser().parse_args(_args(fitted, prompt_files) + ["--no-emit"])
    with pytest.raises(frd.RefusalFitError) as ei:
        frd._kept(fitted, args, "kept")
    assert "without the publication lock" in str(ei.value) and str(frd.publication_lock_path(fitted)) in str(ei.value)

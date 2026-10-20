"""``cpu_verify`` scorer identity (tinycap-verify-rate-limit). With the tree gate gone, a
later tree's verifier could SCORE a store under its own pinned sandbox image / verify timeout while the verified block's
``sandbox_facts`` (the generation plan's) claimed the old ones — scores attributed to the wrong experimental object (the
generator hashes image + timeout into the store identity). The verifier now compares what its RECONSTRUCTED coupled task
will execute (the compose image, the builder's verify timeout, the binding contract version) with what the generation was
hashed under (the sidecar plan's sandbox facts), byte-equal or refused BEFORE any sandbox, and records the executed
identity on the verified block. Pure: no inspect_ai, no modal."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import cpu_verify as CV

IMG_A = "ghcr.io/generality-labs/inspect-eval-bigcodebench:2026-07-28@sha256:" + "a" * 64
IMG_B = "ghcr.io/generality-labs/inspect-eval-bigcodebench:2026-10-01@sha256:" + "b" * 64


def _compose(tmp_path, image, *, name="compose.yaml"):
    p = tmp_path / name
    p.write_text(f"services:\n  default:\n    image: {image}\n    init: true\n    command: tail -f /dev/null\n")
    return p


def test_the_executed_identity_reads_the_tasks_compose_image_and_this_trees_timeout_and_binding(tmp_path, monkeypatch):
    task = SimpleNamespace(sandbox=SimpleNamespace(type=C.BCB_SANDBOX_PROVIDER, config=str(_compose(tmp_path, IMG_B))))
    ex = CV.executed_scorer_identity(task)
    assert ex == {"image": IMG_B, "verify_timeout_s": C.BCB_VERIFY_TIMEOUT_S, "binding_version": C.CPU_VERIFY_BINDING_VERSION,
                  "image_source": "compose"}
    # a task without a compose (a local sandbox, a bare spec, no sandbox): this tree's pinned image, and it says so
    for task in (SimpleNamespace(sandbox=None), SimpleNamespace(sandbox="local"), SimpleNamespace(sandbox=SimpleNamespace(config=None)),
                 SimpleNamespace(sandbox=SimpleNamespace(config=str(tmp_path / "absent.yaml")))):
        ex = CV.executed_scorer_identity(task)
        assert ex["image"] == C.BCB_SANDBOX_IMAGE and ex["image_source"] == "contract"
    monkeypatch.setattr(C, "BCB_SANDBOX_IMAGE", IMG_B)
    assert CV.executed_scorer_identity(SimpleNamespace(sandbox=None))["image"] == IMG_B   # read at call time, not import time
    # a compose naming two distinct images is not one scorer
    two = tmp_path / "two.yaml"
    two.write_text(f"services:\n  a:\n    image: {IMG_A}\n  b:\n    image: {IMG_B}\n")
    with pytest.raises(CV.CpuVerifyError, match="names 2 images"):
        CV.executed_scorer_identity(SimpleNamespace(sandbox=SimpleNamespace(config=str(two))))
    assert CV.SCORER_IDENTITY_FIELDS == ("image", "verify_timeout_s", "binding_version")


def test_the_generation_identity_comes_off_the_sidecar_plans_sandbox_facts_with_the_legacy_binding_default():
    facts = C.bcb_sandbox_facts()
    assert facts["binding_version"] == C.CPU_VERIFY_BINDING_VERSION == "1" == C.CPU_VERIFY_BINDING_VERSION_LEGACY
    gen = CV.generation_scorer_identity(facts)
    assert gen == {"image": C.BCB_SANDBOX_IMAGE, "verify_timeout_s": 30, "binding_version": "1"}
    legacy = {k: v for k, v in facts.items() if k != "binding_version"}              # a sidecar a tree before this change wrote
    assert CV.generation_scorer_identity(legacy)["binding_version"] == C.CPU_VERIFY_BINDING_VERSION_LEGACY
    with pytest.raises(CV.CpuVerifyError, match="carries no sandbox facts"):
        CV.generation_scorer_identity(None)
    with pytest.raises(CV.CpuVerifyError, match=r"lack \['image'\]"):
        CV.generation_scorer_identity({"verify_timeout_s": 30})
    with pytest.raises(CV.CpuVerifyError, match=r"lack \['verify_timeout_s'\]"):
        CV.generation_scorer_identity({"image": IMG_A, "verify_timeout_s": None})


def test_any_difference_in_image_timeout_or_binding_is_named_and_equality_admits_the_verifier():
    gen = {"image": IMG_A, "verify_timeout_s": 30, "binding_version": "1"}
    assert CV.scorer_identity_problems({**gen, "image_source": "compose"}, gen) == []          # extra keys on the executed side are not facts
    assert CV.scorer_identity_problems({**gen, "image": IMG_B}, gen) == [
        f"image: this verifier would execute {IMG_B!r}, the store was generated under {IMG_A!r}"]
    assert CV.scorer_identity_problems({**gen, "verify_timeout_s": 60}, gen) == [
        "verify_timeout_s: this verifier would execute 60, the store was generated under 30"]
    assert CV.scorer_identity_problems({**gen, "binding_version": "2"}, gen) == [
        "binding_version: this verifier would execute '2', the store was generated under '1'"]
    assert len(CV.scorer_identity_problems({"image": IMG_B, "verify_timeout_s": 60, "binding_version": "2"}, gen)) == 3
    # the block records the executed identity beside the generation's facts; the field set and its legacy acceptance
    block = CV.verified_block(mechanism="M2_replay_vehicle", function_call_id="f", started_at=1.0, finished_at=2.0, sandbox_facts=gen,
                              max_sandboxes=4, n_items=1, vehicle_log_sha256="0" * 64, verifier_tree_sha="c" * 40, generation_tree_sha="a" * 40,
                              executed_image=IMG_A, executed_verify_timeout_s=30, binding_version="1")
    assert tuple(block) == C.VERIFICATION_PASS_VERIFIED_FIELDS and len(block) == 14
    assert (block["executed_image"], block["executed_verify_timeout_s"], block["binding_version"]) == (IMG_A, 30, "1")
    assert C.VERIFICATION_PASS_VERIFIED_FIELDS[-5:] == ("verifier_tree_sha", "generation_tree_sha", "executed_image",
                                                         "executed_verify_timeout_s", "binding_version")
    assert C.verified_block_keys_problem(block) is None
    assert C.verified_block_keys_problem({k: v for k, v in block.items() if k in C.VERIFICATION_PASS_VERIFIED_FIELDS_V1}) is None
    assert "'binding_version'" not in C.verified_block_keys_problem({**block, "stray": 1}) or "stray" in C.verified_block_keys_problem({**block, "stray": 1})
    resumed = CV.verified_block(mechanism="M2_replay_vehicle", function_call_id="f", started_at=1.0, finished_at=2.0, sandbox_facts=gen,
                                max_sandboxes=4, n_items=1, vehicle_log_sha256="0" * 64)
    assert resumed["executed_image"] is None and resumed["executed_verify_timeout_s"] is None and resumed["binding_version"] is None
    assert C.REC_VERIFY_SCORER_IDENTITY_MISMATCH == "VERIFY_SCORER_IDENTITY_MISMATCH" not in C.REC_OK_STATUSES
    assert C.REC_VERIFY_CALL_UNAVAILABLE == "VERIFY_CALL_UNAVAILABLE" not in C.REC_OK_STATUSES
    assert C.SCORER_IDENTITY_MISMATCH_MARK == "SCORER_IDENTITY_MISMATCH"


def test_the_verifiers_result_restamp_carries_the_manifests_standup_before_verified_by():
    """The registry's landing-2 refusal (2026-09-30): the CPU verifier's `result=COMPLETE … verified_by=<fcid>` re-stamp
    carried NO `standup=`, while the row runner's earlier line and `manifest.row_engine.standup_id` did; the registry reads
    the LAST line's token and refused every landing-2 store. `result_line` spells the extras in order, so the tail's
    `{"standup": id, "verified_by": fcid}` reads `… standup=<id> verified_by=<fcid>` with `verified_by=` LAST (the
    idempotence check in `_finish_tail` reads `endswith(" verified_by=<fcid>")`)."""
    line = CV.result_line(name="n", status="OK", tree_sha="ab" * 20, extra={"standup": "11111111222222223333333344444444", "verified_by": "fc-v"})
    assert line.endswith(" standup=11111111222222223333333344444444 verified_by=fc-v") and f" result={C.RESULT_COMPLETE} status=OK " in line
    tokens = dict(tok.split("=", 1) for tok in line.split()[1:])
    assert tokens["standup"] == "11111111222222223333333344444444" and tokens["verified_by"] == "fc-v" and tokens["result"] == C.RESULT_COMPLETE
    # the tail's builder threads the manifest's id (source pins)
    src = (Path(__file__).resolve().parents[1] / "src/sisteer/capeval/tinycap/cpu_verify.py").read_text()
    assert 'extra["standup"] = str(standup_id)' in src and 'extra["verified_by"] = str(verified_by)' in src
    assert src.count('standup_id=(str(block.get("standup_id")) if block.get("standup_id") else None))') == 2   # both _finish_tail call sites

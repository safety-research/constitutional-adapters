"""fj29: the generation-only sidecar proof of a misalign attempt, derived from a pulled root (`scripts/sdfbase/seedrep/misalign_generation_proof.py`)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal", REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import misalign_generation_proof as mgp  # noqa: E402
import seedrep_carry_over as co  # noqa: E402
import seedrep_launch_plan as lp  # noqa: E402
import test_seedrep_fj2_carry_over as F  # noqa: E402  -- the synthetic plan / prior archive
import test_seedrep_fj30_generation_proof as G  # noqa: E402  -- the misalign row, the two manifests, the fake store, the real pull's paths

VOL = "/vol/seedrep/aaaaaaa/exp/seedrep_battery_modal_smoke/S5/misalign/s1/shard00/attempts/01TESTTASK-deadbeef0001"
REMOVED = ["openai/huihui-ai/Huihui-Qwen3.6-27B-abliterated", "qwen-abl"]
ADDED = ["anthropic/claude-opus-4-8", "grok-4.6", "openrouter/x-ai/grok-4.6", "opus-4-8"]
TARGET = ["petri__base", "petri__none", "petri__prompted__safety", "qwen35_4b_it"]
POLICY = {"removed": REMOVED, "added": ADDED, "target": TARGET, "manifests": ["m.yaml"], "prev_W": "a" * 40, "this_W": "b" * 40}


def _derive(root, **over):
    kw = {"volume_root": VOL, "removed": REMOVED, "added": ADDED, "target": TARGET, "policy": POLICY}
    kw.update(over)
    return mgp.derive(root, **kw)


def _root(tmp: Path, *, taint: str | None = None) -> Path:
    root = tmp / "root"
    for beh in ("agentic__s1__leaking", "agentic__s1__murder"):
        b = root / "exp" / "misalign" / beh
        b.mkdir(parents=True)
        (b / "response_evidence.jsonl").write_text("".join(json.dumps({"request_id": f"r{i}", "vmodel": "petri__base", "status": 200}) + "\n" for i in range(3)))
        for cell in ("none", "prompted__safety"):
            c = b / cell
            c.mkdir()
            (c / "meta.json").write_text(json.dumps({"cell_id": cell, "virtual_model": f"petri__{cell}", "served_models": [f"petri__{cell}"], "params": {"sampler": "vendor"}}))
            rows = [{"item_id": f"it{i}", "sample_idx": 0, "model_reported": "qwen35_4b_it", "virtual_model": f"petri__{cell}", "response": "ok"} for i in range(4)]
            if taint and cell == "none" and beh == "agentic__s1__murder":
                rows[1]["response"] = f"judged by {taint}"
            (c / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "modal_evidence.json").write_text(json.dumps({"attempt_id": "01TESTTASK-deadbeef0001", "petri": None, "children": []}))
    return root


def test_a_clean_generation_store_proves_zero_uses_and_binds_every_input_by_digest(tmp_path):
    root = _root(tmp_path)
    rec = _derive(root, chain_note="W22 -> launch: dbf326ead judge fallback chain")
    assert rec["ok"] is True and rec["cells"] == 4 and rec["n_rows"] == 16 and rec["n_response_evidence"] == 6
    assert rec["removed_leg"] == {"uses": 0, "legs": REMOVED, "per_leg": {leg: 0 for leg in REMOVED}}
    assert rec["added_legs"]["uses"] == 0 and rec["served_models"] == ["petri__base", "petri__none", "petri__prompted__safety"] and rec["models_reported"] == ["qwen35_4b_it"]
    assert rec["target_models"] == TARGET and rec["foreign_models"] == [] and rec["policy"] == POLICY and rec["bytes_read"] > 0
    assert rec["judge"]["in_attempt"] is False and rec["petri_completion_bundle"] is None and rec["why"] is None
    # every file read is an input, recorded under the VOLUME spelling with the sha256 of its bytes
    assert len(rec["inputs"]) == 2 + 4 * 2 + 1
    for path, digest in rec["inputs"].items():
        assert path.startswith(VOL + "/")
        assert hashlib.sha256((root / path[len(VOL) + 1:]).read_bytes()).hexdigest() == digest
    assert set(rec["per_cell"]) == {"agentic__s1__leaking/none", "agentic__s1__leaking/prompted__safety", "agentic__s1__murder/none", "agentic__s1__murder/prompted__safety"}


def test_a_single_use_of_the_removed_leg_in_any_row_fails_the_proof_and_names_the_cell(tmp_path):
    root = _root(tmp_path, taint="openai/huihui-ai/Huihui-Qwen3.6-27B-abliterated")
    rec = _derive(root)
    assert rec["ok"] is False and rec["removed_leg"]["uses"] == 1 and rec["removed_leg"]["per_leg"]["openai/huihui-ai/Huihui-Qwen3.6-27B-abliterated"] == 1
    assert rec["per_cell"]["agentic__s1__murder/none"]["removed_leg_uses"]["openai/huihui-ai/Huihui-Qwen3.6-27B-abliterated"] == 1
    assert rec["per_cell"]["agentic__s1__leaking/none"]["removed_leg_uses"]["openai/huihui-ai/Huihui-Qwen3.6-27B-abliterated"] == 0
    assert "reached 1 time(s)" in rec["why"]
    # the alias form counts as a use; an ADDED leg's use FAILS the proof too (fj31); a foreign served name fails it
    rec2 = _derive(_root(tmp_path / "b", taint="QWEN-ABL"))
    assert rec2["ok"] is False and rec2["removed_leg"]["per_leg"]["qwen-abl"] == 1
    rec3 = _derive(_root(tmp_path / "c", taint="anthropic/claude-opus-4-8"))
    assert rec3["ok"] is False and rec3["added_legs"]["uses"] == 2 and rec3["added_legs"]["per_leg"] == {"anthropic/claude-opus-4-8": 1, "grok-4.6": 0, "openrouter/x-ai/grok-4.6": 0, "opus-4-8": 1}
    assert "ADDED leg(s) were reached 2 time(s)" in rec3["why"] and "REMOVED" not in rec3["why"]
    rec4 = _derive(_root(tmp_path / "d"), target=["petri__base"])
    assert rec4["ok"] is False and rec4["foreign_models"] == ["petri__none", "petri__prompted__safety", "qwen35_4b_it"] and "not in the archived row's target set" in rec4["why"]


def test_an_incomplete_or_empty_store_proves_nothing(tmp_path):
    root = _root(tmp_path)
    (root / "exp" / "misalign" / "agentic__s1__murder" / "none" / "meta.json").unlink()
    try:
        _derive(root)
    except FileNotFoundError as e:
        assert "a cell without meta.json" in str(e)
    else:
        raise AssertionError("an incomplete cell must refuse")
    empty = tmp_path / "empty" / "root" / "exp" / "misalign"
    empty.mkdir(parents=True)
    rec = _derive(tmp_path / "empty" / "root")
    assert rec["ok"] is False and rec["cells"] == 0 and "nothing to prove" in rec["why"]


def _two_tree_repo(tmp_path: Path):
    """A real git repository with the archived row's manifest at two commits: the prior tree (V1: the huihui auditor leg) and the launch tree (V2:
    the leg removed, opus-4-8 added to the auditor chain, a judge chain added) -- the CLI reads both through `git show`."""
    import subprocess
    repo = tmp_path / "repo"
    (repo / Path(F.MANIFEST).parent).mkdir(parents=True)
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin"}
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    (repo / F.MANIFEST).write_text(G.MANIFEST_V1)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "prev"], check=True, env=env)
    prev = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (repo / F.MANIFEST).write_text(G.MANIFEST_V2)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-am", "this"], check=True, env=env)
    this = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return repo, prev, this


def _archived(tmp_path: Path, prev: str):
    """The prior tree's archived run state as `RS/prior/<W7>/` holds it: its plan with the misalign row (children --out / --only-cells, argv --manifest,
    cells_served_expected), written by the prior world's builders."""
    row = G._misalign_row(prev, prev[:7])
    plan = F._plan(prev, materialisation="draft", rows=[row])
    return F._prior_dir(tmp_path, plan, name="prior"), plan, row


def _pulled_store(tmp_path: Path):
    """The fj30 world's generation-only store (per behaviour a proxy response_evidence.jsonl, per cell meta.json + responses.jsonl, the attempt's
    modal_evidence.json) as a PULLED root on disk, and the same bytes as a reader keyed by volume path (the consumer's side)."""
    files: dict = {}
    G._store(files, VOL)
    files[f"{VOL}/modal_evidence.json"] = json.dumps({"attempt_id": "01TESTTASK-deadbeef0001", "petri": None, "children": []})
    root = tmp_path / "pulled"
    for vpath, text in files.items():
        q = root / vpath[len(VOL) + 1:]
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(text)
    return root, files


def _cli(tmp_path: Path):
    repo, prev, this = _two_tree_repo(tmp_path)
    prior_dir, plan, row = _archived(tmp_path, prev)
    root, files = _pulled_store(tmp_path)
    out = tmp_path / "proof.json"
    args = ["--root", str(root), "--volume-root", VOL, "--out", str(out), "--repo", str(repo), "--prior-dir", str(prior_dir), "--standup-id", G.SID,
            "--prev-w", prev, "--this-w", this, "--chain-note", "synthetic two-tree chain"]
    return dict(repo=repo, prev=prev, this=this, prior_dir=prior_dir, plan=plan, row=row, root=root, files=files, out=out, args=args, git=co.default_git(repo))


def test_the_cli_derives_one_policy_from_the_archived_row_writes_the_record_and_prints_its_digest(tmp_path, capsys):
    w = _cli(tmp_path)
    assert mgp.main(w["args"]) == 0
    rec = json.loads(w["out"].read_text())
    printed = capsys.readouterr().out
    assert rec["ok"] is True and f"sha256={hashlib.sha256(w['out'].read_bytes()).hexdigest()}" in printed and "ok=True cells=4" in printed
    # the policy IS the consumer's -- the same function over the same archived row (cells, target, removed / added, both trees' chains)
    policy = co.generation_proof_policy(w["row"], git=w["git"], prev_W=w["prev"], this_W=w["this"])
    assert rec["policy"] == policy and policy["cells"] == list(G.CELLS) and rec["target_models"] == policy["target"] == sorted(G._vm(c) for c in G.CELLS)
    assert rec["policy"]["removed"] == REMOVED and rec["policy"]["added"] == ADDED and rec["kind"] == co.GENERATION_PROOF_KIND and rec["ruling"] == co.GENERATION_PROOF_RULING
    assert rec["chain"] == "synthetic two-tree chain" and isinstance(rec["bytes_read"], int) and rec["foreign_models"] == []
    # refusals, by name, exit 2: an unreadable root; a --prev-w that is not the archive's W; a row the archive does not hold
    a = w["args"]
    assert mgp.main(a[:1] + [str(tmp_path / "nowhere")] + a[2:]) == 2
    assert mgp.main([x if x != w["prev"] else w["this"] for x in a]) == 2 and "not the archived plan's W" in capsys.readouterr().err
    assert mgp.main([x if x != G.SID else "misalign__s9__base__shard00" for x in a]) == 2 and "has no row" in capsys.readouterr().err


def test_sol_fj33_a_cli_document_is_admitted_by_the_consumer_and_the_pre_fj33_policy_shape_is_refused(tmp_path):
    """The integration end to end: the CLI's document -> `generation_proof_record` over a reader serving the same bytes -> ADMITTED (the policy
    block equal, the proof re-derived); the same document with the CLI's former policy block (chain_policy + target, no `cells`) -> refused by the
    consumer's unchanged equality check; the archived row and any smoke row reconstructed from it derive ONE policy."""
    w = _cli(tmp_path)
    assert mgp.main(w["args"]) == 0
    doc = json.loads(w["out"].read_text())
    kw = dict(source="s5_smoke", root=VOL, evidence={"petri": None}, read_text=w["files"].get, git=w["git"], prev_W=w["prev"], this_W=w["this"])
    record, probs = co.generation_proof_record(doc, prev_row=w["row"], **kw)
    assert probs == [] and record is not None and record["ok"] is True
    assert record["cells"] == 4 and record["removed_leg"]["uses"] == 0 and record["added_legs"]["uses"] == 0 and record["policy"] == doc["policy"]
    assert record["proof_sha256"] == hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert set(record["inputs"]) == set(doc["inputs"]) == set(mgp.expected_inputs(w["row"]["children"], VOL))
    # the smoke rows the evaluator reconstructs (S5 FULL shape; S3 with the item budget appended) derive the same policy and admit the same document
    sident = lp.smoke_identity(w["plan"], "S5", F.S5_WAVE)
    srow = lp.smoke_row(w["row"], sident, max_children=len(w["row"]["children"]), max_timeout_s=int(w["row"]["modal"]["timeout_s"]))
    assert co.generation_proof_policy(srow, git=w["git"], prev_W=w["prev"], this_W=w["this"]) == doc["policy"]
    record_s, probs_s = co.generation_proof_record(doc, prev_row=srow, **kw)
    assert probs_s == [] and record_s["ok"] is True
    # the pre-fj33 CLI's document: chain_policy(...) + target, no `cells` -> the consumer refuses it (its equality check is unchanged)
    old_policy = {k: v for k, v in doc["policy"].items() if k != "cells"}
    old_doc = {**doc, "policy": old_policy}
    record_o, probs_o = co.generation_proof_record(old_doc, prev_row=w["row"], **kw)
    assert record_o is None and any("policy block is not the one derived from the trees" in p_ for p_ in probs_o), probs_o
    # and a document without the labels the CLI now stamps is still admitted (the consumer defaults them), so the stamp is a completeness, not a gate
    record_n, probs_n = co.generation_proof_record({k: v for k, v in doc.items() if k not in ("kind", "ruling")}, prev_row=w["row"], **kw)
    assert probs_n == [] and record_n["ok"] is True


def test_the_real_w22_fixture_is_byte_for_byte_the_cli_over_the_pulled_bytes_and_the_archived_row():
    """Grading-host-local (skipped where the pull or the archive is absent): the checked-in fixture IS what the CLI writes over the
    pulled bytes with the archived row -- byte for byte -- and the archived row's policy equals its smoke row's."""
    import tempfile
    if not (G.PULLED / G.REAL_ROOT[5:] / "modal_evidence.json").is_file() or not (G.RS / "prior" / "56aff57" / "launch_plan_draft.json").is_file():
        pytest.skip("the fj29 pull / the W22 archive are not on this machine")
    prev_W, this_W = "56aff576e2975804602aec7d12ba1c98c805adfd", "0f00f7fae98e8c719617f6d01bbdc4eb2c763c88"
    if any(subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{w}^{{commit}}"], capture_output=True).returncode != 0 for w in (prev_W, this_W)):
        pytest.skip("the W22 / launch shas are not in this clone")
    fixture = REPO / "tests" / "fixtures" / "seedrep_battery" / "misalign_w22_generation_proof.json"
    with tempfile.TemporaryDirectory(prefix="fj33_regen_", dir="/tmp/launch_premint") as d:
        out = Path(d) / "regen.json"
        rc = mgp.main(["--root", str(G.PULLED / G.REAL_ROOT[5:]), "--volume-root", G.REAL_ROOT, "--out", str(out), "--repo", str(REPO),
                       "--prior-dir", str(G.RS / "prior" / "56aff57"), "--standup-id", G.SID, "--prev-w", prev_W, "--this-w", this_W,
                       "--chain-note", json.loads(fixture.read_text())["chain"]])
        assert rc == 0
        assert out.read_bytes() == fixture.read_bytes(), "the CLI's document over the pulled bytes is not the checked-in fixture"
    prior = co.load_prior_source(G.RS / "prior" / "56aff57")
    prow = co.archived_row(prior, G.SID)
    git = co.default_git(REPO)
    pol = co.generation_proof_policy(prow, git=git, prev_W=prev_W, this_W=this_W)
    srow = lp.smoke_row(prow, lp.smoke_identity(prior["plan"], "S5", "42c2793a-8160-462c-86e6-e39687cd6b13"), max_children=len(prow["children"]), max_timeout_s=int(prow["modal"]["timeout_s"]))
    assert co.generation_proof_policy(srow, git=git, prev_W=prev_W, this_W=this_W) == pol == json.loads(fixture.read_text())["policy"]

"""The failure this closes (2026-10-02, exit 75 with two COMPLETE stores): the narrowed registration writes the registered config to
$SHARED/registry/pending/<job>.config.yaml and jobplan stamped extra.config relative to `_checkout_root`'s grandparent fallback —
`pending/<job>.config.yaml` — which the closer resolved against the job's checkout: FileNotFoundError, "the registered config … could not be
read", 0 filed / 2 refused. The closer now resolves a relative stamp against the checkout AND beside the pending file (the writer's path),
naming every path tried when none exists; jobplan stamps an ABSOLUTE path for a config no checkout holds. Reuses the heavy multi-turn
fixtures of tests/test_registry_close_stores.py; the closer runs with a cwd that is neither the checkout nor the pending dir."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sisteer.registry import emit, emit_cli, hooks, synth  # noqa: E402
from test_registry_close_stores import CLOSER, OBJECTS, REPO, SEEDS, TAG, _multiturn_store, _records, _runs, _seeded_registry  # noqa: E402


def _world(tmp_path, monkeypatch):
    for k in (emit.DISABLED_ENV, emit.SPOOL_ENV, emit.PUSH_SCRIPT_ENV, hooks.EMIT_ENV, hooks.RUN_IDENTITY_ENV):
        monkeypatch.delenv(k, raising=False)
    root = tmp_path / "reg"; _seeded_registry(root)
    return root, root / "records"


def _register_narrowed(tmp_path, root, *, job: str, legacy_relative: bool) -> Path:
    """The cells-fix registration: the narrowed config lives in the PENDING dir (no checkout around it); --seeds/--pins explicit, as common.sh passes them."""
    pend = tmp_path / "shared" / "registry" / "pending"; pend.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load((REPO / "configs" / f"sdfbase_full22_{TAG}.yaml").read_text())
    base["explicit_cells"] = list(OBJECTS); base["registry_narrowed_from"] = f"configs/sdfbase_full22_{TAG}.yaml"; base["registry_narrowed_cells"] = list(OBJECTS)
    cfg = pend / f"{job}.config.yaml"; cfg.write_text(yaml.safe_dump(base, sort_keys=False))
    rc = emit_cli.main(["register", "--from-config", str(cfg), "--model-key", TAG, "--evals", "crescendo", "--serving-path", "lora_native", "--campaign", "sdfbase_full22",
                        "--job-id", job, "--job-name", f"f22-att-{TAG}-g5678-crescendo", "--pending-dir", str(pend), "--spool", str(root / "records"), "--code-sha", synth.code_sha("b"),
                        "--substrate", "cluster_a", "--no-push", "--seeds", str(SEEDS), "--pins", str(REPO / "configs" / "sdfbase_full22_pins.json")])
    assert rc == 0
    p = pend / f"{job}.json"; d = json.loads(p.read_text())
    stamped = {r["plan"]["extra"]["config"] for r in d["runs"]}
    assert stamped == {str(cfg)}, stamped                                                   # jobplan now stamps the ABSOLUTE path (no checkout holds it)
    if legacy_relative:                                                                      # the 837552 / 837557 / 837563 pending files: the OLD stamp
        for r in d["runs"]:
            r["plan"]["extra"]["config"] = f"pending/{job}.config.yaml"
        p.write_text(json.dumps(d))
    return p


def _close(pending: Path, spool: Path, wave: Path, cwd: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(CLOSER), "--pending", str(pending), "--wave-root", str(wave), "--tag", TAG, "--spool", str(spool), "--code-sha", synth.code_sha("b"),
                           "--job-name", f"f22-att-{TAG}-g5678-crescendo", *extra], cwd=str(cwd), capture_output=True, text=True)


def test_a_legacy_relative_stamp_is_resolved_beside_the_pending_file_and_the_complete_store_files(tmp_path, monkeypatch):
    root, spool = _world(tmp_path, monkeypatch)
    pending = _register_narrowed(tmp_path, root, job="837557", legacy_relative=True)
    runs = _runs(pending); prun = yaml.safe_load((REPO / "configs" / f"sdfbase_full22_{TAG}.yaml").read_text())["private_run"]
    wave = tmp_path / "exp" / "reproduce" / TAG / "waves" / "mt-u0"; wave.mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"; elsewhere.mkdir()
    # the old closer (any cwd): "could not be read" — the pending dir is where the config IS
    assert (tmp_path / "shared" / "registry" / "pending" / "837557.config.yaml").is_file()
    # expected ids from the registered config, as the closer recomputes them (the fixture needs the real ids)
    sys.path.insert(0, str(REPO / "scripts")); import jailbreak_run as jr
    ids = [str(g.goal_id) for g in jr._eval_goals(jr._load_cfg(str(tmp_path / "shared" / "registry" / "pending" / "837557.config.yaml")), confirmatory=True)]
    for obj in OBJECTS:
        _multiturn_store(wave, prun, "crescendo", obj, nshards=9, goals=len(ids), goal_ids=ids)
    # a stamp that exists NOWHERE refuses by name and names every path tried (the checkout and the pending dir) — before anything is filed
    d = json.loads(pending.read_text()); keep = json.dumps(d)
    for rr in d["runs"]:
        rr["plan"]["extra"]["config"] = "pending/999999.config.yaml"
    pending.write_text(json.dumps(d))
    r = _close(pending, spool, wave, elsewhere); out = r.stdout + r.stderr
    assert r.returncode == 2 and "is not on disk (tried" in out and "999999.config.yaml" in out and "the run stays running" in out and "filed" not in out.split("close:")[-1].split(",")[0].replace("0 filed", ""), out
    pending.write_text(keep)                                                                 # the legacy relative stamp again (837557's shape)
    r = _close(pending, spool, wave, elsewhere); out = r.stdout + r.stderr
    assert r.returncode == 0 and "2 filed" in out and "could not be read" not in out and "is not on disk" not in out, out
    for obj in OBJECTS:
        rid = runs[("crescendo", obj)]["run_id"]
        st = [p for t, p, *_ in _records(spool, rid) if t == "run_state" and p.get("state") == "generated"]
        assert len(st) == 1, (obj, [t for t, *_ in _records(spool, rid)])


def test_a_fresh_narrowed_registration_stamps_the_absolute_path_and_a_checkout_config_stays_relative(tmp_path, monkeypatch):
    root, spool = _world(tmp_path, monkeypatch)
    pending = _register_narrowed(tmp_path, root, job="837558", legacy_relative=False)              # the assertion inside: absolute stamp
    # a config INSIDE a checkout keeps the repo-relative spelling (the wave configs: unchanged behaviour)
    co = tmp_path / "checkout"; (co / "configs").mkdir(parents=True); os.symlink(SEEDS, co / "configs" / "registry")
    cfg = co / "configs" / f"sdfbase_full22_{TAG}.yaml"; cfg.write_text(yaml.safe_dump({"explicit_cells": list(OBJECTS)}))
    pend = tmp_path / "pending2"
    rc = emit_cli.main(["register", "--from-config", str(cfg), "--model-key", TAG, "--evals", "pasttense", "--serving-path", "lora_native", "--campaign", "sdfbase_full22",
                        "--job-id", "837559", "--job-name", f"f22s-{TAG}-static_base_t0", "--pending-dir", str(pend), "--spool", str(root / "records"), "--code-sha", synth.code_sha("b"),
                        "--substrate", "cluster_a", "--no-push"])
    assert rc == 0
    d = json.loads((pend / "837559.json").read_text())
    assert {r["plan"]["extra"]["config"] for r in d["runs"]} == {f"configs/sdfbase_full22_{TAG}.yaml"}

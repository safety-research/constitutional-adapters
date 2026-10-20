"""ONE RUN PER RUNG. The sampled lane's static phases register their runs
at the rung they DRAW — the producer key is the rung's private run name (`sdfbase_full22_<tag>_sampled_t<T>` / `…_cap8192` / `…_cap16384`),
token_cap_tokens the drawn cap (never the instrument's read-at 8192), token_cap_rung_index 0/1/2 and token_cap_ladder_json [4096, 8192, 16384];
jobplan refuses an inconsistent cap / rung / ladder by name (the cap is a RUN_IDENTITY field no patch can correct; the rung and the ladder are
not patchable at all); the closer locates a static store from the run's own key (a rung store is one shard). Before: every phase registered
<job_name>@<job_id> at the instrument's 8192 with an empty ladder — 112 rows drawn at 4,096 read as 8,192."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
from sisteer.registry import emit, emit_cli, hooks, identity, synth  # noqa: E402
from sisteer.registry import jobplan as JP  # noqa: E402
from test_registry_close_stores import OBJECTS, SEEDS, TAG, _close, _records, _runs, _seeded_registry, _static_store  # noqa: E402

LADDER = "4096,8192,16384"
KEY_T0 = f"sdfbase_full22_{TAG}_sampled_t0"


def _register(tmp_path: Path, root: Path, *, job: str, job_name: str, evals: str, flags=(), expect_rc: int = 0) -> Path:
    co = tmp_path / f"checkout_{job}"; (co / "configs").mkdir(parents=True)
    os.symlink(SEEDS, co / "configs" / "registry")
    cfg = co / "configs" / f"sdfbase_full22_{TAG}.yaml"; cfg.write_text(yaml.safe_dump({"explicit_cells": list(OBJECTS)}))
    pend = tmp_path / "pending"
    rc = emit_cli.main(["register", "--from-config", str(cfg), "--model-key", TAG, "--evals", evals, "--serving-path", "lora_native", "--campaign", "sdfbase_full22",
                        "--job-id", job, "--job-name", job_name, "--pending-dir", str(pend), "--spool", str(root / "records"), "--code-sha", synth.code_sha("a"),
                        "--substrate", "cluster_a", "--no-push", *flags])
    assert rc == expect_rc, (rc, expect_rc, flags)   # static-reg-fix-2: a refusal exits 2 — the shell stops the job on it
    return pend / f"{job}.json"


def _plans(pending: Path) -> dict:
    return {(r["route"]["eval_id"], r["route"]["object"]): r["plan"] for r in json.loads(pending.read_text())["runs"]}


def _world(tmp_path, monkeypatch):
    for k in (emit.DISABLED_ENV, emit.SPOOL_ENV, emit.PUSH_SCRIPT_ENV, hooks.EMIT_ENV, hooks.RUN_IDENTITY_ENV):
        monkeypatch.delenv(k, raising=False)
    root = tmp_path / "reg"; _seeded_registry(root)
    return root, root / "records"


def test_the_base_phase_registers_cap_4096_rung_0_rows_under_the_private_run_key_and_the_esc_phase_its_own_rung(tmp_path, monkeypatch):
    root, spool = _world(tmp_path, monkeypatch)
    base = _register(tmp_path, root, job="20001", job_name=f"f22s-{TAG}-{TAG}-static_base_t0", evals="pasttense,grok420",
                     flags=["--producer-key", KEY_T0, "--token-cap", "4096", "--rung-index", "0", "--ladder", LADDER, "--pass-label", "base_cap", "--trial-index", "0"])
    for (ev, obj), p in _plans(base).items():
        assert p["producer_run_key_legacy"] == KEY_T0 and p["token_cap_tokens"] == 4096 and p["token_cap_rung_index"] == 0
        assert p["token_cap_ladder_json"] == [4096, 8192, 16384] and p["pass_label"] == "base_cap" and p["trial_index"] == 0
        assert p["extra"]["rung_registration"] == {"token_cap_tokens": 4096, "token_cap_rung_index": 0, "token_cap_ladder_json": [4096, 8192, 16384], "pass_label": "base_cap", "trial_index": 0}
        assert p["extra"]["seed_instrument_id"] == "static_sampled_vendor_preset_4096_8192" and p["job_name"].endswith("static_base_t0")
    # the esc rung of the same trial: a DIFFERENT run per (eval, object) — its key names the rung, its cap is the drawn 8192, rung 1
    esc = _register(tmp_path, root, job="20002", job_name=f"f22s-{TAG}-se0", evals="pasttense,grok420",
                    flags=["--producer-key", f"{KEY_T0}_cap8192", "--token-cap", "8192", "--rung-index", "1", "--ladder", LADDER, "--pass-label", "ladder_rung", "--trial-index", "0"])
    bp, ep = _plans(base), _plans(esc)
    for k in bp:
        assert ep[k]["producer_run_key_legacy"] == f"{KEY_T0}_cap8192" and ep[k]["token_cap_tokens"] == 8192 and ep[k]["token_cap_rung_index"] == 1 and ep[k]["pass_label"] == "ladder_rung"
        assert ep[k]["instrument_id"] == bp[k]["instrument_id"]                                            # the SAME seeded instrument: rungs differ by key only
        assert ep[k]["n_items_planned"] == bp[k]["n_items_planned"]                                        # a rung store holds every row of the pair (re-drawn or copied forward)
    base_ids = {r["run_id"] for r in json.loads(base.read_text())["runs"]}; esc_ids = {r["run_id"] for r in json.loads(esc.read_text())["runs"]}
    assert base_ids.isdisjoint(esc_ids) and len(base_ids) == len(esc_ids) == 4
    rid = next(iter(base_ids)); recs = _records(spool, rid)
    assert [t for t, *_ in recs][:1] == ["run"] and recs[0][1]["token_cap_tokens"] == 4096 and recs[0][1]["token_cap_rung_index"] == 0      # the run record carries them
    # without the flags a job registers exactly as before: <job_name>@<job_id>, the instrument's cap, no rung, no ladder
    plain = _register(tmp_path, root, job="20003", job_name=f"f22s-{TAG}-{TAG}-static_base_t1", evals="pasttense")
    p = next(iter(_plans(plain).values()))
    assert p["producer_run_key_legacy"] == f"f22s-{TAG}-{TAG}-static_base_t1@20003" and p["token_cap_tokens"] == 8192
    assert "token_cap_rung_index" not in p and "token_cap_ladder_json" not in p and "rung_registration" not in p["extra"]


def test_an_inconsistent_rung_refuses_by_name_and_registers_nothing(tmp_path, monkeypatch, capsys):
    root, spool = _world(tmp_path, monkeypatch)
    cases = [
        (["--token-cap", "4096"], "--token-cap, --rung-index and --ladder go together"),
        (["--token-cap", "5000", "--rung-index", "0", "--ladder", LADDER], "--token-cap 5000 is not ladder[0] of [4096, 8192, 16384]"),
        (["--token-cap", "8192", "--rung-index", "0", "--ladder", LADDER], "--token-cap 8192 is not ladder[0]"),
        (["--token-cap", "4096", "--rung-index", "0", "--ladder", "8192,4096,16384"], "an ascending list"),
        (["--token-cap", "4096", "--rung-index", "0", "--ladder", LADDER, "--pass-label", "ladder_rung"], "does not fit rung index 0"),
        (["--token-cap", "8192", "--rung-index", "1", "--ladder", LADDER, "--pass-label", "basecap"], "--pass-label 'basecap': one of"),
        (["--producer-key", "sdfbase_full22 oops"], "one token of [A-Za-z0-9_.-]"),
    ]
    for i, (flags, frag) in enumerate(cases):
        pend = _register(tmp_path, root, job=f"3000{i}", job_name=f"f22s-{TAG}-x{i}", evals="pasttense", flags=flags, expect_rc=2)
        err = capsys.readouterr().err
        assert not pend.exists() and "self-registration refused" in err and frag in err, (flags, err)
    # the instrument's own ladder must hold the cap: a cap the seeded ladder never lists is refused per eval (the pair lands in `skipped`)
    pend = _register(tmp_path, root, job="30009", job_name=f"f22s-{TAG}-x9", evals="pasttense", flags=["--token-cap", "2048", "--rung-index", "0", "--ladder", "2048,8192"], expect_rc=2)
    err = capsys.readouterr().err
    assert (not pend.exists()) or not json.loads(pend.read_text())["runs"], err
    assert "does not hold the registered rung cap 2048" in err or "nothing registered" in err or "no run" in err, err


def test_the_phase_exports_its_rung_and_common_sh_forwards_it():
    sh = "scripts/sdfbase/full22/sampled/common_sampled.sh"
    def parse(phase, preset=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith("F22_REGISTRY_")}
        env.update(preset or {})
        r = subprocess.run(["bash", "-c", f'source {sh} && f22s_parse_phase "$1" && printf "%s|%s|%s|%s|%s|%s" "${{F22_REGISTRY_TOKEN_CAP-unset}}" "${{F22_REGISTRY_RUNG_INDEX-unset}}" "${{F22_REGISTRY_LADDER-unset}}" "${{F22_REGISTRY_PASS_LABEL-unset}}" "${{F22_REGISTRY_RUN_SUFFIX-unset}}" "${{F22_REGISTRY_TRIAL_INDEX-unset}}"', "_", phase],
                           cwd=REPO, env=env, capture_output=True, text=True)
        assert r.returncode == 0, (phase, r.stderr[-600:])
        return r.stdout.strip().split("|")
    assert parse("static_base_t0") == ["4096", "0", LADDER, "base_cap", "_sampled_t0", "0"]
    assert parse("static_base_t1:pasttense") == ["4096", "0", LADDER, "base_cap", "_sampled_t1", "1"]
    assert parse("static_esc_t1") == ["8192", "1", LADDER, "ladder_rung", "_sampled_t1_cap8192", "1"]
    assert parse("static_esc16_t0") == ["16384", "2", LADDER, "ladder_rung", "_sampled_t0_cap16384", "0"]
    assert parse("controls_t0") == ["unset"] * 6 and parse("capability_t1") == ["unset"] * 6             # other instruments keep their registration
    assert parse("static_esc_t0", {"F22_REGISTRY_TOKEN_CAP": "8192", "F22_REGISTRY_RUNG_INDEX": "1", "F22_REGISTRY_RUN_SUFFIX": "_sampled_t0_cap8192"})[:2] == ["8192", "1"]
    common = (REPO / "scripts/sdfbase/full22/common.sh").read_text()
    i_flags = common.index('rung_flags+=(--producer-key "sdfbase_full22_${key_tag}${F22_REGISTRY_RUN_SUFFIX}")')
    assert common.index('key_tag="$KEY$(') < i_flags < common.index('${rung_flags[@]+"${rung_flags[@]}"} >/dev/null')                      # built after key_tag, passed on the ONE register line
    assert 'rung_flags+=(--token-cap "$F22_REGISTRY_TOKEN_CAP" --rung-index "${F22_REGISTRY_RUNG_INDEX:?' in common and 'rung_flags+=(--pass-label "$F22_REGISTRY_PASS_LABEL")' in common
    sb = (REPO / "scripts/sdfbase/full22/sampled/target_sampled.sbatch").read_text()
    assert 'f22_init "$WAVE_SHA"; f22s_parse_phase "$PHASE"; f22s_resolve_config "$KEY" "$TRIAL" "$VIEW"' in sb                           # the phase is parsed BEFORE the registration fires


def test_the_closer_locates_a_static_store_from_the_runs_key_and_a_rung_store_is_one_shard(tmp_path, monkeypatch):
    root, spool = _world(tmp_path, monkeypatch)
    wave = tmp_path / "exp" / "reproduce" / TAG / "waves" / "static-u0"; wave.mkdir(parents=True)
    # the base run of trial 0, keyed by its private run; the JOB NAME carries no _t<N> (the hand-launched -sb0 spelling) — the key locates the store
    base = _register(tmp_path, root, job="40001", job_name=f"f22s-{TAG}-sb0", evals="pasttense",
                     flags=["--producer-key", KEY_T0, "--token-cap", "4096", "--rung-index", "0", "--ladder", LADDER, "--pass-label", "base_cap", "--trial-index", "0"])
    for obj in OBJECTS:
        _static_store(wave, 0, "pasttense", obj, nshards=9, goals=259, variants=2)
    r = _close(base, spool, wave, "--nshards", "9", job_name=f"f22s-{TAG}-sb0"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "2 filed" in out, out
    rid = _runs(base)[("pasttense", "none")]["run_id"]
    st = [p for t, p, *_ in _records(spool, rid) if t == "run_state" and p.get("state") == "generated"]
    assert len(st) == 1 and '"located_by": "producer_run_key"' in st[0]["note"] and f'"private_run": "{KEY_T0}"' in st[0]["note"] and '"rung_cap": null' in st[0]["note"], st[0]["note"]
    # the 8192 rung of the same trial: its own run, its store ONE shard under the `_cap8192` private run, every row of the pair present
    esc = _register(tmp_path, root, job="40002", job_name=f"f22s-{TAG}-se0", evals="pasttense",
                    flags=["--producer-key", f"{KEY_T0}_cap8192", "--token-cap", "8192", "--rung-index", "1", "--ladder", LADDER, "--pass-label", "ladder_rung", "--trial-index", "0"])
    r = _close(esc, spool, wave, "--nshards", "9", job_name=f"f22s-{TAG}-se0"); out = r.stdout + r.stderr
    assert r.returncode == 2 and "no store directory at" in out and f"{KEY_T0}_cap8192/pasttense" in out, out                              # the rung run never reads the base store
    for obj in OBJECTS:
        d = _static_store(wave, 0, "pasttense", obj, nshards=1, goals=259, variants=2)
    base_dir = wave / "transcripts_results" / KEY_T0; rung_dir = wave / "transcripts_results" / f"{KEY_T0}_cap8192"
    # _static_store wrote the one-shard store INTO the base spelling (shard00 overwritten): move those shard00 dirs into the rung's private run
    rung_dir.mkdir(parents=True)
    for obj in OBJECTS:
        (rung_dir / "pasttense").mkdir(parents=True, exist_ok=True)
        (base_dir / "pasttense" / obj).rename(rung_dir / "pasttense" / obj)
        _static_store(wave, 0, "pasttense", obj, nshards=9, goals=259, variants=2)                      # the base store back, nine shards
        for extra in (rung_dir / "pasttense" / obj).glob("shard0[1-8]"):
            __import__("shutil").rmtree(extra)                                                             # a rung store is shard00 ONLY
    r = _close(esc, spool, wave, "--nshards", "9", job_name=f"f22s-{TAG}-se0"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "2 filed" in out, out
    rid_e = _runs(esc)[("pasttense", "none")]["run_id"]
    st = [p for t, p, *_ in _records(spool, rid_e) if t == "run_state" and p.get("state") == "generated"]
    assert len(st) == 1 and '"rung_cap": 8192' in st[0]["note"] and '"nshards_expected": 1' in st[0]["note"] and f'"private_run": "{KEY_T0}_cap8192"' in st[0]["note"], st[0]["note"]
    pt = [p for t, p, *_ in _records(spool, rid_e) if t == "run_patch"]
    assert pt[-1]["fields"]["store_uri"].endswith(f"{KEY_T0}_cap8192/pasttense/none") and pt[-1]["fields"]["n_items_generated"] == 518

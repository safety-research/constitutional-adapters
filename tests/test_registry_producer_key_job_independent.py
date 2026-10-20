"""A JOB-INDEPENDENT producer key for the job kinds that registered `<job_name>@<job_id>`.

* `f22_registry_producer_key JOB_NAME KEY_TAG [SEGMENTS]` (common.sh) is a pure function of the measurement identity — the per-family prefix
  (the job kind), the (key, view) tag, the family's own segments — `-`-separated, no `@`, one token of [A-Za-z0-9_.-]; a malformed key is a
  refusal (rc 1), never a silent default; the gsm8k rows, the rung-keyed f22s-* jobs and every kind the table does not name
  keep jobplan's `<job_name>@<job_id>` (empty output).
* f22-bonclimb joined the table on 2026-10-11 (the companion of the closer's store512 / climb survivor
  rule): the climb array's four tasks register ONE run per cell (`sdfbase_full22-bonclimb-<tag>`), not one per task — and ONLY for NEW
  registrations: an existing registration's pending file is its identity, `f22_registry_register` keeps it, so every
  run registered under the job-keyed key before the landing keeps its run_id (frozen records).
* `f22_registry_register` puts that key on the planner's argv unless a rung suffix keys the run, and the five families export their identity
  segments BEFORE they register (petri: the pass; multi-turn: the shard group; misalign: mode / shard / baselines; padding: the level [+ shard range]).
* through the REAL registration (common.sh -> emit_cli register over the checkout's seeds): two Petri proxies with different job ids and the same
  measurement register the SAME run ids (the attempt tells attempts apart), while a kind the table does not name still mints job-keyed ids; the
  run-id derivation itself is unchanged (campaign:eval:instrument[:12]:sha16(producer key | object key)).

Synthetic job names and ids throughout; no cluster, no real spool."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, run_f22_registration, synthetic_checkout  # noqa: E402

COMMON = REPO / "scripts/sdfbase/full22/common.sh"
F22 = REPO / "scripts/sdfbase/full22"
TAG = "qwen35_35b_a3b_s1"
CODE_SHA = "9c4fd67577dfc6df5e75fc5dd63d2882bba59d7b"   # == synth.code_sha("a"): a placeholder (forty identical digits) run code_sha is refused


def _key(job: str, tag: str, seg: str = "") -> tuple[int, str, str]:
    r = subprocess.run(["bash", "-c", 'source "$1"; f22_registry_producer_key "$2" "$3" "$4"', "_", str(COMMON), job, tag, seg],
                       capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr


@pytest.mark.parametrize("job,tag,seg,want", [
    ("f22-petri-qwen35_35b_a3b_s1", TAG, "pass-first", f"sdfbase_full22-petri-{TAG}-pass-first"),
    ("f22-petri-qwen35_35b_a3b_s1", TAG, "pass-overflow-rerun", f"sdfbase_full22-petri-{TAG}-pass-overflow-rerun"),
    ("f22-stat-qwen35_35b_a3b_s2r-2", "qwen35_35b_a3b_s2r", "", "sdfbase_full22-stat-2-qwen35_35b_a3b_s2r"),
    ("f22-stat-qwen35_35b_a3b-all", "qwen35_35b_a3b", "", "sdfbase_full22-stat-all-qwen35_35b_a3b"),
    ("f22-att-qwen35_35b_a3b_s2r-crescendo", "qwen35_35b_a3b_s2r", "sh0_1_2_3_4of9", "sdfbase_full22-att-crescendo-qwen35_35b_a3b_s2r-sh0_1_2_3_4of9"),
    ("f22-att-qwen35_35b_a3b_s2r-crescendo", "qwen35_35b_a3b_s2r", "sh5_6_7_8of9", "sdfbase_full22-att-crescendo-qwen35_35b_a3b_s2r-sh5_6_7_8of9"),
    ("f22-att-qwen35_35b_a3b_s2r-crescendo", "qwen35_35b_a3b_s2r", "sh0of259", "sdfbase_full22-att-crescendo-qwen35_35b_a3b_s2r-sh0of259"),   # 282b: the one-goal smoke is its own run
    ("f22-att-goalset-qwen35_35b_a3b-pair", "qwen35_35b_a3b", "sh0_1_2_3_4", "sdfbase_full22-att-goalset-pair-qwen35_35b_a3b-sh0_1_2_3_4"),
    ("f22-att-qwen35_35b_a3b-actorattack", "qwen35_35b_a3b", "sh0_1_2_3_4_5_6_7_8", "sdfbase_full22-att-actorattack-qwen35_35b_a3b-sh0_1_2_3_4_5_6_7_8"),
    ("f22-misalign-em-llama8b", "llama8b", "target-llama8b_org-mode-organisms-sampler-vendor", "sdfbase_full22-misalign-organisms-llama8b-target-llama8b_org-mode-organisms-sampler-vendor"),
    ("f22-misalign-em-llama8b", "llama8b", "target-llama8b_org-mode-organisms-sampler-t1", "sdfbase_full22-misalign-organisms-llama8b-target-llama8b_org-mode-organisms-sampler-t1"),   # 282b: the record rung
    ("f22-misalign-qwen35_35b_a3b_s1", TAG, f"target-{TAG}-mode-full", f"sdfbase_full22-misalign-agentic-{TAG}-target-{TAG}-mode-full"),
    ("f22-misalign-qwen35_4b", "qwen35_4b", "target-qwen35_4b_psr-mode-full", "sdfbase_full22-misalign-agentic-qwen35_4b-target-qwen35_4b_psr-mode-full"),   # 282b: the config's own identity
    ("f22-misalign-qwen35_4b", "qwen35_4b", "target-qwen35_4b-mode-ladder16k-shard1of4", "sdfbase_full22-misalign-agentic-qwen35_4b-target-qwen35_4b-mode-ladder16k-shard1of4"),
    ("f22-misalign-qwen35_4b", "qwen35_4b", "target-qwen35_4b-mode-vendor-root-r2", "sdfbase_full22-misalign-agentic-qwen35_4b-target-qwen35_4b-mode-vendor-root-r2"),   # 282b: a re-sample root
    ("padstat-qwen35_35b_a3b", "qwen35_35b_a3b_u0", "p8192", "sdfbase_full22-padstat-qwen35_35b_a3b_u0-p8192"),
    ("padobj-qwen35_4b-p0-s0-3", "qwen35_4b_s1", "p0-sh0_3_16", "sdfbase_full22-padobj-qwen35_4b_s1-p0-sh0_3_16"),
    # the climb array keys by the kind and the tag — one run per cell across its tasks (job-keyed before)
    ("f22-bonclimb-qwen35_35b_a3b", "qwen35_35b_a3b", "", "sdfbase_full22-bonclimb-qwen35_35b_a3b"),
    ("f22-bonclimb-qwen35_35b_a3b_s1", TAG, "", f"sdfbase_full22-bonclimb-{TAG}"),
    ("f22-bonclimb-qwen35_4b_demo_battery", "qwen35_4b_demo_battery", "", "sdfbase_full22-bonclimb-qwen35_4b_demo_battery"),
    # kept on jobplan's default, by name
    ("g8k-qwen35_35b_a3b-s1-ctl", TAG, "", ""),
    ("f22s-qwen35_35b_a3b-qwen35_35b_a3b_s2r-static_base_t0", "qwen35_35b_a3b_s2r", "", ""),
    ("mmluF-qwen35_35b_a3b-u0", "qwen35_35b_a3b_u0", "", ""),
    ("", TAG, "", ""),
])
def test_the_key_is_a_pure_function_of_the_measurement_identity_and_names_no_job(job, tag, seg, want):
    rc, out, err = _key(job, tag, seg)
    assert rc == 0 and out == want, (job, tag, seg, out, err)
    if want:
        assert "@" not in want and re.fullmatch(r"[A-Za-z0-9_.-]+", want) and want.startswith("sdfbase_full22-")
    # the same identity under another job id is the same key — the job id is not an input
    rc2, out2, _ = _key(job, tag, seg)
    assert (rc2, out2) == (rc, out)


def test_a_malformed_key_is_a_refusal_not_a_silent_default():
    rc, out, err = _key("f22-petri-x", "x@1")
    assert rc == 1 and out == "" and "is not one token of [A-Za-z0-9_.-] without '@'" in err and "'sdfbase_full22-petri-x@1'" in err, err
    rc, out, err = _key("f22-petri-x", "x", "a b")
    assert rc == 1 and out == "" and "segments 'a b'" in err, err
    rc, out, err = _key("f22-petri-x", "x", "pass-first/second")
    assert rc == 1 and out == "", err


def test_the_register_block_puts_the_key_on_the_argv_unless_a_rung_suffix_keys_the_run():
    src = COMMON.read_text()
    body = src[src.index("f22_registry_register(){"):]
    body = body[:body.index("\n}\n")]
    assert 'if [ -n "${F22_REGISTRY_RUN_SUFFIX:-}" ]; then rung_flags+=(--producer-key "sdfbase_full22_${key_tag}${F22_REGISTRY_RUN_SUFFIX}")' in body
    assert 'pkey=$(f22_registry_producer_key "${SLURM_JOB_NAME:-}" "$key_tag" "${F22_REGISTRY_KEY_SEGMENTS:-}")' in body
    assert 'f22_registry_refused "the job-independent producer key could not be formed' in body               # malformed -> the job does not run
    assert '[ -n "$pkey" ] && rung_flags+=(--producer-key "$pkey")' in body
    assert src.index("f22_registry_producer_key(){") < src.index("f22_registry_register(){")                    # defined before its caller


@pytest.mark.parametrize("script,segment_expr,register_marker", [
    ("petri_proxy.sbatch", 'F22_REGISTRY_KEY_SEGMENTS="${F22_REGISTRY_KEY_SEGMENTS:-pass-${PASS:-first}}"', 'f22_init "$WAVE_SHA"; f22_resolve_config "$KEY" "$VIEW"'),
    ("target_attacks.sbatch", 'F22_REGISTRY_KEY_SEGMENTS="${F22_REGISTRY_KEY_SEGMENTS:-sh${SHARDS//,/_}of${NSHARDS}}"', 'f22_resolve_config "$KEY" "$VIEW"'),   # 282b: the shard total
    ("misalign/misalign_proxy.sbatch", 'F22_REGISTRY_KEY_SEGMENTS="${F22_REGISTRY_KEY_SEGMENTS:-target-${TARGET}-mode-${MODE}', "f22_registry_register || true"),   # 282b: the target namespace
    ("static_padding/target_static_padded.sbatch", 'F22_REGISTRY_KEY_SEGMENTS="${F22_REGISTRY_KEY_SEGMENTS:-p${PAD_LIST[0]}}"', "f22_registry_register || true"),
    ("static_padding/target_static_padded_obj.sbatch", 'F22_REGISTRY_KEY_SEGMENTS="${F22_REGISTRY_KEY_SEGMENTS:-p${PAD}-sh${SHARDS_SPEC//[-\\/]/_}}"', "f22_registry_register || true"),
])
def test_each_family_exports_its_identity_segments_before_it_registers(script, segment_expr, register_marker):
    text = (F22 / script).read_text()
    i, j = text.index("export " + segment_expr), text.index(register_marker)
    assert i < j, (script, "the segment export must precede the registration")
    assert subprocess.run(["bash", "-n", str(F22 / script)], capture_output=True).returncode == 0
    # the static job keys by its job name alone (the phase is in the name); the gsm8k row keeps its job-keyed default — neither exports a segment
    for untouched in ("target_static.sbatch", "../gsm8k/gsm8k_thinking.sbatch"):
        assert "F22_REGISTRY_KEY_SEGMENTS" not in (F22 / untouched).read_text(), untouched


def _checkout(root: Path) -> Path:
    cfg = yaml.safe_dump({"explicit_cells": ["none", "prompted__si_constitution"], "model": "Qwen/Qwen3.5-4B"})
    roster = yaml.safe_dump({"targets": {"qwen35_4b": {"views": {"main": {"cells": {"prompted__si_constitution": None}}}}}})
    return synthetic_checkout(root, configs={"configs/sdfbase_full22_qwen35_4b.yaml": cfg, "configs/sdfbase_full22_roster.yaml": roster,
                                             "configs/sdfbase_full22_pins.json": "{}"})


def _registered(shared: Path, job: str) -> dict:
    doc = json.loads((shared / "registry" / "pending" / f"{job}.json").read_text())
    return {r["route"]["object"]: (r["run_id"], r["plan"]["producer_run_key_legacy"]) for r in doc["runs"]}


def test_two_attempts_of_one_measurement_register_the_same_runs_and_an_unnamed_kind_stays_job_keyed(tmp_path, monkeypatch):
    co = _checkout(tmp_path / "a")
    for s in ("shared1", "shared2", "shared3", "shared4"):
        (tmp_path / s).mkdir()                                                     # static-reg-fix-6: the storage root exists before the spool preflight
    monkeypatch.setenv("F22_REGISTRY_KEY_SEGMENTS", "pass-first")                 # what petri_proxy.sbatch exports before it registers
    first = run_f22_registration(co, tmp_path / "shared1", key="qwen35_4b", view="main", cfg_rel="configs/sdfbase_full22_qwen35_4b.yaml",
                                 job_id="4242", job_name="f22-petri-qwen35_4b", sha=CODE_SHA, evals="xstest")
    second = run_f22_registration(co, tmp_path / "shared2", key="qwen35_4b", view="main", cfg_rel="configs/sdfbase_full22_qwen35_4b.yaml",
                                  job_id="4343", job_name="f22-petri-qwen35_4b", sha=CODE_SHA, evals="xstest")
    for r in (first, second):
        assert r.returncode == 0 and "REGISTERED pending=" in r.stdout, r.stdout + r.stderr
    a, b = _registered(tmp_path / "shared1", "4242"), _registered(tmp_path / "shared2", "4343")
    assert set(a) == set(b) == {"none", "prompted__si_constitution"}
    for obj in a:
        assert a[obj] == b[obj], (obj, a[obj], b[obj])                            # same run id, same key, under two job ids
        assert a[obj][1] == "sdfbase_full22-petri-qwen35_4b-pass-first"
    assert len({rid for rid, _k in a.values()}) == 2                               # the object key still tells the runs apart
    # a kind the table does not name: jobplan's default, job-keyed — two job ids, two run ids
    monkeypatch.delenv("F22_REGISTRY_KEY_SEGMENTS", raising=False)
    c = run_f22_registration(co, tmp_path / "shared3", key="qwen35_4b", view="main", cfg_rel="configs/sdfbase_full22_qwen35_4b.yaml",
                             job_id="5151", job_name="mmluF-qwen35_4b-main", sha=CODE_SHA, evals="xstest")
    d = run_f22_registration(co, tmp_path / "shared4", key="qwen35_4b", view="main", cfg_rel="configs/sdfbase_full22_qwen35_4b.yaml",
                             job_id="5252", job_name="mmluF-qwen35_4b-main", sha=CODE_SHA, evals="xstest")
    assert c.returncode == 0 and d.returncode == 0, c.stdout + c.stderr + d.stdout + d.stderr
    cc, dd = _registered(tmp_path / "shared3", "5151"), _registered(tmp_path / "shared4", "5252")
    assert cc["none"][1] == "mmluF-qwen35_4b-main@5151" and dd["none"][1] == "mmluF-qwen35_4b-main@5252" and cc["none"][0] != dd["none"][0]


def test_a_malformed_segment_stops_the_job_before_it_registers(tmp_path, monkeypatch):
    co = _checkout(tmp_path / "a"); (tmp_path / "shared").mkdir()                   # the storage root exists; the refusal under test is the key's
    monkeypatch.setenv("F22_REGISTRY_KEY_SEGMENTS", "pass first")                  # a space: not one token
    r = run_f22_registration(co, tmp_path / "shared", key="qwen35_4b", view="main", cfg_rel="configs/sdfbase_full22_qwen35_4b.yaml",
                             job_id="4242", job_name="f22-petri-qwen35_4b", sha=CODE_SHA, evals="xstest")
    out = r.stdout + r.stderr
    assert "self-registration REFUSED" in out and "the job-independent producer key could not be formed" in out, out
    assert not (tmp_path / "shared" / "registry" / "pending" / "4242.json").exists()


def _rid(pkey: str, obj: str = "none") -> str:
    """The registry's own run-id derivation over a key (identity.run_id: campaign:eval:instrument[:12]:sha16(producer key | object key))."""
    sys.path.insert(0, str(REPO / "src"))
    from sisteer.registry import identity
    return identity.run_id("sdfbase_full22", pkey, obj, "agentic_leaking", "a" * 64)


def test_282b_two_misalign_targets_sharing_model_and_view_are_two_runs_and_a_retry_is_the_same_run():
    """qwen35_4b and qwen35_4b_psr both resolve to KEY=qwen35_4b / VIEW=main / MODE=full and both measure `none` under distinct
    roots — the TARGET namespace in the segment keeps them two runs; the two organisms decode rungs (vendor | t1) and a vendor re-sample root
    are two runs each; the same measurement under another job id (a retry) is the SAME key, hence the same run id."""
    a = _key("f22-misalign-qwen35_4b", "qwen35_4b", "target-qwen35_4b-mode-full")
    b = _key("f22-misalign-qwen35_4b", "qwen35_4b", "target-qwen35_4b_psr-mode-full")
    assert a[0] == b[0] == 0 and a[1] != b[1] and _rid(a[1]) != _rid(b[1]), (a, b)
    v = _key("f22-misalign-em-llama8b", "llama8b", "target-llama8b_org-mode-organisms-sampler-vendor")
    t = _key("f22-misalign-em-llama8b", "llama8b", "target-llama8b_org-mode-organisms-sampler-t1")
    assert v[0] == t[0] == 0 and v[1] != t[1] and _rid(v[1], "em-financial_r32") != _rid(t[1], "em-financial_r32"), (v, t)
    r1 = _key("f22-misalign-qwen35_4b", "qwen35_4b", "target-qwen35_4b-mode-vendor")
    r2 = _key("f22-misalign-qwen35_4b", "qwen35_4b", "target-qwen35_4b-mode-vendor-root-r2")
    assert r1[0] == r2[0] == 0 and r1[1] != r2[1], (r1, r2)
    # a retry: the key names no job — the job NAME is read for the family only, the job ID never — so two attempts agree
    assert _key("f22-misalign-qwen35_4b", "qwen35_4b", "target-qwen35_4b_psr-mode-full") == b
    # the sbatch forms the segment from the TARGET it receives and resolves the sampler BEFORE the export (the organisms branch below reads it)
    text = (F22 / "misalign/misalign_proxy.sbatch").read_text()
    exp = text.index('export F22_REGISTRY_KEY_SEGMENTS="${F22_REGISTRY_KEY_SEGMENTS:-target-${TARGET}-mode-${MODE}')
    assert text.index('EM_ORG_SAMPLER="${EM_ORG_SAMPLER:-vendor}"') < exp < text.index("f22_registry_register || true")
    assert '-sampler-${EM_ORG_SAMPLER}' in text and '-root-${VENDOR_ROOT#misalign_vendor_}' in text and 'TARGET="${1:?target' in text


def test_282b_the_attack_key_carries_the_shard_total_so_a_smoke_and_a_confirmatory_run_are_two_runs():
    """jailbreak_run.py selects goals[shard::nshards]; `sh0` alone aliased the one-goal smoke (nshards 259) and the 9-shard
    run (nshards 9) — the total is in the key; identical values across job ids agree."""
    smoke = _key("f22-att-qwen35_35b_a3b_s2r-crescendo", "qwen35_35b_a3b_s2r", "sh0of259")
    full = _key("f22-att-qwen35_35b_a3b_s2r-crescendo", "qwen35_35b_a3b_s2r", "sh0_1_2_3_4of9")
    assert smoke[0] == full[0] == 0 and smoke[1] != full[1] and _rid(smoke[1], "constv2ctl2-full__psr_ce_all") != _rid(full[1], "constv2ctl2-full__psr_ce_all")
    assert _key("f22-att-qwen35_35b_a3b_s2r-crescendo", "qwen35_35b_a3b_s2r", "sh0_1_2_3_4of9") == full          # another job id, the same measurement
    text = (F22 / "target_attacks.sbatch").read_text()
    assert 'export F22_REGISTRY_KEY_SEGMENTS="${F22_REGISTRY_KEY_SEGMENTS:-sh${SHARDS//,/_}of${NSHARDS}}"' in text
    assert text.index('NSHARDS="${5:-9}"') < text.index('export F22_REGISTRY_KEY_SEGMENTS=')                     # the total is set before the export


# ------------------------------------------------- since 2026-10-11: the climb key is for NEW registrations only


def _checkout_before_d456(root: Path) -> Path:
    """The REAL common.sh as it stood before the climb key landed — this tree's file with the one `f22-bonclimb-*)` case line removed, byte for byte
    otherwise — so the fixture of an EXISTING climb registration is minted exactly as the live ones were (jobplan's `<job_name>@<job_id>`)."""
    co = _checkout(root)
    sh = co / "scripts/sdfbase/full22/common.sh"; text = sh.read_text()
    lines = [ln for ln in text.splitlines(keepends=True) if not (ln.lstrip().startswith("f22-bonclimb-*)") and 'fam="bonclimb"' in ln)]
    assert len(lines) == len(text.splitlines(keepends=True)) - 1, "the bonclimb case line of f22_registry_producer_key is the one to remove"
    sh.write_text("".join(lines))
    return co


def _spool_files(shared: Path) -> list:
    return sorted(str(p.relative_to(shared)) for p in (shared / "registry" / "records").rglob("*.jsonl"))


def test_d456_the_climb_key_changes_identity_for_new_registrations_only_an_existing_registration_keeps_its_run_id(tmp_path, monkeypatch):
    """Over a FIXTURE of an existing registration — job 4444, `f22-bonclimb-qwen35_4b`, registered by the earlier common.sh
    under the job-keyed key — the real registration of the SAME job on this tree keeps the pending file byte for byte (its identity;
    `f22_registry_register` returns on an existing pending file) and writes no record: the run_id is frozen. A NEW job id registers the
    job-independent key (`sdfbase_full22-bonclimb-qwen35_4b`, distinct run ids from the old ones), and another new task of the array — a
    second job id — registers the SAME run ids as the first (one run per cell across the array)."""
    monkeypatch.delenv("F22_REGISTRY_KEY_SEGMENTS", raising=False)
    shared = tmp_path / "shared"; shared.mkdir()
    old_co = _checkout_before_d456(tmp_path / "old")
    r = run_f22_registration(old_co, shared, key="qwen35_4b", view="main", cfg_rel="configs/sdfbase_full22_qwen35_4b.yaml",
                             job_id="4444", job_name="f22-bonclimb-qwen35_4b", sha=CODE_SHA, evals="strongreject_bon")
    assert r.returncode == 0 and "REGISTERED pending=" in r.stdout, r.stdout + r.stderr
    existing = _registered(shared, "4444")
    assert set(existing) == {"none", "prompted__si_constitution"} and all(k == "f22-bonclimb-qwen35_4b@4444" for _rid_, k in existing.values()), existing
    pend = shared / "registry" / "pending" / "4444.json"; frozen_bytes = pend.read_bytes(); frozen_spool = _spool_files(shared)
    assert frozen_spool, "the fixture registration wrote its run records"
    # the same job id on THIS tree (the climb key in the table): the existing registration stands — nothing re-minted, nothing written
    co = _checkout(tmp_path / "new")
    r = run_f22_registration(co, shared, key="qwen35_4b", view="main", cfg_rel="configs/sdfbase_full22_qwen35_4b.yaml",
                             job_id="4444", job_name="f22-bonclimb-qwen35_4b", sha=CODE_SHA, evals="strongreject_bon")
    assert r.returncode == 0 and f"REGISTERED pending={pend}" in r.stdout, r.stdout + r.stderr
    assert pend.read_bytes() == frozen_bytes and _spool_files(shared) == frozen_spool and _registered(shared, "4444") == existing
    # a NEW job id: the job-independent key, run ids distinct from the frozen ones
    r = run_f22_registration(co, shared, key="qwen35_4b", view="main", cfg_rel="configs/sdfbase_full22_qwen35_4b.yaml",
                             job_id="4545", job_name="f22-bonclimb-qwen35_4b", sha=CODE_SHA, evals="strongreject_bon")
    assert r.returncode == 0 and "REGISTERED pending=" in r.stdout, r.stdout + r.stderr
    new = _registered(shared, "4545")
    assert set(new) == set(existing) and all(k == "sdfbase_full22-bonclimb-qwen35_4b" for _rid_, k in new.values()), new
    assert all(new[obj][0] != existing[obj][0] for obj in new) and _registered(shared, "4444") == existing and pend.read_bytes() == frozen_bytes
    # another task of the same array (another job id, the same measurement): the SAME run ids — one run per cell, not one per task
    r = run_f22_registration(co, shared, key="qwen35_4b", view="main", cfg_rel="configs/sdfbase_full22_qwen35_4b.yaml",
                             job_id="4646", job_name="f22-bonclimb-qwen35_4b", sha=CODE_SHA, evals="strongreject_bon")
    assert r.returncode == 0 and "REGISTERED pending=" in r.stdout, r.stdout + r.stderr
    assert _registered(shared, "4646") == new
    # the key itself is a pure function of (kind, tag): the same under any job id, and distinct from the store512 job's key over the same tag
    assert _key("f22-bonclimb-qwen35_4b", "qwen35_4b") == (0, "sdfbase_full22-bonclimb-qwen35_4b", "")
    assert _key("f22-stat-qwen35_4b-3", "qwen35_4b")[1] == "sdfbase_full22-stat-3-qwen35_4b" != _key("f22-bonclimb-qwen35_4b", "qwen35_4b")[1]

"""full22 SAMPLED re-run — lane G2 (the 16,384 extend, the capability battery, the hop-timeout guard).

Everything offline and synthetic (tmp_path fixtures; never real data). Covers:
  * the phase grammar's extend / capability arms and the REFUSED greedy comparator.
  * target_sampled.sbatch's dispatch: one if/elif/else chain closed by a refusal, so a phase kind the
    grammar accepts but the script does not implement exits non-zero instead of reporting "done: 0
    failed drivers" (a silent no-op success).
  * hop_timeout_guard.py, ladder_extend.py, write_ladder.py's decision fold, f22s_check_config's 16k
    assertions, common.sh's thinking read-back, the smoke's --thinking-on, the submit printer's third block.
"""
from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
F22 = REPO / "scripts" / "sdfbase" / "full22"
SAMPLED = F22 / "sampled"
PY = sys.executable
sys.path.insert(0, str(SAMPLED))


def _bash(snippet: str, **env) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", f'source "{SAMPLED}/common_sampled.sh"; {snippet}'],
                          capture_output=True, text=True, cwd=REPO, env={**os.environ, **env})


# --------------------------------------------------------------------------- #
# 1. the phase grammar                                                          #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("phase,want", [
    ("ladder_extend_t0", "ladder_extend 0  ladder_extend_t0"),
    ("ladder_extend_t1", "ladder_extend 1  ladder_extend_t1"),
    ("capability_t0", "capability 0  capability_t0"),
    ("capability_t1", "capability 1  capability_t1"),
])
def test_phase_grammar_parses_the_extend_and_capability_arms(phase, want):
    rc = _bash(f'f22s_parse_phase "{phase}"; echo "$PHASE_KIND $TRIAL ${{FAMS[*]:-}} $PHASE_TAG"')
    assert rc.returncode == 0 and rc.stdout.strip() == want, rc.stdout + rc.stderr


@pytest.mark.parametrize("phase,needle", [
    ("capability_greedy", "no greedy GSM8K comparator"),          # sampled only; the arm is refused, not parsed
    ("ladder_extend_t0:pasttense", "takes no family subset"),
    ("capability_t1:xstest", "takes no family subset"),
    ("ladder_extend_t2", "unknown phase"),
    ("capability_t", "unknown phase"),
])
def test_phase_grammar_refuses_the_greedy_comparator_and_malformed_arms(phase, needle):
    rc = _bash(f'f22s_parse_phase "{phase}"; echo "reached"')
    out = rc.stdout + rc.stderr
    assert "FATAL" in out and needle in out and "reached" not in rc.stdout, out
    # the grammar's usage line no longer advertises the greedy arm
    assert "capability_greedy;" not in out


def test_the_reclassify_helper_has_no_greedy_kind():
    src = (SAMPLED / "common_sampled.sh").read_text()
    assert "gsm8k_greedy)" not in src and "_greedy/{tag}" not in src
    rc = _bash('MPY=python3; REPO_DIR=.; WAVE_ROOT=/nonexistent; f22s_reclassify gsm8k_greedy /tmp/x; echo "rc=$?"')
    assert "unknown kind gsm8k_greedy" in rc.stdout and "rc=2" in rc.stdout


# --------------------------------------------------------------------------- #
# 2. the sbatch dispatch                                                        #
# --------------------------------------------------------------------------- #

SBATCH = SAMPLED / "target_sampled.sbatch"
_ARM_RE = re.compile(r'^\s*(?:if|elif) \[ "\$PHASE_KIND" = (\w+) \]; then', re.M)
_GRAMMAR_RE = re.compile(r"^\s*\w+_t\[01\]\)\s+PHASE_KIND=(\w+)", re.M)

STUB = r'''#!/bin/bash
# a stand-in for common_sampled.sh: every helper the sbatch calls BEFORE its phase blocks, no cluster, no GPU
f22_log(){ echo "[log] $*"; }
f22_fatal(){ f22_log "FATAL: $*"; exit 1; }
f22_attempt_wait(){ :; }                       # the twin-submit grant hook (PR7b): a no-op without SISTEER_ATTEMPT_ID
f22_full22_manifest(){ echo "configs/capeval/manifests/petri_full22_$1.yaml"; }
f22_init(){ WAVE_ROOT="$F22_TEST_ROOT/wave"; SHARED="$F22_TEST_ROOT"; REPO_DIR="$F22_TEST_ROOT"; FIGS_DIR="$F22_TEST_ROOT/figs"; mkdir -p "$WAVE_ROOT" "$FIGS_DIR"; }
f22s_parse_phase(){ PHASE_KIND="$F22_TEST_KIND"; TRIAL=0; FAMS=(); PHASE_TAG="$1"
  case "$PHASE_KIND" in static_*) FAMS=(pasttense codeattack renellm grok420);; esac; }
f22s_resolve_config(){ KEY="$1"; VIEW="${3:-main}"; TAG="$1"; CFG=cfg.yaml; CFG_ESC=esc.yaml; CFG_ESC16=esc16.yaml
  PRIVATE_RUN_BASE=run_t0; PRIVATE_RUN_ESC=run_t0_cap8192; PRIVATE_RUN_ESC16=run_t0_cap16384; SIBLING_TRIAL=1; }
f22_stop_requested(){ return 1; }
f22s_stop_requested(){ return 1; }
f22_driver_venv(){ MPY=python3; }
f22_serving_venv(){ SPY=python3; }
f22s_assert_vllm_mamba_kernels(){ return 0; }
f22s_check_config(){ return 0; }
f22_bank_view(){ ALL_CELLS=(none cellA); TT_TWO_SI_DEFENSES_DIR="$F22_TEST_ROOT/view"; }
f22_result(){ echo "$*" >> "$FIGS_DIR/RESULT"; }
f22_standup(){ echo "STANDUP $*"; return 1; }
f22_standup_stop(){ :; }
f22s_positive_control(){ return 0; }
f22s_at_cap_static(){ return 2; }
f22s_at_cap_controls(){ return 2; }
f22s_rep_ratio_source(){ echo "REP_RATIO_SOURCE"; return 2; }
f22s_hop_guard(){ echo "HOP_GUARD $*"; return 2; }
f22s_reclassify(){ echo "RECLASSIFY $*"; return 2; }
f22s_event(){ :; }
f22s_event_json(){ echo '{}'; }
f22s_write_ladder(){ :; }
f22s_write_ladder_ext(){ :; }
f22s_rung_window(){ echo 18432; }
f22s_gsm8k_window(){ echo 10240; }
F22S_STATIC_CAP=4096; F22S_STATIC_CAP_ESC=8192; F22S_STATIC_CAP_EXT=16384; F22S_CEILING=16384; F22S_ESC_CONC=32
F22S_FLOOR_SAMPLE_K=200; F22S_CONTROLS=(xstest or_bench_hard_1k); F22S_GSM8K_CAP=8192; F22S_DIR=scripts/sdfbase/full22/sampled
F22S_CONTROLS_RUNGS_EXT="512,2048,8192,16384"; F22S_STATIC_RUNGS_EXT="4096,8192,16384"; F22S_GSM8K_RUNGS="8192,16384"
'''


def _grammar_kinds() -> set:
    return set(_GRAMMAR_RE.findall((SAMPLED / "common_sampled.sh").read_text()))


def _dispatched_kinds() -> list:
    return _ARM_RE.findall(SBATCH.read_text())


def _run_sbatch_with_kind(tmp_path: Path, kind: str) -> subprocess.CompletedProcess:
    """Run target_sampled.sbatch with its `source` of common_sampled.sh swapped for the stub above and the
    grammar answering ``kind`` — the dispatch chain is the code under test, nothing else."""
    src = SBATCH.read_text()
    # the PORTABLE bootstrap (paper-companion-repo PR7b; reproduce/lib/common.sh's two-root rule): the site's SISTEER_SHARED_ROOT /
    # SISTEER_EVAL_REPO_ROOT override first, then the two cluster roots — ONE statement, swapped whole for the stub
    src2, n = re.subn(r'^\{ \[ -n "\$\{SISTEER_SHARED_ROOT:-\}" \] && source "\$\{SISTEER_EVAL_REPO_ROOT:-\$SISTEER_SHARED_ROOT/[^\n]*common_sampled\.sh" 2>/dev/null; \} \\\n'
                      r'\s*\|\| source "\$\{SISTEER_EVAL_REPO_ROOT:\?[^\n]*common_sampled\.sh"\n',
                      'source "$F22_TEST_STUB"\n', src, count=1, flags=re.M)
    assert n == 1, "the sbatch no longer sources common_sampled.sh through the portable bootstrap (the SISTEER_SHARED_ROOT override, then SISTEER_EVAL_REPO_ROOT)"
    copy = tmp_path / "target_sampled.sbatch"
    copy.write_text(src2)
    stub = tmp_path / "stub.sh"
    stub.write_text(STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    (tmp_path / "root").mkdir(exist_ok=True)
    env = {**os.environ, "F22_TEST_STUB": str(stub), "F22_TEST_KIND": kind, "F22_TEST_ROOT": str(tmp_path / "root"),
           "SLURM_JOB_ID": "424242", "SLURM_JOB_NAME": "f22s-test", "CUDA_VISIBLE_DEVICES": "", "USER": os.environ.get("USER", "u")}
    return subprocess.run(["bash", str(copy), "gemma4e4b", f"{kind}_t0", "ab" * 20], capture_output=True, text=True,
                          env=env, cwd=REPO, timeout=120)


def test_sbatch_dispatch_is_one_chain_closed_by_a_refusal(tmp_path):
    src = SBATCH.read_text()
    arms = _dispatched_kinds()
    assert arms and arms[0] and len(set(arms)) == len(arms), arms
    # ONE chain: exactly one `if`, the rest `elif`, closed by `else` + f22_fatal naming the kind
    assert len(re.findall(r'^if \[ "\$PHASE_KIND" = ', src, re.M)) == 1
    assert len(re.findall(r'^elif \[ "\$PHASE_KIND" = ', src, re.M)) == len(arms) - 1
    assert re.search(r'^else\n(?:\s*#[^\n]*\n)*\s*f22_fatal "phase kind \$PHASE_KIND \(\$PHASE\) is not implemented', src, re.M)
    grammar = _grammar_kinds()
    assert set(arms) <= grammar, (arms, grammar)
    # EXECUTABLE: a kind the grammar could hand over that has no block exits 1 with the refusal, never "done: 0 failed"
    rc = _run_sbatch_with_kind(tmp_path, "bogus")
    out = rc.stdout + rc.stderr
    assert rc.returncode == 1, out
    assert "FATAL: phase kind bogus (bogus_t0) is not implemented" in out
    assert "done: 0 failed" not in out and "STANDUP" not in out
    # the RESULT line records the refusal's exit code, not a stale rc=0
    result = (tmp_path / "root" / "figs" / "RESULT").read_text()
    assert "rc=1" in result and "exit=1" in result, result
    # every kind the grammar hands over that the sbatch does NOT dispatch refuses the same way
    for kind in sorted(grammar - set(arms)):
        rc = _run_sbatch_with_kind(tmp_path, kind)
        assert rc.returncode == 1 and f"phase kind {kind} ({kind}_t0) is not implemented" in rc.stdout + rc.stderr, kind
    # and every dispatched kind ENTERS its block (the stub's first helper refuses, so the run stops inside the block)
    for kind in arms:
        rc = _run_sbatch_with_kind(tmp_path, kind)
        out = rc.stdout + rc.stderr
        assert "not implemented" not in out and rc.returncode != 0, (kind, out)


# --------------------------------------------------------------------------- #
# 3. common.sh: the thinking regime + hop timeout read-back composes JSON        #
# --------------------------------------------------------------------------- #

COMMON = (F22 / "common.sh").read_text()


def _one_liner(var: str, arg: str) -> str:
    m = re.search(rf'^\s*{re.escape(var)}=\$\("\$SPY" -c "([^"]+)" "\${arg}"\)', COMMON, re.M)
    assert m, f"common.sh no longer reads {var} with a -c one-liner against ${arg}"
    return m.group(1)


def _py(code: str, *args: str) -> str:
    rc = subprocess.run([PY, "-c", code, *args], capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    return rc.stdout.strip()


def test_standup_record_reads_the_hop_timeout_and_verdict_as_json_never_pythons_none(tmp_path):
    """C (the 2026-09-10 review): a standup without upstream_timeout_s was recorded as the repr `None`, so the
    `${…:-null}` default never fired and standup_jobs.jsonl carried a non-JSON line. Both reads now emit JSON."""
    up = _one_liner("TARGET_UPSTREAM_TIMEOUT_S", "ep")
    verdict = _one_liner("think_verdict", "ep")
    verdict_json = _one_liner("think_verdict_json", "think_verdict")
    ep_full = tmp_path / "full.json"
    ep_full.write_text(json.dumps({"thinking_mode": "on", "upstream_timeout_s": 7200.0,
                                   "thinking_certification": {"verdict": "certified"}}))
    ep_bare = tmp_path / "bare.json"
    ep_bare.write_text(json.dumps({"thinking_mode": "off"}))
    assert _py(up, str(ep_full)) == "7200.0" and _py(up, str(ep_bare)) == "null"
    v_full, v_bare = _py(verdict, str(ep_full)), _py(verdict, str(ep_bare))
    assert v_full == "certified" and v_bare == ""          # a bare word for the case arm; empty when absent
    assert _py(verdict_json, v_full) == '"certified"' and _py(verdict_json, v_bare) == "null"
    # the case arm reads the bare word; the record reads the JSON forms
    assert 'case "$think_verdict" in certified|trivially_in_support)' in COMMON
    assert '"thinking_certification_verdict":%s,"upstream_timeout_s":%s' in COMMON
    assert '"${think_verdict_json:-null}" "${TARGET_UPSTREAM_TIMEOUT_S:-null}"' in COMMON
    assert "print(json.load(open(sys.argv[1])).get('upstream_timeout_s'))" not in COMMON   # the repr read is gone
    # the printf template composes VALID JSON with the absent-field values in place
    m = re.search(r"printf '(\{\"job_id\":[^']*)\\n' \\\n\s*(.*?) >> \"\$out/standup_jobs.jsonl\"", COMMON, re.S)
    assert m, "the standup_jobs.jsonl printf moved"
    template, argline = m.group(1), m.group(2)
    n = template.count("%s")
    # 26 since the native-serving default: the record gained `engine`, `serving_path`, `cells_skipped` and
    # `geometry_adjustments` (four %s), and `lens_reduction_mode` became a JSON value (null on a native standup, the
    # quoted mode on a lens one — a hoisted one-word argument, so the count below still reads one word per %s);
    # 27 since the kind-aware default (2026-09-15): `serving_path_selection` (JSON; null when f22_init never ran)
    assert n == len(re.findall(r'"\$[^"]*"|"\$\([^)]*\)"', argline)) == 27, (n, argline)
    args = ["1", "node", "gemma4e4b", "main", "attacks", "capability:short", "ab" * 20, "lens:stacked", "lens", "[]", "null",
            '"stacked"', "57", "10240", "57", "10240",
            "583680", "1.0", "null", "cluster_b", "0.9", "null", "off", "null", "null", "null", "2026-09-10T00:00:00Z"]
    rec = json.loads(template % tuple(args))
    assert rec["thinking_certification_verdict"] is None and rec["upstream_timeout_s"] is None and rec["thinking_mode"] == "off"
    assert rec["serving_path_selection"] is None
    args[25] = '{"mode":"auto","selected":"lens","reason":"every steered cell is lens-only (alphasteer\\u00d72)"}'
    assert json.loads(template % tuple(args))["serving_path_selection"] == {"mode": "auto", "selected": "lens",
                                                                          "reason": "every steered cell is lens-only (alphasteer\u00d72)"}
    assert rec["engine"] == "lens:stacked" and rec["lens_reduction_mode"] == "stacked" and rec["cells_skipped"] == []
    # a native standup's row: no lens mode (null), the engine and the adjustment chain recorded
    args[7], args[8], args[10], args[11] = "lora_native", "lora_native", '[{"kind":"lora_kernel_index_ceiling","from_S":121,"to_S":107}]', "null"
    rec = json.loads(template % tuple(args))
    assert rec["engine"] == "lora_native" and rec["lens_reduction_mode"] is None and rec["geometry_adjustments"][0]["to_S"] == 107
    args[23], args[24] = '"certified"', "7200.0"
    rec = json.loads(template % tuple(args))
    assert rec["thinking_certification_verdict"] == "certified" and rec["upstream_timeout_s"] == 7200.0


# --------------------------------------------------------------------------- #
# 4. hop_timeout_guard.py                                                       #
# --------------------------------------------------------------------------- #

import hop_timeout_guard as hg  # noqa: E402

STATS = ("INFO 09-10 12:00:{s:02d} [loggers.py:123] Engine 000: Avg prompt throughput: 1200.0 tokens/s, "
         "Avg generation throughput: {tps} tokens/s, Running: {run} reqs, Waiting: 0 reqs, GPU KV cache usage: 40.0%\n")


def _vllm_log(path: Path, windows) -> Path:
    path.write_text("".join(STATS.format(s=i % 60, tps=tps, run=run) for i, (tps, run) in enumerate(windows)))
    return path


def _endpoint_json(path: Path, **fields) -> Path:
    path.write_text(json.dumps({"proxy_url": "http://127.0.0.1:7/v1", **fields}))
    return path


LOADED = [(300.0, 60)] * 30            # 30 windows at Running 60, 5.0 tok/s per sequence
PREFILL_BURST = [(0.03, 60)]           # one prefill-heavy window: 0.0005 tok/s per sequence — the spurious floor
NEAR_EMPTY = [(1.0, 2)]                # Running 2: 0.5 tok/s per sequence, below the eligibility floor


def _guard(tmp_path, *extra, windows=None, rung=16384, client="controls"):
    log = _vllm_log(tmp_path / "vllm_server.log", windows if windows is not None else LOADED)
    out = tmp_path / "rec.json"
    rc = subprocess.run([PY, str(SAMPLED / "hop_timeout_guard.py"), "--rung", str(rung), "--client", client,
                         "--vllm-log", str(log), "--out", str(out), "--repo-dir", str(REPO), *extra],
                        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": f"{REPO}/src"})
    rec = json.loads(out.read_text()) if out.exists() else None
    return rc, rec


def test_hop_guard_reads_either_endpoint_key_and_refuses_a_torn_record(tmp_path):
    # the sdfbase standup's key
    ep = _endpoint_json(tmp_path / "ep1.json", upstream_timeout_s=7200)
    rc, rec = _guard(tmp_path, "--endpoint", str(ep))
    assert rc.returncode == 0 and rec["verdict"] == "PASS", rc.stdout + rc.stderr
    assert rec["upstream_timeout_s"] == 7200.0 and rec["upstream_key"] == "upstream_timeout_s"
    assert rec["upstream_keys_found"] == ["upstream_timeout_s"] and rec["phase"] == "post_standup" and rec["sole_check"] is False
    # the tinycap standup's key (capctrl_serving_up.py) — accepted, recorded under its own name
    ep = _endpoint_json(tmp_path / "ep2.json", proxy_upstream_timeout_s=7200.0)
    rc, rec = _guard(tmp_path, "--endpoint", str(ep))
    assert rc.returncode == 0 and rec["upstream_key"] == "proxy_upstream_timeout_s" and rec["upstream_source"].endswith(":proxy_upstream_timeout_s")
    # both present and equal: fine, both named
    ep = _endpoint_json(tmp_path / "ep3.json", upstream_timeout_s=7200, proxy_upstream_timeout_s=7200.0)
    rc, rec = _guard(tmp_path, "--endpoint", str(ep))
    assert rc.returncode == 0 and rec["upstream_keys_found"] == ["proxy_upstream_timeout_s", "upstream_timeout_s"]
    # both present and DISAGREEING: a torn record, refused — never a choice
    ep = _endpoint_json(tmp_path / "ep4.json", upstream_timeout_s=7200, proxy_upstream_timeout_s=600)
    rc, rec = _guard(tmp_path, "--endpoint", str(ep))
    assert rc.returncode == 2 and rec["verdict"] == "REFUSE" and "DISAGREE" in rec["reason"] and "torn" in rec["reason"]
    # neither present, or not positive: undeterminable
    for fields, needle in (({}, "none of"), ({"upstream_timeout_s": None}, "none of"), ({"upstream_timeout_s": 0}, "not a positive"),
                           ({"proxy_upstream_timeout_s": True}, "not a positive")):
        ep = _endpoint_json(tmp_path / "ep5.json", **fields)
        rc, rec = _guard(tmp_path, "--endpoint", str(ep))
        assert rc.returncode == 2 and "undeterminable" in rec["reason"] and needle in rec["reason"], (fields, rec["reason"])
    assert hg.ENDPOINT_UPSTREAM_TIMEOUT_KEYS == ("upstream_timeout_s", "proxy_upstream_timeout_s")   # the peer's two keys
    # a missing / non-JSON endpoint refuses
    rc, rec = _guard(tmp_path, "--endpoint", str(tmp_path / "nope.json"))
    assert rc.returncode == 2 and "no such file" in rec["reason"]


def test_hop_guard_pre_standup_is_explicit_and_never_the_sole_check(tmp_path):
    # neither flag: undeterminable, refused — a rung is not climbed on absent evidence
    rc, rec = _guard(tmp_path)
    assert rc.returncode == 2 and "neither --endpoint" in rec["reason"] and rec["phase"] is None
    # --pre-standup: the configured default by import (proxy.HOP_TIMEOUT_S), stamped as the pre-standup phase
    rc, rec = _guard(tmp_path, "--pre-standup")
    assert rc.returncode == 0 and rec["verdict"] == "PASS", rc.stdout + rc.stderr
    from sisteer.capeval.serving import proxy as px
    assert rec["upstream_timeout_s"] == float(px.HOP_TIMEOUT_S) and rec["phase"] == "pre_standup" and rec["upstream_key"] is None
    assert "PRE-standup" in rec["upstream_source"] and "mandatory before any request" in rec["upstream_source"]
    assert rec["sole_check"] is False and rec["reason"].startswith("[pre_standup]")
    # both flags: they answer different questions — refused
    ep = _endpoint_json(tmp_path / "ep.json", upstream_timeout_s=7200)
    rc, rec = _guard(tmp_path, "--pre-standup", "--endpoint", str(ep))
    assert rc.returncode == 2 and "mutually exclusive" in rec["reason"]
    # the shell caller passes exactly one of the two, and the post-standup call is the one wired with an endpoint
    src = (SAMPLED / "common_sampled.sh").read_text()
    assert 'if [ -n "$ep" ]; then mode=(--endpoint "$ep"); phase=post-standup; else mode=(--pre-standup); phase=pre-standup; fi' in src
    assert '"${mode[@]}"' in src


def test_hop_guard_floor_is_windowed_p05_with_the_raw_min_as_evidence(tmp_path):
    ep = _endpoint_json(tmp_path / "ep.json", upstream_timeout_s=7200)
    # 30 loaded windows + ONE prefill-heavy window + a near-empty window: the raw min (0.0005 tok/s) would demand
    # 16384 / 0.0005 x 2 = 65,536,000 s and refuse a hop that carries the rung with room; p05 over the eligible
    # windows reads 5.0 tok/s (required 6,554 s < 7,200 s) and the raw min rides along as evidence
    rc, rec = _guard(tmp_path, "--endpoint", str(ep), windows=LOADED + PREFILL_BURST + NEAR_EMPTY)
    assert rc.returncode == 0 and rec["verdict"] == "PASS", rc.stdout + rc.stderr
    assert rec["floor_tok_s"] == pytest.approx(5.0) and rec["required_s"] == pytest.approx(16384 / 5.0 * 2.0)
    assert rec["raw_min_tok_s"] == pytest.approx(0.0005) and rec["raw_min_tps"] == 0.03 and rec["raw_min_running"] == 60
    assert rec["floor_n_eligible"] == 31 and rec["floor_n_all"] == 32 and rec["floor_samples"] == 31   # Running 2 is not eligible
    assert rec["floor_quantile"] == 0.05 and rec["floor_min_running"] == 8 and rec["floor_min_samples"] == 10
    assert rec["eligible_min_tok_s"] == pytest.approx(0.0005) and "p5 of per-sequence decode rates" in rec["floor_rule"]
    assert "raw min 0.0005" in rec["reason"] and rec["binding_hop"] == "upstream" and rec["binding_s"] == 7200.0
    assert rec["floor_source"][0]["n_windows"] == 32 and rec["floor_source"][0]["n_eligible"] == 31
    # too few eligible windows: undeterminable, refused (the raw windows alone never decide)
    rc, rec = _guard(tmp_path, "--endpoint", str(ep), windows=[(300.0, 60)] * 5 + NEAR_EMPTY * 20)
    assert rc.returncode == 2 and "only 5 eligible" in rec["reason"] and "undeterminable" in rec["reason"]
    # the knobs are honoured and recorded
    rc, rec = _guard(tmp_path, "--endpoint", str(ep), "--floor-min-running", "1", "--floor-quantile", "0.0", "--floor-min-samples", "3",
                     windows=LOADED + PREFILL_BURST)
    assert rc.returncode == 2 and rec["floor_tok_s"] == pytest.approx(0.0005) and rec["floor_min_running"] == 1
    assert rec["required_s"] == pytest.approx(16384 / 0.0005 * 2.0) and "raise scripts/sdfbase/mt_serving_up.py --upstream-timeout-s" in rec["reason"]
    # no decode window at all (every stats line has zero throughput): undeterminable
    rc, rec = _guard(tmp_path, "--endpoint", str(ep), windows=[(0.0, 60)] * 30)
    assert rc.returncode == 2 and "undeterminable" in rec["reason"]


def test_hop_guard_verdict_arithmetic_and_the_binding_hop(tmp_path):
    from sisteer.capeval.serving import proxy as px
    ep = _endpoint_json(tmp_path / "ep.json", upstream_timeout_s=7200)
    # controls client: AIOHTTP_TOTAL_TIMEOUT_S by AST (7200.0) -> upstream binds at equality
    rc, rec = _guard(tmp_path, "--endpoint", str(ep), client="controls")
    assert rec["client_timeout_s"] == 7200.0 and rec["client_source"].endswith("served_capability_run.py:AIOHTTP_TOTAL_TIMEOUT_S")
    assert rec["binding_hop"] == "upstream" and rec["binding_s"] == 7200.0 and rec["verdict"] == "PASS"
    # static client: proxy.CLIENT_TIMEOUT_S by import (7500) -> upstream (7200) binds
    rc, rec = _guard(tmp_path, "--endpoint", str(ep), client="static")
    assert rec["client_timeout_s"] == float(px.CLIENT_TIMEOUT_S) == 7500.0 and rec["binding_hop"] == "upstream"
    # a SMALLER client hop binds and the refusal names the client constant to raise
    ep_big = _endpoint_json(tmp_path / "ep_big.json", upstream_timeout_s=99999)
    rc, rec = _guard(tmp_path, "--endpoint", str(ep_big), "--margin", "3.0", client="static")
    assert rec["binding_hop"] == "client:static" and rec["binding_s"] == 7500.0
    assert rec["required_s"] == pytest.approx(16384 / 5.0 * 3.0) and rc.returncode == 2 and rec["verdict"] == "REFUSE"
    assert "sisteer.capeval.serving.proxy.CLIENT_TIMEOUT_S" in rec["reason"] and f">= {16384 / 5.0 * 3.0:.0f} s" in rec["reason"]
    # the margin flips the verdict at the boundary: 16384 / 5 x 2.2 = 7209 > 7200
    rc, rec = _guard(tmp_path, "--endpoint", str(ep), "--margin", "2.2")
    assert rc.returncode == 2 and rec["verdict"] == "REFUSE" and rec["binding_hop"] == "upstream"
    assert "SHORT completion" in rec["reason"] and "--upstream-timeout-s" in rec["reason"]
    assert "[hop-timeout-guard] REFUSED" in rc.stderr
    # a smaller rung passes under the same hop
    rc, rec = _guard(tmp_path, "--endpoint", str(ep), "--margin", "2.2", rung=8192)
    assert rc.returncode == 0 and rec["verdict"] == "PASS" and rec["rung"] == 8192
    # bad inputs refuse
    rc, rec = _guard(tmp_path, "--endpoint", str(ep), "--margin", "0.5")
    assert rc.returncode == 2 and "margin" in rec["reason"]
    # the record is ONE JSON line on stdout too, and carries the tool's sha
    rc, rec = _guard(tmp_path, "--endpoint", str(ep))
    line = json.loads(rc.stdout.strip().splitlines()[-1])
    assert line["verdict"] == "PASS" and line["tool"]["sha256"] == rec["tool"]["sha256"] == hg._sha(SAMPLED / "hop_timeout_guard.py")


def test_hop_guard_is_at_least_as_strict_as_the_peers_transport_check():
    """Min hop binds; refuse on undeterminable; read BOTH endpoint key names — pinned on the module."""
    src = (SAMPLED / "hop_timeout_guard.py").read_text()
    assert 'ENDPOINT_UPSTREAM_TIMEOUT_KEYS = ("upstream_timeout_s", "proxy_upstream_timeout_s")' in src
    assert "binding = min(up_s, cl_s)" in src and "torn" in src and "undeterminable" in src
    assert hg.FLOOR_MIN_RUNNING == 8 and hg.FLOOR_QUANTILE == 0.05 and hg.FLOOR_MIN_SAMPLES == 10 and hg.DEFAULT_MARGIN == 2.0


# --------------------------------------------------------------------------- #
# 5. ladder_extend.py + write_ladder.py's decision fold                          #
# --------------------------------------------------------------------------- #

import ladder_decide as ld  # noqa: E402
import ladder_extend as le  # noqa: E402
import write_ladder as wl  # noqa: E402
from test_ladder_decide import (CTL, LOOP_RR, COHERENT, _base_rows, _sibling_rows,  # noqa: E402
                                _write_controls, _write_static)


def _rep_src(tmp_path: Path) -> Path:
    from test_ladder_decide import REP_SRC
    p = tmp_path / "join_scores.py"
    p.write_text("import json\n\n\n" + REP_SRC + "\n\nX = 1\n")
    return p


def _ctl_items_of(kind: str, n: int = 200):
    """kind: 'trunc' (5 coherent at cap -> escalate), 'loops' (3 loops at cap -> exit_le_1pct_with_residual),
    'clean' (no at-cap -> exit_zero), 'trunc_many' (60 coherent at cap: the ceiling residual)."""
    out = []
    for i in range(n):
        iid = f"it{i:04d}"
        if kind == "trunc" and i < 5:
            out.append((iid, COHERENT, "length", 8192))
        elif kind == "trunc_many" and i < 60:
            out.append((iid, COHERENT, "length", 16384))
        elif kind == "loops" and i < 3:
            out.append((iid, LOOP_RR, "length", 8192))
        else:
            out.append((iid, f"terminated reply {i}", "stop", 40))
    return out


def _world(tmp_path: Path, *, with_gsm8k: bool = True) -> dict:
    """Synthetic cluster roots at the 8192 rung + the reclassify summaries ladder_extend reads:
    static: run_t0 (4096) escalated to run_t0_cap8192 (7 at-cap items, 4 truncated -> escalate);
    controls: cellA/xstest truncated (escalate), cellB/xstest loops only (exit_le_1pct_with_residual),
    cellC/xstest clean (exit_zero); gsm8k: cellA/xstest-shaped store truncated (escalate)."""
    rep = _rep_src(tmp_path)
    priv = tmp_path / "private"
    _write_static(priv, "run_t0", _base_rows())
    rows8 = _base_rows(cap=8192)
    _write_static(priv, "run_t0_cap8192", [rows8[0] + rows8[1]], escalation={"cap_from": 4096, "cap_to": 8192, "n_rows": 200})
    _write_static(priv, "run_t1", _sibling_rows())
    sampled = tmp_path / "sampled"
    _write_controls(sampled / "controls_cap8192" / "TAG_t0", "cellA", _ctl_items_of("trunc"), cap=8192)
    _write_controls(sampled / "controls_cap8192" / "TAG_t0", "cellB", _ctl_items_of("loops"), cap=8192)
    _write_controls(sampled / "controls_cap8192" / "TAG_t0", "cellC", _ctl_items_of("clean"), cap=8192)
    out = tmp_path / "reclass"
    assert ld.main(["reclassify", "--kind", "static", "--root", str(priv), "--run", "run_t0", "--sibling-run", "run_t1",
                    "--out-dir", str(out / "static"), "--rep-ratio-source", str(rep)]) == 0
    assert ld.main(["reclassify", "--kind", "controls", "--root", str(sampled), "--tag", "TAG", "--trial", "0",
                    "--out-dir", str(out / "controls"), "--rep-ratio-source", str(rep)]) == 0
    recl = {"static": out / "static" / "reclassify.json", "controls": out / "controls" / "reclassify.json"}
    if with_gsm8k:
        _write_controls(sampled / "gsm8k_thinking_on_cap8192" / "TAG_t0", "cellA", _ctl_items_of("trunc"), cap=8192)
        assert ld.main(["reclassify", "--kind", "controls", "--root", str(sampled), "--tag", "TAG", "--trial", "0",
                        "--root-pattern", "gsm8k_thinking_on_cap{cap}/{tag}_t{trial}",
                        "--out-dir", str(out / "gsm8k"), "--rep-ratio-source", str(rep)]) == 0
        recl["gsm8k"] = out / "gsm8k" / "reclassify.json"
    return {"rep": rep, "priv": priv, "sampled": sampled, "recl": recl}


def _guard_record(path: Path, **over) -> Path:
    rec = {"rung": 16384, "phase": "pre_standup", "client": "controls", "verdict": "PASS", "binding_hop": "upstream",
           "binding_s": 7200.0, "required_s": 6553.6, "upstream_timeout_s": 7200.0, "upstream_key": None,
           "client_timeout_s": 7200.0, "floor_tok_s": 5.0, "raw_min_tok_s": 0.0005, "margin": 2.0, "sole_check": False,
           "reason": "ok"}
    rec.update(over)
    path.write_text(json.dumps(rec))
    return path


def _decide_argv(w, out, *extra, cap="8192", next_cap="16384", events=None):
    argv = ["decide", "--cap", cap, "--tag", "TAG", "--trial", "0", "--out", str(out)]
    if next_cap is not None:
        argv += ["--next-cap", next_cap]
    argv += [x for k, p in w["recl"].items() for x in ("--reclassify", f"{k}={p}")]
    if events is not None:
        argv += ["--events", str(events)]
    return argv + list(extra)


def test_ladder_extend_decides_promotes_and_files_exit_kinds_separately(tmp_path, capsys):
    w = _world(tmp_path)
    g_ctl = _guard_record(tmp_path / "g_ctl.json")
    g_sta = _guard_record(tmp_path / "g_sta.json", client="static", client_timeout_s=7500.0)
    out, events = tmp_path / "ladders" / "decision_16384.json", tmp_path / "ladders" / "events.jsonl"
    rc = le.main(_decide_argv(w, out, "--hop-guard", str(g_ctl), "--hop-guard", str(g_sta), events=events))
    assert rc == 0, capsys.readouterr().err
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    rec = json.loads(out.read_text())
    assert rec["kind"] == "ladder_extend" and rec["rule"] == "item_filtered_v1" and rec["cap_from"] == 8192 and rec["cap_to"] == 16384
    assert rec["at_ceiling"] is False and rec["ceiling"] == 16384 and rec["never_past_ceiling"] is True
    assert rec["decision"] == "rung" and line["decision"] == "rung" and "pairs" not in line
    by = {(p["kind"], p.get("attack") or p["cell"], p.get("cell") if p["kind"] == "static" else p["control"]): p for p in rec["pairs"]}
    assert rec["n_pairs_considered"] == 5 and rec["n_promoted"] == 3
    # static: 4 truncated of 200 (no sibling at 8192) -> escalate, promoted, items file pinned by sha
    st = by[("static", "pasttense", "none")]
    assert st["promoted"] and st["decision"] == "escalate" and st["exit_kind"] is None and st["next_cap"] == 16384
    assert st["counts"]["n_truncated"] == 4 and st["counts"]["n_loop"] == 2 and st["counts"]["n_repetitive"] == 1 and st["counts"]["n_boundary"] == 0
    assert st["items_sha256"] == le._sha(Path(st["items_path"])) and st["private_run"] == "run_t0_cap8192"
    # controls: cellA escalates; cellB's at-cap items are ALL loops -> exit on the 1% rule WITH a residual; cellC clean
    a, b, c = by[("controls", "cellA", CTL)], by[("controls", "cellB", CTL)], by[("controls", "cellC", CTL)]
    assert a["promoted"] and a["counts"]["n_truncated"] == 5 and a["counts"]["share_truncated"] == pytest.approx(0.025)
    assert not b["promoted"] and b["decision"] == "exit_le_1pct" and b["exit_kind"] == "exit_le_1pct_with_residual"
    assert b["residual_count"] == 3 and b["residual_share"] == pytest.approx(0.015) and b["counts"]["n_loop"] == 3 and b["counts"]["n_truncated"] == 0
    assert not c["promoted"] and c["exit_kind"] == "exit_zero" and c["residual_count"] == 0 and c["residual"] is None
    # gsm8k: a controls-shaped kind with its own reclassify
    g = by[("gsm8k", "cellA", CTL)]
    assert g["promoted"] and g["kind"] == "gsm8k" and rec["by_kind"]["gsm8k"]["n_promoted"] == 1
    # the two populations are filed SEPARATELY, with residual share + count each
    assert rec["n_by_exit_kind"] == {"exit_zero": 1, "exit_le_1pct_with_residual": 1, "exit_ceiling": 0}
    assert [p["cell"] for p in rec["exit_le_1pct_with_residual"]] == ["cellB"] and rec["exit_ceiling"] == []
    assert rec["exit_le_1pct_with_residual"][0] == {"kind": "controls", "attack": None, "cell": "cellB", "control": CTL, "cap": 8192,
                                                    "decision": "exit_le_1pct", "exit_kind": "exit_le_1pct_with_residual",
                                                    "residual_count": 3, "residual_share": pytest.approx(0.015), "n_truncated": 0,
                                                    "share_truncated": 0.0, "n_loop": 3, "n_repetitive": 0, "n_boundary": 0, "n_responses": 200}
    assert [p["cell"] for p in rec["exit_zero"]] == ["cellC"] and rec["absent_kinds"] == []
    # the hop guards are CARRIED and were ENFORCED
    assert rec["hop_timeout_guard_enforced"] is True and len(rec["hop_timeout_guard"]) == 2
    assert {g["client"] for g in rec["hop_timeout_guard"]} == {"controls", "static"} and all(g["verdict"] == "PASS" for g in rec["hop_timeout_guard"])
    # the events: one per pair at 8192 with exit_kind + extend_kind, one ladder-level per kind at 16384
    evs = [json.loads(l) for l in events.read_text().splitlines()]
    pair_evs = [e for e in evs if "cell" in e]
    lad_evs = [e for e in evs if "cell" not in e]
    assert len(pair_evs) == 5 and all(e["rung"] == 8192 and e["rule"] == "item_filtered_v1" for e in pair_evs)
    assert {e["extend_kind"] for e in pair_evs} == {"static", "controls", "gsm8k"}
    assert {(e["extend_kind"], e["cell"]): e["exit_kind"] for e in pair_evs if e["kind"] == "controls"} == {
        ("controls", "cellA"): None, ("controls", "cellB"): "exit_le_1pct_with_residual", ("controls", "cellC"): "exit_zero", ("gsm8k", "cellA"): None}
    assert {e["extend_kind"]: e["decision"] for e in lad_evs} == {"static": "rung", "controls": "rung", "gsm8k": "rung"}
    assert all(e["rung"] == 16384 and e["ceiling"] == 16384 and e["hop_timeout_guard"] for e in lad_evs)
    # `list` prints exactly the promoted pairs per kind
    le.main(["list", "--decision", str(out), "--kind", "static"])
    assert capsys.readouterr().out.strip() == f"pasttense\tnone\t{st['items_path']}"
    le.main(["list", "--decision", str(out), "--kind", "controls"])
    assert capsys.readouterr().out.strip() == f"cellA\t{CTL}\t{a['items_path']}"
    le.main(["list", "--decision", str(out), "--kind", "gsm8k"])
    assert capsys.readouterr().out.strip().startswith("cellA\t")
    with pytest.raises(SystemExit):   # gsm8k_greedy is not a kind
        le.main(["list", "--decision", str(out), "--kind", "gsm8k_greedy"])
    assert le.KINDS == ("static", "controls", "gsm8k")


def test_ladder_extend_enforces_the_hop_guards_and_its_refusals(tmp_path, capsys):
    w = _world(tmp_path, with_gsm8k=False)
    out = tmp_path / "d.json"
    ok = _guard_record(tmp_path / "ok.json")
    # a REFUSE record: exit 2, nothing written
    bad = _guard_record(tmp_path / "bad.json", verdict="REFUSE", reason="rung 16384: the binding hop is upstream at 600 s ...")
    assert le.main(_decide_argv(w, out, "--hop-guard", str(ok), "--hop-guard", str(bad))) == 2
    err = capsys.readouterr().err
    assert "no rung is promoted under a transport that cannot carry it" in err and "verdict='REFUSE'" in err and not out.exists()
    # a PASS at the WRONG rung: exit 2
    wrong = _guard_record(tmp_path / "wrong.json", rung=8192)
    assert le.main(_decide_argv(w, out, "--hop-guard", str(wrong))) == 2
    assert "rung=8192 != the rung about to be generated 16384" in capsys.readouterr().err and not out.exists()
    # NO hop guard at all on a climb: exit 2 (absent evidence)
    assert le.main(_decide_argv(w, out)) == 2
    assert "needs at least one --hop-guard" in capsys.readouterr().err and not out.exists()
    # a missing / torn record: exit 2
    assert le.main(_decide_argv(w, out, "--hop-guard", str(tmp_path / "missing.json"))) == 2
    capsys.readouterr()
    # with a PASS the decision is written and the record carries the guard
    assert le.main(_decide_argv(w, out, "--hop-guard", str(ok))) == 0
    assert json.loads(out.read_text())["hop_timeout_guard"][0]["sha256"] == le._sha(ok)
    capsys.readouterr()
    # --next-cap past the ceiling, or not the next rung, refuses
    for nxt in ("32768", "8192"):
        assert le.main(_decide_argv(w, tmp_path / "x.json", "--hop-guard", str(ok), next_cap=nxt)) == 2
        capsys.readouterr()
    # the controls reclassify is required; a foreign kind refuses; an incomplete summary refuses
    argv = _decide_argv(w, tmp_path / "x.json", "--hop-guard", str(ok))
    i = argv.index("--reclassify", argv.index("--reclassify") + 1)
    assert le.main(argv[:i] + argv[i + 2:]) == 2 and "controls=<reclassify.json> is required" in capsys.readouterr().err
    assert le.main(_decide_argv(w, tmp_path / "x.json", "--hop-guard", str(ok), "--reclassify", "gsm8k_greedy=/x")) == 2
    capsys.readouterr()
    summ = json.loads(w["recl"]["controls"].read_text())
    summ["complete"] = False; summ["n_refused"] = 1
    summ["entries"][0].update(status="refused", reason="torn line")
    w["recl"]["controls"].write_text(json.dumps(summ))
    assert le.main(_decide_argv(w, tmp_path / "x.json", "--hop-guard", str(ok))) == 2
    assert "complete=False" in capsys.readouterr().err
    # a summary under another rule refuses
    summ["complete"] = True; summ["n_refused"] = 0; summ["entries"][0].update(status="decided"); summ["rule"] = "all_at_cap_v0"
    w["recl"]["controls"].write_text(json.dumps(summ))
    assert le.main(_decide_argv(w, tmp_path / "x.json", "--hop-guard", str(ok))) == 2
    assert "never acts on a decision made under another rule" in capsys.readouterr().err


def test_ladder_extend_decides_a_view_with_no_controls_leg_only_when_declared(tmp_path, capsys):
    """A view whose roster gives it NO controls leg (llama8b_fullconst) is decided on its static reclassify alone — but
    only under an explicit `--no-controls-leg <reason>`, which the record carries; the flag is refused beside a controls
    reclassify, and an empty reason is refused (2026-09-14)."""
    w = _world(tmp_path, with_gsm8k=False)
    ok = _guard_record(tmp_path / "ok.json")
    argv = _decide_argv(w, tmp_path / "d.json", "--hop-guard", str(ok))
    i = argv.index("--reclassify", argv.index("--reclassify") + 1)
    static_only = argv[:i] + argv[i + 2:]                       # the controls reclassify dropped
    assert le.main(static_only) == 2 and "controls=<reclassify.json> is required" in capsys.readouterr().err
    assert le.main(static_only + ["--no-controls-leg", "  "]) == 2 and "DECLARED reason" in capsys.readouterr().err
    assert le.main(argv + ["--no-controls-leg", "roster D19"]) == 2 and "a view has a controls leg or it does not" in capsys.readouterr().err
    assert le.main(static_only + ["--no-controls-leg", "roster D19: fullconst serves no controls leg"]) == 0
    rec = json.loads((tmp_path / "d.json").read_text())
    assert rec["controls_leg"] == {"present": False, "declared_absent": "roster D19: fullconst serves no controls leg"}
    assert "controls" in rec["absent_kinds"] and "static" in rec["by_kind"] and "controls" not in rec["by_kind"]
    capsys.readouterr()
    assert le.main(_decide_argv(w, tmp_path / "e.json", "--hop-guard", str(ok))) == 0   # the ordinary two-kind decision is unchanged
    assert json.loads((tmp_path / "e.json").read_text())["controls_leg"] == {"present": True}


def test_ladder_extend_ceiling_report_stops_the_ladder_and_reports_residual_share_and_count(tmp_path, capsys):
    """At 16,384 nothing escalates; the residual SHARE and COUNT are reported per pair and the ladder stops."""
    w = _world(tmp_path, with_gsm8k=False)
    # the 16384 rungs: static with the same 7 at-cap items (share 0.02 > 1% -> exit_ceiling); controls cellA with 60
    # coherent at cap (exit_ceiling), cellB loops only (exit_le_1pct_with_residual), cellC clean (exit_zero)
    rows16 = _base_rows(cap=16384)
    _write_static(w["priv"], "run_t0_cap16384", [rows16[0] + rows16[1]], escalation={"cap_from": 8192, "cap_to": 16384, "n_rows": 200})
    _write_controls(w["sampled"] / "controls_cap16384" / "TAG_t0", "cellA", _ctl_items_of("trunc_many"), cap=16384)
    _write_controls(w["sampled"] / "controls_cap16384" / "TAG_t0", "cellB", _ctl_items_of("loops"), cap=16384)
    _write_controls(w["sampled"] / "controls_cap16384" / "TAG_t0", "cellC", _ctl_items_of("clean"), cap=16384)
    out = tmp_path / "reclass16"
    assert ld.main(["reclassify", "--kind", "static", "--root", str(w["priv"]), "--run", "run_t0", "--sibling-run", "run_t1",
                    "--out-dir", str(out / "static"), "--rep-ratio-source", str(w["rep"])]) == 0
    assert ld.main(["reclassify", "--kind", "controls", "--root", str(w["sampled"]), "--tag", "TAG", "--trial", "0",
                    "--out-dir", str(out / "controls"), "--rep-ratio-source", str(w["rep"])]) == 0
    w["recl"] = {"static": out / "static" / "reclassify.json", "controls": out / "controls" / "reclassify.json"}
    capsys.readouterr()
    ceil, events = tmp_path / "ceiling_16384.json", tmp_path / "events.jsonl"
    assert le.main(_decide_argv(w, ceil, cap="16384", next_cap=None, events=events)) == 0, capsys.readouterr().err
    rec = json.loads(ceil.read_text())
    assert rec["at_ceiling"] is True and rec["cap_from"] == 16384 and rec["cap_to"] is None and rec["decision"] == "ceiling_stop"
    assert rec["n_promoted"] == 0 and rec["promoted"] == [] and rec["never_past_ceiling"] is True
    assert rec["n_by_exit_kind"] == {"exit_zero": 1, "exit_le_1pct_with_residual": 1, "exit_ceiling": 2}
    ceiling = {(p["kind"], p.get("attack") or p["cell"]): p for p in rec["exit_ceiling"]}
    assert ceiling[("static", "pasttense")]["residual_count"] == 7 and ceiling[("static", "pasttense")]["residual_share"] == pytest.approx(0.035)
    assert ceiling[("static", "pasttense")]["n_truncated"] == 4 and ceiling[("static", "pasttense")]["exit_kind"] == "exit_ceiling"
    assert ceiling[("controls", "cellA")]["residual_count"] == 60 and ceiling[("controls", "cellA")]["residual_share"] == pytest.approx(0.30)
    assert [p["cell"] for p in rec["exit_le_1pct_with_residual"]] == ["cellB"] and rec["exit_le_1pct_with_residual"][0]["residual_count"] == 3
    assert [p["cell"] for p in rec["exit_zero"]] == ["cellC"]
    assert rec["hop_timeout_guard"] == []   # no rung is generated from a ceiling report; none required
    evs = [json.loads(l) for l in events.read_text().splitlines()]
    assert {e["decision"] for e in evs if "cell" not in e} == {"ceiling_stop"} and all(e["rung"] == 16384 for e in evs)
    assert {e["exit_kind"] for e in evs if "cell" in e} == {"exit_ceiling", "exit_le_1pct_with_residual", "exit_zero"}
    # the ceiling report REFUSES a --cap below the ceiling without --next-cap, and a --next-cap at the ceiling
    assert le.main(_decide_argv(w, tmp_path / "x.json", cap="8192", next_cap=None)) == 2
    assert "is not the ceiling" in capsys.readouterr().err
    assert le.main(_decide_argv(w, tmp_path / "x.json", cap="16384", next_cap="32768")) == 2
    capsys.readouterr()
    # write_ladder folds the ceiling decisions: the static pair reads "16384 + residual 7", exit kinds tallied
    static_events = [e for e in evs if e["kind"] == "static"] + [
        {"kind": "static", "attack": "pasttense", "cell": "none", "rung": 4096, "n_at_cap": 7},
        {"kind": "static", "attack": "pasttense", "cell": "none", "rung": 8192, "n_at_cap": 7},
        {"kind": "static", "attack": "pasttense", "cell": "none", "rung": 16384, "n_at_cap": 7}]
    lad = wl.fold(static_events, "static", [4096, 8192, 16384], None, tag="TAG", trial=0)
    r = lad["cells"]["none"]["pasttense"]
    assert r["n_at_cap"] == {"4096": 7, "8192": 7, "16384": 7} and r["reached_zero_at"] == "16384 + residual 7" and r["residual"] == 7
    assert r["decisions"]["16384"]["exit_kind"] == "exit_ceiling" and r["decisions"]["16384"]["share_at_cap"] == pytest.approx(0.035)
    assert lad["extend_decisions"][0]["decision"] == "ceiling_stop" and lad["extend_decisions"][0]["hop_timeout_guard"] == []
    assert lad["summary"]["exit_kind"]["exit_ceiling"] == 1 and lad["summary"]["residual_at_top"] == 1


def test_write_ladder_folds_decisions_hop_guards_and_filters_by_extend_kind():
    ev = [
        # the controls ladder: cellA escalated at 8192 and reached 16384 with a residual; cellB exited on the 1% rule
        {"kind": "controls", "cell": "cellA", "control": CTL, "rung": 512, "n_at_cap": 40, "root": "/r512"},
        {"kind": "controls", "cell": "cellA", "control": CTL, "rung": 2048, "n_at_cap": 9, "root": "/r2048"},
        {"kind": "controls", "cell": "cellB", "control": CTL, "rung": 512, "n_at_cap": 12, "root": "/r512"},
        {"kind": "controls", "cell": "cellB", "control": CTL, "rung": 2048, "n_at_cap": 4, "root": "/r2048"},
        {"kind": "controls", "cell": "cellA", "control": CTL, "rung": 8192, "n_at_cap": 5, "root": "/r8192"},
        {"kind": "controls", "cell": "cellA", "control": CTL, "rung": 8192, "decision": "escalate", "exit_kind": None,
         "extend_kind": "controls", "n_truncated": 5, "n_loop": 0, "n_repetitive": 0, "n_boundary": 0, "n_at_cap": 5,
         "n_responses": 200, "share_truncated": 0.025, "share_at_cap": 0.025, "rule": "item_filtered_v1", "classifier_version": "item_filtered_v1"},
        {"kind": "controls", "cell": "cellA", "control": CTL, "rung": 16384, "n_at_cap": 2, "root": "/r16384"},
        {"kind": "controls", "cell": "cellA", "control": CTL, "rung": 16384, "decision": "exit_le_1pct", "exit_kind": "exit_le_1pct_with_residual",
         "extend_kind": "controls", "n_truncated": 2, "n_at_cap": 2, "n_responses": 200, "share_truncated": 0.01, "share_at_cap": 0.01},
        {"kind": "controls", "cell": "cellB", "control": CTL, "rung": 8192, "n_at_cap": 3, "root": "/r8192"},
        {"kind": "controls", "cell": "cellB", "control": CTL, "rung": 8192, "decision": "exit_le_1pct", "exit_kind": "exit_le_1pct_with_residual",
         "extend_kind": "controls", "n_truncated": 0, "n_loop": 3, "n_at_cap": 3, "n_responses": 200, "share_truncated": 0.0, "share_at_cap": 0.015},
        {"kind": "controls", "extend_kind": "controls", "rung": 16384, "decision": "rung", "n_pairs_considered": 2, "n_promoted": 1,
         "n_by_exit_kind": {"exit_zero": 0, "exit_le_1pct_with_residual": 1, "exit_ceiling": 0}, "rule": "item_filtered_v1"},
        {"kind": "controls", "rung": 16384, "hop_timeout_guard": {"verdict": "PASS", "phase": "post_standup", "rung": 16384}},
        # the GSM8K readout's events share the file: skipped by the controls fold, folded by the gsm8k fold
        {"kind": "controls", "extend_kind": "gsm8k", "cell": "cellA", "control": "gsm8k", "rung": 8192, "n_at_cap": 4, "root": "/g8192"},
        {"kind": "controls", "extend_kind": "gsm8k", "cell": "cellA", "control": "gsm8k", "rung": 8192, "decision": "escalate", "exit_kind": None,
         "n_truncated": 4, "n_at_cap": 4, "n_responses": 200, "share_truncated": 0.02, "share_at_cap": 0.02},
        {"kind": "controls", "extend_kind": "gsm8k", "rung": 16384, "decision": "rung", "n_pairs_considered": 1, "n_promoted": 1},
        # a static event: the other kind, skipped as before
        {"kind": "static", "attack": "pasttense", "cell": "none", "rung": 4096, "n_at_cap": 0},
    ]
    lad = wl.fold(ev, "controls", [512, 2048, 8192, 16384], None, tag="TAG", trial=0)
    assert lad["extend_kind"] == "controls" and lad["skipped_other_kind_events"] == 4    # 3 gsm8k + 1 static
    a, b = lad["cells"]["cellA"][CTL], lad["cells"]["cellB"][CTL]
    assert a["n_at_cap"] == {"512": 40, "2048": 9, "8192": 5, "16384": 2} and a["reached_zero_at"] == "16384 + residual 2" and a["residual"] == 2
    assert a["decisions"]["8192"]["decision"] == "escalate" and a["decisions"]["8192"]["exit_kind"] is None
    assert a["decisions"]["16384"]["exit_kind"] == "exit_le_1pct_with_residual" and a["decisions"]["16384"]["share_at_cap"] == 0.01
    assert b["reached_zero_at"] == "8192 + residual 3 (exit_le_1pct)" and b["residual"] == 3    # the decision closes the ladder
    assert "gsm8k" not in lad["cells"]["cellA"]
    assert lad["extend_decisions"] == [ev[10]] and lad["hop_timeout_guards"] == [ev[11]]
    assert lad["summary"]["exit_kind"] == {"exit_zero": 0, "exit_le_1pct_with_residual": 2, "exit_ceiling": 0, "promoted": 0, "undecided": 0}
    # the gsm8k fold: its own rungs, its own pairs, the controls' events skipped
    g = wl.fold(ev, "controls", [8192, 16384], None, tag="TAG", trial=0, extend_kind="gsm8k")
    assert g["extend_kind"] == "gsm8k" and list(g["cells"]) == ["cellA"] and list(g["cells"]["cellA"]) == ["gsm8k"]
    gg = g["cells"]["cellA"]["gsm8k"]
    assert gg["n_at_cap"] == {"8192": 4} and gg["reached_zero_at"] == "incomplete"    # escalate with no successor count: the rung is due
    assert g["extend_decisions"] == [ev[14]] and g["summary"]["exit_kind"]["promoted"] == 1 and "hop_timeout_guards" not in g
    # a ladder file of one extend kind never merges another's
    with pytest.raises(SystemExit, match="refusing to merge"):
        wl.fold(ev, "controls", [8192, 16384], g, tag="TAG", trial=0, extend_kind="controls")
    # the CLI flag
    src = (SAMPLED / "write_ladder.py").read_text()
    assert 'ap.add_argument("--extend-kind", default=None, choices=("static", "controls", "gsm8k")' in src


# --------------------------------------------------------------------------- #
# 6. the sbatch blocks (source pins), f22s_check_config's 16k assertions, the smoke's --thinking-on,   #
#    the printer's third block                                                                          #
# --------------------------------------------------------------------------- #

def test_sbatch_dispatches_every_grammar_kind_and_the_two_new_blocks_follow_the_contract():
    src = SBATCH.read_text()
    # every grammar kind has an arm and every arm a grammar entry; every wave's phases share the ONE chain:
    # static_esc16_t<T> (the gate-based static 16,384 rung) and gsm8k_t<T> (the roster-gated GSM8K readout)
    # beside ladder_extend_t<T> / capability_t<T>
    assert set(_dispatched_kinds()) == _grammar_kinds() == {"static_base", "static_esc", "static_esc16", "controls", "ladder_extend",
                                                            "capability", "gsm8k"}
    # --- ladder_extend_t<T>: reclassify -> pre-standup guards -> decide (PINNED decision path) -> promoted pairs only -> ceiling report
    assert 'DECISION="$LADDER_DIR/decision_${F22S_CEILING}.json"; CEILING_REC="$LADDER_DIR/ceiling_${F22S_CEILING}.json"' in src
    assert 'f22s_rep_ratio_source || { RC=1; exit 1; }' in src
    assert 'f22s_reclassify static "$EXT_DIR/reclassify_static" || { RC=1; exit 1; }' in src
    assert 'f22s_reclassify controls "$EXT_DIR/reclassify_controls" || { RC=1; exit 1; }' in src
    assert 'if [ -d "$GSM_ROOT8/cells" ]; then' in src and 'f22s_reclassify gsm8k "$EXT_DIR/reclassify_gsm8k" || { RC=1; exit 1; }' in src
    assert 'f22s_hop_guard "$F22S_CEILING" static "$SERVE_DIR/hop_guard_pre_static.json" || { RC=1; exit 1; }' in src
    assert 'f22s_hop_guard "$F22S_CEILING" controls "$SERVE_DIR/hop_guard_pre_controls.json" || { RC=1; exit 1; }' in src
    assert ('f22s_ladder_extend_decide "$F22S_STATIC_CAP_ESC" "$F22S_CEILING" "$DECISION" "${RECL[@]}" \\\n'
            '      --hop-guard "$SERVE_DIR/hop_guard_pre_static.json" --hop-guard "$SERVE_DIR/hop_guard_pre_controls.json" || { RC=1; exit 1; }') in src
    assert 'if [ "$N_PROMOTED" -eq 0 ]; then' in src and "ZERO promotions" in src
    # the decision precedes every 16384 standup; each standup is followed by the MANDATORY post-standup guard before any driver
    i_decide = src.index('f22s_ladder_extend_decide "$F22S_STATIC_CAP_ESC"')
    for sub, client in (("static_16384", "static"), ("controls_16384", "controls"), ("gsm8k_thinking_on_16384", "controls")):
        i_up = src.index(f'f22_standup attacks "$SERVE_DIR/{sub}"')
        i_guard = src.index(f'"$SERVE_DIR/{sub}/endpoint.json" || {{ RC=1; exit 1; }}')
        assert i_decide < i_up < i_guard, sub
        assert f'f22s_hop_guard "$F22S_CEILING" {client} "$SERVE_DIR/hop_guard_post_' in src[i_up:i_guard + 1]
    assert 'f22_standup attacks "$SERVE_DIR/static_16384" "" families_served:short "$(f22s_rung_window "$F22S_CEILING")" || { RC=1; exit 1; }' in src
    assert 'f22_standup attacks "$SERVE_DIR/controls_16384" "" capability:short "$(f22s_rung_window "$F22S_CEILING")" "" "$CTL_TOP_LOGPROBS" || { RC=1; exit 1; }' in src
    assert 'f22_standup attacks "$SERVE_DIR/gsm8k_thinking_on_16384" "" capability:short "$(f22s_rung_window "$F22S_CEILING")" "" "$CTL_TOP_LOGPROBS" || { RC=1; exit 1; }' in src
    # the drivers: the contract's exact flags, --items from the disposition, into the 16384 roots
    assert '--config "$CFG_ESC16" --attack "$fam" --defense "$cell" --private-root "$PRIVATE_ROOT"' in src
    assert '--max-model-len "$TARGET_L" --concurrency "$F22S_ESC_CONC" --items "$items"' in src
    assert 'f22s_escalation_event "$fam" "$cell" "$PRIVATE_RUN_ESC16"' in src
    assert '--controls-cap "$F22S_CEILING" --escalate-from-root "$(ROOT_OF "$F22S_STATIC_CAP_ESC")" --floor-sample "$F22S_FLOOR_SAMPLE_K" "${items_args[@]}"' in src
    assert '--out-root "$(ROOT_OF "$F22S_CEILING")"' in src
    assert '--controls gsm8k --gsm8k-arm sampled --standup-endpoint "$SERVE_DIR/gsm8k_thinking_on_16384/endpoint.json"' in src
    assert '--controls-cap "$F22S_CEILING" --escalate-from-root "$GSM_ROOT8" --floor-sample "$F22S_FLOOR_SAMPLE_K" --items "${EXT_GSM[$cell]}"' in src
    assert 'GSM_ROOT8="$(f22s_gsm8k_root "$F22S_GSM8K_CAP")"; GSM_ROOT16="$(f22s_gsm8k_root "$F22S_CEILING")"' in src
    # the ceiling report: after every promoted rung, only when nothing failed; the ladder STOPS there
    assert 'f22s_ladder_extend_decide "$F22S_CEILING" "" "$CEILING_REC" "${RECL16[@]}" || { RC=1; exit 1; }' in src
    assert src.index('"$CEILING_REC" "${RECL16[@]}"') > src.index('f22_standup attacks "$SERVE_DIR/gsm8k_thinking_on_16384"')
    assert "the ceiling report is NOT written" in src and "nothing escalates past 16,384" in src
    assert "32768" not in src and src.count("f22s_rung_window \"$F22S_CEILING\"") == 3   # never past the ceiling
    # the folds carry the extended rung lists; the gsm8k ladder folds under its own extend kind
    assert 'f22s_write_ladder_ext static "$LADDER_DIR/static_ladder_t${TRIAL}.json" "$F22S_STATIC_RUNGS_EXT"' in src
    assert 'f22s_write_ladder_ext controls "$LADDER_DIR/ladder.json" "$F22S_CONTROLS_RUNGS_EXT"' in src
    assert 'f22s_write_ladder_ext controls "$LADDER_DIR/gsm8k_ladder.json" "$F22S_GSM8K_RUNGS" "" gsm8k' in src
    assert 'f22s_hop_guard_event static static "$F22S_CEILING"' in src and 'f22s_hop_guard_event controls gsm8k' in src
    # --- capability_t<T>: pre guard -> thinking-ON standup -> positive control -> post guard -> gsm8k sampled only; then MMLU OFF
    i_pre = src.index('f22s_hop_guard "$F22S_GSM8K_CAP" controls "$SERVE_DIR/hop_guard_pre_gsm8k.json" || { RC=1; exit 1; }')
    i_on = src.index('F22_THINKING_ON=1\n  f22_standup attacks "$SERVE_DIR/gsm8k_thinking_on" "" capability:short "$(f22s_gsm8k_window)" "" "$CTL_TOP_LOGPROBS" || { RC=1; exit 1; }')
    i_pc = src.index('f22s_positive_control gsm8k_thinking_on || { RC=1; exit 1; }')
    i_post = src.index('f22s_hop_guard "$F22S_GSM8K_CAP" controls "$SERVE_DIR/hop_guard_post_gsm8k.json" "$SERVE_DIR/gsm8k_thinking_on/endpoint.json" || { RC=1; exit 1; }')
    i_drv = src.index('f22s_launch_cell "gsm8k__${cell}" "$cell" "$MPY"')   # The per-cell launcher (the env prefix rides here)
    # the GSM8K block is gated on the roster's controls list (merge 2026-09-12: a view whose roster omits gsm8k goes straight to
    # MMLU; the `fi` closes that gate), so the thinking-OFF reset is followed by the gate's close before the MMLU comment
    i_off = src.index('f22_standup_stop; F22_THINKING_ON=""\n  fi   # RUN_GSM8K\n  # 2. MMLU')
    # the MMLU standup's window is VIEW-AWARE (f22s_base_window controls: the wave's 2,048 for the published targets; the
    # sdfbase-llama8b per-view controls windows, e.g. shortconst 6,656), not main's literal 2048
    i_mmlu = src.index('f22_standup attacks "$SERVE_DIR/mmlu" "" capability:short "$(f22s_base_window controls)" "" "$CTL_TOP_LOGPROBS" || { RC=1; exit 1; }')
    assert i_pre < i_on < i_pc < i_post < i_drv < i_off < i_mmlu
    assert 'GSM_ROOT="$(f22s_gsm8k_root "$F22S_GSM8K_CAP")"; MMLU_ROOT="$(f22s_mmlu_root)"' in src
    assert ('--defenses-dir "$TT_TWO_SI_DEFENSES_DIR" --out-root "$GSM_ROOT" \\\n'
            '      --controls gsm8k --gsm8k-arm sampled --standup-endpoint "$SERVE_DIR/gsm8k_thinking_on/endpoint.json"') in src
    assert '--out-root "$MMLU_ROOT" \\\n      --controls mmlu --concurrency "$MMLU_CONC" --top-logprobs "$CTL_TOP_LOGPROBS"' in src
    assert 'f22s_assert_gsm8k_items "$GSM_ROOT" "$cell" || { TOTAL_FAIL=$((TOTAL_FAIL + 1)); continue; }' in src
    assert 'f22s_event_json kind=controls extend_kind=gsm8k cell="$cell" control=gsm8k rung="$F22S_GSM8K_CAP"' in src
    assert "--gsm8k-arm greedy" not in src and "capability_greedy" not in src
    # the thinking-ON variable is a plain shell variable (never exported through Slurm), set and cleared around its standups
    code = [l for l in src.splitlines() if not l.lstrip().startswith("#")]
    assert sum(l.count("F22_THINKING_ON=1") for l in code) == 2 and sum(l.count('F22_THINKING_ON=""') for l in code) == 2
    assert "export F22_THINKING_ON" not in src
    assert "--export=" not in src.replace("never --export=", "") and "--mem=0" not in src
    # the helpers the blocks call exist in common_sampled.sh
    common = (SAMPLED / "common_sampled.sh").read_text()
    for fn in ("f22s_ladder_extend_decide", "f22s_decision_field", "f22s_extend_list", "f22s_hop_guard_event", "f22s_gsm8k_root",
               "f22s_mmlu_root", "f22s_write_ladder_ext", "f22s_reclassify", "f22s_hop_guard", "f22s_rep_ratio_source", "f22s_assert_gsm8k_items"):
        assert f"{fn}()" in common, fn
    assert 'f22s_gsm8k_root(){ echo "$WAVE_ROOT/sampled/gsm8k_thinking_on_cap$1/${TAG}_t${TRIAL}"; }' in common   # FIXED root names
    assert 'f22s_mmlu_root(){ echo "$WAVE_ROOT/sampled/mmlu/${TAG}_t${TRIAL}"; }' in common
    assert '--root-pattern "gsm8k_thinking_on_cap{cap}/{tag}_t{trial}"' in common


def test_common_sampled_helpers_run_offline(tmp_path):
    """The shell helpers the new blocks lean on, exercised against synthetic files: the decision field / summary readers,
    the extend list, the hop-guard event, the root names, and f22s_write_ladder_ext's --extend-kind plumbing."""
    dec = tmp_path / "decision_16384.json"
    dec.write_text(json.dumps({"kind": "ladder_extend", "rule": "item_filtered_v1", "decision": "rung", "n_pairs_considered": 3,
                               "n_promoted": 1, "at_ceiling": False, "n_by_exit_kind": {"exit_zero": 1, "exit_le_1pct_with_residual": 1, "exit_ceiling": 0},
                               "by_kind": {"static": {"n_promoted": 1, "n_by_exit_kind": {}}},
                               "promoted": [{"kind": "static", "attack": "pasttense", "cell": "none", "items_path": "/x/items.jsonl"}]}))
    env = dict(MPY=PY, REPO_DIR=str(REPO), TAG="gemma4e4b", TRIAL="0", WAVE_ROOT=str(tmp_path / "wave"), LADDER_DIR=str(tmp_path / "lad"),
               SERVE_DIR=str(tmp_path / "serve"), F22S_FLOOR_SAMPLE_K="200")
    pre = "; ".join(f'{k}="{v}"' for k, v in env.items()) + "; "
    assert _bash(pre + f'f22s_decision_field "{dec}" n_promoted').stdout.strip() == "1"
    assert _bash(pre + f'f22s_decision_field "{dec}" n_by_exit_kind').stdout.strip() == json.dumps({"exit_zero": 1, "exit_le_1pct_with_residual": 1, "exit_ceiling": 0})
    assert _bash(pre + f'f22s_decision_field "{dec}" nope; echo rc=$?').stdout.strip().endswith("rc=1")
    assert "'n_promoted': 1" in _bash(pre + f'f22s_decision_summary "{dec}"').stdout
    assert _bash(pre + f'f22s_extend_list static "{dec}"').stdout.strip() == "pasttense\tnone\t/x/items.jsonl"
    assert _bash(pre + f'f22s_extend_list controls "{dec}"').stdout.strip() == ""
    assert _bash(pre + 'f22s_gsm8k_root 8192; f22s_gsm8k_root 16384; f22s_mmlu_root').stdout.split() == [
        f"{tmp_path}/wave/sampled/gsm8k_thinking_on_cap8192/gemma4e4b_t0", f"{tmp_path}/wave/sampled/gsm8k_thinking_on_cap16384/gemma4e4b_t0",
        f"{tmp_path}/wave/sampled/mmlu/gemma4e4b_t0"]
    # a hop-guard event lands in events.jsonl with kind / extend_kind / rung and the record whole
    rec = tmp_path / "hg.json"
    rec.write_text(json.dumps({"verdict": "PASS", "phase": "post_standup", "rung": 16384, "binding_s": 7200.0}))
    rc = _bash(pre + f'f22s_hop_guard_event controls gsm8k 16384 "{rec}"')
    assert rc.returncode == 0, rc.stdout + rc.stderr
    ev = json.loads((tmp_path / "lad" / "events.jsonl").read_text().strip())
    assert ev["kind"] == "controls" and ev["extend_kind"] == "gsm8k" and ev["rung"] == 16384 and ev["hop_timeout_guard"]["verdict"] == "PASS"
    # the extended fold: the gsm8k ladder under its own extend kind and rungs, the hop guard folded, a count event folded
    (tmp_path / "lad" / "events.jsonl").open("a").write(json.dumps({"kind": "controls", "extend_kind": "gsm8k", "cell": "none",
                                                                   "control": "gsm8k", "rung": 8192, "n_at_cap": 3, "root": "/g"}) + "\n")
    (tmp_path / "serve").mkdir(exist_ok=True)
    rc = _bash(pre + f'f22s_write_ladder_ext controls "{tmp_path}/lad/gsm8k_ladder.json" "8192,16384" "" gsm8k')
    assert rc.returncode == 0, rc.stdout + rc.stderr
    lad = json.loads((tmp_path / "lad" / "gsm8k_ladder.json").read_text())
    assert lad["extend_kind"] == "gsm8k" and lad["rungs"] == [8192, 16384] and lad["cells"]["none"]["gsm8k"]["n_at_cap"] == {"8192": 3}
    assert lad["hop_timeout_guards"][0]["hop_timeout_guard"]["verdict"] == "PASS"
    assert (tmp_path / "serve" / "gsm8k_ladder.json").exists()   # the copy beside the serving dir
    # the controls fold of the same file skips the gsm8k events
    rc = _bash(pre + f'f22s_write_ladder_ext controls "{tmp_path}/lad/ladder.json" "512,2048,8192,16384"')
    assert rc.returncode == 0, rc.stdout + rc.stderr
    lad = json.loads((tmp_path / "lad" / "ladder.json").read_text())
    assert lad["cells"] == {} and lad["skipped_other_kind_events"] == 2 and lad["extend_kind"] == "controls"
    # the escalation event reads the named rung's private_run
    root = tmp_path / "priv" / "run_cap16384" / "pasttense" / "none" / "shard00"
    root.mkdir(parents=True)
    (root / "escalation.json").write_text(json.dumps({"cap_from": 8192, "cap_to": 16384, "n_at_cap_after": 2}))
    rc = _bash(pre + f'PRIVATE_ROOT="{tmp_path}/priv"; PRIVATE_RUN_ESC=run_cap8192; f22s_escalation_event pasttense none run_cap16384')
    assert rc.returncode == 0 and json.loads(rc.stdout.strip())["escalation"]["cap_to"] == 16384
    rc = _bash(pre + f'PRIVATE_ROOT="{tmp_path}/priv"; PRIVATE_RUN_ESC=run_cap8192; f22s_escalation_event pasttense none')
    assert rc.returncode == 3 and "not found" in json.loads(rc.stdout.strip())["problem"]   # the default run is the 8192 one


def test_check_config_asserts_the_16k_rung_config_chains_from_the_8192_rung(tmp_path):
    """f22s_check_config: the _cap16384 config must be the wave config as data except the permitted keys, carry
    static_max_new_tokens 16384 == sampled_decode.static_cap, private_run <base>_cap16384 and escalated_from = the 8192
    rung (never the base); the base config's readouts.gsm8k must say thinking on @ 8192."""
    import yaml
    ok = _bash(f'f22s_resolve_config gemma4e4b 0 >/dev/null; TRIAL=0; f22s_check_config "{PY}"')
    assert ok.returncode == 0 and "sampled configs OK" in ok.stdout and "gsm8k readout on@8192" in ok.stdout, ok.stdout + ok.stderr
    cfg16 = yaml.safe_load((REPO / "configs/sdfbase_full22_gemma4e4b_sampled_t0_cap16384.yaml").read_text())
    assert cfg16["attacks"]["static_max_new_tokens"] == 16384 and cfg16["sampled_decode"]["static_cap"] == 16384
    assert cfg16["sampled_decode"]["escalated_from"] == {"kind": "static", "cap": 8192, "private_run": "sdfbase_full22_gemma4e4b_sampled_t0_cap8192"}

    def refused(mutate, needle):
        c = yaml.safe_load((REPO / "configs/sdfbase_full22_gemma4e4b_sampled_t0_cap16384.yaml").read_text())
        mutate(c)
        bad = tmp_path / "bad16.yaml"
        bad.write_text(yaml.safe_dump(c))
        rc = _bash(f'f22s_resolve_config gemma4e4b 0 >/dev/null; TRIAL=0; CFG_ESC16="{bad}"; f22s_check_config "{PY}"; echo reached')
        out = rc.stdout + rc.stderr
        assert "FATAL" in out and "reached" not in rc.stdout and needle in out, (needle, out[-900:])

    def cap8(c): c["attacks"]["static_max_new_tokens"] = 8192
    refused(cap8, "static_max_new_tokens 8192 != 16384")
    def static_cap(c): c["sampled_decode"]["static_cap"] = 8192
    refused(static_cap, "static_cap != attacks.static_max_new_tokens")
    def chain_from_base(c): c["sampled_decode"]["escalated_from"] = {"kind": "static", "cap": 4096, "private_run": "sdfbase_full22_gemma4e4b_sampled_t0"}
    refused(chain_from_base, "AssertionError")
    def run_name(c): c["private_run"] = "sdfbase_full22_gemma4e4b_sampled_t0_cap8192"
    refused(run_name, "sdfbase_full22_gemma4e4b_sampled_t0_cap8192")
    def trial(c): c["sampled_decode"]["trial"] = 1
    refused(trial, "sampled_decode.trial 1 != this phase's trial 0")
    def extra_key(c): c["some_new_key"] = 1
    refused(extra_key, "differs from")
    # the base config's gsm8k readout block is asserted too
    base = yaml.safe_load((REPO / "configs/sdfbase_full22_gemma4e4b_sampled_t0.yaml").read_text())
    base["sampled_decode"]["readouts"]["gsm8k"]["thinking"] = "off"
    badb = tmp_path / "badbase.yaml"
    badb.write_text(yaml.safe_dump(base))
    rc = _bash(f'f22s_resolve_config gemma4e4b 0 >/dev/null; TRIAL=0; CFG="{badb}"; f22s_check_config "{PY}"; echo reached')
    assert "FATAL" in rc.stdout + rc.stderr and "readouts.gsm8k must be thinking on @8192" in rc.stdout + rc.stderr and "reached" not in rc.stdout


def test_smoke_positive_control_thinking_on_declares_on_lens_routes_only():
    """smoke_positive_control.py --thinking-on (the stopped agent's uncommitted seam, pinned): every LENS-route request
    carries proxy.THINKING_ON_KEY (the bare `none` anchor must not — 400 thinking_on_not_lens), through ONE wrapper that
    probe_fn and the greedy legs both use; check 1 requires petri_proxy.thinking_mode == on; OFF is byte-identical."""
    src = (F22 / "smoke_positive_control.py").read_text()
    assert 'ap.add_argument("--thinking-on", action="store_true"' in src
    assert "from sisteer.capeval.serving.proxy import THINKING_ON_KEY, ProxyClientEngine, load_routing_table" in src
    assert "lens_vmodels = {r.virtual_model for r in routes if bs._is_lens(r)}" in src
    # (the wrapper takes the kind-tag headers since recorder-load-check-kinds; probe_fn goes through its labelled bites form)
    assert ('    def chat(body: dict, headers: dict) -> dict:\n'
            '        if args.thinking_on and body.get("model") in lens_vmodels:\n'
            '            body = {**body, THINKING_ON_KEY: True}\n'
            '        return client.chat_completions(body, headers=headers)') in src
    assert "probe_fn(chat_bites, route.virtual_model, twin.virtual_model)" in src and "probe_fn(client.chat_completions" not in src
    assert "evidence.labelled(chat, POSITIVE_CONTROL_KIND), evidence.labelled(chat, BITES_KIND)" in src
    assert src.count("client.chat_completions(") == 1          # every request goes through the wrapper
    assert 'record("proxy_identity_mode", mode == args.require_lens_mode and ident.get("thinking_mode") == want_thinking,' in src
    assert '"thinking_on": bool(args.thinking_on), "thinking_declaration_key": THINKING_ON_KEY if args.thinking_on else None,' in src
    rc = subprocess.run([PY, str(F22 / "smoke_positive_control.py"), "--help"], capture_output=True, text=True)
    assert rc.returncode == 0 and "--thinking-on" in rc.stdout
    # the job side passes it exactly when the standup was thinking-ON (F22_THINKING_ON), on the standup AND the smoke
    common = (F22 / "common.sh").read_text()
    # the smoke's line also names the wave's ENGINE since native-serving-default (--require-engine "$TARGET_ENGINE",
    # lens:stacked by default): the thinking flag still rides exactly when the standup was thinking-ON
    assert '${F22_THINKING_ON:+--thinking-on} \\' in common
    assert '--out "$out" --require-engine "${TARGET_ENGINE:-lens:stacked}" ${F22_THINKING_ON:+--thinking-on} )' in common
    assert 'local want_thinking="off"; [ -n "${F22_THINKING_ON:-}" ] && want_thinking="on"' in common
    assert 'if [ "$ep_think" != "$want_thinking" ] || [ "$px_think" != "$want_thinking" ]; then' in common
    assert 'case "$think_verdict" in certified|trivially_in_support) ;; *)' in common
    from sisteer.capeval.serving import proxy as px
    assert px.THINKING_ON_KEY == "petri_thinking_on"


def test_submit_printer_third_block_and_the_readme_and_design_carry_the_contract(tmp_path, monkeypatch):
    import submit_sampled as ss
    import yaml
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cluster_site import expect, write_site
    monkeypatch.setenv("SISTEER_CONFIG", str(write_site(tmp_path)))      # each cluster's host, directories and QoS
    a, b = expect(tmp_path, "a"), expect(tmp_path, "b")
    roster = yaml.safe_load((REPO / "configs/sdfbase_full22_roster.yaml").read_text())
    p = ss.plan(roster, ["gemma4e4b", "glm46vflash"], {**ss.DEFAULT_CLUSTER}, 9)
    assert [j.name for j in p["ext"]] == ["f22s-gemma4e4b-cap0", "f22s-gemma4e4b-le0", "f22s-gemma4e4b-cap1", "f22s-gemma4e4b-le1",
                                          "f22s-glm46vflash-cap0", "f22s-glm46vflash-le0", "f22s-glm46vflash-cap1", "f22s-glm46vflash-le1"]
    lay = ss.layout(p, {"cluster_a": 16, "cluster_b": 96})
    assert [j.name for j in lay["cluster_b"]["ext"]] == ["f22s-glm46vflash-cap0", "f22s-glm46vflash-le0", "f22s-glm46vflash-cap1", "f22s-glm46vflash-le1"]
    sha = "cd" * 20
    line = ss.sbatch_line(lay["cluster_b"]["ext"][1], sha, {})
    assert line == (f"ssh {b['host']} 'cd {b['eval']}/{sha} && sbatch {b['sbatch']} "
                    f"--time=16:00:00 -J f22s-glm46vflash-le0 "
                    f"scripts/sdfbase/full22/sampled/target_sampled.sbatch glm46vflash ladder_extend_t0 {sha} 9'")
    line = ss.sbatch_line(lay["cluster_a"]["ext"][0], sha, {})
    assert line == (f"ssh {a['host']} 'cd {a['eval']}/{sha} && sbatch {a['sbatch']} --time=12:00:00 "
                    f"-J f22s-gemma4e4b-cap0 scripts/sdfbase/full22/sampled/target_sampled.sbatch gemma4e4b capability_t0 {sha} 9'")
    text = "\n".join(ss.render(p, lay, sha, {}, offline=True))
    i2, i3, iw = text.index("# --- 2. SECOND BLOCK"), text.index("# --- 3. THIRD BLOCK"), text.index("# --- watchers")
    assert i2 < i3 < iw and "8 third-block jobs" in text and "| 3 (after ctl0 + se0) |" in text and "| 3 (after cap0) |" in text
    assert all(text.index(f" -J f22s-{k}-cap{t} ") < text.index(f" -J f22s-{k}-le{t} ") for k in ("gemma4e4b", "glm46vflash") for t in (0, 1))
    assert not any("--dependency" in l for l in text.splitlines() if l.startswith("ssh "))
    # the docs name the contract
    readme = (SAMPLED / "README.md").read_text()
    for needle in ("## Capability readouts", "gsm8k_thinking_on_cap8192/<TAG>_t<T>", "sampled/mmlu/<TAG>_t<T>",
                   "decision_16384.json", "ceiling_16384.json", "exit_le_1pct_with_residual", "no matched greedy arm", "f22s-<key>-cap<T>",
                   "f22s-<key>-le<T>", "hop_timeout_guard.py", "proxy_upstream_timeout_s", "`capability_greedy` phase is refused"):
        assert needle in readme, needle
    design = (SAMPLED / "DESIGN.md").read_text()
    for needle in ("exit_kind", "ladder_extend.py decide --cap 8192 --next-cap 16384", "decision_16384.json", "ceiling_16384.json"):
        assert needle in design, needle


def test_write_ladder_reads_a_count_only_era_pair_at_its_declared_ceiling_and_never_a_decided_one():
    """2026-09-14 (qwen35_4b_it): the controls_t jobs re-folded ladder.json under the 4-rung list over the 2026-09-10 COUNT-ONLY events (the
    pre-gate 3-rung era whose ceiling was 8,192): a legacy pair 512:500 -> 2048:11 -> 8192:3 read 'incomplete' (16,384 never attempted, no exit
    decision) and the pass-2 planner refused both trial units on it. With the era's ceiling DECLARED (`--count-only-ceiling 8192`; the sbatch's
    controls folds pass it), such a pair reads as left at 8,192 with its residual — the planner's leading integer is the final rung. Never
    inferred: a pair with ANY gate decision (a decision-era job that died before its next gate), a pair whose top count sits below the
    ceiling, or a pair with a gap in its rungs stays 'incomplete'; the governing verdict stays 'undecided' (no gate ran); without the flag the
    fold is byte-identical to before."""
    XS = "xstest"
    def counts(cell, *pairs):
        return [{"kind": "controls", "cell": cell, "control": XS, "rung": r, "n_at_cap": n, "root": f"/r{r}"} for r, n in pairs]
    def gate(cell, rung, n):
        nxt = {512: 2048, 2048: 8192, 8192: 16384}[rung]
        return {"kind": "controls", "cell": cell, "control": XS, "rung": rung, "decision": "escalate", "next_cap": nxt, "exit_kind": None,
                "extend_kind": "controls", "n_truncated": n, "n_loop": 0, "n_repetitive": 0, "n_boundary": 0, "n_at_cap": n, "n_responses": 200,
                "share_truncated": n / 200, "share_at_cap": n / 200, "rule": "item_filtered_v1"}
    ev = (counts("legacy", (512, 500), (2048, 11), (8192, 3))                  # the count-only era: no decision anywhere, top = its ceiling
          + counts("below", (512, 40), (2048, 2))                               # count-only, but its top count sits below the era's ceiling
          + counts("gap", (512, 40), (8192, 2))                                 # count-only with a hole: never contiguous
          + counts("decided", (512, 40)) + [gate("decided", 512, 40)]           # the decision era: escalated twice, then the job died before
          + counts("decided", (2048, 9)) + [gate("decided", 2048, 9)]           # its 8,192 gate — the rung IS due
          + counts("decided", (8192, 5)))
    rungs = [512, 2048, 8192, 16384]
    plain = wl.fold(ev, "controls", rungs, None, tag="TAG", trial=0)
    assert {c: plain["cells"][c][XS]["reached_zero_at"] for c in ("legacy", "below", "gap", "decided")} == {c: "incomplete" for c in ("legacy", "below", "gap", "decided")}
    assert plain["count_only_ceiling"] is None and plain["summary"]["incomplete"] == 4 and "count_only_era_ceiling" not in plain["summary"]
    lad = wl.fold(ev, "controls", rungs, None, tag="TAG", trial=0, count_only_ceiling=8192)
    L = lad["cells"]["legacy"][XS]
    assert L["reached_zero_at"] == f"8192 + residual 3 ({wl.COUNT_ONLY_ERA_NOTE})" and L["residual"] == 3
    assert re.match(r"^(\d+)", L["reached_zero_at"]).group(1) == "8192"      # grade_sampled.ladder_final_rungs reads the leading integer
    assert L["left_ladder_at"] == {"status": "undecided"}                    # the governing verdict is the gate's, and there was none
    assert {c: lad["cells"][c][XS]["reached_zero_at"] for c in ("below", "gap", "decided")} == {c: "incomplete" for c in ("below", "gap", "decided")}
    assert lad["cells"]["decided"][XS]["left_ladder_at"] == {"status": "escalating", "from": 2048, "to": 8192}
    assert lad["count_only_ceiling"] == 8192 and lad["summary"]["incomplete"] == 3 and lad["summary"]["count_only_era_ceiling"] == 1
    # a decided era pair at the same ceiling is unchanged by the flag: the decision closes the ladder, not the era rule
    dec = (counts("closed", (512, 40)) + [gate("closed", 512, 40)] + counts("closed", (2048, 9)) + [gate("closed", 2048, 9)]
           + counts("closed", (8192, 3)) + [{**gate("closed", 8192, 3), "decision": "exit_le_1pct", "next_cap": None, "exit_kind": "exit_le_1pct_with_residual"}])
    assert wl.fold(dec, "controls", rungs, None, tag="TAG", trial=0, count_only_ceiling=8192)["cells"]["closed"][XS]["reached_zero_at"] == "8192 + residual 3 (exit_le_1pct)"
    # the CLI flag, and the sbatch's controls folds pass the era's ceiling (the extend fold by its extend kind; static: never)
    src = (SAMPLED / "write_ladder.py").read_text(); sh = (SAMPLED / "common_sampled.sh").read_text()
    assert 'ap.add_argument("--count-only-ceiling", type=int, default=None' in src
    assert "F22S_COUNT_ONLY_CEILING=8192" in sh and '$(f22s_count_only_flag "$1")' in sh and '$(f22s_count_only_flag "$ext")' in sh
    assert '[ "$1" = controls ] && echo "--count-only-ceiling $F22S_COUNT_ONLY_CEILING"' in sh

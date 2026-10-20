"""Review round 2, P1 — the required 16,384-token static rung is reachable.

Before: `static_esc_t<T>` inspected the 4,096 store and generated only the 8,192 rung; nothing evaluated the 8,192
result, consumed the item-filtered escalation decision, selected the committed `_cap16384.yaml` config or called
`f22s_rung_window 16384`; the plan scheduled one static escalation job. A view with > 1 % coherently truncated
static responses at 8,192 reported a censored 8,192 result while the policy required escalation.

Now: a THIRD static phase `static_esc16_t<T>` reads the BOUND decision (`ladder_gate.py static` over the 8,192 store:
ladder_decide's item-filtered 1 % TRUNCATED verdict + the transport approval), regenerates EXACTLY its escalating items
with the `_cap16384.yaml` config (escalate_static.py --items) after the preflight + standup at `f22s_rung_window 16384`
(fullconst: its own 51,200 window), records 16,384 in `static_ladder_t<T>.json`, and refuses to run without the
decision; submit_sampled.py schedules it after the 8,192 rung for every view with the static leg.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SAMPLED = REPO / "scripts" / "sdfbase" / "full22" / "sampled"
sys.path.insert(0, str(SAMPLED))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))

import submit_sampled as ss  # noqa: E402
import write_ladder as wl  # noqa: E402

ROSTER = yaml.safe_load((REPO / "configs/sdfbase_full22_roster.yaml").read_text())
PY = sys.executable
LOOP_RR = "a b c d " * 6
REPETITIVE = "a b c d e f g h a b c d e f g h"
COHERENT = "one two three four five six seven eight nine ten eleven twelve"
PRESET = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
N_GOALS = 100


# --------------------------------------------------------------------------- a sampled static store (the ladder_decide shape)

def _srow(gid, idx, text, fr, *, trial=0, cap=8192):
    block = {"do_sample": True, **PRESET, "cap": cap, "arm": "vendor_recommended", "seed_base": 20260909, "trial": trial}
    return {"goal_id": gid, "sample_idx": idx, "trial": trial, "variant_key": f"{idx}:{trial}", "family": "pasttense",
            "attack_prompt": f"{gid}/{idx}", "response": text, "decode": block, "sampling_seed": 7, "finish_reason": fr,
            "completion_tokens": cap if fr == "length" else 10, "prompt_tokens": 5, "max_tokens": cap, "at_cap": fr == "length"}


def _write_static(root, run, rows, *, escalation=None):
    d = root / run / "pasttense" / "none" / "shard00"
    d.mkdir(parents=True, exist_ok=True)
    (d / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    goals = {}
    for r in rows:
        goals[r["goal_id"]] = goals.get(r["goal_id"], 0) + 1
    (d / "progress.jsonl").write_text("".join(json.dumps({"goal_id": g, "family": "pasttense", "n_variants": n, "outcome": "done"}) + "\n"
                                              for g, n in goals.items()))
    if escalation is not None:
        (d / "escalation.json").write_text(json.dumps(escalation))
    return d


def _rows(*, cap, loops=1, repetitive=1, coherent=3):
    rows = []
    for g in range(N_GOALS):
        for idx in (0, 1):
            i = g * 2 + idx
            if i < loops:
                rows.append(_srow(f"g{g}", idx, LOOP_RR, "length", cap=cap))
            elif i < loops + repetitive:
                rows.append(_srow(f"g{g}", idx, REPETITIVE, "length", cap=cap))
            elif i < loops + repetitive + coherent:
                rows.append(_srow(f"g{g}", idx, COHERENT, "length", cap=cap))
            else:
                rows.append(_srow(f"g{g}", idx, f"fine answer {g} {idx}", "stop", cap=cap))
    return rows


def _endpoint(tmp_path, **rec):
    p = tmp_path / "endpoint.json"; p.write_text(json.dumps({"proxy_url": "http://127.0.0.1:1/v1", **rec})); return p


def _gate(tmp_path, root, run, cap, endpoint):
    argv = ["static", "--private-root", str(root), "--private-run", run, "--attack", "pasttense", "--defense", "none",
            "--cap", str(cap), "--endpoint", str(endpoint), "--out", str(tmp_path / "gate.json"),
            "--decision-out", str(tmp_path / "decision.json"), "--items-out", str(tmp_path / "items.jsonl")]
    r = subprocess.run([PY, str(SAMPLED / "ladder_gate.py"), *argv], capture_output=True, text=True, cwd=REPO)
    return r.returncode, r.stdout.strip(), r.stderr


# --------------------------------------------------------------------------- the bound decision over the 8,192 store

def test_the_static_gate_escalates_the_8192_stores_truncated_items_to_16384_with_the_transports_approval(tmp_path):
    root = tmp_path / "private"
    _write_static(root, "run_t0_cap8192", _rows(cap=8192))       # 1 loop + 1 repetitive + 3 coherent at cap of 200 -> 1.5 % truncated
    rc, line, err = _gate(tmp_path, root, "run_t0_cap8192", 8192, _endpoint(tmp_path, upstream_timeout_s=7200))
    assert rc == 0, err
    assert line == f"ESCALATE 16384 {tmp_path / 'items.jsonl'}"
    g = json.loads((tmp_path / "gate.json").read_text())
    assert g["kind"] == "static" and g["attack"] == "pasttense" and g["cell"] == "none" and g["private_run"] == "run_t0_cap8192"
    assert g["decision"] == "escalate" and g["next_cap"] == 16384 and g["transport"]["ok"] is True
    assert g["counts"]["n_at_cap"] == 5 and g["counts"]["n_loop"] == 1 and g["counts"]["n_repetitive"] == 1 and g["counts"]["n_truncated"] == 3
    assert g["n_items_escalate"] == 3 and g["decision_tool"]["ladder"] == [4096, 8192, 16384]
    rows = [json.loads(l) for l in (tmp_path / "items.jsonl").read_text().splitlines()]
    header = rows[0]["__ladder_items__"]
    assert header["kind"] == "static" and header["cap"] == 8192 and header["private_run"] == "run_t0_cap8192" and header["n_escalate"] == 3
    esc = {(r["goal_id"], r["sample_idx"]) for r in rows[1:] if r["escalate"]}
    assert esc == {("g1", 0), ("g1", 1), ("g2", 0)} and all(not r["escalate"] for r in rows[1:] if r["class"] in ("loop", "repetitive"))
    # the transport: the old 600 s proxy default cannot carry a 16,384-token draw (6,144 s needed) -> no climb
    rc, line, _ = _gate(tmp_path, root, "run_t0_cap8192", 8192, _endpoint(tmp_path, upstream_timeout_s=600))
    assert rc == 0 and line == "EXIT transport_refused"
    # the ceiling: a 16,384 store with truncated items is UNRESOLVED, never climbed further
    _write_static(root, "run_t0_cap16384", _rows(cap=16384))
    rc, line, _ = _gate(tmp_path, root, "run_t0_cap16384", 16384, _endpoint(tmp_path, upstream_timeout_s=7200))
    assert rc == 0 and line == "EXIT exit_ceiling"
    # a store at another cap than the rung named is a REFUSAL (exit 2), never a decision
    rc, line, err = _gate(tmp_path, root, "run_t0_cap16384", 8192, _endpoint(tmp_path, upstream_timeout_s=7200))
    assert rc == 2 and line == "" and "REFUSED" in err and "not --cap 8192" in err


def test_only_loops_at_8192_never_reach_16384(tmp_path):
    root = tmp_path / "private"
    _write_static(root, "run_t0_cap8192", _rows(cap=8192, loops=3, repetitive=0, coherent=0))
    rc, line, _ = _gate(tmp_path, root, "run_t0_cap8192", 8192, _endpoint(tmp_path, upstream_timeout_s=7200))
    assert rc == 0 and line == "EXIT exit_le_1pct"
    g = json.loads((tmp_path / "gate.json").read_text())
    assert g["counts"]["n_loop"] == 3 and g["counts"]["n_truncated"] == 0 and g["items"] is None


# --------------------------------------------------------------------------- the plan: every view, after the 8,192 rung

def _all_tags():
    """Every (target, view) in the SAMPLED scope (make_sampled_configs.SAMPLED_VIEWS) — the roster carries views the
    sampled lane never planned (e.g. a half-dose view), and the printer refuses a TAG outside its scope."""
    import make_sampled_configs as msc
    return msc.sampled_tags()


def test_the_plan_carries_the_third_phase_for_every_view_with_the_static_leg_after_the_8192_rung(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cluster_site import write_site
    monkeypatch.setenv("SISTEER_CONFIG", str(write_site(tmp_path)))      # the printer reads each cluster's host and QoS from the site
    tags = _all_tags()
    p = ss.plan(ROSTER, tags, {t: "cluster_a" for t in tags}, 9)
    served = {s["key"] for s in p["served"]}
    assert {"llama8b", "llama8b_it", "llama8b_fullconst", "gemma4e4b", "gemma4e4b_it"} <= served
    by16 = {(j.key, j.phase): j for j in p["esc16"]}
    by_ext = {(j.key, j.phase): j for j in p["ext"]}
    for tag in served:
        key, view = ss.split_target(tag)
        from f22_roster import target_caps, view_legs
        has_static = "static" in view_legs(ROSTER, key, view)
        # the third block follows the lane that owns the target's 16,384 rung (the 2026-09-11 reconciliation with main):
        # a target WITH a caps block (this lane) climbs on the GATE (static_esc16_t<T>); every other target is lane G2's
        # capability_t<T> + ladder_extend_t<T> — never both for one target
        caps_lane = target_caps(ROSTER, key) is not None
        for T in (0, 1):
            se = [j for j in p["esc"] if j.key == tag and j.phase == f"static_esc_t{T}"]
            assert bool(se) == has_static, tag
            j = by16.get((tag, f"static_esc16_t{T}"))
            assert (j is not None) == (has_static and caps_lane), (tag, T)
            assert ((tag, f"ladder_extend_t{T}") in by_ext) == (not caps_lane), (tag, T)
            if j is not None:
                assert j.name == f"f22s-{tag}-s16k{T}" and j.block == "esc16" and j.time == "16:00:00"
                assert j.args == (f"{key} static_esc16_t{T} W 9" + ("" if view == "main" else f' "" {view}'))
    # the fullconst view: its 16,384 rung stands up at ITS OWN window 51,200, the other phases at 45,056
    fc = next(s for s in p["served"] if s["key"] == "llama8b_fullconst")
    from sisteer.capeval.serving.geometry import BUDGET_BY_MODEL as _B
    assert fc["geom"]["static_esc16"] == (51200, _B["llama8b"].seqs_at(51200)) == (51200, 9)
    assert fc["geom"]["static_esc"] == (45056, 10) and fc["geom"]["static_base"] == (45056, 10)
    main = next(s for s in p["served"] if s["key"] == "llama8b")
    assert main["geom"]["static_esc16"] == (18432, _B["llama8b"].seqs_at(18432))
    # the layout carries the third block per cluster, and the rendered plan puts it AFTER the second
    lay = ss.layout(p, {"cluster_a": 16, "cluster_b": 16})
    assert [j.name for j in lay["cluster_a"]["esc16"]] == [j.name for j in p["esc16"]]
    text = "\n".join(ss.render(p, lay, "deadbeef" * 5, {}, True))
    i2, i3 = text.index("# --- 2. SECOND BLOCK"), text.index("# --- 3. THIRD BLOCK")
    assert i2 < i3 and "third-block jobs" in text
    for j in p["esc16"]:
        assert text.index(f" -J {j.name} ") > i3
    for j in p["esc"]:
        assert i2 < text.index(f" -J {j.name} ") < i3


# --------------------------------------------------------------------------- the phase itself

def test_the_sbatch_static_esc16_phase_runs_only_on_the_gates_escalate_with_the_16384_config_and_window():
    src = (SAMPLED / "target_sampled.sbatch").read_text()
    common = (SAMPLED / "common_sampled.sh").read_text()
    i0 = src.index('elif [ "$PHASE_KIND" = static_esc16 ]; then')   # one dispatch chain (test_sampled_extend): an elif arm
    i1 = src.index('elif [ "$PHASE_KIND" = controls ]; then')
    body = src[i0:i1]
    # the bound decision governs the climb. The base run's at-cap count appears ONCE, to tell which pairs the 8,192 phase
    # DREW (its own rule): a pair with none left the ladder at 4,096, has no 8,192 store, and is skipped — never gated on
    # a missing store, never climbed on a count. When no pair reached 8,192 the phase exits 0 BEFORE demanding the
    # previous standup's record (that standup never ran); once one did, a missing record is fatal
    assert 'line=$(f22s_static_gate "$PRIVATE_RUN_ESC" "$fam" "$cell" "$F22S_STATIC_CAP_ESC" "$PREV_EP")' in body
    assert body.count("f22s_at_cap_static") == 1 and 'n=$(f22s_at_cap_static "$PRIVATE_RUN_BASE" "$fam" "$cell")' in body
    assert 'if [ "$n" -eq 0 ]; then' in body and 'REACHED["$fam|$cell"]=1' in body and "-gt 0" not in body
    i_reach, i_none = body.index('n=$(f22s_at_cap_static "$PRIVATE_RUN_BASE"'), body.index('if [ "$NREACHED" -eq 0 ]; then')
    i_prev, i_gate = body.index('PREV_EP=$(f22s_prev_endpoint'), body.index('line=$(f22s_static_gate')
    assert i_reach < i_none < i_prev < i_gate and "RC=0; exit 0" in body[i_none:i_prev]
    assert body[i_prev:i_gate].count('[ -n "${REACHED[$fam|$cell]:-}" ] || continue') == 1
    assert body.count('[ -n "${ITEMS16[$fam|$cell]:-}" ] || continue') == 2   # the driver launch and the record read: the gate's items only
    assert 'REFUSING: the ladder gate at rung $F22S_STATIC_CAP_ESC for $fam/$cell was refused' in body and "RC=1; exit 1" in body
    assert '[ "$arg1" = "$F22S_STATIC_CAP_ESC16" ] ||' in body and 'ITEMS16["$fam|$cell"]="$arg2"' in body
    assert 'if [ "$NEED" -eq 0 ]; then' in body and "no standup" in body
    # the transport approval needs the 8,192 rung's recorded hop, found by name (or F22S_PREV_ENDPOINT), else the phase refuses
    assert 'PREV_EP=$(f22s_prev_endpoint "static_esc_t${TRIAL}" static_esc) || f22_fatal' in body
    # the committed _cap16384 config, the gate's items, the 16,384 window, the load check at the rung's cap, the transport re-check
    assert '--config "$CFG_ESC16" --attack "$fam" --defense "$cell" --private-root "$PRIVATE_ROOT"' in body
    assert '--items "${ITEMS16[$fam|$cell]}"' in body
    assert 'WINDOW16=$(f22s_rung_window 16384)' in body and 'f22_window_check "$SERVE_DIR/static_esc16" "$WINDOW16"' in body
    assert 'F22_LOAD_CHECK_CAP=16384 f22_standup attacks "$SERVE_DIR/static_esc16" "" families_served:short "$WINDOW16"' in body
    i_up = body.index('f22_standup attacks "$SERVE_DIR/static_esc16"')
    i_tr = body.index('f22s_ladder_transport "$SERVE_DIR/static_esc16/endpoint.json" 16384')
    i_pc = body.index("f22s_positive_control static_esc16")
    i_drv = body.index('f22s_launch_cell "esc16__${fam}__${cell}"')
    assert i_up < i_tr < i_pc < i_drv
    # the 16,384 rung's record lands in the same static ladder file
    assert 'f22s_escalation_event "$fam" "$cell" "$PRIVATE_RUN_ESC16"' in body
    assert body.count('f22s_write_ladder static "$LADDER_DIR/static_ladder_t${TRIAL}.json"') == 2
    # the helpers and the config resolution
    assert "f22s_static_gate(){" in common and "f22s_prev_endpoint(){" in common
    assert 'CFG_ESC16="configs/sdfbase_full22_${TAG}_sampled_t${trial}_cap16384.yaml"' in common
    assert 'PRIVATE_RUN_ESC16="${PRIVATE_RUN_BASE}_cap16384"' in common and "F22S_STATIC_CAP_ESC16=16384" in common
    assert "static_esc16_t[01]) PHASE_KIND=static_esc16;;" in common


def _bash(snippet, **env):
    return subprocess.run(["bash", "-c", f'source "{SAMPLED}/common_sampled.sh"; {snippet}'], capture_output=True, text=True,
                          cwd=REPO, env={**os.environ, **env})


def test_the_phase_grammar_and_the_config_resolution_carry_the_16384_rung():
    r = _bash('f22s_parse_phase "static_esc16_t1:pasttense,grok420"; echo "$PHASE_KIND $TRIAL ${FAMS[*]} $PHASE_TAG"')
    assert r.returncode == 0 and r.stdout.strip() == "static_esc16 1 pasttense grok420 static_esc16_t1_pasttense-grok420", r.stdout + r.stderr
    r = _bash('f22s_parse_phase "static_esc16_t0"; echo "$PHASE_KIND $TRIAL ${FAMS[*]} $PHASE_TAG"')
    assert r.stdout.strip() == "static_esc16 0 pasttense codeattack renellm grok420 static_esc16_t0"
    for key, view, tag in (("llama8b", "fullconst", "llama8b_fullconst"), ("qwen35_4b", "main", "qwen35_4b")):
        r = _bash(f'f22s_resolve_config {key} 1 {view} >/dev/null; TRIAL=1; echo "$CFG_ESC16|$PRIVATE_RUN_ESC16"; f22s_check_config "{PY}" >/dev/null && echo CHECK_OK')
        assert r.stdout.strip().splitlines() == [f"configs/sdfbase_full22_{tag}_sampled_t1_cap16384.yaml|sdfbase_full22_{tag}_sampled_t1_cap16384", "CHECK_OK"], r.stdout + r.stderr
        cfg16 = yaml.safe_load((REPO / f"configs/sdfbase_full22_{tag}_sampled_t1_cap16384.yaml").read_text())
        assert cfg16["attacks"]["static_max_new_tokens"] == 16384
        assert cfg16["sampled_decode"]["escalated_from"] == {"kind": "static", "cap": 8192, "private_run": f"sdfbase_full22_{tag}_sampled_t1_cap8192"}
    # the fullconst 16,384 rung window is the view's own 51,200; every other key 18,432
    for key, view_window, want in (("llama8b", "45056", "51200"), ("qwen35_4b", "", "18432")):
        r = subprocess.run(["bash", "-c", f'f22_fatal(){{ echo FATAL "$@" >&2; exit 1; }}; KEY={key}; VIEW_WINDOW={view_window}; '
                            f'eval "$(sed -n "/^f22s_rung_window()/,/^}}/p" {SAMPLED}/common_sampled.sh)"; f22s_rung_window 16384'],
                           capture_output=True, text=True)
        assert r.stdout.strip() == want, (key, r.stdout, r.stderr)


def test_the_static_gate_helper_refuses_without_a_bound_decision(tmp_path):
    """The phase cannot run on a guess: f22s_static_gate echoes the gate's line verbatim and returns 2 on a refusal or on
    anything that is not a bound decision — and the phase exits 1 on that return (pinned above)."""
    stub = tmp_path / "mpy"
    stub.write_text("#!/bin/bash\ncat \"$STUB_OUT\"\nexit \"${STUB_RC:-0}\"\n"); stub.chmod(0o755)
    (tmp_path / "serve").mkdir(); (tmp_path / "ladders").mkdir()
    env = {**os.environ, "MPY": str(stub), "REPO_DIR": str(REPO), "F22S_DIR": str(SAMPLED), "SERVE_DIR": str(tmp_path / "serve"),
           "LADDER_DIR": str(tmp_path / "ladders"), "WAVE_ROOT": str(tmp_path), "PRIVATE_ROOT": str(tmp_path / "private"),
           "TAG": "llama8b_fullconst", "TRIAL": "0"}

    def run(text, rc="0"):
        (tmp_path / "out.txt").write_text(text)
        return subprocess.run(["bash", "-c", f'source "{SAMPLED}/common_sampled.sh"; f22_log(){{ echo "LOG $*" >&2; }}; '
                               f'f22s_static_gate sdfbase_full22_llama8b_fullconst_sampled_t0_cap8192 pasttense none 8192 /ep.json; echo "rc=$?"'],
                              capture_output=True, text=True, cwd=REPO, env={**env, "STUB_OUT": str(tmp_path / "out.txt"), "STUB_RC": rc})
    r = run("ESCALATE 16384 /x/items.jsonl\n")
    assert r.stdout.strip().splitlines() == ["ESCALATE 16384 /x/items.jsonl", "rc=0"], r.stdout + r.stderr
    r = run("EXIT exit_le_1pct\n")
    assert r.stdout.strip().splitlines() == ["EXIT exit_le_1pct", "rc=0"]
    r = run("", rc="2")
    assert r.stdout.strip() == "rc=2" and "REFUSED rc=2" in r.stderr
    r = run("7\n")
    assert r.stdout.strip() == "rc=2" and "something other than a bound decision" in r.stderr
    assert (tmp_path / "ladders" / "gates" / "static_cap8192").is_dir()
    # no previous standup record anywhere -> the phase's transport input is undeterminable (the caller refuses)
    r = subprocess.run(["bash", "-c", f'source "{SAMPLED}/common_sampled.sh"; f22s_prev_endpoint static_esc_t0 static_esc; echo "rc=$?"'],
                       capture_output=True, text=True, cwd=REPO, env=env)
    assert r.stdout.strip() == "rc=1"
    (tmp_path / "sampled/serving/static_esc_t0_llama8b_fullconst_123/static_esc").mkdir(parents=True)
    (tmp_path / "sampled/serving/static_esc_t0_llama8b_fullconst_123/static_esc/endpoint.json").write_text("{}")
    r = subprocess.run(["bash", "-c", f'source "{SAMPLED}/common_sampled.sh"; f22s_prev_endpoint static_esc_t0 static_esc'],
                       capture_output=True, text=True, cwd=REPO, env=env)
    assert r.stdout.strip().endswith("static_esc_t0_llama8b_fullconst_123/static_esc/endpoint.json")
    # the TAG is anchored by the job id: for TAG `llama8b` neither the fullconst view's nor the it view's record is this view's
    prev = ["bash", "-c", f'source "{SAMPLED}/common_sampled.sh"; f22s_prev_endpoint static_esc_t0 static_esc; echo "rc=$?"']
    (tmp_path / "sampled/serving/static_esc_t0_llama8b_it_999/static_esc").mkdir(parents=True)
    (tmp_path / "sampled/serving/static_esc_t0_llama8b_it_999/static_esc/endpoint.json").write_text("{}")
    r = subprocess.run(prev, capture_output=True, text=True, cwd=REPO, env={**env, "TAG": "llama8b"})
    assert r.stdout.strip() == "rc=1", r.stdout
    # ... and a family-subset PHASE_TAG of the main view IS (the subset sits before the TAG)
    (tmp_path / "sampled/serving/static_esc_t0_pasttense-grok420_llama8b_7/static_esc").mkdir(parents=True)
    (tmp_path / "sampled/serving/static_esc_t0_pasttense-grok420_llama8b_7/static_esc/endpoint.json").write_text("{}")
    r = subprocess.run(prev, capture_output=True, text=True, cwd=REPO, env={**env, "TAG": "llama8b"})
    assert r.stdout.strip().splitlines() == [str(tmp_path / "sampled/serving/static_esc_t0_pasttense-grok420_llama8b_7/static_esc/endpoint.json"), "rc=0"]
    # F22S_PREV_ENDPOINT names the record explicitly — and must exist
    r = subprocess.run(prev, capture_output=True, text=True, cwd=REPO, env={**env, "F22S_PREV_ENDPOINT": str(tmp_path / "nowhere.json")})
    assert r.stdout.strip() == "rc=1"
    ep = tmp_path / "sampled/serving/static_esc_t0_llama8b_it_999/static_esc/endpoint.json"
    r = subprocess.run(prev, capture_output=True, text=True, cwd=REPO, env={**env, "F22S_PREV_ENDPOINT": str(ep)})
    assert r.stdout.strip().splitlines() == [str(ep), "rc=0"]


def test_write_ladder_folds_the_static_16384_rung_its_decisions_and_where_each_pair_left(tmp_path):
    assert wl.RUNGS["static"] == [4096, 8192, 16384]
    dec = lambda d, **k: {"decision": d, "next_cap": k.get("next_cap"), "n_truncated": k.get("nt", 0), "n_loop": k.get("nl", 0),
                          "n_repetitive": 0, "n_boundary": 0, "share_truncated": k.get("nt", 0) / 200, "transport_ok": k.get("tr"),
                          "n_items_escalate": k.get("nt", 0) if d == "escalate" else 0, "gate": "/g"}
    events = [
        {"kind": "static", "attack": "pasttense", "cell": "none", "rung": 4096, "n_at_cap": 12},
        {"kind": "static_escalation", "attack": "pasttense", "cell": "none",
         "escalation": {"n_rows": 200, "n_at_cap_before": 12, "n_regenerated": 12, "n_at_cap_after": 5, "cap_from": 4096, "cap_to": 8192}, "problem": None},
        {"kind": "static", "attack": "pasttense", "cell": "none", "rung": 8192, "n_at_cap": 5, **dec("escalate", next_cap=16384, nt=3, nl=2, tr=True)},
        {"kind": "static_escalation", "attack": "pasttense", "cell": "none",
         "escalation": {"n_rows": 200, "n_at_cap_before": 5, "n_regenerated": 3, "n_at_cap_after": 2, "cap_from": 8192, "cap_to": 16384}, "problem": None},
        {"kind": "static", "attack": "pasttense", "cell": "none", "rung": 16384, "n_at_cap": 2, **dec("exit_ceiling", nt=1, nl=1)},
        {"kind": "static", "attack": "grok420", "cell": "none", "rung": 4096, "n_at_cap": 3},
        {"kind": "static_escalation", "attack": "grok420", "cell": "none",
         "escalation": {"n_rows": 200, "n_at_cap_before": 3, "n_regenerated": 3, "n_at_cap_after": 1, "cap_from": 4096, "cap_to": 8192}, "problem": None},
        {"kind": "static", "attack": "grok420", "cell": "none", "rung": 8192, "n_at_cap": 1, **dec("exit_le_1pct", nt=0, nl=1)},
    ]
    lad = wl.fold(events, "static", wl.RUNGS["static"], None, tag="llama8b_fullconst", trial=0)
    assert lad["rungs"] == [4096, 8192, 16384]
    p = lad["cells"]["none"]["pasttense"]
    assert p["n_at_cap"] == {"4096": 12, "8192": 5, "16384": 2} and p["reached_zero_at"] == "16384 + residual 2"
    assert set(p["escalations"]) == {"8192", "16384"} and p["escalation"]["cap_to"] == 16384 and p["escalations"]["8192"]["n_regenerated"] == 12
    assert p["left_ladder_at"] == {"status": "left", "rung": 16384, "exit": "exit_ceiling", "n_truncated": 1, "n_loop": 1,
                                   "n_repetitive": 0, "n_boundary": 0, "unresolved_truncation": True}
    g = lad["cells"]["none"]["grok420"]
    # reached_zero_at closes on the EXIT decision at the rung (formatted "<rung> + residual n (<decision>)"), so the raw
    # count history and the governing verdict agree on where the pair left
    assert g["left_ladder_at"]["exit"] == "exit_le_1pct" and g["left_ladder_at"]["rung"] == 8192 and g["reached_zero_at"] == "8192 + residual 1 (exit_le_1pct)"
    assert lad["summary"]["exits"] == {"exit_le_1pct": 1, "exit_ceiling": 1, "transport_refused": 0, "escalating": 0, "undecided": 0}

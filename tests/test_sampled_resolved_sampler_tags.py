"""The 8192 rung's resolved-sampler record (855239 / 855240) read DIVERGED — "within-class mismatch in client: n=6 at
temperature 0.0 / top_p 1.0 / top_k 0" — because the standup's six greedy LOAD-CHECK probes, fired at the CLIENT cap by design, were counted
as client rows. The standup already tags them at the source (load_check.py: X-Tinycap-Task: probe/load_check + load_check.request_evidence.jsonl
beside load_check.json; the positive control likewise) and the reader already carves a tagged load_check at the client cap — the sampled lane's
reader call (common_sampled.sh f22s_record_resolved_sampler) simply never handed the sidecars over. Now the static read runs the reader in kinds
mode with every <standup dir>/*.request_evidence.jsonl as --probe-tags; the controls read keeps its declared caps. Through the REAL shell helper
and the REAL reader over a synthetic standup; the DIVERGED records of 855239 / 855240 stay as written (a re-read is a NEW record)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, VENV_PY  # noqa: E402
from test_record_resolved_sampler import _GREEDY, _lc_tags, _probe_lines, _sp, _static_row, _tag_row  # noqa: E402

SAMPLED = REPO / "scripts/sdfbase/full22/sampled/common_sampled.sh"
VENDOR = dict(t=1.0, tp=0.95, tk=20, pp=0.0)                                                        # the front door's generation_config preset
SENT = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}


def _standup(tmp_path: Path, *, sidecars: bool, positive_control: bool = False, native_controls: bool = False) -> tuple[Path, Path]:
    """A standup dir as the sampled lane leaves it: vllm_server.log with 8 vendor-preset client draws + 6 greedy load-check probes at the same
    cap (S=6), the load check's request_evidence sidecar (and the positive control's, when asked), and the clients' shard00/responses.jsonl."""
    d = tmp_path / "standup"; d.mkdir(parents=True, exist_ok=True)
    lines = [_sp(i, cap=8192, **VENDOR) for i in range(8)] + _probe_lines(cap=8192, n=6, **_GREEDY)
    if positive_control:
        lines += _probe_lines(cap=64, n=4, **_GREEDY) + _probe_lines(cap=48, n=10, **_GREEDY)      # 4 positive-control legs @64; 10 liveness bites @48
    if native_controls:
        lines += _probe_lines(cap=48, n=2, **_GREEDY)                                               # the NATIVE standup's own controls (its sidecar beside the standup records)
    (d / "vllm_server.log").write_text("\n".join(lines) + "\n")
    if sidecars:
        (d / "load_check.request_evidence.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _lc_tags(n=6, cap=8192)))
        if positive_control:
            # the NATIVE standup's positive controls write beside the standup's records; the SAMPLED lane's own positive control writes one
            # level up, named for the phase (positive_control_<phase>.request_evidence.jsonl); a sibling phase's sidecar is not this standup's
            if native_controls:
                (d / "positive_controls.request_evidence.jsonl").write_text("".join(json.dumps(_tag_row("standup_controls", 48, n=i)) + "\n" for i in range(2)))
            (tmp_path / "positive_control_standup.request_evidence.jsonl").write_text("".join(json.dumps(_tag_row("positive_control", 64, n=i)) + "\n" for i in range(4)))
            (tmp_path / "positive_control_static.request_evidence.jsonl").write_text(json.dumps(_tag_row("positive_control", 64, n=9)) + "\n")
            # the low-dose liveness writer labels its greedy text probes probe/bites and records them beside its record in the standup dir
            (d / "liveness_low_dose_constv2ctl2-full__psr_ce_all.request_evidence.jsonl").write_text("".join(json.dumps(_tag_row("bites", 48, vmodel="petri__si_constv2ctl2_full__psr_ce_all", n=i)) + "\n" for i in range(10)))
    sent = tmp_path / "pasttense" / "constv2ctl2-full__psr_ce_all" / "shard00"; sent.mkdir(parents=True, exist_ok=True)
    (sent / "responses.jsonl").write_text("".join(json.dumps(_static_row(i, cap=8192, sampler=SENT)) + "\n" for i in range(8)))
    return d, sent / "responses.jsonl"


def _helper(d: Path, kind: str, sent: Path, *, spy: str) -> subprocess.CompletedProcess:
    """common_sampled.sh f22s_record_resolved_sampler FOR REAL (the function sourced from the real file; f22_log and the two names it reads
    — REPO_DIR, SPY — provided as the sbatch provides them)."""
    script = f"""
set -u
REPO_DIR={REPO}; SPY={spy}
eval "$(sed -n '/^f22s_record_resolved_sampler(){{/,/^}}/p' {SAMPLED})"
f22_log(){{ echo "$*"; }}
f22s_record_resolved_sampler {d} {kind} {sent}
"""
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=600)


def test_the_static_read_hands_every_request_evidence_sidecar_to_the_reader_as_probe_tags_and_the_controls_read_is_unchanged(tmp_path):
    spy = tmp_path / "spy"; argv_out = tmp_path / "argv.json"                                   # ONE executable: the helper runs "$SPY" as a single word
    spy.write_text("#!" + sys.executable + "\nimport json, sys\nif sys.argv[1:2] == ['-']:\n    sys.argv = sys.argv[1:]; exec(compile(sys.stdin.read(), '<stdin>', 'exec'))\nelse:\n    open(" + repr(str(argv_out)) + ", 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"); spy.chmod(0o755)   # a `-` call is the helper's reference read: run it
    d, sent = _standup(tmp_path, sidecars=True, positive_control=True, native_controls=True)
    r = _helper(d, "static", sent, spy=str(spy))
    assert r.returncode == 0, r.stdout + r.stderr
    argv = json.loads(argv_out.read_text().splitlines()[-1])
    assert argv[:3] == ["scripts/sdfbase/record_resolved_sampler.py", "--vllm-log", str(d / "vllm_server.log")]
    assert argv[argv.index("--sent-static") + 1] == str(sent)
    i = argv.index("--probe-expect"); assert argv[i + 1] == "kinds"
    tags = [argv[j + 1] for j, a in enumerate(argv) if a == "--probe-tags"]
    assert tags == [str(d / "liveness_low_dose_constv2ctl2-full__psr_ce_all.request_evidence.jsonl"), str(d / "load_check.request_evidence.jsonl"),
                    str(d / "positive_controls.request_evidence.jsonl"), str(tmp_path / "positive_control_standup.request_evidence.jsonl")]   # the standup's sidecars (sorted) + THIS phase's positive control; the sibling phase's is not passed
    assert "--probe-cap" not in argv and "--client-cap" not in argv                                              # the static read declares no caps (the tags classify)
    assert [argv[j + 1] for j, a in enumerate(argv) if a == "--expect"] == ["temperature=1.0", "top_p=0.95", "top_k=20", "min_p=0.0", "presence_penalty=0.0", "repetition_penalty=1.0"]   # the client reference: every sampler key the sent rows record
    assert argv.index("--probe-expect") < argv.index("--expect") < argv.index("--probe-tags")
    # the controls read: its declared caps as before, no kinds mode, no tags
    r = _helper(d, "controls", sent, spy=str(spy))
    argv = json.loads(argv_out.read_text().splitlines()[-1])
    assert "--probe-expect" not in argv and "--probe-tags" not in argv
    assert argv[argv.index("--sent-controls") + 1] == str(sent) and argv[argv.index("--client-cap") + 1] == "512" and argv.count("--probe-cap") == 3
    # a standup that left no sidecar (a frozen sha before the harness kinds): the static read is exactly the old one
    d2, sent2 = _standup(tmp_path / "old", sidecars=False)
    _helper(d2, "static", sent2, spy=str(spy))
    argv = json.loads(argv_out.read_text().splitlines()[-1])
    assert "--probe-expect" not in argv and "--probe-tags" not in argv and "--expect" not in argv and argv[argv.index("--sent-static") + 1] == str(sent2)
    src = SAMPLED.read_text()
    assert "no client reference in the sent rows — reading without the tags" in src                           # sidecars without a reference: the old read, said so
    src = SAMPLED.read_text()
    assert 'for s in "$out"/*.request_evidence.jsonl; do [ -f "$s" ] && tags+=(--probe-tags "$s"); done' in src
    assert 'args+=(--probe-expect kinds $ref "${tags[@]}")' in src and 'if [ ${#tags[@]} -gt 0 ]; then' in src and 'if [ "$kind" = static ]; then' in src
    assert 's="$(dirname "$out")/positive_control_$(basename "$out").request_evidence.jsonl"; [ -f "$s" ] && tags+=(--probe-tags "$s")' in src


def test_through_the_real_reader_the_tagged_load_check_is_carved_from_the_client_class_and_the_untagged_reading_is_the_false_diverged(tmp_path):
    d, sent = _standup(tmp_path, sidecars=True, positive_control=True)
    r = _helper(d, "static", sent, spy=VENV_PY)
    assert r.returncode == 0 and "resolved sampler" in r.stdout, r.stdout + r.stderr
    rec = json.loads((d / "resolved_sampler.json").read_text())
    assert rec["verdict"] == "AGREES", rec["why"]
    assert rec["classes"]["client"]["verdict"] == "AGREES" and rec["classes"]["client"]["n"] == 8 == rec["n_client_requests"]
    kinds = rec["classes"]["probe"]["kinds"]
    lc = kinds["load_check"]
    assert lc["verdict"] == "AGREES" and lc["n"] == 6 and lc["n_tagged"] == 6 and lc["basis"] == "tag" and lc["reference_kind"] == "greedy"
    assert {k: (v["verdict"], v["n"], v["n_tagged"]) for k, v in kinds.items() if v["n"]} == {
        "load_check": ("AGREES", 6, 6), "positive_control": ("AGREES", 4, 4), "bites": ("AGREES", 10, 10)}   # every harness kind by its tag; the native standup's own controls share bites' cap and are the reader's own fixture elsewhere
    assert rec["n_carved_from_client"] == {"load_check": 6} and rec["n_unclassified_requests"] == 0
    assert "DIVERGED" not in r.stdout and "AGREES" in r.stdout
    # the SAME standup read the old way (no sidecar handed over): the six greedy probes land in the client class — the false DIVERGED of 855239 / 855240
    d2, sent2 = _standup(tmp_path / "old", sidecars=False)
    r2 = _helper(d2, "static", sent2, spy=VENV_PY)
    rec2 = json.loads((d2 / "resolved_sampler.json").read_text())
    assert rec2["verdict"] == "DIVERGED" and rec2["classes"]["client"]["verdict"] == "DIVERGED" and rec2["classes"]["client"]["n"] == 14
    assert "within-class mismatch in client" in r2.stdout

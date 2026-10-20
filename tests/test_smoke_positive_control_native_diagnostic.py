"""smoke_positive_control.py under the NATIVE engine: the greedy
bit-identity check `determinism` is DIAGNOSTIC — recorded in full (status DIAGNOSTIC, `gate: false`, its would-be
reading as `outcome`, the nondeterministic prompts per leg), never gating `all_pass` / the exit status; under a LENS
engine the same run FAILS exactly as before, with the entry's shape byte-identical (no gate / outcome / note keys);
an engine-NEUTRAL failure (greedy_divergence, steering_bites) still fails under native. Driven END-TO-END through
main() over a fake proxy transport (ProxyClientEngine._http_json, the injectable seam), real routing tables, the real
salt-isolation / scale-0-twin / steering-bites helpers.

Why: the native multi-LoRA kernels are measured NOT batch-invariant (a greedy near-tie forks with batch composition),
production decodes by sampling, and admission is functional (multilora_admission rule functional-2026-09-13,
numerics.gate false; user ruling 2026-09-13) — 502002's standup passed every steer-is-live check and every core
control and was refused on 1 of 8 steered greedy prompts repeating non-identically.

Plus the smoke sbatch's FAIL detail (pc_fail_detail / positive_control, RUN under bash): the FAILED controls first,
each with its whole payload, then the DIAGNOSTIC ones, the passing names and the record path — never a truncated log
tail (502002's detail was `tail -5 | head -c 600` of the log and ended mid-JSON inside a PASS entry)."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
F22 = REPO / "scripts" / "sdfbase" / "full22"
for _p in (REPO / "src", REPO / "scripts", F22):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import smoke_positive_control as spc  # noqa: E402
from sisteer.capeval.serving import proxy as px  # noqa: E402
from sisteer.capeval.serving.proxy import RouteEntry  # noqa: E402
from sisteer.eval.attacks.served_target import PROBE_PROMPTS  # noqa: E402

CELL = "constv2-mix-full__lora_r4_ce"          # the smoke's steered cell (502002)
HCELL = "si_constv2_mix_full__lora_r4_ce"       # its harness id (registry_adapter.harness_cell_id)
STEERED_VM, ANCHOR_VM = f"petri__{HCELL}", "petri__none"
NONDET_PROMPT = spc.EXTRA_PROMPTS[2]            # the one prompt that forked on 502002's steered leg
NATIVE, LENS = "lora_native", "lens:stacked"
HASHES = {ANCHOR_VM: "h_none", STEERED_VM: "h_steered"}


def _routes(engine: str) -> tuple[RouteEntry, ...]:
    """The bare anchor, the PATH-MATCHED scale-0 twin (none__lora0 on a native table, none__lens0 on a lens one) and
    the steered cell on that path — the three legs the control drives."""
    path, suffix = ("lora", "__lora0") if engine == NATIVE else ("lens", "__lens0")
    return (RouteEntry(virtual_model=ANCHOR_VM, cell_id="none", cell_class="control",
                       serving_identity="proxy_vllm+passthrough+interleaved", intervention_hash="h_none"),
            RouteEntry(virtual_model=f"petri__none{suffix}", cell_id=f"none{suffix}", cell_class="control",
                       serving_identity=f"proxy_vllm+{path}+interleaved", intervention_hash="h_zero", control_of="none"),
            RouteEntry(virtual_model=STEERED_VM, cell_id=HCELL, cell_class="object",
                       serving_identity=f"proxy_vllm+{path}+interleaved", intervention_hash="h_steered"))


def _table(tmp_path: Path, engine: str) -> Path:
    p = tmp_path / f"routing_table_{engine.replace(':', '_')}.json"
    p.write_text(json.dumps({"version": 1, "base_model": "org/model", "serving_mode": "interleaved",
                             "exact_replay_strategy": None, "base_revision": "r",
                             "routes": [asdict(r) for r in _routes(engine)], "prompted_prefixes": []}))
    return p


class FakeProxy:
    """The proxy's two endpoints over ProxyClientEngine._http_json: /models attests the engine; /chat/completions
    answers greedily and deterministically per (vmodel, prompt) — except that the STEERED leg's SECOND repeat of
    every prompt in ``nondet_steered`` forks (502002's shape: one greedy near-tie flipping under batch composition),
    and ``inert`` makes the steered leg answer exactly like the base (a silently no-op'd object)."""

    def __init__(self, engine: str, *, nondet_steered=(), inert: bool = False):
        self.engine, self.nondet, self.inert = engine, tuple(nondet_steered), inert
        self.calls: dict = {}
        self.ident = ({"engine": {"kind": "lora_native", "adapters": {"none__lora0": {}, HCELL: {}}}, "thinking_mode": "off"}
                      if engine == NATIVE else {"lens_reduction_mode": "stacked", "thinking_mode": "off"})

    def http(self, url: str, payload, headers=None):
        if url.endswith("/models"):
            return {"data": [{"id": r.virtual_model, "owned_by": "petri"} for r in _routes(self.engine)],
                    "petri_proxy": self.ident}
        assert url.endswith("/chat/completions"), url
        assert payload["temperature"] == 0.0 and payload["seed"] == 0, payload
        model, prompt = payload["model"], payload["messages"][-1]["content"]
        n = self.calls[(model, prompt)] = self.calls.get((model, prompt), 0) + 1
        steered = model == STEERED_VM
        text = f"S:{prompt}" if (steered and not self.inert) else f"B:{prompt}"
        if steered and prompt in self.nondet and n == 2:
            text = f"S':{prompt}"                         # the second greedy repeat forks
        return {"choices": [{"message": {"content": text}}], "petri_intervention_hash": HASHES.get(model, "h_zero")}


def _run(tmp_path, monkeypatch, capsys, engine: str, **fake_kw):
    fake = FakeProxy(engine, **fake_kw)
    monkeypatch.setattr(px.ProxyClientEngine, "_http_json",
                        lambda self, url, payload, headers=None: fake.http(url, payload, headers))
    out = tmp_path / f"positive_control_{engine.replace(':', '_')}_{len(fake_kw)}_{fake.inert}.json"
    rc = spc.main(["--proxy-url", "http://fake/v1", "--routing-table", str(_table(tmp_path, engine)),
                   "--cell", CELL, "--out", str(out), "--require-engine", engine])
    rec = json.loads(out.read_text())
    by = {c["check"]: c for c in rec["checks"]}
    return rc, rec, by, capsys.readouterr().out


ALL_CHECKS = ("proxy_identity_mode", "salt_isolation", "determinism", "noise_floor", "steering_bites",
              "intervention_hash_distinct", "greedy_divergence")
LENS_DETERMINISM_KEYS = {"check", "status", "n_prompts", "repeats_per_leg", "nondeterministic_prompts"}


def test_the_dispatch_is_by_engine_name_and_the_vocabulary_is_the_admissions():
    assert spc.DIAGNOSTIC_STATUS == "DIAGNOSTIC"
    assert spc.NATIVE_DIAGNOSTIC_CHECKS == ("determinism",) == spc.diagnostic_checks_for(NATIVE)
    assert spc.diagnostic_checks_for(LENS) == () == spc.diagnostic_checks_for("lens:per_row")
    assert "functional-2026-09-13" in spc.DIAGNOSTIC_NOTES["determinism"] and "never gating" in spc.DIAGNOSTIC_NOTES["determinism"]
    assert set(spc.DIAGNOSTIC_NOTES) == set(spc.NATIVE_DIAGNOSTIC_CHECKS)


def test_native_a_nondeterministic_steered_leg_is_recorded_diagnostic_and_all_pass_holds(tmp_path, monkeypatch, capsys):
    rc, rec, by, out = _run(tmp_path, monkeypatch, capsys, NATIVE, nondet_steered=(NONDET_PROMPT,))
    assert rc == 0 and rec["all_pass"] is True and rec["require_engine"] == NATIVE and rec["engine_problem"] is None
    assert [c["check"] for c in rec["checks"]] == list(ALL_CHECKS)
    det = by["determinism"]
    assert det["status"] == "DIAGNOSTIC" and det["gate"] is False and det["outcome"] == "FAIL"
    # the SAME payload the gate recorded: the nondeterministic prompts per leg, in full
    assert det["n_prompts"] == len(PROBE_PROMPTS) + len(spc.EXTRA_PROMPTS) == 8 and det["repeats_per_leg"] == 2
    assert det["nondeterministic_prompts"] == {"steered": [NONDET_PROMPT], "anchor": [], "twin": []}
    assert det["note"] == spc.DIAGNOSTIC_NOTES["determinism"]
    for name in ALL_CHECKS:
        if name != "determinism":
            assert by[name]["status"] == "PASS" and "gate" not in by[name] and "outcome" not in by[name], name
    # the rows keep the fork legible: the steered repeat differs on exactly that prompt
    flipped = [r["prompt"] for r in rec["rows"] if r["steered_deterministic"] is False]
    assert flipped == [NONDET_PROMPT] and all(r["anchor_deterministic"] and r["twin_deterministic"] for r in rec["rows"])
    # the log line and the summary name it as diagnostic
    assert "[positive-control] DIAGNOSTIC determinism (outcome FAIL, gate=false): " in out
    assert "[positive-control] FAIL determinism" not in out
    assert out.rstrip().splitlines()[-1].startswith("[positive-control] ALL PASS (diagnostic, not gating: determinism=FAIL) -> ")


def test_native_a_deterministic_run_is_still_recorded_diagnostic_not_pass(tmp_path, monkeypatch, capsys):
    """The gate rule depends on the ENGINE, never on the reading: a native run whose legs all repeat byte-identically
    records determinism as DIAGNOSTIC with outcome PASS — a reader knows from the engine alone what gated."""
    rc, rec, by, out = _run(tmp_path, monkeypatch, capsys, NATIVE)
    assert rc == 0 and rec["all_pass"] is True
    det = by["determinism"]
    assert det["status"] == "DIAGNOSTIC" and det["gate"] is False and det["outcome"] == "PASS"
    assert det["nondeterministic_prompts"] == {"steered": [], "anchor": [], "twin": []}
    assert "[positive-control] DIAGNOSTIC determinism (outcome PASS, gate=false): " in out
    assert out.rstrip().splitlines()[-1].startswith("[positive-control] ALL PASS (diagnostic, not gating: determinism=PASS) -> ")


def test_lens_the_same_nondeterministic_run_fails_exactly_as_before(tmp_path, monkeypatch, capsys):
    rc, rec, by, out = _run(tmp_path, monkeypatch, capsys, LENS, nondet_steered=(NONDET_PROMPT,))
    assert rc == 1 and rec["all_pass"] is False and rec["require_engine"] == LENS and rec["is_lens"] is True
    det = by["determinism"]
    assert det["status"] == "FAIL" and set(det) == LENS_DETERMINISM_KEYS          # byte-identical shape: no gate / outcome / note
    assert det["nondeterministic_prompts"] == {"steered": [NONDET_PROMPT], "anchor": [], "twin": []}
    assert all(by[n]["status"] == "PASS" for n in ALL_CHECKS if n != "determinism")
    assert "[positive-control] FAIL determinism: " in out and "DIAGNOSTIC" not in out
    assert out.rstrip().splitlines()[-1].startswith("[positive-control] FAILED -> ")


def test_lens_a_deterministic_run_passes_with_the_summary_line_unchanged(tmp_path, monkeypatch, capsys):
    rc, rec, by, out = _run(tmp_path, monkeypatch, capsys, LENS)
    assert rc == 0 and rec["all_pass"] is True
    assert by["determinism"]["status"] == "PASS" and set(by["determinism"]) == LENS_DETERMINISM_KEYS
    assert all("gate" not in c and "outcome" not in c and "note" not in c for c in rec["checks"])
    assert out.rstrip().splitlines()[-1].startswith("[positive-control] ALL PASS -> ")
    assert "diagnostic" not in out


def test_native_an_engine_neutral_failure_still_fails(tmp_path, monkeypatch, capsys):
    """A silently no-op'd object under the native engine: steering_bites and greedy_divergence FAIL, all_pass is
    False, the exit status is 1 — determinism is diagnostic beside them, not a licence for the others."""
    rc, rec, by, out = _run(tmp_path, monkeypatch, capsys, NATIVE, inert=True)
    assert rc == 1 and rec["all_pass"] is False
    assert by["greedy_divergence"]["status"] == "FAIL" and by["greedy_divergence"]["n_steered_ne_anchor"] == 0
    assert by["steering_bites"]["status"] == "FAIL" and "no difference" in by["steering_bites"]["error"].lower().replace("produce no", "no")
    assert by["determinism"]["status"] == "DIAGNOSTIC" and by["determinism"]["outcome"] == "PASS"
    assert by["intervention_hash_distinct"]["status"] == "PASS" and by["noise_floor"]["status"] == "PASS"
    assert out.rstrip().splitlines()[-1].startswith("[positive-control] FAILED (diagnostic, not gating: determinism=PASS) -> ")


# --- the smoke sbatch's FAIL detail, RUN under bash --------------------------------------------------------------

def _sbatch_fn(name: str) -> str:
    sh = (F22 / "smoke_full22.sbatch").read_text()
    start = sh.index(f"{name}(){{")
    return sh[start:sh.index("\n}\n", start) + 3]


def _bash(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", "set -uo pipefail\n" + script], capture_output=True, text=True)


def test_pc_fail_detail_names_the_failed_controls_first_untruncated_then_the_diagnostics_and_the_record(tmp_path):
    fn = _sbatch_fn("pc_fail_detail")
    assert 'DIAGNOSTIC = "DIAGNOSTIC"' in fn and spc.DIAGNOSTIC_STATUS == "DIAGNOSTIC"   # one vocabulary
    code = "\n".join(l for l in fn.splitlines() if not l.strip().startswith("#"))       # the rationale comment names the old cut
    assert "head -c" not in code and "tail -5" not in code
    big = "x" * 3000                                     # a payload longer than the old 600-byte cut
    rec = tmp_path / "positive_control_static.json"
    log = tmp_path / "positive_control_static.log"
    log.write_text("\n".join(f"[positive-control] PASS check_{i}: {{}}" for i in range(30)) + "\n[positive-control] FAILED -> x\n")
    # 502002's lens-era shape: every check PASS but determinism FAIL
    rec.write_text(json.dumps({"all_pass": False, "checks": [
        {"check": "proxy_identity_mode", "status": "PASS", "engine": "lora_native"},
        {"check": "salt_isolation", "status": "PASS", "n_routes": 3},
        {"check": "determinism", "status": "FAIL", "n_prompts": 8, "repeats_per_leg": 2,
         "nondeterministic_prompts": {"steered": [NONDET_PROMPT], "anchor": [], "twin": []}},
        {"check": "noise_floor", "status": "PASS"}, {"check": "steering_bites", "status": "PASS", "diag": {"arms": {"x": big}}},
        {"check": "intervention_hash_distinct", "status": "PASS"}, {"check": "greedy_divergence", "status": "PASS"}]}))
    r = _bash(f"MPY={sys.executable}\n{fn}\npc_fail_detail {rec} {log}\n")
    assert r.returncode == 0 and "unbound variable" not in r.stderr, (r.stdout, r.stderr)
    line = r.stdout.rstrip("\n")
    assert "\n" not in line
    assert line.startswith('FAILED determinism={"n_prompts": 8, "repeats_per_leg": 2, "nondeterministic_prompts": {"steered": ['), line
    assert json.dumps(NONDET_PROMPT) in line and " | PASS proxy_identity_mode,salt_isolation,noise_floor,steering_bites,intervention_hash_distinct,greedy_divergence | " in line
    assert line.endswith(f"record={rec}") and "DIAGNOSTIC" not in line
    # a native-shape record: an engine-neutral failure first, the diagnostic determinism after it, both WHOLE
    rec.write_text(json.dumps({"all_pass": False, "checks": [
        {"check": "proxy_identity_mode", "status": "PASS"},
        {"check": "determinism", "status": "DIAGNOSTIC", "gate": False, "outcome": "FAIL", "n_prompts": 8,
         "nondeterministic_prompts": {"steered": [NONDET_PROMPT], "anchor": [], "twin": []}, "note": "n"},
        {"check": "greedy_divergence", "status": "FAIL", "n_prompts": 8, "n_steered_ne_anchor": 0, "texts": big}]}))
    line = _bash(f"MPY={sys.executable}\n{fn}\npc_fail_detail {rec} {log}\n").stdout.rstrip("\n")
    assert line.startswith('FAILED greedy_divergence={"n_prompts": 8, "n_steered_ne_anchor": 0, "texts": "' + big + '"} | '
                           'DIAGNOSTIC (recorded, not gating) determinism={"gate": false, "outcome": "FAIL", "n_prompts": 8, '), line
    assert line.index("FAILED greedy_divergence") < line.index("DIAGNOSTIC (recorded") < line.index(" | PASS proxy_identity_mode | record=")
    # the driver exited non-zero with every check PASS (a crash after the record): said so, never a phantom failure
    rec.write_text(json.dumps({"all_pass": True, "checks": [{"check": "greedy_divergence", "status": "PASS"}]}))
    line = _bash(f"MPY={sys.executable}\n{fn}\npc_fail_detail {rec} {log}\n").stdout.rstrip("\n")
    assert line.startswith("FAILED no check FAILED, the driver exited non-zero (all_pass=True) — read the log | PASS greedy_divergence | record=")
    # no record at all (the driver died before writing it): the log's last 20 lines, whole
    line = _bash(f"MPY={sys.executable}\n{fn}\npc_fail_detail {tmp_path}/absent.json {log}\n").stdout.rstrip("\n")
    assert line.startswith(f"no record JSON at {tmp_path}/absent.json (the driver died before writing it); log tail: ")
    assert "check_11:" in line and "check_10:" not in line and line.rstrip().endswith("[positive-control] FAILED -> x")   # 31 log lines, the last 20
    # an unreadable record: named, with the log tail
    rec.write_text("{not json")
    line = _bash(f"MPY={sys.executable}\n{fn}\npc_fail_detail {rec} {log}\n").stdout.rstrip("\n")
    assert line.startswith(f"record {rec} unreadable (") and "log tail: " in line


def test_positive_control_in_the_sbatch_files_the_fail_detail_from_the_record(tmp_path):
    """The whole `positive_control <tag>` function as the smoke runs it: a stub driver that writes 502002's record
    and exits 1 -> `check positive_control_static FAIL <the failed controls first ... record=...>`; exit 0 -> PASS."""
    fns = _sbatch_fn("pc_fail_detail") + _sbatch_fn("positive_control")
    repo = tmp_path / "repo"; (repo / "scripts/sdfbase/full22").mkdir(parents=True)
    stub = repo / "scripts/sdfbase/full22/smoke_positive_control.py"
    stub.write_text("import json,sys\nout=sys.argv[sys.argv.index('--out')+1]\n"
                    "json.dump({'all_pass': False, 'checks': [{'check': 'salt_isolation', 'status': 'PASS'},"
                    " {'check': 'determinism', 'status': 'FAIL', 'n_prompts': 8, 'nondeterministic_prompts': {'steered': ['p'], 'anchor': [], 'twin': []}},"
                    " {'check': 'greedy_divergence', 'status': 'PASS'}]}, open(out, 'w'))\n"
                    "print('[positive-control] FAILED ->', out)\nsys.exit(int(open(sys.argv[0] + '.rc').read()))\n")
    rc_file = stub.parent / "smoke_positive_control.py.rc"
    smoke_root = tmp_path / "smoke"; serve = smoke_root / "serving"; (serve / "drivers").mkdir(parents=True)
    script = (f"MPY={sys.executable}; REPO_DIR={repo}; PROXY_URL=http://127.0.0.1:1/v1; TABLE={tmp_path}/t.json; CELL={CELL}\n"
              f"TARGET_ENGINE=lora_native; SMOKE_ROOT={smoke_root}; SERVE_DIR={serve}\n"
              "check(){ local name=\"$1\" st=\"$2\"; shift 2; printf 'CHECK %s %s %s\\n' \"$name\" \"$st\" \"$*\"; }\n"
              f"{fns}\npositive_control static; echo \"rc=$?\"\n")
    rc_file.write_text("1")
    r = _bash(script)
    assert "unbound variable" not in r.stderr and r.returncode == 0, (r.stdout, r.stderr)
    line = next(l for l in r.stdout.splitlines() if l.startswith("CHECK positive_control_static "))
    assert line.startswith('CHECK positive_control_static FAIL FAILED determinism={"n_prompts": 8, "nondeterministic_prompts": {"steered": ["p"], "anchor": [], "twin": []}} | PASS salt_isolation,greedy_divergence | record=')
    assert line.endswith(f"record={smoke_root}/positive_control_static.json")
    assert (serve / "drivers/positive_control_static.log").read_text().startswith("[positive-control] FAILED -> ")
    rc_file.write_text("0")
    r = _bash(script)
    line = next(l for l in r.stdout.splitlines() if l.startswith("CHECK positive_control_static "))
    assert line.startswith("CHECK positive_control_static PASS salt_isolation=PASS | determinism=FAIL | greedy_divergence=PASS | greedy: ")

"""bcb-bfcl-200-widen S3 consumer (1) / S6 / S7: the lane's launch wrapper
(``scripts/bcb_bfcl_200_widen_launch.py``) — pure composition against the checked-in roster:
refusals (a roster failing its rules, a wave with no runnable cell, an unmet prerequisite named,
a non-runnable ``--cell``), the launch argv (`none` FIRST, the runnable legs, `--sampling
protocol`, `--modes off`, the roster snapshot, the app / suffix naming), the launch-time
snapshot map, the meter invocations in Slice C's (wave, suffix, unit, stage-kind) vocabulary
(parsed by the meter's own parser), the watcher argv, the ladder rerun's two steps
(`--ladder-certify`; `--cell X --ladder-probe-cell X`), the run-id capture. No Modal, no ledger."""
from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import widen_roster as W

REPO = Path(__file__).resolve().parents[1]
SHA = "a" * 40
DERIVED = "constv2-mix-full__lora_r4_ce_minus_ctl"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def Wr():
    return _load("bcb_bfcl_200_widen_launch_under_test", "scripts/bcb_bfcl_200_widen_launch.py")


@pytest.fixture(scope="module")
def meter():
    return _load("bcb_bfcl_200_meter_for_wrapper_test", "scripts/bcb_bfcl_200_meter.py")


ROSTER = W.load_roster()
PINS = W.load_pins()


def test_refusals_name_every_reason_not_to_launch(Wr):
    assert Wr.refusals(ROSTER, PINS, C.MODELS, wave="gemma4e4b_it") == []
    assert Wr.refusals(ROSTER, PINS, C.MODELS, wave="gemma4e4b_main") == []
    # the qwen and nemotron waves wait on their recorded prerequisites (S6e; the parents' copy).
    # State-independent: the shipped roster's `prerequisites_met` moves as each prerequisite is
    # recorded (the Nemotron parents among them), so the unmet face is tested on a copy
    # with the record cleared, and the met face on a copy with it filled.
    unmet = copy.deepcopy(ROSTER)
    for wave in ("qwen35_4b_main", "nemotron3nano4b_main"):
        unmet["waves"][wave]["prerequisites_met"] = []
    (why,) = Wr.refusals(unmet, PINS, C.MODELS, wave="qwen35_4b_main")
    assert "prerequisite NOT met" in why and "S6e" in why and "prerequisites_met" in why
    (why,) = Wr.refusals(unmet, PINS, C.MODELS, wave="nemotron3nano4b_main")
    assert "prerequisite NOT met" in why and "parents copied" in why
    met = copy.deepcopy(ROSTER)
    for wave in ("qwen35_4b_main", "nemotron3nano4b_main"):
        met["waves"][wave]["prerequisites_met"] = list(met["waves"][wave]["prerequisites"])
        assert Wr.refusals(met, PINS, C.MODELS, wave=wave) == []
    # the shipped roster's own state: nemotron's parents are recorded, so it is launchable
    assert Wr.refusals(ROSTER, PINS, C.MODELS, wave="nemotron3nano4b_main") == []
    # a wave with no runnable cell says what is queued and what unblocks it — a SYNTHETIC wave (every
    # entry queued), never a shipped one: the shipped queued waves flip to runnable as objects land
    parked = copy.deepcopy(ROSTER)
    for entry in parked["waves"]["gemma4e4b_it"]["entries"].values():
        if entry.get("status") == "runnable":
            entry.update(status="queued", reason="object not yet fit (synthetic)", unblock="stage the object (synthetic)")
    assert W.validate(parked, PINS, C.MODELS) == []
    (why,) = Wr.refusals(parked, PINS, C.MODELS, wave="gemma4e4b_it")
    assert "no runnable cell" in why and "unblock:" in why and "synthetic" in why
    # since 2026-09-13 the shipped qwen35_4b_it wave is RUNNABLE — the batch-64 LoRA subtraction is staged and pinned — so
    # it launches; the no-runnable-cell refusal is covered by the synthetic `parked` wave above, never by a shipped one
    assert Wr.refusals(ROSTER, PINS, C.MODELS, wave="qwen35_4b_it") == []
    assert Wr.refusals(ROSTER, PINS, C.MODELS, wave="nope")[0].startswith("wave 'nope' is not in the roster")
    # a non-runnable --cell; a roster failing its own rules refuses first
    (why,) = Wr.refusals(ROSTER, PINS, C.MODELS, wave="gemma4e4b_it", cell="safety_refusal__lora_r4")      # a queued cell (ctl_lora is runnable since #107)
    assert "not a runnable cell" in why
    bad = copy.deepcopy(ROSTER)
    bad["waves"]["gemma4e4b_it"]["entries"]["const_lora"]["object"]["sha256"] = "0" * 64
    assert all(p.startswith("roster: ") for p in Wr.refusals(bad, PINS, C.MODELS, wave="gemma4e4b_it"))
    assert W.EXCLUDED_FAMILY not in json.dumps(ROSTER["waves"])


def test_compose_puts_none_first_the_runnable_legs_protocol_sampling_and_the_snapshot(Wr):
    L = Wr.compose(ROSTER, wave="gemma4e4b_it", sha=SHA, roster_sha256="0" * 64, launched_at="2026-09-11T00:00:00Z",
                   stage_kind="smoke", inspect_limit=20, tag="smoke", snapshot_path="/x/snap.json")
    assert (L.app, L.suffix, L.row) == ("tinycap-bcbw-gemma4e4b_it-aaaaaaa", "bcbw-gemma4e4b_it-aaaaaaa-smoke", "gemma4e4b-f22it")
    # round 3: with the invocation's nonce the remote app name is unique per invocation and descriptive —
    # `<base>-<cell|wave>-<hex6>`, within Modal's 64 characters (a long cell is shortened; the nonce is kept)
    named = Wr.compose(ROSTER, wave="gemma4e4b_it", sha=SHA, roster_sha256="0" * 64, launched_at="t", app_token="c0ffee")
    assert named.app == "tinycap-bcbw-gemma4e4b_it-aaaaaaa-wave-c0ffee" and named.env == {C.MODAL_APP_NAME_ENV: named.app}
    assert named.launcher_argv[named.launcher_argv.index("--app-name") + 1] == named.app     # env and --app-name agree
    cell_app = Wr.app_name("nemotron3nano4b_main", SHA, scope="constv2-mix-full__lora_r4_ce_minus_ctl", token="c0ffee")
    assert len(cell_app) <= Wr.MODAL_APP_NAME_MAX == 64 and cell_app.endswith("-c0ffee") and cell_app.startswith("tinycap-bcbw-nemotron3nano4b_main-aaaaaaa-constv2-mix")
    assert Wr.app_name("w", SHA, scope="a b", token="t") == "tinycap-bcbw-w-aaaaaaa-a-b-t"
    assert Wr.app_name("gemma4e4b_it", SHA, scope="x", token="1") != Wr.app_name("gemma4e4b_it", SHA, scope="x", token="2")
    with pytest.raises(SystemExit, match="no room"):
        Wr.app_name("w" * 60, SHA, scope="x", token="t")
    # the roster's runnable set (PSR out by default, the constitution SI in; since 2026-09-14 the batch-64 control LoRA + LoRA subtraction
    # beside the batch-8 constitution LoRA — a mixed set): none, the two prompted cells, three fitted cells
    assert L.cells == tuple(W.runnable_cells(ROSTER, "gemma4e4b_it")) == ("none", "prompted__safety_refusal", "prompted__si_constitution",
                                                                             "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce", DERIVED)
    assert L.controls == ("bigcodebench_200", "bfcl_200") and L.env == {C.MODAL_APP_NAME_ENV: L.app}
    argv = list(L.launcher_argv)
    assert argv[:6] == ["--sha", SHA, "--app-name", L.app, "--pf", "--pass1"]
    assert argv[argv.index("--sampling") + 1] == "protocol" and argv[argv.index("--modes") + 1] == "off"
    # the lane's serving path is COMPOSED on every launch (before this the launcher's entry-point default was trusted
    # and the earlier qwen waves' rows stamp `serving_path unverified`) — the roster reader's ONE constant, native
    assert argv[argv.index("--serving-path") + 1] == "lora_native" == Wr.SERVING_PATH == W.SERVING_PATH == C.SERVING_ENGINE_LORA_NATIVE
    assert L.serving_path_requested == "lora_native" and argv.index("--serving-path") > argv.index("--modes")
    assert argv[argv.index("--cells") + 1] == ",".join(L.cells) and argv[argv.index("--controls") + 1] == "bigcodebench_200,bfcl_200"
    assert argv[argv.index("--out-suffix") + 1] == L.suffix and argv[argv.index("--inspect-limit") + 1] == "20"
    assert argv[argv.index("--roster-snapshot") + 1] == "/x/snap.json" and C.FLAG_CAP_OVERRIDE not in argv
    assert L.modal_argv("/venv/bin/python")[:6] == ["/venv/bin/python", "-m", "modal", "run", "--detach", "scripts/modal/tinycap_served.py"]
    # the snapshot: one block per launched cell, digests recomputing, the fitted cells' generation
    assert set(L.snapshot) == set(L.cells) and all(W.snapshot_problems(b) == [] for b in L.snapshot.values())
    assert L.snapshot["constv2-mix-full__lora_r4_ce"]["object_generation"] == "batch8-provisional"
    assert L.snapshot["none"]["object_generation"] is None and L.snapshot["none"]["object_sha256"] is None
    # without a path the snapshot rides INLINE (parseable by the launcher's reader)
    inline = Wr.compose(ROSTER, wave="gemma4e4b_it", sha=SHA, roster_sha256="0" * 64, launched_at="t")
    doc = W.parse_snapshot_arg(inline.launcher_argv[inline.launcher_argv.index("--roster-snapshot") + 1])
    assert doc == inline.snapshot and "--inspect-limit" not in inline.launcher_argv
    # the qwen main wave: BOTH legs since 2026-09-23 (the live probe PASSED at 80df464); nemotron: BOTH legs since its own
    # live PASS at 8365992 the same day; the gemma main wave: none + the LoRA subtraction (the distilled PSR is out)
    q = Wr.compose(ROSTER, wave="qwen35_4b_main", sha=SHA, roster_sha256="0" * 64, launched_at="t")
    assert q.controls == ("bigcodebench_200", "bfcl_200") and q.cells == ("none", DERIVED) and q.row == "qwen35_4b-f22"
    g = Wr.compose(ROSTER, wave="gemma4e4b_main", sha=SHA, roster_sha256="0" * 64, launched_at="t")
    assert g.cells == tuple(W.runnable_cells(ROSTER, "gemma4e4b_main")) == ("none", DERIVED) and g.row == "gemma4e4b-f22"
    n = Wr.compose(ROSTER, wave="nemotron3nano4b_main", sha=SHA, roster_sha256="0" * 64, launched_at="t")
    assert n.row == "nemotron3nano4b" and n.controls == ("bigcodebench_200", "bfcl_200")
    ll = Wr.compose(ROSTER, wave="llama8b_it", sha=SHA, roster_sha256="0" * 64, launched_at="t")
    assert ll.row == "llama8b-f22it" and ll.controls == ("bigcodebench_200",)                  # BFCL skipped for llama
    with pytest.raises(SystemExit, match="40-hex"):
        Wr.compose(ROSTER, wave="gemma4e4b_it", sha="abc", roster_sha256="0" * 64, launched_at="t")


def test_the_ladder_rerun_is_two_steps_certify_then_per_cell(Wr, tmp_path):
    sel = str(tmp_path / "ladder_selection.json")
    cap = f"{C.inspect_task('bfcl_200')}=16384"
    cert = Wr.compose(ROSTER, wave="gemma4e4b_it", sha=SHA, roster_sha256="0" * 64, launched_at="t", tag="cap16k",
                      cap_override=cap, item_selection=sel, certify=True, snapshot_path="/x/s.json")
    argv = list(cert.launcher_argv)
    assert cert.ladder and cert.certify and C.FLAG_LADDER_CERTIFY in argv and C.FLAG_LADDER_PROBE_CELL not in argv
    assert argv[argv.index(C.FLAG_CAP_OVERRIDE) + 1] == cap and argv[argv.index(C.FLAG_ITEM_SELECTION) + 1] == sel
    ALL = tuple(W.runnable_cells(ROSTER, "gemma4e4b_it"))                                 # none, two prompted cells, three fitted cells
    FITTED = tuple(c for c in ALL if C.parse_launcher_cell(c).kind == C.CELL_KIND_FITTED)
    LORA = "constv2-mix-full__lora_r4_ce"
    assert FITTED == (LORA, "constv2ctl2-full__lora_r4_ce", DERIVED) and len(ALL) == 6
    assert cert.controls == ("bfcl_200",) and cert.cells == ALL == cert.wave_cells       # the fired leg alone
    assert cert.stage_kind == "ladder-probe" and cert.fitted_wave_cells == FITTED
    # code review: the CERTIFIED WAVE's membership rides on every ladder launch, separate from --cells
    assert argv[argv.index(C.FLAG_LADDER_WAVE_CELLS) + 1] == ",".join(ALL)
    one = Wr.compose(ROSTER, wave="gemma4e4b_it", sha=SHA, roster_sha256="0" * 64, launched_at="t", tag="cap16k",
                     cap_override=cap, item_selection=sel, cell=LORA, snapshot_path="/x/s.json")
    argv = list(one.launcher_argv)
    assert one.cells == (LORA,) and argv[argv.index(C.FLAG_LADDER_PROBE_CELL) + 1] == LORA
    assert argv[argv.index("--cells") + 1] == LORA
    assert argv[argv.index(C.FLAG_LADDER_WAVE_CELLS) + 1] == ",".join(ALL) and one.wave_cells == ALL   # narrowed --cells, whole wave
    assert C.FLAG_LADDER_CERTIFY not in argv and set(one.snapshot) == {LORA}
    assert one.stage_kind == "ladder-score" and one.fitted_wave_cells == FITTED
    base = Wr.compose(ROSTER, wave="gemma4e4b_it", sha=SHA, roster_sha256="0" * 64, launched_at="t", tag="cap16k",
                      cap_override=cap, item_selection=sel, cell="none", snapshot_path="/x/s.json")
    assert C.FLAG_LADDER_PROBE_CELL not in base.launcher_argv          # a base cell rides the wave record
    assert base.stage_kind == "ladder-score" and base.launcher_argv[base.launcher_argv.index(C.FLAG_LADDER_WAVE_CELLS) + 1] == ",".join(ALL)
    # an explicit --stage-kind that AGREES is honoured; a plain launch is wave by default, smoke on request
    assert Wr.compose(ROSTER, wave="gemma4e4b_it", sha=SHA, roster_sha256="0" * 64, launched_at="t", cap_override=cap,
                      item_selection=sel, certify=True, stage_kind="ladder-probe").stage_kind == "ladder-probe"
    assert Wr.compose(ROSTER, wave="gemma4e4b_it", sha=SHA, roster_sha256="0" * 64, launched_at="t").stage_kind == "wave"
    assert Wr.compose(ROSTER, wave="gemma4e4b_it", sha=SHA, roster_sha256="0" * 64, launched_at="t", stage_kind="smoke",
                      inspect_limit=5).stage_kind == "smoke"
    for kw, needle in ((dict(cap_override=cap), "travel together"), (dict(certify=True), "is a ladder-rerun flag"),
                       # a plain launch may name cells, but a LADDER scoring step scores ONE
                       (dict(cap_override=cap, item_selection=sel, cell=["none", LORA]), "scores ONE cell"),
                       (dict(cap_override=cap, item_selection=sel, certify=True, cell="none"), "not both"),
                       # code review: `--ladder-rerun` ALONE used to compose a PLAIN full-wave launch
                       (dict(ladder_rerun=True), "launches nothing"),
                       (dict(ladder_reprobe=True), "qualifies a ladder rerun"),
                       # code review: the ladder-* stage kinds are honoured — a disagreement refuses, never a silent override
                       (dict(cap_override=cap, item_selection=sel, certify=True, stage_kind="ladder-score"), "disagrees with this rerun's step ladder-probe"),
                       (dict(cap_override=cap, item_selection=sel, cell="none", stage_kind="ladder-probe"), "disagrees with this rerun's step ladder-score"),
                       (dict(cap_override=cap, item_selection=sel, cell="none", stage_kind="wave"), "disagrees"),
                       (dict(stage_kind="ladder-score"), "is a ladder rerun's stage"),
                       # llama's BFCL leg is BLOCKED (skipped for llama): a BFCL-only rerun of it has no runnable leg (qwen's
                       # legs are runnable, and nemotron's since its own live-probe PASS at 8365992, so the blocked case lives on llama)
                       (dict(cap_override=f"{C.inspect_task('bfcl_200')}=16384", item_selection=sel), "none of which is a runnable leg")):
        with pytest.raises(SystemExit, match=needle):
            Wr.compose(ROSTER, wave="llama8b_it" if "runnable leg" in needle else "gemma4e4b_it", sha=SHA,
                       roster_sha256="0" * 64, launched_at="t", **kw)


def test_meter_invocations_are_the_widen_meters_vocabulary_and_parse_with_its_parser(Wr, meter, tmp_path):
    L = Wr.compose(ROSTER, wave="gemma4e4b_it", sha=SHA, roster_sha256="0" * 64, launched_at="t", stage_kind="smoke", tag="smoke",
                   inspect_limit=20)
    ms = Wr.meter_argvs(L, ledger="~/ledger.jsonl", forecast_h=1.2, model_key="gemma4e4b")       # 1.2 h over six cells = 0.2 h each, exact
    assert len(ms) == len(L.cells) == 6 and all(m[1] == "scripts/bcb_bfcl_200_meter.py" for m in ms)
    for m, cell in zip(ms, L.cells):
        a = meter.build_parser().parse_args(m[2:])
        assert (a.wave, a.suffix, a.unit, a.stage_kind, a.forecast_h) == ("gemma4e4b_it", L.suffix, cell, "smoke", pytest.approx(1.2 / len(L.cells)))
    wave = Wr.compose(ROSTER, wave="gemma4e4b_it", sha=SHA, roster_sha256="0" * 64, launched_at="t")
    with pytest.raises(SystemExit, match="needs --forecast-h"):
        Wr.meter_argvs(wave, ledger="l", forecast_h=None, model_key="gemma4e4b")
    sel = str(tmp_path / "sel.json")
    cap = f"{C.inspect_task('bfcl_200')}=16384"
    cert = Wr.compose(ROSTER, wave="gemma4e4b_it", sha=SHA, roster_sha256="0" * 64, launched_at="t", tag="cap16k",
                      cap_override=cap, item_selection=sel, certify=True)
    probes = Wr.meter_argvs(cert, ledger="l", forecast_h=None, model_key="gemma4e4b", item_selection=sel)
    assert [meter.build_parser().parse_args(m[2:]).unit for m in probes] == [f"probe:{c}" for c in cert.fitted_wave_cells] == [
        "probe:constv2-mix-full__lora_r4_ce", "probe:constv2ctl2-full__lora_r4_ce", f"probe:{DERIVED}"]
    a = meter.build_parser().parse_args(probes[0][2:])
    assert (a.stage_kind, a.model, a.selection, a.fitted_cell) == ("ladder-probe", "gemma4e4b", sel, ["constv2-mix-full__lora_r4_ce"])
    # code review: the meter runs LOCALLY, so a scoring stage is handed the LOCAL verified-cache
    # paths of EVERY fitted wave cell's certificate — for a fitted, a base or a prompted scoring cell alike
    local = {c: f"/cache/rung_probe.{c}.json" for c in cert.fitted_wave_cells}
    for scored in ("constv2-mix-full__lora_r4_ce", "none", "prompted__safety_refusal", "prompted__si_constitution"):
        one = Wr.compose(ROSTER, wave="gemma4e4b_it", sha=SHA, roster_sha256="0" * 64, launched_at="t", tag="cap16k",
                         cap_override=cap, item_selection=sel, cell=scored)
        (score,) = Wr.meter_argvs(one, ledger="l", forecast_h=None, model_key="gemma4e4b", item_selection=sel,
                                  certificate_paths=local)
        a = meter.build_parser().parse_args(score[2:])
        # Fixer C's contract §2: a --cell X launch claims score:<cell> (its OWN claim) with the WAVE's certificates
        assert a.stage_kind == "ladder-score" and a.unit == f"{meter.SCORE_CELL_PREFIX}{scored}" == Wr.unit_of(score)
        assert sorted(a.certificate) == sorted(local.values())
        assert not any(p.startswith(C.VOL_GEN_ROOT) for p in a.certificate)      # never a container-only path
    assert Wr.score_unit(("x",)) == "score:x" and Wr.score_unit(("a", "b")) == "score" == meter.SCORE_UNIT if hasattr(meter, "SCORE_UNIT") else Wr.score_unit(("a", "b")) == "score"
    assert Wr.SCORE_CELL_PREFIX == meter.SCORE_CELL_PREFIX
    # Fixer C's contract §1: a release per admitted unit, in the meter's own spelling
    rel = Wr.release_argvs(one, ledger="~/l.jsonl", units=["score:x", "none"], why="nothing launched")
    assert [Wr.unit_of(r) for r in rel] == ["score:x", "none"]
    for r in rel:
        a = meter.build_parser().parse_args(r[2:])
        assert (a.release, a.wave, a.suffix, a.ledger) == ("nothing launched", one.wave, one.suffix, "~/l.jsonl")
        assert a.forecast_h is None and not a.record
    assert Wr.release_argvs(one, ledger="l", units=[], why="x") == []
    with pytest.raises(SystemExit, match="fetch_certificates first"):            # no / a partial certificate set refuses
        Wr.meter_argvs(one, ledger="l", forecast_h=None, model_key="gemma4e4b", item_selection=sel)
    with pytest.raises(SystemExit, match="fetch_certificates first"):
        Wr.meter_argvs(one, ledger="l", forecast_h=None, model_key="gemma4e4b", item_selection=sel,
                       certificate_paths={"constv2-mix-full__psr_ce_all": "/cache/x.json"})
    with pytest.raises(SystemExit, match="actual selection"):
        Wr.meter_argvs(cert, ledger="l", forecast_h=None, model_key="gemma4e4b")
    # the container paths the wrapper fetches: one per fitted wave cell + the wave record, under _ladder/<suffix>
    remote = Wr.certificate_remote_paths(one)
    assert set(remote) == set(cert.fitted_wave_cells) | {Wr.ADMISSION_KEY}
    assert remote[Wr.ADMISSION_KEY] == f"{C.VOL_GEN_ROOT}/gemma4e4b-f22it/_ladder/{one.suffix}/rung_admission.json"
    assert Wr.volume_relative(remote[Wr.ADMISSION_KEY]) == f"tinycap/gen/gemma4e4b-f22it/_ladder/{one.suffix}/rung_admission.json"
    with pytest.raises(SystemExit, match="not under the volume mount"):
        Wr.volume_relative("/tmp/x.json")


def _wave_certificates(Wr, launch, *, S=16, tamper=None):
    """Producer-shaped: a REAL certificate per fitted wave cell (`rung_probe.build_certificate` over the
    launch's provenance-like facts) and the wave record `build_admission` over them."""
    from sisteer.capeval.tinycap import rung_probe as RP
    sel = {C.inspect_task("bfcl_200"): {"ids": ["a", "b"], "n": 200, "n_selected": 2, "ids_sha256": "0" * 64,
                                        "cap_from": 8192, "cap_to": 16384, "max_prompt_tokens_rung1": 100}}
    certs = {}
    for cell in launch.fitted_wave_cells:
        prov = RP.Provenance(tree_sha="f" * 40, model_slug=launch.row, hf="h", rev="r", cfg_path="c.yaml", cfg_sha256="0" * 64,
                             object_digests={c: "ab" * 32 for c in launch.fitted_wave_cells}, selection=sel,
                             cap_overrides_by_task={C.inspect_task("bfcl_200"): 16384}, batched_token_budget=589_824,
                             budget_source="measured", suffix=launch.suffix, probed_cell=cell)
        certs[cell] = RP.build_certificate(accepted=RP.accepted_block(S, identity=18),
                                           walked=[RP.walked_entry(18, RP.WALK_ACCEPTED)] if S == 18 else
                                           [RP.walked_entry(18, RP.WALK_REFUSED, "KV cache holds 1000 tokens < the 589824-token budget (= 18 x 32768)"),
                                            RP.walked_entry(S, RP.WALK_ACCEPTED)],
                                           provenance=prov, probe_run_id="r", standup_s=1.0)
    admission = RP.build_admission(suffix=launch.suffix, common=RP.accepted_block(S, identity=18), certificates=certs)
    if tamper:
        tamper(certs, admission)
    return certs, admission


def test_the_scoring_certificates_are_fetched_into_a_local_cache_and_verified_against_the_wave_record(Wr, tmp_path):
    """Code review: the meter reads certificates from THIS filesystem, so the
    wrapper fetches the wave's set (`modal volume get`, one file per fitted wave cell + the
    record) and verifies each against the record BEFORE metering — a superseded / edited
    certificate, a record for another suffix or another cell set, or a failed fetch refuses."""
    from sisteer.capeval.tinycap import rung_probe as RP
    sel = str(tmp_path / "sel.json")
    one = Wr.compose(ROSTER, wave="gemma4e4b_it", sha=SHA, roster_sha256="0" * 64, launched_at="t", tag="cap16k",
                     cap_override=f"{C.inspect_task('bfcl_200')}=16384", item_selection=sel, cell="none")
    certs, admission = _wave_certificates(Wr, one)
    volume = {Wr.volume_relative(p): (admission if k == Wr.ADMISSION_KEY else certs[k]) for k, p in Wr.certificate_remote_paths(one).items()}
    calls = []

    def runner(argv, **kw):                                  # a `modal volume get` double: writes the volume's file
        calls.append(argv)
        assert argv[1:6] == ["-m", "modal", "volume", "get", "--force"] and argv[6] == C.MODAL_VOLUME
        remote, local = argv[7], Path(argv[8])
        if remote not in volume:
            return SimpleNamespace(returncode=1, stdout="", stderr=f"No such file: {remote}")
        local.write_text(json.dumps(volume[remote]))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    cache = tmp_path / "cache" / "inv-1"
    paths = Wr.fetch_certificates(one, cache_dir=cache, modal_py="/m/python", runner=runner)
    # one `modal volume get` per fitted wave cell + one for the record (the it wave's THREE fitted cells since 2026-09-14: the batch-8
    # constitution LoRA, the batch-64 control LoRA and the batch-64 LoRA subtraction)
    assert paths == {c: str(cache / RP.certificate_filename(c)) for c in one.fitted_wave_cells} and len(calls) == len(one.fitted_wave_cells) + 1 == 4
    assert all(Path(p).is_file() for p in paths.values()) and (cache / RP.ADMISSION_FILENAME).is_file()
    # the verified cache is what the meter is handed, for a `none` scoring launch too
    (score,) = Wr.meter_argvs(one, ledger="l", forecast_h=None, model_key="gemma4e4b", item_selection=sel, certificate_paths=paths)
    n_fitted = len(one.fitted_wave_cells)
    assert sorted(score[score.index("--certificate") + 1::2][:n_fitted]) == sorted(paths.values())
    # every verification face refuses by name
    def refuses(tamper, needle):
        c2, a2 = _wave_certificates(Wr, one, tamper=tamper)
        d = tmp_path / "bad"
        d.mkdir(exist_ok=True)
        (d / RP.ADMISSION_FILENAME).write_text(json.dumps(a2))
        for cell, cert in c2.items():
            (d / RP.certificate_filename(cell)).write_text(json.dumps(cert))
        with pytest.raises(SystemExit, match=needle):
            Wr.verify_certificate_cache(d, one)
    refuses(lambda c, a: c["constv2-mix-full__lora_r4_ce"].__setitem__("standup_s", 2.0), "not the certificate the record")
    refuses(lambda c, a: a.__setitem__("suffix", "other"), "is for suffix 'other'")
    refuses(lambda c, a: a["cells"].pop("constv2-mix-full__lora_r4_ce"), "names cells")
    refuses(lambda c, a: c["constv2-mix-full__lora_r4_ce"].__setitem__("probed_cell", "none"), "is about 'none'")
    refuses(lambda c, a: a.__setitem__("common", RP.accepted_block(14, identity=18)), "the record's common rung")
    with pytest.raises(SystemExit, match="certify the wave first"):
        Wr.verify_certificate_cache(tmp_path / "empty", one)
    # a fetch that fails names the file and the exact CLI it ran
    volume.pop(Wr.volume_relative(Wr.certificate_remote_paths(one)[Wr.ADMISSION_KEY]))
    with pytest.raises(SystemExit, match="could not fetch .*rung_admission.json"):
        Wr.fetch_certificates(one, cache_dir=tmp_path / "cache2", modal_py="/m/python", runner=runner)


def test_watcher_argv_and_run_id_capture(Wr):
    L = Wr.compose(ROSTER, wave="gemma4e4b_main", sha=SHA, roster_sha256="0" * 64, launched_at="t")
    w = Wr.watcher_argv("aaaaaaa-deadbeef", L, modal_py="/m/python")
    assert w[:2] == ["/m/python", "scripts/tinycap_sandbox_watch.py"]
    assert w[w.index("--run-id") + 1] == "aaaaaaa-deadbeef" and w[w.index("--suffix") + 1] == L.suffix
    assert w[w.index("--model") + 1] == "gemma4e4b-f22" and w[w.index("--cells") + 1] == ",".join(L.cells)
    # the watcher reads the attempt pointers under the NATIVE out root — the same flag the launcher got
    assert w[w.index("--serving-path") + 1] == "lora_native" == L.serving_path_requested
    assert w.index("--serving-path") > w.index("--cells") and "--poll" in w
    assert Wr.run_id_from_log("[tinycap] argv: …\n[tinycap] run id: aaaaaaa-deadbeef\n") == "aaaaaaa-deadbeef"
    assert Wr.run_id_from_log("nothing yet") is None
    # code review: a REUSED suffix log carries the previous attempt's id — the search
    # starts at the byte offset THIS invocation's output begins at, so the watcher gets the NEW id
    old = "== launch t1 modal run …\n[tinycap] run id: aaaaaaa-00000001\n[tinycap] 4/4 container(s) OK\n"
    header = "== launch t2 modal run …\n"
    assert Wr.run_id_from_log(old + header, start=len(old)) is None                 # nothing new yet: never the old id
    assert Wr.run_id_from_log(old + header + "[tinycap] run id: aaaaaaa-00000002\n", start=len(old)) == "aaaaaaa-00000002"
    assert Wr.run_id_from_log(old + header, start=0) == "aaaaaaa-00000001"          # the unguarded read is exactly the bug
    assert Wr.SAMPLING_MODE == C.SAMPLING_PROTOCOL == "protocol" and Wr.LEDGER_DEFAULT.endswith("bcb-bfcl-200-widen/gpu_ledger.jsonl")
    # the CLI refuses a wave that cannot launch, before anything else (no Modal, no ledger)
    with pytest.raises(SystemExit, match="REFUSED"):
        Wr.main(["--wave", "nope", "--sha", SHA, "--dry-run", "--log-dir", "/tmp/bcbw-test-logs"])


def test_a_resumed_launch_waits_on_its_own_output_and_no_run_id_terminates_and_stops_the_app(Wr, tmp_path):
    """The resume face end to end on the log: the wrapper appends its header, records the offset,
    and `wait_for_run_id` reads only what follows; when nothing appears it returns None, and
    `abandon_launch` terminates the local client and stops the detached app BY NAME (code review:
    a `modal run --detach` used to be left running with no watcher armed)."""
    L = Wr.compose(ROSTER, wave="gemma4e4b_main", sha=SHA, roster_sha256="0" * 64, launched_at="t")
    log = tmp_path / f"{L.suffix}.log"
    log.write_text("== launch t1 …\n[tinycap] run id: aaaaaaa-00000001\n")
    with log.open("a") as fh:
        fh.write("== launch t2 …\n")
        fh.flush()
        start = fh.tell()

    class _Proc:
        def __init__(self):
            self.rc, self.terminated, self.killed = None, False, False

        def poll(self):
            return self.rc

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            if self.terminated and not self.killed:
                raise subprocess.TimeoutExpired("modal", timeout)    # a client that ignores SIGTERM
            self.rc = -9
            return self.rc

    clock = {"t": 0.0}
    ticks = []

    def sleep(s):
        ticks.append(s)
        clock["t"] += s
        if len(ticks) == 3:                                             # the launcher prints its id on the third poll
            with log.open("a") as fh:
                fh.write("[tinycap] run id: aaaaaaa-00000002\n")
    proc = _Proc()
    got = Wr.wait_for_run_id(log, proc, start=start, wait_s=60, poll_s=5, clock=lambda: clock["t"], sleep=sleep)
    assert got == "aaaaaaa-00000002" and len(ticks) == 3                # never aaaaaaa-00000001
    # nothing appears within the wait: None, after ceil(wait / poll) polls
    log2 = tmp_path / "quiet.log"
    log2.write_text("== launch t1 …\n[tinycap] run id: aaaaaaa-00000001\n== launch t2 …\n")
    clock["t"] = 0.0
    assert Wr.wait_for_run_id(log2, _Proc(), start=log2.stat().st_size, wait_s=20, poll_s=5,
                              clock=lambda: clock["t"], sleep=lambda s: clock.__setitem__("t", clock["t"] + s)) is None
    # the remote app is reconciled BY NAME: listed, every live row with this description stopped BY ID, listed again
    listing = {"n": 0}
    calls = []

    def runner(argv, **kw):
        calls.append(argv)
        if argv[3:6] == ["app", "list", "--json"]:
            listing["n"] += 1
            rows = [{"app_id": "ap-other", "description": "someone-else", "state": "ephemeral (detached)"},
                    {"app_id": "ap-old", "description": L.app, "state": "stopped"}]
            if listing["n"] == 1:
                rows.append({"app_id": "ap-live", "description": L.app, "state": "ephemeral (detached)"})
            return SimpleNamespace(returncode=0, stdout=json.dumps(rows), stderr="")
        assert argv[3:5] == ["app", "stop"]
        return SimpleNamespace(returncode=0, stdout="stopped", stderr="")
    rec = Wr.abandon_launch(proc, L, modal_py="/m/python", runner=runner)
    assert proc.terminated and proc.killed and rec["terminated"] and rec["killed"] and rec["client_rc"] == -9
    assert [s["app_id"] for s in rec["stops"]] == ["ap-live"] and rec["stops"][0]["argv"][-2:] == ["stop", "ap-live"]
    assert rec["remote_stopped"] is True and rec["live_after"] == [] and rec["problems"] == []
    assert [a[3:6] for a in calls] == [["app", "list", "--json"], ["app", "stop", "ap-live"], ["app", "list", "--json"]]
    # a client already gone is not signalled; nothing live listed under the name -> confirmed without a stop
    gone = _Proc()
    gone.rc = 1
    rec2 = Wr.abandon_launch(gone, L, modal_py="/m/python",
                             runner=lambda argv, **kw: SimpleNamespace(returncode=0, stdout="[]", stderr=""))
    assert not gone.terminated and rec2["client_rc"] == 1 and rec2["stops"] == [] and rec2["remote_stopped"] is True
    assert Wr.live_apps_named([{"description": L.app, "state": "stopping..."}], L.app) != []     # not gone yet
    with pytest.raises(SystemExit, match="not a list"):
        Wr.live_apps_named({"x": 1}, L.app)


def test_admitted_claims_are_released_when_the_invocation_does_not_launch(Wr, tmp_path, monkeypatch):
    """The admission contract (an admission is a CLAIM): both faces through `main` with
    subprocess doubles — (a) the second unit's admission is refused, so the first unit's claim is
    released and the refusal records it; (b) every unit admitted, the launcher never prints a run id,
    so the client is terminated, the app stopped by name and EVERY claim released, all recorded."""
    calls = []
    seen_app = {"name": None}

    class _Proc:
        pid = 4242

        def __init__(self, argv, **kw):
            calls.append(("popen", argv))
            seen_app["name"] = kw["env"][C.MODAL_APP_NAME_ENV]          # the invocation-unique app name the launcher gets
            self.rc = None

        def poll(self):
            return self.rc

        def terminate(self):
            self.rc = -15

        def kill(self):
            self.rc = -9

        def wait(self, timeout=None):
            return self.rc

    def make_runner(refuse_unit=None):
        def runner(argv, **kw):
            calls.append(("run", argv))
            if argv[1] == "scripts/bcb_bfcl_200_meter.py" and "--release" in argv:
                return SimpleNamespace(returncode=0, stdout=f"[meter] released {Wr.unit_of(argv)}\n", stderr="")
            if argv[1] == "scripts/bcb_bfcl_200_meter.py":
                unit = Wr.unit_of(argv)
                if unit == refuse_unit:
                    return SimpleNamespace(returncode=2, stdout="", stderr=f"REFUSED launch of {unit}: CLAIMED\n")
                return SimpleNamespace(returncode=0, stdout=f"[meter] {unit} admitted\n", stderr="")
            if argv[1:5] == ["-m", "modal", "app", "stop"]:
                return SimpleNamespace(returncode=0, stdout="stopped", stderr="")
            if argv[1:6] == ["-m", "modal", "app", "list", "--json"]:
                rows = [{"app_id": "ap-live", "description": seen_app["name"], "state": "ephemeral (detached)"}]
                # the FIRST list shows the app live; after the stop it is gone
                stopped = any(a[1:5] == ["-m", "modal", "app", "stop"] for _, a in calls if isinstance(a, list))
                return SimpleNamespace(returncode=0, stdout=json.dumps([] if stopped else rows), stderr="")
            raise AssertionError(argv)
        return runner

    def meters_run(kind):
        return [Wr.unit_of(argv) for k, argv in calls if k == "run" and argv[1] == "scripts/bcb_bfcl_200_meter.py"
                and (("--release" in argv) == (kind == "release"))]

    base = ["--wave", "gemma4e4b_it", "--sha", SHA, "--forecast-h", "1.0", "--log-dir", str(tmp_path / "logs"),
            "--ledger", str(tmp_path / "ledger.jsonl"), "--modal-py", "/m/python", "--no-watcher"]
    # (a) the second cell's admission is refused: the first cell's claim is released; nothing launched
    monkeypatch.setattr(Wr.subprocess, "run", make_runner(refuse_unit="prompted__safety_refusal"))
    monkeypatch.setattr(Wr.subprocess, "Popen", _Proc)
    with pytest.raises(SystemExit, match="METER REFUSED prompted__safety_refusal .* released none \\(rc 0\\)") as e:
        Wr.main(base)
    assert meters_run("admit") == ["none", "prompted__safety_refusal"] and meters_run("release") == ["none"]
    assert not any(k == "popen" for k, _ in calls)
    rel = next(argv for k, argv in calls if k == "run" and "--release" in argv)
    assert "a later unit's admission was refused (prompted__safety_refusal, rc 2)" in rel[rel.index("--release") + 1]
    # (b) every unit admitted, then NO RUN ID: client terminated, app stopped, all four claims released
    calls.clear()
    monkeypatch.setattr(Wr.subprocess, "run", make_runner())
    monkeypatch.setattr(Wr, "RUN_ID_WAIT_S", 0)
    with pytest.raises(SystemExit) as e:
        Wr.main(base)
    msg = str(e.value)
    assert "NO RUN ID" in msg and "still running after 0 min" in msg and "terminated (rc -15)" in msg
    assert "stop ap-live (ephemeral (detached)) -> rc 0" in msg and "confirmed stopped" in msg
    assert seen_app["name"].startswith("tinycap-bcbw-gemma4e4b_it-aaaaaaa-wave-") and len(seen_app["name"]) == len("tinycap-bcbw-gemma4e4b_it-aaaaaaa-wave-") + 6
    cells = W.runnable_cells(ROSTER, "gemma4e4b_it")                                     # none, two prompted cells, three fitted cells
    assert meters_run("admit") == cells and meters_run("release") == cells and len(cells) == 6
    assert all(f"released {c}" in msg or c in msg for c in cells) and "nothing is left running unwatched or claimed" in msg
    stop_at = next(i for i, (k, argv) in enumerate(calls) if k == "run" and argv[1:5] == ["-m", "modal", "app", "stop"])
    first_release = next(i for i, (k, argv) in enumerate(calls) if k == "run" and "--release" in argv)
    assert stop_at < first_release                                            # abandon (stop the app) BEFORE releasing
    rel = next(argv for k, argv in calls if k == "run" and "--release" in argv)
    assert rel[rel.index("--release") + 1].startswith("never ran: NO RUN ID (still running after 0 min), no GPU fan-out")
    # a failed release is RECORDED in the refusal, never raised over it
    def runner_bad_release(argv, **kw):
        if "--release" in argv:
            return SimpleNamespace(returncode=1, stdout="", stderr="nothing to release")
        return make_runner()(argv, **kw)
    calls.clear()
    monkeypatch.setattr(Wr.subprocess, "run", runner_bad_release)
    with pytest.raises(SystemExit, match="released none \\(rc 1\\)"):
        Wr.main(base)
    # the pure recorder: one record per unit with the rc and the exact argv
    recs = Wr.release_claims(Wr.compose(ROSTER, wave="gemma4e4b_it", sha=SHA, roster_sha256="0" * 64, launched_at="t"),
                             ledger="l", units=["none"], why="w",
                             runner=lambda argv, **kw: SimpleNamespace(returncode=3, stdout="", stderr="boom"))
    assert recs[0]["unit"] == "none" and recs[0]["rc"] == 3 and recs[0]["output"] == "boom" and "--release" in recs[0]["argv"]


def test_local_files_are_invocation_unique_and_two_scoring_cells_interleave_without_sharing(Wr, tmp_path, monkeypatch):
    """Every local file was keyed by the SUFFIX alone, so two wrappers scoring cells
    A and B of one suffix concurrently shared the snapshot (B's single-cell block overwrote A's before
    the launcher read it), the launch log (either could read the other's run id) and the watcher log.
    Now the invocation id keys them all; A's `modal run` double runs B's whole wrapper BEFORE printing
    A's run id, and each side reads only its own files. (B is the constitution-SI prompted cell: the PSR cell is
    out by default, and a scoring launch needs a RUNNABLE cell.)"""
    A, B = "constv2-mix-full__lora_r4_ce", "prompted__si_constitution"
    la = Wr.compose(ROSTER, wave="gemma4e4b_it", sha=SHA, roster_sha256="0" * 64, launched_at="2026-09-11T20:00:00+00:00",
                    tag="cap16k", cap_override=f"{C.inspect_task('bfcl_200')}=16384", item_selection="/x/sel.json", cell=A)
    inv_a = Wr.invocation_id(la.suffix, scope=A, launched_at="2026-09-11T20:00:00+00:00", token="aaa111")
    inv_b = Wr.invocation_id(la.suffix, scope=B, launched_at="2026-09-11T20:00:00+00:00", token="aaa111")
    assert inv_a == f"{la.suffix}.{A}.20260911T200000-aaa111" and inv_a != inv_b
    pa, pb = Wr.invocation_paths(tmp_path, str(tmp_path / "cache"), inv_a), Wr.invocation_paths(tmp_path, str(tmp_path / "cache"), inv_b)
    assert set(pa) == {"snapshot", "launch_log", "watch_log", "dry_log", "cert_cache"}
    assert not (set(map(str, pa.values())) & set(map(str, pb.values())))          # the same suffix + time: disjoint files
    assert all(la.suffix in str(v) for v in pa.values())                          # the suffix still names the volume relationship
    # two random tokens at the same second never collide either
    assert Wr.invocation_id("s", scope="wave", launched_at="t") != Wr.invocation_id("s", scope="wave", launched_at="t")
    # --- the interleaving, through main -------------------------------------------------------------------------------
    sel = tmp_path / "sel.json"
    sel.write_text("{}")
    certs, admission = _wave_certificates(Wr, la)
    volume = {Wr.volume_relative(p): (admission if k == Wr.ADMISSION_KEY else certs[k]) for k, p in Wr.certificate_remote_paths(la).items()}
    seen = {"popen": [], "meters": [], "b_done": False}

    def runner(argv, **kw):
        if argv[1] == "scripts/bcb_bfcl_200_meter.py":
            seen["meters"].append(Wr.unit_of(argv))
            return SimpleNamespace(returncode=0, stdout="admitted\n", stderr="")
        if argv[3:5] == ["volume", "get"]:
            Path(argv[8]).write_text(json.dumps(volume[argv[7]]))
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(argv)

    def args(cell):
        return ["--wave", "gemma4e4b_it", "--sha", SHA, "--ladder-rerun", "--cap-override", f"{C.inspect_task('bfcl_200')}=16384",
                "--item-selection", str(sel), "--tag", "cap16k", "--cell", cell, "--log-dir", str(tmp_path / "logs"),
                "--ledger", str(tmp_path / "ledger.jsonl"), "--cert-cache", str(tmp_path / "cache"), "--modal-py", "/m/python",
                "--no-watcher"]

    class _Proc:
        pid = 1

        def __init__(self, argv, **kw):
            cell = argv[argv.index("--cells") + 1]
            seen["popen"].append(cell)
            fh = kw["stdout"]
            if cell == A and not seen["b_done"]:
                seen["b_done"] = True
                Wr.main(args(B))                       # B's WHOLE wrapper runs while A waits for its run id
            fh.write(f"[tinycap] argv: … --cells {cell}\n[tinycap] run id: aaaaaaa-{'0000000a' if cell == A else '0000000b'}\n")
            fh.flush()
            self.rc = None

        def poll(self):
            return self.rc

    monkeypatch.setattr(Wr.subprocess, "run", runner)
    monkeypatch.setattr(Wr.subprocess, "Popen", _Proc)
    monkeypatch.setattr(Wr, "RUN_ID_POLL_S", 0)
    import io
    from contextlib import redirect_stdout
    out = io.StringIO()
    with redirect_stdout(out):
        assert Wr.main(args(A)) == 0
    text = out.getvalue()
    assert seen["popen"] == [A, B] and seen["meters"] == [f"score:{A}", f"score:{B}"]
    assert f"LAUNCHED {la.suffix} run_id=aaaaaaa-0000000a" in text and f"LAUNCHED {la.suffix} run_id=aaaaaaa-0000000b" in text
    snaps = sorted((tmp_path / "logs").glob("roster_snapshot_*.json"))
    logs = sorted((tmp_path / "logs").glob(f"{la.suffix}.*.log"))
    assert len(snaps) == 2 and len(logs) == 2                                    # one of each per invocation, never shared
    by_cell = {tuple(json.loads(p.read_text())): p for p in snaps}
    assert set(by_cell) == {(A,), (B,)}                                          # A's snapshot is still A's after B ran
    assert A in by_cell[(A,)].name and B in by_cell[(B,)].name
    ids = {p.name: Wr.run_id_from_log(p.read_text()) for p in logs}
    assert sorted(ids.values()) == ["aaaaaaa-0000000a", "aaaaaaa-0000000b"]
    assert all((A in n) == (rid.endswith("a")) for n, rid in ids.items())         # each log carries its own run id
    caches = sorted(p.name for p in (tmp_path / "cache").iterdir())
    assert len(caches) == 2 and all(la.suffix in c for c in caches)               # the certificate cache too


def test_claims_are_retained_unless_the_remote_app_is_confirmed_gone_and_no_gpu_work_started(Wr, tmp_path, monkeypatch):
    """Round 2: on NO RUN ID the claims were released even when `modal app stop` failed —
    the detached app could still be running unwatched while its reservation was gone. Now a release
    needs EVIDENCE: the app reconciled by name and confirmed gone, and no GPU fan-out line in this
    invocation's output; otherwise every claim is RETAINED and the reconcile commands are printed."""
    L = Wr.compose(ROSTER, wave="gemma4e4b_it", sha=SHA, roster_sha256="0" * 64, launched_at="t")
    base = ["--wave", "gemma4e4b_it", "--sha", SHA, "--forecast-h", "1.0", "--log-dir", str(tmp_path / "logs"),
            "--ledger", str(tmp_path / "ledger.jsonl"), "--modal-py", "/m/python", "--no-watcher"]
    calls = []

    seen_app = {"name": None}

    def make_runner(*, stop_rc=0, live_after=False, list_rc=0, rows_live=True):
        def runner(argv, **kw):
            calls.append(argv)
            if argv[1] == "scripts/bcb_bfcl_200_meter.py":
                return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")
            if argv[1:6] == ["-m", "modal", "app", "list", "--json"]:
                n_lists = sum(1 for a in calls if a[1:6] == ["-m", "modal", "app", "list", "--json"])
                live = [{"app_id": "ap-live", "description": seen_app["name"], "state": "ephemeral (detached)"},
                        # a SIBLING invocation's app on the same (wave, sha): live throughout, never a stop target
                        {"app_id": "ap-sibling", "description": Wr.app_name("gemma4e4b_it", SHA, scope="wave", token="s1bl1n"),
                         "state": "ephemeral (detached)"}]
                rows = live if (rows_live and (n_lists == 1 or live_after)) else []
                return SimpleNamespace(returncode=list_rc, stdout=json.dumps(rows) if list_rc == 0 else "", stderr="boom" if list_rc else "")
            if argv[1:5] == ["-m", "modal", "app", "stop"]:
                return SimpleNamespace(returncode=stop_rc, stdout="", stderr="cannot stop" if stop_rc else "")
            raise AssertionError(argv)
        return runner

    class _Proc:
        pid = 7

        def __init__(self, argv, **kw):
            self.rc = None
            self.fh = kw["stdout"]
            seen_app["name"] = kw["env"][C.MODAL_APP_NAME_ENV]

        def poll(self):
            return self.rc

        def terminate(self):
            self.rc = -15

        def wait(self, timeout=None):
            return self.rc

    releases = lambda: [a for a in calls if a[1] == "scripts/bcb_bfcl_200_meter.py" and "--release" in a]
    monkeypatch.setattr(Wr.subprocess, "Popen", _Proc)
    monkeypatch.setattr(Wr, "RUN_ID_WAIT_S", 0)
    # (1) the stop exits non-zero and the app is STILL listed live: retained, the commands printed
    monkeypatch.setattr(Wr.subprocess, "run", make_runner(stop_rc=1, live_after=True))
    with pytest.raises(SystemExit) as e:
        Wr.main(base)
    msg = str(e.value)
    assert "stop ap-live (ephemeral (detached)) -> rc 1" in msg and "meter claims RETAINED (none, prompted__safety_refusal" in msg
    assert "could NOT be confirmed stopped" in msg and "live app(s) remain after the stop: ap-live" in msg
    assert "modal app list --json" in msg and "modal app stop <app_id>" in msg and "--release" in msg and releases() == []
    # (2) the list itself fails: nothing is known about the remote app -> retained
    calls.clear()
    monkeypatch.setattr(Wr.subprocess, "run", make_runner(list_rc=1))
    with pytest.raises(SystemExit, match="RETAINED") as e:
        Wr.main(base)
    assert "rc 1: boom" in str(e.value) and releases() == [] and "no live app listed by name" in str(e.value)
    # (3) the stop succeeds and the second list shows the app gone: released as never ran
    calls.clear()
    monkeypatch.setattr(Wr.subprocess, "run", make_runner())
    with pytest.raises(SystemExit, match="confirmed stopped; meter claims: released none \\(rc 0\\)"):
        Wr.main(base)
    n_cells = len(W.runnable_cells(ROSTER, "gemma4e4b_it"))
    assert len(releases()) == n_cells and all(f"app {seen_app['name']} confirmed stopped" in a[a.index("--release") + 1] for a in releases())
    assert [a[-1] for a in calls if a[1:5] == ["-m", "modal", "app", "stop"]] == ["ap-live"]      # the sibling's app was never stopped
    # (4) no live row under THIS invocation's name (only the sibling's): confirmed without a stop -> released
    calls.clear()
    monkeypatch.setattr(Wr.subprocess, "run", make_runner(rows_live=False))
    with pytest.raises(SystemExit, match="no live app listed by name; confirmed stopped"):
        Wr.main(base)
    assert len(releases()) == n_cells and not [a for a in calls if a[1:5] == ["-m", "modal", "app", "stop"]]
    # (5) the app is gone but the launcher's GPU fan-out line is in THIS invocation's output: billable work
    #     started -> retained, the operator records the actual instead of releasing
    calls.clear()

    class _ProcFanout(_Proc):
        def __init__(self, argv, **kw):
            super().__init__(argv, **kw)
            self.fh.write("[tinycap] launching 4 configuration(s) @ sha aaaaaaa …\n")   # and no run id line (a torn log)
            self.fh.flush()
    monkeypatch.setattr(Wr.subprocess, "Popen", _ProcFanout)
    monkeypatch.setattr(Wr.subprocess, "run", make_runner())
    with pytest.raises(SystemExit, match="RETAINED") as e:
        Wr.main(base)
    assert "GPU work STARTED" in str(e.value) and "--record --actual-h" in str(e.value) and releases() == []
    assert Wr.gpu_work_started("[tinycap] launching 4 configuration(s)") and not Wr.gpu_work_started("[tinycap] run id: x")


def test_the_run_id_is_read_from_the_appended_bytes_not_a_byte_offset_into_decoded_text(Wr, tmp_path):
    """`fh.tell()` is a BYTE offset, and it was applied to `read_text()` CHARACTERS —
    multibyte output before the run-id line skipped past it and a valid retry read as NO RUN ID."""
    log = tmp_path / "l.log"
    old = "== launch t1 … — «previous» —\n[tinycap] run id: aaaaaaa-00000001\n"     # multibyte: … — « »
    log.write_bytes(old.encode("utf-8"))
    with log.open("a", encoding="utf-8") as fh:
        fh.write("== launch t2 … — retry —\n")
        fh.flush()
        start = fh.tell()
    with log.open("a", encoding="utf-8") as fh:
        fh.write("[tinycap] run id: aaaaaaa-00000002\n")
    assert log.read_bytes()[start:].decode("utf-8") != log.read_text(encoding="utf-8")[start:]   # bytes != characters here
    assert Wr.run_id_from_log(log.read_text(), start) is None                          # the bug: the offset skipped the line
    assert Wr.appended_text(log, start) == "[tinycap] run id: aaaaaaa-00000002\n"
    assert Wr.run_id_from_log(Wr.appended_text(log, start)) == "aaaaaaa-00000002"

    class _Proc:
        def poll(self):
            return None
    assert Wr.wait_for_run_id(log, _Proc(), start=start, wait_s=1, poll_s=0, clock=lambda: 0.0, sleep=lambda s: None) == "aaaaaaa-00000002"
    # invalid bytes never raise (a torn write is decoded with replacement)
    log.write_bytes(b"== h\n\xff\xfe[tinycap] run id: aaaaaaa-00000003\n")
    assert Wr.run_id_from_log(Wr.appended_text(log, 5)) == "aaaaaaa-00000003"


def test_abandoning_a_failed_invocation_never_stops_a_live_siblings_app(Wr, tmp_path):
    """Two per-cell scoring invocations on one certified wave used to get the SAME remote
    app name (`app_name` identified only (wave, sha)), so abandoning A stopped B's app and its GPU work and
    reported `remote_stopped=True`. The name now carries the invocation's nonce; the reconcile matches only
    that name — and, when the launcher's output printed the app id, only that id. (B is the constitution-SI
    prompted cell — a RUNNABLE scoring cell; the PSR cell is out by default.)"""
    A, B = "constv2-mix-full__lora_r4_ce", "prompted__si_constitution"
    kw = dict(roster_sha256="0" * 64, launched_at="t", tag="cap16k", cap_override=f"{C.inspect_task('bfcl_200')}=16384",
              item_selection="/x/sel.json")
    la = Wr.compose(ROSTER, wave="gemma4e4b_it", sha=SHA, cell=A, app_token="aaa111", **kw)
    lb = Wr.compose(ROSTER, wave="gemma4e4b_it", sha=SHA, cell=B, app_token="bbb222", **kw)
    assert la.app != lb.app and la.suffix == lb.suffix                              # one suffix, two apps
    # the 28-char cell does not fit whole within Modal's 64: the scope tag is shortened, the nonce kept
    assert la.app.startswith("tinycap-bcbw-gemma4e4b_it-aaaaaaa-constv2-mix-full__lora") and la.app.endswith("-aaa111")
    assert len(la.app) == Wr.MODAL_APP_NAME_MAX == 64
    rows = [{"app_id": "ap-aaaaaaaaaaaaaaaaaaaaaa", "description": la.app, "state": "ephemeral (detached)"},
            {"app_id": "ap-bbbbbbbbbbbbbbbbbbbbbb", "description": lb.app, "state": "ephemeral (detached)"},
            {"app_id": "ap-cccccccccccccccccccccc", "description": Wr.app_name("gemma4e4b_it", SHA), "state": "ephemeral (detached)"}]
    stopped = []

    def runner(argv, **kw_):
        if argv[3:6] == ["app", "list", "--json"]:
            live = [r for r in rows if r["app_id"] not in stopped]
            return SimpleNamespace(returncode=0, stdout=json.dumps(live), stderr="")
        assert argv[3:5] == ["app", "stop"]
        stopped.append(argv[5])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    class _Gone:
        def poll(self):
            return 1
    # A fails (no run id): only A's app is stopped; B's and the legacy-named app stay live
    rec = Wr.abandon_launch(_Gone(), la, modal_py="/m/python", runner=runner)
    assert stopped == ["ap-aaaaaaaaaaaaaaaaaaaaaa"] and rec["remote_stopped"] is True and rec["app_id"] is None
    assert [r["app_id"] for r in rows if r["app_id"] not in stopped] == ["ap-bbbbbbbbbbbbbbbbbbbbbb", "ap-cccccccccccccccccccccc"]
    # the pure matcher: the unique name selects one row; the id narrows it further
    assert [r["app_id"] for r in Wr.live_apps_named(rows, lb.app)] == ["ap-bbbbbbbbbbbbbbbbbbbbbb"]
    assert Wr.live_apps_named(rows, lb.app, app_id="ap-cccccccccccccccccccccc") == []
    # B fails with its app id in the launcher's output: exactly that id is stopped
    stopped.clear()
    appended = "✓ Initialized. View run at https://modal.com/apps/ws/main/ap-bbbbbbbbbbbbbbbbbbbbbb\n✓ Created objects.\n"
    assert Wr.app_id_from_log(appended) == "ap-bbbbbbbbbbbbbbbbbbbbbb" and Wr.app_id_from_log("no id here") is None
    rec = Wr.abandon_launch(_Gone(), lb, modal_py="/m/python", appended=appended, runner=runner)
    assert stopped == ["ap-bbbbbbbbbbbbbbbbbbbbbb"] and rec["remote_stopped"] is True and rec["app_id"] == "ap-bbbbbbbbbbbbbbbbbbbbbb"
    # an app id the launcher printed that the list attributes to ANOTHER description: refused, nothing stopped, not confirmed
    stopped.clear()
    rec = Wr.abandon_launch(_Gone(), la, modal_py="/m/python",
                            appended="View run at https://modal.com/apps/ws/main/ap-cccccccccccccccccccccc", runner=runner)
    assert stopped == [] and rec["remote_stopped"] is False and any("not this invocation's" in p for p in rec["problems"])
    # the reconcile commands name this invocation's app alone
    assert la.app in Wr.reconcile_commands(la, ledger="l", units=["score:x"], modal_py="/m/python")[0]


def test_a_plain_launch_may_be_narrowed_to_named_runnable_cells(Wr, meter):
    """`--cell` on a PLAIN smoke / wave runs exactly the named
    runnable cells — repeatable or comma-separated, `none` first when present, otherwise the given order — so the canary
    gate (`none`, then ONE arm, then the rest) is a sequence of ordinary launches; the snapshot, the meter units and the
    watcher cells follow the launched set; a cell outside the wave's runnable set refuses by name; the stage kind stays
    smoke / wave; the ladder paths are unchanged."""
    two = Wr.compose(ROSTER, wave="gemma4e4b_it", sha=SHA, roster_sha256="0" * 64, launched_at="t", stage_kind="wave",
                     cell="none,constv2-mix-full__lora_r4_ce", snapshot_path="/x/s.json")
    assert two.cells == ("none", "constv2-mix-full__lora_r4_ce") == two.watcher_cells and two.stage_kind == "wave"
    argv = list(two.launcher_argv)
    assert argv[argv.index("--cells") + 1] == "none,constv2-mix-full__lora_r4_ce" and C.FLAG_CAP_OVERRIDE not in argv
    assert set(two.snapshot) == {"none", "constv2-mix-full__lora_r4_ce"} and two.wave_cells == tuple(W.runnable_cells(ROSTER, "gemma4e4b_it"))
    meters = Wr.meter_argvs(two, ledger="/l", forecast_h=0.5, model_key="gemma4e4b")
    assert [Wr.unit_of(m) for m in meters] == ["none", "constv2-mix-full__lora_r4_ce"] and all("--forecast-h" in m and m[m.index("--forecast-h") + 1] == "0.250" for m in meters)
    for m in meters:
        meter.build_parser().parse_args(m[2:])
    assert two.app.startswith("tinycap-bcbw-gemma4e4b_it-aaaaaaa") and Wr.launch_scope(two.cells) == "2cells"
    # repeatable form, and `none` moves FIRST whatever the order given; the rest keep their order
    rep = Wr.compose(ROSTER, wave="gemma4e4b_it", sha=SHA, roster_sha256="0" * 64, launched_at="t",
                     cell=[DERIVED, "prompted__si_constitution,none"], snapshot_path="/x/s.json")
    assert rep.cells == ("none", DERIVED, "prompted__si_constitution")
    # one cell: the scope is the cell (the app name says which); the stage kind is still smoke on request
    one = Wr.compose(ROSTER, wave="gemma4e4b_it", sha=SHA, roster_sha256="0" * 64, launched_at="t", stage_kind="smoke",
                     inspect_limit=5, cell="none", snapshot_path="/x/s.json", app_token="c0ffee")
    assert one.cells == ("none",) and one.stage_kind == "smoke" and one.app == "tinycap-bcbw-gemma4e4b_it-aaaaaaa-none-c0ffee"
    assert Wr.cell_list(None) is None and Wr.cell_list("a, b") == ("a", "b") and Wr.cell_list(["a,b", "c"]) == ("a", "b", "c")
    # refusals: a non-runnable / unknown cell, a duplicate, an empty list — both in `refusals` and in `compose`
    (why,) = Wr.refusals(ROSTER, PINS, C.MODELS, wave="gemma4e4b_it", cell="none,safety_refusal__lora_r4")
    assert "--cell 'safety_refusal__lora_r4' is not a runnable cell" in why
    assert Wr.refusals(ROSTER, PINS, C.MODELS, wave="gemma4e4b_it", cell=["none", DERIVED]) == []
    with pytest.raises(SystemExit, match="is not a runnable cell"):
        Wr.compose(ROSTER, wave="gemma4e4b_it", sha=SHA, roster_sha256="0" * 64, launched_at="t", cell="nope")
    with pytest.raises(SystemExit, match="twice"):
        Wr.cell_list("none,none")
    with pytest.raises(SystemExit, match="names no cell"):
        Wr.cell_list(",")
    # the ladder's one-cell semantics are byte-identical: a single --cell still derives ladder-score and the probe cell
    sel = "/x/sel.json"
    cap = f"{C.inspect_task('bigcodebench_200')}=16384"
    lad = Wr.compose(ROSTER, wave="gemma4e4b_it", sha=SHA, roster_sha256="0" * 64, launched_at="t", cap_override=cap,
                     item_selection=sel, cell=[DERIVED], snapshot_path="/x/s.json")
    assert lad.stage_kind == "ladder-score" and lad.cells == (DERIVED,)
    assert lad.launcher_argv[lad.launcher_argv.index(C.FLAG_LADDER_PROBE_CELL) + 1] == DERIVED
    # the CLI spelling: --cell repeatable and comma-separated, read into the same list
    ns = Wr.build_parser().parse_args(["--wave", "w", "--sha", SHA, "--cell", "none,a", "--cell", "b"])
    assert Wr.cell_list(ns.cell) == ("none", "a", "b")
    # the earlier qwen waves compose the same serving path and, since the live probe PASSED, both legs
    q = Wr.compose(ROSTER, wave="qwen35_4b_main", sha=SHA, roster_sha256="0" * 64, launched_at="t")
    assert q.controls == ("bigcodebench_200", "bfcl_200") and q.serving_path_requested == "lora_native"


def test_the_b64_seed_waves_compose_natively_and_refuse_until_their_staging_is_recorded(Wr):
    """The nine b64 waves through the wrapper: the seed wave composes its eight cells (`none` first) on row
    qwen35_4b-f22s<s> under `--serving-path lora_native` with BOTH legs (BFCL runnable since the live probe PASS at
    80df464); the paired wave's composed cell can be launched alone (the roster's `none` is a free replicate the
    launch script never names); the Arditi wave launches the refusal-steer cell alone; the bank / parents prerequisites
    refuse every launch until they are recorded — the SHIPPED roster records them (staged and
    sha-verified 2026-09-23), so the refusal is exercised on a copy with the records withdrawn (the mechanism, not a defect)."""
    import copy
    met = ROSTER                                              # the shipped roster records every staging prerequisite
    unrec = copy.deepcopy(ROSTER)                             # ... withdrawn here: only the geometry line stays recorded
    for wname, w in unrec["waves"].items():
        if wname.startswith("qwen35_4b_s") and w.get("prerequisites"):
            w["prerequisites_met"] = [p for p in w["prerequisites_met"] if p.startswith("S6e")]
    for s_ in ("1", "2", "3"):
        seed, paired, arditi = f"qwen35_4b_s{s_}", f"qwen35_4b_s{s_}p", f"qwen35_4b_s{s_}_arditi"
        unmet = Wr.refusals(unrec, PINS, C.MODELS, wave=seed)
        # three unmet: the seed bank, the derived object's parents, and the Arditi extra the seed config enumerates too
        assert len(unmet) == 3 and all("prerequisite NOT met" in u for u in unmet) and any("/vol/out/qwen35_4b_base_eb64_s" + s_ in u for u in unmet)
        assert any("Arditi extra staged INSIDE" in u for u in unmet)
        assert Wr.refusals(met, PINS, C.MODELS, wave=seed) == [] == Wr.refusals(met, PINS, C.MODELS, wave=paired)
        L = Wr.compose(met, wave=seed, sha=SHA, roster_sha256="0" * 64, launched_at="t", snapshot_path="/x/s.json")
        assert L.row == f"qwen35_4b-f22s{s_}" and L.cells[0] == "none" and len(L.cells) == 8 and L.controls == ("bigcodebench_200", "bfcl_200")
        argv = list(L.launcher_argv)
        assert argv[argv.index("--serving-path") + 1] == "lora_native" and argv[argv.index("--model") + 1] == f"qwen35_4b-f22s{s_}"
        assert L.snapshot["constv2-mix-full__psr_ce_all"]["object_generation"] == "batch64"
        assert L.snapshot["constv2-mix-full__psr_ce_all"]["object_sha256"] == PINS["qwen35_4b"][f"base64s{s_}__constv2-mix-full__psr_ce_all"]["object_sha256"]
        canary = Wr.compose(met, wave=seed, sha=SHA, roster_sha256="0" * 64, launched_at="t", cell="none", snapshot_path="/x/s.json")
        assert canary.cells == ("none",) and canary.stage_kind == "wave"
        P = Wr.compose(met, wave=paired, sha=SHA, roster_sha256="0" * 64, launched_at="t", cell="constv2-mix-full__psr_ce_all_minus_ctl",
                       snapshot_path="/x/s.json")
        assert P.cells == ("constv2-mix-full__psr_ce_all_minus_ctl",) and P.row == f"qwen35_4b-f22s{s_}p" and P.wave_cells == ("none", "constv2-mix-full__psr_ce_all_minus_ctl")
        # the Arditi wave: runnable; the script launches the refusal-steer cell alone (its anchor is the seed wave's none)
        assert Wr.refusals(met, PINS, C.MODELS, wave=arditi) == []
        unmet_a = Wr.refusals(unrec, PINS, C.MODELS, wave=arditi)        # the extra AND the seed bank its standup registers beside it
        assert len(unmet_a) == 2 and all("prerequisite NOT met" in u for u in unmet_a)
        assert any("Arditi extra staged INSIDE" in u for u in unmet_a) and any("the seed bank" in u for u in unmet_a)
        A = Wr.compose(met, wave=arditi, sha=SHA, roster_sha256="0" * 64, launched_at="t", cell="arditi_allpos", snapshot_path="/x/s.json")
        assert A.cells == ("arditi_allpos",) and A.row == f"qwen35_4b-f22s{s_}" and A.wave_cells == ("none", "arditi_allpos")
        assert A.snapshot["arditi_allpos"]["object_sha256"] == "ba2212f0c2f9ffef2997bdd24831182052560cf980054bcee9cf95ea0db085b2"
        assert A.snapshot["arditi_allpos"]["object_generation"] == "final"



def test_a_plain_launch_may_serve_a_subset_of_the_waves_runnable_legs(Wr):
    """The instruct-direct order runs one cell on one leg (qwen35_4b_it difference LoRA on BFCL alone), so a
    plain launch takes `--legs` — a subset of the wave's runnable legs, in the roster's order; a leg the roster does not run on the wave
    refuses by name, and a ladder rerun never takes the flag (its legs are the fired caps)."""
    kw = dict(sha=SHA, roster_sha256="0" * 64, launched_at="2026-09-11T00:00:00Z")
    L = Wr.compose(ROSTER, wave="gemma4e4b_it", legs="bfcl_200", **kw)
    assert L.controls == ("bfcl_200",) and L.launcher_argv[L.launcher_argv.index("--controls") + 1] == "bfcl_200"
    both = Wr.compose(ROSTER, wave="gemma4e4b_it", legs="bfcl_200,bigcodebench_200", **kw)
    assert both.controls == ("bigcodebench_200", "bfcl_200")                       # the roster's order, never the flag's
    assert Wr.compose(ROSTER, wave="gemma4e4b_it", **kw).controls == ("bigcodebench_200", "bfcl_200")   # default: every runnable leg
    with pytest.raises(Wr.LaunchError, match="not a runnable leg"):
        Wr.compose(ROSTER, wave="llama8b_it", legs="bfcl_200", **kw)              # Llama runs no BFCL
    with pytest.raises(Wr.LaunchError, match="not a runnable leg"):
        Wr.compose(ROSTER, wave="gemma4e4b_it", legs="ifeval", **kw)
    with pytest.raises(Wr.LaunchError, match="plain-launch flag"):
        Wr.compose(ROSTER, wave="gemma4e4b_it", legs="bfcl_200", ladder_rerun=True, cap_override="inspect/bfcl_200=1024",
                   item_selection="/x/sel.json", cell="none", **kw)
    assert "--legs" in Wr.build_parser().format_help()


# --------------------------------------------------------------------------- #
# ROW MODE                                                                     #
# --------------------------------------------------------------------------- #
FULL_ROSTER_PATH = REPO / "configs/capeval/bcb_bfcl_full_roster.yaml"
FULL = W.load_roster(FULL_ROSTER_PATH)
ARDITI = "arditi_allpos"
LORA_MIX, LORA_CTL2 = "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce"
EIGHT = ("none", "prompted__safety_refusal", "prompted__si_constitution", "constv2-mix-full__psr_ce_all", "constv2ctl2-full__psr_ce_all",
         LORA_MIX, LORA_CTL2, DERIVED)


def _row(Wr, row, **kw):
    kw.setdefault("snapshot_path", "/x/snap.json")
    return Wr.compose_row(FULL, row=row, sha=SHA, roster_sha256="0" * 64, launched_at="2026-09-28T00:00:00Z", **kw)


def _after(argv, flag):
    argv = list(argv)
    return argv[argv.index(flag) + 1]


def test_compose_row_groups_the_rows_waves_into_one_launch_with_per_cell_suffixes(Wr):
    """The WAVES of a row = every roster wave whose model_row is the row; qwen s1 = qwen35_4b_s1 + qwen35_4b_s1_arditi (9 cells,
    `none` deduplicated and FIRST, the Arditi cell LAST with its snapshot AND suffix from ITS wave — the solo reader's launch_tag_of
    rule holds unchanged); the row dir is keyed by the PRIMARY wave's suffix; the argv = today's compose argv + the row flags."""
    assert Wr.row_waves(FULL, "qwen35_4b-f22s1") == ("qwen35_4b_s1", "qwen35_4b_s1_arditi")
    assert Wr.row_waves(FULL, "gemma4e4b-f22s1") == ("gemma4e4b_s1",) and Wr.row_waves(FULL, "nope-f22s1") == ()
    assert Wr.primary_wave(("qwen35_4b_s1", "qwen35_4b_s1_arditi"), row="r") == "qwen35_4b_s1"
    with pytest.raises(SystemExit, match="primary wave must be exactly one"):
        Wr.primary_wave(("qwen35_4b_s1", "qwen35_4b_s2"), row="r")
    L = _row(Wr, "qwen35_4b-f22s1", app_token="c0ffee")
    assert L.row_mode and L.row == "qwen35_4b-f22s1" and L.wave == "qwen35_4b_s1" and L.waves == ("qwen35_4b_s1", "qwen35_4b_s1_arditi")
    assert L.cells == EIGHT + (ARDITI,) == L.watcher_cells == L.wave_cells and L.cells[0] == "none" and L.cells.count("none") == 1
    assert L.suffix == "bcbw-qwen35_4b_s1-aaaaaaa-shared" and L.tag == "shared" and L.stage_kind == "wave"
    assert L.suffix_by_cell[ARDITI] == "bcbw-qwen35_4b_s1_arditi-aaaaaaa-shared"
    assert {L.suffix_by_cell[c] for c in EIGHT} == {"bcbw-qwen35_4b_s1-aaaaaaa-shared"}
    assert L.row_dir == f"{C.VOL_GEN_ROOT}/rows/qwen35_4b-f22s1/bcbw-qwen35_4b_s1-aaaaaaa-shared" == Wr.row_dir_for("qwen35_4b-f22s1", L.suffix)
    assert not L.row_dir.startswith(f"{C.VOL_GEN_ROOT}/native/")           # outside native/: never a store to the mirror or the reader
    assert L.app == "tinycap-bcbw-qwen35_4b_s1-aaaaaaa-row-c0ffee" and L.env == {C.MODAL_APP_NAME_ENV: L.app}
    assert L.controls == ("bigcodebench_full", "bfcl_full") and L.serving_path_requested == "lora_native"
    assert not L.ladder and not L.certify and L.repeat_cells == ("none",) and L.concurrency_k == 1 and L.cache_expectation == "cold" and L.cache_persist
    assert L.negative_control_cell is None and not L.warm_only and L.row_unit == "row:qwen35_4b-f22s1"
    # the snapshot: one block per cell, the Arditi block from ITS wave, `none` from the primary wave
    assert set(L.snapshot) == set(L.cells) and all(W.snapshot_problems(b) == [] for b in L.snapshot.values())
    assert L.snapshot[ARDITI]["wave"] == "qwen35_4b_s1_arditi" and L.snapshot["none"]["wave"] == "qwen35_4b_s1" and L.snapshot[LORA_MIX]["wave"] == "qwen35_4b_s1"
    assert L.snapshot[ARDITI]["object_sha256"] and L.snapshot["none"]["object_sha256"] is None
    # the argv: today's compose prefix, then the row flags in the plan's order; the per-cell suffix map parses back to suffix_by_cell
    argv = list(L.launcher_argv)
    assert argv[:6] == ["--sha", SHA, "--app-name", L.app, "--pf", "--pass1"]
    assert _after(argv, "--sampling") == "protocol" and _after(argv, "--model") == "qwen35_4b-f22s1" and _after(argv, "--modes") == "off"
    assert _after(argv, "--cells") == ",".join(L.cells) and _after(argv, "--serving-path") == "lora_native"
    assert _after(argv, "--controls") == "bigcodebench_full,bfcl_full" and _after(argv, "--out-suffix") == L.suffix
    assert _after(argv, "--roster-snapshot") == "/x/snap.json" and "--inspect-limit" not in argv and C.FLAG_CAP_OVERRIDE not in argv
    i = argv.index("--row")
    assert i > argv.index("--roster-snapshot")
    assert argv[i:] == ["--row", "--repeat-cells", "none", "--concurrency-k", "1", "--cache-expectation", "cold",
                        "--out-suffix-by-cell", argv[-1]]
    assert json.loads(argv[-1]) == L.suffix_by_cell and "--no-cache-persist" not in argv and "--row-forecast-h" not in argv and L.row_forecast_h is None
    # with the row's meter claim (the CLI's --forecast-h), the forecast rides to the harness BEFORE the suffix map (integration)
    Lf = _row(Wr, "qwen35_4b-f22s1", app_token="c0ffee", row_forecast_h=4.25)
    fa = list(Lf.launcher_argv)
    assert fa[fa.index("--row"):] == ["--row", "--repeat-cells", "none", "--concurrency-k", "1", "--cache-expectation", "cold",
                                      "--row-forecast-h", "4.25", "--out-suffix-by-cell", fa[-1]] and Lf.row_forecast_h == 4.25
    # the flag spellings are the CONTRACTS' — the harness binds its ROW_*_FLAG names to the same constants (tests/test_tinycap_row_launch_grammar.py)
    assert (Wr.FLAG_ROW, Wr.FLAG_REPEAT_CELLS, Wr.FLAG_NEGATIVE_CONTROL_CELL, Wr.FLAG_WARM_CACHE_ONLY, Wr.FLAG_CONCURRENCY_K,
            Wr.FLAG_CACHE_EXPECTATION, Wr.FLAG_NO_CACHE_PERSIST, Wr.FLAG_OUT_SUFFIX_BY_CELL, Wr.FLAG_ROW_FORECAST_H, Wr.WATCHER_FLAG_ROW_DIR) == (
        C.FLAG_ROW, C.FLAG_REPEAT_CELLS, C.FLAG_NEGATIVE_CONTROL_CELL, C.FLAG_WARM_CACHE_ONLY, C.FLAG_CONCURRENCY_K, C.FLAG_CACHE_EXPECTATION,
        C.FLAG_NO_CACHE_PERSIST, C.FLAG_OUT_SUFFIX_BY_CELL, C.FLAG_ROW_FORECAST_H, "--row-dir") == (
        "--row", "--repeat-cells", "--negative-control-cell", "--warm-cache-only", "--concurrency-k", "--cache-expectation",
        "--no-cache-persist", "--out-suffix-by-cell", "--row-forecast-h", "--row-dir")
    assert L.modal_argv("/venv/bin/python")[:6] == ["/venv/bin/python", "-m", "modal", "run", "--detach", "scripts/modal/tinycap_served.py"]
    # a single-wave row (gemma: 8 cells) and the nemotron row compose the same way; every per-wave Launch is untouched (row_mode False)
    g = _row(Wr, "gemma4e4b-f22s1")
    assert g.waves == ("gemma4e4b_s1",) and g.cells == EIGHT and set(g.suffix_by_cell.values()) == {"bcbw-gemma4e4b_s1-aaaaaaa-shared"}
    n = _row(Wr, "nemotron3nano4b-f22s1")
    assert n.cells == EIGHT and n.row_unit == "row:nemotron3nano4b-f22s1" and n.controls == ("bigcodebench_full", "bfcl_full")
    per_wave = Wr.compose(FULL, wave="gemma4e4b_s1", sha=SHA, roster_sha256="0" * 64, launched_at="t")
    assert not per_wave.row_mode and per_wave.waves == () and per_wave.suffix_by_cell is None and per_wave.row_dir is None and per_wave.tag == ""
    assert "--row" not in per_wave.launcher_argv
    # the inline snapshot rides as JSON, parseable by the roster reader
    inline = _row(Wr, "gemma4e4b-f22s1", snapshot_path=None)
    assert W.parse_snapshot_arg(_after(inline.launcher_argv, "--roster-snapshot")) == inline.snapshot
    # --legs narrows the row's legs (the one-leg launch), in the roster's order; a leg the row does not run refuses by name
    assert _row(Wr, "gemma4e4b-f22s1", legs="bfcl_full").controls == ("bfcl_full",)
    with pytest.raises(SystemExit, match="not a runnable leg of row"):
        _row(Wr, "gemma4e4b-f22s1", legs="bfcl_200")


def test_the_it_row_serves_its_own_none_as_a_cell(Wr):
    """An instruct-checkpoint row is a shareable row like any other — its LoRA cells AND its own `none` (the served
    checkpoint's base route) in one container; the launch set files that `none` as it_none_comparison (derive_launch_set_shared), never an anchor."""
    L = _row(Wr, "qwen35_4b-f22it", app_token="c0ffee")
    assert L.waves == ("qwen35_4b_it",) and L.cells == ("none", LORA_MIX, LORA_CTL2, DERIVED) and L.repeat_cells == ("none",)
    assert L.suffix == "bcbw-qwen35_4b_it-aaaaaaa-shared" and L.row_dir.endswith("/rows/qwen35_4b-f22it/bcbw-qwen35_4b_it-aaaaaaa-shared")
    assert L.snapshot["none"]["object_sha256"] is None and all(L.snapshot[c]["object_sha256"] for c in (LORA_MIX, LORA_CTL2, DERIVED))
    assert _after(L.launcher_argv, "--model") == "qwen35_4b-f22it" and _after(L.launcher_argv, "--cells").startswith("none,")


def test_the_row_scoped_subset_relaunch_grammar_keeps_the_per_cell_suffix_and_the_relaunch_tag(Wr):
    """`--row --cells <c1,…> --tag shared-r<N>` — the cells must be runnable cells of the row; `none` MAY be included;
    each cell's suffix is out_suffix(wave_of_cell, sha, 'shared-r<N>'); the row dir is the r<N> primary suffix's (a distinct reservation
    identity and watcher dir); the repeat defaults to `none` only when the subset serves it."""
    r2 = _row(Wr, "qwen35_4b-f22s1", cells=f"{ARDITI},{LORA_MIX}", tag="shared-r2", app_token="c0ffee")
    assert r2.cells == (ARDITI, LORA_MIX) and r2.stage_kind == "wave" and r2.tag == "shared-r2" and Wr.row_relaunch_n("shared-r2") == 2
    assert r2.suffix_by_cell == {ARDITI: "bcbw-qwen35_4b_s1_arditi-aaaaaaa-shared-r2", LORA_MIX: "bcbw-qwen35_4b_s1-aaaaaaa-shared-r2"}
    assert r2.suffix == "bcbw-qwen35_4b_s1-aaaaaaa-shared-r2" and r2.row_dir.endswith("/rows/qwen35_4b-f22s1/bcbw-qwen35_4b_s1-aaaaaaa-shared-r2")
    assert r2.app == "tinycap-bcbw-qwen35_4b_s1-aaaaaaa-row-2cells-c0ffee" and r2.wave_cells == EIGHT + (ARDITI,)
    assert r2.repeat_cells == () and "--repeat-cells" not in r2.launcher_argv          # no `none` served: nothing to repeat
    assert set(r2.snapshot) == {ARDITI, LORA_MIX} and _after(r2.launcher_argv, "--cells") == f"{ARDITI},{LORA_MIX}"
    with_none = _row(Wr, "qwen35_4b-f22s1", cells=[f"{LORA_MIX},none"], tag="shared-r3")
    assert with_none.cells == ("none", LORA_MIX) and with_none.repeat_cells == ("none",) and with_none.suffix.endswith("-shared-r3")
    # the meter identity of a relaunch is `row:<row>` under the r<N> primary suffix
    (m,) = Wr.meter_argvs(r2, ledger="/l", forecast_h=1.0, model_key="qwen35_4b")
    assert _after(m, "--unit") == "row:qwen35_4b-f22s1" and _after(m, "--suffix") == "bcbw-qwen35_4b_s1-aaaaaaa-shared-r2"
    # refusals: a cell outside the row, a duplicate, a tag outside the grammar (r0, a bare word), an explicit repeat the subset does not serve
    with pytest.raises(SystemExit, match="not a runnable cell of row"):
        _row(Wr, "qwen35_4b-f22s1", cells="nope", tag="shared-r2")
    with pytest.raises(SystemExit, match="twice"):
        _row(Wr, "qwen35_4b-f22s1", cells="none,none", tag="shared-r2")
    for tag in ("shared-r0", "r2", "sharedr2", "shared-", "smoke", "shared-rep2", "shared-warm", "warm-r2"):
        assert Wr.row_tag_problem(tag) and Wr.ROW_TAG_RE.match(tag) is None
        with pytest.raises(SystemExit, match="not a row-lane tag"):
            _row(Wr, "gemma4e4b-f22s1", tag=tag)
    for tag in ("shared", "shared-r1", "shared-r12", "shared-negctl", "warm", "warm2", "warm3", "warm10"):
        assert Wr.row_tag_problem(tag) is None
    # a warm probe RE-RUN is `warm<N>`, N >= 2: `warm1` would alias the first probe, a leading zero would alias `warm2`
    for tag in ("warm1", "warm0", "warm02", "warm-2", "warm2x"):
        assert Wr.row_tag_problem(tag) and Wr.ROW_TAG_RE.match(tag) is None
    assert all(Wr.is_warm_tag(t) for t in ("warm", "warm2", "warm3", "warm10"))
    assert not any(Wr.is_warm_tag(t) for t in ("shared", "shared-negctl", "shared-r1", "warm1", ""))
    with pytest.raises(SystemExit, match="not one of the launched cells"):
        _row(Wr, "qwen35_4b-f22s1", cells=f"{ARDITI},{LORA_MIX}", tag="shared-r2", repeat_cells="none")
    assert Wr.repeat_suffix("bcbw-qwen35_4b_s1-aaaaaaa-shared") == "bcbw-qwen35_4b_s1-aaaaaaa-shared-rep2"


def test_the_repeat_cell_the_negative_control_and_the_warm_probe_compose_their_own_flags_tags_and_stage_kinds(Wr):
    """§1.6.1 / §1.3 / §1.5: the repeat rides `--repeat-cells` (default `none`; '' = none; must be a launched cell); the negative-identity
    control is a SMOKE (`--tag shared-negctl`, a fitted launched cell, `--inspect-limit` required, `--no-cache-persist` forced, stage `smoke`);
    the warm probe (`--tag warm --warm-cache-only`, suffix …-warm) is the whole row's served set with no driver, no repeat, stage `wave`."""
    none_rep = _row(Wr, "gemma4e4b-f22s1", repeat_cells="")
    assert none_rep.repeat_cells == () and "--repeat-cells" not in none_rep.launcher_argv
    two = _row(Wr, "gemma4e4b-f22s1", repeat_cells=("none", LORA_MIX))
    assert two.repeat_cells == ("none", LORA_MIX) and _after(two.launcher_argv, "--repeat-cells") == f"none,{LORA_MIX}"
    with pytest.raises(SystemExit, match="not one of the launched cells"):
        _row(Wr, "gemma4e4b-f22s1", repeat_cells=ARDITI)
    # the negative control
    neg = _row(Wr, "gemma4e4b-f22s1", tag="shared-negctl", negative_control_cell=LORA_CTL2, inspect_limit=5, app_token="c0ffee")
    assert neg.stage_kind == "smoke" and neg.suffix == "bcbw-gemma4e4b_s1-aaaaaaa-shared-negctl" and not neg.cache_persist
    assert neg.negative_control_cell == LORA_CTL2 and neg.cells == EIGHT and neg.app == "tinycap-bcbw-gemma4e4b_s1-aaaaaaa-row-negctl-c0ffee"
    argv = list(neg.launcher_argv)
    assert _after(argv, "--inspect-limit") == "5" and _after(argv, "--negative-control-cell") == LORA_CTL2 and "--no-cache-persist" in argv
    assert argv.index("--negative-control-cell") > argv.index("--row") and "--warm-cache-only" not in argv
    assert set(neg.suffix_by_cell.values()) == {"bcbw-gemma4e4b_s1-aaaaaaa-shared-negctl"} and neg.row_dir.endswith("-shared-negctl")
    with pytest.raises(SystemExit, match="files under --tag shared-negctl"):
        _row(Wr, "gemma4e4b-f22s1", tag="shared", negative_control_cell=LORA_CTL2, inspect_limit=5)
    with pytest.raises(SystemExit, match="is a SMOKE: pass --inspect-limit"):
        _row(Wr, "gemma4e4b-f22s1", tag="shared-negctl", negative_control_cell=LORA_CTL2)
    with pytest.raises(SystemExit, match="not a fitted cell"):
        _row(Wr, "gemma4e4b-f22s1", tag="shared-negctl", negative_control_cell="prompted__si_constitution", inspect_limit=5)
    with pytest.raises(SystemExit, match="not one of the launched cells"):
        _row(Wr, "gemma4e4b-f22s1", tag="shared-negctl", negative_control_cell=LORA_CTL2, inspect_limit=5, cells=f"none,{LORA_MIX}")
    with pytest.raises(SystemExit, match="negative-control smoke's tag"):
        _row(Wr, "gemma4e4b-f22s1", tag="shared-negctl")
    # any --inspect-limit run never persists the cache (a partial shape is not the row's)
    assert not _row(Wr, "gemma4e4b-f22s1", inspect_limit=5).cache_persist and "--no-cache-persist" in _row(Wr, "gemma4e4b-f22s1", inspect_limit=5).launcher_argv
    assert not _row(Wr, "gemma4e4b-f22s1", cache_persist=False).cache_persist
    # the warm probe
    warm = _row(Wr, "qwen35_4b-f22s1", tag="warm", warm_only=True, cache_expectation="cold", app_token="c0ffee")
    assert warm.stage_kind == "wave" and warm.warm_only and warm.repeat_cells == () and warm.cells == EIGHT + (ARDITI,) and warm.cache_persist
    assert warm.suffix == "bcbw-qwen35_4b_s1-aaaaaaa-warm" and warm.app == "tinycap-bcbw-qwen35_4b_s1-aaaaaaa-row-warm-c0ffee" and warm.tag == "warm"
    argv = list(warm.launcher_argv)
    assert "--warm-cache-only" in argv and "--repeat-cells" not in argv and "--negative-control-cell" not in argv and "--no-cache-persist" not in argv
    assert warm.suffix_by_cell[ARDITI] == "bcbw-qwen35_4b_s1_arditi-aaaaaaa-warm"
    with pytest.raises(SystemExit, match="files under --tag warm"):
        _row(Wr, "qwen35_4b-f22s1", tag="shared", warm_only=True)
    with pytest.raises(SystemExit, match="WHOLE served set"):
        _row(Wr, "qwen35_4b-f22s1", tag="warm", warm_only=True, cells="none")
    with pytest.raises(SystemExit, match="runs no driver"):
        _row(Wr, "qwen35_4b-f22s1", tag="warm", warm_only=True, inspect_limit=5)
    with pytest.raises(SystemExit, match="warm probe's tag"):
        _row(Wr, "qwen35_4b-f22s1", tag="warm")
    # a re-probe: `--tag warm2 --warm-cache-only` files under …-warm2 — its own claim and row dir, never the done `-warm` row's
    warm2 = _row(Wr, "qwen35_4b-f22s1", tag="warm2", warm_only=True, cache_expectation="warm", app_token="c0ffee")
    assert warm2.suffix == "bcbw-qwen35_4b_s1-aaaaaaa-warm2" and warm2.tag == "warm2" and warm2.stage_kind == "wave"
    assert warm2.app == "tinycap-bcbw-qwen35_4b_s1-aaaaaaa-row-warm2-c0ffee" and "--warm-cache-only" in warm2.launcher_argv
    assert _after(warm2.launcher_argv, "--cache-expectation") == "warm" and "--repeat-cells" not in warm2.launcher_argv
    assert warm2.suffix_by_cell[ARDITI] == "bcbw-qwen35_4b_s1_arditi-aaaaaaa-warm2"
    with pytest.raises(SystemExit, match="warm probe's tag"):
        _row(Wr, "qwen35_4b-f22s1", tag="warm3")
    with pytest.raises(SystemExit, match="not a row-lane tag"):
        _row(Wr, "qwen35_4b-f22s1", tag="warm1", warm_only=True)
    with pytest.raises(SystemExit, match="two different launches"):
        _row(Wr, "gemma4e4b-f22s1", tag="shared-negctl", warm_only=True, negative_control_cell=LORA_CTL2, inspect_limit=5)
    # the cache expectation is declared, K is >= 1, an explicit stage kind must agree with the tag's
    k2 = _row(Wr, "gemma4e4b-f22s1", concurrency_k=2, cache_expectation="warm")
    assert _after(k2.launcher_argv, "--concurrency-k") == "2" and _after(k2.launcher_argv, "--cache-expectation") == "warm" and k2.concurrency_k == 2
    with pytest.raises(SystemExit, match="must be >= 1"):
        _row(Wr, "gemma4e4b-f22s1", concurrency_k=0)
    with pytest.raises(SystemExit, match="not in"):
        _row(Wr, "gemma4e4b-f22s1", cache_expectation="hot")
    assert _row(Wr, "gemma4e4b-f22s1", stage_kind="wave").stage_kind == "wave"
    with pytest.raises(SystemExit, match="disagrees with the row tag"):
        _row(Wr, "gemma4e4b-f22s1", stage_kind="smoke")
    assert Wr.row_stage_kind("shared") == Wr.row_stage_kind("shared-r2") == Wr.row_stage_kind("warm") == "wave" and Wr.row_stage_kind("shared-negctl") == "smoke"


def test_a_row_launch_makes_one_meter_claim_and_the_watcher_reads_the_row_dir(Wr, meter):
    """ONE reservation per row — `--unit row:<row>` under the row's primary suffix with the WHOLE row's forecast; never a
    per-cell claim (K overlapping drivers would sum the same GPU minute K times). The watcher gets `--row-dir <row dir>`."""
    L = _row(Wr, "qwen35_4b-f22s1")
    ms = Wr.meter_argvs(L, ledger="~/ledger.jsonl", forecast_h=4.25, model_key="qwen35_4b")
    assert len(ms) == 1 and ms[0][1] == "scripts/bcb_bfcl_200_meter.py"
    a = meter.build_parser().parse_args(ms[0][2:])
    assert (a.wave, a.suffix, a.unit, a.stage_kind, a.forecast_h, a.ledger) == ("qwen35_4b_s1", L.suffix, "row:qwen35_4b-f22s1", "wave", 4.25, "~/ledger.jsonl")
    assert Wr.unit_of(ms[0]) == L.row_unit == Wr.row_unit_of("qwen35_4b-f22s1") and Wr.ROW_UNIT_PREFIX == "row:"
    assert meter.reservation_id(a.wave, a.suffix, a.unit) == f"qwen35_4b_s1/{L.suffix}/row:qwen35_4b-f22s1"      # the unit may carry ':'
    with pytest.raises(SystemExit, match="WHOLE row's H200-hours"):
        Wr.meter_argvs(L, ledger="l", forecast_h=None, model_key="qwen35_4b")
    # a smoke / warm probe / relaunch is its own suffix -> its own reservation identity, still ONE claim
    neg = _row(Wr, "gemma4e4b-f22s1", tag="shared-negctl", negative_control_cell=LORA_CTL2, inspect_limit=5)
    (m,) = Wr.meter_argvs(neg, ledger="l", forecast_h=0.25, model_key="gemma4e4b")
    a = meter.build_parser().parse_args(m[2:])
    assert (a.suffix, a.unit, a.stage_kind, a.forecast_h) == ("bcbw-gemma4e4b_s1-aaaaaaa-shared-negctl", "row:gemma4e4b-f22s1", "smoke", 0.25)
    warm = _row(Wr, "qwen35_4b-f22s1", tag="warm", warm_only=True)
    (m,) = Wr.meter_argvs(warm, ledger="l", forecast_h=0.25, model_key="qwen35_4b")
    assert _after(m, "--stage-kind") == "wave" and _after(m, "--suffix") == "bcbw-qwen35_4b_s1-aaaaaaa-warm" and _after(m, "--unit") == "row:qwen35_4b-f22s1"
    # the release of a row claim names the same identity
    (rel,) = Wr.release_argvs(L, ledger="~/l.jsonl", units=[L.row_unit], why="nothing launched")
    a = meter.build_parser().parse_args(rel[2:])
    assert (a.release, a.wave, a.suffix, a.unit) == ("nothing launched", "qwen35_4b_s1", L.suffix, "row:qwen35_4b-f22s1")
    # the watcher: the per-cell flags as today PLUS --row-dir; a per-wave launch never carries --row-dir
    w = Wr.watcher_argv("aaaaaaa-deadbeef", L, modal_py="/m/python")
    assert w[:2] == ["/m/python", "scripts/tinycap_sandbox_watch.py"] and _after(w, "--run-id") == "aaaaaaa-deadbeef"
    assert _after(w, "--suffix") == L.suffix and _after(w, "--model") == "qwen35_4b-f22s1" and _after(w, "--cells") == ",".join(L.cells)
    assert _after(w, "--serving-path") == "lora_native" and _after(w, "--row-dir") == L.row_dir and w.index("--row-dir") < w.index("--poll")
    per_wave = Wr.compose(FULL, wave="gemma4e4b_s1", sha=SHA, roster_sha256="0" * 64, launched_at="t")
    assert "--row-dir" not in Wr.watcher_argv("x", per_wave)
    # the row's per-cell meter claims of the per-wave path are untouched
    ms = Wr.meter_argvs(per_wave, ledger="l", forecast_h=0.8, model_key="gemma4e4b")
    assert [Wr.unit_of(m) for m in ms] == list(per_wave.cells)


def test_row_refusals_and_the_row_cli_are_exclusive_with_the_per_wave_path(Wr, tmp_path):
    """`row_refusals`: a roster failing its rules, an unknown row, waves whose legs disagree (a row engine serves ONE leg set), an
    unmet prerequisite, a `--cells` cell outside the row, a tag outside the grammar. The CLI: exactly one of --row / --wave; --row takes none
    of the per-wave flags and refuses before any subprocess."""
    assert Wr.row_refusals(FULL, PINS, C.MODELS, row="qwen35_4b-f22s1") == []
    assert Wr.row_refusals(FULL, PINS, C.MODELS, row="qwen35_4b-f22it", cells="none,constv2-mix-full__lora_r4_ce", tag="shared-r2") == []
    assert Wr.row_refusals(FULL, PINS, C.MODELS, row="qwen35_4b-f22s1", tag="warm") == [] and Wr.row_refusals(FULL, PINS, C.MODELS, row="qwen35_4b-f22s1", tag="shared-warm") != []
    (why,) = Wr.row_refusals(FULL, PINS, C.MODELS, row="nope-f22s1")
    assert "named by no wave" in why
    whys = Wr.row_refusals(FULL, PINS, C.MODELS, row="gemma4e4b-f22s1", cells="none,zzz", tag="nope")
    assert len(whys) == 2 and "--cells 'zzz' is not a runnable cell of row" in whys[0] and "not a row-lane tag" in whys[1]
    legs = copy.deepcopy(FULL)
    legs["waves"]["qwen35_4b_s1_arditi"]["legs"]["bfcl_full"] = {"status": "blocked", "reason": "synthetic", "unblock": "synthetic"}
    assert W.validate(legs, PINS, C.MODELS) == []
    (why,) = Wr.row_refusals(legs, PINS, C.MODELS, row="qwen35_4b-f22s1")
    assert "qwen35_4b_s1_arditi" in why and "ONE leg set" in why
    with pytest.raises(SystemExit, match="ONE leg set|runs legs"):
        Wr.compose_row(legs, row="qwen35_4b-f22s1", sha=SHA, roster_sha256="0" * 64, launched_at="t", snapshot_path="/x")
    unmet = copy.deepcopy(FULL)
    unmet["waves"]["qwen35_4b_s1_arditi"]["prerequisites"] = ["synthetic prerequisite"]
    unmet["waves"]["qwen35_4b_s1_arditi"]["prerequisites_met"] = []
    if W.validate(unmet, PINS, C.MODELS) == []:
        assert any("prerequisite NOT met" in w for w in Wr.row_refusals(unmet, PINS, C.MODELS, row="qwen35_4b-f22s1"))
    bad = copy.deepcopy(FULL)
    bad["waves"]["gemma4e4b_s1"]["entries"]["const_lora"]["object"]["sha256"] = "0" * 64
    assert all(p.startswith("roster: ") for p in Wr.row_refusals(bad, PINS, C.MODELS, row="gemma4e4b-f22s1"))
    # the CLI
    ns = Wr.build_parser().parse_args(["--row", "qwen35_4b-f22s1", "--sha", SHA, "--cells", "none,a", "--cells", "b", "--tag", "shared-r2",
                                       "--repeat-cells", "none", "--concurrency-k", "2", "--cache-expectation", "warm", "--no-cache-persist",
                                       "--standup-forecast-h", "0.25"])
    assert (ns.row, ns.wave, Wr.cell_list(ns.cells), ns.tag, ns.concurrency_k, ns.cache_expectation, ns.no_cache_persist, ns.standup_forecast_h) == (
        "qwen35_4b-f22s1", None, ("none", "a", "b"), "shared-r2", 2, "warm", True, 0.25)
    assert Wr.build_parser().parse_args(["--wave", "w", "--sha", SHA]).row is None and Wr.build_parser().parse_args(["--row", "r", "--sha", SHA]).repeat_cells is None
    with pytest.raises(SystemExit):
        Wr.build_parser().parse_args(["--row", "r", "--sha", SHA, "--cache-expectation", "hot"])
    both = Wr.build_parser().parse_args(["--row", "r", "--wave", "w", "--sha", SHA])
    assert any("exactly one of --row" in p for p in Wr.row_mode_problems(both))
    neither = Wr.build_parser().parse_args(["--sha", SHA])
    assert any("exactly one of --row" in p for p in Wr.row_mode_problems(neither))
    mixed = Wr.build_parser().parse_args(["--row", "r", "--sha", SHA, "--cell", "x", "--ladder-rerun", "--cap-override", "a=1"])
    assert [p.split(" ")[0] for p in Wr.row_mode_problems(mixed)] == ["--cell", "--ladder-rerun", "--cap-override"]
    wave_with_row_flags = Wr.build_parser().parse_args(["--wave", "w", "--sha", SHA, "--cells", "a", "--warm-cache-only", "--concurrency-k", "2"])
    assert [p.split(" ")[0] for p in Wr.row_mode_problems(wave_with_row_flags)] == ["--cells", "--warm-cache-only", "--concurrency-k"]
    assert Wr.row_mode_problems(Wr.build_parser().parse_args(["--row", "r", "--sha", SHA])) == []
    # main refuses a bad row / tag before anything else (no Modal, no ledger, no subprocess)
    log_dir = str(tmp_path / "logs")
    with pytest.raises(SystemExit, match="REFUSED"):
        Wr.main(["--row", "nope-f22s1", "--sha", SHA, "--roster", str(FULL_ROSTER_PATH), "--dry-run", "--log-dir", log_dir])
    with pytest.raises(SystemExit, match="not a row-lane tag"):
        Wr.main(["--row", "gemma4e4b-f22s1", "--sha", SHA, "--roster", str(FULL_ROSTER_PATH), "--tag", "smoke", "--dry-run", "--log-dir", log_dir])
    with pytest.raises(SystemExit, match="exactly one of --row"):
        Wr.main(["--row", "gemma4e4b-f22s1", "--wave", "gemma4e4b_s1", "--sha", SHA, "--roster", str(FULL_ROSTER_PATH), "--dry-run", "--log-dir", log_dir])
    with pytest.raises(SystemExit, match="per-wave flag"):
        Wr.main(["--row", "gemma4e4b-f22s1", "--sha", SHA, "--roster", str(FULL_ROSTER_PATH), "--cell", "none", "--dry-run", "--log-dir", log_dir])
    assert not (tmp_path / "logs").exists()


def test_a_local_spawn_failure_after_the_meter_claim_releases_the_row_claim(Wr, tmp_path, monkeypatch):
    """Code review R2-3 (MEDIUM): `go` admits `row:<row>` and only THEN opens its log and calls Popen; a missing /
    invalid `--modal-py` raised straight out of Popen, skipped every release, and left the row's forecast hours
    committed under the ceiling with nothing launched. A REAL Popen on a missing executable now refuses by name
    (`SPAWN FAILED before any child`), releases the admitted claim — a spawn that provably did not happen: no child
    exists — records the failure in the launch log, and reports a failed release rather than raising over it."""
    calls = []

    def runner(argv, **kw):
        calls.append(list(argv))
        if argv[1] == "scripts/bcb_bfcl_200_meter.py" and "--release" in argv:
            return SimpleNamespace(returncode=0, stdout=f"[meter] released {Wr.unit_of(argv)}\n", stderr="")
        if argv[1] == "scripts/bcb_bfcl_200_meter.py":
            return SimpleNamespace(returncode=0, stdout=f"[meter] {Wr.unit_of(argv)} admitted\n", stderr="")
        raise AssertionError(argv)
    monkeypatch.setattr(Wr.subprocess, "run", runner)                      # the meter is doubled; Popen is REAL
    missing = str(tmp_path / "no-such-venv" / "bin" / "python")
    base = ["--row", "gemma4e4b-f22s1", "--sha", SHA, "--roster", str(FULL_ROSTER_PATH), "--forecast-h", "3.5", "--log-dir", str(tmp_path / "logs"),
            "--ledger", str(tmp_path / "ledger.jsonl"), "--modal-py", missing, "--no-watcher"]
    with pytest.raises(SystemExit, match="SPAWN FAILED before any child") as e:
        Wr.main(base)
    msg = str(e.value)
    assert "FileNotFoundError" in msg and f"{missing} is missing" in msg and "nothing spawned" in msg
    assert "meter claims: released row:gemma4e4b-f22s1 (rc 0)" in msg and "nothing is left running or claimed" in msg
    meters = [a for a in calls if a[1] == "scripts/bcb_bfcl_200_meter.py"]
    assert [Wr.unit_of(a) for a in meters if "--release" not in a] == ["row:gemma4e4b-f22s1"]         # ONE claim admitted …
    (rel,) = [a for a in meters if "--release" in a]                                                  # … and released once
    assert Wr.unit_of(rel) == "row:gemma4e4b-f22s1" and rel[rel.index("--release") + 1].startswith("never ran: the launcher could not be spawned (FileNotFoundError")
    logs = [p for p in (tmp_path / "logs").glob("*.log") if not p.name.endswith("_watch.log")]
    assert len(logs) == 1 and "== spawn failed: never ran: the launcher could not be spawned" in logs[0].read_text()
    # a failed release is RECORDED in the refusal, never raised over it
    def bad_release(argv, **kw):
        if "--release" in argv:
            calls.append(list(argv))
            return SimpleNamespace(returncode=1, stdout="", stderr="nothing to release")
        return runner(argv, **kw)
    monkeypatch.setattr(Wr.subprocess, "run", bad_release)
    with pytest.raises(SystemExit, match=r"SPAWN FAILED .* released row:gemma4e4b-f22s1 \(rc 1\)"):
        Wr.main(base)
    # an unopenable launch log is the same provable non-spawn: released before any child
    calls.clear()
    monkeypatch.setattr(Wr.subprocess, "run", runner)
    monkeypatch.setattr(Wr.Path, "open", lambda self, *a, **k: (_ for _ in ()).throw(PermissionError(13, "read-only log dir")) if self.suffix == ".log" and "_watch" not in self.name else open(self, *a, **k))
    with pytest.raises(SystemExit, match="SPAWN FAILED before any child: never ran: the launch log .* could not be opened .PermissionError") as e:
        Wr.main(base)
    assert "released row:gemma4e4b-f22s1 (rc 0)" in str(e.value) and [Wr.unit_of(a) for a in calls if "--release" in a] == ["row:gemma4e4b-f22s1"]

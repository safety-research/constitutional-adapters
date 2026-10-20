"""petri-cells (the proxy-side plumbing): the Petri proxy takes the view's cells on positional 5 (the pass stays 4), exports
F22_REGISTRY_CELLS before registration, and hands the named cells to the standup as mt_serving_up --cell (serve `none` implicitly +
exactly the named cells). The native core records WHAT WAS NAMED — `cells_requested_explicit` (roster spelling) and
`cells_requested_explicit_harness` — into the pre-launch record, the final cells_served.json and endpoint.json, and the grading host's
drive (petri_run) freezes exactly those canonical legs, bound to the live attempt through the sibling endpoint.json's
standup_attempt_id, with the controls twin only when `none` is declared (a steered-only view: no `none`, no `none__lora0`). A
baseline-wave proxy is never stood up with a partial --cell (its legs are the whole baseline): the plan refuses by name. Synthetic
records, placeholder ids."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402

sys.path.insert(0, str(REPO / "reproduce" / "lib"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import stage as ST  # noqa: E402
import petri_run as PR  # noqa: E402
import test_lora_serving_up as E  # noqa: E402  (the gemma campaign harness: fake bank, scripted upstream, the REAL bring_up_lora)
from sisteer.capeval import manifest as cm  # noqa: E402
from sisteer.capeval.petri import registry_adapter as ra  # noqa: E402
from sisteer.capeval.serving import native_standup as NS  # noqa: E402
from sisteer.capeval.serving import vllm_server as vs  # noqa: E402
from sisteer.registry import synth  # noqa: E402

KEY = "qwen35_4b_demo"; SHA = synth.code_sha("a"); AB = ["cluster_a", "cluster_b"]   # the front-door model (configured, not run): the SINGLE-PHASE recipe (one wave battery + two dose rungs, no u0); a placeholder run code_sha is refused: synth's derived sha
PHASED = "phased_demo"; VIEW = "s2"                                     # a SYNTHETIC row that RUNS IN PHASES (u0 + s2 + d0p75_relu) on a tmp copy of the table: the steered-wave proxy's subject
RELU = ["constv2ctl2-full__psr_ce_all", "constv2-mix-full__psr_ce_all"]; CSV = ",".join(RELU)
DIST = "constv2-mix-full__psr_ce_all_minus_ctl_distilled"              # the table's difference_relu cell (the one cell of the d0p75_relu view)
PAPER_KEY, PAPER_VIEW = "qwen35_4b", "s1"                              # a PAPER model with a committed wave config, roster block and pins (the registration test)
RELU_H = ["si_constv2ctl2_full__psr_ce_all", "si_constv2_mix_full__psr_ce_all"]
SBATCH = REPO / "scripts" / "sdfbase" / "full22" / "petri_proxy.sbatch"
COMMON = REPO / "scripts" / "sdfbase" / "full22" / "common.sh"
GKEY = "gemma4e4b"                                                   # the campaign the standup harness serves (test_lora_serving_up)


def proxy_cell_flags_for(csv: str) -> list:
    """The REAL proxy path from positional 5 to the standup's argv: petri_proxy.sbatch's own positional line and its `--cell` block, run under
    bash with common.sh sourced (the REAL f22_standup_extra_reset), f22_standup stubbed to print what it would hand mt_serving_up."""
    src = SBATCH.read_text()
    pos = next(l for l in src.splitlines() if 'CELLS_ARG="${5:-}"' in l)
    k = src.index("CELL_FLAGS=()"); i = src.rindex('if [ -n "$CELLS_ARG" ]; then', 0, k); j = src.index("\n", src.index('f22_standup petri "$OUT"', i))   # the --cell block (the registration block above it is tested on its own)
    block = src[i:j]
    script = (f"set -u; export REPO_DIR={REPO} SHARED=/tmp SLURM_JOB_ID=1 SLURM_JOB_NAME=t; source {COMMON}; "
              'f22_log(){ :; }; f22_standup(){ printf "%s\\n" ${F22_STANDUP_EXTRA[@]+"${F22_STANDUP_EXTRA[@]}"}; }; '
              f'OUT=o; PROXY_PORT=1; RERUN_L=""; RC=0; {pos}; {block}')
    r = subprocess.run(["bash", "-c", script, "_", "k", "v", "s", "", csv], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, (r.stdout, r.stderr)
    return [t for t in r.stdout.split("\n") if t]


def _mt():
    spec = importlib.util.spec_from_file_location("_mt_petri_cells_under_test", REPO / "scripts/sdfbase/mt_serving_up.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["_mt_petri_cells_under_test"] = mod; spec.loader.exec_module(mod)
    return mod


def _steered_standup(tmp_path, monkeypatch, cell_flags: list) -> dict:
    """The REAL mt_serving_up main over the REAL bring_up_lora and native core (the vllm child the scripted upstream; the s3r-fix-2 harness)
    on the gemma campaign's config narrowed to a steered-wave shape (explicit_cells = the one object cell, no none; manifest cells = its harness
    id; audit `…_s2r`), with exactly `cell_flags` appended to the argv — what the proxy hands the standup. Returns the captured records."""
    mt = _mt()
    cfg_path, man_path, cfg = E._campaign(GKEY)
    E.cached_or_skip(cfg)
    view, shas = E.fake_bank(tmp_path, cfg)
    cfg2 = dict(cfg); cfg2["explicit_cells"] = [E.GEMMA_CELL]
    cfg2_path = tmp_path / "steered_wave.yaml"; cfg2_path.write_text(yaml.safe_dump(cfg2, sort_keys=False))
    man = yaml.safe_load(Path(man_path).read_text()); man["petri"]["cells"] = [E.GEMMA_HCELL]; man["petri"]["audit"] = f"petri_full22_{GKEY}_s2r"
    man_sw = tmp_path / "steered_wave_manifest.yaml"; man_sw.write_text(yaml.safe_dump(man, sort_keys=False))
    out = tmp_path / "out"; out.mkdir(); stop = out / "STOP"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1"); monkeypatch.delenv("TT_TWO_SI_DEFENSES_DIR", raising=False)
    E.snapshot_seam(monkeypatch, tmp_path, cfg)
    monkeypatch.setattr(socket, "gethostname", lambda: "TESTHOST")
    captured: dict = {"shas": shas, "out": out, "view": view, "manifest": man_sw, "config": cfg2_path}
    monkeypatch.setattr(vs, "launch_server", E.fake_upstream_launch_server(tmp_path / "fu", captured=captured, chat=E.default_chat(cfg["model"]),
                                                                            genuine_name=E.GEMMA_HCELL, chat_delay_s=0.05))

    def snapshot(stop_file, server, **kw):
        captured["endpoint"] = json.loads((out / "endpoint.json").read_text())
        captured["cells_served"] = json.loads((out / "cells_served.json").read_text())
        return 0
    monkeypatch.setattr(mt, "wait_until_stop", snapshot)
    argv = ["--repo-dir", str(REPO), "--config", str(cfg2_path), "--manifest", str(man_sw), "--defenses-dir", str(view), "--out", str(out),
            "--stop-file", str(stop), "--mode", "attacks", "--eval", "families_served:short", "--model-key", GKEY, "--adapter-bank", str(tmp_path / "bank"),
            "--expect-object-sha256", f"{E.GEMMA_CELL}={shas[E.GEMMA_CELL]}", *cell_flags]
    captured["rc"] = mt.main(argv)
    return captured


def _drive_freeze(captured: dict, monkeypatch, out_root: Path):
    """The grading host drive's freeze over the standup's records (petri_run._frozen_enumeration under lora_native with --verify-standup, the sibling
    endpoint.json binding the declaration to the live attempt), the cell registry the drive would build over the served view (the standup's own
    derived shape: the base route + the cell), then write_cells_json. Returns (payload, requested, cells.json path)."""
    view = captured["view"]
    # the REAL registry path (petri_run.load_cell_source -> registry_adapter.traintest_cell_source under the lens and the lora_native policies):
    # the two-SI config the drive derives from is the standup's own derived shape — the base route + the cell — over the served view
    two_si = dict(yaml.safe_load(Path(captured["config"]).read_text())); two_si["explicit_cells"] = ["none", E.GEMMA_CELL]; two_si["defenses_dir"] = str(view)
    two_si_path = captured["out"] / "two_si_drive.yaml"; two_si_path.write_text(yaml.safe_dump(two_si, sort_keys=False))
    monkeypatch.setenv("PETRI_TWO_SI_CONFIG", str(two_si_path)); monkeypatch.setenv("TT_TWO_SI_DEFENSES_DIR", str(view))
    assert PR.CELL_SOURCE_FACTORY is None                                        # nothing injected: the drive builds its registry as in production
    spec = cm.load_petri_spec(captured["manifest"])
    assert PR.steered_wave_audit(spec)
    # the records the drive reads are the ones PULLED from the live standup (the proxy serves until the audit ends; the harness's standup has
    # already torn down and removed endpoint.json) — the live copies captured at READY, the endpoint named through --served-endpoint as the pull does
    live_cs = captured["out"] / "pulled" / "cells_served.json"; live_ep = captured["out"] / "pulled" / "endpoint.json"; live_cs.parent.mkdir(exist_ok=True)
    live_cs.write_text(json.dumps(captured["cells_served"])); live_ep.write_text(json.dumps(captured["endpoint"]))
    args = argparse.Namespace(verify_standup=str(live_cs), served_endpoint=str(live_ep))
    cell_set, payload, native, requested = PR._frozen_enumeration(args, spec, PR.SERVING_PATH_LORA_NATIVE)
    path = PR.write_cells_json(cell_set, spec, out_root, payload=payload)
    return payload, requested, path


@pytest.fixture(scope="module")
def _roster_dir(tmp_path_factory):
    """Wave configs + a roster naming them, under tmp, and a copy of the models table with the SYNTHETIC phased row — what the plan's cells positional
    reads (stage.view_cells_of): the phased row's u0 (its whole prompted-only baseline), the two ReLU parents on its steered view s2, the one composed
    cell on its d0p75_relu; and the demo's own single-phase waves, each carrying its `none` (battery: the whole enumeration; d0p75_relu: none + the composed
    cell). The front door commits none of these: 05 --step configs renders them at run time."""
    import copy
    root = tmp_path_factory.mktemp("petri_cells_roster")
    t = yaml.safe_load((REPO / "reproduce" / "models.yaml").read_text())
    row = copy.deepcopy(t["models"][KEY]); row["view_legacy_prefix"] = PHASED
    row["views"] = {"u0": {"cells": ["none", "prompted__safety_refusal", "prompted__si_constitution"]}, VIEW: {"cells": ["control_relu", "constitution_relu"]},
                    "d0p75_relu": {"cells": ["difference_relu"], "dose": 0.75, "dose_unit": "psr_direction_norm_multiplier", "object_family": "relu"}}
    t["models"][PHASED] = row
    (root / "models.yaml").write_text(yaml.safe_dump(t, sort_keys=False))
    demo_s1 = t["models"][KEY]["views"]["battery"]["cells"]
    views = {PHASED: {"u0": ["none", "prompted__safety_refusal", "prompted__si_constitution"], VIEW: RELU, "d0p75_relu": [DIST]},
             KEY: {"battery": ["none", "prompted__safety_refusal", "prompted__si_constitution", *RELU, DIST], "d0p75_relu": ["none", DIST]}}
    assert {"control_relu", "constitution_relu", "difference_relu"} <= set(demo_s1)
    targets = {k: {"views": {}} for k in views}
    for k, vs in views.items():
        for v, cells in vs.items():
            cfg = root / f"sdfbase_full22_{k}_{v}.yaml"; cfg.write_text(yaml.safe_dump({"explicit_cells": cells}))
            targets[k]["views"][v] = {"config": str(cfg), "cells": {c: (None if c.startswith("prompted__") else f"base__{c}") for c in cells if c != "none"}}
    (root / "roster.yaml").write_text(yaml.safe_dump({"targets": targets}))
    return root


@pytest.fixture(autouse=True)
def _knobs(monkeypatch, _roster_dir):
    monkeypatch.setenv("SISTEER_SHARD_GROUPS", "0,1,2,3,4 5,6,7,8")
    monkeypatch.setenv("SISTEER_MT_ATTACKER_GPUS", "2")
    monkeypatch.setenv("SISTEER_PLAN_SKIP_ARTEFACT_CHECK", "1")     # the plan's steered-wave artefact walk (s3r-fix) is not this test's subject
    monkeypatch.setenv("F22_ROSTER", str(_roster_dir / "roster.yaml"))
    monkeypatch.setenv("SISTEER_MODELS_TABLE", str(_roster_dir / "models.yaml"))   # the table with the synthetic phased row (the demo's rows unchanged)


def test_the_plan_puts_the_cells_on_the_proxys_positional_5_and_the_sbatch_parses_them_there():
    rows = ST.unit_plan(PHASED, VIEW, families=["petri"], sha=SHA, clusters=AB, cells=RELU)
    assert len(rows) == 1 and rows[0]["script"].endswith("petri_proxy.sbatch")
    assert rows[0]["args"] == [PHASED, VIEW, SHA, "", CSV] and rows[0]["env"]["F22_REGISTRY_CELLS"] == CSV and rows[0]["cells"] == RELU   # the pass stays 4 (""), cells 5
    plain = ST.unit_plan(PHASED, VIEW, families=["petri"], sha=SHA, clusters=AB)
    assert plain[0]["args"] == [PHASED, VIEW, SHA, "", CSV] and plain[0]["env"]["F22_REGISTRY_CELLS"] == CSV and plain[0]["cells"] == RELU   # A steered-wave proxy is planned WITH its view's measured cells (the steered view's config's explicit_cells) without --cells
    assert ST.view_cells_of(PHASED, VIEW) == RELU
    with pytest.raises(ValueError, match="are not in"):
        ST.unit_plan(PHASED, VIEW, families=["petri"], sha=SHA, clusters=AB, cells=["none"])                                              # never none: a selection is held to the steered config's cells
    sub = ST.unit_plan(PHASED, VIEW, families=["petri"], sha=SHA, clusters=AB, cells=[RELU[1]])
    assert sub[0]["args"] == [PHASED, VIEW, SHA, "", RELU[1]] and sub[0]["cells"] == [RELU[1]]                                            # a selection narrows the default
    one = ST.unit_plan(PHASED, "d0p75_relu", families=["petri"], sha=SHA, clusters=AB)
    assert one[0]["args"] == [PHASED, "d0p75_relu", SHA, "", DIST] and one[0]["env"]["F22_REGISTRY_CELLS"] == DIST   # the dose audit's one measured cell
    # the SINGLE-PHASE demo: none of its waves is a steered wave — every proxy is planned with the whole enumeration (its own `none` among the legs),
    # no cells positional, no registry cells env; the dose rung included
    for v in ("battery", "d0p75_relu"):
        whole = ST.unit_plan(KEY, v, families=["petri"], sha=SHA, clusters=AB)
        assert whole[0]["args"] == [KEY, v, SHA] and "F22_REGISTRY_CELLS" not in whole[0]["env"] and "none" in ST.view_cells_of(KEY, v), v
    assert ST.CELLS_POSITIONAL["scripts/sdfbase/full22/petri_proxy.sbatch"] == 4
    line = next(l for l in SBATCH.read_text().splitlines() if 'CELLS_ARG="${5:-}"' in l)                                              # the script's OWN positional line
    r = subprocess.run(["bash", "-c", line + '; printf "%s|%s|%s" "$PASS" "$CELLS_ARG" "$VIEW"', "_", *rows[0]["args"]], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout == f"|{CSV}|{VIEW}", r.stdout + r.stderr
    r = subprocess.run(["bash", "-c", line + '; printf "%s|%s" "$PASS" "$CELLS_ARG"', "_", KEY, VIEW, SHA, "overflow-rerun", CSV], capture_output=True, text=True)
    assert r.stdout == f"overflow-rerun|{CSV}"                                                                                                  # the re-run pass carries the same cells
    # the flags the proxy builds for the standup, run as the script builds them
    blk = 'CELLS_ARG="$1"; CELL_FLAGS=(); IFS="," read -ra _cells <<< "$CELLS_ARG"; for c in "${_cells[@]}"; do [ -n "$c" ] && CELL_FLAGS+=(--cell "$c"); done; printf "%s\\n" "${CELL_FLAGS[@]}"'
    r = subprocess.run(["bash", "-c", blk, "_", CSV], capture_output=True, text=True)
    assert r.stdout.split("\n")[:-1] == ["--cell", RELU[0], "--cell", RELU[1]], r.stdout
    # the REAL proxy path from the plan's positional to the standup's argv: the sbatch's positional line + its --cell block under bash, common.sh
    # sourced (the real f22_standup_extra_reset), f22_standup stubbed to print the extra argv it would pass mt_serving_up
    assert proxy_cell_flags_for(CSV) == ["--cell", RELU[0], "--cell", RELU[1]]
    assert proxy_cell_flags_for(DIST) == ["--cell", DIST]
    src = SBATCH.read_text()
    i_pos = src.index('CELLS_ARG="${5:-}"'); i_exp = src.index('export F22_REGISTRY_CELLS="$CELLS_ARG"')
    i_init = src.index('f22_init "$WAVE_SHA"; f22_resolve_config "$KEY" "$VIEW"'); i_reset = src.index('f22_standup_extra_reset "${CELL_FLAGS[@]}"'); i_up = src.index('f22_standup petri "$OUT"')
    assert i_pos < i_exp < i_init < i_reset < i_up                                   # exported BEFORE the registration; the --cell flags BEFORE the standup
    assert 'CELL_FLAGS+=(--cell "$c")' in src


def test_a_baseline_wave_petri_proxy_takes_no_partial_cells_selection():
    """The baseline wave's audit legs (u0 when declared; the single-phase recipe's one wave battery on the demo) are the WHOLE baseline — a partial declaration would narrow it while
    expected_cells advertises the full manifest, and the drive refuses it; the plan refuses first, by name."""
    with pytest.raises(ValueError, match="the baseline wave's audit legs are the whole baseline"):
        ST.unit_plan(KEY, "battery", families=["petri"], sha=SHA, clusters=AB, cells=["prompted__safety_refusal"])
    rows = ST.unit_plan(KEY, "battery", families=["petri"], sha=SHA, clusters=AB)                        # without --cells: the whole baseline, as before
    assert rows[0]["args"] == [KEY, "battery", SHA] and "F22_REGISTRY_CELLS" not in rows[0]["env"]
    # the other baseline-wave units still take a selection (the static rows narrow by cell)
    sel = ST.unit_plan(KEY, "battery", families=["static"], sha=SHA, clusters=AB, cells=["prompted__safety_refusal"])
    assert sel and all(r["cells"] == ["prompted__safety_refusal"] for r in sel if r.get("kind") != "attacker" and r.get("registers", True))


def test_the_named_cells_become_the_proxys_declaration_and_the_drive_freezes_exactly_them_bound_to_the_live_attempt(tmp_path):
    """The writer is the native core (not re-derived here): what --cell named -> `cells_requested_explicit` (roster spelling) and
    `cells_requested_explicit_harness` (harness ids); the drive's reader takes the harness list, binds the served record to the live
    endpoint's attempt, and a steered-wave audit freezes only the declared legs."""
    roster, harness = NS.requested_cells_explicit(RELU)
    assert list(roster) == RELU and list(harness) == RELU_H
    assert NS.requested_cells_explicit(None) == ((), ()) and NS.requested_cells_explicit([RELU[0], f" {RELU[0]} "]) == ((RELU[0],), (RELU_H[0],))   # stripped, de-duplicated
    with pytest.raises(NS.StandupRefused, match="non-empty TrainTest cell id"):
        NS.requested_cells_explicit([RELU[0], " "])                                                                                           # a blank --cell refuses (never silently dropped)
    assert NS.REQUESTED_EXPLICIT_KEY == PR.REQUESTED_ROSTER_KEY == "cells_requested_explicit" and NS.REQUESTED_EXPLICIT_HARNESS_KEY == PR.REQUESTED_HARNESS_KEY == "cells_requested_explicit_harness"
    assert NS.ENDPOINT_ATTEMPT_KEY == PR.STANDUP_ATTEMPT_KEY == "standup_attempt_id"
    # the drive's reader: the harness list the standup declared, bound to the live attempt through the sibling endpoint.json
    state = tmp_path / "state"; state.mkdir()
    served = {"cells_requested": ["none", *RELU], "harness_cells": ["none", *RELU_H], PR.REQUESTED_ROSTER_KEY: RELU, PR.REQUESTED_HARNESS_KEY: RELU_H, PR.STANDUP_ATTEMPT_KEY: "attempt-0001"}
    (state / "cells_served.json").write_text(json.dumps(served)); (state / "endpoint.json").write_text(json.dumps({PR.STANDUP_ATTEMPT_KEY: "attempt-0001", PR.REQUESTED_HARNESS_KEY: RELU_H}))
    assert PR.requested_cells_of(served, state / "cells_served.json") == RELU_H
    assert PR.requested_cells_of({**served, PR.REQUESTED_HARNESS_KEY: []}, "p") is None                 # nothing named: the whole enumeration, as before
    args = argparse.Namespace(served_endpoint=None)
    assert PR.endpoint_record_path(args, state / "cells_served.json") == state / "endpoint.json"
    assert PR.bind_declaration_to_attempt(args, served, state / "cells_served.json") == "attempt-0001"
    stale = {**served, PR.STANDUP_ATTEMPT_KEY: "attempt-0000"}                                        # an earlier attempt's record over the same routes
    assert "attempt attempt-0000 but the live endpoint" in (PR.attempt_binding_problem(stale, json.loads((state / "endpoint.json").read_text()), standup_path="s", endpoint_path="e") or "")
    with pytest.raises(PR.DriverError, match="not bound to the live attempt"):
        PR.bind_declaration_to_attempt(args, stale, state / "cells_served.json")
    explicit = argparse.Namespace(served_endpoint=str(state / "endpoint.json"))                     # the optional explicit override names the same record
    assert PR.endpoint_record_path(explicit, tmp_path / "elsewhere" / "cells_served.json") == state / "endpoint.json"
    # a steered-wave audit drives ONLY the declared legs; the controls twin derives only when `none` is declared
    assert PR.steered_wave_audit(type("S", (), {"audit": f"petri_full22_{KEY}_s2r"})()) and not PR.steered_wave_audit(type("S", (), {"audit": f"petri_full22_{KEY}_battery"})())
    assert not PR.steered_wave_audit(type("S", (), {"audit": f"petri_full22_{KEY}_d0p75_relu"})())      # the single-phase demo's dose audit drives its own none too
    src = (REPO / "scripts" / "petri_run.py").read_text()
    assert 'want = set(requested) | ({c.cell_id for c in cell_set.controls} if "none" in set(requested) else set())' in src
    assert "a steered-only view drives no none and no none__lora0" in src
    # the producer side is the core's, never a front's extra fact (one owner per file)
    assert "requested_cells_fields" not in (REPO / "scripts" / "sdfbase" / "mt_serving_up.py").read_text()
    assert "cells_requested_explicit=requested_explicit" in (REPO / "scripts" / "sdfbase" / "lora_serving_up.py").read_text()


def test_the_real_proxy_path_declares_the_planned_cells_and_the_drive_freezes_exactly_them(tmp_path, monkeypatch):
    """plan -> positional 5 -> the REAL sbatch --cell block (common.sh's f22_standup_extra_reset) -> the REAL mt_serving_up
    standup over bring_up_lora and the native core -> cells_served.json / endpoint.json -> petri_run._frozen_enumeration (--verify-standup, the
    attempt binding) -> write_cells_json: the frozen cells.json carries EXACTLY the planned cell as its canonical leg, controls [] (no none
    declared, so no twin), expected_cells = the manifest's steered universe; the standup served the base route and the twin implicitly but
    DECLARED only the cell. The gemma campaign stands in for the wave (the one object cell the harness can serve for real)."""
    planned = [E.GEMMA_CELL]                                                    # the plan's CSV for a steered-wave proxy: the view's measured cells
    flags = proxy_cell_flags_for(",".join(planned))
    assert flags == ["--cell", E.GEMMA_CELL]
    cap = _steered_standup(tmp_path, monkeypatch, flags)
    assert cap["rc"] == 0, "the steered-only standup must stand up (s3r-fix-2)"
    cs, ep = cap["cells_served"], cap["endpoint"]
    assert cs["cells_requested_explicit"] == planned and cs["cells_requested_explicit_harness"] == [E.GEMMA_HCELL]      # what the proxy DECLARED
    assert "none" in cs["cells_requested"] and set(ep["served_models"]) == {"petri__base", f"petri__{E.GEMMA_HCELL}", f"petri__{E.ZERO}"}   # served ≠ declared
    assert ep[PR.STANDUP_ATTEMPT_KEY] and ep[PR.REQUESTED_HARNESS_KEY] == [E.GEMMA_HCELL]
    payload, requested, path = _drive_freeze(cap, monkeypatch, tmp_path / "drive")
    assert requested == [E.GEMMA_HCELL]
    rec = json.loads(path.read_text())
    for r in (payload, rec):
        assert [c["cell_id"] for c in r["canonical"]] == [E.GEMMA_HCELL] and r["controls"] == [] and r["expected_cells"] == [E.GEMMA_HCELL]
        assert r[PR.REQUESTED_HARNESS_KEY] == [E.GEMMA_HCELL] and r["serving_path"] == PR.SERVING_PATH_LORA_NATIVE and r["audit"].endswith("_s2r")
    assert [ra.traintest_cell_id(c["cell_id"]) for c in rec["canonical"]] == planned                                       # the frozen legs ARE the planned cells
    assert path.name == PR.CELLS_JSON and PR.write_cells_json(*_cell_set_and_spec(cap, payload, monkeypatch), tmp_path / "drive", payload=payload) == path   # immutable, identical re-derivation stands


def _cell_set_and_spec(cap, payload, monkeypatch):
    spec = cm.load_petri_spec(cap["manifest"])                                  # PETRI_TWO_SI_CONFIG / TT_TWO_SI_DEFENSES_DIR still set by _drive_freeze
    args = argparse.Namespace(verify_standup=str(cap["out"] / "pulled" / "cells_served.json"), served_endpoint=str(cap["out"] / "pulled" / "endpoint.json"))
    cell_set, payload2, _n, _r = PR._frozen_enumeration(args, spec, PR.SERVING_PATH_LORA_NATIVE)
    assert payload2 == payload
    return cell_set, spec


def test_an_undeclared_steered_wave_proxy_refuses_at_the_freeze_which_is_why_the_plan_names_the_cells(tmp_path, monkeypatch):
    """The failure this guards: the same real standup with NO --cell (a proxy that declares nothing) records an empty declaration, and the
    drive's freeze on a steered-wave audit refuses by name instead of auditing `none` and the twin."""
    cap = _steered_standup(tmp_path, monkeypatch, [])
    assert cap["rc"] == 0 and cap["cells_served"]["cells_requested_explicit"] == []
    with pytest.raises(PR.DriverError, match="is a steered-wave view and drives only the cells the proxy declared"):
        _drive_freeze(cap, monkeypatch, tmp_path / "drive")
    assert not (tmp_path / "drive" / PR.CELLS_JSON).exists()


def registration_block_for(args: list, inherited: "str | None") -> subprocess.CompletedProcess:
    """The REAL proxy script text from its positional line through the registration export block, run under bash with `args` as the job's
    positionals and `inherited` as a pre-set F22_REGISTRY_CELLS (None = unset); prints the resulting variable (or UNSET)."""
    src = SBATCH.read_text()
    pos = next(l for l in src.splitlines() if 'CELLS_ARG="${5:-}"' in l)
    i = src.index('if [ -n "$CELLS_ARG" ]; then\n  if [ -n "${F22_REGISTRY_CELLS:-}" ]'); j = src.index("\nfi\n", i) + len("\nfi\n")
    block = src[i:j]
    pre = f"export F22_REGISTRY_CELLS={inherited!r}; " if inherited is not None else "unset F22_REGISTRY_CELLS; "
    script = "set -u; " + pre + pos + "\n" + block + '\necho "CELLS=${F22_REGISTRY_CELLS-UNSET}"'
    return subprocess.run(["bash", "-c", script, "_", *args], capture_output=True, text=True, timeout=60)


def test_positional_5_is_the_one_authority_for_the_registration_cells():
    """petri-cells-fix: the value the job REGISTERS is positional 5 — the same value the declaration and the frozen
    legs come from — never an inherited F22_REGISTRY_CELLS. A matching inherited value is harmless; a differing one refuses by name before
    anything is sourced; a whole-baseline proxy (no positional 5) refuses an inherited selection by name and otherwise leaves the variable unset."""
    ok = registration_block_for([KEY, VIEW, SHA, "", CSV], None)
    assert ok.returncode == 0 and ok.stdout.strip() == f"CELLS={CSV}", ok.stdout + ok.stderr
    same = registration_block_for([KEY, VIEW, SHA, "", CSV], CSV)
    assert same.returncode == 0 and same.stdout.strip() == f"CELLS={CSV}", same.stdout + same.stderr
    conflict = registration_block_for([KEY, VIEW, SHA, "", RELU[1]], CSV)                          # a hand retry with the wave's two cells inherited
    assert conflict.returncode == 2 and "CELLS=" not in conflict.stdout
    assert f"positional 5 names '{RELU[1]}' but the inherited F22_REGISTRY_CELLS='{CSV}' differs" in conflict.stderr and "positional 5 is the one authority" in conflict.stderr
    base_inherited = registration_block_for([KEY, "battery", SHA], RELU[0])                               # a baseline-wave proxy with a stale selection in the environment
    assert base_inherited.returncode == 2 and "CELLS=" not in base_inherited.stdout
    assert "no positional 5 (a whole-baseline proxy) but an inherited F22_REGISTRY_CELLS" in base_inherited.stderr
    base = registration_block_for([KEY, "battery", SHA], None)
    assert base.returncode == 0 and base.stdout.strip() == "CELLS=UNSET", base.stdout + base.stderr
    src = SBATCH.read_text()
    assert '${F22_REGISTRY_CELLS:-$CELLS_ARG}' not in src                                           # the inherited-wins default is gone
    assert src.index('export F22_REGISTRY_CELLS="$CELLS_ARG"') < src.index('source "${SISTEER_EVAL_REPO_ROOT') < src.index('f22_init "$WAVE_SHA"')


def test_the_registration_covers_exactly_the_positional_cells_and_an_inherited_conflict_registers_nothing(tmp_path, monkeypatch):
    """Through the REAL registration (common.sh f22_registry_register over the real seeds and a real committed wave config — the paper model's
    battery view, whose roster block and pins the tree carries; the front door commits none — the petri job kind's evals): the sbatch's registration
    export from positional 5 narrows the registered runs to EXACTLY the positional cells — one cell → 7 petri runs on that cell, none on the
    other wave cells — so registered ⊆ declared (the --cell flags from the same positional) ⊆ served (the standup's `none` + declared); and the
    conflicting-inherited case refuses before the export, so nothing registers."""
    import shutil
    from test_static_registration_gated_pairs import _checkout, _run_dirs
    key, view = PAPER_KEY, PAPER_VIEW
    co = _checkout(tmp_path, key)
    for rel in (f"sdfbase_full22_{key}_{view}.yaml", "sdfbase_full22_roster.yaml", "sdfbase_full22_pins.json"):      # the real wave config, roster and pins: a staged object's sha comes from them
        shutil.copy(REPO / "configs" / rel, co / "configs")
    shared = tmp_path / "shared"; (shared / "registry/records").mkdir(parents=True, exist_ok=True)
    for k in ("F22_REGISTRY_CELLS", "F22_REGISTRY_EVALS", "F22_REGISTRY_PAIRS", "SISTEER_REGISTRY_RUN_IDENTITY", "SISTEER_REGISTRY_DISABLED", "REGISTRY_EMIT"):
        monkeypatch.delenv(k, raising=False)
    src = SBATCH.read_text(); pos = next(l for l in src.splitlines() if 'CELLS_ARG="${5:-}"' in l)
    i = src.index('if [ -n "$CELLS_ARG" ]; then\n  if [ -n "${F22_REGISTRY_CELLS:-}" ]'); j = src.index("\nfi\n", i) + len("\nfi\n"); block = src[i:j]

    def register(job: str, args: list, inherited: "str | None"):
        pre = f"export F22_REGISTRY_CELLS={inherited!r}" if inherited is not None else "unset F22_REGISTRY_CELLS"
        script = f"""
            set -u
            cd {co}
            {pre}
            export REPO_DIR={co} SHARED={shared} SISTEER_REGISTRY_SPOOL={shared}/registry/records
            export SISTEER_REGISTRY_SUBSTRATE=slurm SISTEER_REGISTRY_CLUSTER_LABEL=cluster_a CLUSTER=slurm
            export SLURM_JOB_ID={job} SLURM_JOB_NAME=f22-petri-{key}_{view} WAVE_SHA={SHA} SLURMD_NODENAME=TESTHOST
            export KEY={key} VIEW={view} TAG={key}_{view} CFG=configs/sdfbase_full22_{key}_{view}.yaml F22_ROSTER=configs/sdfbase_full22_roster.yaml F22_SERVING_PATH=lora_native
            {pos}
            {block}
            source {REPO / "scripts/sdfbase/full22/common.sh"}
            f22_log(){{ echo "[log] $*"; }}
            f22_registry_register || true
            echo "RC=$? IDENT=${{SISTEER_REGISTRY_RUN_IDENTITY:-}} CELLS=${{F22_REGISTRY_CELLS-UNSET}}"
        """
        env = dict(os.environ)
        r = subprocess.run(["bash", "-c", script, "_", *args], capture_output=True, text=True, env=env, timeout=300)
        return r, shared / "registry" / "pending" / f"{job}.json"

    # the authoritative case: positional 5 = ONE of the wave's cells (the config lists both ReLU parents among its eight)
    r, pend = register("7101", [key, view, SHA, "", RELU[1]], None)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and f"CELLS={RELU[1]}" in r.stdout and pend.is_file(), out
    runs = json.loads(pend.read_text())["runs"]
    objects = {x["plan"]["producer_object_key_legacy"] for x in runs}                                  # the plan's object in roster spelling
    shas = {x["plan"]["object_id"] for x in runs}                                                     # the registry's object_id is the staged object's sha256
    assert objects == {RELU[1]} and len(runs) == 7, (objects, len(runs))                            # 7 petri evals × the one positional cell; nothing on the other
    assert len(shas) == 1 and next(iter(shas)) in (REPO / "configs" / "sdfbase_full22_pins.json").read_text()   # … pinned under the wave's pins
    assert {x["plan"]["eval_id"] for x in runs} == set(ST.PETRI_EVALS)
    declared = proxy_cell_flags_for(RELU[1])
    assert declared == ["--cell", RELU[1]] and objects <= set(declared[1::2]) <= {"none", *RELU}     # registered ⊆ declared ⊆ served (none + declared)
    # the conflict: two of the wave's cells inherited, one named positionally — refused before the export, nothing registered
    r2, pend2 = register("7102", [key, view, SHA, "", RELU[1]], CSV)
    assert r2.returncode == 2 and "differs" in r2.stderr and not pend2.exists() and "RC=" not in r2.stdout, r2.stdout + r2.stderr
    assert sorted(_run_dirs(shared / "registry" / "records")) == sorted(_run_dirs(shared / "registry" / "records"))  # no second job's records
    assert not [d for d in _run_dirs(shared / "registry" / "records") if "7102" in str(d)]

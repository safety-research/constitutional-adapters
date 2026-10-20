"""``scripts/sdfbase/mt_serving_up.py --serving-path``, CPU-only.

The wave standup's TRANSLATION onto the wave front ``lora_serving_up.bring_up_lora`` (itself a front
over the shared core ``native_standup.bring_up_native``), tested on the LAUNCH ARGUMENTS that reach the
scripted upstream (``test_lora_serving_up.fake_upstream_launch_server``: the REAL composed launch, argv
swapped for the stand-in, the REAL ``launch_server`` readiness, proxy and controls) and on the records
— never on intercepted kwargs:

- the DEFAULT is native: the argv carries ``--no-enforce-eager`` + ``--lora-modules``, the records
  ``engine: lora_native`` and the mt provenance beside the core's keys (``serving_config.json`` written
  before the launch); ``--thinking-on --reasoning-parser gemma4`` puts ``--reasoning-parser gemma4`` on
  the argv;
- ``--expect-object-sha256`` ABSENT on the wave path: the digests derive from the bank view's
  ``VIEW_MANIFEST.json`` + the pins (``expected_digests_source: view_manifest+pins``); a drifted object
  refuses;
- each lens-only flag (``--reduction-mode stacked``, ``--require-attested-mode``, ``--lens-salt-policy
  request_unique_all``) refuses by name, exit 2, before any launch;
- ``--serving-path lens`` is GOLDEN-identical to the pre-landing tree's ``main`` on the same stubs
  (``git show HEAD:scripts/sdfbase/mt_serving_up.py``).

Skips (never fetches) when the pinned gemma tokenizer / config is not in the local HF cache.
"""
from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
#: the PRE-LANDING tree the lens goldens compare against: the branch base of native-serving-default (origin/main
#: at branch time, before any serving path existed). A fixed sha, so the golden RUNS on every later tree
#: (`HEAD:` skipped forever once HEAD carried the flag); a checkout without that object skips.
PRE_LANDING_SHA = "079533f1"   # origin/main at the FINAL merge (re-derived 2026-09-14): the lens body must equal MAIN's lens body, and main's
#: misalign lane (ddcf14b2 organisms overlay, ff3cce95 chat_template override) added two keys to the lens cells_served record
#: after the branch's original base ea8bf598 — the golden follows the tree the branch lands on, never an older main
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import test_lora_serving_up as E  # noqa: E402
import test_mt_serving_up_thinking as T  # noqa: E402

from sisteer.capeval.serving import geometry as sg  # noqa: E402,F401  (before any monkeypatch of vs.build_server_launch)
from sisteer.capeval.serving import vllm_server as vs  # noqa: E402

KEY = "gemma4e4b"


def _load_mt(name="_mt_serving_path_under_test", path=None):
    spec = importlib.util.spec_from_file_location(name, path or REPO / "scripts/sdfbase/mt_serving_up.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mt():
    return _load_mt()


def _run_native(mt, tmp_path, monkeypatch, *extra, digests=True, view_manifest=None, pins=None, pre_files=None):
    """The REAL ``mt.main`` under the native default on the gemma campaign pair over a synthetic bank
    (the target cell a REAL lora_all blob), the vllm child the scripted upstream. Returns
    ``(captured, launch)``; the records are read while the standup is READY (endpoint.json is unlinked
    at teardown). ``pre_files`` = ``{name: text}`` written into ``--out`` BEFORE ``main`` runs — what the
    launching shell leaves beside the standup (full22's ``registry_runs.json``, tests/test_bon_long_si_windows.py)."""
    cfg_path, man_path, cfg = E._campaign(KEY)
    E.cached_or_skip(cfg)
    view, shas = E.fake_bank(tmp_path, cfg)
    if view_manifest is not None:
        (view / mt.VIEW_MANIFEST_FILE).write_text(json.dumps(view_manifest(shas)))
    if pins is not None:
        pins_path = tmp_path / "pins.json"
        pins_path.write_text(json.dumps(pins(shas)))
        monkeypatch.setattr(mt, "PINS_FILE", str(pins_path))
    out = tmp_path / "out"
    out.mkdir()
    for name, text in (pre_files or {}).items():
        (out / name).write_text(text)
    stop = out / "STOP"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("TT_TWO_SI_DEFENSES_DIR", raising=False)
    E.snapshot_seam(monkeypatch, tmp_path, cfg)
    # the core reads the node name off the stdlib module (native_standup imports socket); patch it there
    monkeypatch.setattr(socket, "gethostname", lambda: "TESTHOST")
    captured: dict = {"shas": shas, "out": out}
    monkeypatch.setattr(vs, "launch_server", E.fake_upstream_launch_server(
        tmp_path / "fake_upstream", captured=captured, chat=E.default_chat(cfg["model"]),
        genuine_name=E.GEMMA_HCELL, chat_delay_s=0.05))

    def snapshot(stop_file, server, **kw):
        captured["endpoint"] = json.loads((out / "endpoint.json").read_text())
        captured["cells_served"] = json.loads((out / "cells_served.json").read_text())
        captured["serving_config"] = json.loads((out / mt.SERVING_CONFIG_RECORD_FILE).read_text())
        return 0
    monkeypatch.setattr(mt, "wait_until_stop", snapshot)
    argv = ["--repo-dir", str(REPO), "--config", str(cfg_path), "--manifest", str(man_path),
            "--defenses-dir", str(view), "--out", str(out), "--stop-file", str(stop),
            # the wave's mode vocabulary (EVAL_BY_MODE: attacks | petri, what f22_standup passes); --eval pins the
            # static families' geometry entry, and `attacks` composes no agent modality
            "--mode", "attacks", "--eval", "families_served:short", "--model-key", KEY,
            "--cell", E.GEMMA_CELL, "--adapter-bank", str(tmp_path / "bank"), *extra]
    if digests:
        argv += ["--expect-object-sha256", f"{E.GEMMA_CELL}={shas[E.GEMMA_CELL]}"]
    captured["rc"] = mt.main(argv)
    return captured


def test_the_default_is_native_and_the_launch_argv_carries_the_native_pins(mt, tmp_path, monkeypatch):
    cap = _run_native(mt, tmp_path, monkeypatch, "--enable-log-requests", "--driver-max-tokens", "512",
                      "--upstream-timeout-s", "7200")
    assert cap["rc"] == 0
    cmd = list(cap["launch"].cmd)
    assert cap["launch"].lora_native and "--no-enforce-eager" in cmd and "--enforce-eager" not in cmd
    assert "--lora-modules" in cmd and "--enable-log-requests" in cmd
    assert cmd[cmd.index("--max-lora-rank") + 1] == "8" and cap["launch"].env.get("VLLM_PLUGINS") == ""
    assert "--enable-auto-tool-choice" not in cmd                       # --mode attacks: no agent modality
    ep, cs = cap["endpoint"], cap["cells_served"]
    assert ep["engine"] == ep["serving_engine"] == cs["engine"] == "lora_native"
    assert "lens_reduction_mode" not in ep and ep["reduction_mode"] == "not_applicable"
    assert ep["lens_salt_policy"] == "not_applicable" and ep["require_attested_mode"] is None
    assert ep["expected_digests_source"] == "cli" and ep["object_artifact_sha256_by_cell"] == {E.GEMMA_HCELL: cap["shas"][E.GEMMA_CELL]}
    assert "serving_path_selection" not in ep and "serving_path_selection" not in cs     # no --serving-path-selection: the records as before
    # mt's provenance keys ride beside the core's (extra_endpoint_facts), never over one
    assert ep["serving_config_sha256"] == cap["serving_config"]["serving_config_sha256"]
    assert ep["manifest_sha256"] == cap["serving_config"]["manifest_sha256"] and ep["standup_script"] == "mt_serving_up.py"
    assert ep["driver_max_tokens"] == 512 and ep["upstream_timeout_s"] == 7200.0 and ep["proxy_upstream_timeout_s"] == 7200.0
    assert ep["budget_entry_model_key"] == KEY and "bank_objects" in ep and ep["serving_path"] == "lora_native"
    assert ep["enable_log_requests"] is True and ep["mode"] == "attacks" and ep["eval"] == "families_served:short"
    assert ep["standup_front"] == "lora_serving_up" and ep["thinking_mode"] == "off"
    assert cap["serving_config"]["schema"] == mt.SERVING_CONFIG_RECORD_SCHEMA
    assert ep["max_num_seqs"] == 107 and ep["geometry"]["adjustments"][0]["kind"] == "lora_kernel_index_ceiling"
    assert ep["cells_skipped"] == [] and set(ep["served_models"]) == {"petri__base", f"petri__{E.GEMMA_HCELL}", f"petri__{E.ZERO}"}


def test_thinking_on_forwards_the_family_parser_to_the_launch_argv(mt, tmp_path, monkeypatch):
    cap = _run_native(mt, tmp_path, monkeypatch, "--thinking-on", "--reasoning-parser", "gemma4")
    assert cap["rc"] == 0
    cmd = list(cap["launch"].cmd)
    assert cmd[cmd.index("--reasoning-parser") + 1] == "gemma4"
    assert '"enable_thinking": true' in cmd[cmd.index("--default-chat-template-kwargs") + 1]
    assert cap["endpoint"]["thinking_mode"] == "on" and cap["endpoint"]["thinking_declared_cells"] == []
    assert cap["endpoint"]["thinking_certification"]["verdict"] == "trivially_in_support"
    assert cap["endpoint"]["reasoning_parser"] == "gemma4"


def _view_manifest(shas):
    return {"wave": "full22", "target": KEY, "view": "main", "bank": "synthetic",
            "entries": [{"cell_id": E.GEMMA_CELL, "staged_id": f"base__{E.GEMMA_CELL}",
                         "object_sha256": shas[E.GEMMA_CELL], "identity": "staged"}]}


def test_the_digests_derive_from_the_view_manifest_and_the_pins_when_the_flag_is_absent(mt, tmp_path, monkeypatch):
    cap = _run_native(mt, tmp_path, monkeypatch, digests=False, view_manifest=_view_manifest,
                      pins=lambda shas: {KEY: {f"base__{E.GEMMA_CELL}": {"object_sha256": shas[E.GEMMA_CELL]}}})
    assert cap["rc"] == 0
    assert cap["endpoint"]["expected_digests_source"] == "view_manifest+pins"
    assert cap["endpoint"]["object_artifact_sha256_by_cell"] == {E.GEMMA_HCELL: cap["shas"][E.GEMMA_CELL]}
    assert cap["cells_served"]["expected_object_sha256_by_cell"] == cap["endpoint"]["object_artifact_sha256_by_cell"]


def test_a_drifted_pin_or_manifest_refuses_before_any_launch(mt, tmp_path, monkeypatch):
    launched: list = []
    monkeypatch.setattr(vs, "build_server_launch", lambda *a, **k: launched.append(1))
    cap = _run_native(mt, tmp_path / "pin", monkeypatch, digests=False, view_manifest=_view_manifest,
                      pins=lambda shas: {KEY: {f"base__{E.GEMMA_CELL}": {"object_sha256": "ab" * 32}}})
    assert cap["rc"] == 2 and launched == [] and not (cap["out"] / "endpoint.json").exists()
    cap2 = _run_native(mt, tmp_path / "manifest", monkeypatch, digests=False,
                       view_manifest=lambda shas: {**_view_manifest(shas), "entries": []},
                       pins=lambda shas: {KEY: {f"base__{E.GEMMA_CELL}": {"object_sha256": shas[E.GEMMA_CELL]}}})
    assert cap2["rc"] == 2 and launched == []
    # no manifest, no flag: refused by name
    cap3 = _run_native(mt, tmp_path / "nomanifest", monkeypatch, digests=False)
    assert cap3["rc"] == 2 and launched == []


@pytest.mark.parametrize("flag,argv", [
    ("--reduction-mode", ["--reduction-mode", "stacked"]),
    ("--require-attested-mode", ["--require-attested-mode", "stacked"]),
    ("--lens-salt-policy", ["--lens-salt-policy", "request_unique_all"]),
])
def test_each_lens_only_flag_refuses_by_name_under_native(mt, tmp_path, monkeypatch, flag, argv, capsys):
    cfg_path, man_path, cfg = E._campaign(KEY)
    launched: list = []
    monkeypatch.setattr(vs, "build_server_launch", lambda *a, **k: launched.append(1))
    monkeypatch.setattr(vs, "launch_server", lambda *a, **k: launched.append(1))
    rc = mt.main(["--repo-dir", str(REPO), "--config", str(cfg_path), "--manifest", str(man_path),
                  "--defenses-dir", str(tmp_path), "--out", str(tmp_path / "out"), "--stop-file", str(tmp_path / "STOP"),
                  "--mode", "attacks", "--model-key", KEY, *argv])
    assert rc == 2 and launched == []
    text = capsys.readouterr().out
    assert flag in text and "--serving-path lens" in text
    # the DEFAULTS of the same flags are accepted (recorded not_applicable by the core): the parser says so
    ns = mt._build_parser().parse_args(["--repo-dir", "r", "--config", "c", "--manifest", "m", "--defenses-dir", "d",
                                        "--out", "o", "--stop-file", "s", "--mode", "attacks", "--model-key", KEY])
    assert (ns.serving_path, ns.reduction_mode, ns.require_attested_mode, ns.lens_salt_policy) == (None, "per_row", None, "attested")
    assert (ns.text_only_mm, ns.kv_derived_seqs, ns.adapter_bank) == ("off", "off", None)
    assert ns.mamba_backend_contract == "configs/capeval/mamba_backend_contract.json"


def test_serving_path_lens_is_golden_identical_to_the_pre_landing_tree(mt, tmp_path, monkeypatch):
    """The lens body of THIS tree's ``main`` (``--serving-path lens``) equals the pre-landing tree's
    ``main`` (git show HEAD) on the shared stubbed harness: the launch argv / env and the endpoint /
    cells_served records minus the volatile keys."""
    try:
        head_src = subprocess.run(["git", "-C", str(REPO), "show", f"{PRE_LANDING_SHA}:scripts/sdfbase/mt_serving_up.py"],
                                  capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        pytest.skip(f"HEAD's mt_serving_up.py is not readable here: {e}")
    if "--serving-path" in head_src:
        pytest.skip("HEAD already carries the serving path (the golden is the pre-landing tree)")
    head_path = tmp_path / "mt_serving_up_head.py"
    head_path.write_text(head_src)
    head = _load_mt("_mt_head_golden", head_path)
    # T._run appends "--serving-path lens" (this tree's flag); HEAD's parser does not know it — strip it there
    real_main = head.main

    def head_main(argv):
        argv = list(argv)
        i = argv.index("--serving-path")
        del argv[i:i + 2]
        return real_main(argv)
    head.main = head_main
    new, proxy_new, launch_new = T._run(mt, tmp_path / "new", monkeypatch, KEY, *T.M5T_EXTRA)
    old, proxy_old, launch_old = T._run(head, tmp_path / "old", monkeypatch, KEY, *T.M5T_EXTRA)
    # the ports are free-port draws and each run has its OWN out root (the --chat-template path)
    strip = lambda cmd, root: [a.replace(str(root), "<OUT>") for i, a in enumerate(cmd)  # noqa: E731
                               if not (i and cmd[i - 1] == "--port")]
    assert strip(list(launch_new.cmd), tmp_path / "new") == strip(list(launch_old.cmd), tmp_path / "old")
    envn = lambda env, root: {k: v.replace(str(root), "<OUT>") for k, v in env.items() if k != "VLLM_API_KEY"}  # noqa: E731
    assert envn(launch_new.env, tmp_path / "new") == envn(launch_old.env, tmp_path / "old")   # PETRI_LENS_REGISTRY_DIR is under out
    volatile = {"pid", "node", "written_at", "started", "proxy_url", "proxy_port", "vllm_port", "routing_table",
                "steering_attestation", "serving_config", "config", "manifest", "defenses_dir", "cells_served"}

    def norm(rec, root):
        out = {k: v for k, v in rec.items() if k not in volatile}
        # `server_default_policy` is ADDITIVE on both paths since the seed-replicates reseed: the
        # lens records name the policy their launch was composed under (contract_override = the pre-landing launch, byte
        # for byte — the argv / env comparison above still holds it), a key the pre-landing tree never wrote
        for k in ("bank_objects", "server_default_policy"):
            out.pop(k, None)
        return json.loads(json.dumps(out, sort_keys=True, default=str).replace(str(root), "<OUT>"))
    assert norm(new["endpoint"], tmp_path / "new") == norm(old["endpoint"], tmp_path / "old")
    assert norm(new["cells_served"], tmp_path / "new") == norm(old["cells_served"], tmp_path / "old")
    assert "engine" not in new["endpoint"] and new["endpoint"]["lens_reduction_mode"] == "stacked"


# --- the serving-path SELECTION record (the kind-aware default, 2026-09-15) ------------------------------------------

SELECTION = {"mode": "auto", "selected": "lora_native",
             "reason": "1 of 2 steered cells natively servable; skipped by name: si_x__psr_ce_all (psr)",
             "cells_by_disposition": {"passthrough": ["none"], "serve": [E.GEMMA_HCELL], "skip": ["si_x__psr_ce_all"]},
             "skipped": [{"cell_id": "si_x__psr_ce_all", "kind": "psr", "reason": "kind 'psr' is not natively exportable"}],
             "spec_source": "kind_only", "cells_source": "F22_LAUNCH_CELLS", "rule": "kind-aware-2026-09-15"}


def test_the_selection_record_rides_both_native_records_verbatim_and_a_foreign_one_refuses(mt, tmp_path, monkeypatch):
    """``--serving-path-selection FILE`` (what common.sh f22_standup passes after f22_init chose the path): the record
    lands VERBATIM under ``serving_path_selection`` in endpoint.json AND cells_served.json beside the core's keys (an
    extra fact, never a core key — the collision check stays). A record whose ``selected`` is another path, a
    non-record and an unreadable file refuse before any launch: the record must describe THIS standup."""
    f = tmp_path / "selection.json"
    f.write_text(json.dumps(SELECTION))
    cap = _run_native(mt, tmp_path, monkeypatch, "--serving-path-selection", str(f))
    assert cap["rc"] == 0
    assert cap["endpoint"]["serving_path_selection"] == SELECTION == cap["cells_served"]["serving_path_selection"]
    assert cap["endpoint"]["engine"] == "lora_native" and cap["endpoint"]["serving_path"] == "lora_native"
    from sisteer.capeval.serving import native_standup as ns
    assert "serving_path_selection" not in ns._CORE_RECORD_KEYS
    for bad, needle in (({**SELECTION, "selected": "lens"}, "must describe this standup"),
                        ({"selected": "lora_native"}, "not a selection record"),
                        ({**SELECTION, "mode": "guessed"}, "not a selection record"),
                        ([1, 2], "not a selection record")):
        f.write_text(json.dumps(bad))
        sub = tmp_path / f"bad_{needle[:4]}_{len(json.dumps(bad))}"
        sub.mkdir()
        with pytest.raises(SystemExit, match=needle):
            _run_native(mt, sub, monkeypatch, "--serving-path-selection", str(f))
    with pytest.raises(SystemExit, match="unreadable"):
        _run_native(mt, tmp_path / "missing", monkeypatch, "--serving-path-selection", str(tmp_path / "nope.json"))


def test_the_selection_record_rides_both_lens_records_too(mt, tmp_path, monkeypatch):
    """The lens body (``--serving-path lens``) carries the same key on both records when — and only when — the caller
    recorded one (the golden test above pins the flag-less lens records against the pre-landing tree)."""
    rec = {**SELECTION, "selected": "lens", "mode": "explicit", "reason": "F22_SERVING_PATH set by caller",
           "cells_by_disposition": None, "skipped": [], "spec_source": None, "cells_source": None}
    f = tmp_path / "selection_lens.json"
    f.write_text(json.dumps(rec))
    sub = tmp_path / "lens"
    sub.mkdir()
    snap, _proxy, _launch = T._run(mt, sub, monkeypatch, KEY, *T.M5T_EXTRA, "--serving-path-selection", str(f))
    assert snap["endpoint"]["serving_path_selection"] == rec == snap["cells_served"]["serving_path_selection"]
    assert snap["endpoint"]["lens_reduction_mode"] == "stacked" and "engine" not in snap["endpoint"]
    f.write_text(json.dumps({**rec, "selected": "lora_native"}))
    sub2 = tmp_path / "lens_bad"
    sub2.mkdir()
    with pytest.raises(SystemExit, match="must describe this standup"):
        T._run(mt, sub2, monkeypatch, KEY, *T.M5T_EXTRA, "--serving-path-selection", str(f))


def test_the_production_front_declares_what_cell_named_in_both_final_records(mt, tmp_path, monkeypatch):
    """The PRODUCTION standup -- ``mt.main --cell <id>`` over ``lora_serving_up.bring_up_lora``
    and the native core -- writes ``cells_requested_explicit`` (the roster spelling named) and ``cells_requested_explicit_harness`` (its
    harness id) into the FINAL cells_served.json AND endpoint.json beside the served list; ``none`` is served (``cells_requested`` lists
    it first) but not declared, since it was not named. The reader (``petri_run.requested_cells_of``) reads exactly these keys."""
    import petri_run as prun  # noqa: E402  (scripts/ is on sys.path above)
    from sisteer.capeval.serving import native_standup as ns  # noqa: E402  (this module imports the core inside its tests)
    cap = _run_native(mt, tmp_path, monkeypatch)
    assert cap["rc"] == 0
    cs, ep = cap["cells_served"], cap["endpoint"]
    for rec in (cs, ep):
        assert rec[ns.REQUESTED_EXPLICIT_KEY] == [E.GEMMA_CELL] and rec[ns.REQUESTED_EXPLICIT_HARNESS_KEY] == [E.GEMMA_HCELL]
    assert cs.get("stage") != "pre_launch" and cs["cells_requested"][0] == "none" and cs["harness_cells"] == ["none", E.GEMMA_HCELL]
    assert "none" not in cs[ns.REQUESTED_EXPLICIT_KEY]
    assert prun.requested_cells_of(cs, "cells_served.json") == [E.GEMMA_HCELL] == prun.requested_cells_of(ep, "endpoint.json")
    assert prun.attempt_binding_problem(cs, ep, standup_path="cells_served.json", endpoint_path="endpoint.json") is None
    assert ns.REQUESTED_EXPLICIT_KEY in ns._CORE_RECORD_KEYS and ns.REQUESTED_EXPLICIT_HARNESS_KEY in ns._CORE_RECORD_KEYS

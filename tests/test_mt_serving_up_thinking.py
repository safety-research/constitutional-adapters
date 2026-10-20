"""``scripts/sdfbase/mt_serving_up.py`` under thinking ON: the cluster
standup's REAL ``main`` on the pinned campaign configs over a view-shaped
fake bank — every
derivation, the registry and routing-table exports, the launch composition
and the RoutingProxy are the real ones; only the GPU launch (a fake child
that writes the workers' attestation), the upstream client and the aiohttp
thread are replaced, and the stop-wait seam is read to snapshot the records
the standup unlinks at teardown.

Skips (never fetches) when a pinned tokenizer is not in the local HF cache."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from test_sdfbase_full22_wave import DERIVED_ARMS, _stage, _stage_derived
from test_serve_declarations_attestation import (
    _cached_tok,
    _FakeServer,
    _RecordingBackend,
)

from sisteer.capeval.serving import proxy as px
from sisteer.capeval.serving import standup_thinking as st
from sisteer.capeval.serving import vllm_server as vs
from sisteer.capeval.tinycap import contracts as C
from sisteer.eval.defense_cell import arm_of, sha256_file

OBJECT = "constv2-mix-full__lora_r4_ce_minus_ctl"
OBJECT_HARNESS = "si_constv2_mix_full__lora_r4_ce_minus_ctl"
PROMPTED = "prompted__safety_refusal"
NEMOTRON_OFF = [10, 1503, 19464, 1010, 12, 13]
NEMOTRON_PARTIAL = {"switch": True, "moves": None, "off_ids": NEMOTRON_OFF, "on_ids": None}


def _load_mt():
    spec = importlib.util.spec_from_file_location("_mt_thinking_under_test", REPO / "scripts/sdfbase/mt_serving_up.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_mt_thinking_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mt():
    return _load_mt()


def _campaign(key):
    cfg_path = REPO / f"configs/sdfbase_full22_{key}.yaml"
    man_path = REPO / f"configs/capeval/manifests/petri_full22_{key}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    return cfg_path, man_path, cfg


def _fake_bank(tmp_path, cfg):
    """Every OBJECT cell of the config staged the way the view stages it
    (derived arms -> derived_meta.json, the rest a budget sidecar); prompted
    cells have no object. Returns the bank dir and the object cell's sha."""
    bank = tmp_path / "cells_view" / "main"
    for c in cfg["explicit_cells"]:
        if c == "none" or c.startswith("prompted__"):
            continue
        if arm_of(c) in DERIVED_ARMS:
            _stage_derived(bank, c, c)
        else:
            _stage(bank, c, c)
    return bank, sha256_file(bank / OBJECT / "object.pt")


_TOK_PRESENT: dict = {}


def _cached_or_skip(cfg):
    """The skip guard (the pinned tokenizer must be in the local cache), memoised per (model, revision): the guard's
    full tokenizer load — discarded — was ~a quarter of this module's wall time across its repeated standups."""
    key = (cfg["model"], cfg["model_revision"])
    if key not in _TOK_PRESENT:
        _TOK_PRESENT[key] = _cached_tok(*key)       # pytest.skip raises here when absent; nothing is memoised
    return _TOK_PRESENT[key]


def _attestation(launch, *, decls=None, policy="always"):
    """What the workers attest at readiness for the composed launch: stacked
    on both report kinds, the live scheduler geometry echoing the argv, the
    resumed-prefill policy per cell, the (empty) declaration map, the hybrid
    cache dtype the model-facts reader wants."""
    cmd = list(launch.cmd)
    S = int(cmd[cmd.index("--max-num-seqs") + 1])
    L = int(cmd[cmd.index("--max-model-len") + 1])
    install = {"registration_count": 1, "layers": [3], "oproj_layers": [], "plan_layer": 3,
               "cells": ["none__lens0", OBJECT_HARNESS], "ledger_cells": [],
               "lens_reduction_mode": "stacked",
               "resumed_prefill_policy_cells": {"none__lens0": policy, OBJECT_HARNESS: policy},
               "lens_serve_declarations": ({} if decls is None else decls)}
    return {"registry_dir": "/tmp/synthetic/lens_registry", "rpc_method": "sisteer_install_steering",
            "worker_reports": [install],
            "scheduler_geometry": {"worker_reports": [
                {"lens_reduction_mode": "stacked", "max_num_seqs": S, "max_model_len": L,
                 "mamba_ssm_cache_dtype": "auto"}]}}


def _run(mt, tmp_path, monkeypatch, key, *extra, attestation=None, pre_table=None, pre_files=None):
    """The real ``main`` on ``key``'s campaign config with the GPU seams
    stubbed. Returns ``(snapshot, proxy, launch)``: the records as they stood
    while the standup was READY (endpoint.json is unlinked at teardown). ``pre_files`` =
    ``{name: text}`` written into ``--out`` before ``main`` runs (what the launching shell
    leaves beside the standup, e.g. full22's ``registry_runs.json``)."""
    cfg_path, man_path, cfg = _campaign(key)
    _cached_or_skip(cfg)
    bank, _ = _fake_bank(tmp_path, cfg)
    out = tmp_path / "out"
    out.mkdir()
    if pre_table is not None:
        (out / "routing_table.json").write_text(json.dumps(pre_table))
    for name, text in (pre_files or {}).items():
        (out / name).write_text(text)
    stop = out / "STOP"
    stop.touch()                                            # tear down right after READY
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    # `main` pins the lens reduction switch into THIS process's environment for its run (and restores it on
    # the way out — pinned below); touching the variable through monkeypatch FIRST — at its CURRENT value, or
    # its absence — makes pytest restore the pre-test state at teardown whatever the code under test does with
    # it, so a standup run here can never leak the switch into a later module of the same process (the
    # launcher's tests read it; integration I4)
    if mt.REDUCTION_ENV_VAR in os.environ:
        monkeypatch.setenv(mt.REDUCTION_ENV_VAR, os.environ[mt.REDUCTION_ENV_VAR])
    else:
        monkeypatch.delenv(mt.REDUCTION_ENV_VAR, raising=False)
    monkeypatch.setattr(mt.socket, "gethostname", lambda: "TESTHOST")
    captured = {}

    def fake_launch(launch, *, log_path, **_):
        captured["launch"] = launch
        att = attestation(launch) if attestation is not None else _attestation(launch)
        (Path(log_path).parent / vs.STEERING_ATTESTATION_FILE).write_text(json.dumps(att))
        return _FakeServer()
    monkeypatch.setattr(vs, "launch_server", fake_launch)
    monkeypatch.setattr(vs, "_shutdown_child", lambda proc, **kw: None)
    monkeypatch.setattr(mt, "mrope_attest", lambda launch, model_key, **kw: {
        "ok": True, "problems": [], "reports": [{"rotary_cls": "RotaryEmbedding", "patch_applied": False,
                                                  "needs_patch": False, "routed_native_calls": 0}],
        "rpc_method": mt.MROPE_RPC_METHOD, "model_key": model_key, "asked_at": "synthetic"})
    monkeypatch.setattr(mt, "_start_proxy_thread",
                        lambda app, host, port: types.SimpleNamespace(
                            call_soon_threadsafe=lambda f: None, stop=lambda: None))
    monkeypatch.setattr(px, "make_app", lambda proxy: captured.setdefault("proxy", proxy))
    monkeypatch.setattr(px, "HTTPBackend", _RecordingBackend)
    _RecordingBackend.instances.clear()

    def snapshot(stop_file, server, **kw):
        captured["endpoint"] = json.loads((out / "endpoint.json").read_text())
        captured["table"] = json.loads((out / "routing_table.json").read_text())
        captured["cells_served"] = json.loads((out / "cells_served.json").read_text())
        return 0
    monkeypatch.setattr(mt, "wait_until_stop", snapshot)
    argv = ["--repo-dir", str(REPO), "--config", str(cfg_path), "--manifest", str(man_path),
            "--defenses-dir", str(bank), "--out", str(out), "--stop-file", str(stop),
            "--mode", "attacks", "--eval", "capability:gen_long_math", "--model-key", key,
            # the LENS standup this harness exercises (native-serving-default flipped the default to lora_native)
            "--serving-path", "lens",
            "--reduction-mode", "stacked", "--require-attested-mode", "stacked", *extra]
    rc = mt.main(argv)
    assert rc == 0, rc
    captured["out"] = out
    captured["backend"] = _RecordingBackend.instances[-1]
    return captured, captured["proxy"], captured["launch"]


M5T_EXTRA = ["--upstream-timeout-s", "7200", "--lens-salt-policy", "attested"]


# --------------------------------------------------------------------------- #
# the pure helpers                                                             #
# --------------------------------------------------------------------------- #


def test_select_cells_is_the_config_list_without_the_flag_and_none_plus_the_requested_with_it(mt):
    want = ["none", "a__psr_kl_all", OBJECT, PROMPTED]
    assert mt.select_cells(want, None) == want and mt.select_cells(want, []) == want
    # a steered-wave config lists no `none` — the base route is served implicitly, first, never listed
    assert mt.select_cells([OBJECT, PROMPTED], None) == ["none", OBJECT, PROMPTED] and mt.select_cells([OBJECT], []) == ["none", OBJECT]
    assert mt.select_cells(want, ["none"]) == ["none"]
    assert mt.select_cells(want, [PROMPTED]) == ["none", PROMPTED]
    assert mt.select_cells(want, [OBJECT, "none", OBJECT, PROMPTED]) == ["none", OBJECT, PROMPTED]
    # the variant spelling rides through (its BASE must be enumerated)
    assert mt.select_cells(want, [f"{OBJECT}@thinking"]) == ["none", f"{OBJECT}@thinking"]
    with pytest.raises(SystemExit, match="not one of the config's explicit_cells"):
        mt.select_cells(want, ["b__psr_kl_all"])
    with pytest.raises(SystemExit, match="not one of the config's explicit_cells"):
        mt.select_cells(want, ["b__psr_kl_all@thinking"])
    with pytest.raises(SystemExit):
        mt.select_cells(want, [f"{OBJECT}@nope"])


def test_the_new_flags_default_to_the_legacy_behaviour(mt):
    """Without the new flags the standup composes exactly what it composed
    before (the battery's existing standup tests stay green): no parser, no
    digest, no timeout kwarg, `attested`, every explicit cell served."""
    import argparse
    src = (REPO / "scripts/sdfbase/mt_serving_up.py").read_text()
    for flag in ("--cell", "--expect-object-sha256", "--reasoning-parser", "--lens-salt-policy",
                 "--upstream-timeout-s"):
        assert f'"{flag}"' in src, flag
    # the composer receives the parser ONLY from the flag (None by default = the composer's default);
    # the seed-replicates lane (ae212d50) appended the server-default sampling policy kwarg after it,
    # so the parser line now ends in a comma and the call closes on that ADDITIVE flag
    assert "reasoning_parser=args.reasoning_parser," in src
    assert "server_default_policy=args.server_default_policy)" in src
    assert "**st.backend_timeout_kwargs(args.upstream_timeout_s)" in src
    assert st.backend_timeout_kwargs(None) == {}
    ns = argparse.Namespace(cell=None, expect_object_sha256=None, reasoning_parser=None,
                            lens_salt_policy="attested", upstream_timeout_s=None)
    assert ns.reasoning_parser is None and ns.upstream_timeout_s is None


# --------------------------------------------------------------------------- #
# the real standup on the pinned Nemotron config                               #
# --------------------------------------------------------------------------- #


def test_cell_subset_exports_the_named_cells_with_their_own_routes_and_the_served_positions_stamp(mt, tmp_path, monkeypatch):
    """`--cell` for every job of this experiment: the prompted route
    with its verified system-prefix hash, the object route, the base route and
    the derived twin; the FRESH table carries Evals' lens_served_positions stamp
    (`all` for the object cell), the standup hands it to the proxy and echoes it
    into endpoint.json with the passthrough record."""
    snap, proxy, launch = _run(mt, tmp_path, monkeypatch, "nemotron3nano4b",
                               "--cell", PROMPTED, "--cell", OBJECT, *M5T_EXTRA)
    table, ep = snap["table"], snap["endpoint"]
    routes = {r["cell_id"]: r for r in table["routes"]}
    assert set(routes) == {"none", "prompted__safety", OBJECT_HARNESS, "none__lens0"}
    assert routes["prompted__safety"]["virtual_model"] == "petri__prompted__safety"
    assert routes[OBJECT_HARNESS]["virtual_model"] == f"petri__{OBJECT_HARNESS}"
    prefixes = {s["virtual_model"]: s for s in table["prompted_prefixes"]}
    manifest = yaml.safe_load((REPO / "configs/capeval/manifests/petri_full22_nemotron3nano4b.yaml").read_text())
    assert prefixes["petri__prompted__safety"]["sha16"] == manifest["petri"]["prompted_prefix_hashes"]["safety"]
    # Evals' stamp, in the FRESH table and echoed everywhere
    assert table[st.LENS_SERVED_POSITIONS_KEY] == {OBJECT_HARNESS: ["all"]}
    assert ep[st.LENS_SERVED_POSITIONS_KEY] == {OBJECT_HARNESS: ["all"]}
    assert ep[st.LENS_SERVED_POSITIONS_PASSTHROUGH_KEY] == st.PASSTHROUGH_PASSED
    assert proxy.lens_served_positions == {OBJECT_HARNESS: ["all"]}
    assert proxy.models_payload()["petri_proxy"][st.LENS_SERVED_POSITIONS_KEY] == {OBJECT_HARNESS: ["all"]}
    # the requested spellings are recorded, `none` implicit and first
    assert ep["cells_requested"] == ["none", PROMPTED, OBJECT] == snap["cells_served"]["cells_requested"]
    assert snap["cells_served"]["harness_cells"] == ["none", "prompted__safety", OBJECT_HARNESS]
    assert set(ep["served_models"]) == {"petri__base", "petri__prompted__safety", f"petri__{OBJECT_HARNESS}",
                                        "petri__none__lens0"}


def test_the_nemotron_header_fact_is_recorded_partial_and_verbatim_and_the_standup_still_serves(mt, tmp_path, monkeypatch):
    """The exact Nemotron state: a base-only / undeclared
    standup derives the OFF header, the ON derivation is not run for any cell,
    the partial fact + the reason ride the table, endpoint.json and the proxy,
    and NOTHING substitutes a table's ids for the null on_ids."""
    snap, proxy, launch = _run(mt, tmp_path, monkeypatch, "nemotron3nano4b", "--cell", OBJECT, *M5T_EXTRA)
    table, ep = snap["table"], snap["endpoint"]
    assert table[px.THINKING_HEADER_KEY] == NEMOTRON_PARTIAL
    assert table[px.THINKING_HEADER_ERROR_KEY].startswith("ChatMarkerError: ")
    assert ep["thinking_header"] == NEMOTRON_PARTIAL and ep["thinking_header_error"] == table[px.THINKING_HEADER_ERROR_KEY]
    assert snap["cells_served"]["thinking_header"] == NEMOTRON_PARTIAL
    block = proxy.models_payload()["petri_proxy"]
    assert block["thinking_header"] == NEMOTRON_PARTIAL and block["thinking_header_error"].startswith("ChatMarkerError")
    # the registry exported the OFF header ALONE (rule 5): no second header for an undeclared cell
    reg = json.loads((snap["out"] / "lens_registry" / "registry.json").read_text())
    text = json.dumps(reg)
    assert "assistant_header_ids_thinking_on" not in text or '"assistant_header_ids_thinking_on": null' in text
    assert list(reg["assistant_header_ids"]) == NEMOTRON_OFF


def test_the_upstream_timeout_reaches_the_backend_and_the_record_and_is_null_without_the_flag(mt, tmp_path, monkeypatch):
    snap, proxy, _ = _run(mt, tmp_path / "with", monkeypatch, "nemotron3nano4b", "--cell", OBJECT, *M5T_EXTRA)
    assert proxy._backend is snap["backend"] and snap["backend"].timeout_s == 7200.0
    assert snap["endpoint"][C.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S] == 7200.0 == C.PROXY_UPSTREAM_TIMEOUT_S
    snap2, proxy2, _ = _run(mt, tmp_path / "without", monkeypatch, "nemotron3nano4b", "--cell", OBJECT)
    assert proxy2._backend is snap2["backend"] and snap2["backend"].timeout_s is None
    assert snap2["endpoint"][C.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S] is None
    assert "--upstream-timeout-s" not in " ".join(snap2["launch"].cmd)


def test_the_attested_empty_declarations_map_and_the_policy_map_reach_the_proxy(mt, tmp_path, monkeypatch):
    snap, proxy, _ = _run(mt, tmp_path, monkeypatch, "nemotron3nano4b", "--cell", OBJECT, *M5T_EXTRA)
    ep = snap["endpoint"]
    assert proxy.lens_serve_declarations == {} and proxy.lens_serve_declarations_error is None
    assert ep["lens_serve_declarations"] == {} and ep["lens_serve_declarations_error"] is None
    assert ep["resumed_prefill_policy_cells"] == {"none__lens0": "always", OBJECT_HARNESS: "always"}
    assert ep["resumed_prefill_policy_error"] is None
    assert ep["lens_salt_policy"] == "attested" == proxy.lens_salt_policy


def test_endpoint_carries_every_key_the_driver_reads_and_passes_check_serve_attestation(mt, tmp_path, monkeypatch):
    """The keys `check_serve_attestation` / `_served_identity`
    / the ledger read — additive, every legacy key kept — and the REAL
    contracts check accepts the record (in-support shape; the out-of-support
    shape is the driver lane's)."""
    cfg_path, _, cfg = _campaign("nemotron3nano4b")
    _cached_or_skip(cfg)
    bank, sha = _fake_bank(tmp_path / "pre", cfg)
    snap, proxy, launch = _run(mt, tmp_path, monkeypatch, "nemotron3nano4b", "--cell", OBJECT, "--cell", PROMPTED,
                               "--expect-object-sha256", f"{OBJECT}={sha}", *M5T_EXTRA)
    ep = snap["endpoint"]
    for key in ("lens_serve_declarations", "lens_serve_declarations_error", "thinking_header",
                "thinking_header_error", "object_artifact_sha256", "object_artifact_sha256_by_cell",
                "lens_salt_policy", "resumed_prefill_policy_cells", "resumed_prefill_policy_error",
                "chat_template_hash", "reasoning_parser", "proxy_upstream_timeout_s", "cells", "harness_cells",
                "proxy_token", "agent_loop_entry", "lens_reduction_mode", "lens_reduction_mode_error",
                st.LENS_SERVED_POSITIONS_KEY, st.LENS_SERVED_POSITIONS_PASSTHROUGH_KEY, "cells_requested"):
        assert key in ep, key
    # the legacy keys keep their values
    for key in ("proxy_url", "routing_table", "mode", "eval", "model", "model_key", "reduction_mode",
                "max_model_len", "max_num_seqs", "geometry", "petri_agent_loop_entry", "tool_calls",
                "bank_objects", "mrope_attestation", "served_models", "steering_attestation"):
        assert key in ep, key
    assert ep["agent_loop_entry"] is True is ep["petri_agent_loop_entry"]
    assert ep["reasoning_parser"] is None and ep["proxy_token"] is None
    assert ep["lens_reduction_mode"] == "stacked" and ep["lens_reduction_mode_error"] is None
    assert ep["chat_template_hash"] == hashlib.sha256((snap["out"] / "chat_template.jinja").read_bytes()).hexdigest()[:16]
    assert ep["object_artifact_sha256_by_cell"] == {OBJECT_HARNESS: sha} and ep["object_artifact_sha256"] == sha
    assert proxy.object_artifact_sha256_by_cell == {OBJECT_HARNESS: sha}
    assert ep["max_model_len"] == 17664 and ep["geometry"]["max_model_len"] == 17664     # MATH-500's own entry
    # the REAL attestation compare over the record — the driver's own code, the in-support shape
    facts = C.check_serve_attestation(ep, cell=OBJECT, arm=C.THINKING_OFF, out_of_support=False,
                                      model_key="nemotron3nano4b", expected_object_sha256=sha, budget=None)
    assert facts["object_digest"] is not None and facts[C.ENDPOINT_KEY_LENS_SALT_POLICY] == "attested"
    assert C.check_serve_attestation(ep, cell=PROMPTED, arm=C.THINKING_OFF, out_of_support=False,
                                     model_key="nemotron3nano4b", expected_object_sha256=None, budget=None)


def test_a_wrong_object_digest_refuses_before_any_launch(mt, tmp_path, monkeypatch):
    cfg_path, _, cfg = _campaign("nemotron3nano4b")
    _cached_or_skip(cfg)
    launched = []
    monkeypatch.setattr(vs, "launch_server", lambda *a, **k: launched.append(1))
    with pytest.raises(SystemExit, match="registry export refused"):
        _run(mt, tmp_path, monkeypatch, "nemotron3nano4b", "--cell", OBJECT,
             "--expect-object-sha256", f"{OBJECT}={'ab' * 32}", *M5T_EXTRA)
    assert launched == []
    with pytest.raises(SystemExit, match="names cell"):
        _run(mt, tmp_path / "b", monkeypatch, "nemotron3nano4b", "--cell", OBJECT,
             "--expect-object-sha256", f"{PROMPTED}={'ab' * 32}", *M5T_EXTRA)


def test_a_reasoning_parser_outside_the_familys_pin_is_refused_at_launch_composition(mt, tmp_path, monkeypatch):
    cfg_path, _, cfg = _campaign("nemotron3nano4b")
    _cached_or_skip(cfg)
    with pytest.raises(SystemExit, match="NO pinned reasoning parser"):
        _run(mt, tmp_path, monkeypatch, "nemotron3nano4b", "--cell", OBJECT, "--reasoning-parser", "gemma4", *M5T_EXTRA)


def test_the_routing_table_is_regenerated_on_every_start(mt, tmp_path, monkeypatch):
    """The stale-table hazard: a pre-existing table (here a pre-refinement
    one, no lens_served_positions) is OVERWRITTEN by the fresh export."""
    stale = {"version": 1, "base_model": "stale", "serving_mode": "interleaved", "exact_replay_strategy": None,
             "base_revision": None, "routes": [], "prompted_prefixes": []}
    snap, _, _ = _run(mt, tmp_path, monkeypatch, "nemotron3nano4b", "--cell", OBJECT, *M5T_EXTRA, pre_table=stale)
    assert snap["table"]["base_model"] != "stale" and snap["table"]["routes"]
    assert snap["table"][st.LENS_SERVED_POSITIONS_KEY] == {OBJECT_HARNESS: ["all"]}


def test_a_thinking_declaration_on_an_all_position_cell_is_refused_by_name(mt, tmp_path, monkeypatch):
    """`<cell>@thinking` on a positions=all object: the lens's rule 3 refuses the
    declaration as vacuous on an all-position mask (F11) on gemma and qwen alike
    — the mask does not move with the render. (gemma's equal ON header is no
    longer a refusal of its own since the rule of 2026-09-10; a
    header-LOCATED gemma cell declares and serves.) Either way the standup
    refuses BEFORE any launch, naming the reason — never a silent all-position
    serve under a declaration that changes nothing."""
    launched = []
    monkeypatch.setattr(vs, "launch_server", lambda *a, **k: launched.append(1))
    for key, match in (("gemma4e4b", "VACUOUS"), ("qwen35_4b", "vacuous|VACUOUS|declar")):
        cfg_path, _, cfg = _campaign(key)
        _cached_or_skip(cfg)
        with pytest.raises(SystemExit, match=match):
            _run(mt, tmp_path / key, monkeypatch, key, "--cell", f"{OBJECT}@thinking", *M5T_EXTRA)
    assert launched == []


def test_gemma_standup_carries_its_parser_and_a_whole_header_fact(mt, tmp_path, monkeypatch):
    snap, proxy, launch = _run(mt, tmp_path, monkeypatch, "gemma4e4b", "--cell", OBJECT,
                               "--reasoning-parser", "gemma4", *M5T_EXTRA)
    cmd = list(launch.cmd)
    assert "--reasoning-parser" in cmd and cmd[cmd.index("--reasoning-parser") + 1] == "gemma4"
    ep = snap["endpoint"]
    assert ep["reasoning_parser"] == "gemma4"
    fact = snap["table"][px.THINKING_HEADER_KEY]
    assert fact["switch"] is True and fact["moves"] is False and fact["on_ids"] == fact["off_ids"]
    assert snap["table"][px.THINKING_HEADER_ERROR_KEY] is None and ep["thinking_header_error"] is None


# --------------------------------------------------------------------------- #
# the process environment: the reduction switch never leaks past `main`         #
# --------------------------------------------------------------------------- #


def test_importing_the_standup_module_leaves_the_process_environment_unchanged():
    """The module writes ``os.environ`` only inside ``main`` (the standup path), never at import."""
    before = dict(os.environ)
    mod = _load_mt()
    assert dict(os.environ) == before
    assert mod.REDUCTION_ENV_VAR == C.LENS_STACKED_REDUCTIONS_ENV == "SISTEER_LENS_STACKED_REDUCTIONS"


def test_the_standup_restores_the_reduction_switch_it_pins_for_its_own_run(mt, tmp_path, monkeypatch):
    """``main`` pins the switch for its run and puts the caller's state back on exit — unset stays unset,
    a prior value survives — checked right after ``main`` returns, BEFORE monkeypatch's own teardown."""
    monkeypatch.delenv(mt.REDUCTION_ENV_VAR, raising=False)
    captured, _proxy, _launch = _run(mt, tmp_path / "unset", monkeypatch, "qwen35_4b", *M5T_EXTRA)
    assert captured["endpoint"][mt.REDUCTION_ENV_VAR] == "1"          # the run itself served under the pin
    assert mt.REDUCTION_ENV_VAR not in os.environ                      # and left nothing behind
    monkeypatch.setenv(mt.REDUCTION_ENV_VAR, "per_row")
    _run(mt, tmp_path / "prior", monkeypatch, "qwen35_4b", *M5T_EXTRA)
    assert os.environ[mt.REDUCTION_ENV_VAR] == "per_row"               # the caller's prior value, put back


def test_a_standup_test_leaves_no_reduction_switch_behind_for_a_later_module_in_the_same_process(tmp_path):
    """The cross-module leak as the CPU suite saw it (25 launcher tests refusing a per_row standup against a
    stale ``stacked`` expectation): ONE pytest process runs a standup test and THEN a probe module that
    asserts the variable is not in ``os.environ``. The probe must pass whether the standup test ran or was
    skipped (a tokenizer cache is not this test's precondition)."""
    probe = tmp_path / "test_zz_reduction_switch_probe.py"
    probe.write_text("import os\n\n\ndef test_no_reduction_switch_left_behind():\n"
                     f"    assert {C.LENS_STACKED_REDUCTIONS_ENV!r} not in os.environ\n")
    env = {k: v for k, v in os.environ.items() if k != C.LENS_STACKED_REDUCTIONS_ENV}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), env.get("PYTHONPATH", "")) if p)
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-o", "addopts=", "-q", "--no-header",
         f"{Path(__file__)}::test_the_standup_restores_the_reduction_switch_it_pins_for_its_own_run", str(probe)],
        capture_output=True, text=True, cwd=REPO, env=env, timeout=900, check=False)
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-1500:])
    assert "test_no_reduction_switch_left_behind PASSED" in r.stdout or " passed" in r.stdout, r.stdout[-800:]
    assert "failed" not in r.stdout.splitlines()[-1], r.stdout[-800:]


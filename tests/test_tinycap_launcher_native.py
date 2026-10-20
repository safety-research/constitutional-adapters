"""The Modal tinycap launcher under the NATIVE
default — ``scripts/modal/tinycap_served.py`` and ``scripts/modal/capctrl_served.py`` — plus the
explicit-lens argv of the Modal launchers that are not capability / tinycap surfaces (§8.6).

What is pinned here, and only here:

- the two defaults (entry point native, library lens) and the vocabulary shared with the contracts;
- the standup argv on both paths (``--serving-path`` stated on EVERY standup, ``--adapter-bank`` under
  native only) and ``launch_argv`` spelling the flag only when it is not the default;
- ``expected_identity`` under both paths: the engine-aware key set, ``reduction_mode`` recorded
  ``not_applicable`` under native, and the §1.8 kernel ceiling step PRE-COMPUTED (gemma 36 -> 31 at
  the pass-1 footprint; llama8b: no step);
- resume validation refusing native <-> lens by name, accepting native -> native;
- the three copies of the engine-first reduction-mode reader (lens_worker, this launcher,
  capctrl_served) agreeing case by case over W-A's record vocabulary, the native case included;
- the TERMINAL engine conflict at the output boundary (a completed lens root under a native run:
  ``SERVING_ENGINE_CONFLICT``, no standup, no driver, no commit, the root byte-identical);
- the native fake-standup run through ``_run_configuration``: the engine read FIRST off
  endpoint.json and held to the live proxy, the adjustments record validated, the mode recorded;
- smoke planning honouring the requested cells with a unique out-subdir, the preflight planning the
  same, the dry run under both paths;
- the Nemotron certificate key bound to the engine and to the kernel-adjusted served geometry, a lens
  certificate never admitting a native row, a native document discoverable by the real gate;
- ``capctrl_served.py`` taking the same flag and readings; the other Modal launchers naming the lens
  standup explicitly; the inventory of every reduction-mode reader under ``scripts/modal/``.

The lens path's own goldens live in ``test_tinycap_launcher.py`` (they pass ``serving_path="lens"``
explicitly since the default flipped) and stand unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import re
import shutil
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from sisteer.capeval.petri import registry_adapter as ra
from sisteer.capeval.serving import geometry as sg
from sisteer.capeval.tinycap import contracts as C

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_contracts_native_endpoint as TCN  # noqa: E402  (W-A's record vocabulary + endpoint shapes)
import test_tinycap_launcher as TL  # noqa: E402  (the modal stub, the by-path loader, the store builders)

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts/modal/tinycap_served.py"
CAPCTRL_SERVED = REPO / "scripts/modal/capctrl_served.py"
MODAL_DIR = REPO / "scripts/modal"
FIXTURE_LENS_STORE = REPO / "tests/fixtures/tinycap_identity_v1/standard_off_gsm8k.json"

NATIVE, LENS = "lora_native", "lens"
#: a gemma4e4b LoRA cell in the tinycap grammar (arm ``lora_r4`` -> object kind ``lora_all``)
LORA_CELL = "safety_refusal__lora_r4"
#: the smoke cell (the launcher's own spelling) and row
J4_CELL = "constv2-mix-full__lora_r4_ce"
J4_MODEL = "llama8b-f22it"
#: gemma4e4b at the pass-1 footprint: the budget's S and the kernel-lowered S (D 4096, ceiling 524,287 tokens)
GEMMA_L, GEMMA_S, GEMMA_S_LOWERED = 16384, 36, 31
KERNEL_STEP = {"kind": "lora_kernel_index_ceiling", "from_S": 36, "to_S": 31, "D": 4096, "ceiling": 524287}
_ABSENT = object()


@pytest.fixture(scope="module")
def L():
    return TL._load_launcher(TL._stub_modal(), app_name=TL.TEST_APP_NAME, name="tinycap_served_native_under_test")


def _load_by_path(name: str, path: Path, stub):
    """A Modal script loaded by path under ``stub`` as ``modal``; the prior ``modal`` restored."""
    prior = sys.modules.get("modal", _ABSENT)
    sys.modules["modal"] = stub
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if prior is _ABSENT:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prior
    return mod


@pytest.fixture(scope="module")
def capdriver():
    """``scripts/modal/capctrl_served.py`` — the x2 capability wave's Modal launcher."""
    import test_grokmt_served_resume as TG
    return _load_by_path("capctrl_served_native_under_test", CAPCTRL_SERVED, TG._stub_modal())


def _plan(L, cells, *, serving_path=LENS, model=C.MODEL_SLUG, controls=("gsm8k",), modes=("off",), **kw):
    return L.plan_configurations(cells=list(cells), modes=list(modes), controls=list(controls), model=model,
                                 serving_path=serving_path, **kw)


def _expected(L, cfg, *, serving_path, tree=None, **kw):
    """The launcher's own identity derivation for ``cfg`` (the lens one at the launcher's default
    ``per_row`` — the environment's stacked-by-default is not what the stores here record)."""
    geom, _ = L.served_geometry(cfg, 1)
    kw.setdefault("reduction_mode", C.REDUCTION_MODE_LAUNCHER_DEFAULT)
    return L.expected_identity(cfg, geom, pass_no=1, controls=cfg["controls"], samples=C.MATH500_SAMPLES_PER_ITEM,
                               limit=0, inspect_limit=0, tree=tree, tree_sha=TL.FULL_SHA, serving_path=serving_path, **kw)


def _native_store(out: Path, exp) -> Path:
    """A tinycap store the served driver leaves behind a NATIVE standup: the hashed served block
    (``serving_engine`` present, ``reduction_mode`` not_applicable — from ``exp``) and the route
    block's unhashed engine mirror."""
    return TL._store(out, exp, mutate=lambda m: m[C.MANIFEST_KEY_ROUTE].update({C.SERVED_ENGINE_FIELD: NATIVE}))


def _tree_hashes(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


# --------------------------------------------------------------------------- #
# 1. the defaults and the vocabulary                                            #
# --------------------------------------------------------------------------- #


def test_the_two_defaults_and_the_vocabulary_are_the_contracts(L):
    """Entry-point default native (= registry_adapter.DEFAULT_STANDUP_SERVING_PATH);
    library default lens (a direct call composes today's lens launch); every shared literal read off
    the contracts / the core; the standup's parser takes both flags."""
    from sisteer.capeval.serving import native_standup as ns
    assert L.SERVING_PATH_DEFAULT == ra.DEFAULT_STANDUP_SERVING_PATH == NATIVE
    assert L.SERVING_PATH_LIBRARY_DEFAULT == LENS != L.SERVING_PATH_DEFAULT
    assert L.SERVING_PATHS == (NATIVE, LENS) and L.SERVING_PATH_FLAG == "--serving-path"
    assert L.SERVED_ENGINE_FIELD == C.SERVED_ENGINE_FIELD == "serving_engine"
    assert L.REDUCTION_MODE_NOT_APPLICABLE == C.REDUCTION_MODE_NOT_APPLICABLE == "not_applicable"
    assert L.REDUCTION_MODE_NOT_APPLICABLE not in C.REDUCTION_MODES
    assert L.REC_SERVING_ENGINE_MISMATCH == C.REC_SERVING_ENGINE_MISMATCH == "SERVING_ENGINE_MISMATCH"
    assert L.REC_SERVING_ENGINE_CONFLICT == C.REC_SERVING_ENGINE_CONFLICT == "SERVING_ENGINE_CONFLICT"
    assert L.ENDPOINT_KEY_ENGINE == C.ENDPOINT_KEY_ENGINE == "engine"
    assert L.ENGINE_LORA_NATIVE == C.SERVING_ENGINE_LORA_NATIVE == NATIVE
    assert L.ENGINE_LENS == C.SERVING_ENGINE_LENS == LENS and L.ENGINES == C.SERVING_ENGINES
    assert L.ADAPTER_BANK_ROOT_ENV == ns.ADAPTER_BANK_ROOT_ENV == "SISTEER_ADAPTER_BANK_ROOT"
    assert L.ADAPTER_BANK_ROOT == f"{C.VOL_TINYCAP_ROOT}/multilora_adapters" == "/vol/tinycap/multilora_adapters"
    assert L.ADAPTER_BANK_FLAG == "--adapter-bank" and L.OUT_SUBDIR_NATIVE == "native"
    assert L.native_gen_root() == f"{C.VOL_GEN_ROOT}/native" and L.native_gen_root("/x/") == "/x/native"
    assert L.ADJUSTMENT_KINDS == sg.ADJUSTMENT_KINDS and L.GEOMETRY_ADJUSTMENTS_KEY == sg.ADJUSTMENTS_KEY
    assert L.ADJUSTMENT_KERNEL == sg.ADJUSTMENT_KERNEL and L.ADJUSTMENT_KV == sg.ADJUSTMENT_KV
    assert dict(L._LORA_MAX_MODULE_DIM_BY_MODEL_FALLBACK) == dict(sg.LORA_MAX_MODULE_DIM_BY_MODEL)
    assert L.PROXY_IDENTITY_KEY == "petri_proxy" and L.PROXY_IDENTITY_ENGINE_KEY == "engine"
    # the closed vocabulary
    for bad in ("", "native", "LENS", "Lora_native", None, 3):
        with pytest.raises(ValueError, match="--serving-path must be one of"):
            L.check_serving_path_arg(bad)
    assert L.check_serving_path_arg(NATIVE) == NATIVE and L.check_serving_path_arg(LENS) == LENS
    assert L.is_native_path(NATIVE) and not L.is_native_path(LENS)
    assert L.engine_of_serving_path(NATIVE) == NATIVE and L.engine_of_serving_path(LENS) == LENS
    # the plan marker: present iff native (a lens plan is byte-identical), read off the configuration
    assert L.SERVING_PATH_KEY == "serving_path"
    assert L.serving_path_of({}) == LENS and L.serving_path_of({L.SERVING_PATH_KEY: NATIVE}) == NATIVE
    with pytest.raises(ValueError, match="--serving-path must be one of"):
        L.serving_path_of({L.SERVING_PATH_KEY: "vllm"})
    # the standup's parser reads both flags back
    up = TL._load_standup()
    base = ["--repo-dir", "r", "--config", "c", "--cell", "none", "--defenses-dir", "d", "--out", "o", "--stop-file", "s"]
    ns_ = up.build_parser().parse_args([*base, "--serving-path", NATIVE, "--adapter-bank", "/b/k"])
    assert ns_.serving_path == NATIVE and str(ns_.adapter_bank) == "/b/k"
    ns_ = up.build_parser().parse_args([*base, "--serving-path", LENS])
    assert ns_.serving_path == LENS and ns_.adapter_bank is None
    # the adapter bank per model, never without one
    assert L.adapter_bank_for("gemma4e4b") == "/vol/tinycap/multilora_adapters/gemma4e4b"
    assert L.adapter_bank_for("llama8b") == "/vol/tinycap/multilora_adapters/llama8b"
    with pytest.raises(ValueError, match="names none"):
        L.adapter_bank_for(None)


def test_the_disposition_at_plan_time_names_the_kind_and_the_escape_hatch(L):
    """A skip-class object cell (a Class B id whose kind is outside NATIVE_LORA_SERVABLE_KINDS) has
    nothing to serve natively: refused by name at PLAN time, naming ``--serving-path lens``; LoRA
    cells, ``none``, the prompted cells AND — since native-psr-serving — the residual-slot kinds
    (a psr cell) plan natively; the OBJECT's position mode is judged at standup. The verdict is
    W-A's disposition (``defense_cells.native_serving_disposition``)."""
    from sisteer.capeval.petri import defense_cells as dc
    assert L.native_kind_of_cell("none") is None and L.native_kind_of_cell(C.CELL_PROMPTED_SAFETY) is None
    assert L.native_kind_of_cell(LORA_CELL) == "lora_all" == ra.ARM_OBJECT_KINDS["lora_r4"]
    assert L.native_kind_of_cell(J4_CELL) == ra.ARM_OBJECT_KINDS["lora_r4_ce"]
    assert L.native_kind_of_cell(C.CELL_PSRKL_SAFETY) == "psr"
    assert L.native_serving_disposition("psr") == dc.native_serving_disposition("psr", None)
    assert L.native_serving_disposition("psr")[0] == "serve"          # native-psr-serving: the KIND is servable
    assert L.native_serving_disposition("lora_all")[0] == "serve" == dc.native_serving_disposition("lora_all", None)[0]
    for ok in ("none", C.CELL_PROMPTED_SAFETY, LORA_CELL, J4_CELL, "constv2-mix-full__lora_r4_ce_minus_ctl",
               C.CELL_PSRKL_SAFETY):
        assert L.native_cell_refusal(ok) is None, ok
    why = L.native_cell_refusal("antidote")                            # a Class B id outside the servable kinds
    assert why and "antidote" in why and "'antidote'" in why and "--serving-path lens" in why
    assert "user ruling 2026-09-13" not in why                        # the superseded PSR-is-out clause is gone
    assert "NATIVE_LORA_SERVABLE_KINDS" in why
    # the planner: a psr cell plans under EITHER path (the object's mode is judged at standup)
    assert [c["cell"] for c in _plan(L, [C.CELL_PSRKL_SAFETY], serving_path=NATIVE)] == [C.CELL_PSRKL_SAFETY]
    assert [c["cell"] for c in _plan(L, [C.CELL_PSRKL_SAFETY], serving_path=LENS)] == [C.CELL_PSRKL_SAFETY]
    assert [c["cell"] for c in _plan(L, [LORA_CELL, "none", C.CELL_PROMPTED_SAFETY], serving_path=NATIVE)] == [
        LORA_CELL, "none", C.CELL_PROMPTED_SAFETY]


# --------------------------------------------------------------------------- #
# 2. the argv: every standup states its path; the bank rides under native only  #
# --------------------------------------------------------------------------- #


def test_every_standup_argv_states_its_path_and_the_bank_rides_under_native_only(L, tmp_path):
    """``--serving-path <path>`` in the FIXED part right after the readback flag, on BOTH paths
    (the standup's own default is native now — a lens launch that omitted it would silently stand up
    the other engine); ``--adapter-bank <root>/<model key>`` right after it under native ONLY. The lens
    argv is the library default's and equals the historical argv plus those two tokens; the native argv
    differs from the lens one in exactly the path value and the bank."""
    up = TL._load_standup()
    cfgs = _plan(L, [LORA_CELL, "none", C.CELL_PROMPTED_SAFETY], serving_path=NATIVE)
    for cfg in cfgs:
        geom, _ = L.served_geometry(cfg, 1)
        assert geom.model_key == C.MODEL_SLUG == "gemma4e4b"
        bank = L.adapter_bank_for(geom.model_key)
        sha = "a" * 64 if cfg["steered"] else ""
        args = ("/tree", "/tree/cfg.yaml", cfg["cell"], "d", tmp_path, tmp_path / "STOP", geom, 64, cfg["standup_extra"])
        common = dict(object_sha256=sha, declared=L.standup_declared(cfg))
        native = L._standup_cmd(*args, serving_path=NATIVE, adapter_bank=bank, **common)
        lens = L._standup_cmd(*args, serving_path=LENS, **common)
        assert L._standup_cmd(*args, **common) == lens                     # the library default IS the lens argv
        i = native.index(C.FLAG_ENABLE_LOG_REQUESTS)
        assert native[i - 4:i] == ["--ready-timeout-s", "2700", C.FLAG_UPSTREAM_TIMEOUT_S, "7200"]
        assert native[i + 1:i + 5] == [L.SERVING_PATH_FLAG, NATIVE, L.ADAPTER_BANK_FLAG, bank], cfg["name"]
        assert lens[i + 1:i + 3] == [L.SERVING_PATH_FLAG, LENS] and L.ADAPTER_BANK_FLAG not in lens
        assert native[:i + 1] == lens[:i + 1] and native[i + 5:] == lens[i + 3:]          # the ONLY difference
        assert native.count(L.SERVING_PATH_FLAG) == 1 == lens.count(L.SERVING_PATH_FLAG)
        assert native.count(L.ADAPTER_BANK_FLAG) == 1
        tail = native[i + 5:]
        assert tail == ([L.OBJECT_DIGEST_FLAG, sha] if sha else []) + list(cfg["standup_extra"])   # digest, then extras
        ns_n = up.build_parser().parse_args(native[2:])
        assert ns_n.serving_path == NATIVE and str(ns_n.adapter_bank) == bank and ns_n.cell == cfg["cell"]
        ns_l = up.build_parser().parse_args(lens[2:])
        assert ns_l.serving_path == LENS and ns_l.adapter_bank is None
        # a wiring error is a refusal, not a launch
        with pytest.raises(ValueError, match="needs its adapter bank"):
            L._standup_cmd(*args, serving_path=NATIVE, **common)
        with pytest.raises(ValueError, match="offered to a lens standup"):
            L._standup_cmd(*args, serving_path=LENS, adapter_bank=bank, **common)
        with pytest.raises(ValueError, match="--serving-path must be one of"):
            L._standup_cmd(*args, serving_path="native", **common)


def test_launch_argv_spells_the_path_only_when_it_is_not_the_native_default(L):
    """The printed ``modal run`` argv: a lens launch is never implicit (``--serving-path lens`` is
    appended), a native launch spells nothing (the default), an unknown path refuses."""
    base = L.launch_argv("abc1234", pass1=True)
    assert L.SERVING_PATH_FLAG not in base
    assert L.launch_argv("abc1234", pass1=True, serving_path=NATIVE) == base
    lens = L.launch_argv("abc1234", pass1=True, serving_path=LENS)
    assert lens == [*base, L.SERVING_PATH_FLAG, LENS]
    smoke = L.launch_argv("abc1234", smoke=True, cells=f"{LORA_CELL},none")
    assert "--smoke" in smoke and f"{LORA_CELL},none" in smoke and L.SERVING_PATH_FLAG not in smoke
    with pytest.raises(ValueError, match="--serving-path must be one of"):
        L.launch_argv("abc1234", pass1=True, serving_path="vllm")


# --------------------------------------------------------------------------- #
# 3. expected_identity: the engine-aware key set and the pre-computed kernel step (§1.8)  #
# --------------------------------------------------------------------------- #


def test_expected_identity_derives_the_engine_the_mode_and_the_pre_computed_kernel_step(L):
    """Under native the served block gains ``serving_engine`` (through
    ``served_identity_fields(native=True)`` — the driver's own key set) and records
    ``reduction_mode`` as ``not_applicable``; the kernel ceiling step is PRE-COMPUTED so the
    expected S is the S the standup will serve (gemma 36 -> 31 at the pass-1 footprint); under lens
    every derivation is byte-identical to the argument-less call. llama8b: D 4096 at 16384 x 29 —
    no step, ``[]``."""
    cfg_n = _plan(L, [LORA_CELL], serving_path=NATIVE)[0]
    cfg_l = _plan(L, [LORA_CELL], serving_path=LENS)[0]
    assert cfg_n[L.SERVING_PATH_KEY] == NATIVE and L.SERVING_PATH_KEY not in cfg_l
    assert cfg_n["out_dir"] == f"{C.VOL_GEN_ROOT}/native/gemma4e4b/{LORA_CELL}/thinking_off"
    assert cfg_l["out_dir"] == f"{C.VOL_GEN_ROOT}/gemma4e4b/{LORA_CELL}/thinking_off"
    assert ({k: v for k, v in cfg_n.items() if k not in (L.SERVING_PATH_KEY, "out_dir")}
            == {k: v for k, v in cfg_l.items() if k != "out_dir"})                    # the ONE marker + the namespace
    geom, _ = L.served_geometry(cfg_n, 1)
    assert (geom.max_model_len, geom.max_num_seqs, geom.model_key) == (GEMMA_L, GEMMA_S, "gemma4e4b")
    # (the lens derivation at the launcher's default per_row: with no --reduction-mode the identity
    # reads the environment — stacked when unset since E46 — which is not the point here)
    kw = dict(pass_no=1, controls=cfg_n["controls"], samples=1, reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT)
    exp_n = L.expected_identity(cfg_n, geom, serving_path=NATIVE, **kw)
    exp_l = L.expected_identity(cfg_l, geom, **kw)
    assert L.expected_identity(cfg_l, geom, serving_path=LENS, **kw) == exp_l           # lens: byte-identical
    assert tuple(exp_l.served) == C.SERVED_IDENTITY_FIELDS == C.served_identity_fields()
    assert tuple(exp_n.served) == C.served_identity_fields(native=True) == (*C.SERVED_IDENTITY_FIELDS, "serving_engine")
    assert exp_n.served["serving_engine"] == NATIVE and exp_n.served["reduction_mode"] == "not_applicable"
    assert exp_l.served["reduction_mode"] == "per_row" and "serving_engine" not in exp_l.served
    assert exp_n.serving_engine == NATIVE and exp_n.native is True
    assert exp_l.serving_engine == LENS and exp_l.native is False
    # the kernel step, pre-computed: the expected S — and the driver's --concurrency — is the served S
    assert sg.kernel_ceiling_adjustment("gemma4e4b", GEMMA_L, GEMMA_S) == KERNEL_STEP
    assert L.kernel_ceiling_adjustment("gemma4e4b", GEMMA_L, GEMMA_S, sg=sg) == KERNEL_STEP
    assert L.kernel_ceiling_adjustment("gemma4e4b", GEMMA_L, GEMMA_S) == KERNEL_STEP          # the launcher's fallback: the same formula
    assert exp_n.served["max_num_seqs"] == exp_n.served["concurrency"] == GEMMA_S_LOWERED
    assert exp_l.served["max_num_seqs"] == exp_l.served["concurrency"] == GEMMA_S
    for f in C.SERVED_IDENTITY_FIELDS:
        if f not in ("max_num_seqs", "concurrency", "reduction_mode"):
            assert exp_n.served[f] == exp_l.served[f], f
    # a lens reduction-mode argument is not consulted under native (main refuses a non-default one first)
    assert L.expected_identity(cfg_n, geom, serving_path=NATIVE, **{**kw, "reduction_mode": "stacked"}) == exp_n
    assert L.expected_identity(cfg_n, geom, serving_path=NATIVE, **{**kw, "reduction_mode": None}) == exp_n
    g2, s2 = L.expected_geometry_for(geom, native=True)
    assert s2 == KERNEL_STEP and (g2.max_model_len, g2.max_num_seqs) == (GEMMA_L, GEMMA_S_LOWERED)
    assert g2.seqs_source == geom.seqs_source and not g2.is_identity and geom.is_identity
    assert L.expected_geometry_for(geom, native=False) == (geom, None)
    # llama8b: no step at this footprint
    cfg_ll = _plan(L, ["none"], serving_path=NATIVE, model="llama8b")[0]
    gl, _ = L.served_geometry(cfg_ll, 1)
    assert gl.model_key == "llama8b" and (gl.max_model_len, gl.max_num_seqs) == (16384, 29)
    assert sg.kernel_ceiling_adjustment("llama8b", 16384, 29) is None
    assert L.expected_geometry_for(gl, native=True) == (gl, None)
    exp_ll = L.expected_identity(cfg_ll, gl, serving_path=NATIVE, **kw)
    assert exp_ll.served["max_num_seqs"] == 29 and exp_ll.served["serving_engine"] == NATIVE
    # a native expectation needs the model: refused, never a silent identity
    with pytest.raises(L.IdentityUnderivable, match="names no model_key"):
        L.expected_geometry_for(SimpleNamespace(max_model_len=GEMMA_L, max_num_seqs=GEMMA_S, model_key=None), native=True)
    with pytest.raises(L.IdentityUnderivable, match="no pinned LoRA max module dim"):
        L.expected_geometry_for(SimpleNamespace(max_model_len=GEMMA_L, max_num_seqs=GEMMA_S, model_key="nope"), native=True)
    # the expectation at the served S keeps the engine
    assert L.expected_at_served_seqs(exp_n, GEMMA_S_LOWERED) == exp_n
    assert L.expected_at_served_seqs(exp_n, 30).served["serving_engine"] == NATIVE


# --------------------------------------------------------------------------- #
# 4. resume validation and the terminal engine conflict at the output boundary #
# --------------------------------------------------------------------------- #


def test_resume_validation_is_engine_aware_and_refuses_cross_engine_stores_by_name(L, tmp_path):
    """The store-trust exact-keys check compares the mode-AND-engine-aware key set: a native store
    resumes onto a native run and never onto a lens one (or the reverse), refused BY NAME; the route
    block's unhashed engine mirror is held too."""
    tree = TL._tree(tmp_path / "repos", "f6cf462")
    cfg_n = _plan(L, [LORA_CELL], serving_path=NATIVE)[0]
    cfg_l = _plan(L, [LORA_CELL], serving_path=LENS)[0]
    exp_n = _expected(L, cfg_n, serving_path=NATIVE, tree=tree)
    exp_l = _expected(L, cfg_l, serving_path=LENS, tree=tree)
    n_store = _native_store(tmp_path / "n", exp_n)
    l_store = TL._store(tmp_path / "l", exp_l)
    assert L.untrusted_store_reason(n_store, expected=exp_n) is None
    assert L.untrusted_store_reason(l_store, expected=exp_l) is None
    why = L.untrusted_store_reason(n_store, expected=exp_l)
    assert why and "serving_engine" in why and "never resume onto each other" in why and "'lens'" in why
    why = L.untrusted_store_reason(l_store, expected=exp_n)
    assert why and "serving_engine" in why and "never resume onto each other" in why and f"'{NATIVE}'" in why
    # the route mirror: a native store that forgot it, a lens store that names the other engine
    n2 = TL._store(tmp_path / "n2", exp_n)
    assert f"manifest.route.serving_engine None != '{NATIVE}'" in (L.untrusted_store_reason(n2, expected=exp_n) or "")
    l2 = TL._store(tmp_path / "l2", exp_l, mutate=lambda m: m[C.MANIFEST_KEY_ROUTE].update({C.SERVED_ENGINE_FIELD: NATIVE}))
    assert "this run serves the lens" in (L.untrusted_store_reason(l2, expected=exp_l) or "")
    # the resume report files the native store as DONE for the native run only
    for c in cfg_n["controls"]:
        for t in L.expected_tasks(c, "off"):
            TL._finish(n_store, t, L.FULL_ITEM_COUNT_OF_TASK[t])
    assert L.resume_report(tmp_path / "n", cfg_n["controls"], exp_n)["controls"]["gsm8k"]["done_in"] == n_store.name
    assert L.resume_report(tmp_path / "n", cfg_l["controls"], exp_l)["controls"]["gsm8k"]["done_in"] is None
    # the store's engine, read the way the terminal check reads it
    read = lambda d: json.loads((d / "manifest.json").read_text())  # noqa: E731
    assert L.store_serving_engine(read(n_store)) == (NATIVE, "route.serving_engine")
    assert L.store_serving_engine(read(n2)) == (NATIVE, "tinycap.served.serving_engine")
    assert L.store_serving_engine(read(l_store)) == (LENS, "absent (a store from before the native landing: lens by era)")


def test_engine_conflict_at_output_judges_every_store_under_the_root(L, tmp_path):
    """Identity-matching or not: any store under ``<out_dir>/cells/`` served by another engine than
    this run's is a conflict — a real lens fixture store (no engine anywhere: lens by era) under a
    native run included; an unreadable manifest fails closed; a same-engine root is none."""
    tree = TL._tree(tmp_path / "repos", "f6cf462")
    exp_n = _expected(L, _plan(L, [LORA_CELL], serving_path=NATIVE)[0], serving_path=NATIVE, tree=tree)
    exp_l = _expected(L, _plan(L, [LORA_CELL], serving_path=LENS)[0], serving_path=LENS, tree=tree)
    n_store = _native_store(tmp_path / "n", exp_n)
    l_store = TL._store(tmp_path / "l", exp_l)
    assert L.engine_conflict_at_output(tmp_path / "n", exp_n) is None
    assert L.engine_conflict_at_output(tmp_path / "l", exp_l) is None
    assert L.engine_conflict_at_output(tmp_path / "nowhere", exp_n) is None
    c = L.engine_conflict_at_output(tmp_path / "n", exp_l)
    assert c["expected_engine"] == LENS and [s["store"] for s in c["stores"]] == [n_store.name]
    assert c["stores"][0]["engine"] == NATIVE and c["stores"][0]["cell_id"] == LORA_CELL
    c = L.engine_conflict_at_output(tmp_path / "l", NATIVE)                     # an engine name works too
    assert c["stores"][0]["engine"] == LENS and c["stores"][0]["problem"].startswith("served by 'lens' (absent")
    # the REAL fixture store (a lens PSR store from before the landing): a conflict by era under native
    fx = json.loads(FIXTURE_LENS_STORE.read_text())
    d = tmp_path / "fx" / "cells" / fx["cell_dir"]
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(fx["manifest"]))
    assert fx["manifest"]["route"]["serving_identity"] == "proxy_vllm+lens+interleaved"
    assert C.SERVED_ENGINE_FIELD not in fx["manifest"]["route"] and C.SERVED_ENGINE_FIELD not in fx["manifest"]["tinycap"]["served"]
    c = L.engine_conflict_at_output(tmp_path / "fx", NATIVE)
    assert c["stores"][0]["cell_id"] == "safety_refusal__psr_kl_all" and c["stores"][0]["engine"] == LENS
    assert c["stores"][0]["tasks"] == list(fx["manifest"]["tinycap"]["tasks"])
    assert L.engine_conflict_at_output(tmp_path / "fx", LENS) is None
    msg = L.engine_conflict_error(c, tmp_path / "fx")
    assert "never re-generated on the other engine" in msg and "--serving-path lens" in msg and "nothing written" in msg
    # an unreadable manifest: its engine cannot be read -> a conflict (fail closed); an unknown engine too
    bad = tmp_path / "bad" / "cells" / "x"
    bad.mkdir(parents=True)
    (bad / "manifest.json").write_text("{not json")
    c = L.engine_conflict_at_output(tmp_path / "bad", NATIVE)
    assert c["stores"][0]["engine"] is None and "cannot be read" in c["stores"][0]["problem"]
    odd = tmp_path / "odd" / "cells" / "y"
    odd.mkdir(parents=True)
    (odd / "manifest.json").write_text(json.dumps({"route": {C.SERVED_ENGINE_FIELD: "vllm"}}))
    c = L.engine_conflict_at_output(tmp_path / "odd", NATIVE)
    assert c["stores"][0]["engine"] == "vllm" and "unknown engine" in c["stores"][0]["problem"]
    # the whole root is judged, mixed roots included: one lens store beside a native one under native
    shutil.copytree(l_store, tmp_path / "n" / "cells" / l_store.name)
    c = L.engine_conflict_at_output(tmp_path / "n", exp_n)
    assert [s["store"] for s in c["stores"]] == [l_store.name]


# --------------------------------------------------------------------------- #
# 5. the three copies of the engine-first reduction-mode reader                 #
# --------------------------------------------------------------------------- #


def _record_vocabulary():
    """W-A's case list (``tests/test_contracts_native_endpoint.py``), the native case included."""
    KEY, ERR = C.PROXY_IDENTITY_REDUCTION_MODE, C.LENS_REDUCTION_MODE_ERROR_KEY
    why = "scheduler-config worker 0: no lens_reduction_mode in its scheduler-config report"
    return [{KEY: "per_row"}, {KEY: "stacked"},
            {KEY: "per_row", ERR: None}, {KEY: "stacked", ERR: ""},
            {}, {"proxy_url": "http://127.0.0.1:1/v1"},
            {KEY: None, ERR: why}, {ERR: why}, {KEY: None, ERR: ""},
            {KEY: None}, {KEY: None, ERR: None},
            {KEY: "vectorized"}, {KEY: "STACKED"}, {KEY: ""}, {KEY: 1}, {KEY: True},
            None, [], "endpoint.json", 3,
            {"engine": NATIVE}, TCN.core_endpoint(), TCN.core_endpoint(with_row_spelling=False),
            {"engine": NATIVE, KEY: None}, {"engine": NATIVE, KEY: None, ERR: None},
            {"engine": NATIVE, KEY: "stacked"},
            {"engine": NATIVE, KEY: "per_row", ERR: why},
            {"engine": "vllm", KEY: "stacked"},
            {"engine": None, KEY: "per_row"},
            {"engine": LENS, KEY: "stacked"}, {"engine": LENS},
            {C.SERVED_ENGINE_FIELD: LENS, KEY: "per_row"},
            {C.SERVED_ENGINE_FIELD: NATIVE},
            {C.SERVED_ENGINE_FIELD: NATIVE, KEY: "stacked"},
            {"engine": NATIVE, C.SERVED_ENGINE_FIELD: LENS},
            {"engine": LENS, C.SERVED_ENGINE_FIELD: NATIVE},
            {"engine": "LORA_NATIVE"}, {"engine": ""}, {"engine": 1},
            TCN.ident_native(), TCN.ident_lens("stacked"), TCN.ident_lens(None),
            {**TCN.ident_native(), KEY: "stacked"},
            {"engine": {"kind": "lens_plugin"}, KEY: "stacked"},
            TCN.capctrl_endpoint(), TCN.mt_endpoint(mode="per_row"),
            TCN.capctrl_endpoint(mode=None, mode_error="workers disagree")]


def test_the_three_copies_of_the_reduction_mode_reader_agree_including_the_native_case(L, capdriver):
    """Parity pin over W-A's whole record vocabulary: ``lens_worker.endpoint_reduction_mode`` (the
    consumer's), this launcher's ``endpoint_reduction_mode`` (imports nothing — the read site runs
    with the staged tree first on sys.path) and ``capctrl_served.resolve_reduction_mode`` read the
    ENGINE FIRST and agree case by case; the contracts' ``served_reduction_mode`` too."""
    from sisteer.capeval.serving import lens_worker as lw

    def read(fn, record, exc):
        try:
            got = fn(record)
        except exc:
            return None, True
        return (got[0] if isinstance(got, tuple) else got), False

    for record in _record_vocabulary():
        theirs = read(lw.endpoint_reduction_mode, record, lw.ReductionModeUnattested)
        assert read(L.endpoint_reduction_mode, record, L.ReductionModeUnattested) == theirs, record
        assert read(capdriver.resolve_reduction_mode, record, capdriver.ReductionModeUnattested) == theirs, record
        assert read(C.served_reduction_mode, record, ValueError) == theirs, record
    # the native case, spelled out on each copy
    assert L.endpoint_reduction_mode({"engine": NATIVE}) == "not_applicable"
    assert L.endpoint_reduction_mode(TCN.core_endpoint()) == "not_applicable"
    assert capdriver.resolve_reduction_mode({"engine": NATIVE}) == ("not_applicable", "engine")
    assert capdriver.resolve_reduction_mode(TCN.ident_native()) == ("not_applicable", "engine")
    for torn in ({"engine": NATIVE, C.PROXY_IDENTITY_REDUCTION_MODE: "stacked"},
                 {"engine": NATIVE, C.SERVED_ENGINE_FIELD: LENS}, {"engine": "vllm"}):
        with pytest.raises(L.ReductionModeUnattested):
            L.endpoint_reduction_mode(torn)
        with pytest.raises(capdriver.ReductionModeUnattested):
            capdriver.resolve_reduction_mode(torn)
    # the pre-change readings are byte-identical on both copies
    assert L.endpoint_reduction_mode({}) == "per_row" == capdriver.resolve_reduction_mode({})[0]
    assert L.endpoint_reduction_mode({C.PROXY_IDENTITY_REDUCTION_MODE: "stacked"}) == "stacked"
    assert capdriver.resolve_reduction_mode({C.PROXY_IDENTITY_REDUCTION_MODE: "stacked"}) == ("stacked", "attested")
    # the engine is read before the lens keys, in both copies; neither imports from the tree
    for src in (inspect.getsource(L.endpoint_reduction_mode), inspect.getsource(capdriver.resolve_reduction_mode)):
        assert src.index("ENGINE") < src.index("why_none = ") and "import" not in src.split('"""')[2]
    assert capdriver.REDUCTION_MODE_NOT_APPLICABLE == C.REDUCTION_MODE_NOT_APPLICABLE
    assert capdriver.ENDPOINT_ENGINE_KEY == C.ENDPOINT_KEY_ENGINE and capdriver.SERVED_ENGINE_FIELD == C.SERVED_ENGINE_FIELD


def test_the_endpoint_engine_and_the_live_proxy_engine_readers(L, monkeypatch):
    """``endpoint_engine``: ``engine`` (bare or the ``/v1/models`` block's ``{"kind": …}``), else the
    row spelling, else lens BY ERA; a torn pair or an unknown value refuses. ``proxy_engine_kind``:
    the LIVE proxy's ``petri_proxy.engine.kind`` over ``GET <proxy_url>/models`` with the bearer token,
    ``(None, why)`` for every shape that attests no engine."""
    assert L.endpoint_engine({}) == LENS and L.endpoint_engine({"proxy_url": "http://x/v1"}) == LENS
    assert L.endpoint_engine({"engine": NATIVE}) == NATIVE == L.endpoint_engine(TCN.core_endpoint())
    assert L.endpoint_engine(TCN.ident_native()) == NATIVE == L.endpoint_engine({C.SERVED_ENGINE_FIELD: NATIVE})
    assert L.endpoint_engine({"engine": LENS}) == LENS == L.endpoint_engine(TCN.capctrl_endpoint())
    for torn in ({"engine": NATIVE, C.SERVED_ENGINE_FIELD: LENS}, {"engine": "vllm"}, {"engine": ""}, [], None, "ep"):
        with pytest.raises(ValueError):
            L.endpoint_engine(torn)
    assert L.endpoint_is_native(TCN.core_endpoint()) and L.endpoint_is_native(TCN.ident_native())
    assert not L.endpoint_is_native({}) and not L.endpoint_is_native(TCN.capctrl_endpoint())
    seen: list = []
    payload: dict = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(payload).encode()

    def fake_urlopen(req, timeout=None):
        seen.append((req.full_url, req.get_header("Authorization"), timeout))
        if payload.get("__raise__"):
            raise OSError("connection refused")
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    ep = {"proxy_url": "http://127.0.0.1:1/v1/", "proxy_token": "t0k"}
    payload.update({"petri_proxy": {"engine": {"kind": NATIVE, "adapters": []}}})
    assert L.proxy_engine_kind(ep) == (NATIVE, None)
    assert seen[-1] == ("http://127.0.0.1:1/v1/models", "Bearer t0k", 30.0)
    payload.clear()
    payload.update({"petri_proxy": {"engine": {"kind": LENS}}})
    assert L.proxy_engine_kind({"proxy_url": "http://127.0.0.1:1/v1"}, timeout_s=5) == (LENS, None)
    assert seen[-1] == ("http://127.0.0.1:1/v1/models", None, 5)
    payload.clear()
    payload.update({"petri_proxy": {C.PROXY_IDENTITY_REDUCTION_MODE: "stacked"}})       # a lens proxy: no engine block
    kind, why = L.proxy_engine_kind(ep)
    assert kind is None and "attests no engine" in why
    payload.clear()
    payload.update({"data": []})
    assert "exposes no 'petri_proxy'" in L.proxy_engine_kind(ep)[1]
    payload.clear()
    payload.update({"petri_proxy": {"engine": NATIVE}})                                 # a bare string: malformed
    assert "malformed" in L.proxy_engine_kind(ep)[1]
    payload.clear()
    payload.update({"__raise__": True})
    assert "GET http://127.0.0.1:1/v1/models failed: OSError" in L.proxy_engine_kind(ep)[1]
    assert L.proxy_engine_kind({"proxy_url": ""}) == (None, "endpoint.json names no proxy_url")


# --------------------------------------------------------------------------- #
# 6. served_geometry_of_endpoint: the ONE adjustment record (§1.8)               #
# --------------------------------------------------------------------------- #


def test_served_geometry_of_endpoint_honours_a_launch_class_step_with_geometry_semantics(L, tmp_path, monkeypatch):
    """FLATMOUNTVOCABPIN: geometry.py's closed adjustment vocabulary LEADS with `launch_class` (53bd76ed0 / 3f1a19b48 — the pinned site
    record sizes S from the launches whose adapter class dominates the served standup); the launcher mirrors the vocabulary, validates the
    step through the SHARED validator with the standup's attested class, and APPLIES it exactly as the native standup does
    (max_num_seqs -> the class-specific to_S, seqs_source "override"), then the kernel step from the class-adjusted S. A forged to_S and a
    class other than the attested one are refused by name; the launcher's own fallback (no shared validator) refuses the kind by name,
    never passes it through the KV path."""
    spec = importlib.util.spec_from_file_location("test_geometry_adjustment_launch_class", REPO / "tests" / "test_geometry_adjustment_launch_class.py")
    GA = importlib.util.module_from_spec(spec); spec.loader.exec_module(GA)
    GA._install(tmp_path, monkeypatch, wide=True)                       # the classed record: the class step reaches 16, the kernel ceiling lowers it to 14
    geom0, geom_cls, rec, k, block, S_id, S_served = GA._standup_block(served_lora=(1, 3))
    assert rec is not None and rec["kind"] == "launch_class" == L.ADJUSTMENT_LAUNCH_CLASS and k is not None
    assert L.ADJUSTMENT_KINDS == sg.ADJUSTMENT_KINDS == ("launch_class", "lora_kernel_index_ceiling", "kv_shrink")
    ep = {"geometry": dict(block), "engine": NATIVE, "lora_config": {"max_lora_rank": 1, "max_loras": 3}}
    assert L.endpoint_served_lora(ep) == (1, 3) and L.endpoint_served_lora({"geometry": {}}) is None
    served, shrink, problem = L.served_geometry_of_endpoint(ep, geom0)
    assert problem is None and shrink is None
    # the applied effect: the class step's to_S then the kernel step's — ending at the standup's served S; the source is the class step's "override"
    assert (served.max_model_len, served.max_num_seqs) == (GA.L, S_served) and S_served == k["to_S"] and rec["to_S"] == geom_cls.max_num_seqs
    assert rec["to_S"] != rec["from_S"] == geom0.max_num_seqs and served.seqs_source == "override" == geom_cls.seqs_source
    assert not served.is_identity
    srec = L.served_geometry_record(block, served, None, block["adjustments"])
    assert srec["max_num_seqs"] == S_served and srec["seqs_source"] == "override" and srec["adjustments"] == block["adjustments"]
    # a forged class to_S and a class other than the attested one: refused by name through the shared validator
    forged = {**ep, "geometry": {**block, "adjustments": [{**rec, "to_S": rec["to_S"] + 1}, {**k, "from_S": rec["to_S"] + 1}]}}
    _, _, p = L.served_geometry_of_endpoint(forged, geom0)
    assert p and "launch_class" in p
    other = {**ep, "lora_config": {"max_lora_rank": 64, "max_loras": 64}}
    _, _, p = L.served_geometry_of_endpoint(other, geom0)
    assert p and "launch_class" in p and "attests" in p
    # the launcher's own fallback (a tree whose geometry has no shared validator) refuses the kind by name — never a KV reading
    p = L.geometry_adjustment_problem(block, model_key=GA.KEY, L=GA.L, S_identity=S_id, S_served=S_served, sg=SimpleNamespace(), native=True)
    assert p and "'launch_class'" in p and "shared validator" in p and "refusing by name" in p


def test_served_geometry_of_endpoint_reads_the_one_adjustment_record(L):
    """``geometry.adjustments`` judged through the one function (W-A's, ``native=`` from the record's
    own engine): ``[kernel]`` -> the kernel-lowered S (not a KV shrink); ``[]`` on a native record
    where the formula lowers -> a named problem (present IFF it lowers); a kernel step on a lens
    record -> refused; ``[kv]`` and a record without the key -> today's readings, byte for byte;
    every malformed shape a NAMED problem, never 'a smaller S'."""
    cfg = _plan(L, [LORA_CELL], serving_path=NATIVE)[0]
    geom, grec = L.served_geometry(cfg, 1)
    assert (geom.max_model_len, geom.max_num_seqs) == (GEMMA_L, GEMMA_S)

    def ep(S, adjustments=_ABSENT, *, engine=NATIVE, L_=GEMMA_L, **extra):
        g = {"max_model_len": L_, "max_num_seqs": S, **extra}
        if adjustments is not _ABSENT:
            g["adjustments"] = adjustments
        d = {"geometry": g}
        if engine is not None:
            d["engine"] = engine
        return d

    served, shrink, problem = L.served_geometry_of_endpoint(ep(31, [KERNEL_STEP]), geom)
    assert problem is None and shrink is None
    assert (served.max_model_len, served.max_num_seqs) == (GEMMA_L, GEMMA_S_LOWERED)
    assert not served.is_identity and served.seqs_source == geom.seqs_source == "model_budget"
    rec = L.served_geometry_record(grec, served, None, [KERNEL_STEP])
    assert (rec["configured_max_num_seqs"], rec["max_num_seqs"], rec["identity"]) == (GEMMA_S, GEMMA_S_LOWERED, False)
    assert rec["adjustments"] == [KERNEL_STEP] and sg.KV_SHRINK_KEY not in rec and rec["seqs_source"] == "model_budget"
    assert rec["batched_tokens"] == served.batched_tokens
    assert L.endpoint_geometry_adjustments(ep(31, [KERNEL_STEP])) == [KERNEL_STEP] and L.endpoint_geometry_adjustments({}) == []
    # a native record that omitted the kernel step where it lowers: the "iff" half of the rule
    _, _, p = L.served_geometry_of_endpoint(ep(GEMMA_S, []), geom)
    assert p and "must record the 'lora_kernel_index_ceiling' step (36 -> 31" in p
    _, _, p = L.served_geometry_of_endpoint(ep(GEMMA_S), geom)                       # no key at all, native: the same rule
    assert p and "must record the 'lora_kernel_index_ceiling' step" in p
    # a bare smaller S with no record; to_S off the formula; steps reversed; an unknown kind; a repeat
    _, _, p = L.served_geometry_of_endpoint(ep(31, []), geom)
    assert p and ("names no adjustment" in p or "must record" in p)
    _, _, p = L.served_geometry_of_endpoint(ep(30, [{**KERNEL_STEP, "to_S": 30}]), geom)
    assert p and "the formula gives 31" in p
    _, _, p = L.served_geometry_of_endpoint(ep(31, [{"kind": "kv_shrink"}, KERNEL_STEP]), geom)
    assert p and "order" in p
    _, _, p = L.served_geometry_of_endpoint(ep(31, [{"kind": "magic", "from_S": 36, "to_S": 31}]), geom)
    assert p and "not one of" in p
    _, _, p = L.served_geometry_of_endpoint(ep(31, [KERNEL_STEP, KERNEL_STEP]), geom)
    assert p and "twice" in p
    _, _, p = L.served_geometry_of_endpoint(ep(31, "kernel"), geom)
    assert p and "not a list" in p
    _, _, p = L.served_geometry_of_endpoint(ep(31, [KERNEL_STEP], L_=8192), geom)
    assert p and "max_model_len=8192" in p
    # a torn engine spelling on the record is a problem before any geometry reading
    _, _, p = L.served_geometry_of_endpoint({**ep(31, [KERNEL_STEP]), C.SERVED_ENGINE_FIELD: LENS}, geom)
    assert p and "engine cannot be read" in p
    # a LENS record: no kernel step may appear; today's shapes read byte for byte
    _, _, p = L.served_geometry_of_endpoint(ep(31, [KERNEL_STEP], engine=None), geom)
    assert p and "LENS engine" in p
    assert L.served_geometry_of_endpoint(ep(GEMMA_S, engine=None), geom) == (geom, None, None)
    assert L.served_geometry_of_endpoint(ep(GEMMA_S, [], engine=None), geom) == (geom, None, None)
    _, _, p = L.served_geometry_of_endpoint(ep(30, engine=None), geom)
    assert p == f"the standup served max_num_seqs=30, not the resolved {GEMMA_S}, and records no {sg.KV_SHRINK_KEY}"
    # [kv]: the KV-derived shrink, through today's rule, with and without the adjustments spelling
    kv = sg.kv_shrink_record(configured_max_num_seqs=GEMMA_S, max_model_len=GEMMA_L, kv_backed_concurrency=20.7,
                             kv_capacity_tokens=20.7 * GEMMA_L, basis="test")
    kv_step = sg.kv_shrink_adjustment(kv)
    today = L.served_geometry_of_endpoint(ep(20, engine=None, kv_shrink=kv), geom)
    assert today[2] is None and today[0].max_num_seqs == 20 and today[1] == kv
    both = L.served_geometry_of_endpoint(ep(20, [kv_step], engine=None, kv_shrink=kv), geom)
    assert both[2] is None and both[0].max_num_seqs == 20 and both[1] == kv
    _, _, p = L.served_geometry_of_endpoint(ep(20, [], engine=None, kv_shrink=kv), geom)
    assert p and "contradicts itself" in p
    # [kernel, kv]: the KV step from the kernel-lowered S (a native record)
    kv2 = sg.kv_shrink_record(configured_max_num_seqs=GEMMA_S_LOWERED, max_model_len=GEMMA_L, kv_backed_concurrency=20.7,
                              kv_capacity_tokens=20.7 * GEMMA_L, basis="test")
    served, shrink, p = L.served_geometry_of_endpoint(ep(20, [KERNEL_STEP, sg.kv_shrink_adjustment(kv2)], kv_shrink=kv2), geom)
    assert p is None and served.max_num_seqs == 20 and shrink == kv2
    rec = L.served_geometry_record(grec, served, shrink, [KERNEL_STEP, sg.kv_shrink_adjustment(kv2)])
    assert rec[sg.KV_SHRINK_KEY] == kv2 and len(rec["adjustments"]) == 2 and rec["max_num_seqs"] == 20
    # a kv_shrink from the IDENTITY S when a kernel step precedes it: a problem
    _, _, p = L.served_geometry_of_endpoint(ep(20, [KERNEL_STEP, kv_step], kv_shrink=kv), geom)
    assert p and "kv_shrink" in p
    # llama8b: [] at the identity S is the identity (no step is due)
    cfg_ll = _plan(L, ["none"], serving_path=NATIVE, model="llama8b")[0]
    gl, _ = L.served_geometry(cfg_ll, 1)
    assert L.served_geometry_of_endpoint(ep(29, []), gl) == (gl, None, None)
    _, _, p = L.served_geometry_of_endpoint(ep(28, [{"kind": "lora_kernel_index_ceiling", "from_S": 29, "to_S": 28,
                                                     "D": 4096, "ceiling": 524287}]), gl)
    assert p and "does not lower" in p
    # the launcher's fallback rule (a staged tree without W-A's function) says the same things
    g = {"max_model_len": GEMMA_L, "max_num_seqs": 31, "adjustments": [KERNEL_STEP]}
    assert L.geometry_adjustment_problem(g, model_key="gemma4e4b", L=GEMMA_L, S_identity=GEMMA_S, S_served=31) is None
    assert "must record" in L.geometry_adjustment_problem({"adjustments": []}, model_key="gemma4e4b", L=GEMMA_L,
                                                          S_identity=GEMMA_S, S_served=GEMMA_S, native=True)
    assert "LENS engine" in L.geometry_adjustment_problem(g, model_key="gemma4e4b", L=GEMMA_L, S_identity=GEMMA_S,
                                                          S_served=31, native=False)
    assert "not the formula's 31" in L.geometry_adjustment_problem(
        {"adjustments": [{**KERNEL_STEP, "to_S": 30}]}, model_key="gemma4e4b", L=GEMMA_L, S_identity=GEMMA_S, S_served=30)


# --------------------------------------------------------------------------- #
# 7. smoke planning, the dry run under both paths, main's plan-time refusals    #
# --------------------------------------------------------------------------- #


def test_the_native_smoke_honours_the_requested_cells_and_the_lens_smoke_is_byte_identical(L, capsys, monkeypatch):
    """Under native the smoke REQUIRES ``--cells`` (the historical smoke cell is a PSR object) and
    each named cell becomes its own thinking-OFF configuration under a unique ``smoke-<sha8>-<utc>``
    out-subdir; under lens today's PSR smoke plans byte for byte and ``cells`` is not consulted
    (judgement call, W-G: the lens smoke keeps its historical shape)."""
    (s,) = L.smoke_plan()
    assert s["cell"] == C.CELL_PSRKL_SAFETY and L.SERVING_PATH_KEY not in s
    assert L.smoke_plan(serving_path=LENS) == [s] == L.smoke_plan(cells=[LORA_CELL], serving_path=LENS)
    with pytest.raises(ValueError, match="needs --cells"):
        L.smoke_plan(serving_path=NATIVE)
    # native-psr-serving: the historical psr smoke cell PLANS natively too (its mode is judged at standup)
    (psr_native,) = L.smoke_plan(cells=[C.CELL_PSRKL_SAFETY], serving_path=NATIVE)
    assert psr_native["cell"] == C.CELL_PSRKL_SAFETY and psr_native[L.SERVING_PATH_KEY] == NATIVE
    cfgs = L.smoke_plan(cells=[LORA_CELL, "none"], serving_path=NATIVE)
    assert [c["cell"] for c in cfgs] == [LORA_CELL, "none"]
    assert all(c["mode"] == "off" and c[L.SERVING_PATH_KEY] == NATIVE for c in cfgs)
    assert [c["name"] for c in cfgs] == ["tinycap-e4b-lora_r4_safety-think_off", "tinycap-e4b-base-think_off"]
    assert L.plan_for(smoke=True, cells=[LORA_CELL, "none"], serving_path=NATIVE) == cfgs
    assert L.plan_for(smoke=True, cells=[LORA_CELL, "none"]) == [s]                    # lens (library default): the PSR smoke
    assert "PSR object with no PEFT form" in L.smoke_cells_required_message()
    # main: a native smoke without cells refuses BEFORE any plan or Modal call
    monkeypatch.setattr(L, "plan_for", lambda **k: pytest.fail("a plan was made"))
    with pytest.raises(SystemExit, match="needs --cells"):
        L.main(app_name=L.APP_NAME, sha="abc1234", smoke=True, dry_run=True)
    monkeypatch.undo()
    # the unique out-subdir (§5's per-job scratch claim)
    assert L.smoke_out_subdir("abcdef0123456789", now=0) == "smoke-abcdef01-19700101T000000Z"
    assert L.smoke_out_subdir("abc1234", now=86400.0) == "smoke-abc1234-19700102T000000Z"
    assert L.SMOKE_OUT_SUBDIR == "smoke"
    # the native smoke dry run: two configurations, the unique subdir, the native lines
    L.main(app_name=L.APP_NAME, sha="abc1234", smoke=True, cells=f"{LORA_CELL},none", dry_run=True)
    out = capsys.readouterr().out
    assert "2 GPU configuration container(s)" in out and "limit 5, inspect_limit 5" in out
    m = re.search(r"out roots under …/(smoke-abc1234-\d{8}T\d{6}Z)/", out)
    assert m, out
    assert out.count("serving path: 'lora_native'") == 2
    assert out.count(f"kernel ceiling step (§1.8, pre-computed): S {GEMMA_S} -> {GEMMA_S_LOWERED} at L {GEMMA_L}") == 2
    standups = [ln for ln in out.splitlines() if ln.startswith("   standup: ")]
    assert len(standups) == 2
    assert all(f" --serving-path lora_native --adapter-bank {L.ADAPTER_BANK_ROOT}/gemma4e4b" in ln for ln in standups)
    assert all(ln.count("--serving-path") == 1 for ln in standups)
    assert f"--out-root {C.VOL_GEN_ROOT}/native/gemma4e4b/{LORA_CELL}/thinking_off/{m.group(1)} " in out
    assert f"--out-root {C.VOL_GEN_ROOT}/native/gemma4e4b/none/thinking_off/{m.group(1)} " in out
    # the standup is asked the budget's S; the driver drives — and the identity records — the served S
    assert out.count(f"--max-num-seqs {GEMMA_S} ") == 2 and out.count(f"--concurrency {GEMMA_S_LOWERED} ") == 2
    assert f"--concurrency {GEMMA_S} " not in out
    assert out.count('"serving_engine": "lora_native"') == 2 and out.count('"reduction_mode": "not_applicable"') == 2
    assert "reduction_mode: 'per_row'" in out and "SERVING_ENGINE_MISMATCH" in out and "SERVING_ENGINE_CONFLICT" in out
    argv_line = next(ln for ln in out.splitlines() if ln.startswith("[tinycap] argv:"))
    assert L.SERVING_PATH_FLAG not in argv_line and f"--smoke --cells {LORA_CELL},none" in argv_line
    assert f"[tinycap] serving path: lora_native ({L.SERVING_PATH_FLAG}; engine lora_native; native out roots under {C.VOL_GEN_ROOT}/native/)" in out
    # the lens smoke dry run: today's `smoke` subdir, the flag spelled, no kernel step
    L.main(app_name=L.APP_NAME, sha="abc1234", smoke=True, serving_path=LENS, dry_run=True)
    out = capsys.readouterr().out
    assert "1 GPU configuration container(s)" in out and "out roots under …/smoke/" in out
    assert "serving path: 'lens'" in out and "kernel ceiling" not in out and "--serving-path lens" in out
    assert f"--out-root {C.VOL_GEN_ROOT}/gemma4e4b/{C.CELL_PSRKL_SAFETY}/thinking_off/smoke " in out
    assert "[tinycap] serving path: lens (--serving-path; engine lens; the lens standup)" in out


def test_the_native_dry_run_and_mains_plan_time_refusals(L, capsys, monkeypatch):
    """A pass-1 dry run under the default (native): the native lines per configuration, the out roots
    under the ``native/`` namespace; a PSR cell — the historical default battery included — refuses
    by name naming the escape hatch; a lens reduction mode under native refuses; an unknown path
    refuses; a plan / dry-run disagreement on the path refuses."""
    L.main(app_name=L.APP_NAME, sha="abc1234", pass1=True, cells=f"{LORA_CELL},none", modes="off", controls="gsm8k",
           dry_run=True)
    out = capsys.readouterr().out
    assert "2 GPU configuration container(s)" in out and out.count("serving path: 'lora_native'") == 2
    assert f"--out-root {C.VOL_GEN_ROOT}/native/gemma4e4b/none/thinking_off " in out
    assert f"--out-root {C.VOL_GEN_ROOT}/native/gemma4e4b/{LORA_CELL}/thinking_off " in out
    assert f"adapter bank {L.ADAPTER_BANK_ROOT}/gemma4e4b" in out
    assert "a store of another engine under the out root is SERVING_ENGINE_CONFLICT, terminal" in out
    # the same launch under lens: the historical roots, the flag spelled
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path=LENS, pass1=True, cells=f"{LORA_CELL},none", modes="off",
           controls="gsm8k", dry_run=True)
    out = capsys.readouterr().out
    assert f"--out-root {C.VOL_GEN_ROOT}/gemma4e4b/none/thinking_off " in out and "/native/" not in out
    assert out.count("serving path: 'lens'") == 2 and "--serving-path lens" in out
    # native-psr-serving: a PSR cell under the default PLANS natively (the object's position mode is judged
    # at standup, not here) — the default battery (TEST_CELLS names one) too
    L.main(app_name=L.APP_NAME, sha="abc1234", dry_run=True, pass1=True, cells=C.CELL_PSRKL_SAFETY, modes="off",
           controls="gsm8k")
    out = capsys.readouterr().out
    assert "1 GPU configuration container(s)" in out and out.count("serving path: 'lora_native'") == 1
    assert f"--out-root {C.VOL_GEN_ROOT}/native/gemma4e4b/{C.CELL_PSRKL_SAFETY}/thinking_off " in out
    L.main(app_name=L.APP_NAME, sha="abc1234", dry_run=True, pass1=True)
    assert "GPU configuration container(s)" in capsys.readouterr().out
    # a lens fact beside the native path
    with pytest.raises(SystemExit, match="is a lens fact the native engine cannot attest"):
        L.main(app_name=L.APP_NAME, sha="abc1234", pass1=True, cells="none", modes="off", controls="gsm8k",
               reduction_mode="stacked", dry_run=True)
    with pytest.raises(SystemExit, match="--serving-path must be one of"):
        L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="native", pass1=True, dry_run=True)
    # describe_plan holds the plan's path to the launch's
    cfgs = _plan(L, ["none"], serving_path=NATIVE)
    with pytest.raises(ValueError, match="two readings of one fact"):
        L.describe_plan(cfgs, limit=5, inspect_limit=5, samples=1, pass_no=1, serving_path=LENS)
    with pytest.raises(ValueError, match="two readings of one fact"):
        L.describe_plan(_plan(L, ["none"]), limit=5, inspect_limit=5, samples=1, pass_no=1, serving_path=NATIVE)


def test_the_preflight_plans_the_same_native_configurations_and_the_fitted_smoke_needs_pf(L, monkeypatch, capsys):
    """ONE ``plan_args`` mapping (``serving_path`` a planning input): the preflight's plan equals the
    launched configurations by name, key and out root — the native smoke over ``--cells <lora>,none``
    is TWO configurations (the fitted cell and the zero-adapter-only ``none``); a fitted cell launches
    only behind ``--pf`` (E40 H1); ``none`` alone needs none."""
    got: dict = {}

    def _remote(sha, plan_args, expect):
        got["plan_args"] = plan_args
        return TL._preflight_report(L, L.plan_for(**plan_args))
    launched: list = []
    monkeypatch.setattr(L, "preflight", SimpleNamespace(remote=_remote))
    monkeypatch.setattr(L, "run_configuration", TL._spawn_only(TL._done, lambda c, k: launched.append(c)))
    L.main(app_name=L.APP_NAME, sha="abc1234", pf=True, smoke=True, cells=f"{LORA_CELL},none")
    capsys.readouterr()
    pa = got["plan_args"]
    assert pa["serving_path"] == NATIVE and pa["smoke"] is True and pa["cells"] == [LORA_CELL, "none"]
    pf_plan = L.plan_for(**pa)
    assert [c["name"] for c in pf_plan] == [c["name"] for c in launched] == [
        "tinycap-e4b-lora_r4_safety-think_off", "tinycap-e4b-base-think_off"]
    assert [c["geometry_keys"] for c in pf_plan] == [c["geometry_keys"] for c in launched]
    assert all(c[L.SERVING_PATH_KEY] == NATIVE for c in launched)
    assert all(c["out_dir"].startswith(f"{C.VOL_GEN_ROOT}/native/gemma4e4b/") and "/smoke-abc1234-" in c["out_dir"]
               for c in launched)
    assert [c["out_dir"] for c in pf_plan] == [c["out_dir"].rsplit("/", 1)[0] for c in launched]   # the same roots, isolated
    # without --pf the fitted cell refuses before any Modal call; `none` alone launches
    launched.clear()
    with pytest.raises(SystemExit, match="launch only behind --pf"):
        L.main(app_name=L.APP_NAME, sha="abc1234", smoke=True, cells=f"{LORA_CELL},none")
    assert not launched
    L.main(app_name=L.APP_NAME, sha="abc1234", smoke=True, cells="none")
    capsys.readouterr()
    assert [c["cell"] for c in launched] == ["none"] and launched[0][L.SERVING_PATH_KEY] == NATIVE
    # a pass-1 native launch: the same mapping, native
    launched.clear()
    L.main(app_name=L.APP_NAME, sha="abc1234", pf=True, pass1=True, cells=f"{LORA_CELL},none", modes="off", controls="gsm8k")
    capsys.readouterr()
    assert got["plan_args"]["serving_path"] == NATIVE and got["plan_args"]["smoke"] is False
    assert [c["name"] for c in L.plan_for(**got["plan_args"])] == [c["name"] for c in launched]
    # the explicit lens launch records lens and marks nothing
    launched.clear()
    L.main(app_name=L.APP_NAME, sha="abc1234", pf=True, serving_path=LENS, smoke=True, controls="gsm8k")
    capsys.readouterr()
    assert got["plan_args"]["serving_path"] == LENS and [c["cell"] for c in launched] == [C.CELL_PSRKL_SAFETY]
    assert L.SERVING_PATH_KEY not in launched[0]


def test_the_preflight_requires_the_standup_flag_and_keys_the_nemotron_certificate_by_engine(L):
    """Source pins on the Modal preflight body: the staged tree's standup must take ``--serving-path``
    (a pre-flip sha is refused by name, never launched) and ``--adapter-bank`` under native; the
    Nemotron certificate is looked up at the ENGINE-bound key."""
    src = LAUNCHER.read_text()
    body = src[src.index("def preflight("):src.index("def _upload_tree(")]
    i = body.index("def standup_flags():")
    flags = body[i:body.index("chk(", i)]
    assert "SERVING_PATH_FLAG]" in flags and "need.append(ADAPTER_BANK_FLAG)" in flags
    assert "a pre-flip tree" in flags and 'is_native_path(str(plan_args.get("serving_path")' in flags
    j = body.index("def backend_attested():")
    att = body[j:body.index("chk(", j)]
    assert 'serving_engine = engine_of_serving_path(str(plan_args.get("serving_path") or SERVING_PATH_LIBRARY_DEFAULT))' in att
    assert "serving_engine=serving_engine)" in att and '"serving_engine": serving_engine' in att
    # main hands the path into the plan, the argv, the probe and the dry run; the worker reads the plan marker
    m = src[src.index("def main("):]
    assert '"serving_path": serving_path}' in m and "serving_path=serving_path)" in m
    assert "row.slug, serving_path)" in m                                    # the Nemotron probe's argument
    w = src[src.index("def _run_configuration("):src.index("def _producer_conflict_error(")]
    assert "serving_path = serving_path_of(cfg)" in w and "conflict = engine_conflict_at_output(out_dir, engine)" in w
    assert w.index("conflict = engine_conflict_at_output") < w.index("lane_roster") < w.index("expected = expected_identity(")
    assert w.index("served_engine = endpoint_engine(endpoint)") < w.index("attested = endpoint_reduction_mode(endpoint")
    assert "env[ADAPTER_BANK_ROOT_ENV] = ADAPTER_BANK_ROOT" in w
    assert "nemotron_worker_admission(row, tree=tree, tree_sha=tree_sha, geom=geom, reduction_mode=reduction_mode,\n" \
           "                                          serving_engine=engine)" in w


# --------------------------------------------------------------------------- #
# 7b. the worker: a native run through _run_configuration against a fake standup  #
# --------------------------------------------------------------------------- #


def _native_world(L, tmp_path, monkeypatch, *, record, cell=LORA_CELL, serving_path=NATIVE):
    """A LoRA configuration planned under ``serving_path`` with a staged tree and a fake standup
    attesting ``record`` (merged over the minimum the launcher reads — ``geometry`` replaced whole
    when given). The standup dir is the REAL ``/tmp/srv_<name>``, cleared here and by
    :func:`TL._run_fake`."""
    cfg = _plan(L, [cell], serving_path=serving_path)[0]
    shutil.rmtree(f"/tmp/srv_{cfg['name']}", ignore_errors=True)
    repos = tmp_path / "repos"
    tree = TL._tree(repos, "f6cf462")
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    monkeypatch.delenv(C.LENS_STACKED_REDUCTIONS_ENV, raising=False)
    out = tmp_path / "gen"
    cfg = dict(cfg, out_dir=str(out))
    TL._FakeStandupProc.record = dict(record)
    monkeypatch.setattr(L.subprocess, "Popen", TL._FakeStandupProc)
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    return cfg, out, tree


#: what the native core writes for gemma4e4b at the pass-1 footprint: the engine (both spellings),
#: the bank, the kernel-lowered S and the ONE adjustments record
NATIVE_EP = {"engine": NATIVE, C.SERVED_ENGINE_FIELD: NATIVE, "adapter_bank": "/vol/tinycap/multilora_adapters/gemma4e4b",
             "geometry": {"max_model_len": GEMMA_L, "max_num_seqs": GEMMA_S_LOWERED, "adjustments": [KERNEL_STEP]}}


def test_a_native_run_holds_the_standup_to_the_engine_and_the_adjustments_record(L, tmp_path, monkeypatch, capsys):
    """The worker reads the plan marker (never a kwarg), asks the standup for the budget's S with
    ``--serving-path lora_native --adapter-bank …`` and the bank root in its env, and the moment
    endpoint.json appears: the served geometry through the §1.8 record (36 -> 31, not a KV shrink),
    the ENGINE first (the record's, then the live proxy's), the mode recorded ``not_applicable`` with
    no lens key consulted; the driver is driven at the served S. The run ends at the fake ledger."""
    cfg, out, tree = _native_world(L, tmp_path, monkeypatch, record=NATIVE_EP)
    monkeypatch.setattr(L, "proxy_engine_kind", lambda ep, **k: (NATIVE, None))
    rec = TL._run_fake(L, cfg)
    assert rec["status"] == "LEDGER_FAILED", rec.get("error")           # past the standup, the engine, the geometry and the mode
    assert rec["serving_path"] == NATIVE and rec[C.SERVED_ENGINE_FIELD] == NATIVE == rec["served_engine"]
    assert rec["proxy_engine_kind"] == NATIVE and rec["reduction_mode_requested"] == "per_row"
    assert rec[C.REDUCTION_MODE_FIELD] == "not_applicable"
    assert rec["reduction_mode_attestation"] == (f"native engine ({NATIVE}): no lens reduction mode — 'not_applicable', "
                                                 f"no {C.PROXY_IDENTITY_REDUCTION_MODE} consulted")
    assert rec["expected_geometry_adjustments"] == [KERNEL_STEP] == rec["geometry_adjustments"] == rec["geometry"]["adjustments"]
    assert (rec["geometry"]["configured_max_num_seqs"], rec["geometry"]["max_num_seqs"]) == (GEMMA_S, GEMMA_S_LOWERED)
    assert rec["geometry"]["identity"] is False and sg.KV_SHRINK_KEY not in rec["geometry"]
    assert rec["geometry"]["seqs_source"] == "model_budget"
    served = rec["expected_identity"]["served"]
    assert served["serving_engine"] == NATIVE and served["reduction_mode"] == "not_applicable"
    assert served["max_num_seqs"] == served["concurrency"] == GEMMA_S_LOWERED
    assert rec["expected_identity"] == rec["expected_identity_configured"]        # the prediction held: a no-op re-statement
    argv = rec["standup_argv"]
    i = argv.index(L.SERVING_PATH_FLAG)
    assert argv[i + 1:i + 4] == [NATIVE, L.ADAPTER_BANK_FLAG, "/vol/tinycap/multilora_adapters/gemma4e4b"]
    assert argv[argv.index("--max-num-seqs") + 1] == str(GEMMA_S)                # the standup is ASKED the budget's S
    # the driver (composed after the ledger the fake kills) is driven at the SERVED geometry: the
    # worker rebinds `geom` to the served one before composing it — pinned on the source
    w = LAUNCHER.read_text()
    w = w[w.index("def _run_configuration("):w.index("def _producer_conflict_error(")]
    assert w.index("geom = geom_served") < w.index('rec["driver_argv"] = cmd')
    assert TL._FakeStandupProc.last_env[L.ADAPTER_BANK_ROOT_ENV] == L.ADAPTER_BANK_ROOT
    assert TL._FakeStandupProc.last_env[C.LENS_STACKED_REDUCTIONS_ENV] == "per_row"
    sgj = json.loads((out / L.SERVED_GEOMETRY_FILENAME).read_text())
    assert sgj["serving_path"] == NATIVE and sgj[C.SERVED_ENGINE_FIELD] == NATIVE
    assert sgj["endpoint"]["engine"] == NATIVE and sgj["geometry"]["adjustments"] == [KERNEL_STEP]
    out_text = capsys.readouterr().out
    assert f"[geometry] adjusted S: the standup served {GEMMA_L} x {GEMMA_S_LOWERED} under" in out_text
    assert "KV-derived S" not in out_text
    # (a) the live proxy attests another engine than the record: torn down before generation
    monkeypatch.setattr(L, "proxy_engine_kind", lambda ep, **k: (LENS, None))
    rec = TL._run_fake(L, cfg)
    assert rec["status"] == "SERVING_ENGINE_MISMATCH" and rec["proxy_engine_kind"] == LENS and "standup_tail" in rec
    assert "the record does not vouch for the proxy answering" in rec["error"]
    monkeypatch.setattr(L, "proxy_engine_kind", lambda ep, **k: (None, "GET http://127.0.0.1:1/v1/models failed: OSError"))
    rec = TL._run_fake(L, cfg)
    assert rec["status"] == "SERVING_ENGINE_MISMATCH" and "GET http://127.0.0.1:1/v1/models failed" in rec["error"]
    # (b) a LENS standup (today's record: no engine key) under the native run
    monkeypatch.setattr(L, "proxy_engine_kind", lambda ep, **k: pytest.fail("the proxy is not consulted for a lens record"))
    cfg_b, _, _ = _native_world(L, tmp_path / "b", monkeypatch, record={})
    rec = TL._run_fake(L, cfg_b)
    assert rec["status"] == "SERVING_ENGINE_MISMATCH"
    assert f"attests engine='lens' but this run's --serving-path {NATIVE} expects '{NATIVE}'" in rec["error"]
    assert rec["served_engine"] == LENS and "standup_tail" in rec
    # (c) a NATIVE standup under a LENS launch (the record's kernel step holds; the engine does not)
    cfg_c, _, _ = _native_world(L, tmp_path / "c", monkeypatch, record=NATIVE_EP, serving_path=LENS)
    assert L.SERVING_PATH_KEY not in cfg_c
    rec = TL._run_fake(L, cfg_c)
    assert rec["status"] == "SERVING_ENGINE_MISMATCH" and rec["serving_path"] == LENS and rec["served_engine"] == NATIVE
    assert f"attests engine='{NATIVE}' but this run's --serving-path lens expects 'lens'" in rec["error"]
    assert rec["standup_argv"][rec["standup_argv"].index(L.SERVING_PATH_FLAG) + 1] == LENS
    assert L.ADAPTER_BANK_FLAG not in rec["standup_argv"]
    # (d) a torn native record that STATES a lens mode: refused, neither fact read
    monkeypatch.setattr(L, "proxy_engine_kind", lambda ep, **k: (NATIVE, None))
    cfg_d, _, _ = _native_world(L, tmp_path / "d", monkeypatch, record={**NATIVE_EP, C.PROXY_IDENTITY_REDUCTION_MODE: "stacked"})
    rec = TL._run_fake(L, cfg_d)
    assert rec["status"] == "REDUCTION_MODE_UNATTESTED" and "torn record" in rec["error"] and C.REDUCTION_MODE_FIELD not in rec
    # (e) a native standup that served the budget's S and named no kernel step: the §1.8 rule, by name
    cfg_e, _, _ = _native_world(L, tmp_path / "e", monkeypatch, record={
        **NATIVE_EP, "geometry": {"max_model_len": GEMMA_L, "max_num_seqs": GEMMA_S, "adjustments": []}})
    rec = TL._run_fake(L, cfg_e)
    assert rec["status"] == C.REC_SERVED_GEOMETRY_MISMATCH == "SERVED_GEOMETRY_MISMATCH"
    assert "must record the 'lora_kernel_index_ceiling' step (36 -> 31" in rec["error"]
    # ... and one that lowered to another S than the formula's
    cfg_f, _, _ = _native_world(L, tmp_path / "f", monkeypatch, record={
        **NATIVE_EP, "geometry": {"max_model_len": GEMMA_L, "max_num_seqs": 30, "adjustments": [{**KERNEL_STEP, "to_S": 30}]}})
    rec = TL._run_fake(L, cfg_f)
    assert rec["status"] == "SERVED_GEOMETRY_MISMATCH" and "the formula gives 31" in rec["error"]
    # (g) a lens reduction mode handed to a native worker: refused before any standup
    cfg_g, _, _ = _native_world(L, tmp_path / "g", monkeypatch, record=NATIVE_EP)
    monkeypatch.setattr(L.subprocess, "Popen", lambda *a, **k: pytest.fail("a standup was started"))
    rec = L._run_configuration(cfg_g, "f6cf462", reduction_mode="stacked", object_digests=TL._digests_for(cfg_g))
    assert rec["status"] == "REDUCTION_MODE_MISMATCH" and "is a lens fact" in rec["error"] and "standup_argv" not in rec
    assert rec["serving_path"] == NATIVE and rec[C.SERVED_ENGINE_FIELD] == NATIVE
    # (h) a configuration whose plan marker is malformed refuses by name, before anything
    rec = L._run_configuration(dict(cfg_g, **{L.SERVING_PATH_KEY: "vllm"}), "f6cf462", object_digests=TL._digests_for(cfg_g))
    assert rec["status"] == "SERVING_ENGINE_MISMATCH" and "--serving-path must be one of" in rec["error"]


def test_the_terminal_engine_conflict_refuses_before_any_write_commit_or_standup(L, tmp_path, monkeypatch):
    """A completed LENS root under a NATIVE run — a synthetic lens store AND the real
    lens fixture store beside it — is ``SERVING_ENGINE_CONFLICT``, terminal: no standup, no driver, no
    ``VOL.commit``, no sidecar, the root byte-identical, and NO re-plan (the store is never filed as
    remaining work to be regenerated on the other engine). The reverse refuses the same way. A
    complete NATIVE store under the native run is DONE with no GPU minute."""
    cfg, out, tree = _native_world(L, tmp_path, monkeypatch, record={})
    exp_l = _expected(L, _plan(L, [LORA_CELL], serving_path=LENS)[0], serving_path=LENS, tree=tree)
    l_store = TL._store(out, exp_l)
    for c in cfg["controls"]:
        for t in L.expected_tasks(c, "off"):
            TL._finish(l_store, t, L.FULL_ITEM_COUNT_OF_TASK[t])
    fx = json.loads(FIXTURE_LENS_STORE.read_text())
    fxd = out / "cells" / fx["cell_dir"]
    fxd.mkdir(parents=True)
    (fxd / "manifest.json").write_text(json.dumps(fx["manifest"]))
    before = _tree_hashes(out)
    commits: list = []
    monkeypatch.setattr(L, "VOL", SimpleNamespace(commit=lambda: commits.append(1), reload=lambda: None))
    monkeypatch.setattr(L.subprocess, "Popen", lambda *a, **k: pytest.fail("a standup / driver / ledger was started"))
    rec = L._run_configuration(cfg, "f6cf462", object_digests=TL._digests_for(cfg))
    assert rec["status"] == "SERVING_ENGINE_CONFLICT" == C.REC_SERVING_ENGINE_CONFLICT
    assert rec["engine_conflict"]["expected_engine"] == NATIVE
    assert sorted(s["store"] for s in rec["engine_conflict"]["stores"]) == sorted([l_store.name, fx["cell_dir"]])
    assert all(s["engine"] == LENS for s in rec["engine_conflict"]["stores"])
    assert "never re-generated on the other engine" in rec["error"] and "nothing written, no standup" in rec["error"]
    assert "resume" not in rec and "resume_prestandup" not in rec and "expected_identity" not in rec   # no re-plan
    assert "standup_argv" not in rec and "controls_remaining" not in rec
    assert _tree_hashes(out) == before and commits == []
    assert not (out / "lane_roster.json").exists() and not (out / L.SERVED_GEOMETRY_FILENAME).exists()
    assert rec["serving_path"] == NATIVE and rec[C.SERVED_ENGINE_FIELD] == NATIVE and rec["tree_sha"] == TL.FULL_SHA
    # the reverse: a native store under a lens run
    exp_n = _expected(L, cfg, serving_path=NATIVE, tree=tree)
    out2 = tmp_path / "gen2"
    n_store = _native_store(out2, exp_n)
    before2 = _tree_hashes(out2)
    cfg_l = dict(_plan(L, [LORA_CELL], serving_path=LENS)[0], out_dir=str(out2))
    rec = L._run_configuration(cfg_l, "f6cf462", object_digests=TL._digests_for(cfg_l))
    assert rec["status"] == "SERVING_ENGINE_CONFLICT" and rec["engine_conflict"]["expected_engine"] == LENS
    assert [s["store"] for s in rec["engine_conflict"]["stores"]] == [n_store.name]
    assert _tree_hashes(out2) == before2 and commits == []
    # a same-engine complete native store: DONE before any standup, the engine on the identity
    out3 = tmp_path / "gen3"
    done = _native_store(out3, exp_n)
    for c in cfg["controls"]:
        for t in L.expected_tasks(c, "off"):
            TL._finish(done, t, L.FULL_ITEM_COUNT_OF_TASK[t])
    rec = L._run_configuration(dict(cfg, out_dir=str(out3)), "f6cf462", object_digests=TL._digests_for(cfg))
    assert rec["status"] == "DONE" and rec["controls_remaining"] == [] and "standup_s" not in rec
    assert rec["resume"]["controls"]["gsm8k"]["done_in"] == done.name
    assert rec["expected_identity"]["served"]["serving_engine"] == NATIVE
    assert rec["expected_identity"]["served"]["max_num_seqs"] == GEMMA_S_LOWERED       # the pre-computed S found the store
    assert rec["expected_geometry_adjustments"] == [KERNEL_STEP]
    # the same store is NOT a lens run's: refused by name at the resume (and the conflict fires first)
    rec = L._run_configuration(dict(cfg_l, out_dir=str(out3)), "f6cf462", object_digests=TL._digests_for(cfg_l))
    assert rec["status"] == "SERVING_ENGINE_CONFLICT"


#: the lane roster's runnable LoRA entry on the llama8b waves (``diff_lora``: a lora_subtraction cell)
J4_LANE_CELL = "constv2-mix-full__lora_r4_ce_minus_ctl"


def test_the_j4_smoke_shape_on_llama8b_plans_two_native_configurations(L, capsys):
    """The smoke as TWO native configurations — the fitted LoRA cell and the
    zero-adapter-only ``none`` — with llama8b's bank, NO kernel step (D 4096 at 16384 x 29) and the
    ``native/`` namespace. Two launchable shapes, both pinned: the LANE row ``llama8b-f22it`` with the
    lane roster's runnable LoRA entry (``constv2-mix-full__lora_r4_ce_minus_ctl``; a lane row needs
    ``--roster-snapshot``, one block per planned cell), and the plain ``llama8b`` row with
    ``constv2-mix-full__lora_r4_ce``. The plan's literal command (``--model llama8b-f22it --cells
    constv2-mix-full__lora_r4_ce,none``) is refused by name on BOTH counts: no snapshot, and the lane
    roster lists no such entry for ``llama8b_it``."""
    import test_tinycap_widen_launcher as TW
    from sisteer.capeval.tinycap import widen_roster as W
    entries = W.entries(TW.ROSTER, "llama8b_it")
    assert {e["cell_id"] for e in entries.values() if e.get("status") == "runnable"} == {"none", J4_LANE_CELL}
    assert J4_CELL not in {e["cell_id"] for e in entries.values()}
    assert L.native_cell_refusal(J4_LANE_CELL) is None and L.native_kind_of_cell(J4_LANE_CELL) == "lora_subtraction"
    # the plan's literal command: a lane row without its snapshot refuses by name ...
    with pytest.raises(SystemExit, match="--roster-snapshot"):
        L.main(app_name=L.APP_NAME, sha="abc1234", smoke=True, model=J4_MODEL, cells=f"{J4_CELL},none", dry_run=True)
    # ... and with the roster's blocks, the plan's cell has none (one block per planned cell)
    with pytest.raises(SystemExit, match="one block per planned cell"):
        L.main(app_name=L.APP_NAME, sha="abc1234", smoke=True, model=J4_MODEL, cells=f"{J4_CELL},none",
               roster_snapshot=json.dumps({"none": TW._snap("llama8b_it", "none")}), dry_run=True)
    capsys.readouterr()

    def check(model, cell, **kw):
        L.main(app_name=L.APP_NAME, sha="abc1234", smoke=True, model=model, cells=f"{cell},none", dry_run=True, **kw)
        out = capsys.readouterr().out
        assert "2 GPU configuration container(s)" in out and out.count("== tinycap-") == 2
        assert f" cell={cell} " in out and " cell=none " in out
        assert out.count("serving path: 'lora_native'") == 2
        assert out.count("kernel ceiling step (§1.8, pre-computed): none (S 29 x L 16384 fits") == 2
        standups = [ln for ln in out.splitlines() if ln.startswith("   standup: ")]
        assert len(standups) == 2
        assert all(f" --serving-path lora_native --adapter-bank {L.ADAPTER_BANK_ROOT}/llama8b" in ln for ln in standups)
        assert all("--model-key llama8b" in ln and "--max-num-seqs 29" in ln for ln in standups)
        assert out.count("--concurrency 29 ") == 2
        m = re.search(r"out roots under …/(smoke-abc1234-\d{8}T\d{6}Z)/", out)
        assert m and f"--out-root {C.VOL_GEN_ROOT}/native/{model}/{cell}/thinking_off/{m.group(1)} " in out
        assert f"--out-root {C.VOL_GEN_ROOT}/native/{model}/none/thinking_off/{m.group(1)} " in out
        assert out.count('"serving_engine": "lora_native"') == 2 and out.count('"reduction_mode": "not_applicable"') == 2
        assert f"{L.OBJECT_DIGEST_FLAG} <preflight artifact_sha256>" in out      # the fitted cell is held to --pf's digest
        argv_line = next(ln for ln in out.splitlines() if ln.startswith("[tinycap] argv:"))
        assert f"--model {model} --smoke --cells {cell},none" in argv_line and L.SERVING_PATH_FLAG not in argv_line
        return out
    out = check(J4_MODEL, J4_LANE_CELL, roster_snapshot=json.dumps({
        "none": TW._snap("llama8b_it", "none"), J4_LANE_CELL: TW._snap("llama8b_it", "diff_lora")}))
    assert "--defenses-dir /vol/objects/sdfbase/full22/llama8b/it " in out
    out = check("llama8b", J4_CELL)
    assert "--defenses-dir /vol/objects/scale-12/llama8b " in out and "--config <tree>/configs/two_si.yaml" in out


# --------------------------------------------------------------------------- #
# 8. the Nemotron duty certificate binds the engine                            #
# --------------------------------------------------------------------------- #


def test_the_nemotron_certificate_key_binds_the_engine_and_the_adjusted_geometry(L, tmp_path, monkeypatch):
    """A lens key is today's key byte for byte (every certificate on the volume keeps its address);
    the native key carries ``serving_engine: lora_native``, ``reduction_mode: not_applicable`` and the
    §1.8 kernel-adjusted SERVED geometry (2528 x 256 -> 165 for nemotron at the 2048 entry); a lens
    certificate never admits a native row (refused NAMING the engine and the remedy); a native
    document is discoverable by the REAL ``nemotron_worker_admission`` through the real reader."""
    import test_tinycap_launcher_port as TP
    from sisteer.capeval.serving import mamba_backend as mb
    row = C.model_row(TP.NEMO)
    tree = TP._tree(tmp_path, row)
    geom = SimpleNamespace(max_model_len=2528, max_num_seqs=256, model_key="nemotron3nano4b")
    common = dict(tree=str(tree), tree_sha="a" * 40, geom=geom)
    lens_key = L.nemotron_attestation_key(row, reduction_mode="stacked", **common)
    assert lens_key == L.nemotron_attestation_key(row, reduction_mode="stacked", serving_engine=LENS, **common)
    lens_inputs = L.nemotron_attestation_key_inputs(row, reduction_mode="stacked", **common)
    assert "serving_engine" not in lens_inputs and (lens_inputs["max_model_len"], lens_inputs["max_num_seqs"]) == (2528, 256)
    assert lens_key == mb.attestation_cache_key(**lens_inputs) == mb.attestation_cache_key(
        image_specs=L.image_install_specs(), wheel_sha256=mb.wheel_sha256_of(L.VLLM_WHEEL),
        contract_sha256=mb.contract_sha256(tree / mb.CONTRACT_RELPATH), tree_sha="a" * 40, model_id=row.hf,
        model_revision=row.rev, serving_config_sha256=L.sha256_file(tree / row.cfg), max_model_len=2528,
        max_num_seqs=256, reduction_mode="stacked")
    # the native key: engine-bound, mode not_applicable, the kernel-adjusted S (D 5120: 419,430 // 2528 = 165)
    assert sg.kernel_ceiling_adjustment("nemotron3nano4b", 2528, 256) == {
        "kind": "lora_kernel_index_ceiling", "from_S": 256, "to_S": 165, "D": 5120, "ceiling": 419430}
    native_inputs = L.nemotron_attestation_key_inputs(row, reduction_mode="per_row", serving_engine=NATIVE, **common)
    assert native_inputs["serving_engine"] == NATIVE and native_inputs["reduction_mode"] == "not_applicable"
    assert (native_inputs["max_model_len"], native_inputs["max_num_seqs"]) == (2528, 165)
    assert {k: v for k, v in native_inputs.items() if k not in ("serving_engine", "reduction_mode", "max_num_seqs")} == \
           {k: v for k, v in lens_inputs.items() if k not in ("reduction_mode", "max_num_seqs")}
    native_key = L.nemotron_attestation_key(row, reduction_mode="per_row", serving_engine=NATIVE, **common)
    assert native_key != lens_key and len(native_key) == 64
    assert native_key == mb.attestation_cache_key(**native_inputs) == L.nemotron_attestation_cache_key(native_inputs)
    assert native_key == L.nemotron_attestation_key(row, reduction_mode="stacked", serving_engine=NATIVE, **common)  # no lens mode consulted
    # the key geometry
    assert L.nemotron_key_geometry(geom, LENS) is geom
    assert (L.nemotron_key_geometry(geom, NATIVE).max_model_len, L.nemotron_key_geometry(geom, NATIVE).max_num_seqs) == (2528, 165)
    for bad in ("vllm", "", None):
        with pytest.raises(ValueError, match="serving_engine must be one of"):
            L.nemotron_key_geometry(geom, bad)
        with pytest.raises(ValueError, match="serving_engine must be one of"):
            L.nemotron_attestation_key_inputs(row, reduction_mode="stacked", serving_engine=bad, **common)
    # a malformed native input set never canonicalises
    with pytest.raises(ValueError):
        L.nemotron_attestation_cache_key({**native_inputs, "reduction_mode": "stacked"})
    with pytest.raises(ValueError):
        L.nemotron_attestation_cache_key({**lens_inputs, "reduction_mode": "not_applicable"})
    # admission through the REAL gate: a LENS certificate on the volume does not admit a NATIVE row
    root = tmp_path / "nemo_backend"
    monkeypatch.setattr(L, "NEMO_BACKEND_ROOT", str(root))
    minimal = {"schema": "sisteer.mamba_backend_attestation/2", "admitted": True, "packages_ok": True,
               "packages": {"ok": True}, "static": {"ok": True, "discovery": {}}, "live": {"ok": True, "reports": []}}
    lens_path = L.nemotron_attestation_path(lens_key)
    lens_path.parent.mkdir(parents=True)
    lens_path.write_text(json.dumps({**minimal, "cache_key": lens_key}))
    adm = L.nemotron_worker_admission(row, reduction_mode="stacked", serving_engine=NATIVE, **common)
    assert adm["ok"] is False and adm["cache_key"] == native_key and adm["serving_engine"] == NATIVE
    assert adm["path"] == str(L.nemotron_attestation_path(native_key)) != str(lens_path)
    assert any(p.startswith(f"[{NATIVE}] no Nemotron backend attestation") for p in adm["problems"])
    assert any(f"run {L.FLAG_NEMOTRON_PROBE} under --serving-path {NATIVE} first" in p for p in adm["problems"])
    # ... while the lens run reads its own certificate at today's address
    adm_l = L.nemotron_worker_admission(row, reduction_mode="stacked", **common)
    assert adm_l["cache_key"] == lens_key and adm_l["path"] == str(lens_path) and adm_l["serving_engine"] == LENS
    assert not any("no Nemotron backend attestation" in p for p in adm_l["problems"])
    # a NATIVE-produced document (the probe's fold, the engine recorded) is discoverable by the real gate
    doc = mb.attestation_document(packages={"ok": True}, static={"ok": True, "discovery": {}},
                                  live={"ok": True, "reports": [], "expected_n_mamba_layers": 1},
                                  duty={"sm_duty_mean": 0.9, "tok_s": 100.0, "n_seqs": 64, "max_tokens": 512, "gpu_index": "0",
                                        "served": {"max_num_seqs": 165}},
                                  peer_duty={"sm_duty_mean": 0.9, "tok_s": 100.0, "n_seqs": 64, "max_tokens": 512,
                                             "gpu_index": "0", "served": {"max_num_seqs": 165}},
                                  peer_key="qwen35_4b", peer_image="image_nemotron", ratio=mb.duty_ratio(),
                                  cache_key=native_key, key_inputs=native_inputs, geometry="2048", standups=[],
                                  extra={C.SERVED_ENGINE_FIELD: NATIVE, "serving_path": NATIVE,
                                         "key_geometry": {"max_model_len": 2528, "max_num_seqs": 165, "requested_max_num_seqs": 256},
                                         "geometry_adjustments": [sg.kernel_ceiling_adjustment("nemotron3nano4b", 2528, 256)]})
    assert doc[C.SERVED_ENGINE_FIELD] == NATIVE and doc["key_inputs"]["serving_engine"] == NATIVE
    assert doc["key_geometry"]["max_num_seqs"] == 165 and doc["geometry_adjustments"][0]["to_S"] == 165
    native_path = L.nemotron_attestation_path(native_key)
    native_path.parent.mkdir(parents=True)
    native_path.write_text(json.dumps(doc, default=str))
    adm = L.nemotron_worker_admission(row, reduction_mode="stacked", serving_engine=NATIVE, **common)
    assert adm["path"] == str(native_path) and adm["cache_key"] == native_key
    assert not any("no Nemotron backend attestation" in p or "attestation is for cache key" in p for p in adm["problems"])
    # a non-Nemotron row certifies nothing on either engine
    assert L.nemotron_worker_admission(C.model_row(TP.QWEN), reduction_mode="stacked", serving_engine=NATIVE, **common) is None
    # the probe: both standups on the requested path, the engine folded into the document
    src = LAUNCHER.read_text()
    body = src[src.index("def _nemotron_probe("):src.index("def nemotron_backend_probe(")]
    assert body.count("serving_path=serving_path,") >= 1 and "adapter_bank=(adapter_bank_for(" in body
    # the additive block rides the NATIVE document alone — the lens certificate at the unchanged lens key folds its
    # inputs byte for byte (test_tinycap_lens_records_golden holds it)
    assert "native_extra = ({SERVED_ENGINE_FIELD: serving_engine, \"serving_path\": serving_path," in body
    assert '**({"extra": native_extra} if native else {}))' in body
    assert "key_geom = nemotron_key_geometry(geom, serving_engine)" in body
    # the pre-computed-S equality is the NATIVE key's rule only — under lens the key is the requested
    # geometry and the served S may sit below it through the KV shrink (test_nemotron_probe_cpu executes both)
    assert "if native and int(peer_geom.max_num_seqs) != int(key_geom.max_num_seqs):" in body
    assert 'env[ADAPTER_BANK_ROOT_ENV] = ADAPTER_BANK_ROOT' in body
    sig = inspect.signature(L.nemotron_backend_probe)
    assert sig.parameters["serving_path"].default == LENS                   # the library default: the lens probe byte for byte


# --------------------------------------------------------------------------- #
# 9. capctrl_served.py: the same flag and readings                             #
# --------------------------------------------------------------------------- #


def test_capctrl_served_takes_the_same_flag_and_readings(capdriver, tmp_path, monkeypatch):
    """The x2 capability wave's launcher: the same two defaults, ``--serving-path`` on every standup
    and the bank under native only; its registered CELLS are PSR objects, refused by name under the
    default BEFORE any Modal call; the engine read FIRST at its read site with the same exit shape as
    the reduction-mode refusal."""
    assert capdriver.SERVING_PATH_DEFAULT == NATIVE == ra.DEFAULT_STANDUP_SERVING_PATH
    assert capdriver.SERVING_PATH_FLAG == "--serving-path" and capdriver.SERVING_PATHS == (NATIVE, LENS)
    assert capdriver.ADAPTER_BANK_ROOT == "/vol/tinycap/multilora_adapters" and capdriver.ADAPTER_BANK_FLAG == "--adapter-bank"
    assert capdriver.ADAPTER_BANK_ROOT_ENV == "SISTEER_ADAPTER_BANK_ROOT"
    assert capdriver.REC_SERVING_ENGINE_MISMATCH == C.REC_SERVING_ENGINE_MISMATCH
    assert capdriver.engine_of_serving_path(NATIVE) == NATIVE and capdriver.engine_of_serving_path(LENS) == LENS
    geom = SimpleNamespace(eval="capability", model_key="llama8b", max_model_len=4096, max_num_seqs=4, window_source="table",
                           seqs_source="model_budget", overridden=False, over_budget=False, is_identity=True)
    args = ("/tree", "cfg.yaml", "none", "ddir", tmp_path, tmp_path / "STOP", geom, 64)
    lens = capdriver._standup_cmd(*args)
    assert lens[-2:] == ["--serving-path", LENS] and "--adapter-bank" not in lens
    assert capdriver._standup_cmd(*args, serving_path=LENS) == lens
    native = capdriver._standup_cmd(*args, serving_path=NATIVE, adapter_bank="/vol/tinycap/multilora_adapters/llama8b")
    assert native[-4:] == ["--serving-path", NATIVE, "--adapter-bank", "/vol/tinycap/multilora_adapters/llama8b"]
    assert native[:-4] == lens[:-2]
    with pytest.raises(ValueError, match="needs its adapter bank"):
        capdriver._standup_cmd(*args, serving_path=NATIVE)
    with pytest.raises(ValueError, match="offered to a lens standup"):
        capdriver._standup_cmd(*args, adapter_bank="/b")
    up = TL._load_standup()
    ns_ = up.build_parser().parse_args(native[2:])
    assert ns_.serving_path == NATIVE and str(ns_.adapter_bank) == "/vol/tinycap/multilora_adapters/llama8b"
    # the wave's cells are PSR objects: they PLAN natively since native-psr-serving (the object's position
    # mode is judged at standup); a Class B id outside the servable kinds still refuses, naming the hatch
    assert capdriver.CELLS == ("safety_refusal__psr_kl_all", "si_constitution__psr_kl_all")
    for ok in ("none", "prompted__safety_refusal", LORA_CELL, *capdriver.CELLS):
        assert capdriver.native_cell_refusal(ok) is None, ok
    why = capdriver.native_cell_refusal("antidote")
    assert why and "antidote" in why and "--serving-path lens" in why
    # the signatures, before the Modal doubles go in: the cell function keeps the library default
    # (a direct call is today's lens cell), main the entry-point default
    assert inspect.signature(capdriver.run_cell).parameters["serving_path"].default == LENS
    assert inspect.signature(capdriver._run_cell).parameters["serving_path"].default == LENS
    assert inspect.signature(capdriver.main).parameters["serving_path"].default == NATIVE
    monkeypatch.setattr(capdriver, "stage_corpora", SimpleNamespace(remote=lambda *a, **k: pytest.fail("Modal was called")))
    monkeypatch.setattr(capdriver, "preflight", SimpleNamespace(remote=lambda *a, **k: pytest.fail("Modal was called")))
    # native-psr-serving: under the entry-point default the wave's psr cells PLAN natively — every cell is
    # handed the native path (before this spec the same call refused the whole wave, naming the hatch)
    calls: list = []
    monkeypatch.setattr(capdriver, "run_cell", SimpleNamespace(map=lambda specs, kwargs: calls.append((list(specs), kwargs)) or []))
    capdriver.main(sha="abc1234", fanout=True)
    specs, kwargs = calls[-1]
    assert kwargs["serving_path"] == NATIVE and len(specs) == len(capdriver.CELLS) * len(capdriver.MODELS)
    with pytest.raises(SystemExit, match="--serving-path must be one of"):
        capdriver.main(sha="abc1234", fanout=True, serving_path="native")
    # under the explicit lens path the wave runs as before and hands the path to every cell
    capdriver.main(sha="abc1234", fanout=True, serving_path=LENS)
    specs, kwargs = calls[-1]
    assert kwargs["serving_path"] == LENS and len(specs) == len(capdriver.CELLS) * len(capdriver.MODELS)
    # the read site: the engine first, the same exit shape (status, stop file, teardown, commit)

    class _Standup:
        def __init__(self):
            self.calls: list = []

        def wait(self, timeout=None):
            self.calls.append(("wait", timeout))
            return 0

        def kill(self):
            self.calls.append("kill")

    def refusal(endpoint, path, **kw):
        rec = {"model": "llama8b", "cell": "none"}
        stop, standup, commits = tmp_path / f"STOP-{len(list(tmp_path.iterdir()))}", _Standup(), []
        out = capdriver._served_engine_refusal(endpoint, serving_path=path, rec=rec, ep_path="ep.json", stop=stop,
                                               standup=standup, commit=lambda: commits.append(1), **kw)
        return out, rec, stop, standup, commits

    out, rec, stop, standup, commits = refusal({C.PROXY_IDENTITY_REDUCTION_MODE: "stacked"}, NATIVE)   # a lens endpoint under native
    assert out is rec and rec["status"] == "SERVING_ENGINE_MISMATCH" and rec["served_engine"] == LENS
    assert "did not serve the engine the launch asked for" in rec["serving_engine_error"]
    assert stop.exists() and standup.calls == [("wait", capdriver.STANDUP_TEARDOWN_TIMEOUT_S)] and commits == [1]
    out, rec, stop, standup, commits = refusal({C.PROXY_IDENTITY_REDUCTION_MODE: "stacked"}, LENS)     # lens under lens: proceeds
    assert out is None and rec["served_engine"] == LENS and not stop.exists() and standup.calls == [] and commits == []
    out, rec, *_ = refusal({"engine": NATIVE}, LENS)                                                     # native under lens
    assert out is rec and rec["status"] == "SERVING_ENGINE_MISMATCH" and rec["served_engine"] == NATIVE
    monkeypatch.setattr(capdriver, "proxy_engine_kind", lambda ep, **k: (NATIVE, None))
    out, rec, stop, *_ = refusal({"engine": NATIVE, "proxy_url": "http://x/v1"}, NATIVE)                # native under native: the live proxy agrees
    assert out is None and rec["served_engine"] == NATIVE and rec["proxy_engine_kind"] == NATIVE and not stop.exists()
    monkeypatch.setattr(capdriver, "proxy_engine_kind", lambda ep, **k: (None, "GET failed: refused"))
    out, rec, stop, *_ = refusal({"engine": NATIVE, "proxy_url": "http://x/v1"}, NATIVE)
    assert out is rec and rec["status"] == "SERVING_ENGINE_MISMATCH" and "GET failed: refused" in rec["serving_engine_error"]
    assert "does not vouch for the proxy answering" in rec["serving_engine_error"] or "attests engine=" in rec["serving_engine_error"]
    out, rec, *_ = refusal({"engine": NATIVE, C.SERVED_ENGINE_FIELD: LENS}, NATIVE)                      # torn
    assert out is rec and "torn record" in rec["serving_engine_error"]
    # the cell body: the engine check runs BEFORE the mode resolver, the bank rides in env + argv under native
    src = CAPCTRL_SERVED.read_text()
    body = src[src.index("def _run_cell("):src.index("def run_cell(")]
    assert body.index("_served_engine_refusal(endpoint") < body.index("_resolve_served_reduction_mode(endpoint")
    assert "env[ADAPTER_BANK_ROOT_ENV] = ADAPTER_BANK_ROOT" in body
    assert 'adapter_bank=(f"{ADAPTER_BANK_ROOT}/{model}" if native else None)' in body
    assert '"serving_path": serving_path, "serving_engine": engine_of_serving_path(serving_path)' in body
    assert "if SERVING_PATH_FLAG not in text:" in src[src.index("def preflight("):]


# --------------------------------------------------------------------------- #
# 10. the other Modal launchers name the lens standup; the inventory            #
# --------------------------------------------------------------------------- #


def test_the_non_tinycap_modal_launchers_name_the_lens_standup_explicitly(L):
    """§8.6: ``grokmt_served``, ``fst_multiturn``, ``petri_agentloop_control`` and
    ``misalign_vendor_shards`` are not capability / tinycap surfaces; each composes
    ``--serving-path lens`` EXPLICITLY (the standups' own default is native now) and carries a TODO
    naming the native reading it lacks."""
    import test_fst_multiturn as TF
    import test_grokmt_served_resume as TG
    grokmt = _load_by_path("grokmt_served_native_inventory", MODAL_DIR / "grokmt_served.py", TG._stub_modal())
    geom = TF._geom(eval="petri", model_key="gemma4e4b")
    cmd = grokmt._target_standup_cmd("/tree", Path("/r/srv"), Path("/r/srv/STOP"), geom)
    assert cmd[-2:] == ["--serving-path", LENS] and cmd.count("--serving-path") == 1 and "--adapter-bank" not in cmd
    assert cmd[1] == "/tree/scripts/capctrl_serving_up.py"
    fst = TF.load_driver()
    tt = TF.TT
    digests = {tt: "ab" * 32, f"{tt}@all_turns": "ab" * 32, f"{tt}@all_positions": "ab" * 32}
    cmd = fst._target_standup_cmd("/tree", Path("/r/srv"), Path("/r/srv/STOP"), TF._geom(),
                                  cells=list(fst.DEFAULT_CELLS), digests_by_tt=digests)
    assert cmd[-2:] == ["--serving-path", LENS] and cmd.count("--serving-path") == 1 and "--adapter-bank" not in cmd
    up = TL._load_standup()
    assert up.build_parser().parse_args(cmd[2:]).serving_path == LENS
    text = (MODAL_DIR / "petri_agentloop_control.py").read_text()
    body = text[text.index("def control("):]
    assert '"--serving-path", "lens"' in body and body.index("--petri-agent-loop") < body.index('"--serving-path", "lens"')
    text = (MODAL_DIR / "misalign_vendor_shards.py").read_text()
    i = text.index("scripts/sdfbase/mt_serving_up.py")
    argv = text[i:text.index("env=env", i)]
    assert '"--serving-path", "lens"' in argv and '"--reduction-mode", "stacked"' in argv
    for name in ("grokmt_served.py", "fst_multiturn.py", "petri_agentloop_control.py", "misalign_vendor_shards.py"):
        t = (MODAL_DIR / name).read_text()
        assert "native-serving-default" in t and "TODO" in t and t.count('"--serving-path", "lens"') == 1, name


def test_every_reduction_mode_reader_under_scripts_modal_handles_the_engine_or_names_the_lens(L):
    """The inventory: every Modal launcher that reads the lens reduction-mode fact
    (``lens_reduction_mode`` / ``PROXY_IDENTITY_REDUCTION_MODE``) either reads the ENGINE FIRST
    (``endpoint_engine`` + ``not_applicable``: exactly the two capability / tinycap surfaces) or
    names the lens standup explicitly in its argv."""
    readers, engine_aware, explicit_lens = [], set(), set()
    # `seedrep_served.standup_job_line` COPIES the endpoint's own `lens_reduction_mode` fact verbatim into the
    # cluster-shape `standup_jobs.jsonl` line -- a record passthrough that decides nothing on the value (None on a native
    # engine). Exempt exactly that spelling; any other mention would make the file a reader and trip the pin below.
    record_passthrough = {"seedrep_served.py": 'endpoint.get("lens_reduction_mode")'}
    for p in sorted(MODAL_DIR.glob("*.py")):
        t = p.read_text()
        if "lens_reduction_mode" not in t and "PROXY_IDENTITY_REDUCTION_MODE" not in t:
            continue
        if p.name in record_passthrough:
            stripped = t.replace(record_passthrough[p.name], "").replace('"lens_reduction_mode": ', "")
            assert "PROXY_IDENTITY_REDUCTION_MODE" not in stripped, p.name
            assert stripped.count("lens_reduction_mode") <= 1, (p.name, "a reader, not a passthrough")   # the docstring mention
            continue
        readers.append(p.name)
        if "def endpoint_engine(" in t and "not_applicable" in t and "def proxy_engine_kind(" in t:
            engine_aware.add(p.name)
        if '"--serving-path", "lens"' in t:
            explicit_lens.add(p.name)
        assert p.name in engine_aware or p.name in explicit_lens, f"{p.name} reads the lens reduction mode and neither " \
            "reads the engine first nor names --serving-path lens"
    assert set(readers) >= {"tinycap_served.py", "capctrl_served.py", "fst_multiturn.py", "misalign_vendor_shards.py"}
    assert engine_aware == {"tinycap_served.py", "capctrl_served.py"}
    assert engine_aware.isdisjoint(explicit_lens)
    # every capctrl_serving_up / mt_serving_up argv composed under scripts/modal states a path
    for p in sorted(MODAL_DIR.glob("*.py")):
        t = p.read_text()
        n_standups = t.count("scripts/capctrl_serving_up.py") + t.count("scripts/sdfbase/mt_serving_up.py")
        if n_standups and p.name not in ("tinycap_served.py", "capctrl_served.py"):
            if "subprocess.Popen" in t or "run_commands" in t:
                assert '"--serving-path", "lens"' in t or "--serving-path" in t, p.name

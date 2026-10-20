"""The capctrl FRONT over the shared native core, CPU-only.

``scripts/capctrl_serving_up.bring_up(serving_path="lora_native")`` — the single-cell / tinycap
standup under its DEFAULT serving path — driven through the REAL ``native_standup.bring_up_native``
with the scripted lora-native upstream of ``test_lora_serving_up`` (``fake_upstream_launch_server``:
the REAL composed launch, argv swapped for the stand-in, the REAL ``launch_server`` readiness and
attestation, the REAL RoutingProxy on an aiohttp thread, the five positive controls over HTTP, the
routing recorder over the dev RPC route). Only ``vs.launch_server`` and the exporter's snapshot seam
 are faked — NEVER ``bring_up_lora`` / ``bring_up_native``.

Pinned: a fitted cell + ``none`` with ``--tool-calls --manifest <the family's Petri manifest>`` puts
``--enable-auto-tool-choice`` + the family's tool parser on the native LAUNCH ARGV and records
``tool_calls: true``; capctrl's default-manifest TEMPLATE replacement (the run's model / revision,
the SERVED template's hash, ``--enable-lora`` stripped) is honoured before the native flag pass
strips ``--enforce-eager``; ``--cell none`` alone is a zero-adapter-only standup; ``--cell <psr>``
refuses naming ``--serving-path lens``; with neither cluster root the bank resolves per §1.9 (the
env root, the explicit flag, a refusal naming all three); ``--serving-path lens`` composes the argv and
the endpoint record GOLDEN-identically to the pre-landing tree (the same stubbed lens bring-up run
through ``git show HEAD:scripts/capctrl_serving_up.py``).

Skips (never fetches) when the pinned tokenizer / config is not in the local HF cache.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
#: the PRE-LANDING tree the lens goldens compare against: the branch base of native-serving-default (origin/main
#: at branch time, before any serving path existed). A fixed sha, so the golden RUNS on every later tree
#: (`HEAD:` skipped forever once HEAD carried the flag); a checkout without that object skips.
PRE_LANDING_SHA = "ea8bf598"
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import test_lora_serving_up as E  # noqa: E402
from test_serve_declarations_attestation import NEMOTRON, _att, _cached_tok, _stubbed_bring_up  # noqa: E402

from sisteer.capeval import manifest as cm  # noqa: E402
from sisteer.capeval.petri import registry_adapter as ra  # noqa: E402
from sisteer.capeval.serving import native_standup as ns  # noqa: E402
from sisteer.capeval.serving import proxy as px  # noqa: E402
from sisteer.capeval.serving import serving_attestation as sa  # noqa: E402
from sisteer.capeval.serving import geometry as sg  # noqa: E402,F401  (before any monkeypatch of vs.build_server_launch)
from sisteer.capeval.serving import vllm_server as vs  # noqa: E402

KEY = "gemma4e4b"
EVAL = "families_served:short"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def capctrl():
    return _load("_native_front_capctrl_serving_up", "scripts/capctrl_serving_up.py")


def _prepare(tmp_path, monkeypatch, *, genuine=E.GEMMA_HCELL, chat=None):
    """The gemma campaign config over a synthetic bank (the certification cell a REAL lora_all blob),
    the snapshot seam, the scripted upstream behind the REAL launch_server. Returns (cfg_path, cfg,
    view, shas, captured)."""
    cfg_path, man_path, cfg = E._campaign(KEY)
    E.cached_or_skip(cfg)
    view, shas = E.fake_bank(tmp_path, cfg)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("TT_TWO_SI_DEFENSES_DIR", raising=False)
    E.snapshot_seam(monkeypatch, tmp_path, cfg)
    captured: dict = {"shas": shas}
    monkeypatch.setattr(vs, "launch_server", E.fake_upstream_launch_server(
        tmp_path / "fake_upstream", captured=captured, chat=chat or E.default_chat(cfg["model"], genuine),
        genuine_name=genuine, chat_delay_s=0.05))
    return cfg_path, man_path, cfg, view, shas, captured


def _geom(capctrl):
    return capctrl.resolve_standup_geometry(EVAL, None, None, model_key=KEY)


def test_a_fitted_cell_with_tool_calls_on_the_family_manifest_composes_the_agent_modality_natively(
        capctrl, tmp_path, monkeypatch):
    cfg_path, man_path, cfg, view, shas, cap = _prepare(tmp_path, monkeypatch)
    out = tmp_path / "out"
    st = capctrl.bring_up(repo=REPO, config_path=cfg_path, cell=[E.GEMMA_CELL, "none"], defenses_dir=view, out=out,
                          geom=_geom(capctrl), manifest=str(man_path), tool_calls=True, model_key=KEY,
                          expect_object_sha256=f"{E.GEMMA_CELL}={shas[E.GEMMA_CELL]}",
                          adapter_bank=tmp_path / "bank", serving_path="lora_native", probe_max_tokens=8)
    try:
        assert isinstance(st, ns.LoraStandup) and st.front == "capctrl_serving_up"
        cmd = list(cap["launch"].cmd)
        # the AGENT modality on the native LAUNCH ARGV: explicit tool_calls -> the family's pinned parser
        assert "--enable-auto-tool-choice" in cmd
        assert cmd[cmd.index("--tool-call-parser") + 1] == cm.load_petri_spec(man_path).tool_parser == "gemma4"
        assert "--no-enforce-eager" in cmd and "--enforce-eager" not in cmd and "--lora-modules" in cmd
        assert cmd[-4:] == ["--gpu-memory-utilization", "0.9", "--max-logprobs", "64"]   # capctrl's splice
        ep = st.endpoint(EVAL)
        assert ep["tool_calls"] is True and ep["engine"] == ep["serving_engine"] == "lora_native"
        assert ep["cell"] == E.GEMMA_CELL and ep["harness_cell"] == E.GEMMA_HCELL
        assert ep["cells"] == ["none", E.GEMMA_CELL] and ep["harness_cells"] == ["none", E.GEMMA_HCELL]
        assert ep["object_artifact_sha256_by_cell"] == {E.GEMMA_HCELL: shas[E.GEMMA_CELL]}
        assert ep["object_artifact_sha256"] == shas[E.GEMMA_CELL] and ep["expected_digests_source"] == "cli"
        # native-positions-resolve: the plain LoRA adapter attests `all` from its KIND; the exporter stamps it on the
        # route AND into the engine-neutral served_positions map (the lens map stays empty on a native table), and the
        # endpoint record copies the map — what the served driver reads first
        table = json.loads((out / ns.ROUTING_TABLE_FILE).read_text())
        served_route = next(r for r in table["routes"] if r["cell_id"] == E.GEMMA_HCELL)
        assert served_route["petri_position_mode"] == "all" and table[px.LENS_SERVED_POSITIONS_KEY] == {}
        assert table[px.SERVED_POSITIONS_KEY] == {E.GEMMA_HCELL: ["all"]} == ep[ns.SERVED_POSITIONS_KEY]
        assert px.SERVED_POSITIONS_KEY == ns.SERVED_POSITIONS_KEY and ns.SERVED_POSITIONS_KEY in ns._CORE_RECORD_KEYS
        assert "petri_position_mode" not in next(r for r in table["routes"] if r["cell_id"] == E.ZERO)
        assert ep["eval"] == EVAL and ep["mode"] is None and ep["manifest_flags_dropped"] == ["--enforce-eager"]
        assert ep["max_num_seqs"] == 107 and ep["geometry"]["adjustments"][0]["kind"] == "lora_kernel_index_ceiling"
        # the lowering moved S only: the WINDOW keeps its table source (a re-passed L had stamped `override`, which the
        # capability readers refuse before they read the adjustment)
        assert ep["geometry"]["window_source"] == "table" and ep["geometry"]["seqs_source"] == "override"
        assert ep["adapter_bank"] == str((tmp_path / "bank").resolve())
        assert "lens_reduction_mode" not in ep and ep["reduction_mode"] == "not_applicable"
        assert ep["mrope_attestation"] is None and ep["mamba_backend_attestation"] is None
        assert "not a nemotron_h target" in ep["mamba_backend_attestation_error"]
        assert (out / "endpoint.json").is_file() and (out / "positive_controls.json").is_file()
        pc = json.loads((out / "positive_controls.json").read_text())
        assert pc["passed"] is True and all(pc["controls"][n]["passed"] is True for n in ns.CONTROL_NAMES
                                             if n != "boundary_refusal")
        assert pc["controls"]["boundary_refusal"]["verdict"] == ns.NOT_APPLICABLE     # a LoRA-only standup (O49)
        # the shared shape capctrl's main reads at teardown
        assert st.server_rc() is None and st.mrope_attestation is None and callable(st.tear_down)
        assert st.proxy.models_payload()["petri_proxy"][px.ENGINE_KEY]["kind"] == "lora_native"
    finally:
        st.tear_down()
    assert not (out / "endpoint.json").exists()


def test_the_default_manifest_is_a_template_whose_target_and_served_hash_are_replaced(capctrl, tmp_path, monkeypatch):
    """capctrl's effective spec (:850-863) stands: the default twosi manifest pins another model and template
    hash; the front replaces target / revision / template hash with the run's (the core then holds the
    served bytes to that hash) and strips --enable-lora before the native pass strips --enforce-eager."""
    cfg_path, man_path, cfg, view, shas, cap = _prepare(tmp_path, monkeypatch)
    default = cm.load_petri_spec(REPO / "configs/capeval/manifests/petri_twosi.yaml")
    assert default.target_model != cfg["model"]
    st = capctrl.bring_up(repo=REPO, config_path=cfg_path, cell=E.GEMMA_CELL, defenses_dir=view, out=tmp_path / "out",
                          geom=_geom(capctrl), model_key=KEY, expect_object_sha256=shas[E.GEMMA_CELL],
                          adapter_bank=tmp_path / "bank", serving_path="lora_native", probe_max_tokens=8)
    try:
        ep = st.endpoint()
        served_sha16 = cm.sha256_hex16((tmp_path / "out" / "chat_template.jinja").read_bytes())
        assert ep["chat_template_hash"] == served_sha16 != default.chat_template_hash
        assert ep["model"] == cfg["model"] and ep["model_revision"] == cfg["model_revision"]
        assert ep["tool_calls"] is False and "--enable-auto-tool-choice" not in cap["launch"].cmd
        assert ep["manifest_flags_dropped"] == [f for f in default.serving_launch_flags if f == "--enforce-eager"]
        assert "--enable-lora" not in ep["manifest_flags_dropped"]        # stripped by the template step, as today
        assert st.manifest_path == REPO / "configs/capeval/manifests/petri_twosi.yaml"
    finally:
        st.tear_down()


def test_cell_none_alone_is_a_zero_adapter_only_standup(capctrl, tmp_path, monkeypatch):
    cfg_path, man_path, cfg, view, shas, cap = _prepare(tmp_path, monkeypatch)
    st = capctrl.bring_up(repo=REPO, config_path=cfg_path, cell="none", defenses_dir=view, out=tmp_path / "out",
                          geom=_geom(capctrl), model_key=KEY, adapter_bank=tmp_path / "bank",
                          serving_path="lora_native", probe_max_tokens=8)
    try:
        ep = st.endpoint()
        assert st.zero_adapter_only and ep["lora_cells"] == [] and ep["adapters"] == {}
        assert ep["id_name_map"] == {"1": E.ZERO} and list(cap["launch"].lora_modules) == [E.ZERO]
        assert set(ep["served_models"]) == {"petri__base", f"petri__{E.ZERO}"}
        assert ep["cells"] == ["none"] and ep["cell"] == "none" and ep["object_artifact_sha256"] is None
        pc = json.loads((tmp_path / "out" / "positive_controls.json").read_text())
        assert pc["passed"] is True
        assert pc["controls"]["steering_bites"]["verdict"] == pc["controls"]["selection_honored"]["verdict"] == "not_applicable"
        assert pc["controls"]["noop_certificate"]["passed"] is True and pc["controls"]["adapter_digests"]["passed"] is True
    finally:
        st.tear_down()


def test_an_explicit_psr_cell_refuses_naming_the_escape_hatch_and_a_lens_only_flag_refuses(capctrl, tmp_path, monkeypatch):
    # STAGE-1 WORLD (native-psr-serving step 7 flipped both flags): the fake bank's psr cell is a
    # RESPONSE-fit blob, which the disposition SKIPS by name only while the position class is
    # unarmed — armed, it serves (and this bring_up would refuse on the deliberately wrong
    # expect_object_sha256 instead). tests/test_lora_serving_up.py pins the same fact both ways.
    from sisteer.capeval.petri import defense_cells as _dc
    from sisteer.capeval.serving import native_residual as _nr
    monkeypatch.setattr(_dc, "NATIVE_STAGE2_ARMED", False)
    monkeypatch.setattr(_nr, "STAGE2_POSITION_CLASSES", False)
    cfg_path, man_path, cfg, view, shas, cap = _prepare(tmp_path, monkeypatch)
    launched: list = []
    monkeypatch.setattr(vs, "build_server_launch", lambda *a, **k: launched.append(1))
    with pytest.raises(ns.StandupRefused, match="--serving-path lens") as ei:
        capctrl.bring_up(repo=REPO, config_path=cfg_path, cell=E.PSR_CELL, defenses_dir=view, out=tmp_path / "psr",
                         geom=_geom(capctrl), model_key=KEY, expect_object_sha256="cd" * 32,
                         adapter_bank=tmp_path / "bank", serving_path="lora_native")
    assert ei.value.rc == 2 and "psr" in str(ei.value) and launched == []
    with pytest.raises(ns.StandupRefused, match="--lens-salt-policy"):
        capctrl.bring_up(repo=REPO, config_path=cfg_path, cell="none", defenses_dir=view, out=tmp_path / "salt",
                         geom=_geom(capctrl), model_key=KEY, adapter_bank=tmp_path / "bank",
                         serving_path="lora_native", lens_salt_policy="request_unique_all")
    with pytest.raises(ValueError, match="serving-path"):
        capctrl.bring_up(repo=REPO, config_path=cfg_path, cell="none", defenses_dir=view, out=tmp_path / "bad",
                         geom=_geom(capctrl), model_key=KEY, serving_path="native")
    assert launched == []


def test_the_adapter_bank_resolves_per_1_9_with_neither_cluster_root(capctrl, tmp_path, monkeypatch):
    """Off-cluster (no shared root) the bank is the explicit flag, else $SISTEER_ADAPTER_BANK_ROOT/
    <model_key> (the tinycap launcher's /vol/tinycap/multilora_adapters), else a refusal naming all three —
    BEFORE any launch is composed."""
    cfg_path, man_path, cfg, view, shas, cap = _prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(ns, "shared_root", lambda: None)
    monkeypatch.delenv(ns.ADAPTER_BANK_ROOT_ENV, raising=False)
    common = dict(repo=REPO, config_path=cfg_path, cell="none", defenses_dir=view, geom=_geom(capctrl),
                  model_key=KEY, serving_path="lora_native", probe_max_tokens=8)
    with pytest.raises(ns.StandupRefused) as ei:
        capctrl.bring_up(out=tmp_path / "none", **common)
    for source in ("--adapter-bank", "SISTEER_SHARED_ROOT", ns.ADAPTER_BANK_ROOT_ENV):
        assert source in str(ei.value), source
    assert "launch" not in cap                                          # refused before any launch
    # the env root resolves under <root>/<model_key>
    monkeypatch.setenv(ns.ADAPTER_BANK_ROOT_ENV, str(tmp_path / "vol"))
    st = capctrl.bring_up(out=tmp_path / "env", **common)
    try:
        assert st.endpoint()["adapter_bank"] == str((tmp_path / "vol" / KEY).resolve())
        assert Path(st.zero_adapter["dir"]).parent == (tmp_path / "vol" / KEY).resolve()
    finally:
        st.tear_down()
    # the explicit flag wins over both
    st2 = capctrl.bring_up(out=tmp_path / "explicit", adapter_bank=tmp_path / "flag", **common)
    try:
        assert st2.endpoint()["adapter_bank"] == str((tmp_path / "flag").resolve())
    finally:
        st2.tear_down()


def test_main_parses_the_serving_path_and_the_bank_and_defaults_to_the_entry_point_default(capctrl):
    p = capctrl.build_parser()
    base = ["--repo-dir", str(REPO), "--config", "c.yaml", "--cell", "none", "--defenses-dir", "d", "--out", "o",
            "--stop-file", "s"]
    assert p.parse_args(base).serving_path is None and p.parse_args(base).adapter_bank is None
    assert p.parse_args(base + ["--serving-path", "lens"]).serving_path == "lens"
    assert p.parse_args(base + ["--serving-path", "lora_native", "--adapter-bank", "/vol/x"]).adapter_bank == "/vol/x"
    with pytest.raises(SystemExit):
        p.parse_args(base + ["--serving-path", "native"])
    src = (REPO / "scripts/capctrl_serving_up.py").read_text()
    assert "serving_path = args.serving_path or ns.DEFAULT_STANDUP_SERVING_PATH" in src
    assert "serving_path=serving_path, adapter_bank=args.adapter_bank" in src
    assert ns.DEFAULT_STANDUP_SERVING_PATH == ra.DEFAULT_STANDUP_SERVING_PATH == "lora_native"
    # the library default is the entry-point default: a bare bring_up() goes native
    import inspect
    assert inspect.signature(capctrl.bring_up).parameters["serving_path"].default is None
    assert "serving_path = DEFAULT_STANDUP_SERVING_PATH" in src
    # the import-time surface stays free of sisteer (the fronts import the core lazily)
    head = src.split("\ndef free_port", 1)[0]
    assert "from sisteer" not in head and "import sisteer" not in head


# --------------------------------------------------------------------------- #
# --serving-path lens: the pre-landing lens body, byte for byte                #
# --------------------------------------------------------------------------- #

_VOLATILE = ("pid", "node", "written_at", "started")


def _lens_golden(capctrl_mod, tmp_path, monkeypatch, **kw):
    """The stubbed lens bring-up (the shared harness) on ``capctrl_mod``: the composed argv and the
    endpoint record minus the volatile keys."""
    st, proxy = _stubbed_bring_up(capctrl_mod, tmp_path, monkeypatch, attest=_att({}, {}), **kw)
    ep = {k: v for k, v in st.endpoint("capability").items() if k not in _VOLATILE}
    ep["routing_table"] = Path(ep["routing_table"]).name
    return list(st.launch.cmd), dict(st.launch.env), ep


def test_serving_path_lens_is_golden_identical_to_the_pre_landing_tree(capctrl, tmp_path, monkeypatch):
    """The lens argv / env / endpoint of THIS tree's ``bring_up(serving_path="lens")`` equal the pre-landing
    tree's ``bring_up()`` (``git show HEAD:scripts/capctrl_serving_up.py``), on the same stubs — the lens
    path composes byte-identically. Skips when HEAD's copy is not readable (a shallow / detached checkout)."""
    _cached_tok(*NEMOTRON)
    try:
        head_src = subprocess.run(["git", "-C", str(REPO), "show", f"{PRE_LANDING_SHA}:scripts/capctrl_serving_up.py"],
                                  capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        pytest.skip(f"HEAD's capctrl_serving_up.py is not readable here: {e}")
    if "serving_path" in head_src:
        pytest.skip("HEAD already carries the serving path (the golden is the pre-landing tree)")
    head_path = tmp_path / "capctrl_serving_up_head.py"
    head_path.write_text(head_src)
    spec = importlib.util.spec_from_file_location("_capctrl_head_golden", head_path)
    head = importlib.util.module_from_spec(spec)
    sys.modules["_capctrl_head_golden"] = head
    spec.loader.exec_module(head)
    # the shared harness names the path it exercises (serving_path="lens"); the pre-landing bring_up has no such
    # keyword, so the golden strips it there — every other keyword reaches both trees identically
    real_bring_up = head.bring_up
    head.bring_up = lambda **kw: real_bring_up(**{k: v for k, v in kw.items() if k != "serving_path"})
    argv_new, env_new, ep_new = _lens_golden(capctrl, tmp_path / "new", monkeypatch, serving_path="lens")
    argv_old, env_old, ep_old = _lens_golden(head, tmp_path / "old", monkeypatch)
    # the ports are free-port draws and each run has its OWN out root (the --chat-template path, the records'
    # paths); everything else on the argv is the same bytes
    strip = lambda cmd, root: [a.replace(str(root), "<OUT>") for i, a in enumerate(cmd)  # noqa: E731
                               if not (i and cmd[i - 1] == "--port")]
    assert strip(argv_new, tmp_path / "new") == strip(argv_old, tmp_path / "old")
    envn = lambda env, root: {k: v.replace(str(root), "<OUT>") for k, v in env.items() if k != "VLLM_API_KEY"}  # noqa: E731
    assert envn(env_new, tmp_path / "new") == envn(env_old, tmp_path / "old")       # PETRI_LENS_REGISTRY_DIR is under out
    for k in ("proxy_url",):
        ep_new.pop(k, None); ep_old.pop(k, None)
    norm = lambda ep, root: json.loads(json.dumps(ep, default=str).replace(str(root), "<OUT>"))  # noqa: E731
    assert norm(ep_new, tmp_path / "new") == norm(ep_old, tmp_path / "old")
    assert "engine" not in ep_new and "serving_engine" not in ep_new           # a lens record names no engine

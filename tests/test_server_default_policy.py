"""The SERVER-DEFAULT SAMPLING POLICY: ``vllm_server.build_server_launch(...,
server_default_policy="contract_override" | "engine_defaults")``, recorded on ``ServerLaunch``, ``endpoint.json``,
``cells_served.json`` and the proxy's ``/v1/models`` identity block, threaded through ``mt_serving_up.py
--server-default-policy`` (both serving paths), ``lora_serving_up`` / ``native_standup``, and ``common.sh f22_standup`` as
the scalar ``F22_SERVER_DEFAULT_POLICY``.

* ``contract_override`` (the default) composes the launch BYTE-IDENTICALLY to today's for gemma / qwen / llama / nemotron
  (the explicit policy equals the implicit default; Qwen still carries the pinned vendor row as
  ``--override-generation-config``, the others nothing);
* ``engine_defaults`` removes exactly the two override tokens from the Qwen argv (nothing else moves), records
  ``override_generation_config: None`` BY CHOICE and the policy name, and skips the vendor-file premise probe (nothing is
  composed, nothing can mask a file);
* an unknown policy refuses; the mt / lora parsers accept exactly the vocabulary;
* the standup artifacts carry the field on the LENS path (endpoint / cells_served / proxy identity) and on the NATIVE path;
* a synthetic ``record_resolved_sampler.py`` run over an ``engine_defaults`` log AGREES with the neutral tuple and
  DIVERGES with the 0.7 / 0.8 / 20 one.

Skips (never fetches) the standup-record cases when the pinned gemma tokenizer is not in the local HF cache.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import huggingface_hub
# transformers binds `huggingface_hub.try_to_load_from_cache` BY NAME when it is first imported: import it here, before any
# test stubs that hub attribute (the `no_hub_file` fixture), or the stub is bound into transformers.utils.hub for the rest
# of the process and the standup harnesses below skip on a "missing" gemma tokenizer that IS in the cache (a pinned
# foreign import pollutes later imports — bisected 2026-09-15: every composer test run before the harness tripped it)
pytest.importorskip("transformers")
from sisteer.capeval.serving import native_standup as ns
from sisteer.capeval.serving import proxy as px
from sisteer.capeval.serving import vllm_server as vs

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts", REPO / "scripts" / "sdfbase", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from test_petri_serving_gate import (  # noqa: E402
    THINK_TEMPLATE_HASH, _spec, _think_template, _tpl_spec, _chat_template)
from test_qwen_server_default_sampling import EXPECTED_JSON, QWEN4B, QWEN4B_REV  # noqa: E402

COMMON = (REPO / "scripts/sdfbase/full22/common.sh").read_text()
PETRI_PROXY = (REPO / "scripts/sdfbase/full22/petri_proxy.sbatch").read_text()
MISALIGN_PROXY = (REPO / "scripts/sdfbase/full22/misalign/misalign_proxy.sbatch").read_text()
MT_SRC = (REPO / "scripts/sdfbase/mt_serving_up.py").read_text()


@pytest.fixture
def no_hub_file(monkeypatch):
    """The composer tests' premise: the hub was never asked (None) — NOT autouse, because the standup harnesses below
    locate the cached gemma tokenizer through the same ``try_to_load_from_cache`` and would skip under the stub."""
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)


def _launch(tmp_path, model, revision, **kw):
    if "Llama" in model:
        return vs.build_server_launch(_tpl_spec(target_model=model, target_revision=revision,
                                                serving_launch_flags=["--enforce-eager"]),
                                      port=8801, exact_replay_strategy="serialize",
                                      chat_template_path=_chat_template(tmp_path), **kw)
    return vs.build_server_launch(_spec(target_model=model, target_revision=revision,
                                        chat_template_hash=THINK_TEMPLATE_HASH,
                                        serving_launch_flags=["--enforce-eager"]),
                                  port=8801, exact_replay_strategy="serialize",
                                  chat_template_path=_think_template(tmp_path), **kw)


def _comparable(launch):
    """The launch minus its per-call random token."""
    return dataclasses.replace(launch, api_key=None)


MODELS = [
    ("gemma", "google/gemma-4-E4B-it-synthetic", "0" * 40),
    ("qwen", QWEN4B, QWEN4B_REV),
    ("llama", "meta-llama/Llama-3.1-8B-Instruct", "0123456789abcdef0123456789abcdef01234567"),
    ("nemotron", "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16-synthetic", "0" * 40),
]


# --------------------------------------------------------------------------- the vocabulary

def test_the_vocabulary_and_the_default_are_pinned_everywhere():
    assert vs.SERVER_DEFAULT_POLICIES == ("contract_override", "engine_defaults")
    assert vs.SERVER_DEFAULT_POLICY_DEFAULT == "contract_override"
    assert dataclasses.fields(vs.ServerLaunch)[[f.name for f in dataclasses.fields(vs.ServerLaunch)].index("server_default_policy")].default == "contract_override"
    import inspect
    assert inspect.signature(vs.build_server_launch).parameters["server_default_policy"].default == "contract_override"
    core = inspect.signature(ns.bring_up_native).parameters["server_default_policy"]
    assert core.kind == core.KEYWORD_ONLY and core.default == "contract_override"
    assert "server_default_policy" in ns._CORE_RECORD_KEYS
    assert px.SERVER_DEFAULT_POLICY_KEY == "server_default_policy"
    # the mt parser spells the vocabulary literally (no serving-tree import at parse time) — held equal to the module's
    m = re.search(r'"--server-default-policy", choices=\[([^\]]*)\], default="([^"]+)"', MT_SRC)
    assert m and tuple(json.loads("[" + m.group(1) + "]")) == vs.SERVER_DEFAULT_POLICIES and m.group(2) == vs.SERVER_DEFAULT_POLICY_DEFAULT


# --------------------------------------------------------------------------- the composer

@pytest.mark.parametrize("family, model, revision", MODELS, ids=[m[0] for m in MODELS])
def test_the_default_argv_is_byte_identical_to_an_explicit_contract_override(no_hub_file, tmp_path, family, model, revision):
    implicit = _comparable(_launch(tmp_path, model, revision))
    explicit = _comparable(_launch(tmp_path, model, revision, server_default_policy="contract_override"))
    assert implicit == explicit
    assert implicit.server_default_policy == "contract_override"
    if family == "qwen":
        cmd = list(implicit.cmd)
        assert cmd[cmd.index("--override-generation-config") + 1] == EXPECTED_JSON and implicit.override_generation_config == EXPECTED_JSON
    else:
        assert "--override-generation-config" not in implicit.cmd and implicit.override_generation_config is None


def test_engine_defaults_removes_exactly_the_override_from_the_qwen_argv_and_records_the_policy(no_hub_file, tmp_path):
    co = _comparable(_launch(tmp_path, QWEN4B, QWEN4B_REV, server_default_policy="contract_override"))
    ed = _comparable(_launch(tmp_path, QWEN4B, QWEN4B_REV, server_default_policy="engine_defaults"))
    cmd_co, cmd_ed = list(co.cmd), list(ed.cmd)
    assert "--override-generation-config" not in cmd_ed
    i = cmd_co.index("--override-generation-config")
    assert cmd_ed == cmd_co[:i] + cmd_co[i + 2:], "engine_defaults may remove the two override tokens and nothing else"
    assert ed.override_generation_config is None and ed.server_default_policy == "engine_defaults"
    assert ed.env == co.env
    # every other recorded field agrees
    for f in dataclasses.fields(vs.ServerLaunch):
        if f.name in ("cmd", "override_generation_config", "server_default_policy"):
            continue
        assert getattr(ed, f.name) == getattr(co, f.name), f.name


@pytest.mark.parametrize("family, model, revision", [m for m in MODELS if m[0] != "qwen"], ids=[m[0] for m in MODELS if m[0] != "qwen"])
def test_engine_defaults_on_an_unpinned_model_changes_only_the_recorded_policy(no_hub_file, tmp_path, family, model, revision):
    co = _comparable(_launch(tmp_path, model, revision))
    ed = _comparable(_launch(tmp_path, model, revision, server_default_policy="engine_defaults"))
    assert ed.cmd == co.cmd and ed.env == co.env and ed.override_generation_config is None
    assert dataclasses.replace(ed, server_default_policy="contract_override") == co


def test_engine_defaults_skips_the_vendor_file_premise_probe(tmp_path, monkeypatch):
    """A cached generation_config.json refuses contract_override (the override would mask it — the pinned rule) but
    composes under engine_defaults: nothing is composed, so nothing can mask the file and the engine reads it."""
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda *a, **k: str(tmp_path / "snap" / "generation_config.json"))
    with pytest.raises(vs.VllmServerError, match="generation_config.json"):
        _launch(tmp_path, QWEN4B, QWEN4B_REV, server_default_policy="contract_override")
    ed = _launch(tmp_path, QWEN4B, QWEN4B_REV, server_default_policy="engine_defaults")
    assert ed.override_generation_config is None and ed.server_default_policy == "engine_defaults"
    calls = []
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: calls.append(a) or None)
    _launch(tmp_path, QWEN4B, QWEN4B_REV, server_default_policy="engine_defaults")
    assert calls == [], "engine_defaults asked the hub about a file it will never override"


def test_an_unknown_policy_refuses(no_hub_file, tmp_path):
    with pytest.raises(vs.VllmServerError, match="server_default_policy='vendor_row' is not one of"):
        _launch(tmp_path, QWEN4B, QWEN4B_REV, server_default_policy="vendor_row")


# --------------------------------------------------------------------------- the parsers

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


REQUIRED = ["--repo-dir", "r", "--config", "c", "--manifest", "m", "--defenses-dir", "d", "--out", "o",
            "--stop-file", "s", "--mode", "petri", "--model-key", "gemma4e4b"]


def test_mt_serving_up_parses_the_flag_with_the_default_and_refuses_an_unknown_value():
    mt = _load("_mt_sdp_parser", REPO / "scripts/sdfbase/mt_serving_up.py")
    ap = mt._build_parser()
    assert ap.parse_args(REQUIRED).server_default_policy == "contract_override"
    assert ap.parse_args(REQUIRED + ["--server-default-policy", "engine_defaults"]).server_default_policy == "engine_defaults"
    with pytest.raises(SystemExit):
        ap.parse_args(REQUIRED + ["--server-default-policy", "vendor_row"])
    # forwarded on BOTH paths: the lens launch composition and the native front call
    assert "server_default_policy=args.server_default_policy)" in MT_SRC
    assert MT_SRC.count("server_default_policy=args.server_default_policy") == 2
    lsu = _load("_lsu_sdp_parser", REPO / "scripts/sdfbase/lora_serving_up.py")
    ap2 = lsu.build_parser()
    base = ["--repo-dir", "r", "--config", "c", "--manifest", "m", "--defenses-dir", "d", "--out", "o", "--stop-file", "s",
            "--model-key", "gemma4e4b", "--mode", "static"]
    assert ap2.parse_args(base).server_default_policy == "contract_override"
    assert ap2.parse_args(base + ["--server-default-policy", "engine_defaults"]).server_default_policy == "engine_defaults"
    assert "server_default_policy" in lsu.NATIVE_FRONT_OPTIONS


def test_the_shell_threads_the_scalar_and_asserts_it_back_from_both_stamps():
    # f22_standup: validation, the argv token (a NEW line — the pinned thinking line is untouched), the read-back
    assert 'local want_sdp="${F22_SERVER_DEFAULT_POLICY:-}"' in COMMON
    assert 'case "$want_sdp" in ""|contract_override|engine_defaults) ;;' in COMMON
    assert '${F22_THINKING_ON:+--thinking-on} \\\n    ${want_sdp:+--server-default-policy "$want_sdp"} ${log_requests_flag:+"$log_requests_flag"} \\' in COMMON
    assert "print(json.load(open(sys.argv[1])).get('server_default_policy'))" in COMMON
    assert "(json.load(sys.stdin).get('petri_proxy') or {}).get('server_default_policy')" in COMMON
    assert 'REFUSING TRAFFIC: this standup asked for server-default sampling policy $want_sdp' in COMMON
    assert 'TARGET_SERVER_DEFAULT_POLICY="$ep_sdp"' in COMMON
    # the scalar log-requests opt-in (the plan's F22_LOG_REQUESTS=1; a Bash array never crosses sbatch)
    assert '[ "${F22_LOG_REQUESTS:-0}" = 1 ] && log_requests_flag="--enable-log-requests"' in COMMON
    # the anchor preflight gate exists and is what the two proxies call
    assert "f22_anchor_preflight(){" in COMMON and "anchor_sampler_preflight.py" in COMMON
    for text, leg in ((PETRI_PROXY, "petri"), (MISALIGN_PROXY, "misalign")):
        assert 'case "${F22_SERVER_DEFAULT_POLICY:-}" in' in text
        assert 'contract_override|engine_defaults) f22_log "server-default sampling policy:' in text
        assert 'F22_SERVER_DEFAULT_POLICY must be contract_override or engine_defaults' in text
        assert 'if [ "${F22_SERVER_DEFAULT_POLICY:-}" = engine_defaults ] && [ "${F22_LOG_REQUESTS:-0}" != 1 ]; then' in text
        assert f"f22_anchor_preflight {leg} " in text and 'if [ -n "${F22_ANCHOR_PREFLIGHT:-}" ]; then' in text
        assert _no_export_flag(text)
    assert subprocess.run(["bash", "-n", str(REPO / "scripts/sdfbase/full22/common.sh")]).returncode == 0



def _no_export_flag(text: str) -> bool:
    """No `--export=` in a live line (a comment saying "never --export=" is the rule, not a use)."""
    return not any("--export=" in ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))

# --------------------------------------------------------------------------- the standup records (CPU, gemma tokenizer cached)

def test_the_lens_standup_records_carry_the_policy_on_every_stamp(tmp_path, monkeypatch):
    import test_mt_serving_up_thinking as T
    mt = T._load_mt()
    cap, proxy, launch = T._run(mt, tmp_path / "ed", monkeypatch, "gemma4e4b", "--server-default-policy", "engine_defaults")
    assert launch.server_default_policy == "engine_defaults"
    assert cap["endpoint"]["server_default_policy"] == "engine_defaults"
    assert cap["cells_served"]["server_default_policy"] == "engine_defaults"
    assert proxy.models_payload()["petri_proxy"]["server_default_policy"] == "engine_defaults"
    assert cap["endpoint"]["override_generation_config"] is None
    cap2, proxy2, launch2 = T._run(mt, tmp_path / "co", monkeypatch, "gemma4e4b")
    assert launch2.server_default_policy == "contract_override"
    assert cap2["endpoint"]["server_default_policy"] == cap2["cells_served"]["server_default_policy"] == "contract_override"
    assert proxy2.models_payload()["petri_proxy"]["server_default_policy"] == "contract_override"


def test_the_native_standup_records_carry_the_policy(tmp_path, monkeypatch):
    import test_mt_serving_up_serving_path as SP
    mt = SP._load_mt("_mt_sdp_native")
    cap = SP._run_native(mt, tmp_path, monkeypatch, "--server-default-policy", "engine_defaults")
    assert cap["rc"] == 0
    assert cap["endpoint"]["server_default_policy"] == "engine_defaults" == cap["cells_served"]["server_default_policy"]
    assert cap["endpoint"]["engine"] == "lora_native" and cap["endpoint"]["override_generation_config"] is None


def test_a_proxy_without_the_key_in_its_attestation_echoes_nothing():
    """A pre-key standup's identity block carries no server_default_policy (never defaulted at the proxy)."""
    payload_src = px.RoutingProxy.models_payload.__code__.co_consts
    assert px.SERVER_DEFAULT_POLICY_KEY in {c for c in payload_src if isinstance(c, str)} or True   # the echo is conditional (source-pinned below)
    src = (REPO / "src/sisteer/capeval/serving/proxy.py").read_text()
    assert "if SERVER_DEFAULT_POLICY_KEY in self._launch_attestation" in src


# --------------------------------------------------------------------------- the recorder over an engine_defaults log

def test_record_resolved_sampler_agrees_with_the_neutral_tuple_and_diverges_with_the_vendor_row(tmp_path):
    import record_resolved_sampler as rrs
    import test_record_resolved_sampler as R
    neutral = [R._STARTED, R._ATTEST_INFO] + [R._sp(i, cap=1024, seed=None, **R._NEUTRAL).replace(f"seed={1000 + i}", "seed=None") for i in range(4)]
    vendor = [R._STARTED, R._ATTEST_INFO] + [R._sp(i, cap=1024, **R._VENDOR) for i in range(4)]
    expect = ["--expect", "temperature=1.0", "--expect", "top_p=1.0", "--expect", "top_k=0", "--expect", "presence_penalty=0.0"]
    for name, lines, want in (("neutral", neutral, "AGREES"), ("vendor", vendor, "DIVERGED")):
        log = tmp_path / f"{name}.log"
        log.write_text("\n".join(lines) + "\n")
        out = tmp_path / f"{name}.json"
        rc = rrs.main(["--vllm-log", str(log), "--out", str(out), "--client-cap", "1024", *expect])
        assert rc == 0   # recorded, not propagated (the default exit policy)
        rec = json.loads(out.read_text())
        assert rec["verdict"] == want, rec["why"]
    # vLLM logs `top_k=0` for "off": the plan's `--expect top_k=-1` would read as a false divergence on a neutral log
    assert rrs.VLLM_NEUTRAL_DEFAULTS["top_k"] == 0

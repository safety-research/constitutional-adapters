"""The composer's LORA-NATIVE mode and its readiness contract.

- COMPOSE: the exact argv and env under ``lora_native=True``
  (``--no-enforce-eager`` present and ``--enforce-eager`` absent,
  ``--enable-lora --max-loras N --max-lora-rank <fitted> --max-cpu-loras
  N``, ONE ``--lora-modules``, the plugin-free extension + the middleware,
  ``VLLM_PLUGINS == ""``, ``VLLM_LENS_DISABLE == "1"``, dev mode,
  ``PYTHONPATH``, no registry env); every refusal (a lens registry, no
  adapters, a manifest ``--enforce-eager`` / ``--no-enforce-eager`` /
  ``--max-cpu-loras``); the rank fit; the qwen manifest's unowned flags
  surviving; and THE LENS+LORA LAUNCH BYTE-IDENTICAL to a golden argv
  captured from the composer BEFORE this mode existed (9486b825).
- READINESS, against the scripted fake server (``_FAKE_SERVER`` from
  test_petri_serving_gate, extended to list LoRA cards with roots and to
  answer ``sisteer_scheduler_config`` with canned engine / LoRA keys): an
  agreeing server writes ``serving_attestation.json`` with the §1.3.2
  keys (``id_name_map``, ``name_binding.asserted``, per-adapter
  ``lora_int_id``); each of {enforce_eager True, cudagraph NONE, plugin
  imported, count mismatch, per-name fingerprint mismatch, two adapters
  permuted between ids, two roots swapped between names, a root
  elsewhere, a missing card, a duplicated card, a non-bf16 dtype, a
  clamped geometry, a KV shortfall, a LoRA-config drift, a missing RPC
  route} refuses readiness, shuts the child down and leaves NO
  attestation. ``cudagraph_mode`` and ``lora_dtype`` are RECORDED, not
  asserted, and a test pins that too.
"""

import dataclasses
import json
import re
import socket
import sys
import time
from pathlib import Path

import pytest
import torch

from sisteer.capeval.serving import lora_export as lx
from sisteer.capeval.serving import native_worker as nw
from sisteer.capeval.serving import serving_attestation as sa
from sisteer.capeval.serving import vllm_server as vs

sys.path.insert(0, str(Path(__file__).parent))
from test_petri_serving_gate import (  # noqa: E402
    REV, _chat_template, _default_capture_sizes, _fake_lora_native_launch,
    _flat_registry, _native_sched_report, _tiny_peft_adapter, _tpl_spec)


def _bank(tmp_path, *, r_a=4, r_b=4):
    bank = tmp_path / "bank"
    return {"si_a__lora": _tiny_peft_adapter(bank / "a", r=r_a, seed=1),
            "si_b__lora": _tiny_peft_adapter(bank / "b", r=r_b, seed=2)}


def _native(tmp_path, adapters, **over):
    kw = dict(port=8801, lora_modules=adapters, exact_replay_strategy="serialize",
              chat_template_path=_chat_template(tmp_path), lora_native=True)
    kw.update(over)
    spec = kw.pop("spec", None) or _tpl_spec(serving_launch_flags=[])
    return vs.build_server_launch(spec, **kw)


# --------------------------------------------------------------------------
# compose
# --------------------------------------------------------------------------


def test_lora_native_composes_the_exact_argv_and_env(tmp_path):
    adapters = _bank(tmp_path)
    zero = lx.export_zero_adapter(
        tmp_path / "bank" / "zero", base_model="meta-llama/Llama-3.1-8B-Instruct",
        rank=1, target_modules={"model.layers.0.self_attn.o_proj": (24, 32),
                                "model.layers.1.self_attn.o_proj": (24, 32)})
    adapters["none__lora0"] = zero
    launch = _native(tmp_path, adapters)
    cmd = list(launch.cmd)
    assert cmd[:3] == ["vllm", "serve", "meta-llama/Llama-3.1-8B-Instruct"]
    assert "--no-enforce-eager" in cmd and "--enforce-eager" not in cmd
    assert cmd.count("--no-enforce-eager") == 1
    i = cmd.index("--enable-lora")
    assert cmd[i:i + 9] == ["--enable-lora", "--max-loras", "3",
                            "--max-lora-rank", "8", "--max-cpu-loras", "3",
                            # the ADAPTED CLASS: the one suffix set
                            # every registered adapter adapts, nargs="+" spelled
                            "--lora-target-modules", "o_proj"]
    assert cmd.count("--lora-target-modules") == 1
    assert launch.lora_target_modules == ("o_proj",)
    assert cmd.count("--lora-modules") == 1
    j = cmd.index("--lora-modules")
    assert cmd[j + 1:j + 4] == [f"none__lora0={zero.resolve()}",
                                f"si_a__lora={adapters['si_a__lora'].resolve()}",
                                f"si_b__lora={adapters['si_b__lora'].resolve()}"]
    assert cmd[cmd.index("--worker-extension-cls") + 1] == \
        vs.NATIVE_WORKER_EXTENSION_CLS == nw.WORKER_EXTENSION_CLS
    assert cmd[cmd.index("--middleware") + 1] == vs.AUTH_MIDDLEWARE_CLS
    assert vs.SYNC_SCHEDULING_FLAG in cmd and "--enable-prefix-caching" in cmd
    # env: plugin OUT (set empty, not omitted), the plugin's own disable
    # switch, dev mode for the RPC, the sisteer tree — and NO registry
    assert launch.env == {
        "PYTHONPATH": launch.env["PYTHONPATH"],
        "VLLM_PLUGINS": "", vs.LENS_DISABLE_ENV: "1",
        vs.SERVER_DEV_MODE_ENV: "1"}
    assert (Path(launch.env["PYTHONPATH"]) / "sisteer" / "__init__.py").is_file()
    assert vs.LENS_REGISTRY_DIR_ENV not in launch.env
    # the launch record
    assert launch.lora_native is True
    assert launch.lora_modules == ("none__lora0", "si_a__lora", "si_b__lora")
    assert launch.lora_adapter_dirs == {
        n: str(p.resolve()) for n, p in adapters.items()}
    assert (launch.max_loras, launch.max_lora_rank, launch.max_cpu_loras) == (3, 8, 3)
    assert launch.revision == REV and launch.lens_registry_dir is None
    assert launch.api_key and launch.api_key not in cmd
    # a bare --enable-lora restatement is still the documented duplicate
    again = _native(tmp_path, adapters,
                    spec=_tpl_spec(serving_launch_flags=["--enable-lora"]))
    assert again.cmd.count("--enable-lora") == 1


def _with_q_proj_factors(adapter_dir: Path, *, r=4, layers=(0, 1), d_in=24, d_out=32, seed=0) -> Path:
    """Add q_proj factors to a tiny o_proj adapter's FILE (and name q_proj in
    its config): the class an adapter adapts is read off its factors
    (``lora_export.adapter_target_class``),
    so a two-suffix class has to exist in the safetensors, not just the config."""
    from safetensors.torch import load_file, save_file
    g = torch.Generator().manual_seed(seed + 100)
    t = load_file(str(adapter_dir / lx.ADAPTER_WEIGHTS))
    for layer in layers:
        m = f"base_model.model.model.layers.{layer}.self_attn.q_proj"
        t[f"{m}.lora_A.weight"] = torch.randn(r, d_in, generator=g)
        t[f"{m}.lora_B.weight"] = torch.randn(d_out, r, generator=g) * 0.1
    save_file(t, str(adapter_dir / lx.ADAPTER_WEIGHTS))
    cfg = json.loads((adapter_dir / lx.ADAPTER_CONFIG).read_text())
    cfg["target_modules"] = sorted(set(cfg["target_modules"]) | {"q_proj"})
    (adapter_dir / lx.ADAPTER_CONFIG).write_text(json.dumps(cfg, indent=2, sort_keys=True))
    return adapter_dir


def test_lora_native_composes_the_adapted_class_and_refuses_disagreement(tmp_path):
    """With ``lora_target_modules=None`` vLLM 0.25.1 wraps EVERY linear
    projection (kernels dispatched at gate_up_proj's 20,480-wide packed
    stride under a ceiling sized for o_proj's 4,096), so the launch
    restricts the wrappers to the class the adapters adapt — ONE set, else
    refused naming every adapter's; an adapter whose config states no class
    refuses too. The zero adapter's un-deduplicated ``["o_proj", "o_proj",
    ...]`` is the same class. The class is the FILE's (its factor keys),
    never the config∪file union: a config that declares a module with no
    factor behind it does not widen ``--lora-target-modules``, and a
    disagreement has to be in the factors."""
    adapters = _bank(tmp_path)
    zero = lx.export_zero_adapter(
        tmp_path / "bank" / "zero", base_model="meta-llama/Llama-3.1-8B-Instruct",
        rank=1, target_modules={"model.layers.0.self_attn.o_proj": (24, 32),
                                "model.layers.1.self_attn.o_proj": (24, 32)})
    assert json.loads((zero / lx.ADAPTER_CONFIG).read_text())["target_modules"] == ["o_proj", "o_proj"]
    launch = _native(tmp_path, dict(adapters, none__lora0=zero))
    assert launch.lora_target_modules == ("o_proj",)
    assert list(launch.cmd)[list(launch.cmd).index("--lora-target-modules") + 1] == "o_proj"
    assert vs._adapted_module_class({"a": str(adapters["si_a__lora"]), "z": str(zero)}) == ("o_proj",)
    # a config that DECLARES q_proj while the file carries o_proj factors only is the o_proj class: the
    # launch composes on the file, never on the union (before: `--lora-target-modules o_proj q_proj`, LoRA buffers and
    # kernels at every q_proj wrapper no adapter adapts)
    wide_cfg = _tiny_peft_adapter(tmp_path / "bank" / "widecfg", r=4, seed=5, target_modules=("o_proj", "q_proj"))
    assert vs._adapted_module_class({"w": str(wide_cfg)}) == ("o_proj",)
    launch = _native(tmp_path, dict(adapters, si_wide__lora=wide_cfg))
    assert launch.lora_target_modules == ("o_proj",)
    # two adapters whose FILES adapt different classes cannot share one restriction
    odd = _with_q_proj_factors(_tiny_peft_adapter(tmp_path / "bank" / "odd", r=4, seed=5), seed=5)
    with pytest.raises(vs.VllmServerError,
                       match=r"registered adapters disagree on their adapted module class: "
                             r"\{'si_a__lora': \['o_proj'\], 'si_b__lora': \['o_proj'\], "
                             r"'si_odd__lora': \['o_proj', 'q_proj'\]\}") as err:
        _native(tmp_path, dict(adapters, si_odd__lora=odd))
    assert "one lora-native launch restricts the LoRA execution wrappers to ONE suffix set" in str(err.value)
    # ... and a class of two suffixes (in the FILE) composes both, sorted
    both = {"si_q__lora": _with_q_proj_factors(_tiny_peft_adapter(tmp_path / "bank" / "q", r=4, seed=6), seed=6)}
    launch = _native(tmp_path, both)
    cmd = list(launch.cmd)
    j = cmd.index("--lora-target-modules")
    assert cmd[j + 1:j + 3] == ["o_proj", "q_proj"] and launch.lora_target_modules == ("o_proj", "q_proj")
    # an unstated class refuses at compose time
    for targets, shown in ((), "[]"), (("",), "['']"), ((1,), "[1]"):
        bad = _tiny_peft_adapter(tmp_path / "bank" / "bad", r=4, seed=7, target_modules=targets)
        with pytest.raises(vs.VllmServerError,
                           match=rf"adapter 'si_bad__lora' .*target_modules={re.escape(shown)} is not a "
                                 r"non-empty list of module suffixes"):
            _native(tmp_path, {"si_bad__lora": bad})
    # the golden gemma geometry at D = 4096 still composes at 107 x 4864 — WITH the class
    bank = tmp_path / "wide_bank"
    wide = _tiny_peft_adapter(bank / "wide", r=4, layers=(0,), d_in=4096, d_out=2560, seed=3)
    ok = _native(tmp_path, {"si_w__lora": wide}, max_num_seqs=107, max_model_len=4864)
    assert (ok.lora_max_module_dim, ok.lora_kernel_index_ceiling) == (4096, 524_287)
    assert ok.max_num_batched_tokens == 107 * 4864 == 520_448 <= 524_287
    assert ok.lora_target_modules == ("o_proj",) and "--lora-target-modules" in ok.cmd


def test_lora_native_refusals_name_the_reason(tmp_path):
    adapters = _bank(tmp_path)
    with pytest.raises(vs.VllmServerError, match="plugin forces enforce_eager"):
        _native(tmp_path, adapters, lens_registry_dir=_flat_registry(tmp_path))
    with pytest.raises(vs.VllmServerError, match="no lora_modules"):
        _native(tmp_path, {})
    with pytest.raises(vs.VllmServerError, match="no lora_modules"):
        _native(tmp_path, ())
    # every full22 manifest carries --enforce-eager for the lens; under
    # lora_native NEITHER spelling is a tolerated restatement, and the
    # refusal names the mode and what to do
    with pytest.raises(vs.VllmServerError,
                       match="'--enforce-eager'\\] under lora_native.*RECORD the drop"):
        _native(tmp_path, adapters, spec=_tpl_spec())   # ["--enable-lora", "--enforce-eager"]
    with pytest.raises(vs.VllmServerError, match="'--no-enforce-eager'\\] under lora_native"):
        _native(tmp_path, adapters,
                spec=_tpl_spec(serving_launch_flags=["--no-enforce-eager"]))
    # the newly owned options refuse in every spelling, on every mode —
    # including the deployment-time LoRA RESTRICTION (--lora-target-modules:
    # filters the execution wrappers, not the fingerprinted CPU factors) and
    # the two REAL 0.25.1 capture-surface options, spelled exactly as the
    # pinned parser registers them (engine/arg_utils.py:1478-1482 at the
    # tag: `--cudagraph-capture-sizes`, `--max-cudagraph-capture-size`);
    # `--cuda-graph-sizes` is the misspelling the earlier list carried
    for flags in (["--max-cpu-loras", "4"], ["--max-cpu-loras=4"], ["--max_cpu_loras", "4"],
                  ["--lora-dtype", "float16"], ["--fully-sharded-loras"],
                  ["--no-fully-sharded-loras"], ["--cuda-graph-sizes", "1", "2"],
                  ["--compilation-config", '{"mode": 0}'],
                  ["--lora-target-modules", "o_proj"], ["--lora_target_modules", "o_proj"],
                  ["--lora-target-modules=o_proj"],
                  ["--cudagraph-capture-sizes", "1", "2", "4"],
                  ["--cudagraph_capture_sizes", "1"],
                  ["--max-cudagraph-capture-size", "1"], ["--max-cudagraph-capture-size=1"],
                  ["--max_cudagraph_capture_size", "1"]):
        with pytest.raises(vs.VllmServerError, match="wrapper-owned launch setting"):
            _native(tmp_path, adapters, spec=_tpl_spec(serving_launch_flags=flags))
        with pytest.raises(vs.VllmServerError, match="wrapper-owned launch setting"):
            vs.build_server_launch(
                _tpl_spec(serving_launch_flags=["--enforce-eager"] + flags), port=8801,
                exact_replay_strategy="serialize",
                chat_template_path=_chat_template(tmp_path))
    for owned in ("--lora-target-modules", "--cudagraph-capture-sizes",
                  "--max-cudagraph-capture-size"):
        assert owned in vs._WRAPPER_OWNED_OPTIONS
    # the lens tuple keeps its shape; the native one is empty
    assert vs._PIN_RESTATEMENTS == ("--enforce-eager",)
    assert vs._LORA_NATIVE_PIN_RESTATEMENTS == ()
    # a non-loopback bind refuses like a lens launch
    with pytest.raises(vs.VllmServerError, match="LORA-NATIVE launch bound to"):
        _native(tmp_path, adapters, host="10.0.0.5")


def test_cudagraph_capture_sizes_for_is_vllms_default_list_plus_S():
    """The composed capture list is vLLM's DEFAULT list for S (config/vllm.py
    _set_cudagraph_sizes @0.25.1: [1, 2, 4] + multiples of 8 below 256 +
    multiples of 16 from 256, all <= min(2S, 512)) plus S itself, sorted and
    de-duplicated. KNOWN-BAD first: S=51 — the 2026-09-16 Modal H200 launch —
    is NOT a default size (48 and 56 are), so vLLM captured FULL decode graphs
    only up to 48 and every 49-51 sequence step ran PIECEWISE; the composed
    list contains 51 and every default size below it."""
    d51, m51 = _default_capture_sizes(51)
    assert m51 == 96 and d51[-1] == 96 and 51 not in d51 and 48 in d51 and 56 in d51
    c51 = vs.cudagraph_capture_sizes_for(51)
    assert 51 in c51 and set(x for x in d51 if x < 51) <= set(c51)
    assert c51 == tuple(sorted(set(d51) | {51}))
    assert list(c51) == sorted(set(c51))                       # sorted, no duplicates
    assert c51[c51.index(48):c51.index(48) + 3] == (48, 51, 56)
    # S=107 (the gemma boundary): contains 107, default cap min(214, 512) = 214 -> largest 208
    d107, m107 = _default_capture_sizes(107)
    c107 = vs.cudagraph_capture_sizes_for(107)
    assert 107 in c107 and c107 == tuple(sorted(set(d107) | {107})) and c107[-1] == m107 == 208
    # S=48 is already a default size: membership unchanged (the list IS the default)
    d48, _ = _default_capture_sizes(48)
    assert vs.cudagraph_capture_sizes_for(48) == tuple(d48)
    # S=8 (the fake server's geometry) and S=1 (the serialize default)
    assert vs.cudagraph_capture_sizes_for(8) == (1, 2, 4, 8, 16) == tuple(_default_capture_sizes(8)[0])
    assert vs.cudagraph_capture_sizes_for(1) == (1, 2)
    # above vLLM's 512 default cap the default list stops at 512 and S rides beside it
    c600 = vs.cudagraph_capture_sizes_for(600)
    assert c600[-1] == 600 and 512 in c600 and 528 not in c600 and c600[-2] == 512
    for bad in (0, -1, True, False, "51", 51.0, None):
        with pytest.raises(vs.VllmServerError, match="cudagraph_capture_sizes_for"):
            vs.cudagraph_capture_sizes_for(bad)
    assert vs.VLLM_DEFAULT_CUDAGRAPH_CAPTURE_CAP == 512


def test_lora_native_composes_the_capture_list_with_max_num_seqs_in_it(tmp_path):
    """The lora-native argv carries ONE ``--cudagraph-capture-sizes`` whose
    values are ``cudagraph_capture_sizes_for(S)`` (S itself a member), spelled
    as the pinned parser reads them (nargs="+", type=int — arg_utils.
    collection_to_kwargs for list[int]); no ``--max-cudagraph-capture-size``
    beside it (vLLM raises when the two disagree; it pins the max to the list's
    largest entry itself); the list is recorded on the launch; S is not moved.
    A lens launch (eager) composes nothing here — its argv is untouched."""
    import argparse
    adapters = _bank(tmp_path)
    for S in (51, 107, 48):
        launch = _native(tmp_path, adapters, max_num_seqs=S)
        cmd = list(launch.cmd)
        composed = vs.cudagraph_capture_sizes_for(S)
        assert S in composed and launch.cudagraph_capture_sizes == composed
        assert launch.max_num_seqs == S                            # never clamped or moved
        assert cmd[cmd.index("--max-num-seqs") + 1] == str(S)
        assert cmd.count("--cudagraph-capture-sizes") == 1
        i = cmd.index("--cudagraph-capture-sizes")
        toks = []
        for t in cmd[i + 1:]:
            if t.startswith("-"):
                break
            toks.append(t)
        assert toks == [str(n) for n in composed]
        assert "--max-cudagraph-capture-size" not in cmd and "--compilation-config" not in cmd
        # parses exactly as vLLM's pinned parser registers the option
        ap = argparse.ArgumentParser()
        ap.add_argument("--cudagraph-capture-sizes", type=int, nargs="+")
        ns, _rest = ap.parse_known_args(cmd[i:])
        assert ns.cudagraph_capture_sizes == list(composed)
    # the lens launch: no capture flag, an empty record
    lens = vs.build_server_launch(
        _tpl_spec(serving_launch_flags=["--enforce-eager"]), port=8801,
        lens_registry_dir=_flat_registry(tmp_path), exact_replay_strategy="serialize",
        chat_template_path=_chat_template(tmp_path))
    assert "--cudagraph-capture-sizes" not in lens.cmd and lens.cudagraph_capture_sizes == ()
    # a manifest that tries to hand its own list is still refused as wrapper-owned
    with pytest.raises(vs.VllmServerError, match="wrapper-owned launch setting"):
        _native(tmp_path, adapters, max_num_seqs=51,
                spec=_tpl_spec(serving_launch_flags=["--cudagraph-capture-sizes", "51"]))


def test_build_server_launch_is_the_last_writer_of_max_num_seqs():
    """The guarantee "max_num_seqs is a captured size" holds wherever S is finally
    DECIDED because exactly one function WRITES ``--max-num-seqs`` on every
    standup path — ``build_server_launch`` — and it composes the capture list
    beside it: a budget-derived S (``mt_serving_up.py`` sizes ``geom.max_num_seqs``
    from the per-model batched-token budget when ``f22_standup``'s caller passes
    no width, $6 empty), a geometry-table S and an explicit ``--max-num-seqs``
    all arrive as its ``max_num_seqs`` argument; ``mt_serving_up.launch_cmd_extras``
    appends only ``--gpu-memory-utilization`` / ``--max-logprobs`` /
    ``--enable-log-requests`` after it, and no standup-path module spells the
    argv pair itself."""
    repo = Path(vs.__file__).resolve().parents[4]
    writer = re.compile(r'"--max-num-seqs",\s*str\(')
    assert len(writer.findall((repo / "src/sisteer/capeval/serving/vllm_server.py").read_text())) == 1
    for rel in ("src/sisteer/capeval/serving/native_standup.py",
                "scripts/sdfbase/mt_serving_up.py", "scripts/capctrl_serving_up.py"):
        text = (repo / rel).read_text()
        assert not writer.search(text), rel
        assert '"--cudagraph-capture-sizes"' not in text, rel
    mt = (repo / "scripts/sdfbase/mt_serving_up.py").read_text()
    extras = mt[mt.index("def launch_cmd_extras("):mt.index("\ndef ", mt.index("def launch_cmd_extras(") + 1)]
    assert "max-num-seqs" not in extras and "cudagraph" not in extras
    assert "max_num_seqs=geom.max_num_seqs" in mt                  # the budget-/table-/explicit-derived S, handed over
    assert "vs.build_server_launch(" in mt
    # common.sh hands an explicit width ($6) to mt_serving_up.py, never to vllm directly
    common = (repo / "scripts/sdfbase/full22/common.sh").read_text()
    assert '${gS:+--max-num-seqs "$gS"}' in common and "vllm serve" not in common


def test_cudagraph_coverage_problems_hold_the_full_surface_to_max_num_seqs():
    """The readiness rule behind the composer's list. KNOWN-BAD first: the
    2026-09-16 report — S=51, vLLM's default list (max 96), FULL keys stopping
    at 48 — passes both pre-existing checks (max 96 >= 51, largest 96 >= 51)
    and is refused ONLY by the FULL check, by name; the composed list (51 a
    member, a FULL key at 51) passes; every other deviation refuses by name."""
    sizes, mx = _default_capture_sizes(51)

    def rep(**over):
        r = {"cudagraph_mode": "FULL_AND_PIECEWISE",
             "cudagraph_capture_sizes": list(sizes), "max_cudagraph_capture_size": mx,
             "cudagraph_full_capture_sizes": [x for x in sizes if x <= 51],
             "cudagraph_full_capture_max": 48, "uniform_decode_query_len": 1,
             "model_runner_surface": "v1"}
        r.update(over)
        return r

    probs = sa.cudagraph_coverage_problems([rep()], max_num_seqs=51)
    assert len(probs) == 1, probs
    assert "largest FULL CUDA-graph capture size 48 < the composed max_num_seqs 51" in probs[0]
    assert "49..51 sequences pad to the next PIECEWISE size" in probs[0]
    # the fix: S in the list -> a FULL key at S; the composed entries all resolved
    composed = vs.cudagraph_capture_sizes_for(51)
    good = rep(cudagraph_capture_sizes=list(composed), max_cudagraph_capture_size=96,
               cudagraph_full_capture_sizes=[x for x in composed if x <= 51],
               cudagraph_full_capture_max=51)
    assert sa.cudagraph_coverage_problems([good], max_num_seqs=51,
                                          composed_capture_sizes=composed) == []
    # the same S with a wider FULL surface also passes (>=, not ==)
    assert sa.cudagraph_coverage_problems([good], max_num_seqs=40) == []
    # the engine did not take the composed list: 51 missing from the resolved sizes
    probs = sa.cudagraph_coverage_problems([rep()], max_num_seqs=51, composed_capture_sizes=composed)
    assert any("composed --cudagraph-capture-sizes entries [51] are MISSING" in x for x in probs)
    # plain FULL mode: every capture size has a FULL key, the max check already covers S
    full_mode = rep(cudagraph_mode="FULL", cudagraph_full_capture_sizes=list(sizes),
                    cudagraph_full_capture_max=96)
    assert sa.cudagraph_coverage_problems([full_mode], max_num_seqs=51) == []
    # NO FULL key (PIECEWISE-only): refused by name, the mode named
    probs = sa.cudagraph_coverage_problems(
        [rep(cudagraph_mode="PIECEWISE", cudagraph_full_capture_sizes=[],
             cudagraph_full_capture_max=None)], max_num_seqs=51)
    assert len(probs) == 1 and "NO FULL CUDA-graph capture size" in probs[0] \
        and "'PIECEWISE'" in probs[0]
    # a pre-landing worker extension (no FULL keys reported) refuses — never a quiet pass
    old = rep()
    for k in ("cudagraph_full_capture_sizes", "cudagraph_full_capture_max",
              "uniform_decode_query_len"):
        old.pop(k)
    probs = sa.cudagraph_coverage_problems([old], max_num_seqs=51)
    assert len(probs) == 1 and "MISSING from the resolved report" in probs[0] \
        and "cudagraph_full_capture_max" in probs[0]
    # two readings of one dispatcher disagreeing
    probs = sa.cudagraph_coverage_problems([rep(cudagraph_full_capture_max=40)], max_num_seqs=51)
    assert len(probs) == 1 and "disagrees with cudagraph_full_capture_sizes" in probs[0]
    probs = sa.cudagraph_coverage_problems(
        [rep(cudagraph_full_capture_sizes=[], cudagraph_full_capture_max=48)], max_num_seqs=51)
    assert len(probs) == 1 and "disagrees with" in probs[0]
    # a FULL size the capture list never had
    probs = sa.cudagraph_coverage_problems(
        [rep(cudagraph_full_capture_sizes=[1, 2, 4, 8, 16, 24, 32, 40, 48, 50],
             cudagraph_full_capture_max=50)], max_num_seqs=51)
    assert any("FULL capture sizes [50] are not in cudagraph_capture_sizes" in x for x in probs)
    assert any("largest FULL CUDA-graph capture size 50 < the composed max_num_seqs 51" in x
               for x in probs)
    # speculative decode would make a step S x q tokens: the rule is in tokens
    probs = sa.cudagraph_coverage_problems(
        [rep(uniform_decode_query_len=2, cudagraph_full_capture_sizes=list(sizes),
             cudagraph_full_capture_max=96)], max_num_seqs=51)
    assert len(probs) == 1 and "x uniform_decode_query_len 2 = 102 tokens" in probs[0]
    for bad_q in (0, True, "1", None):
        probs = sa.cudagraph_coverage_problems([rep(uniform_decode_query_len=bad_q)], max_num_seqs=51)
        assert len(probs) == 1 and "uniform_decode_query_len resolved to" in probs[0]
    probs = sa.cudagraph_coverage_problems([rep(cudagraph_full_capture_sizes="48")], max_num_seqs=51)
    assert len(probs) == 1 and "expected a list of ints" in probs[0]
    probs = sa.cudagraph_coverage_problems([rep(cudagraph_full_capture_max="48")], max_num_seqs=51)
    assert len(probs) == 1 and "not an int or None" in probs[0]
    # WHICH runner surface answered (Slurm 521717: gemma runs the V2 runner): unsaid refuses; a surface that is
    # not the attested class's kind refuses; either kind, agreeing, passes
    good_v2 = dict(good, model_runner_surface="v2", model_runner_kind="v2",
                   model_runner_class="vllm.v1.worker.gpu.model_runner.GPUModelRunner")
    assert sa.cudagraph_coverage_problems([good_v2], max_num_seqs=51) == []
    assert sa.cudagraph_coverage_problems([dict(good, model_runner_kind="v1")], max_num_seqs=51) == []
    r = dict(good); r.pop("model_runner_surface")
    probs = sa.cudagraph_coverage_problems([r], max_num_seqs=51)
    assert len(probs) == 1 and "model_runner_surface resolved to None" in probs[0]
    probs = sa.cudagraph_coverage_problems([dict(good_v2, model_runner_surface="v1")], max_num_seqs=51)
    assert len(probs) == 1 and "read off the v1 runner surface but the model runner class is the v2 runner" in probs[0]


def test_fitted_max_lora_rank(tmp_path):
    def d(name, r, **kw):
        return _tiny_peft_adapter(tmp_path / name, r=r, **kw)
    assert vs.fitted_max_lora_rank([d("r4", 4), d("r1", 1)]) == 8
    assert vs.fitted_max_lora_rank([d("r8", 8)]) == 8
    assert vs.fitted_max_lora_rank([d("r9", 9)]) == 16
    assert vs.fitted_max_lora_rank([d("r1b", 1)]) == 1
    assert vs.fitted_max_lora_rank([d("r64", 64), d("r4b", 4)]) == 64
    assert vs.fitted_max_lora_rank([d("r300", 300)]) == 320
    assert vs.LEGAL_LORA_RANK_CAPS == (1, 8, 16, 32, 64, 128, 256, 320, 512)
    with pytest.raises(vs.VllmServerError, match="no adapter_config.json"):
        vs.fitted_max_lora_rank([tmp_path / "nowhere"])
    with pytest.raises(vs.VllmServerError, match="exceeds the largest legal"):
        vs.fitted_max_lora_rank([d("r513", 4, extra_config={"r": 513})])
    with pytest.raises(vs.VllmServerError, match="not a positive int"):
        vs.fitted_max_lora_rank([d("r0", 4, extra_config={"r": 0})])
    with pytest.raises(vs.VllmServerError, match="no adapter directories"):
        vs.fitted_max_lora_rank([])
    # ... and the composer uses it: rank-9 -> cap 16 on argv
    launch = _native(tmp_path, {"a": d("ca", 9), "b": d("cb", 4)})
    assert list(launch.cmd)[list(launch.cmd).index("--max-lora-rank") + 1] == "16"
    assert launch.max_lora_rank == 16


def test_unowned_manifest_flags_survive_under_lora_native(tmp_path):
    """The qwen manifests' --chat-template-content-format and
    --limit-mm-per-prompt (and gemma's) are NOT owned and must pass
    through under lora_native exactly as on the lens;
    the manifest's --enforce-eager is stripped by the standup first."""
    flags = ["--chat-template-content-format", "string",
             "--limit-mm-per-prompt", '{"image":0,"video":0}',
             "--gpu-memory-utilization", "0.9", "--max-logprobs", "20"]
    launch = _native(tmp_path, _bank(tmp_path),
                     spec=_tpl_spec(serving_launch_flags=flags))
    cmd = list(launch.cmd)
    assert cmd[-len(flags):] == flags
    for opt in ("--chat-template-content-format", "--limit-mm-per-prompt",
                "--gpu-memory-utilization", "--max-logprobs"):
        assert opt not in vs._WRAPPER_OWNED_OPTIONS


#: the lens+lora launch the composer at 9486b825 emitted for
#: test_petri_serving_gate's fixtures (captured on the unmodified worktree
#: before the lora-native mode was added; <TMP> = the fixture dir,
#: <SISTEER> = the sisteer tree's parent). Every lens launch must still
#: compose EXACTLY this.
_GOLDEN_LENS_LORA_CMD = [
    "vllm", "serve", "meta-llama/Llama-3.1-8B-Instruct",
    "--revision", "0123456789abcdef0123456789abcdef01234567",
    "--dtype", "bfloat16", "--host", "127.0.0.1", "--port", "8123",
    "--max-model-len", "16384", "--max-num-seqs", "1",
    "--max-num-batched-tokens", "16384",
    "--long-prefill-token-threshold", "0", "--max-num-partial-prefills", "1",
    "--max-long-partial-prefills", "1", "--no-async-scheduling",
    "--enforce-eager", "--enable-prefix-caching", "--no-enable-auto-tool-choice",
    "--chat-template", "<TMP>/chat_template.jinja",
    "--enable-lora", "--max-loras", "2", "--max-lora-rank", "64",
    "--lora-modules", "cellA=<TMP>/adapters/cellA", "cellB=<TMP>/adapters/cellB",
    "--worker-extension-cls",
    "sisteer.capeval.serving.lens_worker.SisteerWorkerExtension",
    "--middleware", "sisteer.capeval.serving.vllm_auth.SisteerTokenAuthMiddleware",
]
_GOLDEN_LENS_LORA_ENV = {"PYTHONPATH": "<SISTEER>", "VLLM_PLUGINS": "activations",
                         "PETRI_LENS_REGISTRY_DIR": "<TMP>/reg",
                         "VLLM_SERVER_DEV_MODE": "1"}


def test_legacy_lens_lora_launch_is_byte_identical_to_the_golden(tmp_path):
    import sisteer

    reg = _flat_registry(tmp_path)
    tpl = _chat_template(tmp_path)
    a = tmp_path / "adapters" / "cellA"
    b = tmp_path / "adapters" / "cellB"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    launch = vs.build_server_launch(
        _tpl_spec(), port=8123, lens_registry_dir=reg,
        lora_modules={"cellB": b, "cellA": a},        # unsorted on purpose
        exact_replay_strategy="serialize", chat_template_path=tpl)
    sis = str(Path(sisteer.__file__).resolve().parents[1])
    cmd = [c.replace(str(tmp_path), "<TMP>") for c in launch.cmd]
    env = {k: v.replace(str(tmp_path), "<TMP>").replace(sis, "<SISTEER>")
           for k, v in launch.env.items()}
    assert cmd == _GOLDEN_LENS_LORA_CMD
    assert env == _GOLDEN_LENS_LORA_ENV
    # the legacy path records its literal cap and composes no CPU count
    assert launch.lora_native is False
    assert launch.lora_modules == ("cellA", "cellB")
    assert (launch.max_loras, launch.max_lora_rank, launch.max_cpu_loras) == (2, 64, None)
    # ... and a plain launch (no registry, no adapters) is the golden minus
    # the lora and lens blocks
    plain = vs.build_server_launch(
        _tpl_spec(serving_launch_flags=["--enforce-eager"]), port=8123,
        exact_replay_strategy="serialize", chat_template_path=tpl)
    pcmd = [c.replace(str(tmp_path), "<TMP>") for c in plain.cmd]
    assert pcmd == _GOLDEN_LENS_LORA_CMD[:_GOLDEN_LENS_LORA_CMD.index("--enable-lora")]
    assert plain.env == {}


# --------------------------------------------------------------------------
# expectations
# --------------------------------------------------------------------------


def test_expected_scheduler_state_gains_the_lora_native_keys(tmp_path):
    launch = _native(tmp_path, _bank(tmp_path), max_num_seqs=8)
    exp = vs.expected_scheduler_state(launch)
    assert exp["enforce_eager"] is False
    assert exp["cudagraph_mode_is_none"] is False
    assert exp["lens_plugin_imported"] is False
    assert (exp["lora_max_loras"], exp["lora_max_lora_rank"],
            exp["lora_max_cpu_loras"], exp["lora_loaded_count"]) == (2, 8, 2, 2)
    # the composed adapted class, asserted EQUAL — no longer None
    assert exp["lora_target_modules"] == ["o_proj"]
    assert "cudagraph_mode" not in exp and "lora_dtype" not in exp  # recorded only
    # the equality validator refuses None (every projection wrapped) and a
    # wider or different resolved class, by name
    for got, shown in ((None, "None"), (["o_proj", "gate_up_proj"], "['o_proj', 'gate_up_proj']"),
                       (["qkv_proj"], "['qkv_proj']")):
        rep = _native_sched_report(vs.expected_native_lora_state(launch), lora_target_modules=got)
        with pytest.raises(vs.VllmServerError,
                           match=rf"lora_target_modules resolved to {re.escape(shown)}, composed \['o_proj'\]"):
            vs.validate_scheduler_reports([rep], exp)
    # a lora-native launch without the class refuses the expectation
    unclassed = dataclasses.replace(launch, lora_target_modules=())
    with pytest.raises(vs.VllmServerError, match="no composed --lora-target-modules set"):
        vs.expected_scheduler_state(unclassed)
    # the capture surface is held by validate_cudagraph_coverage (>=), not
    # by the equality validator: neither key is in the expectation
    assert "max_cudagraph_capture_size" not in exp and "cudagraph_capture_sizes" not in exp
    assert exp["max_num_batched_tokens"] == 8 * 16384 and exp["enable_prefix_caching"] is True
    # a lens launch's expectation is unchanged: none of the new keys
    lens = vs.build_server_launch(
        _tpl_spec(serving_launch_flags=["--enforce-eager"]), port=8801,
        lens_registry_dir=_flat_registry(tmp_path), exact_replay_strategy="serialize",
        chat_template_path=_chat_template(tmp_path))
    assert not any(k.startswith(("lora", "enforce", "cudagraph", "lens"))
                   for k in vs.expected_scheduler_state(lens))
    # a hand-built lora-native launch without the LoRA config refuses
    bare = vs.ServerLaunch(cmd=("x",), env={}, host="127.0.0.1", port=1, model="m",
                           max_num_seqs=8, max_num_batched_tokens=8 * 16384,
                           max_model_len=16384, prefix_caching=True, lora_native=True)
    with pytest.raises(vs.VllmServerError, match="no composed LoRA config"):
        vs.expected_scheduler_state(bare)
    with pytest.raises(vs.VllmServerError, match="not lora_native"):
        vs.expected_native_lora_state(lens)
    # a directory that cannot be fingerprinted refuses the expectation
    scaled = {"si_s__lora": _tiny_peft_adapter(tmp_path / "s", r=4, lora_alpha=8)}
    with pytest.raises(vs.VllmServerError, match="non-unit LoRA scaling"):
        vs.expected_native_lora_state(_native(tmp_path, scaled))


def test_a_fingerprint_contract_refusal_never_spawns_the_child(tmp_path, monkeypatch):
    """The per-name expectation (the adapter fingerprints off the
    directories) is computed BEFORE ``Popen``: a launch whose adapter
    declares ``lora_alpha != r`` refuses with the contract's words and no
    child is ever started (the readiness path used to spawn first and
    refuse only after the RPC), and neither attestation file exists."""
    scaled = {"si_s__lora": _tiny_peft_adapter(tmp_path / "s", r=4, lora_alpha=8)}
    launch = _fake_lora_native_launch(tmp_path, adapters=scaled, scheduler_results=[])

    def boom(*a, **kw):
        raise AssertionError("Popen must not be called for a launch the fingerprint "
                             "contract refuses")
    monkeypatch.setattr(vs.subprocess, "Popen", boom)
    with pytest.raises(vs.VllmServerError, match="non-unit LoRA scaling"):
        vs.launch_server(launch, log_path=tmp_path / "srv.log",
                         ready_timeout_s=5, install_timeout_s=5)
    assert not (tmp_path / vs.SERVING_ATTESTATION_FILE).exists()
    assert not (tmp_path / vs.STEERING_ATTESTATION_FILE).exists()


# --------------------------------------------------------------------------
# readiness against the fake server
# --------------------------------------------------------------------------


def _port_free(port):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                time.sleep(0.2)
        except OSError:
            return True
    return False


def _launch(tmp_path, launch):
    return vs.launch_server(launch, log_path=tmp_path / "srv.log",
                            ready_timeout_s=30, install_timeout_s=10)


def test_lora_native_readiness_writes_the_attestation(tmp_path):
    adapters = _bank(tmp_path)
    launch = _fake_lora_native_launch(tmp_path, adapters=adapters)
    proc = _launch(tmp_path, launch)
    try:
        path = tmp_path / vs.SERVING_ATTESTATION_FILE
        text = path.read_text()
        att = sa.read_serving_attestation(path)   # the STRICT reader accepts it
        assert not (tmp_path / vs.STEERING_ATTESTATION_FILE).exists()
        assert att["kind"] == "lora_native" and att["schema_version"] == 2   # native-psr-serving: v2
        assert att[sa.RESIDUAL_SLOT_KEY]["armed"] is False                     # a LoRA-only launch
        assert att["adapters"]["si_a__lora"]["residual_fingerprint"] is None
        assert att["adapters"]["si_a__lora"]["position_mode"] is None
        assert att["model"] == launch.model and att["revision"] == REV
        assert att["server_pid"] == proc.pid and att["port"] == launch.port
        assert len(att["nonce"]) == 32 and "registry_index_sha16" not in att
        assert att["expected"] == vs.expected_scheduler_state(launch)
        exp = vs.expected_native_lora_state(launch)
        assert att["expected_native_lora"] == exp
        assert att["id_name_map"] == {"1": "si_a__lora", "2": "si_b__lora"}
        assert att["name_binding"]["asserted"] is True
        assert "lora_id_counter" in att["name_binding"]["basis"]
        # the fake reports vLLM's default capture surface for S = 8
        # (min(2S, 512) = 16): both RESOLVED values are on the record
        assert att["engine"] == {"enforce_eager": False,
                                 "cudagraph_mode": "FULL_AND_PIECEWISE",
                                 "cudagraph_specialize_lora": True,
                                 "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
                                 "max_cudagraph_capture_size": 16,
                                 "lens_plugin_imported": False,
                                 "enable_prefix_caching": True,
                                 # the FULL surface (2026-09-16): the dispatcher's
                                 # FULL keys stop at S = 8
                                 "cudagraph_full_capture_sizes": [1, 2, 4, 8],
                                 "cudagraph_full_capture_max": 8}
        assert set(att["engine"]) == set(sa.ENGINE_KEYS) | set(sa.ENGINE_CAPTURE_COVERAGE_KEYS)
        # the composed capture list rode the launch and every entry resolved
        assert launch.cudagraph_capture_sizes == (1, 2, 4, 8, 16)
        assert att["lora_config"] == {"max_loras": 2, "max_lora_rank": 8,
                                      "max_cpu_loras": 2,
                                      "lora_dtype": "torch.bfloat16",
                                      "fully_sharded_loras": False,
                                      "target_modules": ["o_proj"]}
        # the adapted class and the executing strides are on the record
        #, consistent with the ceiling's D
        assert att[sa.LORA_TARGET_MODULES_KEY] == ["o_proj"]
        assert att["execution_modules"]["strides"] == {
            "model.layers.0.self_attn.o_proj": {"in": 24, "out": 32},
            "model.layers.1.self_attn.o_proj": {"in": 24, "out": 32}}
        assert att["execution_modules"]["stride_max"] == 32 == att[sa.LORA_MAX_MODULE_DIM_KEY]
        assert att[sa.LORA_KERNEL_CEILING_KEY] == (2**31 - 1) // 32
        # the EXECUTION set: every fingerprinted module carries a wrapper
        assert set(att["execution_modules"]) == set(sa.EXECUTION_MODULES_KEYS)
        assert att["execution_modules"]["module_keys"] == sorted(
            {k for a in exp["adapters"].values() for k in a["module_keys"]})
        assert att["execution_modules"]["n_modules"] == 2
        assert att["execution_modules"]["other"] == []
        # ... and its EXACT names: every fingerprinted factor is bound by
        # the module of its own exact name
        assert att["execution_modules"]["module_names"] == [
            "model.layers.0.self_attn.o_proj", "model.layers.1.self_attn.o_proj"]
        for w in att["worker_adapters"][0]:
            assert w["execution_bindings"] == {
                m: m for m in att["execution_modules"]["module_names"]}
        for name, block in att["adapters"].items():
            d = Path(adapters[name]).resolve()
            assert block["dir"] == str(d) == block["models_root"]
            assert block["config_sha256"] == sa.sha256_file(d / lx.ADAPTER_CONFIG)
            assert block["weights_sha256"] == sa.sha256_file(d / lx.ADAPTER_WEIGHTS)
            assert block["fingerprint_bf16"] == exp["adapters"][name]["fingerprint_bf16"]
            assert block["lora_int_id"] == exp["adapters"][name]["lora_int_id"]
            assert block["r"] == 4 and block["petri_zero_adapter"] is False
            assert len(block["petri_intervention_hash"]) == 16
        assert att["scheduler_geometry"]["rpc_method"] == vs.SCHEDULER_RPC_METHOD
        assert att["scheduler_geometry"]["n_workers"] == 1
        assert [a["id"] for a in att["worker_adapters"][0]] == [1, 2]
        assert set(att["worker_adapters"][0][0]) == set(sa.WORKER_ADAPTER_KEYS)
        assert launch.api_key not in text
        # the sub-shape the proxy is handed
        ident = sa.engine_identity(att)
        assert ident["kind"] == "lora_native" and ident["enforce_eager"] is False
        assert ident["cudagraph_mode"] == "FULL_AND_PIECEWISE"
        assert set(ident["adapters"]["si_a__lora"]) == {
            "config_sha256", "weights_sha256", "fingerprint_bf16", "lora_int_id",
            "petri_intervention_hash", "petri_zero_adapter",
            "petri_source_kind", "residual_fingerprint", "position_mode"}   # native-psr-serving
        assert ident["residual_slot"] == {"armed": False, "n_workers": 1}   # the attested worker count travels
        assert ident["adapters"]["si_a__lora"]["petri_intervention_hash"] == \
            att["adapters"]["si_a__lora"]["petri_intervention_hash"]
        assert ident["adapters"]["si_a__lora"]["petri_zero_adapter"] is False
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_lora_native_readiness_records_but_does_not_assert_mode_and_dtype(tmp_path):
    """`cudagraph_mode` (vLLM resolves FULL / PIECEWISE / FULL_AND_PIECEWISE
    per platform) and the resolved `lora_dtype` string are RECORDED, not
    asserted — a non-NONE mode with a different name still attests."""
    adapters = _bank(tmp_path)
    names = tuple(sorted(adapters))
    dirs = {n: str(Path(p).resolve()) for n, p in adapters.items()}
    exp = sa.expected_native_lora_state_for(names, dirs)
    launch = _fake_lora_native_launch(
        tmp_path, adapters=adapters,
        scheduler_results=[_native_sched_report(exp, cudagraph_mode="FULL_DECODE_ONLY",
                                                lora_dtype="bfloat16")])
    proc = _launch(tmp_path, launch)
    try:
        att = sa.read_serving_attestation(tmp_path / vs.SERVING_ATTESTATION_FILE)
        assert att["engine"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
        assert att["lora_config"]["lora_dtype"] == "bfloat16"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def _perturbations(tmp_path, adapters):
    """(label, scheduler_results | None, models_data | None, match) — each
    ONE deviation from a healthy server that must refuse readiness."""
    names = tuple(sorted(adapters))
    dirs = {n: str(Path(p).resolve()) for n, p in adapters.items()}
    exp = sa.expected_native_lora_state_for(names, dirs)
    model = "meta-llama/Llama-3.1-8B-Instruct"
    healthy_models = [{"id": model}] + [{"id": n, "root": dirs[n]} for n in names]

    def rep(**over):
        return _native_sched_report(exp, **over)

    def rep_adapters(mutate):
        r = rep()
        mutate(r["lora_adapters"])
        return r

    def wider_wrapper(module, d_in, d_out):
        """The healthy report plus ONE extra or widened execution wrapper."""
        r = rep()
        if module not in r["lora_execution_module_names"]:
            r["lora_execution_modules_other"] = [module]
        r["lora_execution_module_strides"][module] = {"in": d_in, "out": d_out}
        r["lora_execution_module_stride_max"] = max(
            max(v["in"], v["out"]) for v in r["lora_execution_module_strides"].values())
        return r

    def swap_fp(ads):
        ads[0]["fingerprint_bf16"], ads[1]["fingerprint_bf16"] = \
            ads[1]["fingerprint_bf16"], ads[0]["fingerprint_bf16"]

    def wrong_fp(ads):
        ads[0]["fingerprint_bf16"] = "0" * 64

    def f16(ads):
        ads[1]["tensor_dtype"] = "torch.float16"

    def drop_one(ads):
        del ads[1]

    a, b = names
    return [
        ("eager", [rep(enforce_eager=True)], None, "enforce_eager resolved to True"),
        ("cudagraph none", [rep(cudagraph_mode="NONE", cudagraph_mode_is_none=True)],
         None, "cudagraph_mode_is_none resolved to True"),
        ("plugin imported", [rep(lens_plugin_imported=True)], None, "lens_plugin_imported"),
        ("count mismatch", [rep(lora_loaded_count=1)], None, "lora_loaded_count"),
        ("one adapter missing", [rep_adapters(drop_one)], None,
         "ids \\[1\\] != the expected \\[1, 2\\]"),
        ("fingerprint mismatch", [rep_adapters(wrong_fp)], None,
         f"adapter '{a}' \\(lora_int_id 1\\) reports fingerprint_bf16 '0{{64}}'"),
        ("permuted ids", [rep_adapters(swap_fp)], None,
         f"expected for adapter '{b}' \\(lora_int_id 2\\)"),
        ("non-bf16 dtype", [rep_adapters(f16)], None, "tensor_dtype 'torch.float16'"),
        ("lora config drift", [rep(lora_max_lora_rank=64)], None,
         "lora_max_lora_rank resolved to 64, composed 8"),
        ("clamped geometry", [rep(max_num_batched_tokens=8192)], None, "silent clamp"),
        ("kv shortfall", [rep(kv_num_gpu_blocks=4096)], None, "KV cache holds"),
        ("swapped roots", None,
         [{"id": model}, {"id": a, "root": dirs[b]}, {"id": b, "root": dirs[a]}],
         "binding does not match"),
        ("root elsewhere", None,
         [{"id": model}, {"id": a, "root": str(tmp_path / "elsewhere")},
          {"id": b, "root": dirs[b]}],
         f"adapter '{a}': /v1/models root"),
        ("missing card", None, [{"id": model}, {"id": a, "root": dirs[a]}],
         f"adapter '{b}' listed 0 times"),
        ("duplicated card", None, healthy_models + [{"id": b, "root": dirs[b]}],
         f"adapter '{b}' listed 2 times"),
        ("no root", None, [{"id": model}, {"id": a}, {"id": b, "root": dirs[b]}],
         "carries no root"),
        # code-review findings (2026-09-11): the EXECUTION set excludes a
        # fingerprinted layer (a --lora-target-modules restriction filters
        # the wrappers, not the CPU factors the fingerprint hashes) ...
        ("execution set excludes a layer",
         [rep(lora_execution_module_keys=list(exp["adapters"][a]["module_keys"])[:-1])],
         None, f"adapter '{a}' \\(lora_int_id 1\\): 1 of 2 fingerprinted module\\(s\\) "
               "\\['L1.self_attn.o_proj'\\] have NO LoRA execution layer"),
        # ... the resolved restriction is NOT the composed adapted class
        # (None means every projection wrapped) ...
        ("target modules unrestricted", [rep(lora_target_modules=None)], None,
         "lora_target_modules resolved to None, composed \\['o_proj'\\]"),
        # ... a wrapper OUTSIDE the class (gemma's packed gate_up_proj at its
        # 20,480-wide stride) executes under a ceiling sized for o_proj ...
        ("wrapper outside the adapted class",
         [wider_wrapper("model.layers.0.mlp.gate_up_proj", 2560, 20480)], None,
         "carry a LoRA wrapper whose suffix is NOT in the composed --lora-target-modules "
         "\\['o_proj'\\]: \\['model.layers.0.mlp.gate_up_proj'\\]"),
        # ... and a wrapper inside the class but wider than D
        ("wrapper wider than D",
         [wider_wrapper("model.layers.1.self_attn.o_proj", 4096, 32)], None,
         "execution module 'model.layers.1.self_attn.o_proj' carries a LoRA wrapper with stride "
         "4,096 \\(in 4,096 x out 32\\) > the composed lora_max_module_dim 32"),
        # ... and a capture max below max_num_seqs: graphs-on on the record,
        # eager for every full-width decode batch in production
        ("capture max below max_num_seqs",
         [rep(cudagraph_capture_sizes=[1, 2, 4], max_cudagraph_capture_size=4)], None,
         "max_cudagraph_capture_size 4 < the composed max_num_seqs 8"),
        ("capture list stops short of the max",
         [rep(cudagraph_capture_sizes=[1, 2, 4], max_cudagraph_capture_size=16)], None,
         "largest cudagraph_capture_size 4 < the composed max_num_seqs 8"),
        # ... the FULL surface stopping below max_num_seqs while the capture list
        # and its max read fine (2026-09-16: S=51 on the default list, FULL
        # largest 48 — 49-51 sequence steps PIECEWISE with attention eager) ...
        ("FULL capture stops below max_num_seqs",
         [rep(cudagraph_full_capture_sizes=[1, 2, 4], cudagraph_full_capture_max=4)], None,
         "largest FULL CUDA-graph capture size 4 < the composed max_num_seqs 8"),
        # ... no FULL key at all (a PIECEWISE-only engine under a graphs-on record) ...
        ("no FULL capture at all",
         [rep(cudagraph_full_capture_sizes=[], cudagraph_full_capture_max=None)], None,
         "NO FULL CUDA-graph capture size"),
        # ... and a resolved list that dropped a composed entry (the fake launch composes
        # cudagraph_capture_sizes_for(8) = [1, 2, 4, 8, 16])
        ("resolved list lacks a composed size",
         [rep(cudagraph_capture_sizes=[1, 2, 4, 8], max_cudagraph_capture_size=8)], None,
         "composed --cudagraph-capture-sizes entries \\[16\\] are MISSING"),
    ]


@pytest.mark.parametrize("case", range(25))
def test_lora_native_readiness_refuses_each_deviation(tmp_path, case):
    adapters = _bank(tmp_path)
    label, sched, models, match = _perturbations(tmp_path, adapters)[case]
    launch = _fake_lora_native_launch(tmp_path, adapters=adapters,
                                      scheduler_results=sched, models_data=models)
    with pytest.raises(vs.VllmServerError, match=match):
        _launch(tmp_path, launch)
    assert not (tmp_path / vs.SERVING_ATTESTATION_FILE).exists(), label
    assert not (tmp_path / vs.STEERING_ATTESTATION_FILE).exists(), label
    assert _port_free(launch.port), f"{label}: child not shut down"


def test_lora_native_readiness_refuses_without_the_rpc_route_or_token(tmp_path):
    adapters = _bank(tmp_path)
    launch = _fake_lora_native_launch(tmp_path, adapters=adapters, mount_rpc=False)
    with pytest.raises(vs.VllmServerError, match="not mounted"):
        _launch(tmp_path, launch)
    assert not (tmp_path / vs.SERVING_ATTESTATION_FILE).exists()
    assert _port_free(launch.port)
    launch = _fake_lora_native_launch(tmp_path, adapters=adapters, api_key="")
    with pytest.raises(vs.VllmServerError, match="lora-native launch carries no api_key"):
        _launch(tmp_path, launch)


def test_lora_native_readiness_refuses_a_dying_child(tmp_path):
    adapters = _bank(tmp_path)
    launch = _fake_lora_native_launch(tmp_path, adapters=adapters, die_after_rpc=True)
    with pytest.raises(vs.VllmServerError,
                       match="could not be reached|after the lora-native attestation"):
        _launch(tmp_path, launch)
    assert not (tmp_path / vs.SERVING_ATTESTATION_FILE).exists()


def test_lora_native_readiness_clears_a_stale_serving_attestation(tmp_path):
    adapters = _bank(tmp_path)
    stale = tmp_path / vs.SERVING_ATTESTATION_FILE
    stale.write_text(json.dumps({"kind": "lora_native", "nonce": "stale"}))
    launch = _fake_lora_native_launch(tmp_path, adapters=adapters,
                                      scheduler_results=[_native_sched_report(
                                          sa.expected_native_lora_state_for(
                                              tuple(sorted(adapters)),
                                              {n: str(p.resolve()) for n, p in adapters.items()}),
                                          enforce_eager=True)])
    with pytest.raises(vs.VllmServerError):
        _launch(tmp_path, launch)
    assert not stale.exists()


# --------------------------------------------------------------------------
# the strict reader
# --------------------------------------------------------------------------


def test_read_serving_attestation_is_strict(tmp_path):
    adapters = _bank(tmp_path)
    launch = _fake_lora_native_launch(tmp_path, adapters=adapters)
    proc = _launch(tmp_path, launch)
    try:
        path = tmp_path / vs.SERVING_ATTESTATION_FILE
        good = json.loads(path.read_text())
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    def write(mutate):
        att = json.loads(json.dumps(good))
        mutate(att)
        p = tmp_path / "att.json"
        p.write_text(json.dumps(att))
        return p

    assert sa.read_serving_attestation(write(lambda a: None))["kind"] == "lora_native"
    with pytest.raises(sa.ServingAttestationError, match="kind='steering'"):
        sa.read_serving_attestation(write(lambda a: a.__setitem__("kind", "steering")))
    with pytest.raises(sa.ServingAttestationError, match="asserted is not true"):
        sa.read_serving_attestation(write(lambda a: a["name_binding"].__setitem__("asserted", False)))
    with pytest.raises(sa.ServingAttestationError, match="lacks required keys \\['id_name_map'\\]"):
        sa.read_serving_attestation(write(lambda a: a.pop("id_name_map")))
    with pytest.raises(sa.ServingAttestationError, match="enforce_eager=True"):
        sa.read_serving_attestation(write(lambda a: a["engine"].__setitem__("enforce_eager", True)))
    with pytest.raises(sa.ServingAttestationError, match="id_name_map .* disagrees"):
        sa.read_serving_attestation(write(lambda a: a["id_name_map"].__setitem__("1", "si_b__lora")))
    with pytest.raises(sa.ServingAttestationError, match="schema_version=3"):
        sa.read_serving_attestation(write(lambda a: a.__setitem__("schema_version", 3)))
    # native-psr-serving: a schema-2 record without its residual slot block refuses; a schema-1
    # record (every certification before) reads without one
    with pytest.raises(sa.ServingAttestationError, match="lacks the residual_slot block"):
        sa.read_serving_attestation(write(lambda a: a.pop(sa.RESIDUAL_SLOT_KEY)))

    def _as_v1(a):
        a["schema_version"] = 1
        a.pop(sa.RESIDUAL_SLOT_KEY)
        for b in a["adapters"].values():
            for k in ("petri_source_kind", "residual_fingerprint", "position_mode"):
                b.pop(k)
    v1 = sa.read_serving_attestation(write(_as_v1))
    assert v1["schema_version"] == 1 and sa.RESIDUAL_SLOT_KEY not in v1
    assert sa.engine_identity(v1)["residual_slot"] == {"armed": False}
    assert sa.engine_identity(v1)["adapters"]["si_a__lora"]["residual_fingerprint"] is None
    with pytest.raises(sa.ServingAttestationError, match="no serving attestation"):
        sa.read_serving_attestation(tmp_path / "missing.json")
    # the FULL capture surface (2026-09-16) is ADDITIVE: a record written before the
    # keys existed reads unchanged; half a surface or a malformed value refuses
    assert set(sa.ENGINE_CAPTURE_COVERAGE_KEYS) <= set(good["engine"])

    def _pre_landing(a):
        for k in sa.ENGINE_CAPTURE_COVERAGE_KEYS:
            a["engine"].pop(k)
    pre = sa.read_serving_attestation(write(_pre_landing))
    assert not (set(sa.ENGINE_CAPTURE_COVERAGE_KEYS) & set(pre["engine"]))
    assert sa.engine_identity(pre) == sa.engine_identity(good)      # not routing identity
    with pytest.raises(sa.ServingAttestationError, match="half a FULL capture surface"):
        sa.read_serving_attestation(write(lambda a: a["engine"].pop("cudagraph_full_capture_max")))
    with pytest.raises(sa.ServingAttestationError, match="is not a list of ints"):
        sa.read_serving_attestation(write(lambda a: a["engine"].__setitem__(
            "cudagraph_full_capture_sizes", "8")))
    with pytest.raises(sa.ServingAttestationError, match="is not an int or None"):
        sa.read_serving_attestation(write(lambda a: a["engine"].__setitem__(
            "cudagraph_full_capture_max", "8")))
    with pytest.raises(sa.ServingAttestationError, match="disagrees with"):
        sa.read_serving_attestation(write(lambda a: a["engine"].__setitem__(
            "cudagraph_full_capture_max", 4)))
    # a lens steering attestation handed to the strict reader refuses
    lens_like = tmp_path / "steering_attestation.json"
    lens_like.write_text(json.dumps({"registry_dir": "x", "rpc_method": "sisteer_install_steering",
                                     "n_workers": 1, "worker_reports": [], "nonce": "a" * 32}))
    with pytest.raises(sa.ServingAttestationError, match="lacks required keys"):
        sa.read_serving_attestation(lens_like)
    assert vs.SERVING_ATTESTATION_FILE == sa.SERVING_ATTESTATION_FILE == "serving_attestation.json"
    assert vs.SERVING_ATTESTATION_FILE != vs.STEERING_ATTESTATION_FILE


# --------------------------------------------------------------------------
# the LoRA kernel index ceiling (gemma canary job 488932)
# --------------------------------------------------------------------------


def test_lora_native_refuses_a_geometry_over_the_lora_kernel_index_ceiling(tmp_path):
    """Job 488932: gemma-4-E4B at the table's 121 x 4864 = 588,544 batched
    tokens died at warmup with an illegal memory access — the pinned LoRA
    Triton kernels index the full-budget dummy batch with int32 offsets and
    588,544 x 4096 (the global-attention o_proj width) is 2.41e9 > 2^31-1.
    The composer now REFUSES such a geometry by name (never clamps); the
    boundary composes (107 x 4864 = 520,448 <= 524,287) and one more
    sequence refuses; the ceiling and D ride on the launch."""
    bank = tmp_path / "bank"
    # gemma-4-E4B's global-attention o_proj (8 x 512 -> 2560) beside a local one (2048 -> 2560)
    wide = _tiny_peft_adapter(bank / "wide", r=4, layers=(0,), d_in=4096, d_out=2560, seed=3)
    narrow = _tiny_peft_adapter(bank / "narrow", r=4, layers=(1,), d_in=2048, d_out=2560, seed=4)
    adapters = {"si_wide__lora": wide, "si_narrow__lora": narrow}
    with pytest.raises(vs.VllmServerError) as ei:
        _native(tmp_path, adapters, max_num_seqs=121, max_model_len=4864)
    msg = str(ei.value)
    assert "refusing to compose a LORA-NATIVE launch at max_num_seqs=121 x max_model_len=4864 = 588,544 batched tokens" in msg
    assert "widest adapted module of the registered adapters has 4096 in/out features (si_wide__lora:" in msg
    assert "588,544 x 4096 = 2,410,676,224 elements > 2^31-1 = 2,147,483,647" in msg
    assert "lora_shrink_op.py, lora_expand_op.py" in msg and "int32" in msg
    assert "illegal memory access" in msg and "kernel_warmup -> flashinfer_autotune -> _dummy_run" in msg
    assert "job 488932" in msg
    assert "lora_kernel_index_ceiling is 524,287 tokens ((2^31-1) // 4096)" in msg
    assert "largest legal max_num_seqs at max_model_len=4864 is 107" in msg
    assert "geometry_override_reason='lora_kernel_index_ceiling'" in msg and "never clamps" in msg
    # the boundary: 107 x 4864 = 520,448 composes; 108 x 4864 = 525,312 refuses
    ok = _native(tmp_path, adapters, max_num_seqs=107, max_model_len=4864)
    assert ok.max_num_batched_tokens == 520_448
    assert (ok.lora_max_module_dim, ok.lora_kernel_index_ceiling) == (4096, 524_287)
    assert "--max-num-batched-tokens" in ok.cmd or ok.max_num_batched_tokens == 107 * 4864
    with pytest.raises(vs.VllmServerError, match="= 525,312 batched tokens"):
        _native(tmp_path, adapters, max_num_seqs=108, max_model_len=4864)
    # the leaf's arithmetic, pinned for the three certification targets at L=4864
    assert sa.LORA_KERNEL_INDEX_LIMIT == 2**31 - 1
    assert sa.lora_kernel_index_ceiling(4096) == 524_287 and 524_287 // 4864 == 107      # gemma (121 -> 107), qwen (148 -> 107)
    assert sa.lora_kernel_index_ceiling(5120) == 419_430 and 419_430 // 4864 == 86       # nemotron (161 -> 86)
    assert 107 * 4864 == 520_448 and 86 * 4864 == 418_304
    with pytest.raises(sa.ServingAttestationError, match="positive int"):
        sa.lora_kernel_index_ceiling(0)
    # the header-only dims reader and the widest-module pick
    assert sa.adapter_module_dims(wide) == {"base_model.model.model.layers.0.self_attn.o_proj": (4096, 2560)}
    assert sa.adapter_module_dims(narrow) == {"base_model.model.model.layers.1.self_attn.o_proj": (2048, 2560)}
    dims = {"si_wide__lora": sa.adapter_module_dims(wide), "si_narrow__lora": sa.adapter_module_dims(narrow)}
    assert sa.max_module_dim(dims) == (4096, "si_wide__lora:base_model.model.model.layers.0.self_attn.o_proj")
    # a tiny bank (24 x 32) composes far under the ceiling, as every existing test does
    small = _native(tmp_path, _bank(tmp_path))
    assert small.lora_max_module_dim == 32 and small.lora_kernel_index_ceiling == (2**31 - 1) // 32


def test_the_ceiling_applies_only_under_lora_native(tmp_path):
    """A LENS(+lora) launch composes no ceiling — those kernels never run
    there — and its argv is the golden one (the byte-identity test above)."""
    reg = _flat_registry(tmp_path)
    a = tmp_path / "adapters" / "cellA"
    a.mkdir(parents=True)
    launch = vs.build_server_launch(
        _tpl_spec(), port=8123, lens_registry_dir=reg, lora_modules={"cellA": a},
        exact_replay_strategy="serialize", chat_template_path=_chat_template(tmp_path))
    assert launch.lora_native is False
    assert launch.lora_max_module_dim is None and launch.lora_kernel_index_ceiling is None
    plain = vs.build_server_launch(_tpl_spec(serving_launch_flags=["--enforce-eager"]), port=8123,
                                   exact_replay_strategy="serialize", chat_template_path=_chat_template(tmp_path),
                                   max_num_seqs=121, max_model_len=4864)
    assert plain.max_num_batched_tokens == 588_544 and plain.lora_kernel_index_ceiling is None
    # the attestation's additive keys: absent on a hand-built launch, refused when inconsistent
    att = {"lora_kernel_index_ceiling": 524_287, "lora_max_module_dim": 4096}
    sa._validate_kernel_ceiling_keys(att["lora_kernel_index_ceiling"], att["lora_max_module_dim"], where="t")
    with pytest.raises(sa.ServingAttestationError, match="!= \\(2\\^31-1\\) //"):
        sa._validate_kernel_ceiling_keys(524_288, 4096, where="t")
    with pytest.raises(sa.ServingAttestationError, match="not a positive int"):
        sa._validate_kernel_ceiling_keys(524_287, None, where="t")

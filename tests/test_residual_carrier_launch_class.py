"""A residual-slot CARRIER adapts no module class (the steered serving canary, 2026-10-01).

The native standup's zero-adapter class and the composer's ``--lora-target-modules``
(``vllm_server._adapted_module_class``) read ONE rule, ``lora_export.launch_target_class``:
every GENUINE adapter must adapt the same class; a residual-slot carrier — the rank-1 ZERO
o_proj pair ``lora_export._export_residual_slot`` writes beside the payload — never decides the
class and rides INSIDE the genuine adapters' (vLLM resets its slot at every wrapper it has no
factor for), while a carrier OUTSIDE the class, or genuine adapters that disagree, refuse by
name. Before this a psr carrier beside an attn_mlp (expert-bearing) LoRA refused the standup
(job 817770). Also: the canary sbatch's registry flag is honoured by every registry entry point
of ``scripts/sdfbase/full22/common.sh``."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sisteer.capeval.serving import lora_export as lx  # noqa: E402
from sisteer.capeval.serving import serving_attestation as sa  # noqa: E402
from sisteer.capeval.serving import vllm_server as vs  # noqa: E402
import steered_canary as sc  # noqa: E402  (the derived pair's canary marker)

CODE_SHA = "9c4fd67577dfc6df5e75fc5dd63d2882bba59d7b"   # == synth.code_sha("a"): a placeholder (forty identical digits) run code_sha is refused; the guard's WAVE_SHA
PFX = "base_model.model.model.layers"


def _adapter(d: Path, *, modules, r=4, experts=False, carrier=False, layers=(0, 1)) -> Path:
    """A synthetic PEFT adapter dir whose FILE adapts ``modules`` (+ both experts halves when
    ``experts``); a ``carrier`` adds the residual payload keys and the petri_residual_slot block."""
    d.mkdir(parents=True)
    g = torch.Generator().manual_seed(7)
    t = {}
    for layer in layers:
        for m in modules:
            site = "self_attn" if m in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
            t[f"{PFX}.{layer}.{site}.{m}.lora_A.weight"] = torch.zeros(r, 16) if carrier else torch.randn(r, 16, generator=g)
            t[f"{PFX}.{layer}.{site}.{m}.lora_B.weight"] = torch.zeros(24, r) if carrier else torch.randn(24, r, generator=g)
        if experts:
            t[f"{PFX}.{layer}.mlp.experts.base_layer.lora_A.weight"] = torch.randn(8 * r, 16, generator=g)
            t[f"{PFX}.{layer}.mlp.experts.base_layer.lora_B.weight"] = torch.randn(8, 48, r, generator=g)
            t[f"{PFX}.{layer}.mlp.experts.lora_A.weight"] = torch.randn(8 * r, 24, generator=g)
            t[f"{PFX}.{layer}.mlp.experts.lora_B.weight"] = torch.randn(8, 16, r, generator=g)
        if carrier:
            t[f"{sa.RESIDUAL_KEY_PREFIX}layers.{layer}.direction"] = torch.randn(16, generator=g)
    save_file(t, str(d / lx.ADAPTER_WEIGHTS))
    cfg = {"peft_type": "LORA", "r": r, "lora_alpha": r, "target_modules": sorted(modules), "bias": "none"}
    if carrier:
        cfg[vs.RESIDUAL_SLOT_CONFIG_BLOCK] = {"schema": 1, "kind": "psr", "position_mode": "all"}
    (d / lx.ADAPTER_CONFIG).write_text(json.dumps(cfg, indent=2, sort_keys=True))
    return d


# --------------------------------------------------------------------------- the pure rule
def test_launch_target_class_rule():
    attn_mlp = ("down_proj", "experts", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj")
    assert lx.launch_target_class({"lora": (attn_mlp, False), "relu": (("o_proj",), True)}) == attn_mlp
    assert lx.launch_target_class({"relu": (("o_proj",), True)}) == ("o_proj",)
    assert lx.launch_target_class({}) == ("o_proj",)
    assert lx.launch_target_class({"a": (("o_proj",), False), "b": (("o_proj",), False)}) == ("o_proj",)
    with pytest.raises(lx.LoraExportError, match="registered adapters disagree on their adapted module class"):
        lx.launch_target_class({"a": (("o_proj",), False), "b": (("q_proj",), False), "relu": (("o_proj",), True)})
    with pytest.raises(lx.LoraExportError, match="carrier.*outside the launch class"):
        lx.launch_target_class({"a": (("q_proj", "k_proj"), False), "relu": (("o_proj",), True)})
    with pytest.raises(lx.LoraExportError, match="carriers of this launch disagree"):
        lx.launch_target_class({"r1": (("o_proj",), True), "r2": (("q_proj",), True)})


# --------------------------------------------------------------------------- the composer
def test_composer_admits_a_carrier_inside_an_expert_bearing_class_and_still_refuses_disagreement(tmp_path):
    lora = _adapter(tmp_path / "lora", modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"), experts=True)
    zero = _adapter(tmp_path / "zero", modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"), experts=True, r=1)
    relu = _adapter(tmp_path / "relu", modules=("o_proj",), r=1, carrier=True)
    want = ("down_proj", "experts", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj")
    assert lx.adapter_target_class(relu) == ("o_proj",) and lx.adapter_target_class(lora) == want
    # the carrier + the expert-bearing LoRA (+ the class zero) compose the LoRA's class
    assert vs._adapted_module_class({"si_x__lora_r4_ce": str(lora), "si_x__psr_ce_all": str(relu), "none__lora0": str(zero)}) == want
    # carriers alone keep the paper's o_proj class
    assert vs._adapted_module_class({"si_x__psr_ce_all": str(relu)}) == ("o_proj",)
    # two GENUINE adapters of different classes still refuse by the old name
    wo = _adapter(tmp_path / "wo", modules=("o_proj",))
    with pytest.raises(vs.VllmServerError, match="registered adapters disagree on their adapted module class"):
        vs._adapted_module_class({"a": str(lora), "b": str(wo)})
    # a carrier whose modules lie outside the genuine class refuses by name (a genuine q_proj-only LoRA beside an o_proj carrier)
    q_only = _adapter(tmp_path / "q", modules=("q_proj",))
    with pytest.raises(vs.VllmServerError, match="carrier.*outside the launch class"):
        vs._adapted_module_class({"a": str(q_only), "relu": str(relu)})
    # a plain (non-carrier) rank-1 o_proj adapter beside the attn_mlp LoRA is a genuine disagreement, never admitted as a carrier
    plain = _adapter(tmp_path / "plain", modules=("o_proj",), r=1)
    with pytest.raises(vs.VllmServerError, match="registered adapters disagree"):
        vs._adapted_module_class({"a": str(lora), "p": str(plain)})


def test_standup_zero_class_reads_the_same_rule():
    """The zero-adapter class block of bring_up_native marks carriers by petri_source_kind in RESIDUAL_SLOT_KINDS
    and derives the class through lora_export.launch_target_class (the composer's rule) — pinned by text, the
    block runs only inside a GPU standup."""
    src = (REPO / "src" / "sisteer" / "capeval" / "serving" / "native_standup.py").read_text()
    i = src.index("# A RESIDUAL-SLOT CARRIER IS NOT A LoRA OF THE CLASS")
    block = src[i:i + 2600]
    assert 'str(b.get("petri_source_kind")) in RESIDUAL_SLOT_KINDS' in block
    assert "zero_class = lx.launch_target_class(classes)" in block
    assert "carriers_in_class" in block and "ride inside it" in block
    assert 'distinct = sorted(set(classes.values()))' not in block
    comp = (REPO / "src" / "sisteer" / "capeval" / "serving" / "vllm_server.py").read_text()
    j = comp.index("def _adapted_module_class(")
    assert "return _lx.launch_target_class(sets)" in comp[j:j + 4000]
    assert "RESIDUAL_SLOT_CONFIG_BLOCK in cfg" in comp[j:j + 4000]


# --------------------------------------------------------------------------- the registry guard (352956c83)
F22 = REPO / "scripts" / "sdfbase" / "full22" / "common.sh"
import test_registry_native_wiring as trw  # noqa: E402  (its stub cluster: spool, pending, a registering config)


def _registry_script(co, *, key="qwen35_4b", view="main", cfg=None, pre="") -> str:
    return f"""
set -u
REPO_DIR="{REPO}"; WAVE_SHA="{CODE_SHA}"; CLUSTER=cluster_a
KEY={key}; VIEW={view}; CFG="{cfg or co['cfg']}"; F22_SERVING_PATH=lora_native
{pre}
source {F22}
f22_registry_state running RUNNING; echo "rc_state=$?"
f22_registry_register; echo "rc_reg=$?"; echo "ident=${{SISTEER_REGISTRY_RUN_IDENTITY:-unset}}"
f22_registry_result "COMPLETED exit=0"; echo "rc_res=$?"
"""


def _run(script: str, env: dict):
    r = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    return r


def _run_rc(script: str, env: dict):
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=300)


def test_canary_script_identity_refuses_registration_without_any_flag(tmp_path):
    """The P1 shape: a canary IDENTITY (its script) + an inherited F22_REGISTRY_EVALS and NO flag -> refused before any
    registration call; the job name is neutral (not the production kinds, not scanary*)."""
    st = trw._stub_cluster(tmp_path)
    env = dict(st["env"], SLURM_JOB_NAME="my-own-launcher", F22_REGISTRY_EVALS="xstest or_bench_hard", F22_JOB_SCRIPT="/x/y/steered_canary.sbatch")
    env.pop("F22_REGISTRY_CANARY", None)
    r = _run(_registry_script(st["checkout"]), env)
    assert "rc_state=0" in r.stdout and "rc_reg=0" in r.stdout and "ident=unset" in r.stdout and "rc_res=0" in r.stdout
    assert r.stderr.count("canary identity (job script 'steered_canary.sbatch')") == 3
    assert not (st["shared"] / "registry" / "pending" / "4242.json").exists()
    assert not any((st["shared"] / "registry" / "records").iterdir())


def test_the_steered_canary_own_name_and_script_agree(tmp_path):
    st = trw._stub_cluster(tmp_path)
    env = dict(st["env"], SLURM_JOB_NAME="scanary-qwen35_4b", F22_REGISTRY_EVALS="xstest", F22_JOB_SCRIPT="/r/scripts/sdfbase/steered_canary.sbatch", F22_REGISTRY_CANARY="1")
    r = _run(_registry_script(st["checkout"]), env)
    assert "ident=unset" in r.stdout and "canary identity (job script 'steered_canary.sbatch'; F22_REGISTRY_CANARY=1)" in r.stderr
    assert not (st["shared"] / "registry" / "pending" / "4242.json").exists()


@pytest.mark.parametrize("name", ["f22s-qwen35_4b-canary-static", "f22s-qwen35_4b-canary-ctl"])
def test_the_sampled_waves_canary_named_production_runs_register_normally(tmp_path, name):
    """A job NAME never suppresses the registry — the sampled wave's production runs carry `-canary-` in
    their names and must register as before, with no warning."""
    st = trw._stub_cluster(tmp_path)
    env = dict(st["env"], SLURM_JOB_NAME=name, F22_JOB_SCRIPT="/r/scripts/sdfbase/full22/sampled/target_static_sampled.sbatch")
    r = _run(_registry_script(st["checkout"]), env)
    pend = st["shared"] / "registry" / "pending" / "4242.json"
    assert pend.exists() and f"ident={pend}" in r.stdout
    assert "canary identity" not in r.stderr and "contradiction" not in r.stderr and "WARNING" not in r.stderr
    assert {x["route"]["eval_id"] for x in json.loads(pend.read_text())["runs"]} >= {"pasttense", "codeattack", "renellm", "grok420"}


def test_a_canary_looking_name_on_a_production_script_fails_fast(tmp_path):
    st = trw._stub_cluster(tmp_path)
    env = dict(st["env"], SLURM_JOB_NAME="scanary-qwen35_4b", F22_REGISTRY_EVALS="xstest", F22_JOB_SCRIPT="/r/scripts/sdfbase/full22/target_static.sbatch")
    r = _run_rc(_registry_script(st["checkout"]), env)
    out = r.stdout + r.stderr   # f22_fatal logs through f22_log (stdout) before exiting
    assert r.returncode != 0 and "FATAL: registry: contradiction" in out and "scanary*" in out and "rc_state" not in r.stdout
    assert not (st["shared"] / "registry" / "pending" / "4242.json").exists()
    assert not any((st["shared"] / "registry" / "records").iterdir())


def test_a_canary_script_under_a_production_job_name_fails_fast(tmp_path):
    st = trw._stub_cluster(tmp_path)
    env = dict(st["env"], SLURM_JOB_NAME="f22-petri-qwen35_4b", F22_JOB_SCRIPT="/x/steered_canary.sbatch")
    r = _run_rc(_registry_script(st["checkout"]), env)
    out = r.stdout + r.stderr
    assert r.returncode != 0 and "FATAL: registry: contradiction" in out and "production job name 'f22-petri-qwen35_4b'" in out and "rc_state" not in r.stdout
    assert not (st["shared"] / "registry" / "pending" / "4242.json").exists()


def test_production_identity_registers_even_with_the_flag_set(tmp_path):
    """The guard is not fooled the other way: the flag never decides — a production identity registers, with a WARNING."""
    st = trw._stub_cluster(tmp_path)
    env = dict(st["env"], SLURM_JOB_NAME="f22-petri-qwen35_4b", F22_REGISTRY_CANARY="1")
    r = _run(_registry_script(st["checkout"]), env)
    pend = st["shared"] / "registry" / "pending" / "4242.json"
    assert pend.exists() and f"ident={pend}" in r.stdout
    assert "WARNING: F22_REGISTRY_CANARY=1 is set but this job's identity is not a canary's" in r.stderr
    assert "canary identity" not in r.stderr


def test_canary_script_and_config_marker_are_identities_too(tmp_path):
    st = trw._stub_cluster(tmp_path)
    # (2) the job's own script named *_canary.sbatch (scontrol's Command= in production; F22_JOB_SCRIPT here)
    env = dict(st["env"], SLURM_JOB_NAME="alt-launcher", F22_REGISTRY_EVALS="xstest", F22_JOB_SCRIPT="/x/y/steered_canary.sbatch")
    r = _run(_registry_script(st["checkout"]), env)
    assert "ident=unset" in r.stdout and "canary identity (job script 'steered_canary.sbatch')" in r.stderr
    assert not (st["shared"] / "registry" / "pending" / "4242.json").exists()
    # (3) a resolved config carrying the `canary:` marker steered_canary.py derive stamps first
    derived = tmp_path / "derived.yaml"
    cfg_src = json.loads(json.dumps({"explicit_cells": ["none", "synth__lora_r4"]}))
    import yaml
    new_cfg, _m, _c = sc.derive_pair(cfg_src, {"petri": {"cells": ["none"]}}, cells=["none", "synth__lora_r4"],
                                     harness_cells=["none", "synth__lora_r4"], defenses_dir="/v", namespace="steered_canary_x")
    derived.write_text(yaml.safe_dump(new_cfg, sort_keys=False))
    env = dict(st["env"], SLURM_JOB_NAME="alt-launcher", F22_REGISTRY_EVALS="xstest", F22_JOB_SCRIPT="/x/y/some_wave.sbatch")
    r = _run(_registry_script(st["checkout"], cfg=str(derived)), env)
    assert "ident=unset" in r.stdout and f"canary identity (config {derived} carries the canary marker)" in r.stderr
    assert not (st["shared"] / "registry" / "pending" / "4242.json").exists()


def test_identity_is_the_script_or_the_marker_never_the_name():
    """f22_canary_identity reads the script and the config marker only; f22_canary_name is the steered canary's own
    `scanary*` cross-check — `-canary-` inside a production name is NOT a canary signal."""
    script = """
set -u
source {F22}
for s in /a/steered_canary.sbatch /a/other_canary.sbatch /a/target_static.sbatch /a/canary.sbatch; do
  if F22_JOB_SCRIPT="$s" CFG= SLURM_JOB_ID= f22_canary_identity >/dev/null; then echo "$(basename $s)=canary"; else echo "$(basename $s)=prod"; fi
done
for n in scanary-qwen35_4b scanary f22s-qwen35_4b-canary-static f22s-qwen35_4b-canary-ctl x-canary-y canary-foo f22-petri-qwen35_4b; do
  if SLURM_JOB_NAME="$n" f22_canary_name; then echo "$n=cname"; else echo "$n=not"; fi
done
""".replace("{F22}", str(F22))
    env = {"PATH": os.environ.get("PATH", ""), "HOME": "/nonexistent", "USER": "nobody-synthetic", "SHARED": "/nonexistent"}
    r = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    got = dict(ln.split("=", 1) for ln in r.stdout.split() if "=" in ln)
    assert got == {"steered_canary.sbatch": "canary", "other_canary.sbatch": "canary", "target_static.sbatch": "prod", "canary.sbatch": "prod",
                   "scanary-qwen35_4b": "cname", "scanary": "cname", "f22s-qwen35_4b-canary-static": "not", "f22s-qwen35_4b-canary-ctl": "not",
                   "x-canary-y": "not", "canary-foo": "not", "f22-petri-qwen35_4b": "not"}

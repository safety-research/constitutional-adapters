"""The GATED-DELTANET kernel overlay of the serving venv (the steered canary, 2026-10-01): transformers' Qwen3.5 GDN
layer falls back to pure torch when flash-linear-attention / causal-conv1d are missing — the canary's standup log carried
that line from the SERVING venv while the fit venv has both. The overlay rides the nemotron precedent: a separate venv
sisteer-petri-vllm-gdn-<hash8 of base+overlay>, the base requirements untouched (a live venv is never rebuilt),
selected per model key by f22_serving_venv_for_key (qwen35_* -> gdn; nemotron3nano4b -> nemotron; else the base)."""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
F22 = REPO / "scripts" / "sdfbase" / "full22" / "common.sh"
BASE = REPO / "configs" / "capeval" / "vllm-requirements.txt"
GDN = REPO / "configs" / "capeval" / "vllm-requirements-gdn.txt"
NEMO = REPO / "configs" / "capeval" / "vllm-requirements-nemotron.txt"


def _pins(path: Path) -> dict:
    out = {}
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and "==" in ln and not ln.startswith("--") and " @ " not in ln:
            name, ver = ln.split("==", 1)      # `--extra-index-url` and the sha-pinned `vllm @ <url>` are not pins
            out[name] = ver
    return out


def test_the_overlay_pins_the_fit_envs_kernels_and_nothing_else():
    pins = _pins(GDN)
    assert pins == {"fla-core": "0.5.2", "flash-linear-attention": "0.5.2", "causal-conv1d": "1.6.2.post1"}
    # the fit env's pins (the scripts/make_fit_configs.py note) and the nemotron overlay's causal-conv1d, byte for byte
    mk = (REPO / "scripts" / "make_fit_configs.py").read_text()
    assert "flash-linear-attention" in mk and "0.5.2" in mk and "causal-conv1d 1.6.2.post1" in mk
    assert _pins(NEMO)["causal-conv1d"] == pins["causal-conv1d"]
    # the base stays untouched by this overlay: none of the three is pinned there (the live venv's hash does not move)
    assert not ({"fla-core", "flash-linear-attention", "causal-conv1d"} & set(_pins(BASE)))
    text = GDN.read_text()
    assert "--no-build-isolation --no-deps" in text and "fla-core==0.5.2 (listed)" in text and "triton>=3.3" in text


def _funcs(*names) -> str:
    sh = F22.read_text()
    out = []
    for name in names:
        start = sh.index(f"{name}(){{")
        end = sh.index("\n}\n", start) + 3
        out.append(sh[start:end])
    return "\n".join(out)


def test_variant_per_key_and_the_venv_naming():
    """f22_serving_venv_variant_for_key maps the hybrids to their overlays; `f22_serving_venv gdn` names
    sisteer-petri-vllm-gdn-<hash8 of base + overlay> (the nemotron recipe, run not grepped; uv is stubbed)."""
    funcs = _funcs("f22_serving_venv_variant_for_key", "f22_serving_venv_for_key", "f22_serving_venv")
    h_gdn = hashlib.sha256(BASE.read_bytes() + GDN.read_bytes()).hexdigest()[:8]
    h_base = hashlib.sha256(BASE.read_bytes()).hexdigest()[:8]
    body = funcs[funcs.index("f22_serving_venv(){"):]
    naming = body[body.index("{") + 1:body.index("(\n    flock")]
    script = (f"set -u\nf22_log(){{ :; }}\nf22_fatal(){{ echo FATAL \"$@\"; exit 1; }}\nflock(){{ :; }}\nuv(){{ :; }}\n"
              f"REPO_DIR={REPO}; USER=tester\n{funcs}\n"
              "f22_serving_venv() { local variant=\"${1:-}\"; " + naming + "echo \"$SERVE_VENV|$SERVE_REQS_SHA|$overlay\"; }\n"
              "for k in qwen35_4b qwen35_9b nemotron3nano4b gemma4e4b llama8b qwen35_4b_demo; do echo \"$k=$(f22_serving_venv_variant_for_key $k)\"; done\n"
              "f22_serving_venv gdn; f22_serving_venv_for_key qwen35_4b; f22_serving_venv_for_key gemma4e4b\n")
    env = {"PATH": os.environ.get("PATH", ""), "HOME": "/nonexistent", "USER": "tester"}
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, (r.stdout, r.stderr)
    lines = r.stdout.strip().splitlines()
    mapping = dict(ln.split("=", 1) for ln in lines[:6])
    # the front door's demo key (reproduce/models.yaml qwen35_4b_demo, a Qwen3.5 hybrid) lands on the gdn overlay by the same qwen35_* rule
    assert mapping == {"qwen35_4b": "gdn", "qwen35_9b": "gdn", "nemotron3nano4b": "nemotron",
                       "gemma4e4b": "", "llama8b": "", "qwen35_4b_demo": "gdn"}
    gdn_line, by_key_line, base_line = lines[6], lines[7], lines[8]
    assert gdn_line.split("|")[0] == f"/home/tester/venvs/sisteer-petri-vllm-gdn-{h_gdn}" and gdn_line.split("|")[2] == str(GDN)
    assert by_key_line == gdn_line                                   # the selector lands on the same venv
    assert base_line.split("|")[0] == f"/home/tester/venvs/sisteer-petri-vllm-{h_base}" and base_line.split("|")[2] == ""


def test_the_overlay_import_check_is_per_variant_and_asks_transformers_too():
    sh = F22.read_text()
    i = sh.index('case "$tag" in'); block = sh[i:i + 2500]
    assert "-gdn)" in block and "import fla, causal_conv1d" in block and "from fla.ops.gated_delta_rule import chunk_gated_delta_rule" in block
    assert "from causal_conv1d import causal_conv1d_fn, causal_conv1d_update" in block
    assert "is_flash_linear_attention_available() and is_causal_conv1d_available()" in block
    assert "import mamba_ssm, causal_conv1d" in block                 # the nemotron check is unchanged, under its own case
    # the overlay install is the nemotron recipe (source build against the venv's torch): ONE command serves both variants,
    # common.sh's f22_overlay_install, called once by f22_serving_venv (tests/test_serving_overlay_install.py dry-runs it)
    assert len(re.findall(r"f22_overlay_install \"\$SERVE_VENV/bin/python\" \"\$overlay\" \|\| exit 1", sh)) == 1
    assert len(re.findall(r"CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE uv pip install --no-config --python \"\$py\"", sh)) == 1


def test_the_steered_canary_selects_its_venv_by_key():
    text = (REPO / "scripts" / "sdfbase" / "steered_canary.sbatch").read_text()
    assert 'f22_serving_venv_for_key "$KEY"' in text and "f22_serving_venv nemotron" not in text


# --------------------------------------------------------------------------- central selection + the record
def _naming_shell(extra: str = "") -> str:
    """A bash harness running the REAL f22_serving_venv naming (uv / flock stubbed, the bootstrap body skipped)."""
    funcs = _funcs("f22_serving_venv_variant_for_key", "f22_serving_venv_for_key", "f22_serving_venv", "f22_log")
    body = funcs[funcs.index("f22_serving_venv(){"):]
    body = body[:body.index("\n}\n") + 3]
    naming = body[body.index("{") + 1:body.index("(\n    flock")]
    return (f"set -u\nf22_fatal(){{ echo FATAL \"$@\"; exit 1; }}\nflock(){{ :; }}\nuv(){{ :; }}\nf22_driver_venv(){{ :; }}\nf22_bank_view(){{ :; }}\nMPY=python3\n"
            f"REPO_DIR={REPO}; USER=tester\n{funcs}\n"
            "f22_serving_venv() { local variant reqs=\"$REPO_DIR/configs/capeval/vllm-requirements.txt\" overlay=\"\" tag=\"\" h hf lockdir sent\n"
            + naming[naming.index("  if [ $# -ge 1 ]"):] + "echo \"$SERVE_VENV|${variant:-base}\"; }\n" + extra)


def test_a_bare_call_resolves_the_variant_from_the_jobs_key_and_the_production_lines_follow(tmp_path):
    """target_static / target_attacks / petri_proxy / bon_climb call `f22_serving_venv` BARE: a qwen35_* job must land on the
    gdn venv, a nemotron job on nemotron, gemma on the base — run through the production sbatch's exact line."""
    h_gdn = hashlib.sha256(BASE.read_bytes() + GDN.read_bytes()).hexdigest()[:8]
    h_nemo = hashlib.sha256(BASE.read_bytes() + NEMO.read_bytes()).hexdigest()[:8]
    h_base = hashlib.sha256(BASE.read_bytes()).hexdigest()[:8]
    for sbatch in ("target_static.sbatch", "petri_proxy.sbatch", "target_attacks.sbatch", "bon_climb.sbatch"):
        line = next(ln for ln in (REPO / "scripts" / "sdfbase" / "full22" / sbatch).read_text().splitlines()
                    if ln.strip().startswith("f22_driver_venv; f22_serving_venv"))
        assert line.strip() == 'f22_driver_venv; f22_serving_venv; f22_bank_view "$MPY"', (sbatch, line)
        script = _naming_shell(f"for k in qwen35_4b nemotron3nano4b gemma4e4b llama8b; do KEY=$k; {line}; done\n"
                               "KEY=qwen35_4b; f22_serving_venv \"\"\nKEY=gemma4e4b; f22_serving_venv nemotron\nunset KEY; f22_serving_venv\n")
        env = {"PATH": os.environ.get("PATH", ""), "HOME": "/nonexistent", "USER": "tester"}
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=120)
        assert r.returncode == 0, (sbatch, r.stdout, r.stderr)
        got = [ln.split("|") for ln in r.stdout.strip().splitlines() if ln.startswith("/home/tester/")]
        # the four production lines by key (gdn / nemotron / base / base), then an EXPLICIT "" (= the base, whatever the key), an
        # explicit `nemotron` on a gemma key, and no key at all (= the base)
        assert [g[1] for g in got] == ["gdn", "nemotron", "base", "base", "base", "nemotron", "base"]
        assert got[0][0] == f"/home/tester/venvs/sisteer-petri-vllm-gdn-{h_gdn}"
        assert got[1][0] == f"/home/tester/venvs/sisteer-petri-vllm-nemotron-{h_nemo}" == got[5][0]
        assert got[2][0] == got[3][0] == got[4][0] == got[6][0] == f"/home/tester/venvs/sisteer-petri-vllm-{h_base}"
        assert "serving venv variant from KEY=qwen35_4b: 'gdn'" in r.stdout and "KEY=unbound: 'base'" in r.stdout
    # the identity exports every serving record reads
    sh = F22.read_text()
    assert 'export SISTEER_SERVING_VENV="$SERVE_VENV" SISTEER_SERVING_VENV_VARIANT="${tag#-}" SISTEER_SERVING_REQS_SHA256="$SERVE_REQS_SHA"' in sh
    assert 'export SISTEER_SERVING_BASE_REQS_SHA256="$(sha256sum "$reqs" | cut -c1-64)"' in sh   # the engine identity's base layer


def test_every_serving_record_names_its_venv(monkeypatch):
    """serving_attestation.serving_venv_facts rides in the attestation (build_serving_attestation), endpoint.json and
    cells_served.json (native_standup._common_facts, a core record key): the process's prefix parsed, the exported identity,
    the kernels importable here."""
    import sys
    from sisteer.capeval.serving import native_standup as ns
    from sisteer.capeval.serving import serving_attestation as sa
    assert sa.SERVING_VENV_KEY == "serving_venv" and sa.SERVING_VENV_KEY in ns._CORE_RECORD_KEYS
    src_sa = (REPO / "src" / "sisteer" / "capeval" / "serving" / "serving_attestation.py").read_text()
    i = src_sa.index("def build_serving_attestation("); assert "SERVING_VENV_KEY: serving_venv_facts()," in src_sa[i:i + 12000]
    src_ns = (REPO / "src" / "sisteer" / "capeval" / "serving" / "native_standup.py").read_text()
    i = src_ns.index("def _common_facts("); assert "sa.SERVING_VENV_KEY: sa.serving_venv_facts()," in src_ns[i:i + 6000]
    assert sa.parse_serving_venv_name("sisteer-petri-vllm-gdn-1a2b3c4d") == {"campaign_venv": True, "variant": "gdn", "hash8": "1a2b3c4d"}
    assert sa.parse_serving_venv_name("sisteer-petri-vllm-nemotron-deadbeef") == {"campaign_venv": True, "variant": "nemotron", "hash8": "deadbeef"}
    assert sa.parse_serving_venv_name("sisteer-petri-vllm-6b72fc94") == {"campaign_venv": True, "variant": None, "hash8": "6b72fc94"}
    assert sa.parse_serving_venv_name(".venv") == {"campaign_venv": False, "variant": None, "hash8": None}
    env = {"SISTEER_SERVING_VENV": "/home/x/venvs/sisteer-petri-vllm-gdn-1a2b3c4d", "SISTEER_SERVING_VENV_VARIANT": "gdn",
           "SISTEER_SERVING_REQS_SHA256": "f" * 64}
    f = sa.serving_venv_facts(prefix="/home/x/venvs/sisteer-petri-vllm-gdn-1a2b3c4d", environ=env)
    assert f["name"] == "sisteer-petri-vllm-gdn-1a2b3c4d" and f["variant"] == "gdn" and f["hash8"] == "1a2b3c4d" and f["campaign_venv"]
    assert f["exported"] == {"venv": env["SISTEER_SERVING_VENV"], "variant": "gdn", "requirements_sha256": "f" * 64} and f["exported_matches_prefix"]
    assert set(f["kernels"]) == {"fla", "causal_conv1d", "mamba_ssm"} and all({"importable", "version"} == set(v) for v in f["kernels"].values())
    # the engine identity {vllm version, BASE requirements digest, flashinfer version} rides in the facts
    assert sa.ENGINE_IDENTITY_KEY == "engine_identity" and set(f[sa.ENGINE_IDENTITY_KEY]) == set(sa.ENGINE_IDENTITY_FIELDS) == {"vllm_version", "base_requirements_sha256", "flashinfer_version"}
    env2b = dict(env, SISTEER_SERVING_BASE_REQS_SHA256="a" * 64)
    assert sa.serving_venv_facts(prefix="/home/x/venvs/sisteer-petri-vllm-gdn-1a2b3c4d", environ=env2b)[sa.ENGINE_IDENTITY_KEY]["base_requirements_sha256"] == "a" * 64
    assert f[sa.ENGINE_IDENTITY_KEY]["base_requirements_sha256"] is None     # not exported here
    assert sa.FLASHINFER_DISTS[0] == "flashinfer-python"                      # the installed dist name (0.6.13 in every campaign venv)
    # the base venv exports an EMPTY variant -> None; a prefix that is not the exported venv is flagged
    f2 = sa.serving_venv_facts(prefix="/home/x/venvs/sisteer-petri-vllm-6b72fc94", environ={"SISTEER_SERVING_VENV": "/elsewhere", "SISTEER_SERVING_VENV_VARIANT": ""})
    assert f2["variant"] is None and f2["exported"]["variant"] is None and f2["exported_matches_prefix"] is False
    # this process: never raises, reports its own prefix
    f3 = sa.serving_venv_facts()
    assert f3["prefix"] == sys.prefix and isinstance(f3["kernels"]["fla"]["importable"], bool)

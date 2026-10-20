"""The Qwen base seed-replicate battery: the BoN driver's three lane declarations reach bon_served_run.py as
SCALAR env vars through target_static.sbatch — F22_BON_DECODE_PRESET, F22_BON_DECODE_SEED_POLICY, F22_BON_DECODE_SEED_BASE —
because a Bash ARRAY never crosses sbatch (Slurm's implicit export carries scalars only). The launch line is RENDERED
under bash (the sbatch's own text, a stub f22_queue_launch printing its argv), not re-typed: unset = today's argv byte
for byte; set = today's argv + exactly the three flags, which the driver's own parser accepts. The smoke's line
(smoke_full22.sbatch) forwards the same three when set and is byte-identical when unset."""
from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
F22 = REPO / "scripts/sdfbase/full22"
for p in (F22, REPO / "scripts" / "sdfbase" / "seedrep"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import bon_served_run as br  # noqa: E402
import eval_seed as es  # noqa: E402

VARS = ('MPY=/venv/bin/python; CFG=configs/sdfbase_full22_qwen35_4b_s1.yaml; cell=constv2-mix-full__psr_ce_all; PRIVATE_ROOT=/wave/transcripts_results; '
        'PROXY_URL=http://127.0.0.1:4321/v1; TABLE=/wave/serving/routing_table.json; TARGET_L=4864; BON_CONC=24; TARGET_ENGINE=lora_native; '
        'KEY=qwen35_4b; SERVE_DIR=/wave/serving; REPO_DIR=/repo; ')
STUBS = ('f22_queue_launch(){ shift; for a in "$@"; do printf "%s\\n" "$a"; done; }; f22_stop_requested(){ return 1; }; ')
#: today's argv, byte for byte (the golden the unset render must equal)
GOLDEN_WAVE = ["/venv/bin/python", "scripts/sdfbase/full22/bon_served_run.py",
               "--config", "configs/sdfbase_full22_qwen35_4b_s1.yaml", "--defense", "constv2-mix-full__psr_ce_all", "--confirmatory",
               "--private-root", "/wave/transcripts_results", "--proxy-url", "http://127.0.0.1:4321/v1",
               "--routing-table", "/wave/serving/routing_table.json", "--max-model-len", "4864", "--max-concurrency", "24",
               "--require-engine", "lora_native", "--bon-decode-preset", "vendor_preset", "--model-key", "qwen35_4b"]
#: (the wave's default preset is main's `vendor_preset` since 2026-09-17 — the served model's registry row;
#: the lane's runs name `anchor_match` explicitly, scripts/sdfbase/seedrep/select_cells.py)
GOLDEN_SMOKE = ["/venv/bin/python", "scripts/sdfbase/full22/bon_served_run.py",
                "--config", "configs/sdfbase_full22_qwen35_4b_s1.yaml", "--defense", "constv2-mix-full__psr_ce_all", "--confirmatory",
                "--private-root", "/wave/transcripts_results", "--proxy-url", "http://127.0.0.1:4321/v1",
                "--routing-table", "/wave/serving/routing_table.json", "--max-model-len", "4864",
                "--max-concurrency", "40", "--limit-goals", "2", "--require-engine", "lora_native"]
LANE_ENV = {"F22_BON_DECODE_PRESET": "anchor_match", "F22_BON_DECODE_SEED_POLICY": "replicate_v1",
            "F22_BON_DECODE_SEED_BASE": str(es.seed_base(1))}


def _snippet(src: str, start: str) -> str:
    """The launch command starting at ``start`` through its last continuation line, verbatim from the sbatch."""
    i = src.index(start)
    lines = src[i:].splitlines()
    out = []
    for line in lines:
        out.append(line)
        if not line.rstrip().endswith("\\"):
            break
    return "\n".join(out)


def _render(snippet: str, env: dict) -> list:
    exports = "".join(f"export {k}={shlex.quote(v)}; " for k, v in env.items())
    r = subprocess.run(["bash", "-c", "set -u; " + VARS + STUBS + exports + snippet], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


def test_the_wave_job_forwards_exactly_the_three_scalars_and_is_byte_identical_when_unset():
    src = (F22 / "target_static.sbatch").read_text()
    snippet = _snippet(src, 'f22_queue_launch "bon__${cell}"')
    assert "--decode-seed-policy" in snippet and "--decode-seed-base" in snippet and "--bon-decode-preset" in snippet
    unset = _render(snippet, {})
    assert unset == GOLDEN_WAVE
    lane = _render(snippet, LANE_ENV)
    assert lane == GOLDEN_WAVE[:-4] + ["--bon-decode-preset", "anchor_match", "--model-key", "qwen35_4b",
                                       "--decode-seed-policy", "replicate_v1", "--decode-seed-base", str(es.seed_base(1))]
    # the driver's own parser accepts both renders and the three values land where the lane expects them
    ns = br.build_parser().parse_args(lane[2:])
    assert (ns.bon_decode_preset, ns.decode_seed_policy, ns.decode_seed_base) == ("anchor_match", "replicate_v1", es.seed_base(1))
    ns0 = br.build_parser().parse_args(unset[2:])
    assert (ns0.bon_decode_preset, ns0.decode_seed_policy, ns0.decode_seed_base) == ("vendor_preset", "bon_seed_v1", None)
    # a preset alone, or a policy alone: each rides independently (the scalars are independent declarations)
    only_preset = _render(snippet, {"F22_BON_DECODE_PRESET": "anchor_match"})
    assert only_preset == GOLDEN_WAVE[:-4] + ["--bon-decode-preset", "anchor_match", "--model-key", "qwen35_4b"]
    only_policy = _render(snippet, {"F22_BON_DECODE_SEED_POLICY": "bon_seed_v1"})
    assert only_policy == GOLDEN_WAVE + ["--decode-seed-policy", "bon_seed_v1"]
    # the env names are read as SCALARS and never assigned by the job; no --export= anywhere in the sbatch
    for name in LANE_ENV:
        assert re.search(r"\$\{" + name + r"[:}]", src), name
        assert not re.search(r"^\s*" + name + "=", src, flags=re.M), name
        assert not re.search(r"^\s*(declare -a|local -a)\s+" + name, src, flags=re.M), name
    assert not any("--export=" in l for l in src.splitlines() if not l.lstrip().startswith("#"))   # named in prose only
    assert subprocess.run(["bash", "-n", str(F22 / "target_static.sbatch")], capture_output=True).returncode == 0


def test_the_smoke_forwards_the_same_three_when_set_and_is_byte_identical_when_unset():
    src = (F22 / "smoke_full22.sbatch").read_text()
    i = src.index('"$MPY" scripts/sdfbase/full22/bon_served_run.py')
    body = src[i:]
    body = body[:body.index(") >")]                       # the subshell's command, up to its redirection
    snippet = STUBS + "f22_queue_launch x " + body.replace("\\\n", " ")
    unset = _render(snippet, {})
    assert unset == GOLDEN_SMOKE
    lane = _render(snippet, LANE_ENV)
    assert lane == GOLDEN_SMOKE + ["--bon-decode-preset", "anchor_match", "--decode-seed-policy", "replicate_v1",
                                   "--decode-seed-base", str(es.seed_base(1))]
    ns = br.build_parser().parse_args(lane[2:])
    assert (ns.bon_decode_preset, ns.decode_seed_policy, ns.decode_seed_base, ns.limit_goals) == ("anchor_match", "replicate_v1", es.seed_base(1), 2)
    assert subprocess.run(["bash", "-n", str(F22 / "smoke_full22.sbatch")], capture_output=True).returncode == 0


def test_the_wave_job_forwards_the_vendor_preset_and_the_literal_seed_base():
    """In the Qwen seed-replicate native battery, the BoN line is the SAME sbatch text with
    F22_BON_DECODE_PRESET=vendor and the view's LITERAL training seed as F22_BON_DECODE_SEED_BASE — no sbatch change;
    the driver's parser accepts the render and `vendor` resolves THE REGISTRY ROW at launch (tests/test_bon_vendor_preset.py)."""
    src = (F22 / "target_static.sbatch").read_text()
    snippet = _snippet(src, 'f22_queue_launch "bon__${cell}"')
    for s in (1, 2, 3):
        lane = _render(snippet, {"F22_BON_DECODE_PRESET": "vendor", "F22_BON_DECODE_SEED_POLICY": "replicate_v1",
                                 "F22_BON_DECODE_SEED_BASE": str(s)})
        assert lane == GOLDEN_WAVE[:-4] + ["--bon-decode-preset", "vendor", "--model-key", "qwen35_4b",
                                           "--decode-seed-policy", "replicate_v1", "--decode-seed-base", str(s)]
        ns = br.build_parser().parse_args(lane[2:])
        assert (ns.bon_decode_preset, ns.model_key, ns.decode_seed_policy, ns.decode_seed_base) == ("vendor", "qwen35_4b", "replicate_v1", s)
    smoke = (F22 / "smoke_full22.sbatch").read_text()
    i = smoke.index('"$MPY" scripts/sdfbase/full22/bon_served_run.py')
    body = smoke[i:]
    body = body[:body.index(") >")]
    sm = _render(STUBS + "f22_queue_launch x " + body.replace("\\\n", " "), {"F22_BON_DECODE_PRESET": "vendor"})
    assert sm == GOLDEN_SMOKE + ["--bon-decode-preset", "vendor"]           # the smoke line has no --model-key of its own: the driver refuses it at launch
    with pytest.raises(SystemExit, match="requires --model-key"):
        br.resolve_bon_decode("vendor", br.build_parser().parse_args(sm[2:]).model_key)

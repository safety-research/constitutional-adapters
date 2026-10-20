"""Regression guards for the torch CPU/GPU flavor split.

Static, CPU-only, no network:
- pyproject shape: torch/accelerate live ONLY in the flavor extras, the
  conflicts + per-extra index routing that make a flavor-less sync install
  no torch (loud ImportError, never the job-61380 silent-CPU path);
- uv.lock: NO PyPI-sourced torch fork anywhere (the strongest guard — a
  future relock that reintroduced it would silently hand cluster nodes a
  default-CUDA torch), and the cu128 fork is exactly 2.11.0+cu128;
- sbatch flavor audit, on ACTIVE LOGICAL COMMANDS (comments and heredoc
  bodies stripped, backslash continuations joined, quoted `bash -c`
  payloads audited as their own shell): every cluster `uv sync` is
  frozen + cu128-pinned with no unfrozen fallback and its OWN rc handling,
  every `uv run` carries --no-sync, every entry point runs an active cu128
  guard BEFORE its first workload, and at most one tagged audit exemption
  of EACH kind exists: the proof job's deliberate bare-sync loud-failure
  demo (all sync checks waived), and its prod-lock twin sync (flavor-less
  BY DESIGN against the production lockfile, where torch is a base dep;
  only the cu128-extra requirement is waived, frozen + rc handling stay
  enforced) — plus exactly the two petri serving scripts under
  OUT_OF_LOCKFILE_TAG (their venv is out-of-lockfile torch cu129, so the
  cu128 entry-point guard does not apply);
- stage2_common-based sbatch headers document the mandatory submit-time
  SHA pin so the incantations cannot drift back to unpinned — scope
  follows the source chain (capeval_common.sh and any deeper helper
  included), and the pin must derive from each job's effective
  S2_REPO_DIR, never a sibling checkout whose HEAD can differ.
"""

import itertools
import os
import re
import shutil
import subprocess
import textwrap
import tomllib
from pathlib import Path
from typing import NamedTuple
from front_door_testlib import MATCHERS, SCAN, token_hits

#: one tier of the publication scan's token table, matched as the scan matches it (the tokens are never spelled here)
STORAGE_TOKENS = SCAN.Matcher([r for r in MATCHERS["rows"] if r.tier == "storage"])

REPO = Path(__file__).resolve().parents[1]
SBATCH_DIR = REPO / "scripts" / "sbatch"
EXEMPT_TAG = "# audit-exempt: loud-failure-demo"
# The twin-run proof's prod-env twin syncs a throwaway venv from the
# PRODUCTION repo's lockfile, where torch is a base dependency — that sync
# is flavor-less by design and cannot produce the silent-CPU env (and the
# job asserts torch.cuda.is_available() under srun right after). The tag
# waives ONLY the cu128-extra requirement; frozen + rc handling still apply.
PROD_TWIN_TAG = "# audit-exempt: prod-lock-twin"
# The petri serving scripts (petri_gate / petri_proxy) run from the
# requirements-hash-versioned vllm venv installed OUTSIDE the project
# lockfile — pinned at torch cu129 (cu129 runs on driver 570 via CUDA
# 12.x minor-version compatibility), so the cu128 entry-point guard does
# not apply: asserting 2.11.0+cu128 there would be wrong, and the 61380
# silent-CPU path cannot occur (no `uv sync` anywhere; a missing venv
# fails `uv run --no-sync` loudly). File-level, entry-point audit only;
# the flavor audit (sync/run hygiene) still applies in full.
OUT_OF_LOCKFILE_TAG = "# audit-exempt: out-of-lockfile-venv"
EXEMPT_TAGS = (EXEMPT_TAG, PROD_TWIN_TAG)


def _pyproject() -> dict:
    return tomllib.loads((REPO / "pyproject.toml").read_text())


def _dep_name(spec: str) -> str:
    return re.match(r"[A-Za-z0-9_.-]+", spec).group(0).lower()


# ------------------------------------------------------------- pyproject


def test_torch_and_accelerate_are_not_base_deps():
    base = {_dep_name(d) for d in _pyproject()["project"]["dependencies"]}
    assert "torch" not in base, (
        "torch back in base deps — a bare sync would install it and the "
        "loud-failure guard is gone")
    assert "accelerate" not in base, (
        "accelerate back in base deps — it hard-requires torch and drags "
        "a PyPI torch into the no-extra lock fork (validated 2026-08-15)")


def test_flavor_extras_pin_torch_and_carry_accelerate():
    opt = _pyproject()["project"]["optional-dependencies"]
    for flavor in ("cpu", "cu128"):
        deps = opt[flavor]
        assert "torch==2.11.*" in deps, flavor
        assert any(_dep_name(d) == "accelerate" for d in deps), flavor
    lc = opt["longctx"]
    assert "sisteer[cu128]" in lc, "longctx must imply the GPU flavor"
    assert "torch==2.11.*" in lc, (
        "longctx needs its own torch pin — without it the longctx source "
        "entry is illegal and the longctx-sans-cu128 fork resolves PyPI "
        "torch")
    assert any(d.startswith("flash-attn==2.8.3.post1") for d in lc)
    # the fused Mamba kernels (REQUIRED on Mamba-hybrid targets, user directive
    # 2026-09-09) ride the SAME shape: a GPU-only extra with its own torch pin,
    # exact pins on the PyPI sdists uv compiles, the cp312/linux/x86_64 marker
    # that keeps the CUDA-only extra empty elsewhere
    mb = opt["mamba"]
    assert "sisteer[cu128]" in mb and "torch==2.11.*" in mb
    marker = ("python_full_version >= '3.12' and python_full_version < '3.13' "
              "and sys_platform == 'linux' and platform_machine == 'x86_64'")
    kernels = [d for d in mb if d.startswith(("mamba-ssm==", "causal-conv1d=="))]
    assert sorted(d.split(" ;")[0] for d in kernels) == ["causal-conv1d==1.6.2.post1", "mamba-ssm==2.3.1"]
    assert all(marker in d for d in kernels), kernels


def test_cpu_conflicts_with_cu128_and_longctx():
    groups = [{e["extra"] for e in grp}
              for grp in _pyproject()["tool"]["uv"]["conflicts"]]
    assert {"cpu", "cu128"} in groups
    assert {"cpu", "longctx"} in groups
    assert {"cpu", "mamba"} in groups


def test_torch_sources_route_each_extra_to_its_index():
    uv = _pyproject()["tool"]["uv"]
    routing = {s["extra"]: s["index"] for s in uv["sources"]["torch"]}
    assert routing == {
        "cpu": "pytorch-cpu",
        "cu128": "pytorch-cu128",
        "longctx": "pytorch-cu128",   # belt-and-suspenders
        "mamba": "pytorch-cu128",     # same belt for the fused-kernel extra
    }
    indexes = {i["name"]: i for i in uv["index"]}
    assert indexes["pytorch-cpu"]["url"] == "https://download.pytorch.org/whl/cpu"
    assert indexes["pytorch-cu128"]["url"] == "https://download.pytorch.org/whl/cu128"
    assert indexes["pytorch-cpu"]["explicit"] is True
    assert indexes["pytorch-cu128"]["explicit"] is True


#: The two kernel sdists exactly as PyPI serves them (sdist-only releases; checked 2026-10-12).
MAMBA_PINS = {"mamba-ssm": "2.3.1", "causal-conv1d": "1.6.2.post1"}
MAMBA_SDIST_SHA256 = {
    "mamba-ssm": "4d529477ad94753962216d583fc8f1c127c717b7d7c875d6bbb9376366d0d761",
    "causal-conv1d": "245e314ea21064ded7a5bf6b3b842b644aa6f92e45cecfe3e935629744c35ff4",
}


def test_mamba_extra_is_the_pinned_pypi_sdists_with_no_path_source():
    """The `mamba` extra installs the pinned PyPI releases, compiled by uv
    against the lockfile's torch, NOT the self-built path wheels under
    wheels/ that mamba_build.sbatch used to deposit. PyPI carries only
    sdists for both and upstream's prebuilt GitHub wheels stop at torch
    2.10 (the lock pins 2.11.0+cu128), so the sdist's setup.py compiles --
    pinned here: no [tool.uv.sources] entry for either package, both
    versions pinned in the extra, uv.lock resolving both from pypi.org as
    sdists with the known hashes (never a wheel, never a path), and the
    lock's manifest carrying the dependency-metadata that lets
    causal-conv1d (dynamic sdist metadata) lock without a build."""
    py = _pyproject()
    src = py["tool"]["uv"]["sources"]
    assert "mamba-ssm" not in src and "causal-conv1d" not in src, sorted(src)
    # flash-attn (the longctx extra) STAYS a path wheel: fa_build.sbatch's product
    assert src["flash-attn"] == {"path": "wheels/flash_attn-2.8.3.post1-cp312-cp312-linux_x86_64.whl"}
    assert "wheels/*.whl" in (REPO / ".gitignore").read_text().splitlines()
    pins = {_dep_name(d): d.split(" ;")[0].split("==")[1]
            for d in py["project"]["optional-dependencies"]["mamba"] if "==" in d and not d.startswith("torch")}
    assert pins == MAMBA_PINS, pins
    lock = tomllib.loads((REPO / "uv.lock").read_text())
    by_name = {p["name"]: p for p in lock["package"]}
    for name, version in MAMBA_PINS.items():
        entry = by_name[name]
        assert entry["version"] == version, (name, entry["version"])
        assert entry["source"] == {"registry": "https://pypi.org/simple"}, (name, entry["source"])
        assert "wheels" not in entry and "sdist" in entry, (name, sorted(entry))
        assert entry["sdist"]["hash"] == f"sha256:{MAMBA_SDIST_SHA256[name]}", (name, entry["sdist"])
        assert entry["sdist"]["url"].startswith("https://files.pythonhosted.org/"), entry["sdist"]["url"]
        # the sisteer requires-dist pins the version, with no path
        (req,) = [r for r in by_name["sisteer"]["metadata"]["requires-dist"] if r["name"] == name]
        assert req.get("specifier") == f"=={version}" and "path" not in req, req
    # causal-conv1d's static metadata, declared once in pyproject and mirrored by the lock's manifest
    (meta,) = py["tool"]["uv"]["dependency-metadata"]
    assert meta == {"name": "causal-conv1d", "version": "1.6.2.post1",
                    "requires-dist": ["torch", "packaging", "ninja"], "requires-python": ">=3.9"}, meta
    assert lock["manifest"]["dependency-metadata"] == [meta], lock["manifest"].get("dependency-metadata")


def test_mamba_sdists_build_against_the_runtime_torch_and_are_forced_to_compile():
    """uv's build settings for the two sdists: torch at the RUNTIME's exact
    distribution (match-runtime -- the build env must never pull a PyPI torch
    of its own) plus the sdists' setuptools / wheel / ninja / packaging, and
    the FORCE_BUILD switches so neither setup.py guesses a GitHub wheel URL at
    build time. The kernel extra's torch still routes to the cu128 index."""
    uv = _pyproject()["tool"]["uv"]
    for name in MAMBA_PINS:
        deps = uv["extra-build-dependencies"][name]
        assert {"requirement": "torch", "match-runtime": True} in deps, (name, deps)
        names = {_dep_name(d) for d in deps if isinstance(d, str)}
        assert {"setuptools", "wheel", "ninja", "packaging"} <= names, (name, names)
    assert uv["extra-build-variables"]["mamba-ssm"] == {"MAMBA_FORCE_BUILD": "TRUE"}
    assert uv["extra-build-variables"]["causal-conv1d"] == {"CAUSAL_CONV1D_FORCE_BUILD": "TRUE"}
    assert {s["extra"]: s["index"] for s in uv["sources"]["torch"]}["mamba"] == "pytorch-cu128"


def test_mamba_build_sbatch_warms_the_shared_uv_cache_and_proves_the_kernels_are_used():
    """scripts/sbatch/mamba_build.sbatch is the once-per-cluster cache warm-up
    plus the USED-not-importable proof: a frozen cu128+mamba sync of a scratch
    env against the SHARED UV_CACHE_DIR (the one s2_init reads) with a
    VALIDATED CUDA 12 toolkit selected by the shared cuda_toolkit.sh (the same
    file s2_init sources, so the fit path compiles with what this job warmed
    with) and a bounded MAX_JOBS set before it, then the tiny-model proof
    (scripts/mamba_tiny_proof.py) and, with a config, the campaign-seam check
    with its --fallback baseline. It no longer builds wheels itself nor copies
    anything into wheels/. Two failure modes are pinned here: SHARED is
    cluster-DETECTED (never a hardcoded shared root, which made the job look
    for its checkout, hf_env.sh and cache under a root that does not exist on
    cluster B -- the detection block itself runs under test_sbatch_lint's
    harness), and when no CUDA 12 toolkit exists (cluster B ships 13.0 only)
    the job installs CUDA 12.8.1 under $SHARED/cuda-12.8 from NVIDIA's
    runfile, pinned by URL + size + published md5 + sha256, BEFORE the sync,
    instead of letting torch's major-mismatch error kill the build."""
    sb = (REPO / "scripts" / "sbatch" / "mamba_build.sbatch").read_text()
    code = "\n".join(l for l in sb.splitlines() if l.strip() and not l.lstrip().startswith("#"))
    hardcoded = [l for l in code.splitlines() if l.startswith("SHARED=") and token_hits(l, STORAGE_TOKENS)]
    assert not hardcoded, hardcoded          # the finding: hardcoded root
    assert 'SHARED="${SISTEER_SHARED_ROOT:?' in code   # the job environment names it (the front door exports it from the site config)
    assert 'export UV_CACHE_DIR=$SHARED/.cache/uv' in code
    assert "export MAX_JOBS=16" in code
    assert '[ -s "$SHARED/scripts/hf_env.sh" ]' in code and 'source "$SHARED/scripts/hf_env.sh"' in code
    # the toolkit: the shared selector, then the pinned install when it finds none
    assert 'source "$CODE/scripts/sbatch/cuda_toolkit.sh"' in code
    assert "if ! s2_cuda12_select; then" in code
    assert 'CUDA_TOOLKIT_DIR="${S2_CUDA_SHARED_TOOLKIT:-$SHARED/cuda-12.8}"' in code
    assert "CUDA_RUNFILE_URL=https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run" in code
    assert "CUDA_RUNFILE_SIZE=5382238770" in code
    assert "CUDA_RUNFILE_MD5=e281945bf4ea42788db29457ebc1e6ae" in code
    assert "CUDA_RUNFILE_SHA256=228f6bcaf5b7618d032939f431914fc92d0e5ed39ebe37098a24502f26a19797" in code
    for check in ('[ "$size" = "$CUDA_RUNFILE_SIZE" ]', '| sha256sum -c -', '| md5sum -c -'):
        assert check in code, check
    assert '--toolkit --toolkitpath="$CUDA_TOOLKIT_DIR"' in code and "--silent" in code
    for no_root in ("sudo", "apt-get", "apt install", "dpkg"):   # user-space only, nothing under /usr
        assert no_root not in code, no_root
    # the old unconditional default is GONE: CUDA_HOME is only ever set by the selector
    assert "export CUDA_HOME=" not in code
    sync = "uv sync --frozen --extra cu128 --extra mamba"
    assert sync in code
    for before in ('export UV_CACHE_DIR=$SHARED/.cache/uv', "export MAX_JOBS=16",
                   'source "$CODE/scripts/sbatch/cuda_toolkit.sh"', "if ! s2_cuda12_select; then",
                   '--toolkitpath="$CUDA_TOOLKIT_DIR"'):
        assert code.index(before) < code.index(sync), before
    assert "mamba_tiny_proof.py" in code and "mamba_kernels_check.py" in code and "--fallback" in code
    assert code.index(sync) < code.index("mamba_tiny_proof.py") < code.index("mamba_kernels_check.py")
    # retired: the pip wheel build, the hand-rolled build deps, the deposit into wheels/
    for gone in ("pip wheel", "--no-build-isolation", "uv pip install", 'cp -v', '/wheels/', "MAMBA_FORCE_BUILD=TRUE"):
        assert gone not in code, gone
    subprocess.run(["bash", "-n", str(REPO / "scripts" / "sbatch" / "mamba_build.sbatch")], check=True)


def test_submit_mamba_build_takes_qos_and_the_private_log_dir_from_the_job_env(tmp_path):
    """scripts/sbatch/submit_mamba_build.sh: #SBATCH headers cannot branch per
    cluster, so the wrapper takes the cluster's QoS and shared root from the
    job environment the front door composes from the site block
    (SISTEER_CLUSTER_QOS, SISTEER_SHARED_ROOT), creates the PRIVATE log dir
    ($SHARED/exp/jailbreak/_logs, mode 0700; Slurm never creates log parents,
    and a missing one means the job NEVER STARTS), and overrides --qos /
    --output / --error at submit time. Flags before `--` go to sbatch,
    positionals after it to mamba_build.sbatch. sbatch hands the wrapper's
    environment to the job, which refuses without the shared root and the
    project directory, so without either (or the QoS) the wrapper refuses by
    name and submits nothing. Run for real with a fake sbatch on PATH and tmp
    roots."""
    sb_dir = tmp_path / "sbatch_dir"
    sb_dir.mkdir()
    wrapper = sb_dir / "submit_mamba_build.sh"
    shutil.copy2(REPO / "scripts" / "sbatch" / "submit_mamba_build.sh", wrapper)
    wrapper.chmod(0o755)
    (sb_dir / "mamba_build.sbatch").write_text("#!/bin/bash\n")   # the script it submits
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "sbatch").write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    (fake_bin / "sbatch").chmod(0o755)

    def site(label):
        root = tmp_path / f"shared_{label}"
        return {"SISTEER_SHARED_ROOT": str(root), "SISTEER_PROJECT_ROOT": str(root / "project"),
                "SISTEER_CLUSTER_QOS": f"qos_{label}"}

    def run(*args, **extra_env):
        env = {k: v for k, v in os.environ.items() if k != "QOS" and not k.startswith("SISTEER_")}
        env.update(PATH=f"{fake_bin}:{env['PATH']}", **extra_env)
        return subprocess.run([str(wrapper), *args], capture_output=True, text=True, env=env, timeout=30)

    # one cluster's job environment
    r = run("--time=4:00:00", "--", "/some/checkout", "configs/x.yaml", **site("a"))
    assert r.returncode == 0, r.stderr
    logdir = tmp_path / "shared_a" / "exp" / "jailbreak" / "_logs"
    assert logdir.is_dir() and (logdir.stat().st_mode & 0o777) == 0o700
    assert r.stdout.splitlines() == [
        "--qos=qos_a", f"--output={logdir}/%x_%j.out", f"--error={logdir}/%x_%j.err",
        "--time=4:00:00", str(sb_dir / "mamba_build.sbatch"), "/some/checkout", "configs/x.yaml"]
    # no flags, no positionals; QOS from the body env wins over the cluster's
    r = run(QOS="low", **site("a"))
    assert r.returncode == 0 and r.stdout.splitlines()[0] == "--qos=low"
    assert r.stdout.splitlines()[-1] == str(sb_dir / "mamba_build.sbatch")
    # the other cluster's job environment: its QoS and its log dir
    r = run("--", "/ws/checkout", **site("b"))
    assert r.returncode == 0, r.stderr
    b_logdir = tmp_path / "shared_b" / "exp" / "jailbreak" / "_logs"
    assert b_logdir.is_dir()
    assert r.stdout.splitlines()[:3] == ["--qos=qos_b", f"--output={b_logdir}/%x_%j.out",
                                         f"--error={b_logdir}/%x_%j.err"]
    assert r.stdout.splitlines()[-1] == "/ws/checkout"
    # without any one of the settings: refuses by name, submits nothing
    for missing in site("a"):
        r = run(**{k: v for k, v in site("a").items() if k != missing})
        assert r.returncode != 0 and missing in r.stderr and r.stdout == "", (missing, r.stderr)


def test_the_fit_path_and_the_warm_up_select_the_cuda_toolkit_through_one_file():
    """s2_init's mamba branch and mamba_build.sbatch source the SAME
    scripts/sbatch/cuda_toolkit.sh and call the same s2_cuda12_select, so the
    toolkit a fit compiles with on a cold cache is the one the warm-up used;
    neither carries a CUDA_HOME default of its own any more."""
    s2 = (REPO / "scripts" / "sbatch" / "stage2_common.sh").read_text()
    sb = (REPO / "scripts" / "sbatch" / "mamba_build.sbatch").read_text()
    s2_code = "\n".join(l for l in s2.splitlines() if l.strip() and not l.lstrip().startswith("#"))
    sb_code = "\n".join(l for l in sb.splitlines() if l.strip() and not l.lstrip().startswith("#"))
    assert 'source "$S2_REPO_DIR/scripts/sbatch/cuda_toolkit.sh"' in s2_code
    assert 'source "$CODE/scripts/sbatch/cuda_toolkit.sh"' in sb_code
    assert "s2_cuda12_select" in s2_code and "s2_cuda12_select" in sb_code
    for code in (s2_code, sb_code):
        assert "/usr/local/cuda" not in code       # the probe list lives in the one helper
        assert "export CUDA_HOME=" not in code
    helper = (REPO / "scripts" / "sbatch" / "cuda_toolkit.sh").read_text()
    assert "s2_cuda12_select()" in helper and 'S2_CUDA_MAJOR="${S2_CUDA_MAJOR:-12}"' in helper
    subprocess.run(["bash", "-n", str(REPO / "scripts" / "sbatch" / "cuda_toolkit.sh")], check=True)


def test_python_version_pins_cp312():
    """The flash-attn wheel and the cluster venvs are cp312; an unpinned
    fresh sync can land on 3.11 and silently diverge."""
    assert (REPO / ".python-version").read_text().strip() == "3.12"


# ---------------------------------------------------------------- uv.lock


def test_lock_has_no_pypi_torch_fork():
    lock = tomllib.loads((REPO / "uv.lock").read_text())
    torches = [p for p in lock["package"] if p["name"] == "torch"]
    assert torches, "no torch in the lock at all?"
    for p in torches:
        registry = p.get("source", {}).get("registry", "")
        assert "pypi.org" not in registry, (
            f"PyPI-sourced torch {p['version']} in uv.lock — a relock "
            "reintroduced the no-extra fork; re-apply the §1.1 mitigations")
        assert "download.pytorch.org" in registry, (p["version"], registry)
    assert any(p["version"] == "2.11.0+cu128" for p in torches), (
        "cu128 torch fork missing or version drifted")


# ------------------------------------------------------------ sbatch audit
#
# The audits below operate on ACTIVE LOGICAL COMMANDS, not raw text: a small
# shell lexer strips comment lines and trailing comments, drops heredoc
# BODIES (the introducer line stays), joins backslash continuations and
# quote-spanning newlines into the one command bash executes, and splices
# quoted `bash -c` payloads in as their own shell context with fresh
# `set -e` state (fa_build's guard lives inside an `srun bash -c` build
# shell). The raw-substring scans these replace were mutation-vulnerable:
# a commented-out guard, a guard moved after the workload, a `uv \`-split
# sync, a heredoc body, and a `set -e`..`set +e` window all slipped past
# them (adversarial review, 2026-08-15). The mutation tests at the bottom
# pin each of those catches against the REAL current scripts.

_SYNC_RE = re.compile(r"(?:^|[\s({;])uv\s+sync\b")
_RUN_RE = re.compile(r"(?:^|[\s({;])uv\s+run\b")
# Both present in one active command = the runtime assert itself (the
# s2_assert_cu128 / fa_build-inline python block).
_GUARD_MARKERS = ("torch.cuda.is_available()", "2.11.0+cu128")
# A workload invocation: anything that runs project code.
_WORKLOAD_RE = re.compile(r"(?:^|[\s;({])(?:s2_run\b|uv\s+run\b|python[0-9.]*\b)")


class _Cmd(NamedTuple):
    where: str    # "file.sbatch:12" or "file.sbatch:26[bash -c]:3"
    text: str     # the logical command (bash -c payloads -> __PAYLOAD__)
    exempt: str   # the audit-exempt tag carried in a trailing comment, or ""
    ctx: int      # shell context: the script, or one bash -c payload


def _logical_lines(text: str) -> list[tuple[int, str, bool]]:
    """Lex shell source into (lineno, command, exempt) triples of ACTIVE
    logical commands. Comment-only lines and trailing comments are removed
    (an EXEMPT_TAGS member in a trailing comment sets `exempt` to that
    tag), heredoc BODIES are
    skipped, and backslash continuations / quote-spanning newlines are
    joined, so the command text is what bash actually executes. The quote
    tracking is deliberately modest (no `$(...)` nesting model) — enough
    for these scripts; the mutation tests pin the cases that matter."""
    lines = text.split("\n")
    out: list[tuple[int, str, bool]] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        start = i + 1
        buf: list[str] = []
        exempt = ""
        heredocs: list[str] = []
        quote: str | None = None
        while i < len(lines):
            line = lines[i]
            j = 0
            continued = False
            while j < len(line):
                c = line[j]
                if quote == "'":
                    if c == "'":
                        quote = None
                    buf.append(c)
                    j += 1
                elif quote == '"':
                    if c == "\\":
                        if j == len(line) - 1:   # \<newline> joins in dquotes
                            continued = True
                            j += 1
                        else:
                            buf.append(line[j:j + 2])
                            j += 2
                    else:
                        if c == '"':
                            quote = None
                        buf.append(c)
                        j += 1
                elif c == "\\":
                    if j == len(line) - 1:       # line continuation
                        continued = True
                        j += 1
                    else:
                        buf.append(line[j:j + 2])
                        j += 2
                elif c in "'\"":
                    quote = c
                    buf.append(c)
                    j += 1
                elif c == "#" and (not buf or buf[-1] in " \t;&|"):
                    for tag in EXEMPT_TAGS:
                        if tag in line[j:]:
                            exempt = tag
                    j = len(line)                # trailing comment
                elif line.startswith("<<<", j):
                    # here-STRING: consume whole, or the re-test at the
                    # second `<` sees `<< word` and registers a bogus
                    # heredoc whose terminator never comes — swallowing
                    # the rest of the file (bit capeval_common.sh's
                    # `read <<< "$rest"`, hiding its source of
                    # stage2_common.sh from the submit-pin scope walk)
                    buf.append("<<<")
                    j += 3
                elif line.startswith("<<", j):
                    m = re.match(r"<<-?\s*(?:'([^']+)'|\"([^\"]+)\"|(\S+))",
                                 line[j:])
                    if m:
                        heredocs.append(m.group(1) or m.group(2) or m.group(3))
                        buf.append(line[j:j + m.end()])
                        j += m.end()
                    else:
                        buf.append(c)
                        j += 1
                else:
                    buf.append(c)
                    j += 1
            if continued:
                i += 1
                continue
            if quote is not None:                # newline inside quotes
                buf.append("\n")
                i += 1
                continue
            break
        i += 1
        for term in heredocs:                    # skip heredoc BODIES
            while i < len(lines) and lines[i].strip() != term:
                i += 1
            i += 1
        cmd = "".join(buf).strip()
        if cmd:
            out.append((start, cmd, exempt))
    return out


def _split_words(cmd: str) -> list[tuple[str, int, int]]:
    """Split a logical command into (unquoted_word, start, end) spans —
    bash-style: adjacent quoted/unquoted segments concatenate into one
    word, with the quote characters removed from the unquoted form."""
    words: list[tuple[str, int, int]] = []
    j, n = 0, len(cmd)
    while j < n:
        if cmd[j] in " \t\n":
            j += 1
            continue
        start = j
        unq: list[str] = []
        quote: str | None = None
        while j < n:
            c = cmd[j]
            if quote == "'":
                if c == "'":
                    quote = None
                else:
                    unq.append(c)
                j += 1
            elif quote == '"':
                if c == "\\" and j + 1 < n and cmd[j + 1] in '$`"\\':
                    unq.append(cmd[j + 1])
                    j += 2
                else:
                    if c == '"':
                        quote = None
                    else:
                        unq.append(c)
                    j += 1
            elif c in " \t\n":
                break
            elif c in "'\"":
                quote = c
                j += 1
            elif c == "\\" and j + 1 < n:
                unq.append(cmd[j + 1])
                j += 2
            else:
                unq.append(c)
                j += 1
        words.append(("".join(unq), start, j))
    return words


def _bash_c_payload(cmd: str) -> tuple[str, str] | None:
    """If this command hands a quoted script to `bash -c` (or `sh -c`),
    return (the command with the payload replaced by __PAYLOAD__, the
    payload script) so the payload is audited as its own shell rather
    than as outer-shell text."""
    words = _split_words(cmd)
    for k in range(len(words) - 2):
        w = words[k][0]
        if w in ("bash", "sh") or w.endswith("/bash") or w.endswith("/sh"):
            if words[k + 1][0] == "-c":
                payload, s, e = words[k + 2]
                return cmd[:s] + "__PAYLOAD__" + cmd[e:], payload
    return None


def _script_entries(path: Path) -> list[_Cmd]:
    """Flatten a script into execution-ordered active commands, splicing
    each `bash -c` payload's commands in (as a fresh shell context) right
    after the command that launches them."""
    entries: list[_Cmd] = []
    ctx_counter = itertools.count()

    def walk(src: str, prefix: str, ctx: int) -> None:
        for lineno, cmd, exempt in _logical_lines(src):
            split = _bash_c_payload(cmd)
            if split is not None:
                outer, payload = split
                entries.append(_Cmd(f"{prefix}:{lineno}", outer, exempt, ctx))
                walk(payload, f"{prefix}:{lineno}[bash -c]",
                     next(ctx_counter))
            else:
                entries.append(_Cmd(f"{prefix}:{lineno}", cmd, exempt, ctx))

    walk(path.read_text(), path.name, next(ctx_counter))
    return entries


def _errexit_after(cmd: str, errexit: bool) -> bool:
    """Track `set -e` / `set +e` state, including flag bundles
    (`set -euo pipefail`) and the -o/+o errexit spellings. Any other
    command leaves the state alone."""
    toks = cmd.split()
    if not toks or toks[0] != "set":
        return errexit
    k = 1
    while k < len(toks):
        t = toks[k]
        if t in ("-o", "+o") and k + 1 < len(toks):
            if toks[k + 1] == "errexit":
                errexit = t == "-o"
            k += 2
            continue
        if re.fullmatch(r"-[A-Za-z]+", t) and "e" in t:
            errexit = True
        elif re.fullmatch(r"\+[A-Za-z]+", t) and "e" in t:
            errexit = False
        k += 1
    return errexit


def _flavor_violations(sbatch_dir: Path) -> list[str]:
    """Every ACTIVE cluster `uv sync` must be frozen + cu128-pinned with no
    unfrozen fallback and ITS OWN rc handling — a same-command `||` handler
    (not `|| true`), a `$?` capture in the immediately-following commands,
    or errexit genuinely ACTIVE at the sync in its own shell context (an
    earlier `set -e` with no interposed `set +e`; a bash -c payload starts
    fresh). Every active `uv run` must carry --no-sync. EXEMPT_TAG skips a
    command's sync checks entirely (the bare-sync loud-failure demo);
    PROD_TWIN_TAG waives only the cu128-extra requirement (a sync against
    the production lockfile is flavor-less by design) — frozen and rc
    handling stay enforced. Tag counts are capped per kind elsewhere."""
    violations: list[str] = []
    files = sorted(p for p in sbatch_dir.iterdir() if p.is_file())
    assert files, f"{sbatch_dir} is empty?"
    for f in files:
        entries = _script_entries(f)
        errexit: dict[int, bool] = {}
        for i, e in enumerate(entries):
            errexit[e.ctx] = _errexit_after(e.text, errexit.get(e.ctx, False))
            if e.exempt == EXEMPT_TAG:
                continue
            sync = _SYNC_RE.search(e.text)
            if sync:
                if re.search(r"\|\|\s*uv\s+sync\b", e.text):
                    violations.append(
                        f"{e.where}: unfrozen `|| uv sync` fallback (D6 "
                        "removed these — both-halves-fail ran stale envs "
                        "silently)")
                for seg in re.split(r"\|\||&&", e.text):
                    if not _SYNC_RE.search(seg):
                        continue
                    if "--frozen" not in seg:
                        violations.append(f"{e.where}: unfrozen uv sync")
                    if ("--extra cu128" not in seg
                            and e.exempt != PROD_TWIN_TAG):
                        violations.append(
                            f"{e.where}: flavor-less uv sync — silent-CPU "
                            "risk")
                handled = (
                    re.search(r"\|\|(?!\s*(?:true|:)\b)",
                              e.text[sync.end():]) is not None
                    or any("$?" in later.text
                           for later in entries[i + 1:i + 4])
                    or errexit[e.ctx])
                if not handled:
                    violations.append(
                        f"{e.where}: uv sync result is not checked (no "
                        "same-command `||` handler, no `$?` capture, and "
                        "errexit is not active at the sync in this shell)")
            if _RUN_RE.search(e.text) and "--no-sync" not in e.text:
                violations.append(
                    f"{e.where}: cluster `uv run` without --no-sync can "
                    "mutate the asserted env mid-job")
    return violations


def _entry_point_violations(sbatch_dir: Path) -> list[str]:
    """Every *.sbatch must run an ACTIVE cu128 guard BEFORE its first
    workload invocation: `s2_init` for stage2_common-based scripts (all of
    them since the legacy port deleted cu128_env.sh; the capeval
    scripts source stage2_common.sh one level down through
    capeval_common.sh — pinned by test_capeval_stage2_common.py — so
    sourcing THAT wrapper counts — the submit-pin audit's source-chain
    walk below verifies that chain against the live files), or the inline
    variant — a frozen cu128
    sync followed by the runtime assert (fa_build's lives inside its
    `srun bash -c` build shell and is picked up through the payload
    splice). File-level exception: OUT_OF_LOCKFILE_TAG (the petri serving
    scripts) — their venv is the requirements-hash-versioned vllm venv
    installed OUTSIDE the lockfile at torch cu129, so the
    cu128 assert would be factually wrong there, and the 61380 silent-CPU
    path cannot occur (they never `uv sync`; a missing venv fails
    `uv run --no-sync` loudly)."""
    violations: list[str] = []
    for f in sorted(sbatch_dir.glob("*.sbatch")):
        if OUT_OF_LOCKFILE_TAG in f.read_text():
            continue
        entries = _script_entries(f)

        def first(pred):
            return next((i for i, e in enumerate(entries) if pred(e)), None)

        src = first(lambda e: re.search(
            r"(?:^|[\s;])(?:source|\.)\s+\S*"
            r"(?:stage2_common|capeval_common)\.sh", e.text))
        if src is not None:
            guard = first(lambda e: re.match(r"s2_init\b", e.text))
            what = "s2_init"
        else:
            sync = first(lambda e: not e.exempt and _SYNC_RE.search(e.text)
                         and "--frozen" in e.text
                         and "--extra cu128" in e.text)
            assert_ = first(
                lambda e: all(m in e.text for m in _GUARD_MARKERS))
            guard = (assert_ if sync is not None and assert_ is not None
                     and sync < assert_ else None)
            what = "an inline frozen-cu128-sync + runtime-assert block"
        if guard is None or (src is not None and guard < src):
            violations.append(
                f"{f.name}: no ACTIVE cu128 guard ({what}) — a commented-"
                "out or missing guard leaves the 61380 silent-CPU path "
                "open")
            continue
        workload = first(lambda e: _WORKLOAD_RE.search(e.text)
                         and not all(m in e.text for m in _GUARD_MARKERS))
        if workload is not None and workload < guard:
            violations.append(
                f"{f.name}: first workload ({entries[workload].where}) "
                f"runs BEFORE the cu128 guard ({entries[guard].where})")
    return violations


def _exempt_tags(root: Path) -> list[tuple[str, str]]:
    """Every audit-exempt tag under root, as (tag, "file:line") pairs."""
    tagged = []
    for f in sorted(root.rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts:
            text = f.read_text(errors="ignore")  # skip binary noise
            for i, line in enumerate(text.splitlines(), 1):
                for tag in (*EXEMPT_TAGS, OUT_OF_LOCKFILE_TAG):
                    if tag in line:
                        tagged.append((tag, f"{f.name}:{i}"))
    return tagged


# ----------------------------------------------------- the audits themselves


def test_sbatch_flavor_audit():
    """Every ACTIVE cluster sync is frozen + cu128 + rc-checked with no
    unfrozen fallback; every active `uv run` is --no-sync; the one tagged
    exemption is the proof job's bare-sync demo."""
    assert _flavor_violations(SBATCH_DIR) == []


def test_every_sbatch_entry_point_asserts_the_gpu_flavor():
    """Every job script reaches the runtime assert — CUDA available +
    torch 2.11.0+cu128, before the workload — via s2_init (every
    stage2_common-based script, the 8 ported legacy scripts included)
    or the inline block (fa_build's in-build variant)."""
    assert _entry_point_violations(SBATCH_DIR) == []


def test_the_legacy_helper_stays_deleted():
    """The legacy cu128_env.sh helper was deleted when its last users
    (the 8 legacy scripts) were ported onto stage2_common's s2_init. A
    revived copy would be an unaudited second bootstrap path — the
    entry-point audit no longer recognizes `cu128_sync_and_assert` as a
    guard, so any script that sourced it would fail the audit anyway; this
    pins the file itself."""
    assert not (SBATCH_DIR / "cu128_env.sh").exists(), (
        "cu128_env.sh is back — port the script onto stage2_common's "
        "s2_init instead (D29/D30)")


def test_at_most_one_audit_exemption_per_kind_in_the_repo():
    tagged = _exempt_tags(REPO / "scripts")
    for tag in EXEMPT_TAGS:
        hits = [where for t, where in tagged if t == tag]
        assert len(hits) <= 1, (
            f"only ONE command may carry {tag!r} (the proof job's bare-sync "
            f"demo / prod-lock twin sync respectively): {hits}")
    # The out-of-lockfile venv exemption is carried by exactly the petri
    # serving scripts (see OUT_OF_LOCKFILE_TAG) — every new carrier is a
    # DECISION, recorded here, never a copy-paste. Each of these runs
    # from the requirements-hash-versioned vllm venv (torch cu129)
    # rather than the project lockfile, because the vLLM stack cannot
    # lock in-project; that is the whole reason the tag exists. Adding a
    # name here without that being true is the drift this assertion is
    # for.
    ool = sorted({where.split(":")[0] for t, where in tagged
                  if t == OUT_OF_LOCKFILE_TAG})
    # A SET of recorded carriers (each a decision, commented), not an exact
    # list: branches land carriers independently (geom_sweep.sbatch on the
    # geom-sweep branch, lens_vectorize_gate.sbatch on lens-vectorize), so
    # a recorded carrier may be absent on a given checkout — what must never
    # happen is an UNRECORDED carrier, asserted below.
    allowed = {"attacker_pool.sbatch",         # RETIRED Slurm attacker-pool launcher (exits 64 before any cluster code): it served from the dedicated vllm attacker venv and never ran `uv sync`, so the cu128 guard does not apply; tagged since it was written, recorded 2026-09-06
               "bon_negprobe.sbatch",          # served steering-bites +/- gate
                   "geom_sweep.sbatch",            # per-model budget + served decode sweep (vllm lane venv + matplotlib, 2026-09-03)
                   "geom_budget_window.sbatch",    # budget-only walk of ONE model at ONE explicit window + S ladder (the MATH-500 17,664 re-measure): the SAME requirements-hash geom-sweep serving venv as geom_sweep.sbatch, no sweep phase
                   "lens_vectorize_gate.sbatch",   # the lens-vectorize GPU bit-identity gate (two serving venvs, same lane)
                   "lens_stacked_ab.sbatch",       # lens-grouped-scope: served A/B of stacked vs per-row reductions (same serving venvs as the gate, 2026-09-04)
                   "lens_stacked_microbench.sbatch",  # lens-grouped-scope: the hook microbench (the geom-sweep serving venv; torch only)
                   "petri_artifact_gate.sbatch",   # real-artifact gate
                   "petri_bon_fanout.sbatch",      # served-BoN fan-out gate
                   "petri_bon_pathA.sbatch",       # served-BoN path-A gate
                   "petri_bon_prompted.sbatch",    # served-BoN prompted gate
                   "petri_bon_smoke_integration.sbatch",  # served-BoN integration smoke
                   "petri_bon_speedup.sbatch",     # served-BoN speedup gate
                   "petri_cache_isolation_gate.sbatch",  # cache-iso gate
                   "petri_families_prompted.sbatch",  # non-BoN families engine A/B
                   "petri_families_served_integration.sbatch",  # fully-served sweep integration (pre-existing carrier, missed at the integrate-all merge)
                   "petri_gate.sbatch",            # the launch gate
                   "petri_gate_psr.sbatch",        # the live gate, psr_kl-scoped (2026-08-29)
                   "petri_lora_smoke.sbatch",      # lora o_proj-site serving smoke (2026-08-28)
                   "petri_probe.sbatch",           # vllm-lens compat probe
                   "petri_proxy.sbatch",           # the routing proxy
                   "petri_steered_smoke.sbatch",   # steered serving smoke
                   "qwen9b_probe.sbatch",          # qwen3.5-9B serving viability probe
                   "qwen_gate.sbatch",             # the qwen serving gate
                   "qwen_wave.sbatch",             # qwen served wave (one condition/job)
                   # the sdf-base-steering Slurm fit lane: all three run
                   # from / build the PREBUILT sdfbase-fit venv on shared
                   # storage (pip from the lockfile export + hybrid-stack
                   # pins; never a uv sync), so the frozen-sync guard is
                   # factually wrong for them; the two GPU jobs assert the
                   # cu128 torch on a live CUDA device before their
                   # workload (exit 92), the CPU builder asserts the wheel
                   # identity in its bake step
                   "sdfbase_derived_build.sbatch",  # derived pairs, prebuilt venv + runtime CUDA assert
                   "sdfbase_fit_cell.sbatch",      # fit cells, prebuilt venv + runtime CUDA assert
        # the four onboarding/HPO/probe carriers run from the PREBUILT sdfbase-fit / serving venvs
        # like sdfbase_fit_cell.sbatch and carry the same in-body cu128 runtime assert.
        "memprobe_fit.sbatch",           # FD 23 fit-memory ladder (sdfbase-fit venv)
        "fit_refusal_direction.sbatch",  # Arditi diff-in-means capture (sdfbase-fit venv)
        "hpo_trial.sbatch",              # HPO trial array (sdfbase-fit venv)
        "serving_probe.sbatch",          # native multi-LoRA serving probe (serving venv, vllm-requirements sha)
        # the data-parallel fit's REAL 2-GPU-vs-1-GPU parity smoke runs the production trainer from
        # the SAME prebuilt sdfbase-fit venv as sdfbase_fit_cell.sbatch, one leg plain and one under
        # torchrun, with the same in-body cu128 runtime assert (exit 92)
        "ddp_parity_smoke.sbatch",       # parity smoke (sdfbase-fit venv, torchrun leg)
                   "sdfbase_fit_env.sbatch",       # builds that venv (CPU-only; wheel-identity assert)
                   "swp_attacker_pool.sbatch",     # vllm attacker venv
                   "twosi_fits_llama.sbatch",      # prebaked twosweep-fit venv (pod-fanout convention)
                   }
    unrecorded = sorted(set(ool) - allowed)
    assert not unrecorded, (
        f"unrecorded {OUT_OF_LOCKFILE_TAG!r} carrier(s) {unrecorded} — a new carrier is a "
        "DECISION: add it to `allowed` with its reason, or drop the tag")


# ----------------------------------------------- submit-pin header audit
#
# s2_init (stage2_common.sh) refuses any submission that does not carry
# --comment=sha:<HEAD of the job's effective S2_REPO_DIR>, so a script's
# documented submit incantation must (a) exist, (b) carry the pin, and
# (c) derive the sha from the SAME checkout s2_init will compare against.
# Scope follows the source chain: a script is covered when it reaches
# stage2_common.sh directly OR transitively through any sourced helper in
# scripts/sbatch/ (capeval_common.sh today; a future third-level helper is
# covered the same way). The raw-substring scope this replaces never saw
# the capeval recipes — they mention only capeval_common.sh — so their
# headers were invisible to the sweep while the real submissions would
# have died at s2_init (caught 2026-08-18). The right-checkout check (c)
# is the other half of that bug class: a capeval pin derived from the
# cluster's repo mirror (`$CLUSTER_REPO_MIRROR`) passes the sha:<40-hex>
# format check and then FATALs on HEAD-equality whenever the two
# checkouts differ.

_REPO_ASSIGN_RE = re.compile(
    r"^(?:export\s+)?(S2_ROOT|S2_REPO_ROOT|S2_REPO_DIR)=(\"[^\"]*\"|\S+)$")
_SOURCE_RE = re.compile(r"(?:^|[\s;&|(])(?:source|\.)\s+(\S+)")
_DOC_SUBMIT_RE = re.compile(r"^#\s*(?:\d+\.\s+)?(sbatch\s.*)$")
_DOC_PIN_RE = re.compile(r"--comment=sha:\$\(git -C (\S+) rev-parse HEAD\)")
_REFUSE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):\?[^{}]*\}")
_BRACED_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}(?![A-Za-z0-9_])")


def _canon_path(v: str) -> str:
    """A job variable's refusal form ${NAME:?message} and its braced form ${NAME} both name $NAME."""
    return _BRACED_RE.sub(r"$\1", _REFUSE_RE.sub(r"$\1", v))


def _resolve_repo_value(value: str, env: dict[str, str]) -> str:
    """Statically resolve a repo-variable assignment's RHS the way a
    default submission evaluates it: quotes dropped; ${NAME:-default}
    takes the already-derived value when NAME is one we track and set,
    and the default otherwise (seam variables like CAPEVAL_REPO_DIR are
    unset at submit time); $S2_ROOT / $S2_REPO_ROOT / $S2_REPO_DIR substitute; every
    other variable ($SHARED, $USER) stays symbolic, matching how the
    headers spell the incantation."""
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        v = v[1:-1]
    v = _canon_path(v)
    brace = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^{}]*)\}")
    while (m := brace.search(v)) is not None:
        v = v[:m.start()] + env.get(m.group(1), m.group(2)) + v[m.end():]
    for name, val in env.items():
        v = v.replace("${%s}" % name, val).replace("$" + name, val)
    return v


def _walk_submit_env(path: Path, sbatch_dir: Path, env: dict[str, str],
                     seen: set[Path]) -> bool:
    """Walk a script's ACTIVE commands in execution order, following every
    sourced helper that lives in sbatch_dir (matched by basename — the
    scripts source cluster paths whose tail file is this repo's copy).
    Records S2_REPO_ROOT / S2_REPO_DIR assignments into env; returns
    whether the chain reaches stage2_common.sh (= this submission passes
    through s2_init's pin gate)."""
    reached = path.name == "stage2_common.sh"
    for _, cmd, _ in _logical_lines(path.read_text()):
        m = _REPO_ASSIGN_RE.match(cmd)
        if m:
            env[m.group(1)] = _resolve_repo_value(m.group(2), env)
            continue
        for sm in _SOURCE_RE.finditer(cmd):
            base = Path(sm.group(1).strip("'\"")).name
            helper = sbatch_dir / base
            if (base.endswith(".sh") and helper.is_file()
                    and helper != path and helper not in seen):
                seen.add(helper)
                if _walk_submit_env(helper, sbatch_dir, env, seen):
                    reached = True
    return reached


def _documented_submits(text: str) -> list[tuple[int, str]]:
    """The header submit incantations: comment lines whose command word is
    `sbatch` (an enumeration prefix like '4.' allowed), with backslash
    continuations joined across the following comment lines, and kept only
    when the joined command names a *.sbatch script — which drops prose
    mentions like 'never via sbatch --export'."""
    lines = text.split("\n")
    out: list[tuple[int, str]] = []
    i = 0
    while i < len(lines):
        m = _DOC_SUBMIT_RE.match(lines[i])
        if not m:
            i += 1
            continue
        start, cmd = i + 1, m.group(1).strip()
        while cmd.endswith("\\") and i + 1 < len(lines):
            i += 1
            cont = lines[i].lstrip()
            if not cont.startswith("#"):
                break
            cmd = cmd[:-1].rstrip() + " " + cont.lstrip("#").strip()
        if re.search(r"\S+\.sbatch\b", cmd):
            out.append((start, cmd))
        i += 1
    return out


def _pin_header_violations(sbatch_dir: Path) -> tuple[list[str], int]:
    """Violations of the submit-pin header contract across every file that
    reaches stage2_common.sh, plus the count of submit-documenting files
    checked. Helpers (.sh) are checked when they document a submit line;
    job scripts (.sbatch) must document one."""
    violations: list[str] = []
    checked = 0
    for f in sorted(p for p in sbatch_dir.iterdir()
                    if p.is_file() and p.suffix in (".sbatch", ".sh")):
        env: dict[str, str] = {}
        if not _walk_submit_env(f, sbatch_dir, env, {f}):
            continue  # never hits s2_init's pin gate (fa_build, petri_*)
        assert "S2_REPO_DIR" in env, (
            f"{f.name}: reached stage2_common.sh but its S2_REPO_DIR "
            "default was not discoverable — the audit's static model "
            "needs updating")
        expected = env["S2_REPO_DIR"].rstrip("/")
        docs = _documented_submits(f.read_text())
        if not docs:
            if f.suffix == ".sbatch":
                violations.append(
                    f"{f.name}: no documented submit line in header")
            continue
        checked += 1
        for lineno, cmd in docs:
            where = f"{f.name}:{lineno}"
            if "--comment=sha:" not in cmd:
                violations.append(f"{where}: unpinned submit line")
                continue
            m = _DOC_PIN_RE.search(cmd)
            if m is None:
                violations.append(
                    f"{where}: pin is not derived from a checkout — want "
                    "--comment=sha:$(git -C <effective S2_REPO_DIR> "
                    "rev-parse HEAD)")
            elif _canon_path(m.group(1)).rstrip("/") != expected:
                violations.append(
                    f"{where}: pin derives from '{m.group(1)}' but this "
                    f"job's effective S2_REPO_DIR is '{expected}' — that "
                    "pin passes the format check and FATALs on "
                    "HEAD-equality whenever the checkouts differ")
    return violations, checked


def test_stage2_common_sbatch_headers_document_the_pin():
    """Round 3, widened: every submit incantation documented by a script
    that reaches s2_init's pin gate — directly or through sourced helpers
    like capeval_common.sh — must carry the mandatory --comment=sha: pin
    AND derive it from the job's effective S2_REPO_DIR. s2_init refuses
    unpinned submissions and FATALs wrong-checkout pins on HEAD-equality,
    so a bad incantation in a header is a job that dies at start."""
    violations, checked = _pin_header_violations(SBATCH_DIR)
    assert violations == []
    # the stage2/jailbreak/proof scripts + the ported legacy ones + the 3
    # capeval recipes + the 2 sourced helpers' own documented incantations:
    # 35 since the stage-2 IFEval-type matrix and its sibling scripts were
    # removed with the third-party suites they ran, and the capeval vLLM
    # bring-up script with them.
    assert checked >= 35, (
        f"only {checked} pin-gated submit-documenting scripts found — did "
        "the source-chain scope regress?")


# ------------------------------------------------------------- lexer pins


def test_shell_lexer_semantics():
    """The lexer models what bash executes: comment lines and trailing
    comments go (an EXEMPT_TAG in one is still recorded), heredoc bodies
    go, backslash continuations join, and quotes are respected."""
    src = textwrap.dedent("""\
        # uv sync --frozen in a comment is not a command
        uv \\
          sync --frozen --extra cu128 || exit 1
        cat <<'EOF'
        uv sync in a heredoc body is data, not a command
        EOF
        read -r x <<< "$y"
        echo 'a # in quotes is not a comment'
        echo done  # audit-exempt: loud-failure-demo
        """)
    cmds = _logical_lines(src)
    assert [re.sub(r"\s+", " ", c) for _, c, _ in cmds] == [
        "uv sync --frozen --extra cu128 || exit 1",
        "cat <<'EOF'",
        # a here-STRING is not a heredoc: nothing after it is swallowed
        # (a bogus '$y' heredoc here once hid the rest of the file)
        'read -r x <<< "$y"',
        "echo 'a # in quotes is not a comment'",
        "echo done",
    ]
    assert [(n, ex) for n, _, ex in cmds] == [
        (2, ""), (4, ""), (7, ""), (8, ""), (9, EXEMPT_TAG)]


def test_fa_build_payload_is_audited_as_its_own_shell():
    """fa_build's guard lives INSIDE an `srun bash -c` build shell: the
    splice must surface its frozen cu128 sync and runtime assert as active
    commands, in a context separate from the outer script (so the
    payload's `set -e` never vouches for outer syncs, and vice versa)."""
    entries = _script_entries(SBATCH_DIR / "fa_build.sbatch")
    payload = [e for e in entries if "[bash -c]" in e.where]
    assert payload, "fa_build's srun bash -c payload was not spliced in"
    assert any(_SYNC_RE.search(e.text) and "--frozen" in e.text
               and "--extra cu128" in e.text for e in payload)
    assert any(all(m in e.text for m in _GUARD_MARKERS) for e in payload)
    outer_ctxs = {e.ctx for e in entries if "[bash -c]" not in e.where}
    assert all(e.ctx not in outer_ctxs for e in payload)


# ---------------------------------------------------------- mutation tests
#
# Each test copies the REAL scripts/sbatch, applies one regression the old
# raw-substring audit provably missed (or must keep catching), and asserts
# the audit now fails — so the audit's claims stay executable.


def _mutated_sbatch(tmp_path: Path, mutate) -> Path:
    dst = tmp_path / "sbatch"
    shutil.copytree(SBATCH_DIR, dst)
    mutate(dst)
    return dst


def _edit(dst: Path, name: str, old: str, new: str) -> None:
    p = dst / name
    text = p.read_text()
    assert old in text, f"mutation anchor missing from {name}: {old!r}"
    p.write_text(text.replace(old, new, 1))


def test_audit_catches_a_fully_commented_out_guard(tmp_path):
    """The old substring scan matched the assert markers inside comments.
    Commenting out fa_build's whole build block — markers intact as
    text — must now fail the entry-point audit, as must a commented-out
    s2_init call in a ported legacy script."""
    def comment_block(dst):
        p = dst / "fa_build.sbatch"
        lines = p.read_text().splitlines(keepends=True)
        start = next(i for i, l in enumerate(lines)
                     if l.startswith("srun bash -c '"))
        end = next(i for i in range(start + 1, len(lines))
                   if lines[i].rstrip() == "'")
        p.write_text("".join(
            "#" + l if start <= i <= end else l
            for i, l in enumerate(lines)))

    dst = _mutated_sbatch(tmp_path, comment_block)
    assert any(v.startswith("fa_build.sbatch: no ACTIVE cu128 guard")
               for v in _entry_point_violations(dst))

    dst2 = _mutated_sbatch(tmp_path / "legacy", lambda d: _edit(
        d, "frontier.sbatch", '\ns2_init "$@"\n',
        '\n# s2_init "$@"\n'))
    assert any(v.startswith("frontier.sbatch: no ACTIVE cu128 guard")
               for v in _entry_point_violations(dst2))


def test_audit_catches_a_guard_moved_after_the_workload(tmp_path):
    """A guard that runs after the workload guards nothing — the workload
    already ran on whatever env was lying around."""
    dst = _mutated_sbatch(tmp_path, lambda d: _edit(
        d, "frontier.sbatch", 's2_init "$@"\n',
        'srun uv run --no-sync python -c pass\ns2_init "$@"\n'))
    assert any("runs BEFORE the cu128 guard" in v
               for v in _entry_point_violations(dst))


def test_audit_sees_through_backslash_split_syncs(tmp_path):
    """`uv \\<newline>sync` is the same command to bash; the old regex
    never saw it, so a split sync escaped every check. Split-and-degraded
    must fail; split-but-correct must still pass (the join is real
    analysis, not a ban on continuations)."""
    anchor = 'uv sync --frozen --extra cu128 "${uv_extra_flags[@]}"'
    dst = _mutated_sbatch(tmp_path, lambda d: _edit(
        d, "stage2_common.sh", anchor,
        'uv \\\n  sync --frozen "${uv_extra_flags[@]}"'))
    assert any("flavor-less uv sync" in v for v in _flavor_violations(dst))

    dst_ok = _mutated_sbatch(tmp_path / "ok", lambda d: _edit(
        d, "stage2_common.sh", anchor,
        'uv \\\n  sync --frozen --extra cu128 "${uv_extra_flags[@]}"'))
    assert _flavor_violations(dst_ok) == []
    assert _entry_point_violations(dst_ok) == []


def test_audit_catches_unfrozen_fallback_reintroduction(tmp_path):
    """The `|| uv sync` fallbacks were removed (both-halves-fail ran stale
    envs silently); reintroducing one must trip both the fallback ban and
    the per-segment --frozen check."""
    dst = _mutated_sbatch(tmp_path, lambda d: _edit(
        d, "stage2_common.sh",
        'uv sync --frozen --extra cu128 "${uv_extra_flags[@]}"',
        'uv sync --frozen --extra cu128 "${uv_extra_flags[@]}"'
        ' || uv sync --extra cu128'))
    v = _flavor_violations(dst)
    assert any("unfrozen `|| uv sync` fallback" in x for x in v)
    assert any("unfrozen uv sync" in x for x in v)


def test_audit_catches_a_missing_no_sync(tmp_path):
    dst = _mutated_sbatch(tmp_path, lambda d: _edit(
        d, "frontier.sbatch", "s2_run uv run --no-sync python",
        "s2_run uv run python"))
    assert any("without --no-sync" in v for v in _flavor_violations(dst))


def test_audit_catches_a_set_e_set_plus_e_window(tmp_path):
    """The old audit accepted ANY earlier textual `set -e` as rc handling.
    A `set +e` between it and the sync voids errexit — fa_build's payload
    sync has no other handler, so this must now be a violation."""
    dst = _mutated_sbatch(tmp_path, lambda d: _edit(
        d, "fa_build.sbatch", "  set -e\n", "  set -e\n  set +e\n"))
    assert any("uv sync result is not checked" in v
               for v in _flavor_violations(dst))


def test_audit_catches_a_second_exempt_tag(tmp_path):
    """The exemption tags waive sync checks, so each kind's count is capped
    at one repo-wide; a second tag of a kind must be visible to the
    counter."""
    dst = _mutated_sbatch(tmp_path, lambda d: _edit(
        d, "frontier.sbatch", 's2_init "$@"\n',
        f's2_init "$@"  {EXEMPT_TAG}\n'))
    demo = [w for t, w in _exempt_tags(dst) if t == EXEMPT_TAG]
    assert len(demo) == 2, (
        "the second loud-failure-demo tag must fail the per-kind cap")


def test_prod_twin_tag_waives_only_the_cu128_extra(tmp_path):
    """PROD_TWIN_TAG exists for ONE sync that is flavor-less by design
    (production lockfile, torch as base dep). It must not become a
    general escape hatch: dropping --frozen on the tagged sync is still a
    violation."""
    dst = _mutated_sbatch(tmp_path, lambda d: _edit(
        d, "torch_split_proof.sbatch",
        f"uv sync --frozen --no-install-project  {PROD_TWIN_TAG}",
        f"uv sync --no-install-project  {PROD_TWIN_TAG}"))
    assert any("unfrozen uv sync" in v for v in _flavor_violations(dst))
    # ...while the untouched tagged sync is accepted (covered by
    # test_sbatch_flavor_audit on the real tree, asserted here for locality):
    assert not any("flavor-less" in v and "torch_split_proof" in v
                   for v in _flavor_violations(SBATCH_DIR))


def test_pin_header_audit_catches_a_stripped_capeval_pin(tmp_path):
    """The capeval recipes reach stage2_common.sh only THROUGH
    capeval_common.sh — the old raw-substring scope never saw them, so an
    unpinned capeval submit line passed the sweep while s2_init would have
    FATALed the real submission (the exact blind spot of 2026-08-18).
    Stripping the pin from one recipe must now fail the audit."""
    dst = _mutated_sbatch(tmp_path, lambda d: _edit(
        d, "capeval_s2.sbatch",
        "#   sbatch --comment=sha:$(git -C ${SISTEER_PROJECT_ROOT}-capeval-repo"
        " rev-parse HEAD) \\\n"
        "#          --qos=<gpu-avail> --array=$CAPEVAL_S2_ARRAY_PILOT",
        "#   sbatch --qos=<gpu-avail> --array=$CAPEVAL_S2_ARRAY_PILOT"))
    violations, _ = _pin_header_violations(dst)
    assert any("capeval_s2.sbatch" in v and "unpinned submit line" in v
               for v in violations)


def test_pin_header_audit_catches_a_wrong_checkout_pin(tmp_path):
    """The other half of the bug class: a capeval pin derived from the
    cluster's repo mirror passes the sha:<40-hex> format check and then
    FATALs on HEAD-equality whenever the two checkouts differ (the norm).
    The audit must resolve the effective S2_REPO_DIR THROUGH
    capeval_common.sh's export and refuse the mismatch."""
    dst = _mutated_sbatch(tmp_path, lambda d: _edit(
        d, "capeval_calib.sbatch",
        "git -C ${SISTEER_PROJECT_ROOT}-capeval-repo rev-parse HEAD",
        "git -C $SISTEER_PROJECT_ROOT/repo rev-parse HEAD"))
    violations, _ = _pin_header_violations(dst)
    assert any("capeval_calib.sbatch" in v
               and "effective S2_REPO_DIR is '$SISTEER_PROJECT_ROOT-capeval-repo'"
               in v for v in violations)
    # the same holds for the isolated tsplit lane, whose S2_REPO_ROOT
    # override the audit must also follow:
    dst2 = _mutated_sbatch(tmp_path / "tsplit", lambda d: _edit(
        d, "torch_split_proof.sbatch",
        "git -C ${SISTEER_PROJECT_ROOT}-tsplit/repo rev-parse HEAD",
        "git -C $SISTEER_PROJECT_ROOT/repo rev-parse HEAD"))
    violations2, _ = _pin_header_violations(dst2)
    assert any("torch_split_proof.sbatch" in v
               and "'$SISTEER_PROJECT_ROOT-tsplit/repo'" in v
               for v in violations2)


def test_pin_header_audit_follows_new_indirection_levels(tmp_path):
    """The scope is the CLASS, not the instance: a future recipe that
    reaches stage2_common.sh through a THIRD-level helper (recipe ->
    new helper -> capeval_common.sh -> stage2_common.sh) is in scope
    without anyone updating a hard-coded helper list."""
    def add_level(d: Path) -> None:
        (d / "capeval_extra_common.sh").write_text(
            "# hypothetical third-level helper\n"
            'source "${SISTEER_PROJECT_ROOT}-capeval-repo/scripts/sbatch/'
            'capeval_common.sh"\n')
        (d / "capeval_extra.sbatch").write_text(
            "#!/bin/bash\n"
            "#   sbatch --qos=x scripts/sbatch/capeval_extra.sbatch\n"
            'source "${SISTEER_PROJECT_ROOT}-capeval-repo/scripts/sbatch/'
            'capeval_extra_common.sh"\n'
            's2_init "$@"\n')

    dst = _mutated_sbatch(tmp_path, add_level)
    violations, _ = _pin_header_violations(dst)
    assert any(v.startswith("capeval_extra.sbatch")
               and "unpinned submit line" in v for v in violations)

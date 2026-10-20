"""`reproduce/02_prepare_model.sh --step pools` is re-runnable past its ctl2 step.

The chain used to run `build_ctl2_pools.py build … --out <pools_ctl2>` unconditionally, and that builder REFUSES a
non-empty --out by contract, so a re-run after a later failure died ~27 minutes in. Now `ctl2_pools_reuse.py decide`
answers ONE line (`REUSE <dir>` / `BUILD <dir>`) the chain reads with `read -r`, the build runs only in the BUILD branch
(into the canonical dir, or a fresh stamped sibling when an incomplete build sits there), `commit` records the finished dir
as CURRENT, and the hard-link loop re-points the volume at `$CTL2_DIR`. Both tool calls take the constv2-mix-full manifest
the chain JUST rebuilt (`--budget-manifest`): a recorded build cut against a different budget (its `budget_source`'s
pool_sha256 / official_tokens — never the manifest FILE's sha, which changes with the checkout) refuses at the decision, and a
seeded rebuild beside a partial must reproduce the partial's recorded pool sha256s before `commit` re-points anything. The
recorded build's RECIPE (its manifests' `recipe` — salts, control-input shas, tokenizer pins, version; never the checkout's
commit) must equal today's builder's too: a complete CURRENT build cut under another recipe is BUILT beside, never reused,
and the volume is re-pointed at the fresh selection.

Two halves, both against the NEUTRAL site example (no cluster, no ssh):
(a) the `--dry-run` ssh line carries the new shape and no unexpanded placeholder.
(b) EXECUTION FORM — the `bash -c '…'` body is extracted from that line and RUN under fakes (a fake fit-venv python that
    dispatches on its script argument: the real ctl2_pools_reuse.py for the decision, synthetic builders for the rest),
    four times over one state: BUILD into the root -> REUSE -> BUILD into a stamped sibling past a partial root -> REUSE
    the sibling; then two refusals: the mix-full pool "changed" under a complete CURRENT build (no builder call, links
    untouched) and a seeded "rebuild" beside a partial that yields different bytes (commit refuses, no pointer, links
    untouched, sidecar left). The builder call log and the volume links' inodes are the evidence.
(c) THE STAGE LIVE (no --dry-run) through the `local:` seam (`CLUSTER_SSH=local:<dir>` makes repro_ssh run every remote
    string with `bash -c` on this host), so the pools step's DONE-marker check runs for real:
    the marker written after a chain certifies `recipe=<digest> budget=<pool_sha256>:<official_tokens>` — what
    `ctl2_pools_reuse.py identity` read off the volume — and a later run honours it only when both are today's; a marker
    certifying another recipe or budget, or none (a legacy marker), is ARCHIVED under <state>/…/superseded/<utc>/DONE.pools
    (a FRESH microsecond-stamped directory per archival — lib/common.sh repro_superseded_dir; moved, never deleted) and the
    chain re-decides by name: REUSE when the CURRENT build's recipe is today's, BUILD into a
    stamped sibling when the build itself was cut under another recipe, and the new marker certifies the new identities.
    The marker is LOCAL to a checkout's state dir while the build it certifies is SHARED: so the check
    also reads the RECORDED side — the CURRENT build's own recipe / budget, which another checkout may have rebuilt and
    re-pointed — and `volume_links`, whether the volume's ctl2 files are the CURRENT build's inodes (`identity --volume`);
    a recorded disagreement, a build with no readable record, or a volume holding a STALE copy or a superseded build's
    link is each said by name, archived, and re-decided; and the mark after a chain refuses unless the volume holds the
    build. The volume holds the build as hard links (`volume_links=ok`, the marker's `links=hard`) OR as byte-equal copies
    (`ok:copy` / `links=copy` — the loop is `ln -f … || cp -f …`, so a site whose volume is on another filesystem than
    the build copies; an `ok`-only mark made such a site unable to ever write DONE.pools). The
    cross-device flow runs LIVE too, under a fake `ln` on PATH that answers EXDEV for the volume dir (deterministic on
    every host), and once more against a REAL second filesystem (/dev/shm) where one is writable: run 1 completes with
    `links=copy`, run 2 honours it, a stale copy is refused by name and REUSE's `cp -f` refreshes it to the CURRENT bytes.
Synthetic everything (tests/test_no_personal_identifiers.py scans this tree)."""
from __future__ import annotations

import grp
import hashlib
import json
import os
import pwd
import re
import shutil
import stat
import sys
import subprocess
import tempfile
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO, VENV_PY, assert_no_tokens, run_stage  # noqa: E402
from test_front_door_dry_run import PLACEHOLDER  # noqa: E402  (binds the regex only — no test function of that module)

STAGE = "02_prepare_model.sh"
MODEL = "qwen35_4b"
CELLS = ("constv2ctl2-heldout", "constv2ctl2-full")
POOL_FILES = tuple(f"pairs_{c}{suf}" for c in CELLS for suf in (".jsonl", ".manifest.json"))
TOOL = "scripts/sdfbase/ctl2_pools_reuse.py"
BUDGET_MANIFEST = "pairs_constv2-mix-full.manifest.json"


# ------------------------------------------------------------------------------------------------------------------
# sites
# ------------------------------------------------------------------------------------------------------------------
def _site(root: Path, shared: Path | None = None, ssh: str | None = None) -> Path:
    site = root / "site"
    shutil.copytree(REPRO / "site.example", site)
    with open(site / "config.env", "a") as fh:
        fh.write(f"\nSISTEER_STATE_DIR={root / 'state'}\nSISTEER_PYTHON={VENV_PY}\nSISTEER_REGISTRY_ROOT={root / 'registry'}\n")
    env_file = site / "clusters/cluster_a.env"
    if shared is not None:
        # the executing half needs the primary cluster's shared root somewhere this process may write
        text = env_file.read_text()
        new, n = re.subn(r"^CLUSTER_SHARED_ROOT=\S+", f"CLUSTER_SHARED_ROOT={shared}", text, count=1, flags=re.M)
        assert n == 1, text[:300]
        env_file.write_text(new)
    if ssh is not None:
        # the live half: `local:<dir>` makes repro_ssh run the remote string with `bash -c` on THIS host (lib/common.sh's seam)
        text = env_file.read_text()
        new, n = re.subn(r"^CLUSTER_SSH=\S+", f"CLUSTER_SSH={ssh}", text, count=1, flags=re.M)
        assert n == 1, text[:300]
        env_file.write_text(new)
    return site


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    return _site(tmp_path_factory.mktemp("s02rr"))


@pytest.fixture(scope="module")
def exec_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("s02rrx")
    (root / "shared").mkdir()
    _site(root, shared=root / "shared")
    return root


def _scan(out: str, root: Path) -> str:
    """The runner's OWN paths and account are not hard-coded identifiers: normalise them before the token scan."""
    me = pwd.getpwuid(os.getuid()).pw_name
    mygrp = grp.getgrgid(os.getgid()).gr_name
    s = out.replace(str(REPO), "<repo>").replace(sys.executable, "<python>").replace(VENV_PY, "<python>")
    s = s.replace(str(root), "<tmp>").replace(os.path.expanduser("~"), "<home>")
    return re.sub(rf"\b({re.escape(me)}|{re.escape(mygrp)})\b", "<user>", s)


def _dry_ssh_line(site: Path, env=None) -> str:
    r = run_stage(STAGE, "--dry-run", "--step", "pools", "--model", MODEL, site=site, env=env, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    assert_no_tokens(_scan(out, site.parent), STAGE)
    lines = [ln for ln in out.splitlines() if ln.startswith("DRY-RUN: ssh ")]
    assert len(lines) == 1, lines
    return lines[0]


def _chain_body(ssh_line: str) -> str:
    """The body of the remote `bash -c '…'` (the printed line closes it with the chain's quote + the DRY-RUN wrapper's)."""
    _, sep, rest = ssh_line.partition("bash -c '")
    assert sep, ssh_line[:200]
    assert rest.endswith("''"), ssh_line[-120:]
    body = rest[:-2]
    assert "'" not in body, "a single quote inside the chain would end the remote quoting early"
    return body


# ------------------------------------------------------------------------------------------------------------------
# (a) the dry-run line
# ------------------------------------------------------------------------------------------------------------------
def test_dry_run_line_decides_builds_conditionally_commits_and_links_the_current_dir(site):
    line = _dry_ssh_line(site)
    assert not PLACEHOLDER.search(line), line[:400]
    body = _chain_body(line)
    ctl2 = "/scratch/example/shared/exp/reproduce/qwen35_4b/02_prepare_model/pools_ctl2"
    pools = "/scratch/example/shared/exp/reproduce/qwen35_4b/02_prepare_model/pools"
    # the decision: one line from the fit venv's interpreter, read as two words, a refusal stops the chain
    m_dec = re.search(rf"CTL2_DECISION=\$\((\S+/envs/sdfbase-fit/bin/python) {re.escape(TOOL)} decide --root {re.escape(ctl2)} "
                      rf"--budget-manifest {re.escape(pools)}/{re.escape(BUDGET_MANIFEST)}\) \|\| exit 3; "
                      rf'read -r CTL2_MODE CTL2_DIR <<<"\$CTL2_DECISION"; echo "ctl2: \$CTL2_MODE \$CTL2_DIR"; case "\$CTL2_MODE" in ', body)
    assert m_dec, body[:1500]
    py = m_dec.group(1)
    # the case: REUSE says so and builds nothing; BUILD mkdirs the decided dir, builds INTO it, preflights IT, commits IT;
    # anything else refuses. Inside the branch the commands are `;`-separated so `set -e` stops at the first failing builder.
    reuse = re.search(r'REUSE\) echo "ctl2: reusing the complete, preflight-PASSED, byte-verified build at \$CTL2_DIR \(D98 i\)";; BUILD\) ', body)
    assert reuse, body[:1500]
    build_branch = body[reuse.end():body.index(";; *) ", reuse.end())]
    assert build_branch.startswith('mkdir -p "$CTL2_DIR"; ')
    steps = build_branch.split("; ")
    assert steps[1].startswith(f"{py} scripts/sdfbase/build_ctl2_pools.py build ") and ' --out "$CTL2_DIR" ' in steps[1] + " ", steps[1]
    assert f"--budget-manifest {pools}/pairs_constv2-mix-full.manifest.json" in steps[1] and f"--split-out {pools}/ctl2_heldout_split.json" in steps[1]
    assert steps[2] == f'{py} scripts/sdfbase/build_pools.py preflight --pools "$CTL2_DIR"', steps[2]
    assert steps[3] == f'{py} {TOOL} commit --root {ctl2} --dir "$CTL2_DIR" --budget-manifest {pools}/{BUDGET_MANIFEST}', steps[3]
    # decide, the builder and commit read the SAME rebuilt mix-full manifest (the budget identity is read once per run)
    assert body.count(f"--budget-manifest {pools}/{BUDGET_MANIFEST}") == 3
    assert len(steps) == 4, steps
    assert '*) echo "REFUSED: ctl2_pools_reuse.py decide answered $CTL2_DECISION" >&2; exit 3;; esac; ' in body
    # no fixed ctl2 target anywhere: the builder and its preflight name $CTL2_DIR only, and the early mkdir no longer pre-creates it
    assert f"--out {ctl2}" not in body and f"preflight --pools {ctl2}" not in body
    assert body.count("build_ctl2_pools.py build") == 1
    mk = re.search(r"; mkdir -p (\S+) (\S+); ", body)
    assert mk and mk.group(1) == pools and mk.group(2).endswith("/vol/pool/qwen35_4b_base"), mk.group(0) if mk else body[:600]
    # the link loop reads the CURRENT dir, after the case and before the token sidecars
    loop = re.search(rf"for f in {re.escape(pools)}/pairs_\*\.jsonl {re.escape(pools)}/pairs_\*\.manifest\.json "
                     r'\$CTL2_DIR/pairs_\*\.jsonl \$CTL2_DIR/pairs_\*\.manifest\.json; do ln -f "\$f" (\S+)/ 2>/dev/null \|\| cp -f "\$f" \1/; done', body)
    assert loop, body[-2500:]
    assert body.index("esac; ") < loop.start() < body.index("precompute-tokens")
    # the untouched steps of the chain, in order, before the decision
    for i, needle in enumerate(("build_heldout_split.py", "build_pools.py build ", f"build_pools.py preflight --pools {pools};")):
        assert body.index(needle) < m_dec.start(), needle
    # the header + the say line name the skip
    text = (REPRO / STAGE).read_text()
    assert re.search(r"^#   pools .*SKIPPED when a complete,\n#\s+preflight-PASSED, byte-verified ctl2 build exists", text, re.M), "header"


def test_dry_run_line_honours_an_empty_cpu_srun_and_the_two_models(site):
    # the executing half relies on this: SISTEER_CPU_SRUN="" gives the bare `bash -c` (04_fit_objects.sh's semantics)
    line = _dry_ssh_line(site, env={"SISTEER_CPU_SRUN": ""})
    assert line.startswith("DRY-RUN: ssh cluster-a-login ' bash -c '"), line[:120]
    r = run_stage(STAGE, "--dry-run", "--step", "pools", "--model", "qwen35_4b_demo", site=site, timeout=600)
    assert r.returncode == 0, (r.stdout + r.stderr)[-1500:]
    assert (f"{TOOL} decide --root /scratch/example/shared/exp/reproduce/qwen35_4b_demo/02_prepare_model/pools_ctl2 "
            f"--budget-manifest /scratch/example/shared/exp/reproduce/qwen35_4b_demo/02_prepare_model/pools/{BUDGET_MANIFEST})") in r.stdout


# ------------------------------------------------------------------------------------------------------------------
# (b) execution form: run the extracted chain under fakes, four times over one state
# ------------------------------------------------------------------------------------------------------------------
FAKE_TOOLS = textwrap.dedent('''\
    """The synthetic builders the fake fit-venv python runs for every script but ctl2_pools_reuse.py. Each call is logged
    (FAKE_LOG, one JSON line) and produces exactly what the later chain steps need."""
    import hashlib, json, os, sys
    from pathlib import Path

    CELLS = ("constv2ctl2-heldout", "constv2ctl2-full")
    args = sys.argv[1:]
    script, sub = args[0], (args[1] if len(args) > 1 else "")
    with open(os.environ["FAKE_LOG"], "a") as fh:
        fh.write(json.dumps({"script": script, "sub": sub, "argv": args}) + "\\n")

    def opt(name):
        return args[args.index(name) + 1]

    def write_pool(d, cell, prefix, n=3, salt="", extra=None):
        rows = [{"doc_id": f"{prefix}{i:04d}", "instruction_id": cell, "prompt": "p", "pos": f"x{salt}{i}"} for i in range(n)]
        data = "".join(json.dumps(r) + "\\n" for r in rows).encode()          # deterministic: a rebuild reproduces the bytes
        (d / f"pairs_{cell}.jsonl").write_bytes(data)
        m = {"cell": cell, "pool_sha256": hashlib.sha256(data).hexdigest(), "n_docs": n,
             "doc_ids_in_order": [r["doc_id"] for r in rows], "official_tokens": 10 * n,
             "generator": {"commit": os.environ.get("FAKE_GENERATOR_SHA", "a" * 40)},   # the checkout's sha: the FILE changes per run
             **(extra or {})}
        (d / f"pairs_{cell}.manifest.json").write_text(json.dumps(m, indent=1) + "\\n")
        return m

    if script.endswith("build_heldout_split.py"):
        out = Path(opt("--out")); out.parent.mkdir(parents=True, exist_ok=True); out.write_text("{}\\n")
    elif script.endswith("build_pools.py") and sub == "build":
        out = Path(opt("--out")); out.mkdir(parents=True, exist_ok=True)
        for cell in opt("--cells").split(","):
            # the constitutional pools overwrite in place; FAKE_MIX_SALT "changes" the mix-full pool's bytes (a new budget)
            write_pool(out, cell, "doc-", salt=os.environ.get("FAKE_MIX_SALT", "") if cell == "constv2-mix-full" else "")
    elif script.endswith("build_pools.py") and sub == "preflight":
        d = Path(opt("--pools")); pools = sorted(d.glob("pairs_*.jsonl"))
        if not pools:
            print(f"REFUSED: no pairs_*.jsonl pools under {d}", file=sys.stderr); sys.exit(2)
        for p in pools:                                                          # build_pools.run_preflight's stamps
            mp = p.with_name(p.stem + ".manifest.json"); m = json.loads(mp.read_text())
            m["preflight"] = {"max_len": 8192, "tokenizers": {"fake": {"repo": "TESTHOST/fake", "revision": "0" * 40}}}
            mp.write_text(json.dumps(m, indent=1) + "\\n")
        (d / "preflight_report.json").write_text(json.dumps({"max_len": 8192, "violations": []}) + "\\n")
    elif script.endswith("build_ctl2_pools.py") and sub == "build":
        out = Path(opt("--out"))
        if out.exists() and any(out.iterdir()):                                 # refuse_nonempty_out, the real contract
            print(f"REFUSED: --out {out} is not empty", file=sys.stderr); sys.exit(2)
        bm = Path(opt("--budget-manifest"))
        if not bm.is_file():                                                     # read_budget, the real contract
            print(f"REFUSED: budget manifest {bm} missing", file=sys.stderr); sys.exit(2)
        b = json.loads(bm.read_text())
        budget_source = {"cell": b["cell"], "manifest": str(bm), "manifest_sha256": hashlib.sha256(bm.read_bytes()).hexdigest(),
                         "pool_sha256": b["pool_sha256"], "official_tokens": b["official_tokens"]}
        out.mkdir(parents=True, exist_ok=True)
        # the RECIPE the real builder records (build_ctl2_pools.recipe_identity, through the REAL module of this checkout,
        # so the chain's REUSE path on the next run compares an EQUAL recipe): the release's pinned bytes, the builder's
        # default salts / split (the chain line passes no --salt), this checkout's tokenizer pins; a fake rig's provenance
        sys.path.insert(0, os.path.join(os.environ["FAKE_REPO"], "scripts", "sdfbase"))
        import build_ctl2_pools as b2
        recipe = b2.recipe_identity(shas=dict(b2.RELEASE_SHAS), pinned=True, release_label=b2.CORPUS_RELEASE,
                                    salt=opt("--salt") if "--salt" in args else b2.hsplit.SALT_HELDOUT,
                                    eval_salt=opt("--eval-salt") if "--eval-salt" in args else b2.hsplit.SALT_EVAL,
                                    frac=b2.hsplit.FRAC, eval_n=b2.hsplit.EVAL_N, tokenizer_pins=b2.load_tokenizer_pins())
        recipe["tokenizers"]["rig"] = {"fake": True}
        perturb = os.environ.get("FAKE_CTL2_PERTURB", "")                        # a cell whose "seeded" rebuild yields other bytes
        ms = {cell: write_pool(out, cell, "ctl2-doc-", salt="perturbed" if cell == perturb else "",
                               extra={"recipe": recipe, **({"budget_source": budget_source} if cell == "constv2ctl2-full" else {})})
              for cell in CELLS}
        (out / "pools_summary.json").write_text(json.dumps({"recipe": recipe, "cells": {c: {"pool_sha256": m["pool_sha256"]} for c, m in ms.items()}}) + "\\n")
    elif script.endswith("train_two_si_objects.py") and sub == "precompute-tokens":
        p = Path(opt("--pairs"))
        if not p.is_file():                                                      # the sidecar needs the LINKED pool
            print(f"REFUSED: no pairs file at {p}", file=sys.stderr); sys.exit(2)
    else:
        print(f"fake python: unexpected call {args}", file=sys.stderr); sys.exit(97)
    ''')


class Rig:
    """The chain's expected layout under the site's (tmp) shared root, plus the fake interpreter and the call log."""

    def __init__(self, root: Path, body: str):
        self.root, self.body = root, body
        self.co = Path(re.search(r"; cd (\S+);", body).group(1))
        self.py = Path(re.search(r"; (\S+/envs/sdfbase-fit/bin/python) scripts/", body).group(1))
        self.hf_env = Path(re.search(r"source (\S+/hf_env\.sh)", body).group(1))
        self.ctl2 = Path(re.search(rf"{re.escape(TOOL)} decide --root (\S+) --budget-manifest (\S+)\)", body).group(1))
        self.pools = Path(re.search(r"; mkdir -p (\S+) (\S+); ", body).group(1))
        self.vol = Path(re.search(r'do ln -f "\$f" (\S+)/ ', body).group(1))
        shared = root / "shared"
        for p in (self.co, self.py, self.hf_env, self.ctl2, self.pools, self.vol):
            assert str(p).startswith(str(shared) + "/"), (p, shared)
        self.co.mkdir(parents=True, exist_ok=True)
        self.hf_env.parent.mkdir(parents=True, exist_ok=True); self.hf_env.write_text(":\n")
        fakes = root / "fakes"; fakes.mkdir(exist_ok=True)
        (fakes / "fake_tools.py").write_text(FAKE_TOOLS)
        self.log = root / "fake_calls.jsonl"
        self.py.parent.mkdir(parents=True, exist_ok=True)
        self.py.write_text(textwrap.dedent(f'''\
            #!/bin/bash
            # fake fit-venv python: the REAL decision tool from this checkout, synthetic builders for everything else
            case "$1" in
              {TOOL}) shift; exec {sys.executable} {REPO / TOOL} "$@";;
              *) exec {sys.executable} {fakes / "fake_tools.py"} "$@";;
            esac
            '''))
        self.py.chmod(self.py.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        (root / "home").mkdir(exist_ok=True)

    def run(self, **env_extra) -> subprocess.CompletedProcess:
        self.log.write_text("")
        env = {"PATH": "/usr/bin:/bin", "HOME": str(self.root / "home"), "FAKE_LOG": str(self.log), "FAKE_REPO": str(REPO),
               **env_extra}
        r = subprocess.run(["bash", "-c", self.body], capture_output=True, text=True, env=env, cwd=str(self.root), timeout=600)
        assert_no_tokens(_scan(r.stdout + r.stderr, self.root), "pools chain")
        return r

    def calls(self) -> list:
        return [json.loads(ln) for ln in self.log.read_text().splitlines() if ln.strip()]

    def ctl2_builds(self) -> list:
        return [c for c in self.calls() if c["script"].endswith("build_ctl2_pools.py") and c["sub"] == "build"]

    def preflights(self) -> list:
        return [c["argv"][c["argv"].index("--pools") + 1] for c in self.calls() if c["script"].endswith("build_pools.py") and c["sub"] == "preflight"]

    def pointer(self) -> dict:
        return json.loads(self.ctl2.with_name(self.ctl2.name + ".CURRENT").read_text())

    def vol_inodes(self) -> dict:
        return {f: os.stat(self.vol / f).st_ino for f in POOL_FILES}

    def budget(self) -> dict:
        """The CURRENT budget identity: the rebuilt mix-full manifest's pool_sha256 + official_tokens."""
        m = json.loads((self.pools / BUDGET_MANIFEST).read_text())
        return {"pool_sha256": m["pool_sha256"], "official_tokens": m["official_tokens"]}

    @staticmethod
    def expect(target: Path) -> Path:
        return target.with_name(target.name + ".expect.json")

    @staticmethod
    def expect_checked(target: Path) -> Path:
        return target.with_name(target.name + ".expect.checked.json")

    @staticmethod
    def inodes(d: Path) -> dict:
        return {f: os.stat(d / f).st_ino for f in POOL_FILES}


@pytest.fixture(scope="module")
def rig(exec_root):
    line = _dry_ssh_line(exec_root / "site", env={"SISTEER_CPU_SRUN": ""})
    return Rig(exec_root, _chain_body(line))


def test_chain_executes_build_reuse_rebuild_reuse(rig):
    # ---- run 1: nothing exists -> BUILD into the root, builder + preflight run, commit writes the pointer, links land
    r1 = rig.run()
    assert r1.returncode == 0, r1.stdout[-2000:] + r1.stderr[-2000:]
    assert f"ctl2: BUILD {rig.ctl2}" in r1.stdout, r1.stdout
    assert "reusing" not in r1.stdout
    builds = rig.ctl2_builds()
    assert len(builds) == 1 and builds[0]["argv"][builds[0]["argv"].index("--out") + 1] == str(rig.ctl2), builds
    assert rig.preflights() == [str(rig.pools), str(rig.ctl2)], rig.preflights()
    assert rig.pointer()["dir"] == str(rig.ctl2)
    assert (rig.ctl2 / "preflight_report.json").is_file()
    budget = rig.budget()
    assert rig.pointer()["budget"] == budget and rig.pointer()["reproduced_from"] is None
    full = json.loads((rig.ctl2 / "pairs_constv2ctl2-full.manifest.json").read_text())
    assert {k: full["budget_source"][k] for k in budget} == budget           # the fake builder recorded the budget it read
    assert not list(rig.ctl2.parent.glob("*.expect*"))
    root_inodes = rig.inodes(rig.ctl2)
    assert rig.vol_inodes() == root_inodes                                       # HARD links, not copies
    for si in ("constv2-mix-10m", "constv2-mix-full", "constv2-heldout"):
        assert (rig.vol / f"pairs_{si}.jsonl").is_file()                          # the constitutional pools still ride the loop
    sidecars = [c for c in rig.calls() if c["sub"] == "precompute-tokens"]
    assert len(sidecars) == 5 and all(Path(c["argv"][c["argv"].index("--pairs") + 1]).parent == rig.vol for c in sidecars)
    # the chain never pre-creates the ctl2 dir before the decision: only the pools dir and the volume ride the early mkdir
    assert re.search(rf"; mkdir -p {re.escape(str(rig.pools))} {re.escape(str(rig.vol))}; ", rig.body)

    # ---- run 2: everything complete -> REUSE, the builder is NOT called, no ctl2 preflight, the links stay
    pointer_before = rig.pointer()
    manifest_before = (rig.pools / BUDGET_MANIFEST).read_bytes()
    r2 = rig.run(FAKE_GENERATOR_SHA="b" * 40)                                    # another checkout: the manifest FILE differs
    assert (rig.pools / BUDGET_MANIFEST).read_bytes() != manifest_before and rig.budget() == pointer_before["budget"]
    assert r2.returncode == 0, r2.stdout[-2000:] + r2.stderr[-2000:]
    assert f"ctl2: REUSE {rig.ctl2}" in r2.stdout and f"reusing the complete, preflight-PASSED, byte-verified build at {rig.ctl2}" in r2.stdout
    assert rig.ctl2_builds() == [], rig.ctl2_builds()
    assert rig.preflights() == [str(rig.pools)], rig.preflights()                # the constitutional preflight only
    assert not [c for c in rig.calls() if c["script"] == TOOL]                  # (the real tool is exec'd, not logged)
    assert rig.pointer() == pointer_before
    assert rig.vol_inodes() == root_inodes
    assert len([c for c in rig.calls() if c["sub"] == "precompute-tokens"]) == 5

    # ---- run 3: the root is a PARTIAL build (pointer gone, report gone) -> BUILD into a stamped sibling; the partial stays
    rig.ctl2.with_name(rig.ctl2.name + ".CURRENT").unlink()
    (rig.ctl2 / "preflight_report.json").unlink()
    partial_before = {p.name: p.read_bytes() for p in sorted(rig.ctl2.iterdir())}
    stamped = rig.ctl2.with_name(rig.ctl2.name + ".build-job4242")
    r3 = rig.run(SLURM_JOB_ID="4242")
    assert r3.returncode == 0, r3.stdout[-2000:] + r3.stderr[-2000:]
    assert f"ctl2: BUILD {stamped}" in r3.stdout, r3.stdout
    assert "incomplete ctl2 build" in r3.stderr and "left in place" in r3.stderr and str(rig.ctl2) in r3.stderr, r3.stderr
    builds = rig.ctl2_builds()
    assert len(builds) == 1 and builds[0]["argv"][builds[0]["argv"].index("--out") + 1] == str(stamped), builds
    assert rig.preflights() == [str(rig.pools), str(stamped)], rig.preflights()
    assert {p.name: p.read_bytes() for p in sorted(rig.ctl2.iterdir())} == partial_before   # untouched, not deleted
    assert rig.pointer()["dir"] == str(stamped) and rig.pointer()["slurm_job_id"] == "4242"
    stamped_inodes = rig.inodes(stamped)
    assert rig.vol_inodes() == stamped_inodes                                    # `ln -f` RE-POINTED the volume
    assert not set(stamped_inodes.values()) & set(root_inodes.values())
    assert rig.inodes(rig.ctl2) == root_inodes                                   # the partial's own files are still its own
    # the seeded builder reproduces the bytes: the rebuilt pools equal the partial's
    for f in POOL_FILES:
        if f.endswith(".jsonl"):
            assert (stamped / f).read_bytes() == (rig.ctl2 / f).read_bytes()
    # … and commit CHECKED that against the partial's recorded shas: the sidecar decide wrote was consumed (renamed, not
    # deleted) and the pointer names where the expectation came from
    assert "must reproduce" in r3.stderr and str(rig.expect(stamped)) in r3.stderr, r3.stderr
    assert not rig.expect(stamped).exists() and rig.expect_checked(stamped).is_file()
    exp = json.loads(rig.expect_checked(stamped).read_text())
    assert exp["previous_dir"] == str(rig.ctl2) and exp["budget"] == rig.budget() and exp["skipped"] == {}
    assert exp["cells"] == {c: json.loads((rig.ctl2 / f"pairs_{c}.manifest.json").read_text())["pool_sha256"] for c in CELLS}
    assert rig.pointer()["reproduced_from"] == str(rig.ctl2) and rig.pointer()["budget"] == rig.budget()

    # ---- run 4: REUSE the stamped dir through the pointer; no builder call; the links stay on it
    r4 = rig.run(SLURM_JOB_ID="4343")
    assert r4.returncode == 0, r4.stdout[-2000:] + r4.stderr[-2000:]
    assert f"ctl2: REUSE {stamped}" in r4.stdout, r4.stdout
    assert rig.ctl2_builds() == [] and rig.preflights() == [str(rig.pools)]
    assert rig.vol_inodes() == stamped_inodes
    assert not stamped.with_name(rig.ctl2.name + ".build-job4343").exists()


def test_chain_refuses_when_the_recorded_build_disagrees_with_its_manifests(rig):
    """An edited pool under the CURRENT pointer stops the chain at the decision (exit 3, `REFUSED:`), before any builder
    or link runs — the recorded build is neither rebuilt over nor deleted."""
    cur = Path(rig.pointer()["dir"])
    pool = cur / "pairs_constv2ctl2-full.jsonl"
    original = pool.read_bytes()
    try:
        with open(pool, "ab") as fh:
            fh.write(b'{"doc_id": "ctl2-doc-9999", "instruction_id": "constv2ctl2-full", "prompt": "p", "pos": "extra"}\n')
        r = rig.run()
        assert r.returncode == 3, (r.returncode, r.stdout[-800:], r.stderr[-800:])
        assert "REFUSED:" in r.stderr and "constv2ctl2-full" in r.stderr and "sha256" in r.stderr, r.stderr
        assert "ctl2:" not in r.stdout                                           # no decision line was read
        assert rig.ctl2_builds() == [] and rig.preflights() == [str(rig.pools)]
        assert not [c for c in rig.calls() if c["sub"] == "precompute-tokens"]  # the chain stopped before the sidecars
        assert pool.read_bytes() != original and cur.is_dir()                    # nothing deleted, nothing rebuilt over
    finally:
        pool.write_bytes(original)


def test_chain_refuses_a_current_build_cut_against_a_different_mix_full_budget(rig):
    """The mix-full pool "changes" (new bytes -> a new pool_sha256 in the manifest the chain just rebuilt) while the CURRENT
    ctl2 build records the old one: the decision refuses by name, BEFORE any ctl2 builder or link runs. The recorded build
    is neither rebuilt over nor deleted, the pointer and the volume's ctl2 links are untouched."""
    ptr_before = rig.pointer()
    cur = Path(ptr_before["dir"])
    assert (cur / "preflight_report.json").is_file()                             # a COMPLETE, CURRENT build
    vol_before, cur_before = rig.vol_inodes(), {p.name: p.read_bytes() for p in sorted(cur.iterdir())}
    old = ptr_before["budget"]
    try:
        r = rig.run(FAKE_MIX_SALT="v2")
        new = rig.budget()                                                       # the chain DID rebuild mix-full: new bytes
        assert new["pool_sha256"] != old["pool_sha256"] and new["official_tokens"] == old["official_tokens"]
        assert r.returncode == 3, (r.returncode, r.stdout[-800:], r.stderr[-800:])
        assert "REFUSED:" in r.stderr and "different constv2-mix-full budget" in r.stderr and str(cur) in r.stderr, r.stderr
        assert old["pool_sha256"] in r.stderr and new["pool_sha256"] in r.stderr, r.stderr    # BOTH budgets named
        assert "neither reused, rebuilt over nor deleted" in r.stderr and "move it aside by hand" in r.stderr, r.stderr
        assert "ctl2:" not in r.stdout                                           # no decision line was read
        assert rig.ctl2_builds() == [] and rig.preflights() == [str(rig.pools)]  # no ctl2 builder, no ctl2 preflight
        assert not [c for c in rig.calls() if c["sub"] == "precompute-tokens"]  # the chain stopped before the links + sidecars
        assert rig.vol_inodes() == vol_before and rig.pointer() == ptr_before
        assert {p.name: p.read_bytes() for p in sorted(cur.iterdir())} == cur_before
        assert not list(rig.ctl2.parent.glob("*.expect.json"))                  # nothing written for a REUSE candidate
    finally:
        # the SAME budget again (the mix-full pool rebuilt to its original bytes) -> REUSE: the refusal was about the budget
        r = rig.run()
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        assert f"ctl2: REUSE {cur}" in r.stdout and rig.budget() == old and rig.vol_inodes() == vol_before


def test_chain_refuses_a_seeded_rebuild_that_does_not_reproduce_the_partials_recorded_bytes(rig):
    """A partial root with recorded shas (the pointer gone) -> BUILD into a stamped sibling with an expectation sidecar; the
    fake builder's "seeded rebuild" yields DIFFERENT full-cell bytes -> commit refuses by name (cell, both shas, the previous
    dir): no pointer is written, the volume's ctl2 links still point at the last CURRENT build, the sidecar is left in place
    beside the built-but-uncommitted dir for a human to inspect."""
    ptr = rig.ctl2.with_name(rig.ctl2.name + ".CURRENT")
    last_current = Path(rig.pointer()["dir"])
    ptr.unlink()
    (rig.ctl2 / "preflight_report.json").unlink(missing_ok=True)               # the root is a PARTIAL build (from run 3)
    recorded = {c: json.loads((rig.ctl2 / f"pairs_{c}.manifest.json").read_text())["pool_sha256"] for c in CELLS}
    vol_before = rig.vol_inodes()
    assert vol_before == rig.inodes(last_current)
    target = rig.ctl2.with_name(rig.ctl2.name + ".build-job5151")
    r = rig.run(SLURM_JOB_ID="5151", FAKE_CTL2_PERTURB="constv2ctl2-full")
    assert r.returncode == 3, (r.returncode, r.stdout[-800:], r.stderr[-800:])
    # the decision and the build went ahead (a partial is stepped around) …
    assert f"ctl2: BUILD {target}" in r.stdout, r.stdout
    builds = rig.ctl2_builds()
    assert len(builds) == 1 and builds[0]["argv"][builds[0]["argv"].index("--out") + 1] == str(target), builds
    assert rig.preflights() == [str(rig.pools), str(target)], rig.preflights()
    produced = json.loads((target / "pairs_constv2ctl2-full.manifest.json").read_text())["pool_sha256"]
    assert produced != recorded["constv2ctl2-full"]
    # … and commit refused the rebuild by name
    assert "REFUSED:" in r.stderr and "seeded rebuild of constv2ctl2-full" in r.stderr and str(target) in r.stderr, r.stderr
    assert produced in r.stderr and recorded["constv2ctl2-full"] in r.stderr and f"previous build at {rig.ctl2}" in r.stderr, r.stderr
    assert "not reproducing the recorded bytes" in r.stderr and "NOT re-pointed" in r.stderr, r.stderr
    assert not ptr.exists()                                                      # no pointer written
    assert rig.vol_inodes() == vol_before                                        # the link loop never ran: links still on the last CURRENT
    assert not [c for c in rig.calls() if c["sub"] == "precompute-tokens"]
    assert rig.expect(target).is_file() and not rig.expect_checked(target).exists()   # the sidecar is left for inspection
    exp = json.loads(rig.expect(target).read_text())
    assert exp["cells"] == recorded and exp["previous_dir"] == str(rig.ctl2)
    assert target.is_dir() and (target / "preflight_report.json").is_file()      # the built dir stays too (inspect both by hand)
    assert rig.inodes(rig.ctl2) != rig.inodes(target)                            # the partial's own files are still its own


def test_chain_builds_fresh_beside_a_complete_current_build_cut_under_another_recipe(rig):
    """SAME mix-full pool bytes (the budget identity unchanged), a complete CURRENT build whose manifests (and pointer)
    record a different heldout salt — a build an older checkout's builder carved differently: the decision is BUILD into a
    stamped sibling, NOT REUSE; the old dir is left untouched; commit moves the pointer to the fresh build and the link
    loop re-points the volume at ITS inodes, so the old selection is not linked as current. Then the fresh build REUSEs."""
    # re-establish a complete CURRENT build through the chain itself (the previous test left the pointer gone, the root
    # partial): BUILD past the partial into job6060, the seeded rebuild reproduces its bytes, commit writes the pointer
    r0 = rig.run(SLURM_JOB_ID="6060")
    assert r0.returncode == 0, r0.stdout[-2000:] + r0.stderr[-2000:]
    cur = Path(rig.pointer()["dir"])
    assert cur == rig.ctl2.with_name(rig.ctl2.name + ".build-job6060") and (cur / "preflight_report.json").is_file()
    cur_inodes = rig.inodes(cur)
    assert rig.vol_inodes() == cur_inodes
    today = rig.pointer()["recipe"]["salts"]["heldout"]
    other = "sdfbase-heldout-v0-other"
    assert today and other != today
    for c in CELLS:                                                              # both cell manifests record the other salt …
        mp = cur / f"pairs_{c}.manifest.json"
        m = json.loads(mp.read_text())
        m["recipe"]["salts"]["heldout"] = other
        mp.write_text(json.dumps(m, indent=1) + "\n")
    ptr_path = rig.ctl2.with_name(rig.ctl2.name + ".CURRENT")                    # … and so does the pointer that recorded them
    ptr = json.loads(ptr_path.read_text())
    ptr["recipe"]["salts"]["heldout"] = other
    ptr_path.write_text(json.dumps(ptr) + "\n")
    budget_before = rig.budget()
    cur_before = {p.name: p.read_bytes() for p in sorted(cur.iterdir())}
    stamped = rig.ctl2.with_name(rig.ctl2.name + ".build-job6161")
    diff = f"salts.heldout: {other} (recorded) != {today} (current)"
    r = rig.run(SLURM_JOB_ID="6161")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert rig.budget() == budget_before                                         # the mix-full bytes did NOT change: the recipe alone
    assert f"ctl2: BUILD {stamped}" in r.stdout and "reusing" not in r.stdout, r.stdout
    assert "different recipe (salts.heldout)" in r.stderr and diff in r.stderr, r.stderr
    assert str(cur) in r.stderr and "left in place" in r.stderr and "NOT reused" in r.stderr and "REFUSED" not in r.stderr, r.stderr
    builds = rig.ctl2_builds()
    assert len(builds) == 1 and builds[0]["argv"][builds[0]["argv"].index("--out") + 1] == str(stamped), builds
    assert rig.preflights() == [str(rig.pools), str(stamped)], rig.preflights()
    assert {p.name: p.read_bytes() for p in sorted(cur.iterdir())} == cur_before   # the old selection: untouched, not deleted
    assert rig.inodes(cur) == cur_inodes                                         # its files are still its own
    p = rig.pointer()
    assert p["dir"] == str(stamped) and p["slurm_job_id"] == "6161" and p["previous_dir"] == str(cur) and p["reproduced_from"] is None
    assert p["recipe"]["salts"]["heldout"] == today and p["recipe_changed"] == [diff]
    new_inodes = rig.inodes(stamped)
    assert rig.vol_inodes() == new_inodes                                        # `ln -f` RE-POINTED the volume at the fresh build …
    assert not set(new_inodes.values()) & set(cur_inodes.values())              # … and the old selection is NOT linked as current
    assert not rig.expect(stamped).exists() and rig.expect_checked(stamped).is_file()
    exp = json.loads(rig.expect_checked(stamped).read_text())
    assert exp["cells"] == {} and exp["previous_dir"] == str(cur) and set(exp["skipped"]) == set(CELLS)
    assert all(v.startswith("recipe changed (salts.heldout)") for v in exp["skipped"].values()) and exp["recipe_changed"] == [diff]
    assert len([c for c in rig.calls() if c["sub"] == "precompute-tokens"]) == 5
    # the fresh build records today's recipe -> the next run REUSEs it, no builder call, the links stay
    r2 = rig.run(SLURM_JOB_ID="6262")
    assert r2.returncode == 0, r2.stdout[-2000:] + r2.stderr[-2000:]
    assert f"ctl2: REUSE {stamped}" in r2.stdout and rig.ctl2_builds() == [] and rig.vol_inodes() == new_inodes
    assert rig.pointer() == p



# ------------------------------------------------------------------------------------------------------------------
# (c) the stage LIVE through the `local:` seam: the DONE marker certifies the volume's identities and is validated
# ------------------------------------------------------------------------------------------------------------------
sys.path.insert(0, str(REPO / "scripts/sdfbase"))
import ctl2_pools_reuse as cpr  # noqa: E402  (recipe_digest: the identity a marker certifies)

MARKER_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z) (\S+) on (\S+) recipe=([0-9a-f]{64}) budget=([0-9a-f]{64}):(\d+) links=(hard|copy)$")
#: an archived marker's directory: the UTC stamp to the microsecond (repro_superseded_dir), `-NN` only when that stamp was taken
ARCHIVE_DIR_RE = r"\d{8}T\d{6}\.\d{6}Z(-\d{2})?"
HONOURED = "pools: done (marker) — its certified recipe and budget are today's; skipping"
NOT_HONOURED = "pools: the DONE marker is not honoured — "


class Live:
    """The REAL `02_prepare_model.sh --step pools` run live against a site whose primary cluster is the `local:` seam: every
    repro_ssh string (the chain under the empty CPU-srun wrapper, the two identity reads) runs with `bash -c` on this host
    under the Rig's fakes (the fake fit-venv python runs the REAL ctl2_pools_reuse.py, synthetic builders for the rest)."""

    def __init__(self, root: Path, cross_device: bool = False):
        self.root = root
        (root / "shared").mkdir()
        (root / "fakeslurm").mkdir()
        self.site = _site(root, shared=root / "shared", ssh=f"local:{root / 'fakeslurm'}")
        # the same paths the live run derives, read off the dry run (which touches no state)
        self.rig = Rig(root, _chain_body(_dry_ssh_line(self.site, env={"SISTEER_CPU_SRUN": ""})))
        self.state = root / "state" / MODEL / "02_prepare_model"
        self.marker = self.state / "DONE.pools"
        assert not (root / "state").exists()
        # cross_device: a site whose $VOL is on another filesystem than $CTL2. Emulated by a fake `ln` first on the PATH of
        # the live run (run_stage inherits this process's environment, the `local:` seam's `bash -c` inherits the stage's):
        # `ln -f <src> <vol>/` fails the way EXDEV does, so the chain's `|| cp -f` fallback copies; any other ln is real.
        # A fake rather than a real second filesystem so the flow runs on every host; the real EXDEV is exercised once more
        # by test_live_a_real_cross_device_volume_completes_and_marks_links_copy where /dev/shm is a second filesystem.
        self.ln_log = root / "fake_ln_calls.log"
        self.env_base: dict = {}
        if cross_device:
            real_ln = shutil.which("ln")
            assert real_ln, "no ln on PATH"
            fakebin = root / "fakebin"
            fakebin.mkdir()
            (fakebin / "ln").write_text(textwrap.dedent(f'''\
                #!/bin/bash
                # the cross-device volume: linking INTO {self.rig.vol} fails like EXDEV; every other ln is the real one
                printf "%s\\n" "$*" >> "${{FAKE_LN_LOG:-/dev/null}}"
                for a in "$@"; do
                  case "$a" in {self.rig.vol}|{self.rig.vol}/) echo "ln: failed to create hard link: Invalid cross-device link" >&2; exit 1;; esac
                done
                exec {real_ln} "$@"
                '''))
            (fakebin / "ln").chmod(0o755)
            self.env_base = {"PATH": f"{fakebin}:{os.environ['PATH']}", "FAKE_LN_LOG": str(self.ln_log)}

    def run(self, **env_extra) -> subprocess.CompletedProcess:
        self.rig.log.write_text("")
        self.ln_log.write_text("")
        env = {"SISTEER_CPU_SRUN": "", "FAKE_LOG": str(self.rig.log), "FAKE_REPO": str(REPO), **self.env_base, **env_extra}
        r = run_stage(STAGE, "--step", "pools", "--model", MODEL, site=self.site, env=env, timeout=600)
        out = r.stdout + r.stderr
        assert_no_tokens(_scan(out, self.root), "live pools step")
        assert "DRY-RUN" not in out and "dry run" not in out, out[-800:]
        return r

    def marker_fields(self) -> dict:
        m = MARKER_RE.match(self.marker.read_text().rstrip("\n"))
        assert m, self.marker.read_text()
        return {"utc": m.group(1), "vol": m.group(2), "cluster": m.group(3), "recipe": m.group(4), "budget": f"{m.group(5)}:{m.group(6)}",
                "links": m.group(7)}

    def identity(self, volume: bool = True) -> dict:
        """The REAL tool's identity line for the rig's root + the mix-full manifest the chain last rebuilt (+ the volume the
        stage names, as the stage's two reads do)."""
        vol = ["--volume", str(self.rig.vol)] if volume else []
        r = subprocess.run([sys.executable, str(REPO / TOOL), "identity", "--root", str(self.rig.ctl2),
                            "--budget-manifest", str(self.rig.pools / BUDGET_MANIFEST), *vol], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return dict(t.split("=", 1) for t in r.stdout.strip().split(" "))

    def superseded(self) -> list:
        return sorted(self.state.glob("superseded/*/DONE.pools"))

    def ln_calls(self) -> list:
        """Every `ln` argv the chain issued (cross_device only: the fake logs before refusing or delegating)."""
        return [ln for ln in self.ln_log.read_text().splitlines() if ln.strip()]

    def vol_shas(self) -> dict:
        return {f: hashlib.sha256((self.rig.vol / f).read_bytes()).hexdigest() for f in POOL_FILES}

    def rewrite_marker(self, **fields) -> str:
        """The marker with one or more `key=value` tokens replaced; returns the new text."""
        text = self.marker.read_text()
        for k, v in fields.items():
            text, n = re.subn(rf"\b{k}=\S+", f"{k}={v}", text, count=1)
            assert n == 1, text
        self.marker.write_text(text)
        return text


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    return Live(tmp_path_factory.mktemp("s02live"))


def test_live_run_1_runs_the_chain_and_the_marker_certifies_the_volumes_recorded_identities(live):
    assert not live.marker.exists()
    r = live.run()
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    assert f"ctl2: BUILD {live.rig.ctl2}" in r.stdout and "skipping" not in r.stdout and NOT_HONOURED not in r.stdout
    builds = live.rig.ctl2_builds()
    assert len(builds) == 1 and builds[0]["argv"][builds[0]["argv"].index("--out") + 1] == str(live.rig.ctl2)
    assert live.rig.preflights() == [str(live.rig.pools), str(live.rig.ctl2)]
    assert len([c for c in live.rig.calls() if c["sub"] == "precompute-tokens"]) == 5
    assert live.rig.pointer()["dir"] == str(live.rig.ctl2) and live.rig.vol_inodes() == live.rig.inodes(live.rig.ctl2)
    # the marker: `<utc> <vol> on <cluster> recipe=<64 hex> budget=<64 hex>:<int>`, equal to what identity reads off the volume
    f = live.marker_fields()
    assert f["vol"] == str(live.rig.vol) and f["cluster"] == "cluster_a"
    ident = live.identity()
    assert f["recipe"] == ident["recorded_recipe"] == ident["declared_recipe"] == cpr.recipe_digest(live.rig.pointer()["recipe"])
    b = live.rig.budget()
    assert f["budget"] == ident["recorded_budget"] == ident["budget"] == f"{b['pool_sha256']}:{b['official_tokens']}"
    assert ident["current_dir"] == str(live.rig.ctl2)
    assert ident["volume_links"] == "ok"                                     # the volume's four ctl2 files ARE the build's inodes
    assert f["links"] == "hard"                                              # … and the marker records that form (same-device site)
    assert not live.superseded()
    # the stage told the reader what it honoured/wrote: the only "mark" is the real one (no dry-run wording, asserted in run)
    assert "02_prepare_model done" in r.stdout


def test_live_run_2_honours_the_marker_whose_identities_are_todays_and_runs_no_chain(live):
    before = live.marker.read_bytes()
    ptr, inodes = live.rig.pointer(), live.rig.vol_inodes()
    r = live.run(FAKE_GENERATOR_SHA="b" * 40)                                # another checkout would rewrite the manifest FILE — no chain runs at all
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    assert HONOURED in r.stdout and NOT_HONOURED not in r.stdout, r.stdout
    assert "ctl2:" not in r.stdout and "pools:" in r.stdout
    assert live.rig.calls() == [], live.rig.calls()                         # no builder, no preflight, no sidecar (identity execs the real tool, unlogged)
    assert live.marker.read_bytes() == before and live.rig.pointer() == ptr and live.rig.vol_inodes() == inodes
    assert not live.superseded()


def test_live_run_3_a_marker_certifying_another_recipe_is_archived_and_the_chain_re_decides_reuse(live):
    other = "f" * 64
    text = live.rewrite_marker(recipe=other)
    ptr, inodes = live.rig.pointer(), live.rig.vol_inodes()
    declared = live.identity()["declared_recipe"]
    r = live.run()
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    why = f"{NOT_HONOURED}the marker certifies recipe {other} but this checkout's builder declares {declared} (the ctl2 recipe changed since the marker was written)"
    assert why in r.stdout and HONOURED not in r.stdout, r.stdout
    # archived, never deleted: <state>/<model>/02_prepare_model/superseded/<utc to the microsecond>/DONE.pools holds the exact old text
    arch = live.superseded()
    assert len(arch) == 1 and arch[0].parent.parent == live.state / "superseded" and re.fullmatch(ARCHIVE_DIR_RE, arch[0].parent.name), arch
    assert arch[0].read_text() == text and f"archived to {arch[0]} (moved, never deleted)" in r.stdout
    # the chain ran and decided by name: the CURRENT build's recipe IS today's -> REUSE, no builder, the links re-made on the same inodes
    assert f"ctl2: REUSE {live.rig.ctl2}" in r.stdout and live.rig.ctl2_builds() == [] and live.rig.preflights() == [str(live.rig.pools)]
    assert len([c for c in live.rig.calls() if c["sub"] == "precompute-tokens"]) == 5
    assert live.rig.pointer() == ptr and live.rig.vol_inodes() == inodes
    # a NEW marker certifying the current identities
    f = live.marker_fields()
    assert f["recipe"] == declared and f["budget"] == live.identity()["budget"] and live.marker.read_text() != text


def test_live_run_4_a_changed_recipe_certified_by_the_marker_is_rebuilt_beside_and_the_marker_recertified(live):
    """The build on the volume AND the marker both belong to an older checkout's recipe (another heldout salt): the marker
    mismatches today's declared recipe -> archived -> the chain runs -> decide BUILDs a stamped sibling (the old dir
    untouched), the volume is re-pointed, and the new marker certifies the NEW recipe."""
    cur = Path(live.rig.pointer()["dir"])
    assert cur == live.rig.ctl2 and (cur / "preflight_report.json").is_file()
    today = live.rig.pointer()["recipe"]["salts"]["heldout"]
    other = "sdfbase-heldout-v0-other"
    assert today and other != today
    for c in CELLS:
        mp = cur / f"pairs_{c}.manifest.json"
        m = json.loads(mp.read_text())
        m["recipe"]["salts"]["heldout"] = other
        mp.write_text(json.dumps(m, indent=1) + "\n")
    ptr_path = live.rig.ctl2.with_name(live.rig.ctl2.name + ".CURRENT")
    ptr = json.loads(ptr_path.read_text())
    ptr["recipe"]["salts"]["heldout"] = other
    ptr_path.write_text(json.dumps(ptr) + "\n")
    old_digest = cpr.recipe_digest(ptr["recipe"])
    ident_before = live.identity()
    assert ident_before["recorded_recipe"] == old_digest != ident_before["declared_recipe"]
    text = live.rewrite_marker(recipe=old_digest)                               # what that older checkout's run would have certified
    cur_before = {p.name: p.read_bytes() for p in sorted(cur.iterdir())}
    cur_inodes = live.rig.inodes(cur)
    stamped = live.rig.ctl2.with_name(live.rig.ctl2.name + ".build-job7171")
    diff = f"salts.heldout: {other} (recorded) != {today} (current)"
    r = live.run(SLURM_JOB_ID="7171")
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    assert f"{NOT_HONOURED}the marker certifies recipe {old_digest} but this checkout's builder declares {ident_before['declared_recipe']}" in r.stdout
    arch = live.superseded()
    assert len(arch) == 2 and arch[-1].read_text() == text
    # the chain: BUILD beside (the old selection left in place, never linked as current), commit re-points, links follow
    assert f"ctl2: BUILD {stamped}" in r.stdout and "reusing" not in r.stdout, r.stdout
    assert "different recipe (salts.heldout)" in r.stderr and diff in r.stderr and "REFUSED" not in r.stderr, r.stderr
    builds = live.rig.ctl2_builds()
    assert len(builds) == 1 and builds[0]["argv"][builds[0]["argv"].index("--out") + 1] == str(stamped)
    assert {p.name: p.read_bytes() for p in sorted(cur.iterdir())} == cur_before and live.rig.inodes(cur) == cur_inodes
    p = live.rig.pointer()
    assert p["dir"] == str(stamped) and p["previous_dir"] == str(cur) and p["recipe_changed"] == [diff] and p["recipe"]["salts"]["heldout"] == today
    new_inodes = live.rig.inodes(stamped)
    assert live.rig.vol_inodes() == new_inodes and not set(new_inodes.values()) & set(cur_inodes.values())
    # the new marker certifies the NEW recipe (today's declared == the fresh build's recorded), never the old digest
    f = live.marker_fields()
    ident = live.identity()
    assert ident["current_dir"] == str(stamped) and ident["recorded_recipe"] == ident["declared_recipe"] == cpr.recipe_digest(p["recipe"])
    assert f["recipe"] == ident["recorded_recipe"] != old_digest and f["budget"] == ident["recorded_budget"] == ident["budget"]


def test_live_run_5_a_legacy_marker_with_no_identities_is_archived_and_the_chain_runs(live):
    legacy = f"2026-01-01T00:00:00Z {live.rig.vol} on cluster_a\n"                 # what repro_mark_done wrote before this change
    live.marker.write_text(legacy)
    ptr, inodes = live.rig.pointer(), live.rig.vol_inodes()
    r = live.run(SLURM_JOB_ID="7272")
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    assert f"{NOT_HONOURED}the marker certifies no recipe/budget identities (written before markers carried them)" in r.stdout, r.stdout
    arch = live.superseded()
    assert len(arch) == 3 and arch[-1].read_text() == legacy
    assert f"ctl2: REUSE {ptr['dir']}" in r.stdout and live.rig.ctl2_builds() == []
    assert live.rig.pointer() == ptr and live.rig.vol_inodes() == inodes
    f = live.marker_fields()
    ident = live.identity()
    assert f["recipe"] == ident["declared_recipe"] and f["budget"] == ident["budget"]
    assert not stamped_exists(live, "7272")


def test_live_run_6_a_marker_certifying_another_budget_is_archived_and_the_chain_runs(live):
    f0 = live.marker_fields()
    other = f"{'e' * 64}:{f0['budget'].split(':')[1]}"
    text = live.rewrite_marker(budget=other)
    r = live.run(SLURM_JOB_ID="7373")
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    assert f"{NOT_HONOURED}the marker certifies budget {other} but the current constv2-mix-full budget is {f0['budget']}" in r.stdout, r.stdout
    arch = live.superseded()
    assert len(arch) == 4 and arch[-1].read_text() == text
    assert "ctl2: REUSE " in r.stdout and live.rig.ctl2_builds() == []
    assert live.marker_fields()["budget"] == f0["budget"] and live.marker_fields()["recipe"] == f0["recipe"]
    # and then the honoured path again, unchanged
    r = live.run()
    assert r.returncode == 0 and HONOURED in r.stdout and live.rig.calls() == []


def stamped_exists(live: Live, job: str) -> bool:
    return live.rig.ctl2.with_name(live.rig.ctl2.name + f".build-job{job}").exists()


def test_the_stage_script_reads_the_marker_as_designed():
    """The shell side, pinned by text where the live runs cannot see it: the marker is validated on the primary cluster's
    login host through repro_ssh (no bare ssh, no srun for a two-manifest read), archived by mv (never rm), and written
    from the RECORDED identities after the chain — refusing to mark when they are not the declared ones."""
    text = (REPRO / STAGE).read_text()
    assert 'if repro_done "$MODEL" "$STAGE" pools; then repro_say "pools: done (marker) — skipping"' not in text
    assert re.search(r'^\s*IDENTITY="cd \$CO && \$PY scripts/sdfbase/ctl2_pools_reuse\.py identity --root \$CTL2 --budget-manifest \$POOLS/pairs_constv2-mix-full\.manifest\.json --volume \$VOL"$', text, re.M)
    assert text.count('repro_ssh "$PRIMARY" "$IDENTITY"') == 2                  # the check before the chain, the read after it — BOTH with --volume
    assert re.search(r'^\s*mv -f "\$MARKER" "\$arch/DONE\.pools" \|\| repro_fatal ', text, re.M) and "rm -f \"$MARKER\"" not in text
    # the directory comes from lib/common.sh repro_superseded_dir (a fresh one per archival); a second-resolution `$(date …)` name
    # here made two archivals in one second overwrite each other (gate 2026-10-09, tests 6–12 of this module cascading)
    assert 'arch="$(repro_superseded_dir "$(repro_state_dir "$MODEL")/$STAGE/superseded")" || repro_fatal ' in text
    assert 'superseded/$(date' not in text and 'mkdir -p "$arch"' not in text
    assert 'repro_mark_done "$MODEL" "$STAGE" pools "$VOL on $PRIMARY recipe=$rec_recipe budget=$rec_budget links=$held"' in text
    assert '[ "$rec_recipe" = "$declared" ] && [ "$rec_budget" = "$budget" ]' in text
    assert 'pools: done (marker) — dry run: the marker is not validated; skipping' in text
    assert "'" not in re.search(r'^\s*IDENTITY="([^"]*)"', text, re.M).group(1)
    # the honour condition reads every side by name, the declared side before the recorded one (runs 3/4 name this checkout's
    # builder, run 8 the shared build), the volume's links last; the mark after the chain requires the links too
    pools_step = text[text.index("if want pools; then"):text.index("if want memprobe; then")]
    for needle in ('elif [ "$m_recipe" != "$declared" ]', 'elif [ "$m_budget" != "$budget" ]', 'elif [ "$cur_dir" = none ]',
                   'elif [ "$rec_recipe" = none ]', 'elif [ "$rec_budget" = none ]', 'elif [ "$m_recipe" != "$rec_recipe" ]',
                   'elif [ "$m_budget" != "$rec_budget" ]', 'elif [ "$links" != ok ] && [ "$links" != ok:copy ]'):
        assert pools_step.count(needle) == 1, needle
    assert pools_step.index('"$m_recipe" != "$declared"') < pools_step.index('"$cur_dir" = none') < pools_step.index('"$rec_recipe" = none') \
        < pools_step.index('"$m_recipe" != "$rec_recipe"') < pools_step.index('"$links" != ok ]')
    assert 'records no readable recipe' in pools_step and 'but the CURRENT build $cur_dir records $rec_recipe' in pools_step
    # both forms the loop can leave are accepted, before AND after the chain; the marker records which (links=hard / links=copy)
    assert 'does not hold the CURRENT build $cur_dir (neither hard links nor byte-equal copies; volume_links=$links)' in pools_step
    assert 'as hard links (volume_links=$links)' not in pools_step        # the `ok`-only wording is gone from both refusals
    mark = pools_step[pools_step.index('ident="$(repro_ssh "$PRIMARY" "$IDENTITY")" || repro_fatal'):]
    assert 'case "$links" in ok) held=hard;; ok:copy) held=copy;;' in mark
    assert mark.index('case "$links" in') < mark.index('repro_mark_done "$MODEL" "$STAGE" pools "$VOL')
    assert "does not hold the CURRENT ctl2 build (neither hard links nor byte-equal copies; volume_links=$links)" in mark
    assert "neither its ln -f nor its cp -f fallback landed" in mark
    assert "no DONE marker written; inspect $VOL and $CTL2.CURRENT on $PRIMARY" in mark
    # the loop line itself is unchanged: `ln -f … || cp -f …`, unconditional after the case (REUSE and BUILD alike)
    assert pools_step.count('do ln -f \\"\\$f\\" $VOL/ 2>/dev/null || cp -f \\"\\$f\\" $VOL/; done"') == 1
    assert pools_step.index("esac\"") < pools_step.index('do ln -f') < pools_step.index("precompute-tokens")
    # nothing outside the pools step changed shape for this: the other steps still mark through repro_done / repro_mark_done alone
    assert "volume_links" not in text[:text.index("if want pools; then")] and "volume_links" not in text[text.index("if want memprobe; then"):]


def _frozen_date_bin(root: Path) -> Path:
    """A `date` for the front of PATH that answers every archive-stamp format (`%Y%m%dT%H%M%S…`) for ONE fixed instant and
    delegates everything else to the real date — so two archivals fall in the same second AND the same microsecond, whatever
    the stamp's resolution, while repro_say's prefixes and the marker's own utc stay real."""
    real = shutil.which("date")
    assert real, "no date on PATH"
    fakebin = root / "fakebin_date"
    fakebin.mkdir()
    (fakebin / "date").write_text(textwrap.dedent(f'''\
        #!/bin/bash
        case "$*" in *%Y%m%dT%H%M%S*) exec {real} -u -d @1790000000 "$@";; *) exec {real} "$@";; esac
        '''))
    (fakebin / "date").chmod(0o755)
    return fakebin


def test_two_archivals_in_one_second_land_in_two_directories_and_the_first_is_kept(tmp_path_factory):
    """The gate of 2026-10-09 (main 2dedae50f): tests 6–12 of this module went red on one run and green on the next. The
    archive directory was `superseded/$(date -u +%Y%m%dT%H%M%SZ)` — to the second — so on a fast node two not-honoured runs
    in one second named the SAME directory, `mkdir -p` accepted it and the second `mv -f` wrote the first archived marker
    over; the cumulative count every later test asserts was then one short. With `date` frozen (a fake on PATH, one instant
    for every stamp format) two archivals land in TWO directories — the microsecond stamp, then its `-02` sibling — each
    holding the marker text it archived, the stage saying each path; nothing archived is ever written over."""
    root = tmp_path_factory.mktemp("s02samesec")
    live = Live(root)
    path = f"{_frozen_date_bin(root)}:{os.environ['PATH']}"
    r = live.run(PATH=path)
    assert r.returncode == 0 and live.marker.exists() and not live.superseded(), r.stdout[-2500:] + r.stderr[-2500:]
    first = live.rewrite_marker(recipe="a" * 64)
    r1 = live.run(PATH=path)
    assert r1.returncode == 0 and NOT_HONOURED in r1.stdout, r1.stdout[-2500:] + r1.stderr[-2500:]
    second = live.rewrite_marker(recipe="b" * 64)
    assert second != first
    r2 = live.run(PATH=path)
    assert r2.returncode == 0 and NOT_HONOURED in r2.stdout, r2.stdout[-2500:] + r2.stderr[-2500:]
    arch = live.superseded()
    assert len(arch) == 2, [str(a) for a in arch]
    names = [a.parent.name for a in arch]
    assert re.fullmatch(r"\d{8}T\d{6}\.\d{6}Z", names[0]) and names[1] == f"{names[0]}-02", names     # the same instant, twice
    assert arch[0].read_text() == first and arch[1].read_text() == second                          # the first archive is intact
    assert f"archived to {arch[0]} (moved, never deleted)" in r1.stdout and f"archived to {arch[1]} (moved, never deleted)" in r2.stdout
    assert live.marker.read_text() not in (first, second) and live.marker_fields()["recipe"] not in ("a" * 64, "b" * 64)


def test_repro_superseded_dir_reserves_a_fresh_microsecond_stamped_directory_per_call(tmp_path):
    """lib/common.sh repro_superseded_dir, the function the stage archives through: under a frozen `date` three calls print and
    create <base>/<stamp>, <stamp>-02, <stamp>-03 (a lexical sort is the archival order; no call reuses an existing directory);
    under the real clock two calls back to back print two distinct directories in order; a base that cannot be a directory
    (a file) prints nothing and returns non-zero; under REPRO_DRY_RUN=1 it creates nothing, prints nothing and refuses by name
    (its stdout is the path the caller moves into, so a dry run cannot print a note there — the stage's own dry-run branch never
    reaches it, test_dry_run_with_a_marker_skips_without_validating_and_touches_nothing)."""
    base = tmp_path / "state" / "superseded"
    env = {**os.environ, "PATH": f"{_frozen_date_bin(tmp_path)}:{os.environ['PATH']}"}
    script = f"set -uo pipefail; source {REPO}/reproduce/lib/common.sh; for i in 1 2 3; do repro_superseded_dir {base} || echo FAIL; done"
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0 and "FAIL" not in r.stdout, r.stdout + r.stderr
    printed = r.stdout.split()
    assert len(printed) == 3 and len(set(printed)) == 3 and all(Path(d).is_dir() for d in printed), printed
    names = [Path(d).name for d in printed]
    assert re.fullmatch(r"\d{8}T\d{6}\.\d{6}Z", names[0]) and names[1:] == [f"{names[0]}-02", f"{names[0]}-03"], names
    assert sorted(p.name for p in base.iterdir()) == names                                        # sorted = archival order
    # the real clock: two calls back to back are two directories, the later one sorting last
    script = f"set -uo pipefail; source {REPO}/reproduce/lib/common.sh; repro_superseded_dir {tmp_path}/real; repro_superseded_dir {tmp_path}/real"
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=os.environ, timeout=60)
    two = r.stdout.split()
    assert r.returncode == 0 and len(two) == 2 and two[0] < two[1] and all(re.fullmatch(ARCHIVE_DIR_RE, Path(d).name) for d in two), r.stdout + r.stderr
    # a base that is a file: nothing printed, non-zero
    (tmp_path / "afile").write_text("")
    script = f"set -uo pipefail; source {REPO}/reproduce/lib/common.sh; out=$(repro_superseded_dir {tmp_path}/afile); echo rc=$? out=[$out]"
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=os.environ, timeout=60)
    assert "rc=1 out=[]" in r.stdout, r.stdout + r.stderr
    # a dry run: nothing created, nothing printed, non-zero by name (the exported flag survives sourcing the libs — fail-closed)
    script = f"set -uo pipefail; export REPRO_DRY_RUN=1; source {REPO}/reproduce/lib/common.sh; out=$(repro_superseded_dir {tmp_path}/dry); echo rc=$? out=[$out]"
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=os.environ, timeout=60)
    assert "rc=1 out=[]" in r.stdout and "repro_superseded_dir: refused under --dry-run" in r.stderr and not (tmp_path / "dry").exists(), r.stdout + r.stderr


def test_dry_run_with_a_marker_skips_without_validating_and_touches_nothing(tmp_path):
    """--dry-run never reads the cluster or moves state: an existing marker skips (said as unvalidated), no ssh line at
    all; without a marker the identity read after the chain is only NAMED (a note, not a `DRY-RUN: ssh` line — every ssh
    line of --step pools carries the CPU-srun wrapper, test_stage02_prepare_model_fixes pins that) and the mark is printed."""
    site = _site(tmp_path)
    state = tmp_path / "state" / MODEL / "02_prepare_model"
    r = run_stage(STAGE, "--dry-run", "--step", "pools", "--model", MODEL, site=site, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2000:]
    assert_no_tokens(_scan(out, tmp_path), STAGE)
    assert len([ln for ln in out.splitlines() if ln.startswith("DRY-RUN: ssh ")]) == 1
    note = [ln for ln in out.splitlines() if ln.startswith("note: dry run: after the chain")]
    assert len(note) == 1 and "scripts/sdfbase/ctl2_pools_reuse.py identity --root /scratch/example/shared/exp/reproduce/qwen35_4b/02_prepare_model/pools_ctl2 --budget-manifest /scratch/example/shared/exp/reproduce/qwen35_4b/02_prepare_model/pools/pairs_constv2-mix-full.manifest.json" in note[0], note
    assert "recipe= and budget= tokens plus links=hard|copy (how the volume holds the build)" in note[0] and not PLACEHOLDER.search(note[0])
    assert "--volume /scratch/example/shared/vol/pool/qwen35_4b_base" in note[0], note[0]     # both identity reads check the volume
    assert f"DRY-RUN: mark {state}/DONE.pools" in out and not (tmp_path / "state").exists()
    # a marker present: skipped unvalidated, nothing read, nothing moved
    state.mkdir(parents=True)
    (state / "DONE.pools").write_text("2026-01-01T00:00:00Z /scratch/example/shared/vol/pool/x on cluster_a\n")
    r = run_stage(STAGE, "--dry-run", "--step", "pools", "--model", MODEL, site=site, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2000:]
    assert "pools: done (marker) — dry run: the marker is not validated; skipping" in out
    assert "DRY-RUN: ssh " not in out and NOT_HONOURED not in out and "identity" not in out
    assert sorted(p.name for p in state.iterdir()) == ["DONE.pools"]


def test_live_run_7_a_failed_identity_read_archives_the_marker_and_the_chain_rebuilds_the_budget_manifest(live):
    """The mix-full manifest is gone (someone cleared the pools dir): identity REFUSES (exit 3) before the chain, the marker
    is archived with the rc named, and the chain runs — it rebuilds the manifest, decides REUSE, and a fresh marker lands."""
    f0 = live.marker_fields()
    text = live.marker.read_text()
    (live.rig.pools / BUDGET_MANIFEST).unlink()
    ptr, inodes = live.rig.pointer(), live.rig.vol_inodes()
    r = live.run()
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    assert f"{NOT_HONOURED}the identity read on cluster_a failed (rc 3; its reason is above)" in r.stdout, r.stdout
    assert "REFUSED: budget manifest" in r.stderr and "missing" in r.stderr, r.stderr
    arch = live.superseded()
    assert len(arch) == 5 and arch[-1].read_text() == text
    assert f"ctl2: REUSE {ptr['dir']}" in r.stdout and live.rig.ctl2_builds() == [] and (live.rig.pools / BUDGET_MANIFEST).is_file()
    assert live.rig.pointer() == ptr and live.rig.vol_inodes() == inodes
    assert live.marker_fields()["recipe"] == f0["recipe"] and live.marker_fields()["budget"] == f0["budget"]


# ------------------------------------------------------------------------------------------------------------------
# (c, continued) the RECORDED side and the volume's links: the marker is local, the build is shared
# ------------------------------------------------------------------------------------------------------------------
def _copy_in_place(p: Path) -> None:
    """Replace the file at ``p`` with a COPY of itself: identical bytes, its own inode (what `cp -f` leaves where `ln -f`
    should have landed, or what someone copying a pool by hand leaves)."""
    data = p.read_bytes()
    p.unlink()
    p.write_bytes(data)


def test_live_run_8_another_checkouts_rebuild_re_pointed_the_shared_build_so_the_marker_is_not_honoured_and_recertified(live):
    """After this checkout's run, ANOTHER checkout (a changed heldout salt) rebuilt ctl2 into its own stamped sibling, moved
    $CTL2.CURRENT to it and `ln -f`'d the volume's ctl2 files onto ITS inodes. This checkout's marker still equals its own
    declared recipe and the unchanged mix-full budget — the check on 17a20bcb5 skipped here and left the fits reading the
    other checkout's selection. Now the RECORDED side disagrees: not honoured by name (which build, both digests), archived,
    the chain runs, decide BUILDs a fresh sibling under today's recipe (the other checkout's dir untouched, never linked as
    current), and the marker written afterwards certifies the volume's ACTUAL build."""
    f0 = live.marker_fields()
    text = live.marker.read_text()
    cur = Path(live.rig.pointer()["dir"])
    assert (cur / "preflight_report.json").is_file()
    ident0 = live.identity()
    assert ident0["current_dir"] == str(cur) and ident0["recorded_recipe"] == f0["recipe"] == ident0["declared_recipe"]
    assert ident0["volume_links"] == "ok"
    # --- the other checkout's work: a sibling build with the same coherent bytes whose manifests record ITS heldout salt, the
    # pointer moved onto it (recording that recipe), the volume's ctl2 files re-linked onto its inodes
    other_dir = live.rig.ctl2.with_name(live.rig.ctl2.name + ".build-job8080")
    shutil.copytree(cur, other_dir)
    today = live.rig.pointer()["recipe"]["salts"]["heldout"]
    other_salt = "sdfbase-heldout-v0-another-checkout"
    assert today and other_salt != today
    for c in CELLS:
        mp = other_dir / f"pairs_{c}.manifest.json"
        m = json.loads(mp.read_text())
        m["recipe"]["salts"]["heldout"] = other_salt
        mp.write_text(json.dumps(m, indent=1) + "\n")
    ptr_path = live.rig.ctl2.with_name(live.rig.ctl2.name + ".CURRENT")
    ptr = json.loads(ptr_path.read_text())
    ptr["dir"], ptr["slurm_job_id"] = str(other_dir), "8080"
    ptr["recipe"]["salts"]["heldout"] = other_salt
    ptr_path.write_text(json.dumps(ptr) + "\n")
    for f in POOL_FILES:
        (live.rig.vol / f).unlink()
        os.link(other_dir / f, live.rig.vol / f)
    other_digest = cpr.recipe_digest(json.loads((other_dir / f"pairs_{CELLS[1]}.manifest.json").read_text())["recipe"])
    assert other_digest != f0["recipe"]
    other_before = {p.name: p.read_bytes() for p in sorted(other_dir.iterdir())}
    other_inodes = live.rig.inodes(other_dir)
    assert live.rig.vol_inodes() == other_inodes and not set(other_inodes.values()) & set(live.rig.inodes(cur).values())
    # this checkout's marker STILL equals its declared recipe and the mix-full budget — the two identities the old check compared —
    # while the volume's CURRENT build records the other checkout's recipe, and the volume does hold that build
    ident1 = live.identity()
    assert f0["recipe"] == ident1["declared_recipe"] and f0["budget"] == ident1["budget"]
    assert ident1["current_dir"] == str(other_dir) and ident1["recorded_recipe"] == other_digest and ident1["volume_links"] == "ok"
    stamped = live.rig.ctl2.with_name(live.rig.ctl2.name + ".build-job8181")
    diff = f"salts.heldout: {other_salt} (recorded) != {today} (current)"
    r = live.run(SLURM_JOB_ID="8181")
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    why = f"{NOT_HONOURED}the marker certifies recipe {f0['recipe']} but the CURRENT build {other_dir} records {other_digest}"
    assert why in r.stdout and HONOURED not in r.stdout, r.stdout
    assert "another checkout rebuilt ctl2 under a different recipe" in r.stdout
    arch = live.superseded()
    assert len(arch) == 6 and arch[-1].read_text() == text and f"archived to {arch[-1]} (moved, never deleted)" in r.stdout
    # the chain ran and decided by name: the CURRENT build's selection is not today's builder's -> BUILD beside, the builder ran,
    # commit moved the pointer, the link loop re-pointed the volume; the other checkout's dir is untouched and not linked
    assert f"ctl2: BUILD {stamped}" in r.stdout and "reusing" not in r.stdout, r.stdout
    assert "different recipe (salts.heldout)" in r.stderr and diff in r.stderr and str(other_dir) in r.stderr and "REFUSED" not in r.stderr, r.stderr
    builds = live.rig.ctl2_builds()
    assert len(builds) == 1 and builds[0]["argv"][builds[0]["argv"].index("--out") + 1] == str(stamped), builds
    assert live.rig.preflights() == [str(live.rig.pools), str(stamped)]
    assert {p.name: p.read_bytes() for p in sorted(other_dir.iterdir())} == other_before and live.rig.inodes(other_dir) == other_inodes
    p = live.rig.pointer()
    assert p["dir"] == str(stamped) and p["slurm_job_id"] == "8181" and p["previous_dir"] == str(other_dir)
    assert p["recipe_changed"] == [diff] and p["recipe"]["salts"]["heldout"] == today
    new_inodes = live.rig.inodes(stamped)
    assert live.rig.vol_inodes() == new_inodes and not set(new_inodes.values()) & set(other_inodes.values())
    assert len([c for c in live.rig.calls() if c["sub"] == "precompute-tokens"]) == 5
    # the marker written afterwards certifies the VOLUME's actual build — today's recipe, recorded == declared — and the
    # volume holds it; never the other checkout's digest
    f = live.marker_fields()
    ident = live.identity()
    assert ident["current_dir"] == str(stamped) and ident["volume_links"] == "ok"
    assert f["recipe"] == ident["recorded_recipe"] == ident["declared_recipe"] == cpr.recipe_digest(p["recipe"]) == f0["recipe"]
    assert f["recipe"] != other_digest and f["budget"] == ident["recorded_budget"] == ident["budget"] == f0["budget"]


def test_live_run_9_a_current_build_recording_no_readable_recipe_is_not_honoured_and_the_chain_refuses_by_name(live):
    """The CURRENT build's manifests lose their `recipe` block (recorded_recipe=none): the marker — still equal to today's
    declared recipe and budget — is not honoured ("records no readable recipe"), archived, and the chain runs; decide then
    REFUSES that build by name (no recipe record: neither reused nor rebuilt over), so the stage fails and writes NO marker,
    nothing is built or re-linked. With the block restored, the next run (no marker) re-decides REUSE and re-certifies."""
    f0 = live.marker_fields()
    text = live.marker.read_text()
    cur = Path(live.rig.pointer()["dir"])
    saved = {}
    for c in CELLS:
        mp = cur / f"pairs_{c}.manifest.json"
        saved[c] = mp.read_bytes()
        m = json.loads(saved[c])
        del m["recipe"]
        mp.write_text(json.dumps(m, indent=1) + "\n")
    ident0 = live.identity()
    assert ident0["current_dir"] == str(cur) and ident0["recorded_recipe"] == "none" and ident0["volume_links"] == "ok"
    assert ident0["recorded_budget"] == f0["budget"] == ident0["budget"] and f0["recipe"] == ident0["declared_recipe"]   # the old check would skip
    ptr, inodes = live.rig.pointer(), live.rig.vol_inodes()
    r = live.run(SLURM_JOB_ID="9090")
    assert r.returncode != 0, r.stdout[-2500:] + r.stderr[-2500:]
    why = f"{NOT_HONOURED}the CURRENT build at {cur} records no readable recipe (the marker certifies recipe {f0['recipe']}; the build cannot say what cut it)"
    assert why in r.stdout and HONOURED not in r.stdout, r.stdout
    arch = live.superseded()
    assert len(arch) == 7 and arch[-1].read_text() == text
    # the chain ran to the decision, which refused by name: nothing built, nothing linked, no sidecars, no marker
    assert "REFUSED:" in r.stderr and "records no recipe" in r.stderr and str(cur) in r.stderr and "never silently reused" in r.stderr, r.stderr
    assert "REFUSED: the pools chain failed on cluster_a" in r.stderr and "nothing marked done" in r.stderr, r.stderr
    assert "ctl2:" not in r.stdout and live.rig.ctl2_builds() == [] and live.rig.preflights() == [str(live.rig.pools)]
    assert not [c for c in live.rig.calls() if c["sub"] == "precompute-tokens"]
    assert not live.marker.exists() and not stamped_exists(live, "9090")
    assert live.rig.pointer() == ptr and live.rig.vol_inodes() == inodes
    # restore the block (the build itself was never touched): no marker -> the chain runs -> REUSE -> a marker certifying it
    for c in CELLS:
        (cur / f"pairs_{c}.manifest.json").write_bytes(saved[c])
    r = live.run(SLURM_JOB_ID="9191")
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    assert NOT_HONOURED not in r.stdout and HONOURED not in r.stdout and f"ctl2: REUSE {cur}" in r.stdout, r.stdout
    assert live.rig.ctl2_builds() == [] and live.rig.pointer() == ptr and live.rig.vol_inodes() == inodes
    f = live.marker_fields()
    assert f["recipe"] == f0["recipe"] and f["budget"] == f0["budget"] and len(live.superseded()) == 7
    assert live.identity()["volume_links"] == "ok"


def test_live_run_10_a_byte_equal_copy_is_ok_copy_and_honoured_while_a_stale_copy_is_refused_by_name_and_re_linked(live):
    """(a) The volume's heldout pool is a COPY of the CURRENT build's (same bytes, its own inode): identity says
    volume_links=ok:copy — the fits read the same bytes — and the marker is honoured, no chain (under the earlier
    `ok`-only rule this was a refusal, and a site whose volume is on another filesystem could never mark).
    (b) The copy goes STALE (other bytes): volume_links=pairs_constv2ctl2-heldout.jsonl:not_the_current_builds_file, the
    marker is not honoured with that reason verbatim, archived, the chain runs (REUSE — the build itself is fine) and its
    `ln -f` loop re-LINKS the volume on this same-device site: the inodes are equal afterwards, identity says ok, and the
    marker is rewritten with links=hard."""
    f0 = live.marker_fields()
    text = live.marker.read_text()
    cur = Path(live.rig.pointer()["dir"])
    name = "pairs_constv2ctl2-heldout.jsonl"
    _copy_in_place(live.rig.vol / name)
    assert (live.rig.vol / name).read_bytes() == (cur / name).read_bytes()
    assert os.stat(live.rig.vol / name).st_ino != os.stat(cur / name).st_ino
    ident0 = live.identity()
    assert ident0["volume_links"] == "ok:copy" and ident0["recorded_recipe"] == f0["recipe"] == ident0["declared_recipe"]
    ptr = live.rig.pointer()
    # (a) honoured: no chain, nothing archived, the marker untouched, the copy still a copy (nothing re-linked it)
    r = live.run(SLURM_JOB_ID="1010")
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    assert HONOURED in r.stdout and NOT_HONOURED not in r.stdout and "ctl2:" not in r.stdout, r.stdout
    assert live.rig.calls() == [] and live.marker.read_text() == text and len(live.superseded()) == 7
    assert os.stat(live.rig.vol / name).st_ino != os.stat(cur / name).st_ino and live.rig.pointer() == ptr
    assert not stamped_exists(live, "1010")
    # (b) the copy goes stale (its own inode: writing it touches no build file)
    with open(live.rig.vol / name, "ab") as fh:
        fh.write(b'{"doc_id": "ctl2-doc-9999", "instruction_id": "constv2ctl2-heldout", "prompt": "p", "pos": "stale"}\n')
    assert (live.rig.vol / name).read_bytes() != (cur / name).read_bytes()
    reason = f"{name}:not_the_current_builds_file"
    assert live.identity()["volume_links"] == reason
    r = live.run(SLURM_JOB_ID="1020")
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    why = f"{NOT_HONOURED}the volume {live.rig.vol} on cluster_a does not hold the CURRENT build {cur} (neither hard links nor byte-equal copies; volume_links={reason})"
    assert why in r.stdout and HONOURED not in r.stdout, r.stdout
    arch = live.superseded()
    assert len(arch) == 8 and arch[-1].read_text() == text
    assert f"ctl2: REUSE {cur}" in r.stdout and live.rig.ctl2_builds() == [] and live.rig.preflights() == [str(live.rig.pools)]
    assert live.rig.vol_inodes() == live.rig.inodes(cur)                         # `ln -f` re-linked the stale copy away
    assert live.rig.pointer() == ptr and not stamped_exists(live, "1020")
    assert live.identity()["volume_links"] == "ok"
    f = live.marker_fields()
    assert live.marker.is_file() and f["recipe"] == f0["recipe"] and f["budget"] == f0["budget"] and f["utc"] >= f0["utc"]
    assert f["links"] == "hard"                                                  # a same-device site links; the marker says so


def test_live_run_11_an_unchanged_state_is_still_honoured_with_no_chain_call(live):
    before = live.marker.read_bytes()
    ptr, inodes = live.rig.pointer(), live.rig.vol_inodes()
    r = live.run(FAKE_GENERATOR_SHA="c" * 40)
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    assert HONOURED in r.stdout and NOT_HONOURED not in r.stdout and "ctl2:" not in r.stdout, r.stdout
    assert live.rig.calls() == [], live.rig.calls()
    assert live.marker.read_bytes() == before and live.rig.pointer() == ptr and live.rig.vol_inodes() == inodes
    assert len(live.superseded()) == 8


def test_live_run_12_a_marker_from_before_links_was_recorded_is_still_honoured(live):
    """708a7db63's markers carry recipe= and budget= only (they could only ever certify a hard-linked volume). links= is a
    RECORD of how the volume held the build when marked, not an identity the honour rule compares — so such a marker is
    honoured while the identities agree and the volume holds the build today; nothing archived, the marker left as it is
    (the next chain run writes the token)."""
    text = live.marker.read_text()
    stripped = re.sub(r" links=\S+", "", text)
    assert stripped != text and MARKER_RE.match(stripped.rstrip("\n")) is None
    live.marker.write_text(stripped)
    try:
        r = live.run()
        assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
        assert HONOURED in r.stdout and NOT_HONOURED not in r.stdout and "ctl2:" not in r.stdout, r.stdout
        assert live.rig.calls() == [] and live.marker.read_text() == stripped and len(live.superseded()) == 8
    finally:
        live.marker.write_text(text)


def test_live_a_real_cross_device_volume_completes_and_marks_links_copy(tmp_path_factory):
    """The REAL EXDEV, inverting the earlier rule (where this site could never write DONE.pools): $VOL resolves
    onto tmpfs while the build is on the pytest tmp root, so coreutils' `ln -f` fails with EXDEV and the loop's `cp -f`
    copies. Run 1 completes: the volume holds byte-equal copies on another device, identity says ok:copy, the marker is
    written with links=copy; run 2 honours it with no chain. The same verdicts the fake-ln flow (`xdev` below) reaches on
    every host. Needs a second filesystem: /dev/shm (tmpfs) when it differs from the pytest tmp root."""
    shm = Path("/dev/shm")
    root = tmp_path_factory.mktemp("s02xdev")
    if not shm.is_dir() or not os.access(shm, os.W_OK) or os.stat(shm).st_dev == os.stat(root).st_dev:
        pytest.skip("no second filesystem to put the volume on")
    live = Live(root)
    other_fs = Path(tempfile.mkdtemp(prefix="s02xdev-vol-", dir=shm))
    try:
        live.rig.vol.parent.mkdir(parents=True, exist_ok=True)
        live.rig.vol.symlink_to(other_fs)                                        # $VOL resolves onto tmpfs: `ln -f` -> EXDEV -> `cp -f`
        r = live.run(SLURM_JOB_ID="1111")
        assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
        assert f"ctl2: BUILD {live.rig.ctl2}" in r.stdout and "02_prepare_model done" in r.stdout, r.stdout
        assert "REFUSED" not in r.stderr and NOT_HONOURED not in r.stdout, r.stderr
        assert live.rig.pointer()["dir"] == str(live.rig.ctl2)
        for f in POOL_FILES:
            assert (other_fs / f).read_bytes() == (live.rig.ctl2 / f).read_bytes()
            assert os.stat(other_fs / f).st_dev != os.stat(live.rig.ctl2 / f).st_dev
        ident = live.identity()
        assert ident["volume_links"] == "ok:copy" and ident["current_dir"] == str(live.rig.ctl2)
        f = live.marker_fields()
        assert f["links"] == "copy" and f["recipe"] == ident["recorded_recipe"] == ident["declared_recipe"] and f["budget"] == ident["budget"]
        assert not live.superseded()
        # run 2: honoured, no chain
        text = live.marker.read_text()
        r = live.run()
        assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
        assert HONOURED in r.stdout and NOT_HONOURED not in r.stdout and live.rig.calls() == [] and live.marker.read_text() == text
    finally:
        shutil.rmtree(other_fs, ignore_errors=True)


# ------------------------------------------------------------------------------------------------------------------
# (d) the cross-device site LIVE, deterministically: a fake `ln` on PATH answers EXDEV for the volume
# ------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def xdev(tmp_path_factory):
    return Live(tmp_path_factory.mktemp("s02xdevln"), cross_device=True)


def test_xdev_run_1_a_cross_device_volume_completes_with_byte_equal_copies_and_the_marker_says_links_copy(xdev):
    """Run 1 on a site whose volume is on another filesystem: the chain's `ln -f` is refused for every file of the loop (the
    fake ln logs each and exits 1 the way EXDEV does), `cp -f` copies, the volume holds byte-equal copies with their own
    inodes, identity says ok:copy and the mark ACCEPTS it: DONE.pools is written with links=copy — where 708a7db63 refused
    every run of such a site."""
    assert not xdev.marker.exists()
    r = xdev.run(SLURM_JOB_ID="2121")
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    assert f"ctl2: BUILD {xdev.rig.ctl2}" in r.stdout and "02_prepare_model done" in r.stdout and NOT_HONOURED not in r.stdout, r.stdout
    assert "REFUSED" not in r.stderr, r.stderr
    builds = xdev.rig.ctl2_builds()
    assert len(builds) == 1 and builds[0]["argv"][builds[0]["argv"].index("--out") + 1] == str(xdev.rig.ctl2)
    assert xdev.rig.pointer()["dir"] == str(xdev.rig.ctl2)
    assert len([c for c in xdev.rig.calls() if c["sub"] == "precompute-tokens"]) == 5   # the sidecars read the COPIED pools
    # the fake ln saw every `ln -f <src> <vol>/` of the loop and refused each: the four ctl2 files among them
    calls = xdev.ln_calls()
    assert calls and all(c.startswith("-f ") and c.endswith(f" {xdev.rig.vol}/") for c in calls), calls
    assert {Path(c.split(" ")[1]).name for c in calls} >= set(POOL_FILES), calls
    for f in POOL_FILES:
        assert (xdev.rig.vol / f).read_bytes() == (xdev.rig.ctl2 / f).read_bytes()      # the CURRENT dir's bytes …
        assert os.stat(xdev.rig.vol / f).st_ino != os.stat(xdev.rig.ctl2 / f).st_ino      # … as copies, not links
    ident = xdev.identity()
    assert ident["volume_links"] == "ok:copy" and ident["current_dir"] == str(xdev.rig.ctl2)
    f = xdev.marker_fields()
    assert f["links"] == "copy" and f["vol"] == str(xdev.rig.vol) and f["cluster"] == "cluster_a"
    assert f["recipe"] == ident["recorded_recipe"] == ident["declared_recipe"] and f["budget"] == ident["recorded_budget"] == ident["budget"]
    assert not xdev.superseded()


def test_xdev_run_2_honours_the_links_copy_marker_with_no_chain_call(xdev):
    before = xdev.marker.read_bytes()
    ptr, shas = xdev.rig.pointer(), xdev.vol_shas()
    r = xdev.run(FAKE_GENERATOR_SHA="d" * 40)
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    assert HONOURED in r.stdout and NOT_HONOURED not in r.stdout and "ctl2:" not in r.stdout, r.stdout
    assert xdev.rig.calls() == [] and xdev.ln_calls() == []                     # no builder, no sidecar, no loop
    assert xdev.marker.read_bytes() == before and xdev.rig.pointer() == ptr and xdev.vol_shas() == shas
    assert not xdev.superseded()


def test_xdev_run_3_a_stale_copy_is_refused_by_name_and_reuse_re_copies_the_current_bytes(xdev):
    """One volume pool goes STALE (other bytes than the CURRENT build's): identity says
    volume_links=pairs_constv2ctl2-full.jsonl:not_the_current_builds_file, the marker is not honoured with that reason,
    archived, the chain runs — REUSE, no builder — and its loop (which runs on REUSE too) has `ln -f` refused again and
    `cp -f` re-copy the file: its sha256 equals the CURRENT dir's afterwards, identity says ok:copy, and the marker is
    rewritten with links=copy."""
    f0 = xdev.marker_fields()
    text = xdev.marker.read_text()
    cur = Path(xdev.rig.pointer()["dir"])
    name = "pairs_constv2ctl2-full.jsonl"
    assert os.stat(xdev.rig.vol / name).st_ino != os.stat(cur / name).st_ino    # a copy: writing it touches no build file
    with open(xdev.rig.vol / name, "ab") as fh:
        fh.write(b'{"doc_id": "ctl2-doc-9999", "instruction_id": "constv2ctl2-full", "prompt": "p", "pos": "stale"}\n')
    stale = hashlib.sha256((xdev.rig.vol / name).read_bytes()).hexdigest()
    current = hashlib.sha256((cur / name).read_bytes()).hexdigest()
    assert stale != current
    reason = f"{name}:not_the_current_builds_file"
    assert xdev.identity()["volume_links"] == reason
    ptr = xdev.rig.pointer()
    r = xdev.run(SLURM_JOB_ID="2323")
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    why = f"{NOT_HONOURED}the volume {xdev.rig.vol} on cluster_a does not hold the CURRENT build {cur} (neither hard links nor byte-equal copies; volume_links={reason})"
    assert why in r.stdout and HONOURED not in r.stdout, r.stdout
    arch = xdev.superseded()
    assert len(arch) == 1 and arch[0].read_text() == text and f"archived to {arch[0]} (moved, never deleted)" in r.stdout
    assert f"ctl2: REUSE {cur}" in r.stdout and xdev.rig.ctl2_builds() == [] and xdev.rig.preflights() == [str(xdev.rig.pools)]
    # the loop ran on REUSE: ln refused for the stale file too, cp -f re-copied it — the CURRENT bytes are back on the volume
    assert any(Path(c.split(" ")[1]).name == name for c in xdev.ln_calls()), xdev.ln_calls()
    assert hashlib.sha256((xdev.rig.vol / name).read_bytes()).hexdigest() == current
    for f in POOL_FILES:
        assert (xdev.rig.vol / f).read_bytes() == (cur / f).read_bytes()
        assert os.stat(xdev.rig.vol / f).st_ino != os.stat(cur / f).st_ino
    assert xdev.identity()["volume_links"] == "ok:copy"
    assert xdev.rig.pointer() == ptr and not stamped_exists(xdev, "2323")
    assert len([c for c in xdev.rig.calls() if c["sub"] == "precompute-tokens"]) == 5
    f = xdev.marker_fields()
    assert f["links"] == "copy" and f["recipe"] == f0["recipe"] and f["budget"] == f0["budget"] and f["utc"] >= f0["utc"]
    # and honoured again
    r = xdev.run()
    assert r.returncode == 0 and HONOURED in r.stdout and xdev.rig.calls() == [] and len(xdev.superseded()) == 1

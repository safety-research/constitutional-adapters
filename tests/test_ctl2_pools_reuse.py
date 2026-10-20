"""scripts/sdfbase/ctl2_pools_reuse.py: the ONE-line REUSE/BUILD decision that makes
`02 --step pools` re-runnable past its ctl2 step, driven through the CLI with the same interpreter (argv, exit codes and the
stdout shape are what the remote chain's `read -r MODE DIR` sees). Both subcommands take the constv2-mix-full manifest the
chain just rebuilt (`--budget-manifest`) and compare its SEMANTIC budget identity (pool_sha256 + official_tokens, never the
manifest file's sha) with the recorded build's `budget_source`; a build stepped around leaves an
expectation sidecar the seeded rebuild must reproduce before `commit` re-points anything. The recorded build's RECIPE (its
manifests' `recipe`: version, control-input shas, id prefix, salts, split, tokenizer pins — never the checkout's commit) is
compared with today's builder's (`current_recipe`): equal -> REUSE, different -> BUILD a fresh stamped sibling (the old dir
untouched, never linked as current), absent -> refused by name. Synthetic pool trees only: placeholder
shas, invented rows, no real data."""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "scripts/sdfbase/ctl2_pools_reuse.py"
sys.path.insert(0, str(ROOT / "scripts/sdfbase"))
import build_ctl2_pools as b2  # noqa: E402
import ctl2_pools_reuse as cpr  # noqa: E402

CELLS = cpr.CELLS
BUDGET_CELL = cpr.BUDGET_CELL
STAMP_RE = re.compile(r"^\d{8}T\d{6}Z-pid\d+$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

#: the synthetic CURRENT budget: the mix-full pool's bytes (their sha is the identity) and its official token count
BUDGET_POOL = b'{"doc_id": "doc-0000", "instruction_id": "constv2-mix-full", "prompt": "p", "pos": "x0"}\n'
BUDGET_SHA = hashlib.sha256(BUDGET_POOL).hexdigest()
BUDGET_TOKENS = 12_345
OTHER_SHA = hashlib.sha256(BUDGET_POOL + b"changed\n").hexdigest()
DEFAULT = object()
#: today's DECLARED recipe (the builder's constants + this checkout's corpora/TOKENIZERS.json): what a build made by this
#: checkout's builder records, and what decide/commit compare a recorded build against
CURRENT_RECIPE = cpr.current_recipe()
PINS_FILE = ROOT / "corpora/TOKENIZERS.json"
FIRST_TOK = sorted(json.loads(PINS_FILE.read_text())["tokenizers"])[0]


def altered(recipe: dict, path: str, value) -> dict:
    """A deep copy of ``recipe`` with the dotted-path leaf ``path`` set to ``value``."""
    out = copy.deepcopy(recipe)
    node = out
    parts = path.split(".")
    for k in parts[:-1]:
        node = node[k]
    node[parts[-1]] = value
    return out


def leaf(recipe: dict, path: str):
    node = recipe
    for k in path.split("."):
        node = node[k]
    return node


# ------------------------------------------------------------------------------------------------------------------
# synthetic builds
# ------------------------------------------------------------------------------------------------------------------
def budget_source(pool_sha256=BUDGET_SHA, official_tokens=BUDGET_TOKENS, *, manifest_sha256="0" * 64) -> dict:
    """What build_ctl2_pools.read_budget records on the FULL cell's manifest (the shape of its return)."""
    return {"cell": BUDGET_CELL, "manifest": f"~/pools/pairs_{BUDGET_CELL}.manifest.json", "manifest_sha256": manifest_sha256,
            "pool_sha256": pool_sha256, "official_tokens": official_tokens, "limiting_tokenizer": "fake", "n_docs": 1}


def write_budget_manifest(path: Path, *, pool_sha256=BUDGET_SHA, official_tokens=BUDGET_TOKENS, cell=BUDGET_CELL,
                          generator_sha="a" * 40) -> Path:
    """A constv2-mix-full pool manifest as the chain leaves it before the decision: built (generator = the checkout's commit)
    and preflight-stamped. ``generator_sha`` changes the FILE's bytes without changing the budget identity."""
    m = {"cell": cell, "pool_sha256": pool_sha256, "official_tokens": official_tokens, "n_docs": 1, "doc_ids_in_order": ["doc-0000"],
         "generator": {"commit": generator_sha, "script": "build_pools.py"},
         "preflight": {"max_len": 8192, "tokenizers": {"fake": {"repo": "TESTHOST/fake", "revision": "0" * 40}}}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m, indent=1) + "\n")
    return path


def budget_manifest_for(root: Path, **kw) -> Path:
    """The current mix-full manifest beside a root (<root.parent>/pools/…), rewritten on every call the way the chain
    rewrites it every run."""
    return write_budget_manifest(Path(root).parent / "pools" / f"pairs_{BUDGET_CELL}.manifest.json", **kw)


def _rows(cell: str, n: int, salt: str = ""):
    # verify_pools requires instruction_id == cell and the ctl2- doc-id prefix
    return [{"doc_id": f"ctl2-doc-{i:04d}", "instruction_id": cell, "prompt": "p", "pos": f"x{salt}{i}"} for i in range(n)]


def write_cell(d: Path, cell: str, *, n=3, salt="", preflight=True, budget=DEFAULT, drop_pool=False,
               recipe=DEFAULT) -> dict:
    """One cell's pool + manifest (pool_sha256 = sha256 of the pool bytes). The FULL cell's manifest records ``budget``
    as its budget_source (the default synthetic budget; None omits it, the way build_cell's first write does). BOTH
    cells' manifests record ``recipe`` (today's declared recipe by default, the way the builder records it; None omits
    it, the way a build from before recipes were recorded looks)."""
    rows = _rows(cell, n, salt)
    data = "".join(json.dumps(r) + "\n" for r in rows).encode()
    if not drop_pool:
        (d / f"pairs_{cell}.jsonl").write_bytes(data)
    m = {"cell": cell, "pool_sha256": hashlib.sha256(data).hexdigest(), "n_docs": len(rows),
         "doc_ids_in_order": [r["doc_id"] for r in rows], "official_tokens": 10 * len(rows)}
    if cell == cpr.FULL_CELL:
        src = budget_source() if budget is DEFAULT else budget
        if src is not None:
            m["budget_source"] = src
    rec = CURRENT_RECIPE if recipe is DEFAULT else recipe
    if rec is not None:
        m["recipe"] = copy.deepcopy(rec)
    if preflight:
        m["preflight"] = {"max_len": 8192, "tokenizers": {"fake": {"repo": "TESTHOST/fake", "revision": "0" * 40}}}
    (d / f"pairs_{cell}.manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    return m


def write_build(d: Path, *, preflight=True, report=True, violations=(), summary=True, n=3, salt="", drop_pool=None,
                budget=DEFAULT, recipe=DEFAULT) -> Path:
    """A ctl2 build dir: both cells' pools + manifests, pools_summary.json and a preflight_report.json — each piece
    switchable off to make the dir INCOMPLETE in one named way; ``budget`` is the full cell's recorded budget_source,
    ``recipe`` both cells' recorded recipe (today's by default)."""
    d.mkdir(parents=True, exist_ok=True)
    for cell in CELLS:
        stamped = preflight is True or (isinstance(preflight, str) and preflight != cell)
        write_cell(d, cell, n=n, salt=salt, preflight=stamped, budget=budget, drop_pool=(drop_pool == cell), recipe=recipe)
    if summary:
        (d / "pools_summary.json").write_text(json.dumps({"cells": {c: {} for c in CELLS}}) + "\n")
    if report:
        (d / "preflight_report.json").write_text(json.dumps({"max_len": 8192, "violations": list(violations)}) + "\n")
    return d


def manifest_sha(d: Path, cell: str) -> str:
    return json.loads((d / f"pairs_{cell}.manifest.json").read_text())["pool_sha256"]


def snapshot(d: Path) -> dict:
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())} if d.exists() else {}


def cli(*args, env_extra=None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("SLURM_JOB_ID", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run([sys.executable, str(TOOL), *args], capture_output=True, text=True, env=env, cwd=str(ROOT))


def decide(root: Path, budget: Path | None = None, **kw) -> subprocess.CompletedProcess:
    budget = budget or budget_manifest_for(root)
    return cli("decide", "--root", str(root), "--budget-manifest", str(budget), **kw)


def commit(root: Path, built: Path, budget: Path | None = None, **kw) -> subprocess.CompletedProcess:
    budget = budget or budget_manifest_for(root)
    return cli("commit", "--root", str(root), "--dir", str(built), "--budget-manifest", str(budget), **kw)


def one_line(r: subprocess.CompletedProcess) -> str:
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert len(lines) == 1, r.stdout          # the chain does `read -r MODE DIR` on stdout: exactly one line, nothing else
    return lines[0]


def refused(r: subprocess.CompletedProcess) -> str:
    assert r.returncode == cpr.RC_REFUSED, (r.returncode, r.stdout, r.stderr)
    assert r.stdout == "", r.stdout            # a refusal leaves stdout EMPTY so `read -r` never sees half a decision
    assert r.stderr.startswith("REFUSED: "), r.stderr
    return r.stderr


# ------------------------------------------------------------------------------------------------------------------
# contracts shared with the builder
# ------------------------------------------------------------------------------------------------------------------
def test_cell_names_are_the_builders():
    assert (cpr.HELDOUT_CELL, cpr.FULL_CELL) == (b2.HELDOUT_CELL, b2.FULL_CELL)
    assert cpr.BUDGET_CELL == b2.BUDGET_CELL == "constv2-mix-full"
    assert cpr.pointer_path(Path("/x/pools_ctl2")) == Path("/x/pools_ctl2.CURRENT")
    assert cpr.fresh_build_dir(Path("/x/pools_ctl2"), "job7") == Path("/x/pools_ctl2.build-job7")
    assert cpr.expect_path(Path("/x/pools_ctl2.build-job7")) == Path("/x/pools_ctl2.build-job7.expect.json")
    assert cpr.expect_checked_path(Path("/x/pools_ctl2.build-job7")) == Path("/x/pools_ctl2.build-job7.expect.checked.json")
    # the identity keys are exactly the two read_budget fields the module docstring argues for
    assert cpr.BUDGET_KEYS == ("pool_sha256", "official_tokens")


def test_the_synthetic_budget_source_has_read_budgets_shape(tmp_path):
    """The fixture's budget_source is what the REAL read_budget returns for the fixture's manifest (keys + identity values),
    so the comparisons below exercise the shape the builder records."""
    mp = budget_manifest_for(tmp_path / "pools_ctl2")
    real = b2.read_budget(mp)
    fake = budget_source()
    assert set(fake) <= set(real), (set(fake) - set(real))
    assert {k: real[k] for k in cpr.BUDGET_KEYS} == {k: fake[k] for k in cpr.BUDGET_KEYS}
    assert real["manifest_sha256"] != fake["manifest_sha256"]        # the file's sha is NOT part of the identity


# ------------------------------------------------------------------------------------------------------------------
# decide
# ------------------------------------------------------------------------------------------------------------------
def test_missing_or_empty_root_builds_into_the_root(tmp_path):
    root = tmp_path / "pools_ctl2"
    assert one_line(decide(root)) == f"BUILD {root}"
    assert not root.exists()                   # decide names the target; the chain mkdir -p's it
    root.mkdir()
    assert one_line(decide(root)) == f"BUILD {root}"
    assert not list(tmp_path.glob("pools_ctl2*.expect*"))            # nothing to expect from nothing


def test_complete_consistent_build_is_reused_untouched(tmp_path):
    root = write_build(tmp_path / "pools_ctl2")
    before = snapshot(root)
    r = decide(root)
    assert one_line(r) == f"REUSE {root}"
    assert r.stderr == ""
    assert snapshot(root) == before
    assert not cpr.pointer_path(root).exists()  # decide never writes a pointer; only commit does


def test_the_budget_manifest_is_required_and_read_on_every_path(tmp_path):
    root = tmp_path / "pools_ctl2"
    r = cli("decide", "--root", str(root))
    assert r.returncode == 2 and "--budget-manifest" in r.stderr, r.stderr          # argparse: required
    # a first run (nothing to reuse) still reads it: a missing / foreign / unreadable manifest refuses before anything builds
    err = refused(decide(root, tmp_path / "pools" / "nowhere.manifest.json"))
    assert "budget manifest" in err and "missing" in err and BUDGET_CELL in err, err
    err = refused(decide(root, budget_manifest_for(root, cell="constv2-mix-10m")))
    assert "constv2-mix-10m" in err and f"not {BUDGET_CELL!r}" in err, err
    bad = tmp_path / "pools" / "bad.manifest.json"
    bad.write_text("{not json")
    assert "unreadable" in refused(decide(root, bad))
    zero = budget_manifest_for(root, official_tokens=0)
    assert "official_tokens=0" in refused(decide(root, zero))
    assert one_line(decide(root)) == f"BUILD {root}"


@pytest.mark.parametrize("field", ["official_tokens", "pool_sha256"])
def test_a_complete_build_cut_against_a_different_budget_refuses_by_name(tmp_path, field):
    root = write_build(tmp_path / "pools_ctl2")
    before = snapshot(root)
    other = {"official_tokens": BUDGET_TOKENS + 1, "pool_sha256": OTHER_SHA}[field]
    current = budget_manifest_for(root, **{field: other})
    err = refused(decide(root, current))
    assert str(root) in err and f"different {BUDGET_CELL} budget" in err, err
    recorded, now = {"official_tokens": (str(BUDGET_TOKENS), str(BUDGET_TOKENS + 1)), "pool_sha256": (BUDGET_SHA, OTHER_SHA)}[field]
    assert recorded in err and now in err, err                       # BOTH values are named
    assert "budget_source" in err and str(current) in err, err
    assert "neither reused, rebuilt over nor deleted" in err and "move it aside by hand" in err and ".CURRENT" in err, err
    assert snapshot(root) == before and not cpr.pointer_path(root).exists()
    assert not list(tmp_path.glob("pools_ctl2*.expect*"))
    # the same budget again -> REUSE (the refusal was about the budget, not the dir)
    assert one_line(decide(root)) == f"REUSE {root}"


def test_the_identity_is_semantic_a_rebuilt_manifest_file_with_the_same_budget_reuses(tmp_path):
    """The chain rewrites the mix-full manifest every run (generator = the checkout's commit; the preflight stamp), so its
    FILE sha differs between runs; the recorded build is still the same budget."""
    root = write_build(tmp_path / "pools_ctl2", budget=budget_source(manifest_sha256="f" * 64))
    current = budget_manifest_for(root, generator_sha="b" * 40)
    assert hashlib.sha256(current.read_bytes()).hexdigest() != "f" * 64
    assert one_line(decide(root, current)) == f"REUSE {root}"


def test_a_complete_build_without_budget_source_is_unverifiable_and_refuses(tmp_path):
    root = write_build(tmp_path / "pools_ctl2", budget=None)
    before = snapshot(root)
    err = refused(decide(root))
    assert "no budget_source" in err and "unverifiable" in err and str(root) in err and BUDGET_CELL in err, err
    assert "neither reused, rebuilt over nor deleted" in err, err
    assert snapshot(root) == before
    # a budget_source with an unusable identity refuses too
    write_build(root, budget=budget_source(pool_sha256=""))
    assert "pool_sha256=''" in refused(decide(root))


def test_a_pointer_whose_recorded_budget_differs_refuses_as_a_belt(tmp_path):
    root = write_build(tmp_path / "pools_ctl2")
    one_line(commit(root, root))
    ptr = cpr.pointer_path(root)
    rec = json.loads(ptr.read_text())
    assert rec["budget"] == {"pool_sha256": BUDGET_SHA, "official_tokens": BUDGET_TOKENS}
    rec["budget"]["official_tokens"] = BUDGET_TOKENS + 7                # the dir's manifest still agrees; the pointer does not
    ptr.write_text(json.dumps(rec) + "\n")
    err = refused(decide(root))
    assert str(ptr) in err and str(BUDGET_TOKENS + 7) in err and str(BUDGET_TOKENS) in err and str(root) in err, err
    # a pointer from before budgets were recorded carries none and is judged by the dir's manifest alone
    del rec["budget"]
    ptr.write_text(json.dumps(rec) + "\n")
    assert one_line(decide(root)) == f"REUSE {root}"


@pytest.mark.parametrize("breakage", ["no_report", "no_preflight_stamp_heldout", "no_preflight_stamp_full", "violations",
                                      "no_summary", "no_pool_full"])
def test_incomplete_build_goes_to_a_stamped_sibling_and_the_partial_stays(tmp_path, breakage):
    root = tmp_path / "pools_ctl2"
    kw = {"no_report": dict(report=False), "no_preflight_stamp_heldout": dict(preflight=cpr.HELDOUT_CELL),
          "no_preflight_stamp_full": dict(preflight=cpr.FULL_CELL),
          "violations": dict(violations=[{"pool": "pairs_constv2ctl2-full.jsonl", "line": 1, "doc_id": "ctl2-doc-0000"}]),
          "no_summary": dict(summary=False), "no_pool_full": dict(drop_pool=cpr.FULL_CELL)}[breakage]
    write_build(root, **kw)
    before = snapshot(root)
    r = decide(root)
    mode, where = one_line(r).split(" ", 1)
    assert mode == "BUILD"
    target = Path(where)
    assert target.parent == root.parent and target.name.startswith("pools_ctl2.build-"), where
    assert STAMP_RE.match(target.name[len("pools_ctl2.build-"):]), where
    assert not target.exists()                 # named, not created
    assert snapshot(root) == before            # the partial build is left exactly as it was
    assert "incomplete" in r.stderr and str(root) in r.stderr and "left in place" in r.stderr, r.stderr
    assert r.stderr.startswith("note: ")       # an aside, never a refusal
    # every partial here carries both manifests with a pool_sha256 and the current budget: the sidecar records both cells
    side = cpr.expect_path(target)
    assert side.is_file() and side.parent == root.parent, side
    exp = json.loads(side.read_text())
    assert exp["cells"] == {cell: manifest_sha(root, cell) for cell in CELLS}
    assert exp["previous_dir"] == str(root) and exp["target"] == str(target) and exp["skipped"] == {}
    assert exp["budget"] == {"pool_sha256": BUDGET_SHA, "official_tokens": BUDGET_TOKENS} and UTC_RE.match(exp["utc"])
    assert not list(tmp_path.glob("*.tmp.*"))  # atomic
    assert "must reproduce" in r.stderr and str(side) in r.stderr, r.stderr


def test_the_stamp_is_the_slurm_job_id_under_slurm(tmp_path):
    root = write_build(tmp_path / "pools_ctl2", report=False)
    assert one_line(decide(root, env_extra={"SLURM_JOB_ID": "4242"})) == f"BUILD {root}.build-job4242"
    # an empty SLURM_JOB_ID is "not under Slurm"
    line = one_line(decide(root, env_extra={"SLURM_JOB_ID": ""}))
    assert STAMP_RE.match(line.split(".build-", 1)[1]), line


def test_the_sidecar_records_only_cells_whose_bytes_are_expected_to_reproduce(tmp_path):
    # (a) the partial's full manifest carries no budget_source (build_cell's first write): heldout only, the skip named
    root = write_build(tmp_path / "pools_ctl2", report=False, budget=None)
    r = decide(root, env_extra={"SLURM_JOB_ID": "1"})
    assert one_line(r) == f"BUILD {root}.build-job1"
    exp = json.loads(cpr.expect_path(Path(f"{root}.build-job1")).read_text())
    assert exp["cells"] == {cpr.HELDOUT_CELL: manifest_sha(root, cpr.HELDOUT_CELL)}
    assert "no budget_source" in exp["skipped"][cpr.FULL_CELL]
    assert f"no reproduction expectation for {cpr.FULL_CELL}" in r.stderr and "no budget_source" in r.stderr, r.stderr
    # (b) the partial was cut against a DIFFERENT budget: heldout only, the skip names both budgets; no refusal (stepped around)
    write_build(root, report=False, budget=budget_source(official_tokens=99))
    r = decide(root, env_extra={"SLURM_JOB_ID": "2"})
    assert one_line(r) == f"BUILD {root}.build-job2"
    exp = json.loads(cpr.expect_path(Path(f"{root}.build-job2")).read_text())
    assert exp["cells"] == {cpr.HELDOUT_CELL: manifest_sha(root, cpr.HELDOUT_CELL)}
    assert "different" in exp["skipped"][cpr.FULL_CELL] and "99" in exp["skipped"][cpr.FULL_CELL] and str(BUDGET_TOKENS) in exp["skipped"][cpr.FULL_CELL]
    assert "different" in r.stderr, r.stderr
    # (c) a partial with no manifest at all (a stray file): BUILD, no sidecar
    stray = tmp_path / "other_ctl2"
    stray.mkdir()
    (stray / "pairs_constv2ctl2-heldout.jsonl").write_text("{}\n")
    r = decide(stray, env_extra={"SLURM_JOB_ID": "3"})
    assert one_line(r) == f"BUILD {stray}.build-job3"
    assert not cpr.expect_path(Path(f"{stray}.build-job3")).exists()
    assert "no reproduction expectation" in r.stderr


def test_a_pre_existing_sidecar_for_the_target_refuses_like_a_pre_existing_target(tmp_path):
    root = write_build(tmp_path / "pools_ctl2", report=False)
    target = Path(f"{root}.build-job5")
    before = snapshot(root)
    cpr.expect_path(target).write_text("{}\n")
    err = refused(decide(root, env_extra={"SLURM_JOB_ID": "5"}))
    assert str(cpr.expect_path(target)) in err and "already exists" in err and str(root) in err, err
    assert cpr.expect_path(target).read_text() == "{}\n" and snapshot(root) == before and not target.exists()
    cpr.expect_path(target).unlink()
    cpr.expect_checked_path(target).write_text("{}\n")
    err = refused(decide(root, env_extra={"SLURM_JOB_ID": "5"}))
    assert str(cpr.expect_checked_path(target)) in err and "already exists" in err, err


def test_edited_pool_bytes_refuse_by_cell_and_delete_nothing(tmp_path):
    root = write_build(tmp_path / "pools_ctl2")
    pool = root / f"pairs_{cpr.FULL_CELL}.jsonl"
    with open(pool, "a") as fh:
        fh.write(json.dumps({"doc_id": "ctl2-doc-9999", "instruction_id": cpr.FULL_CELL, "prompt": "p", "pos": "extra"}) + "\n")
    before = snapshot(root)
    err = refused(decide(root))
    assert cpr.FULL_CELL in err and "sha256" in err and str(root) in err, err
    assert "never" in err or "neither rebuilt over nor deleted" in err, err
    assert snapshot(root) == before            # a disagreeing build is neither rebuilt over nor deleted
    # a wrong instruction_id is the builder's own verify refusal, surfaced the same way
    root2 = write_build(tmp_path / "other_ctl2")
    pool2 = root2 / f"pairs_{cpr.HELDOUT_CELL}.jsonl"
    rows = [json.loads(ln) for ln in pool2.read_text().splitlines()]
    rows[0]["instruction_id"] = "constv2-heldout"
    pool2.write_text("".join(json.dumps(r) + "\n" for r in rows))
    err2 = refused(decide(root2))
    assert "instruction_id" in err2, err2


def test_a_root_with_whitespace_is_refused(tmp_path):
    root = tmp_path / "pools ctl2"
    err = refused(decide(root))
    assert "whitespace" in err


# ------------------------------------------------------------------------------------------------------------------
# commit + the pointer
# ------------------------------------------------------------------------------------------------------------------
def test_commit_writes_the_pointer_and_a_later_decide_reuses_the_pointed_dir(tmp_path):
    root = write_build(tmp_path / "pools_ctl2", report=False)           # the canonical dir is a PARTIAL build
    built = write_build(tmp_path / "pools_ctl2.build-job7", salt="b")   # the fresh build the chain just finished
    r = commit(root, built, env_extra={"SLURM_JOB_ID": "7"})
    assert one_line(r) == f"CURRENT {built}"
    ptr = cpr.pointer_path(root)
    assert ptr.is_file() and ptr == tmp_path / "pools_ctl2.CURRENT"
    rec = json.loads(ptr.read_text())
    assert rec["dir"] == str(built) and rec["root"] == str(root) and rec["slurm_job_id"] == "7"
    assert UTC_RE.match(rec["utc"])
    for cell in CELLS:
        assert rec["cells"][cell] == manifest_sha(built, cell)
    assert rec["budget"] == {"pool_sha256": BUDGET_SHA, "official_tokens": BUDGET_TOKENS}
    assert rec["budget_manifest"] == str(budget_manifest_for(root)) and rec["reproduced_from"] is None   # no sidecar was there
    assert not list(tmp_path.glob("pools_ctl2.CURRENT.tmp.*"))         # atomic: tmp + os.replace, nothing left behind
    # the pointer, not the partial root, is what the next run reuses
    assert one_line(decide(root)) == f"REUSE {built}"
    # a second commit of the same dir is idempotent (a re-run that rebuilt nothing may still commit)
    assert one_line(commit(root, built)) == f"CURRENT {built}"
    assert json.loads(ptr.read_text())["dir"] == str(built)


def test_commit_refuses_an_incomplete_or_inconsistent_dir_and_writes_no_pointer(tmp_path):
    root = tmp_path / "pools_ctl2"
    partial = write_build(tmp_path / "pools_ctl2.build-job1", report=False)
    err = refused(commit(root, partial))
    assert "not a complete ctl2 build" in err and "preflight_report.json missing" in err, err
    assert not cpr.pointer_path(root).exists()
    bad = write_build(tmp_path / "pools_ctl2.build-job2")
    (bad / f"pairs_{cpr.HELDOUT_CELL}.jsonl").write_bytes(b'{"doc_id": "ctl2-doc-0000", "instruction_id": "constv2ctl2-heldout", "prompt": "p", "pos": "changed"}\n')
    err = refused(commit(root, bad))
    assert cpr.HELDOUT_CELL in err, err
    assert not cpr.pointer_path(root).exists()
    # a missing dir
    err = refused(commit(root, tmp_path / "nowhere"))
    assert "not a directory" in err


@pytest.mark.parametrize("field", ["official_tokens", "pool_sha256"])
def test_commit_refuses_a_build_whose_recorded_budget_differs_from_the_current_one(tmp_path, field):
    root = tmp_path / "pools_ctl2"
    built = write_build(tmp_path / "pools_ctl2.build-job8")
    other = {"official_tokens": BUDGET_TOKENS + 1, "pool_sha256": OTHER_SHA}[field]
    current = budget_manifest_for(root, **{field: other})
    err = refused(commit(root, built, current))
    assert str(built) in err and f"different {BUDGET_CELL} budget" in err and "neither recorded as CURRENT" in err, err
    assert str(other) in err and {"official_tokens": str(BUDGET_TOKENS), "pool_sha256": BUDGET_SHA}[field] in err, err
    assert not cpr.pointer_path(root).exists()
    # a build with no budget_source is unverifiable at commit too
    err = refused(commit(root, write_build(tmp_path / "pools_ctl2.build-job9", budget=None)))
    assert "no budget_source" in err and "unverifiable" in err
    assert not cpr.pointer_path(root).exists()


def test_commit_refuses_a_rebuild_that_does_not_reproduce_the_recorded_bytes_then_records_one_that_does(tmp_path):
    root = write_build(tmp_path / "pools_ctl2", report=False)           # a partial: both manifests record shas, the current budget
    r = decide(root, env_extra={"SLURM_JOB_ID": "7"})
    target = Path(f"{root}.build-job7")
    assert one_line(r) == f"BUILD {target}"
    side = cpr.expect_path(target)
    side_bytes = side.read_bytes()
    # the "seeded rebuild": heldout reproduces, the full cell does NOT
    write_build(target)
    write_cell(target, cpr.FULL_CELL, salt="not-the-same")
    produced, recorded = manifest_sha(target, cpr.FULL_CELL), manifest_sha(root, cpr.FULL_CELL)
    assert produced != recorded
    before = snapshot(target)
    err = refused(commit(root, target, env_extra={"SLURM_JOB_ID": "7"}))
    assert f"seeded rebuild of {cpr.FULL_CELL}" in err and str(target) in err, err
    assert produced in err and recorded in err and f"previous build at {root}" in err, err
    assert "not reproducing the recorded bytes" in err and "NOT recorded as CURRENT" in err and "NOT re-pointed" in err, err
    assert not cpr.pointer_path(root).exists()                          # nothing written …
    assert side.is_file() and side.read_bytes() == side_bytes           # … the sidecar left exactly as it was
    assert not cpr.expect_checked_path(target).exists()
    assert snapshot(target) == before and snapshot(root)                # both dirs untouched (inspect by hand)
    # the builder reproducing the bytes: commit records CURRENT, the pointer carries the budget and the provenance, the
    # sidecar is renamed (never deleted)
    write_cell(target, cpr.FULL_CELL)
    assert manifest_sha(target, cpr.FULL_CELL) == recorded
    assert one_line(commit(root, target, env_extra={"SLURM_JOB_ID": "7"})) == f"CURRENT {target}"
    rec = json.loads(cpr.pointer_path(root).read_text())
    assert rec["dir"] == str(target) and rec["reproduced_from"] == str(root)
    assert rec["budget"] == {"pool_sha256": BUDGET_SHA, "official_tokens": BUDGET_TOKENS}
    assert rec["cells"] == {cell: manifest_sha(target, cell) for cell in CELLS}
    assert not side.exists() and cpr.expect_checked_path(target).read_bytes() == side_bytes
    assert one_line(decide(root, env_extra={"SLURM_JOB_ID": "7"})) == f"REUSE {target}"


def test_commit_refuses_a_sidecar_written_against_another_budget_or_malformed(tmp_path):
    root = write_build(tmp_path / "pools_ctl2", report=False)
    target = Path(f"{root}.build-job6")
    one_line(decide(root, env_extra={"SLURM_JOB_ID": "6"}))
    write_build(target)
    side = cpr.expect_path(target)
    exp = json.loads(side.read_text())
    exp["budget"]["official_tokens"] = BUDGET_TOKENS + 3                # the budget moved between the decision and the commit
    side.write_text(json.dumps(exp) + "\n")
    err = refused(commit(root, target))
    assert "changed between the decision and this commit" in err and str(BUDGET_TOKENS + 3) in err and str(BUDGET_TOKENS) in err, err
    assert not cpr.pointer_path(root).exists() and side.is_file()
    side.write_text("{not json")
    assert "unreadable" in refused(commit(root, target))
    side.write_text(json.dumps({"cells": {}}) + "\n")
    assert "malformed" in refused(commit(root, target))
    exp["budget"]["official_tokens"] = BUDGET_TOKENS
    exp["cells"] = {"constv2ctl2-other": "0" * 64}
    side.write_text(json.dumps(exp) + "\n")
    assert "unknown cell" in refused(commit(root, target))
    assert not cpr.pointer_path(root).exists()


def test_dangling_pointer_refuses(tmp_path):
    root = write_build(tmp_path / "pools_ctl2")
    gone = tmp_path / "pools_ctl2.build-job3"
    cpr.pointer_path(root).write_text(json.dumps({"dir": str(gone), "utc": "2026-01-01T00:00:00Z", "cells": {}}) + "\n")
    err = refused(decide(root))
    assert "dangling" in err and str(gone) in err and str(cpr.pointer_path(root)) in err, err
    # a pointer naming an EMPTY dir is dangling too (commit only ever records a complete build)
    gone.mkdir()
    err = refused(decide(root))
    assert "dangling" in err and "empty" in err, err
    # an unreadable pointer refuses rather than falling back to the root
    cpr.pointer_path(root).write_text("{not json")
    err = refused(decide(root))
    assert "unreadable" in err


def test_an_existing_stamped_target_refuses(tmp_path):
    root = write_build(tmp_path / "pools_ctl2", report=False)
    (tmp_path / "pools_ctl2.build-job99").mkdir()
    before = snapshot(root)
    err = refused(decide(root, env_extra={"SLURM_JOB_ID": "99"}))
    assert str(tmp_path / "pools_ctl2.build-job99") in err and "already exists" in err, err
    assert snapshot(root) == before
    assert not cpr.expect_path(tmp_path / "pools_ctl2.build-job99").exists()   # refused before any sidecar


def test_pointer_recorded_sha_that_differs_from_the_dir_refuses(tmp_path):
    root = write_build(tmp_path / "pools_ctl2")
    one_line(commit(root, root))
    # a build swapped out consistently (pool AND manifest) under its pointer: verify_pools agrees, the pointer does not
    write_build(root, salt="swapped")
    err = refused(decide(root))
    assert "pointer" in err and str(cpr.pointer_path(root)) in err and "sha256" in err, err


def test_pure_pieces(tmp_path):
    assert cpr.completeness(tmp_path / "missing") == "not a directory"
    d = write_build(tmp_path / "d")
    assert cpr.completeness(d) is None
    (d / "preflight_report.json").write_text(json.dumps({"max_len": 8192}) + "\n")       # no violations key at all
    assert "violations" in cpr.completeness(d)
    (d / "preflight_report.json").write_text("nope")
    assert "unreadable" in cpr.completeness(d)
    assert cpr.build_stamp({"SLURM_JOB_ID": " 12 "}) == "job12"
    assert cpr.build_stamp({}, now=0, pid=5) == "19700101T000000Z-pid5"
    assert cpr.budget_identity({"pool_sha256": "ab", "official_tokens": 3, "extra": 1}, where="x") == {"pool_sha256": "ab", "official_tokens": 3}
    for bad in ({"pool_sha256": "ab", "official_tokens": True}, {"pool_sha256": "ab", "official_tokens": "3"},
                {"pool_sha256": None, "official_tokens": 3}, "not a dict"):
        with pytest.raises(cpr.Refusal):
            cpr.budget_identity(bad, where="x")


# ------------------------------------------------------------------------------------------------------------------
# the recipe identity (P1 follow-up): what was sampled and how, independent of the budget
# ------------------------------------------------------------------------------------------------------------------
def test_current_recipe_is_the_builders_declared_identity_and_carries_no_commit():
    cur = CURRENT_RECIPE
    assert list(cur) == ["version", "release", "release_bytes_pinned", "corpus_shas", "id_prefix", "cells", "budget_cell",
                         "salts", "split", "tokenizers"]
    assert cur["version"] == b2.CTL2_RECIPE_VERSION == "ctl2-recipe-v1"
    # the chain always builds PINNED: the release's bytes, its label
    assert cur["release"] == b2.CORPUS_RELEASE and cur["release_bytes_pinned"] is True and cur["corpus_shas"] == b2.RELEASE_SHAS
    assert cur["id_prefix"] == b2.CTL2_ID_PREFIX == "ctl2-" and cur["cells"] == list(CELLS) and cur["budget_cell"] == BUDGET_CELL
    # the salts / split are the builder's DEFAULTS (the chain line passes no --salt) …
    sig = inspect.signature(b2.build_ctl2).parameters
    assert cur["salts"] == {"heldout": sig["salt"].default, "eval": sig["eval_salt"].default, "draw": b2.bp.SALT_CTL}
    assert cur["split"] == {"frac": sig["frac"].default, "eval_n": sig["eval_n"].default}
    # … which are the build_heldout_split constants the module declares from (a drift of either is caught here)
    hs = b2.hsplit
    assert cur["salts"] == {"heldout": hs.SALT_HELDOUT, "eval": hs.SALT_EVAL, "draw": b2.bp.SALT_CTL}
    assert cur["split"] == {"frac": hs.FRAC, "eval_n": hs.EVAL_N}
    # the tokenizer pins are this checkout's corpora/TOKENIZERS.json: repo + revision + per-file sha256 per family, the
    # file named repo-relative (never an absolute path) with its own sha256 as provenance; no rig (this tool stages nothing)
    pins = json.loads(PINS_FILE.read_text())["tokenizers"]
    tok = cur["tokenizers"]
    assert list(tok) == ["pins_file", "pins_sha256", "pins"]
    assert tok["pins_file"] == "corpora/TOKENIZERS.json" and tok["pins_sha256"] == hashlib.sha256(PINS_FILE.read_bytes()).hexdigest()
    assert tok["pins"] == {n: {"repo": t["repo"], "revision": t["revision"], "files": {f["name"]: f["sha256"] for f in t["files"]}}
                           for n, t in pins.items()}
    assert set(tok["pins"]) == {"gemma", "llama3", "qwen"}
    # NEVER the checkout's commit, nothing of the generator provenance; a JSON round trip is the identity (manifests carry JSON)
    text = json.dumps(cur)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True).stdout.strip()
    assert len(head) == 40 and head not in text
    assert not any(k in text for k in ("sisteer_commit", "corpus_generator_commit", "generator", "dirty", str(Path.home())))
    assert json.loads(text) == cur
    assert cpr.current_recipe() == cur                                  # deterministic


def test_compare_recipe_names_each_differing_leaf_and_skips_provenance_only():
    cur = CURRENT_RECIPE
    assert cpr.RECIPE_PROVENANCE_ONLY == ("tokenizers.pins_sha256", "tokenizers.rig")
    assert cpr.compare_recipe(copy.deepcopy(cur), cur) == []
    rec = altered(cur, "salts.heldout", "sdfbase-heldout-v0")
    assert cpr.compare_recipe(rec, cur) == [f"salts.heldout: sdfbase-heldout-v0 (recorded) != {cur['salts']['heldout']} (current)"]
    # provenance only: the pins FILE's sha and the staged rig's own manifest are recorded, never compared
    rec = altered(cur, "tokenizers.pins_sha256", "0" * 64)
    rec["tokenizers"]["rig"] = {"tokenizers": {"llama3": {"snapshot": "x"}}}
    assert cpr.compare_recipe(rec, cur) == []
    # a leaf the recorded block lacks is named as not recorded
    rec = copy.deepcopy(cur)
    del rec["split"]["eval_n"]
    assert cpr.compare_recipe(rec, cur) == [f"split.eval_n: (not recorded) != {cur['split']['eval_n']} (current)"]
    # a leaf beyond the declared ones is ignored (an algorithm change is what the version bump is for)
    rec = copy.deepcopy(cur)
    rec["extra"] = 1
    rec["salts"]["more"] = "x"
    assert cpr.compare_recipe(rec, cur) == []
    assert cpr.compare_recipe("nope", cur) == ["recipe: nope (recorded) is not a JSON object"]
    # several differences, sorted by path; non-string leaves in JSON
    diffs = cpr.compare_recipe(altered(altered(cur, "version", "ctl2-recipe-v0"), "cells", ["a"]), cur)
    assert diffs == [f"cells: [\"a\"] (recorded) != {json.dumps(list(CELLS))} (current)",
                     "version: ctl2-recipe-v0 (recorded) != ctl2-recipe-v1 (current)"]
    pin = f"tokenizers.pins.{FIRST_TOK}.revision"
    assert cpr.compare_recipe(altered(cur, pin, "f" * 40), cur) == [f"{pin}: {'f' * 40} (recorded) != {leaf(cur, pin)} (current)"]


RECIPE_CHANGES = {
    "salt": ("salts.heldout", "sdfbase-heldout-v0"),
    "corpus_sha": ("corpus_shas.control_advice", "1" * 64),
    "tokenizer_pin": (f"tokenizers.pins.{FIRST_TOK}.revision", "f" * 40),
    "version": ("version", "ctl2-recipe-v0"),
}


@pytest.mark.parametrize("change", sorted(RECIPE_CHANGES))
def test_a_complete_build_under_a_different_recipe_is_not_reused_but_built_beside(tmp_path, change):
    """SAME budget identity, complete and byte-coherent, but the recorded recipe differs in one field: decide says BUILD
    into a stamped sibling (never REUSE), names the field on stderr, leaves the old dir untouched and writes a sidecar
    with NO cell expectations (the bytes are expected to differ) that says why."""
    path, value = RECIPE_CHANGES[change]
    root = write_build(tmp_path / "pools_ctl2", recipe=altered(CURRENT_RECIPE, path, value))
    before = snapshot(root)
    r = decide(root, env_extra={"SLURM_JOB_ID": "11"})
    target = Path(f"{root}.build-job11")
    assert one_line(r) == f"BUILD {target}"                              # NOT REUSE
    assert r.stderr.startswith("note: ") and "REFUSED" not in r.stderr
    expected_diff = f"{path}: {value} (recorded) != {leaf(CURRENT_RECIPE, path)} (current)"
    assert expected_diff in r.stderr, r.stderr
    assert str(root) in r.stderr and f"different recipe ({path})" in r.stderr and "left in place" in r.stderr, r.stderr
    assert "NOT reused" in r.stderr and str(target) in r.stderr and "no reproduction expectation" in r.stderr, r.stderr
    assert snapshot(root) == before and not target.exists() and not cpr.pointer_path(root).exists()
    side = cpr.expect_path(target)
    exp = json.loads(side.read_text())
    assert exp["cells"] == {} and exp["previous_dir"] == str(root) and exp["target"] == str(target)
    assert set(exp["skipped"]) == set(CELLS)
    assert all(v.startswith(f"recipe changed ({path})") and "expected to differ" in v for v in exp["skipped"].values())
    assert exp["recipe_changed"] == [expected_diff]
    assert exp["budget"] == {"pool_sha256": BUDGET_SHA, "official_tokens": BUDGET_TOKENS} and UTC_RE.match(exp["utc"])
    # the same dir recording today's recipe -> REUSE: the decision was about the recipe, not the dir
    side.unlink()
    write_build(root)
    r = decide(root, env_extra={"SLURM_JOB_ID": "11"})
    assert one_line(r) == f"REUSE {root}" and r.stderr == ""


def test_after_a_recipe_change_the_fresh_build_commits_and_supersedes_the_old_dir(tmp_path):
    root = write_build(tmp_path / "pools_ctl2", recipe=altered(CURRENT_RECIPE, "salts.eval", "sdfbase-heldout-eval-v0"))
    before = snapshot(root)
    target = Path(f"{root}.build-job12")
    assert one_line(decide(root, env_extra={"SLURM_JOB_ID": "12"})) == f"BUILD {target}"
    side_bytes = cpr.expect_path(target).read_bytes()
    write_build(target, salt="today")                                    # the fresh build: other bytes (expected), today's recipe
    assert manifest_sha(target, cpr.FULL_CELL) != manifest_sha(root, cpr.FULL_CELL)
    assert one_line(commit(root, target, env_extra={"SLURM_JOB_ID": "12"})) == f"CURRENT {target}"
    rec = json.loads(cpr.pointer_path(root).read_text())
    assert rec["dir"] == str(target) and rec["previous_dir"] == str(root) and rec["reproduced_from"] is None
    assert rec["recipe"] == CURRENT_RECIPE
    assert rec["recipe_changed"] == [f"salts.eval: sdfbase-heldout-eval-v0 (recorded) != {CURRENT_RECIPE['salts']['eval']} (current)"]
    assert not cpr.expect_path(target).exists() and cpr.expect_checked_path(target).read_bytes() == side_bytes
    assert snapshot(root) == before                                      # the old selection: untouched, never deleted
    assert one_line(decide(root, env_extra={"SLURM_JOB_ID": "13"})) == f"REUSE {target}"
    # a partial stepped around records no recipe change on the pointer
    root2 = write_build(tmp_path / "other_ctl2", report=False)
    t2 = Path(f"{root2}.build-job12")
    assert one_line(decide(root2, env_extra={"SLURM_JOB_ID": "12"})) == f"BUILD {t2}"
    write_build(t2)
    one_line(commit(root2, t2))
    rec2 = json.loads(cpr.pointer_path(root2).read_text())
    assert rec2["previous_dir"] == rec2["reproduced_from"] == str(root2) and rec2["recipe_changed"] is None


def test_a_complete_build_without_a_recipe_record_refuses_by_name(tmp_path):
    root = write_build(tmp_path / "pools_ctl2", recipe=None)             # built before recipes were recorded
    before = snapshot(root)
    err = refused(decide(root))
    assert "no recipe record" in err and "built before the recipe was recorded" in err and str(root) in err, err
    assert "cannot be checked against today's builder" in err and "neither reused, rebuilt over nor deleted" in err, err
    assert "move it aside by hand" in err and ".CURRENT" in err and "never silently reused" in err, err
    assert snapshot(root) == before and not list(tmp_path.glob("pools_ctl2*.expect*")) and not cpr.pointer_path(root).exists()
    # the heldout manifest lacking the block while the full cell carries it: mixed manifests, refused
    write_build(root)
    mh = root / f"pairs_{cpr.HELDOUT_CELL}.manifest.json"
    m = json.loads(mh.read_text())
    del m["recipe"]
    mh.write_text(json.dumps(m, indent=1) + "\n")
    err = refused(decide(root))
    assert "records no recipe while" in err and "tampered or mixed" in err, err
    # … and the two cells disagreeing on a field
    m["recipe"] = altered(CURRENT_RECIPE, "salts.draw", "sdfbase-ctl-draw-v0")
    mh.write_text(json.dumps(m, indent=1) + "\n")
    err = refused(decide(root))
    assert "disagree on the recipe (salts.draw)" in err and "inspect them by hand" in err, err
    # commit says the same about a build lacking one: nothing recorded as CURRENT
    err = refused(commit(root, write_build(tmp_path / "pools_ctl2.build-job9", recipe=None)))
    assert "no recipe record" in err and "recorded as CURRENT" in err and not cpr.pointer_path(root).exists(), err


def test_commit_refuses_a_build_whose_recipe_is_not_this_checkouts(tmp_path):
    root = tmp_path / "pools_ctl2"
    built = write_build(tmp_path / "pools_ctl2.build-job8", recipe=altered(CURRENT_RECIPE, "id_prefix", "ctl-"))
    err = refused(commit(root, built))
    assert f"{built}: the build's recorded recipe differs from this checkout's builder" in err, err
    assert "id_prefix: ctl- (recorded) != ctl2- (current)" in err, err
    assert "tampered or the build ran from another checkout" in err and "nothing recorded as CURRENT" in err and "NOT re-pointed" in err, err
    assert not cpr.pointer_path(root).exists()
    # provenance-only differences do not refuse: the pins FILE's sha and the staged rig's own manifest ride along as recorded
    rec = altered(CURRENT_RECIPE, "tokenizers.pins_sha256", "0" * 64)
    rec["tokenizers"]["rig"] = {"tokenizers": {"llama3": {"snapshot": "x", "files": {"tokenizer.json": "1" * 64}}}}
    built2 = write_build(tmp_path / "pools_ctl2.build-job10", recipe=rec)
    assert one_line(commit(root, built2)) == f"CURRENT {built2}"
    assert json.loads(cpr.pointer_path(root).read_text())["recipe"] == rec   # the pointer carries the DIR's recorded recipe, provenance included
    assert one_line(decide(root)) == f"REUSE {built2}"


def test_the_pointer_carries_the_recipe_and_a_tampered_pointer_recipe_refuses_as_a_belt(tmp_path):
    root = write_build(tmp_path / "pools_ctl2")
    one_line(commit(root, root))
    ptr = cpr.pointer_path(root)
    rec = json.loads(ptr.read_text())
    assert rec["recipe"] == CURRENT_RECIPE
    rec["recipe"]["salts"]["heldout"] = "sdfbase-heldout-v7"              # the dir's manifests still say today's; the pointer does not
    ptr.write_text(json.dumps(rec) + "\n")
    err = refused(decide(root))
    assert str(ptr) in err and "recorded a recipe that differs from the one its dir" in err and str(root) in err, err
    assert f"salts.heldout: sdfbase-heldout-v7 (recorded) != {CURRENT_RECIPE['salts']['heldout']} (current)" in err, err
    assert "inspect both by hand" in err and not list(tmp_path.glob("pools_ctl2*.expect*")), err
    # a pointer from before recipes were recorded carries none and is judged by the dir alone
    del rec["recipe"]
    ptr.write_text(json.dumps(rec) + "\n")
    assert one_line(decide(root)) == f"REUSE {root}"


def test_a_byte_incoherent_complete_build_refuses_even_when_its_recipe_differs(tmp_path):
    """The recipe record lives in the manifests; a manifest whose pool bytes disagree with it is not trusted to say what
    recipe cut them — bytes first, so this is a refusal, not a step-around."""
    root = write_build(tmp_path / "pools_ctl2", recipe=altered(CURRENT_RECIPE, "salts.heldout", "sdfbase-heldout-v0"))
    with open(root / f"pairs_{cpr.FULL_CELL}.jsonl", "a") as fh:
        fh.write(json.dumps({"doc_id": "ctl2-doc-9999", "instruction_id": cpr.FULL_CELL, "prompt": "p", "pos": "extra"}) + "\n")
    before = snapshot(root)
    err = refused(decide(root, env_extra={"SLURM_JOB_ID": "14"}))
    assert "sha256" in err and cpr.FULL_CELL in err and "neither rebuilt over nor deleted" in err, err
    assert snapshot(root) == before and not cpr.expect_path(Path(f"{root}.build-job14")).exists()


def test_a_recipe_change_with_an_existing_target_or_sidecar_refuses_like_the_partial_path(tmp_path):
    root = write_build(tmp_path / "pools_ctl2", recipe=altered(CURRENT_RECIPE, "version", "ctl2-recipe-v0"))
    target = tmp_path / "pools_ctl2.build-job15"
    target.mkdir()
    before = snapshot(root)
    err = refused(decide(root, env_extra={"SLURM_JOB_ID": "15"}))
    assert str(target) in err and "already exists" in err and "different recipe (version)" in err and str(root) in err, err
    target.rmdir()
    cpr.expect_path(target).write_text("{}\n")
    err = refused(decide(root, env_extra={"SLURM_JOB_ID": "15"}))
    assert str(cpr.expect_path(target)) in err and "already exists" in err, err
    assert snapshot(root) == before and cpr.expect_path(target).read_text() == "{}\n"


# --------------------------------------------------------------------------------------------------------------
# identity (the DONE marker's certified identities) + recipe_digest
# --------------------------------------------------------------------------------------------------------------
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def identity(root: Path, budget: Path | None = None, volume: Path | None = None, **kw) -> subprocess.CompletedProcess:
    budget = budget or budget_manifest_for(root)
    vol = ["--volume", str(volume)] if volume is not None else []
    return cli("identity", "--root", str(root), "--budget-manifest", str(budget), *vol, **kw)


def parse_identity(line: str) -> dict:
    """The one line as the stage script reads it: space-separated `key=value` tokens, in IDENTITY_KEYS order."""
    toks = line.split(" ")
    assert [t.split("=", 1)[0] for t in toks] == list(cpr.IDENTITY_KEYS), line
    return dict(t.split("=", 1) for t in toks)


def link_volume(vol: Path, d: Path) -> Path:
    """The chain's link loop for the ctl2 files: `ln -f <d>/pairs_* <vol>/` — hard links, replacing whatever is there."""
    vol.mkdir(parents=True, exist_ok=True)
    for name in cpr.CELL_FILES:
        (vol / name).unlink(missing_ok=True)
        os.link(d / name, vol / name)
    return vol


def copy_in_place(p: Path) -> None:
    """Replace the file at ``p`` with a COPY of itself: identical bytes, its own inode."""
    data = p.read_bytes()
    p.unlink()
    p.write_bytes(data)


def copy_volume(vol: Path, d: Path) -> Path:
    """The loop's cross-device fallback for the ctl2 files: `cp -f <d>/pairs_* <vol>/` — byte-equal copies, own inodes."""
    vol.mkdir(parents=True, exist_ok=True)
    for name in cpr.CELL_FILES:
        (vol / name).unlink(missing_ok=True)
        (vol / name).write_bytes((d / name).read_bytes())
    return vol


#: the token's grammar as the tool's docstring spells it: one word, `:` joining the two-part forms
VOLUME_LINKS_RE = re.compile(
    r"^(ok|ok:copy|unchecked|no_current_dir|"
    r"pairs_(constv2ctl2-heldout|constv2ctl2-full)\.(jsonl|manifest\.json):(missing_in_volume|missing_in_current_dir|not_the_current_builds_file))$")


def test_recipe_digest_ignores_provenance_only_leaves_and_changes_with_any_compared_leaf():
    cur = CURRENT_RECIPE
    d = cpr.recipe_digest(cur)
    assert HEX64.match(d) and cpr.recipe_digest(copy.deepcopy(cur)) == d
    # the canonical form: sorted keys, compact separators, the two provenance-only leaves removed
    stripped = copy.deepcopy(cur)
    del stripped["tokenizers"]["pins_sha256"]
    stripped["tokenizers"].pop("rig", None)
    assert d == hashlib.sha256(json.dumps(stripped, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    # provenance only: an edit to the pins FILE's sha (its `_doc`) and the staged rig's own manifest never change it …
    rec = altered(cur, "tokenizers.pins_sha256", "0" * 64)
    rec["tokenizers"]["rig"] = {"tokenizers": {"llama3": {"snapshot": "x"}}}
    assert cpr.recipe_digest(rec) == d and cpr.compare_recipe(rec, cur) == []
    # … while every compared leaf (a salt, a corpus sha, a pin's revision, the version) does
    for name, (path, value) in sorted(RECIPE_CHANGES.items()):
        assert cpr.recipe_digest(altered(cur, path, value)) != d, name
    assert cpr.recipe_digest(altered(cur, "split.frac", 0.5)) != d
    # a JSON round trip (the manifests carry JSON) and the key order are not part of the identity
    assert cpr.recipe_digest(json.loads(json.dumps(cur))) == d
    assert cpr.recipe_digest({k: cur[k] for k in reversed(list(cur))}) == d
    with pytest.raises(cpr.Refusal):
        cpr.recipe_digest("nope")


def test_identity_on_a_fresh_commit_reports_the_declared_identities_equal_to_the_recorded_ones(tmp_path):
    root = write_build(tmp_path / "pools_ctl2")
    one_line(commit(root, root))
    before = snapshot(root)
    r = identity(root)
    ident = parse_identity(one_line(r))
    assert r.stderr == ""                                                # reads only; the one line is the whole answer
    assert ident["declared_recipe"] == cpr.recipe_digest(CURRENT_RECIPE) and HEX64.match(ident["declared_recipe"])
    assert ident["budget"] == f"{BUDGET_SHA}:{BUDGET_TOKENS}"
    assert ident["current_dir"] == str(root)
    assert ident["recorded_recipe"] == ident["declared_recipe"]         # what the marker will certify == today's
    assert ident["recorded_budget"] == ident["budget"]
    assert ident["volume_links"] == "unchecked"                          # no --volume: the token is still there, and says so
    assert snapshot(root) == before and json.loads(cpr.pointer_path(root).read_text())["dir"] == str(root)
    assert cpr.identity_line(ident) == r.stdout.strip()
    # the CURRENT dir follows the pointer: a committed stamped sibling is what identity reads next
    target = Path(f"{root}.build-job31")
    write_build(target)
    assert one_line(commit(root, target, env_extra={"SLURM_JOB_ID": "31"})) == f"CURRENT {target}"
    ident2 = parse_identity(one_line(identity(root)))
    assert ident2["current_dir"] == str(target) and ident2["recorded_recipe"] == ident["declared_recipe"]
    assert ident2["recorded_budget"] == ident["budget"]
    # without a pointer the root itself is the CURRENT dir
    cpr.pointer_path(root).unlink()
    assert parse_identity(one_line(identity(root)))["current_dir"] == str(root)


def test_identity_on_a_root_with_no_build_reports_none_for_the_recorded_side(tmp_path):
    root = tmp_path / "pools_ctl2"
    ident = parse_identity(one_line(identity(root)))
    assert ident["declared_recipe"] == cpr.recipe_digest(CURRENT_RECIPE) and ident["budget"] == f"{BUDGET_SHA}:{BUDGET_TOKENS}"
    assert (ident["current_dir"], ident["recorded_recipe"], ident["recorded_budget"]) == ("none", "none", "none")
    assert not root.exists()                                             # nothing created
    root.mkdir()                                                         # an EMPTY root is no build either
    assert parse_identity(one_line(identity(root)))["current_dir"] == "none"
    assert one_line(decide(root)) == f"BUILD {root}"                     # the same state decide calls a first run


def test_identity_refuses_a_missing_budget_manifest_and_reads_everything_else_without_judging(tmp_path):
    root = write_build(tmp_path / "pools_ctl2")
    err = refused(identity(root, tmp_path / "pools" / "nowhere.manifest.json"))
    assert "budget manifest" in err and "missing" in err and BUDGET_CELL in err, err
    bad = tmp_path / "pools" / "bad.manifest.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{not json")
    assert "unreadable" in refused(identity(root, bad))
    # a complete build with NO recipe record and no budget_source: `none`, not a refusal (decide is the judge)
    write_build(root, recipe=None, budget=None)
    ident = parse_identity(one_line(identity(root)))
    assert ident["current_dir"] == str(root) and ident["recorded_recipe"] == "none" and ident["recorded_budget"] == "none"
    assert "no budget_source" in refused(decide(root))
    # a recorded recipe that DIFFERS in a salt digests differently from the declared one — the stage's marker check keys on this
    other = altered(CURRENT_RECIPE, "salts.heldout", "sdfbase-heldout-v0")
    write_build(root, recipe=other)
    ident = parse_identity(one_line(identity(root)))
    assert ident["recorded_recipe"] == cpr.recipe_digest(other) != ident["declared_recipe"]
    assert ident["recorded_budget"] == ident["budget"]
    # a budget_source that differs is reported as recorded, not judged
    write_build(root, budget=budget_source(official_tokens=99))
    assert parse_identity(one_line(identity(root)))["recorded_budget"] == f"{BUDGET_SHA}:99"
    # an unreadable pointer refuses (as decide does); whitespace in a path refuses (the line is read as tokens)
    cpr.pointer_path(root).write_text("{nope")
    assert "unreadable" in refused(identity(root))
    cpr.pointer_path(root).unlink()
    assert "whitespace" in refused(identity(tmp_path / "pools ctl2", budget_manifest_for(root)))
    assert "identity" in cli("--help").stdout


# --------------------------------------------------------------------------------------------------------------------
# identity --volume: does the volume the fits read HOLD the CURRENT build?
# --------------------------------------------------------------------------------------------------------------------
def test_the_volume_links_token_is_the_last_token_and_its_grammar_is_pinned():
    """The stage reads the line as space-separated tokens and `volume_links` by name: it is always present, always LAST
    (so every earlier token keeps its position), and every value it can take is one word of the grammar
    `ok | ok:copy | unchecked | no_current_dir | pairs_<cell>.<jsonl|manifest.json>:<why>` (the tool's docstring)."""
    assert cpr.IDENTITY_KEYS == ("declared_recipe", "budget", "current_dir", "recorded_recipe", "recorded_budget", "volume_links")
    assert cpr.IDENTITY_KEYS[-1] == "volume_links"
    assert cpr.CELL_FILES == ("pairs_constv2ctl2-heldout.jsonl", "pairs_constv2ctl2-heldout.manifest.json",
                              "pairs_constv2ctl2-full.jsonl", "pairs_constv2ctl2-full.manifest.json")
    for v in (cpr.LINKS_OK, cpr.LINKS_OK_COPY, cpr.LINKS_UNCHECKED, cpr.LINKS_NO_CURRENT_DIR, cpr.LINKS_MISSING_IN_VOLUME,
              cpr.LINKS_MISSING_IN_CURRENT, cpr.LINKS_NOT_CURRENT, *cpr.CELL_FILES):
        assert v and not any(ch.isspace() for ch in v) and "=" not in v, v
    assert (cpr.LINKS_OK, cpr.LINKS_OK_COPY, cpr.LINKS_UNCHECKED, cpr.LINKS_NO_CURRENT_DIR) == ("ok", "ok:copy", "unchecked", "no_current_dir")
    assert (cpr.LINKS_MISSING_IN_VOLUME, cpr.LINKS_MISSING_IN_CURRENT, cpr.LINKS_NOT_CURRENT) == (
        "missing_in_volume", "missing_in_current_dir", "not_the_current_builds_file")
    # every value the tool can produce is a word of the grammar; nothing else is (ok:copy is the ONLY `ok:` form — the `:`
    # after a `pairs_` file introduces a reason, never a qualifier)
    for v in (cpr.LINKS_OK, cpr.LINKS_OK_COPY, cpr.LINKS_UNCHECKED, cpr.LINKS_NO_CURRENT_DIR):
        assert VOLUME_LINKS_RE.match(v), v
    for f in cpr.CELL_FILES:
        for why in (cpr.LINKS_MISSING_IN_VOLUME, cpr.LINKS_MISSING_IN_CURRENT, cpr.LINKS_NOT_CURRENT):
            assert VOLUME_LINKS_RE.match(f"{f}:{why}"), (f, why)
    for bad in ("", "ok:hard", "ok:", "copy", "ok copy", "ok:copy:copy", "pairs_constv2ctl2-full.jsonl:copy",
                "pairs_constv2ctl2-full.jsonl", "pairs_other.jsonl:missing_in_volume", "not_the_current_builds_file"):
        assert not VOLUME_LINKS_RE.match(bad), bad
    # the docstring spells the same grammar
    doc = cpr.__doc__
    assert "``volume_links=<ok | ok:copy | unchecked | no_current_dir | pairs_<cell>.<jsonl|manifest.json>:<why>>``" in doc
    assert 'volume_links = "ok" | "ok:copy" | "unchecked" | "no_current_dir" | <file> ":" <why>' in doc
    assert '<why>        = "missing_in_volume" | "missing_in_current_dir" | "not_the_current_builds_file"' in doc
    ident = {k: "x" for k in cpr.IDENTITY_KEYS}
    ident["volume_links"] = "ok"
    assert cpr.identity_line(ident).split(" ")[-1] == "volume_links=ok"
    assert cpr.identity_line(ident).split(" ")[:5] == ["declared_recipe=x", "budget=x", "current_dir=x", "recorded_recipe=x", "recorded_budget=x"]
    ident["volume_links"] = "ok:copy"
    assert cpr.identity_line(ident).split(" ")[-1] == "volume_links=ok:copy" and len(cpr.identity_line(ident).split(" ")) == 6


def test_identity_volume_ok_on_a_fresh_commit_whose_volume_is_hard_linked_to_it(tmp_path):
    root = write_build(tmp_path / "pools_ctl2")
    one_line(commit(root, root))
    vol = link_volume(tmp_path / "vol" / "pool" / "fitkey", root)
    r = identity(root, volume=vol)
    line = one_line(r)
    ident = parse_identity(line)
    assert r.stderr == "" and ident["volume_links"] == "ok", line
    assert line.split(" ")[-1] == "volume_links=ok" and len(line.split(" ")) == 6      # last token; the first five unchanged
    assert line.split(" ")[:5] == one_line(identity(root)).split(" ")[:5]
    assert ident["current_dir"] == str(root) and ident["recorded_recipe"] == ident["declared_recipe"]
    # the volume's extra files (the constitutional pools ride the same loop) do not matter: only the four ctl2 files are checked
    (vol / "pairs_constv2-mix-full.jsonl").write_bytes(b"whatever\n")
    assert parse_identity(one_line(identity(root, volume=vol)))["volume_links"] == "ok"
    # the pure piece agrees
    assert cpr.volume_links(vol, root) == "ok"


def test_identity_volume_says_ok_copy_for_byte_equal_copies_and_names_a_stale_copy_first_file_first(tmp_path):
    """Same bytes, different inode — `cp -f` where `ln -f` hit EXDEV (the loop's cross-device fallback), or a copy someone
    made: the volume holds the current build's BYTES and the fits read the same -> `ok:copy` (under the
    `ok`-only rule a site whose volume is on another filesystem could never mark). Different inode AND different bytes — a
    STALE copy — is a reason, the first such file in cell order, pool before manifest."""
    root = write_build(tmp_path / "pools_ctl2")
    one_line(commit(root, root))
    vol = link_volume(tmp_path / "vol", root)
    full_pool = "pairs_constv2ctl2-full.jsonl"
    copy_in_place(vol / full_pool)                                       # identical bytes, its own inode
    assert (vol / full_pool).read_bytes() == (root / full_pool).read_bytes()
    assert os.stat(vol / full_pool).st_ino != os.stat(root / full_pool).st_ino
    r = identity(root, volume=vol)
    line = one_line(r)
    ident = parse_identity(line)
    assert r.stderr == "" and ident["volume_links"] == "ok:copy", r.stdout                 # a MIXED set: three links + one equal copy
    assert line.split(" ")[-1] == "volume_links=ok:copy" and len(line.split(" ")) == 6
    assert line.split(" ")[:5] == one_line(identity(root)).split(" ")[:5]
    # EVERY file a byte-equal copy (the whole loop fell back to cp -f): still ok:copy, from the CLI and the pure piece
    copy_volume(vol, root)
    assert all(os.stat(vol / f).st_ino != os.stat(root / f).st_ino for f in cpr.CELL_FILES)
    assert all((vol / f).read_bytes() == (root / f).read_bytes() for f in cpr.CELL_FILES)
    assert parse_identity(one_line(identity(root, volume=vol)))["volume_links"] == "ok:copy"
    assert cpr.volume_links(vol, root) == "ok:copy"
    # one copy whose bytes DIFFER (stale: the build was rebuilt after the copy was taken) is named, by file
    (vol / full_pool).write_bytes((root / full_pool).read_bytes() + b'{"doc_id": "ctl2-doc-9999", "instruction_id": "constv2ctl2-full", "prompt": "p", "pos": "stale"}\n')
    assert parse_identity(one_line(identity(root, volume=vol)))["volume_links"] == f"{full_pool}:not_the_current_builds_file"
    # the FIRST stale file in cell order (heldout before full), pool before manifest: a stale heldout manifest is named over the full pool
    held_m = "pairs_constv2ctl2-heldout.manifest.json"
    (vol / held_m).write_bytes(b"{}\n")
    assert parse_identity(one_line(identity(root, volume=vol)))["volume_links"] == f"{held_m}:not_the_current_builds_file"
    assert cpr.volume_links(vol, root) == f"{held_m}:not_the_current_builds_file"
    # re-copying (the loop's `cp -f`, which runs on REUSE too) makes it ok:copy again; re-linking (`ln -f`) makes it ok
    copy_volume(vol, root)
    assert parse_identity(one_line(identity(root, volume=vol)))["volume_links"] == "ok:copy"
    link_volume(vol, root)
    assert parse_identity(one_line(identity(root, volume=vol)))["volume_links"] == "ok"
    # every value seen here is a word of the grammar
    for v in ("ok", "ok:copy", f"{full_pool}:not_the_current_builds_file", f"{held_m}:not_the_current_builds_file"):
        assert VOLUME_LINKS_RE.match(v), v


def test_volume_links_hashes_only_the_files_that_are_not_the_same_inode(tmp_path, monkeypatch):
    """The sha256 is the price of the copy case only: a hard-linked volume (the common same-device site) is four stats and
    no bytes read; a copy costs the copy and its original; a stale copy stops the walk at the first reason."""
    root = write_build(tmp_path / "pools_ctl2")
    one_line(commit(root, root))
    vol = link_volume(tmp_path / "vol", root)
    hashed: list[str] = []
    real = cpr._sha256_file
    monkeypatch.setattr(cpr, "_sha256_file", lambda q: (hashed.append(Path(q).name), real(q))[1])
    assert cpr.volume_links(vol, root) == "ok" and hashed == []
    copy_in_place(vol / "pairs_constv2ctl2-full.manifest.json")
    assert cpr.volume_links(vol, root) == "ok:copy"
    assert hashed == ["pairs_constv2ctl2-full.manifest.json"] * 2                         # the copy and its original only
    hashed.clear()
    (vol / "pairs_constv2ctl2-heldout.jsonl").unlink()
    (vol / "pairs_constv2ctl2-heldout.jsonl").write_bytes(b"stale\n")
    assert cpr.volume_links(vol, root) == "pairs_constv2ctl2-heldout.jsonl:not_the_current_builds_file"
    assert hashed == ["pairs_constv2ctl2-heldout.jsonl"] * 2                              # stopped at the first reason
    assert real(root / "pairs_constv2ctl2-full.jsonl") == hashlib.sha256((root / "pairs_constv2ctl2-full.jsonl").read_bytes()).hexdigest()


def test_identity_refuses_by_name_when_a_volume_copy_cannot_be_read_to_compare(tmp_path):
    """A different inode whose bytes cannot be read is neither `ok:copy` nor a stale-copy verdict: the tool refuses by name
    (exit 3, stdout empty) the way it does for an unreadable pointer, and the stage says the identity read failed."""
    if os.geteuid() == 0:
        pytest.skip("root reads a mode-000 file")
    root = write_build(tmp_path / "pools_ctl2")
    one_line(commit(root, root))
    vol = link_volume(tmp_path / "vol", root)
    name = "pairs_constv2ctl2-full.jsonl"
    copy_in_place(vol / name)
    (vol / name).chmod(0)
    try:
        err = refused(identity(root, volume=vol))
        assert "volume_links: cannot read" in err and name in err, err
    finally:
        (vol / name).chmod(0o644)
    assert parse_identity(one_line(identity(root, volume=vol)))["volume_links"] == "ok:copy"


def test_identity_volume_follows_the_pointer_so_a_link_left_on_a_superseded_build_is_not_the_current_builds_file(tmp_path):
    """The finding's shape at the tool level: the volume was linked to the root's build; a stamped sibling was committed as
    CURRENT (another checkout's rebuild) and the links were NOT re-pointed — the volume holds the superseded selection."""
    root = write_build(tmp_path / "pools_ctl2")
    one_line(commit(root, root))
    vol = link_volume(tmp_path / "vol", root)
    assert parse_identity(one_line(identity(root, volume=vol)))["volume_links"] == "ok"
    target = Path(f"{root}.build-job77")
    write_build(target, salt="other")                                    # other bytes, today's recipe: a valid sibling
    assert one_line(commit(root, target, env_extra={"SLURM_JOB_ID": "77"})) == f"CURRENT {target}"
    ident = parse_identity(one_line(identity(root, volume=vol)))
    assert ident["current_dir"] == str(target)
    assert (vol / "pairs_constv2ctl2-heldout.jsonl").read_bytes() != (target / "pairs_constv2ctl2-heldout.jsonl").read_bytes()
    assert ident["volume_links"] == "pairs_constv2ctl2-heldout.jsonl:not_the_current_builds_file", ident   # another selection: other bytes
    link_volume(vol, target)                                             # the chain's loop re-points the volume at $CTL2_DIR
    assert parse_identity(one_line(identity(root, volume=vol)))["volume_links"] == "ok"


def test_identity_volume_names_a_missing_file_missing_in_volume_and_a_missing_volume_dir_likewise(tmp_path):
    root = write_build(tmp_path / "pools_ctl2")
    one_line(commit(root, root))
    vol = link_volume(tmp_path / "vol", root)
    (vol / "pairs_constv2ctl2-full.manifest.json").unlink()
    ident = parse_identity(one_line(identity(root, volume=vol)))
    assert ident["volume_links"] == "pairs_constv2ctl2-full.manifest.json:missing_in_volume", ident
    # a byte-equal copy is NOT a reason: the heldout pool a copy and the full manifest missing names the missing file
    held_pool = "pairs_constv2ctl2-heldout.jsonl"
    copy_in_place(vol / held_pool)
    assert parse_identity(one_line(identity(root, volume=vol)))["volume_links"] == "pairs_constv2ctl2-full.manifest.json:missing_in_volume"
    # both a STALE copy and a missing file: the earlier file in CELL_FILES order wins
    (vol / held_pool).write_bytes(b"stale\n")
    assert parse_identity(one_line(identity(root, volume=vol)))["volume_links"] == f"{held_pool}:not_the_current_builds_file"
    # a volume dir that does not exist at all: its first file is missing
    ident = parse_identity(one_line(identity(root, volume=tmp_path / "nowhere")))
    assert ident["volume_links"] == "pairs_constv2ctl2-heldout.jsonl:missing_in_volume", ident
    assert not (tmp_path / "nowhere").exists()                          # reads only
    # a file the CURRENT dir lacks (an incomplete build the pointer still names) is said as the current dir's, not the volume's
    link_volume(vol, root)
    (root / "pairs_constv2ctl2-full.jsonl").unlink()
    ident = parse_identity(one_line(identity(root, volume=vol)))
    assert ident["volume_links"] == "pairs_constv2ctl2-full.jsonl:missing_in_current_dir", ident


def test_identity_volume_says_no_current_dir_when_nothing_is_built(tmp_path):
    root = tmp_path / "pools_ctl2"
    vol = tmp_path / "vol"
    ident = parse_identity(one_line(identity(root, volume=vol)))
    assert (ident["current_dir"], ident["recorded_recipe"], ident["recorded_budget"], ident["volume_links"]) == (
        "none", "none", "none", "no_current_dir")
    root.mkdir()                                                         # an EMPTY root is no build either
    vol.mkdir()
    assert parse_identity(one_line(identity(root, volume=vol)))["volume_links"] == "no_current_dir"
    # … and a volume that happens to hold some files still has no current build to be linked to
    (vol / "pairs_constv2ctl2-heldout.jsonl").write_bytes(b"x\n")
    assert parse_identity(one_line(identity(root, volume=vol)))["volume_links"] == "no_current_dir"
    assert one_line(decide(root)) == f"BUILD {root}"
    # without --volume the same states say unchecked
    assert parse_identity(one_line(identity(root)))["volume_links"] == "unchecked"
    assert "--volume" in cli("identity", "--help").stdout


# --------------------------------------------------------------------------------------------------------------
# a PARTIAL build's recipe gates its byte expectations
# --------------------------------------------------------------------------------------------------------------
def test_a_partial_under_the_current_recipe_records_both_cells_expectations_as_before(tmp_path):
    root = write_build(tmp_path / "pools_ctl2", report=False)           # today's recipe on both manifests
    r = decide(root, env_extra={"SLURM_JOB_ID": "41"})
    target = Path(f"{root}.build-job41")
    assert one_line(r) == f"BUILD {target}"
    exp = json.loads(cpr.expect_path(target).read_text())
    assert exp["cells"] == {c: manifest_sha(root, c) for c in CELLS} and exp["skipped"] == {} and "recipe_changed" not in exp
    assert "must reproduce" in r.stderr and "no recipe recorded" not in r.stderr and "recipe changed" not in r.stderr, r.stderr
    found = cpr.recorded_cell_shas(root, cpr.read_budget_manifest(budget_manifest_for(root)), lambda s: None,
                                   declared=CURRENT_RECIPE)
    assert found.cells == exp["cells"] and found.recipe_why is None and found.recipe_changed is None


@pytest.mark.parametrize("change", sorted(RECIPE_CHANGES))
def test_a_partial_under_a_different_recipe_records_no_expectations_and_the_fresh_build_commits(tmp_path, change):
    """THE case that used to fail: an interrupted build, then a recipe change (a salt, a corpus sha, a pin, the version).
    The partial's recorded shas are the OLD recipe's bytes; decide recorded them as expectations anyway, the new, valid
    build failed check_expectation at commit and `.CURRENT` never moved. Now the expectations are skipped BY NAME (stderr
    and the sidecar's `skipped`), and commit of the fresh build succeeds — the pointer moves."""
    path, value = RECIPE_CHANGES[change]
    root = write_build(tmp_path / "pools_ctl2", report=False, recipe=altered(CURRENT_RECIPE, path, value))
    before = snapshot(root)
    r = decide(root, env_extra={"SLURM_JOB_ID": "42"})
    target = Path(f"{root}.build-job42")
    assert one_line(r) == f"BUILD {target}" and "REFUSED" not in r.stderr
    diff = f"{path}: {value} (recorded) != {leaf(CURRENT_RECIPE, path)} (current)"
    why = f"recipe changed ({diff}): the rebuilt bytes are expected to differ"
    assert "incomplete ctl2 build" in r.stderr and why in r.stderr, r.stderr
    for c in CELLS:
        assert f"no reproduction expectation for {c} — {why}" in r.stderr, r.stderr
    assert str(cpr.expect_path(target)) in r.stderr and "must reproduce" not in r.stderr
    exp = json.loads(cpr.expect_path(target).read_text())
    assert exp["cells"] == {} and exp["previous_dir"] == str(root) and exp["target"] == str(target)
    assert exp["recipe_changed"] == [diff] and set(exp["skipped"]) == set(CELLS)
    assert all(v == why for v in exp["skipped"].values())
    assert exp["budget"] == {"pool_sha256": BUDGET_SHA, "official_tokens": BUDGET_TOKENS}
    assert snapshot(root) == before and not target.exists()
    # the fresh build: OTHER bytes (expected), today's recipe -> commit SUCCEEDS and the pointer moves
    write_build(target, salt="today")
    for c in CELLS:
        assert manifest_sha(target, c) != manifest_sha(root, c)
    assert one_line(commit(root, target, env_extra={"SLURM_JOB_ID": "42"})) == f"CURRENT {target}"
    rec = json.loads(cpr.pointer_path(root).read_text())
    assert rec["dir"] == str(target) and rec["previous_dir"] == str(root) and rec["reproduced_from"] is None
    assert rec["recipe_changed"] == [diff] and rec["recipe"] == CURRENT_RECIPE
    assert not cpr.expect_path(target).exists() and json.loads(cpr.expect_checked_path(target).read_text()) == exp
    assert snapshot(root) == before                                      # the partial: untouched, never deleted
    assert one_line(decide(root, env_extra={"SLURM_JOB_ID": "43"})) == f"REUSE {target}"


def test_control_the_expectations_the_recipe_check_skips_are_exactly_what_made_commit_refuse(tmp_path):
    """The same differing-recipe partial with its shas recorded anyway (a sidecar written the way decide used to write
    it) makes commit refuse the valid fresh build by name — the failure finding (2) removes."""
    root = write_build(tmp_path / "pools_ctl2", report=False, recipe=altered(CURRENT_RECIPE, "salts.heldout", "sdfbase-heldout-v0"))
    target = Path(f"{root}.build-job46")
    current = cpr.read_budget_manifest(budget_manifest_for(root))
    cpr.write_expectation(target, previous_dir=root, cells={c: manifest_sha(root, c) for c in CELLS}, skipped={}, current=current)
    write_build(target, salt="today")
    err = refused(commit(root, target, env_extra={"SLURM_JOB_ID": "46"}))
    assert "seeded rebuild of" in err and "not reproducing the recorded bytes" in err and "NOT re-pointed" in err, err
    assert not cpr.pointer_path(root).exists()


def test_a_partial_with_no_recipe_record_records_no_expectations_and_says_so(tmp_path):
    root = write_build(tmp_path / "pools_ctl2", report=False, recipe=None)
    before = snapshot(root)
    r = decide(root, env_extra={"SLURM_JOB_ID": "44"})
    target = Path(f"{root}.build-job44")
    assert one_line(r) == f"BUILD {target}" and "REFUSED" not in r.stderr
    assert "no recipe recorded on the partial build" in r.stderr and "carry no recipe block" in r.stderr, r.stderr
    for c in CELLS:
        assert f"no reproduction expectation for {c} — no recipe recorded on the partial build" in r.stderr, r.stderr
    exp = json.loads(cpr.expect_path(target).read_text())
    assert exp["cells"] == {} and exp["previous_dir"] == str(root) and "recipe_changed" not in exp
    assert set(exp["skipped"]) == set(CELLS)
    assert all(v.startswith("no recipe recorded on the partial build (") for v in exp["skipped"].values())
    assert all(f"pairs_{c}.manifest.json" in v for c in CELLS for v in exp["skipped"].values())
    # the fresh build commits (even with the SAME bytes: nothing was expected), the pointer records the superseded partial
    write_build(target)
    assert one_line(commit(root, target, env_extra={"SLURM_JOB_ID": "44"})) == f"CURRENT {target}"
    rec = json.loads(cpr.pointer_path(root).read_text())
    assert rec["dir"] == str(target) and rec["previous_dir"] == str(root) and rec["reproduced_from"] is None
    assert rec["recipe_changed"] is None and snapshot(root) == before
    # only ONE manifest present (the heldout's, unrecorded): still named; the missing full manifest keeps its own reason
    root2 = write_build(tmp_path / "other_ctl2", report=False, recipe=None)
    (root2 / f"pairs_{cpr.FULL_CELL}.manifest.json").unlink()
    r = decide(root2, env_extra={"SLURM_JOB_ID": "45"})
    assert one_line(r) == f"BUILD {root2}.build-job45"
    exp = json.loads(cpr.expect_path(Path(f"{root2}.build-job45")).read_text())
    assert exp["cells"] == {} and exp["skipped"][cpr.HELDOUT_CELL].startswith("no recipe recorded on the partial build (pairs_constv2ctl2-heldout.manifest.json carry")
    assert exp["skipped"][cpr.FULL_CELL] == f"pairs_{cpr.FULL_CELL}.manifest.json missing"
    # a stray dir with no manifest at all: nothing to expect, no recipe to compare, no sidecar (unchanged)
    stray = tmp_path / "stray_ctl2"
    stray.mkdir()
    (stray / f"pairs_{cpr.HELDOUT_CELL}.jsonl").write_text("{}\n")
    r = decide(stray, env_extra={"SLURM_JOB_ID": "47"})
    assert one_line(r) == f"BUILD {stray}.build-job47" and not cpr.expect_path(Path(f"{stray}.build-job47")).exists()
    assert "no recipe recorded" not in r.stderr and "recipe changed" not in r.stderr, r.stderr


def test_a_partial_whose_two_manifests_disagree_on_the_recipe_skips_by_the_union_of_differing_fields(tmp_path):
    """One cell's manifest records today's recipe, the other's an older salt: the partial is not one recipe's, so every
    cell is skipped and the differing field is named (the union over both manifests' comparisons)."""
    root = write_build(tmp_path / "pools_ctl2", report=False)
    write_cell(root, cpr.FULL_CELL, preflight=True, recipe=altered(CURRENT_RECIPE, "salts.eval", "sdfbase-heldout-eval-v0"))
    r = decide(root, env_extra={"SLURM_JOB_ID": "48"})
    target = Path(f"{root}.build-job48")
    assert one_line(r) == f"BUILD {target}"
    diff = f"salts.eval: sdfbase-heldout-eval-v0 (recorded) != {CURRENT_RECIPE['salts']['eval']} (current)"
    exp = json.loads(cpr.expect_path(target).read_text())
    assert exp["cells"] == {} and exp["recipe_changed"] == [diff] and set(exp["skipped"]) == set(CELLS)
    assert all(v == f"recipe changed ({diff}): the rebuilt bytes are expected to differ" for v in exp["skipped"].values())
    # the pure verdict, directly: equal -> (None, None); several differing manifests -> the sorted union
    m_ok = {"recipe": copy.deepcopy(CURRENT_RECIPE)}
    m_a = {"recipe": altered(CURRENT_RECIPE, "version", "ctl2-recipe-v0")}
    m_b = {"recipe": altered(CURRENT_RECIPE, "salts.eval", "sdfbase-heldout-eval-v0")}
    assert cpr.partial_recipe_verdict([("h", m_ok), ("f", m_ok)], CURRENT_RECIPE) == (None, None)
    why, diffs = cpr.partial_recipe_verdict([("h", m_a), ("f", m_b)], CURRENT_RECIPE)
    assert diffs == [diff, "version: ctl2-recipe-v0 (recorded) != ctl2-recipe-v1 (current)"] and why.startswith("recipe changed (")
    why, diffs = cpr.partial_recipe_verdict([("h", {"cell": "x"}), ("f", {"recipe": "nope"})], CURRENT_RECIPE)
    assert diffs is None and why.startswith("no recipe recorded on the partial build (h, f carry no recipe block)")
    # a manifest recording today's recipe beside one recording NONE: the partial is not one recipe's either — but the
    # verdict is about the recorded blocks: the present one is equal, so the bytes it describes are expected (the
    # missing block's cell is still skipped for its own reason when its sha is unreadable; here both record shas)
    assert cpr.partial_recipe_verdict([("h", m_ok), ("f", {"cell": "x"})], CURRENT_RECIPE) == (None, None)

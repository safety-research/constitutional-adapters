"""The six inode rules, by construction.

Rule 1: ``stage`` keeps the tree's TARBALL only and every consumer (``resolve_staged_tree``) unpacks it
node-local with the ``TREE_SHA`` verified — nothing is extracted into the volume. Rule 2: a full-shape
per-cell store is exactly 40 inodes on a row engine (today's 36 + ``standup/`` ×3), counted on the REAL
mirrored gemma ``none`` store and on the synthetic store the shared helper writes through the real
writers. Rule 3: the compile cache is module D's (its own suite pins the tarball + node-local env). Rule
4: the wave-close planner refuses without a PASS gate at the sha and a mirrored store / row dir, names
the parked stores relaunches.json names, this wave's ``-negctl`` stores, the preflight artefacts and the
row dirs — never the tree archive (shared at its sha; refused by kind on both halves); the container
half acts only on the planner's approved inventory, prints it and deletes only under confirm, all or nothing.
Rule 5: ``inode_budget_verdict`` refuses at >= 90 % (a synthetic 90.0 % / 89.9 %-plus-forecast / 40.7 %
reading), ``forecast_inodes`` follows the plan's formula, the preflight carries the check and ``main``
prints the number on every preflighted path and refuses before any spawn. Rule 6: the launcher never
sets the registry's run-identity variable.

Loads the launcher the way ``test_tinycap_launcher`` does (a modal stub, the app name in the env)."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
from test_tinycap_launcher import _load_launcher, _stub_modal  # noqa: E402
from tinycap_store_shape import (BCB_FULL, BFCL_FULL, ROW_ENGINE_STORE_INODES, SOLO_FULL_SHAPE_INODES,  # noqa: E402
                                 SOLO_FULL_SHAPE_WITH_POINTER_INODES, STANDUP_DIR, STANDUP_FILES, assert_store_inodes,
                                 count_inodes, full_shape_store, inode_census, normalised_layout)
from _env import host_site_path  # noqa: E402  (tests/ helper: a host-only data path through the site config)

LAUNCHER = REPO / "scripts/modal/tinycap_served.py"
CLOSE_WAVE = REPO / "scripts/modal/tinycap_close_wave.py"
TEST_APP_NAME = "tinycap-inode-rules-under-test"
#: the canonical mirror's gemma `none` full store (read-only; copied into a temp dir before anything touches it)
MIRROR_STORE = host_site_path("$STATE/romp/candidate-si-refit/run-state/full22/bigcodebench_full/native/gemma4e4b-f22s1/"
                              "none/thinking_off/bcbw-gemma4e4b_s1-8ba0b48")
FULL = "f6cf462" + "0" * 33
OTHER = "0" * 40
SHA7 = FULL[:7]
ROW = "gemma4e4b-f22s1"
LORA = "constv2-mix-full__lora_r4_ce"
SUFFIX = f"bcbw-gemma4e4b_s1-{SHA7}-shared"


@pytest.fixture(scope="module")
def L():
    return _load_launcher(_stub_modal(), app_name=TEST_APP_NAME, name="tinycap_served_inode_rules")


def _load_close_wave():
    name = "tinycap_close_wave_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, CLOSE_WAVE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _tarball(repos: Path, name: str, full: str | None, *, roster: bool = True) -> Path:
    """``<repos>/sisteer-<name>.tar.gz`` — a sisteer tree archive (``src/sisteer``, the full22 roster) with the
    ``TREE_SHA`` stamp at its root naming ``full`` (``None`` = no stamp: a tarball from before the discipline)."""
    repos.mkdir(parents=True, exist_ok=True)
    p = repos / f"sisteer-{name}.tar.gz"

    def add(tf, arc, text):
        data = text.encode("utf-8")
        info = tarfile.TarInfo(arc)
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    with tarfile.open(p, "w:gz") as tf:
        add(tf, "src/sisteer/__init__.py", "x = 1\n")
        if roster:
            add(tf, C.FULL22_ROSTER_REL, "targets:\n  qwen35_4b:\n    views:\n      main: {}\n      it: {}\n")
        if full is not None:
            add(tf, C.TREE_SHA_FILENAME, full + "\n")
    return p


# --------------------------------------------------------------------------- #
# rule 1: tarball-only staging, node-local unpack, TREE_SHA verified            #
# --------------------------------------------------------------------------- #


def test_resolve_staged_tree_unpacks_the_tarball_node_local_and_writes_nothing_on_the_volume(L, tmp_path, monkeypatch):
    monkeypatch.delenv(C.ENV_FULL22_ROSTER, raising=False)
    repos, local = tmp_path / "vol" / "repos", tmp_path / "node-local"
    tarball = _tarball(repos, SHA7, FULL)
    tree, sha = L.resolve_staged_tree(SHA7, repos, local_root=local)
    assert (tree, sha) == (local / f"sisteer-{FULL}", FULL)
    assert (tree / "src" / "sisteer" / "__init__.py").is_file() and (tree / C.TREE_SHA_FILENAME).read_text() == FULL + "\n"
    # the volume side is untouched: the tarball is the tree's ONE inode, no extracted dir beside it
    assert sorted(repos.iterdir()) == [tarball] and count_inodes(repos) == 2
    assert os.environ[C.ENV_FULL22_ROSTER] == str(tree / C.FULL22_ROSTER_REL)      # the roster binds off the local tree
    # idempotent: a second resolution (the same container) returns the tree without re-extracting it
    marker = tree / "resolved-once"
    marker.write_text("x")
    assert L.resolve_staged_tree(SHA7, repos, local_root=local) == (tree, FULL) and marker.is_file()
    # the FULL sha finds the abbreviated tarball through its TREE_SHA member (a merge chained with a producer sha)
    assert L.resolve_staged_tree(FULL, repos, local_root=local) == (tree, FULL)
    # and a torn earlier unpack (no stamp) is replaced whole, never trusted
    (tree / C.TREE_SHA_FILENAME).unlink()
    assert L.resolve_staged_tree(SHA7, repos, local_root=local) == (tree, FULL)
    assert (tree / C.TREE_SHA_FILENAME).is_file() and not marker.exists()
    # the env override re-points the local root (the tests' seam); the default is node-local /tmp/tree
    monkeypatch.setenv(L.TREE_LOCAL_ROOT_ENV, str(tmp_path / "env-root"))
    assert L.resolve_staged_tree(SHA7, repos)[0] == tmp_path / "env-root" / f"sisteer-{FULL}"
    assert L.TREE_LOCAL_ROOT == "/tmp/tree" and not str(L.TREE_LOCAL_ROOT).startswith("/vol")


def test_resolve_staged_tree_verifies_the_stamp_and_refuses_a_foreign_missing_or_ambiguous_one(L, tmp_path, monkeypatch):
    monkeypatch.delenv(C.ENV_FULL22_ROSTER, raising=False)
    local = tmp_path / "node-local"
    # a tarball whose stamp names ANOTHER commit than the one it is addressed by
    repos = tmp_path / "r1"
    _tarball(repos, SHA7, OTHER)
    with pytest.raises(L.TreeShaMismatch, match="addressed as"):
        L.resolve_staged_tree(SHA7, repos, local_root=local)
    assert not local.exists() or not any(local.iterdir())                  # nothing unpacked from a refused tarball
    # no stamp at all: a tarball uploaded before the producer discipline
    repos = tmp_path / "r2"
    _tarball(repos, SHA7, None)
    with pytest.raises(L.TreeShaMissing, match=C.TREE_SHA_FILENAME):
        L.resolve_staged_tree(SHA7, repos, local_root=local)
    assert L.tarball_tree_sha(repos / f"sisteer-{SHA7}.tar.gz") is None
    # nothing staged under any spelling
    with pytest.raises(L.NoStagedTree, match="no staged tree"):
        L.resolve_staged_tree("deadbee", repos, local_root=local)
    # two tarballs claiming one sha under two names: ambiguous, refused by name (never a coin flip)
    repos = tmp_path / "r3"
    _tarball(repos, FULL[:8], FULL)
    _tarball(repos, FULL[:9], FULL)
    with pytest.raises(L.TreeShaMismatch, match="several"):
        L.resolve_staged_tree(SHA7, repos, local_root=local)
    # a tarball named for another commit is never even opened for an unrelated sha (the name screen)
    repos = tmp_path / "r4"
    _tarball(repos, "abcdef1", OTHER)
    with pytest.raises(L.NoStagedTree):
        L.resolve_staged_tree(SHA7, repos, local_root=local)
    assert L.tarball_tree_sha(tmp_path / "absent.tar.gz") is None


def test_a_tree_extracted_before_rule_1_still_wins_over_its_tarball(L, tmp_path, monkeypatch):
    monkeypatch.delenv(C.ENV_FULL22_ROSTER, raising=False)
    repos, local = tmp_path / "repos", tmp_path / "node-local"
    _tarball(repos, SHA7, FULL)
    legacy = repos / f"sisteer-{SHA7}"
    (legacy / "src" / "sisteer").mkdir(parents=True)
    (legacy / C.TREE_SHA_FILENAME).write_text(FULL + "\n")
    assert L.resolve_staged_tree(SHA7, repos, local_root=local) == (legacy, FULL)
    assert not local.exists()                                              # nothing unpacked: the legacy tree is used as is
    step = L.stage_tree_step(SHA7, repos_root=repos, local_root=local)
    assert step["on_volume"] == L.TREE_ON_VOLUME_EXTRACTED_LEGACY and step["extracted"] is False and step["tree"] == str(legacy)


def test_stage_tree_step_reports_tarball_only_and_every_consumer_resolves_the_same_way(L, tmp_path, monkeypatch):
    monkeypatch.delenv(C.ENV_FULL22_ROSTER, raising=False)          # a resolution binds the roster env; never leak a tmp path
    repos, local = tmp_path / "repos", tmp_path / "node-local"
    with pytest.raises(FileNotFoundError, match="--upload-tree"):
        L.stage_tree_step(SHA7, repos_root=repos, local_root=local)
    tarball = _tarball(repos, SHA7, FULL)
    step = L.stage_tree_step(SHA7, repos_root=repos, local_root=local)
    assert step == {"tree": str(local / f"sisteer-{FULL}"), "extracted": True, "tree_sha": FULL, "tarball": str(tarball),
                    "on_volume": L.TREE_ON_VOLUME_TARBALL_ONLY, "inodes_on_volume": 1, "unpacked_to": "node-local"}
    assert count_inodes(repos) == 2                                        # the repos dir + the tarball, nothing else
    assert L.staged_tarball_path(SHA7, repos) == tarball
    src = LAUNCHER.read_text()
    stage_body = src[src.index("def stage(sha: str"):src.index("OBJECT_IDENTITY_TIMEOUT_S")]
    assert 'step("tree", lambda: stage_tree_step(sha))' in stage_body and "extractall" not in stage_body
    pf = src[src.index("def preflight(sha: str"):src.index("def close_wave(")]
    assert "_tree_path, _tree_full = resolve_staged_tree(sha)" in pf and 'chk("tree_sha_stamp", tree_sha_stamp)' in pf
    # the ONE remaining volume-path spelling of a tree is the preflight's refusal fallback (the check names the remedy)
    assert src.count('tree = f"{C.VOL_REPOS_ROOT}/sisteer-{sha}"') == 1
    assert "tree = str(resolve_staged_tree(sha)[0])" in src[src.index("def check_protocol_sampling("):]


# --------------------------------------------------------------------------- #
# rule 2: a full-shape per-cell store is exactly 40 (36 today + standup/ x3)     #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not MIRROR_STORE.is_dir(), reason="the canonical mirror is not on this machine")
def test_a_real_full_shape_solo_store_counts_36_inodes_and_40_with_the_standup_files(tmp_path):
    """The inode count, measured: the mirrored gemma `none` full store is 12 directories (the root
    included) + 24 files = 36; a COPY (the mirror is read-only) + the row engine's standup/ (1 + 3) = exactly 40."""
    census = inode_census(MIRROR_STORE)
    assert census == {"dirs": 12, "files": 24, "total": SOLO_FULL_SHAPE_INODES}
    copy = tmp_path / "store"
    shutil.copytree(MIRROR_STORE, copy, symlinks=True)
    assert count_inodes(copy) == SOLO_FULL_SHAPE_INODES
    (copy / STANDUP_DIR).mkdir()
    for name in STANDUP_FILES:
        (copy / STANDUP_DIR / name).write_text("x")
    assert assert_store_inodes(copy, ROW_ENGINE_STORE_INODES)["total"] == 40
    # one more file anywhere in the store breaks the bound (zero margin) — the assertion says so
    (copy / "one_more.json").write_text("{}")
    with pytest.raises(AssertionError, match="!= 40"):
        assert_store_inodes(copy, ROW_ENGINE_STORE_INODES)


def test_the_synthetic_full_shape_store_is_todays_layout_through_the_real_writers_and_counts_exactly_40(tmp_path):
    cell_dir = full_shape_store(tmp_path / "row-engine")
    assert assert_store_inodes(tmp_path / "row-engine", ROW_ENGINE_STORE_INODES) == {"dirs": 13, "files": 27, "total": 40}
    for task in (BCB_FULL, BFCL_FULL):                         # written by completion.finish_task, read back by its readers
        assert completion.is_done(cell_dir, task, require_outcomes=True)
        assert completion.read_task_outcomes(cell_dir, task)["n_attempted_items"] == 2
    assert full_shape_store(tmp_path / "solo", standup=False) and count_inodes(tmp_path / "solo") == SOLO_FULL_SHAPE_INODES
    assert count_inodes(full_shape_store(tmp_path / "solo-pointer", standup=False, solo_pointer=True).parents[1]) \
        == SOLO_FULL_SHAPE_WITH_POINTER_INODES
    if MIRROR_STORE.is_dir():
        # the synthetic solo store's layout IS the real one's, name for name (the variables masked)
        assert normalised_layout(tmp_path / "solo") == normalised_layout(MIRROR_STORE)


def test_count_inodes_agrees_between_the_launcher_and_the_shared_helper(L, tmp_path):
    full_shape_store(tmp_path / "s")
    assert L.count_inodes(tmp_path / "s") == count_inodes(tmp_path / "s") == 40
    assert L.count_inodes(tmp_path / "absent") == count_inodes(tmp_path / "absent") == 0
    assert L.count_inodes(tmp_path / "s" / "RESULT") == 1
    assert L.INODES_PER_CELL_STORE == ROW_ENGINE_STORE_INODES == 40


# --------------------------------------------------------------------------- #
# rule 5: the per-launch budget check                                          #
# --------------------------------------------------------------------------- #


def test_inode_budget_verdict_refuses_at_the_floor_and_states_the_number_either_way(L):
    total = 500_000
    at_floor = L.inode_budget_verdict(450_000, total, 0)                       # exactly 90.0 % used, nothing added
    assert at_floor["ok"] is False and at_floor["status"] == L.INODE_BUDGET_EXCEEDED == "INODE_BUDGET_EXCEEDED"
    assert at_floor["pct_used"] == 90.0 and at_floor["pct_after"] == 90.0 and at_floor["headroom_inodes"] == 0
    assert "90.00 %" in at_floor["refusal"] and "tinycap_close_wave.py" in at_floor["refusal"]
    crossing = L.inode_budget_verdict(449_400, total, 700)                     # 89.88 % used; the launch crosses to 90.02 %
    assert crossing["ok"] is False and crossing["pct_used"] == 89.88 and crossing["pct_after"] == 90.02
    assert crossing["headroom_inodes"] == -100
    under = L.inode_budget_verdict(449_400, total, 100)                        # 89.9 % after: under the floor
    assert under["ok"] is True and under["status"] == "OK" and under["refusal"] is None and under["pct_after"] == 89.9
    today = L.inode_budget_verdict(203_500, total, 483)                        # the volume after the 09-28 sweeps
    assert today["ok"] and today["pct_used"] == 40.7 and today["pct_after"] == 40.8 and today["inodes_after"] == 203_983
    assert today["floor_pct"] == 90.0 and today["inodes_free"] == 296_500
    # a percent passed as the floor refuses by name (never a 9,000 % floor); so do impossible readings
    with pytest.raises(ValueError, match=r"0\.90"):
        L.inode_budget_verdict(1, total, 0, floor=90)
    with pytest.raises(ValueError, match="positive"):
        L.inode_budget_verdict(0, 0, 0)
    with pytest.raises(ValueError, match=">= 0"):
        L.inode_budget_verdict(1, total, -1)
    with pytest.raises(ValueError, match="within"):
        L.inode_budget_verdict(total + 1, total, 0)
    # the line main prints, and the refusal predicate
    assert L.inode_budget_line(today) == ("[tinycap] inodes: used 203,500 of 500,000 (40.7 %); forecast +483 -> 40.8 % "
                                          "(floor 90 %) OK")
    assert L.inode_budget_line(at_floor).endswith("(floor 90 %) REFUSED (INODE_BUDGET_EXCEEDED)")
    assert "not probed" in L.inode_budget_line(None) and "not probed" in L.inode_budget_line({})
    assert L.inode_budget_refused(at_floor) and not L.inode_budget_refused(today) and not L.inode_budget_refused(None)


def test_forecast_inodes_follows_the_plans_formula(L):
    """The row dir's bound is an ENUMERATION (landing 2 follow-up): 19 base — the dir, standup/ (1 + 3 + the persisted 3 +
    the compile-cache head + factors), config/ (1 + 1), watch/ (1 + the GPU pointer + the run's verifiers record),
    generation/, spills/, gpu_interval.json, row_rec.json — and 10 per cell — the ledger log, the row sweep, the
    verifier's pointer / record / sweep, the sidecar / marker / archived log / row_engine__<label>.json copy, the spill
    ledger. Landing 1's 12 + 7N left the record copies, the ledger logs, the spills and the verifiers record uncounted."""
    f = L.forecast_inodes(9, staged=True, n_new_adapters=8)      # the qwen s1 row: 9 stores, one row dir, 8 fitted cells
    assert f["total"] == 9 * 40 + (19 + 10 * 9) + 0 + 8 * 5 + 8 == 517
    assert (f["stores"], f["row_dir"], f["tarball"], f["adapters"], f["compile_cache"]) == (360, 109, 0, 40, 8)
    assert L.forecast_inodes(9, staged=False, n_new_adapters=8)["total"] == 518            # + the tarball
    assert L.forecast_inodes(0, staged=True)["total"] == 19 + 8                              # a warm probe: a row dir, the cache
    assert L.forecast_inodes(3, staged=True, n_rows=3)["row_dir"] == 3 * 19 + 10 * 3         # three solo cells: a row dir each
    with pytest.raises(ValueError, match="negative"):
        L.forecast_inodes(-1, staged=True)
    assert (L.INODES_PER_CELL_STORE, L.ROW_DIR_INODES_BASE, L.ROW_DIR_INODES_PER_CELL, L.INODES_PER_ADAPTER,
            L.COMPILE_CACHE_INODES_MAX, L.INODE_FLOOR_DEFAULT) == (40, 19, 10, 5, 8, 0.90)
    # the enumeration behind the two constants, by name (a file added to the row dir is added HERE too)
    src = (REPO / "scripts/modal/tinycap_served.py").read_text(encoding="utf-8")
    for name in ("verifiers_<run id>.json", "row_engine__<label>.json", "vllm_server.head.log", "cache_key_factors.json",
                 "verifier_spills.jsonl", "ledger__<cell>.log"):
        assert name in src[src.index("ROW_DIR_INODES_BASE = ") - 1400:src.index("ROW_DIR_INODES_BASE = ")], name


def test_probe_volume_inodes_reads_statvfs_on_the_mount_and_a_synthetic_90_percent_reading_refuses(L, monkeypatch, tmp_path):
    """The statvfs probe (the w2_volume_probe method) against a synthetic >= 90 % reading: the verdict
    refuses; the probe refuses a filesystem with no inode table, and refuses to read a path that is not the mount."""
    readings = {str(tmp_path): SimpleNamespace(f_files=500_000, f_ffree=50_000)}
    monkeypatch.setattr(L.os, "statvfs", lambda p: readings[str(p)])
    probe = L.probe_volume_inodes(str(tmp_path))
    assert (probe["inodes_total"], probe["inodes_used"], probe["inodes_free"], probe["pct_used"]) == (500_000, 450_000, 50_000, 90.0)
    verdict = L.inode_budget_verdict(probe["inodes_used"], probe["inodes_total"], L.forecast_inodes(8, staged=True)["total"])
    assert verdict["ok"] is False and verdict["status"] == "INODE_BUDGET_EXCEEDED"
    readings[str(tmp_path)] = SimpleNamespace(f_files=0, f_ffree=0)
    with pytest.raises(ValueError, match="no inode table"):
        L.probe_volume_inodes(str(tmp_path))
    with pytest.raises(FileNotFoundError, match="not mounted"):
        L.probe_volume_inodes(str(tmp_path / "not-a-mount"))
    assert L.inodes_from_statvfs(SimpleNamespace(f_files=10, f_ffree=4), mount="/m")["pct_used"] == 60.0


def test_the_preflight_carries_the_budget_check_and_main_prints_the_number_then_refuses_before_any_spawn(L):
    src = LAUNCHER.read_text()
    pf = src[src.index("def preflight(sha: str"):src.index("def close_wave(")]
    assert "chk(INODE_BUDGET_CHECK, inode_budget)" in pf and "probe = probe_volume_inodes()" in pf
    assert "rep[INODE_BUDGET_FIELD] = {**verdict" in pf and L.INODE_BUDGET_FIELD == "inode_budget"
    # the check is the LAST one: the number is read after every other door, still before any GPU
    assert pf.index("chk(INODE_BUDGET_CHECK, inode_budget)") > pf.index('chk("out_root_writable", writable)')
    main = src[src.index("def main(sha: str"):]
    i_print = main.index("print(inode_budget_line(r.get(INODE_BUDGET_FIELD)))")
    i_refuse = main.index("if inode_budget_refused(r.get(INODE_BUDGET_FIELD)):")
    i_ok = main.index('if not r.get("ok"):\n            raise SystemExit("[tinycap] PREFLIGHT FAILED')
    i_spawn = main.index("fan_out(")
    assert i_print < i_refuse < i_ok < i_spawn                       # printed on every preflighted path, refused before any spawn
    assert f"[tinycap] {{INODE_BUDGET_EXCEEDED}}" in main


# --------------------------------------------------------------------------- #
# rule 6: registry emission stays off in the launcher                          #
# --------------------------------------------------------------------------- #


def test_the_launcher_never_sets_the_registry_run_identity_variable():
    from sisteer.registry import hooks as H
    name = "SISTEER_REGISTRY_" + "RUN_IDENTITY"
    assert getattr(H, "RUN_IDENTITY_ENV", name) == name or name in Path(H.__file__).read_text()
    assert name not in LAUNCHER.read_text()


# --------------------------------------------------------------------------- #
# rule 4: the wave-close prune — the planner and the container half            #
# --------------------------------------------------------------------------- #


def _row(cell: str, **over):
    r = {"model_key": "gemma4e4b", "view": "s1", "cell": cell, "tag": "gemma4e4b_s1", "bcbw_wave": "gemma4e4b_s1",
         "tuple_id": f"s1/{cell}", "sha": FULL, "suffix": SUFFIX, "row_dir": f"rows/{ROW}/{SUFFIX}",
         "view_shared": "gemma4e4b_s1_shared", "comparison_kind": "shared_vs_solo"}
    r.update(over)
    return r


def _records(tmp_path: Path, *, gate_sha=FULL, gate_verdict="PASS", drop_gate_for=(), mirror_drop=(), row_dir_mirrored=True,
             rows=None):
    """A synthetic records world: the launch set (two evals, two cells), gate records, two PROVENANCE files, relaunches."""
    rows = rows if rows is not None else [_row("none"), _row(LORA)]
    launch_set = {"schema": "launch_sets/v1", "trees": [{"sha": FULL, "role": "the landing tree"}],
                  "evals": {"bigcodebench": {"launch": rows}, "bfcl": {"launch": rows}}}
    records = tmp_path / "records"
    records.mkdir(parents=True, exist_ok=True)
    for r in rows:
        if r["cell"] in drop_gate_for:
            continue
        (records / f"gate_{r['tag']}_{r['cell']}_{r['suffix']}.json").write_text(json.dumps(
            {"wave": r["tag"], "cell": r["cell"], "suffix": r["suffix"], "row": ROW, "sha": gate_sha, "verdict": gate_verdict,
             "problems": [] if gate_verdict == "PASS" else ["done marker missing: bfcl_full"]}))
    rels = [f"native/{ROW}/{r['cell']}/thinking_off/{r['suffix']}" for r in rows]
    provs = []
    for ev in ("bigcodebench", "bfcl"):
        provs.append({"schema": "w4_mirror_provenance/1", "eval": ev, "source": f"{ev}_full_shared",
                      "stores_mirrored": [x for x in rels if not any(x.endswith(f"/{c}/thinking_off/{SUFFIX}") for c in mirror_drop)],
                      "records": ([f"records/rows/{ROW}/{SUFFIX}/row_rec.json", f"records/rows/{ROW}/{SUFFIX}/watch/attempt_fc-row.json"]
                                  if row_dir_mirrored else []) + ["records/launch_sets_shared.json"]})
    relaunches = {
        f"gemma4e4b_s1/{LORA}": {"tag": "r2", "sha": FULL, "parked_suffix": f"bcbw-gemma4e4b_s1-{SHA7}-shared",
                                 "why": "worker loss after none completed; the fitted store was INCOMPLETE (D209), parked"},
        "llama8b_s1/none": {"tag": "r2", "sha": OTHER, "parked_suffix": "bcbw-llama8b_s1-0000000", "why": "another wave"},
        "gemma4e4b_s1/none": {"tag": "r2", "sha": FULL, "why": "nothing was created — no parked suffix"}}
    return launch_set, records, provs, relaunches


def test_the_close_wave_planner_names_the_tarball_parked_and_negctl_stores_preflight_artefacts_and_row_dirs(L, tmp_path):
    CW = _load_close_wave()
    launch_set, records, provs, relaunches = _records(tmp_path)
    rep = CW.plan(sha=FULL, launch_set=launch_set, records_dir=records, provenances=provs, relaunches=relaunches, now=1.0)
    assert rep["ok"] is True and rep["refusals"] == [] and len(rep["rows"]) == 2
    assert rep["notes"][0] == "relaunch gemma4e4b_s1/none: names no parked_suffix — nothing to prune by name" and len(rep["notes"]) == 2
    inv = rep["inventory"]
    assert inv["schema"] == CW.INVENTORY_SCHEMA == L.CLOSE_WAVE_INVENTORY_SCHEMA and inv["sha"] == FULL and inv["created_at"] == 1.0
    by_kind = {}
    for e in inv["entries"]:
        by_kind.setdefault(e["kind"], []).append(e)
    assert tuple(CW.KINDS) == L.CLOSE_WAVE_KINDS
    # the tree archive is shared by every wave at the sha — a wave QUEUED there has no row dir yet — so it is NEVER in
    # a wave's inventory: no tarball kind, the block says why, a note says it is kept
    assert "tarball" not in by_kind and inv["tree_archive"] == {"pruned": False, "reason": "shared at sha; pruned only by an explicit cross-wave operation (not this landing)"}
    assert any(n == f"tree archive tinycap/repos/sisteer-{FULL}.tar.gz kept: {CW.TREE_ARCHIVE_REASON} (D23)" for n in rep["notes"])
    assert "tarball" not in CW.KINDS and "tarball" not in L.CLOSE_WAVE_KINDS and CW.KIND_TARBALL_RETIRED == L.CLOSE_WAVE_KIND_TARBALL_RETIRED == "tarball"
    assert [e["rel"] for e in by_kind["parked_store"]] == [f"tinycap/gen/native/{ROW}/{LORA}/thinking_off/bcbw-gemma4e4b_s1-{SHA7}-shared"]
    assert "D209" in by_kind["parked_store"][0]["why"]
    # code review R3-2: the smoke's stores by THIS wave's suffix (a `-r2` / `-rep2` row folds to its primary), never `*-<sha7>-*negctl`
    assert [e["rel"] for e in by_kind["negctl_store"]] == [f"tinycap/gen/native/{ROW}/*/thinking_off/{SUFFIX}-negctl"] and by_kind["negctl_store"][0]["glob"] is True
    assert CW.negctl_globs([{"model_row": ROW, "suffix": SUFFIX + "-r2"}, {"model_row": ROW, "suffix": SUFFIX + "-rep2"}, {"model_row": ROW, "suffix": SUFFIX},
                            {"model_row": ROW, "suffix": "bcbw-gemma4e4b_s1-0000000-shared"}], SHA7) == [(ROW, f"{SUFFIX}-negctl")]
    assert [e["rel"] for e in by_kind["preflight_artefact"]] == [f"tinycap/gen/_preflight/{ROW}/{FULL}", "tinycap/gen/_preflight/probe"]
    assert [e["rel"] for e in by_kind["row_dir"]] == [f"tinycap/gen/rows/{ROW}/{SUFFIX}"]
    # code review R3-1: a PASS plan carries the planner's mark — the verdict and the digest the container recomputes
    assert inv[C.CLOSE_WAVE_PLAN_VERDICT_KEY] == "PASS" and inv[C.CLOSE_WAVE_PLAN_DIGEST_KEY] == C.close_wave_inventory_digest(inv)
    assert C.close_wave_inventory_approval_problems(inv) == [] and inv["would_prune_once"] is None
    # every entry passes the container's admission rule (the same check both halves run)
    assert L.close_wave_inventory_problems(inv) == []
    # the other wave's relaunch and the entry without a parked suffix name nothing
    assert all("llama8b" not in e["rel"] and "/none/thinking_off/" not in e["rel"] for e in inv["entries"])
    assert CW.meter_decision_text(FULL, 123) == f"close-wave {SHA7}: pruned 123 inodes"
    argv = CW.modal_run_argv(inv, confirm=True, modal_python="/x/python", launcher="/repo/scripts/modal/tinycap_served.py")
    assert argv[:4] == ["/x/python", "-m", "modal", "run"] and argv[4] == "/repo/scripts/modal/tinycap_served.py::close_wave"
    assert argv[5] == "--inventory-json" and json.loads(argv[6]) == json.loads(json.dumps(inv)) and argv[-1] == "--confirm"
    # a tarball entry is refused BY KIND on both halves, however the inventory was assembled (the planner's own mark included) —
    # the queued-wave case (a second wave at this sha, not started, no row dir) is covered by construction: no path can name the archive
    tar_entry = {"kind": "tarball", "rel": f"tinycap/repos/sisteer-{FULL}.tar.gz", "why": "hand-added"}
    (why,) = CW.inventory_refusals({**inv, "entries": inv["entries"] + [tar_entry]})
    assert why.startswith(f"entry {len(inv['entries'])}: tarball 'tinycap/repos/sisteer-{FULL}.tar.gz' — the tree archive is shared at sha") and "not this wave's to prune" in why
    assert CW.inventory_refusals(inv) == []
    marked = _approved({**inv, "entries": inv["entries"] + [tar_entry]})
    (cp,) = L.close_wave_inventory_problems(marked)
    assert cp == (f"entry {len(inv['entries'])}: tarball 'tinycap/repos/sisteer-{FULL}.tar.gz': the tree archive is shared at sha; pruned only by an "
                  "explicit cross-wave operation (not this landing) — never a wave's to prune (D23); refused by kind")
    assert "unknown kind" not in cp                                                            # by kind, by name — not as a stranger
    # the retired flags are gone from the CLI (argparse refuses them) and the plan takes no such keyword
    import inspect
    assert not {"prune_tree_archive", "known_launch_sets", "extra_tarballs"} & set(inspect.signature(CW.plan).parameters)
    src = CLOSE_WAVE.read_text()
    assert "tree_archive_consumers" not in src and "--known-launch-set" not in src and "--extra-tarball" not in src and "--also-prune-tree-archive" not in src


def test_the_close_wave_planner_refuses_without_a_pass_gate_at_the_sha_or_a_mirrored_store_or_row_dir(L, tmp_path):
    CW = _load_close_wave()

    def refusals(**kw):
        launch_set, records, provs, relaunches = _records(tmp_path / str(len(list(tmp_path.iterdir()))), **kw)
        rep = CW.plan(sha=FULL, launch_set=launch_set, records_dir=records, provenances=provs, relaunches=relaunches)
        assert rep["ok"] is False and rep["inventory"]["would_prune_once"] == rep["refusals"]
        # code review R3-1: a REFUSED plan's inventory is marked REFUSED, carries no digest, and the CONTAINER refuses to act on it —
        # `modal run …::close_wave --inventory-json <the saved --out's inventory> --confirm` deletes nothing
        inv = rep["inventory"]
        assert inv[C.CLOSE_WAVE_PLAN_VERDICT_KEY] == "REFUSED" and C.CLOSE_WAVE_PLAN_DIGEST_KEY not in inv
        problems = L.close_wave_inventory_problems(inv)
        assert any(p.startswith("the planner REFUSED this inventory (would_prune_once lists") for p in problems)
        assert any("no planner PASS verdict (plan_verdict='REFUSED')" in p for p in problems) and any("plan_digest missing" in p for p in problems)
        return rep["refusals"]

    store = f"native/{ROW}/{LORA}/thinking_off/{SUFFIX}"
    assert refusals(drop_gate_for=(LORA,)) == [f"{store}: no gate record"]
    (r,) = [x for x in refusals(gate_sha=OTHER) if store in x]
    assert "not this wave's" in r
    assert any("reads 'FAIL', not PASS (done marker missing: bfcl_full)" in x for x in refusals(gate_verdict="FAIL"))
    assert refusals(mirror_drop=(LORA,)) == [f"{store}: not in stores_mirrored of the bigcodebench mirror",
                                             f"{store}: not in stores_mirrored of the bfcl mirror"]
    assert refusals(row_dir_mirrored=False) == [f"row dir rows/{ROW}/{SUFFIX}: not mirrored (no PROVENANCE records/rows/{ROW}/{SUFFIX}/… "
                                                "entry or row_dirs_mirrored)"]
    # a row that declares no row dir refuses: the prune cannot guess which row dir a cell rode
    assert any("declares no row_dir" in x for x in refusals(rows=[_row("none", row_dir=None), _row(LORA)]))
    # rows at another tree are another wave's: nothing to close at this sha (and this wave's relaunch entry, whose wave no
    # row places any more, is refused by name rather than guessed)
    none_here = refusals(rows=[_row("none", sha=OTHER)])
    assert f"the launch set has no row at sha {FULL} — nothing to close" in none_here
    assert any("cannot place the parked store" in x for x in none_here) and len(none_here) == 2
    # a mirror for an eval the rows file is required (a store is mirrored per eval root)
    launch_set, records, provs, relaunches = _records(tmp_path / "one-prov")
    rep = CW.plan(sha=FULL, launch_set=launch_set, records_dir=records, provenances=provs[:1], relaunches=relaunches)
    assert rep["refusals"] and all("no PROVENANCE given for eval 'bfcl'" in x for x in rep["refusals"])
    with pytest.raises(CW.PlanError, match="40-hex"):
        CW.plan(sha=SHA7, launch_set=launch_set, records_dir=records, provenances=provs, relaunches={})
    # the CLI: a dry run writes the plan and exits 0; a refusal exits 2 and touches nothing
    (tmp_path / "ls.json").write_text(json.dumps(launch_set))
    (tmp_path / "rl.json").write_text(json.dumps(relaunches))
    for i, p in enumerate(provs):
        (tmp_path / f"prov{i}.json").write_text(json.dumps(p))
    argv = ["--sha", FULL, "--launch-set", str(tmp_path / "ls.json"), "--records-dir", str(records), "--relaunches",
            str(tmp_path / "rl.json"), "--provenance", str(tmp_path / "prov0.json"), "--provenance", str(tmp_path / "prov1.json"),
            "--out", str(tmp_path / "plan.json")]
    assert CW.main(argv) == CW.EXIT_OK and json.loads((tmp_path / "plan.json").read_text())["ok"] is True
    assert CW.main(argv[:-2] + ["--confirm"]) == CW.EXIT_REFUSED                # --confirm without --app-name
    assert CW.main(argv[:10]) == CW.EXIT_REFUSED                                 # one provenance only: refused
    # through the CLI: the archive-prune flags no longer exist (argparse refuses them); a plan never names a tarball
    for retired in (["--also-prune-tree-archive"], ["--known-launch-set", str(tmp_path / "ls.json")], ["--extra-tarball", "x.tar.gz"]):
        with pytest.raises(SystemExit):
            CW.main(argv + retired)
    assert CW.main(argv) == CW.EXIT_OK
    written = json.loads((tmp_path / "plan.json").read_text())
    assert written["inventory"]["tree_archive"] == {"pruned": False, "reason": CW.TREE_ARCHIVE_REASON} and not any(e["kind"] == "tarball" for e in written["inventory"]["entries"])


def _approved(inv: dict) -> dict:
    """A hand-built inventory with the planner's mark (code review R3-1) — what `tinycap_close_wave.approve_inventory` stamps on a PASS plan."""
    body = {k: v for k, v in inv.items() if k not in (C.CLOSE_WAVE_PLAN_VERDICT_KEY, C.CLOSE_WAVE_PLAN_DIGEST_KEY)}
    return {**body, C.CLOSE_WAVE_PLAN_VERDICT_KEY: C.CLOSE_WAVE_PLAN_PASS, C.CLOSE_WAVE_PLAN_DIGEST_KEY: C.close_wave_inventory_digest(body)}


def test_close_wave_inventory_problems_name_every_inadmissible_entry(L):
    ok = _approved({"schema": L.CLOSE_WAVE_INVENTORY_SCHEMA, "sha": FULL, "entries": []})
    assert L.close_wave_inventory_problems(ok) == []
    assert L.close_wave_inventory_problems(_approved({**ok, "schema": "x"})) == [f"schema 'x' != {L.CLOSE_WAVE_INVENTORY_SCHEMA!r}"]
    assert L.close_wave_inventory_problems(_approved({**ok, "sha": SHA7})) == [f"sha {SHA7!r} is not a full 40-hex commit"]
    assert L.close_wave_inventory_problems("nope") == ["inventory is not an object"]
    # code review R3-1: the planner's mark is required at the container boundary — a REFUSED plan's inventory (would_prune_once),
    # a missing / other verdict, a missing digest, an inventory edited after the planner passed it (digest mismatch), each by name
    bare = {"schema": L.CLOSE_WAVE_INVENTORY_SCHEMA, "sha": FULL, "entries": []}
    assert L.close_wave_inventory_problems(bare) == [
        "no planner PASS verdict (plan_verdict=None) — the container acts only on an inventory the planner passed",
        "plan_digest missing — the entries are not the ones the planner passed (edited, or assembled by hand)"]
    held = L.close_wave_inventory_problems({**bare, "would_prune_once": ["store x: no gate record", "row dir y: not mirrored"],
                                            C.CLOSE_WAVE_PLAN_VERDICT_KEY: "REFUSED"})
    assert held[0] == ("the planner REFUSED this inventory (would_prune_once lists 2 reason(s): store x: no gate record; row dir y: not mirrored) "
                       "— not actionable; re-plan") and len(held) == 3
    edited = dict(ok, entries=[{"kind": "preflight_artefact", "rel": "tinycap/gen/_preflight/probe", "why": "added by hand"}])
    assert L.close_wave_inventory_problems(edited) == ["plan_digest does not match the inventory — the entries are not the ones the planner passed (edited, or assembled by hand)"]
    assert L.close_wave_inventory_problems(_approved(edited)) == [] and C.close_wave_inventory_digest({**ok, "created_at": 99.0}) == ok[C.CLOSE_WAVE_PLAN_DIGEST_KEY]
    bad = [{"kind": "cache", "rel": "tinycap/cache/compile/x.tar.gz"},                                   # an unknown kind
           {"kind": "preflight_artefact", "rel": "/vol/tinycap/gen/_preflight/probe"},                 # absolute
           {"kind": "preflight_artefact", "rel": "tinycap/gen/../objects/x"},                          # escapes the root
           {"kind": "tarball", "rel": f"tinycap/repos/sisteer-{FULL}.tar.gz"},                        # The retired kind, well-formed path
           {"kind": "parked_store", "rel": f"tinycap/gen/native/{ROW}/none/thinking_off"},              # not a store path
           {"kind": "parked_store", "rel": f"tinycap/gen/native/{ROW}/none/thinking_off/bcbw-x-0000000"},  # another sha7
           {"kind": "negctl_store", "rel": f"tinycap/gen/native/{ROW}/none/thinking_off/bcbw-x-{SHA7}-shared"},  # not negctl
           {"kind": "preflight_artefact", "rel": "tinycap/gen/native/x"},                              # not under _preflight
           {"kind": "row_dir", "rel": f"tinycap/gen/rows/{ROW}"},                                      # no suffix
           {"kind": "row_dir", "rel": f"tinycap/gen/rows/{ROW}/bcbw-x-0000000-shared"},                # another sha7
           {"kind": "row_dir", "rel": "tinycap/objects/rows/x/y"},                                    # outside gen
           "not an object"]
    problems = L.close_wave_inventory_problems(_approved({**ok, "entries": bad}))
    assert len(problems) == len(bad) and [p.split(":")[0] for p in problems] == [f"entry {i}" for i in range(len(bad))]
    assert "unknown kind" in problems[0] and "volume-relative" in problems[1] and "'..'" in problems[2]
    assert "never a wave's to prune (D23); refused by kind" in problems[3] and "native/<row>/<cell>" in problems[4] and "sha7" in problems[5]
    assert "negctl" in problems[6] and "_preflight" in problems[7] and "<row>/<suffix>" in problems[8] and "sha7" in problems[9]
    assert "not under tinycap/gen/" in problems[10] and "not an object" in problems[11]
    # a glob entry is admitted by shape without the sha7 rule (the container re-checks every match)
    assert L.close_wave_inventory_problems(_approved({**ok, "entries": [
        {"kind": "negctl_store", "rel": f"tinycap/gen/native/{ROW}/*/thinking_off/*-{SHA7}-*negctl", "glob": True}]})) == []


def test_close_wave_prints_the_inventory_and_deletes_only_under_confirm(L, tmp_path):
    """The container half over a fake mounted volume: the inventory first, with per-path inode counts; a dry run
    deletes nothing; confirm deletes exactly the present admissible paths and reports the inodes freed; an unmarked
    or REFUSED inventory does nothing (R3-1); one refused expanded entry deletes nothing (all or nothing, R3-1); the
    tree archives on the volume — this tree's and another's — are bystanders no entry kind can name."""
    vol = tmp_path / "vol"
    repos = vol / "tinycap" / "repos"
    gen = vol / "tinycap" / "gen"
    mine = _tarball(repos, SHA7, FULL)
    other = _tarball(repos, SHA7 + "x", OTHER)                       # matched by the sha7 glob, names ANOTHER tree
    parked = gen / "native" / ROW / LORA / "thinking_off" / f"bcbw-gemma4e4b_s1-{SHA7}-shared"
    full_shape_store(parked, cell=LORA)
    negctl = gen / "native" / ROW / "none" / "thinking_off" / f"bcbw-gemma4e4b_s1-{SHA7}-shared-negctl"
    full_shape_store(negctl, standup=True)
    canonical = gen / "native" / ROW / "none" / "thinking_off" / "bcbw-gemma4e4b_s1-8ba0b48"      # another wave's store
    full_shape_store(canonical, standup=False)
    pf = gen / "_preflight" / ROW / FULL
    pf.mkdir(parents=True)
    (pf / "prompt_budget.json").write_text("{}")
    (gen / "_preflight" / "probe").write_text("ok")
    row_dir = gen / "rows" / ROW / SUFFIX
    (row_dir / "watch").mkdir(parents=True)
    (row_dir / "row_rec.json").write_text("{}")
    (row_dir / "watch" / "attempt_fc-row.json").write_text("{}")
    inv = _approved({"schema": L.CLOSE_WAVE_INVENTORY_SCHEMA, "sha": FULL, "entries": [
        {"kind": "parked_store", "rel": f"tinycap/gen/native/{ROW}/{LORA}/thinking_off/bcbw-gemma4e4b_s1-{SHA7}-shared", "why": "p"},
        {"kind": "negctl_store", "rel": f"tinycap/gen/native/{ROW}/*/thinking_off/*-{SHA7}-*negctl", "why": "n", "glob": True},
        {"kind": "preflight_artefact", "rel": f"tinycap/gen/_preflight/{ROW}/{FULL}", "why": "pf"},
        {"kind": "preflight_artefact", "rel": "tinycap/gen/_preflight/probe", "why": "pf"},
        {"kind": "row_dir", "rel": f"tinycap/gen/rows/{ROW}/{SUFFIX}", "why": "r"}]})
    # code review R3-1: the SAME inventory without the planner's mark (the shape a hand-assembled or REFUSED-plan inventory has) does nothing
    unmarked = {k: v for k, v in inv.items() if k not in (C.CLOSE_WAVE_PLAN_VERDICT_KEY, C.CLOSE_WAVE_PLAN_DIGEST_KEY)}
    refused_plan = L._close_wave_apply({**unmarked, "would_prune_once": ["x: no gate record"], C.CLOSE_WAVE_PLAN_VERDICT_KEY: "REFUSED"}, confirm=True, volume_root=vol, log=lambda s: None)
    assert refused_plan["ok"] is False and refused_plan["deleted"] == [] and any("the planner REFUSED this inventory" in p for p in refused_plan["problems"])
    assert L._close_wave_apply(unmarked, confirm=True, volume_root=vol, log=lambda s: None)["ok"] is False
    assert mine.is_file() and parked.is_dir() and negctl.is_dir() and row_dir.is_dir()
    # code review R3-1: every expanded entry validates BEFORE any deletion — one refused present entry (a glob that over-matches: it takes
    # another wave's canonical store, whose suffix carries no sha7) beside the admissible present paths deletes NOTHING
    lines = []
    with_bad = _approved({**unmarked, "entries": unmarked["entries"] + [
        {"kind": "negctl_store", "rel": f"tinycap/gen/native/{ROW}/none/thinking_off/*", "why": "over-matching glob", "glob": True}]})
    partial = L._close_wave_apply(with_bad, confirm=True, volume_root=vol, log=lines.append)
    assert partial["ok"] is False and partial["deleted"] == [] and partial["inodes_freed"] == 0
    assert partial["problems"] == [f"tinycap/gen/native/{ROW}/none/thinking_off/bcbw-gemma4e4b_s1-8ba0b48: negctl_store "
                                   f"'tinycap/gen/native/{ROW}/none/thinking_off/bcbw-gemma4e4b_s1-8ba0b48': the suffix does not carry this wave's sha7 {SHA7}"]
    assert any(ln.startswith("[close-wave] REFUSED: 1 present entry fail(s) validation — nothing deleted (all or nothing)") for ln in lines)
    assert mine.is_file() and parked.is_dir() and negctl.is_dir() and pf.is_dir() and row_dir.is_dir() and other.is_file() and canonical.is_dir()
    # the tree archive is a BYSTANDER of every per-wave close — a second wave queued at this sha (not started: no row dir, no store)
    # depends on `mine`; no entry kind can name it, an approved inventory carrying a tarball entry is refused by kind and deletes nothing
    with_tar = _approved({**unmarked, "entries": unmarked["entries"] + [{"kind": "tarball", "rel": f"tinycap/repos/sisteer-{SHA7}.tar.gz", "why": "t"}]})
    refused_tar = L._close_wave_apply(with_tar, confirm=True, volume_root=vol, log=lambda s: None)
    assert refused_tar["ok"] is False and refused_tar["deleted"] == [] and any("refused by kind" in p and "(D23)" in p for p in refused_tar["problems"])
    assert mine.is_file() and other.is_file() and parked.is_dir()
    # three statvfs readings: the dry run's "before", then the confirmed run's "before" and "after"
    readings = iter([{"inodes_used": 1000, "inodes_total": 5000}, {"inodes_used": 1000, "inodes_total": 5000},
                     {"inodes_used": 1000 - 40 - 40 - 2 - 1 - 4, "inodes_total": 5000}])
    lines = []
    dry = L._close_wave_apply(inv, confirm=False, volume_root=vol, probe=lambda: next(readings), log=lines.append)
    assert dry["ok"] is True and dry["confirm"] is False and dry["deleted"] == [] and dry["inodes_freed"] == 0
    assert mine.is_file() and parked.is_dir() and negctl.is_dir() and canonical.is_dir() and row_dir.is_dir()
    present = {e["rel"]: e for e in dry["entries"] if e["present"]}
    assert set(present) == {f"tinycap/gen/native/{ROW}/{LORA}/thinking_off/bcbw-gemma4e4b_s1-{SHA7}-shared",
                            f"tinycap/gen/native/{ROW}/none/thinking_off/bcbw-gemma4e4b_s1-{SHA7}-shared-negctl",
                            f"tinycap/gen/_preflight/{ROW}/{FULL}", "tinycap/gen/_preflight/probe", f"tinycap/gen/rows/{ROW}/{SUFFIX}"}
    assert present[f"tinycap/gen/native/{ROW}/{LORA}/thinking_off/bcbw-gemma4e4b_s1-{SHA7}-shared"]["inodes"] == 40
    assert lines[0].startswith(f"[close-wave] sha {FULL}: 5 present path(s), 87 inode(s) would be deleted")
    # confirm: exactly the admissible present paths go; BOTH tarballs (this tree's and another's) and the canonical store stay
    lines.clear()
    done = L._close_wave_apply(inv, confirm=True, volume_root=vol, probe=lambda: next(readings), log=lines.append)
    assert done["ok"] is True and done["confirm"] is True and done["inodes_freed"] == 40 + 40 + 2 + 1 + 4 == 87
    assert sorted(done["deleted"]) == sorted(present)
    assert done["statvfs_delta"] == 87 and done["statvfs_before"]["inodes_used"] == 1000
    assert not parked.exists() and not negctl.exists() and not pf.exists() and not row_dir.exists()
    assert mine.is_file() and other.is_file() and canonical.is_dir() and count_inodes(canonical) == 36
    # an inadmissible inventory does nothing at all
    assert L._close_wave_apply(_approved({**inv, "sha": SHA7}), confirm=True, volume_root=vol)["ok"] is False
    # the decorated function is the CPU entry `close_wave(inventory_json, confirm)` over VOL_PATH, wired to the same apply
    src = LAUNCHER.read_text()
    body = src[src.index("def close_wave(inventory_json: str, confirm: bool = False)"):src.index("def _upload_tree(")]
    assert "_close_wave_apply(inventory, confirm=bool(confirm), volume_root=VOL_PATH, probe=probe_volume_inodes)" in body
    assert "VOL.commit()" in body


def test_tarball_tree_sha_is_memoised_per_archive_bytes(L, tmp_path, monkeypatch):
    """The TREE_SHA member is a full gzip pass; the same archive (path, size, mtime_ns) is read once, a rewritten
    one is read again, a missing one is None without a cache entry; resolve_staged_tree's abbreviation path hands the sha it read to
    unpack_tree_local (no second pass)."""
    import tarfile
    repos = tmp_path / "repos"
    tb = _tarball(repos, SHA7 + "abc", FULL)
    opens = []
    real_open = tarfile.open
    monkeypatch.setattr(tarfile, "open", lambda *a, **k: (opens.append(a[0] if a else k.get("name")) or real_open(*a, **k)))
    assert L.tarball_tree_sha(tb) == FULL and L.tarball_tree_sha(tb) == FULL and L.tarball_tree_sha(str(tb)) == FULL
    assert len([o for o in opens if str(o) == str(tb)]) == 1                                  # ONE gzip pass for three calls
    assert L.tarball_tree_sha(repos / "absent.tar.gz") is None and not any(str(k[0]).endswith("absent.tar.gz") for k in L._TARBALL_TREE_SHA_CACHE)
    tb2 = _tarball(repos, SHA7 + "abc", OTHER)                                                # rewritten: another commit, re-read
    assert L.tarball_tree_sha(tb2) == OTHER
    # the abbreviation path (a tarball NAMED by a shorter prefix than the sha asked for): the sha read once for the match is
    # handed to unpack_tree_local — the only open left is the unpack's own
    repos2 = tmp_path / "repos2"
    tb3 = _tarball(repos2, FULL[:8], FULL)
    assert L.tarball_tree_sha(tb3) == FULL
    opens.clear()
    tree, full = L.resolve_staged_tree(FULL[:9], repos2, local_root=tmp_path / "local")
    assert full == FULL and (tree / "src" / "sisteer").is_dir()
    assert [str(o) for o in opens] == [str(tb3)]                                                # ONE open: the unpack; no second sha read

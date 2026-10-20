"""CPU-only contract tests for scripts/modal/petri_smoke.py, the Modal port of
scripts/sbatch/petri_steered_smoke.sbatch.

The launcher is imported with a synthetic Modal stub (no network, no
container). What is pinned here, and why each pin exists:

* the wrapper's command equals the FROZEN sbatch's ``srun ... python
  scripts/petri_steered_smoke.py smoke ...`` line flag for flag, parsed from
  the sbatch text by this file independently of the wrapper's own parser
  (which must agree) — drift in either file is caught;
* the wrapper's outcome vocabulary (marker names, phase keys, model pin,
  defaults) AND its restated gate (the BI=0 waiver set, the instrument-failure
  marks, the certified geometries, check_state / split_gating_failures) equal
  the FROZEN smoke script's own constants and functions; the KV-coverage
  verdict is the IMPORTED authoritative ``vllm_server.kv_coverage_problem``,
  never a replica;
* the unavailable-marker trap: ``smoke_unavailable.json`` + exit 0 is a
  FAILURE, as are ``smoke_failed.json``, a nonzero exit, a silent exit 0 and
  an incomplete record; PASS is the smoke's COMPLETE gate — one negative test
  per predicate — and never stricter: the base comparator throughput legs
  (evidence only in the smoke) are advisory;
* the two REQUIRED objects are the pinned ORIGINALS: hashed at preflight and
  in the GPU container (``verify_object``), and held to the same hashes in
  the record (``arm_problems``) — a same-kind look-alike is refused at both;
* the REAL 2026-09-06 Modal record (tests/fixtures/
  modal_petri_smoke_record_completed.json — the first canary at ad6c3cb,
  scrubbed of the volume id and container hostname, otherwise verbatim)
  replays to PASS with the two base legs advisory-missing, and flips to FAIL
  when a gating check is failed; the lens throughput key names the wrapper
  demands are pinned from that record, not assumed;
* evidence handling: every run publishes only under its own
  ``attempts/<attempt-id>/`` and a completed attempt is promoted by an atomic
  ``current.json`` replace (two simulated concurrent runs never touch each
  other's files; ``current`` names exactly one); PUBLICATION IS PART OF THE
  EXIT CONTRACT and the pointer is its LAST durable write — the complete
  attempt (RESULT ``published=true attempt=<id>``, record, evidence) is
  synced, committed and byte-verified while nothing names it, THEN
  ``current.json`` is replaced and read back, and nothing touches the attempt
  afterwards; a PASS exits 0 only with ``publication_ok`` and
  ``current.attempt`` naming this attempt; a failure injected into the final
  sync (once, or sticky), the promotion, the promotion's commit or the
  pointer's read-back makes the returned record PUBLISH_FAILED with exit 1,
  never a PASS pointer beside PUBLISH_FAILED files (the round-4 tree); the
  grading host mirror pulls all-or-nothing into ``versions/<pull-id>/`` behind an
  atomically replaced ``current`` symlink (one unreadable file leaves the
  previous mirror byte-identical; promotion only after every file landed AND
  the pointed attempt's own record and RESULT agree with the pointer, the
  record's ``current`` EQUAL to the pointer as a whole — a hand-built PASS
  pointer over PUBLISH_FAILED files, and a pointer differing in any single
  field (sha, image, result, path, promoted_at, status, verdict, app), is
  refused untouched); publication first QUIESCES the periodic sync worker
  (one lock, bounded join, ``is_alive`` checked) and FAILS CLOSED — nothing
  written, no pointer — when a stalled periodic sync outlives the bound; the
  pull after a run is BOUND to the pointer that run installed (two
  self-consistent PASS pointers for one sha, the record belonging to A while
  the volume's ``current.json`` names B → the mirror is left untouched, the
  outcome is MIRROR_SUPERSEDED, exit 2 — distinct from PUBLISH_FAILED's 1 —
  and ``--harvest`` binds to the pointer in that run's record); every local
  mirror transaction of a sha runs under one cross-process flock (two
  barrier-synchronized pulls of one sha: the second acquires only after the
  first released, each caller's ``promoted`` is what ``current`` resolved to
  under its own lock, GC at the retention limit never removes the current;
  a lock held past the bound is MIRROR_FAILED with the mirror untouched); a
  ``--harvest`` of a PUBLISH_FAILED record exits 1 without refreshing the
  mirror, and the mirror garbage-collects by an explicit
  pull sequence, never by the id's lexical order (same-second ids with the
  current one sorting in the middle leave exactly ``keep`` versions);
* the mandatory ``modal run -n petri-smoke-<shortsha>`` name check, and that
  ``--go`` cannot reach a GPU spawn past a failed preflight.
"""
from __future__ import annotations

import contextlib
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from front_door_testlib import MATCHERS, SCAN, token_hits

#: one tier of the publication scan's token table, matched as the scan matches it (the tokens are never spelled here)
STORAGE_TOKENS = SCAN.Matcher([r for r in MATCHERS["rows"] if r.tier == "storage"])

REPO = Path(__file__).resolve().parents[1]
SBATCH = REPO / "scripts/sbatch/petri_steered_smoke.sbatch"
SMOKE = REPO / "scripts/petri_steered_smoke.py"
REAL_RECORD = REPO / "tests/fixtures/modal_petri_smoke_record_completed.json"
#: the sha the real record was produced at (its provenance.checkout_sha)
REAL_SHA = "ad6c3cb90890df4548da616852c2bb1670b7f00b"


class _Chain:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: self


class _App:
    def __init__(self, name=None):
        self.name = name
        self.description = name
        self.app_id = "ap-TEST"

    def function(self, **_kwargs):
        return lambda fn: fn

    def local_entrypoint(self, **_kwargs):
        return lambda fn: fn


class _Volume:
    @classmethod
    def from_name(cls, *_args, **_kwargs):
        return cls()

    def commit(self):
        return None

    def reload(self):
        return None


def _modal_stub() -> types.ModuleType:
    mod = types.ModuleType("modal")
    mod.is_local = lambda: True
    mod.App = _App
    mod.Volume = _Volume
    mod.Secret = types.SimpleNamespace(from_name=lambda name: {"secret": name})
    mod.Image = types.SimpleNamespace(from_registry=lambda *_a, **_k: _Chain(),
                                      debian_slim=lambda *_a, **_k: _Chain())
    mod.FunctionCall = types.SimpleNamespace(from_id=lambda _id: None)
    return mod


def _load(name: str, rel: str, *, stub_modal: bool):
    old = sys.modules.get("modal")
    if stub_modal:
        sys.modules["modal"] = _modal_stub()
    try:
        spec = importlib.util.spec_from_file_location(name, REPO / rel)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if stub_modal:
            if old is None:
                sys.modules.pop("modal", None)
            else:
                sys.modules["modal"] = old


@pytest.fixture(scope="module")
def port():
    return _load("_modal_petri_smoke_test", "scripts/modal/petri_smoke.py", stub_modal=True)


@pytest.fixture(scope="module")
def smoke():
    """The FROZEN smoke script (stdlib-only at import; nothing runs)."""
    return _load("_petri_steered_smoke_test", "scripts/petri_steered_smoke.py", stub_modal=False)


@pytest.fixture(scope="module")
def real_record() -> dict:
    return json.loads(REAL_RECORD.read_text())


@pytest.fixture(scope="module")
def shared_kv_verdict():
    """The AUTHORITATIVE verdict, imported straight from this checkout's src."""
    sys.path.insert(0, str(REPO / "src"))
    try:
        from sisteer.capeval.serving.vllm_server import kv_coverage_problem
    finally:
        sys.path.remove(str(REPO / "src"))
    return kv_coverage_problem


def _sbatch_tokens_independent() -> list:
    """This file's own parse of the sbatch's command: the line naming the
    smoke script, backslash continuations joined, shell-split."""
    lines = SBATCH.read_text().splitlines()
    idx = [i for i, l in enumerate(lines)
           if "scripts/petri_steered_smoke.py smoke" in l and not l.lstrip().startswith("#")]
    assert len(idx) == 1, idx
    i = idx[0]
    joined = ""
    while lines[i].rstrip().endswith("\\"):
        joined += lines[i].rstrip()[:-1] + " "
        i += 1
    joined += lines[i]
    return shlex.split(joined)


# --- the command ---------------------------------------------------------------


def test_sbatch_command_is_the_known_launcher_plus_the_script():
    toks = _sbatch_tokens_independent()
    assert toks[:6] == ["srun", "--ntasks=1", "uv", "run", "--no-sync", "python"]
    assert toks[6:8] == ["scripts/petri_steered_smoke.py", "smoke"]


#: synthetic volume paths for the two objects: the site names the real ones, these name nothing anywhere
OBJ = {"static_r_matched": "/vol/objects/jailbreak/example_set/static_r_matched/object.pt",
       "alphasteer_official": "/vol/objects/jailbreak/example_set/alphasteer_official/object.pt"}
#: the sbatch's shell variable for each object, as its required-setting lines name them
SBATCH_OBJECT_VAR = {"$STATIC_R_OBJECT": "static_r_matched", "$ALPHASTEER_OBJECT": "alphasteer_official"}
#: the site key each object is named by (the smoke's, the sbatch's and the launcher's alike)
OBJECT_KEY = {"static_r_matched": "SISTEER_PETRI_SMOKE_STATIC_R_OBJECT",
              "alphasteer_official": "SISTEER_PETRI_SMOKE_ALPHASTEER_OBJECT"}


def test_wrapper_argv_equals_sbatch_flag_for_flag(port):
    toks = _sbatch_tokens_independent()[6:]          # from the script path on
    out_dir = "/root/petri-smoke-work/<sha>/<attempt>/steered_smoke"
    expected = []
    for t in toks:
        if t == "$SMOKE_OUT_DIR":
            expected.append(out_dir)
        elif t in SBATCH_OBJECT_VAR:
            expected.append(OBJ[SBATCH_OBJECT_VAR[t]])
        else:
            # the sbatch spells no object path of its own: the site names both
            assert not t.startswith("$"), f"unexpected shell variable in the sbatch command: {t}"
            expected.append(t)
    got = port.smoke_argv("/usr/bin/python", out_dir, OBJ["static_r_matched"], OBJ["alphasteer_official"])
    assert got[0] == "/usr/bin/python"
    assert got[1:] == expected, "\n".join(("wrapper:", *got[1:], "sbatch:", *expected))
    flags = dict(zip(expected[2::2], expected[3::2]))
    assert flags["--manifest"] == port.MANIFEST
    assert flags["--model"] == port.MODEL_HF
    assert flags["--revision"] == port.MODEL_REV
    assert flags["--static-r-object"] == OBJ["static_r_matched"]
    assert flags["--alphasteer-object"] == OBJ["alphasteer_official"]
    assert flags["--static-r-scale"] == port.STATIC_R_SCALE == "0.05"
    assert flags["--response-arm"] == port.RESPONSE_ARM == "on"
    assert flags["--fidelity-batch-invariant"] == port.FIDELITY_BATCH_INVARIANT == "0"
    assert flags["--fidelity-bi1-probe"] == port.FIDELITY_BI1_PROBE == "on"
    assert not any(token_hits(x, STORAGE_TOKENS) or x.startswith("$") for x in got), got


def _sbatch_object_lines() -> list:
    """The sbatch's code lines that read the two object keys into its object variables."""
    return [ln for ln in SBATCH.read_text().splitlines()
            if not ln.lstrip().startswith("#") and ln.startswith(("STATIC_R_OBJECT=", "ALPHASTEER_OBJECT="))]


def test_the_sbatch_takes_both_objects_from_the_site_keys_and_refuses_by_name_without_them(port, smoke):
    """The sbatch names no object: two lines read the site's keys from the job environment into the variables its command
    serves, before the venv bootstrap, and each refuses by name when its key is missing. A value passes as written (a $SHARED
    in it is the smoke's to expand), and the keys are the ones the smoke and the launcher read."""
    lines = _sbatch_object_lines()
    assert [ln.split("=", 1)[0] for ln in lines] == ["STATIC_R_OBJECT", "ALPHASTEER_OBJECT"], lines
    for ln, arm in zip(lines, ("static_r_matched", "alphasteer_official")):
        assert ln.split("=", 1)[1].startswith(f'"${{{OBJECT_KEY[arm]}:?set {OBJECT_KEY[arm]}, '), ln
    code = [ln for ln in SBATCH.read_text().splitlines() if not ln.lstrip().startswith("#")]
    assert max(code.index(ln) for ln in lines) < min(i for i, ln in enumerate(code) if "uv venv" in ln or "VLLM_REQS=" in ln)
    assert set(SBATCH_OBJECT_VAR) == set(port.SBATCH_OBJECT_VARS) and port.SBATCH_OBJECT_VARS == SBATCH_OBJECT_VAR
    assert port.OBJECT_KEYS == OBJECT_KEY == {"static_r_matched": smoke.STATIC_R_OBJECT_KEY,
                                              "alphasteer_official": smoke.ALPHASTEER_OBJECT_KEY}
    script = "set -uo pipefail\n" + "\n".join(lines) + '\nprintf "%s|%s" "$STATIC_R_OBJECT" "$ALPHASTEER_OBJECT"\n'
    base = {"PATH": "/usr/bin:/bin"}

    def run(**env):
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={**base, **env}, timeout=30)
    r = run()
    assert r.returncode != 0 and f"set {OBJECT_KEY['static_r_matched']}, the static_r_matched object.pt" in r.stderr, r.stderr
    assert r.stdout == ""
    r = run(**{OBJECT_KEY["static_r_matched"]: "/data/a/object.pt"})
    assert r.returncode != 0 and f"set {OBJECT_KEY['alphasteer_official']}, the alphasteer_official object.pt" in r.stderr
    vals = {OBJECT_KEY["static_r_matched"]: "$SHARED/exp/jailbreak/example_set/static_r_matched/object.pt",
            OBJECT_KEY["alphasteer_official"]: "/data/b/object.pt"}
    r = run(**vals)
    assert r.returncode == 0, r.stderr
    assert r.stdout == f"{vals[OBJECT_KEY['static_r_matched']]}|{vals[OBJECT_KEY['alphasteer_official']]}"


def test_wrapper_parser_agrees_with_independent_parse(port):
    text = SBATCH.read_text()
    assert port.sbatch_smoke_tokens(text) == _sbatch_tokens_independent()[6:]
    translated = port.sbatch_tokens_translated(text, "<out-dir>", OBJ)
    assert translated == port.smoke_argv("python", "<out-dir>", OBJ["static_r_matched"], OBJ["alphasteer_official"])[1:]
    # the sbatch's object variables stand for the objects this run serves, so translating them needs those objects
    with pytest.raises(ValueError, match="SISTEER_PETRI_SMOKE_STATIC_R_OBJECT"):
        port.sbatch_tokens_translated(text, "<out-dir>")
    # an older tree's sbatch spelled the objects' cluster paths: they translate by the documented rule
    older = text.replace('"$STATIC_R_OBJECT"', '"$SHARED/exp/jailbreak/example_set/static_r_matched/object.pt"').replace(
        '"$ALPHASTEER_OBJECT"', '"$SHARED/exp/jailbreak/example_set/alphasteer_official/object.pt"')
    assert older != text
    assert port.sbatch_tokens_translated(older, "<out-dir>") == translated
    with pytest.raises(ValueError, match="no `scripts/petri_steered_smoke.py smoke`"):
        port.sbatch_smoke_tokens("#!/bin/bash\necho nothing\n")
    with pytest.raises(ValueError, match="launcher prefix changed"):
        port.sbatch_smoke_tokens("python scripts/petri_steered_smoke.py smoke --x y\n")


def test_object_paths_map_cluster_to_volume_by_the_documented_rule(port):
    assert port.cluster_object_to_volume(
        "$SHARED/exp/jailbreak/example_set/static_r_matched/object.pt") == OBJ["static_r_matched"]
    assert port.cluster_object_to_volume(
        "${SHARED}/exp/jailbreak/example_set/alphasteer_official/object.pt") == OBJ["alphasteer_official"]
    with pytest.raises(ValueError):
        port.cluster_object_to_volume("/shared/someone/exp/jailbreak/x/object.pt")
    with pytest.raises(ValueError):
        port.cluster_object_to_volume("$SHARED/exp/other/x/object.pt")


def _isolated_site(tmp_path: Path, body: str = "") -> dict:
    """An environment whose site config is a file of this test's own (empty unless ``body``): the host's is never read."""
    cfg = tmp_path / "site" / "config.env"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(body)
    return {"SISTEER_CONFIG": str(cfg)}


def test_object_on_volume_reads_the_flag_then_the_site_key_and_refuses_by_name(port, tmp_path):
    """The launcher's objects: the flag, else the site key (the environment, then the site config). A cluster path maps to the
    volume, a /vol/ path is taken as it is, and anything else, or nothing at all, refuses by name: there is no default object."""
    key = OBJECT_KEY["static_r_matched"]
    cluster = "$SHARED/exp/jailbreak/example_set/static_r_matched/object.pt"
    env = _isolated_site(tmp_path / "none")
    for name in OBJ:
        with pytest.raises(SystemExit, match=rf"no {name} object: pass {port.OBJECT_FLAGS[name]}, or set {OBJECT_KEY[name]} "):
            port.object_on_volume(name, env=env)
    assert port.object_on_volume("static_r_matched", env={**env, key: cluster}) == OBJ["static_r_matched"]
    assert port.object_on_volume("static_r_matched", env={**env, key: "/vol/elsewhere/object.pt"}) == "/vol/elsewhere/object.pt"
    in_file = _isolated_site(tmp_path / "file", f"{key}='{cluster}'\n")
    assert port.object_on_volume("static_r_matched", env=in_file) == OBJ["static_r_matched"]
    # the flag wins over the key; the environment over the file
    assert port.object_on_volume("static_r_matched", "/vol/given/object.pt", env=in_file) == "/vol/given/object.pt"
    assert port.object_on_volume("static_r_matched", env={**in_file, key: "/vol/env/object.pt"}) == "/vol/env/object.pt"
    with pytest.raises(SystemExit, match=rf"{key}='/data/x/object.pt' names neither a cluster path"):
        port.object_on_volume("static_r_matched", env={**env, key: "/data/x/object.pt"})
    with pytest.raises(SystemExit, match=r"--static-r-object='/data/x/object.pt' names neither"):
        port.object_on_volume("static_r_matched", "/data/x/object.pt", env=env)
    with pytest.raises(SystemExit, match=rf"{key} could not be read"):
        port.object_on_volume("static_r_matched", env={"SISTEER_CONFIG": str(tmp_path / "missing.env")})


# --- the restated gate is the frozen smoke's -------------------------------------


def test_wrapper_pins_equal_the_frozen_smokes_constants(port, smoke):
    assert port.MODEL_HF == smoke.DEFAULT_MODEL
    assert port.MODEL_REV == smoke.DEFAULT_REVISION
    assert float(port.STATIC_R_SCALE) == smoke.DEFAULT_STATIC_R_SCALE
    assert port.RESPONSE_ARM == smoke.DEFAULT_RESPONSE_ARM
    assert port.FIDELITY_BATCH_INVARIANT == smoke.CERTIFIED_BATCH_INVARIANT == port.CERTIFIED_BATCH_INVARIANT
    assert port.UNAVAILABLE_MARKER == smoke.UNAVAILABLE_MARKER
    assert port.FAILURE_MARKER == smoke.FAILURE_MARKER
    assert port.SMOKE_RECORD == smoke.SMOKE_RECORD
    assert port.EVIDENCE_FILE == smoke.EVIDENCE_FILE
    assert port.PHASE_CERTIFIED == smoke.PHASE_CERTIFIED
    assert port.PHASE_PETRI == smoke.PHASE_PETRI
    assert port.OBJECT_KEYS == {"static_r_matched": smoke.STATIC_R_OBJECT_KEY,
                                "alphasteer_official": smoke.ALPHASTEER_OBJECT_KEY}
    assert port.REQUIRED_ARMS == ("static_r_matched", "alphasteer_official")
    # THE GATE'S OWN CONSTANTS, read off the frozen script
    assert port.BI0_EXPECTED_FAILURES == tuple(smoke.BI0_EXPECTED_FAILURES)
    assert port.INSTRUMENT_FAILURE_MARKS == tuple(smoke._INSTRUMENT_FAILURE_MARKS)
    assert port.GEOMETRIES == {"fidelity_2048": (smoke.MAX_MODEL_LEN, smoke.MAX_NUM_SEQS),
                               "fidelity_petri_16384": (smoke.PETRI_MAX_MODEL_LEN, smoke.PETRI_MAX_NUM_SEQS)}
    assert port.GEOMETRIES == {"fidelity_2048": (2048, 32), "fidelity_petri_16384": (16384, 24)}
    assert set(port.NON_WAIVABLE_CHECKS) == set(port.FIDELITY_CHECKS) - set(port.BI0_EXPECTED_FAILURES)
    assert port.EXIT_STATUS_GOVERNED_BY == (smoke.PHASE_CERTIFIED, smoke.PHASE_PETRI)


def test_the_frozen_smoke_takes_its_objects_from_the_flags_or_the_site_and_refuses_by_name(smoke, monkeypatch, tmp_path,
                                                                                            capsys):
    """The smoke names no object of its own: each comes from its flag, else its site key (the environment, then the site
    config), with $SHARED expanded against the job's root; nothing named, or a root that does not resolve, refuses by name
    (argparse's exit 2) before the run starts."""
    seen = []
    monkeypatch.setattr(smoke, "cmd_smoke", lambda args: seen.append((args.static_r_object, args.alphasteer_object)) or 0)
    for k, v in _isolated_site(tmp_path).items():
        monkeypatch.setenv(k, v)
    for key in (smoke.STATIC_R_OBJECT_KEY, smoke.ALPHASTEER_OBJECT_KEY, "SISTEER_SHARED_ROOT", "CLUSTER_SHARED_ROOT"):
        monkeypatch.delenv(key, raising=False)
    argv = ["smoke", "--manifest", "m.yaml", "--out-dir", str(tmp_path / "out")]
    with pytest.raises(SystemExit) as e:
        smoke.main(argv)
    assert e.value.code == 2 and seen == []
    assert f"--static-r-object is not given and {smoke.STATIC_R_OBJECT_KEY} is not set" in capsys.readouterr().err
    monkeypatch.setenv(smoke.STATIC_R_OBJECT_KEY, "$SHARED/exp/jailbreak/example_set/static_r_matched/object.pt")
    monkeypatch.setenv(smoke.ALPHASTEER_OBJECT_KEY, "$SHARED/exp/jailbreak/example_set/alphasteer_official/object.pt")
    with pytest.raises(SystemExit) as e:          # the key spells $SHARED and no root resolves it
        smoke.main(argv)
    assert e.value.code == 2 and seen == [] and "SISTEER_SHARED_ROOT" in capsys.readouterr().err
    monkeypatch.setenv("SISTEER_SHARED_ROOT", str(tmp_path / "shared"))
    assert smoke.main(argv) == 0
    assert seen[-1] == (f"{tmp_path}/shared/exp/jailbreak/example_set/static_r_matched/object.pt",
                        f"{tmp_path}/shared/exp/jailbreak/example_set/alphasteer_official/object.pt")
    assert smoke.main(argv + ["--static-r-object", "/data/a/object.pt", "--alphasteer-object", "/data/b/object.pt"]) == 0
    assert seen[-1] == ("/data/a/object.pt", "/data/b/object.pt")
    # the worker subcommands never consult the object keys
    monkeypatch.delenv(smoke.STATIC_R_OBJECT_KEY)
    monkeypatch.setattr(smoke, "cmd_worker_base", lambda args: 0)
    assert smoke.main(["worker-base", "--model", "m", "--revision", "r", "--mode", "eager", "--out", str(tmp_path / "w")]) == 0


def test_check_state_and_split_agree_with_the_smoke(port, smoke):
    for v in ("passed", "indeterminate: engine_determinism invalidated the report", "batch_invariance: 3 rows",
              "selection_honored: no evidence gathered — a missing probe", ""):
        assert port.check_state(v) == smoke.check_state(v)
    failures = ["batch_invariance: 3 of 15 rows differ from their same-size reference",
                "selection_honored: output half: 2 rows differ",
                "selection_honored: no evidence gathered — a missing probe is not a passed check",
                "engine_determinism: run2 differs at token 7",
                "evidence: unknown evidence key(s) ['x'] — allowed keys ..."]
    for bi in ("0", "1"):
        assert port.split_gating_failures(failures, batch_invariant=bi) == \
            tuple(smoke.split_gating_failures(failures, batch_invariant=bi))
    gating, waived = port.split_gating_failures(failures, batch_invariant="0")
    assert waived == failures[:2]
    assert gating == failures[2:]        # the instrument failure gates under a waivable name


def test_fidelity_check_names_match_the_committed_evaluator(port):
    sys.path.insert(0, str(REPO / "src"))
    try:
        from sisteer.capeval.serving import fidelity
    finally:
        sys.path.remove(str(REPO / "src"))
    tuples = [v for v in vars(fidelity).values()
              if isinstance(v, (tuple, list)) and set(v) == set(port.FIDELITY_CHECKS)]
    assert tuples, "fidelity.py no longer names the five checks the wrapper demands"
    assert tuple(tuples[0]) == port.FIDELITY_CHECKS
    src = (REPO / "src/sisteer/capeval/serving/fidelity.py").read_text()
    for mark in port.INSTRUMENT_FAILURE_MARKS:
        assert mark in src, f"fidelity.py no longer emits {mark!r}"


#: a small table of kv_cache blocks (the smoke's legacy fact names) — uniform,
#: hybrid, unreadable, malformed — for the authoritative-verdict equality test
_KV_TABLE = [
    ("uniform covered 24-way", {"kv_num_cache_groups": 1, "num_gpu_blocks": 35874, "block_size": 16,
                                "kv_cache_groups": [{"spec": "FullAttentionSpec", "block_size": 16}]}, 24),
    ("uniform 32-way shortfall (job 230089 @0.95)", {"kv_num_cache_groups": 1, "num_gpu_blocks": 32546,
                                                     "block_size": 16,
                                                     "kv_cache_groups": [{"spec": "FullAttentionSpec"}]}, 32),
    ("uniform blocks unreadable", {"kv_num_cache_groups": 1, "num_gpu_blocks": None, "block_size": 16}, 24),
    ("no group count", {"num_gpu_blocks": 35874, "block_size": 16}, 24),
    ("bool group count", {"kv_num_cache_groups": True, "num_gpu_blocks": 1, "block_size": 16}, 1),
    ("hybrid legit", {"kv_num_cache_groups": 2, "kv_max_concurrency": 30.5, "num_gpu_blocks": 10, "block_size": 16,
                      "kv_cache_groups": [{"spec": "FullAttentionSpec", "block_size": 16, "num_layers": 8},
                                          {"spec": "MambaSpec", "block_size": 16, "num_layers": 24}]}, 24),
    ("hybrid short", {"kv_num_cache_groups": 2, "kv_max_concurrency": 20.0, "num_gpu_blocks": 10, "block_size": 16,
                      "kv_cache_groups": [{"spec": "FullAttentionSpec"}, {"spec": "MambaSpec"}]}, 24),
    ("hybrid MALFORMED groups, right length, ample concurrency",
     {"kv_num_cache_groups": 2, "kv_max_concurrency": 30.5, "num_gpu_blocks": 10, "block_size": 16,
      "kv_cache_groups": [{}, {}]}, 24),
    ("hybrid groups without spec", {"kv_num_cache_groups": 2, "kv_max_concurrency": 30.5,
                                    "kv_cache_groups": [{"block_size": 16}, {"block_size": 16}]}, 24),
    ("hybrid wrong length", {"kv_num_cache_groups": 2, "kv_max_concurrency": 30.5,
                             "kv_cache_groups": [{"spec": "FullAttentionSpec"}]}, 24),
    ("hybrid inf concurrency", {"kv_num_cache_groups": 2, "kv_max_concurrency": float("inf"),
                                "kv_cache_groups": [{"spec": "A"}, {"spec": "B"}]}, 24),
]


def test_kv_verdict_is_the_authoritative_function(port, shared_kv_verdict):
    """The wrapper IMPORTS vllm_server.kv_coverage_problem (the smoke's adapter
    over the record's legacy fact names) — its verdict equals the authoritative
    one on every row, including the malformed-groups row a replica once
    accepted."""
    assert port.authoritative_kv_coverage_problem(_KV_TABLE[0][1], max_num_seqs=24, max_model_len=16384) is None
    assert port.authoritative_kv_coverage_problem(_KV_TABLE[5][1], max_num_seqs=24, max_model_len=16384) is None
    bad = port.authoritative_kv_coverage_problem(_KV_TABLE[7][1], max_num_seqs=24, max_model_len=16384)
    assert bad and "per-group structure could not be read" in bad, bad           # [{}, {}] -> FAIL
    short = port.authoritative_kv_coverage_problem(_KV_TABLE[1][1], max_num_seqs=32, max_model_len=16384)
    assert "KV cache holds 520736 tokens < the 524288-token budget (= 32 x 16384)" in short

    def adapt(kv):
        rep = dict(kv)
        rep["kv_num_gpu_blocks"] = kv.get("num_gpu_blocks")
        rep["kv_block_size"] = kv.get("block_size")
        return rep
    for name, kv, seqs in _KV_TABLE:
        ours = port.authoritative_kv_coverage_problem(kv, max_num_seqs=seqs, max_model_len=16384)
        theirs = shared_kv_verdict(adapt(kv), max_num_seqs=seqs, max_model_len=16384)
        assert ours == theirs, (name, ours, theirs)
    # no replica of the verdict's logic lives in the wrapper any more
    src = (REPO / "scripts/modal/petri_smoke.py").read_text()
    assert "def kv_coverage_problem" not in src
    assert "from sisteer.capeval.serving.vllm_server import kv_coverage_problem" in src


def test_kv_verdict_fails_closed_when_the_authoritative_function_is_unavailable(port, monkeypatch):
    def boom():
        raise ImportError("no sisteer on this path")
    monkeypatch.setattr(port, "_import_kv_verdict", boom)
    prob = port.authoritative_kv_coverage_problem(_KV_TABLE[0][1], max_num_seqs=24, max_model_len=16384)
    assert prob and "authoritative KV-coverage verdict unavailable" in prob and "does not pass" in prob
    assert any("verdict unavailable" in p for p in _p(port))      # a whole record fails on it
    assert str(REPO / "src") in port.SISTEER_SRC_CANDIDATES


# --- the environment -----------------------------------------------------------


def test_env_translation_drops_cluster_vars_and_sets_the_volume_cache(port):
    base = {k: "inherited" for k in port.SCRUB_ENV}
    base.update(KEEP="yes", CUDA_VISIBLE_DEVICES="0", HF_HOME="/somewhere/else")
    env = port.smoke_env(base, "/root/trees/sisteer-abc")
    assert all(k not in env for k in port.SCRUB_ENV)
    assert "NCCL_SOCKET_IFNAME" in port.SCRUB_ENV and "VLLM_PLUGINS" in port.SCRUB_ENV
    assert env["KEEP"] == "yes"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["HF_HOME"] == "/vol/hf"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert env["PYTHONPATH"] == "/root/trees/sisteer-abc/src"


def test_images_are_the_lens_gates_certified_recipe(port):
    assert set(port.IMAGES) == {"tf5141", "tf5150"}
    assert port.DEFAULT_IMAGE == "tf5141"
    assert port.TRANSFORMERS_PIN == {"tf5141": "5.14.1", "tf5150": "5.15.0"}
    assert port.UV_VERSION == "0.12.1"
    assert port.SERVING_EXCLUDE_NEWER == "2026-08-03T00:00:00Z"
    reqs = port.VLLM_REQUIREMENTS.read_text()
    assert "vllm-0.25.1+cu129" in reqs and "vllm-lens==1.2.1" in reqs
    for variant, rel in port.REQS_FILE.items():
        assert (REPO / rel).is_file(), rel
    assert port.SMOKE_TIMEOUT_S == 3600
    assert port.SMOKE_DEADLINE_S < port.SMOKE_TIMEOUT_S
    assert set(port.SMOKE_FN) == set(port.PREFLIGHT) == set(port.IMAGES)


# --- object identity (H2) ----------------------------------------------------------


def test_pinned_object_hashes_are_the_originals_the_record_names(port, real_record):
    for a, h in port.OBJECT_SHA256.items():
        assert len(h) == 64 and int(h, 16) >= 0
    assert port.OBJECT_SHA256["static_r_matched"].startswith("4cc6b82a38c11cfc")
    assert port.OBJECT_SHA256["alphasteer_official"].startswith("2c3477acf7e8dfc0")
    arms = {a["cell_id"]: a for a in real_record["arm_derivations"]}
    for a in port.REQUIRED_ARMS:
        assert arms[a]["original_sha16"] == port.OBJECT_SHA256[a][:16]
    # the ARTIFACT hash, not the served one: static_r is served DERIVED at 0.05
    assert arms["static_r_matched"]["served_sha16"] != arms["static_r_matched"]["original_sha16"]
    assert arms["alphasteer_official"]["served_sha16"] == arms["alphasteer_official"]["original_sha16"]


def test_verify_object_accepts_only_the_pinned_bytes(port, monkeypatch, tmp_path):
    good = tmp_path / "static_r_matched" / "object.pt"
    good.parent.mkdir()
    good.write_bytes(b"the real static_r_matched artifact bytes")
    wrong = tmp_path / "lookalike.pt"
    wrong.write_bytes(b"a same-kind static_r object fitted on other data")   # same kind, different bytes
    monkeypatch.setitem(port.OBJECT_SHA256, "static_r_matched", hashlib.sha256(good.read_bytes()).hexdigest())
    got = port.verify_object("static_r_matched", str(good))
    assert got == {"path": str(good), "size": good.stat().st_size, "sha256": port.OBJECT_SHA256["static_r_matched"]}
    with pytest.raises(AssertionError, match="hashes to .* not the pinned original .* refusing"):
        port.verify_object("static_r_matched", str(wrong))
    with pytest.raises(AssertionError, match=r"absent at .* \(the site names it by SISTEER_PETRI_SMOKE_STATIC_R_OBJECT\)"):
        port.verify_object("static_r_matched", str(tmp_path / "nope" / "object.pt"))
    empty = tmp_path / "empty.pt"
    empty.write_bytes(b"")
    with pytest.raises(AssertionError, match="absent"):
        port.verify_object("static_r_matched", str(empty))
    assert port.sha256_file(good) == port.OBJECT_SHA256["static_r_matched"]


def test_a_wrong_object_in_the_record_fails_classification(port, real_record, tmp_path):
    doc = copy.deepcopy(real_record)
    for a in doc["arm_derivations"]:
        if a["cell_id"] == "alphasteer_official":
            a["original_sha16"] = a["served_sha16"] = "9c25cdf2e1251dc3"     # a same-kind alphasteer, other bytes
    problems = port.positive_artifact_problems(doc, expected_sha=REAL_SHA)
    assert any("alphasteer_official" in p and "DIFFERENT object was served" in p for p in problems), problems
    _write(tmp_path, "smoke_record.json", doc)
    _write(tmp_path, "smoke_evidence.json", {})
    assert port.classify_outcome(str(tmp_path), 0, expected_sha=REAL_SHA)["verdict"] == "FAIL"
    doc = copy.deepcopy(real_record)
    next(a for a in doc["arm_derivations"] if a["cell_id"] == "static_r_matched")["served_base_scale"] = 1.0
    assert any("static_r_matched: not served DERIVED at the deployment lambda" in p
               for p in port.arm_problems(doc))
    doc = copy.deepcopy(real_record)
    next(a for a in doc["arm_derivations"] if a["cell_id"] == "alphasteer_official")["deploy_scale"] = -0.3
    assert any("alphasteer_official: not served at its stored base_scale" in p for p in port.arm_problems(doc))


# --- the outcome: synthetic shapes ----------------------------------------------

SHA = "1" * 40
FIVE = ("engine_determinism", "scale0_noop", "steering_bites", "batch_invariance", "selection_honored")
SR16 = "4cc6b82a38c11cfc"
AL16 = "2c3477acf7e8dfc0"


def _phase(bi: str = "0", gating: bool = True) -> dict:
    return {"outcome": "evaluated", "gating": gating, "certifies": gating, "governs_exit_status": gating,
            "requested_batch_invariant": bi, "resolved_batch_invariant": bi, "batch_invariant_as_requested": True,
            "is_the_certified_configuration": gating and bi == "0", "engine_reached_ready": True, "child_exit": 0,
            "invalidated": False, "passed": True, "check_states": {c: "passed" for c in FIVE},
            "failed_checks": [], "expected_failures": [], "unexpected_failures": [],
            "waived_failures": [] if gating else None,
            "selection_direct": {"verdict": "PASS", "n_rows": 15, "n_mismatched": 0}}


def _geom(mml: int, mns: int, util: float, n_blocks: int) -> dict:
    tokens = n_blocks * 16
    return {"gpu_memory_utilization": util, "max_model_len": mml, "max_num_seqs": mns,
            "max_num_batched_tokens": mml * mns,
            "kv_cache": {"block_size": 16, "num_gpu_blocks": n_blocks, "kv_num_cache_groups": 1,
                         "kv_cache_groups": [{"block_size": 16, "num_layers": 32, "page_size_bytes": 65536,
                                              "spec": "FullAttentionSpec"}],
                         "kv_cache_tokens": tokens, "full_length_concurrency": round(tokens / mml, 2),
                         "kv_max_concurrency": tokens / mml, "covers_full_budget": tokens >= mml * mns}}


def _complete_record(**over) -> dict:
    """A synthetic smoke_record.json carrying every positive artifact (the
    real record's shape), base comparator legs measured too."""
    rep = {"passed": True, "invalidated": False, "failures": [], "checks": {c: "passed" for c in FIVE},
           "diagnostics": {}}
    leg = {"tokens_per_s": 1234.5, "prompts_per_min": 60.0}
    doc = {
        "status": "completed",
        "fidelity": copy.deepcopy(rep),
        "fidelity_petri_geometry": copy.deepcopy(rep),
        "fidelity_waived_failures": [],
        "fidelity_petri_geometry_waived_failures": [],
        "fidelity_phases": {
            "gating_phase_key": "bi0_certified", "certified_phase_key": "bi0_certified",
            "petri_geometry_phase_key": "bi0_certified_petri16k", "certified_petri_phase_key": "bi0_certified_petri16k",
            "exit_status_governed_by": ["bi0_certified", "bi0_certified_petri16k"],
            "expected_failures_at_bi0": ["batch_invariance", "selection_honored"],
            "phases": {"bi0_certified": _phase(), "bi0_certified_petri16k": _phase(),
                       "bi1_probe": _phase(bi="1", gating=False)},
        },
        "certified_geometries": {"fidelity_2048": _geom(2048, 32, 0.85, 49580),
                                 "fidelity_petri_16384": _geom(16384, 24, 0.9, 35874)},
        "arm_derivations": [
            {"cell_id": "static_r_matched", "kind": "static_r", "required": True, "derived": True,
             "deploy_scale": 0.05, "original_base_scale": 1.0, "served_base_scale": 0.05,
             "original_sha16": SR16, "served_sha16": "d93e5b5941e35730"},
            {"cell_id": "alphasteer_official", "kind": "alphasteer", "required": True, "derived": False,
             "deploy_scale": None, "original_base_scale": 1.0, "served_base_scale": 1.0,
             "original_sha16": AL16, "served_sha16": AL16},
            {"cell_id": "alphasteer_official_scale-0.3", "kind": "alphasteer", "required": False, "derived": True,
             "deploy_scale": -0.3, "original_base_scale": 1.0, "served_base_scale": -0.3,
             "original_sha16": AL16, "served_sha16": "9c25cdf2e1251dc3"},
        ],
        "selection_direct": {"verdict": "PASS"},
        "selection_direct_petri_geometry": {"verdict": "PASS"},
        "throughput": {
            "lens": {"lens_unselected": dict(leg), "lens_steered__static_r_matched": dict(leg),
                     "lens_steered__alphasteer_official": dict(leg)},
            "base_graphs_prefix_cache": dict(leg),
            "base_eager_no_prefix_cache": dict(leg),
        },
        "response_mode_arm": {"requested": "on", "served_arm_ids": ["static_r_matched_response"]},
        "provenance": {"checkout_sha": SHA, "served_model": "meta-llama/Llama-3.1-8B-Instruct",
                       "served_revision": "0e9e39f249a16976918f6564b8830bc894c89659"},
        "chat_markers_shim": False,
    }
    doc.update(over)
    return doc


def _write(out: Path, name: str, doc) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(doc))


def _p(port, **over) -> list:
    return port.positive_artifact_problems(_complete_record(**over), expected_sha=SHA)


def test_unavailable_marker_with_exit_zero_is_a_failure(port, tmp_path):
    _write(tmp_path, "smoke_unavailable.json",
           {"status": "smoke unavailable", "reason": "arm 'static_r_matched': no object file", "stage": "objects"})
    _write(tmp_path, "smoke_evidence.json", {"provenance": {}})
    res = port.classify_outcome(str(tmp_path), 0, expected_sha=SHA)
    assert res["verdict"] == "UNAVAILABLE" and res["ok"] is False
    assert "no object file" in res["reason"] and "NOT a pass" in res["reason"]
    assert port.modal_exit_code({"verdict": "UNAVAILABLE", "status": "DONE"}) == 1


def test_failed_marker_nonzero_exit_and_silent_exit_are_failures(port, tmp_path):
    a = tmp_path / "a"
    _write(a, "smoke_failed.json", {"status": "smoke FAILED (steered serving did not hold)",
                                    "reason": "steering_bites: static_r_matched changed nothing"})
    r = port.classify_outcome(str(a), 1, expected_sha=SHA)
    assert r["verdict"] == "FAIL" and "changed nothing" in r["reason"]
    b = tmp_path / "b"
    b.mkdir()
    r = port.classify_outcome(str(b), 137, expected_sha=SHA)
    assert r["verdict"] == "FAIL" and "exit 137" in r["reason"]
    c = tmp_path / "c"
    c.mkdir()
    r = port.classify_outcome(str(c), 0, expected_sha=SHA)      # the 50-second COMPLETED shape
    assert r["verdict"] == "FAIL" and "no smoke_record.json" in r["reason"]
    d = tmp_path / "d"
    _write(d, "smoke_record.json", _complete_record())
    _write(d, "smoke_evidence.json", {})
    r = port.classify_outcome(str(d), 0, timed_out=True, expected_sha=SHA)
    assert r["verdict"] == "TIMEOUT" and r["ok"] is False


def test_complete_record_passes_and_each_positive_artifact_is_demanded(port, tmp_path):
    good = tmp_path / "good"
    _write(good, "smoke_record.json", _complete_record())
    _write(good, "smoke_evidence.json", {"provenance": {}})
    res = port.classify_outcome(str(good), 0, expected_sha=SHA)
    assert res["verdict"] == "PASS" and res["ok"] is True and res["problems"] == []
    assert res["advisory_missing"] == {} and "every advisory leg measured" in res["reason"]
    assert res["positives"]["petri16k_kv_cache_tokens"] == 573984
    assert res["positives"]["tokens_per_s"]["lens_steered__alphasteer_official"] == 1234.5
    assert port.modal_exit_code({"verdict": "PASS", "status": "DONE"}) == 1   # a PASS not yet PUBLISHED

    def problems(**over):
        return port.positive_artifact_problems(_complete_record(**over), expected_sha=SHA)

    assert problems() == []
    assert any("status" in p for p in problems(status="running"))
    rep = _complete_record()["fidelity"]
    rep["checks"]["engine_determinism"] = "engine_determinism: run 2 differs at token 7"
    rep["failures"] = [rep["checks"]["engine_determinism"]]
    rep["passed"] = False
    assert any("engine_determinism" in p and "gates at any BI" in p for p in problems(fidelity=rep))
    rep = _complete_record()["fidelity_petri_geometry"]
    del rep["checks"]["selection_honored"]
    assert any("lacks" in p for p in problems(fidelity_petri_geometry=rep))
    rep = _complete_record()["fidelity"]
    rep["invalidated"] = "engine_determinism"
    assert any("invalidated" in p for p in problems(fidelity=rep))
    ph = _complete_record()["fidelity_phases"]
    ph["phases"]["bi0_certified_petri16k"]["outcome"] = "MISSING: the lens-active engine could not init"
    assert any("bi0_certified_petri16k" in p for p in problems(fidelity_phases=ph))
    ph = _complete_record()["fidelity_phases"]
    ph["gating_phase_key"] = "bi1_gating_NOT_certified"
    assert any("gating_phase_key" in p for p in problems(fidelity_phases=ph))
    assert any("certified_geometries.fidelity_petri_16384: missing" in p for p in problems(certified_geometries={}))
    arms = [a for a in _complete_record()["arm_derivations"] if a["cell_id"] != "alphasteer_official"]
    assert any("alphasteer_official" in p for p in problems(arm_derivations=arms))
    tp = _complete_record()["throughput"]
    tp["lens"]["lens_steered__static_r_matched"] = {"missing": "MISSING: not measured"}
    assert any("lens_steered__static_r_matched" in p for p in problems(throughput=tp))
    prov = dict(_complete_record()["provenance"], checkout_sha="2" * 40)
    assert any("checkout_sha" in p for p in problems(provenance=prov))
    prov = dict(_complete_record()["provenance"], served_model="NousResearch/Meta-Llama-3.1-8B-Instruct")
    assert any("served" in p for p in problems(provenance=prov))


# --- H1: the COMPLETE gate, one negative per predicate ------------------------------


def test_gate_waives_only_the_two_composition_checks_and_records_the_waiver(port):
    text = "batch_invariance: 3 of 15 rows differ from their same-size reference (first divergence at token 12)"
    rep = _complete_record()["fidelity"]
    rep["checks"]["batch_invariance"] = text
    rep["failures"] = [text]
    rep["passed"] = False
    ph = _complete_record()["fidelity_phases"]
    ph["phases"]["bi0_certified"]["check_states"]["batch_invariance"] = "failed"
    ph["phases"]["bi0_certified"]["failed_checks"] = ["batch_invariance"]
    ph["phases"]["bi0_certified"]["expected_failures"] = ["batch_invariance"]
    ph["phases"]["bi0_certified"]["waived_failures"] = [text]
    ph["phases"]["bi0_certified"]["passed"] = False
    assert _p(port, fidelity=rep, fidelity_waived_failures=[text], fidelity_phases=ph) == []
    assert any("did not record as waived" in p for p in _p(port, fidelity=rep, fidelity_waived_failures=[],
                                                              fidelity_phases=ph))
    rep2 = _complete_record()["fidelity"]
    rep2["checks"]["scale0_noop"] = "scale0_noop: 2 rows differ from unsteered"
    rep2["failures"] = [rep2["checks"]["scale0_noop"]]
    rep2["passed"] = False
    probs = _p(port, fidelity=rep2, fidelity_waived_failures=[rep2["checks"]["scale0_noop"]])
    assert any("scale0_noop" in p and "gates at any BI" in p for p in probs)
    assert any("failures gating at BI=0" in p for p in probs)
    rep3 = _complete_record()["fidelity"]
    rep3["failures"] = [text]
    rep3["checks"]["batch_invariance"] = text
    assert any("disagrees" in p for p in _p(port, fidelity=rep3, fidelity_waived_failures=[text]))
    rep4 = _complete_record()["fidelity"]
    rep4["checks"]["selection_honored"] = "selection_honored: output half: 1 row differs"
    assert any("no failure string names it" in p for p in _p(port, fidelity=rep4))


def test_gate_never_waives_an_instrument_failure_under_any_name(port):
    text = ("selection_honored: no evidence gathered — a missing probe is not a passed check "
            "(smoke_evidence lacks selection_honored)")
    rep = _complete_record()["fidelity_petri_geometry"]
    rep["checks"]["selection_honored"] = text
    rep["failures"] = [text]
    rep["passed"] = False
    probs = _p(port, fidelity_petri_geometry=rep, fidelity_petri_geometry_waived_failures=[text])
    assert any("INSTRUMENT failure, never waivable" in p for p in probs), probs
    assert any("failures gating at BI=0" in p for p in probs)
    for mark in port.INSTRUMENT_FAILURE_MARKS:
        t = f"batch_invariance: {mark} (something)"
        r = _complete_record()["fidelity"]
        r["checks"]["batch_invariance"] = t
        r["failures"] = [t]
        r["passed"] = False
        assert any("INSTRUMENT" in p for p in _p(port, fidelity=r, fidelity_waived_failures=[t])), mark
    r = _complete_record()["fidelity"]
    r["failures"] = ["evidence: unknown evidence key(s) ['extra'] — allowed keys are the five checks"]
    r["passed"] = False
    assert any("failures gating at BI=0" in p for p in _p(port, fidelity=r))
    r = _complete_record()["fidelity"]
    r["checks"]["steering_bites"] = "indeterminate: engine_determinism failed, so this comparison is meaningless"
    assert any("indeterminate" in p for p in _p(port, fidelity=r))


def test_gate_fails_on_selection_direct_failed_in_either_phase_or_at_top_level(port):
    failed = {"verdict": "FAILED", "n_rows": 15, "n_mismatched": 2, "aggregate_n_mismatched": 2,
              "aggregate_n_observations": 30}
    assert any("selection_direct FAILED" in p for p in _p(port, selection_direct=failed))
    assert any("selection_direct_petri_geometry FAILED" in p for p in _p(port, selection_direct_petri_geometry=failed))
    ph = _complete_record()["fidelity_phases"]
    ph["phases"]["bi0_certified_petri16k"]["selection_direct"] = failed
    assert any("bi0_certified_petri16k.selection_direct FAILED" in p for p in _p(port, fidelity_phases=ph))
    missing = {"verdict": "MISSING: not one selected row could be correlated to the worker's record"}
    assert _p(port, selection_direct=missing) == []
    assert port.advisory_missing(_complete_record(selection_direct=missing))["selection_direct"].startswith("MISSING")
    assert _p(port, selection_direct_petri_geometry={"verdict": "PARTIAL: 3 of 15 rows unobserved"}) == []


def test_gate_requires_full_kv_coverage_at_both_geometries_not_tokens_gt_zero(port):
    geoms = _complete_record()["certified_geometries"]
    geoms["fidelity_petri_16384"]["kv_cache"]["covers_full_budget"] = False
    assert any("covers_full_budget=False" in p for p in _p(port, certified_geometries=geoms))
    geoms = _complete_record()["certified_geometries"]
    geoms["fidelity_petri_16384"]["kv_cache"].update(num_gpu_blocks=1, block_size=1, kv_cache_tokens=1,
                                                     full_length_concurrency=0.0, kv_max_concurrency=0.0,
                                                     covers_full_budget=True)
    probs = _p(port, certified_geometries=geoms)
    assert any("KV cache holds 1 tokens < the 393216-token budget" in p for p in probs), probs
    assert any("full_length_concurrency=0.0 < 24" in p for p in probs)
    assert any("kv_cache_tokens=1 < 393216" in p for p in probs)
    geoms = _complete_record()["certified_geometries"]
    geoms["fidelity_petri_16384"]["max_num_seqs"] = 32
    geoms["fidelity_petri_16384"]["max_num_batched_tokens"] = 32 * 16384
    assert any("recorded 16384x32 != 16384x24" in p for p in _p(port, certified_geometries=geoms))
    geoms = _complete_record()["certified_geometries"]
    geoms["fidelity_2048"]["kv_cache"].update(num_gpu_blocks=100, kv_cache_tokens=1600,
                                              full_length_concurrency=0.78, covers_full_budget=False)
    probs = _p(port, certified_geometries=geoms)
    assert any("fidelity_2048.kv_cache: KV cache holds 1600 tokens < the 65536-token budget" in p for p in probs)
    geoms = _complete_record()["certified_geometries"]
    geoms["fidelity_2048"]["max_num_batched_tokens"] = 8192
    assert any("chunked prefill could bind" in p for p in _p(port, certified_geometries=geoms))
    geoms = _complete_record()["certified_geometries"]
    geoms["fidelity_petri_16384"]["kv_cache"]["full_length_concurrency"] = 12.0
    assert any("full_length_concurrency=12.0 < 24" in p for p in _p(port, certified_geometries=geoms))
    geoms = _complete_record()["certified_geometries"]
    del geoms["fidelity_petri_16384"]["kv_cache"]["kv_num_cache_groups"]
    assert any("could not be read" in p for p in _p(port, certified_geometries=geoms))
    # a hybrid block with the declared group count but MALFORMED entries: the
    # authoritative verdict refuses it (a replica once accepted it)
    geoms = _complete_record()["certified_geometries"]
    geoms["fidelity_petri_16384"]["kv_cache"].update(kv_num_cache_groups=2, kv_cache_groups=[{}, {}],
                                                     kv_max_concurrency=30.0)
    assert any("per-group structure could not be read" in p for p in _p(port, certified_geometries=geoms))


def test_gate_validates_every_phase_flag_including_resolved_bi_and_exit_governance(port):
    def with_phase(pk="bi0_certified", **fields):
        ph = _complete_record()["fidelity_phases"]
        ph["phases"][pk].update(fields)
        return _p(port, fidelity_phases=ph)
    assert any("resolved_batch_invariant='1'" in p for p in with_phase(resolved_batch_invariant="1"))
    assert any("requested_batch_invariant='1'" in p for p in with_phase(requested_batch_invariant="1"))
    assert any("batch_invariant_as_requested=False" in p for p in with_phase(batch_invariant_as_requested=False))
    assert any("governs_exit_status=False" in p for p in with_phase("bi0_certified_petri16k", governs_exit_status=False))
    assert any("is_the_certified_configuration=False" in p for p in with_phase(is_the_certified_configuration=False))
    assert any("certifies=False" in p for p in with_phase(certifies=False))
    assert any("gating=False" in p for p in with_phase(gating=False))
    assert any("engine_reached_ready=False" in p for p in with_phase(engine_reached_ready=False))
    assert any("child_exit=3" in p for p in with_phase(child_exit=3))
    assert any("invalidated=True" in p for p in with_phase(invalidated=True))
    assert any("unexpected_failures=['scale0_noop']" in p for p in with_phase(unexpected_failures=["scale0_noop"]))
    assert any("check_states.steering_bites='failed'" in p
               for p in with_phase(check_states={**{c: "passed" for c in FIVE}, "steering_bites": "failed"}))
    assert any("check_states lacks" in p for p in with_phase(check_states={"engine_determinism": "passed"}))
    assert any("outcome='MISSING: not reached'" in p for p in with_phase(outcome="MISSING: not reached"))
    ph = _complete_record()["fidelity_phases"]
    del ph["phases"]["bi0_certified_petri16k"]
    assert any("bi0_certified_petri16k: missing" in p for p in _p(port, fidelity_phases=ph))
    ph = _complete_record()["fidelity_phases"]
    ph["exit_status_governed_by"] = ["bi0_certified"]
    assert any("exit_status_governed_by=['bi0_certified']" in p for p in _p(port, fidelity_phases=ph))
    for key in ("certified_phase_key", "petri_geometry_phase_key", "certified_petri_phase_key"):
        ph = _complete_record()["fidelity_phases"]
        ph[key] = "bi1_probe"
        assert any(f"fidelity_phases.{key}='bi1_probe'" in p for p in _p(port, fidelity_phases=ph)), key


def test_base_comparator_legs_are_advisory_not_gating(port, tmp_path):
    tp = _complete_record()["throughput"]
    reason = "MISSING: comparator child exit 4, stage engine_init: comparator config assertions failed"
    tp["base_graphs_prefix_cache"] = {"missing": reason}
    del tp["base_eager_no_prefix_cache"]
    doc = _complete_record(throughput=tp)
    assert port.positive_artifact_problems(doc, expected_sha=SHA) == []
    adv = port.advisory_missing(doc)
    assert adv == {"throughput.base_graphs_prefix_cache": reason,
                   "throughput.base_eager_no_prefix_cache": "MISSING: no leg recorded"}
    _write(tmp_path, "smoke_record.json", doc)
    _write(tmp_path, "smoke_evidence.json", {})
    res = port.classify_outcome(str(tmp_path), 0, expected_sha=SHA)
    assert res["verdict"] == "PASS" and res["advisory_missing"] == adv
    assert "advisory legs MISSING (evidence only, not gating)" in res["reason"]
    line = port.result_line({"verdict": "PASS", "status": "DONE", "smoke_rc": 0, "elapsed_s": 1.0,
                             "app": "petri-smoke-1111111", "sha": SHA, "image": "tf5141",
                             "reason": res["reason"], "advisory_missing": adv})
    assert line.startswith("PASS exit=0 ") and "advisory_missing=" in line and reason in line
    assert set(port.ADVISORY_ARTIFACTS) == {f"throughput.{leg}" for leg in port.ADVISORY_BASE_LEGS}


def test_record_without_evidence_file_is_incomplete(port, tmp_path):
    _write(tmp_path, "smoke_record.json", _complete_record())
    res = port.classify_outcome(str(tmp_path), 0, expected_sha=SHA)
    assert res["verdict"] == "FAIL" and "smoke_evidence.json missing" in res["reason"]


def test_positive_artifacts_table_names_what_the_code_demands(port):
    table = port.POSITIVE_ARTIFACTS
    assert set(table) == {"smoke_record.json", "smoke_evidence.json"}
    rec_keys = table["smoke_record.json"]
    for key in ("status", "fidelity", "fidelity_petri_geometry", "fidelity_phases", "arm_derivations",
                "throughput.lens", "provenance", "certified_geometries"):
        assert key in rec_keys
    assert any("selection_direct" in k for k in rec_keys)
    joined = " ".join(str(v) for v in rec_keys.values())
    for needle in ("covers_full_budget", "kv_cache_tokens >= seqs x len", "no evidence gathered",
                   "RESOLVED VLLM_BATCH_INVARIANT", "governs_exit_status", "OBJECT_SHA256"):
        assert needle in joined, needle
    for c in port.NON_WAIVABLE_CHECKS:
        assert c in rec_keys["fidelity"]
    assert not any("base_" in k for k in rec_keys), "base comparator rungs must not be positive artifacts"
    assert port.LENS_LEGS_REQUIRED == ("lens_unselected", "lens_steered__static_r_matched",
                                       "lens_steered__alphasteer_official")
    assert port.ADVISORY_BASE_LEGS == ("base_graphs_prefix_cache", "base_eager_no_prefix_cache")


def test_result_line_and_exit_code(port):
    rec = {"verdict": "FAIL", "status": "DONE", "smoke_rc": 1, "elapsed_s": 900.2, "app": "petri-smoke-1111111",
           "sha": SHA, "image": "tf5141", "reason": "smoke_failed.json: x"}
    line = port.result_line(rec)
    assert line.startswith("FAIL exit=1 ") and line.endswith("\n") and f"sha={SHA}" in line
    assert "advisory_missing=" not in line
    assert port.modal_exit_code(rec) == 1
    assert port.modal_exit_code({"verdict": "PASS", "status": "TIMEOUT"}) == 1
    assert port.modal_exit_code({"status": "SETUP_FAILED", "verdict": "SETUP_FAILED"}) == 1
    assert port.modal_exit_code({"status": "ATTEMPT_DIR_EXISTS", "verdict": "ATTEMPT_DIR_EXISTS"}) == 1
    assert port.modal_exit_code({"status": "PUBLISH_FAILED", "verdict": "PUBLISH_FAILED", "smoke_verdict": "PASS",
                                 "attempt": "a", "current": {"attempt": "a"}}) == 1
    assert port.modal_exit_code({"verdict": "PASS", "status": "DONE"}) == 1   # a PASS not yet PUBLISHED
    published = {"verdict": "PASS", "status": "DONE", "attempt": "1700000000-1-aaaaaaaa", "publication_ok": True,
                 "current": {"attempt": "1700000000-1-aaaaaaaa"}}
    assert port.modal_exit_code(published) == 0
    # the RESULT line names the attempt and whether ITS publication completed
    assert " attempt=None published=false reason=" in line
    assert " attempt=1700000000-1-aaaaaaaa published=true reason=" in port.result_line(published)
    assert "published=" not in port.result_line(published, publication=False)
    assert " attempt=1700000000-1-aaaaaaaa reason=" in port.result_line(published, publication=False)


# --- the outcome: the REAL record, replayed -------------------------------------


def test_real_record_is_scrubbed_and_from_the_canary_sha(real_record):
    text = REAL_RECORD.read_text()
    assert "vo-TESTVOLUME" in text and "vo-i96" not in text
    assert real_record["provenance"]["node"] == "TESTHOST"
    assert real_record["provenance"]["checkout_sha"] == REAL_SHA
    assert real_record["provenance"]["served_model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert real_record["status"] == "completed"
    for line in text.splitlines():
        assert "/home/" not in line and not token_hits(line, STORAGE_TOKENS), line


def test_real_record_replays_to_pass_with_base_legs_advisory(port, real_record, tmp_path):
    _write(tmp_path, "smoke_record.json", real_record)
    _write(tmp_path, "smoke_evidence.json", {"provenance": {}})
    res = port.classify_outcome(str(tmp_path), 0, expected_sha=REAL_SHA)
    assert res["verdict"] == "PASS" and res["ok"] is True, res["reason"]
    assert res["problems"] == []
    assert set(res["advisory_missing"]) == {"throughput.base_graphs_prefix_cache",
                                            "throughput.base_eager_no_prefix_cache",
                                            "selection_direct", "selection_direct_petri_geometry"}
    for leg in port.ADVISORY_BASE_LEGS:
        reason = res["advisory_missing"][f"throughput.{leg}"]
        assert reason.startswith("MISSING: comparator child exit 4, stage engine_init: comparator config "
                                 "assertions failed: resolved model '/__modal/volumes/")
        assert "!= configured 'meta-llama/Llama-3.1-8B-Instruct'" in reason
    assert res["advisory_missing"]["selection_direct"].startswith("MISSING: not one selected row")
    assert "advisory legs MISSING (evidence only, not gating)" in res["reason"]
    pos = res["positives"]
    assert pos["fidelity_checks"] == {c: "passed" for c in FIVE}
    assert pos["fidelity_petri_checks"] == {c: "passed" for c in FIVE}
    assert pos["fidelity_passed"] is True and pos["fidelity_petri_passed"] is True
    assert pos["fidelity_waived_failures"] == [] and pos["fidelity_petri_waived_failures"] == []
    assert pos["phase_outcomes"] == {"bi0_certified": "evaluated", "bi0_certified_petri16k": "evaluated",
                                     "bi1_probe": "evaluated"}
    assert pos["petri16k_kv_cache_tokens"] == 573984 and pos["petri16k_full_length_concurrency"] == 35.03
    assert pos["arms"]["static_r_matched"]["required"] is True and pos["arms"]["alphasteer_official"]["required"] is True
    assert pos["tokens_per_s"]["base_graphs_prefix_cache"] is None
    assert pos["tokens_per_s"]["lens_unselected"] > 0
    assert pos["response_mode_arm_served"] == ["static_r_matched_response"]
    assert pos["chat_markers_shim"] is False
    rec = {"verdict": res["verdict"], "status": "DONE", "smoke_rc": 0, "elapsed_s": 940.4,
           "app": "petri-smoke-ad6c3cb", "sha": REAL_SHA, "image": "tf5141", "reason": res["reason"],
           "advisory_missing": res["advisory_missing"]}
    line = port.result_line(rec)
    assert line.startswith("PASS exit=0 ") and "advisory_missing=" in line
    assert "throughput.base_graphs_prefix_cache" in line and "throughput.base_eager_no_prefix_cache" in line
    assert port.modal_exit_code(rec) == 1, "a classified PASS is not exit 0 until its publication completed"
    a = "1700000000-1-aaaaaaaa"
    assert port.modal_exit_code({**rec, "attempt": a, "publication_ok": True, "current": {"attempt": a}}) == 0
    for pk in port.EXIT_STATUS_GOVERNED_BY:
        ph = real_record["fidelity_phases"]["phases"][pk]
        assert ph["resolved_batch_invariant"] == "0" and ph["governs_exit_status"] is True
        assert ph["is_the_certified_configuration"] is True and ph["unexpected_failures"] == []
    for key, (mml, mns) in port.GEOMETRIES.items():
        kv = real_record["certified_geometries"][key]["kv_cache"]
        assert kv["covers_full_budget"] is True and kv["kv_cache_tokens"] >= mml * mns
        assert port.authoritative_kv_coverage_problem(kv, max_num_seqs=mns, max_model_len=mml) is None


def test_lens_throughput_keys_are_pinned_from_the_real_record(port, real_record):
    lens = real_record["throughput"]["lens"]
    for leg in port.LENS_LEGS_REQUIRED:
        assert leg in lens, leg
        assert isinstance(lens[leg]["tokens_per_s"], (int, float)) and lens[leg]["tokens_per_s"] > 0
    assert set(port.LENS_LEGS_REQUIRED) == {"lens_unselected", "lens_steered__static_r_matched",
                                            "lens_steered__alphasteer_official"}
    assert set(lens) == {
        "lens_unselected", "lens_unselected__bi1",
        "lens_steered__static_r_matched", "lens_steered__static_r_matched__bi1",
        "lens_steered__alphasteer_official", "lens_steered__alphasteer_official__bi1",
        "lens_steered__alphasteer_official_scale-0.3", "lens_steered__alphasteer_official_scale-0.3__bi1",
        "lens_steered__static_r_matched_response",
    }
    legacy = real_record["throughput"]["suites"]["legacy_short_distinct"]["legs"]
    assert set(port.LENS_LEGS_REQUIRED) <= set(legacy)
    for leg in port.ADVISORY_BASE_LEGS:
        assert leg in real_record["throughput"] and "missing" in real_record["throughput"][leg]
        assert leg in legacy and "missing" in legacy[leg]
    sp = real_record["throughput"]["suites"]["shared_prefix"]["legs"]
    assert all("missing" in v for k, v in sp.items() if "__base_" in k)
    assert all("tokens_per_s" in v for k, v in sp.items() if "__lens_" in k)


def test_real_record_with_a_gating_check_failed_is_a_failure(port, real_record, tmp_path):
    for key, check in (("fidelity", "engine_determinism"), ("fidelity", "steering_bites"),
                       ("fidelity_petri_geometry", "scale0_noop")):
        doc = copy.deepcopy(real_record)
        doc[key]["checks"][check] = f"{check}: 4 of 15 rows differ"
        doc[key]["failures"] = [doc[key]["checks"][check]]
        doc[key]["passed"] = False
        out = tmp_path / f"{key}-{check}"
        _write(out, "smoke_record.json", doc)
        _write(out, "smoke_evidence.json", {})
        res = port.classify_outcome(str(out), 0, expected_sha=REAL_SHA)
        assert res["verdict"] == "FAIL", (key, check)
        assert any(check in p and key in p for p in res["problems"]), res["problems"]
    doc = copy.deepcopy(real_record)
    doc["fidelity_phases"]["phases"]["bi0_certified_petri16k"]["outcome"] = "MISSING: not reached"
    assert any("bi0_certified_petri16k" in p for p in port.positive_artifact_problems(doc, expected_sha=REAL_SHA))
    assert any("checkout_sha" in p for p in port.positive_artifact_problems(real_record, expected_sha="2" * 40))


# --- evidence on the volume: attempts + current.json ----------------------


def _fill(work: Path, tag: str) -> None:
    (work / "steered_smoke").mkdir(parents=True, exist_ok=True)
    (work / "steered_smoke" / "smoke_record.json").write_text(json.dumps({"status": "completed", "run": tag}))
    (work / "steered_smoke" / "smoke_evidence.json").write_text("{}")
    (work / "smoke.log").write_text(f"log of {tag}\n")


def _snapshot(root: Path) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.name == ".mirror.lock":         # lock plumbing (MIRROR_LOCK), not mirror content
            continue
        rel = str(p.relative_to(root))
        if p.is_symlink():
            out[rel] = ("link", os.readlink(p))
        elif p.is_file():
            out[rel] = ("file", p.read_bytes())
        else:
            out[rel] = ("dir", None)
    return out


def test_attempt_ids_are_collision_proof_and_paths_are_per_attempt(port, monkeypatch, tmp_path):
    ids = {port.attempt_id() for _ in range(2000)}
    assert len(ids) == 2000
    epoch, pid, rnd = next(iter(ids)).split("-")
    assert epoch.isdigit() and pid.isdigit() and len(rnd) == 8 and int(rnd, 16) >= 0
    monkeypatch.setattr(port, "RESULTS_ROOT", "/vol/petri-smoke")
    monkeypatch.setattr(port, "WORK_ROOT", "/root/petri-smoke-work")
    vol_dir, attempt_dir, work_dir = port.attempt_paths(SHA, "1-2-abcd1234")
    assert vol_dir == Path("/vol/petri-smoke") / SHA
    assert attempt_dir == vol_dir / "attempts" / "1-2-abcd1234"
    assert work_dir == Path("/root/petri-smoke-work") / SHA / "1-2-abcd1234"
    assert port.ATTEMPTS_SUBDIR == "attempts" and port.CURRENT_POINTER == "current.json"


def test_two_concurrent_attempts_never_touch_each_other_and_current_names_exactly_one(port, monkeypatch, tmp_path):
    """Two runs on ONE sha, interleaved at every step: each publishes only under
    its own attempts/<id>/; current.json names exactly one attempt; neither
    run's files are changed by the other's publish."""
    monkeypatch.setattr(port, "RESULTS_ROOT", str(tmp_path / "vol"))
    monkeypatch.setattr(port, "WORK_ROOT", str(tmp_path / "work"))
    a, b = "1700000000-11-aaaaaaaa", "1700000000-22-bbbbbbbb"
    vol_dir, adir, awork = port.attempt_paths(SHA, a)
    _, bdir, bwork = port.attempt_paths(SHA, b)
    _fill(awork, "A")
    _fill(bwork, "B")
    adir.mkdir(parents=True, exist_ok=False)
    bdir.mkdir(parents=True, exist_ok=False)
    rec_a = {"sha": SHA, "app": "petri-smoke-1111111", "image": "tf5141", "verdict": "PASS", "status": "DONE",
             "smoke_rc": 0, "elapsed_s": 900.0, "reason": "A ok", "attempt": a}
    rec_b = {"sha": SHA, "app": "petri-smoke-1111111", "image": "tf5141", "verdict": "FAIL", "status": "DONE",
             "smoke_rc": 1, "elapsed_s": 800.0, "reason": "B failed", "attempt": b}
    # interleave: A's periodic sync, B's periodic sync, A's final publish, B's final publish
    port._sync_results(awork, adir)
    port._sync_results(bwork, bdir)
    snap_a_before_b = None
    port._publish(rec_a, awork, adir, vol_dir, a, promote=True)
    snap_a_before_b = _snapshot(adir)
    port._publish(rec_b, bwork, bdir, vol_dir, b, promote=True)
    # A's attempt dir is byte-identical after B published; B's holds only B's files
    assert _snapshot(adir) == snap_a_before_b
    assert (adir / "steered_smoke" / "smoke_record.json").read_text() == json.dumps({"status": "completed", "run": "A"})
    assert (bdir / "steered_smoke" / "smoke_record.json").read_text() == json.dumps({"status": "completed", "run": "B"})
    assert (adir / "smoke.log").read_text() == "log of A\n" and (bdir / "smoke.log").read_text() == "log of B\n"
    # nothing at the top level but attempts/ and current.json; the pointer names exactly one attempt (the last)
    assert sorted(p.name for p in vol_dir.iterdir()) == ["attempts", "current.json"]
    assert sorted(p.name for p in (vol_dir / "attempts").iterdir()) == [a, b]
    ptr = json.loads((vol_dir / "current.json").read_text())
    assert ptr["attempt"] == b and ptr["path"] == f"attempts/{b}" and ptr["verdict"] == "FAIL"
    assert ptr["result"].startswith("FAIL exit=1 ") and rec_b["current"]["attempt"] == b
    assert rec_a["current"]["attempt"] == a               # A's own pointer content, from its promotion
    # both wrapper records carry their own attempt and its RESULT line
    wa = json.loads((adir / "wrapper_record.json").read_text())
    assert wa["attempt"] == a and wa["current"]["attempt"] == a
    assert (adir / "RESULT").read_text().startswith("PASS exit=0 ")
    assert (bdir / "RESULT").read_text().startswith("FAIL exit=1 ")
    assert f" attempt={a} published=true " in (adir / "RESULT").read_text()
    assert f" attempt={b} published=true " in (bdir / "RESULT").read_text()
    assert rec_a["publication_ok"] is True and rec_b["publication_ok"] is True
    assert port.modal_exit_code(rec_a) == 0 and port.modal_exit_code(rec_b) == 1
    # re-promoting A flips the pointer atomically; no temp pointer files linger
    port._promote_current(vol_dir, a, rec_a)
    assert json.loads((vol_dir / "current.json").read_text())["attempt"] == a
    assert not [p for p in vol_dir.iterdir() if p.name.startswith(".current.json.")]


def test_run_smoke_fails_closed_when_the_attempt_dir_already_exists(port, monkeypatch, tmp_path):
    monkeypatch.setattr(port, "RESULTS_ROOT", str(tmp_path / "vol"))
    monkeypatch.setattr(port, "WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setattr(port, "attempt_id", lambda: "fixed-id")
    vol_dir, attempt_dir, _ = port.attempt_paths(SHA, "fixed-id")
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "RESULT").write_text("someone else's attempt\n")
    before = _snapshot(vol_dir)
    rec = port._run_smoke(SHA, "petri-smoke-1111111", "/vol/x.pt", "/vol/y.pt", "tf5141")
    assert rec["status"] == rec["verdict"] == "ATTEMPT_DIR_EXISTS"
    assert port.modal_exit_code(rec) == 1
    assert _snapshot(vol_dir) == before                    # nothing published anywhere
    assert not (tmp_path / "work").exists()
    assert "pins" not in rec and "argv" not in rec and rec["current"] is None
    assert not (vol_dir / "current.json").exists()


def test_publish_without_promotion_leaves_current_alone(port, monkeypatch, tmp_path):
    monkeypatch.setattr(port, "RESULTS_ROOT", str(tmp_path / "vol"))
    monkeypatch.setattr(port, "WORK_ROOT", str(tmp_path / "work"))
    a = "1700000000-1-aaaaaaaa"
    vol_dir, adir, awork = port.attempt_paths(SHA, a)
    adir.mkdir(parents=True)
    rec = {"sha": SHA, "app": "x", "image": "tf5141", "verdict": "SETUP_FAILED", "status": "SETUP_FAILED",
           "smoke_rc": None, "elapsed_s": 3.0, "reason": "REQUIRED object hashes differ", "attempt": a}
    port._publish(rec, awork, adir, vol_dir, a, promote=False)
    assert (adir / "RESULT").read_text().startswith("SETUP_FAILED exit=None ")
    assert f" attempt={a} published=false " in (adir / "RESULT").read_text()
    assert json.loads((adir / "wrapper_record.json").read_text())["attempt"] == a
    assert not (vol_dir / "current.json").exists() and rec.get("current") is None
    assert "publish_error" not in rec and rec["publication_ok"] is False and rec["status"] == "SETUP_FAILED"
    assert port.modal_exit_code(rec) == 1


# --- publication is part of the exit contract ---------------------------------------------


def _completed_pass(a: str) -> dict:
    return {"sha": SHA, "app": "petri-smoke-1111111", "image": "tf5141", "verdict": "PASS", "status": "DONE",
            "smoke_rc": 0, "elapsed_s": 900.0, "reason": "smoke_record.json complete", "attempt": a, "current": None}


def _prepare_attempt(port, monkeypatch, tmp_path, a: str):
    monkeypatch.setattr(port, "RESULTS_ROOT", str(tmp_path / "vol"))
    monkeypatch.setattr(port, "WORK_ROOT", str(tmp_path / "work"))
    vol_dir, adir, awork = port.attempt_paths(SHA, a)
    _fill(awork, "A")
    adir.mkdir(parents=True, exist_ok=False)
    return vol_dir, adir, awork


def test_a_published_pass_exits_zero_and_its_result_says_so(port, monkeypatch, tmp_path):
    """The happy path: the COMPLETE attempt (RESULT published=true attempt=<id>,
    the record carrying the pointer, the evidence) synced + committed and
    byte-verified FIRST — no current.json exists during any sync — then the
    pointer replaced, byte-identical to what the record carries, and read back.
    Only the record carrying every one of those facts is exit 0."""
    a = "1700000000-1-aaaaaaaa"
    vol_dir, adir, awork = _prepare_attempt(port, monkeypatch, tmp_path, a)
    commits, pointer_seen_during_sync = [], []
    monkeypatch.setattr(port.VOL, "commit", lambda: commits.append(1))
    real_sync = port._sync_results

    def sync(work, vol):
        pointer_seen_during_sync.append((vol_dir / "current.json").exists())
        return real_sync(work, vol)
    monkeypatch.setattr(port, "_sync_results", sync)
    rec = _completed_pass(a)
    port._publish(rec, awork, adir, vol_dir, a, promote=True)
    assert pointer_seen_during_sync == [False], "the pointer is the LAST durable write: one sync, before it"
    assert rec["publication_ok"] is True and "publish_error" not in rec
    assert rec["status"] == "DONE" and rec["verdict"] == "PASS" and rec["current"]["attempt"] == a
    assert port.modal_exit_code(rec) == 0
    line = (adir / "RESULT").read_text()
    assert line.startswith("PASS exit=0 ") and f" attempt={a} published=true " in line
    assert (adir / "RESULT").read_bytes() == (awork / "RESULT").read_bytes()
    landed = json.loads((adir / "wrapper_record.json").read_text())
    assert landed["publication_ok"] is True and landed["current"]["attempt"] == a
    ptr = json.loads((vol_dir / "current.json").read_text())
    assert ptr == rec["current"] == landed["current"], "the pointer is byte-identical to what the record carries"
    assert port.pointed_attempt_problems(vol_dir, ptr, sha=SHA) == [], "the mirror's agreement check accepts the real tree"
    assert port.required_attempt_files("PASS") == ["RESULT", "wrapper_record.json",
                                                    "steered_smoke/smoke_record.json",
                                                    "steered_smoke/smoke_evidence.json"]
    assert port.required_attempt_files("FAIL") == ["RESULT", "wrapper_record.json"]
    assert all((adir / rel).is_file() for rel in port.required_attempt_files("PASS"))
    assert len(commits) == 2, "the attempt's sync, then the pointer — each committed"
    assert not [p for p in vol_dir.iterdir() if p.name.startswith(".current.json.")]
    # the contract is EXPLICIT: drop any one publication fact and the exit is 1
    assert port.modal_exit_code({**rec, "publication_ok": False}) == 1
    assert port.modal_exit_code({k: v for k, v in rec.items() if k != "publication_ok"}) == 1
    assert port.modal_exit_code({**rec, "current": {**rec["current"], "attempt": "1700000000-9-other000"}}) == 1
    assert port.modal_exit_code({**rec, "current": None}) == 1
    assert port.modal_exit_code({k: v for k, v in rec.items() if k != "attempt"}) == 1


def _install_remote_from_dir(port, monkeypatch, sha: str, sha_dir: Path) -> None:
    """Serve a REAL local volume tree (what _publish produced under tmp_path) as
    the fake remote, so the mirror is exercised against the publisher's output."""
    files = {f"/petri-smoke/{sha}/{p.relative_to(sha_dir).as_posix()}": p.read_bytes()
             for p in sha_dir.rglob("*") if p.is_file()} if sha_dir.is_dir() else {}
    # _prepare_attempt pointed RESULTS_ROOT at the tmp volume for the publisher; the mirror
    # lists the REMOTE under volume_client_path(RESULTS_ROOT), so it must be the /vol path again
    monkeypatch.setattr(port, "RESULTS_ROOT", "/vol/petri-smoke")
    _install_remote(port, monkeypatch, sha, files)


@pytest.mark.parametrize("inject", ["final_sync_once", "final_sync_sticky", "promote", "commit_promotion",
                                    "readback_other"])
def test_a_pass_whose_publication_fails_is_publish_failed_and_exits_nonzero(port, monkeypatch, tmp_path, inject):
    """Failure injected into the final sync (ONCE — the round-4 tree: the recovery
    rewrite then succeeds; or sticky), the promotion before its replace, the
    promotion's commit after its replace, or a read-back naming another attempt:
    the returned record is status = verdict = PUBLISH_FAILED (smoke_verdict keeps
    PASS), publication_ok false, current None, exit 1 — and the VOLUME is never
    a PASS pointer beside PUBLISH_FAILED files: either nothing names the attempt
    (its files say PUBLISH_FAILED, or its final files sit orphaned), or the
    pointer names it and its files were left EXACTLY as the pointer describes.
    The mirror, served the real tree, never promotes an inconsistent one."""
    a = "1700000000-1-aaaaaaaa"
    vol_dir, adir, awork = _prepare_attempt(port, monkeypatch, tmp_path, a)
    calls = []
    real_sync = port._sync_results
    if inject in ("final_sync_once", "final_sync_sticky"):
        def sync(work, vol):
            calls.append(1)
            if len(calls) == 1 or inject == "final_sync_sticky":   # the ONE final sync (before the pointer)
                raise OSError("volume write failed")
            return real_sync(work, vol)
        monkeypatch.setattr(port, "_sync_results", sync)
    elif inject == "promote":
        def promote(*_a, **_k):
            raise OSError("temp write failed")                    # before any replace
        monkeypatch.setattr(port, "_promote_current", promote)
    elif inject == "commit_promotion":
        def commit():
            calls.append(1)
            if len(calls) == 2:                                    # the promotion's commit, AFTER its replace
                raise RuntimeError("commit failed")
        monkeypatch.setattr(port.VOL, "commit", commit)
    else:
        # a lost write: the old pointer, naming another attempt, survives os.replace
        vol_dir.mkdir(parents=True, exist_ok=True)
        (vol_dir / "current.json").write_text(json.dumps({"attempt": "1700000000-9-other000"}))
        monkeypatch.setattr(os, "replace", lambda src, dst: Path(src).unlink())
    rec = _completed_pass(a)
    port._publish(rec, awork, adir, vol_dir, a, promote=True)
    assert rec["status"] == rec["verdict"] == "PUBLISH_FAILED" and rec["smoke_verdict"] == "PASS"
    assert rec["publication_ok"] is False and rec["current"] is None and rec["publish_error"]
    assert "publication failed" in rec["reason"] and "verdict was PASS" in rec["reason"]
    assert port.modal_exit_code(rec) == 1
    assert port.result_line(rec).startswith("PUBLISH_FAILED exit=0 ") and " published=false " in port.result_line(rec)
    assert not [p for p in vol_dir.iterdir() if p.name.startswith(".current.json.")]
    pointer = json.loads((vol_dir / "current.json").read_text()) if (vol_dir / "current.json").is_file() else None
    landed_result = (adir / "RESULT").read_text() if (adir / "RESULT").is_file() else None
    landed_rec = json.loads((adir / "wrapper_record.json").read_text()) if (adir / "wrapper_record.json").is_file() else None
    if inject == "commit_promotion":
        # the pointer WAS replaced (its commit failed): the attempt is left EXACTLY as it is —
        # final files that agree with the pointer — never rewritten to PUBLISH_FAILED
        assert rec["pointer_may_be_installed"] is True and len(calls) == 2
        assert pointer["attempt"] == a and pointer["verdict"] == "PASS"
        assert landed_result.startswith("PASS exit=0 ") and f" attempt={a} published=true " in landed_result
        assert landed_rec["verdict"] == "PASS" and landed_rec["publication_ok"] is True
        assert landed_rec["current"] == pointer
        assert port.pointed_attempt_problems(vol_dir, pointer, sha=SHA) == []
    else:
        # nothing names this attempt; it was rewritten (best effort) to say why
        assert rec["pointer_may_be_installed"] is False
        assert pointer is None or pointer["attempt"] != a
        if inject == "final_sync_sticky":
            assert landed_result is None and landed_rec is None, "a dead sync lands nothing"
            assert "could not be synced either" in rec["publish_error"]
        else:
            assert landed_result.startswith("PUBLISH_FAILED exit=0 ") and " published=false " in landed_result
            assert landed_rec["verdict"] == landed_rec["status"] == "PUBLISH_FAILED"
            assert landed_rec["smoke_verdict"] == "PASS" and landed_rec["publication_ok"] is False
            assert landed_rec["current"] is None
        if inject == "readback_other":
            assert pointer == {"attempt": "1700000000-9-other000"} and "not in place" in rec["publish_error"]
        if inject == "promote":
            assert pointer is None
    # the local work dir's RESULT never claims PASS either
    assert (awork / "RESULT").read_text().startswith("PUBLISH_FAILED " if inject != "commit_promotion" else "PASS ")
    # the mirror, served THIS tree: it never promotes an inconsistent one
    _install_remote_from_dir(port, monkeypatch, SHA, vol_dir)
    got = port.mirror_results(SHA, tmp_path / "mirror")
    if inject == "commit_promotion":
        # consistent by construction — the volume holds a verified publication the run
        # could not confirm; the mirror may take it (main() does not ask it to, see the
        # harvest test), and what it takes agrees with the pointer
        assert got["promoted"] is True and got["pointer"]["attempt"] == a
    else:
        assert got["promoted"] is False and "error" in got
        assert not (tmp_path / "mirror" / SHA / "current").exists()
        if inject == "readback_other":
            assert "lacks" in got["error"]          # the other attempt's files are not there
        else:
            assert "no current.json" in got["error"]


def test_round4_tree_a_single_final_sync_failure_then_recovery_never_leaves_a_pass_pointer(port, monkeypatch, tmp_path):
    """Pin (a), with a PREVIOUS published attempt in place: the final sync fails
    exactly once and the PUBLISH_FAILED recovery succeeds. The pointer still
    names the previous attempt, this attempt's files say PUBLISH_FAILED, the
    exit is 1, and the mirror promotes the PREVIOUS attempt — never this one."""
    prev, a = "1700000000-1-pppppppp", "1700000000-2-aaaaaaaa"
    vol_dir, pdir, pwork = _prepare_attempt(port, monkeypatch, tmp_path, prev)
    rec_prev = {**_completed_pass(prev), "verdict": "FAIL", "smoke_rc": 1, "reason": "smoke_failed.json: x"}
    port._publish(rec_prev, pwork, pdir, vol_dir, prev, promote=True)
    assert rec_prev["publication_ok"] is True
    before = _snapshot(pdir)
    _, adir, awork = port.attempt_paths(SHA, a)
    _fill(awork, "A")
    adir.mkdir(parents=True, exist_ok=False)
    real_sync, calls = port._sync_results, []

    def sync(work, vol):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("volume write failed once")
        return real_sync(work, vol)
    monkeypatch.setattr(port, "_sync_results", sync)
    rec = _completed_pass(a)
    port._publish(rec, awork, adir, vol_dir, a, promote=True)
    assert rec["status"] == "PUBLISH_FAILED" and rec["smoke_verdict"] == "PASS" and port.modal_exit_code(rec) == 1
    assert len(calls) == 2, "the failed final sync, then the successful PUBLISH_FAILED recovery"
    ptr = json.loads((vol_dir / "current.json").read_text())
    assert ptr["attempt"] == prev and ptr["verdict"] == "FAIL", "the pointer never moved"
    assert _snapshot(pdir) == before, "the previous attempt is untouched"
    assert (adir / "RESULT").read_text().startswith("PUBLISH_FAILED exit=0 ")
    assert json.loads((adir / "wrapper_record.json").read_text())["status"] == "PUBLISH_FAILED"
    # a fresh mirror pull takes the previous attempt (consistent), not this one
    _install_remote_from_dir(port, monkeypatch, SHA, vol_dir)
    got = port.mirror_results(SHA, tmp_path / "mirror")
    assert got["promoted"] is True and got["pointer"]["attempt"] == prev
    cur = tmp_path / "mirror" / SHA / "current"
    assert (cur / "attempts" / prev / "RESULT").read_text().startswith("FAIL exit=1 ")
    assert (cur / "attempts" / a / "RESULT").read_text().startswith("PUBLISH_FAILED ")
    # and main() would not even ask for the pull after a PUBLISH_FAILED run
    assert port.mirror_after_run(rec, SHA, tmp_path / "mirror2", no_mirror=False) is None
    assert not (tmp_path / "mirror2").exists()


def test_publication_fails_closed_when_the_periodic_sync_does_not_quiesce(port, monkeypatch, tmp_path):
    """A periodic sync held past the quiesce bound and released
    only after the publish attempt. Publication writes NOTHING (volume or work
    dir), installs no pointer, and the record is PUBLISH_FAILED naming the
    un-quiesced sync; once released, the worker finishes and exits, and the
    attempt on the volume is byte-identical to what it was before the attempt."""
    a = "1700000000-1-aaaaaaaa"
    vol_dir, adir, awork = _prepare_attempt(port, monkeypatch, tmp_path, a)
    port._sync_results(awork, adir)                       # an earlier periodic copy already landed
    before = _snapshot(adir)
    started, release, calls = threading.Event(), threading.Event(), []
    real_sync = port._sync_results

    def stalled_sync(work, vol):
        calls.append(threading.current_thread().name)
        started.set()
        assert release.wait(10), "the test releases the stalled sync after the publish attempt"
        return real_sync(work, vol)
    monkeypatch.setattr(port, "_sync_results", stalled_sync)
    monkeypatch.setattr(port, "SYNC_QUIESCE_S", 0.2)
    worker = port.SyncWorker(awork, adir, every_s=0.01).start()
    assert started.wait(5), "the worker's periodic sync started"
    rec = _completed_pass(a)
    t0 = time.monotonic()
    port._publish(rec, awork, adir, vol_dir, a, promote=True, syncer=worker)
    assert time.monotonic() - t0 < 5, "the bounded join returned; publication did not block on the stalled sync"
    assert worker.thread.is_alive(), "the periodic sync is still stalled — that is the case under test"
    assert rec["periodic_sync_quiesced"] is False
    assert rec["status"] == rec["verdict"] == "PUBLISH_FAILED" and rec["smoke_verdict"] == "PASS"
    assert rec["publication_ok"] is False and rec["current"] is None and rec["pointer_may_be_installed"] is False
    assert "did not quiesce" in rec["publish_error"] and "did not quiesce" in rec["reason"]
    assert port.modal_exit_code(rec) == 1
    assert not (vol_dir / "current.json").exists()
    assert not (awork / "RESULT").exists() and not (awork / "wrapper_record.json").exists(), "nothing written locally"
    assert _snapshot(adir) == before, "nothing written to the attempt on the volume"
    assert calls == ["sync"], "only the stalled periodic sync ever ran; the final sync never started"
    # release the stalled worker: it completes its copy and exits — and still nothing names the attempt
    release.set()
    worker.thread.join(5)
    assert not worker.thread.is_alive()
    assert not (vol_dir / "current.json").exists() and _snapshot(adir) == before
    assert calls == ["sync"]
    # the same failure on an attempt that did not complete: publish_error only, status kept
    rec2 = {**_completed_pass(a), "verdict": "TIMEOUT", "status": "TIMEOUT"}
    stuck = port.SyncWorker(awork, adir, every_s=0.01)
    stuck.thread = threading.Thread(target=lambda: release.wait(10) and time.sleep(0.5), name="sync", daemon=True)
    stuck.thread.start()
    port._publish(rec2, awork, adir, vol_dir, a, promote=False, syncer=stuck)
    assert rec2["status"] == "TIMEOUT" and rec2["periodic_sync_quiesced"] is False
    assert "did not quiesce" in rec2["publish_error"] and "smoke_verdict" not in rec2
    assert _snapshot(adir) == before and not (awork / "RESULT").exists()
    stuck.thread.join(5)
    # a worker that never started counts as quiesced
    assert port.SyncWorker(awork, adir).quiesce(0.1) is True


def test_publication_proceeds_under_the_lock_once_the_periodic_sync_has_ended(port, monkeypatch, tmp_path):
    """The happy path with a live worker: it has synced at least once, quiesces
    promptly on request, and the final sync + pointer then run on the main
    thread under the worker's lock with the worker proven dead — a PASS, exit 0."""
    a = "1700000000-1-aaaaaaaa"
    vol_dir, adir, awork = _prepare_attempt(port, monkeypatch, tmp_path, a)
    real_sync, seen = port._sync_results, []

    def sync(work, vol):
        seen.append((threading.current_thread().name, worker.lock.locked(), worker.thread.is_alive()))
        return real_sync(work, vol)
    monkeypatch.setattr(port, "_sync_results", sync)
    worker = port.SyncWorker(awork, adir, every_s=0.01).start()
    deadline = time.monotonic() + 5
    while not any(n == "sync" for n, _, _ in seen) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert any(n == "sync" for n, _, _ in seen), "the worker synced periodically"
    rec = _completed_pass(a)
    port._publish(rec, awork, adir, vol_dir, a, promote=True, syncer=worker)
    assert rec["periodic_sync_quiesced"] is True and not worker.thread.is_alive()
    assert rec["publication_ok"] is True and rec["status"] == "DONE" and port.modal_exit_code(rec) == 0
    assert json.loads((vol_dir / "current.json").read_text())["attempt"] == a
    periodic = [x for x in seen if x[0] == "sync"]
    final = [x for x in seen if x[0] != "sync"]
    assert periodic and all(locked for _, locked, _ in periodic), "every periodic sync ran under the lock"
    assert len(final) == 1 and final[0][1] is True and final[0][2] is False, \
        "the one final sync ran under the lock, after the worker had ended"
    assert seen.index(final[0]) == len(seen) - 1, "no periodic sync after the final one"


def test_promote_current_reads_the_pointer_back_and_raises_on_a_mismatch(port, monkeypatch, tmp_path):
    a = "1700000000-1-aaaaaaaa"
    vol_dir = tmp_path / "vol" / SHA
    vol_dir.mkdir(parents=True)
    (vol_dir / "current.json").write_text(json.dumps({"attempt": "1700000000-9-other000"}))
    rec = _completed_pass(a)
    # a lost write (another attempt's pointer survives): a PLAIN error — this pointer is not in place
    monkeypatch.setattr(os, "replace", lambda src, dst: Path(src).unlink())
    with pytest.raises(RuntimeError, match="not in place") as ex:
        port._promote_current(vol_dir, a, rec)
    assert not isinstance(ex.value, port.PointerInstalledError)
    assert json.loads((vol_dir / "current.json").read_text())["attempt"] == "1700000000-9-other000"
    assert not [p for p in vol_dir.iterdir() if p.name.startswith(".current.json.")]
    monkeypatch.undo()
    # a failing commit AFTER the replace: PointerInstalledError, the pointer in place
    monkeypatch.setattr(port.VOL, "commit", lambda: (_ for _ in ()).throw(RuntimeError("commit failed")))
    with pytest.raises(port.PointerInstalledError, match="replaced but not confirmed"):
        port._promote_current(vol_dir, a, rec)
    assert json.loads((vol_dir / "current.json").read_text())["attempt"] == a
    monkeypatch.undo()
    # the pointer passed in is written byte-identical and returned
    pointer = port.current_pointer(vol_dir, a, rec)
    ptr = port._promote_current(vol_dir, a, rec, pointer)
    assert ptr is pointer and json.loads((vol_dir / "current.json").read_text()) == pointer and ptr["attempt"] == a
    assert "published=" not in ptr["result"] and f" attempt={a} " in ptr["result"]
    assert issubclass(port.PointerInstalledError, RuntimeError)


# --- the grading host mirror ---------------------------------------------------


def _remote(sha: str, attempt: str, verdict: str = "PASS", *, complete: bool = True, extra=None) -> dict:
    """A fake volume tree for one sha: current.json -> attempts/<attempt> whose own
    RESULT and wrapper_record.json AGREE with the pointer (what _publish writes),
    plus an older attempt; ``complete=False`` drops the smoke record a PASS must
    carry; ``extra`` overrides / adds files."""
    base = f"/petri-smoke/{sha}"
    result = (f"{verdict} exit=0 elapsed_s=900.0 app=petri-smoke-1111111 sha={sha} image=tf5141 "
              f"attempt={attempt} published=true reason=x")
    ptr = {"attempt": attempt, "path": f"attempts/{attempt}", "verdict": verdict, "status": "DONE",
           "app": "petri-smoke-1111111", "sha": sha, "image": "tf5141",
           "result": result.replace(" published=true", ""), "promoted_at": 1700000000.5}
    record = {"attempt": attempt, "verdict": verdict, "status": "DONE", "publication_ok": True, "current": ptr}
    files = {
        f"{base}/current.json": json.dumps(ptr).encode(),
        f"{base}/attempts/{attempt}/RESULT": (result + "\n").encode(),
        f"{base}/attempts/{attempt}/wrapper_record.json": json.dumps(record).encode(),
        f"{base}/attempts/{attempt}/smoke.log": b"log\n",
        f"{base}/attempts/0000000000-1-older/RESULT": b"UNAVAILABLE exit=0 ...\n",
        f"{base}/attempts/0000000000-1-older/steered_smoke/smoke_unavailable.json": b"{}",
    }
    if complete:
        files[f"{base}/attempts/{attempt}/steered_smoke/smoke_record.json"] = b'{"status": "completed"}'
        files[f"{base}/attempts/{attempt}/steered_smoke/smoke_evidence.json"] = b"{}"
    files.update(extra or {})
    return files


def _install_remote(port, monkeypatch, sha: str, files: dict, *, fail_on: str | None = None):
    src = f"/petri-smoke/{sha}"
    entries = [types.SimpleNamespace(path=p.lstrip("/"), type=types.SimpleNamespace(name="FILE")) for p in files]
    entries.append(types.SimpleNamespace(path=f"petri-smoke/{sha}/attempts", type=types.SimpleNamespace(name="DIRECTORY")))
    monkeypatch.setattr(port.VOL, "listdir",
                        lambda path, recursive: entries if path == src and recursive else [], raising=False)

    def read(path):
        if fail_on and path.endswith(fail_on):
            raise OSError(f"transient read failure on {path}")
        return files[path]
    monkeypatch.setattr(port, "_vol_read", read)


def test_mirror_promotes_a_versioned_pull_atomically_via_the_current_symlink(port, monkeypatch, tmp_path):
    a1 = "1700000001-1-aaaaaaaa"
    _install_remote(port, monkeypatch, SHA, _remote(SHA, a1))
    got = port.mirror_results(SHA, tmp_path)
    assert got["promoted"] is True and got["pointer"]["attempt"] == a1
    cur = tmp_path / SHA / "current"
    assert cur.is_symlink() and os.readlink(cur) == os.path.join("versions", got["current_target"])
    assert Path(got["version"]).name == got["current_target"]
    assert json.loads((cur / "current.json").read_text())["attempt"] == a1
    assert (cur / "attempts" / a1 / "steered_smoke" / "smoke_record.json").read_bytes() == b'{"status": "completed"}'
    assert (cur / "attempts" / "0000000000-1-older" / "steered_smoke" / "smoke_unavailable.json").is_file()
    assert sorted(got["pulled"]) == sorted(p[len(f"/petri-smoke/{SHA}/"):] for p in _remote(SHA, a1))
    assert not [p for p in (tmp_path / SHA).iterdir() if p.name.startswith(".current.")]
    # a second pull (new attempt promoted remotely) -> new version, symlink repointed, old version kept
    a2 = "1700000002-2-bbbbbbbb"
    _install_remote(port, monkeypatch, SHA, _remote(SHA, a2, verdict="FAIL"))
    got2 = port.mirror_results(SHA, tmp_path)
    assert got2["promoted"] and got2["current_target"] != got["current_target"]
    assert json.loads((cur / "current.json").read_text())["attempt"] == a2
    assert sorted(p.name for p in (tmp_path / SHA / "versions").iterdir()) == \
        sorted([got["current_target"], got2["current_target"]])


def test_mirror_read_failure_leaves_the_previous_mirror_byte_identical(port, monkeypatch, tmp_path):
    a1 = "1700000001-1-aaaaaaaa"
    _install_remote(port, monkeypatch, SHA, _remote(SHA, a1))
    first = port.mirror_results(SHA, tmp_path)
    assert first["promoted"]
    before = _snapshot(tmp_path / SHA)
    # the remote now has a new promoted attempt, but ONE of its files fails to read
    # (the LAST one listed, so every other file had already landed in the temp version)
    a2 = "1700000002-2-bbbbbbbb"
    files = _remote(SHA, a2)
    last = list(files)[-1]
    _install_remote(port, monkeypatch, SHA, files, fail_on=last.rsplit("/", 1)[-1])
    got = port.mirror_results(SHA, tmp_path)
    assert "error" in got and "read failed for" in got["error"] and got["promoted"] is False
    assert got["note"] == "existing mirror left untouched"
    assert _snapshot(tmp_path / SHA) == before, "a failed pull must not change one byte of the mirror"
    assert json.loads((tmp_path / SHA / "current" / "current.json").read_text())["attempt"] == a1
    assert len(list((tmp_path / SHA / "versions").iterdir())) == 1, "no partial version left behind"


def test_mirror_refuses_an_incomplete_promoted_attempt_and_a_pointerless_tree(port, monkeypatch, tmp_path):
    a1 = "1700000001-1-aaaaaaaa"
    _install_remote(port, monkeypatch, SHA, _remote(SHA, a1))
    assert port.mirror_results(SHA, tmp_path)["promoted"]
    before = _snapshot(tmp_path / SHA)
    # a PASS pointer whose attempt lacks the smoke record it claims to rest on
    a2 = "1700000002-2-bbbbbbbb"
    _install_remote(port, monkeypatch, SHA, _remote(SHA, a2, complete=False))
    got = port.mirror_results(SHA, tmp_path)
    assert got["promoted"] is False and "lacks" in got["error"] and "smoke_record.json" in got["error"]
    assert _snapshot(tmp_path / SHA) == before
    # a FAIL pointer needs only RESULT + wrapper_record.json (no smoke record) -> promoted
    _install_remote(port, monkeypatch, SHA, _remote(SHA, a2, verdict="FAIL", complete=False))
    assert port.mirror_results(SHA, tmp_path)["promoted"] is True
    before = _snapshot(tmp_path / SHA)
    # no current.json at all (a pre-attempts layout): nothing promoted, mirror untouched
    files = {k: v for k, v in _remote(SHA, a2).items() if not k.endswith("current.json")}
    _install_remote(port, monkeypatch, SHA, files)
    got = port.mirror_results(SHA, tmp_path)
    assert got["promoted"] is False and "no current.json" in got["error"]
    assert _snapshot(tmp_path / SHA) == before
    # a listing failure: untouched too
    monkeypatch.setattr(port.VOL, "listdir", lambda path, recursive: (_ for _ in ()).throw(OSError("down")),
                        raising=False)
    got = port.mirror_results(SHA, tmp_path)
    assert got["promoted"] is False and "listing failed" in got["error"]
    assert _snapshot(tmp_path / SHA) == before
    assert port.required_mirror_files({"path": "attempts/x", "verdict": "PASS"}) == [
        "attempts/x/RESULT", "attempts/x/wrapper_record.json",
        "attempts/x/steered_smoke/smoke_record.json", "attempts/x/steered_smoke/smoke_evidence.json"]
    assert port.required_mirror_files({"path": "attempts/x", "verdict": "FAIL"}) == [
        "attempts/x/RESULT", "attempts/x/wrapper_record.json"]


def test_mirror_bound_to_the_runs_pointer_refuses_a_concurrent_replacement(port, monkeypatch, tmp_path):
    """Round 6: two self-consistent PASS pointers for one sha. The run's record
    belongs to attempt A (its pointer installed and read back); by the time
    of the pull the volume's current.json names attempt B. Bound to A's
    pointer, the pull is refused as superseded — the mirror byte-identical,
    the outcome MIRROR_SUPERSEDED with exit 2 — while an unbound pull, or one
    bound to B's own pointer, takes B. The happy path (pointer unchanged) is
    unchanged."""
    base = f"/petri-smoke/{SHA}"
    a, b = "1700000001-1-aaaaaaaa", "1700000002-2-bbbbbbbb"
    remote_a = _remote(SHA, a)
    ptr_a = json.loads(remote_a[f"{base}/current.json"])
    rec_a = {"verdict": "PASS", "status": "DONE", "attempt": a, "publication_ok": True, "current": ptr_a,
             "sha": SHA, "reason": "ok"}
    # the happy path: A published, current.json still A -> bound pull promotes A, exit 0
    _install_remote(port, monkeypatch, SHA, remote_a)
    got = port.mirror_after_run(rec_a, SHA, tmp_path, no_mirror=False)
    assert got["promoted"] is True and got["superseded"] is False and got["pointer"] == ptr_a
    assert port.local_exit(rec_a, got, no_mirror=False) == (0, "PASS")
    assert json.loads((tmp_path / SHA / "current" / "current.json").read_text()) == ptr_a
    before = _snapshot(tmp_path / SHA)
    # meanwhile: attempt B (also self-consistent) replaced current.json; A's files are still there
    a_files = {k: v for k, v in remote_a.items() if f"/attempts/{a}/" in k}
    remote_b = _remote(SHA, b, extra=a_files)
    ptr_b = json.loads(remote_b[f"{base}/current.json"])
    assert ptr_b != ptr_a and port.pointed_attempt_problems is not None
    _install_remote(port, monkeypatch, SHA, remote_b)
    got = port.mirror_after_run(rec_a, SHA, tmp_path, no_mirror=False)
    assert got["promoted"] is False and got["superseded"] is True
    assert "concurrent publication" in got["error"] and repr(b) in got["error"] and repr(a) in got["error"]
    assert got["pointer"] == ptr_b and got["expected_pointer"] == ptr_a
    assert got["note"] == "existing mirror left untouched"
    assert _snapshot(tmp_path / SHA) == before, "the mirror holds no evidence this run did not install"
    assert json.loads((tmp_path / SHA / "current" / "current.json").read_text()) == ptr_a
    assert len(list((tmp_path / SHA / "versions").iterdir())) == 1, "no partial version left behind"
    assert port.local_exit(rec_a, got, no_mirror=False) == (2, "MIRROR_SUPERSEDED")
    assert port.modal_exit_code(rec_a) == 0, "the publication itself succeeded; only the mirror is withheld"
    # a pointer differing in ONE field is superseded too (whole-object equality)
    got = port.mirror_results(SHA, tmp_path, expected_pointer={**ptr_b, "promoted_at": ptr_b["promoted_at"] + 1})
    assert got["promoted"] is False and got["superseded"] is True and _snapshot(tmp_path / SHA) == before
    # B's own run, bound to B's pointer, takes B; an unbound pull takes whatever is consistent
    rec_b = {**rec_a, "attempt": b, "current": ptr_b}
    got = port.mirror_after_run(rec_b, SHA, tmp_path, no_mirror=False)
    assert got["promoted"] is True and got["superseded"] is False
    assert json.loads((tmp_path / SHA / "current" / "current.json").read_text()) == ptr_b
    assert port.local_exit(rec_b, got, no_mirror=False) == (0, "PASS")
    assert port.mirror_results(SHA, tmp_path)["promoted"] is True
    # superseded is not the same outcome as PUBLISH_FAILED, and never 1
    assert port.EXIT_MIRROR == 2 and issubclass(port.MirrorSuperseded, RuntimeError)


def test_mirror_transactions_on_one_sha_are_serialized_by_the_cross_process_lock(port, monkeypatch, tmp_path):
    """Round 7: two pulls of ONE sha at once — two threads, and flock is per open
    file description so they contend exactly as two processes do — with an
    interleaving barrier: t1 holds the lock inside its pull while t2 arrives;
    t2 cannot acquire until t1 has released; each caller's `promoted` is what
    `current` resolved to under ITS lock (its own version, verified); the
    mirror sits at the retention limit throughout and GC never removes the
    current version."""
    a = "1700000001-1-aaaaaaaa"
    remote = _remote(SHA, a)
    _install_remote(port, monkeypatch, SHA, remote)
    for _ in range(2):
        assert port.mirror_results(SHA, tmp_path, keep=2)["promoted"]        # at the retention limit
    ptr = json.loads(remote[f"/petri-smoke/{SHA}/current.json"])
    names = ("t1", "t2")
    attempt = {n: threading.Event() for n in names}
    acquired = {n: threading.Event() for n in names}
    in_pull = {n: threading.Event() for n in names}
    release = {n: threading.Event() for n in names}
    events, events_lock = [], threading.Lock()

    def log(*ev):
        with events_lock:
            events.append(ev)
    real_lock, real_read = port._mirror_lock, port._vol_read

    @contextlib.contextmanager
    def traced_lock(sha_root, timeout=None):
        n = threading.current_thread().name
        log(n, "attempt")
        attempt[n].set()
        with real_lock(sha_root, timeout):
            log(n, "acquired")
            acquired[n].set()
            try:
                yield
            finally:
                log(n, "releasing")

    def traced_read(path):
        n = threading.current_thread().name
        if not in_pull[n].is_set():                   # hold the FIRST read of each pull until released
            in_pull[n].set()
            assert release[n].wait(10), f"{n} released by the test"
        return real_read(path)
    monkeypatch.setattr(port, "_mirror_lock", traced_lock)
    monkeypatch.setattr(port, "_vol_read", traced_read)
    results = {}

    def run(n):
        results[n] = port.mirror_results(SHA, tmp_path, keep=2, expected_pointer=ptr)
    t1 = threading.Thread(target=run, args=("t1",), name="t1")
    t2 = threading.Thread(target=run, args=("t2",), name="t2")
    t1.start()
    assert acquired["t1"].wait(5) and in_pull["t1"].wait(5), "t1 holds the lock, inside its pull"
    t2.start()
    assert attempt["t2"].wait(5), "t2 reached the lock"
    time.sleep(0.3)
    assert not acquired["t2"].is_set() and t2.is_alive() and "t1" not in results, \
        "t2 blocks on the lock for as long as t1's transaction is open"
    release["t1"].set()
    t1.join(10)
    assert acquired["t2"].wait(5), "t2 acquires once t1 released"
    release["t2"].set()
    t2.join(10)
    assert not t1.is_alive() and not t2.is_alive()
    r1, r2 = results["t1"], results["t2"]
    # each caller's promotion is what current resolved to under ITS lock: its own version, verified
    for r in (r1, r2):
        assert r["promoted"] is True and r["verified"] is True and r["pointer"] == ptr
        assert r["current_target"] == Path(r["version"]).name
        assert r["current_target"] not in r["removed_versions"]
    assert r1["version"] != r2["version"] and r2["pull_seq"] == r1["pull_seq"] + 1
    # exactly one transaction open at a time; t1 released before t2 acquired
    seq = list(events)
    assert seq.index(("t1", "releasing")) < seq.index(("t2", "acquired"))
    holder = None
    for n, e in seq:
        if e == "acquired":
            assert holder is None, f"{n} acquired while {holder} held the lock"
            holder = n
        elif e == "releasing":
            assert holder == n
            holder = None
    # the final current is t2's version; t1's survives (keep=2); every removed version was older than both
    cur = tmp_path / SHA / "current"
    assert os.readlink(cur) == os.path.join("versions", r2["current_target"])
    remaining = sorted(p.name for p in (tmp_path / SHA / "versions").iterdir())
    assert remaining == sorted([r1["current_target"], r2["current_target"]])
    assert json.loads((cur / "current.json").read_text()) == ptr
    assert r1["removed_versions"] and r2["removed_versions"]
    assert not set(r1["removed_versions"]) & set(remaining) and not set(r2["removed_versions"]) & set(remaining)
    assert not [p for p in (tmp_path / SHA).iterdir() if p.name.startswith(".current.")]


def test_mirror_lock_timeout_is_mirror_failed_and_leaves_the_mirror_untouched(port, monkeypatch, tmp_path):
    a = "1700000001-1-aaaaaaaa"
    remote = _remote(SHA, a)
    _install_remote(port, monkeypatch, SHA, remote)
    assert port.mirror_results(SHA, tmp_path)["promoted"]
    ptr = json.loads(remote[f"/petri-smoke/{SHA}/current.json"])
    before = _snapshot(tmp_path / SHA)
    lock_file = tmp_path / SHA / port.MIRROR_LOCK
    assert lock_file.is_file()
    fd = os.open(str(lock_file), os.O_RDWR)                  # another process's transaction, held
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        monkeypatch.setattr(port, "MIRROR_LOCK_TIMEOUT_S", 0.2)
        t0 = time.monotonic()
        got = port.mirror_results(SHA, tmp_path, expected_pointer=ptr)
        elapsed = time.monotonic() - t0
        assert got["promoted"] is False and got["lock_timeout"] is True and got["superseded"] is False
        assert "mirror.lock" in got["error"] and "another mirror transaction" in got["error"]
        assert 0.15 <= elapsed < 5, "bounded: the wait is the timeout, not forever"
        assert got["note"] == "existing mirror left untouched" and got["version"] is None
        assert _snapshot(tmp_path / SHA) == before
        assert len(list((tmp_path / SHA / "versions").iterdir())) == 1, "no version allocated while locked out"
        rec = {"verdict": "PASS", "status": "DONE", "attempt": a, "publication_ok": True, "current": ptr}
        assert port.local_exit(rec, got, no_mirror=False) == (2, "MIRROR_FAILED")
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    # released: the same pull now goes through, verified
    got = port.mirror_results(SHA, tmp_path, expected_pointer=ptr)
    assert got["promoted"] is True and got["verified"] is True and got["lock_timeout"] is False
    assert issubclass(port.MirrorLockTimeout, RuntimeError)


_R4_RESULT = ("PUBLISH_FAILED exit=0 elapsed_s=900.0 app=petri-smoke-1111111 sha={sha} image=tf5141 "
              "attempt={a} published=false reason=publication failed\n")


@pytest.mark.parametrize("break_", ["round4_tree", "other_attempt", "other_verdict", "status_not_done",
                                    "publication_not_ok", "current_names_other", "result_token", "result_attempt",
                                    "result_not_published", "record_unreadable", "pointer_status",
                                    "pointer_sha", "pointer_image", "pointer_result", "pointer_path",
                                    "pointer_promoted_at", "pointer_verdict", "pointer_app"])
def test_mirror_refuses_a_pointer_whose_attempt_disagrees_with_it(port, monkeypatch, tmp_path, break_):
    """Pin (b): existence is not agreement. A hand-built PASS pointer over
    PUBLISH_FAILED RESULT + record (the round-4 tree), every single-field
    disagreement in the record or RESULT, and — round 5 — a pointer that
    differs from the record's ``current`` in ANY pointer-only field (sha,
    image, result, path, promoted_at, status, verdict, app: exactly what
    ``_promote_current`` would have rejected on read-back) is refused, the
    partial version removed, the existing mirror byte-identical."""
    a1 = "1700000001-1-aaaaaaaa"
    _install_remote(port, monkeypatch, SHA, _remote(SHA, a1))
    assert port.mirror_results(SHA, tmp_path)["promoted"]
    before = _snapshot(tmp_path / SHA)
    a2 = "1700000002-2-bbbbbbbb"
    base = f"/petri-smoke/{SHA}"
    good = _remote(SHA, a2)
    ptr = json.loads(good[f"{base}/current.json"])
    rec = json.loads(good[f"{base}/attempts/{a2}/wrapper_record.json"])
    result = good[f"{base}/attempts/{a2}/RESULT"].decode()
    if break_ == "round4_tree":
        rec.update(verdict="PUBLISH_FAILED", status="PUBLISH_FAILED", publication_ok=False, current=None,
                   smoke_verdict="PASS")
        result = _R4_RESULT.format(sha=SHA, a=a2)
    elif break_ == "other_attempt":
        rec["attempt"] = a1
    elif break_ == "other_verdict":
        rec["verdict"] = "FAIL"
    elif break_ == "status_not_done":
        rec["status"] = "TIMEOUT"
    elif break_ == "publication_not_ok":
        rec["publication_ok"] = False
    elif break_ == "current_names_other":
        rec["current"] = {"attempt": a1}
    elif break_ == "result_token":
        result = result.replace("PASS exit=0", "FAIL exit=1", 1)
    elif break_ == "result_attempt":
        result = result.replace(f"attempt={a2}", f"attempt={a1}")
    elif break_ == "result_not_published":
        result = result.replace("published=true", "published=false")
    elif break_ == "record_unreadable":
        rec = "not json {"
    elif break_ == "pointer_status":
        ptr["status"] = "TIMEOUT"
    elif break_ == "pointer_sha":
        ptr["sha"] = "2" * 40
    elif break_ == "pointer_image":
        ptr["image"] = "tf5150"
    elif break_ == "pointer_result":
        ptr["result"] = ptr["result"].replace("reason=x", "reason=y")
    elif break_ == "pointer_path":
        ptr["path"] = f"attempts//{a2}"        # resolves to the SAME dir, so only the agreement check can catch it
    elif break_ == "pointer_promoted_at":
        ptr["promoted_at"] = 1700000001.5
    elif break_ == "pointer_verdict":
        ptr["verdict"] = "FAIL"
    else:
        ptr["app"] = "petri-smoke-2222222"
    files = {**good, f"{base}/current.json": json.dumps(ptr).encode(),
             f"{base}/attempts/{a2}/wrapper_record.json": (rec if isinstance(rec, str) else json.dumps(rec)).encode(),
             f"{base}/attempts/{a2}/RESULT": result.encode()}
    _install_remote(port, monkeypatch, SHA, files)
    got = port.mirror_results(SHA, tmp_path)
    assert got["promoted"] is False and "disagrees with the pointer" in got["error"], got
    assert got["note"] == "existing mirror left untouched"
    assert _snapshot(tmp_path / SHA) == before
    assert json.loads((tmp_path / SHA / "current" / "current.json").read_text())["attempt"] == a1
    assert len(list((tmp_path / SHA / "versions").iterdir())) == 1, "no partial version left behind"
    if break_.startswith("pointer_") and break_ not in ("pointer_status", "pointer_verdict"):
        assert "record current != pointer" in got["error"] and break_[len("pointer_"):] in got["error"]
    if break_ == "pointer_path":
        assert "pointer path" in got["error"]
    if break_ == "pointer_sha":
        assert "the mirrored sha" in got["error"]
    # the un-broken tree IS accepted, so the refusal above is the disagreement and nothing else
    _install_remote(port, monkeypatch, SHA, good)
    assert port.mirror_results(SHA, tmp_path)["promoted"] is True
    good_ptr = json.loads(good[f"{base}/current.json"])
    assert port.pointed_attempt_problems(tmp_path / SHA / "current", good_ptr, sha=SHA) == []
    # and the sha being mirrored is checked against the pointer's, independently of the record
    assert any("mirrored sha" in pr for pr in port.pointed_attempt_problems(tmp_path / SHA / "current", good_ptr,
                                                                           sha="3" * 40))


def test_mirror_garbage_collects_old_versions_but_never_the_current_one(port, monkeypatch, tmp_path):
    targets = []
    for i in range(4):
        a = f"17000000{i:02d}-1-{'a' * 8}"
        _install_remote(port, monkeypatch, SHA, _remote(SHA, a))
        got = port.mirror_results(SHA, tmp_path, keep=2)
        assert got["promoted"]
        targets.append(got["current_target"])
    versions = sorted(p.name for p in (tmp_path / SHA / "versions").iterdir())
    assert len(versions) == 2 and targets[-1] in versions
    assert os.readlink(tmp_path / SHA / "current") == os.path.join("versions", targets[-1])
    assert got["removed_versions"] and targets[-1] not in got["removed_versions"]


def test_mirror_gc_orders_by_pull_sequence_not_by_the_ids_lexical_order(port, monkeypatch, tmp_path):
    """Four pulls inside ONE second: the version ids share the epoch and their
    tails sort d, a, c, b — so the CURRENT (fourth) pull lands in the lexical
    MIDDLE. Lexical GC put it inside old[:-keep], skipped it as current and
    left keep+1 versions (the flaky test of 2026-09-06). Ordered by pull
    sequence: exactly ``keep`` remain, they are the two NEWEST pulls, the
    current one among them, and the removed ones are the two oldest."""
    tags = iter(["1700000000-1-dddddddd", "1700000000-1-aaaaaaaa", "1700000000-1-cccccccc", "1700000000-1-bbbbbbbb"])
    monkeypatch.setattr(port, "attempt_id", lambda: next(tags))
    versions = tmp_path / SHA / "versions"
    removed_per_pull, remaining_per_pull = [], []
    for i in range(4):
        _install_remote(port, monkeypatch, SHA, _remote(SHA, f"17000000{i:02d}-1-{'e' * 8}"))
        got = port.mirror_results(SHA, tmp_path, keep=2)
        assert got["promoted"] and got["pull_seq"] == i + 1
        removed_per_pull.append(got["removed_versions"])
        remaining_per_pull.append(sorted(p.name for p in versions.iterdir()))
    # GC runs on every pull: the third removes d (the OLDEST pull, though lexically LAST —
    # lexical GC would have removed a instead), the fourth removes a; exactly `keep` remain each time
    assert removed_per_pull == [[], [], ["1700000000-1-dddddddd"], ["1700000000-1-aaaaaaaa"]]
    assert remaining_per_pull[2] == ["1700000000-1-aaaaaaaa", "1700000000-1-cccccccc"]
    assert got["current_target"] == "1700000000-1-bbbbbbbb"
    remaining = remaining_per_pull[3]
    assert remaining == ["1700000000-1-bbbbbbbb", "1700000000-1-cccccccc"]
    assert os.readlink(tmp_path / SHA / "current") == os.path.join("versions", "1700000000-1-bbbbbbbb")
    assert [port.pull_seq(versions / n) for n in remaining] == [4, 3]
    assert json.loads((versions / "1700000000-1-bbbbbbbb" / port.MIRROR_PULL_META).read_text())["tag"] == \
        "1700000000-1-bbbbbbbb"
    # a version without the meta file (a pull from before sequencing) sorts oldest
    assert port.pull_seq(tmp_path / "nowhere") == 0
    # keep falsy: no GC at all
    _install_remote(port, monkeypatch, SHA, _remote(SHA, "1700000004-1-eeeeeeee"))
    monkeypatch.setattr(port, "attempt_id", lambda: "1700000000-1-00000000")
    got = port.mirror_results(SHA, tmp_path, keep=0)
    assert got["promoted"] and got["removed_versions"] == [] and len(list(versions.iterdir())) == 3


def test_mirror_refuses_a_pre_versions_current_directory(port, monkeypatch, tmp_path):
    (tmp_path / SHA / "current").mkdir(parents=True)          # an old-layout real directory
    (tmp_path / SHA / "current" / "RESULT").write_text("old")
    _install_remote(port, monkeypatch, SHA, _remote(SHA, "1700000001-1-aaaaaaaa"))
    before = _snapshot(tmp_path / SHA)
    got = port.mirror_results(SHA, tmp_path)
    assert got["promoted"] is False and "not a symlink" in got["error"]
    assert _snapshot(tmp_path / SHA) == before


# --- the entrypoint ------------------------------------------------------------


def test_requires_exact_descriptive_run_name(port):
    port.app.description = port.DEFAULT_APP_NAME
    with pytest.raises(SystemExit, match="modal run -n petri-smoke-1111111"):
        port.main(sha=SHA, pf=True)
    with pytest.raises(SystemExit, match="FULL 40-hex"):
        port.main(sha="1234567", pf=True)
    with pytest.raises(SystemExit, match="--image must be one of"):
        port.main(sha=SHA, pf=True, image="tf9999")
    assert port.expected_run_name(SHA) == "petri-smoke-1111111"


def test_go_never_spawns_past_a_failed_preflight(port, monkeypatch):
    port.app.description = port.expected_run_name(SHA)
    spawned = []
    objs = {"static_r_object": OBJ["static_r_matched"], "alphasteer_object": OBJ["alphasteer_official"]}
    monkeypatch.setattr(port, "run_preflight",
                        lambda variant, sha, s, a: {"ok": False, "checks": {
                            "objects": {"ok": False, "err": "REQUIRED object ... hashes to ..., not the pinned original"},
                            "tree": {"ok": True}}})
    monkeypatch.setattr(port, "spawn_smoke", lambda *a, **k: spawned.append(a))
    with pytest.raises(SystemExit, match=r"PREFLIGHT FAILED \(\['objects'\]\)"):
        port.main(sha=SHA, go=True, **objs)
    assert spawned == []
    monkeypatch.setattr(port, "run_preflight", lambda variant, sha, s, a: {"ok": True, "checks": {}})
    port.main(sha=SHA, pf=True, **objs)
    assert spawned == []


def test_pf_and_go_take_the_objects_from_the_site_and_refuse_by_name_before_any_call(port, monkeypatch, tmp_path):
    """Nothing names an object by default: with neither the flags nor the site keys, --pf / --go / --stage --go refuse by name
    before anything is staged, preflighted or spawned. With the keys set (the site's cluster paths), the preflight is handed the
    objects' volume paths."""
    port.app.description = port.expected_run_name(SHA)
    calls = []
    monkeypatch.setattr(port, "run_preflight", lambda variant, sha, s, a: calls.append(("pf", s, a)) or {"ok": True, "checks": {}})
    monkeypatch.setattr(port, "spawn_smoke", lambda *a, **k: calls.append(("spawn",) + a))
    monkeypatch.setattr(port, "shallow_clone_tarball", lambda *a, **k: calls.append(("stage",) + a))
    for k, v in _isolated_site(tmp_path).items():
        monkeypatch.setenv(k, v)
    for key in OBJECT_KEY.values():
        monkeypatch.delenv(key, raising=False)
    for flags in ({"pf": True}, {"go": True}, {"stage": True, "go": True}):
        with pytest.raises(SystemExit, match=r"no static_r_matched object: pass --static-r-object, or set "
                                             r"SISTEER_PETRI_SMOKE_STATIC_R_OBJECT"):
            port.main(sha=SHA, **flags)
    with pytest.raises(SystemExit, match=r"no alphasteer_official object: pass --alphasteer-object, or set "
                                         r"SISTEER_PETRI_SMOKE_ALPHASTEER_OBJECT"):
        port.main(sha=SHA, pf=True, static_r_object=OBJ["static_r_matched"])
    assert calls == []
    monkeypatch.setenv(OBJECT_KEY["static_r_matched"], "$SHARED/exp/jailbreak/example_set/static_r_matched/object.pt")
    monkeypatch.setenv(OBJECT_KEY["alphasteer_official"], "$SHARED/exp/jailbreak/example_set/alphasteer_official/object.pt")
    port.main(sha=SHA, pf=True)
    assert calls == [("pf", OBJ["static_r_matched"], OBJ["alphasteer_official"])]


def test_harvest_of_a_publish_failed_attempt_exits_one_and_does_not_mirror(port, monkeypatch, tmp_path):
    """Pin (d): ``--harvest`` of a PUBLISH_FAILED record exits 1 and never asks
    the mirror to pull (its attempt is not current, or cannot be confirmed as
    such). A PUBLISHED FAIL still pulls and exits 1; a PUBLISHED PASS pulls
    and returns; ``--no-mirror`` never pulls."""
    a = "1700000000-1-aaaaaaaa"
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"call": "fc-TEST", "sha": SHA, "app": "petri-smoke-1111111"}))
    pulled = []
    mirror_reply = {"promoted": True, "superseded": False}

    def fake_mirror(sha, root, **k):
        pulled.append((sha, str(root), k.get("expected_pointer")))
        return {**mirror_reply, "sha": sha}
    monkeypatch.setattr(port, "mirror_results", fake_mirror)
    records = {}
    monkeypatch.setattr(port.modal.FunctionCall, "from_id",
                        lambda call_id: types.SimpleNamespace(get=lambda: records[call_id]))
    mroot = str(tmp_path / "mirror")
    ptr = {"attempt": a, "path": f"attempts/{a}", "verdict": "FAIL", "status": "DONE", "sha": SHA,
           "promoted_at": 1700000000.5}
    records["fc-TEST"] = {"verdict": "PUBLISH_FAILED", "status": "PUBLISH_FAILED", "smoke_verdict": "PASS",
                          "attempt": a, "current": None, "publication_ok": False, "sha": SHA,
                          "publish_error": "OSError: volume write failed", "reason": "publication failed"}
    with pytest.raises(SystemExit) as ex:
        port.main(harvest=str(state), mirror_root=mroot)
    assert ex.value.code == 1 and pulled == []
    # a run with no pointer of its own (TIMEOUT here) refreshes nothing either
    records["fc-TEST"] = {"verdict": "TIMEOUT", "status": "TIMEOUT", "attempt": a, "current": None,
                          "publication_ok": False, "sha": SHA, "reason": "deadline"}
    with pytest.raises(SystemExit) as ex:
        port.main(harvest=str(state), mirror_root=mroot)
    assert ex.value.code == 1 and pulled == []
    # a PUBLISHED FAIL pulls, BOUND to the pointer in ITS record, and exits 1
    records["fc-TEST"] = {"verdict": "FAIL", "status": "DONE", "attempt": a, "publication_ok": True,
                          "current": ptr, "sha": SHA, "reason": "smoke_failed.json: x"}
    with pytest.raises(SystemExit) as ex:
        port.main(harvest=str(state), mirror_root=mroot)
    assert ex.value.code == 1 and pulled == [(SHA, mroot, ptr)]
    # a PUBLISHED PASS pulls, bound to its pointer, and returns
    ptr_pass = {**ptr, "verdict": "PASS"}
    records["fc-TEST"] = {"verdict": "PASS", "status": "DONE", "attempt": a, "publication_ok": True,
                          "current": ptr_pass, "sha": SHA, "reason": "ok"}
    assert port.main(harvest=str(state), mirror_root=mroot) is None
    assert pulled == [(SHA, mroot, ptr), (SHA, mroot, ptr_pass)]
    assert port.main(harvest=str(state), mirror_root=mroot, no_mirror=True) is None
    assert len(pulled) == 2
    # the SAME published PASS, but the volume's current.json was replaced meanwhile: exit 2, MIRROR_SUPERSEDED
    mirror_reply.update(promoted=False, superseded=True, error="concurrent publication replaced it")
    with pytest.raises(SystemExit) as ex:
        port.main(harvest=str(state), mirror_root=mroot)
    assert ex.value.code == port.EXIT_MIRROR == 2 and len(pulled) == 3
    # any other refused/failed pull after a PASS: exit 2 too (MIRROR_FAILED), never a silent 0
    mirror_reply.update(promoted=False, superseded=False, error="listing failed")
    with pytest.raises(SystemExit) as ex:
        port.main(harvest=str(state), mirror_root=mroot)
    assert ex.value.code == 2
    # the same gate + exit serve the --go path
    assert port.mirror_after_run({"status": "PUBLISH_FAILED"}, SHA, Path(mroot), no_mirror=False) is None
    assert port.mirror_after_run({"status": "DONE", "verdict": "FAIL"}, SHA, Path(mroot), no_mirror=False) is None, \
        "a DONE record without its pointer is not bound to anything: no pull"
    assert port.mirror_after_run({"status": "DONE", "verdict": "FAIL", "current": ptr}, SHA, Path(mroot),
                                 no_mirror=False)["sha"] == SHA
    assert pulled[-1] == (SHA, mroot, ptr)
    pass_rec = {"verdict": "PASS", "status": "DONE", "attempt": a, "publication_ok": True, "current": ptr_pass}
    assert port.local_exit(pass_rec, {"promoted": True, "superseded": False}, no_mirror=False) == (0, "PASS")
    assert port.local_exit(pass_rec, None, no_mirror=True) == (0, "PASS (not mirrored: --no-mirror)")
    assert port.local_exit(pass_rec, {"promoted": False, "superseded": True}, no_mirror=False) == \
        (2, "MIRROR_SUPERSEDED")
    assert port.local_exit(pass_rec, {"promoted": False, "superseded": False}, no_mirror=False) == (2, "MIRROR_FAILED")
    assert port.local_exit(pass_rec, None, no_mirror=False) == (2, "MIRROR_FAILED")
    assert port.local_exit({**pass_rec, "verdict": "FAIL"}, {"promoted": True}, no_mirror=False) == (1, "FAIL")
    assert port.local_exit(records["fc-TEST"] | {"verdict": "PUBLISH_FAILED", "status": "PUBLISH_FAILED",
                                                 "publication_ok": False}, None, no_mirror=False) == \
        (1, "PUBLISH_FAILED")


def test_go_path_never_reads_a_dunder_name_off_a_modal_function():
    """The first canary (2026-09-06) crashed AFTER spawning the GPU function:
    ``SMOKE_FN[image].__name__`` — a Modal Function object has no ``__name__``,
    so the local entrypoint died, Modal stopped the app, and the spawned smoke
    was killed. The spawn line names the image key instead; nothing on the
    ``--go`` path may read ``__name__`` off a Function."""
    src = REPO / "scripts" / "modal" / "petri_smoke.py"
    text = src.read_text()
    assert "SMOKE_FN[image].__name__" not in text
    assert ".__name__" not in text.split("def main(")[1]


def test_results_and_mirror_layout(port):
    assert port.RESULTS_ROOT == "/vol/petri-smoke"
    assert port.volume_client_path(port.RESULTS_ROOT) == "/petri-smoke"
    assert port.MIRROR_ROOT == Path.home() / "figs" / "petri-smoke"
    assert port.OUT_SUBDIR == "steered_smoke"
    assert port.MIRROR_KEEP_VERSIONS >= 2 and port.MIRROR_PULL_META == ".mirror-pull.json"
    assert port.MIRROR_LOCK == ".mirror.lock" and port.MIRROR_LOCK_TIMEOUT_S > 0
    with pytest.raises(ValueError):
        port.volume_client_path("/tmp/not-the-volume")

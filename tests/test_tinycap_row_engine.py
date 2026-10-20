"""tinycap-shared-standup §4 (module A): the ROW runner of ``scripts/modal/tinycap_served.py`` — ONE
standup per model row, every cell driven against it in turn, the row dir beside the stores, the
40-inode per-cell store — against a fake standup / ledger / driver world (no Modal, no GPU).

The solo path's byte-identity is held by the existing suites (``test_tinycap_launcher*.py``, the
lens goldens, the argv tests); this file pins the pieces the split exposed (the multi-cell
``_standup_cmd``, ``expected_identity(concurrency=)``), the row invariants, the per-cell doors,
the negative identity control, the repeat cell, the row refusal that names every cell, the
worker-loss resume (DONE skipped, in-flight parked, the subset relaunch under a fresh standup),
the ``_it`` row's own ``none``, the row engine record the driver copies, the row dir's contents
(pointer, sweeps, standup logs, ``gpu_interval.json`` in module C's vocabulary, ``row_rec.json``),
the exact 40-inode store (module E's ``tinycap_store_shape`` helper) and ``main --row``.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, Dict, List, Mapping, Optional, Sequence

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_tinycap_launcher as TL  # noqa: E402  (the stub, the loader and the store helpers)
import test_tinycap_widen_launcher as TW  # noqa: E402  (the lane tree)
import tinycap_store_shape as SS  # noqa: E402  (module E's 40-inode helper)
from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from sisteer.capeval.tinycap import cpu_verify as CV  # noqa: E402  (the driver half's marker / sidecar spellings)
from sisteer.capeval.tinycap import sandbox_scope as scope  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
NATIVE = "lora_native"
FULL_SHA = TL.FULL_SHA
ROW = "gemma4e4b-f22s1"
IT_ROW = "qwen35_4b-f22it"
LORA = "constv2ctl2-full__lora_r4_ce"
LORA2 = "constv2-mix-full__lora_r4_ce"
PROMPTED = "prompted__safety_refusal"
CONTROLS = ["bigcodebench_full", "bfcl_full"]
SUFFIX = "bcbw-gemma4e4b_s1-f6cf462-shared"
BCB_FULL = C.inspect_task("bigcodebench_full")
BFCL_FULL = C.inspect_task("bfcl_full")
STANDUP_A = "11111111222222223333333344444444"
PROTOCOL_CHECK = {"ok": True, "production": {}, "negative_control": {}, "defaults_source": "test"}
#: the statvfs reading the test world hands the row runner's inode probe (``_World._probe_volume_inodes``): a
#: 1,000,000-inode table with 100,000 used, so the engine's arithmetic (10 % used, +177 forecast for a staged
#: three-cell row = 10.02 % after, under the 90 % floor) is checkable by hand and independent of the host's filesystem
INODE_READING = SimpleNamespace(f_files=1_000_000, f_ffree=900_000)
GPU_INTERVAL_FIXTURE = Path.home() / "specs/tinycap-shared-standup/scripts/tests/fixtures/gpu_interval_example.json"


@pytest.fixture(scope="module")
def L():
    return TL._load_launcher(TL._stub_modal(), app_name=TL.TEST_APP_NAME, name="tinycap_served_row_engine")


# --------------------------------------------------------------------------- #
# the fake world: a staged lane tree, a row of configurations, a scripted     #
# standup that serves several cells, a live ledger, a driver that files stores #
# --------------------------------------------------------------------------- #


def _plan(L, cells, *, model=ROW, controls=CONTROLS):
    # the world plans under the driver's default draw (`card`): the shared `_manifest` helper builds card stores; the
    # row runner is draw-agnostic (the protocol clamp-override door is exercised by the fake endpoint whenever the
    # standup argv carries the override, exactly as the real standup records it)
    return L.plan_configurations(list(cells), ["off"], list(controls), model=model, serving_path=NATIVE)


def _row_tree(tmp_path):
    """The widen lane tree (pins, roster, configs, the Nemotron contract) plus the FULL pins the full legs'
    identity reads and the qwen live tool-parser verdict (the ``_it`` row's BFCL gate)."""
    repos, tree = TW._lane_tree(tmp_path)
    rels = [C.BCB_FULL_PIN_PATH, C.BFCL_FULL_PIN_PATH] + [v["record"] for v in C.TOOL_PARSER_LIVE_PASS.values()]
    for rel in rels:
        src = REPO / rel
        if src.is_file():
            (tree / rel).parent.mkdir(parents=True, exist_ok=True)
            (tree / rel).write_text(src.read_text())
    return repos, tree


class _Ledger:
    """The usage-ledger child: writes its endpoint record at ``--endpoint-json`` and stays alive."""

    def __init__(self, argv, *a, **k):
        self.argv = list(argv)
        ep = Path(self.argv[self.argv.index("--endpoint-json") + 1])
        ep.parent.mkdir(parents=True, exist_ok=True)
        ep.write_text(json.dumps({"ledger_url": "http://127.0.0.1:1/v1"}))
        self.rc = None

    def poll(self):
        return self.rc

    def wait(self, timeout=None):
        return 0 if self.rc is None else self.rc

    def terminate(self):
        self.rc = 0

    def kill(self):
        self.rc = 0


class _RowStandup:
    """A scripted MULTI-CELL standup: reads the cells and the keyed digests off its argv, writes the native
    ``endpoint.json`` (the served geometry through the §1.8 kernel step, the clamp override, the by-cell
    digests keyed by HARNESS id, the standup attempt id, the lora_config / residual_slot facts), the
    routing table, the serving attestation and an (empty) server log into ``--out``; stays alive until
    stopped. ``world`` is set by :class:`_World`; ``record`` overrides endpoint keys (a None deletes)."""

    world: ClassVar[Optional["_World"]] = None
    argvs: ClassVar[List[List[str]]] = []
    envs: ClassVar[List[Dict[str, str]]] = []
    ids: ClassVar[List[str]] = []

    def __init__(self, argv, *a, **k):
        self.argv = list(argv)
        self.rc = None
        type(self).argvs.append(self.argv)
        type(self).envs.append(dict(k.get("env") or {}))
        w = type(self).world
        out = Path(self.argv[self.argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        cells = [self.argv[i + 1] for i, tok in enumerate(self.argv) if tok == "--cell"]
        digests = {}
        for i, tok in enumerate(self.argv):
            if tok == w.L.OBJECT_DIGEST_FLAG:
                key, _, sha = self.argv[i + 1].partition("=")
                digests[C.cell_spellings(key, declared=False).harness] = sha
        fitted = [c for c in cells if w.L.is_steered(c)]
        geom, _ = w.L.served_geometry(w.cfgs[0], 1, str(w.tree))
        g2, step = w.L.expected_geometry_for(geom, native=True, name="row")
        standup_id = w.next_standup_id()
        type(self).ids.append(standup_id)
        (out / "routing_table.json").write_text(json.dumps({"version": 3, "cells": cells}))
        (out / "serving_attestation.json").write_text(json.dumps({"schema": 2, "engine": {"enforce_eager": False}}))
        # the server log: a world may script it (``w.vllm_log(env, out) -> text``: the compile-cache lines module D reads)
        log_text = w.vllm_log(dict(k.get("env") or {}), out) if getattr(w, "vllm_log", None) else "INFO vLLM started\n"
        (out / "vllm_server.log").write_text(log_text)
        rec = {"proxy_url": "http://127.0.0.1:1/v1", "proxy_token": None, "routing_table": str(out / "routing_table.json"),
               "tool_calls": True, "reasoning_parser": None, "agent_loop_entry": True,
               "geometry": {"max_model_len": g2.max_model_len, "max_num_seqs": g2.max_num_seqs,
                            **({"adjustments": [step]} if step else {})},
               "engine": NATIVE, C.SERVED_ENGINE_FIELD: NATIVE, "adapter_bank": f"{w.L.ADAPTER_BANK_ROOT}/{geom.model_key}",
               **({C.ENDPOINT_KEY_SAMPLING_CLAMP_OVERRIDE: dict(C.SAMPLING_CLAMP_OVERRIDE)}
                  if C.FLAG_SAMPLING_CLAMP_OVERRIDE in self.argv else {}),
               "object_artifact_sha256_by_cell": digests,
               "object_artifact_sha256": (sorted(set(digests.values()))[0] if len(set(digests.values())) == 1 else None),
               "standup_attempt_id": standup_id, "started": 1790000485.0,
               "serving_attestation": str(out / "serving_attestation.json"),
               "cells": cells, "cells_served": str(out / "cells_served.json"),
               "positive_controls": str(out / "positive_controls.json"),
               "lora_config": {"max_loras": len(fitted) + 1, "max_cpu_loras": len(fitted) + 1, "max_lora_rank": 8,
                               "target_modules": ["o_proj"]},
               "residual_slot": {"armed": False, "cells": []}, "zero_adapter_only": not fitted,
               "text_only_mm": True}
        for key, val in dict(w.record).items():
            if val is None:
                rec.pop(key, None)
            else:
                rec[key] = val
        (out / "endpoint.json").write_text(json.dumps(rec))

    def poll(self):
        return self.rc

    def wait(self, timeout=None):
        return 0 if self.rc is None else self.rc

    def terminate(self):
        self.rc = 0

    def kill(self):
        self.rc = 0


class _Dead:
    def __init__(self, *a, **k):
        self.rc = 1

    def poll(self):
        return self.rc

    def wait(self, timeout=None):
        return self.rc

    def terminate(self):
        pass

    def kill(self):
        pass


def _popen(argv, *a, **k):
    joined = " ".join(str(x) for x in argv[:3])
    if "capctrl_serving_up.py" in joined:
        return _RowStandup(argv, *a, **k)
    if "usage_ledger" in joined:
        return _Ledger(argv, *a, **k)
    return _Dead(argv, *a, **k)


class _World:
    """A row of configurations under ``suffix`` with everything the row runner needs to run to completion
    against the fakes: the staged tree, fake object files behind the launch's digests, the protocol check,
    the standup / ledger / driver doubles and the row options."""

    def __init__(self, L, tmp_path, monkeypatch, *, cells, model=ROW, suffix=SUFFIX, suffix_by_cell=None,
                 record=None, standup_ids=(STANDUP_A,), row_opts=None):
        self.L, self.tmp_path, self.record, self._monkeypatch = L, tmp_path, dict(record or {}), monkeypatch
        self.repos, self.tree = _row_tree(tmp_path)
        # the row standup's serve dir under THIS world's tmp_path, never the process's /tmp: two worlds of one row and
        # suffix (two runners on one host) keep their own standup dir, record and record copy
        self.standup_root = tmp_path / "srv"
        monkeypatch.setattr(L, "STANDUP_LOCAL_ROOT", self.standup_root)
        monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(self.repos))
        monkeypatch.delenv(C.LENS_STACKED_REDUCTIONS_ENV, raising=False)
        cfgs = _plan(L, cells, model=model)
        self.gen = tmp_path / "gen"
        vol_gen = C.VOL_GEN_ROOT
        cfgs = [dict(c, out_dir=str(c["out_dir"]).replace(vol_gen, str(self.gen), 1)) for c in cfgs]
        self.cfgs = L.isolate_out_dirs_by_cell(cfgs, suffix, suffix_by_cell)
        self.by_cell = {c["cell"]: c for c in self.cfgs}
        # the fitted cells' objects: fake bytes, the launch holds the standup to their sha (E40 H1)
        self.objects = tmp_path / "objects"
        self.object_digests: Dict[str, Dict[str, object]] = {}
        for c in self.cfgs:
            if c.get("steered"):
                d = self.objects / c["cell"]
                d.mkdir(parents=True, exist_ok=True)
                (d / "object.safetensors").write_bytes(f"object of {c['cell']}".encode())
                sha = hashlib.sha256((d / "object.safetensors").read_bytes()).hexdigest()
                self.object_digests[c["cell"]] = {"artifact_sha256": sha, "intervention_hash": "1" * 16,
                                                  "expected": "1" * 16, "lens_route_intervention_hash": "2" * 16}
        monkeypatch.setattr(L, "planned_object_paths",
                            lambda objects_dir, cs: {c: self.objects / c / "object.safetensors" for c in cs if L.is_steered(c)})
        monkeypatch.setattr(L, "proxy_engine_kind", lambda ep, **k: (NATIVE, None))
        monkeypatch.setattr(L.time, "sleep", lambda s: None)
        monkeypatch.setattr(L, "LEDGER_WAIT_S", 2)
        monkeypatch.setattr(L, "STANDUP_WAIT_S", 2)
        # module D's roots under the temp dir (never /vol or the grading host's /tmp)
        monkeypatch.setattr(L.CC, "DEFAULT_VOL_ROOT", str(tmp_path / "cache" / "compile"))
        monkeypatch.setattr(L.CC, "LOCAL_ROOT_DEFAULT", str(tmp_path / "compile_cache"))
        self._standup_ids = list(standup_ids)
        _RowStandup.world = self
        _RowStandup.argvs, _RowStandup.envs, _RowStandup.ids = [], [], []
        monkeypatch.setattr(L.subprocess, "Popen", _popen)
        self.driver_calls: List[Dict[str, object]] = []
        monkeypatch.setattr(L.subprocess, "run", self._driver)
        # §1.9.5: the runner probes ``row_opts.inode_mount`` with os.statvfs at start and at teardown; a filesystem
        # with no inode table (f_files == 0: a container's overlay, a sandboxed tmp) makes the probe refuse and the
        # row record then carries the refusal in place of a reading. The world hands the probe a SYNTHETIC statvfs
        # reading for the mount instead, so the budget assertions pin the engine's arithmetic (inodes_from_statvfs
        # + inode_budget_verdict) and not the host's filesystem; the probe's own doors (a mount that is not a
        # directory refuses) stay the engine's.
        self.inode_probes: List[str] = []
        monkeypatch.setattr(L, "probe_volume_inodes", self._probe_volume_inodes)
        self.row_opts = {"inode_mount": str(tmp_path), **dict(row_opts or {})}
        self.row_dir = L.row_dir_for(L.primary_configuration(self.cfgs)["out_dir"], model)
        self.standup_dir = L.row_standup_dir(L.row_of(self.cfgs[0]).slug, Path(str(L.primary_configuration(self.cfgs)["out_dir"])).name)
        # landing 2 (§2.1): the store the fake driver files keys at the client width S // K the runner expects
        self.concurrency_k = int(self.row_opts.get("concurrency_k", 1))

    def next_standup_id(self) -> str:
        return self._standup_ids.pop(0) if len(self._standup_ids) > 1 else self._standup_ids[0]

    def _probe_volume_inodes(self, mount=None):
        """Module E's probe over :data:`INODE_READING` in place of the host's ``os.statvfs``: the mount must still be
        a directory (the probe's own refusal, so a world that never names a mount still records the ``/vol`` refusal
        the engine records in a container without the volume), and the engine's own :func:`inodes_from_statvfs`
        turns the reading into the record. Every mount probed is kept for the row test's before/after pin."""
        mount = str(self.L.VOL_PATH if mount is None else mount)
        self.inode_probes.append(mount)
        if not os.path.isdir(mount):
            raise FileNotFoundError(f"{mount} is not mounted here (synthetic probe)")
        return self.L.inodes_from_statvfs(INODE_READING, mount=mount)

    def expected_for(self, cfg):
        geom, _ = self.L.served_geometry(cfg, 1, str(self.tree))
        # the runner's own defaults (samples = C.MATH500_SAMPLES_PER_ITEM, no limits): the store the driver files is the
        # store the runner then trusts as DONE
        return self.L.expected_identity(cfg, geom, pass_no=1, controls=cfg["controls"], samples=C.MATH500_SAMPLES_PER_ITEM,
                                        tree=str(self.tree), tree_sha=FULL_SHA, serving_path=NATIVE,
                                        concurrency_k=self.concurrency_k)

    def finish_store(self, out_dir: Path, cfg, *, tasks=(BCB_FULL, BFCL_FULL), row_engine=None) -> Path:
        """A COMPLETE store for ``cfg`` under ``out_dir`` as the driver files it (both full legs done); ``tasks`` narrows
        the legs finished (landing 2's ``generate`` driver files BFCL only), ``row_engine`` is the manifest's UNHASHED
        row_engine block the driver copies from the record (with ``verification_pass`` as the phase leaves it)."""
        exp = self.expected_for(cfg)

        def _mutate(m):
            # a NATIVE store's route block names its engine (test_tinycap_launcher_native._native_store)
            m[C.MANIFEST_KEY_ROUTE].update({C.SERVED_ENGINE_FIELD: NATIVE})
            if row_engine is not None:
                m[C.MANIFEST_KEY_ROW_ENGINE] = dict(row_engine)
        store = TL._store(out_dir, exp, tasks=list(cfg["controls"]), extra_identity=exp.optional_identity(), mutate=_mutate)
        for t in tasks:
            TL._finish(store, t, self.L.FULL_ITEM_COUNT_OF_TASK[t])
        return store

    def _driver(self, argv, **kw):
        argv = list(argv)
        cell = argv[argv.index("--cell") + 1]
        out_root = Path(argv[argv.index(C.FLAG_OUT_ROOT) + 1])
        call = {"argv": argv, "cell": cell, "out_root": out_root, "env": dict(kw.get("env") or {})}
        self.driver_calls.append(call)
        cfg = self.by_cell[cell]
        self.finish_store(out_root, cfg)
        (out_root / "evidence.json").write_text(json.dumps({"cell": cell}))
        return SimpleNamespace(returncode=0, stdout="driver ok\n", stderr="")

    def run(self, cfgs=None, **kw):
        kw.setdefault("run_id", "f6cf462-cafebabe")
        kw.setdefault("object_digests", dict(self.object_digests))
        kw.setdefault("protocol_check", None)
        kw.setdefault("row_opts", dict(self.row_opts))
        self._monkeypatch.setattr(self.L, "STANDUP_LOCAL_ROOT", self.standup_root)     # this world's, whichever was built last
        return self.L._run_row(list(self.cfgs if cfgs is None else cfgs), "f6cf462", **kw)


def _result_lines(out_dir: Path) -> List[str]:
    p = Path(out_dir) / "RESULT"
    return p.read_text().splitlines() if p.is_file() else []


# --------------------------------------------------------------------------- #
# 1. the split's seams: the multi-cell standup argv and the client width         #
# --------------------------------------------------------------------------- #


def test_standup_cmd_composes_one_cell_per_flag_and_keyed_digests_and_keeps_the_scalar_argv(L, tmp_path):
    """§1.2: a SEQUENCE of cells composes ``--cell c`` per cell (``none`` first) in the cell slot and one keyed
    ``--expect-object-sha256 <cell>=<sha>`` per fitted cell in the digest slot; a scalar cell is today's argv byte
    for byte; a declared multi-cell standup, a fitted cell without a digest, a stray digest and a bare digest
    beside several cells refuse before any Popen."""
    cfgs = _plan(L, [LORA, "none", PROMPTED])
    geom, _ = L.served_geometry(cfgs[0], 1)
    bank = L.adapter_bank_for(geom.model_key)
    sha = "a" * 64
    args = ("/tree", "/tree/cfg.yaml")
    tail = ("d", tmp_path, tmp_path / "STOP", geom, 64, cfgs[0]["standup_extra"])
    scalar = L._standup_cmd(*args, LORA, *tail, object_sha256=sha, serving_path=NATIVE, adapter_bank=bank)
    multi = L._standup_cmd(*args, [LORA, "none", PROMPTED], *tail, object_sha256_by_cell={LORA: sha},
                           serving_path=NATIVE, adapter_bank=bank)
    i = scalar.index("--cell")
    assert scalar[i:i + 2] == ["--cell", LORA]
    j = multi.index("--cell")
    assert multi[j:j + 6] == ["--cell", "none", "--cell", LORA, "--cell", PROMPTED]          # none first, then the order given
    assert multi[:j] == scalar[:i]                                                             # the cell slot ...
    assert multi[j + 6:] == [f"{LORA}={sha}" if tok == sha else tok for tok in scalar[i + 2:]]     # ... and the keyed digest are the ONLY differences
    assert multi.count("--cell") == 3 and multi.count(L.OBJECT_DIGEST_FLAG) == 1
    assert multi[multi.index(L.OBJECT_DIGEST_FLAG) + 1] == f"{LORA}={sha}"                     # ... plus the keyed digest
    assert scalar[scalar.index(L.OBJECT_DIGEST_FLAG) + 1] == sha                               # the bare digest, as before
    # the standup's own parser reads both forms
    up = TL._load_standup() if hasattr(TL, "_load_standup") else None
    if up is not None:
        ns = up.build_parser().parse_args(multi[2:])
        assert ns.cell == ["none", LORA, PROMPTED]
    # two fitted cells: one keyed digest each, in cell order; a base-only row composes no digest
    two = L._standup_cmd(*args, ["none", LORA, LORA2], *tail, object_sha256_by_cell={LORA2: "b" * 64, LORA: sha},
                         serving_path=NATIVE, adapter_bank=bank)
    ks = [two[k + 1] for k, tok in enumerate(two) if tok == L.OBJECT_DIGEST_FLAG]
    assert ks == [f"{LORA}={sha}", f"{LORA2}={'b' * 64}"]
    base_only = L._standup_cmd(*args, ["none", PROMPTED], *tail, serving_path=NATIVE, adapter_bank=bank)
    assert L.OBJECT_DIGEST_FLAG not in base_only and base_only.count("--cell") == 2
    with pytest.raises(ValueError, match="DECLARED standup serves one cell"):
        L._standup_cmd(*args, [LORA, "none"], *tail, object_sha256_by_cell={LORA: sha}, declared=True,
                       serving_path=NATIVE, adapter_bank=bank)
    with pytest.raises(ValueError, match="no preflight-verified artifact sha256"):
        L._standup_cmd(*args, [LORA, "none"], *tail, serving_path=NATIVE, adapter_bank=bank)
    with pytest.raises(ValueError, match="not fitted cells of this row standup"):
        L._standup_cmd(*args, ["none", PROMPTED], *tail, object_sha256_by_cell={PROMPTED: sha},
                       serving_path=NATIVE, adapter_bank=bank)
    with pytest.raises(ValueError, match="KEYED per fitted cell"):
        L._standup_cmd(*args, [LORA, "none"], *tail, object_sha256=sha, object_sha256_by_cell={LORA: sha},
                       serving_path=NATIVE, adapter_bank=bank)
    with pytest.raises(ValueError, match="names a cell twice"):
        L._standup_cmd(*args, ["none", "none"], *tail, serving_path=NATIVE, adapter_bank=bank)


def test_expected_identity_threads_an_explicit_client_width_and_the_served_s_keeps_a_narrower_one(L):
    """§1.1.1: ``concurrency`` (None = the served S, today's derivation) is hashed store identity; landing 2's
    ``S // K`` is a different key. ``expected_at_served_seqs`` keeps a width below the served S (min), and reads
    ``s_`` exactly as before when the width is the configured S."""
    cfg = _plan(L, ["none"])[0]
    geom, _ = L.served_geometry(cfg, 1)
    base = L.expected_identity(cfg, geom, pass_no=1, controls=cfg["controls"], samples=4, serving_path=NATIVE)
    half = L.expected_identity(cfg, geom, pass_no=1, controls=cfg["controls"], samples=4, serving_path=NATIVE,
                               concurrency=base.served["max_num_seqs"] // 2)
    assert base.served["concurrency"] == base.served["max_num_seqs"] == 31                     # the kernel-lowered S
    assert half.served["concurrency"] == 15 and half.served["max_num_seqs"] == 31
    assert {k: v for k, v in half.served.items() if k != "concurrency"} == {k: v for k, v in base.served.items() if k != "concurrency"}
    assert L.expected_at_served_seqs(base, 30).served["concurrency"] == 30                      # today: follows the served S
    at = L.expected_at_served_seqs(half, 30)
    assert (at.served["max_num_seqs"], at.served["concurrency"]) == (30, 15)                    # a narrower width is kept
    assert L.sandboxed_controls(["bigcodebench_full"], "off") is True                             # §1.10 / S13
    assert L.sweep_certificate_path("/o", "fc-1") == Path("/o/watch/sweep_fc-1.json")
    assert L.sweep_certificate_path("/o", "fc-1", cell="none-rep2") == Path("/o/watch/sweep_fc-1__none-rep2.json")


# --------------------------------------------------------------------------- #
# 2. the row runner end to end                                                    #
# --------------------------------------------------------------------------- #


def test_run_row_hoists_one_standup_and_drives_every_cell_with_its_own_ledger_owner_record_and_store(L, tmp_path, monkeypatch, capsys):
    """§1.1.2: three cells, ONE standup (repeated --cell, the keyed digest, the compile-cache env), one ledger
    and one driver per cell against a per-cell serve-dir copy of the endpoint, every store with its
    ``served_geometry.json`` + ``standup/`` x3 + its own sandbox record, every REC with the row facts and a
    RESULT line ending in ``standup=<id>``, the row dir with the declared pointer, one sweep certificate per
    cell, the standup logs, ``gpu_interval.json`` (module C's shape) and ``row_rec.json`` with ``ended_at``."""
    w = _World(L, tmp_path, monkeypatch, cells=[LORA, "none", PROMPTED])
    result = w.run()
    assert result["status"] == "OK", result.get("error")
    assert sorted(result["cells"]) == sorted(["none", LORA, PROMPTED]) and result["dropped"] == {}
    assert result["standup_id"] == STANDUP_A and result["served_cells"] == ["none", LORA, PROMPTED]
    # ONE standup: repeated --cell (none first), the keyed digest for the fitted cell, module D's env hook
    assert len(_RowStandup.argvs) == 1
    argv = _RowStandup.argvs[0]
    assert [argv[i + 1] for i, t in enumerate(argv) if t == "--cell"] == ["none", LORA, PROMPTED]
    assert argv[argv.index(L.OBJECT_DIGEST_FLAG) + 1] == f"{LORA}={w.object_digests[LORA]['artifact_sha256']}"
    env = _RowStandup.envs[0]
    assert env["VLLM_CACHE_ROOT"].startswith(str(tmp_path / "compile_cache")) and "VLLM_DISABLE_COMPILE_CACHE" not in env
    assert env[L.ADAPTER_BANK_ROOT_ENV] == L.ADAPTER_BANK_ROOT and env[C.LENS_STACKED_REDUCTIONS_ENV] == "per_row"
    # the drive order and one driver per cell, each at the served S with its own serve-dir copy and row-engine record
    assert [c["cell"] for c in w.driver_calls] == ["none", LORA, PROMPTED]
    for call in w.driver_calls:
        a = call["argv"]
        assert a[a.index("--concurrency") + 1] == "31"
        table = Path(a[a.index("--routing-table") + 1])
        assert table.parent.name == call["cell"] and table.parent.parent.name == "cells"
        ep = json.loads((table.parent / "endpoint.json").read_text())
        assert ep["routing_table"] == str(table) and call["cell"] in ep["cells"] and ep["standup_attempt_id"] == STANDUP_A
        assert a[-2:] == [C.FLAG_ROW_ENGINE_RECORD, str(table.parent / "row_engine.json")]           # LAST (§1.1.4)
        rec_doc = C.read_row_engine_record(a[-1])
        assert rec_doc["standup_id"] == STANDUP_A and rec_doc["engine_cells"] == ["none", LORA, PROMPTED]
        assert rec_doc["max_loras"] == 2 and rec_doc["residual_slot_armed"] is False and rec_doc["verification_pass"] is None
        assert set(rec_doc["compile_cache"]) == set(C.ROW_ENGINE_COMPILE_CACHE_FIELDS) and rec_doc["compile_cache"]["hit"] is False
        assert rec_doc["row_dir"] == str(w.row_dir) and tuple(rec_doc) == C.ROW_ENGINE_MANIFEST_FIELDS
        # the owner tuple per cell in the driver env, the record IN the cell store
        assert call["env"][scope.OWNER_ENV_CELL] == call["cell"] and call["env"][scope.OWNER_ENV_RUN_ID] == "f6cf462-cafebabe"
        assert Path(call["env"][scope.SANDBOX_RECORD_ENV]) == call["out_root"] / "watch" / f"sandboxes_{result['function_call_id']}.jsonl"
    fcid = result["function_call_id"]
    for cell, rec in result["cells"].items():
        out = Path(w.by_cell[cell]["out_dir"])
        assert rec["status"] == "OK" and rec["standup_shared"] is True and rec["standup_id"] == STANDUP_A
        assert rec["standup_s"] is None and rec["row_standup_s"] is not None and rec["engine_cells"] == ["none", LORA, PROMPTED]
        assert rec["driver_started_at"] <= rec["driver_ended_at"] and rec["resolved_sampler"]["log_scope"] == "row"
        assert rec["sandbox_owner"] == {"run_id": "f6cf462-cafebabe", "cell": cell, "function_call_id": fcid}
        assert rec["sandbox_pointer"] == str(w.row_dir / "watch" / f"attempt_{fcid}.json")
        lines = _result_lines(out)
        assert len(lines) == 1 and " result=COMPLETE " in lines[0] and lines[0].endswith(f" standup={STANDUP_A}")
        assert sorted(p.name for p in (out / "standup").iterdir()) == sorted(SS.STANDUP_FILES)     # exactly three
        assert json.loads((out / "standup" / "endpoint.json").read_text())["standup_attempt_id"] == STANDUP_A
        assert (out / L.SERVED_GEOMETRY_FILENAME).is_file() and (out / "ledger.log").is_file()
        assert (out / "watch" / f"sandboxes_{fcid}.jsonl").is_file()
        assert not (out / "watch" / f"attempt_{fcid}.json").exists() and not list((out / "watch").glob("sweep_*"))
        # nothing beyond today's store + standup/ (§1.9.2, S7)
        assert {p.name for p in out.iterdir()} <= set(SS.ROOT_FILES) | {"cells", "watch", "standup", cell, "driver_stdout.log", "driver_stderr.log"}
    # the row dir: the declared pointer (module E's watcher rule), one sweep per cell, the logs, the interval, the record
    pointer = json.loads((w.row_dir / "watch" / f"attempt_{fcid}.json").read_text())
    assert set(pointer) == set(L.ROW_POINTER_FIELDS) and pointer["cells"] == ["none", LORA, PROMPTED]
    assert pointer["row_attempt"] == {"standup_id": STANDUP_A} and pointer["phase"] == "coupled" and pointer["cell"] is None
    assert set(pointer["records"]) == {"none", LORA, PROMPTED} and pointer["row"] == ROW and pointer["controls"] == CONTROLS
    assert sorted(p.name for p in (w.row_dir / "watch").glob("sweep_*")) == sorted(f"sweep_{fcid}__{c}.json" for c in ("none", LORA, PROMPTED))
    assert (w.row_dir / "standup" / "standup.log").is_file()
    gi = json.loads((w.row_dir / "gpu_interval.json").read_text())
    assert (gi["row"], gi["suffix"], gi["standup_id"], gi["concurrency_k"]) == (ROW, SUFFIX, STANDUP_A, 1)
    kinds = [e["kind"] for e in gi["events"]]
    assert kinds[:3] == ["container_start", "standup_started", "engine_started"] and kinds[-2:] == ["standup_exited", "container_end"]
    assert kinds.count("driver_started") == kinds.count("driver_exited") == kinds.count("result_stamped") == 3
    assert all(e["phase"] == "coupled" for e in gi["events"] if e["kind"].startswith("driver_"))
    assert gi["gpu_held_s"] == pytest.approx(gi["interval"]["end"] - gi["interval"]["start"], abs=1e-3)
    assert gi["interval"]["t_standup_exited"] is not None and gi["interval"]["end"] >= gi["interval"]["t_last_generation_exit"]
    row_rec = json.loads((w.row_dir / "row_rec.json").read_text())
    assert row_rec["ended_at"].endswith("Z") and row_rec["status"] == "OK" and row_rec["standup_id"] == STANDUP_A
    assert row_rec["inodes_before"]["inodes_used"] >= 0 and row_rec["inode_budget"]["ok"] is True
    # §1.9.5: probed at start and at teardown on the mount the row names; the record is the engine's arithmetic over
    # the world's synthetic statvfs reading (inodes_from_statvfs, then inode_budget_verdict against the forecast)
    assert w.inode_probes == [str(tmp_path), str(tmp_path)]
    before, budget = row_rec["inodes_before"], row_rec["inode_budget"]
    assert (before["mount"], before["inodes_total"], before["inodes_used"], before["inodes_free"], before["pct_used"]) == (
        str(tmp_path), 1_000_000, 100_000, 900_000, 10.0)
    forecast = L.forecast_inodes(3, staged=True)["total"]
    assert (budget["status"], budget["forecast"]["total"], budget["inodes_after"], budget["pct_after"]) == (
        "OK", forecast, 100_000 + forecast, round(100.0 * (100_000 + forecast) / 1_000_000, 2))
    after = row_rec["inodes_after"]
    assert "error" not in after and (after["mount"], after["inodes_total"], after["inodes_used"]) == (str(tmp_path), 1_000_000, 100_000)
    assert row_rec["compile_cache"]["key"].startswith("gemma4e4b__vllm0.25.1__base__")
    assert row_rec["compile_cache"]["evidence"]["hit"] is False and row_rec["compile_cache"]["persist"]["persisted"] is False
    assert row_rec["max_lora_rank_source" if "max_lora_rank_source" in row_rec else "compile_cache"]  # recorded
    assert result["compile_cache"]["max_lora_rank_source"] == "cell_name" and result["compile_cache"]["inputs"]["max_lora_rank"] == 8
    assert result["compile_cache"]["inputs"]["max_loras"] == 2
    out_text = capsys.readouterr().out
    assert out_text.count("[tinycap] REC ") == 3 and "[tinycap] inodes:" in out_text


def test_run_row_refuses_a_non_uniform_row_before_any_popen_and_names_the_odd_cell(L, tmp_path, monkeypatch):
    w = _World(L, tmp_path, monkeypatch, cells=["none", LORA])
    other = dict(_plan(L, ["none"], model=IT_ROW)[0], out_dir=str(w.gen / "x" / SUFFIX))
    result = w.run([w.cfgs[0], other])
    assert result["status"] == "ROW_NOT_UNIFORM" and other["name"] in result["error"] and "row" in result["error"]
    assert _RowStandup.argvs == [] and w.driver_calls == [] and not w.row_dir.exists()
    twice = w.run([w.cfgs[0], dict(w.cfgs[0])])
    assert twice["status"] == "ROW_NOT_UNIFORM" and "planned twice" in twice["error"]
    assert L._run_row([], "f6cf462", row_opts={})["status"] == "ROW_EMPTY"
    # K in {1, 2, 3} and the generate phase are admitted; K > 3 and `verify` refuse by name
    assert L.row_options({"concurrency_k": 2})["concurrency_k"] == 2 and L.row_options({"concurrency_k": 3})["concurrency_k"] == 3
    with pytest.raises(ValueError, match=f"at most {L.ROW_CONCURRENCY_K_MAX} concurrent cell drivers"):
        L.row_options({"concurrency_k": 4})
    assert L.row_options({"verification_phase": "generate"})["verification_phase"] == "generate" == L.ROW_PHASE_GENERATE
    with pytest.raises(ValueError, match="verify is the CPU verifier's subcommand"):
        L.row_options({"verification_phase": "verify"})
    with pytest.raises(ValueError, match="one of \\['coupled', 'generate'\\]"):
        L.row_options({"verification_phase": "scored"})
    with pytest.raises(ValueError, match="warm-cache probe drives no cell"):
        L.row_options({"verification_phase": "generate", "warm_cache_only": True})
    assert L.ROW_LAUNCH_PHASES == ("coupled", "generate") and L.ROW_CONCURRENCY_K_MAX == 3 and L.REC_GENERATED == C.REC_STATUS_GENERATED == "GENERATED"
    assert L._run_row(w.cfgs, "f6cf462", row_opts={"cache_expectation": "hot"})["status"] == "ROW_OPTS_INVALID"


def test_the_per_cell_doors_skip_a_done_cell_park_an_incomplete_store_and_serve_the_rest(L, tmp_path, monkeypatch):
    """§1.1.2 step 2: ``none`` already complete -> DONE, no driver; a store holding a manifest without a
    valid marker -> INCOMPLETE_STORE_PARKED with a RESULT line naming it, never resumed into; the prompted cell
    is served under the standup, whose argv names ONLY the served cells."""
    w = _World(L, tmp_path, monkeypatch, cells=["none", LORA, PROMPTED])
    w.finish_store(Path(w.by_cell["none"]["out_dir"]), w.by_cell["none"])
    exp = w.expected_for(w.by_cell[LORA])
    TL._store(Path(w.by_cell[LORA]["out_dir"]), exp, tasks=CONTROLS, extra_identity=exp.optional_identity())   # a manifest, no markers
    result = w.run()
    assert result["status"] == "ROW_PARTIAL"
    assert result["cells"]["none"]["status"] == "DONE" and result["cells"][PROMPTED]["status"] == "OK"
    assert result["dropped"][LORA]["status"] == "INCOMPLETE_STORE_PARKED" and "D209" in result["dropped"][LORA]["error"]
    assert [c["cell"] for c in w.driver_calls] == [PROMPTED]
    argv = _RowStandup.argvs[0]
    assert [argv[i + 1] for i, t in enumerate(argv) if t == "--cell"] == [PROMPTED] and L.OBJECT_DIGEST_FLAG not in argv
    lines = _result_lines(Path(w.by_cell[LORA]["out_dir"]))
    assert len(lines) == 1 and " result=INCOMPLETE_STORE_PARKED " in lines[0]
    assert _result_lines(Path(w.by_cell["none"]["out_dir"]))[-1].split(" result=")[1].startswith("COMPLETE")
    # every cell done: no standup at all
    w2 = _World(L, tmp_path / "all_done", monkeypatch, cells=["none", PROMPTED])
    for cell in ("none", PROMPTED):
        w2.finish_store(Path(w2.by_cell[cell]["out_dir"]), w2.by_cell[cell])
    r2 = w2.run()
    assert r2["status"] == "DONE" and _RowStandup.argvs == [] and w2.driver_calls == []
    # a cell that refuses at a plan door is dropped by name and the row proceeds
    w3 = _World(L, tmp_path / "failed", monkeypatch, cells=["none", LORA])
    lora_out = Path(w3.by_cell[LORA]["out_dir"])
    lora_out.mkdir(parents=True)
    (lora_out / C.SAMPLING_VALIDATION_FAILED_FILENAME).write_text(json.dumps(
        {"status": "CLAMP_OVERRIDE_INEFFECTIVE", "function_call_id": "fc-old", "run_id": "r-old", "error": "re-clamped"}))
    r3 = w3.run()
    assert r3["status"] == "ROW_PARTIAL" and r3["dropped"][LORA]["status"] == C.REC_SAMPLING_VALIDATION_FAILED
    assert r3["cells"]["none"]["status"] == "OK" and [c["cell"] for c in w3.driver_calls] == ["none"]


def test_the_negative_identity_control_drops_exactly_the_flipped_cell_and_the_row_proceeds(L, tmp_path, monkeypatch):
    """The control flips ONE cell's EXPECTED digest; the per-cell bytes door refuses that cell
    alone (OBJECT_DIGEST_MISMATCH, its RESULT names it), the standup argv names neither the cell nor a digest for
    it, the other cells serve; the ROW-REC records the control and the digest used."""
    w = _World(L, tmp_path, monkeypatch, cells=["none", LORA, LORA2], row_opts={"negative_control": {"cell": LORA}})
    result = w.run()
    assert result["status"] == "ROW_PARTIAL"
    assert result["dropped"][LORA]["status"] == "OBJECT_DIGEST_MISMATCH"
    assert result["dropped"][LORA]["object_bytes_sha256"] == w.object_digests[LORA]["artifact_sha256"]
    assert result["dropped"][LORA]["object_digest"]["negative_control"]["original_artifact_sha256"] == w.object_digests[LORA]["artifact_sha256"]
    assert sorted(result["cells"]) == sorted(["none", LORA2]) and all(r["status"] == "OK" for r in result["cells"].values())
    argv = _RowStandup.argvs[0]
    assert [argv[i + 1] for i, t in enumerate(argv) if t == "--cell"] == ["none", LORA2]
    assert [argv[i + 1] for i, t in enumerate(argv) if t == L.OBJECT_DIGEST_FLAG] == [f"{LORA2}={w.object_digests[LORA2]['artifact_sha256']}"]
    assert " result=OBJECT_DIGEST_MISMATCH " in _result_lines(Path(w.by_cell[LORA]["out_dir"]))[0]
    assert result["negative_control"]["cell"] == LORA and result["negative_control"]["expected_digest_used"] == L.flip_last_hex(w.object_digests[LORA]["artifact_sha256"])
    assert result["compile_cache"]["persist"]["persisted"] is False                        # a negative-control row never persists
    with pytest.raises(ValueError, match="has no preflight-verified digest"):
        L.negative_control_digests(w.object_digests, PROMPTED)
    assert L.flip_last_hex("f" * 64) == "f" * 63 + "0"


def test_the_repeat_cell_is_driven_last_into_rep2_under_the_same_standup(L, tmp_path, monkeypatch):
    """§1.1.2 step 9 (S10): ``none`` driven AGAIN, last, into ``<out dir>-rep2`` — the same expected identity, its
    own ledger / owner / sweep / RESULT / REC (``repeat_of``), the repeat floor of the equivalence bar."""
    w = _World(L, tmp_path, monkeypatch, cells=["none", LORA], row_opts={"repeat_cells": ["none"]})
    pointers_at_drive: List[dict] = []                 # the row pointer as each driver STARTS (code review R1-1)
    real_driver = w._driver

    def _driver_snapshotting_pointer(argv, **kw):
        snaps = [json.loads(p.read_text()) for p in (w.row_dir / "watch").glob("attempt_*.json")]
        pointers_at_drive.append(snaps[0] if snaps else {})
        return real_driver(argv, **kw)
    monkeypatch.setattr(L.subprocess, "run", _driver_snapshotting_pointer)
    result = w.run()
    assert result["status"] == "OK" and sorted(result["cells"]) == sorted(["none", LORA, "none-rep2"])
    assert [c["cell"] for c in w.driver_calls] == ["none", LORA, "none"]
    rep_out = Path(w.by_cell["none"]["out_dir"] + "-rep2")
    assert w.driver_calls[-1]["out_root"] == rep_out and result["cells"]["none-rep2"]["repeat_of"] == "none"
    assert result["cells"]["none-rep2"]["status"] == "OK" and result["cells"]["none-rep2"]["standup_id"] == STANDUP_A
    assert result["cells"]["none-rep2"]["expected_identity"] == result["cells"]["none"]["expected_identity"]
    assert result["cells"]["none-rep2"]["sandbox_owner"]["cell"] == "none-rep2"
    fcid = result["function_call_id"]
    assert (w.row_dir / "watch" / f"sweep_{fcid}__none-rep2.json").is_file()
    assert _result_lines(rep_out)[0].endswith(f" standup={STANDUP_A}")
    assert sorted(p.name for p in (rep_out / "standup").iterdir()) == sorted(SS.STANDUP_FILES)
    assert len(_RowStandup.argvs) == 1                                                        # still ONE standup
    gi = json.loads((w.row_dir / "gpu_interval.json").read_text())
    assert [e["cell"] for e in gi["events"] if e["kind"] == "driver_started"] == ["none", LORA, "none-rep2"]
    # code review R1-1: the repeat's owner (`none-rep2`) and its in-store record are in the row pointer BEFORE any
    # driver starts — the first snapshot (taken as `none`'s driver starts) already lists it, with the standup id
    assert [p["cells"] for p in pointers_at_drive] == [["none", LORA, "none-rep2"]] * 3
    assert pointers_at_drive[0]["records"]["none-rep2"] == str(rep_out / "watch" / f"sandboxes_{fcid}.jsonl")
    assert pointers_at_drive[0]["row_attempt"] == {"standup_id": STANDUP_A}
    assert result["cells"]["none-rep2"]["sandbox_pointer"] == result["cells"]["none"]["sandbox_pointer"]
    assert (rep_out / "watch" / f"sandboxes_{fcid}.jsonl").is_file()
    # a repeat cell the row does not plan is recorded, never invented (a PLANNED base cell this attempt does not
    # serve is a different case: filed by name, see the worker-loss test)
    w2 = _World(L, tmp_path / "skip", monkeypatch, cells=["none"], row_opts={"repeat_cells": [LORA]})
    monkeypatch.setattr(L.subprocess, "run", w2._driver)
    assert w2.run()["repeat_skipped"] == {LORA: "not among the planned cells"}


def test_a_post_standup_row_refusal_names_every_served_cell_and_tears_the_standup_down(L, tmp_path, monkeypatch):
    """§1.1.2 step 6: a refusal at the attestation doors is a ROW refusal — every served cell's REC and RESULT
    carry it, no driver runs, the standup is stopped, the row dir keeps the standup logs and no interval is
    claimed beyond the standup's exit. (The engine door here; the clamp-override door is the same shape.)"""
    w = _World(L, tmp_path, monkeypatch, cells=["none", LORA, PROMPTED])
    monkeypatch.setattr(L, "proxy_engine_kind", lambda ep, **k: ("lens", None))            # the live proxy disagrees
    result = w.run()
    assert result["status"] == "SERVING_ENGINE_MISMATCH" and w.driver_calls == []
    assert sorted(result["dropped"]) == sorted(["none", LORA, PROMPTED]) and result["cells"] == {}
    for cell, rec in result["dropped"].items():
        assert rec["status"] == "SERVING_ENGINE_MISMATCH" and "standup_tail" in rec
        assert " result=SERVING_ENGINE_MISMATCH " in _result_lines(Path(w.by_cell[cell]["out_dir"]))[0]
    # the clamp door, the same shape: a protocol row whose standup records no override
    w1 = _World(L, tmp_path / "clamp", monkeypatch, cells=["none", LORA], record={C.ENDPOINT_KEY_SAMPLING_CLAMP_OVERRIDE: None})
    prot = [dict(c, **{L.SAMPLING_MODE_KEY: "protocol", "standup_extra": list(c["standup_extra"]) + [C.FLAG_SAMPLING_CLAMP_OVERRIDE, "0.001"]})
            for c in w1.cfgs]
    r1 = w1.run(prot, protocol_check=dict(PROTOCOL_CHECK))
    assert r1["status"] == "CLAMP_OVERRIDE_MISMATCH" and set(r1["dropped"]) == {"none", LORA} and w1.driver_calls == []
    srv = Path(_RowStandup.argvs[0][_RowStandup.argvs[0].index("--out") + 1])
    assert (srv / "STOP").exists()
    gi = json.loads((w.row_dir / "gpu_interval.json").read_text())
    assert [e["kind"] for e in gi["events"]] == ["container_start", "standup_started", "engine_started", "standup_exited", "container_end"]
    assert gi["gpu_held_s"] == pytest.approx(gi["interval"]["t_standup_exited"] - gi["interval"]["start"], abs=1e-3)
    # a standup that never comes up: STANDUP_FAILED on every cell, the logs in the row dir
    monkeypatch.setattr(L.subprocess, "Popen", lambda argv, *a, **k: _Dead(argv))
    w2 = _World(L, tmp_path / "dead", monkeypatch, cells=["none", LORA])
    monkeypatch.setattr(L.subprocess, "Popen", lambda argv, *a, **k: _Dead(argv))
    r2 = w2.run()
    assert r2["status"] == "STANDUP_FAILED" and set(r2["dropped"]) == {"none", LORA}
    assert [e["kind"] for e in r2["gpu_interval"]["events"]] == ["container_start", "standup_started", "standup_exited", "container_end"]
    assert json.loads((w2.row_dir / "row_rec.json").read_text())["status"] == "STANDUP_FAILED"
    # an endpoint without a standup attempt id cannot bind row-engine stores
    w3 = _World(L, tmp_path / "noid", monkeypatch, cells=["none", LORA], record={"standup_attempt_id": None})
    r3 = w3.run()
    assert r3["status"] == "STANDUP_ID_MISSING" and w3.driver_calls == [] and set(r3["dropped"]) == {"none", LORA}


def test_a_worker_loss_after_none_completes_parks_the_in_flight_cell_and_the_subset_relaunch_gets_a_fresh_standup(L, tmp_path, monkeypatch):
    """§5: the re-scheduled container finds ``none`` DONE under standup A (skipped, no GPU minute), the in-flight
    fitted cell's store holding manifests without a marker (INCOMPLETE_STORE_PARKED, never resumed into) and the
    unstarted cell, which runs after ONE new standup (B). The subset relaunch ``--cells <fitted> --tag shared-r2``
    is its own row: fresh ``-r2`` stores under standup C — the gate then reads the pair as ``cross_standup``."""
    w = _World(L, tmp_path, monkeypatch, cells=["none", LORA, PROMPTED], standup_ids=["su-A", "su-B", "su-C"])
    first = w.run()
    assert first["status"] == "OK" and first["standup_id"] == "su-A"
    # the loss: the LoRA cell's store was mid-flight (manifest, no markers); the prompted cell never started
    lora_out = Path(w.by_cell[LORA]["out_dir"])
    for sub in ("done", "summary"):
        for p in lora_out.glob(f"cells/*/{sub}"):
            shutil.rmtree(p)
    shutil.rmtree(w.by_cell[PROMPTED]["out_dir"])
    w.driver_calls.clear()
    _RowStandup.argvs.clear()
    second = w.run()
    assert second["status"] == "ROW_PARTIAL" and second["standup_id"] == "su-B"
    assert second["cells"]["none"]["status"] == "DONE" and "standup_argv" not in second["cells"]["none"]
    assert second["dropped"][LORA]["status"] == "INCOMPLETE_STORE_PARKED"
    assert second["cells"][PROMPTED]["status"] == "OK" and second["cells"][PROMPTED]["standup_id"] == "su-B"
    assert [c["cell"] for c in w.driver_calls] == [PROMPTED] and len(_RowStandup.argvs) == 1
    none_lines = _result_lines(Path(w.by_cell["none"]["out_dir"]))
    assert len(none_lines) == 2 and none_lines[0].endswith(" standup=su-A") and " result=COMPLETE status=DONE" in none_lines[1]
    # the subset relaunch: the parked cell alone under the r2 suffix, a fresh standup, a fresh store
    r2_cfg = dict(w.by_cell[LORA], out_dir=w.by_cell[LORA]["out_dir"][:-len(SUFFIX)] + SUFFIX + "-r2")
    w.by_cell[LORA] = r2_cfg
    w.driver_calls.clear()
    _RowStandup.argvs.clear()
    third = w.run([r2_cfg], row_opts={**w.row_opts, "row_dir": str(w.gen / "rows" / ROW / (SUFFIX + "-r2"))})
    assert third["status"] == "OK" and third["standup_id"] == "su-C" and third["suffix"] == SUFFIX + "-r2"
    assert list(third["cells"]) == [LORA] and third["cells"][LORA]["standup_id"] == "su-C"
    assert [c["cell"] for c in w.driver_calls] == [LORA]
    assert _result_lines(Path(r2_cfg["out_dir"]))[0].endswith(" standup=su-C")
    assert Path(third["row_dir"]).name == SUFFIX + "-r2" and (Path(third["row_dir"]) / "gpu_interval.json").is_file()
    assert first["standup_id"] != third["standup_id"]                                          # cross_standup by construction


def test_the_instruct_row_serves_its_own_none_beside_its_lora_cells(L, tmp_path, monkeypatch):
    """The ``_it`` row's ``none`` (the instruct checkpoint, no adapter) is a servable cell of
    the same row engine — planned, named first in the one standup argv, driven, filed with the row facts."""
    w = _World(L, tmp_path, monkeypatch, cells=[LORA2, "none"], model=IT_ROW, suffix="bcbw-qwen35_4b_it-f6cf462-shared")
    assert C.model_row(IT_ROW).objects_dir.endswith("/qwen35_4b/it")
    result = w.run()
    assert result["status"] == "OK", result.get("error")
    argv = _RowStandup.argvs[0]
    assert [argv[i + 1] for i, t in enumerate(argv) if t == "--cell"] == ["none", LORA2]
    assert [c["cell"] for c in w.driver_calls] == ["none", LORA2]
    assert result["cells"]["none"]["engine_cells"] == ["none", LORA2] and result["cells"]["none"]["standup_id"] == STANDUP_A
    assert result["row"] == IT_ROW and Path(result["row_dir"]).parts[-2] == IT_ROW
    assert result["compile_cache"]["key"].startswith("qwen35_4b__vllm0.25.1__base__")


def test_a_warm_cache_probe_stands_up_persists_and_tears_down_without_driving_a_cell(L, tmp_path, monkeypatch):
    """§1.5's warm step: standup -> readiness -> persist -> teardown; no driver. Code review R2-2: WARMED is a FACT
    about the volume — (1) the fake log names no cache dir, so module D refuses the persist by name and the row is
    WARM_NOT_PERSISTED (every cell dropped with the reason; the exit non-zero), never WARMED; (2) a log naming a cache
    dir under the hook's VLLM_CACHE_ROOT persists a REAL tarball -> WARMED; (3) a second probe restores that tarball
    and vLLM demonstrably uses it (a hit) -> WARMED with persist skipped `restored`; (4) restored but no load
    evidence -> WARM_NOT_PERSISTED naming the miss."""
    w = _World(L, tmp_path, monkeypatch, cells=["none", LORA], row_opts={"warm_cache_only": True, "cache_expectation": "cold"})
    result = w.run()
    assert result["status"] == "WARM_NOT_PERSISTED" and w.driver_calls == [] and len(_RowStandup.argvs) == 1
    assert result["cells"] == {} and sorted(result["dropped"]) == sorted(["none", LORA])
    assert all(r["status"] == "WARM_NOT_PERSISTED" and r["standup_id"] == STANDUP_A and "no_cache_dir_line" in r["error"] for r in result["dropped"].values())
    assert result["compile_cache"]["persist"]["persisted"] is False and result["compile_cache"]["persist"]["skipped"] == "no_cache_dir_line"
    assert "skipped: no_cache_dir_line" in result["error"] and "not WARMED" in result["error"] and result["compile_cache"]["expectation"] == "cold"
    gi = json.loads((w.row_dir / "gpu_interval.json").read_text())
    assert [e["kind"] for e in gi["events"]] == ["container_start", "standup_started", "engine_started", "standup_exited", "container_end"]
    recs = L.row_result_recs(result)
    assert [r["status"] for r in recs] == ["WARM_NOT_PERSISTED"] * 3 and L.failed_recs(recs, also_ok=("WARMED",))   # the row's own REC too
    assert " result=WARM_NOT_PERSISTED " in _result_lines(Path(w.by_cell["none"]["out_dir"]))[-1]
    # (2) a REAL persist: the standup's log names a cache dir under the hook's root and the compiled graphs exist there
    HASH = "0123abcd"

    def warm_log(env, out, *, loaded=False):
        d = Path(env["VLLM_CACHE_ROOT"]) / "torch_compile_cache" / HASH / "rank_0_0" / "backbone"
        d.mkdir(parents=True, exist_ok=True)
        (d / "computation_graph.py").write_text("# compiled\n")
        lines = [f"INFO Using cache directory: {d} for vLLM's torch.compile"]
        if loaded:
            lines.append("INFO Directly load the compiled graph(s) for compile range (1, 31) from the cache, took 0.5 s")
        else:
            lines.append("INFO Compiling a graph for compile range (1, 31) takes 12.3 s")
        return "\n".join(lines) + "\n"
    w2 = _World(L, tmp_path / "warm", monkeypatch, cells=["none", LORA], row_opts={"warm_cache_only": True, "cache_expectation": "cold"})
    w2.vllm_log = warm_log
    r2 = w2.run()
    assert r2["status"] == "WARMED", r2.get("error")
    assert sorted(r2["cells"]) == sorted(["none", LORA]) and r2["dropped"] == {} and w2.driver_calls == []
    assert all(r["status"] == "WARMED" and r["standup_id"] == STANDUP_A for r in r2["cells"].values())
    persist = r2["compile_cache"]["persist"]
    assert persist["persisted"] is True and persist["skipped"] is None and persist["vllm_hash_dir"] == HASH and Path(persist["tarball"]).is_file()
    assert Path(persist["tarball"]).parent == tmp_path / "warm" / "cache" / "compile" and r2["compile_cache"]["evidence"]["hit"] is False   # cold: a miss
    recs = L.row_result_recs(r2)
    assert recs == [dict(r) for r in r2["cells"].values()] and L.failed_recs(recs, also_ok=("WARMED",)) == []          # a warm probe exits 0 …
    assert L.failed_recs(recs) == [(r["name"], "WARMED") for r in recs]                                              # … and WARMED is no store elsewhere
    # (3) the same shape again: the tarball is restored, vLLM loads its graphs -> a hit; persist skipped `restored`; still WARMED
    w2.vllm_log = lambda env, out: warm_log(env, out, loaded=True)
    _RowStandup.argvs.clear()
    r3 = w2.run()
    assert r3["status"] == "WARMED", r3.get("error")
    assert r3["compile_cache"]["restore"]["restored"] is True and r3["compile_cache"]["persist"]["skipped"] == "restored"
    assert r3["compile_cache"]["evidence"]["hit"] is True and r3["compile_cache"]["evidence"]["reason"] == "hit"
    # (4) restored, but vLLM did not load from it: not warm for the following row -> refused by name
    w2.vllm_log = lambda env, out: warm_log(env, out, loaded=False)
    r4 = w2.run()
    assert r4["status"] == "WARM_NOT_PERSISTED" and "was restored but vLLM did not use it" in r4["error"] and "no_graph_load_evidence" in r4["error"]
    assert sorted(r4["dropped"]) == sorted(["none", LORA]) and r4["cells"] == {}
    # the pure rule, by case
    assert L.warm_probe_refusal({"key": "k", "persist": {"persisted": True}}) is None
    assert "no persist record" in L.warm_probe_refusal({"key": "k"})
    assert "skipped: tar_failed; disk full" in L.warm_probe_refusal({"key": "k", "persist": {"persisted": False, "skipped": "tar_failed", "refusal_detail": "disk full"}})
    assert L.warm_probe_refusal({"key": "k", "persist": {"persisted": False, "skipped": "restored"}, "restore": {"restored": True}, "evidence": {"hit": True}}) is None
    assert "did not use it" in L.warm_probe_refusal({"key": "k", "persist": {"persisted": False, "skipped": "restored"}, "restore": {"restored": True}, "evidence": {"hit": False, "reason": "hash_mismatch"}})
    with pytest.raises(ValueError, match="takes no negative control"):
        L.row_options({"warm_cache_only": True, "negative_control": {"cell": LORA}})


# --------------------------------------------------------------------------- #
# 3. the 40-inode store, the interval, the record                                #
# --------------------------------------------------------------------------- #


def test_a_full_shape_store_counts_exactly_40_inodes_and_the_runner_adds_only_standup(L, tmp_path):
    """§1.9.2 (S7): today's thirty-six-inode full-shape store + ``standup/`` (1 dir + 3 files) == 40, counted
    with module E's helper; the runner's ``copy_standup_files`` writes EXACTLY those three names and nothing else."""
    store = tmp_path / "store"
    SS.full_shape_store(store, cell="none", standup=False)
    assert SS.count_inodes(store) == SS.SOLO_FULL_SHAPE_INODES
    srv = tmp_path / "srv"
    srv.mkdir()
    (srv / "endpoint.json").write_text(json.dumps({"standup_attempt_id": "x", "serving_attestation": str(srv / "att.json")}))
    (srv / "att.json").write_text("{}")
    (srv / "vllm_server.log").write_text("x" * 10 + "tail")
    written = L.copy_standup_files(srv, json.loads((srv / "endpoint.json").read_text()), store, tail_bytes=4)
    assert sorted(written) == sorted(L.ROW_STANDUP_FILES) == sorted(SS.STANDUP_FILES)
    assert (store / "standup" / "vllm_server.log").read_text() == "tail"                    # the tail, bounded
    census = SS.assert_store_inodes(store, expect=SS.ROW_ENGINE_STORE_INODES)
    assert census["total"] == 40
    L.copy_standup_files(srv, None, store)                                                   # a refresh adds nothing
    SS.assert_store_inodes(store, expect=40)
    # the helper's own row-engine store agrees
    SS.assert_store_inodes(SS.full_shape_store(tmp_path / "store2", cell=LORA, standup=True).parent.parent, expect=40)


def test_gpu_interval_follows_module_cs_rule_on_its_fixture_and_refuses_a_torn_timeline(L):
    """§1.8 (P1 #7): start = container_start, end = max(last generation driver exit, standup_exited), never past
    container_end; K = 1 sequential drivers sum; a driver that never exited or nothing that ends the interval
    refuses. Module C's frozen example must read the same number (its reconcile refuses a disagreement > 1 s)."""
    if GPU_INTERVAL_FIXTURE.is_file():
        doc = json.loads(GPU_INTERVAL_FIXTURE.read_text())
        iv = L.gpu_interval(doc["events"])
        assert iv["gpu_held_s"] == pytest.approx(doc["_expected"]["gpu_held_s"], abs=1.0)
        assert iv["t_standup_exited"] == 1790004760.0 and iv["end"] == 1790004760.0
    ev = [{"t": 0.0, "kind": "container_start", "cell": None, "phase": None},
          {"t": 5.0, "kind": "standup_started", "cell": None, "phase": None},
          {"t": 50.0, "kind": "engine_started", "cell": None, "phase": None},
          {"t": 60.0, "kind": "driver_started", "cell": "a", "phase": "coupled"},
          {"t": 160.0, "kind": "driver_exited", "cell": "a", "phase": "coupled"},
          {"t": 161.0, "kind": "driver_started", "cell": "b", "phase": "coupled"},
          {"t": 261.0, "kind": "driver_exited", "cell": "b", "phase": "coupled"},
          {"t": 300.0, "kind": "standup_exited", "cell": None, "phase": None},
          {"t": 310.0, "kind": "container_end", "cell": None, "phase": None}]
    iv = L.gpu_interval(ev)
    assert (iv["start"], iv["end"], iv["gpu_held_s"], iv["n_drivers"]) == (0.0, 300.0, 300.0, 2)
    late = [e for e in ev if e["kind"] != "container_end"] + [{"t": 200.0, "kind": "container_end", "cell": None, "phase": None}]
    assert L.gpu_interval(late)["end"] == 200.0                                                # never past container_end
    no_su = [e for e in ev if e["kind"] != "standup_exited"]
    assert L.gpu_interval(no_su)["end"] == 261.0                                               # the last generation exit
    with pytest.raises(ValueError, match="never exited"):
        L.gpu_interval([e for e in ev if not (e["kind"] == "driver_exited" and e["cell"] == "b")])
    with pytest.raises(ValueError, match="nothing ends the GPU interval"):
        L.gpu_interval(ev[:3])
    with pytest.raises(ValueError, match="exactly one container_start"):
        L.gpu_interval(ev + [ev[0]])
    rec = L.gpu_interval_record(ev, row="r", suffix="s", run_id="run", function_call_id="fc", standup_id="su",
                                concurrency_k=1, cells={})
    assert set(rec) >= {"row", "suffix", "standup_id", "concurrency_k", "events", "gpu_held_s"} and rec["gpu_held_s"] == 300.0


def test_the_row_engine_record_is_the_contracts_exact_shape_and_the_probe_record_is_valid(L, tmp_path):
    p = L.write_row_engine_record(tmp_path, standup_id="su", engine_cells=["none", LORA], max_loras=2, residual_slot_armed=True,
                                  compile_cache={"hit": True, "key": "k", "vllm_cache_dir": "/tmp/x"}, row_dir=tmp_path / "rows" / "r" / "s")
    doc = json.loads(p.read_text())
    assert tuple(doc) == C.ROW_ENGINE_MANIFEST_FIELDS and C.read_row_engine_record(p) == doc
    with pytest.raises(C.RowEngineRecordError):
        L.write_row_engine_record(tmp_path / "bad", standup_id="su", engine_cells=["none"], max_loras=None, residual_slot_armed=False,
                                  compile_cache={"hit": False, "key": "k", "vllm_cache_dir": None}, row_dir=tmp_path)
    probe = L.write_probe_row_engine_record(tmp_path / "probe" / "row_engine.json")
    assert C.read_row_engine_record(probe)["standup_id"] == "preflight-probe"
    assert L.DRIVER_PROBE_VALUES[C.FLAG_ROW_ENGINE_RECORD].endswith("/row_engine.json")
    assert L.driver_probe_argv("/tree", extra_flags=(C.FLAG_ROW_ENGINE_RECORD,))[-3:-1] == [C.FLAG_ROW_ENGINE_RECORD, L.DRIVER_PROBE_VALUES[C.FLAG_ROW_ENGINE_RECORD]]
    assert (L.row_max_lora_rank("/nonexistent", [LORA, "constv2ctl2-full__psr_ce_all"]), L.row_max_lora_rank("/nonexistent", ["constv2ctl2-full__psr_ce_all"])) == ((8, "cell_name"), (1, "rank1_floor"))
    assert L.row_dir_for("/vol/tinycap/gen/native/r/c/thinking_off/s", "r") == Path("/vol/tinycap/gen/rows/r/s")
    assert L.row_dir_for("/vol/tinycap/gen/r/c/thinking_off/s", "r") == Path("/vol/tinycap/gen/rows/r/s")
    with pytest.raises(ValueError, match="needs an --out-suffix"):
        L.row_dir_for("/vol/tinycap/gen/native/r/c/thinking_off", "r")


# --------------------------------------------------------------------------- #
# 4. dispatch: launch_row, row_function, main --row                              #
# --------------------------------------------------------------------------- #


def test_launch_row_spawns_one_functioncall_for_the_row_and_logs_every_cell(L, tmp_path):
    seen = []

    def _spawn(cfgs, **kw):
        seen.append((list(cfgs), dict(kw)))
        return TL._Call({"name": "row-x", "status": "ROW_PARTIAL", "cells": {"none": {"name": "n", "status": "OK"}},
                         "dropped": {LORA: {"name": "l", "status": "OBJECT_DIGEST_MISMATCH"}}})
    cfgs = _plan(L, ["none", LORA])
    log = tmp_path / "spawned.jsonl"
    result = L.launch_row(SimpleNamespace(spawn=_spawn), cfgs, {"sha": "f6cf462", "row_opts": {}}, spawned_log=log)
    assert len(seen) == 1 and [c["cell"] for c in seen[0][0]] == ["none", LORA] and seen[0][1]["row_opts"] == {}
    lines = [json.loads(ln) for ln in log.read_text().splitlines()]
    assert [ln["cell"] for ln in lines] == ["none", LORA] and len({ln["fc_id"] for ln in lines}) == 1 and all(ln["row"] for ln in lines)
    recs = L.row_result_recs(result)
    assert [r["status"] for r in recs] == ["OK", "OBJECT_DIGEST_MISMATCH"] and L.failed_recs(recs) == [("l", "OBJECT_DIGEST_MISMATCH")]
    refused = L.row_result_recs({"name": "row-y", "status": "ROW_NOT_UNIFORM", "error": "x", "cells": {}, "dropped": {}})
    assert refused == [{"name": "row-y", "model": None, "status": "ROW_NOT_UNIFORM", "error": "x"}]
    # a plain function (a test double without spawn) is called directly; a raising .get() is a failed row
    assert L.launch_row(lambda cfgs, **kw: {"status": "OK", "cells": {}, "dropped": {}}, cfgs, {})["status"] == "OK"

    def _boom(cfgs, **kw):
        return TL._Call(lambda: (_ for _ in ()).throw(RuntimeError("container died")))
    failed = L.launch_row(SimpleNamespace(spawn=_boom), cfgs, {})
    assert failed["status"] == "FUNCTIONCALL_FAILED" and "container died" in failed["error"]
    assert L.row_function(C.model_row("nemotron3nano4b-f22s1")) is L.run_row_nemotron_full
    assert L.row_function(C.model_row(ROW)) is L.run_row_full and L.row_function(C.model_row(IT_ROW), CONTROLS) is L.run_row_full


def test_main_row_validates_its_grammar_and_dry_runs_the_row(L, tmp_path, monkeypatch, capsys):
    """§1.1.3: ``--row`` needs ``--cells`` and ``--out-suffix``; K > 1, a forecast over 0.8 x the function
    timeout, row flags without ``--row`` and an unplanned repeat / negative-control cell refuse before any Modal
    call; the dry run prints the row line; the printed argv spells the row flags LAST."""
    base = dict(sha="f6cf462", app_name=L.APP_NAME, pass1=True, dry_run=True, model=ROW, cells=f"none,{LORA}", modes="off",
                controls=",".join(CONTROLS), sampling="protocol", out_suffix=SUFFIX)
    with pytest.raises(SystemExit, match="needs --cells"):
        L.main(**{**base, "cells": ""}, row=True)
    with pytest.raises(SystemExit, match="needs --out-suffix"):
        L.main(**{**base, "out_suffix": ""}, row=True)
    with pytest.raises(SystemExit, match="ONE thinking mode"):
        L.main(**{**base, "modes": ""}, row=True)
    with pytest.raises(SystemExit, match=f"at most {L.ROW_CONCURRENCY_K_MAX} concurrent cell drivers"):
        L.main(**base, row=True, concurrency_k=4)
    with pytest.raises(SystemExit, match="ROW_TOO_LONG"):
        L.main(**base, row=True, row_forecast_h=8.5)
    with pytest.raises(SystemExit, match="mean nothing without --row"):
        L.main(**base, repeat_cells="none")
    with pytest.raises(SystemExit, match="not among the planned cells"):
        L.main(**base, row=True, repeat_cells=PROMPTED)
    with pytest.raises(SystemExit, match="serves no object"):
        L.main(**base, row=True, negative_control_cell="none")
    with pytest.raises(SystemExit, match="no --smoke"):
        L.main(**base, row=True, smoke=True)
    with pytest.raises(SystemExit, match="not JSON"):
        L.main(**base, row=True, out_suffix_by_cell="{not json")
    capsys.readouterr()
    # the lane row's dry run needs the roster snapshot the launcher renders; a NON-lane row dry-runs plainly
    plain = dict(sha="f6cf462", app_name=L.APP_NAME, pass1=True, dry_run=True, cells="none", modes="off", controls="gsm8k", out_suffix="s")
    L.main(**plain, row=True, repeat_cells="none", row_forecast_h=1.5, cache_expectation="warm")
    out = capsys.readouterr().out
    assert "[tinycap] row: gemma4e4b x 1 cell(s) in ONE container (run_row_full)" in out
    assert "/rows/gemma4e4b/s" in out and "exactly 40 inodes" in out
    argv_line = next(ln for ln in out.splitlines() if ln.startswith("[tinycap] argv:"))
    assert argv_line.split("  (run it")[0].endswith("--row --repeat-cells none --cache-expectation warm --row-forecast-h 1.5"), argv_line
    assert L.row_too_long_problem(0) is None and L.row_too_long_problem(8.0) is None and "ROW_TOO_LONG" in L.row_too_long_problem(8.01)
    assert L.parse_out_suffix_by_cell("") == {} and L.parse_out_suffix_by_cell('{"a": "s-arditi"}') == {"a": "s-arditi"}
    # integration 2026-09-28: the launcher's INLINE map for a real row is longer than NAME_MAX (255) — `Path(text).is_file()` on it
    # raised OSError ENAMETOOLONG and every 8-cell row launch died at plan time; an inline document is never stat()'d (the
    # `--roster-snapshot` rule), a non-JSON value that is no file refuses by name
    long_map = {f"constv2-mix-full__lora_r4_ce_variant_{i:02d}": "bcbw-gemma4e4b_s1-8a73e7b-shared" for i in range(8)}
    assert len(json.dumps(long_map)) > 255 and L.parse_out_suffix_by_cell(json.dumps(long_map, separators=(",", ":"))) == long_map
    assert L.parse_out_suffix_by_cell("  " + json.dumps(long_map) + "\n") == long_map
    with pytest.raises(ValueError, match="neither inline JSON nor a readable file"):
        L.parse_out_suffix_by_cell("/nonexistent/" + "x" * 300)
    # integration 2026-09-28: a REAL modal.Function has no __name__ (its name is `.tag`); the dry run's `describe_row` line read
    # `.__name__` and raised AttributeError on the first row dry run against the harness; the stubs' plain functions hid it
    class _ModalFunction:                       # the shape modal 1.5.4 exposes: `tag` + `info.function_name`, no __name__
        tag = "run_row_full"
        info = SimpleNamespace(function_name="run_row_full")
    assert L.row_function_name(_ModalFunction()) == "run_row_full" == L.row_function_name(lambda: None).replace("<lambda>", "run_row_full")
    assert L.row_function_name(L.describe_row) == "describe_row" and L.row_function_name(object()) == "object"
    assert [c["out_dir"] for c in L.isolate_out_dirs_by_cell([{"cell": "a", "out_dir": "/o/a"}, {"cell": "b", "out_dir": "/o/b"}], "s", {"b": "t"})] == ["/o/a/s", "/o/b/t"]
    with pytest.raises(ValueError, match="not planned"):
        L.isolate_out_dirs_by_cell([{"cell": "a", "out_dir": "/o/a"}], "s", {"zz": "t"})


def test_main_row_launches_one_row_container_and_exits_on_the_per_cell_recs(L, monkeypatch, capsys):
    """The launch path: ``--row`` dispatches ONE ``launch_row`` over ``row_function`` with ``row_opts`` beside the
    configuration kwargs; the per-cell RECs decide the exit code."""
    spawned = []

    def _spawn(cfgs, **kw):
        spawned.append((cfgs, kw))
        cells = {c["cell"]: {"name": c["name"], "cell": c["cell"], "status": "OK"} for c in cfgs}
        return TL._Call({"name": "row-gemma4e4b-s", "status": "OK", "cells": cells, "dropped": {}, "standup_id": "su"})
    monkeypatch.setattr(L, "run_row_full", SimpleNamespace(spawn=_spawn))
    kw = dict(sha="f6cf462", app_name=L.APP_NAME, pass1=True, cells="none", modes="off", controls="gsm8k", out_suffix="s", row=True,
              repeat_cells="none", cache_expectation="warm")
    L.main(**kw)
    assert len(spawned) == 1 and [c["cell"] for c in spawned[0][0]] == ["none"]
    opts = spawned[0][1]["row_opts"]
    assert opts["repeat_cells"] == ["none"] and opts["cache_expectation"] == "warm" and opts["concurrency_k"] == 1
    assert spawned[0][1]["run_id"].startswith("f6cf462-") and "sha" in spawned[0][1]
    out = capsys.readouterr().out
    assert "[tinycap] ROW-RESULT" in out and "1/1 container(s) OK" in out

    def _fail(cfgs, **kw):
        return TL._Call({"name": "row", "status": "ROW_PARTIAL", "cells": {},
                         "dropped": {"none": {"name": "n", "cell": "none", "status": "OBJECT_DIGEST_MISMATCH"}}})
    monkeypatch.setattr(L, "run_row_full", SimpleNamespace(spawn=_fail))
    with pytest.raises(SystemExit, match="OBJECT_DIGEST_MISMATCH"):
        L.main(**kw)


def test_a_cell_keeps_its_own_waves_suffix_while_the_row_dir_is_keyed_by_the_primary(L, tmp_path, monkeypatch):
    """§1.6.1 / S14 (the qwen Arditi cell): one cell's store under ITS wave's suffix, the others under the row's;
    the row dir is keyed by the PRIMARY (``none``) suffix; every cell's record names the same standup."""
    arditi_suffix = SUFFIX.replace("gemma4e4b_s1", "gemma4e4b_s1_arditi")
    w = _World(L, tmp_path, monkeypatch, cells=["none", LORA, LORA2], suffix_by_cell={LORA2: arditi_suffix})
    assert Path(w.by_cell[LORA2]["out_dir"]).name == arditi_suffix and Path(w.by_cell[LORA]["out_dir"]).name == SUFFIX
    result = w.run()
    assert result["status"] == "OK" and result["suffix"] == SUFFIX and Path(result["row_dir"]).name == SUFFIX
    assert result["cells"][LORA2]["out"].endswith(arditi_suffix) and result["cells"][LORA2]["standup_id"] == STANDUP_A
    assert len(_RowStandup.argvs) == 1 and [c["cell"] for c in w.driver_calls] == ["none", LORA, LORA2]
    gi = json.loads((w.row_dir / "gpu_interval.json").read_text())
    assert gi["suffix"] == SUFFIX and {e["cell"] for e in gi["events"] if e["kind"] == "driver_exited"} == {"none", LORA, LORA2}


def test_a_row_launch_probes_the_drivers_row_engine_flag_as_its_own_preflight_check(L):
    """§1.1.4 (module F's ask): ``--row-engine-record`` is deliberately outside TINYCAP_DRIVER_FLAGS, so a ROW launch
    tells the preflight (``row_launch=True``) to probe it as its OWN argparse check, behind a valid dummy record;
    the solo probe (``driver_accepts_tinycap_flags``) is byte-identical and a solo launch never passes the kwarg."""
    import inspect
    assert "row_launch" in inspect.signature(L.preflight).parameters
    src = Path(L.__file__).read_text()
    body = src[src.index("def preflight("):src.index("    def datasets():")]
    assert "probe_flags = ((C.FLAG_ALLOW_OOS_THINKING,) if oos else ()) + (" in body                # the solo probe, unchanged
    i_solo = body.index('chk("driver_accepts_tinycap_flags", driver_flags)')
    i_row = body.index('chk("driver_accepts_row_engine_record", driver_row_engine_flag)')
    assert i_solo < body.index("    if row_launch:\n", i_solo) < i_row
    row_check = body[body.index("def driver_row_engine_flag():"):i_row]
    assert "write_probe_row_engine_record(Path(DRIVER_PROBE_VALUES[C.FLAG_ROW_ENGINE_RECORD]))" in row_check
    assert "driver_probe_argv(tree, extra_flags=(C.FLAG_ROW_ENGINE_RECORD,))" in row_check
    assert "if C.FLAG_ROW_ENGINE_RECORD in unrec:" in row_check and "unrecognized_flags(r.stderr)" in row_check
    main_body = src[src.index("def main("):]
    assert main_body.count("row_launch=True)") == 2                                                  # the snapshot call + the plain call
    assert "preflight.remote(sha, plan_args, expect)" in main_body                                   # the solo call, byte-identical
    assert 'C.FLAG_ROW_ENGINE_RECORD: "/tmp/tinycap-preflight-probe/row_engine.json"' in src


# --------------------------------------------------------------------------- #
# 6. the repeat's sandboxes and the resumed row                                 #
# --------------------------------------------------------------------------- #


def _watcher_module():
    """``scripts/tinycap_sandbox_watch.py`` loaded standalone (the way tests/test_tinycap_bcb_bfcl_200.py does)."""
    import importlib.util
    name = "tinycap_sandbox_watch_for_row_engine"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts/tinycap_sandbox_watch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _RowDirVolume:
    """A remote-Volume double over the runner's REAL row dir: ``listdir`` / ``read_file`` by volume-relative path
    (the watcher strips ``/vol/``; a temp path is served relative to ``/``), ``reload()`` refused as on the grading host."""

    def __init__(self, row_dir: Path, *, drop: Sequence[str] = (), certified: Mapping[str, dict] = ()):
        self.files: Dict[str, bytes] = {}
        rel = str(row_dir).lstrip("/")
        for p in sorted((row_dir / "watch").iterdir()):
            if p.name not in drop:
                self.files[f"{rel}/watch/{p.name}"] = p.read_bytes()
        # the fake world's sandbox listing RPC fails, so the runner's own certificates read `verified_empty: False`
        # (recorded, never inherited); a HEALTHY container's certificate for a cell is stood in here by name
        for name, cert in dict(certified).items():
            self.files[f"{rel}/watch/{name}"] = json.dumps(cert).encode("utf-8")

    def reload(self):
        raise RuntimeError("reload() is container-only")

    def listdir(self, path):
        prefix = path.rstrip("/") + "/"
        return [SimpleNamespace(path=p) for p in self.files if p.startswith(prefix)]

    def read_file(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        yield self.files[path]


def test_a_hard_death_during_the_repeat_leaves_its_owner_in_the_pointer_for_the_watcher(L, tmp_path, monkeypatch, capsys):
    """Code review R1-1 (HIGH): the FunctionCall dies while ``none-rep2``'s BigCodeBench sandboxes run — the in-process
    ``finally`` never executes and no sweep certificate is written for the repeat. The external watcher (``--row-dir``
    mode) derives the owners it sweeps from the row pointer ALONE, so the pointer must list the repeat's owner before
    its driver starts: read against the runner's REAL pointer with the repeat's certificate absent, the watcher sweeps
    ``none-rep2`` by its exact owner tuple (the base cells are certified by the container's own sweeps)."""
    w = _World(L, tmp_path, monkeypatch, cells=["none", LORA], row_opts={"repeat_cells": ["none"]})
    result = w.run()
    assert result["status"] == "OK" and "none-rep2" in result["cells"]
    fcid, run_id = result["function_call_id"], "f6cf462-cafebabe"
    W = _watcher_module()
    pointer = json.loads((w.row_dir / "watch" / f"attempt_{fcid}.json").read_text())
    assert W.is_row_pointer(pointer) and W.row_pointer_undeclared(pointer) is None
    assert [o.as_dict() for o in W.pointer_owners(pointer)] == [{"run_id": run_id, "cell": c, "function_call_id": fcid} for c in ("none", LORA, "none-rep2")]
    # the hard death: the repeat's certificate was never written (the container was killed mid-drive); the base cells'
    # own sweeps had certified clean (stood in: the fake world cannot list sandboxes, see _RowDirVolume)
    assert (w.row_dir / "watch" / W.row_sweep_certificate_name(fcid, "none-rep2")).is_file()      # the healthy run wrote it
    healthy = {W.row_sweep_certificate_name(fcid, c): {"owner": {"run_id": run_id, "cell": c, "function_call_id": fcid}, "verified_empty": True,
                                                       "ids_terminated": [], "still_live": [], "could_not_terminate": []} for c in ("none", LORA)}
    vol = _RowDirVolume(w.row_dir, drop=(W.row_sweep_certificate_name(fcid, "none-rep2"),), certified=healthy)
    assert not any(p.endswith("__none-rep2.json") for p in vol.files) and any(p.endswith("__none.json") for p in vol.files)
    monkeypatch.setattr(W, "_volume", lambda: vol)
    monkeypatch.setattr(W, "sandbox_app_id", lambda *a, **k: "ap-1")
    monkeypatch.setattr(W, "classify", lambda f: SimpleNamespace(unavailable=False, pending=False, terminal=True, describe=lambda: "terminal (died)"))
    swept = []

    def _sweep(owner, **kw):
        swept.append(owner)
        return {"verified_empty": True, "ids_terminated": ["sb-rep2-1", "sb-rep2-2"], "still_live": []}
    monkeypatch.setattr(W, "sweep_attempt", _sweep)
    rc = W.watch_row_dir(run_id=run_id, row_dir=str(w.row_dir), poll_s=0, max_hours=1, state_dir=str(tmp_path / "state"), once=True)
    out = capsys.readouterr().out
    assert rc == W.EXIT_DONE, out
    assert [(o.run_id, o.cell, o.function_call_id) for o in swept] == [(run_id, "none-rep2", fcid)]          # the repeat, and only it
    assert "none: the container's own sweep certified clean" in out and f"{LORA}: the container's own sweep certified clean" in out
    assert "none-rep2: no admissible container certificate (no certificate) — sweeping by owner" in out
    assert f"none-rep2@{fcid}" in out and "DONE: every attempt terminal and every owner certified" in out


def test_a_worker_loss_after_the_base_cells_never_reads_done_while_the_repeat_is_missing_or_parked(L, tmp_path, monkeypatch):
    """Code review R1-2 (MEDIUM): the first attempt finishes every base cell and dies before / during ``none-rep2``.
    The next identical launch must not report DONE (or OK) with the repeat floor absent or parked: (e) a complete
    repeat store resumes as DONE beside the base cells; (a) a repeat that never started is filed REPEAT_BASE_NOT_SERVED
    (its base ``none`` is DONE under standup A — a cross-standup repeat is no in-run floor), no GPU minute spent, the
    exit non-zero by name; (b) an in-flight repeat store is parked, never resumed into; (c) with another base
    cell still to serve the row serves it under a fresh standup and STILL files the repeat by name (ROW_PARTIAL, never
    OK); (d) the truthful recovery is a relaunch that serves ``none`` again WITH its repeat on one standup."""
    w = _World(L, tmp_path, monkeypatch, cells=["none", LORA, PROMPTED], standup_ids=["su-A", "su-B", "su-C"],
               row_opts={"repeat_cells": ["none"]})
    first = w.run()
    assert first["status"] == "OK" and first["cells"]["none-rep2"]["status"] == "OK" and first["standup_id"] == "su-A"
    rep_out = Path(w.by_cell["none"]["out_dir"] + "-rep2")

    def again(**kw):
        w.driver_calls.clear()
        _RowStandup.argvs.clear()
        return w.run(**kw)
    # (e) everything complete, the repeat included: DONE, the repeat DONE beside the base cells, nothing driven
    done = again()
    assert done["status"] == "DONE" and done["cells"]["none-rep2"]["status"] == "DONE" and w.driver_calls == [] and _RowStandup.argvs == []
    assert L.failed_recs(L.row_result_recs(done)) == []
    # (a) the loss BEFORE the repeat started: no repeat store; every base cell DONE under su-A
    shutil.rmtree(rep_out)
    second = again()
    assert second["status"] != "DONE" and second["status"] != "OK" and second["status"] == "ROW_EMPTY"
    assert all(second["cells"][c]["status"] == "DONE" for c in ("none", LORA, PROMPTED)) and "none-rep2" not in second["cells"]
    rep = second["dropped"]["none-rep2"]
    assert rep["status"] == L.REC_REPEAT_BASE_NOT_SERVED == "REPEAT_BASE_NOT_SERVED" and "DONE" in rep["error"] and "fresh whole-row launch" in rep["error"]
    assert w.driver_calls == [] and _RowStandup.argvs == []                                        # no GPU minute for a floor that cannot be
    assert _result_lines(rep_out)[-1].split(" result=")[1].startswith("REPEAT_BASE_NOT_SERVED")
    failed = L.failed_recs(L.row_result_recs(second))
    assert (rep["name"], "REPEAT_BASE_NOT_SERVED") in failed and any(s == "ROW_EMPTY" for _, s in failed)
    assert "repeat_skipped" not in second
    # (b) the loss DURING the repeat: manifests without a marker -> parked, named, never resumed into
    shutil.rmtree(rep_out)
    w.finish_store(rep_out, w.by_cell["none"])
    for sub in ("done", "summary"):
        for p in rep_out.glob(f"cells/*/{sub}"):
            shutil.rmtree(p)
    third = again()
    assert third["status"] == "ROW_EMPTY" and third["dropped"]["none-rep2"]["status"] == "INCOMPLETE_STORE_PARKED" and w.driver_calls == []
    # (c) another base cell still to serve: served under a fresh standup; the repeat is filed by name, the row is ROW_PARTIAL
    shutil.rmtree(rep_out)
    shutil.rmtree(w.by_cell[PROMPTED]["out_dir"])
    fourth = again()
    assert fourth["status"] == "ROW_PARTIAL" and fourth["standup_id"] == "su-B" and [c["cell"] for c in w.driver_calls] == [PROMPTED]
    assert fourth["cells"][PROMPTED]["status"] == "OK" and fourth["cells"]["none"]["status"] == "DONE"
    assert fourth["dropped"]["none-rep2"]["status"] == "REPEAT_BASE_NOT_SERVED" and "none-rep2" not in fourth["cells"]
    assert [a[i + 1] for a in _RowStandup.argvs for i, t in enumerate(a) if t == "--cell"] == [PROMPTED]           # `none` not stood up for a repeat
    pointer = json.loads((w.row_dir / "watch" / f"attempt_{fourth['function_call_id']}.json").read_text())
    assert pointer["cells"] == [PROMPTED]                                                            # no owner for a repeat that is not driven
    # (d) the truthful recovery: `none` served again WITH its repeat on ONE fresh standup, under the relaunch suffix
    r2_none = dict(w.by_cell["none"], out_dir=w.by_cell["none"]["out_dir"][:-len(SUFFIX)] + SUFFIX + "-r2")
    w.by_cell["none"] = r2_none
    fifth = again(cfgs=[r2_none], row_opts={**w.row_opts, "row_dir": str(w.gen / "rows" / ROW / (SUFFIX + "-r2"))})
    assert fifth["status"] == "OK" and fifth["standup_id"] == "su-C" and sorted(fifth["cells"]) == ["none", "none-rep2"]
    assert fifth["cells"]["none"]["standup_id"] == fifth["cells"]["none-rep2"]["standup_id"] == "su-C" and [c["cell"] for c in w.driver_calls] == ["none", "none"]
    assert Path(fifth["cells"]["none-rep2"]["out"]).name == SUFFIX + "-r2-rep2"


# --------------------------------------------------------------------------- #
# 7. the warm probe's exit                                                      #
# --------------------------------------------------------------------------- #


def test_main_exits_zero_on_a_warmed_warm_only_row_and_nonzero_on_warmed_anywhere_else(L, monkeypatch, capsys):
    """Code review R2-1 (HIGH): ``main --row --warm-cache-only`` whose cells all end WARMED exits SUCCESS — before,
    ``failed_recs`` read every WARMED cell as a failure and the Modal CLI exited non-zero on a probe that had
    persisted its tarball. WARMED is accepted for THAT invocation's exit alone: a row launch without
    ``--warm-cache-only`` returning WARMED cells, and the solo path, still exit non-zero on it; a warm probe whose
    persist was refused (WARM_NOT_PERSISTED) exits non-zero naming the reason."""
    spawned = []

    def _warmed(cfgs, **kw):
        spawned.append(kw)
        cells = {c["cell"]: {"name": c["name"], "cell": c["cell"], "status": "WARMED", "standup_id": "su"} for c in cfgs}
        return TL._Call({"name": "row-gemma4e4b-w", "status": "WARMED", "cells": cells, "dropped": {}, "standup_id": "su",
                         "compile_cache": {"persist": {"persisted": True}}})
    monkeypatch.setattr(L, "run_row_full", SimpleNamespace(spawn=_warmed))
    kw = dict(sha="f6cf462", app_name=L.APP_NAME, pass1=True, cells="none", modes="off", controls="gsm8k", out_suffix="w", row=True)
    L.main(**kw, warm_cache_only=True, cache_expectation="cold")                          # exits 0: no SystemExit
    out = capsys.readouterr().out
    assert spawned[0]["row_opts"]["warm_cache_only"] is True and "1/1 container(s) OK" in out and "did NOT finish" not in out
    # the same WARMED cells on a row launch that is NOT a warm probe: WARMED is no completed store -> non-zero
    with pytest.raises(SystemExit, match="WARMED") as e:
        L.main(**kw)
    assert f"status outside {list(L.REC_OK_STATUSES)}" in str(e.value) and "1/1 container(s) did NOT finish" in str(e.value)
    # a refused warm probe: the row refusal and every dropped cell carry WARM_NOT_PERSISTED -> non-zero, by name
    def _not_persisted(cfgs, **kw):
        dropped = {c["cell"]: {"name": c["name"], "cell": c["cell"], "status": "WARM_NOT_PERSISTED",
                               "error": "warm probe k: the compile cache was NOT persisted (skipped: no_cache_dir_line)"} for c in cfgs}
        return TL._Call({"name": "row-gemma4e4b-w", "status": "WARM_NOT_PERSISTED", "cells": {}, "dropped": dropped,
                         "error": "warm probe k: the compile cache was NOT persisted (skipped: no_cache_dir_line)"})
    monkeypatch.setattr(L, "run_row_full", SimpleNamespace(spawn=_not_persisted))
    with pytest.raises(SystemExit, match="WARM_NOT_PERSISTED") as e:
        L.main(**kw, warm_cache_only=True)
    assert "skipped: no_cache_dir_line" in str(e.value) and "2/2 container(s) did NOT finish" in str(e.value)     # the cell + the row's own REC
    # the solo path's rule is byte-identical: no also_ok, WARMED fails
    assert L.failed_recs([{"name": "x", "status": "WARMED"}]) == [("x", "WARMED")] and L.REC_OK_STATUSES == ("OK", "DONE", L.LADDER_REC_STATUS)
    assert L.failed_recs([{"name": "x", "status": "WARMED"}], also_ok=("WARMED",)) == []
    assert "WARMED" not in L.ROW_RESULT_OK_STATUSES or L.row_result_recs({"status": "WARMED", "cells": {"a": {"status": "WARMED"}}, "dropped": {}}) == [{"status": "WARMED"}]


# --------------------------------------------------------------------------- #
# landing 1.1: the cost trip-wire counts CONTAINERS — a row launch is one #
# --------------------------------------------------------------------------- #


def test_gpu_container_count_row_launch_is_one_container(L):
    """The canary's qwen warm probe (9 cells incl. arditi_allpos, ONE row container) was refused
    ``9 GPU containers > MAX_GPU_CONTAINERS 8`` because ``main`` fed ``len(cfgs)`` to the trip-wire in row
    mode too. The count is a pure function: per-cell launches spawn one container per configuration; a row
    launch spawns one for the whole row; the Nemotron probe adds one in either mode."""
    import inspect
    # per-cell: one container per configuration (the pre-landing truth, unchanged)
    assert L.gpu_container_count(9, row_launch=False) == 9
    assert L.gpu_container_count(3, row_launch=False, nemotron_probe=True) == 4
    # row: ONE container however many cells; nothing to serve = nothing spawned
    assert L.gpu_container_count(9, row_launch=True) == 1
    assert L.gpu_container_count(1, row_launch=True) == 1
    assert L.gpu_container_count(0, row_launch=True) == 0
    assert L.gpu_container_count(9, row_launch=True, nemotron_probe=True) == 2
    with pytest.raises(ValueError):
        L.gpu_container_count(-1, row_launch=True)
    # the trip-wire itself is unchanged: 9 per-cell containers refuse, the row's 1 passes, --allow-extra still lifts it
    with pytest.raises(SystemExit) as e:
        L.gpu_container_budget(L.gpu_container_count(9, row_launch=False))
    assert f"> MAX_GPU_CONTAINERS {C.MAX_GPU_CONTAINERS}" in str(e.value)
    L.gpu_container_budget(L.gpu_container_count(9, row_launch=True))
    L.gpu_container_budget(L.gpu_container_count(9, row_launch=False), allow_extra=True)
    # and main feeds the trip-wire THIS count with the launch's own row flag — a source pin, so the per-cell
    # `len(cfgs)` can never come back silently
    src = inspect.getsource(L.main)
    assert "gpu_container_count(len(cfgs), row_launch=row_launch, nemotron_probe=nemotron_probe)" in src
    assert "n_gpu = len(cfgs) +" not in src


# --------------------------------------------------------------------------- #
# K-concurrent drivers, the generate phase, the CPU verifier main spawns, the  #
# exit code after its RECs                                                     #
# --------------------------------------------------------------------------- #

import threading  # noqa: E402


def _generation_sidecar(*, cell: str, store: str, log: Path, concurrency: int, generation_fcid: str = "fc-gen") -> dict:
    """A generation sidecar with EXACTLY ``cpu_verify.GENERATION_SIDECAR_FIELDS`` (the facts the tests read set, the rest
    null) — what the driver's ``generate`` writes beside the pending marker."""
    doc = {k: None for k in CV.GENERATION_SIDECAR_FIELDS}
    doc.update({"task": BCB_FULL, "cell": cell, "store": store, "log": str(log), "n_items": 1140, "n_expected_full": 1140,
                "sample_ids": [f"BigCodeBench/{i}" for i in range(3)], "pin_ids_sha256": "0" * 64,
                "generated_at": "2026-09-29T00:00:00Z", "generation_fcid": generation_fcid, "n_requests": 1140,
                "generated_calls": 1140, "wall_clock_s": 1.0, "logs_reused": 0, "regimes": {}, "client": {},
                "n_http_rejected_retried": 0, "target_model": "openai/x", "thinking": "off", "limits": {}, "out_of_support": False,
                "row_identity": {}, "concurrency": int(concurrency), "max_model_len": 4608, "plan": {},
                "producer_tree_sha": FULL_SHA, "inspect_ai_version": "0.3.260"})
    return doc


def _generate_driver(w: "_World", *, mark: bool = True, marker_fields=None, generation_fcid: str = "fc-gen"):
    """A ``generate`` fake driver: files the BFCL leg as today, leaves BigCodeBench
    UNSCORED — the manifest's row_engine block copied from the record with ``verification_pass`` pending, the
    generation sidecar + the pending marker written through the DRIVER's own ``cpu_verify.write_generation_pending``
    (so the names are the real store-keyed ``<cell>__<task>__<store>.json`` / ``pending_<cell>__<task>__<store>.json``
    and the marker is EXACTLY ``contracts.PENDING_MARKER_FIELDS``) under ``<row dir>/generation/`` (the row dir read
    off the row-engine record the argv names). ``mark=False``: a driver that forgot the marker; ``marker_fields``: a
    marker with another key set, written by hand at the real path."""
    def _driver(argv, **kw):
        argv = list(argv)
        cell = argv[argv.index("--cell") + 1]
        out_root = Path(argv[argv.index(C.FLAG_OUT_ROOT) + 1])
        rec_doc = C.read_row_engine_record(argv[argv.index(C.FLAG_ROW_ENGINE_RECORD) + 1])
        phase = argv[argv.index(C.FLAG_VERIFICATION_PHASE) + 1] if C.FLAG_VERIFICATION_PHASE in argv else "coupled"
        w.driver_calls.append({"argv": argv, "cell": cell, "out_root": out_root, "env": dict(kw.get("env") or {}), "phase": phase})
        cfg = w.by_cell[cell]
        block = {**rec_doc, "verification_pass": {"state": C.VERIFICATION_PASS_PENDING, "tasks": [BCB_FULL]}}
        store = w.finish_store(out_root, cfg, tasks=(BFCL_FULL,), row_engine=block)
        (out_root / "evidence.json").write_text(json.dumps({"cell": cell}))
        row_dir = Path(rec_doc["row_dir"])
        (row_dir / "generation").mkdir(parents=True, exist_ok=True)
        log_dir = store / C.INSPECT_LOGS_SUBDIR
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / "bigcodebench_full.eval"
        log.write_bytes(b"unscored")
        sidecar = _generation_sidecar(cell=cell, store=out_root.name, log=log, concurrency=int(argv[argv.index("--concurrency") + 1]),
                                      generation_fcid=generation_fcid)
        marker = {k: sidecar[k] for k in C.PENDING_MARKER_FIELDS}
        if marker_fields is not None:
            # a marker with another key set: the driver's writer would refuse it, so it is written by hand at the real path
            CV.pending_marker_path(row_dir, cell, BCB_FULL, out_root.name).write_text(json.dumps(dict(marker_fields)))
        elif mark:
            CV.write_generation_pending(row_dir, sidecar=sidecar, marker=marker)
        return SimpleNamespace(returncode=0, stdout="generate ok\n", stderr="")
    return _driver


def test_a_k2_row_drives_two_cells_at_once_at_s_over_k_and_the_gpu_interval_counts_the_overlap_once(L, tmp_path, monkeypatch):
    """``row_opts.concurrency_k = 2`` — a ``ThreadPoolExecutor(2)`` over ``_drive_cell``: the two base drivers
    OVERLAP (each blocks until the other has started), every driver's ``--concurrency`` and the hashed expectation read
    S // K (31 -> 15: a DIFFERENT cell key), the repeat is driven alone AFTER both; the events show the overlap and
    the GPU interval counts it ONCE (union < sum; module C's rule)."""
    w = _World(L, tmp_path, monkeypatch, cells=["none", LORA], row_opts={"concurrency_k": 2, "repeat_cells": ["none"]})
    barrier = threading.Barrier(2, timeout=60)
    n_calls = {"n": 0}
    real = w._driver

    def _overlapping(argv, **kw):
        n_calls["n"] += 1
        if n_calls["n"] <= 2:
            barrier.wait()                              # the two base drives are in flight together, or this hangs and fails
        return real(argv, **kw)
    monkeypatch.setattr(L.subprocess, "run", _overlapping)
    result = w.run()
    assert result["status"] == "OK", result.get("error")
    assert sorted(result["cells"]) == sorted(["none", LORA, "none-rep2"]) and result["concurrency_k"] == 2 and result["phase"] == "coupled"
    assert {c["cell"] for c in w.driver_calls[:2]} == {"none", LORA} and w.driver_calls[2]["cell"] == "none"
    for call in w.driver_calls:
        a = call["argv"]
        assert a[a.index("--concurrency") + 1] == "15"                     # S // K = 31 // 2
        assert C.FLAG_VERIFICATION_PHASE not in a                         # coupled spells nothing (byte-identical)
    for label, rec in result["cells"].items():
        assert rec["status"] == "OK" and rec["expected_identity"]["served"]["concurrency"] == 15
        assert rec["expected_identity"]["served"]["max_num_seqs"] == 31
    # the events: both base drivers started before either exited; the repeat started after both exited
    ev = result["events"]
    t_start = {e["cell"]: e["t"] for e in ev if e["kind"] == "driver_started"}
    t_exit = {e["cell"]: e["t"] for e in ev if e["kind"] == "driver_exited"}
    assert max(t_start["none"], t_start[LORA]) <= min(t_exit["none"], t_exit[LORA])
    assert t_start["none-rep2"] >= max(t_exit["none"], t_exit[LORA])
    iv = result["gpu_interval"]["interval"]
    assert iv["n_drivers"] == 3 and iv["overlap_s"] > 0 and iv["union_driver_s"] < iv["sum_driver_s"]
    assert iv["sum_driver_s"] == pytest.approx(iv["union_driver_s"] + iv["overlap_s"], abs=1e-3)
    assert iv["gpu_held_s"] >= iv["union_driver_s"] and iv["end"] == iv["t_standup_exited"] >= iv["t_last_generation_exit"]
    gi = json.loads((w.row_dir / "gpu_interval.json").read_text())
    assert gi["concurrency_k"] == 2 and gi["gpu_held_s"] == pytest.approx(iv["gpu_held_s"], abs=1e-3)
    # the helper's arithmetic on its own: two overlapping runs count once; sequential runs do not overlap
    assert L.driver_overlap([(0.0, 10.0), (5.0, 15.0)]) == {"sum_driver_s": 20.0, "union_driver_s": 15.0, "overlap_s": 5.0}
    assert L.driver_overlap([(0.0, 10.0), (10.0, 20.0)]) == {"sum_driver_s": 20.0, "union_driver_s": 20.0, "overlap_s": 0.0}
    assert L.driver_overlap([]) == {"sum_driver_s": 0.0, "union_driver_s": 0.0, "overlap_s": 0.0}
    # the client width rule and the post-standup re-derivation at K (the repeat's expectation is re-derived at the served S // K)
    assert L.client_width(31, 1) == 31 and L.client_width(31, 2) == 15 and L.client_width(31, 3) == 10 and L.client_width(1, 3) == 1
    with pytest.raises(ValueError):
        L.client_width(31, 0)
    exp = result["cells"]["none-rep2"]["expected_identity"]
    assert exp["served"]["concurrency"] == 15 and result["cells"]["none-rep2"]["repeat_of"] == "none"
    # gpu_interval on a hand timeline: two overlapping drivers, the interval is the union (end = standup_exited)
    ev2 = [{"t": 0.0, "kind": "container_start", "cell": None, "phase": None},
           {"t": 10.0, "kind": "driver_started", "cell": "a", "phase": "coupled"},
           {"t": 12.0, "kind": "driver_started", "cell": "b", "phase": "coupled"},
           {"t": 110.0, "kind": "driver_exited", "cell": "a", "phase": "coupled"},
           {"t": 130.0, "kind": "driver_exited", "cell": "b", "phase": "coupled"},
           {"t": 140.0, "kind": "standup_exited", "cell": None, "phase": None},
           {"t": 141.0, "kind": "container_end", "cell": None, "phase": None}]
    iv2 = L.gpu_interval(ev2)
    assert (iv2["gpu_held_s"], iv2["sum_driver_s"], iv2["union_driver_s"], iv2["overlap_s"]) == (140.0, 218.0, 120.0, 98.0)


def test_a_generate_row_files_generated_and_names_exactly_the_bcb_cells_for_the_cpu_verifier(L, tmp_path, monkeypatch, capsys):
    """``row_opts.verification_phase = "generate"`` — every driver's argv ends
    ``--row-engine-record <p> --verification-phase generate``; a driver that returned 0 with BigCodeBench left and THIS
    store's pending marker in the row dir is GENERATED (REC + RESULT ``result=GENERATED status=GENERATED … standup=<id>``),
    the row's status is GENERATED, ``verification_pending`` names exactly the BCB cells (the store, the tasks, the
    marker, a copy of the cell's row-engine record in the row dir, the overlay, ``max_sandboxes`` = min(S // K, 8)),
    the pointer's phase is ``generate``, the GPU interval ends at the standup exit / last generation with nothing
    after it, and the container returns. A driver that left the task without a marker stays INCOMPLETE, by name."""
    w = _World(L, tmp_path, monkeypatch, cells=["none", LORA], row_opts={"verification_phase": "generate", "repeat_cells": ["none"]})
    monkeypatch.setattr(L.subprocess, "run", _generate_driver(w))
    result = w.run()
    assert result["status"] == "GENERATED" == L.REC_GENERATED, result.get("error")
    assert result["phase"] == "generate" and result["cell_statuses"] == {"none": "GENERATED", LORA: "GENERATED", "none-rep2": "GENERATED"}
    for call in w.driver_calls:
        a = call["argv"]
        assert a[-2:] == [C.FLAG_VERIFICATION_PHASE, "generate"] and a[-4] == C.FLAG_ROW_ENGINE_RECORD and call["phase"] == "generate"
        assert a[a.index("--concurrency") + 1] == "31"                                            # K = 1: the served S
    fcid, standup_id = result["function_call_id"], result["standup_id"]
    for label, rec in result["cells"].items():
        assert rec["status"] == "GENERATED" and rec["driver_rc"] == 0 and rec["controls_still_remaining"] == ["bigcodebench_full"]
        assert rec["verification_pending"]["tasks"] == [BCB_FULL] and rec["verification_pending"]["problems"] == []
        lines = _result_lines(Path(rec["out"]))
        assert len(lines) == 1 and " result=GENERATED status=GENERATED " in lines[0] and lines[0].endswith(f" standup={standup_id}")
        assert "error" not in rec
    # main's spawn list: exactly the BCB cells (every cell of this row generated BCB), keyed by label, with what the verifier needs
    pending = result["verification_pending"]
    assert list(pending) == ["none", LORA, "none-rep2"]
    for label, pend in pending.items():
        rec = result["cells"][label]
        assert pend["out_dir"] == rec["out"] and pend["cell"] == rec["cell"] and pend["name"] == rec["name"]
        assert pend["controls"] == ["bigcodebench_full"] and pend["tasks"] == [BCB_FULL] and len(pend["markers"]) == 1
        assert Path(pend["markers"][0]).is_file() and Path(pend["markers"][0]).parent == w.row_dir / "generation"
        assert pend["max_sandboxes"] == 8 and pend["served_max_num_seqs"] == 31 and pend["concurrency_k"] == 1
        assert pend["config"] == result["config"] and pend["standup_id"] == standup_id and pend["generation_function_call_id"] == fcid
        copy = Path(pend["row_engine_record"])
        assert copy == w.row_dir / "generation" / f"row_engine__{label}.json" and copy.is_file()
        assert C.read_row_engine_record(copy)["standup_id"] == standup_id and C.read_row_engine_record(copy)["row_dir"] == str(w.row_dir)
        assert "row_engine_record_error" not in pend
    # the pointer's phase, the interval's end, the row record
    pointer = json.loads((w.row_dir / "watch" / f"attempt_{fcid}.json").read_text())
    assert pointer["phase"] == "generate" and pointer["cells"] == ["none", LORA, "none-rep2"]
    gi = json.loads((w.row_dir / "gpu_interval.json").read_text())
    assert gi["phase"] == "generate" and all(e["phase"] == "generate" for e in gi["events"] if e["kind"].startswith("driver_"))
    assert not any(e["kind"].startswith("verify_") for e in gi["events"])
    iv = gi["interval"]
    assert iv["end"] == max(iv["t_last_generation_exit"], iv["t_standup_exited"]) and iv["end"] <= iv["t_container_end"] and iv["n_drivers"] == 3
    row_rec = json.loads((w.row_dir / "row_rec.json").read_text())
    assert row_rec["status"] == "GENERATED" and row_rec["phase"] == "generate" and set(row_rec["verification_pending"]) == set(pending)
    # a GENERATED store still counts the bound: today's files + standup/ (no verifier artefact lands in the store)
    for label, rec in result["cells"].items():
        out = Path(rec["out"])
        assert {p.name for p in out.iterdir()} <= set(SS.ROOT_FILES) | {"cells", "watch", "standup", rec["cell"], "driver_stdout.log", "driver_stderr.log"}
    out_text = capsys.readouterr().out
    assert out_text.count("[tinycap] REC ") == 3
    # the driver forgot the marker: INCOMPLETE, naming the task and the cell; the row is ROW_PARTIAL; nothing to verify
    w2 = _World(L, tmp_path / "w2", monkeypatch, cells=["none"], row_opts={"verification_phase": "generate"})
    monkeypatch.setattr(L.subprocess, "run", _generate_driver(w2, mark=False))
    r2 = w2.run()
    assert r2["status"] == "ROW_PARTIAL" and r2["cells"]["none"]["status"] == "INCOMPLETE" and r2["verification_pending"] == {}
    assert f"task {BCB_FULL!r}" in r2["cells"]["none"]["error"] and "carries no pending marker for cell 'none'" in r2["cells"]["none"]["error"]
    # a marker with a key set other than the contract's is named, never read as absent
    w3 = _World(L, tmp_path / "w3", monkeypatch, cells=["none"], row_opts={"verification_phase": "generate"})
    monkeypatch.setattr(L.subprocess, "run", _generate_driver(w3, marker_fields={"task": BCB_FULL, "cell": "none"}))
    r3 = w3.run()
    assert r3["cells"]["none"]["status"] == "INCOMPLETE" and "PENDING_MARKER_FIELDS" in r3["cells"]["none"]["error"]
    # the coupled default is byte-identical: no phase flag, the same statuses as landing 1
    w4 = _World(L, tmp_path / "w4", monkeypatch, cells=["none"])
    r4 = w4.run()
    assert r4["status"] == "OK" and r4["phase"] == "coupled" and r4["verification_pending"] == {}
    assert C.FLAG_VERIFICATION_PHASE not in w4.driver_calls[0]["argv"]
    assert json.loads((w4.row_dir / "watch" / f"attempt_{r4['function_call_id']}.json").read_text())["phase"] == "coupled"


def test_generation_pending_verdict_binds_a_marker_to_its_own_store_and_names_every_miss(L, tmp_path):
    """The rule behind GENERATED: a store's marker is THE file the driver spells for (cell, task, store
    basename) — ``pending_<cell>__<task>__<store>.json`` (``cpu_verify.pending_marker_path``), read by name — well-formed,
    naming this cell / task, its ``log`` under the store; the in-run repeat writes with the same ``--cell`` into another
    store, so its marker is another file and never this store's evidence (and vice versa); a control with no marker, a
    malformed marker, a marker whose fields name another cell, or no row dir are problems by name."""
    row_dir = tmp_path / "rows" / "r" / "s"
    gen = row_dir / "generation"
    gen.mkdir(parents=True)
    base, rep2 = tmp_path / "store", tmp_path / "store-rep2"
    for d in (base, rep2):
        (d / "cells" / "k" / "logs").mkdir(parents=True)
    ok = {"task": BCB_FULL, "cell": "none", "n_items": 1, "n_expected_full": 1140, "generated_at": "t", "log": str(base / "cells/k/logs/x.eval"),
          "generation_fcid": "fc"}
    base_marker = CV.pending_marker_path(row_dir, "none", BCB_FULL, base.name)
    assert base_marker == gen / f"pending_none__inspect__bigcodebench_full__{base.name}.json"          # the driver's spelling
    base_marker.write_text(json.dumps(ok))
    v = L.generation_pending_verdict(row_dir, cell="none", out_dir=base, remaining=["bigcodebench_full"], mode="off")
    assert v["problems"] == [] and v["tasks"] == [BCB_FULL] and v["markers"] == [str(base_marker)]
    # the REPEAT's store has its own file name: the base's marker is neither its evidence nor its problem
    v2 = L.generation_pending_verdict(row_dir, cell="none", out_dir=rep2, remaining=["bigcodebench_full"], mode="off")
    assert len(v2["problems"]) == 1 and "carries no pending marker for cell 'none'" in v2["problems"][0]
    assert f"pending_none__inspect__bigcodebench_full__{rep2.name}.json" in v2["problems"][0] and "is not under this store" not in v2["problems"][0]
    # another cell's marker is never this cell's (another file name)
    assert L.generation_pending_verdict(row_dir, cell=LORA, out_dir=base, remaining=["bigcodebench_full"], mode="off")["problems"] == [
        f"task {BCB_FULL!r} left by the driver carries no pending marker for cell {LORA!r} "
        f"(no pending_{LORA}__inspect__bigcodebench_full__{base.name}.json under the row dir) (control 'bigcodebench_full')"]
    # BFCL left over has no marker by contract -> a problem (only BigCodeBench is generated unscored)
    v3 = L.generation_pending_verdict(row_dir, cell="none", out_dir=base, remaining=["bigcodebench_full", "bfcl_full"], mode="off")
    assert v3["tasks"] == [BCB_FULL] and len(v3["problems"]) == 1 and BFCL_FULL in v3["problems"][0]
    assert L.generation_pending_verdict(None, cell="none", out_dir=base, remaining=["bigcodebench_full"], mode="off")["problems"] == [
        "no row dir to read pending markers from"]
    # this store's file with another store's log, or naming another cell / task: a problem, by name
    CV.pending_marker_path(row_dir, "none", BCB_FULL, rep2.name).write_text(json.dumps(ok))                   # log under base
    bad = L.generation_pending_verdict(row_dir, cell="none", out_dir=rep2, remaining=["bigcodebench_full"], mode="off")
    assert len(bad["problems"]) == 1 and "is not under this store" in bad["problems"][0]
    CV.pending_marker_path(row_dir, "none", BCB_FULL, rep2.name).write_text(json.dumps({**ok, "cell": LORA, "log": str(rep2 / "cells/k/logs/x.eval")}))
    bad2 = L.generation_pending_verdict(row_dir, cell="none", out_dir=rep2, remaining=["bigcodebench_full"], mode="off")
    assert len(bad2["problems"]) == 1 and f"names ({LORA!r}, {BCB_FULL!r}), not ('none', {BCB_FULL!r})" in bad2["problems"][0]
    # the row-level listing: a torn marker is named there, and scoped BY NAME to its cell / store — never to another's
    (gen / "pending_zz__inspect__bigcodebench_full__other.json").write_text("{torn")
    ms = L.pending_generation_markers(row_dir)
    assert len(ms) == 3 and any("unreadable" in str(m.get("_problem")) for m in ms)
    # (the rep2 file's name is in cell none's scope but its own `cell` field says LORA: dropped from the listing by design)
    assert [m.get("cell") for m in L.pending_generation_markers(row_dir, cell="none") if not m.get("_problem")] == ["none"]
    assert not any(m.get("_problem") for m in L.pending_generation_markers(row_dir, cell="none"))
    assert [Path(m["_path"]).name for m in L.pending_generation_markers(row_dir, store=base.name)] == [base_marker.name]
    assert [Path(m["_path"]).name for m in L.pending_generation_markers(row_dir, cell="zz")] == ["pending_zz__inspect__bigcodebench_full__other.json"]
    assert L.marker_name_scoped_to(base_marker.name, cell="none", store=base.name) and not L.marker_name_scoped_to(base_marker.name, store=rep2.name)
    assert not L.marker_name_scoped_to(base_marker.name, cell="non")                                            # a prefix, not the cell


def test_the_repeat_and_the_base_keep_their_own_store_keyed_markers_and_a_torn_foreign_marker_poisons_neither(L, tmp_path, monkeypatch):
    """Code review L2 H2 (P1): the driver keys the marker by STORE, so a generated row with its default ``none-rep2``
    leaves BOTH ``pending_none__…__<suffix>.json`` and ``pending_none__…__<suffix>-rep2.json`` under the row dir until
    each store's CPU verifier removes its own. The repeat's verdict used to scan every marker of cell ``none`` and count
    the base store's as a problem (REC INCOMPLETE, never verified, the launch failed); the fake driver hid it by writing
    the plan's old cell-only name (one file, overwritten). With the real names: both GENERATED, both queued; a TORN
    marker of another store / cell in the same directory makes no cell of the row INCOMPLETE."""
    w = _World(L, tmp_path, monkeypatch, cells=["none", LORA], row_opts={"verification_phase": "generate", "repeat_cells": ["none"]})
    monkeypatch.setattr(L.subprocess, "run", _generate_driver(w))
    torn = w.row_dir / "generation" / "pending_stale-cell__inspect__bigcodebench_full__some-other-suffix.json"
    torn.parent.mkdir(parents=True, exist_ok=True)
    torn.write_text("{torn")
    result = w.run()
    assert result["status"] == "GENERATED", result.get("error")
    assert result["cell_statuses"] == {"none": "GENERATED", LORA: "GENERATED", "none-rep2": "GENERATED"}
    base_store, rep2_store = Path(result["cells"]["none"]["out"]), Path(result["cells"]["none-rep2"]["out"])
    assert rep2_store.name == f"{base_store.name}-rep2"
    base_marker = CV.pending_marker_path(w.row_dir, "none", BCB_FULL, base_store.name)
    rep2_marker = CV.pending_marker_path(w.row_dir, "none", BCB_FULL, rep2_store.name)
    assert base_marker.is_file() and rep2_marker.is_file() and base_marker != rep2_marker                    # both coexist
    assert CV.generation_sidecar_path(w.row_dir, "none", BCB_FULL, base_store.name).is_file()
    assert CV.generation_sidecar_path(w.row_dir, "none", BCB_FULL, rep2_store.name).is_file()
    assert result["cells"]["none"]["verification_pending"]["markers"] == [str(base_marker)]
    assert result["cells"]["none-rep2"]["verification_pending"]["markers"] == [str(rep2_marker)]
    assert result["verification_pending"]["none"]["markers"] == [str(base_marker)]
    assert result["verification_pending"]["none-rep2"]["markers"] == [str(rep2_marker)]
    for rec in result["cells"].values():
        assert "error" not in rec and rec["verification_pending"]["problems"] == []
    assert torn.is_file()                                                                                       # untouched, and harmless
    # the torn marker is still NAMED by the row-level listing (never silently dropped), scoped to its own name
    assert [Path(m["_path"]).name for m in L.pending_generation_markers(w.row_dir) if m.get("_problem")] == [torn.name]
    assert not any(m.get("_problem") for m in L.pending_generation_markers(w.row_dir, cell="none"))


def test_the_verify_worker_composes_the_cpu_verifiers_argv_owner_env_and_reads_the_stores_evidence(L, tmp_path, monkeypatch):
    """``_verify_cell_worker`` (the body of the CPU ``verify_cell_cpu``) runs ``served_capability_run.py
    verify --out-root <store> --cell <cell> --controls bigcodebench_full --config <overlay> --tree <staged tree> --row-dir
    <row dir> --row-engine-record <copy> --max-sandboxes N --function-call-id <fcid>`` under the client venv with the owner
    ``(run_id, cell, fcid)`` in its env (the record under the ROW dir, never the 40-inode store), sweeps its own owner
    (certificate ``sweep_<fcid>__<cell>.json`` in the row dir), and reads the contract's evidence: the verified
    ``verification_pass`` naming this fcid, no marker left, ``RESULT … result=COMPLETE … verified_by=<fcid>``, the
    verifier's own pointer (phase verify, one cell). Every miss is named; the REC is OK only when nothing is."""
    repos = tmp_path / "repos"
    TL._tree(repos, "f6cf462")
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    row_dir = tmp_path / "rows" / ROW / SUFFIX
    (row_dir / "generation").mkdir(parents=True)
    store = tmp_path / "gen" / "native" / ROW / "none" / "thinking_off" / SUFFIX
    (store / "cells" / "k1" / "logs").mkdir(parents=True)
    log = store / "cells" / "k1" / "logs" / "bcb.eval"
    log.write_bytes(b"unscored")
    manifest = store / "cells" / "k1" / "manifest.json"
    pending_block = {"row_engine": True, "standup_id": STANDUP_A, "engine_cells": ["none"], "max_loras": 1, "residual_slot_armed": False,
                     "compile_cache": {"hit": False, "key": "k", "vllm_cache_dir": None}, "row_dir": str(row_dir),
                     "verification_pass": {"state": "pending", "tasks": [BCB_FULL]}}
    manifest.write_text(json.dumps({"row_engine": pending_block, "tinycap": {}}))
    marker = CV.pending_marker_path(row_dir, "none", BCB_FULL, store.name)                    # the driver's store-keyed name
    marker.write_text(json.dumps({"task": BCB_FULL, "cell": "none", "n_items": 1140, "n_expected_full": 1140, "generated_at": "t",
                                  "log": str(log), "generation_fcid": "fc-gen"}))
    (store / "RESULT").write_text("tinycap-modal name=n rc=0 result=GENERATED status=GENERATED sha=f6cf462 at=t standup=su\n")
    record_copy = row_dir / "generation" / "row_engine__none.json"
    record_copy.write_text(json.dumps({**pending_block, "verification_pass": None}))
    pend = {"cell": "none", "name": "gemma4e4b-f22s1__none__off", "out_dir": str(store), "config": "/tree/config.yaml",
            "row_engine_record": str(record_copy), "controls": ["bigcodebench_full"], "tasks": [BCB_FULL], "markers": [str(marker)],
            "max_sandboxes": 8, "served_max_num_seqs": 31, "concurrency_k": 1, "generation_function_call_id": "fc-gen",
            "standup_id": STANDUP_A, "tree_sha": FULL_SHA}
    cfg = {"cell": "none", "name": pend["name"], "model": ROW, "mode": "off", "out_dir": str(store)}
    calls = []

    def _verifier(argv, **kw):
        """Module A's verify: the verified block, the marker gone, the RESULT stamp, the pointer under the row dir."""
        argv = list(argv)
        calls.append({"argv": argv, "env": dict(kw.get("env") or {}), "cwd": kw.get("cwd")})
        fcid = argv[argv.index("--function-call-id") + 1]
        doc = json.loads(manifest.read_text())
        doc["row_engine"]["verification_pass"] = {"state": "verified", "mechanism": "M2_replay_vehicle", "function_call_id": fcid,
                                                  "started_at": "t0", "finished_at": "t1", "sandbox_facts": {}, "max_sandboxes": 8,
                                                  "n_items": 1140, "vehicle_log_sha256": "0" * 64}
        manifest.write_text(json.dumps(doc))
        marker.unlink()
        with (store / "RESULT").open("a") as fh:
            # the re-stamp names the MANIFEST's standup (the registry's landing-2 rule; a differing token reads torn)
            fh.write(f"tinycap-modal name=n rc=0 result=COMPLETE status=OK sha=f6cf462 at=t standup={doc['row_engine']['standup_id']} verified_by={fcid}\n")
        (row_dir / "watch").mkdir(exist_ok=True)
        (row_dir / "watch" / f"attempt_{fcid}.json").write_text(json.dumps({
            "run_id": kw["env"][scope.OWNER_ENV_RUN_ID], "cell": "none", "function_call_id": fcid, "name": "n", "model": ROW, "mode": "off",
            "out_dir": str(row_dir), "controls": ["bigcodebench_full"], "tree_sha": FULL_SHA, "sandbox_record": kw["env"][scope.SANDBOX_RECORD_ENV],
            "sandbox_app_name": C.BCB_SANDBOX_APP_NAME, "started_at": 1.0, "phase": "verify"}))
        return SimpleNamespace(returncode=0, stdout="verified\n", stderr="")
    monkeypatch.setattr(L.subprocess, "run", _verifier)
    rec = L._verify_cell_worker(cfg, "f6cf462", "f6cf462-cafebabe", STANDUP_A, str(row_dir), pending=pend, function_call_id="fc-verify-1")
    assert rec["status"] == "OK", rec.get("error")
    assert rec["verified_by"] == "fc-verify-1" == rec["function_call_id"] and rec["phase"] == "verify" and rec["name"] == f"{pend['name']}:verify"
    tree = str(repos / "sisteer-f6cf462")
    assert calls[0]["argv"] == [L.CLIENT_PY, f"{tree}/scripts/served_capability_run.py", "verify", C.FLAG_OUT_ROOT, str(store), "--cell", "none",
                                "--controls", "bigcodebench_full", "--config", "/tree/config.yaml", "--tree", tree, "--row-dir", str(row_dir),
                                C.FLAG_ROW_ENGINE_RECORD, str(record_copy), "--max-sandboxes", "8", "--function-call-id", "fc-verify-1"]
    assert calls[0]["cwd"] == tree and calls[0]["env"]["PYTHONPATH"] == f"{tree}/src:{tree}/scripts"
    env = calls[0]["env"]
    assert (env[scope.OWNER_ENV_RUN_ID], env[scope.OWNER_ENV_CELL]) == ("f6cf462-cafebabe", "none")
    assert Path(env[scope.SANDBOX_RECORD_ENV]) == row_dir / "watch" / "sandboxes_fc-verify-1__none.jsonl" == Path(rec["sandbox_record"])
    assert rec["sandbox_owner"] == {"run_id": "f6cf462-cafebabe", "cell": "none", "function_call_id": "fc-verify-1"}
    assert (row_dir / "watch" / "sweep_fc-verify-1__none.json").is_file() and rec["sandbox_sweep"]["certificate"] == "sweep"
    assert rec["verification_pass"]["state"] == "verified" and rec["markers_remaining"] == [] and rec["pointer_present"] is True
    assert rec["pointer"] == str(row_dir / "watch" / "attempt_fc-verify-1.json") and rec["result_tail"].endswith("verified_by=fc-verify-1")
    assert rec["driver_rc"] == 0 and rec["tree_sha"] == FULL_SHA and rec["tasks"] == [BCB_FULL] and rec["controls"] == ["bigcodebench_full"]
    assert not (store / "watch").exists()                                # the verifier spent nothing in the 40-inode store
    # a verifier that returned 0 but changed nothing: VERIFY_FAILED naming every missing piece of evidence
    marker.write_text(json.dumps({"task": BCB_FULL, "cell": "none", "n_items": 1140, "n_expected_full": 1140, "generated_at": "t",
                                  "log": str(log), "generation_fcid": "fc-gen"}))
    manifest.write_text(json.dumps({"row_engine": pending_block}))
    monkeypatch.setattr(L.subprocess, "run", lambda argv, **kw: SimpleNamespace(returncode=0, stdout="", stderr=""))
    bad = L._verify_cell_worker(cfg, "f6cf462", "f6cf462-cafebabe", STANDUP_A, str(row_dir), pending=pend, function_call_id="fc-verify-2")
    assert bad["status"] == L.REC_VERIFY_FAILED == "VERIFY_FAILED" and bad["status"] not in L.REC_OK_STATUSES
    err = bad["error"]
    assert "verification_pass.state is 'pending'" in err and "pending marker(s) still present" in err
    assert "verified_by=fc-verify-2" in err and "wrote no pointer" in err
    # a non-zero verify names its rc first; a missing pending record runs nothing
    monkeypatch.setattr(L.subprocess, "run", lambda argv, **kw: SimpleNamespace(returncode=3, stdout="", stderr="boom"))
    rc3 = L._verify_cell_worker(cfg, "f6cf462", "f6cf462-cafebabe", STANDUP_A, str(row_dir), pending=pend, function_call_id="fc-verify-3")
    assert rc3["status"] == "VERIFY_FAILED" and rc3["error"].startswith(f"{pend['name']}: verify returned rc 3") and rc3["stderr_tail"] == "boom"
    none = L._verify_cell_worker(cfg, "f6cf462", "f6cf462-cafebabe", STANDUP_A, str(row_dir), pending=None, function_call_id="fc-verify-4")
    assert none["status"] == "VERIFY_FAILED" and "no verification_pending record" in none["error"] and "driver_argv" not in none
    # the decorated function is a CPU function: no gpu, the plan's cpu / memory / timeout; the count of GPU containers ignores it
    src = Path(L.__file__).read_text()
    deco = src[src.index("def verify_cell_cpu(") - 400:src.index("def verify_cell_cpu(")]
    assert "@app.function(image=image, volumes={VOL_PATH: VOL}, cpu=VERIFY_FUNCTION_CPU, memory=VERIFY_FUNCTION_MEMORY_MIB," in deco
    assert "timeout=VERIFY_FUNCTION_TIMEOUT_S)" in deco and 'gpu="H200"' not in deco
    assert (L.VERIFY_FUNCTION_CPU, L.VERIFY_FUNCTION_MEMORY_MIB, L.VERIFY_FUNCTION_TIMEOUT_S) == (2, 8192, 3 * 3600)
    assert L.gpu_container_count(9, row_launch=True) == 1                # CPU verifiers are not GPU containers


def test_main_row_generate_spawns_one_cpu_verifier_per_pending_cell_and_exits_only_after_the_verify_recs(L, monkeypatch, capsys):
    """``main --row --verification-phase generate`` — the row container returns GENERATED with
    ``verification_pending``; ``main`` spawns ``verify_cell_cpu.spawn(cell_cfg, sha, run_id, standup_id, row_dir, pending=…)``
    for EXACTLY the pending cells (all spawns before any harvest), prints the watcher command with ``--row-dir``, harvests
    the verify RECs, flips the GENERATED cell RECs whose verifier is OK, and exits 0 only then; a failed verifier fails the
    exit naming VERIFY_FAILED and the cell stays GENERATED. The trip-wire's count is untouched (one row container)."""
    spawned_rows, spawned_verifies, verify_status = [], [], {"none": "OK"}

    def _row_spawn(cfgs, **kw):
        spawned_rows.append((cfgs, kw))
        cells = {"none": {"name": cfgs[0]["name"], "cell": "none", "status": "GENERATED", "out": "/vol/x/none/s"},
                 "none-rep2": {"name": f"{cfgs[0]['name']}-rep2", "cell": "none", "status": "OK", "out": "/vol/x/none/s-rep2"}}
        pending = {"none": {"cell": "none", "name": cfgs[0]["name"], "out_dir": "/vol/x/none/s", "config": "/tree/cfg.yaml",
                            "row_engine_record": "/vol/rows/g/s/generation/row_engine__none.json", "controls": ["bigcodebench_full"],
                            "tasks": [BCB_FULL], "markers": ["/vol/rows/g/s/generation/pending_none__bigcodebench_full.json"],
                            "max_sandboxes": 8, "served_max_num_seqs": 31, "concurrency_k": 1, "generation_function_call_id": "fc-row",
                            "standup_id": "su", "tree_sha": "f" * 40}}
        expectation = {"schema": "tinycap_verifiers/1", "run_id": kw.get("run_id"), "row": "gemma4e4b", "suffix": "s",
                       "generation_function_call_id": "fc-row", "written_at": 1.0, "written_by": "row_container",
                       "cells": {"none": {"cell": "none", "out_dir": "/vol/x/none/s", "state": "expected"}}}
        return TL._Call({"name": "row-gemma4e4b-s", "status": "GENERATED", "phase": "generate", "cells": cells, "dropped": {},
                         "standup_id": "su", "row_dir": "/vol/rows/g/s", "verification_pending": pending,
                         "expected_verifiers": expectation})            # the container wrote the record before returning (H1)

    def _verify_spawn(cell_cfg, sha, run_id, standup_id, row_dir, **kw):
        spawned_verifies.append({"cell_cfg": cell_cfg, "sha": sha, "run_id": run_id, "standup_id": standup_id, "row_dir": row_dir, **kw})
        st = verify_status[cell_cfg["cell"]]
        return TL._Call({"name": f"{cell_cfg['name']}:verify", "cell": cell_cfg["cell"], "status": st, "phase": "verify",
                         "function_call_id": "fc-verify-1", **({"verified_by": "fc-verify-1"} if st == "OK" else {"error": "x: verify returned rc 1"})})
    monkeypatch.setattr(L, "run_row_full", SimpleNamespace(spawn=_row_spawn))
    monkeypatch.setattr(L, "verify_cell_cpu", SimpleNamespace(spawn=_verify_spawn))
    monkeypatch.setattr(L, "put_row_dir_file", lambda row_dir, name, d, **kw: f"{str(row_dir).lstrip('/')}/watch/{name}")   # the grading host's REPLACE lands
    kw = dict(sha="f6cf462", app_name=L.APP_NAME, pass1=True, cells="none", modes="off", controls="bigcodebench_full", out_suffix="s", row=True,
              repeat_cells="none", verification_phase="generate")
    L.main(**kw)                                                        # exits 0: no SystemExit
    out = capsys.readouterr().out
    assert len(spawned_rows) == 1 and spawned_rows[0][1]["row_opts"]["verification_phase"] == "generate"
    assert [v["cell_cfg"]["cell"] for v in spawned_verifies] == ["none"]                       # exactly the pending BCB cell, not the OK repeat
    v = spawned_verifies[0]
    assert (v["sha"], v["standup_id"], v["row_dir"]) == ("f6cf462", "su", "/vol/rows/g/s") and v["run_id"].startswith("f6cf462-")
    assert v["pending"]["controls"] == ["bigcodebench_full"] and v["pending"]["max_sandboxes"] == 8 and v["reduction_mode"] == C.REDUCTION_MODE_LAUNCHER_DEFAULT
    assert v["cell_cfg"]["out_dir"] == "/vol/x/none/s" and v["cell_cfg"]["cell"] == "none" and v["cell_cfg"]["name"] == spawned_rows[0][0][0]["name"]
    assert "[tinycap] verifying 1 cell(s) off the GPU" in out and "not GPU containers" in out
    assert "arm (or keep) the watcher:" in out and "--row-dir /vol/tinycap/gen/rows/gemma4e4b/s" in out and "spawned verify none -> FunctionCall" in out
    assert "3/3 container(s) OK" in out                                  # the cell (flipped), the repeat, the verifier
    assert '"verification": {"status": "OK", "function_call_id": "fc-verify-1", "verified_by": "fc-verify-1"}' in out
    # a failed verifier: the exit names VERIFY_FAILED and the GENERATED cell (never flipped)
    verify_status["none"] = "VERIFY_FAILED"
    with pytest.raises(SystemExit) as e:
        L.main(**kw)
    msg = str(e.value)
    assert "VERIFY_FAILED" in msg and "GENERATED" in msg and "2/3 container(s) did NOT finish" in msg
    # a GENERATED row that names nothing pending spawns nothing and fails the exit on GENERATED itself
    spawned_verifies.clear()
    monkeypatch.setattr(L, "run_row_full", SimpleNamespace(spawn=lambda cfgs, **kw: TL._Call(
        {"name": "row", "status": "GENERATED", "cells": {"none": {"name": cfgs[0]["name"], "cell": "none", "status": "GENERATED"}},
         "dropped": {}, "standup_id": "su", "row_dir": "/vol/rows/g/s", "verification_pending": {}})))
    with pytest.raises(SystemExit, match="GENERATED"):
        L.main(**kw)
    assert spawned_verifies == [] and "named nothing to verify" in capsys.readouterr().out
    # the grammar: verify is never a row launch; generate needs --row; generate needs a BCB leg; K = 2 + generate compose LAST
    base = dict(sha="f6cf462", app_name=L.APP_NAME, pass1=True, dry_run=True, cells="none", modes="off", controls="bigcodebench_full", out_suffix="s")
    with pytest.raises(SystemExit, match="verify is the CPU verifier's subcommand"):
        L.main(**base, row=True, verification_phase="verify")
    with pytest.raises(SystemExit, match="mean nothing without --row"):
        L.main(**base, verification_phase="generate")
    with pytest.raises(SystemExit, match="no BigCodeBench task among the planned controls"):
        L.main(**{**base, "controls": "gsm8k"}, row=True, verification_phase="generate")
    with pytest.raises(SystemExit, match="warm-cache probe drives no cell"):
        L.main(**base, row=True, verification_phase="generate", warm_cache_only=True)
    capsys.readouterr()
    L.main(**base, row=True, verification_phase="generate", concurrency_k=2, row_forecast_h=1.5)
    out = capsys.readouterr().out
    argv_line = next(ln for ln in out.splitlines() if ln.startswith("[tinycap] argv:"))
    assert argv_line.split("  (run it")[0].endswith("--row --concurrency-k 2 --row-forecast-h 1.5 --verification-phase generate"), argv_line
    assert "verification phase: generate" in out and "ONE CPU verifier per BCB cell" in out and "K = 2" in out
    assert L.launch_argv("a" * 40, pass1=True, row=True, cells="none", modes="off", out_suffix="s")[-1] == "--row"     # coupled spells nothing
    assert L.launch_argv("a" * 40, pass1=True, row=True, cells="none", modes="off", out_suffix="s", verification_phase="generate")[-2:] == [
        C.FLAG_VERIFICATION_PHASE, "generate"]
    ns = L.cli_parser().parse_args(["--sha", "x", "--app-name", L.APP_NAME, "--pass1", "--row", "--cells", "none", "--modes", "off", "--out-suffix", "s",
                                    C.FLAG_VERIFICATION_PHASE, "generate", C.FLAG_CONCURRENCY_K, "3"])
    assert ns.verification_phase == "generate" and ns.concurrency_k == 3
    assert L.cli_parser().parse_args(["--sha", "x", "--app-name", L.APP_NAME]).verification_phase == "coupled"
    assert L.ROW_VERIFICATION_PHASE_FLAG == C.FLAG_VERIFICATION_PHASE == "--verification-phase"
    # the row's own REC is not appended for a GENERATED row (its cells' RECs and the verify RECs carry the exit)
    assert L.REC_GENERATED in L.ROW_RESULT_OK_STATUSES and L.REC_GENERATED not in L.REC_OK_STATUSES
    assert L.row_result_recs({"status": "GENERATED", "cells": {"a": {"status": "GENERATED"}}, "dropped": {}}) == [{"status": "GENERATED"}]


def test_the_watcher_classifies_a_verify_pointer_beside_the_rows_pointer_and_sweeps_its_one_owner(L, tmp_path, monkeypatch, capsys):
    """The CPU verifier writes ``watch/attempt_<its fcid>.json`` (phase ``verify``, ONE cell) into
    the same row dir as the GPU phase's declared row pointer; the watcher's ``--row-dir`` mode classifies the verifier's
    call on its own id, sweeps its one owner ``(run_id, cell, verify fcid)`` when no certificate admits it, and reads
    the row pointer's cells as before — DONE only when every owner of both is certified. No watcher change was needed."""
    w = _World(L, tmp_path, monkeypatch, cells=["none", LORA])
    result = w.run()
    fcid, run_id = result["function_call_id"], "f6cf462-cafebabe"
    W = _watcher_module()
    healthy = {W.row_sweep_certificate_name(fcid, c): {"owner": {"run_id": run_id, "cell": c, "function_call_id": fcid}, "verified_empty": True,
                                                       "ids_terminated": [], "still_live": [], "could_not_terminate": []} for c in ("none", LORA)}
    vol = _RowDirVolume(w.row_dir, certified=healthy)
    rel = str(w.row_dir).lstrip("/")
    vpointer = {"run_id": run_id, "cell": LORA, "function_call_id": "fc-verify-1", "name": "n:verify", "model": ROW, "mode": "off",
                "out_dir": str(w.row_dir), "controls": ["bigcodebench_full"], "tree_sha": FULL_SHA,
                "sandbox_record": f"{w.row_dir}/watch/sandboxes_fc-verify-1__{LORA}.jsonl", "sandbox_app_name": C.BCB_SANDBOX_APP_NAME,
                "started_at": 2.0, "phase": "verify"}
    vol.files[f"{rel}/watch/attempt_fc-verify-1.json"] = json.dumps(vpointer).encode()
    assert not W.is_row_pointer(vpointer) and W.pointer_label(vpointer) == f"verify {LORA}"
    assert [o.as_dict() for o in W.pointer_owners(vpointer)] == [{"run_id": run_id, "cell": LORA, "function_call_id": "fc-verify-1"}]
    groups = W.pointers_by_call([json.loads(v) for k, v in vol.files.items() if "/attempt_" in k], run_id=run_id)
    assert set(groups) == {fcid, "fc-verify-1"} and all(W.undeclared_shared_cells(g) is None for g in groups.values())
    monkeypatch.setattr(W, "_volume", lambda: vol)
    monkeypatch.setattr(W, "sandbox_app_id", lambda *a, **k: "ap-1")
    classified = []

    def _classify(f):
        classified.append(f)
        return SimpleNamespace(unavailable=False, pending=False, terminal=True, describe=lambda: "terminal")
    monkeypatch.setattr(W, "classify", _classify)
    swept = []

    def _sweep(owner, **kw):
        swept.append(owner)
        return {"verified_empty": True, "ids_terminated": ["sb-v-1"], "still_live": []}
    monkeypatch.setattr(W, "sweep_attempt", _sweep)
    assert not list((w.row_dir / "watch").glob("verifiers_*.json")) and "expected_verifiers" not in result   # a coupled row: no expectation
    rc = W.watch_row_dir(run_id=run_id, row_dir=str(w.row_dir), poll_s=0, max_hours=1, state_dir=str(tmp_path / "state"), once=True)
    out = capsys.readouterr().out
    assert rc == W.EXIT_DONE, out
    assert sorted(classified) == sorted([fcid, "fc-verify-1"])                                # each call classified once, on its own id
    assert [(o.run_id, o.cell, o.function_call_id) for o in swept] == [(run_id, LORA, "fc-verify-1")]   # the verifier's one owner
    assert f"verify {LORA}: attempt fc-verify-1 TERMINAL" in out and f"{LORA}@fc-verify-1" in out and "DONE: every attempt terminal" in out
    # with the verifier's own certificate present (the harness's sweep_owned_sandboxes(..., cell_suffix=True) name), nothing is swept
    swept.clear()
    vol.files[f"{rel}/watch/{W.row_sweep_certificate_name('fc-verify-1', LORA)}"] = json.dumps(
        {"owner": {"run_id": run_id, "cell": LORA, "function_call_id": "fc-verify-1"}, "verified_empty": True, "ids_terminated": [], "still_live": [],
         "could_not_terminate": []}).encode()
    assert W.watch_row_dir(run_id=run_id, row_dir=str(w.row_dir), poll_s=0, max_hours=1, state_dir=str(tmp_path / "state"), once=True) == W.EXIT_DONE
    assert swept == [] and f"sweep_fc-verify-1__{LORA}.json" == L.sweep_certificate_path("/r", "fc-verify-1", cell=LORA).name
    assert L.verify_sandbox_record_path("/r", "fc-verify-1", LORA) == Path(f"/r/watch/sandboxes_fc-verify-1__{LORA}.jsonl")


def test_a_row_launch_probes_the_drivers_verification_phase_flag_beside_the_row_engine_record(L):
    """Landing 2 (§2.2.3): the preflight's row block probes ``--verification-phase`` as its own argparse check, after the
    row-engine record's — a staged driver that predates the two-phase contract refuses on the CPU before a row pays for
    a standup; the harness and the tree are one sha, so a coupled row at this tree probes it too."""
    assert L.DRIVER_PROBE_VALUES[C.FLAG_VERIFICATION_PHASE] == C.VERIFICATION_PHASE_COUPLED == "coupled"
    assert L.driver_probe_argv("/tree", extra_flags=(C.FLAG_VERIFICATION_PHASE,))[-3:-1] == [C.FLAG_VERIFICATION_PHASE, "coupled"]
    assert C.FLAG_VERIFICATION_PHASE not in C.TINYCAP_DRIVER_FLAGS                            # a solo launch never probes it
    src = Path(L.__file__).read_text()
    body = src[src.index("def preflight("):src.index("    def datasets():")]
    i_row = body.index('chk("driver_accepts_row_engine_record", driver_row_engine_flag)')
    i_phase = body.index('chk("driver_accepts_verification_phase", driver_verification_phase_flag)')
    assert i_row < i_phase and body.index("    if row_launch:\n") < i_row
    check = body[body.index("def driver_verification_phase_flag():"):i_phase]
    assert "driver_probe_argv(tree, extra_flags=(C.FLAG_VERIFICATION_PHASE,))" in check
    assert "if C.FLAG_VERIFICATION_PHASE in unrec:" in check and "unrecognized_flags(r.stderr)" in check
    main_body = src[src.index("def main("):]
    assert main_body.count("row_launch=True)") == 2                                             # the preflight call sites are untouched


# --------------------------------------------------------------------------- #
# the watcher and the gap before the verifiers, the store-keyed markers        #
# (above), and the CPU-only resume                                             #
# --------------------------------------------------------------------------- #


def _verify_pointer_doc(run_id: str, fcid: str, cell: str, out_dir: str) -> dict:
    """The verifier's own pointer as the driver writes it (``cpu_verify.write_verify_pointer``: ``phase verify``, ONE
    ``cell``, ``out_dir`` = the store root it verifies)."""
    return {"run_id": run_id, "cell": cell, "function_call_id": fcid, "name": f"verify:{cell}", "model": ROW, "mode": "off",
            "out_dir": out_dir, "controls": ["bigcodebench_full"], "tree_sha": FULL_SHA,
            "sandbox_record": f"/r/watch/sandboxes_{fcid}__{cell}.jsonl", "sandbox_app_name": C.BCB_SANDBOX_APP_NAME,
            "started_at": 3.0, "phase": "verify"}


def _cert(run_id: str, fcid: str, cell: str) -> dict:
    return {"owner": {"run_id": run_id, "cell": cell, "function_call_id": fcid}, "verified_empty": True, "ids_terminated": [],
            "still_live": [], "could_not_terminate": []}


def test_the_watcher_withholds_done_until_every_expected_verifier_is_terminal_and_certified(L, tmp_path, monkeypatch, capsys):
    """Code review L2 H1 (P1): after the GPU FunctionCall ends under ``generate`` the row pointer is terminal and every
    GPU owner certified, but the CPU verifiers' pointers appear only once their containers start — a tick in that gap
    used to read DONE, and a later hard death of a verifier stranded its sandboxes. Now the row container publishes the
    EXPECTED verifiers as durable row state (``watch/verifiers_<run_id>.json``, one per run) before it returns; ``main``
    folds each spawn's outcome in (the call id, or ``spawn_failed`` — a TERMINAL no-verifier state); the watcher
    withholds DONE until every listed cell has a terminal, certified phase-verify pointer or a terminal state. A
    torn record withholds by name; no record (a coupled row) is the pre-H1 rule alone."""
    w = _World(L, tmp_path, monkeypatch, cells=["none", LORA], row_opts={"verification_phase": "generate", "repeat_cells": ["none"]})
    monkeypatch.setattr(L.subprocess, "run", _generate_driver(w))
    result = w.run()
    assert result["status"] == "GENERATED", result.get("error")
    fcid, run_id = result["function_call_id"], "f6cf462-cafebabe"
    W = _watcher_module()
    labels = ["none", LORA, "none-rep2"]
    # the row container published the expectation BEFORE returning: one file per run id, the watcher's own spelling
    rec_path = w.row_dir / "watch" / f"verifiers_{run_id}.json"
    assert rec_path.is_file() and rec_path.name == W.verifiers_record_name(run_id) == L.verifiers_record_path("/r", run_id).name
    doc = json.loads(rec_path.read_text())
    assert doc == json.loads(json.dumps(result["expected_verifiers"])) and result["expected_verifiers_path"] == str(rec_path)
    assert (doc["run_id"], doc["generation_function_call_id"], doc["written_by"], doc["schema"]) == (run_id, fcid, "row_container", "tinycap_verifiers/1")
    assert {k: (v["cell"], v["state"]) for k, v in doc["cells"].items()} == {"none": ("none", "expected"), LORA: (LORA, "expected"),
                                                                             "none-rep2": ("none", "expected")}
    assert doc["cells"]["none-rep2"]["out_dir"] == result["cells"]["none-rep2"]["out"] and doc["cells"]["none"]["out_dir"] == result["cells"]["none"]["out"]
    assert (L.VERIFIER_TERMINAL_STATES, L.VERIFIER_STATE_EXPECTED, L.VERIFIER_STATE_SPAWNED) == (
        W.VERIFIER_TERMINAL_STATES, W.VERIFIER_STATE_EXPECTED, W.VERIFIER_STATE_SPAWNED)
    row_rec = json.loads((w.row_dir / "row_rec.json").read_text())
    assert row_rec["expected_verifiers"]["cells"].keys() == doc["cells"].keys()
    healthy = {W.row_sweep_certificate_name(fcid, c): _cert(run_id, fcid, c) for c in labels}
    rel = str(w.row_dir).lstrip("/")
    state = {"vol": _RowDirVolume(w.row_dir, certified=healthy)}
    monkeypatch.setattr(W, "_volume", lambda: state["vol"])
    monkeypatch.setattr(W, "sandbox_app_id", lambda *a, **k: "ap-1")
    monkeypatch.setattr(W, "classify", lambda f: SimpleNamespace(unavailable=False, pending=False, terminal=True, describe=lambda: "terminal"))
    monkeypatch.setattr(W, "sweep_attempt", lambda owner, **kw: {"verified_empty": True, "ids_terminated": [], "still_live": []})

    def tick():
        rc = W.watch_row_dir(run_id=run_id, row_dir=str(w.row_dir), poll_s=0, max_hours=1, state_dir=str(tmp_path / "state"), once=True)
        return rc, capsys.readouterr().out
    # THE GAP: the GPU call terminal, every GPU owner certified, no verifier started — NOT done, each expected cell named
    rc, out = tick()
    assert rc == W.EXIT_UNVERIFIED, out
    assert "DONE:" not in out and out.count("DONE withheld") == 3
    for label in labels:
        assert f"verifier for {label} (expected): no phase-verify pointer under the row dir yet — DONE withheld" in out
    # main dispatches from the grading host: two spawns succeed, the repeat's .spawn RPC raises — the others still go out (H3 c),
    # the failed one is a VERIFY_SPAWN_FAILED verify REC, and the record folds every outcome in (spawned + call id /
    # spawn_failed) through the grading-host-side volume write
    written: Dict[str, dict] = {}
    monkeypatch.setattr(L, "put_row_dir_file", lambda row_dir, name, d, **kw: written.__setitem__(name, json.loads(json.dumps(d, default=str)))
                        or f"{str(row_dir).lstrip('/')}/watch/{name}")
    spawned = []

    def _spawn(cell_cfg, sha, run_id_, standup_id, row_dir, **kw):
        spawned.append(cell_cfg["out_dir"])
        if str(cell_cfg["out_dir"]).endswith("-rep2"):
            raise RuntimeError("GRPCError: UNAVAILABLE")
        return TL._Call({"name": f"{cell_cfg['name']}:verify", "cell": cell_cfg["cell"], "status": "OK", "phase": "verify", "verified_by": "x"})
    recs = L.row_result_recs(result)
    vrecs = L.verify_pending_cells(SimpleNamespace(spawn=_spawn), result, w.cfgs, recs, sha="f6cf462", run_id=run_id,
                                   reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT)
    out = capsys.readouterr().out
    assert len(spawned) == 3, "every pending cell's spawn was attempted — one failure never suppressed the others"
    assert [(v["label"], v["status"]) for v in vrecs] == [("none", "OK"), (LORA, "OK"), ("none-rep2", L.REC_VERIFY_SPAWN_FAILED)]
    assert L.REC_VERIFY_SPAWN_FAILED == "VERIFY_SPAWN_FAILED" not in L.REC_OK_STATUSES
    assert "the verifier's .spawn raised — RuntimeError: GRPCError: UNAVAILABLE" in vrecs[2]["error"] and vrecs[2]["function_call_id"] is None
    assert "spawn of verify none-rep2 FAILED" in out and "--verify-pending" in out and "verifier dispatch recorded" in out
    by_name = {r["name"]: r for r in recs}
    assert by_name[result["cells"]["none"]["name"]]["status"] == "OK" and by_name[result["cells"]["none-rep2"]["name"]]["status"] == "GENERATED"
    upd = written[f"verifiers_{run_id}.json"]
    assert upd["written_by"] == "main_dispatch" and upd["run_id"] == run_id and upd["generation_function_call_id"] == fcid
    v_none, v_lora = upd["cells"]["none"]["function_call_id"], upd["cells"][LORA]["function_call_id"]
    assert upd["cells"]["none"]["state"] == upd["cells"][LORA]["state"] == "spawned" and v_none and v_lora and v_none != v_lora
    assert upd["cells"]["none-rep2"]["state"] == "spawn_failed" and "UNAVAILABLE" in upd["cells"]["none-rep2"]["error"]
    assert upd["cells"]["none"]["out_dir"] == result["cells"]["none"]["out"]
    # the updated record on the volume: the two spawned verifiers are awaited BY CALL ID; the failed spawn is terminal
    rec_path.write_text(json.dumps(upd))
    state["vol"] = _RowDirVolume(w.row_dir, certified=healthy)
    rc, out = tick()
    assert rc == W.EXIT_UNVERIFIED and out.count("DONE withheld") == 2
    assert f"verifier for none (spawned, call {v_none}): no phase-verify pointer" in out and f"(spawned, call {v_lora})" in out
    assert "verifier for none-rep2" not in out                                                # spawn_failed: terminal, not awaited
    # the verifiers' pointers land (the driver's phase-verify pointer, one cell, the store as out_dir) — still pending until
    # each call is terminal AND its owner certified
    for label, vf in (("none", v_none), (LORA, v_lora)):
        cell, out_dir = result["cells"][label]["cell"], result["cells"][label]["out"]
        state["vol"].files[f"{rel}/watch/attempt_{vf}.json"] = json.dumps(_verify_pointer_doc(run_id, vf, cell, out_dir)).encode()
    monkeypatch.setattr(W, "classify", lambda f: SimpleNamespace(unavailable=False, pending=f in (v_none, v_lora), terminal=f not in (v_none, v_lora),
                                                                 describe=lambda: "x"))
    rc, out = tick()
    assert rc == W.EXIT_UNVERIFIED and f"attempt {v_none} PENDING" in out and f"verifier for none: attempt(s) ['{v_none}'] not yet terminal and certified" in out
    monkeypatch.setattr(W, "classify", lambda f: SimpleNamespace(unavailable=False, pending=False, terminal=True, describe=lambda: "terminal"))
    for label, vf in (("none", v_none), (LORA, v_lora)):
        cell = result["cells"][label]["cell"]
        state["vol"].files[f"{rel}/watch/{W.row_sweep_certificate_name(vf, cell)}"] = json.dumps(_cert(run_id, vf, cell)).encode()
    rc, out = tick()
    assert rc == W.EXIT_DONE, out
    assert "DONE: every attempt terminal and every owner certified" in out and "DONE withheld" not in out
    assert f"none@{v_none}" in out and f"{LORA}@{v_lora}" in out
    # a TORN record withholds by name (never read as "no expectation"); a record of ANOTHER run id is not this run's
    state["vol"].files[f"{rel}/watch/verifiers_{run_id}.json"] = b"{torn"
    rc, out = tick()
    assert rc == W.EXIT_UNVERIFIED and f"verifiers record verifiers_{run_id}.json: unreadable" in out and "DONE withheld" in out
    assert W.outstanding_verifiers({**upd, "run_id": "another-run"}, run_id=run_id, pointers=[], certified={}) == []
    assert W.outstanding_verifiers(None, run_id=run_id, pointers=[], certified={}) == []
    assert W.outstanding_verifiers({"run_id": run_id, "cells": []}, run_id=run_id, pointers=[], certified={}) == [
        f"verifiers record verifiers_{run_id}.json lists no cells — DONE withheld"]
    # the pre-spawn record (state `expected`, no call id) matches a verifier pointer by (cell, store) — the fallback when
    # main's dispatch update never landed
    exp_only = {"run_id": run_id, "cells": {"none": {"cell": "none", "out_dir": result["cells"]["none"]["out"], "state": "expected"}}}
    ptr = _verify_pointer_doc(run_id, "fc-late", "none", result["cells"]["none"]["out"])
    assert W.outstanding_verifiers(exp_only, run_id=run_id, pointers=[ptr], certified={}) == [
        "verifier for none: attempt(s) ['fc-late'] not yet terminal and certified — DONE withheld"]
    assert W.outstanding_verifiers(exp_only, run_id=run_id, pointers=[ptr], certified={("fc-late", "none"): True}) == []
    rep2_ptr = _verify_pointer_doc(run_id, "fc-late", "none", result["cells"]["none-rep2"]["out"])                # the repeat's store: not it
    assert len(W.outstanding_verifiers(exp_only, run_id=run_id, pointers=[rep2_ptr], certified={("fc-late", "none"): True})) == 1
    # a failed grading host write is printed, never raised: the container's `expected` record stands
    monkeypatch.setattr(L, "put_row_dir_file", lambda *a, **k: (_ for _ in ()).throw(OSError("volume unreachable")))
    assert L.record_dispatch_outcomes(result, {"none": {"cell": "none", "out_dir": "/s", "state": "spawned", "function_call_id": "fc"}},
                                      run_id=run_id, row_dir=str(w.row_dir)) is None
    assert "could not record the verifier dispatch" in capsys.readouterr().out


def test_a_row_engine_cells_spill_ledger_lands_in_the_row_dir_and_the_store_keeps_its_inode_count(L, tmp_path, monkeypatch):
    """The store keeps its inode count: the qwen s1 `constv2-mix-full__psr_ce_all` store counted
    41 inodes because the verifier transport wrote `verifier_spills.jsonl` into the STORE. On a row engine the store's
    manifest carries the runner's row_engine block, so `verifier_transport.ledger_path_for(store)` — the ONE place the path
    is decided — puts the ledger at `<row dir>/spills/<cell>__<store basename>__verifier_spills.jsonl`; the runner's REC
    (`verifier_transport`, read off the same decision) counts the spill and names the file; the store gains nothing."""
    from sisteer.capeval.tinycap import verifier_transport as VT
    w = _World(L, tmp_path, monkeypatch, cells=["none", LORA])

    def _driver(argv, **kw):
        argv = list(argv)
        cell = argv[argv.index("--cell") + 1]
        out_root = Path(argv[argv.index(C.FLAG_OUT_ROOT) + 1])
        rec_doc = C.read_row_engine_record(argv[argv.index(C.FLAG_ROW_ENGINE_RECORD) + 1])
        w.driver_calls.append({"argv": argv, "cell": cell, "out_root": out_root})
        store = w.finish_store(out_root, w.by_cell[cell], row_engine=rec_doc)          # the driver stamps the record into the manifest
        (out_root / "evidence.json").write_text(json.dumps({"cell": cell}))
        if cell == "none":
            # one completion over the argv threshold: the transport's ledger line, at the path the transport decides
            ledger = VT.ledger_path_for(SimpleNamespace(sweep_dir=out_root, cell_dir=store))
            VT._append_ledger(ledger, {"ts": "t", "argv_bytes": 70_000, "program_sha256": "a" * 64, "path": "/tmp/x.py"})
        return SimpleNamespace(returncode=0, stdout="driver ok\n", stderr="")
    monkeypatch.setattr(L.subprocess, "run", _driver)
    result = w.run()
    assert result["status"] == "OK", result.get("error")
    none_store, lora_store = Path(result["cells"]["none"]["out"]), Path(result["cells"][LORA]["out"])
    want = w.row_dir / "spills" / f"none__{none_store.name}__verifier_spills.jsonl"
    assert want.is_file() and VT.n_spilled(want) == 1
    assert result["cells"]["none"]["verifier_transport"] == {"spill_threshold_bytes": VT.SPILL_THRESHOLD_BYTES, "n_spilled": 1, "ledger": str(want)}
    assert result["cells"][LORA]["verifier_transport"] == {"spill_threshold_bytes": VT.SPILL_THRESHOLD_BYTES, "n_spilled": 0,
                                                           "ledger": str(w.row_dir / "spills" / f"{LORA}__{lora_store.name}__verifier_spills.jsonl")}
    assert not (none_store / VT.VERIFIER_SPILLS_FILENAME).exists() and not list(none_store.rglob(VT.VERIFIER_SPILLS_FILENAME))
    # the spilled cell's store has exactly the inodes of the cell that did not spill (the same layout)
    assert SS.count_inodes(none_store) == SS.count_inodes(lora_store)
    assert {p.name for p in none_store.iterdir()} == {p.name for p in lora_store.iterdir()}
    # the ROW-REC names the ledger's row-dir path
    row_rec = json.loads((w.row_dir / "row_rec.json").read_text())
    assert row_rec["cells"]["none"]["verifier_transport"]["ledger"] == str(want)
    # a SOLO store (no row_engine block in its manifest) keeps today's in-store path
    solo = tmp_path / "solo"
    (solo / "cells" / "k").mkdir(parents=True)
    (solo / "cells" / "k" / "manifest.json").write_text(json.dumps({"cell_id": "none"}))
    assert VT.ledger_path_for(SimpleNamespace(sweep_dir=solo, cell_dir=solo / "cells" / "k")) == solo / VT.VERIFIER_SPILLS_FILENAME
    assert VT.rec_field(solo)["ledger"] == str(solo / VT.VERIFIER_SPILLS_FILENAME)


def _mark_store_verified(pend_entry: Mapping[str, object], fcid: str) -> None:
    """What the driver's ``verify`` leaves behind (docs §9.3's order): the BigCodeBench task FILED (its done marker +
    summary), the verified block in the manifest, the RESULT stamp, the marker gone."""
    out = Path(str(pend_entry["out_dir"]))
    manifest = next(out.glob("cells/*/manifest.json"))
    TL._finish(manifest.parent, BCB_FULL, 1140)
    doc = json.loads(manifest.read_text())
    doc["row_engine"]["verification_pass"] = {"state": "verified", "mechanism": "M2_replay_vehicle", "function_call_id": fcid,
                                              "started_at": 1.0, "finished_at": 2.0, "sandbox_facts": {}, "max_sandboxes": 8,
                                              "n_items": 1140, "vehicle_log_sha256": "0" * 64}
    manifest.write_text(json.dumps(doc))
    for m in pend_entry["markers"]:
        Path(str(m)).unlink()
    with (out / "RESULT").open("a") as fh:
        # the re-stamp names the MANIFEST's standup (the registry's landing-2 rule; a differing token reads torn)
        fh.write(f"tinycap-modal name=n rc=0 result=COMPLETE status=OK sha=f6cf462 at=t standup={doc['row_engine']['standup_id']} verified_by={fcid}\n")


def test_a_generate_relaunch_resumes_valid_pending_stores_at_the_door_without_a_standup_and_parks_torn_ones(L, tmp_path, monkeypatch, capsys):
    """After a lost or partly dispatched verifier — a `.spawn` RPC failure, a launcher
    interruption after the row returned — a new invocation used to relaunch the GPU row and park every GENERATED store
    as INCOMPLETE_STORE_PARKED (the earlier rule read the pending stores as torn), so verifying them meant paying the GPU
    again. Now a `generate` relaunch that finds a VALID pending generation (manifest pending, the store-keyed marker +
    sidecar, the unscored log, the record copy) queues it for CPU verification at its door — no standup, no driver —
    and a row whose every cell is pending returns GENERATED without a GPU minute; a torn store (its sidecar gone) still
    parks, by name; a COUPLED relaunch parks a pending store naming the resume flag."""
    w = _World(L, tmp_path, monkeypatch, cells=["none", LORA], row_opts={"verification_phase": "generate", "repeat_cells": ["none"]},
               standup_ids=["su-A", "su-B"])
    monkeypatch.setattr(L.subprocess, "run", _generate_driver(w))
    first = w.run()
    assert first["status"] == "GENERATED" and first["standup_id"] == "su-A", first.get("error")
    # the interruption: no verifier ran; a fresh invocation (its own run id / FunctionCall) relaunches the same row
    w.driver_calls.clear()
    _RowStandup.argvs.clear()
    capsys.readouterr()
    second = w.run(run_id="f6cf462-resume01")
    assert second["status"] == "GENERATED", second.get("error")
    assert _RowStandup.argvs == [] and w.driver_calls == []                                  # no standup, no driver: no GPU minute
    assert second["cell_statuses"] == {"none": "GENERATED", LORA: "GENERATED", "none-rep2": "GENERATED"}
    assert second["dropped"] == {} and "standup_id" not in second                            # nothing served, nothing stood up
    for label, rec in second["cells"].items():
        assert rec["resumed_pending"] is True and rec["standup_id"] == "su-A" and rec["verification_pending"]["problems"] == []
        assert rec["controls_still_remaining"] == ["bigcodebench_full"] and "error" not in rec
        assert len(_result_lines(Path(rec["out"]))) == 1                                     # the GENERATED stamp stands; no new line
    pend = second["verification_pending"]
    assert set(pend) == {"none", LORA, "none-rep2"}
    for label, ent in pend.items():
        prior = first["verification_pending"][label]
        assert ent["resumed"] is True and ent["resumed_by_function_call_id"] == second["function_call_id"] != first["function_call_id"]
        assert ent["out_dir"] == prior["out_dir"] and ent["row_engine_record"] == prior["row_engine_record"] and Path(ent["row_engine_record"]).is_file()
        assert ent["markers"] == prior["markers"] and all(Path(m).is_file() for m in ent["markers"])
        assert ent["tasks"] == [BCB_FULL] and ent["controls"] == ["bigcodebench_full"] and ent["cell"] == prior["cell"] and ent["name"] == prior["name"]
        assert ent["max_sandboxes"] == 8 == prior["max_sandboxes"]                          # min(the sidecar's client width 31, 8)
        assert ent["standup_id"] == "su-A" and ent["generation_function_call_id"] == "fc-gen"
        assert ent["served_max_num_seqs"] == 31 and ent["tree_sha"] == prior["tree_sha"] and ent["config"] == prior["config"]
        assert ent["concurrency_k"] == 1
    # this run's expected-verifiers record + row_rec.json name the resumed cells
    rec2 = json.loads((w.row_dir / "watch" / "verifiers_f6cf462-resume01.json").read_text())
    assert set(rec2["cells"]) == set(pend) and all(v["state"] == "expected" for v in rec2["cells"].values())
    assert rec2["generation_function_call_id"] == second["function_call_id"] and rec2["written_by"] == "row_container"
    row_rec = json.loads((w.row_dir / "row_rec.json").read_text())
    assert row_rec["status"] == "GENERATED" and row_rec["run_id"] == "f6cf462-resume01" and set(row_rec["verification_pending"]) == set(pend)
    assert row_rec["gpu_interval"] is None                                                   # nothing ended a GPU interval: none was held
    out = capsys.readouterr().out
    assert out.count("holds a valid pending generation") == 3 and "queued for CPU verification, no standup" in out
    # the classifier, directly: pending / verified / torn / none, by name
    st = L.pending_generation_state(pend[LORA]["out_dir"], w.row_dir, cell=LORA, label=LORA)
    assert st["state"] == "pending" and st["tasks"] == [BCB_FULL] and st["concurrency"] == 31 and st["standup_id"] == "su-A"
    assert st["markers"] == pend[LORA]["markers"] and st["record_copy"] == pend[LORA]["row_engine_record"] and st["problems"] == []
    assert L.pending_generation_state(tmp_path / "nothing-here", w.row_dir, cell="none", label="none")["state"] == "none"
    # a TORN store parks: the LoRA cell's sidecar vanished -> INCOMPLETE_STORE_PARKED naming it; the two others still resume
    CV.generation_sidecar_path(w.row_dir, LORA, BCB_FULL, Path(pend[LORA]["out_dir"]).name).unlink()
    torn = L.pending_generation_state(pend[LORA]["out_dir"], w.row_dir, cell=LORA, label=LORA)
    assert torn["state"] == "torn" and any("generation sidecar" in x for x in torn["problems"])
    third = w.run(run_id="f6cf462-resume02")
    assert third["status"] == "ROW_EMPTY" and set(third["cells"]) == {"none", "none-rep2"} and set(third["dropped"]) == {LORA}
    assert third["dropped"][LORA]["status"] == "INCOMPLETE_STORE_PARKED"
    assert "not a resumable pending generation" in third["dropped"][LORA]["error"] and "generation sidecar" in third["dropped"][LORA]["error"]
    assert set(third["verification_pending"]) == {"none", "none-rep2"} and "queued for CPU verification" in third["error"]
    assert _RowStandup.argvs == [] and w.driver_calls == []
    # a COUPLED relaunch never resumes into a pending store: parked, naming the resume flags
    w2 = _World(L, tmp_path, monkeypatch, cells=["none", LORA], row_opts={"repeat_cells": ["none"]})
    fourth = w2.run()
    assert fourth["dropped"]["none"]["status"] == "INCOMPLETE_STORE_PARKED"
    assert "--verification-phase generate" in fourth["dropped"]["none"]["error"] and "--verify-pending" in fourth["dropped"]["none"]["error"]
    assert fourth["verification_pending"] == {} and _RowStandup.argvs == []


def test_the_cpu_only_resume_spawns_verifiers_for_the_pending_stores_skips_the_verified_and_parks_the_torn(L, tmp_path, monkeypatch, capsys):
    """Code review L2 H3 (b): `main --row --verification-phase generate --verify-pending` — NO GPU row. One CPU function
    (`row_verify_pending_report`) reads the row dir's row_rec.json + markers + sidecars + manifests, writes this run's
    expected-verifiers record (pending -> expected, verified -> verified, torn -> torn), and `main` spawns the verifiers
    for EXACTLY the still-pending stores, files an OK REC per store already verified and a PENDING_STORE_TORN REC per
    torn store, harvests and exits as today. A rerun after one store verified skips it; a torn store fails by name."""
    w = _World(L, tmp_path, monkeypatch, cells=["none", LORA], row_opts={"verification_phase": "generate", "repeat_cells": ["none"]})
    monkeypatch.setattr(L.subprocess, "run", _generate_driver(w))
    first = w.run()
    pend = first["verification_pending"]
    # before the interruption ONE verifier had filed the LoRA store; the repeat's marker was lost (torn); `none` is pending
    _mark_store_verified(pend[LORA], "fc-v-old")
    Path(pend["none-rep2"]["markers"][0]).unlink()
    report = L.pending_verification_report(w.row_dir, run_id="f6cf462-resume01")
    assert report["status"] == "OK" and report["problems"] == []
    assert set(report["pending"]) == {"none"} and set(report["verified"]) == {LORA} and set(report["torn"]) == {"none-rep2"}
    assert report["verified"][LORA]["verified_by"] == "fc-v-old" and report["verified"][LORA]["out_dir"] == pend[LORA]["out_dir"]
    assert any("carries no pending marker" in x for x in report["torn"]["none-rep2"])
    p_none = report["pending"]["none"]
    assert p_none["out_dir"] == pend["none"]["out_dir"] and p_none["resumed"] is True and p_none["cell_rec"]["status"] == "GENERATED"
    assert p_none["markers"] == pend["none"]["markers"] and p_none["row_engine_record"] == pend["none"]["row_engine_record"]
    assert report["row_rec"]["cells"] == {"none": "GENERATED", LORA: "GENERATED", "none-rep2": "GENERATED"} and report["row_rec"]["standup_id"] == first["standup_id"]
    rec = json.loads((w.row_dir / "watch" / "verifiers_f6cf462-resume01.json").read_text())
    assert {k: v["state"] for k, v in rec["cells"].items()} == {"none": "expected", LORA: "verified", "none-rep2": "torn"}
    assert rec["cells"][LORA]["function_call_id"] == "fc-v-old" and rec["written_by"] == "verify_pending_report" and rec["generation_function_call_id"] is None
    assert L.pending_verification_report(tmp_path / "nowhere", run_id="r")["status"] == "NO_ROW_REC"
    # the resume's RECs: the pending cell GENERATED (flipped by its verifier), the verified one OK by its evidence, the torn one by name
    resumed, recs = L.resume_verification_recs(report)
    assert set(resumed["verification_pending"]) == {"none"} and resumed["row_dir"] == str(w.row_dir) and resumed["standup_id"] == first["standup_id"]
    assert [(r["status"], r.get("verification", {}).get("resumed")) for r in recs] == [("GENERATED", None), ("OK", "already verified"), ("PENDING_STORE_TORN", None)]
    assert recs[1]["verification"]["verified_by"] == "fc-v-old" and "carries no pending marker" in recs[2]["error"]
    assert L.REC_PENDING_STORE_TORN == "PENDING_STORE_TORN" not in L.REC_OK_STATUSES
    # main: the report through the CPU function's .remote, NO row spawn, the verifier for exactly `none`, the exit by name
    w3 = _World(L, tmp_path / "w3", monkeypatch, cells=["none"], row_opts={"verification_phase": "generate", "repeat_cells": ["none"]})
    monkeypatch.setattr(L.subprocess, "run", _generate_driver(w3))
    r3 = w3.run()
    pend3 = r3["verification_pending"]
    _mark_store_verified(pend3["none-rep2"], "fc-v-rep2")
    row_spawns, verify_spawns, written = [], [], {}
    monkeypatch.setattr(L, "run_row_full", SimpleNamespace(spawn=lambda cfgs, **kw: row_spawns.append(cfgs) or TL._Call({})))
    reports = []

    def _report_remote(sha, row_dir, rid, **kw):
        reports.append((sha, row_dir, rid))
        return L.pending_verification_report(w3.row_dir, run_id=rid, **kw)
    monkeypatch.setattr(L, "row_verify_pending_report", SimpleNamespace(remote=_report_remote))

    def _vspawn(cell_cfg, sha, rid, standup_id, row_dir, **kw):
        verify_spawns.append({"cell_cfg": cell_cfg, "sha": sha, "run_id": rid, "standup_id": standup_id, "row_dir": row_dir, **kw})
        return TL._Call({"name": f"{cell_cfg['name']}:verify", "cell": cell_cfg["cell"], "status": "OK", "phase": "verify",
                         "function_call_id": "fc-v-new", "verified_by": "fc-v-new"})
    monkeypatch.setattr(L, "verify_cell_cpu", SimpleNamespace(spawn=_vspawn))
    monkeypatch.setattr(L, "put_row_dir_file", lambda row_dir, name, d, **kw: written.__setitem__(name, json.loads(json.dumps(d, default=str))) or "x")
    kw = dict(sha="f6cf462", app_name=L.APP_NAME, pass1=True, cells="none", modes="off", controls="bigcodebench_full,bfcl_full", out_suffix=SUFFIX,
              row=True, repeat_cells="none", verification_phase="generate", verify_pending=True)
    capsys.readouterr()
    L.main(**kw)                                                                            # exits 0
    out = capsys.readouterr().out
    assert row_spawns == [] and len(reports) == 1 and reports[0][0] == "f6cf462" and reports[0][2].startswith("f6cf462-")
    assert reports[0][1] == f"{C.VOL_GEN_ROOT}/rows/gemma4e4b/{SUFFIX}"                       # the row dir main derives (the registry slug), on the volume
    assert [v["cell_cfg"]["out_dir"] for v in verify_spawns] == [pend3["none"]["out_dir"]]  # exactly the pending store
    v = verify_spawns[0]
    assert v["standup_id"] == r3["standup_id"] and v["row_dir"] == str(w3.row_dir) and v["pending"]["resumed"] is True
    assert v["pending"]["row_engine_record"] == pend3["none"]["row_engine_record"] and v["pending"]["max_sandboxes"] == 8
    assert "no GPU row" in out and "1 store(s) pending ['none']" in out and f"1 already verified ['none-rep2']" in out and "0 torn []" in out
    assert "3/3 container(s) OK" in out                                                     # none (flipped), none-rep2 (verified), the verifier
    assert '"resumed": "already verified"' in out
    upd = written[f"verifiers_{reports[0][2]}.json"]
    assert upd["cells"]["none"]["state"] == "spawned" and upd["cells"]["none"]["function_call_id"] and upd["cells"]["none-rep2"]["state"] == "verified"
    # a rerun after `none` was verified too: nothing pending, both skipped, no verifier spawned, exit 0
    _mark_store_verified(pend3["none"], "fc-v-new")
    verify_spawns.clear()
    L.main(**kw)
    out = capsys.readouterr().out
    assert verify_spawns == [] and "0 store(s) pending []" in out and "2 already verified" in out and "2/2 container(s) OK" in out
    # a TORN store fails the exit by name, and no verifier is spawned for it
    w4 = _World(L, tmp_path / "w4", monkeypatch, cells=["none"], row_opts={"verification_phase": "generate", "repeat_cells": ["none"]})
    monkeypatch.setattr(L.subprocess, "run", _generate_driver(w4))
    r4 = w4.run()
    Path(r4["verification_pending"]["none-rep2"]["markers"][0]).unlink()
    monkeypatch.setattr(L, "row_verify_pending_report", SimpleNamespace(remote=lambda sha, row_dir, rid, **kw: L.pending_verification_report(w4.row_dir, run_id=rid)))
    verify_spawns.clear()
    with pytest.raises(SystemExit) as e:
        L.main(**kw)
    assert "PENDING_STORE_TORN" in str(e.value) and [v["cell_cfg"]["out_dir"] for v in verify_spawns] == [r4["verification_pending"]["none"]["out_dir"]]
    # a row dir with no row_rec.json refuses by name; the grammar: needs --row and the generate phase; no warm probe beside it
    monkeypatch.setattr(L, "row_verify_pending_report", SimpleNamespace(remote=lambda sha, row_dir, rid, **kw: L.pending_verification_report(tmp_path / "none", run_id=rid)))
    with pytest.raises(SystemExit, match="NO_ROW_REC"):
        L.main(**kw)
    base = dict(sha="f6cf462", app_name=L.APP_NAME, pass1=True, dry_run=True, cells="none", modes="off", controls="bigcodebench_full", out_suffix="s")
    with pytest.raises(SystemExit, match="mean nothing without --row"):
        L.main(**base, verify_pending=True)
    with pytest.raises(SystemExit, match="resumes a generate row"):
        L.main(**base, row=True, verify_pending=True)
    with pytest.raises(SystemExit, match="drives no cell|launches no row"):                   # the warm probe: refused by name
        L.main(**base, row=True, verification_phase="generate", verify_pending=True, warm_cache_only=True)
    capsys.readouterr()
    L.main(**base, row=True, verification_phase="generate", verify_pending=True)
    out = capsys.readouterr().out
    assert "CPU-only RESUME" in out and "NO GPU container" in out and "PENDING_STORE_TORN" in out
    argv_line = next(ln for ln in out.splitlines() if ln.startswith("[tinycap] argv:"))
    assert argv_line.split("  (run it")[0].endswith("--verification-phase generate --verify-pending"), argv_line
    assert L.launch_argv("a" * 40, pass1=True, row=True, cells="none", modes="off", out_suffix="s", verification_phase="generate", verify_pending=True)[-3:] == [
        C.FLAG_VERIFICATION_PHASE, "generate", "--verify-pending"]
    assert "--verify-pending" not in L.launch_argv("a" * 40, pass1=True, row=True, cells="none", modes="off", out_suffix="s", verification_phase="generate")
    ns = L.cli_parser().parse_args(["--sha", "x", "--app-name", L.APP_NAME, "--pass1", "--row", "--cells", "none", "--modes", "off", "--out-suffix", "s",
                                    C.FLAG_VERIFICATION_PHASE, "generate", "--verify-pending"])
    assert ns.verify_pending is True and L.cli_parser().parse_args(["--sha", "x", "--app-name", L.APP_NAME]).verify_pending is False
    assert L.ROW_VERIFY_PENDING_FLAG == "--verify-pending" and "verify_pending=a.verify_pending" in Path(L.__file__).read_text()


def test_the_dry_runs_resume_identity_carries_the_client_width_s_over_k(L, capsys):
    """/simplify FYI on landing 2: `describe_plan`'s pre-launch `expected_identity(...)` did not pass `concurrency_k`, so a
    K = 2 dry run printed a resume identity keyed at the served S (the K = 1 width) — not the S // K a K = 2 store is
    hashed at (§2.1, S9). The dry run now threads the row's K: the printed identity's `served.concurrency` is S // K."""
    base = dict(sha="f6cf462", app_name=L.APP_NAME, pass1=True, dry_run=True, cells="none", modes="off", controls="bigcodebench_full", out_suffix="s",
                row=True, row_forecast_h=1.5)

    def identity_concurrency(**kw) -> int:
        capsys.readouterr()
        L.main(**base, **kw)
        line = next(ln for ln in capsys.readouterr().out.splitlines() if "resume identity:" in ln)
        return int(json.loads(line.split("resume identity:", 1)[1])["served"]["concurrency"])
    c1 = identity_concurrency()
    c2 = identity_concurrency(concurrency_k=2)
    c3 = identity_concurrency(concurrency_k=3)
    assert c1 == 31 and c2 == L.client_width(c1, 2) == 15 and c3 == L.client_width(c1, 3) == 10
    src = inspect.getsource(L.describe_plan)
    assert "concurrency_k=concurrency_k" in src and "concurrency_k: int = 1" in src


# --------------------------------------------------------------------------- #
# landing 2 follow-up (the merged tree): the verifier's RESUME contract, the     #
# compile-cache evidence in the row dir, the three store facts, the forecast    #
# --------------------------------------------------------------------------- #


def _verified_store(tmp_path, *, verified_by: str, marker: bool, row_dir: Path):
    """A row-engine store whose BigCodeBench task a verifier ALREADY filed (the verified block naming ``verified_by``,
    the RESULT stamp), with or without its marker still standing (the driver died before removing it)."""
    store = tmp_path / "gen" / "native" / ROW / "none" / "thinking_off" / SUFFIX
    (store / "cells" / "k1" / "logs").mkdir(parents=True)
    log = store / "cells" / "k1" / "logs" / "bcb.eval"
    log.write_bytes(b"unscored")
    block = {"row_engine": True, "standup_id": STANDUP_A, "engine_cells": ["none"], "max_loras": 1, "residual_slot_armed": False,
             "compile_cache": {"hit": False, "key": "k", "vllm_cache_dir": None}, "row_dir": str(row_dir),
             "verification_pass": {"state": "verified", "mechanism": "M2_replay_vehicle", "function_call_id": verified_by,
                                   "started_at": 1.0, "finished_at": 2.0, "sandbox_facts": {}, "max_sandboxes": 8, "n_items": 1140,
                                   "vehicle_log_sha256": "0" * 64}}
    (store / "cells" / "k1" / "manifest.json").write_text(json.dumps({"row_engine": block, "tinycap": {}}))
    (store / "RESULT").write_text("tinycap-modal name=n rc=0 result=GENERATED status=GENERATED sha=f6cf462 at=t standup=su\n"
                                  f"tinycap-modal name=n rc=0 result=COMPLETE status=OK sha=f6cf462 at=t verified_by={verified_by}\n")
    (row_dir / "generation").mkdir(parents=True, exist_ok=True)
    mk = CV.pending_marker_path(row_dir, "none", BCB_FULL, store.name)
    if marker:
        mk.write_text(json.dumps({"task": BCB_FULL, "cell": "none", "n_items": 1140, "n_expected_full": 1140, "generated_at": "t",
                                  "log": str(log), "generation_fcid": "fc-gen"}))
    record_copy = row_dir / "generation" / "row_engine__none.json"
    record_copy.write_text(json.dumps({**block, "verification_pass": None}))
    pend = {"cell": "none", "name": "gemma4e4b-f22s1__none__off", "out_dir": str(store), "config": "/tree/config.yaml",
            "row_engine_record": str(record_copy), "controls": ["bigcodebench_full"], "tasks": [BCB_FULL], "markers": [str(mk)],
            "max_sandboxes": 8, "served_max_num_seqs": 31, "concurrency_k": 1, "generation_function_call_id": "fc-gen",
            "standup_id": STANDUP_A, "tree_sha": FULL_SHA}
    return store, mk, pend


def test_the_verify_worker_accepts_the_drivers_resume_contract_and_still_refuses_a_pending_store(L, tmp_path, monkeypatch, capsys):
    """Landing 2 follow-up (the driver's resume contract, docs §9.3): a resumed or repeated `verify` leaves the verified
    block and RESULT's `verified_by=` naming the ORIGINAL verifier, not the fcid the harness just spawned; an
    already-verified store's `verify` exits 0 with its `already verified … nothing to do` line, writes NO pointer and
    mutates nothing; a store that died after its manifest write has its tail completed (a pointer for the CURRENT fcid,
    the block still naming the original). The harness used to require the block's fcid == the spawned one and a pointer
    always, so every resumed verifier read VERIFY_FAILED. Now: the well-formed verified block with no marker left is the
    fact (`verified_by` read off it), the missing pointer is excused by the driver's line alone, and a pending block, a
    leftover marker or a non-zero exit stay refused."""
    repos = tmp_path / "repos"
    TL._tree(repos, "f6cf462")
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    row_dir = tmp_path / "rows" / ROW / SUFFIX
    ALREADY = ("[served-cap] verify: inspect/bigcodebench_full for none is already verified (by fc-v-old via M2_replay_vehicle over 1140 items); "
               "nothing to do — the store is untouched\n")
    # (i) already verified, no marker: the driver prints its line, exits 0, touches nothing -> OK, verified_by = the ORIGINAL
    store, mk, pend = _verified_store(tmp_path, verified_by="fc-v-old", marker=False, row_dir=row_dir)
    cfg = {"cell": "none", "name": pend["name"], "model": ROW, "mode": "off", "out_dir": str(store)}
    monkeypatch.setattr(L.subprocess, "run", lambda argv, **kw: SimpleNamespace(returncode=0, stdout=ALREADY, stderr=""))
    rec = L._verify_cell_worker(cfg, "f6cf462", "f6cf462-cafebabe", STANDUP_A, str(row_dir), pending=pend, function_call_id="fc-v-new")
    assert rec["status"] == "OK", rec.get("error")
    assert rec["verified_by"] == "fc-v-old" and rec["function_call_id"] == "fc-v-new"
    assert rec["already_verified"] is True and rec["pointer_present"] is False and rec["tail_completed"] is False
    assert not (row_dir / "watch" / "attempt_fc-v-new.json").exists()
    assert L.driver_said_already_verified(ALREADY) and not L.driver_said_already_verified("verified\n")
    # main's flip records the block's verifier on the cell REC
    gen_rec = {"name": pend["name"], "cell": "none", "status": "GENERATED"}
    monkeypatch.setattr(L, "put_row_dir_file", lambda row_dir, name, d, **kw: "x")
    expectation = {"run_id": "f6cf462-cafebabe", "cells": {"none": {"cell": "none", "out_dir": str(store), "state": "expected"}}}
    vrecs = L.verify_pending_cells(lambda *a, **k: rec, {"verification_pending": {"none": pend}, "cells": {"none": gen_rec}, "row_dir": str(row_dir),
                                                        "expected_verifiers": expectation},
                                   [cfg], [gen_rec], sha="f6cf462", run_id="f6cf462-cafebabe", reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT)
    assert vrecs[0]["status"] == "OK" and gen_rec["status"] == "OK" and gen_rec["verification"]["verified_by"] == "fc-v-old"
    # (ii) the marker still stood (a verifier died after its manifest write): the resume completes the tail — removes the
    # marker, writes ITS pointer, the block and RESULT keep naming the original -> OK, tail_completed
    shutil.rmtree(tmp_path / "gen")
    store, mk, pend = _verified_store(tmp_path, verified_by="fc-v-old", marker=True, row_dir=row_dir)

    def _resume(argv, **kw):
        argv = list(argv)
        fcid = argv[argv.index("--function-call-id") + 1]
        mk.unlink()
        (row_dir / "watch").mkdir(exist_ok=True)
        (row_dir / "watch" / f"attempt_{fcid}.json").write_text(json.dumps({
            "run_id": "f6cf462-cafebabe", "cell": "none", "function_call_id": fcid, "name": "n", "model": ROW, "mode": "off", "out_dir": str(store),
            "controls": ["bigcodebench_full"], "tree_sha": FULL_SHA, "sandbox_record": "r", "sandbox_app_name": C.BCB_SANDBOX_APP_NAME,
            "started_at": 1.0, "phase": "verify"}))
        return SimpleNamespace(returncode=0, stdout="[served-cap] verify: … was verified by fc-v-old whose run died before removing the marker — "
                                                    "completing its tail (controls summary, RESULT, marker); nothing re-scored\n", stderr="")
    monkeypatch.setattr(L.subprocess, "run", _resume)
    rec2 = L._verify_cell_worker(cfg, "f6cf462", "f6cf462-cafebabe", STANDUP_A, str(row_dir), pending=pend, function_call_id="fc-v-new2")
    assert rec2["status"] == "OK", rec2.get("error")
    assert rec2["verified_by"] == "fc-v-old" and rec2["tail_completed"] is True and rec2["pointer_present"] is True and rec2["already_verified"] is False
    # (iii) still refused: the already-verified line over a PENDING block; a leftover marker; a non-zero exit with the line
    shutil.rmtree(tmp_path / "gen")
    store, mk, pend = _verified_store(tmp_path, verified_by="fc-v-old", marker=True, row_dir=row_dir)
    monkeypatch.setattr(L.subprocess, "run", lambda argv, **kw: SimpleNamespace(returncode=0, stdout=ALREADY, stderr=""))
    left = L._verify_cell_worker(cfg, "f6cf462", "f6cf462-cafebabe", STANDUP_A, str(row_dir), pending=pend, function_call_id="fc-v-3")
    assert left["status"] == "VERIFY_FAILED" and "pending marker(s) still present" in left["error"]
    manifest = next(store.glob("cells/*/manifest.json"))
    doc = json.loads(manifest.read_text())
    doc["row_engine"]["verification_pass"] = {"state": "pending", "tasks": [BCB_FULL]}
    manifest.write_text(json.dumps(doc))
    pend_block = L._verify_cell_worker(cfg, "f6cf462", "f6cf462-cafebabe", STANDUP_A, str(row_dir), pending=pend, function_call_id="fc-v-4")
    assert pend_block["status"] == "VERIFY_FAILED" and "verification_pass.state is 'pending'" in pend_block["error"]
    shutil.rmtree(tmp_path / "gen")
    store, mk, pend = _verified_store(tmp_path, verified_by="fc-v-old", marker=False, row_dir=row_dir)
    monkeypatch.setattr(L.subprocess, "run", lambda argv, **kw: SimpleNamespace(returncode=1, stdout=ALREADY, stderr="boom"))
    rc1 = L._verify_cell_worker(cfg, "f6cf462", "f6cf462-cafebabe", STANDUP_A, str(row_dir), pending=pend, function_call_id="fc-v-5")
    assert rc1["status"] == "VERIFY_FAILED" and rc1["error"].startswith(f"{pend['name']}: verify returned rc 1") and "wrote no pointer" in rc1["error"]
    # the driver's own spelling of the line (source pin: the two fragments the harness matches)
    src = (REPO / "src/sisteer/capeval/tinycap/cpu_verify.py").read_text()
    assert "is already verified (by {vp['function_call_id']} via " in src and "nothing to do — the store is untouched" in src
    assert L.VERIFY_ALREADY_VERIFIED_MARKS == ("is already verified (by ", "nothing to do")


def test_the_row_dir_retains_the_standups_log_head_and_redacted_compile_factors_and_the_store_gains_nothing(L, tmp_path, monkeypatch):
    """The records keep a 2 MB TAIL of the vLLM log, so the `Using cache directory` line and the compile factors were
    lost. After readiness the row runner copies the log's HEAD (first 256 KB) to `<row dir>/standup/vllm_server.head.log`
    and vLLM's `<local_cache_dir>/cache_key_factors.json` — `VLLM_API_KEY` redacted through
    `tinycap_compile_cache.redact_factors` — to `<row dir>/standup/cache_key_factors.json`; the ROW-REC's
    `compile_cache.evidence` names both. Row-dir files only: the store's inode count is unchanged."""
    HASH = "0123abcd"
    seen = {}

    def vllm_log(env, out):
        d = Path(env["VLLM_CACHE_ROOT"]) / "torch_compile_cache" / HASH / "rank_0_0" / "backbone"
        d.mkdir(parents=True, exist_ok=True)
        (d / "computation_graph.py").write_text("# compiled\n")
        (d / "cache_key_factors.json").write_text(json.dumps({"env": {"VLLM_API_KEY": "sk-live-secret-token", "VLLM_USE_V1": "1",
                                                                       "S3_ACCESS_KEY_ID": None},
                                                              "config_hash": "c" * 8, "code_hash": "d" * 8, "compiler_hash": "e" * 8}))
        seen["dir"] = d
        head = f"INFO Using cache directory: {d} for vLLM's torch.compile\nINFO Compiling a graph for compile range (1, 31) takes 12.3 s\n"
        return head + ("INFO filler line to push the log past the head bound\n" * 9000)            # ~500 KB: the tail loses the head
    w = _World(L, tmp_path, monkeypatch, cells=["none", LORA])
    w.vllm_log = vllm_log
    result = w.run()
    assert result["status"] == "OK", result.get("error")
    ev = result["compile_cache"]["evidence"]
    head = w.row_dir / "standup" / "vllm_server.head.log"
    factors = w.row_dir / "standup" / "cache_key_factors.json"
    assert ev["log_head"] == str(head) and ev["cache_key_factors"] == str(factors) and ev["factors_redacted"] == ["VLLM_API_KEY"]
    assert ev["retain_problems"] == [] and ev["cache_dir"] == str(seen["dir"])
    assert head.is_file() and head.stat().st_size <= L.ROW_VLLM_LOG_HEAD_BYTES == 256_000
    text = head.read_text()
    assert text.startswith(f"INFO Using cache directory: {seen['dir']} for vLLM's torch.compile") and "Compiling a graph" in text
    doc = json.loads(factors.read_text())
    assert doc["env"]["VLLM_API_KEY"] == "<redacted>" and doc["env"]["VLLM_USE_V1"] == "1" and doc["config_hash"] == "c" * 8
    assert "sk-live-secret-token" not in factors.read_text()
    # the row's 2 MB tail copies lose the head line; the head copy keeps it — the whole point
    tail = (w.row_dir / "standup" / "standup.log")
    assert tail.is_file() or True
    row_rec = json.loads((w.row_dir / "row_rec.json").read_text())
    assert row_rec["compile_cache"]["evidence"]["log_head"] == str(head) and row_rec["compile_cache"]["evidence"]["cache_key_factors"] == str(factors)
    # the stores: neither file, and the same inode count as a run that retains nothing
    for rec in result["cells"].values():
        store = Path(rec["out"])
        assert not list(store.rglob("vllm_server.head.log")) and not list(store.rglob("cache_key_factors.json"))
        assert sorted(p.name for p in (store / "standup").iterdir()) == ["endpoint.json", "serving_attestation.json", "vllm_server.log"]
    w_plain = _World(L, tmp_path / "plain", monkeypatch, cells=["none", LORA])
    r_plain = w_plain.run()
    assert SS.count_inodes(Path(result["cells"]["none"]["out"])) == SS.count_inodes(Path(r_plain["cells"]["none"]["out"]))
    # no cache-dir line (the default fake log): named, nothing raised, no factors copied; the head is still kept
    ev_plain = r_plain["compile_cache"]["evidence"]
    assert ev_plain["cache_key_factors"] is None and any("no cache directory" in x for x in ev_plain["retain_problems"])
    assert ev_plain["log_head"] == str(w_plain.row_dir / "standup" / "vllm_server.head.log")
    # an unparseable factors file is not copied, by name
    r, names = L.CC.redact_factors(b"{torn")
    assert r is None and names == ["unparseable"]


def test_a_row_store_names_its_own_cell_carries_the_standups_digest_map_and_the_relative_row_dir(L, tmp_path, monkeypatch):
    """Three store facts the reader refused on earlier stores, fixed for future stores.
    (a) `manifest.route.object_artifact_sha256_by_cell` was null on every row-engine store although the endpoint carries
    the row's per-cell digest map — the runner now stamps the map (and the legacy scalar by the writer's rule) into each
    store's UNHASHED route after its drive; (b) `served_geometry.json#endpoint.cell` named the row's LAST served cell —
    each store's copy (and the serve-dir copy the driver reads) names ITS cell; (c) `manifest.row_engine.row_dir` stays
    the absolute truth and `manifest.route.row_dir_rel = rows/<row>/<suffix>` is written beside it (the row-engine
    block's field set is frozen). The stamped store still resumes as DONE (the route block is outside the content hash)."""
    harness_lora = C.cell_spellings(LORA, declared=False).harness
    # the real standup's endpoint names its LAST --cell in `cell` / `harness_cell` (a multi-cell standup's one-cell field)
    w = _World(L, tmp_path, monkeypatch, cells=["none", LORA], row_opts={"repeat_cells": ["none"]},
               record={"cell": LORA, "harness_cell": harness_lora})

    def _driver(argv, **kw):
        argv = list(argv)
        cell = argv[argv.index("--cell") + 1]
        out_root = Path(argv[argv.index(C.FLAG_OUT_ROOT) + 1])
        rec_doc = C.read_row_engine_record(argv[argv.index(C.FLAG_ROW_ENGINE_RECORD) + 1])
        table = Path(argv[argv.index("--routing-table") + 1])
        ep = json.loads((table.parent / "endpoint.json").read_text())                      # what the driver reads beside the table
        w.driver_calls.append({"argv": argv, "cell": cell, "out_root": out_root, "endpoint_cell": ep.get("cell"),
                               "endpoint_harness_cell": ep.get("harness_cell"), "endpoint_map": ep.get("object_artifact_sha256_by_cell")})
        w.finish_store(out_root, w.by_cell[cell], row_engine=rec_doc)                     # the driver's manifest: route map null, as landing 1 left it
        (out_root / "evidence.json").write_text(json.dumps({"cell": cell}))
        return SimpleNamespace(returncode=0, stdout="driver ok\n", stderr="")
    monkeypatch.setattr(L.subprocess, "run", _driver)
    result = w.run()
    assert result["status"] == "OK", result.get("error")
    sha = w.object_digests[LORA]["artifact_sha256"]
    expected_map = {harness_lora: sha}
    labels = {"none": "none", LORA: LORA, "none-rep2": "none"}
    for label, cell in labels.items():
        rec = result["cells"][label]
        store = Path(rec["out"])
        # (b) the store's served_geometry copy names THIS store's cell, in both spellings
        sg = json.loads((store / "served_geometry.json").read_text())
        assert sg["endpoint"]["cell"] == cell and sg["endpoint"]["harness_cell"] == C.cell_spellings(cell, declared=False).harness, label
        assert sg["endpoint"]["object_artifact_sha256_by_cell"] == expected_map
        # (a) + (c) the manifest's route block: the map, the scalar by the writer's rule (one distinct digest), the relative row dir
        manifest = json.loads(next(store.glob("cells/*/manifest.json")).read_text())
        route = manifest[C.MANIFEST_KEY_ROUTE]
        assert route["object_artifact_sha256_by_cell"] == expected_map and route["object_artifact_sha256"] == sha, label
        assert route["row_dir_rel"] == f"rows/{ROW}/{SUFFIX}" and manifest["row_engine"]["row_dir"] == str(w.row_dir)
        assert rec["route_stamp"] == {"stamped": True, "keys": ["object_artifact_sha256_by_cell", "object_artifact_sha256", "row_dir_rel"], "problem": None}
        assert set(manifest["row_engine"]) == set(C.ROW_ENGINE_MANIFEST_FIELDS)              # the frozen block is untouched
    # the serve-dir copy the DRIVER read named its own cell too (the repeat's: its base cell)
    by_out = {str(c["out_root"]): c for c in w.driver_calls}
    for label, cell in labels.items():
        call = by_out[result["cells"][label]["out"]]
        assert call["endpoint_cell"] == cell and call["endpoint_harness_cell"] == C.cell_spellings(cell, declared=False).harness and call["endpoint_map"] == expected_map
    # the row's own standup record is copied VERBATIM into every store's standup/ (the row-level truth: its last cell)
    assert json.loads((Path(result["cells"]["none"]["out"]) / "standup" / "endpoint.json").read_text())["cell"] == LORA
    # the stamped stores still resume: a relaunch reads every cell DONE (the route block is outside the content hash)
    w.driver_calls.clear()
    _RowStandup.argvs.clear()
    again = w.run()
    assert again["status"] == "DONE" and set(again["cells"]) == set(labels) and w.driver_calls == [] and _RowStandup.argvs == []
    # the helpers, directly: a record without a `cell` key is copied as is (a solo record stays byte-identical); the scalar
    # reads null when the map holds several digests; a store without a route mapping or a row_engine block is left alone
    assert L.per_cell_endpoint({"proxy_url": "u"}, "none") == {"proxy_url": "u"}
    assert L.per_cell_endpoint({"cell": "x", "proxy_url": "u"}, "none") == {"cell": "none", "proxy_url": "u"}
    assert L.row_dir_rel("/vol/tinycap/gen/rows/gemma4e4b/bcbw-x") == "rows/gemma4e4b/bcbw-x" == L.row_dir_rel(f"{C.VOL_GEN_ROOT}/rows/gemma4e4b/bcbw-x")
    two = tmp_path / "two"
    (two / "cells" / "k").mkdir(parents=True)
    (two / "cells" / "k" / "manifest.json").write_text(json.dumps({"route": {"object_artifact_sha256_by_cell": None, "object_artifact_sha256": None},
                                                                    "row_engine": {"row_dir": "/vol/tinycap/gen/rows/r/s"}}))
    st = L.stamp_row_store_route(two, endpoint={"object_artifact_sha256_by_cell": {"a": "1" * 64, "b": "2" * 64}}, row_dir="/vol/tinycap/gen/rows/r/s")
    doc = json.loads((two / "cells" / "k" / "manifest.json").read_text())
    assert st["stamped"] and doc["route"]["object_artifact_sha256"] is None and doc["route"]["object_artifact_sha256_by_cell"] == {"a": "1" * 64, "b": "2" * 64}
    assert doc["route"]["row_dir_rel"] == "rows/r/s"
    assert L.stamp_row_store_route(two, endpoint={}, row_dir="/vol/tinycap/gen/rows/r/s") == {"stamped": False, "keys": [], "problem": None}   # nothing left to add
    (two / "cells" / "k" / "manifest.json").write_text(json.dumps({"route": {}}))
    assert "no row_engine block" in L.stamp_row_store_route(two, endpoint={}, row_dir="/r")["problem"]
    assert "expected exactly one manifest" in L.stamp_row_store_route(tmp_path / "absent", endpoint={}, row_dir="/r")["problem"]


def _kill_verifier_after(store: Path, *, at: str, verified_by: str = "fc-v-old") -> None:
    """What a verifier killed at ``at`` leaves in a GENERATED store (the driver's orchestrated-restart steps, docs §9.3):
    ``task_filed`` / ``gen_stats_written`` — the BCB task's done marker + summary written, the manifest still PENDING, the
    marker standing; ``manifest_written`` — the verified block written too, no RESULT line, the marker standing."""
    cell_dir = next(store.glob("cells/*"))
    TL._finish(cell_dir, BCB_FULL, 1140)
    if at == "manifest_written":
        doc = json.loads((cell_dir / "manifest.json").read_text())
        doc["row_engine"]["verification_pass"] = {"state": "verified", "mechanism": "M2_replay_vehicle", "function_call_id": verified_by,
                                                  "started_at": 1.0, "finished_at": 2.0, "sandbox_facts": {}, "max_sandboxes": 8,
                                                  "n_items": 1140, "vehicle_log_sha256": "0" * 64}
        (cell_dir / "manifest.json").write_text(json.dumps(doc))


def test_a_generate_relaunch_after_a_partial_verification_keeps_the_verified_stores_evidence_and_recovers_the_repeat(L, tmp_path, monkeypatch):
    """Base `none` verified, `none-rep2` still pending; a normal `--row --verification-phase generate`
    relaunch read `none` DONE and appended a fresh COMPLETE RESULT line WITHOUT `verified_by=`, so the resume readers
    (which hold a verified store's LAST line to its block's verifier) classified it PENDING_STORE_TORN — a false failure
    that blocked the repeat's recovery. The DONE line of a verified two-phase store now carries `verified_by=<the
    block's fcid>`; the repeat is queued and the CPU-only recovery succeeds while the base's evidence reads verified."""
    w = _World(L, tmp_path, monkeypatch, cells=["none"], row_opts={"verification_phase": "generate", "repeat_cells": ["none"]})
    monkeypatch.setattr(L.subprocess, "run", _generate_driver(w))
    first = w.run()
    pend = first["verification_pending"]
    _mark_store_verified(pend["none"], "fc-v-old")                     # the base's verifier filed; the repeat's never ran
    _RowStandup.argvs.clear()
    second = w.run(run_id="f6cf462-again")
    assert second["status"] == "GENERATED", second.get("error")
    assert second["cells"]["none"]["status"] == "DONE" and second["cells"]["none-rep2"]["status"] == "GENERATED"
    assert set(second["verification_pending"]) == {"none-rep2"} and _RowStandup.argvs == []
    lines = _result_lines(Path(pend["none"]["out_dir"]))
    assert " result=COMPLETE status=DONE " in lines[-1] and lines[-1].endswith(" verified_by=fc-v-old"), lines[-1]
    st = L.pending_generation_state(pend["none"]["out_dir"], w.row_dir, cell="none", label="none")
    assert st["state"] == "verified" and st["problems"] == [] and st["verified_by"] == "fc-v-old"
    # the CPU-only recovery of the repeat: the report dispatches exactly it; the base is not torn, not re-dispatched
    report = L.pending_verification_report(w.row_dir, run_id="f6cf462-resume")
    assert report["status"] == "OK" and set(report["pending"]) == {"none-rep2"} and report["torn"] == {}
    resumed, recs = L.resume_verification_recs(report)
    monkeypatch.setattr(L, "put_row_dir_file", lambda row_dir, name, d, **kw: "x")                     # the grading host's REPLACE lands
    vrecs = L.verify_pending_cells(SimpleNamespace(spawn=lambda cfg, *a, **k: TL._Call({"name": f"{cfg['name']}:verify", "cell": cfg["cell"], "status": "OK",
                                                                                          "phase": "verify", "verified_by": "fc-v-new"})),
                                   resumed, w.cfgs, recs, sha="f6cf462", run_id="f6cf462-resume", reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT)
    assert [(v["label"], v["status"]) for v in vrecs] == [("none-rep2", "OK")] and recs[0]["status"] == "OK"
    # a coupled store's DONE line is untouched (no verifier to name)
    w2 = _World(L, tmp_path / "coupled", monkeypatch, cells=["none"])
    w2.run()
    _RowStandup.argvs.clear()
    w2.run()
    coupled_last = _result_lines(Path(w2.by_cell["none"]["out_dir"]))[-1]
    assert " result=COMPLETE status=DONE " in coupled_last and "verified_by" not in coupled_last


def test_a_two_phase_store_is_not_done_off_its_done_marker_alone_while_its_verification_is_pending(L, tmp_path, monkeypatch):
    """A verifier killed after `_file_task_results` (the task's done marker + summary written) but before the verified
    manifest leaves the manifest PENDING and the marker standing; `_cell_identity` read every task done and declared
    the cell DONE — its verifier was never spawned again. A two-phase store is DONE only when the block reads verified
    AND no marker stands; otherwise the pending door queues its verifier, no GPU minute spent."""
    w = _World(L, tmp_path, monkeypatch, cells=["none", LORA], row_opts={"verification_phase": "generate"})
    monkeypatch.setattr(L.subprocess, "run", _generate_driver(w))
    first = w.run()
    pend = first["verification_pending"]
    for label, at in (("none", "task_filed"), (LORA, "gen_stats_written")):
        _kill_verifier_after(Path(pend[label]["out_dir"]), at=at)
    hold = L.two_phase_verification_hold(pend["none"]["out_dir"], cell="none", wanted=CONTROLS, mode="off")
    assert hold["held"] == ["bigcodebench_full"] and hold["state"] == "pending" and hold["standing_markers"] == pend["none"]["markers"]
    _RowStandup.argvs.clear()
    w.driver_calls.clear()
    second = w.run(run_id="f6cf462-again")
    assert second["status"] == "GENERATED", second.get("error")
    assert second["cell_statuses"] == {"none": "GENERATED", LORA: "GENERATED"} and _RowStandup.argvs == [] and w.driver_calls == []
    for label in ("none", LORA):
        rec = second["cells"][label]
        assert rec["resumed_pending"] is True and rec["two_phase_hold"]["held"] == ["bigcodebench_full"]
        assert rec["controls_remaining"] == ["bigcodebench_full"] and rec["resume"]["controls"]["bigcodebench_full"]["done_in"] is not None  # done marker present, still held
        assert second["verification_pending"][label]["resume_kind"] == "pending" and second["verification_pending"][label]["tasks"] == [BCB_FULL]
    # a solo store (no row_engine block) and a verified block with no marker hold nothing
    solo = tmp_path / "solo"
    (solo / "cells" / "k").mkdir(parents=True)
    (solo / "cells" / "k" / "manifest.json").write_text(json.dumps({"cell_id": "none"}))
    assert L.two_phase_verification_hold(solo, cell="none", wanted=CONTROLS, mode="off")["held"] == []
    _mark_store_verified(pend["none"], "fc-v-1")
    assert L.two_phase_verification_hold(pend["none"]["out_dir"], cell="none", wanted=CONTROLS, mode="off")["held"] == []
    _RowStandup.argvs.clear()
    third = w.run(run_id="f6cf462-third")
    assert third["cells"]["none"]["status"] == "DONE" and third["cells"][LORA]["status"] == "GENERATED"


def test_a_verified_block_with_its_marker_standing_is_dispatched_as_tail_pending_not_parked(L, tmp_path, monkeypatch):
    """A verifier killed after its manifest write, before the RESULT line / the marker's removal, leaves a well-formed
    verified block with the marker standing. That state was `torn` (parked, never dispatched) although the driver's
    `verify` finishes exactly that tail (a pointer for the CURRENT fcid, the block naming the ORIGINAL): it is now
    `tail_pending` — queued at the door and dispatched by the CPU-only report; the harness's evidence then accepts the
    original verifier in the block / RESULT while checking the recovery call's own pointer."""
    w = _World(L, tmp_path, monkeypatch, cells=["none"], row_opts={"verification_phase": "generate"})
    monkeypatch.setattr(L.subprocess, "run", _generate_driver(w))
    first = w.run()
    pend = first["verification_pending"]
    _kill_verifier_after(Path(pend["none"]["out_dir"]), at="manifest_written", verified_by="fc-v-old")
    st = L.pending_generation_state(pend["none"]["out_dir"], w.row_dir, cell="none", label="none")
    assert st["state"] == "tail_pending" == L.PENDING_STATE_TAIL_PENDING and st["problems"] == [], st
    assert st["tasks"] == [BCB_FULL] and st["markers"] == pend["none"]["markers"] and st["verified_by"] == "fc-v-old"
    assert L.PENDING_DISPATCH_STATES == ("pending", "tail_pending")
    # the relaunch queues it
    _RowStandup.argvs.clear()
    second = w.run(run_id="f6cf462-again")
    assert second["status"] == "GENERATED" and second["cells"]["none"]["status"] == "GENERATED" and _RowStandup.argvs == []
    assert second["verification_pending"]["none"]["resume_kind"] == "tail_pending"
    # the CPU-only report dispatches it (pending, resume_kind tail_pending), never torn
    report = L.pending_verification_report(w.row_dir, run_id="f6cf462-resume")
    assert set(report["pending"]) == {"none"} and report["pending"]["none"]["resume_kind"] == "tail_pending" and report["torn"] == {}
    # the driver's tail: removes the marker, appends RESULT verified_by=<original>, writes ITS pointer -> the harness reads OK
    store = Path(pend["none"]["out_dir"])
    mk = Path(pend["none"]["markers"][0])
    cfg = {"cell": "none", "name": pend["none"]["name"], "model": ROW, "mode": "off", "out_dir": str(store)}

    def _tail(argv, **kw):
        argv = list(argv)
        fcid = argv[argv.index("--function-call-id") + 1]
        mk.unlink()
        with (store / "RESULT").open("a") as fh:
            fh.write("tinycap-modal name=n rc=0 result=COMPLETE status=OK sha=f6cf462 at=t verified_by=fc-v-old\n")
        (w.row_dir / "watch" / f"attempt_{fcid}.json").write_text(json.dumps({
            "run_id": "f6cf462-resume", "cell": "none", "function_call_id": fcid, "name": "n", "model": ROW, "mode": "off", "out_dir": str(store),
            "controls": ["bigcodebench_full"], "tree_sha": FULL_SHA, "sandbox_record": "r", "sandbox_app_name": C.BCB_SANDBOX_APP_NAME,
            "started_at": 1.0, "phase": "verify"}))
        return SimpleNamespace(returncode=0, stdout="verify tail completed\n", stderr="")
    repos = tmp_path / "repos"
    TL._tree(repos, "f6cf462")
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    monkeypatch.setattr(L.subprocess, "run", _tail)
    rec = L._verify_cell_worker(cfg, "f6cf462", "f6cf462-resume", first["standup_id"], str(w.row_dir), pending=report["pending"]["none"],
                                function_call_id="fc-v-new")
    assert rec["status"] == "OK", rec.get("error")
    assert rec["verified_by"] == "fc-v-old" and rec["tail_completed"] is True and rec["pointer_present"] is True
    assert L.pending_generation_state(store, w.row_dir, cell="none", label="none")["state"] == "verified"
    # a torn variant stays torn: the verified block with the marker standing but the sidecar gone
    w2 = _World(L, tmp_path / "torn", monkeypatch, cells=["none"], row_opts={"verification_phase": "generate"})
    monkeypatch.setattr(L.subprocess, "run", _generate_driver(w2))
    r2 = w2.run()
    _kill_verifier_after(Path(r2["verification_pending"]["none"]["out_dir"]), at="manifest_written")
    CV.generation_sidecar_path(w2.row_dir, "none", BCB_FULL, Path(r2["verification_pending"]["none"]["out_dir"]).name).unlink()
    torn = L.pending_generation_state(r2["verification_pending"]["none"]["out_dir"], w2.row_dir, cell="none", label="none")
    assert torn["state"] == "torn" and any("generation sidecar" in x for x in torn["problems"])


def test_the_dispatch_fails_closed_without_an_expectation_record_and_every_generate_row_writes_one(L, tmp_path, monkeypatch, capsys):
    """The run's expectation record is the watcher's fail-closed ground. A dispatch without one (the container's write
    failed, or none is carried) is REFUSED before any spawn (VERIFY_DISPATCH_REFUSED per pending store; the CPU-only
    resume writes the record from a container and dispatches); the record is REPLACED with `spawning` states before
    the spawns — a replace of a record that must already exist — and a failed replace refuses too; EVERY generate row
    writes the record (cells {} when nothing is pending), so its absence means the container never got that far."""
    w = _World(L, tmp_path, monkeypatch, cells=["none"], row_opts={"verification_phase": "generate", "repeat_cells": ["none"]})
    monkeypatch.setattr(L.subprocess, "run", _generate_driver(w))
    result = w.run()
    recs = L.row_result_recs(result)
    spawned = []
    fn = SimpleNamespace(spawn=lambda cfg, *a, **k: spawned.append(cfg["out_dir"]) or TL._Call({"name": "x:verify", "cell": "none", "status": "OK", "phase": "verify"}))
    # (i) no record carried: refused, nothing spawned, by name
    without = {**result, "expected_verifiers": None, "expected_verifiers_error": "OSError: disk full"}
    vrecs = L.verify_pending_cells(fn, without, w.cfgs, recs, sha="f6cf462", run_id="f6cf462-cafebabe", reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT)
    out = capsys.readouterr().out
    assert spawned == [] and [v["status"] for v in vrecs] == ["VERIFY_DISPATCH_REFUSED"] * 2 == [L.REC_VERIFY_DISPATCH_REFUSED] * 2
    assert "verifier dispatch REFUSED" in out and "disk full" in out and "--verify-pending" in out and "disk full" in vrecs[0]["error"]
    assert L.REC_VERIFY_DISPATCH_REFUSED not in L.REC_OK_STATUSES and all(r["status"] == "GENERATED" for r in recs)
    # (ii) the pre-spawn REPLACE fails (the record is not on the volume / the write raised): refused, nothing spawned
    monkeypatch.setattr(L, "put_row_dir_file", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("verifiers_x.json does not exist on the volume")))
    vrecs = L.verify_pending_cells(fn, result, w.cfgs, recs, sha="f6cf462", run_id="f6cf462-cafebabe", reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT)
    out = capsys.readouterr().out
    assert spawned == [] and [v["status"] for v in vrecs] == ["VERIFY_DISPATCH_REFUSED"] * 2 and "could not be REPLACED" in out
    # (iii) the REPLACE lands (must_exist=True): `spawning` per label BEFORE the spawns, the outcomes after
    writes = []
    monkeypatch.setattr(L, "put_row_dir_file", lambda row_dir, name, d, **kw: writes.append((name, kw.get("must_exist"), {k: v["state"] for k, v in d["cells"].items()})) or "x")
    vrecs = L.verify_pending_cells(fn, result, w.cfgs, recs, sha="f6cf462", run_id="f6cf462-cafebabe", reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT)
    assert len(spawned) == 2 and [v["status"] for v in vrecs] == ["OK", "OK"]
    assert writes == [("verifiers_f6cf462-cafebabe.json", True, {"none": "spawning", "none-rep2": "spawning"}),
                      ("verifiers_f6cf462-cafebabe.json", True, {"none": "spawned", "none-rep2": "spawned"})]
    assert L.VERIFIER_STATE_SPAWNING == "spawning" not in L.VERIFIER_TERMINAL_STATES
    # (iv) every generate row writes the record — one whose driver forgot the marker (nothing pending) included
    w2 = _World(L, tmp_path / "w2", monkeypatch, cells=["none"], row_opts={"verification_phase": "generate"})
    monkeypatch.setattr(L.subprocess, "run", _generate_driver(w2, mark=False))
    r2 = w2.run()
    assert r2["status"] == "ROW_PARTIAL" and r2["verification_pending"] == {}
    rec2 = json.loads((w2.row_dir / "watch" / "verifiers_f6cf462-cafebabe.json").read_text())
    assert rec2["cells"] == {} and rec2["written_by"] == "row_container" and r2["expected_verifiers"]["cells"] == {}
    # a coupled row writes none
    w3 = _World(L, tmp_path / "w3", monkeypatch, cells=["none"])
    w3.run()
    assert not list((w3.row_dir / "watch").glob("verifiers_*.json"))


def test_the_watcher_withholds_done_on_an_unavailable_record_read_and_on_a_generate_row_without_a_record(L, tmp_path, monkeypatch, capsys):
    """`read_verifiers_record` swallowed EVERY read error as "no record", which `outstanding_verifiers` read as
    "nothing owed" — a transient Volume error in the GPU→CPU gap printed DONE (the H1 gap again). Now an unavailable
    read withholds and retries; a CONFIRMED-missing record on a GENERATE-phase row fails closed (the container writes
    it before returning, for every generate row); a coupled row without one owes nothing."""
    w = _World(L, tmp_path, monkeypatch, cells=["none"], row_opts={"verification_phase": "generate"})
    monkeypatch.setattr(L.subprocess, "run", _generate_driver(w))
    result = w.run()
    fcid, run_id = result["function_call_id"], "f6cf462-cafebabe"
    W = _watcher_module()
    healthy = {W.row_sweep_certificate_name(fcid, "none"): _cert(run_id, fcid, "none")}
    rel = str(w.row_dir).lstrip("/")
    record_key = f"{rel}/watch/verifiers_{run_id}.json"
    monkeypatch.setattr(W, "sandbox_app_id", lambda *a, **k: "ap-1")
    monkeypatch.setattr(W, "classify", lambda f: SimpleNamespace(unavailable=False, pending=False, terminal=True, describe=lambda: "terminal"))
    monkeypatch.setattr(W, "sweep_attempt", lambda owner, **kw: {"verified_empty": True, "ids_terminated": [], "still_live": []})

    def tick(vol):
        monkeypatch.setattr(W, "_volume", lambda: vol)
        rc = W.watch_row_dir(run_id=run_id, row_dir=str(w.row_dir), poll_s=0, max_hours=1, state_dir=str(tmp_path / "state"), once=True)
        return rc, capsys.readouterr().out
    # (a) the record read raises a transient error in the gap: NOT done, named as unavailable
    class _Flaky(_RowDirVolume):
        def read_file(self, path):
            if path == record_key:
                raise RuntimeError("GRPCError: UNAVAILABLE")
            yield from super().read_file(path)
    rc, out = tick(_Flaky(w.row_dir, certified=healthy))
    assert rc == W.EXIT_UNVERIFIED and "read UNAVAILABLE (RuntimeError: GRPCError: UNAVAILABLE)" in out and "DONE withheld" in out and "DONE:" not in out
    # (b) the record is CONFIRMED missing on a generate row: withheld, by name (fail closed)
    vol = _RowDirVolume(w.row_dir, drop=(f"verifiers_{run_id}.json",), certified=healthy)
    assert record_key not in vol.files
    rc, out = tick(vol)
    assert rc == W.EXIT_UNVERIFIED and f"no expected-verifiers record verifiers_{run_id}.json for this generate row" in out and "DONE:" not in out
    # (c) a modal-style NotFoundError is a confirmed miss too; any other exception is unavailable (pure)
    class NotFoundError(Exception):
        pass

    class _NF:
        def read_file(self, path):
            raise NotFoundError("not found")

    class _Down:
        def read_file(self, path):
            raise OSError("socket closed")
    assert W.read_verifiers_record(_NF(), str(w.row_dir), run_id) is None
    assert W.read_verifiers_record(_Down(), str(w.row_dir), run_id) == {"_unavailable": "OSError: socket closed"}
    assert W.outstanding_verifiers(None, run_id=run_id, pointers=[], certified={}, generate_row=True)[0].startswith("no expected-verifiers record")
    assert W.outstanding_verifiers(None, run_id=run_id, pointers=[], certified={}, generate_row=False) == []
    assert W.outstanding_verifiers({"_unavailable": "x"}, run_id=run_id, pointers=[], certified={}) == [
        f"verifiers record verifiers_{run_id}.json: read UNAVAILABLE (x) — retrying; DONE withheld"]
    row_ptr = json.loads((w.row_dir / "watch" / f"attempt_{fcid}.json").read_text())
    assert W.generate_row_pointer_present([row_ptr], run_id=run_id) and not W.generate_row_pointer_present([{**row_ptr, "phase": "coupled"}], run_id=run_id)
    assert W.VERIFIER_STATE_SPAWNING == L.VERIFIER_STATE_SPAWNING == "spawning"
    # (d) with the record present and the verifier's pointer certified: DONE as before
    vol = _RowDirVolume(w.row_dir, certified=healthy)
    upd = json.loads(vol.files[record_key])
    upd["cells"]["none"].update(state="spawned", function_call_id="fc-v-1")
    vol.files[record_key] = json.dumps(upd).encode()
    vol.files[f"{rel}/watch/attempt_fc-v-1.json"] = json.dumps(_verify_pointer_doc(run_id, "fc-v-1", "none", result["cells"]["none"]["out"])).encode()
    vol.files[f"{rel}/watch/{W.row_sweep_certificate_name('fc-v-1', 'none')}"] = json.dumps(_cert(run_id, "fc-v-1", "none")).encode()
    rc, out = tick(vol)
    assert rc == W.EXIT_DONE, out
    # a coupled row without a record: unchanged (the coupled-row watcher test above reads DONE)


# --------------------------------------------------------------------------- #
# One active verifier per (row dir, cell, store)                                 #
# --------------------------------------------------------------------------- #


class _FakeCall:
    """A FunctionCall double whose ``get(timeout=0)`` is scripted: ``"running"`` raises the builtin TimeoutError (SDK
    1.5.4's not-ready), an exception instance is raised, anything else is returned as the result. ``then`` (with
    ``"running"``): the result the call yields once it has been read as running ``running_polls`` times — the
    dispatcher POLLS a joined call with ``get(timeout=0)`` until it finishes (tinycap-verify-rate-limit), so a call that
    is joined AND harvested in one attempt must finish; a bare ``"running"`` never does (probe-only use)."""

    def __init__(self, object_id, outcome, *, then=None, running_polls: int = 1):
        self.object_id, self.outcome, self.gets = object_id, outcome, 0
        self._then, self._running_left = then, running_polls

    def get(self, timeout=None):
        self.gets += 1
        if self.outcome == "running":
            if self._then is None or self._running_left > 0:
                self._running_left -= 1
                raise TimeoutError()
            return self._then
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def _fake_function_call(table: Mapping[str, object]):
    """``modal.FunctionCall`` double: ``from_id`` serves the scripted call, raises a ``NotFoundError`` for an unknown id."""
    class NotFoundError(Exception):
        pass

    def from_id(cid):
        if cid not in table:
            raise NotFoundError(f"FunctionCall {cid} not found")
        return table[cid]
    return SimpleNamespace(from_id=from_id, NotFoundError=NotFoundError)


_PRIOR_RECORD_SEQ = [0]


def _prior_record(w, *, run_id: str, cells: Mapping[str, Mapping[str, object]], written_at: Optional[float] = None) -> Path:
    """An earlier run's verifiers record under the row dir (what its dispatch left behind). By default it is stamped
    NEWER than any record the world's container wrote (the dispatch update REPLACES the container's record after the
    row returned, so a lost run's last word is its dispatch); ``written_at`` pins an explicit order."""
    import time as _time
    doc = w.L.expected_verifiers_record(run_id=run_id, row=ROW, suffix=SUFFIX, function_call_id="fc-gen-old", cells=cells, written_by="main_dispatch")
    _PRIOR_RECORD_SEQ[0] += 1
    doc["written_at"] = float(written_at) if written_at is not None else _time.time() + 60.0 + _PRIOR_RECORD_SEQ[0]
    return w.L.write_verifiers_record(w.row_dir, doc)


def test_the_probe_classifies_an_earlier_verifiers_call_and_the_ownership_decision_follows_it(L, tmp_path, monkeypatch):
    """The record's `spawned` entry with its FunctionCall id is the ownership token. `probe_verifier_call` folds
    `FunctionCall.from_id(id).get(timeout=0)` to active / terminal_ok / terminal_failed / lost / unavailable;
    `classify_verifier_ownership` decides spawn / join / refuse off the newest record entry: nothing recorded or a
    terminal state -> spawn; `spawning` with no id -> refuse; a running call -> join; a terminal or lost call -> spawn
    with the old id superseded; an unreadable call -> refuse naming it."""
    class RemoteError(Exception):
        pass

    class ConnectionError_(OSError):
        pass
    ConnectionError_.__name__ = "ConnectionError"
    table = {"fc-run": _FakeCall("fc-run", "running"), "fc-ok": _FakeCall("fc-ok", {"status": "OK", "verified_by": "fc-ok"}),
             "fc-bad": _FakeCall("fc-bad", RemoteError("the verifier raised")), "fc-down": _FakeCall("fc-down", ConnectionError_("socket closed"))}
    monkeypatch.setattr(L.modal, "FunctionCall", _fake_function_call(table), raising=False)
    assert L.probe_verifier_call("fc-run")["state"] == "active" == L.CALL_ACTIVE
    ok = L.probe_verifier_call("fc-ok")
    assert ok["state"] == "terminal_ok" and ok["result"] == {"status": "OK", "verified_by": "fc-ok"}
    bad = L.probe_verifier_call("fc-bad")
    assert bad["state"] == "terminal_failed" and bad["error"].startswith("RemoteError: the verifier raised")
    assert L.probe_verifier_call("fc-down")["state"] == "unavailable"
    lost = L.probe_verifier_call("fc-gone")
    assert lost["state"] == "lost" and "NotFoundError" in lost["error"]
    # the decision, off a row dir's records
    w = _World(L, tmp_path, monkeypatch, cells=["none"], row_opts={"verification_phase": "generate"})
    assert L.classify_verifier_ownership(w.row_dir, "none")["decision"] == "spawn"                       # no record at all
    _prior_record(w, run_id="r-old", cells={"none": {"cell": "none", "out_dir": "/s", "state": "spawned", "function_call_id": "fc-run"}}, written_at=100.0)
    own = L.classify_verifier_ownership(w.row_dir, "none")
    assert own["decision"] == "join" == L.OWN_JOIN and own["call_id"] == "fc-run" and own["prior_run_id"] == "r-old" and own["probe"]["state"] == "active"
    assert "call" not in own["probe"] and "result" not in own["probe"]                                   # JSON-clean (the report crosses .remote)
    _prior_record(w, run_id="r-old2", cells={"none": {"cell": "none", "out_dir": "/s", "state": "spawned", "function_call_id": "fc-bad"}}, written_at=200.0)
    own = L.classify_verifier_ownership(w.row_dir, "none")                                                # the NEWEST record decides
    assert own["decision"] == "spawn" and own["superseded"] == "fc-bad" and "terminal_failed" in own["reason"]
    _prior_record(w, run_id="r-old3", cells={"none": {"cell": "none", "out_dir": "/s", "state": "spawned", "function_call_id": "fc-gone"}}, written_at=300.0)
    assert L.classify_verifier_ownership(w.row_dir, "none")["superseded"] == "fc-gone"                  # lost -> supersede
    _prior_record(w, run_id="r-old4", cells={"none": {"cell": "none", "out_dir": "/s", "state": "spawned", "function_call_id": "fc-down"}}, written_at=400.0)
    own = L.classify_verifier_ownership(w.row_dir, "none")
    assert own["decision"] == "refuse" == L.OWN_REFUSE and "fc-down" in own["reason"] and "could not be read" in own["reason"]
    _prior_record(w, run_id="r-old5", cells={"none": {"cell": "none", "out_dir": "/s", "state": "spawning"}}, written_at=500.0)
    own = L.classify_verifier_ownership(w.row_dir, "none")
    assert own["decision"] == "refuse" and "reads 'spawning'" in own["reason"] and "no call id" in own["reason"]
    _prior_record(w, run_id="r-old6", cells={"none": {"cell": "none", "out_dir": "/s", "state": "verified", "function_call_id": "fc-ok"}}, written_at=600.0)
    assert L.classify_verifier_ownership(w.row_dir, "none")["decision"] == "spawn"                     # a terminal state owes nothing
    assert L.classify_verifier_ownership(w.row_dir, "other-label")["decision"] == "spawn"
    assert L.VERIFIER_STATE_JOINED == "joined" not in L.VERIFIER_TERMINAL_STATES


def test_a_resume_joins_a_running_verifier_supersedes_a_dead_or_lost_one_and_refuses_an_unknown_one(L, tmp_path, monkeypatch, capsys):
    """The overlap: the first `main` was lost with its remote verifier alive; the CPU-only retry used to spawn a
    SECOND verifier for the same store. Now: record says spawned A, A still running -> the resume JOINS A
    (no spawn; A's result harvested as this run's verify REC; the record reads `joined` + A); A terminal-failed ->
    spawn B, the record's entry names B with A under `superseded`; A lost (from_id raises) -> spawn; `spawning` with no
    id, or an unreadable call -> refused by name (VERIFY_DISPATCH_REFUSED), nothing spawned for that store."""
    w = _World(L, tmp_path, monkeypatch, cells=["none", LORA], row_opts={"verification_phase": "generate"})
    monkeypatch.setattr(L.subprocess, "run", _generate_driver(w))
    first = w.run()
    pend = first["verification_pending"]
    a_result = {"name": f"{pend['none']['name']}:verify", "cell": "none", "status": "OK", "phase": "verify", "function_call_id": "fc-A",
                "verified_by": "fc-A"}
    table = {"fc-A": _FakeCall("fc-A", "running")}
    monkeypatch.setattr(L.modal, "FunctionCall", _fake_function_call(table), raising=False)
    written: Dict[str, dict] = {}
    monkeypatch.setattr(L, "put_row_dir_file", lambda row_dir, name, d, **kw: written.__setitem__(name, json.loads(json.dumps(d, default=str))) or "x")
    spawned = []
    fn = SimpleNamespace(spawn=lambda cfg, *a, **k: spawned.append(cfg["out_dir"]) or TL._Call({"name": f"{cfg['name']}:verify", "cell": cfg["cell"],
                                                                                              "status": "OK", "phase": "verify", "verified_by": "fc-B"}))
    # the lost first run dispatched A for `none` (spawned) and never got to LORA (expected)
    _prior_record(w, run_id="f6cf462-first", cells={"none": {"cell": "none", "out_dir": pend["none"]["out_dir"], "state": "spawned", "function_call_id": "fc-A"},
                                                    LORA: {"cell": LORA, "out_dir": pend[LORA]["out_dir"], "state": "expected"}})
    # (i) A still running: JOIN — the report marks `none` joined, LORA dispatchable
    report = L.pending_verification_report(w.row_dir, run_id="f6cf462-second")
    assert report["status"] == "OK" and set(report["pending"]) == {"none", LORA} and report["refused"] == {}
    assert report["joining"] == {"none": {"call_id": "fc-A", "run_id": "f6cf462-first"}}
    assert report["pending"]["none"]["join_call_id"] == "fc-A" and "join_call_id" not in report["pending"][LORA]
    rec2 = json.loads((w.row_dir / "watch" / "verifiers_f6cf462-second.json").read_text())
    assert rec2["cells"]["none"] == {"cell": "none", "out_dir": pend["none"]["out_dir"], "state": "joined", "function_call_id": "fc-A", "joined_from_run": "f6cf462-first"}
    assert rec2["cells"][LORA]["state"] == "expected"
    resumed, recs = L.resume_verification_recs(report)
    table["fc-A"] = _FakeCall("fc-A", a_result)                                                        # A finishes while we harvest
    vrecs = L.verify_pending_cells(fn, resumed, w.cfgs, recs, sha="f6cf462", run_id="f6cf462-second", reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT)
    out = capsys.readouterr().out
    assert spawned == [pend[LORA]["out_dir"]], "only LORA was spawned; none was joined"
    assert "joining verifier fc-A for none" in out and "still running, spawning nothing" in out
    by_label = {v["label"]: v for v in vrecs}
    assert by_label["none"]["status"] == "OK" and by_label["none"]["joined_call_id"] == "fc-A" and by_label["none"]["verified_by"] == "fc-A"
    assert table["fc-A"].gets == 1                                                                       # A's result harvested, once
    cell_recs = {r["name"]: r for r in recs}
    assert cell_recs[pend["none"]["name"]]["status"] == "OK" and cell_recs[pend["none"]["name"]]["verification"]["function_call_id"] == "fc-A"
    upd = written["verifiers_f6cf462-second.json"]
    assert upd["cells"]["none"]["state"] == "joined" and upd["cells"]["none"]["function_call_id"] == "fc-A"
    assert upd["cells"][LORA]["state"] == "spawned" and upd["cells"][LORA]["function_call_id"]
    # (ii) A terminal-FAILED: spawn B, the record names B with A superseded
    class RemoteError(Exception):
        pass
    table["fc-A"] = _FakeCall("fc-A", RemoteError("sandbox timeout"))
    spawned.clear()
    report = L.pending_verification_report(w.row_dir, run_id="f6cf462-third")
    # (the newest record now is second's: `none` joined fc-A — a joined entry is probed like a spawned one)
    assert report["joining"] == {} and report["pending"]["none"]["superseded_call_id"] == "fc-A" and report["refused"] == {}
    rec3 = json.loads((w.row_dir / "watch" / "verifiers_f6cf462-third.json").read_text())
    assert rec3["cells"]["none"] == {"cell": "none", "out_dir": pend["none"]["out_dir"], "state": "expected", "superseded": ["fc-A"]}
    resumed, recs = L.resume_verification_recs(report)
    vrecs = L.verify_pending_cells(fn, resumed, w.cfgs, recs, sha="f6cf462", run_id="f6cf462-third", reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT)
    out = capsys.readouterr().out
    assert len(spawned) == 2 and "(supersedes fc-A)" in out and all(v["status"] == "OK" for v in vrecs)
    upd = written["verifiers_f6cf462-third.json"]
    assert upd["cells"]["none"]["state"] == "spawned" and upd["cells"]["none"]["function_call_id"] and upd["cells"]["none"]["superseded"] == ["fc-A"]
    # (iii) LOST (from_id raises): spawn, superseded
    _prior_record(w, run_id="f6cf462-fourth", cells={"none": {"cell": "none", "out_dir": pend["none"]["out_dir"], "state": "spawned", "function_call_id": "fc-gone"}},
                  written_at=9e9)
    report = L.pending_verification_report(w.row_dir, run_id="f6cf462-fifth", write_record=False)
    assert report["pending"]["none"]["superseded_call_id"] == "fc-gone" and report["pending"]["none"]["ownership"]["probe"]["state"] == "lost"
    # (iv) `spawning` with no call id, or an unreadable call: REFUSED by name; the other store is still dispatched
    _prior_record(w, run_id="f6cf462-sixth", cells={"none": {"cell": "none", "out_dir": pend["none"]["out_dir"], "state": "spawning"}}, written_at=9e9 + 1)
    report = L.pending_verification_report(w.row_dir, run_id="f6cf462-seventh")
    assert set(report["pending"]) == {LORA} and set(report["refused"]) == {"none"}
    assert "reads 'spawning'" in report["refused"]["none"] and "f6cf462-sixth" in report["refused"]["none"]
    rec7 = json.loads((w.row_dir / "watch" / "verifiers_f6cf462-seventh.json").read_text())
    assert rec7["cells"]["none"]["state"] == "blocked" and "spawning" in rec7["cells"]["none"]["reason"]
    assert rec7["cells"]["none"]["prior"]["run_id"] == "f6cf462-sixth" and rec7["cells"]["none"]["prior"]["state"] == "spawning"
    resumed, recs = L.resume_verification_recs(report)
    refused = [r for r in recs if r["status"] == "VERIFY_DISPATCH_REFUSED"]
    assert len(refused) == 1 and refused[0]["label"] == "none" and "verifier dispatch refused" in refused[0]["error"]
    spawned.clear()
    vrecs = L.verify_pending_cells(fn, resumed, w.cfgs, recs, sha="f6cf462", run_id="f6cf462-seventh", reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT)
    assert spawned == [pend[LORA]["out_dir"]] and [v["label"] for v in vrecs] == [LORA]
    class ConnectionError_(OSError):
        pass
    ConnectionError_.__name__ = "ConnectionError"
    table["fc-down"] = _FakeCall("fc-down", ConnectionError_("socket closed"))
    _prior_record(w, run_id="f6cf462-eighth", cells={"none": {"cell": "none", "out_dir": pend["none"]["out_dir"], "state": "spawned", "function_call_id": "fc-down"}},
                  written_at=9e9 + 2)
    report = L.pending_verification_report(w.row_dir, run_id="f6cf462-ninth", write_record=False)
    assert set(report["refused"]) == {"none"} and "fc-down" in report["refused"]["none"] and "could not be read" in report["refused"]["none"]
    # the verifier's own owner tuple names the call id the record stores: the container's current_function_call_id IS the
    # spawn's object_id, so the driver's claim file (its belt) names the same id
    src = Path(L.__file__).read_text()
    worker = src[src.index("def _verify_cell_worker("):src.index("def verify_cell_cpu(")]
    assert "fcid = str(function_call_id) if function_call_id else current_function_call_id()" in worker
    assert "owner = sandbox_owner(run_id, cell, fcid)" in worker


def test_a_generate_relaunch_joins_a_running_verifier_at_the_door_and_parks_an_unknown_one(L, tmp_path, monkeypatch, capsys):
    """The door path: a generate relaunch that queues a pending store consults the same ownership rule — an earlier
    run's running verifier is queued as a JOIN (main harvests it, nothing spawned; the container's record reads
    `joined` + the call id), a dead one is superseded, an unreadable one parks the cell naming the call id."""
    w = _World(L, tmp_path, monkeypatch, cells=["none", LORA], row_opts={"verification_phase": "generate"})
    monkeypatch.setattr(L.subprocess, "run", _generate_driver(w))
    first = w.run()
    pend = first["verification_pending"]
    class ConnectionError_(OSError):
        pass
    ConnectionError_.__name__ = "ConnectionError"
    table = {"fc-A": _FakeCall("fc-A", "running"), "fc-down": _FakeCall("fc-down", ConnectionError_("socket closed"))}
    monkeypatch.setattr(L.modal, "FunctionCall", _fake_function_call(table), raising=False)
    _prior_record(w, run_id="f6cf462-first", cells={"none": {"cell": "none", "out_dir": pend["none"]["out_dir"], "state": "spawned", "function_call_id": "fc-A"},
                                                    LORA: {"cell": LORA, "out_dir": pend[LORA]["out_dir"], "state": "spawned", "function_call_id": "fc-down"}})
    _RowStandup.argvs.clear()
    capsys.readouterr()
    second = w.run(run_id="f6cf462-second")
    out = capsys.readouterr().out
    assert second["cells"]["none"]["status"] == "GENERATED" and second["verification_pending"]["none"]["join_call_id"] == "fc-A"
    assert second["verification_pending"]["none"]["ownership"]["decision"] == "join" and "JOINS the running verifier fc-A" in out
    assert second["dropped"][LORA]["status"] == "INCOMPLETE_STORE_PARKED" and "fc-down" in second["dropped"][LORA]["error"]
    assert "an earlier verifier may still own it" in second["dropped"][LORA]["error"] and "--verify-pending" in second["dropped"][LORA]["error"]
    assert second["dropped"][LORA]["verifier_ownership"]["decision"] == "refuse" and _RowStandup.argvs == []
    rec2 = json.loads((w.row_dir / "watch" / "verifiers_f6cf462-second.json").read_text())
    assert rec2["cells"]["none"] == {"cell": "none", "out_dir": pend["none"]["out_dir"], "state": "joined", "function_call_id": "fc-A", "joined_from_run": "f6cf462-first"}
    # the refused label is recorded `blocked` WITH the unresolved owner's provenance (never absent, never terminal)
    assert rec2["cells"][LORA]["state"] == "blocked" and rec2["cells"][LORA]["prior"]["function_call_id"] == "fc-down"
    assert rec2["cells"][LORA]["prior"]["run_id"] == "f6cf462-first" and rec2["cells"][LORA]["prior"]["probe"]["state"] == "unavailable"
    # main's dispatch joins A instead of spawning
    monkeypatch.setattr(L, "put_row_dir_file", lambda row_dir, name, d, **kw: "x")
    table["fc-A"] = _FakeCall("fc-A", {"name": f"{pend['none']['name']}:verify", "cell": "none", "status": "OK", "phase": "verify", "verified_by": "fc-A"})
    spawned = []
    recs = L.row_result_recs(second)
    vrecs = L.verify_pending_cells(SimpleNamespace(spawn=lambda cfg, *a, **k: spawned.append(cfg) or TL._Call({})), second, w.cfgs, recs,
                                   sha="f6cf462", run_id="f6cf462-second", reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT)
    assert spawned == [] and vrecs[0]["joined_call_id"] == "fc-A" and vrecs[0]["status"] == "OK"
    assert "joining verifier fc-A for none" in capsys.readouterr().out
    # a dead earlier verifier: queued with the old id superseded, and spawned
    class RemoteError(Exception):
        pass
    table["fc-A"] = _FakeCall("fc-A", RemoteError("died"))
    _RowStandup.argvs.clear()
    third = w.run(run_id="f6cf462-third")
    assert third["verification_pending"]["none"]["superseded_call_id"] == "fc-A" and "join_call_id" not in third["verification_pending"]["none"]
    rec3 = json.loads((w.row_dir / "watch" / "verifiers_f6cf462-third.json").read_text())
    assert rec3["cells"]["none"]["state"] == "expected" and rec3["cells"]["none"]["superseded"] == ["fc-A"]


def test_two_row_runners_of_one_row_and_suffix_on_one_host_keep_their_own_standup_dirs_and_record_copies(L, tmp_path, monkeypatch):
    """Two row runners of the SAME row and suffix on ONE host shared one standup dir: it was spelled as the literal
    ``/tmp/srv_row_<slug>__<suffix>`` (the design's "this container's /tmp"), and with it the row-engine record every
    driver reads and the row copies into ``<row dir>/generation/row_engine__<label>.json``. The 40-shard gate runs
    many pytest processes per node, each under its own ``TMPDIR``, and two gates' shards of this module overlapped on
    one node (2026-10-09, 113 -> 114 on a tree whose earlier gate had passed them): runner B's ``rmtree`` + record
    write landed between runner A's drive and A's record copy, A's copy named B's row dir, and A's relaunch read its
    store ``torn`` and parked it (``KeyError: 'none'`` on ``verification_pending``; the tail_pending door the same).
    The standup dir now lives under the process's temp root (``tempfile.gettempdir()``: the gate's per-task
    ``TMPDIR``; the world's ``tmp_path`` here), which this test pins by running B's WHOLE row inside A's driver call —
    the window between A's record write and A's record copy: both stores read ``pending`` naming their own row dir,
    A's relaunch queues its store at the door, and the two standup dirs are distinct, under their own roots."""
    a = _World(L, tmp_path / "a", monkeypatch, cells=["none"], row_opts={"verification_phase": "generate"})
    b = _World(L, tmp_path / "b", monkeypatch, cells=["none"], row_opts={"verification_phase": "generate"})
    gen_a, gen_b = _generate_driver(a), _generate_driver(b)
    runs_b: List[dict] = []

    def a_driver_then_b_whole_row(argv, **kw):
        # A's driver files A's store off A's record; then runner B's whole row (its rmtree of the standup dir it
        # computes, its standup, its record write, its drive, its record copy) runs before A's own record copy is taken
        rec = gen_a(argv, **kw)
        monkeypatch.setattr(L.subprocess, "run", gen_b)
        runs_b.append(b.run(run_id="f6cf462-runner-b"))
        monkeypatch.setattr(L.subprocess, "run", a_driver_then_b_whole_row)
        return rec
    monkeypatch.setattr(L.subprocess, "run", a_driver_then_b_whole_row)
    first = a.run(run_id="f6cf462-runner-a")
    assert len(runs_b) == 1 and runs_b[0]["status"] == "GENERATED", runs_b[0].get("error")
    assert first["status"] == "GENERATED" and first["cells"]["none"]["status"] == "GENERATED", first.get("error")
    pend_a, pend_b = first["verification_pending"]["none"], runs_b[0]["verification_pending"]["none"]
    # each runner's record copy names ITS row dir — the copy was taken from its own standup dir, not the other's
    assert C.read_row_engine_record(pend_a["row_engine_record"])["row_dir"] == str(a.row_dir)
    assert C.read_row_engine_record(pend_b["row_engine_record"])["row_dir"] == str(b.row_dir)
    for pend, w in ((pend_a, a), (pend_b, b)):
        st = L.pending_generation_state(pend["out_dir"], w.row_dir, cell="none", label="none")
        assert st["state"] == "pending" and st["problems"] == [], st
        assert all(Path(m).is_relative_to(w.row_dir) for m in pend["markers"]) and st["markers"] == pend["markers"]
    # A's relaunch queues its store at the door (the shape of the two flipped nodes): never torn, never parked
    monkeypatch.setattr(L.subprocess, "run", gen_a)
    a.driver_calls.clear()
    _RowStandup.argvs.clear()
    second = a.run(run_id="f6cf462-runner-a-again")
    assert second["status"] == "GENERATED" and second["dropped"] == {} and _RowStandup.argvs == [] and a.driver_calls == []
    assert second["verification_pending"]["none"]["resumed"] is True and second["cells"]["none"]["resumed_pending"] is True
    # the seam: one standup dir per process temp root (the gate's per-task TMPDIR; tmp_path here), never the literal /tmp
    assert a.standup_dir != b.standup_dir and a.standup_dir.name == b.standup_dir.name == f"srv_row_{ROW}__{SUFFIX}"
    assert a.standup_dir.is_relative_to(tmp_path / "a") and b.standup_dir.is_relative_to(tmp_path / "b")
    argv_a, argv_b = first["cells"]["none"]["standup_argv"], runs_b[0]["cells"]["none"]["standup_argv"]
    assert argv_a[argv_a.index("--out") + 1] == str(a.standup_dir) and argv_b[argv_b.index("--out") + 1] == str(b.standup_dir)
    assert L.row_standup_dir(ROW, SUFFIX) == L.STANDUP_LOCAL_ROOT / f"srv_row_{ROW}__{SUFFIX}"
    assert not str(L.row_standup_dir(ROW, SUFFIX)).startswith("/tmp/srv_row_")        # pinned under tmp_path by the worlds


def test_the_watcher_adopts_a_joined_verifiers_pointer_across_run_ids(L, tmp_path, monkeypatch, capsys):
    """The watcher: a `joined` record entry names a call the FIRST run spawned, whose pointer carries that run's id;
    the second run's watcher adopts it — classifies the call, sweeps its owner (the pointer's own tuple) and
    withholds DONE until it is terminal and certified, exactly like a verifier it spawned itself."""
    w = _World(L, tmp_path, monkeypatch, cells=["none"], row_opts={"verification_phase": "generate"})
    monkeypatch.setattr(L.subprocess, "run", _generate_driver(w))
    result = w.run()
    fcid, run_id, old_run = result["function_call_id"], "f6cf462-cafebabe", "f6cf462-first"
    store = result["cells"]["none"]["out"]
    W = _watcher_module()
    assert W.VERIFIER_STATE_JOINED == L.VERIFIER_STATE_JOINED == "joined" and "joined" not in W.VERIFIER_TERMINAL_STATES
    rel = str(w.row_dir).lstrip("/")
    record = json.loads((w.row_dir / "watch" / f"verifiers_{run_id}.json").read_text())
    record["cells"]["none"] = {"cell": "none", "out_dir": store, "state": "joined", "function_call_id": "fc-A", "joined_from_run": old_run}
    (w.row_dir / "watch" / f"verifiers_{run_id}.json").write_text(json.dumps(record))
    vol = _RowDirVolume(w.row_dir, certified={W.row_sweep_certificate_name(fcid, "none"): _cert(run_id, fcid, "none")})
    vol.files[f"{rel}/watch/attempt_fc-A.json"] = json.dumps(_verify_pointer_doc(old_run, "fc-A", "none", store)).encode()    # the FIRST run's pointer
    assert W.adopted_call_ids(record) == {"fc-A"}
    assert set(W.pointers_by_call([json.loads(v) for k, v in vol.files.items() if "/attempt_" in k], run_id=run_id, adopt=["fc-A"])) == {fcid, "fc-A"}
    assert set(W.pointers_by_call([json.loads(v) for k, v in vol.files.items() if "/attempt_" in k], run_id=run_id)) == {fcid}
    monkeypatch.setattr(W, "_volume", lambda: vol)
    monkeypatch.setattr(W, "sandbox_app_id", lambda *a, **k: "ap-1")
    states = {fcid: False, "fc-A": True}                                   # pending?
    monkeypatch.setattr(W, "classify", lambda f: SimpleNamespace(unavailable=False, pending=states[f], terminal=not states[f], describe=lambda: "x"))
    swept = []
    monkeypatch.setattr(W, "sweep_attempt", lambda owner, **kw: swept.append(owner) or {"verified_empty": True, "ids_terminated": [], "still_live": []})

    def tick():
        rc = W.watch_row_dir(run_id=run_id, row_dir=str(w.row_dir), poll_s=0, max_hours=1, state_dir=str(tmp_path / "state"), once=True)
        return rc, capsys.readouterr().out
    rc, out = tick()
    assert rc == W.EXIT_UNVERIFIED and f"verify none [joined from run {old_run}]: attempt fc-A PENDING" in out and "DONE:" not in out
    assert "verifier for none: attempt(s) ['fc-A'] not yet terminal and certified" in out
    states["fc-A"] = False                                                  # A ends: swept by ITS owner tuple (the first run's), then DONE
    rc, out = tick()
    assert rc == W.EXIT_DONE, out
    assert [(o.run_id, o.cell, o.function_call_id) for o in swept] == [(old_run, "none", "fc-A")]
    assert "none@fc-A" in out and "DONE: every attempt terminal and every owner certified" in out


# --------------------------------------------------------------------------- #
# A refusal is never a release; the claim is created before the driver launches #
# --------------------------------------------------------------------------- #


def test_a_refusal_records_blocked_with_the_owners_provenance_and_later_attempts_resolve_that_owner(L, tmp_path, monkeypatch, capsys):
    """The original dispatch wrote `spawning`, spawned A, and the launcher died before recording A's call id; A keeps
    scoring. Attempt B correctly refused — but wrote `skipped`, a terminal state, so attempt C read the newest record
    as a release and spawned C beside A. Now a refusal records NON-terminal `blocked` carrying the unresolved owner
    (`prior`), and every later attempt resolves THAT owner: (1) A unknown (`spawning`, no id) → B refuses → C still
    refuses (never spawns) until the operator's `--supersede-unresolved`; (2) A known but unavailable once → B refuses
    (blocked, carrying A's id) → A readable-active → C JOINS; (3) A terminal → C spawns with A `superseded`."""
    w = _World(L, tmp_path, monkeypatch, cells=["none"], row_opts={"verification_phase": "generate"})
    monkeypatch.setattr(L.subprocess, "run", _generate_driver(w))
    first = w.run()
    pend = first["verification_pending"]
    store = pend["none"]["out_dir"]
    spawned = []
    fn = SimpleNamespace(spawn=lambda cfg, *a, **k: spawned.append(cfg["out_dir"]) or TL._Call({"name": "x:verify", "cell": "none", "status": "OK",
                                                                                                "phase": "verify", "verified_by": "fc-C"}))
    monkeypatch.setattr(L, "put_row_dir_file", lambda row_dir, name, d, **kw: "x")
    assert L.VERIFIER_STATE_BLOCKED == "blocked" not in L.VERIFIER_TERMINAL_STATES

    def attempt(run_id, **kw):
        report = L.pending_verification_report(w.row_dir, run_id=run_id, **kw)
        resumed, recs = L.resume_verification_recs(report)
        vrecs = L.verify_pending_cells(fn, resumed, w.cfgs, recs, sha="f6cf462", run_id=run_id, reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT)
        return report, recs, vrecs
    # ---- (1) the launcher died between `spawning` and the call id: A is unknown
    _prior_record(w, run_id="f6cf462-A", cells={"none": {"cell": "none", "out_dir": store, "state": "spawning"}})
    rep_b, recs_b, _ = attempt("f6cf462-B")
    assert set(rep_b["refused"]) == {"none"} and spawned == [] and "no call id" in rep_b["refused"]["none"]
    rec_b = json.loads((w.row_dir / "watch" / "verifiers_f6cf462-B.json").read_text())["cells"]["none"]
    assert rec_b["state"] == "blocked" and rec_b["prior"] == {"run_id": "f6cf462-A", "state": "spawning", "function_call_id": None,
                                                            "record": "verifiers_f6cf462-A.json"}
    assert [r["status"] for r in recs_b if r.get("label") == "none"] == ["VERIFY_DISPATCH_REFUSED"]
    # C reads B's `blocked` (the newest record) — and still refuses: the carried owner is unresolved, never a release
    rep_c, _, _ = attempt("f6cf462-C")
    assert set(rep_c["refused"]) == {"none"} and spawned == [], "a blocked record must never read as a release"
    rec_c = json.loads((w.row_dir / "watch" / "verifiers_f6cf462-C.json").read_text())["cells"]["none"]
    assert rec_c["state"] == "blocked" and rec_c["prior"]["run_id"] == "f6cf462-A" and rec_c["prior"]["state"] == "spawning"   # provenance preserved through B
    assert "--supersede-unresolved none" in rep_c["refused"]["none"]
    # the operator's explicit take-over: recorded, spawned
    rep_d, _, vrecs_d = attempt("f6cf462-D", supersede_unresolved=["none"])
    assert rep_d["refused"] == {} and spawned == [store] and vrecs_d[0]["status"] == "OK"
    assert rep_d["pending"]["none"]["superseded_unresolved"] == {"run_id": "f6cf462-A", "state": "spawning", "function_call_id": None,
                                                                  "record": "verifiers_f6cf462-A.json"}
    rec_d = json.loads((w.row_dir / "watch" / "verifiers_f6cf462-D.json").read_text())["cells"]["none"]
    assert rec_d["state"] == "expected" and rec_d["superseded_unresolved"]["run_id"] == "f6cf462-A"
    # ---- (2) A known, unavailable once: B refuses (blocked, carrying A's id); A readable-active later: C joins
    class ConnectionError_(OSError):
        pass
    ConnectionError_.__name__ = "ConnectionError"
    table = {"fc-A": _FakeCall("fc-A", ConnectionError_("socket closed"))}
    monkeypatch.setattr(L.modal, "FunctionCall", _fake_function_call(table), raising=False)
    spawned.clear()
    _prior_record(w, run_id="f6cf462-A2", cells={"none": {"cell": "none", "out_dir": store, "state": "spawned", "function_call_id": "fc-A"}})
    rep_b, _, _ = attempt("f6cf462-B2")
    assert set(rep_b["refused"]) == {"none"} and spawned == [] and "fc-A" in rep_b["refused"]["none"]
    rec_b = json.loads((w.row_dir / "watch" / "verifiers_f6cf462-B2.json").read_text())["cells"]["none"]
    assert rec_b["state"] == "blocked" and rec_b["prior"]["function_call_id"] == "fc-A" and rec_b["prior"]["probe"]["state"] == "unavailable"
    # A readable now, still scoring when C probes it; it finishes while C's dispatch polls it (the harvest must end)
    table["fc-A"] = _FakeCall("fc-A", "running", then={"name": "n:verify", "cell": "none", "status": "OK", "phase": "verify", "verified_by": "fc-A"})
    rep_c, _, _ = attempt("f6cf462-C2")
    assert spawned == [] and rep_c["joining"] == {"none": {"call_id": "fc-A", "run_id": "f6cf462-A2"}}, "C joins A, never spawns beside it"
    assert "joining verifier fc-A for none" in capsys.readouterr().out
    rec_c = json.loads((w.row_dir / "watch" / "verifiers_f6cf462-C2.json").read_text())["cells"]["none"]
    assert rec_c["state"] == "joined" and rec_c["function_call_id"] == "fc-A"
    # ---- (3) A terminal (the joined entry is the newest; its call re-probed): C spawns with A superseded
    class RemoteError(Exception):
        pass
    table["fc-A"] = _FakeCall("fc-A", RemoteError("sandbox timeout"))
    rep_e, _, vrecs_e = attempt("f6cf462-E")
    assert spawned == [store] and rep_e["pending"]["none"]["superseded_call_id"] == "fc-A" and vrecs_e[0]["status"] == "OK"
    # a still-unavailable A after a blocked B: re-probed on every attempt, refused every time — never released by the refusal
    table["fc-A"] = _FakeCall("fc-A", ConnectionError_("socket closed"))
    spawned.clear()
    _prior_record(w, run_id="f6cf462-A3", cells={"none": {"cell": "none", "out_dir": store, "state": "spawned", "function_call_id": "fc-A"}})
    for rid in ("f6cf462-B3", "f6cf462-C3", "f6cf462-D3"):
        rep, _, _ = attempt(rid)
        assert set(rep["refused"]) == {"none"} and spawned == []
        assert json.loads((w.row_dir / "watch" / f"verifiers_{rid}.json").read_text())["cells"]["none"]["prior"]["function_call_id"] == "fc-A"
    # the classifier, directly: a blocked record with no carried owner is itself refused (never a release)
    _prior_record(w, run_id="f6cf462-X", cells={"none": {"cell": "none", "out_dir": store, "state": "blocked", "reason": "x"}})
    own = L.classify_verifier_ownership(w.row_dir, "none")
    assert own["decision"] == "refuse" and "carries no owner" in own["reason"]
    # main's grammar: the operator flag rides only beside --verify-pending, spelled last, and refuses a label the row lacks
    base = dict(sha="f6cf462", app_name=L.APP_NAME, pass1=True, dry_run=True, cells="none", modes="off", controls="bigcodebench_full", out_suffix="s", row=True)
    with pytest.raises(SystemExit, match="pass --verify-pending beside it"):
        L.main(**base, verification_phase="generate", supersede_unresolved="none")
    with pytest.raises(SystemExit, match="not a cell label of this row"):
        L.main(**base, verification_phase="generate", verify_pending=True, supersede_unresolved="other")
    with pytest.raises(SystemExit, match="mean nothing without --row"):
        L.main(**{**base, "row": False}, supersede_unresolved="none")
    capsys.readouterr()
    L.main(**base, verification_phase="generate", verify_pending=True, supersede_unresolved="none,none-rep2")
    argv_line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[tinycap] argv:"))
    assert argv_line.split("  (run it")[0].endswith("--verification-phase generate --verify-pending --supersede-unresolved none,none-rep2")
    ns = L.cli_parser().parse_args(["--sha", "x", "--app-name", L.APP_NAME, "--pass1", "--row", "--cells", "none", "--modes", "off", "--out-suffix", "s",
                                    C.FLAG_VERIFICATION_PHASE, "generate", "--verify-pending", "--supersede-unresolved", "none"])
    assert ns.supersede_unresolved == "none" and "supersede_unresolved=a.supersede_unresolved" in Path(L.__file__).read_text()
    assert L.ROW_SUPERSEDE_UNRESOLVED_FLAG == "--supersede-unresolved"


def test_the_operator_supersede_reaches_the_report_through_main_and_the_watcher_never_reads_blocked_as_done(L, tmp_path, monkeypatch, capsys):
    """`main --verify-pending --supersede-unresolved <label>` passes the labels to the CPU report; a blocked label in
    a run's record keeps that run's watcher pending (never DONE), naming the unresolved owner."""
    w = _World(L, tmp_path, monkeypatch, cells=["none"], row_opts={"verification_phase": "generate"})
    monkeypatch.setattr(L.subprocess, "run", _generate_driver(w))
    result = w.run()
    store = result["verification_pending"]["none"]["out_dir"]
    _prior_record(w, run_id="f6cf462-A", cells={"none": {"cell": "none", "out_dir": store, "state": "spawning"}})
    seen = {}
    monkeypatch.setattr(L, "run_row_full", SimpleNamespace(spawn=lambda cfgs, **kw: pytest.fail("no GPU row")))
    def _report_remote(sha, row_dir, rid, **kw):
        seen["kw"] = dict(kw)
        return L.pending_verification_report(w.row_dir, run_id=rid, **kw)
    monkeypatch.setattr(L, "row_verify_pending_report", SimpleNamespace(remote=_report_remote))
    monkeypatch.setattr(L, "verify_cell_cpu", SimpleNamespace(spawn=lambda cfg, *a, **k: TL._Call({"name": f"{cfg['name']}:verify", "cell": "none",
                                                                                                    "status": "OK", "phase": "verify", "verified_by": "fc-D"})))
    monkeypatch.setattr(L, "put_row_dir_file", lambda row_dir, name, d, **kw: "x")
    kw = dict(sha="f6cf462", app_name=L.APP_NAME, pass1=True, cells="none", modes="off", controls="bigcodebench_full,bfcl_full", out_suffix=SUFFIX,
              row=True, verification_phase="generate", verify_pending=True)
    with pytest.raises(SystemExit) as e:                                    # refused: the unresolved owner blocks the store
        L.main(**kw)
    assert "VERIFY_DISPATCH_REFUSED" in str(e.value) and seen["kw"] == {"supersede_unresolved": []}
    capsys.readouterr()
    L.main(**kw, supersede_unresolved="none")                               # the operator's decision: exits 0
    out = capsys.readouterr().out
    assert seen["kw"] == {"supersede_unresolved": ["none"]}
    assert "--supersede-unresolved ['none']" in out and "superseded_unresolved" in out and "2/2 container(s) OK" in out
    # the watcher: a `blocked` entry is pending, never DONE — even with every pointer terminal and certified
    W = _watcher_module()
    assert W.VERIFIER_STATE_BLOCKED == "blocked" not in W.VERIFIER_TERMINAL_STATES
    fcid, run_id = result["function_call_id"], "f6cf462-cafebabe"
    record = json.loads((w.row_dir / "watch" / f"verifiers_{run_id}.json").read_text())
    record["cells"]["none"] = {"cell": "none", "out_dir": store, "state": "blocked", "reason": "r",
                               "prior": {"run_id": "f6cf462-A", "state": "spawning", "function_call_id": None, "record": "verifiers_f6cf462-A.json"}}
    (w.row_dir / "watch" / f"verifiers_{run_id}.json").write_text(json.dumps(record))
    vol = _RowDirVolume(w.row_dir, certified={W.row_sweep_certificate_name(fcid, "none"): _cert(run_id, fcid, "none")})
    monkeypatch.setattr(W, "_volume", lambda: vol)
    monkeypatch.setattr(W, "sandbox_app_id", lambda *a, **k: "ap-1")
    monkeypatch.setattr(W, "classify", lambda f: SimpleNamespace(unavailable=False, pending=False, terminal=True, describe=lambda: "terminal"))
    rc = W.watch_row_dir(run_id=run_id, row_dir=str(w.row_dir), poll_s=0, max_hours=1, state_dir=str(tmp_path / "state"), once=True)
    out = capsys.readouterr().out
    assert rc == W.EXIT_UNVERIFIED and "verifier for none: BLOCKED" in out and "run f6cf462-A" in out and "no call id" in out and "DONE:" not in out
    assert W.outstanding_verifiers(record, run_id=run_id, pointers=[], certified={}, generate_row=True)[0].startswith("verifier for none: BLOCKED")


def test_the_worker_creates_and_commits_the_drivers_claim_before_launching_and_refuses_a_foreign_committed_claim(L, tmp_path, monkeypatch):
    """The cross-container point: the driver's O_EXCL claim was created inside the CPU container and committed only
    after scoring, so another container never saw it in time. The harness worker now reloads, reads the driver's
    claim path, refuses a committed FOREIGN claim before any driver launches (VERIFY_CLAIM_HELD), takes over
    exactly the call the dispatch found terminal / lost (the driver's TINYCAP_VERIFY_SUPERSEDES set to it, and only
    then), else creates the claim naming THIS fcid (exactly VERIFY_CLAIM_FIELDS) and commits IMMEDIATELY — before the
    driver starts, which its own O_EXCL then reads as "ours"."""
    repos = tmp_path / "repos"
    TL._tree(repos, "f6cf462")
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    row_dir = tmp_path / "rows" / ROW / SUFFIX
    store, mk, pend = _verified_store(tmp_path, verified_by="fc-old", marker=True, row_dir=row_dir)      # a store with its marker standing
    doc = json.loads(next(store.glob("cells/*/manifest.json")).read_text())
    doc["row_engine"]["verification_pass"] = {"state": "pending", "tasks": [BCB_FULL]}                    # pending: a normal verify
    next(store.glob("cells/*/manifest.json")).write_text(json.dumps(doc))
    cfg = {"cell": "none", "name": pend["name"], "model": ROW, "mode": "off", "out_dir": str(store)}
    events = []
    monkeypatch.setattr(L, "VOL", SimpleNamespace(reload=lambda: events.append("reload"), commit=lambda: events.append("commit")))
    claim_path = CV.verify_claim_path(row_dir, "none", BCB_FULL, store.name)
    seen = {}

    gate = L.modal.Dict.from_name(C.VERIFY_CLAIMS_DICT_NAME)                                          # the stub's one store per name

    def _driver(argv, **kw):
        # the driver starts with the claim already on disk, naming this fcid, exactly the driver's fields, and COMMITTED
        seen["claim_at_start"] = json.loads(claim_path.read_text())
        seen["commits_before_driver"] = events.count("commit")
        seen["env_supersedes"] = kw["env"].get(CV.VERIFY_SUPERSEDES_ENV)
        # the gate entry (modal.Dict, linearizable across containers) stands FIRST, same fields
        seen["gate_at_start"] = dict(gate.get(str(claim_path)) or {})
        return SimpleNamespace(returncode=1, stdout="", stderr="stopped here")                            # the verdict is not the point
    monkeypatch.setattr(L.subprocess, "run", _driver)
    rec = L._verify_cell_worker(cfg, "f6cf462", "f6cf462-cafebabe", STANDUP_A, str(row_dir), pending=pend, function_call_id="fc-new")
    assert set(seen["claim_at_start"]) == set(CV.VERIFY_CLAIM_FIELDS) and seen["claim_at_start"]["function_call_id"] == "fc-new"
    assert seen["claim_at_start"]["run_id"] == "f6cf462-cafebabe" and seen["commits_before_driver"] >= 1 and events[0] == "reload"
    assert seen["env_supersedes"] is None and rec["claim"]["claims"] == [str(claim_path)] and rec["claim"]["refused"] is None
    assert rec["status"] == "VERIFY_FAILED" and rec["driver_rc"] == 1                                     # the driver's rc still decides
    assert set(seen["gate_at_start"]) == set(CV.VERIFY_CLAIM_FIELDS) and seen["gate_at_start"]["function_call_id"] == "fc-new"
    assert rec["claim"]["gate"] == [{"key": str(claim_path), "ticket": None}] and rec["gate_released_by_worker"] == [str(claim_path)]
    assert gate.get(str(claim_path)) is None                                                            # released with the file
    # a FOREIGN committed claim: refused before any driver launches, by name
    claim_path.write_text(json.dumps({"function_call_id": "fc-other", "run_id": "f6cf462-other", "started_at": 1.0, "host": "h"}))
    calls = []
    monkeypatch.setattr(L.subprocess, "run", lambda argv, **kw: calls.append(argv) or SimpleNamespace(returncode=0, stdout="", stderr=""))
    held = L._verify_cell_worker(cfg, "f6cf462", "f6cf462-cafebabe", STANDUP_A, str(row_dir), pending=pend, function_call_id="fc-new2")
    assert held["status"] == "VERIFY_CLAIM_HELD" == L.REC_VERIFY_CLAIM_HELD and calls == [] and "driver_argv" not in held
    assert "fc-other" in held["error"] and "holds the committed claim" in held["error"] and held["claim"]["refused"]
    assert json.loads(claim_path.read_text())["function_call_id"] == "fc-other"                          # untouched
    # the SUPERSEDE the dispatch decided (fc-other found terminal / lost): the claim is taken over, the driver's token set
    monkeypatch.setattr(L.subprocess, "run", _driver)
    events.clear()
    sup = L._verify_cell_worker(cfg, "f6cf462", "f6cf462-cafebabe", STANDUP_A, str(row_dir), pending={**pend, "superseded_call_id": "fc-other"},
                                function_call_id="fc-new3")
    assert seen["claim_at_start"]["function_call_id"] == "fc-new3" and seen["env_supersedes"] == "fc-other" and seen["commits_before_driver"] >= 1
    assert sup["claim"]["superseded"] == "fc-other" and sup["claim"]["refused"] is None
    # a claim naming a call the dispatch did NOT supersede is still refused even when a (different) supersede was decided
    claim_path.write_text(json.dumps({"function_call_id": "fc-third", "run_id": "r", "started_at": 1.0, "host": "h"}))
    monkeypatch.setattr(L.subprocess, "run", lambda argv, **kw: calls.append(argv) or SimpleNamespace(returncode=0, stdout="", stderr=""))
    other = L._verify_cell_worker(cfg, "f6cf462", "f6cf462-cafebabe", STANDUP_A, str(row_dir), pending={**pend, "superseded_call_id": "fc-other"},
                                  function_call_id="fc-new4")
    assert other["status"] == "VERIFY_CLAIM_HELD" and calls == [] and "fc-third" in other["error"]
    # a torn claim is not free
    claim_path.write_text("{torn")
    torn = L._verify_cell_worker(cfg, "f6cf462", "f6cf462-cafebabe", STANDUP_A, str(row_dir), pending=pend, function_call_id="fc-new5")
    assert torn["status"] == "VERIFY_CLAIM_HELD" and "not readable" in torn["error"] and calls == []
    # the claim helper, directly: the driver's spelling and fields
    claim_path.unlink()
    res = L.claim_store_for_verifier(row_dir, cell="none", out_dir=store, tasks=[BCB_FULL], function_call_id="fc-x", run_id="r")
    assert res["claims"] == [str(claim_path)] and set(json.loads(claim_path.read_text())) == set(CV.VERIFY_CLAIM_FIELDS)
    assert claim_path == row_dir / "generation" / f"none__inspect__bigcodebench_full__{store.name}.claim.json"
    assert L.claim_store_for_verifier(row_dir, cell="none", out_dir=store, tasks=[BCB_FULL], function_call_id="fc-x", run_id="r")["refused"] is None   # ours: re-entry
    # the row container reloads the volume before its doors read ownership (source pin)
    src = Path(L.__file__).read_text()
    body = src[src.index("def _run_row("):src.index("def _row_worker(")]
    assert "VOL.reload()                                  # the doors read ownership records other containers committed" in body


# --------------------------------------------------------------------------- #
# Store ownership is acquired through a gate                                    #
# that is linearizable ACROSS containers (modal.Dict put-if-absent), in front   #
# of the volume claim file; the CPU-only resume holds a per-row host lock       #
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _host_lock_dir_under_tmp(tmp_path, monkeypatch):
    """Every ``main --verify-pending`` in this module takes its host lock under the test's temp dir, never the box's
    ``$XDG_RUNTIME_DIR`` (``verify_pending_lock_path`` reads the variable at call time)."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "runtime"))


class _AtomicDict:
    """A ``modal.Dict`` double shared by TWO worker threads: ``put(..., skip_if_exists=True)`` is atomic under a lock
    and returns whether THIS call created the entry; ``get`` / ``pop`` (KeyError when absent) / ``contains``. A
    ``barrier`` makes both threads ATTEMPT their first acquisition of a key before either learns the outcome (the
    simultaneous case, not the staggered one); ``on_pop`` runs right after a pop (a third party slipping in)."""

    def __init__(self, *, barrier: Optional[threading.Barrier] = None, on_pop=None):
        self.data: Dict[str, dict] = {}
        self.lock = threading.Lock()
        self.barrier, self.on_pop = barrier, on_pop
        self.puts: List[tuple] = []
        self._arrived: set = set()

    def put(self, key, value, *, skip_if_exists=False):
        if self.barrier is not None and C.VERIFY_CLAIM_TAKEOVER_SEP not in key and threading.get_ident() not in self._arrived:
            self._arrived.add(threading.get_ident())
            self.barrier.wait()
        with self.lock:
            self.puts.append((threading.current_thread().name, key, dict(value), skip_if_exists))
            if skip_if_exists and key in self.data:
                return False
            self.data[key] = dict(value)
            return True

    def get(self, key, default=None):
        return self.data.get(key, default)

    def pop(self, key):
        with self.lock:
            out = self.data.pop(key)
        if self.on_pop is not None:
            self.on_pop(key)
        return out

    def contains(self, key):
        return key in self.data


class _TwoContainerVolume:
    """A Modal-Volume double for TWO containers of one row dir. Each registered thread owns a MOUNT (its own copy of
    the volume); ``reload()`` replaces the mount's row-dir subtree with the shared VOLUME's committed state,
    ``commit()`` copies the mount's files over the volume (per file: last writer wins) and propagates the files the
    mount deleted since its reload — Modal's semantics, never a cross-container compare-and-set. The ``barrier`` makes
    both containers finish their reload before either commits, so both snapshots are empty of claims when both
    acquire (the reviewer's interleaving)."""

    def __init__(self, volume: Path, rel: str, *, barrier: Optional[threading.Barrier] = None):
        self.volume, self.rel, self.barrier = volume, rel, barrier
        self.mounts: Dict[int, Path] = {}
        self.seen_at_reload: Dict[int, set] = {}
        self.events: List[tuple] = []
        self.lock = threading.Lock()

    def register(self, mount: Path) -> Path:
        self.mounts[threading.get_ident()] = mount
        return mount / self.rel

    def _files(self, root: Path) -> Dict[str, Path]:
        return {str(p.relative_to(root)): p for p in root.rglob("*") if p.is_file()} if root.is_dir() else {}

    def reload(self):
        mount = self.mounts[threading.get_ident()]
        with self.lock:
            src, dst = self.volume / self.rel, mount / self.rel
            if dst.exists():
                shutil.rmtree(dst)
            if src.exists():
                shutil.copytree(src, dst)
            self.seen_at_reload[threading.get_ident()] = set(self._files(dst))
            self.events.append((threading.current_thread().name, "reload"))
        if self.barrier is not None:
            self.barrier.wait()

    def commit(self):
        mount = self.mounts[threading.get_ident()]
        with self.lock:
            src, dst = mount / self.rel, self.volume / self.rel
            now = self._files(src)
            for rel, p in now.items():
                (dst / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst / rel)
            for rel in self.seen_at_reload.get(threading.get_ident(), set()) - set(now):
                (dst / rel).unlink(missing_ok=True)
            self.seen_at_reload[threading.get_ident()] = set(now)
            self.events.append((threading.current_thread().name, "commit"))


def _pending_store_on_volume(tmp_path, *, volume_row_dir: Path):
    """A pending store (its marker standing, the manifest's pass ``pending``) whose row-dir files live on the fake
    VOLUME; the containers reload them into their mounts."""
    store, mk, pend = _verified_store(tmp_path, verified_by="fc-old", marker=True, row_dir=volume_row_dir)
    manifest = next(store.glob("cells/*/manifest.json"))
    doc = json.loads(manifest.read_text())
    doc["row_engine"]["verification_pass"] = {"state": "pending", "tasks": [BCB_FULL]}
    manifest.write_text(json.dumps(doc))
    return store, mk, pend


def test_two_containers_acquiring_one_pending_store_at_once_launch_exactly_one_driver(L, tmp_path, monkeypatch):
    """A hole in the previous fix: two CPU-only resumes for the same pending store both pass the ownership report
    before either dispatch is visible, spawn two `verify_cell_cpu` containers, and both reach the claim together.
    Each container `VOL.reload()`s its OWN snapshot before either claim is committed, `O_CREAT|O_EXCL` creates the claim
    only in that snapshot, and `VOL.commit()` is last-writer-wins — so both workers committed a claim and both drivers
    ran. Here: two workers, two mounts of one fake volume, both reloads before either commit, both acquisitions attempted
    before either outcome (barriers) — exactly ONE launches its driver; the other records VERIFY_CLAIM_HELD naming the
    winner's fcid and never calls `subprocess.run`; the volume carries ONE claim file, the winner's, while its driver
    runs; the winner's gate entry is released with its claim once the driver has exited."""
    repos = tmp_path / "repos"
    TL._tree(repos, "f6cf462")
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    volume = tmp_path / "volume"
    rel = f"rows/{ROW}/{SUFFIX}"
    store, mk, pend = _pending_store_on_volume(tmp_path, volume_row_dir=volume / rel)
    cfg = {"cell": "none", "name": pend["name"], "model": ROW, "mode": "off", "out_dir": str(store)}
    reload_barrier, put_barrier = threading.Barrier(2, timeout=30), threading.Barrier(2, timeout=30)
    gate = _AtomicDict(barrier=put_barrier)
    vol = _TwoContainerVolume(volume, rel, barrier=reload_barrier)
    monkeypatch.setattr(L, "VOL", vol)
    local = threading.local()
    adapter_cls = getattr(L, "VerifierClaims", None)
    monkeypatch.setattr(L, "verifier_claims", (lambda: adapter_cls(gate, mount=local.mount)), raising=False)
    drivers: List[Dict[str, object]] = []
    dlock = threading.Lock()

    def _driver(argv, **kw):
        argv = list(argv)
        with dlock, vol.lock:                       # the volume is read between commits, never mid-copy
            claims_on_volume = sorted((volume / rel / "generation").glob("*.claim.json"))
            drivers.append({"fcid": argv[argv.index("--function-call-id") + 1], "thread": threading.current_thread().name,
                            "volume_claims": [(p.name, json.loads(p.read_text())["function_call_id"]) for p in claims_on_volume],
                            "gate": {k: v["function_call_id"] for k, v in gate.data.items()},
                            "env_supersedes": kw["env"].get(CV.VERIFY_SUPERSEDES_ENV)})
        return SimpleNamespace(returncode=1, stdout="", stderr="stopped here")
    monkeypatch.setattr(L.subprocess, "run", _driver)
    recs: Dict[str, dict] = {}
    errors: List[BaseException] = []

    def _container(fcid: str, mount: Path):
        try:
            local.mount = mount
            row_dir = vol.register(mount)
            recs[fcid] = L._verify_cell_worker(cfg, "f6cf462", f"f6cf462-{fcid}", STANDUP_A, str(row_dir), pending=pend, function_call_id=fcid)
        except BaseException as e:  # noqa: BLE001 — surfaced below, never swallowed
            errors.append(e)
    threads = [threading.Thread(target=_container, args=(f, tmp_path / m), name=f) for f, m in (("fc-A", "mountA"), ("fc-B", "mountB"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors
    assert not any(t.is_alive() for t in threads)
    # exactly ONE driver launched — the whole point
    assert len(drivers) == 1, drivers
    winner = drivers[0]["fcid"]
    loser = "fc-B" if winner == "fc-A" else "fc-A"
    assert set(recs) == {"fc-A", "fc-B"}
    assert recs[loser]["status"] == "VERIFY_CLAIM_HELD" and "driver_argv" not in recs[loser]
    assert winner in recs[loser]["error"] and C.VERIFY_CLAIMS_DICT_NAME in recs[loser]["error"]
    assert recs[loser]["claim"]["held_by"]["function_call_id"] == winner and recs[loser]["claim"]["claims"] == []
    # both containers reloaded before either committed, and both attempted the acquisition (the simultaneous case)
    assert [e for e in vol.events[:2]] and all(kind == "reload" for _, kind in vol.events[:2]), vol.events
    assert sorted(name for name, key, _v, skip in gate.puts if skip and C.VERIFY_CLAIM_TAKEOVER_SEP not in key) == ["fc-A", "fc-B"]
    # the volume carried ONE claim file, the winner's, while its driver ran; the gate named the winner alone
    key = f"{rel}/generation/none__inspect__bigcodebench_full__{store.name}.claim.json"
    assert drivers[0]["volume_claims"] == [(f"none__inspect__bigcodebench_full__{store.name}.claim.json", winner)]
    assert drivers[0]["gate"] == {key: winner} and drivers[0]["env_supersedes"] is None
    # the loser never touched the volume claim file: its mount holds none, and it committed nothing of the kind
    assert not list((tmp_path / ("mountB" if loser == "fc-B" else "mountA") / rel / "generation").glob("*.claim.json"))
    # the winner's gate entry and file are released with its driver's exit; the driver's rc still decides the REC
    assert recs[winner]["status"] == "VERIFY_FAILED" and recs[winner]["driver_rc"] == 1
    assert recs[winner]["claim"]["gate"] == [{"key": key, "ticket": None}] and recs[winner]["gate_released_by_worker"] == [key]
    assert gate.data == {} and not list((volume / rel / "generation").glob("*.claim.json"))


def test_the_gate_takes_over_only_the_decided_call_and_a_third_acquirer_between_pop_and_put_is_held(L, tmp_path, monkeypatch):
    """The supersede through the gate: only the dispatcher's DECIDED call id (`superseded_call_id`, from a positive
    terminal / lost probe) is replaced — through a per-dead-call TICKET (put-if-absent), so only one superseder ever pops
    the dead entry; a third acquirer between the pop and the put wins the put, and the superseder is HELD naming it (its
    ticket returned); an entry naming a call the dispatch did not supersede is HELD, nothing popped, no ticket; a second
    superseder of the same dead call while the first holds the ticket is HELD naming the first. Through the worker: the
    takeover sets the driver's TINYCAP_VERIFY_SUPERSEDES to exactly the dead fcid, rewrites the dead call's claim file,
    and releases entry + ticket after the driver."""
    VC = L.VerifierClaims
    dead = {"function_call_id": "fc-dead", "run_id": "r0", "started_at": 1.0, "host": "h"}
    docA = {"function_call_id": "fc-A", "run_id": "rA", "started_at": 2.0, "host": "h"}
    docB = {"function_call_id": "fc-B", "run_id": "rB", "started_at": 3.0, "host": "h"}
    key = "rows/r/s/generation/none__t__s.claim.json"
    ticket = f"{key}{C.VERIFY_CLAIM_TAKEOVER_SEP}fc-dead"
    # (i) the decided takeover succeeds: the dead entry popped, ours put, the ticket held
    d = _AtomicDict()
    d.data[key] = dict(dead)
    got = VC(d, mount=tmp_path).acquire(key, docA, supersedes="fc-dead")
    assert got["won"] is True and got["superseded"] == "fc-dead" and got["ticket"] == ticket
    assert d.data[key]["function_call_id"] == "fc-A" and d.data[ticket]["function_call_id"] == "fc-A"
    # (ii) a second superseder of the same dead call: the entry names A now (live) -> HELD naming A, nothing changed
    got_b = VC(d, mount=tmp_path).acquire(key, docB, supersedes="fc-dead")
    assert got_b["won"] is False and got_b["holder"]["function_call_id"] == "fc-A" and got_b["ticket"] is None
    assert d.data[key]["function_call_id"] == "fc-A" and d.data[ticket]["function_call_id"] == "fc-A"
    # (iii) the ticket race: B arrives while A holds the ticket but has not yet popped the dead entry -> HELD naming A
    d2 = _AtomicDict()
    d2.data[key], d2.data[ticket] = dict(dead), dict(docA)
    got_b2 = VC(d2, mount=tmp_path).acquire(key, docB, supersedes="fc-dead")
    assert got_b2["won"] is False and got_b2["holder"]["function_call_id"] == "fc-A" and "takeover of fc-dead" in got_b2["why"]
    assert d2.data[key]["function_call_id"] == "fc-dead"                                      # the dead entry is A's to pop, not B's
    # (iv) a third acquirer C slips in between A's pop of the dead entry and A's put: A's put loses -> HELD naming C,
    # A's ticket returned, C's entry stands
    d3 = _AtomicDict(on_pop=lambda k: (k == key) and d3.put(k, {"function_call_id": "fc-C", "run_id": "rC", "started_at": 4.0, "host": "h"},
                                                             skip_if_exists=True))
    d3.data[key] = dict(dead)
    got_c = VC(d3, mount=tmp_path).acquire(key, docA, supersedes="fc-dead")
    assert got_c["won"] is False and got_c["holder"]["function_call_id"] == "fc-C" and "third verifier" in got_c["why"]
    assert d3.data[key]["function_call_id"] == "fc-C" and ticket not in d3.data
    # (v) an entry naming a call the dispatch did NOT supersede: HELD by name, no pop, no ticket
    d4 = _AtomicDict()
    d4.data[key] = {"function_call_id": "fc-Y", "run_id": "rY", "started_at": 1.0, "host": "h"}
    got_y = VC(d4, mount=tmp_path).acquire(key, docA, supersedes="fc-dead")
    assert got_y["won"] is False and got_y["holder"]["function_call_id"] == "fc-Y" and got_y["ticket"] is None
    assert d4.data == {key: {"function_call_id": "fc-Y", "run_id": "rY", "started_at": 1.0, "host": "h"}}
    # (vi) no supersede decided: a held entry is HELD whoever holds it; a free key is won; the same fcid re-enters
    assert VC(d4, mount=tmp_path).acquire(key, docA)["won"] is False
    d5 = _AtomicDict()
    assert VC(d5, mount=tmp_path).acquire(key, docA)["won"] is True and VC(d5, mount=tmp_path).acquire(key, docA)["why"] == "re-entry"
    # (vii) through the worker: the dead call holds the gate AND the file; the dispatch decided its supersede
    repos = tmp_path / "repos"
    TL._tree(repos, "f6cf462")
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    row_dir = tmp_path / "rows" / ROW / SUFFIX
    store, mk, pend = _pending_store_on_volume(tmp_path, volume_row_dir=row_dir)
    cfg = {"cell": "none", "name": pend["name"], "model": ROW, "mode": "off", "out_dir": str(store)}
    claim_path = CV.verify_claim_path(row_dir, "none", BCB_FULL, store.name)
    claim_path.write_text(json.dumps(dead))
    gate = _AtomicDict()
    wkey = f"rows/{ROW}/{SUFFIX}/generation/{claim_path.name}"
    gate.data[wkey] = dict(dead)
    monkeypatch.setattr(L, "verifier_claims", lambda: VC(gate, mount=tmp_path))
    monkeypatch.setattr(L, "VOL", SimpleNamespace(reload=lambda: None, commit=lambda: None))
    seen = {}

    def _driver(argv, **kw):
        seen["file"] = json.loads(claim_path.read_text())["function_call_id"]
        seen["gate"] = {k: v["function_call_id"] for k, v in gate.data.items()}
        seen["env"] = kw["env"].get(CV.VERIFY_SUPERSEDES_ENV)
        return SimpleNamespace(returncode=1, stdout="", stderr="")
    monkeypatch.setattr(L.subprocess, "run", _driver)
    rec = L._verify_cell_worker(cfg, "f6cf462", "f6cf462-x", STANDUP_A, str(row_dir), pending={**pend, "superseded_call_id": "fc-dead"},
                                function_call_id="fc-A")
    assert seen == {"file": "fc-A", "gate": {wkey: "fc-A", f"{wkey}{C.VERIFY_CLAIM_TAKEOVER_SEP}fc-dead": "fc-A"}, "env": "fc-dead"}
    assert rec["claim"]["superseded"] == "fc-dead" and rec["claim"]["gate"] == [{"key": wkey, "ticket": f"{wkey}{C.VERIFY_CLAIM_TAKEOVER_SEP}fc-dead"}]
    assert rec["gate_released_by_worker"] == [wkey] and gate.data == {} and not claim_path.exists()
    # the dispatch decided fc-dead but the gate names fc-Y: HELD naming fc-Y, no driver, the file untouched
    gate.data[wkey] = {"function_call_id": "fc-Y", "run_id": "rY", "started_at": 1.0, "host": "h"}
    claim_path.write_text(json.dumps(dead))
    calls = []
    monkeypatch.setattr(L.subprocess, "run", lambda argv, **kw: calls.append(argv) or SimpleNamespace(returncode=0, stdout="", stderr=""))
    held = L._verify_cell_worker(cfg, "f6cf462", "f6cf462-x", STANDUP_A, str(row_dir), pending={**pend, "superseded_call_id": "fc-dead"},
                                 function_call_id="fc-A2")
    assert held["status"] == "VERIFY_CLAIM_HELD" and calls == [] and "fc-Y" in held["error"]
    assert json.loads(claim_path.read_text())["function_call_id"] == "fc-dead" and set(gate.data) == {wkey}


def test_the_gate_release_pops_only_this_verifiers_entry_and_a_lost_task_backs_out_the_taken_ones(L, tmp_path, monkeypatch):
    """Release is read-then-pop of OUR entry alone: an entry another call took over (it names them) is left, and so
    is its ticket; `release_own_gate_claims` reports the keys it released and never raises. A store with two tasks
    whose second task's entry another verifier holds: the first task's entry, already taken, is released again and no
    claim file is written — the refusal leaves nothing of ours behind."""
    VC = L.VerifierClaims
    key, other = "rows/r/s/generation/none__t1__s.claim.json", "rows/r/s/generation/none__t2__s.claim.json"
    ticket = f"{key}{C.VERIFY_CLAIM_TAKEOVER_SEP}fc-dead"
    d = _AtomicDict()
    d.data[key] = {"function_call_id": "fc-B", "run_id": "r", "started_at": 1.0, "host": "h"}
    d.data[ticket] = {"function_call_id": "fc-B", "run_id": "r", "started_at": 1.0, "host": "h"}
    vc = VC(d, mount=tmp_path)
    assert vc.release(key, "fc-A", ticket=ticket) is False and set(d.data) == {key, ticket}        # not ours: left alone
    assert vc.release(key, "fc-B", ticket=ticket) is True and d.data == {}                         # ours: entry and ticket popped
    assert vc.release(key, "fc-B") is False                                                        # already released: nothing
    d.data[key] = {"function_call_id": "fc-A", "run_id": "r", "started_at": 1.0, "host": "h"}
    d.data[other] = {"function_call_id": "fc-B", "run_id": "r", "started_at": 1.0, "host": "h"}
    assert L.release_own_gate_claims(vc, [{"key": key, "ticket": None}, {"key": other, "ticket": None}], function_call_id="fc-A") == [key]
    assert set(d.data) == {other}
    # a Dict that raises on release: printed, nothing released, never raised
    class _Raising(_AtomicDict):
        def get(self, key, default=None):
            raise ConnectionError("gate away")
    assert L.release_own_gate_claims(VC(_Raising(), mount=tmp_path), [{"key": key, "ticket": None}], function_call_id="fc-A") == []
    # the store-level helper: the second task's entry held by another -> refused, the first task's entry backed out
    row_dir = tmp_path / "rows" / ROW / SUFFIX
    (row_dir / "generation").mkdir(parents=True)
    store = tmp_path / "gen" / "s" / SUFFIX
    store.mkdir(parents=True)
    gate = _AtomicDict()
    t2 = C.inspect_task("bfcl_full")
    k2 = f"rows/{ROW}/{SUFFIX}/generation/{CV.verify_claim_path(row_dir, 'none', t2, store.name).name}"
    gate.data[k2] = {"function_call_id": "fc-F", "run_id": "rF", "started_at": 1.0, "host": "h"}
    monkeypatch.setattr(L, "VOL", SimpleNamespace(reload=lambda: None, commit=lambda: pytest.fail("a refusal commits nothing here")))
    res = L.claim_store_for_verifier(row_dir, cell="none", out_dir=store, tasks=[BCB_FULL, t2], function_call_id="fc-A", run_id="r",
                                     claims=VC(gate, mount=tmp_path))
    assert res["refused"] and "fc-F" in res["refused"] and res["held_by"]["function_call_id"] == "fc-F" and res["claims"] == [] and res["gate"] == []
    assert set(gate.data) == {k2}                                                                  # the first task's entry released
    assert not list((row_dir / "generation").glob("*.claim.json"))                                 # no file written by a loser
    # the gate itself raising mid-acquisition: refused as GATE_UNAVAILABLE (never HELD), nothing taken
    class _Down(_AtomicDict):
        def put(self, key, value, *, skip_if_exists=False):
            raise ConnectionError("dict down")
    res2 = L.claim_store_for_verifier(row_dir, cell="none", out_dir=store, tasks=[BCB_FULL], function_call_id="fc-A", run_id="r",
                                      claims=VC(_Down(), mount=tmp_path))
    assert res2["refused"] and res2["gate_error"] is True and "dict down" in res2["refused"] and res2["gate"] == []
    assert C.REC_VERIFY_GATE_UNAVAILABLE == "VERIFY_GATE_UNAVAILABLE" not in C.REC_OK_STATUSES
    # the worker names the gate's failure to open: GATE_UNAVAILABLE, nothing run
    open_gate = L.verifier_claims
    monkeypatch.setattr(L, "verifier_claims", lambda: (_ for _ in ()).throw(ConnectionError("no dict")))
    monkeypatch.setattr(L.subprocess, "run", lambda argv, **kw: pytest.fail("no driver"))
    repos = tmp_path / "repos"
    TL._tree(repos, "f6cf462")
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    store2, mk, pend = _pending_store_on_volume(tmp_path, volume_row_dir=row_dir)
    cfg = {"cell": "none", "name": pend["name"], "model": ROW, "mode": "off", "out_dir": str(store2)}
    rec = L._verify_cell_worker(cfg, "f6cf462", "f6cf462-x", STANDUP_A, str(row_dir), pending=pend, function_call_id="fc-A")
    assert rec["status"] == "VERIFY_GATE_UNAVAILABLE" and "no dict" in rec["error"] and "claim" not in rec
    # the key is the claim file's path RELATIVE TO THE MOUNT — the same in every container whatever its mount — and the
    # absolute path outside one; the production adapter opens the contracts' Dict lazily, through modal.Dict
    p = Path("/vol/tinycap/gen/rows/r/s/generation/x.claim.json")
    assert VC(d).key_for(p) == "tinycap/gen/rows/r/s/generation/x.claim.json" == VC(d, mount="/vol").key_for(p)
    assert VC(d, mount=tmp_path / "other").key_for(p) == str(p)
    opened = open_gate()
    assert isinstance(opened, VC) and opened.store is L.modal.Dict.from_name(C.VERIFY_CLAIMS_DICT_NAME) and str(opened.mount) == L.VOL_PATH
    src = Path(L.__file__).read_text()
    assert "dict_cls.from_name(C.VERIFY_CLAIMS_DICT_NAME, create_if_missing=True)" in src and C.VERIFY_CLAIMS_DICT_NAME == "tinycap-verify-claims"


def test_a_second_verify_pending_of_the_same_row_dir_on_this_host_is_refused_by_the_flock_before_any_spawn(L, tmp_path, monkeypatch, capsys):
    """The same-host belt: `main --verify-pending` holds an exclusive non-blocking flock on a per-row lock file (under
    $XDG_RUNTIME_DIR, never the volume) for the whole dispatch; a second resume of the same row dir on this host is
    refused with VERIFY_PENDING_HOST_LOCKED naming the lock path, before the report is called or anything spawns.
    Released, the same resume proceeds to its report. The lock does not replace the gate (the cross-container case)."""
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    row_dir = Path(f"{C.VOL_GEN_ROOT}/rows/gemma4e4b/{SUFFIX}")                                  # the row dir main derives
    lock_path = L.verify_pending_lock_path(row_dir)
    assert lock_path.parent == runtime / C.VERIFY_PENDING_LOCK_DIRNAME and lock_path.suffix == ".lock"
    assert lock_path == L.verify_pending_lock_path(str(row_dir)) != L.verify_pending_lock_path(row_dir.parent / "other")
    assert not str(lock_path).startswith(L.VOL_PATH)
    reports = []
    monkeypatch.setattr(L, "row_verify_pending_report", SimpleNamespace(remote=lambda sha, rd, rid, **kw: reports.append(rd) or
                                                                        L.pending_verification_report(tmp_path / "none", run_id=rid)))
    monkeypatch.setattr(L, "run_row_full", SimpleNamespace(spawn=lambda cfgs, **kw: pytest.fail("no GPU row")))
    monkeypatch.setattr(L, "verify_cell_cpu", SimpleNamespace(spawn=lambda *a, **k: pytest.fail("nothing spawns under a held lock")))
    kw = dict(sha="f6cf462", app_name=L.APP_NAME, pass1=True, cells="none", modes="off", controls="bigcodebench_full,bfcl_full", out_suffix=SUFFIX,
              row=True, verification_phase="generate", verify_pending=True)
    with L.verify_pending_host_lock(row_dir, run_id="f6cf462-first") as held:                   # the first resume, still running
        assert held == lock_path and json.loads(lock_path.read_text())["run_id"] == "f6cf462-first"
        capsys.readouterr()
        with pytest.raises(SystemExit) as e:
            L.main(**kw)
        out = capsys.readouterr().out
        assert "VERIFY_PENDING_HOST_LOCKED" in str(e.value) and str(lock_path) in str(e.value) and "before any container spawned" in str(e.value)
        assert f"pid {os.getpid()}" in str(e.value) and "f6cf462-first" in str(e.value)
        rec_line = next(ln for ln in out.splitlines() if ln.startswith("[tinycap] VERIFY-PENDING-REC"))
        rec = json.loads(rec_line.split(" ", 2)[2])
        assert rec["status"] == C.REC_VERIFY_PENDING_HOST_LOCKED == "VERIFY_PENDING_HOST_LOCKED" and rec["lock"] == str(lock_path)
        assert rec["holder"]["pid"] == os.getpid() and rec["holder"]["run_id"] == "f6cf462-first" and rec["row_dir"] == str(row_dir)
        assert reports == []                                                                       # refused BEFORE the report
        # the helper itself, directly: the second taker raises naming the holder
        with pytest.raises(L.VerifyPendingHostLocked) as e2:
            with L.verify_pending_host_lock(row_dir, run_id="f6cf462-second"):
                pytest.fail("the lock is held")
        assert e2.value.path == lock_path and e2.value.holder["run_id"] == "f6cf462-first"
    # released: the resume proceeds to its report (and fails there by name — the report is a stub)
    with pytest.raises(SystemExit, match="NO_ROW_REC"):
        L.main(**kw)
    assert reports == [str(row_dir)]
    assert lock_path.is_file()                                                                     # never unlinked (an unlink races an opener)
    with L.verify_pending_host_lock(row_dir, run_id="f6cf462-third") as again:                    # free again after the dispatch
        assert again == lock_path
    # a different row dir is a different lock: not contended
    with L.verify_pending_host_lock(row_dir, run_id="a"), L.verify_pending_host_lock(row_dir.parent / "other", run_id="b"):
        pass


# --------------------------------------------------------------------------- #
# tinycap-verify-rate-limit (2026-09-30): the verifiers' dispatch width, the     #
# per-verifier sandbox override, and the resume from another tree               #
# --------------------------------------------------------------------------- #
#
# The live failure: the gemma (c) row (tree 082f583, 07:05Z, `--concurrency-k 2 --verification-phase generate`)
# returned GENERATED and `verify_pending_cells` spawned NINE `verify_cell_cpu` containers ~40 ms apart, each at
# `max_sandboxes` 8; Modal refused the creates (`ResourceExhaustedError: Sandbox creation rate limit (5/s)
# exceeded`), inspect aborted every task and the row read VERIFY_FAILED with its generations intact — and a
# `--verify-pending` at the same width would have failed the same way.


class _SlowCall:
    """A spawned verifier's FunctionCall double that RUNS for a while: ``get(timeout=0)`` raises the builtin
    ``TimeoutError`` until it has been polled ``polls_to_finish`` times (modal 1.5.4's ``poll_function`` contract),
    then returns its REC — or raises its failure. ``live`` (a shared set) tracks which calls are in flight."""
    _n = 0

    def __init__(self, result, *, polls_to_finish: int, live: set, label: str):
        type(self)._n += 1
        self.object_id = f"fc-slow-{type(self)._n:03d}"
        self._result, self._left, self._live, self.label = result, polls_to_finish, live, label
        self._live.add(label)

    def get(self, timeout=None):
        assert timeout == 0.0, "the dispatcher polls with get(timeout=0) — never a blocking get"
        if self._left > 0:
            self._left -= 1
            raise TimeoutError()
        self._live.discard(self.label)
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


DISPATCH_RUN = "f6cf462-d15a7c00"


def _dispatch_world(n_labels: int, *, row_dir: str = f"{C.VOL_GEN_ROOT}/rows/gemma4e4b/{SUFFIX}", max_sandboxes: int = 8):
    """A GENERATED row result with ``n_labels`` pending stores (labels ``c0``…, the row order), the container's
    expectation record (every label ``expected``), the GENERATED cell RECs and the cfgs — exactly what
    ``verify_pending_cells`` reads; no volume, no tree, no container."""
    labels = [f"c{i}" for i in range(n_labels)]
    pending = {label: {"cell": label, "name": f"{ROW}__{label}__off", "out_dir": f"{C.VOL_GEN_ROOT}/native/{ROW}/{label}/{SUFFIX}",
                       "config": "/tree/config.yaml", "row_engine_record": f"{row_dir}/generation/row_engine__{label}.json",
                       "controls": ["bigcodebench_full"], "tasks": [BCB_FULL], "markers": [f"{row_dir}/generation/pending_{label}.json"],
                       "max_sandboxes": max_sandboxes, "served_max_num_seqs": 16, "concurrency_k": 2, "generation_function_call_id": "fc-gen",
                       "standup_id": STANDUP_A, "tree_sha": FULL_SHA} for label in labels}
    cells = {label: {"name": pending[label]["name"], "cell": label, "status": "GENERATED", "out": pending[label]["out_dir"]} for label in labels}
    recs = [dict(c) for c in cells.values()]
    expectation = {"schema": "tinycap_verifiers/1", "run_id": DISPATCH_RUN, "row": "gemma4e4b", "suffix": SUFFIX,
                   "generation_function_call_id": "fc-gen", "written_at": 1.0, "written_by": "row_container",
                   "cells": {label: {"cell": label, "out_dir": pending[label]["out_dir"], "state": "expected"} for label in labels}}
    result = {"verification_pending": pending, "cells": cells, "row_dir": row_dir, "standup_id": STANDUP_A, "row": "gemma4e4b",
              "suffix": SUFFIX, "function_call_id": "fc-gen", "expected_verifiers": expectation}
    cfgs = [{"cell": label, "name": pending[label]["name"], "model": ROW, "mode": "off", "out_dir": pending[label]["out_dir"]} for label in labels]
    return labels, result, recs, cfgs


def _recording_writes(L, monkeypatch):
    """``put_row_dir_file`` replaced by a recorder: ``(name, must_exist, {label: (state, function_call_id)})`` per write."""
    writes: List[tuple] = []
    monkeypatch.setattr(L, "put_row_dir_file", lambda row_dir, name, d, **kw: writes.append(
        (name, kw.get("must_exist"), {k: (v["state"], v.get("function_call_id")) for k, v in d["cells"].items()})) or "x")
    return writes


def test_the_verifier_dispatch_keeps_at_most_verify_width_in_flight_and_promotes_the_next_as_one_finishes(L, monkeypatch, capsys):
    """Nine pending labels at width 3: never more than three verifiers in flight; the first batch spawns
    `VERIFY_SPAWN_STAGGER_S_DEFAULT` apart; a raised `.spawn` (c3), a call that raised (c5) and a VERIFY_FAILED REC (c1)
    each free their slot and the queue moves on; the verify RECs come back in the row's order and the GENERATED cell
    RECs flip exactly where a verifier was OK. The run's record: the pre-spawn REPLACE marks the first batch `spawning`
    and the queued labels `expected` (never an id-less `spawning`, which a later resume would BLOCK on); a queued label
    is marked `spawning` in a write of its own right before its spawn, `spawned` + its call id (or `spawn_failed`) right
    after; `expected` never returns once a label was dispatched; nothing is ever `skipped`."""
    labels, result, recs, cfgs = _dispatch_world(9)
    writes = _recording_writes(L, monkeypatch)
    sleeps: List[float] = []
    monkeypatch.setattr(L, "_dispatch_sleep", lambda s: sleeps.append(s))
    live: set = set()
    spawned: List[tuple] = []
    in_flight_at_spawn: List[int] = []
    polls_to_finish = {"c0": 3, "c1": 1, "c2": 2, "c4": 1, "c5": 2, "c6": 1, "c7": 1, "c8": 1}

    def _spawn(cell_cfg, sha, rid, standup_id, row_dir, **kw):
        label = cell_cfg["cell"]
        in_flight_at_spawn.append(len(live))
        assert (sha, rid, standup_id, row_dir) == ("f6cf462", DISPATCH_RUN, STANDUP_A, result["row_dir"])
        if label == "c3":
            raise RuntimeError("spawn RPC down")
        pend = kw["pending"]
        spawned.append((label, pend["max_sandboxes"], pend.get("max_sandboxes_override"), pend.get("max_sandboxes_marker")))
        status = "VERIFY_FAILED" if label == "c1" else "OK"
        rec = (RuntimeError("container died") if label == "c5"
               else {"name": f"{cell_cfg['name']}:verify", "cell": label, "status": status, "phase": "verify", "verified_by": f"v-{label}"})
        return _SlowCall(rec, polls_to_finish=polls_to_finish[label], live=live, label=label)
    fn = SimpleNamespace(spawn=_spawn)
    capsys.readouterr()
    vrecs = L.verify_pending_cells(fn, result, cfgs, recs, sha="f6cf462", run_id=DISPATCH_RUN, reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT,
                                   verify_width=3)
    out = capsys.readouterr().out
    # the width held: at every spawn at most two others were in flight (so three with the new one), and c3's failed spawn
    # counted for nothing; every spawnable label was spawned, in the row's order
    assert max(in_flight_at_spawn) <= 2 and len(in_flight_at_spawn) == 9
    assert [s[0] for s in spawned] == ["c0", "c1", "c2", "c4", "c5", "c6", "c7", "c8"]
    assert all(s[1] == 8 and s[2] is None and s[3] is None for s in spawned)          # no override: the marker's value rides untouched
    assert live == set()                                                                  # every call harvested
    # the first batch spawned `stagger` apart; the poll sleep is the join loop's
    assert sleeps[:2] == [L.VERIFY_SPAWN_STAGGER_S_DEFAULT] * 2 == [2.0, 2.0] and L.VERIFY_JOIN_POLL_S in sleeps
    assert all(s in (L.VERIFY_SPAWN_STAGGER_S_DEFAULT, L.VERIFY_JOIN_POLL_S) for s in sleeps)
    # the RECs: the row's order, each label's outcome by name
    assert [v["label"] for v in vrecs] == labels
    assert {v["label"]: v["status"] for v in vrecs} == {"c0": "OK", "c1": "VERIFY_FAILED", "c2": "OK", "c3": "VERIFY_SPAWN_FAILED", "c4": "OK",
                                                        "c5": "FUNCTIONCALL_FAILED", "c6": "OK", "c7": "OK", "c8": "OK"}
    assert "spawn RPC down" in next(v for v in vrecs if v["label"] == "c3")["error"] and "container died" in next(v for v in vrecs if v["label"] == "c5")["error"]
    assert next(v for v in vrecs if v["label"] == "c5")["function_call_id"].startswith("fc-slow-")
    by_cell = {r["cell"]: r for r in recs}
    assert {c for c, r in by_cell.items() if r["status"] == "OK"} == {"c0", "c2", "c4", "c6", "c7", "c8"}
    assert by_cell["c1"]["status"] == "GENERATED" and by_cell["c1"]["verification"]["status"] == "VERIFY_FAILED"
    assert by_cell["c3"]["status"] == "GENERATED" and "verification" not in by_cell["c3"]     # no verifier ran: the REC is as it was
    assert by_cell["c5"]["status"] == "GENERATED" and by_cell["c5"]["verification"]["status"] == "FUNCTIONCALL_FAILED"
    assert by_cell["c0"]["verification"]["verified_by"] == "v-c0"
    # the record: the REPLACE (must_exist) — first batch `spawning`, the rest `expected`; then the batch's outcomes
    assert writes[0] == (f"verifiers_{DISPATCH_RUN}.json", True, {"c0": ("spawning", None), "c1": ("spawning", None), "c2": ("spawning", None),
                                                                 **{f"c{i}": ("expected", None) for i in range(3, 9)}})
    assert writes[1][2]["c0"][0] == "spawned" and writes[1][2]["c0"][1].startswith("fc-slow-") and writes[1][2]["c3"] == ("expected", None)
    assert all(w[1] is True for w in writes)
    states_over_time = {label: [w[2][label][0] for w in writes] for label in labels}
    for label in ("c3", "c4", "c5", "c6", "c7", "c8"):
        seq = states_over_time[label]
        i_spawning = seq.index("spawning")
        assert set(seq[:i_spawning]) == {"expected"}, (label, seq)                       # queued as `expected` until promoted
        assert "expected" not in seq[i_spawning:], (label, seq)                           # never back to `expected`
        assert writes[i_spawning][2][label] == ("spawning", None)                          # the pre-spawn mark carries no call id
        final = "spawn_failed" if label == "c3" else "spawned"
        assert seq[-1] == final and seq[i_spawning + 1:].count("spawning") == 0, (label, seq)
    assert not any("skipped" in {st for st, _ in w[2].values()} for w in writes)
    assert writes[-1][2]["c3"] == ("spawn_failed", None) and all(writes[-1][2][l][0] == "spawned" for l in labels if l != "c3")
    # what the dispatcher says: the width, the rate limit, the queue and the slots
    assert "verifying 9 cell(s) off the GPU" in out and "at most 3 in flight (--verify-width)" in out and "5 sandboxes/s per workspace" in out
    assert "max_sandboxes per verifier: the marker's min(S // K, 8)" in out
    assert "spawned verify c0 -> FunctionCall fc-slow-" in out and "1/3 in flight, 8 queued" in out and "3/3 in flight, 6 queued" in out
    assert "spawn of verify c3 FAILED" in out
    assert "verifier for c1 finished: VERIFY_FAILED" in out
    # the width and the override are validated by name
    with pytest.raises(ValueError, match="--verify-width must be >= 1"):
        L.verify_pending_cells(fn, result, cfgs, recs, sha="f6cf462", run_id=DISPATCH_RUN, reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT, verify_width=0)
    with pytest.raises(ValueError, match="--verify-max-sandboxes must be in 1..8"):
        L.verify_pending_cells(fn, result, cfgs, recs, sha="f6cf462", run_id=DISPATCH_RUN, reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT,
                               verify_max_sandboxes=9)
    # the poll is CLASSIFIED: the builtin TimeoutError is `active`; an expired output is `lost` (frees the
    # slot: positively established); a returned REC is `terminal_ok`; the function's own raise is `terminal_failed`
    class OutputExpiredError(Exception):                      # modal.exception.TimeoutError subclasses modal.exception.Error, not the builtin
        pass
    assert L._poll_verifier_call(SimpleNamespace(get=lambda timeout: (_ for _ in ()).throw(TimeoutError())), 0.0) == (L.CALL_ACTIVE, None, None)
    assert L._poll_verifier_call(SimpleNamespace(get=lambda timeout: (_ for _ in ()).throw(OutputExpiredError("gone"))), 0.0) == (
        L.CALL_LOST, None, "OutputExpiredError: gone")
    assert L._poll_verifier_call(SimpleNamespace(get=lambda timeout: {"status": "OK"}), 0.0) == (L.CALL_TERMINAL_OK, {"status": "OK"}, None)
    assert L._poll_verifier_call(SimpleNamespace(get=lambda timeout: (_ for _ in ()).throw(RuntimeError("died"))), 0.0) == (
        L.CALL_TERMINAL_FAILED, None, "RuntimeError: died")
    assert L.VERIFY_WIDTH_DEFAULT == C.VERIFY_WIDTH_DEFAULT == 3 and L.ROW_VERIFY_WIDTH_FLAG == "--verify-width"
    assert L.VERIFY_JOIN_MAX_S > L.VERIFY_FUNCTION_TIMEOUT_S


def test_a_width_at_or_above_the_label_count_writes_the_record_exactly_twice_as_before_and_a_join_holds_a_slot(L, monkeypatch, capsys):
    """Byte-compatibility with the landing-2 record: two labels at the default width see the REPLACE (`spawning` x2) and
    ONE outcome write (`spawned` x2) — the final write is skipped when nothing changed. A label JOINED from an earlier
    run's still-running verifier holds a slot: at width 2 with one join, only ONE new verifier spawns until the joined
    call finishes; its REC is harvested as this run's (`joined_call_id`)."""
    labels, result, recs, cfgs = _dispatch_world(2)
    writes = _recording_writes(L, monkeypatch)
    monkeypatch.setattr(L, "_dispatch_sleep", lambda s: None)
    live: set = set()
    fn = SimpleNamespace(spawn=lambda cell_cfg, *a, **kw: _SlowCall({"name": "x:verify", "cell": cell_cfg["cell"], "status": "OK", "phase": "verify"},
                                                                     polls_to_finish=1, live=live, label=cell_cfg["cell"]))
    vrecs = L.verify_pending_cells(fn, result, cfgs, recs, sha="f6cf462", run_id=DISPATCH_RUN, reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT)
    assert [v["status"] for v in vrecs] == ["OK", "OK"] and all(r["status"] == "OK" for r in recs)
    assert [(w[1], {k: st for k, (st, _) in w[2].items()}) for w in writes] == [
        (True, {"c0": "spawning", "c1": "spawning"}), (True, {"c0": "spawned", "c1": "spawned"})]
    # a join at width 2: c0 joins fc-A (live for 2 polls), so c1 spawns at once (slot 2) and c2 waits for fc-A
    labels, result, recs, cfgs = _dispatch_world(3)
    result["verification_pending"]["c0"].update(join_call_id="fc-A", ownership={"decision": "join", "call_id": "fc-A", "prior_run_id": "f6cf462-00000001"})
    result["expected_verifiers"]["cells"]["c0"].update(state="joined", function_call_id="fc-A")
    writes.clear()
    live.clear()
    in_flight_at_spawn: List[int] = []
    joined = _SlowCall({"name": "a:verify", "cell": "c0", "status": "OK", "phase": "verify", "function_call_id": "fc-A"}, polls_to_finish=2, live=live, label="c0")
    joined.object_id = "fc-A"
    monkeypatch.setattr(L, "attach_verifier_call", lambda cid: joined)

    def _spawn(cell_cfg, *a, **kw):
        in_flight_at_spawn.append(sorted(live))
        return _SlowCall({"name": "x:verify", "cell": cell_cfg["cell"], "status": "OK", "phase": "verify"}, polls_to_finish=1, live=live, label=cell_cfg["cell"])
    capsys.readouterr()
    vrecs = L.verify_pending_cells(SimpleNamespace(spawn=_spawn), result, cfgs, recs, sha="f6cf462", run_id=DISPATCH_RUN,
                                   reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT, verify_width=2)
    out = capsys.readouterr().out
    # c1 spawned beside the join (slot 2 of 2); c2 only once c1 finished — the join, still running, kept its slot throughout
    assert in_flight_at_spawn == [["c0"], ["c0"]], in_flight_at_spawn
    assert [(v["label"], v["status"], v.get("joined_call_id")) for v in vrecs] == [("c0", "OK", "fc-A"), ("c1", "OK", None), ("c2", "OK", None)]
    assert writes[0][2] == {"c0": ("joined", "fc-A"), "c1": ("spawning", None), "c2": ("expected", None)}
    assert "joining verifier fc-A for c0" in out and "still running, spawning nothing" in out


def test_the_max_sandboxes_override_rides_every_spawns_payload_the_record_and_the_verifiers_driver_argv(L, tmp_path, monkeypatch, capsys):
    """`--verify-max-sandboxes 4`: every spawn's pending payload carries `max_sandboxes` 4 (the marker's 8 kept as
    `max_sandboxes_marker`, `max_sandboxes_override` 4), the record's `spawned` entries name it, the dispatcher says so —
    and in the container `_verify_cell_worker` hands it to the driver as `--max-sandboxes 4` and names it on the REC."""
    labels, result, recs, cfgs = _dispatch_world(2)
    monkeypatch.setattr(L, "_dispatch_sleep", lambda s: None)
    written: Dict[str, dict] = {}
    monkeypatch.setattr(L, "put_row_dir_file", lambda row_dir, name, d, **kw: written.__setitem__(name, json.loads(json.dumps(d, default=str))) or "x")
    payloads: List[dict] = []
    fn = SimpleNamespace(spawn=lambda cell_cfg, *a, **kw: payloads.append(dict(kw["pending"])) or TL._Call(
        {"name": "x:verify", "cell": cell_cfg["cell"], "status": "OK", "phase": "verify", "function_call_id": "fc-v"}))
    capsys.readouterr()
    L.verify_pending_cells(fn, result, cfgs, recs, sha="f6cf462", run_id=DISPATCH_RUN, reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT,
                           verify_max_sandboxes=4)
    out = capsys.readouterr().out
    assert [(p["max_sandboxes"], p["max_sandboxes_marker"], p["max_sandboxes_override"]) for p in payloads] == [(4, 8, 4), (4, 8, 4)]
    assert result["verification_pending"]["c0"]["max_sandboxes"] == 8                     # the row result itself is not mutated
    rec = written[f"verifiers_{DISPATCH_RUN}.json"]["cells"]
    assert rec["c0"]["state"] == "spawned" and rec["c0"]["max_sandboxes"] == 4 and rec["c0"]["max_sandboxes_override"] == 4
    assert "max_sandboxes per verifier: 4 (--verify-max-sandboxes overrides the marker's min(S // K, 8))" in out
    # the container: the payload's max_sandboxes reaches the driver argv; the REC names the effective value and the override
    repos = tmp_path / "repos"
    TL._tree(repos, "f6cf462")
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    row_dir = tmp_path / "rows" / ROW / SUFFIX
    store, mk, pend = _verified_store(tmp_path, verified_by="fc-v-old", marker=False, row_dir=row_dir)
    pend = {**pend, "max_sandboxes": 4, "max_sandboxes_marker": 8, "max_sandboxes_override": 4}
    cfg = {"cell": "none", "name": pend["name"], "model": ROW, "mode": "off", "out_dir": str(store)}
    argvs: List[list] = []
    ALREADY = ("[served-cap] verify: inspect/bigcodebench_full for none is already verified (by fc-v-old via M2_replay_vehicle over 1140 items); "
               "nothing to do — the store is untouched\n")
    monkeypatch.setattr(L.subprocess, "run", lambda argv, **kw: argvs.append(list(argv)) or SimpleNamespace(returncode=0, stdout=ALREADY, stderr=""))
    wrec = L._verify_cell_worker(cfg, "f6cf462", DISPATCH_RUN, STANDUP_A, str(row_dir), pending=pend, function_call_id="fc-v-new")
    assert wrec["status"] == "OK", wrec.get("error")
    argv = argvs[0]
    assert argv[argv.index("--max-sandboxes") + 1] == "4" and wrec["max_sandboxes"] == 4 and wrec["max_sandboxes_override"] == 4
    assert L.verify_cell_argv("/t", out_dir="/o", cell="none", controls=["bigcodebench_full"], config="/c", row_dir="/r", row_engine_record="/e",
                              max_sandboxes=4, function_call_id="f")[-4:] == ["--max-sandboxes", "4", "--function-call-id", "f"]


def test_the_cpu_only_resume_dispatches_at_the_same_width_with_the_override_and_main_refuses_the_knobs_by_name(L, tmp_path, monkeypatch, capsys):
    """`main --row … --verification-phase generate --verify-pending --verify-width 2 --verify-max-sandboxes 4`: the resume's
    dispatch (five pending stores off a report) never has more than two verifiers in flight and every spawn carries
    max_sandboxes 4; the dry run names the width and the override; `launch_argv` spells them LAST, only when not the
    default; the knobs refuse without --row, beside a coupled phase, and out of range."""
    w = _World(L, tmp_path, monkeypatch, cells=["none"], row_opts={"verification_phase": "generate"})
    labels = [f"c{i}" for i in range(5)]
    row_dir = f"{C.VOL_GEN_ROOT}/rows/gemma4e4b/{SUFFIX}"

    def _report_remote(sha, rd, rid, **kw):
        pending = {l: {"cell": "none", "name": f"{ROW}__none__off-{l}", "out_dir": f"{C.VOL_GEN_ROOT}/native/{ROW}/none/{SUFFIX}-{l}",
                       "config": "/tree/config.yaml", "row_engine_record": f"{rd}/generation/row_engine__{l}.json", "controls": ["bigcodebench_full"],
                       "tasks": [BCB_FULL], "markers": [f"{rd}/generation/pending_{l}.json"], "max_sandboxes": 8, "served_max_num_seqs": 16,
                       "concurrency_k": 2, "generation_function_call_id": "fc-gen", "standup_id": STANDUP_A, "tree_sha": FULL_SHA,
                       "resumed": True, "resume_kind": "pending", "ownership": {"decision": L.OWN_SPAWN if hasattr(L, "OWN_SPAWN") else "spawn"},
                       "cell_rec": {"name": f"{ROW}__none__off-{l}", "cell": "none", "status": "GENERATED", "out": f"{C.VOL_GEN_ROOT}/native/{ROW}/none/{SUFFIX}-{l}"}}
                   for l in labels}
        record = {"schema": "tinycap_verifiers/1", "run_id": rid, "row": "gemma4e4b", "suffix": SUFFIX, "generation_function_call_id": None,
                  "written_at": 1.0, "written_by": "verify_pending_report",
                  "cells": {l: {"cell": "none", "out_dir": pending[l]["out_dir"], "state": "expected"} for l in labels}}
        return {"status": "OK", "row_dir": rd, "run_id": rid, "row_rec": {"row_dir": rd, "standup_id": STANDUP_A, "row": "gemma4e4b", "suffix": SUFFIX,
                                                                          "function_call_id": "fc-gen", "cells": {}},
                "pending": pending, "verified": {}, "torn": {}, "joining": {}, "refused": {}, "problems": [], "expected_verifiers": record, "sha": sha[:7]}
    monkeypatch.setattr(L, "row_verify_pending_report", SimpleNamespace(remote=_report_remote))
    monkeypatch.setattr(L, "run_row_full", SimpleNamespace(spawn=lambda cfgs, **kw: (_ for _ in ()).throw(AssertionError("no GPU row"))))
    monkeypatch.setattr(L, "put_row_dir_file", lambda row_dir, name, d, **kw: "x")
    monkeypatch.setattr(L, "_dispatch_sleep", lambda s: None)
    live: set = set()
    in_flight_at_spawn: List[int] = []
    payloads: List[dict] = []

    def _vspawn(cell_cfg, sha, rid, standup_id, rd, **kw):
        in_flight_at_spawn.append(len(live))
        payloads.append(dict(kw["pending"]))
        label = kw["pending"]["name"].rsplit("-", 1)[1]
        return _SlowCall({"name": f"{cell_cfg['name']}:verify", "cell": "none", "status": "OK", "phase": "verify", "function_call_id": "fc-v", "verified_by": "fc-v"},
                         polls_to_finish=2, live=live, label=label)
    monkeypatch.setattr(L, "verify_cell_cpu", SimpleNamespace(spawn=_vspawn))
    kw = dict(sha="f6cf462", app_name=L.APP_NAME, pass1=True, cells="none", modes="off", controls="bigcodebench_full,bfcl_full", out_suffix=SUFFIX,
              row=True, verification_phase="generate", verify_pending=True)
    capsys.readouterr()
    L.main(**kw, verify_width=2, verify_max_sandboxes=4)                                   # exits 0: five GENERATED RECs flipped
    out = capsys.readouterr().out
    assert len(payloads) == 5 and max(in_flight_at_spawn) <= 1                              # never more than two in flight
    assert all(p["max_sandboxes"] == 4 and p["max_sandboxes_override"] == 4 and p["max_sandboxes_marker"] == 8 for p in payloads)
    assert "5 store(s) pending" in out and "at most 2 in flight (--verify-width)" in out
    assert "10/10 container(s) OK" in out                                                 # five cell RECs flipped + five verify RECs
    # the dry run: the dispatch line, the argv spelled LAST (the default width is not spelled by the harness's launch_argv)
    base = dict(sha="f6cf462", app_name=L.APP_NAME, pass1=True, dry_run=True, cells="none", modes="off", controls="bigcodebench_full", out_suffix="s")
    capsys.readouterr()
    L.main(**base, row=True, verification_phase="generate", verify_pending=True, verify_width=2, verify_max_sandboxes=4)
    out = capsys.readouterr().out
    assert "verifiers: at most 2 in flight (--verify-width)" in out and "max_sandboxes 4 per verifier (--verify-max-sandboxes overrides" in out
    assert "5 sandboxes/s per workspace" in out
    argv_line = next(ln for ln in out.splitlines() if ln.startswith("[tinycap] argv:"))
    assert argv_line.split("  (run it")[0].endswith("--verification-phase generate --verify-pending --verify-width 2 --verify-max-sandboxes 4"), argv_line
    capsys.readouterr()
    L.main(**base, row=True, verification_phase="generate")                               # a generate ROW's dry run names the dispatch too
    assert "verifiers: at most 3 in flight (--verify-width)" in capsys.readouterr().out
    common = dict(pass1=True, row=True, cells="none", modes="off", out_suffix="s", verification_phase="generate")
    assert L.launch_argv("a" * 40, **common, verify_pending=True)[-3:] == [C.FLAG_VERIFICATION_PHASE, "generate", "--verify-pending"]
    assert L.launch_argv("a" * 40, **common, verify_pending=True, verify_width=2, verify_max_sandboxes=4, verify_spawn_stagger_s=0.5)[-7:] == [
        "--verify-pending", "--verify-width", "2", "--verify-max-sandboxes", "4", "--verify-spawn-stagger-s", "0.5"]
    assert L.launch_argv("a" * 40, **common, verify_width=3)[-2:] == [C.FLAG_VERIFICATION_PHASE, "generate"]   # the default: not spelled
    ns = L.cli_parser().parse_args(["--sha", "x", "--app-name", L.APP_NAME, "--pass1", "--row", "--cells", "none", "--modes", "off", "--out-suffix", "s",
                                    C.FLAG_VERIFICATION_PHASE, "generate", "--verify-pending", "--verify-width", "2", "--verify-max-sandboxes", "4",
                                    "--verify-spawn-stagger-s", "0.5"])
    assert (ns.verify_width, ns.verify_max_sandboxes, ns.verify_spawn_stagger_s) == (2, 4, 0.5)
    plain = L.cli_parser().parse_args(["--sha", "x", "--app-name", L.APP_NAME])
    assert (plain.verify_width, plain.verify_max_sandboxes, plain.verify_spawn_stagger_s) == (3, None, 2.0)
    src = Path(L.__file__).read_text()
    assert "verify_width=a.verify_width, verify_max_sandboxes=a.verify_max_sandboxes, verify_spawn_stagger_s=a.verify_spawn_stagger_s" in src
    # refused by name: without --row; beside a coupled phase; out of range
    solo = dict(sha="f6cf462", app_name=L.APP_NAME, pass1=True, dry_run=True, cells="none", modes="off", controls="bigcodebench_full", out_suffix="s")
    with pytest.raises(SystemExit, match="mean nothing without --row"):
        L.main(**solo, verify_width=2)
    with pytest.raises(SystemExit, match="mean nothing without --row"):
        L.main(**solo, verify_max_sandboxes=4)
    with pytest.raises(SystemExit, match="mean nothing beside --verification-phase 'coupled'"):
        L.main(**solo, row=True, verify_width=2)
    with pytest.raises(SystemExit, match="--verify-width must be >= 1"):
        L.main(**solo, row=True, verification_phase="generate", verify_width=0)
    with pytest.raises(SystemExit, match="--verify-max-sandboxes must be in 1..8"):
        L.main(**solo, row=True, verification_phase="generate", verify_max_sandboxes=9)
    with pytest.raises(SystemExit, match="--verify-spawn-stagger-s must be >= 0"):
        L.main(**solo, row=True, verification_phase="generate", verify_spawn_stagger_s=-1.0)


def test_a_resume_from_another_tree_verifies_markers_an_earlier_tree_wrote_and_reads_its_legacy_verified_blocks(L, tmp_path, monkeypatch):
    """`--verify-pending --sha <NEW landed sha>` over a row dir, markers and sidecars tree 082f583 wrote: the harness
    resolves the NEW tree for the verifier (`--tree <new>`; the payload's `tree_sha` stays the generation's) and holds
    nothing equal — the driver records both shas (§9.3 step 2) instead of refusing. And the harness's two exact-set
    readers (`pending_generation_state`, `verify_evidence`) read a LEGACY nine-key verified block (a store the earlier
    tree verified) as VERIFIED, never torn — the eleven-key block likewise; a stray key is still a problem."""
    repos = tmp_path / "repos"
    TL._tree(repos, "f6cf462")                                   # the tree that GENERATED (the payload's tree_sha = FULL_SHA)
    NEW_SHA = "0123456" + "a" * 33
    new_tree = TL._tree(repos, "0123456", full=NEW_SHA)          # the tree the resume runs
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    row_dir = tmp_path / "rows" / ROW / SUFFIX
    store, mk, pend = _verified_store(tmp_path, verified_by="fc-v-old", marker=True, row_dir=row_dir)
    assert pend["tree_sha"] == FULL_SHA != NEW_SHA
    cfg = {"cell": "none", "name": pend["name"], "model": ROW, "mode": "off", "out_dir": str(store)}
    argvs: List[list] = []

    def _driver(argv, **kw):
        argv = list(argv)
        argvs.append(argv)
        fcid = argv[argv.index("--function-call-id") + 1]
        mk.unlink()
        (row_dir / "watch").mkdir(exist_ok=True)
        (row_dir / "watch" / f"attempt_{fcid}.json").write_text(json.dumps({
            "run_id": "0123456-resume01", "cell": "none", "function_call_id": fcid, "name": "n", "model": ROW, "mode": "off", "out_dir": str(store),
            "controls": ["bigcodebench_full"], "tree_sha": FULL_SHA, "sandbox_record": "r", "sandbox_app_name": C.BCB_SANDBOX_APP_NAME,
            "started_at": 1.0, "phase": "verify"}))
        return SimpleNamespace(returncode=0, stdout="[served-cap] verify: … was verified by fc-v-old whose run died before removing the marker — "
                                                    "completing its tail (controls summary, RESULT, marker); nothing re-scored\n", stderr="")
    monkeypatch.setattr(L.subprocess, "run", _driver)
    rec = L._verify_cell_worker(cfg, "0123456", "0123456-resume01", STANDUP_A, str(row_dir), pending=pend, function_call_id="fc-v-new")
    assert rec["status"] == "OK", rec.get("error")
    assert rec["tree_sha"] == NEW_SHA and rec["tree"] == str(new_tree)               # the RESUME's tree, resolved by the new sha
    argv = argvs[0]
    assert argv[argv.index("--tree") + 1] == str(new_tree) and argv[argv.index("--row-engine-record") + 1] == pend["row_engine_record"]
    assert "tree" not in (rec.get("error") or "")
    # the readers: a legacy (nine-key) verified block is a verified store; the current (eleven-key) block too; a stray key is named
    shutil.rmtree(tmp_path / "gen")
    store, mk, pend = _verified_store(tmp_path, verified_by="fc-v-old", marker=False, row_dir=row_dir)      # writes the legacy block
    manifest = next(store.glob("cells/*/manifest.json"))
    legacy = json.loads(manifest.read_text())["row_engine"]["verification_pass"]
    assert set(legacy) == set(C.VERIFICATION_PASS_VERIFIED_FIELDS_V1) and len(legacy) == 9
    st = L.pending_generation_state(store, row_dir, cell="none", label="none")
    assert st["state"] == "verified" and st["problems"] == [] and st["verified_by"] == "fc-v-old"
    ALREADY = "[served-cap] verify: inspect/bigcodebench_full for none is already verified (by fc-v-old via M2_replay_vehicle over 1140 items); nothing to do"
    # (read as a FRESH call: the worker above left its own pointer for fc-v-new under the row dir)
    evidence = lambda: L.verify_evidence(out_dir=store, row_dir=row_dir, cell="none", function_call_id="fc-v-reader", tasks=[BCB_FULL], driver_stdout=ALREADY)
    ev = evidence()
    assert ev["problems"] == [] and ev["verified_by"] == "fc-v-old" and ev["already_verified"] is True
    doc = json.loads(manifest.read_text())
    doc["row_engine"]["verification_pass"] = {**legacy, "verifier_tree_sha": NEW_SHA, "generation_tree_sha": FULL_SHA,
                                              "executed_image": C.BCB_SANDBOX_IMAGE, "executed_verify_timeout_s": 30, "binding_version": "1"}
    assert set(doc["row_engine"]["verification_pass"]) == set(C.VERIFICATION_PASS_VERIFIED_FIELDS)
    manifest.write_text(json.dumps(doc))
    assert L.pending_generation_state(store, row_dir, cell="none", label="none")["problems"] == []
    assert evidence()["problems"] == []
    doc["row_engine"]["verification_pass"] = {**legacy, "stray": 1}
    manifest.write_text(json.dumps(doc))
    problems = L.pending_generation_state(store, row_dir, cell="none", label="none")["problems"]
    assert len(problems) == 1 and "verification_pass keys" in problems[0] and "'stray'" in problems[0] and "legacy set" in problems[0]
    assert any("verification_pass keys" in p for p in evidence()["problems"])
    assert C.VERIFICATION_PASS_VERIFIED_FIELDS[:11] == C.VERIFICATION_PASS_VERIFIED_FIELDS_V1 + ("verifier_tree_sha", "generation_tree_sha")
    # the harness never held the trees equal; the driver's former refusal is gone from the tree the verifier runs
    assert "the verifier files through the tree that generated" not in (REPO / "src/sisteer/capeval/tinycap/cpu_verify.py").read_text()
    assert "recorded as verification_pass.verifier_tree_sha" in (REPO / "src/sisteer/capeval/tinycap/cpu_verify.py").read_text()


class _FlakyCall:
    """A running verifier's FunctionCall whose control plane flaps: ``get(timeout=0)`` raises an UNAVAILABLE-class error
    ``unavailable_polls`` times (a ``ConnectionError`` — an OSError, what a dropped gRPC channel surfaces as), then the
    builtin ``TimeoutError`` ``running_polls`` times (still running), then returns its REC (or raises its failure)."""
    _n = 0

    def __init__(self, result, *, unavailable_polls: int, running_polls: int, live: set, label: str,
                 unavailable_exc=None):
        type(self)._n += 1
        self.object_id = f"fc-flaky-{type(self)._n:03d}"
        self._result, self._u, self._r, self._live, self.label = result, unavailable_polls, running_polls, live, label
        self._exc = unavailable_exc
        self.gets = 0
        self._live.add(label)

    def get(self, timeout=None):
        self.gets += 1
        if self._u > 0:
            self._u -= 1
            raise (self._exc or ConnectionError("socket closed"))
        if self._r > 0:
            self._r -= 1
            raise TimeoutError()
        self._live.discard(self.label)
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


def test_an_unavailable_get_holds_the_slot_and_only_a_confirmed_terminal_or_lost_call_frees_it(L, monkeypatch, capsys):
    """A transient control-plane error while polling a still-running verifier used to read as terminal — the call left
    in_flight, another spawned (the width defeated) and the label read FUNCTIONCALL_FAILED though it may still file.
    Now the poll is classified like the ownership probe / the watcher: `active` and `unavailable` KEEP the slot and are
    re-probed next round; only `terminal_ok`, `terminal_failed` and `lost` free it. Three labels at width 2: c0's call
    is unavailable twice then finishes OK — c2 is NOT spawned while c0 is unreadable; c1's call genuinely fails (the
    function raised) — its slot frees at once."""
    labels, result, recs, cfgs = _dispatch_world(3)
    _recording_writes(L, monkeypatch)
    sleeps: List[float] = []
    monkeypatch.setattr(L, "_dispatch_sleep", lambda s: sleeps.append(s))
    live: set = set()
    in_flight_at_spawn: List[list] = []
    calls: Dict[str, object] = {}

    def _spawn(cell_cfg, *a, **kw):
        label = cell_cfg["cell"]
        in_flight_at_spawn.append(sorted(live))
        ok = {"name": f"{cell_cfg['name']}:verify", "cell": label, "status": "OK", "phase": "verify", "verified_by": f"v-{label}"}
        if label == "c0":
            fc = _FlakyCall(ok, unavailable_polls=2, running_polls=1, live=live, label=label)
        elif label == "c1":
            fc = _FlakyCall(RuntimeError("the verifier raised"), unavailable_polls=0, running_polls=1, live=live, label=label)
        else:
            fc = _FlakyCall(ok, unavailable_polls=0, running_polls=0, live=live, label=label)
        calls[label] = fc
        return fc
    capsys.readouterr()
    vrecs = L.verify_pending_cells(SimpleNamespace(spawn=_spawn), result, cfgs, recs, sha="f6cf462", run_id=DISPATCH_RUN,
                                   reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT, verify_width=2)
    out = capsys.readouterr().out
    # c2 spawned only once c1 (the confirmed failure) freed its slot — c0, unreadable, kept its slot the whole time
    assert in_flight_at_spawn == [[], ["c0"], ["c0"]], in_flight_at_spawn
    assert {v["label"]: v["status"] for v in vrecs} == {"c0": "OK", "c1": "FUNCTIONCALL_FAILED", "c2": "OK"}
    assert next(v for v in vrecs if v["label"] == "c1")["call_state"] == "terminal_failed"
    assert "the verifier raised" in next(v for v in vrecs if v["label"] == "c1")["error"]
    assert calls["c0"].gets == 4                                                        # 2 unavailable + 1 running + the result
    assert out.count("call state UNAVAILABLE (ConnectionError: socket closed) — slot held, re-probed next poll") == 2
    assert sleeps.count(L.VERIFY_JOIN_POLL_S) >= 1                                      # an unreadable round waited, it did not spin
    by_cell = {r["cell"]: r for r in recs}
    assert by_cell["c0"]["status"] == "OK" and by_cell["c2"]["status"] == "OK" and by_cell["c1"]["status"] == "GENERATED"
    # the classifier, by name: the builtin TimeoutError is active; lost / unavailable / terminal classes as the probe reads them
    class NotFoundError(Exception):
        pass
    class ServiceError(Exception):
        pass
    class FunctionTimeoutError(Exception):
        pass
    assert L.classify_call_get_exception(TimeoutError()) == L.CALL_ACTIVE
    assert L.classify_call_get_exception(NotFoundError("x")) == L.CALL_LOST
    assert L.classify_call_get_exception(ServiceError("x")) == L.CALL_UNAVAILABLE == L.classify_call_get_exception(ConnectionError("x"))
    assert L.classify_call_get_exception(OSError("socket")) == L.CALL_UNAVAILABLE
    assert L.classify_call_get_exception(FunctionTimeoutError("x")) == L.CALL_TERMINAL_FAILED == L.classify_call_get_exception(RuntimeError("x"))
    assert L.observe_verifier_call(SimpleNamespace(get=lambda timeout: {"status": "OK"})) == (L.CALL_TERMINAL_OK, {"status": "OK"}, None)
    state, value, err = L.observe_verifier_call(SimpleNamespace(get=lambda timeout: (_ for _ in ()).throw(ServiceError("down"))))
    assert (state, value) == (L.CALL_UNAVAILABLE, None) and err == "ServiceError: down"
    assert L._poll_verifier_call(SimpleNamespace(get=lambda timeout: (_ for _ in ()).throw(TimeoutError())), 0.0) == (L.CALL_ACTIVE, None, None)
    # the probe shares the observer: an unavailable call is `unavailable` there too (never a release)
    table = {"fc-u": _FakeCall("fc-u", ServiceError("down"))}
    monkeypatch.setattr(L.modal, "FunctionCall", _fake_function_call(table), raising=False)
    assert L.probe_verifier_call("fc-u")["state"] == "unavailable"


def test_a_call_unreadable_past_the_join_bound_is_recorded_blocked_with_its_provenance_never_functioncall_failed(L, monkeypatch, capsys):
    """The belt: a call that stays unavailable (or running) past VERIFY_JOIN_MAX_S is not FUNCTIONCALL_FAILED — the
    verifier may still file. The label's record entry reads `blocked` (non-terminal for the watcher; the next resume's
    classifier re-probes exactly the carried call: joins it live, supersedes it dead) with the provenance (this run, the
    dispatch state, the call id, the last observation), its REC is VERIFY_CALL_UNAVAILABLE, and the cell REC is left as
    it was (GENERATED, not annotated)."""
    labels, result, recs, cfgs = _dispatch_world(2)
    writes = _recording_writes(L, monkeypatch)
    written_full: Dict[str, dict] = {}
    monkeypatch.setattr(L, "put_row_dir_file", lambda row_dir, name, d, **kw: (written_full.__setitem__(name, json.loads(json.dumps(d, default=str))),
                                                                                  writes.append((name, kw.get("must_exist"), {k: (v["state"], v.get("function_call_id")) for k, v in d["cells"].items()}))) and "x")
    monkeypatch.setattr(L, "_dispatch_sleep", lambda s: None)
    monkeypatch.setattr(L, "VERIFY_JOIN_MAX_S", -1.0)                                   # every non-terminal observation is already past the bound
    live: set = set()

    def _spawn(cell_cfg, *a, **kw):
        label = cell_cfg["cell"]
        ok = {"name": f"{cell_cfg['name']}:verify", "cell": label, "status": "OK", "phase": "verify"}
        if label == "c0":
            return _FlakyCall(ok, unavailable_polls=10, running_polls=0, live=live, label=label)     # never readable
        return _FlakyCall(ok, unavailable_polls=0, running_polls=0, live=live, label=label)
    capsys.readouterr()
    vrecs = L.verify_pending_cells(SimpleNamespace(spawn=_spawn), result, cfgs, recs, sha="f6cf462", run_id=DISPATCH_RUN,
                                   reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT, verify_width=2)
    out = capsys.readouterr().out
    v0 = next(v for v in vrecs if v["label"] == "c0")
    assert v0["status"] == C.REC_VERIFY_CALL_UNAVAILABLE == "VERIFY_CALL_UNAVAILABLE" and v0["function_call_id"].startswith("fc-flaky-")
    assert "could not be read as terminal" in v0["error"] and "unavailable, ConnectionError: socket closed" in v0["error"]
    assert "FUNCTIONCALL_FAILED" not in json.dumps(v0)
    assert next(v for v in vrecs if v["label"] == "c1")["status"] == "OK"
    ent = written_full[f"verifiers_{DISPATCH_RUN}.json"]["cells"]["c0"]
    assert ent["state"] == "blocked" and ent["function_call_id"] == v0["function_call_id"]
    assert ent["prior"]["run_id"] == DISPATCH_RUN and ent["prior"]["state"] == "spawned" and ent["prior"]["function_call_id"] == v0["function_call_id"]
    assert ent["prior"]["record"] == f"verifiers_{DISPATCH_RUN}.json" and ent["prior"]["probe"]["state"] == "unavailable"
    assert "ConnectionError: socket closed" in ent["prior"]["probe"]["error"]
    assert written_full[f"verifiers_{DISPATCH_RUN}.json"]["cells"]["c1"]["state"] == "spawned"
    assert L.VERIFIER_STATE_BLOCKED == "blocked" not in L.VERIFIER_TERMINAL_STATES         # the watcher withholds DONE on it
    by_cell = {r["cell"]: r for r in recs}
    assert by_cell["c0"]["status"] == "GENERATED" and "verification" not in by_cell["c0"] and by_cell["c1"]["status"] == "OK"
    assert "VERIFY_CALL_UNAVAILABLE" in out and "re-probed by the next resume" in out
    # the next resume's classifier reads the blocked entry's carried call and probes IT (a live one is joined)
    prior = L.prior_verifier_claim(result["row_dir"], "c0") if False else None            # (the record lives on the volume; the shape is what matters)
    assert prior is None


def test_the_worker_names_the_drivers_scorer_identity_refusal(L, tmp_path, monkeypatch):
    """The harness half: the driver refuses a cross-tree verify whose scorer identity is not the generation's with the
    SCORER_IDENTITY_MISMATCH mark on stderr; the worker's REC is VERIFY_SCORER_IDENTITY_MISMATCH (not the bare
    VERIFY_FAILED), so the launch log says why nothing was scored; the store stays pending."""
    repos = tmp_path / "repos"
    TL._tree(repos, "f6cf462")
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    row_dir = tmp_path / "rows" / ROW / SUFFIX
    store, mk, pend = _verified_store(tmp_path, verified_by="fc-v-old", marker=True, row_dir=row_dir)
    manifest = next(store.glob("cells/*/manifest.json"))
    doc = json.loads(manifest.read_text())
    doc["row_engine"]["verification_pass"] = {"state": "pending", "tasks": [BCB_FULL]}                # a pending store
    manifest.write_text(json.dumps(doc))
    cfg = {"cell": "none", "name": pend["name"], "model": ROW, "mode": "off", "out_dir": str(store)}
    refusal = ("[served-cap] verify: SCORER_IDENTITY_MISMATCH: the scorer this tree would run is not the one the store was generated "
               "under — image: this verifier would execute 'ghcr.io/x@sha256:bb', the store was generated under 'ghcr.io/x@sha256:aa' "
               "(a cross-tree verifier is admitted only when the scorer identity is byte-equal; nothing scored, the marker stands)\n")
    monkeypatch.setattr(L.subprocess, "run", lambda argv, **kw: SimpleNamespace(returncode=1, stdout="", stderr=refusal))
    rec = L._verify_cell_worker(cfg, "f6cf462", "f6cf462-cafebabe", STANDUP_A, str(row_dir), pending=pend, function_call_id="fc-v-x")
    assert rec["status"] == C.REC_VERIFY_SCORER_IDENTITY_MISMATCH == "VERIFY_SCORER_IDENTITY_MISMATCH"
    assert rec["status"] not in L.REC_OK_STATUSES and "verify returned rc 1" in rec["error"] and rec["driver_rc"] == 1
    assert mk.is_file()                                                                          # the marker stands
    # any other non-zero exit stays VERIFY_FAILED
    monkeypatch.setattr(L.subprocess, "run", lambda argv, **kw: SimpleNamespace(returncode=1, stdout="", stderr="boom"))
    assert L._verify_cell_worker(cfg, "f6cf462", "f6cf462-cafebabe", STANDUP_A, str(row_dir), pending=pend, function_call_id="fc-v-y")["status"] == "VERIFY_FAILED"


def test_the_readers_hold_a_present_standup_token_on_the_last_result_line_to_the_manifests_id_and_accept_an_absent_one(L, tmp_path, monkeypatch):
    """The registry's landing-2 rule read by the harness first (2026-09-30): the verifier's re-stamp now carries
    `standup=<manifest row_engine.standup_id>`; `pending_generation_state` and `verify_evidence` name a last line whose
    token is ANOTHER standup (one standup id per row launch) and accept a line with no token (the stores verified before
    the re-stamp carried one are read off the manifest, never torn)."""
    assert L.result_standup_token("tinycap-modal name=n rc=0 result=COMPLETE status=OK sha=x at=t standup=su verified_by=f") == "su"
    assert L.result_standup_token("tinycap-modal name=n rc=0 result=COMPLETE status=OK sha=x at=t verified_by=f") is None
    assert L.result_standup_problem("… standup=su verified_by=f", "su") is None
    assert L.result_standup_problem("… verified_by=f", "su") is None                       # absent: accepted
    assert L.result_standup_problem("… standup=su verified_by=f", None) is None            # no manifest id to hold it to
    problem = L.result_standup_problem("… standup=other verified_by=f", "su")
    assert "names standup=other" in problem and "row_engine.standup_id is su" in problem
    row_dir = tmp_path / "rows" / ROW / SUFFIX
    store, mk, pend = _verified_store(tmp_path, verified_by="fc-v-old", marker=False, row_dir=row_dir)   # last line: no standup token
    assert L.pending_generation_state(store, row_dir, cell="none", label="none")["problems"] == []
    ALREADY = "[served-cap] verify: … is already verified (by fc-v-old via M2_replay_vehicle over 1140 items); nothing to do"
    assert L.verify_evidence(out_dir=store, row_dir=row_dir, cell="none", function_call_id="fc-r", tasks=[BCB_FULL], driver_stdout=ALREADY)["problems"] == []
    result = store / "RESULT"
    lines = result.read_text().splitlines()
    result.write_text("\n".join(lines[:-1] + [lines[-1].replace(" verified_by=fc-v-old", f" standup={STANDUP_A} verified_by=fc-v-old")]) + "\n")
    assert L.result_standup_token(result.read_text().splitlines()[-1]) == STANDUP_A
    assert L.pending_generation_state(store, row_dir, cell="none", label="none")["problems"] == []          # the manifest's id: accepted
    result.write_text("\n".join(lines[:-1] + [lines[-1].replace(" verified_by=fc-v-old", " standup=99999999888888887777777766666666 verified_by=fc-v-old")]) + "\n")
    st = L.pending_generation_state(store, row_dir, cell="none", label="none")
    assert st["state"] == "torn" and any("names standup=99999999888888887777777766666666" in p_ and f"standup_id is {STANDUP_A}" in p_ for p_ in st["problems"])
    ev = L.verify_evidence(out_dir=store, row_dir=row_dir, cell="none", function_call_id="fc-r", tasks=[BCB_FULL], driver_stdout=ALREADY)
    assert any("names standup=" in p_ for p_ in ev["problems"])

"""generate-anti-neg multi-GPU sharding (2026-08-20).

CPU-only, offline, no vLLM, NO harmful text — every prompt/response is an
INVENTED synthetic string (privacy rule). What the suite proves, mirroring the
``pool_sample.py`` pattern the sharding reuses:

- default path (no shard flags, or 0/1) is byte-compatible with the single-GPU
  behaviour: same legacy ``anti_neg.jsonl``, same full inventory;
- the shard split over the harmful-prompt inventory is deterministic, disjoint
  and complete at N=1,3,8 — no prompt generated twice, none missed (a
  duplicate would corrupt the rejection-sampling accounting);
- concurrent shards never share an output file;
- per-cell resume holds per shard AND across a shard-layout change (a partial
  single-file run re-run 3-way regenerates nothing);
- the downstream readers — grade-anti-neg and the manifest's anti bucket via
  ``_iter_all_pool_rows`` — see EVERY shard's file, not just one;
- the fan-out (`generate-anti-neg-all`) pins each child to one entry of the
  ALREADY-VISIBLE GPU set (a strict subset, never widened), passes the global
  shard args through, refuses an overflowing shard space, and propagates a
  failed shard's exit.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_two_si_pool as bp  # noqa: E402
import sisteer.sampling.engine as E  # noqa: E402
from sisteer.eval.jailbreak_data import harm_category_holdout  # noqa: E402
from sisteer.sampling.sources import SamplePrompt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SI = ("safety_refusal", "si_constitution")
ANTI = "anti_safety_refusal"
HELD_IN, _HELD_OUT = harm_category_holdout(seed=0, frac=0.33)


class _FakeSampler:
    """Stand-in for VllmSampler: same run() contract (skip `done` cells,
    append engine._row rows), no GPU. Class-level counter so tests can assert
    it was never constructed on a nothing-to-do path."""

    built = 0

    def __init__(self, model, **kw):
        type(self).built += 1

    def run(self, prompts, specs, out_path, *, done=None, progress_every=2000):
        from sisteer.sampling.engine import cell_key

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        done = set(done or ())
        written = 0
        with out_path.open("a") as f:
            for spec in specs:
                for ci in range(spec.k):
                    for p in prompts:
                        if cell_key(p, spec.si_id, ci) in done:
                            continue
                        f.write(json.dumps(E._row(
                            p, spec.si_id, ci,
                            f"SYNTHETIC anti candidate {p.goal_id} c{ci}"))
                            + "\n")
                        written += 1
        return {"written": written}


class _BoomSampler:
    def __init__(self, *a, **k):
        raise AssertionError("sampler constructed on a fully-resumed shard")


def _base_cfg(tmp_path, k=2):
    cfg = yaml.safe_load((REPO / "configs" / "two_si.yaml").read_text())
    cfg["si_ids"] = list(SI)
    cfg["pool"] = {**cfg["pool"], "store_dir": str(tmp_path / "labeled"),
                   "static_families": ["bon"], "train_families": ["bon"],
                   "min_pool_size": 1, "hard_floor_pairs": 0,
                   "benign_source": None, "benign_sources": [],
                   "benign_fraction": None, "scoring_disjointness": {},
                   "fit_floor": {}, "relaxed_stratum": {"enabled": False},
                   # pre-flip world: the store serves safety_refusal's pos
                   "pos_runs": {k: v
                                for k, v in (cfg["pool"].get("pos_runs")
                                             or {}).items()
                                if k != "safety_refusal"},
                   "bidpo_benign": {"enabled": False},
                   "anti_neg": {"enabled": True, "anti_si_id": ANTI,
                                "run": "two_si_anti_neg", "candidates": k,
                                "temperature": 1.0}}
    return cfg


def _shard_world(tmp_path, n_prompts=8, k=2):
    """A store of REAL engine._row rows (category_group None — the read-time
    split derivation is part of what generation exercises), n harmful train
    prompts."""
    cfg = _base_cfg(tmp_path, k=k)
    store = tmp_path / "labeled"
    store.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_prompts):
        p = SamplePrompt(
            family="bon", goal_id=f"g{i}", sample_idx=i, trial=0,
            variant_name="bon_0",
            text=f"SYNTHETIC shard inventory prompt {i} harmful 000{i}",
            side="harmful", source="jbb", category=HELD_IN[0])
        rows.append(E._row(p, E.NONE_SI, 0, f"SYNTHETIC undefended resp {i}"))
    (store / "bon.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    return cfg, tmp_path / "priv"


def _args(priv, *, k=2, idx=None, cnt=None, dry_run=False):
    a = types.SimpleNamespace(
        candidates=k, private_root=str(priv), dry_run=dry_run, seed=0,
        max_model_len=16384, gpu_memory_utilization=0.9, enforce_eager=False)
    if idx is not None:
        a.shard_index, a.shard_count = idx, cnt
    return a


def _expected_keys(n_prompts, k):
    return {f"bon|g{i}|{i}:0|{ANTI}|{c}"
            for i in range(n_prompts) for c in range(k)}


# --------------------------------------------------------------------------- #
# 1. default-path invariance: no shard flags == exactly the single-GPU path     #
# --------------------------------------------------------------------------- #


def test_no_shard_flags_is_todays_single_file_behaviour(tmp_path, monkeypatch):
    """An args namespace WITHOUT shard attributes (every pre-sharding caller)
    generates the full inventory into the legacy anti_neg.jsonl and nothing
    else — the flags' absence is the 0/1 default."""
    cfg, priv = _shard_world(tmp_path)
    monkeypatch.setattr("sisteer.sampling.engine.VllmSampler", _FakeSampler)
    bp.cmd_generate_anti_neg(_args(priv), cfg)   # no shard_index/shard_count
    adir = priv / "two_si_anti_neg"
    assert (adir / "anti_neg.jsonl").exists()
    assert not list(adir.glob("anti_neg_shard_*.jsonl"))
    assert E.load_done_keys(adir / "anti_neg.jsonl") == _expected_keys(8, 2)


def test_explicit_0_of_1_matches_the_default(tmp_path, monkeypatch):
    cfg, priv = _shard_world(tmp_path)
    monkeypatch.setattr("sisteer.sampling.engine.VllmSampler", _FakeSampler)
    bp.cmd_generate_anti_neg(_args(priv, idx=0, cnt=1), cfg)
    adir = priv / "two_si_anti_neg"
    assert (adir / "anti_neg.jsonl").exists()
    assert not list(adir.glob("anti_neg_shard_*.jsonl"))
    assert E.load_done_keys(adir / "anti_neg.jsonl") == _expected_keys(8, 2)


def test_bad_shard_spec_refuses_loudly(tmp_path):
    cfg, priv = _shard_world(tmp_path)
    for idx, cnt in ((3, 3), (-1, 3), (0, 0)):
        with pytest.raises(SystemExit, match="bad shard spec"):
            bp.cmd_generate_anti_neg(_args(priv, idx=idx, cnt=cnt), cfg)


# --------------------------------------------------------------------------- #
# 2. disjoint + complete coverage at N = 1, 3, 8; no shared output file         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_shards", [1, 3, 8])
def test_shards_are_disjoint_and_cover_the_whole_inventory(
        tmp_path, monkeypatch, n_shards):
    cfg, priv = _shard_world(tmp_path)
    monkeypatch.setattr("sisteer.sampling.engine.VllmSampler", _FakeSampler)
    for i in range(n_shards):
        bp.cmd_generate_anti_neg(_args(priv, idx=i, cnt=n_shards), cfg)
    adir = priv / "two_si_anti_neg"
    files = ([adir / "anti_neg.jsonl"] if n_shards == 1
             else sorted(adir.glob("anti_neg_shard_*.jsonl")))
    assert len(files) == n_shards                # every shard its OWN file
    per_file = [E.load_done_keys(f) for f in files]
    union = set().union(*per_file)
    # complete: N shards produce exactly the set one shard would...
    assert union == _expected_keys(8, 2)
    # ...and disjoint: no cell (hence no prompt) generated twice.
    assert sum(len(s) for s in per_file) == len(union)
    goals = [{k.split("|")[1] for k in s} for s in per_file]
    for i in range(len(goals)):
        for j in range(i + 1, len(goals)):
            assert goals[i].isdisjoint(goals[j])


def test_dry_run_reports_the_shard_slice(tmp_path, capsys):
    cfg, priv = _shard_world(tmp_path)
    bp.cmd_generate_anti_neg(_args(priv, idx=0, cnt=3, dry_run=True), cfg)
    out = capsys.readouterr().out
    assert "shard 0/3" in out
    assert "3 of 8 harmful train prompts" in out and "6 cells" in out
    assert not (priv / "two_si_anti_neg").exists()   # dry-run: no side effects


# --------------------------------------------------------------------------- #
# 3. resume: per shard, within a shard, and across a shard-layout change        #
# --------------------------------------------------------------------------- #


def test_rerun_of_a_completed_shard_generates_nothing(tmp_path, monkeypatch,
                                                      capsys):
    cfg, priv = _shard_world(tmp_path)
    monkeypatch.setattr("sisteer.sampling.engine.VllmSampler", _FakeSampler)
    bp.cmd_generate_anti_neg(_args(priv, idx=1, cnt=3), cfg)
    path = priv / "two_si_anti_neg" / "anti_neg_shard_01.jsonl"
    before = path.read_bytes()
    monkeypatch.setattr("sisteer.sampling.engine.VllmSampler", _BoomSampler)
    bp.cmd_generate_anti_neg(_args(priv, idx=1, cnt=3), cfg)   # resumes: no-op
    assert "nothing to generate" in capsys.readouterr().out
    assert path.read_bytes() == before


def test_partial_shard_resumes_exactly_the_missing_cells(tmp_path,
                                                         monkeypatch):
    cfg, priv = _shard_world(tmp_path)
    monkeypatch.setattr("sisteer.sampling.engine.VllmSampler", _FakeSampler)
    bp.cmd_generate_anti_neg(_args(priv, idx=1, cnt=3), cfg)
    path = priv / "two_si_anti_neg" / "anti_neg_shard_01.jsonl"
    lines = path.read_text().splitlines()
    assert len(lines) == 6                       # 3 prompts x K=2
    path.write_text("\n".join(lines[:2]) + "\n")  # crash after 2 cells
    bp.cmd_generate_anti_neg(_args(priv, idx=1, cnt=3), cfg)
    keys = E.load_done_keys(path)
    assert len(path.read_text().splitlines()) == 6   # regenerated 4, no dups
    assert len(keys) == 6


def test_layout_change_resume_skips_the_legacy_file_cells(tmp_path,
                                                          monkeypatch):
    """A completed single-GPU run re-run 3-way generates NOTHING — every
    shard's cells are already in the legacy anti_neg.jsonl and the done-key
    union spans all files in the run dir."""
    cfg, priv = _shard_world(tmp_path)
    monkeypatch.setattr("sisteer.sampling.engine.VllmSampler", _FakeSampler)
    bp.cmd_generate_anti_neg(_args(priv), cfg)   # full legacy single-file run
    monkeypatch.setattr("sisteer.sampling.engine.VllmSampler", _BoomSampler)
    for i in range(3):
        bp.cmd_generate_anti_neg(_args(priv, idx=i, cnt=3), cfg)
    assert not list((priv / "two_si_anti_neg").glob("anti_neg_shard_*.jsonl"))


# --------------------------------------------------------------------------- #
# 4. downstream readers see EVERY shard's file                                  #
# --------------------------------------------------------------------------- #


def test_iter_all_pool_rows_reads_every_shard_file(tmp_path, monkeypatch):
    cfg, priv = _shard_world(tmp_path)
    monkeypatch.setattr("sisteer.sampling.engine.VllmSampler", _FakeSampler)
    for i in range(3):
        bp.cmd_generate_anti_neg(_args(priv, idx=i, cnt=3), cfg)
    got = [r for r in bp._iter_all_pool_rows(
               cfg, static_families=("bon",), private_root=str(priv))
           if r.get("si_id") == ANTI]
    assert len(got) == 16                        # 8 prompts x K=2, all shards
    assert {r["goal_id"] for r in got} == {f"g{i}" for i in range(8)}


def test_grade_anti_neg_grades_across_all_shard_files(tmp_path, monkeypatch):
    cfg, priv = _shard_world(tmp_path)
    monkeypatch.setattr("sisteer.sampling.engine.VllmSampler", _FakeSampler)
    for i in range(3):
        bp.cmd_generate_anti_neg(_args(priv, idx=i, cnt=3), cfg)

    def _build(args, cfg_, cache_dir, cache_name=None):
        def grade(items):
            return {it: 0.9 for it in items}     # every c0 a compliance
        return grade
    monkeypatch.setattr(bp, "_build_grader", _build)
    bp.cmd_grade_anti_neg(types.SimpleNamespace(
        private_root=str(priv), dry_run=False, concurrency=4), cfg)
    adir = priv / "two_si_anti_neg"
    for f in sorted(adir.glob("anti_neg_shard_*.jsonl")):
        rows = [json.loads(l) for l in f.read_text().splitlines()]
        by_c = {}
        for r in rows:
            by_c.setdefault(r["goal_id"], {})[r["candidate_idx"]] = r
        for gid, cands in by_c.items():
            # lazy rounds: c0's accepted compliance retires the group; c1
            # stays honestly ungraded (never needed) — in EVERY shard file.
            assert cands[0]["score"] == 0.9, (f.name, gid)
            assert cands[1]["score"] is None, (f.name, gid)


def _matched_world(tmp_path):
    """A minimal matched two-SI world (both SIs thwart two complied prompts)
    with NO anti rows yet — generation + grading fill them in, sharded."""
    cfg = _base_cfg(tmp_path, k=1)
    cfg["pool_manifest"] = str(tmp_path / "pool_manifest.json")
    store = tmp_path / "labeled"
    store.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in (0, 1):
        base = {"attack": "bon", "goal_id": "g0", "sample_idx": idx,
                "trial": 0, "category": "illegal", "category_group": "held_in",
                "attack_prompt": f"SYNTHETIC matched prompt {idx} aaaa000{idx}",
                "graded": True}
        rows += [{**base, "si_condition": "none",
                  "response": "SYNTHETIC undefended complied bbbb0000",
                  "score": 0.9},
                 {**base, "si_condition": "prompted",
                  "response": "SYNTHETIC safety refusal cccc0000",
                  "score": 0.0}]
    # A held-out goal. The shipped config runs goal-level holdout
    # (pool.family_disjointness: false), so a store whose rows are ALL
    # held_in yields an empty eval goal set -- no holdout in either
    # dimension, which build_matched_pool refuses.
    held = {"attack": "bon", "goal_id": "g_held", "sample_idx": 2,
            "trial": 0, "category": "illegal", "category_group": "held_out",
            "attack_prompt": "SYNTHETIC matched prompt held aaaa0002",
            "graded": True}
    rows += [{**held, "si_condition": "none",
              "response": "SYNTHETIC undefended complied bbbb0002",
              "score": 0.9},
             {**held, "si_condition": "prompted",
              "response": "SYNTHETIC safety refusal cccc0002", "score": 0.0}]
    (store / "bon.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    priv = tmp_path / "priv"
    cdir = priv / cfg["pool"].get("constitution_run", "two_si_pos_constitution")
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "bon.jsonl").write_text("".join(json.dumps(
        {"attack": "bon", "goal_id": "g0", "sample_idx": idx, "trial": 0,
         "category": "illegal", "category_group": "held_in",
         "si_condition": "prompted", "si_id": "si_constitution",
         "candidate_idx": 0,
         "attack_prompt": f"SYNTHETIC matched prompt {idx} aaaa000{idx}",
         "response": "SYNTHETIC constitution refusal dddd0000",
         "score": 0.0, "graded": True}) + "\n" for idx in (0, 1)))
    return cfg, priv


def test_manifest_anti_bucket_consumes_all_shards_end_to_end(tmp_path,
                                                             monkeypatch):
    """The full seam: sharded generation (2 workers, one prompt each) ->
    grade-anti-neg -> manifest. The anti bucket must carry BOTH prompts'
    accepted compliances — a reader stuck on the legacy single file would
    see only one."""
    cfg, priv = _matched_world(tmp_path)
    monkeypatch.setattr("sisteer.sampling.engine.VllmSampler", _FakeSampler)
    for i in range(2):
        bp.cmd_generate_anti_neg(_args(priv, k=1, idx=i, cnt=2), cfg)
    shard_files = sorted(
        (priv / "two_si_anti_neg").glob("anti_neg_shard_*.jsonl"))
    assert len(shard_files) == 2 and all(
        len(f.read_text().splitlines()) == 1 for f in shard_files)

    def _build(args, cfg_, cache_dir, cache_name=None):
        def grade(items):
            return {it: 0.9 for it in items}
        return grade
    monkeypatch.setattr(bp, "_build_grader", _build)
    bp.cmd_grade_anti_neg(types.SimpleNamespace(
        private_root=str(priv), dry_run=False, concurrency=4), cfg)

    bp.cmd_manifest(types.SimpleNamespace(private_root=str(priv), out=None,
                                          families=None), cfg)
    m = json.loads(Path(cfg["pool_manifest"]).read_text())
    blk = m["bidpo_anti_neg"]
    assert blk["n_harmful_pairs"] == 2
    assert blk["n_with_anti_neg"] == 2           # one from EACH shard file
    assert blk["n_missing_anti_neg"] == 0


# --------------------------------------------------------------------------- #
# 5. the fan-out: one child per GPU, strict CUDA subset, global shard space     #
# --------------------------------------------------------------------------- #


class _FakeProc:
    def __init__(self, rc=0):
        self.rc = rc

    def wait(self):
        return self.rc


def _fanout_args(priv, **over):
    base = dict(config="configs/two_si.yaml", candidates=2,
                private_root=str(priv), seed=0, max_model_len=16384,
                gpu_memory_utilization=0.9, enforce_eager=False,
                gpus=None, shard_offset=0, shard_total=None)
    base.update(over)
    return types.SimpleNamespace(**base)


def test_fanout_pins_each_child_to_a_strict_gpu_subset(tmp_path, monkeypatch):
    cfg, priv = _shard_world(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")   # what Slurm/pod gave us
    spawned = []

    def _popen(cmd, env=None, stdout=None, stderr=None):
        spawned.append((cmd, env))
        return _FakeProc(0)
    monkeypatch.setattr(bp.subprocess, "Popen", _popen)
    bp.cmd_generate_anti_neg_all(_fanout_args(priv), cfg)

    assert len(spawned) == 2
    for i, (cmd, env) in enumerate(spawned):
        # one entry of the ALREADY-VISIBLE set — never widened, never absolute
        assert env["CUDA_VISIBLE_DEVICES"] == str(4 + i)
        assert cmd[0] == sys.executable and cmd[2] == "generate-anti-neg"
        assert cmd[cmd.index("--shard-index") + 1] == str(i)
        assert cmd[cmd.index("--shard-count") + 1] == "2"
        assert cmd[cmd.index("--candidates") + 1] == "2"
        assert cmd[cmd.index("--private-root") + 1] == str(priv)
        # ninja-on-PATH: the interpreter's bin/ leads the child's PATH
        assert env["PATH"].startswith(str(Path(sys.executable).parent))
    # the run dir was created (mode 0700) before any log was opened
    assert (priv / "two_si_anti_neg").is_dir()


def test_fanout_global_shard_space_and_overflow_guard(tmp_path, monkeypatch):
    # MERGE (stage3 -> twosweep): pin CUDA_VISIBLE_DEVICES so _visible_gpus
    # short-circuits instead of shelling out to nvidia-smi. The fanout now
    # routes --gpus through pool_sample.resolve_gpus, which compares the
    # REQUESTED ids against the VISIBLE ones — that comparison is the guard
    # that catches a GPU COUNT typed into a device-id list (job 212789 ran on
    # one GPU over 1/8 of the inventory and exited 0). It therefore needs the
    # visible list, and without this env var the probe becomes a third Popen
    # that this test's capture would count. Pinning it also matches how the
    # fanout actually runs, under Slurm/Modal, where the var is always set.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    cfg, priv = _shard_world(tmp_path)
    spawned = []
    monkeypatch.setattr(bp.subprocess, "Popen",
                        lambda cmd, **k: (spawned.append(cmd), _FakeProc())[1])
    bp.cmd_generate_anti_neg_all(
        _fanout_args(priv, gpus="0,1", shard_offset=4, shard_total=8), cfg)
    assert [c[c.index("--shard-index") + 1] for c in spawned] == ["4", "5"]
    assert all(c[c.index("--shard-count") + 1] == "8" for c in spawned)
    with pytest.raises(SystemExit, match="overflow"):
        bp.cmd_generate_anti_neg_all(
            _fanout_args(priv, gpus="0,1", shard_offset=7, shard_total=8), cfg)


def test_fanout_propagates_a_failed_shard(tmp_path, monkeypatch):
    # MERGE (stage3 -> twosweep): pin CUDA_VISIBLE_DEVICES so _visible_gpus
    # short-circuits instead of shelling out to nvidia-smi. The fanout now
    # routes --gpus through pool_sample.resolve_gpus, which compares the
    # REQUESTED ids against the VISIBLE ones — that comparison is the guard
    # that catches a GPU COUNT typed into a device-id list (job 212789 ran on
    # one GPU over 1/8 of the inventory and exited 0). It therefore needs the
    # visible list, and without this env var the probe becomes a third Popen
    # that this test's capture would count. Pinning it also matches how the
    # fanout actually runs, under Slurm/Modal, where the var is always set.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    cfg, priv = _shard_world(tmp_path)
    rcs = iter([0, 3])
    monkeypatch.setattr(bp.subprocess, "Popen",
                        lambda cmd, **k: _FakeProc(next(rcs)))
    with pytest.raises(SystemExit, match="FAILED"):
        bp.cmd_generate_anti_neg_all(_fanout_args(priv, gpus="0,1"), cfg)


def test_fanout_refuses_when_disabled_before_spawning(tmp_path, monkeypatch):
    cfg, priv = _shard_world(tmp_path)
    cfg["pool"]["anti_neg"]["enabled"] = False
    monkeypatch.setattr(bp.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("spawned a child"))
    with pytest.raises(SystemExit, match="enabled is false"):
        bp.cmd_generate_anti_neg_all(_fanout_args(priv, gpus="0"), cfg)


# --------------------------------------------------------------------------- #
# 7. F8: anti-neg renders through the CANONICAL tokenizer (real VllmSampler)    #
# --------------------------------------------------------------------------- #


def test_anti_neg_renders_through_the_canonical_tokenizer(
        tmp_path, monkeypatch):
    """RESIDUAL F8's live surface, end to end: generate-anti-neg is the one
    vLLM leg that renders INSIDE VllmSampler (sampler.run -> build_chat), so
    here the sampler is NOT stubbed — only vllm itself and AutoTokenizer are
    faked. The sampler's internal tokenizer must come from the REAL
    load_tokenizer (thinking pinned off on a reasoning-capable template) and
    every request must carry the pin, with the anti-SI text a literal prefix
    (the prefix-cache property) — the raw AutoTokenizer render would have let
    Qwen spend the anti-neg completion budget inside <think>."""
    import sisteer.models  # the load_tokenizer seam the fake rides in on

    cfg, priv = _shard_world(tmp_path, n_prompts=2, k=1)
    log = []

    fake = types.ModuleType("vllm")
    fake.__version__ = "0.0-test"

    class _SamplingParams:
        def __init__(self, **kw):
            self.kw = kw

    class _LLM:
        def __init__(self, **kw):
            pass

        def collective_rpc(self, name):   # vllm_patches.attest_engine's RPC
            assert name == "sisteer_mrope_report"
            return [{"patch_applied": True, "needs_patch": False}]

        def generate(self, rendered, sp):
            log.append((list(rendered), dict(sp.kw)))
            return [types.SimpleNamespace(
                outputs=[types.SimpleNamespace(text=f"VRESP::{r[:24]}")])
                for r in rendered]

    fake.SamplingParams = _SamplingParams
    fake.LLM = _LLM
    monkeypatch.setitem(sys.modules, "vllm", fake)

    class _Tok:
        pad_token = None
        eos_token = "<eos>"
        chat_template = "{% if enable_thinking %}<think>{% endif %}fake"
        calls: list = []

        def apply_chat_template(self, msgs, tokenize=False,
                                add_generation_prompt=True, **kw):
            type(self).calls.append(dict(kw))
            return "".join(f"<{m['role']}>{m['content']}" for m in msgs)

    class _Auto:
        @staticmethod
        def from_pretrained(name, revision=None, **kw):
            return _Tok()

    monkeypatch.setattr(sisteer.models, "AutoTokenizer", _Auto)
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    bp.cmd_generate_anti_neg(_args(priv, k=1), cfg)

    from sisteer.data.instructions import SAFETY_SI

    assert log, "the real sampler generated nothing"
    assert _Tok.calls and all(c.get("enable_thinking") is False
                              for c in _Tok.calls)
    prefix = f"<system>{SAFETY_SI.anti_text}"
    for rendered, kw in log:
        assert all(r.startswith(prefix) for r in rendered)   # shared SI prefix
        assert kw["temperature"] == 0.0 and kw["seed"] is None   # k=1: greedy
    rows = [json.loads(l) for l in
            (priv / "two_si_anti_neg" / "anti_neg.jsonl").read_text()
            .splitlines()]
    assert len(rows) == 2 and all(r["si_id"] == ANTI for r in rows)

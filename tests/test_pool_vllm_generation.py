"""T15 — vLLM pool generation (engine ruling 2026-08-25).

CPU-only, offline, vLLM MOCKED (the package is deliberately absent from the
repo venv), NO harmful text — every prompt/response is an INVENTED synthetic
string (privacy rule). What the suite proves:

- engine selection: `--engine` flag > `pool.engine` config > the leg's
  historical default (hf for generate-pos/generate-benign, vllm for
  generate-anti-neg), and the hf branch still routes byte-for-byte as before;
- lazy import: an absent vLLM fails with a CLEAN, actionable error naming the
  install path and the hf fallback — and ONLY on the real engine=vllm
  generation path (planning / --dry-run never touch it);
- sampling parity: every vLLM request carries the HF-teacher parity set the
  vllm-generate-pos lane pinned — temperature (pool default 1.0), top_p 0.9
  (02c3c52), top_k 50 (bfdd5d8), per-candidate seed base+1000*ci (7931794) —
  and the greedy pass zeroes the lot;
- determinism + resume: candidate indexing is stable and contiguous from the
  plan's `start`, identical worlds produce identical request sequences, a
  resumed run regenerates NOTHING it already holds, and every written cell is
  verified to land exactly once (the sequential-path analogue of a shard-merge
  check — the pos leg deliberately has no shards to merge);
- provenance: every generation records engine + vLLM version + sampling params
  + model/revision in its run dir; a resume with different provenance refuses;
  `manifest` embeds the records, refuses an engine-mixed pool (unless
  --allow-engine-mix records the sanctioned contingency confound), and
  emit-pairs copies the block into pairs_meta.json;
- canonical rendering (F8): the vLLM legs render through the harness's OWN
  load_tokenizer + build_chat_prompt (enable_thinking pinned off, Gemma
  system-role fold), never the engine's raw AutoTokenizer, and the effective
  template kwargs + a rendered-prefix hash are recorded in the sidecar;
- torn-tail safety (F9): a killed append never permanently drops a candidate —
  the plan tracks the exact index SET, the torn tail is repaired under a lock
  before any append, and post-generation verification demands the EXACT
  expected (prompt, candidate) set, not just no-duplicates;
- sidecar completeness (F12): once anything is recorded, a rows-bearing leg
  (the labelled store included) with no or an invalid sidecar refuses the
  manifest; --allow-unrecorded-provenance records the confound instead;
- the pilot CLI (F10): generate-pos --max-prompts (length-representative
  slice) + --pilot-pairs-out (ungraded {prompt,pos,neg,instruction_id} pairs,
  never inside the repo) — the exact command line pod_pilot.sh step 5 runs.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_two_si_pool as bp  # noqa: E402
import sisteer.sampling.engine as E  # noqa: E402
from test_stage3_pool import (  # noqa: E402 — the shared synthetic worlds
    SI3,
    _phtest_csv,
    _seed_benign_run,
    _seed_pos_runs,
    _stage3_cfg,
    _store_row,
    _stub_generation,
    _write_jsonl,
)

IRR = "si_irrelevant"


# --------------------------------------------------------------------------- #
# fakes: a vllm module + a VllmSampler that log every request                   #
# --------------------------------------------------------------------------- #


class _FakeSamplingParams:
    def __init__(self, **kw):
        self.kw = kw


class _FakeLLM:
    def __init__(self, log):
        self.log = log

    def collective_rpc(self, name):   # vllm_patches.attest_engine's RPC
        assert name == "sisteer_mrope_report"
        return [{"patch_applied": True, "needs_patch": False}]

    def generate(self, rendered, sp):
        self.log.append((list(rendered), dict(sp.kw)))
        return [types.SimpleNamespace(
            outputs=[types.SimpleNamespace(text=f"VRESP::{r}")])
            for r in rendered]


class _FakeTok:
    """The canonical-render surface (F8): the vLLM legs now render through the
    REAL ``sisteer.models.load_tokenizer`` + ``build_chat_prompt`` pipeline
    (thinking pin, Gemma system-role fold), so the fake must quack like an
    AutoTokenizer — pad/eos tokens, a ``chat_template`` string (load_tokenizer
    pins ``enable_thinking: False`` when the template mentions it) and an
    ``apply_chat_template`` that RECORDS the kwargs it received. Renders flat
    so responses differ per condition (mirrors the hf stub's convention)."""

    def __init__(self, chat_template="{{fake}}", reject_system=False):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.chat_template = chat_template
        self.reject_system = reject_system
        self.template_calls: list[dict] = []

    def apply_chat_template(self, msgs, tokenize=False,
                            add_generation_prompt=True, **kw):
        self.template_calls.append(dict(kw))
        if self.reject_system and msgs and msgs[0]["role"] == "system":
            raise ValueError("System role not supported")   # the Gemma shape
        system = msgs[0]["content"][:12] if msgs[0]["role"] == "system" else "nosys"
        return f"{system}|{msgs[-1]['content']}"


def _install_fake_vllm(monkeypatch, log, version="0.0-test", tok=None):
    fake = types.ModuleType("vllm")
    fake.__version__ = version
    fake.SamplingParams = _FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake)

    # F8: the render tokenizer comes from sisteer.models.load_tokenizer, so
    # the fake is injected at AutoTokenizer.from_pretrained — the REAL
    # load_tokenizer pipeline (pad default, enable_thinking pin) runs on it.
    ftok = tok if tok is not None else _FakeTok()

    class _FakeAuto:
        @staticmethod
        def from_pretrained(name, revision=None, **kw):
            return ftok

    monkeypatch.setattr("sisteer.models.AutoTokenizer", _FakeAuto)

    class _FakeSampler:
        built = 0

        def __init__(self, model, *, revision=None, dtype=None,
                     max_model_len=16384, gpu_memory_utilization=0.90,
                     seed=0, max_new_tokens=512, enforce_eager=False):
            type(self).built += 1
            self.tok = _FakeTok()      # the engine's RAW tokenizer, unused by
            self.llm = _FakeLLM(log)   # the render path since F8
            self.seed = seed
            self.max_new_tokens = max_new_tokens

    monkeypatch.setattr(E, "VllmSampler", _FakeSampler)
    return _FakeSampler


def _pos_args(priv, **kw):
    base = dict(private_root=str(priv), dry_run=False, batch_size=4,
                si=IRR, candidates=2, engine="vllm", seed=0,
                max_model_len=16384, gpu_memory_utilization=0.90,
                enforce_eager=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _irr_rows(priv, cfg):
    path = priv / bp._pos_runs(cfg)[IRR] / "bon.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _sidecar(priv, cfg, si=IRR):
    return json.loads(
        (priv / bp._pos_runs(cfg)[si] / bp.GEN_PROVENANCE_NAME).read_text())


def _write_prov(d: Path, engine: str, **over):
    """A hand-written provenance sidecar for manifest-side tests."""
    prov = {"engine": engine, "leg": "test-leg", "model": "m",
            "model_revision": "r0",
            "vllm_version": "0.0-test" if engine == "vllm" else None,
            "sampling_params": {"temperature": 1.0, "top_p": 0.9, "top_k": 50}}
    prov.update(over)
    d.mkdir(parents=True, exist_ok=True)
    (d / bp.GEN_PROVENANCE_NAME).write_text(json.dumps(prov))
    return prov


# --------------------------------------------------------------------------- #
# engine selection + lazy import                                                #
# --------------------------------------------------------------------------- #


def test_absent_vllm_fails_cleanly_naming_the_install_and_the_fallback(
        tmp_path, monkeypatch):
    cfg = _stage3_cfg(tmp_path, n_prompts=1)
    monkeypatch.setitem(sys.modules, "vllm", None)   # certain-absent
    with pytest.raises(SystemExit) as exc:
        bp.cmd_generate_pos(_pos_args(tmp_path / "priv"), cfg)
    msg = str(exc.value)
    assert "vllm" in msg and "--engine hf" in msg
    assert "vllm-requirements" in msg                # names the install path
    # failed BEFORE any row or provenance write (the empty 0700 run dir that
    # private_run_dir pre-creates is content-free and harmless)
    run_dir = tmp_path / "priv" / bp._pos_runs(cfg)[IRR]
    assert not list(run_dir.glob("*.jsonl"))
    assert not (run_dir / bp.GEN_PROVENANCE_NAME).exists()


def test_dry_run_and_hf_engine_never_import_vllm(tmp_path, monkeypatch):
    cfg = _stage3_cfg(tmp_path, n_prompts=1)
    monkeypatch.setitem(sys.modules, "vllm", None)
    # planning with engine=vllm is CPU-only — no vllm import
    bp.cmd_generate_pos(_pos_args(tmp_path / "priv", dry_run=True), cfg)
    # the hf engine generates rows without ever touching vllm
    _stub_generation(monkeypatch)
    bp.cmd_generate_pos(_pos_args(tmp_path / "priv", engine="hf"), cfg)
    rows = _irr_rows(tmp_path / "priv", cfg)
    assert len(rows) == 2 and {r["candidate_idx"] for r in rows} == {0, 1}


def test_engine_resolution_flag_beats_config_beats_default(
        tmp_path, monkeypatch):
    cfg = _stage3_cfg(tmp_path, n_prompts=1)
    # default (no flag attr, no config key) -> hf: the pre-ruling route
    calls = _stub_generation(monkeypatch)
    bp.cmd_generate_pos(types.SimpleNamespace(
        private_root=str(tmp_path / "priv"), dry_run=False, batch_size=4,
        si=IRR, candidates=2), cfg)
    assert calls["rendered"], "default engine must still route hf"
    assert _sidecar(tmp_path / "priv", cfg)["engine"] == "hf"

    # config pool.engine=vllm routes vllm (proved by the vllm-only refusal)
    cfg2 = _stage3_cfg(tmp_path / "w2", n_prompts=1)
    cfg2["pool"]["engine"] = "vllm"
    monkeypatch.setitem(sys.modules, "vllm", None)
    with pytest.raises(SystemExit, match="engine=vllm"):
        bp.cmd_generate_pos(types.SimpleNamespace(
            private_root=str(tmp_path / "w2priv"), dry_run=False,
            batch_size=4, si=IRR, candidates=2), cfg2)

    # an explicit --engine hf overrides the config's vllm
    calls2 = _stub_generation(monkeypatch)
    bp.cmd_generate_pos(types.SimpleNamespace(
        private_root=str(tmp_path / "w2priv"), dry_run=False, batch_size=4,
        si=IRR, candidates=2, engine="hf"), cfg2)
    assert calls2["rendered"]

    # and an unknown engine refuses loudly
    with pytest.raises(SystemExit, match="unknown engine"):
        bp.cmd_generate_pos(_pos_args(tmp_path / "priv", engine="tgi"), cfg)


def test_pos_declaration_key_selects_and_a_contradiction_refuses(
        tmp_path, monkeypatch):
    """pool.pos_generation_engine (the sibling of the ed01306 neg-side key):
    with no flag and no pool.engine it SELECTS the engine — the stage3 base
    config declares vllm and sets no pool.engine, and must run vllm here
    exactly as on the branch that introduced the key — and an explicit
    selection that contradicts it REFUSES (config and code disagreeing about
    what produced a pool is the +25.8pp ambiguity; run the hf contingency by
    flipping the declaration, not by silently overriding it)."""
    cfg = _stage3_cfg(tmp_path, n_prompts=1)
    cfg["pool"]["pos_generation_engine"] = "vllm"
    # declaration alone routes vllm (proved by the vllm-only refusal)
    monkeypatch.setitem(sys.modules, "vllm", None)
    with pytest.raises(SystemExit, match="engine=vllm"):
        bp.cmd_generate_pos(types.SimpleNamespace(
            private_root=str(tmp_path / "priv"), dry_run=False,
            batch_size=4, si=IRR, candidates=2), cfg)
    # --engine hf against a vllm declaration: refused, naming the key
    with pytest.raises(SystemExit, match="pos_generation_engine"):
        bp.cmd_generate_pos(_pos_args(tmp_path / "priv", engine="hf"), cfg)
    # pool.engine=hf against a vllm declaration: refused the same way
    cfg["pool"]["engine"] = "hf"
    with pytest.raises(SystemExit, match="pos_generation_engine"):
        bp.cmd_generate_pos(types.SimpleNamespace(
            private_root=str(tmp_path / "priv"), dry_run=False,
            batch_size=4, si=IRR, candidates=2), cfg)
    # a CONSISTENT hf declaration runs the contingency, recorded as ever
    cfg["pool"]["pos_generation_engine"] = "hf"
    calls = _stub_generation(monkeypatch)
    bp.cmd_generate_pos(types.SimpleNamespace(
        private_root=str(tmp_path / "priv"), dry_run=False, batch_size=4,
        si=IRR, candidates=2), cfg)
    assert calls["rendered"]
    assert _sidecar(tmp_path / "priv", cfg)["engine"] == "hf"


# --------------------------------------------------------------------------- #
# sampling parity: the pinned HF-teacher values ride every request              #
# --------------------------------------------------------------------------- #


def test_sampled_requests_carry_the_pinned_parity_values(tmp_path, monkeypatch):
    cfg = _stage3_cfg(tmp_path, n_prompts=1)
    priv, log = tmp_path / "priv", []
    _install_fake_vllm(monkeypatch, log)
    bp.cmd_generate_pos(_pos_args(priv, candidates=2), cfg)

    cap = int(cfg["completion_tokens"])
    assert [kw for _r, kw in log] == [
        # candidate ci draws at base_seed + 1000*ci — resume-stable (7931794)
        {"temperature": 1.0, "top_p": 0.9, "top_k": 50,
         "max_tokens": cap, "seed": 0},
        {"temperature": 1.0, "top_p": 0.9, "top_k": 50,
         "max_tokens": cap, "seed": 1000},
    ]
    rows = _irr_rows(priv, cfg)
    assert {r["candidate_idx"] for r in rows} == {0, 1}
    assert all(r["si_id"] == IRR and r["score"] is None
               and r["graded"] is False for r in rows)
    # schema parity with the hf path's rows: the store's group label rides
    assert all(r["category_group"] == "held_in" for r in rows)


def test_config_overrides_flow_into_the_request(tmp_path, monkeypatch):
    cfg = _stage3_cfg(tmp_path, n_prompts=1)
    # a per-model config pins ITS teacher's values; the mechanism must obey
    cfg["pool"]["pos_sampling_temperature"] = 0.7
    cfg["pool"]["pos_sampling_top_p"] = 0.95
    cfg["pool"]["pos_sampling_top_k"] = 20
    log = []
    _install_fake_vllm(monkeypatch, log)
    bp.cmd_generate_pos(_pos_args(tmp_path / "priv", candidates=2, seed=5), cfg)
    assert log[0][1]["temperature"] == 0.7
    assert log[0][1]["top_p"] == 0.95 and log[0][1]["top_k"] == 20
    assert [kw["seed"] for _r, kw in log] == [5, 1005]


def test_configured_k_forces_a_sampled_candidate_zero(tmp_path, monkeypatch):
    # cost-staged rounds: --candidates 1 under the config's K=8 scheme must
    # DRAW candidate 0, never the greedy mode (the GenSpec.sample rationale)
    cfg = _stage3_cfg(tmp_path, n_prompts=1)
    assert int(cfg["pool"]["pos_candidates"][IRR]) > 1     # the shipped scheme
    log = []
    _install_fake_vllm(monkeypatch, log)
    bp.cmd_generate_pos(_pos_args(tmp_path / "priv", candidates=1), cfg)
    assert log[0][1]["temperature"] == 1.0 and log[0][1]["seed"] == 0


def test_greedy_single_draw_zeroes_the_sampler(tmp_path, monkeypatch):
    cfg = _stage3_cfg(tmp_path, n_prompts=1)
    cfg["pool"]["pos_candidates"] = {IRR: 1}               # a true K=1 scheme
    log = []
    _install_fake_vllm(monkeypatch, log)
    bp.cmd_generate_pos(_pos_args(tmp_path / "priv", candidates=1), cfg)
    assert [kw for _r, kw in log] == [
        {"temperature": 0.0, "top_p": 1.0, "top_k": -1,
         "max_tokens": int(cfg["completion_tokens"]), "seed": None}]


# --------------------------------------------------------------------------- #
# determinism + resume                                                          #
# --------------------------------------------------------------------------- #


def test_candidate_indexing_resumes_from_the_plans_start(tmp_path, monkeypatch):
    cfg = _stage3_cfg(tmp_path, n_prompts=1)
    priv = tmp_path / "priv"
    run = bp._pos_runs(cfg)[IRR]
    # an existing (ungraded) candidate 0 on disk: the plan's start moves to 1
    _write_jsonl(priv / run / "bon.jsonl", [
        _store_row("bon", "g0", 0, 0, "prompted", "SYNTHETIC harmful 0",
                   "SYNTHETIC prior candidate 0", None, si_id=IRR, cidx=0)])
    log = []
    _install_fake_vllm(monkeypatch, log)
    bp.cmd_generate_pos(_pos_args(priv, candidates=3), cfg)
    # only the missing ABSOLUTE indices {1, 2}, each at ITS OWN seed — a
    # resumed cell draws exactly the seed a fresh run would have given it
    assert [kw["seed"] for _r, kw in log] == [1000, 2000]
    assert sorted(r["candidate_idx"] for r in _irr_rows(priv, cfg)) == [0, 1, 2]


def test_identical_worlds_generate_identical_request_sequences(
        tmp_path, monkeypatch):
    seqs = []
    for w in ("w1", "w2"):
        cfg = _stage3_cfg(tmp_path / w, n_prompts=2)
        log = []
        _install_fake_vllm(monkeypatch, log)
        bp.cmd_generate_pos(_pos_args(tmp_path / w / "priv", candidates=2), cfg)
        seqs.append(log)
    assert seqs[0] == seqs[1]                              # byte-equal plans


def test_resume_regenerates_nothing_and_a_k_raise_generates_the_delta(
        tmp_path, monkeypatch):
    cfg = _stage3_cfg(tmp_path, n_prompts=2)
    priv, log = tmp_path / "priv", []
    sampler_cls = _install_fake_vllm(monkeypatch, log)
    args = _pos_args(priv, candidates=2)
    bp.cmd_generate_pos(args, cfg)
    n_rows, n_reqs = len(_irr_rows(priv, cfg)), len(log)
    assert n_rows == 4                                     # 2 prompts x K=2

    # full resume: nothing to do — no engine built, no request, no new row
    bp.cmd_generate_pos(args, cfg)
    assert len(log) == n_reqs and len(_irr_rows(priv, cfg)) == n_rows
    assert sampler_cls.built == 1

    # raising K generates ONLY candidate 2, at its own absolute seed
    args.candidates = 3
    bp.cmd_generate_pos(args, cfg)
    assert [kw["seed"] for _r, kw in log[n_reqs:]] == [2000]
    rows = _irr_rows(priv, cfg)
    assert len(rows) == 6
    cells = {(r["goal_id"], r["candidate_idx"]) for r in rows}
    assert len(cells) == 6                                 # no duplicate cell


def test_every_cell_lands_exactly_once_and_duplicates_refuse(tmp_path):
    # the sequential-path analogue of a shard-merge verification: the pos leg
    # deliberately has no shards (one engine covers it at vLLM speed), so the
    # verified property is per-cell uniqueness of the written file
    ok = tmp_path / "ok.jsonl"
    row = {"attack": "bon", "goal_id": "g0", "sample_idx": 0, "trial": 0,
           "si_id": IRR, "candidate_idx": 0}
    _write_jsonl(ok, [row, {**row, "candidate_idx": 1}])
    bp._verify_pos_cells(ok, tag="t")                      # unique: passes
    dup = tmp_path / "dup.jsonl"
    _write_jsonl(dup, [row, dict(row)])
    with pytest.raises(SystemExit, match="duplicate generated cell"):
        bp._verify_pos_cells(dup, tag="t")


def test_non_numeric_variant_key_refuses_before_generating(tmp_path):
    with pytest.raises(SystemExit, match="non-numeric variant_key"):
        bp._require_engine_keyable(("bon", "g0", "orderhash"), tag="t")


# --------------------------------------------------------------------------- #
# provenance: recorded on write, strict on resume                               #
# --------------------------------------------------------------------------- #


def test_provenance_sidecar_records_engine_version_params_model(
        tmp_path, monkeypatch):
    cfg = _stage3_cfg(tmp_path, n_prompts=1)
    cfg["model_revision"] = "deadbeef"
    priv, log = tmp_path / "priv", []
    _install_fake_vllm(monkeypatch, log, version="9.9-fake")
    bp.cmd_generate_pos(_pos_args(priv, candidates=2), cfg)
    prov = _sidecar(priv, cfg)
    assert prov["engine"] == "vllm"
    assert prov["vllm_version"] == "9.9-fake"
    assert prov["model"] == cfg["model"]
    assert prov["model_revision"] == "deadbeef"
    sp = prov["sampling_params"]
    assert (sp["temperature"], sp["top_p"], sp["top_k"]) == (1.0, 0.9, 50)
    assert sp["base_seed"] == 0 and sp["sample"] is True
    assert prov["info"]["k"] == 2


def test_resume_with_different_params_or_engine_fails_loudly(
        tmp_path, monkeypatch):
    cfg = _stage3_cfg(tmp_path, n_prompts=1)
    priv, log = tmp_path / "priv", []
    _install_fake_vllm(monkeypatch, log)
    bp.cmd_generate_pos(_pos_args(priv, candidates=2), cfg)

    # a K raise is work-to-do again; a drifted seed changes the candidate
    # distribution -> the resume must refuse, not silently mix
    with pytest.raises(SystemExit, match="REFUSING to resume"):
        bp.cmd_generate_pos(_pos_args(priv, candidates=3, seed=7), cfg)

    # and a flipped ENGINE on the same run dir refuses on the hf branch too
    _stub_generation(monkeypatch)
    with pytest.raises(SystemExit, match="REFUSING to resume"):
        bp.cmd_generate_pos(_pos_args(priv, candidates=3, engine="hf"), cfg)

    # the sanctioned delta workflow (same engine, same params, higher K) rides
    bp.cmd_generate_pos(_pos_args(priv, candidates=3), cfg)
    assert sorted(r["candidate_idx"] for r in _irr_rows(priv, cfg)) == [0, 1, 2]


def test_legacy_rows_without_a_sidecar_warn_and_are_recorded(
        tmp_path, monkeypatch, capsys):
    cfg = _stage3_cfg(tmp_path, n_prompts=1)
    priv = tmp_path / "priv"
    run = bp._pos_runs(cfg)[IRR]
    _write_jsonl(priv / run / "bon.jsonl", [
        _store_row("bon", "g0", 0, 0, "prompted", "SYNTHETIC harmful 0",
                   "SYNTHETIC legacy candidate", None, si_id=IRR, cidx=0)])
    log = []
    _install_fake_vllm(monkeypatch, log)
    bp.cmd_generate_pos(_pos_args(priv, candidates=2), cfg)
    assert "UNRECORDED" in _sidecar(priv, cfg)["note"]
    assert "no recorded engine provenance" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# manifest: fields embedded, engine mixing refused                              #
# --------------------------------------------------------------------------- #


def _manifest_world(tmp_path):
    cfg = _stage3_cfg(tmp_path)
    priv = tmp_path / "priv"
    _seed_pos_runs(priv, cfg)
    _seed_benign_run(priv, cfg)
    return cfg, priv


def _prov_all_legs(cfg, priv, engines=None):
    """Sidecars for EVERY rows-bearing leg of `_manifest_world` (two pos runs,
    the benign run, the labelled store) — required since F12: once anything is
    recorded, a rows-bearing leg with no sidecar refuses. ``engines``
    overrides per leg name."""
    engines = engines or {}
    runs = bp._pos_runs(cfg)
    for si, run in runs.items():
        _write_prov(priv / run, engines.get(f"pos:{si}", "vllm"))
    _write_prov(priv / bp._benign_cfg(cfg)[1],
                engines.get("benign", "vllm"))
    _write_prov(Path(cfg["pool"]["store_dir"]),
                engines.get("store", "vllm"))


def test_manifest_embeds_engine_version_params_per_leg(tmp_path):
    cfg, priv = _manifest_world(tmp_path)
    _prov_all_legs(cfg, priv)
    bp.cmd_manifest(types.SimpleNamespace(private_root=str(priv), out=None),
                    cfg)
    m = json.loads(Path(cfg["pool_manifest"]).read_text())
    gp = m["generation_provenance"]
    assert gp["engines"] == ["vllm"]
    assert set(gp["legs"]) == {"pos:safety_refusal", "pos:si_constitution",
                               f"pos:{IRR}", "benign", "store"}
    leg = gp["legs"][f"pos:{IRR}"]
    assert leg["engine"] == "vllm" and leg["vllm_version"] == "0.0-test"
    assert leg["sampling_params"]["top_k"] == 50
    assert leg["model"] == "m" and leg["model_revision"] == "r0"
    # the embedded hash covers the block (write_manifest verified it)
    assert m["manifest_sha256"]


def test_manifest_without_sidecars_stays_byte_compatible(tmp_path):
    cfg, priv = _manifest_world(tmp_path)
    bp.cmd_manifest(types.SimpleNamespace(private_root=str(priv), out=None),
                    cfg)
    m = json.loads(Path(cfg["pool_manifest"]).read_text())
    assert "generation_provenance" not in m       # nothing recorded, nothing added


def test_manifest_refuses_an_engine_mixed_pool_unless_recorded(tmp_path):
    cfg, priv = _manifest_world(tmp_path)
    _prov_all_legs(cfg, priv, engines={f"pos:{IRR}": "hf"})
    with pytest.raises(SystemExit, match="mix engines"):
        bp.cmd_manifest(types.SimpleNamespace(private_root=str(priv),
                                              out=None), cfg)
    assert not Path(cfg["pool_manifest"]).exists()     # nothing frozen

    # --allow-engine-mix records the confound instead
    bp.cmd_manifest(types.SimpleNamespace(private_root=str(priv), out=None,
                                          allow_engine_mix=True), cfg)
    gp = json.loads(Path(cfg["pool_manifest"]).read_text())[
        "generation_provenance"]
    assert gp["engines"] == ["hf", "vllm"]
    assert gp["engine_mix_allowed"] is True and "confound" in gp["engine_mix_note"]


def test_manifest_refuses_a_rows_bearing_leg_with_no_sidecar(tmp_path):
    """F12: in a config that DECLARES its engine (every sweep config), a leg
    holding generation rows but no sidecar may not silently sit out the
    engine check — the labelled store (whose sample-all writer predates
    sidecars) was exactly the leg that let an HF store freeze
    engines:["vllm"]."""
    cfg, priv = _manifest_world(tmp_path)
    cfg["pool"]["engine"] = "vllm"          # the sweep configs' declaration
    runs = bp._pos_runs(cfg)
    _write_prov(priv / runs["safety_refusal"], "vllm")
    _write_prov(priv / runs["si_constitution"], "vllm")
    _write_prov(priv / runs[IRR], "vllm")
    # benign run + labelled store: generation rows, NO sidecar
    with pytest.raises(SystemExit, match="NO provenance sidecar"):
        bp.cmd_manifest(types.SimpleNamespace(private_root=str(priv),
                                              out=None), cfg)
    assert not Path(cfg["pool_manifest"]).exists()     # nothing frozen

    # --allow-unrecorded-provenance records the confound instead of omitting
    bp.cmd_manifest(types.SimpleNamespace(
        private_root=str(priv), out=None,
        allow_unrecorded_provenance=True), cfg)
    gp = json.loads(Path(cfg["pool_manifest"]).read_text())[
        "generation_provenance"]
    assert gp["unrecorded_legs"] == ["benign", "store"]
    assert gp["unrecorded_provenance_allowed"] is True
    assert "confound" in gp["unrecorded_note"].lower()
    assert gp["legs"]["store"]["engine"] == "unrecorded"
    assert gp["legs"]["benign"]["engine"] == "unrecorded"
    assert gp["engines"] == ["vllm"]      # known engines only, mix rule intact


def test_manifest_legacy_config_tolerates_unrecorded_legs(tmp_path):
    """A config with NO declared pool.engine is the pre-ruling world: its
    fixture-era rows predate sidecars entirely, so a partially recorded world
    freezes with only the recorded legs embedded (the pre-F12 behaviour) —
    the strict completeness rule binds only where an engine is declared."""
    cfg, priv = _manifest_world(tmp_path)
    assert not cfg["pool"].get("engine")             # the legacy world
    _write_prov(priv / bp._pos_runs(cfg)[IRR], "hf")
    bp.cmd_manifest(types.SimpleNamespace(private_root=str(priv), out=None),
                    cfg)
    gp = json.loads(Path(cfg["pool_manifest"]).read_text())[
        "generation_provenance"]
    assert set(gp["legs"]) == {f"pos:{IRR}"}
    assert gp["engines"] == ["hf"] and "unrecorded_legs" not in gp


def test_manifest_refuses_a_schema_invalid_sidecar_even_with_the_flag(
        tmp_path):
    cfg, priv = _manifest_world(tmp_path)
    _prov_all_legs(cfg, priv)
    store = Path(cfg["pool"]["store_dir"])
    # a present-but-partial record is corruption, not absence: no flag helps
    (store / bp.GEN_PROVENANCE_NAME).write_text(json.dumps({"engine": "vllm"}))
    with pytest.raises(SystemExit, match="schema-invalid"):
        bp.cmd_manifest(types.SimpleNamespace(
            private_root=str(priv), out=None,
            allow_unrecorded_provenance=True), cfg)
    (store / bp.GEN_PROVENANCE_NAME).write_text("{not json")
    with pytest.raises(SystemExit, match="unreadable"):
        bp.cmd_manifest(types.SimpleNamespace(
            private_root=str(priv), out=None,
            allow_unrecorded_provenance=True), cfg)


def test_emit_pairs_sidecar_carries_the_provenance(tmp_path):
    cfg, priv = _manifest_world(tmp_path)
    _prov_all_legs(cfg, priv)
    bp.cmd_manifest(types.SimpleNamespace(private_root=str(priv), out=None),
                    cfg)
    out = tmp_path / "pairs_out"
    bp.cmd_emit_pairs(types.SimpleNamespace(
        manifest=None, private_root=str(priv), out_dir=str(out),
        dry_run=False), cfg)
    meta = json.loads((out / bp.PAIRS_META_NAME).read_text())
    assert meta["generation_provenance"]["engines"] == ["vllm"]
    assert set(meta["generation_provenance"]["legs"]) == {
        "pos:safety_refusal", "pos:si_constitution", f"pos:{IRR}",
        "benign", "store"}


# --------------------------------------------------------------------------- #
# the other two legs: benign (both engines) and anti-neg (vllm-only)            #
# --------------------------------------------------------------------------- #


def test_generate_benign_vllm_is_greedy_resumable_and_schema_identical(
        tmp_path, monkeypatch):
    cfg = _stage3_cfg(tmp_path)
    csv, _prompts = _phtest_csv(tmp_path)
    priv, log = tmp_path / "priv", []
    _install_fake_vllm(monkeypatch, log)
    args = types.SimpleNamespace(private_root=str(priv), dry_run=False,
                                 batch_size=4, phtest_path=csv,
                                 engine="vllm", seed=0)
    bp.cmd_generate_benign(args, cfg)

    path = priv / bp._benign_cfg(cfg)[1] / "phtest.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(rows) == 8                       # 2 prompts x (none + 3 SIs)
    assert all(r["xstest_label"] is None and r["score"] is None for r in rows)
    assert {r.get("si_id") for r in rows if r.get("si_id")} == set(SI3)
    assert all(r["candidate_idx"] == 0 for r in rows if r.get("si_id"))
    # GREEDY parity with the hf generate_batch default: one sampler config
    assert all(kw == {"temperature": 0.0,
                      "max_tokens": int(cfg["completion_tokens"])}
               for _r, kw in log)
    prov = json.loads((path.parent / bp.GEN_PROVENANCE_NAME).read_text())
    assert prov["engine"] == "vllm" and prov["sampling_params"]["greedy"]

    # resume: a second run regenerates nothing
    n = len(log)
    bp.cmd_generate_benign(args, cfg)
    assert len(log) == n
    assert len(path.read_text().splitlines()) == 8


def test_generate_anti_neg_refuses_the_hf_engine(tmp_path):
    cfg = _stage3_cfg(tmp_path)
    with pytest.raises(SystemExit, match="anti-neg"):
        bp.cmd_generate_anti_neg(types.SimpleNamespace(engine="hf"), cfg)
    with pytest.raises(SystemExit, match="anti-neg"):
        bp.cmd_generate_anti_neg_all(types.SimpleNamespace(engine="hf"), cfg)


# --------------------------------------------------------------------------- #
# F8: vLLM prompts render through the harness's canonical tokenizer             #
# --------------------------------------------------------------------------- #


def test_vllm_render_uses_the_canonical_tokenizer_and_pins_thinking_off(
        tmp_path, monkeypatch):
    """The engine's raw AutoTokenizer render let a reasoning-capable template
    (Qwen3) burn the completion budget inside <think>; the fix routes every
    vLLM render through the REAL load_tokenizer (which pins
    enable_thinking=False) + build_chat_prompt."""
    cfg = _stage3_cfg(tmp_path, n_prompts=1)
    priv, log = tmp_path / "priv", []
    ftok = _FakeTok(
        chat_template="{% if enable_thinking %}<think>{% endif %}fake")
    _install_fake_vllm(monkeypatch, log, tok=ftok)
    bp.cmd_generate_pos(_pos_args(priv, candidates=2), cfg)

    # every render (the provenance prefix probe included) forwarded the pin
    assert ftok.template_calls
    assert all(c.get("enable_thinking") is False for c in ftok.template_calls)
    # the requests were rendered by the canonical path (system-turn render)
    assert all("|" in r for rendered, _kw in log for r in rendered)
    # and the sidecar records the effective render, auditable after the fact
    pr = _sidecar(priv, cfg)["prompt_render"]
    assert pr["template_kwargs"] == {"enable_thinking": False}
    assert "load_tokenizer" in pr["renderer"]
    assert set(pr["rendered_prefix_sha256"]) == {IRR}


def test_vllm_render_folds_the_system_role_when_the_template_rejects_it(
        tmp_path, monkeypatch):
    """The Gemma shape: the template raises on a system role, and
    apply_chat_template_folding folds the SI into the first user turn — the
    raw engine render would have crashed (or dropped the SI)."""
    from sisteer.eval.jailbreak import resolve_instruction

    cfg = _stage3_cfg(tmp_path, n_prompts=1)
    priv, log = tmp_path / "priv", []
    _install_fake_vllm(monkeypatch, log, tok=_FakeTok(reject_system=True))
    bp.cmd_generate_pos(_pos_args(priv, candidates=1), cfg)

    rendered = log[0][0][0]
    assert rendered.startswith("nosys|")            # no system turn survived
    si_head = resolve_instruction(IRR).text.strip()[:40]
    assert si_head in rendered                      # the SI rides in the user turn
    assert "SYNTHETIC attack prompt" in rendered    # so does the prompt


def test_resume_with_a_drifted_prompt_render_refuses(tmp_path, monkeypatch):
    cfg = _stage3_cfg(tmp_path, n_prompts=1)
    priv, log = tmp_path / "priv", []
    _install_fake_vllm(monkeypatch, log,
                       tok=_FakeTok(chat_template="enable_thinking aware"))
    bp.cmd_generate_pos(_pos_args(priv, candidates=2), cfg)

    # same engine, same sampler, higher K — but the effective render changed
    # (the thinking pin vanished): appending would mix render distributions
    _install_fake_vllm(monkeypatch, log, tok=_FakeTok())
    with pytest.raises(SystemExit, match="REFUSING to resume"):
        bp.cmd_generate_pos(_pos_args(priv, candidates=3), cfg)


def test_generate_benign_vllm_renders_through_the_canonical_tokenizer(
        tmp_path, monkeypatch):
    cfg = _stage3_cfg(tmp_path)
    csv, _prompts = _phtest_csv(tmp_path)
    priv, log = tmp_path / "priv", []
    ftok = _FakeTok(chat_template="enable_thinking aware")
    _install_fake_vllm(monkeypatch, log, tok=ftok)
    bp.cmd_generate_benign(types.SimpleNamespace(
        private_root=str(priv), dry_run=False, batch_size=4,
        phtest_path=csv, engine="vllm", seed=0), cfg)
    assert ftok.template_calls
    assert all(c.get("enable_thinking") is False for c in ftok.template_calls)
    prov = json.loads((priv / bp._benign_cfg(cfg)[1]
                       / bp.GEN_PROVENANCE_NAME).read_text())
    pr = prov["prompt_render"]
    assert pr["template_kwargs"] == {"enable_thinking": False}
    # one auditable prefix hash per condition: none + every SI
    assert set(pr["rendered_prefix_sha256"]) == {"none", *SI3}


# --------------------------------------------------------------------------- #
# F9: a torn tail never permanently drops a candidate                           #
# --------------------------------------------------------------------------- #


def test_torn_candidate_zero_is_regenerated_on_resume(tmp_path, monkeypatch):
    """The kill -> torn candidate 0 -> resume world: the torn line left the
    file holding candidates {1} with a HOLE at 0. The old plan
    (max(existing)+1) masked the hole forever while the duplicates-only
    verification passed; the fix tracks the exact index SET, regenerates
    candidate 0 (at ITS OWN seed) and verifies the exact expected set."""
    cfg = _stage3_cfg(tmp_path, n_prompts=1)
    priv = tmp_path / "priv"
    path = priv / bp._pos_runs(cfg)[IRR] / "bon.jsonl"
    cand1 = _store_row("bon", "g0", 0, 0, "prompted", "SYNTHETIC harmful 0",
                       "SYNTHETIC surviving candidate 1", None,
                       si_id=IRR, cidx=1)
    torn = json.dumps(dict(cand1, candidate_idx=0))[:47]   # cut mid-JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{torn}\n{json.dumps(cand1)}\n")

    log = []
    _install_fake_vllm(monkeypatch, log)
    bp.cmd_generate_pos(_pos_args(priv, candidates=2), cfg)

    # ONLY candidate 0 was generated, at the seed a fresh run would give it
    assert [kw["seed"] for _r, kw in log] == [0]
    # the interior torn line stays (readers skip it) — read tolerantly, the
    # way every production reader does
    rows = bp._read_jsonl_rows(path)
    assert sorted(r["candidate_idx"] for r in rows) == [0, 1]
    # a plain resume now regenerates nothing
    n = len(log)
    bp.cmd_generate_pos(_pos_args(priv, candidates=2), cfg)
    assert len(log) == n


def test_torn_unterminated_tail_is_repaired_before_any_append(
        tmp_path, monkeypatch):
    """A kill mid-append leaves an UNTERMINATED fragment at EOF; appending
    after it would merge the next row into one malformed line (losing BOTH).
    The repair truncates the fragment under a lock before generation."""
    cfg = _stage3_cfg(tmp_path, n_prompts=1)
    priv = tmp_path / "priv"
    path = priv / bp._pos_runs(cfg)[IRR] / "bon.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"attack": "bon", "goal_id": "g0", "cand')   # no newline

    log = []
    _install_fake_vllm(monkeypatch, log)
    bp.cmd_generate_pos(_pos_args(priv, candidates=2), cfg)

    lines = path.read_text().splitlines()
    for line in lines:
        json.loads(line)                        # every surviving line parses
    rows = _irr_rows(priv, cfg)
    assert sorted(r["candidate_idx"] for r in rows) == [0, 1]
    assert [kw["seed"] for _r, kw in log] == [0, 1000]


def test_repair_torn_tail_unit(tmp_path):
    p = tmp_path / "f.jsonl"
    row = json.dumps({"a": 1})

    # a complete final row missing only its newline is TERMINATED, not lost
    p.write_text(f'{{"b": 2}}\n{row}')
    assert bp._repair_torn_jsonl_tail(p, tag="t") is True
    assert p.read_text() == f'{{"b": 2}}\n{row}\n'

    # a torn fragment is truncated; the intact rows survive
    p.write_text(f'{row}\n{{"c": 3, "tor')
    assert bp._repair_torn_jsonl_tail(p, tag="t") is True
    assert p.read_text() == f"{row}\n"

    # a clean file (and a repaired one) is untouched
    assert bp._repair_torn_jsonl_tail(p, tag="t") is False
    assert bp._repair_torn_jsonl_tail(tmp_path / "absent.jsonl",
                                      tag="t") is False


def test_verify_pos_cells_exact_expected_set_fails_on_a_hole(tmp_path):
    f = tmp_path / "f.jsonl"
    row = {"attack": "bon", "goal_id": "g0", "sample_idx": 0, "trial": 0,
           "si_id": IRR, "candidate_idx": 1}
    _write_jsonl(f, [row])
    pid = ("bon", "g0", "0:0")
    with pytest.raises(SystemExit, match="HOLE"):
        bp._verify_pos_cells(f, tag="t", si=IRR, expected={pid: {0, 1}})
    bp._verify_pos_cells(f, tag="t", si=IRR, expected={pid: {1}})  # exact: ok
    with pytest.raises(SystemExit, match="do NOT match"):
        bp._verify_pos_cells(f, tag="t", si=IRR, expected={pid: {2}})


# --------------------------------------------------------------------------- #
# F10: the pilot's generate-pos CLI (--max-prompts / --pilot-pairs-out)         #
# --------------------------------------------------------------------------- #


def test_generate_pos_pilot_cli_signature_parses_and_dispatches(
        tmp_path, monkeypatch):
    """pod_pilot.sh step 5 invokes EXACTLY this command line (FIX-E's
    contract) — it must parse and hand both pilot knobs to the command."""
    import yaml

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(_stage3_cfg(tmp_path)))
    seen = {}
    monkeypatch.setattr(bp, "cmd_generate_pos",
                        lambda args, cfg: seen.update(args=args, cfg=cfg))
    monkeypatch.setattr(sys, "argv", [
        "build_two_si_pool.py", "generate-pos",
        "--config", str(cfg_path), "--si", "si_constitution",
        "--private-root", str(tmp_path / "store"),
        "--max-prompts", "192",
        "--pilot-pairs-out", str(tmp_path / "pool" / "pilot_pairs.jsonl")])
    bp.main()
    a = seen["args"]
    assert a.si == "si_constitution"
    assert a.max_prompts == 192
    assert a.pilot_pairs_out.endswith("pilot_pairs.jsonl")
    assert seen["cfg"]["pool"]["static_families"] == ["bon"]


def test_length_representative_slice_spans_the_length_spectrum():
    none_prompt = {("bon", f"g{i}", "0:0"): {"prompt": "x" * n}
                   for i, n in enumerate([5, 50, 500, 5000, 50000])}
    pids = list(none_prompt)
    picks = bp._length_representative_pids(pids, none_prompt, 3)
    assert sorted(len(none_prompt[p]["prompt"]) for p in picks) == [
        5, 500, 50000]                          # shortest, middle, longest
    assert picks == sorted(picks)               # returned in plan order
    assert bp._length_representative_pids(pids, none_prompt, 99) == sorted(pids)
    with pytest.raises(SystemExit, match="--max-prompts"):
        bp._length_representative_pids(pids, none_prompt, 0)


def test_max_prompts_caps_generation_and_pilot_pairs_are_emitted(
        tmp_path, monkeypatch):
    from test_stage3_pool import _hneg, _hprompt

    cfg = _stage3_cfg(tmp_path, n_prompts=5)
    priv, log = tmp_path / "priv", []
    _install_fake_vllm(monkeypatch, log)
    out = tmp_path / "pilot" / "pilot_pairs.jsonl"
    args = _pos_args(priv, candidates=2, max_prompts=2,
                     pilot_pairs_out=str(out))
    bp.cmd_generate_pos(args, cfg)

    rows = _irr_rows(priv, cfg)
    # 2 of 5 prompts, evenly spaced over the (tie-broken) length ordering
    assert {r["goal_id"] for r in rows} == {"g0", "g4"}
    assert len(rows) == 4                                   # 2 prompts x K=2

    pairs = [json.loads(l) for l in out.read_text().splitlines()]
    assert [p["goal_id"] for p in pairs] == ["g0", "g4"]
    for p in pairs:
        i = int(p["goal_id"][1:])
        assert p["prompt"] == _hprompt(i)
        assert p["neg"] == _hneg(i)                # the undefended store reply
        assert p["pos"].startswith("VRESP::")      # this SI's generation
        assert p["instruction_id"] == IRR
        assert p["candidate_idx"] == 0             # the lowest-index candidate
        assert p["ungraded"] is True and p["pilot_only"] is True
    assert (out.stat().st_mode & 0o777) == 0o600   # private-store discipline

    # resume: nothing regenerated, the pairs artifact refreshed byte-identically
    n, before = len(log), out.read_text()
    bp.cmd_generate_pos(args, cfg)
    assert len(log) == n
    assert out.read_text() == before


def test_pilot_pairs_refuse_a_path_inside_the_repo(tmp_path, monkeypatch):
    cfg = _stage3_cfg(tmp_path, n_prompts=1)
    log = []
    sampler_cls = _install_fake_vllm(monkeypatch, log)
    with pytest.raises(SystemExit, match="refusing to write pilot pairs"):
        bp.cmd_generate_pos(_pos_args(
            tmp_path / "priv", candidates=1,
            pilot_pairs_out=str(bp.REPO / "pilot_pairs.jsonl")), cfg)
    # refused BEFORE any GPU work or write — attack text never nears the repo
    assert sampler_cls.built == 0 and not log
    assert not (bp.REPO / "pilot_pairs.jsonl").exists()


def test_hf_smoke_the_pre_ruling_route_is_unchanged(tmp_path, monkeypatch):
    """The retained-contingency smoke: no engine anywhere (bare args, bare
    config) drives the exact pre-ruling hf flow — same row count, same
    candidate indices, same delta-only top-up — now plus a provenance record."""
    cfg = _stage3_cfg(tmp_path, n_prompts=1)
    priv = tmp_path / "priv"
    calls = _stub_generation(monkeypatch)
    args = types.SimpleNamespace(private_root=str(priv), dry_run=False,
                                 batch_size=4, si=IRR, candidates=2)
    bp.cmd_generate_pos(args, cfg)
    rows = _irr_rows(priv, cfg)
    assert len(rows) == 2 and {r["candidate_idx"] for r in rows} == {0, 1}
    assert calls["rendered"]                       # generated via generate_batch
    args.candidates = 3
    bp.cmd_generate_pos(args, cfg)
    assert {r["candidate_idx"] for r in _irr_rows(priv, cfg)} == {0, 1, 2}
    prov = _sidecar(priv, cfg)
    assert prov["engine"] == "hf" and prov["vllm_version"] is None
    assert prov["sampling_params"]["top_p"] == "hf-generation-config-inherited"

"""The stage-3 pool SAMPLING lane:
``src/sisteer/sampling/{sources,engine}.py`` + ``scripts/pool_sample.py``.

CPU-only, offline, no torch, no vLLM, no GPU, no real data — every prompt is an
INVENTED synthetic string. What the suite proves:

- the CPU inventory (``sources``) crosses goals with the pure static families
  at the right per-goal fan-out, keys every prompt by the ``(family, goal_id,
  variant_key)`` triple the pool builder joins on, and is deterministic in the
  seed;
- ``wildjailbreak_harmful`` enters as its own standalone family (sample_idx 0,
  one variant per row) via the shared split reader (monkeypatched — the gated
  corpus is never downloaded here);
- ``inventory_counts`` reports per-side / per-family / per-source counts and no
  text;
- the GPU engine's PURE helpers: the resume ``cell_key``, ``load_done_keys``
  (round-trips a JSONL and tolerates a torn tail), the store-row schema
  ``build_two_si_pool`` reads (si_condition/si_id mapping, explicitly ungraded),
  and ``build_chat``'s shared-prefix property that makes vLLM prefix caching hit.
"""

from __future__ import annotations

import json

import pytest

from sisteer.eval.jailbreak_data import Goal
from sisteer.sampling import sources as S
from sisteer.sampling import engine as E


def _goal(gid: str, text: str, cat: str = "illegal", dataset: str = "jbb") -> Goal:
    return Goal(goal_id=gid, text=text, category=cat, dataset=dataset,
                revision="deadbeef", split="harmful")


GOALS = [
    _goal("g0", "SYNTHETIC benign-looking request number one 11111111"),
    _goal("g1", "SYNTHETIC benign-looking request number two 22222222"),
]


# --------------------------------------------------------------------------- #
# harmful: the static families over synthetic goals                            #
# --------------------------------------------------------------------------- #


def test_harmful_fanout_and_prompt_identity():
    """bon (8 variants/goal) + encoding cross every goal; each SamplePrompt
    carries the (family, goal_id, variant_key) triple the matched pool joins on,
    and as_meta() drops the live-only text."""
    ps = S.harmful_attack_prompts(GOALS, ["bon", "encoding"], seed=0)
    by_fam = {}
    for p in ps:
        by_fam.setdefault(p.family, []).append(p)
        assert p.side == "harmful"
        assert p.source == "jbb"          # from goal.dataset
        assert p.prompt_id == (p.family, p.goal_id, p.variant_key)
        assert p.variant_key == f"{p.sample_idx}:{p.trial}"
        assert "text" not in p.as_meta()  # text never lands in a manifest
    # bon is 8 variants per goal (DEFAULT_BON_VARIANTS), over 2 goals -> 16
    assert len(by_fam["bon"]) == 2 * S.DEFAULT_BON_VARIANTS
    # encoding produces >=1 variant per goal (both goals covered)
    assert {p.goal_id for p in by_fam["encoding"]} == {"g0", "g1"}


def test_harmful_fanout_is_deterministic_in_seed():
    a = S.harmful_attack_prompts(GOALS, ["bon", "encoding"], seed=0)
    b = S.harmful_attack_prompts(GOALS, ["bon", "encoding"], seed=0)
    assert [p.prompt_id for p in a] == [p.prompt_id for p in b]
    assert [p.text for p in a] == [p.text for p in b]


def test_subenv_family_without_a_bank_raises_never_silently_empty():
    """amplegcg_plus needs a pre-generated (model-produced) bank; absent one,
    refuse loudly rather than emit a zero-count family (the stage-2 narrowing
    this spec exists to prevent). Only amplegcg_plus is on this axis."""
    assert S.SUBENV_FAMILIES == frozenset({"amplegcg_plus"})
    with pytest.raises(RuntimeError, match="prompt bank"):
        S.harmful_attack_prompts(GOALS, ["amplegcg_plus"], seed=0)


# --------------------------------------------------------------------------- #
# harmful: wildjailbreak_harmful as its own standalone family                  #
# --------------------------------------------------------------------------- #

WJB_HARMFUL = [
    "SYNTHETIC already-adversarial attack prompt alpha 33333333",
    "SYNTHETIC already-adversarial attack prompt bravo 44444444",
    "SYNTHETIC already-adversarial attack prompt charlie 55555555",
]


def test_wildjailbreak_harmful_is_a_standalone_family(monkeypatch):
    import sisteer.data.content_sets as cs
    monkeypatch.setattr(cs, "load_wildjailbreak_split",
                        lambda data_type, **kw: list(WJB_HARMFUL))
    ps = S.wildjailbreak_harmful_prompts(seed=0)
    assert len(ps) == len(WJB_HARMFUL)
    for p in ps:
        assert p.family == S.WILDJAILBREAK_HARMFUL_FAMILY
        assert p.side == "harmful" and p.source == "wildjailbreak"
        assert p.sample_idx == 0 and p.variant_name == "wjb_adversarial"
    # deterministic subsample to n, seed-stable
    sub1 = S.wildjailbreak_harmful_prompts(n=2, seed=1)
    monkeypatch.setattr(cs, "load_wildjailbreak_split",
                        lambda data_type, **kw: list(WJB_HARMFUL))
    sub2 = S.wildjailbreak_harmful_prompts(n=2, seed=1)
    assert len(sub1) == 2 and [p.text for p in sub1] == [p.text for p in sub2]


def test_inventory_counts_reports_sides_families_sources_and_no_text(monkeypatch):
    import sisteer.data.content_sets as cs
    monkeypatch.setattr(cs, "load_wildjailbreak_split",
                        lambda data_type, **kw: list(WJB_HARMFUL))
    # a fake benign loader so build_inventory needs no gated data
    monkeypatch.setitem(cs.BENIGN_LOADERS, "phtest",
                        lambda n=None, **kw: ["SYNTHETIC benign one aaaaaaaa",
                                              "SYNTHETIC benign two bbbbbbbb"])
    inv = S.build_inventory(goals=GOALS, attack_families=["bon"],
                            benign_families=["phtest"], seed=0)
    counts = S.inventory_counts(inv)
    assert counts["by_side"]["harmful"] == 2 * S.DEFAULT_BON_VARIANTS + len(WJB_HARMFUL)
    assert counts["by_side"]["benign"] == 2
    assert set(counts["by_family"]) == {"bon", "wildjailbreak_harmful", "phtest"}
    assert counts["n_prompts"] == sum(counts["by_family"].values())


# --------------------------------------------------------------------------- #
# GPU engine — the PURE helpers (resume keying + row schema)                    #
# --------------------------------------------------------------------------- #


def _sp(family="bon", gid="g0", sample_idx=3, trial=0):
    return S.SamplePrompt(
        family=family, goal_id=gid, sample_idx=sample_idx, trial=trial,
        variant_name="bon_3", text="SYNTHETIC attack prompt zzzzzzzz",
        side="harmful", source="jbb", category="illegal")


def test_row_schema_matches_the_pool_builder_contract():
    p = _sp()
    none_row = E._row(p, E.NONE_SI, 0, "SYNTHETIC undefended completion")
    si_row = E._row(p, "safety_refusal", 0, "SYNTHETIC refusal")
    # si_condition mapping + explicit ungraded three-state
    assert none_row["si_condition"] == "none" and none_row["si_id"] is None
    assert si_row["si_condition"] == "prompted" and si_row["si_id"] == "safety_refusal"
    for r in (none_row, si_row):
        assert r["attack"] == "bon" and r["goal_id"] == "g0"
        assert r["sample_idx"] == 3 and r["trial"] == 0
        assert r["score"] is None and r["graded"] is False   # generated, awaiting grade


def test_cell_key_and_load_done_keys_round_trip_and_tolerate_a_torn_tail(tmp_path):
    p = _sp()
    out = tmp_path / "shard_00.jsonl"
    rows = [E._row(p, E.NONE_SI, 0, "r0"),
            E._row(p, "safety_refusal", 0, "r1"),
            E._row(p, "si_irrelevant", 2, "r2")]     # candidate_idx 2 (raised K)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write('{"attack": "bon", "goal_id": "g0"')  # torn tail from a SIGKILL
    done = E.load_done_keys(out)
    assert E.cell_key(p, E.NONE_SI, 0) in done
    assert E.cell_key(p, "safety_refusal", 0) in done
    assert E.cell_key(p, "si_irrelevant", 2) in done
    # the torn last line is skipped, not fatal, and never a phantom key
    assert len(done) == 3
    # an un-generated cell is absent -> resume regenerates exactly it
    assert E.cell_key(p, "si_irrelevant", 3) not in done


def test_build_chat_puts_the_si_as_a_literal_prefix():
    """The SI must be a literal PREFIX of the rendered string so vLLM prefix
    caching hits across every prompt under that SI. The dummy tokenizer's
    ``apply_chat_template`` deliberately accepts NO extra kwargs: a plain
    (marker-less) tokenizer must get exactly the historical call — any leaked
    template kwarg would TypeError here (the F8 backward-compat boundary)."""
    class _Tok:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return "".join(f"<{m['role']}>{m['content']}" for m in msgs)

    tok = _Tok()
    si = "SYNTHETIC long system instruction " * 3
    a = E.build_chat(tok, si, "user request one")
    b = E.build_chat(tok, si, "user request two")
    common = f"<system>{si}"
    assert a.startswith(common) and b.startswith(common)
    # no system turn when there is no SI (the undefended pass)
    assert E.build_chat(tok, None, "u").startswith("<user>")


# --------------------------------------------------------------------------- #
# GPU engine — canonical rendering (F8): marker kwargs + system-role fold       #
# --------------------------------------------------------------------------- #


class _CanonTok:
    """Quacks like an AutoTokenizer for the F8 render tests: a
    ``chat_template`` string (the REAL ``load_tokenizer`` pins
    ``enable_thinking: False`` when the template mentions it), pad/eos
    defaults, an ``apply_chat_template`` that RECORDS the kwargs each render
    received, and an optional Gemma-shaped system-role rejection."""

    def __init__(self, chat_template="{{fake}}", reject_system=False):
        self.pad_token = None
        self.eos_token = "<eos>"
        self.chat_template = chat_template
        self.reject_system = reject_system
        self.calls: list[dict] = []

    def apply_chat_template(self, msgs, tokenize=False,
                            add_generation_prompt=True, **kw):
        self.calls.append(dict(kw))
        if self.reject_system and msgs and msgs[0]["role"] == "system":
            raise ValueError("System role not supported")   # the Gemma shape
        return "".join(f"<{m['role']}>{m['content']}" for m in msgs)


SI_TEXT = "SYNTHETIC system instruction for render tests 99999999"


def test_build_chat_applies_marker_kwargs_and_matches_the_canonical_render():
    """A tokenizer carrying the harness marker (what load_tokenizer sets on
    reasoning-capable templates) renders with enable_thinking=False AND
    byte-identically to models.build_chat_prompt — the canonical path whose
    prefix hash the provenance sidecar records (F8)."""
    from sisteer.models import build_chat_prompt

    tok = _CanonTok()
    tok._sisteer_template_kwargs = {"enable_thinking": False}
    out = E.build_chat(tok, SI_TEXT, "user request")
    assert out.startswith(f"<system>{SI_TEXT}")
    assert tok.calls and all(c == {"enable_thinking": False} for c in tok.calls)
    assert out == build_chat_prompt(tok, "user request", system=SI_TEXT)


def test_build_chat_without_the_marker_passes_no_kwargs():
    """A raw tokenizer (no marker) keeps today's exact call: zero template
    kwargs — the backward-compat boundary for every other VllmSampler lane."""
    tok = _CanonTok()
    out = E.build_chat(tok, SI_TEXT, "user request")
    assert out == f"<system>{SI_TEXT}<user>user request"
    assert tok.calls == [{}]


def test_build_chat_folds_the_system_role_when_the_template_rejects_it():
    """The Gemma shape: the template raises on a system role; the fold puts
    the SI into the first user turn (still a literal shared PREFIX, so prefix
    caching keeps hitting) instead of crashing the anti-neg leg."""
    tok = _CanonTok(reject_system=True)
    a = E.build_chat(tok, SI_TEXT, "user request one")
    b = E.build_chat(tok, SI_TEXT, "user request two")
    assert "<system>" not in a
    common = f"<user>{SI_TEXT}"
    assert a.startswith(common) and b.startswith(common)
    assert "user request one" in a and "user request two" in b


def _fake_vllm_module(monkeypatch, log):
    """A CPU-only vllm stand-in for exercising the REAL VllmSampler.__init__
    and run(): LLM records every (rendered, sampling-kwargs) request."""
    import sys
    import types

    fake = types.ModuleType("vllm")
    fake.__version__ = "0.0-test"

    class SamplingParams:
        def __init__(self, **kw):
            self.kw = kw

    class LLM:
        def __init__(self, **kw):
            self.kw = kw

        def collective_rpc(self, name):   # vllm_patches.attest_engine's RPC
            assert name == "sisteer_mrope_report"
            return [{"patch_applied": True, "needs_patch": False}]

        def generate(self, rendered, sp):
            log.append((list(rendered), dict(sp.kw)))
            import types as _t
            return [_t.SimpleNamespace(
                outputs=[_t.SimpleNamespace(text=f"VRESP::{r}")])
                for r in rendered]

    fake.SamplingParams = SamplingParams
    fake.LLM = LLM
    monkeypatch.setitem(sys.modules, "vllm", fake)
    return fake


def _patch_auto_tokenizer(monkeypatch, tok):
    """Inject ``tok`` at sisteer.models.AutoTokenizer, so the REAL
    load_tokenizer pipeline (pad default, enable_thinking pin) runs on it —
    the same seam test_pool_vllm_generation uses."""
    class _Auto:
        @staticmethod
        def from_pretrained(name, revision=None, **kw):
            return tok

    monkeypatch.setattr("sisteer.models.AutoTokenizer", _Auto)


def test_vllmsampler_builds_the_canonical_tokenizer_and_pins_thinking(
        tmp_path, monkeypatch):
    """VllmSampler's internal tokenizer now comes from
    sisteer.models.load_tokenizer, so a reasoning-capable template (Qwen3.8)
    carries the marker and every run() render pins enable_thinking=False —
    the anti-neg leg (the one leg that renders INSIDE the sampler) can no
    longer spend its completion budget inside <think>."""
    log = []
    _fake_vllm_module(monkeypatch, log)
    ftok = _CanonTok(
        chat_template="{% if enable_thinking %}<think>{% endif %}fake")
    _patch_auto_tokenizer(monkeypatch, ftok)
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    sampler = E.VllmSampler("synthetic/model-for-tests", revision="r0")
    # the REAL load_tokenizer ran: thinking pinned, pad defaulted to eos
    assert sampler.tok is ftok
    assert ftok._sisteer_template_kwargs == {"enable_thinking": False}
    assert ftok.pad_token == "<eos>"

    out = tmp_path / "shard_00.jsonl"
    res = sampler.run([_sp()], [E.GenSpec("safety_refusal", SI_TEXT, k=1)], out)
    assert res["written"] == 1
    assert ftok.calls and all(c.get("enable_thinking") is False
                              for c in ftok.calls)
    # the SI stays a literal prefix of the engine's actual request
    (rendered, kw), = log
    assert rendered[0].startswith(f"<system>{SI_TEXT}")
    assert kw["temperature"] == 0.0 and kw["seed"] is None     # k=1: greedy
    row = json.loads(out.read_text())
    assert row["response"] == f"VRESP::{rendered[0]}"


def test_vllmsampler_folds_the_system_role_for_a_gemma_shaped_template(
        tmp_path, monkeypatch):
    """RESIDUAL F8, Gemma half: a template that rejects the system role gets
    the SI folded into the first user turn inside sampler.run — the raw
    render would have crashed the anti-neg generation."""
    log = []
    _fake_vllm_module(monkeypatch, log)
    ftok = _CanonTok(reject_system=True)
    _patch_auto_tokenizer(monkeypatch, ftok)
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    sampler = E.VllmSampler("synthetic/model-for-tests")
    out = tmp_path / "shard_00.jsonl"
    sampler.run([_sp()], [E.GenSpec("safety_refusal", SI_TEXT, k=1)], out)
    (rendered, _kw), = log
    assert "<system>" not in rendered[0]
    assert rendered[0].startswith(f"<user>{SI_TEXT}")           # folded, prefix kept
    assert "SYNTHETIC attack prompt" in rendered[0]


# --------------------------------------------------------------------------- #
# gen-bank CLI surface (scripts/pool_sample.py)                                #
# --------------------------------------------------------------------------- #


def _load_pool_sample():
    """Import scripts/pool_sample.py as a module (it is a script, not a
    package). Each call returns a fresh module object."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "pool_sample_cli", root / "scripts" / "pool_sample.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gen_bank_subcommand_is_wired_with_amplegcg_defaults(monkeypatch):
    import sys

    ps = _load_pool_sample()
    captured: dict = {}
    monkeypatch.setattr(ps, "cmd_gen_bank",
                        lambda args: captured.update(vars(args)))
    monkeypatch.setattr(sys, "argv",
                        ["pool_sample.py", "gen-bank", "--out", "/tmp/bank.json"])
    ps.main()
    assert captured["family"] == "amplegcg_plus"
    assert captured["generator_model"] == ps.AMPLEGCG_MODEL
    assert captured["n_suffixes"] == 8
    assert captured["out"] == "/tmp/bank.json"
    # _common args ride along too — _goals() reads harmbench_splits/seed.
    assert "harmbench_splits" in captured and "seed" in captured


def test_gen_bank_requires_an_out_path(monkeypatch):
    import sys

    ps = _load_pool_sample()
    monkeypatch.setattr(sys, "argv", ["pool_sample.py", "gen-bank"])
    with pytest.raises(SystemExit):        # argparse: --out is required
        ps.main()


def test_gen_bank_refuses_a_family_it_has_no_generator_for():
    import argparse

    ps = _load_pool_sample()
    args = argparse.Namespace(
        family="bon", out="/tmp/x.json", generator_model=ps.AMPLEGCG_MODEL,
        n_suffixes=8, seed=0, harmbench_splits="val,test")
    # the guard fires before any goal load or model download — only
    # amplegcg_plus has a generator; every other family is a CPU transform.
    with pytest.raises(SystemExit, match="no generator"):
        ps.cmd_gen_bank(args)


# --------------------------------------------------------------------------- #
# F12: sample writes a provenance sidecar; merge propagates it to the store    #
# --------------------------------------------------------------------------- #


class _RowWritingSampler:
    """Stand-in for VllmSampler with the same run() contract, appending real
    engine._row rows so merge has genuine input — no GPU, no vLLM."""

    def __init__(self, model, **kw):
        pass

    def run(self, prompts, specs, out_path, *, done=None, progress_every=2000):
        from pathlib import Path

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        done = set(done or ())
        written = 0
        with out_path.open("a") as f:
            for spec in specs:
                for ci in range(spec.k):
                    for p in prompts:
                        if E.cell_key(p, spec.si_id, ci) in done:
                            continue
                        f.write(json.dumps(E._row(
                            p, spec.si_id, ci,
                            f"SYNTHETIC sampled resp {p.goal_id} c{ci}"))
                            + "\n")
                        written += 1
        return {"written": written}


def _sample_args(out_dir, **over):
    import types

    base = dict(
        out_dir=str(out_dir), shard_index=0, shard_count=1, limit=None,
        model="synthetic/model-for-tests", revision="r0", dtype="bfloat16",
        max_model_len=16384, gpu_memory_utilization=0.90, seed=0,
        max_new_tokens=64, enforce_eager=False, temperature=1.0,
        sis="safety_refusal", candidates=None, no_none=False)
    base.update(over)
    return types.SimpleNamespace(**base)


def _sample_world(tmp_path, monkeypatch, **arg_over):
    """A ready-to-run cmd_sample world: tiny synthetic inventory, stubbed
    sampler, fake vllm module + fake AutoTokenizer (a thinking-aware template,
    so the recorded render must carry the pin)."""
    import sys
    import types

    ps = _load_pool_sample()
    monkeypatch.setattr(ps, "_inventory", lambda args: [_sp()])
    monkeypatch.setattr("sisteer.sampling.engine.VllmSampler",
                        _RowWritingSampler)
    fake = types.ModuleType("vllm")
    fake.__version__ = "0.0-test"
    monkeypatch.setitem(sys.modules, "vllm", fake)
    _patch_auto_tokenizer(monkeypatch, _CanonTok(
        chat_template="{% if enable_thinking %}<think>{% endif %}fake"))
    return ps, _sample_args(tmp_path / "shards", **arg_over)


def _load_pool_builder():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import build_two_si_pool as bp

    return bp


def test_cmd_sample_writes_a_schema_valid_provenance_sidecar(
        tmp_path, monkeypatch):
    """RESIDUAL F12: the sample leg records engine + vLLM version + sampling
    params + model/revision + the canonical prompt render, in exactly the
    schema the manifest validates — via build_two_si_pool's own helpers."""
    ps, args = _sample_world(tmp_path, monkeypatch)
    ps.cmd_sample(args)
    bp = _load_pool_builder()

    prov = json.loads(
        (tmp_path / "shards" / bp.GEN_PROVENANCE_NAME).read_text())
    for field in bp._PROV_SCHEMA_REQUIRED:
        assert field in prov, f"sidecar missing required field {field!r}"
    assert prov["engine"] == "vllm"
    assert prov["vllm_version"] == "0.0-test"
    assert prov["model"] == "synthetic/model-for-tests"
    assert prov["model_revision"] == "r0"
    sp = prov["sampling_params"]
    assert sp["temperature"] == 1.0 and sp["base_seed"] == 0
    assert sp["max_tokens"] == 64 and "seed_rule" in sp
    # the canonical render is recorded: pin + one prefix hash per condition
    pr = prov["prompt_render"]
    assert pr["template_kwargs"] == {"enable_thinking": False}
    assert "load_tokenizer" in pr["renderer"]
    assert set(pr["rendered_prefix_sha256"]) == {"none", "safety_refusal"}
    # per-SI K rides in info (outside the strict set: K raises are sanctioned)
    assert prov["info"]["k_by_si"] == {"none": 1, "safety_refusal": 1}
    # and rows were generated alongside it
    assert (tmp_path / "shards" / "shard_00.jsonl").stat().st_size > 0


def test_cmd_sample_resume_with_drifted_provenance_refuses(
        tmp_path, monkeypatch):
    """A resume whose seed (or any strict field) drifted refuses BEFORE the
    engine builds — the arm-time guard every other generation leg runs."""
    ps, args = _sample_world(tmp_path, monkeypatch)
    ps.cmd_sample(args)
    with pytest.raises(SystemExit, match="different"):
        ps.cmd_sample(_sample_args(tmp_path / "shards", seed=7))
    # same args resume rides (byte-identical record)
    ps.cmd_sample(args)


def test_cmd_merge_propagates_the_sidecar_and_the_manifest_gather_accepts_it(
        tmp_path, monkeypatch):
    """The full lane: sample (sidecar in the out-dir) -> merge (sidecar in the
    STORE, the leg the manifest audits) -> _gather_generation_provenance
    accepts a genuine single-engine store under an engine-declaring config —
    and refuses the SAME store with the sidecar removed, proving the sidecar
    is exactly what unblocks the manifest."""
    import types

    ps, args = _sample_world(tmp_path, monkeypatch)
    ps.cmd_sample(args)
    store = tmp_path / "store"
    ps.cmd_merge(types.SimpleNamespace(out_dir=str(tmp_path / "shards"),
                                       store_dir=str(store)))
    bp = _load_pool_builder()
    assert (store / "bon.jsonl").exists()          # rows merged
    prov = json.loads((store / bp.GEN_PROVENANCE_NAME).read_text())
    assert prov["engine"] == "vllm"                # sidecar propagated intact

    # the sweep shape: a pos leg with its own recorded sidecar + the store
    monkeypatch.delenv("TT_TWO_SI_STORE_DIR", raising=False)
    cfg = {"model": "synthetic/model-for-tests",
           "si_ids": ["safety_refusal", "si_constitution"],
           "pool": {"store_dir": str(store), "engine": "vllm"}}
    pos_dir = tmp_path / "priv" / "two_si_pos_constitution"
    pos_dir.mkdir(parents=True)
    (pos_dir / bp.GEN_PROVENANCE_NAME).write_text(json.dumps(
        {"engine": "vllm", "vllm_version": "0.0-test",
         "model": "synthetic/model-for-tests", "model_revision": "r0",
         "sampling_params": {"temperature": 1.0}}))
    gp = bp._gather_generation_provenance(cfg, str(tmp_path / "priv"))
    assert gp["engines"] == ["vllm"]               # single-engine: mix check passes
    assert set(gp["legs"]) == {"store", "pos:si_constitution"}
    assert gp["legs"]["store"]["model_revision"] == "r0"

    # negative control: store rows without the sidecar refuse under a
    # declared engine (the exact F12 world — sibling legs recorded, the
    # sample-all store not) — the sidecar is what unblocks the manifest
    (store / bp.GEN_PROVENANCE_NAME).unlink()
    with pytest.raises(SystemExit, match="NO provenance sidecar"):
        bp._gather_generation_provenance(cfg, str(tmp_path / "priv"))


def test_cmd_merge_refuses_a_store_with_conflicting_provenance(
        tmp_path, monkeypatch):
    """Merging vllm-generated shards into a store recorded under a different
    engine is the silent engine-mixing the ruling forbids — merge refuses
    BEFORE any row lands."""
    import types

    ps, args = _sample_world(tmp_path, monkeypatch)
    ps.cmd_sample(args)
    bp = _load_pool_builder()
    store = tmp_path / "store"
    store.mkdir()
    conflicting = json.loads(
        (tmp_path / "shards" / bp.GEN_PROVENANCE_NAME).read_text())
    conflicting["engine"] = "hf"
    (store / bp.GEN_PROVENANCE_NAME).write_text(json.dumps(conflicting))
    with pytest.raises(SystemExit, match="different"):
        ps.cmd_merge(types.SimpleNamespace(out_dir=str(tmp_path / "shards"),
                                           store_dir=str(store)))
    assert not (store / "bon.jsonl").exists()      # nothing mixed in


def test_cmd_merge_without_a_sidecar_keeps_todays_behaviour(
        tmp_path, monkeypatch, capsys):
    """Legacy shards (generated before provenance recording) merge exactly as
    before — rows land, no sidecar appears, and the gap is said out loud."""
    import types

    out_dir = tmp_path / "shards"
    out_dir.mkdir()
    (out_dir / "shard_00.jsonl").write_text(
        json.dumps(E._row(_sp(), E.NONE_SI, 0, "SYNTHETIC legacy resp"))
        + "\n")
    ps = _load_pool_sample()
    store = tmp_path / "store"
    ps.cmd_merge(types.SimpleNamespace(out_dir=str(out_dir),
                                       store_dir=str(store)))
    bp = _load_pool_builder()
    assert (store / "bon.jsonl").exists()
    assert not (store / bp.GEN_PROVENANCE_NAME).exists()
    assert "predate provenance recording" in capsys.readouterr().err

"""GCG — white-box greedy-coordinate-gradient attack (Zou et al. 2307.15043).

CPU only (tiny-llama + a random dummy steering object + a generate seam; no
network, no harmful text). Covers the contract the task requires:

- (a) the registry resolves ``gcg`` and it is NOT in GATED_ATTACKS (no attacker);
- (b) the candidate filter rejects non-ASCII / non-round-trip tokens;
- (c) one optimisation step computes a (non-increasing) loss and returns a valid
      suffix of UNCHANGED token length;
- (d) gradients flow through an ACTIVE steering object (non-zero input-embedding
      gradient with a dummy object applied, and the loss changes with the object
      in the graph) — the load-bearing white-box property;
- (e) resume skips a goal with a terminal progress record and drops the partial
      response rows of an interrupted goal;
- (f) forward accounting reports optimiser forwards SEPARATELY from the real
      ``n_target_forwards``.
"""

import json

import pytest
import torch

from sisteer.eval.jailbreak_data import Goal


# -- (a) registry --------------------------------------------------------------


def test_gcg_registered_and_not_gated():
    from sisteer.eval.attacks import GATED_ATTACKS, JB_ATTACKS, get_attack

    assert "gcg" in JB_ATTACKS
    assert callable(get_attack("gcg"))          # resolves to run_gcg
    assert get_attack("gcg").__name__ == "run_gcg"
    # GCG is white-box, target-only: it needs NO attacker model + no PyRIT.
    assert "gcg" not in GATED_ATTACKS


# -- (b) candidate filter ------------------------------------------------------


class _FakeTok:
    """A minimal tokenizer stub for the filter: id -> decoded text, and a
    ``__call__`` that round-trips a string back to its id only when EXACTLY one
    id decodes to it (else it 'splits', so the round trip fails)."""

    def __init__(self, table):
        self.table = table

    def decode(self, ids):
        return "".join(self.table.get(int(i), "") for i in ids)

    def batch_decode(self, id_lists):
        return [self.decode(ids) for ids in id_lists]

    def __call__(self, s, add_special_tokens=False):
        matches = [i for i, t in self.table.items() if t == s]
        return {"input_ids": ([matches[0]] if len(matches) == 1 else [0, 0])}


def test_candidate_filter_rejects_non_ascii_and_non_roundtrip():
    from sisteer.eval.attacks.gcg import _allowed_token_mask, _token_is_allowed

    # 0="" (empty), 1="!" (ok), 2="é" (non-ascii), 3="ok" (ok),
    # 4/5="amb" (ambiguous -> re-encode splits -> non-round-trip).
    tok = _FakeTok({0: "", 1: "!", 2: "é", 3: "ok", 4: "amb", 5: "amb"})

    assert _token_is_allowed(tok, 1) is True         # ascii + single round trip
    assert _token_is_allowed(tok, 3) is True
    assert _token_is_allowed(tok, 2) is False        # non-ASCII
    assert _token_is_allowed(tok, 4) is False        # non-round-trip (ambiguous)
    assert _token_is_allowed(tok, 0) is False        # empty decode

    mask = _allowed_token_mask(tok, 6)
    assert mask.tolist() == [False, True, False, True, False, False]


# -- shared helpers for the tiny-model tests -----------------------------------


def _dummy_vector_obj(model, layer=1, scale_seed=0):
    """A random steering object through the exact VectorObject/steering path the
    'vector' arm uses (so the white-box test exercises the real hook)."""
    from sisteer.methods import VectorObject
    from sisteer.models import hidden_size
    from sisteer.vectors import SteeringVector

    g = torch.Generator().manual_seed(scale_seed)
    vec = torch.randn(hidden_size(model), generator=g)
    sv = SteeringVector(vectors={layer: vec}, method="distill_soft",
                        model_name="", instruction_id="")
    return VectorObject(sv)


# -- (c) one optimisation step -------------------------------------------------


def test_gcg_step_computes_loss_and_keeps_suffix_length(tiny_llama):
    from sisteer.eval.attacks.gcg import _build_gcg_prompt, gcg_step

    model, tok = tiny_llama
    prompt = _build_gcg_prompt(tok, None, "hello there", "Sure")
    suffix_ids = tok("! ! !", return_tensors="pt",
                     add_special_tokens=False)["input_ids"][0].long()
    gen = torch.Generator().manual_seed(0)

    new_suffix, best_loss, grad_loss = gcg_step(
        model, None, 1.0, prompt, suffix_ids, topk=8, search_width=8,
        allowed_mask=None, gen=gen, eval_batch_size=8)

    assert new_suffix.shape == suffix_ids.shape          # unchanged token length
    assert new_suffix.dtype == torch.long
    import math
    assert math.isfinite(best_loss) and math.isfinite(grad_loss)
    # the current suffix is always a candidate, so loss never increases.
    assert best_loss <= grad_loss + 1e-4


# -- (d) white-box: gradients flow through an ACTIVE steering object ------------


def test_gradients_flow_through_active_object(tiny_llama):
    from sisteer.eval.attacks.gcg import _build_gcg_prompt, token_gradients

    model, tok = tiny_llama
    obj = _dummy_vector_obj(model)
    prompt = _build_gcg_prompt(tok, obj, "hello there", "Sure")  # inject is identity
    suffix_ids = tok("! ! !", return_tensors="pt",
                     add_special_tokens=False)["input_ids"][0].long()

    loss_obj, grad_obj = token_gradients(model, obj, 1.0, prompt, suffix_ids)
    loss_none, grad_none = token_gradients(model, None, 1.0, prompt, suffix_ids)

    # THE crux: a non-zero input-embedding gradient with the object ACTIVE.
    assert torch.isfinite(grad_obj).all()
    assert grad_obj.abs().sum().item() > 0.0
    assert grad_obj.shape[0] == suffix_ids.numel()       # [L, V]
    # the object is really in the forward graph (loss + gradient both shift).
    assert abs(loss_obj - loss_none) > 1e-5
    assert not torch.allclose(grad_obj, grad_none)


# -- resume / accounting: a fast seam for the real generations -----------------


def _seam_defense(model, tok, text="I'm sorry, I can't help with that."):
    """An LLMHFSteered whose real generations are a fixed (refusal) string via
    the generate seam — so run_gcg's optimiser still uses the real model while
    the periodic ``defense.generate`` is instant and deterministic."""
    from sisteer.eval.jailbreak import LLMHFSteered

    calls = {"n": 0}

    def seam(rendered, **kw):
        calls["n"] += len(rendered)
        return [text for _ in rendered]

    d = LLMHFSteered(model, tok, mode="none", generate_fn=seam)
    return d, calls


# -- (e) resume ----------------------------------------------------------------


def test_run_gcg_resume_skips_done_and_drops_partial(tiny_llama, tmp_path):
    from sisteer.eval.attacks.gcg import (
        _load_covered_goals,
        load_persisted_responses,
        run_gcg,
    )

    model, tok = tiny_llama
    out = tmp_path / "run"
    kw = dict(n_steps=1, search_width=2, topk=4, check_every=1,
              filter_candidates=False, target_max_new_tokens=4)

    g0 = Goal("g0", "benign goal zero", "illegal", "harmbench", "rev")
    g1 = Goal("g1", "benign goal one", "illegal", "harmbench", "rev")

    d, _ = _seam_defense(model, tok)
    run_gcg(d, [g0], out_dir=out, **kw)
    assert _load_covered_goals(out / "progress.jsonl") == {"g0"}
    g0_resps = load_persisted_responses(out / "responses.jsonl")["g0"]
    assert len(g0_resps) >= 1

    # simulate an interrupted g1: a partial response row with NO progress record.
    with (out / "responses.jsonl").open("a") as f:
        f.write(json.dumps({"goal_id": "g1", "order": 0,
                            "response": "PARTIAL_INTERRUPTED"}) + "\n")

    d2, _ = _seam_defense(model, tok)
    result = run_gcg(d2, [g0, g1], out_dir=out, **kw)

    # g0 already terminal -> skipped; only g1 re-run.
    assert result["n_goals_run"] == 1
    assert _load_covered_goals(out / "progress.jsonl") == {"g0", "g1"}
    resps = load_persisted_responses(out / "responses.jsonl")
    # the partial g1 row was dropped and g1 regenerated cleanly (order from 0).
    assert "PARTIAL_INTERRUPTED" not in resps["g1"]
    assert resps["g0"] == g0_resps                        # g0 untouched


# -- (f) forward accounting: optimiser forwards are SEPARATE from target ones ---


def test_forward_accounting_separates_optimizer_from_target(tiny_llama, tmp_path):
    from sisteer.eval.attacks.gcg import run_gcg

    model, tok = tiny_llama
    d, _ = _seam_defense(model, tok)  # refusal seam -> no early stop, runs all steps
    g = Goal("g0", "benign goal zero", "illegal", "harmbench", "rev")

    result = run_gcg(d, [g], out_dir=tmp_path / "run", n_steps=3, search_width=5,
                     topk=4, check_every=3, filter_candidates=False,
                     target_max_new_tokens=4)

    rec = result["per_goal"][0]
    # optimiser cost: steps * search_width, recorded SEPARATELY from real queries.
    assert rec["n_optimizer_steps"] == 3
    assert rec["n_optimizer_forwards"] == 15               # 3 * 5
    assert result["n_optimizer_forwards_total"] == 15
    # exactly ONE real generation (check_every == n_steps == 3), counted at the
    # LLMHFSteered boundary — NOT the optimiser forwards.
    assert rec["n_target_forwards"] == 1
    assert d.n_target_forwards == 1
    assert rec["capped"] is False                          # GCG is exempt from the cap
    # the two accountings are not conflatable.
    assert rec["n_optimizer_forwards"] != rec["n_target_forwards"]


# -- hygiene: the GCG sbatch DEFENSES match the config (index-decode safety) ----


def test_gcg_sbatch_defenses_match_config():
    """The GCG sbatch's DEFENSES array must equal the config defenses IN ORDER
    (a reorder silently mislabels every shard), and it is a SEPARATE file from
    jailbreak_attacks.sbatch so it can never perturb the live-jobs array stride.

    Same explicitly named exceptions as test_jailbreak_run's config-parity
    test: the arms registered in methods.ARMS after the single-SI jailbreak
    config was frozen but not wired into it — the alphakl rungs and the
    diffmean floor — pinned to the registry tail so any OTHER registry
    growth still forces the config + sbatch update. bidpo_lora (2026-08-21,
    two-SI matrix only) joined them, and the cheap-estimator
    caa/actadd/actadd_front arms sit at the very tail; lora_r4 (2026-08-28,
    the lora arm's post-re-tune alias, not wired into this frozen axis)
    joined pinned to the slot right after its sibling lora_r1_all instead
    of the tail. (cast / circuit_breaker used to be named here too; both
    left the registry entirely 2026-08-20.) The stage2-evals did / adasteer
    arms came OUT of this set on 2026-08-19 when they were wired into the
    defense axis. The whole registry tail (not-wired + the wired
    did/adasteer) stays pinned so any OTHER registry growth still forces
    the config + sbatch update."""
    import re
    from pathlib import Path

    import yaml

    repo = Path(__file__).resolve().parents[1]
    from sisteer.eval.jailbreak import COMPILED

    _NOT_WIRED_ARMS = ("alphakl_null", "alphakl_soft", "alphakl_fisher",
                       "diffmean", "bidpo_lora", "caa_static_r",
                       # cheap-estimator arms (IFEval-only; appended last
                       # at the stage2-evals merge): never jailbreak
                       # defenses
                       "caa", "actadd", "actadd_front",
                       # psr_xfer_* transferred-PSR arms (merged 2026-09-02):
                       # serving/eval-lane only, trainer refuses in-repo —
                       # never jailbreak defenses, in NO DEFENSES array
                       "psr_xfer_from_e2b", "psr_xfer_from_e2b_rw1",
                       "psr_xfer_from_e4b", "psr_xfer_from_e4b_rw1",
                       # sdf-base-steering document-CE arms (2026-09-01,
                       # S_TWO_SI only; appended LAST at the sdfbase-mt merge,
                       # after psr_xfer_*): never single-SI jailbreak defenses
                       "psr_ce_all", "lora_r4_ce",
                       # sdf-base-steering DERIVED arms (2026-09-06, after
                       # the CE arms they decompose): serving-lane only and
                       # not trainable at all (methods.ARMS refuses their
                       # build and their train), so in NO DEFENSES array
                       "psr_ce_all_minus_ctl_distilled",
                       "lora_r4_ce_minus_ctl",
                       # ... and the UNDISTILLED paired write (kind
                       # psr_paired_write; relu-differenced-native-qwen,
                       # 2026-09-17), appended LAST: the same DERIVED,
                       # serving-lane-only arm (methods.DERIVED_ARMS names
                       # all three), so in NO DEFENSES array either
                       "psr_ce_all_minus_ctl")
    assert COMPILED[-20:] == ("alphakl_null", "alphakl_soft",
                              "alphakl_fisher", "diffmean", "bidpo_lora",
                              "caa_static_r", "did", "adasteer",
                              "caa", "actadd", "actadd_front",
                              # psr_xfer_* appended (merged 2026-09-02)
                              "psr_xfer_from_e2b", "psr_xfer_from_e2b_rw1",
                              "psr_xfer_from_e4b", "psr_xfer_from_e4b_rw1",
                              # the document-CE arms after them (sdfbase-mt merge)
                              "psr_ce_all", "lora_r4_ce",
                              # and the DERIVED arms after those (2026-09-06)
                              "psr_ce_all_minus_ctl_distilled",
                              "lora_r4_ce_minus_ctl",
                              # then the UNDISTILLED paired write (2026-09-17)
                              "psr_ce_all_minus_ctl")
    # The registry tail in RELATIVE order (each arm strictly before the
    # next) — a literal tail slice would move every time an arm is appended,
    # and appending is the rule; the derived arm is asserted separately as
    # the FINAL registered arm and the final unwired one (2026-09-10).
    _TAIL_ORDER = ("alphakl_null", "alphakl_soft",
                   "alphakl_fisher", "diffmean", "bidpo_lora",
                   "caa_static_r", "did", "adasteer",
                   "caa", "actadd", "actadd_front",
                   # psr_xfer_* appended after the cheap-estimator arms (merged 2026-09-02)
                   "psr_xfer_from_e2b", "psr_xfer_from_e2b_rw1",
                   "psr_xfer_from_e4b", "psr_xfer_from_e4b_rw1",
                   # the sdf-base DERIVED LoRA arm (2026-09-10), then the
                   # UNDISTILLED paired write appended LAST (2026-09-17)
                   "lora_r4_ce_minus_ctl", "psr_ce_all_minus_ctl")
    _idx = [COMPILED.index(a) for a in _TAIL_ORDER]
    assert _idx == sorted(_idx) and len(set(_idx)) == len(_idx)
    assert COMPILED[-2:] == ("lora_r4_ce_minus_ctl", "psr_ce_all_minus_ctl") == _NOT_WIRED_ARMS[-2:]
    # the rename alias sits exactly one past its sibling, nowhere else
    assert COMPILED[COMPILED.index("lora_r1_all") + 1] == "lora_r4"
    _UNWIRED_ARMS = set(_NOT_WIRED_ARMS) | {"lora_r4"}
    wired = [a for a in COMPILED if a not in _UNWIRED_ARMS]

    cfg = yaml.safe_load((repo / "configs/stage2_jailbreak.yaml").read_text())
    expected = ["none", "prompted", *wired]
    assert cfg["defenses"] == expected

    text = (repo / "scripts/sbatch/jailbreak_gcg.sbatch").read_text()
    m = re.search(r"^DEFENSES=\((.*?)\)\s*$", text, re.MULTILINE)
    assert m, "no DEFENSES=(...) array in jailbreak_gcg.sbatch"
    assert m.group(1).split() == expected
    # it dispatches gcg, and does NOT define its own ATTACKS array (which would
    # mean iterating attacks / perturbing a shared stride) — the comment may
    # REFERENCE the other script's array, but there is no such assignment here.
    assert "--attack gcg" in text
    assert not re.search(r"^ATTACKS=\(", text, re.MULTILINE)

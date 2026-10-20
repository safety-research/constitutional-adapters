"""Tier-0 tests: coarsened-KL float64 goldens at
1e-12 incl. p_tail=0 and q_tail=−inf ⇒ +inf; digest-addressed cache
round-trip + sidecar-verify refusals; identical-model KL == 0;
assistant-mask goldens on the Llama-3.1 template incl. the
boundary-retokenization loud failure; NLL windowing + per-window
nll_sum/token_count rows; the CPU tiny-model end-to-end tier0 run
producing real rows through the CellStore.

Base lane only."""

import json
import math

import pytest
import torch
import yaml

from pathlib import Path

from sisteer.capeval import corpora as cco
from sisteer.capeval import store as cs
from sisteer.capeval.serving.hf_engine import HFEngine
from sisteer.capeval.tiers import tier0 as t0
from sisteer.vectors import SteeringVector

TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"  # conftest pin

FIXTURES = Path(__file__).parent / "fixtures" / "capeval"
ROOT = Path(__file__).parents[1]
LLAMA31 = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA31_REV = "0e9e39f249a16976918f6564b8830bc894c89659"

NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# coarsened-KL goldens: |V|=8, top-2 atoms, float64, 1e-12 (§4.3)
# ---------------------------------------------------------------------------


def _dist(probs):
    """log-probs + tail log-mass for an 8-token dist with top-2 atoms
    (atoms = indices 0,1 by construction)."""
    assert len(probs) == 8 and abs(math.fsum(probs) - 1.0) < 1e-15
    logp = [math.log(p) if p > 0 else NEG_INF for p in probs[:2]]
    tail_mass = math.fsum(probs[2:])
    logp_tail = math.log(tail_mass) if tail_mass > 0 else NEG_INF
    return logp, logp_tail


def _expected_kl(p_probs, q_probs):
    """Independent hand computation over the 3-cell coarsening (atom0,
    atom1, tail) with math.fsum — no torch."""
    terms = []
    for pp, qq in [(p_probs[0], q_probs[0]), (p_probs[1], q_probs[1]),
                   (math.fsum(p_probs[2:]), math.fsum(q_probs[2:]))]:
        if pp == 0:
            continue
        if qq == 0:
            return float("inf")
        terms.append(pp * (math.log(pp) - math.log(qq)))
    return math.fsum(terms)


def test_kl_golden_generic_case_to_1e12():
    p = [0.40, 0.30, 0.10, 0.05, 0.05, 0.04, 0.03, 0.03]
    q = [0.25, 0.25, 0.20, 0.10, 0.05, 0.05, 0.05, 0.05]
    lp, lpt = _dist(p)
    lq, lqt = _dist(q)
    got = t0.kl_coarse_position(lp, lpt, lq, lqt)
    assert got == pytest.approx(_expected_kl(p, q), abs=1e-12)
    assert got > 0


def test_kl_golden_identical_dist_is_exactly_zero():
    p = [0.40, 0.30, 0.10, 0.05, 0.05, 0.04, 0.03, 0.03]
    lp, lpt = _dist(p)
    assert t0.kl_coarse_position(lp, lpt, lp, lpt) == 0.0


def test_kl_golden_p_tail_zero_contributes_zero():
    # all base mass on the two atoms: logp_tail = -inf must contribute 0
    p = [0.7, 0.3, 0, 0, 0, 0, 0, 0]
    q = [0.25, 0.25, 0.20, 0.10, 0.05, 0.05, 0.05, 0.05]
    lp, lpt = _dist(p)
    lq, lqt = _dist(q)
    assert lpt == NEG_INF
    got = t0.kl_coarse_position(lp, lpt, lq, lqt)
    expected = math.fsum([0.7 * (math.log(0.7) - math.log(0.25)),
                          0.3 * (math.log(0.3) - math.log(0.25))])
    assert got == pytest.approx(expected, abs=1e-12)
    assert math.isfinite(got)


def test_kl_golden_q_tail_neg_inf_with_p_tail_positive_is_plus_inf():
    p = [0.40, 0.30, 0.10, 0.05, 0.05, 0.04, 0.03, 0.03]  # p_tail > 0
    q = [0.7, 0.3, 0, 0, 0, 0, 0, 0]                      # q_tail = 0
    lp, lpt = _dist(p)
    lq, lqt = _dist(q)
    assert lqt == NEG_INF and lpt != NEG_INF
    got = t0.kl_coarse_position(lp, lpt, lq, lqt)
    assert got == float("inf")  # +inf, never capped (F4)


def test_kl_golden_q_atom_zero_with_p_atom_positive_is_plus_inf():
    lp, lpt = _dist([0.40, 0.30, 0.10, 0.05, 0.05, 0.04, 0.03, 0.03])
    lq = [math.log(0.5), NEG_INF]  # q gives atom 1 zero mass
    lqt = math.log(0.5)
    assert t0.kl_coarse_position(lp, lpt, lq, lqt) == float("inf")


def test_kl_refuses_non_float64():
    with pytest.raises(t0.Tier0Error, match="float64"):
        t0.kl_coarse_window(
            torch.zeros(1, 2, dtype=torch.float32),
            torch.zeros(1, dtype=torch.float64),
            torch.zeros(1, 2, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64))


def test_kl_refuses_nan_but_keeps_inf(tmp_path):
    """NaN atoms are a hard numerical failure and refuse
    loudly (NaN > κ is False, so it would otherwise evade the red flag);
    the legitimate ±inf log-mass behavior is untouched."""
    z2 = torch.zeros(1, 2, dtype=torch.float64)
    z1 = torch.zeros(1, dtype=torch.float64)
    nan2 = torch.full((1, 2), float("nan"), dtype=torch.float64)
    with pytest.raises(t0.Tier0Error, match="NaN"):
        t0.kl_coarse_window(nan2, z1, z2, z1)
    with pytest.raises(t0.Tier0Error, match="NaN"):
        t0.kl_coarse_window(z2, z1, nan2, z1)
    # -inf p atoms (zero mass) still contribute zero, no refusal
    lp = torch.tensor([[math.log(0.5), NEG_INF]], dtype=torch.float64)
    got = t0.kl_coarse_window(lp, torch.tensor([math.log(0.5)]).double(),
                              lp, torch.tensor([math.log(0.5)]).double())
    assert float(got[0]) == 0.0


# ---------------------------------------------------------------------------
# digest-keyed cache: round-trip, atomic-publish discipline, refusals (F5)
# ---------------------------------------------------------------------------


def _spec(**over):
    base = dict(model_revision="0" * 40, engine="hf", corpus_id="tiny",
                corpus_revision="local", slice_hash="ab" * 8,
                tokenizer_hash="cd" * 8, dtype="torch.float32", top_k=2,
                code_version=t0.TIER0_CODE_VERSION)
    base.update(over)
    return t0.cache_spec(**base)


def _arrays(n=5, k=2):
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 8, (n, k), generator=g, dtype=torch.int32)
    logp = -torch.rand(n, k, generator=g, dtype=torch.float64)
    tail = -torch.rand(n, generator=g, dtype=torch.float64)
    return ids, logp, tail


def test_cache_roundtrip_and_digest_key(tmp_path):
    spec = _spec()
    ids, logp, tail = _arrays()
    st = t0.publish_atom_cache(tmp_path, spec, ids, logp, tail)
    digest = t0.cache_digest(spec)
    assert st.name == f"{digest}.safetensors"
    assert (tmp_path / f"{digest}.json").exists()
    ids2, logp2, tail2 = t0.load_atom_cache(tmp_path, spec,
                                            expected_positions=5)
    assert torch.equal(ids, ids2)
    assert torch.equal(logp, logp2)  # float64 end-to-end, no round-trip loss
    assert torch.equal(tail, tail2)
    # digest moves with every field (the field list IS the identity)
    for k, v in [("model_revision", "1" * 40), ("engine", "vllm"),
                 ("top_k", 3), ("code_version", 99),
                 ("tokenizer_hash", "ee" * 8), ("dtype", "torch.bfloat16"),
                 ("slice_hash", "ff" * 8), ("corpus_revision", "x"),
                 ("corpus_id", "other")]:
        assert t0.cache_digest(_spec(**{k: v})) != digest, k


def test_cache_digest_refuses_wrong_field_set():
    spec = dict(_spec())
    spec["extra"] = 1
    with pytest.raises(t0.Tier0Error, match="extra"):
        t0.cache_digest(spec)
    spec2 = dict(_spec())
    del spec2["engine"]
    with pytest.raises(t0.Tier0Error, match="missing"):
        t0.cache_digest(spec2)


def test_cache_consumer_refuses_spec_mismatch(tmp_path):
    spec = _spec()
    ids, logp, tail = _arrays()
    t0.publish_atom_cache(tmp_path, spec, ids, logp, tail)
    # tamper the sidecar's engine field: consumer must refuse, naming both
    _, sc = t0.cache_paths(tmp_path, spec)
    side = json.loads(sc.read_text())
    side["engine"] = "vllm"
    sc.write_text(json.dumps(side))
    with pytest.raises(t0.CacheMismatchError, match="engine.*vllm.*hf"):
        t0.load_atom_cache(tmp_path, spec)


def test_cache_consumer_refuses_missing_sidecar(tmp_path):
    spec = _spec()
    ids, logp, tail = _arrays()
    t0.publish_atom_cache(tmp_path, spec, ids, logp, tail)
    _, sc = t0.cache_paths(tmp_path, spec)
    sc.unlink()  # tensors without sidecar = publish-order violation
    with pytest.raises(t0.CacheMismatchError, match="sidecar"):
        t0.load_atom_cache(tmp_path, spec)


def test_cache_consumer_refuses_position_count_drift(tmp_path):
    spec = _spec()
    ids, logp, tail = _arrays(n=5)
    t0.publish_atom_cache(tmp_path, spec, ids, logp, tail)
    with pytest.raises(t0.CacheMismatchError, match="window geometry"):
        t0.load_atom_cache(tmp_path, spec, expected_positions=7)


def test_cache_publish_refuses_wrong_dtypes(tmp_path):
    spec = _spec()
    ids, logp, tail = _arrays()
    with pytest.raises(t0.Tier0Error, match="float64"):
        t0.publish_atom_cache(tmp_path, spec, ids, logp.float(), tail)


def test_cache_publish_reuses_winner_never_clobbers(tmp_path):
    """A second publisher of the same digest reuses the
    winner's files (bit-identical afterwards) instead of renaming over
    them, and leaves no temporary files behind."""
    spec = _spec()
    ids, logp, tail = _arrays()
    st = t0.publish_atom_cache(tmp_path, spec, ids, logp, tail)
    before = st.read_bytes()
    st2 = t0.publish_atom_cache(tmp_path, spec, ids, logp - 1.0, tail)
    assert st2 == st
    assert st.read_bytes() == before  # the winner's cache is untouched
    assert not list(Path(tmp_path).glob("*.tmp"))
    loaded_ids, loaded_logp, _ = t0.load_atom_cache(tmp_path, spec,
                                                    expected_positions=5)
    assert torch.equal(loaded_logp, logp)


def test_cache_publish_serializes_on_the_digest_lock(tmp_path):
    """Publication holds an exclusive flock on
    <digest>.lock — while another writer holds it, publish blocks, and
    completes once it is released (event-based, no fixed tmp names to
    rename out from under each other)."""
    import fcntl
    import threading

    spec = _spec()
    ids, logp, tail = _arrays()
    digest = t0.cache_digest(spec)
    lock_path = Path(tmp_path) / f"{digest}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    holder = open(lock_path, "w")
    fcntl.flock(holder, fcntl.LOCK_EX)

    published = threading.Event()

    def worker():
        t0.publish_atom_cache(tmp_path, spec, ids, logp, tail)
        published.set()

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    assert not published.wait(0.3), "publish proceeded under a held lock"
    fcntl.flock(holder, fcntl.LOCK_UN)
    holder.close()
    assert published.wait(10), "publish never completed after lock release"
    th.join(timeout=10)
    t0.load_atom_cache(tmp_path, spec, expected_positions=5)


def test_cache_missing_is_a_loud_refusal(tmp_path):
    with pytest.raises(t0.Tier0Error, match="calibration trial 0"):
        t0.load_atom_cache(tmp_path, _spec())


# ---------------------------------------------------------------------------
# assistant-content mask
# ---------------------------------------------------------------------------


def _llama31_tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(LLAMA31, revision=LLAMA31_REV)
    except Exception as e:  # loud skip: names what must be fetched
        pytest.skip(
            f"Llama-3.1 tokenizer not loadable ({type(e).__name__}) — run "
            "scripts/capeval_fetch_data.py (or any hf_hub_download of "
            f"{LLAMA31}@{LLAMA31_REV[:8]}) on this machine first; the "
            "template goldens require the real tokenizer")


TWO_TURN = [{"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello there"}]
MULTI_TURN = [{"role": "user", "content": "Name a color."},
              {"role": "assistant", "content": "Blue."},
              {"role": "user", "content": "Another?"},
              {"role": "assistant", "content": "Green and gold."}]
EMPTY_TURN = [{"role": "user", "content": "Say nothing."},
              {"role": "assistant", "content": ""},
              {"role": "user", "content": "Ok now speak."},
              {"role": "assistant", "content": "Words."}]


def test_mask_golden_two_turn_llama31():
    tok = _llama31_tokenizer()
    full, pos = t0.assistant_content_mask(tok, TWO_TURN)
    # pinned exact indices on the Llama-3.1 template @ 0e9e39f2:
    # 39 tokens; content = ["Hello", " there"]; the terminal <|eot_id|>
    # at 38 is EXCLUDED, as is every header/system scaffold token.
    assert len(full) == 39
    assert pos == [36, 37]
    assert tok.decode([full[36]]) == "Hello"
    assert tok.decode([full[37]]) == " there"
    assert full[38] == tok.convert_tokens_to_ids("<|eot_id|>")


def test_mask_golden_multi_assistant_turn_llama31():
    tok = _llama31_tokenizer()
    full, pos = t0.assistant_content_mask(tok, MULTI_TURN)
    assert len(full) == 58
    assert pos == [39, 40, 53, 54, 55, 56]
    assert [tok.decode([full[p]]) for p in pos] == [
        "Blue", ".", "Green", " and", " gold", "."]


def test_mask_golden_empty_content_turn_llama31():
    tok = _llama31_tokenizer()
    full, pos = t0.assistant_content_mask(tok, EMPTY_TURN)
    # the empty assistant turn contributes NO positions; only the final
    # turn's content ("Words", ".") is masked in.
    assert len(full) == 55
    assert pos == [52, 53]
    assert [tok.decode([full[p]]) for p in pos] == ["Words", "."]


def _tiny_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TINY_LLAMA)


# A structural replica of the Llama-3 header/eot scaffold, expressed with
# ADDED special tokens on the tiny sentencepiece tokenizer — always runs
# (no gated download), exercising the same mask logic as the real
# template: headers in the generation-prompt render, content + terminal
# EOT in the suffix, EOT stripped as a special token.
REPLICA_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>' }}"
    "{{ message['content'] }}{{ '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>' }}"
    "{% endif %}")


def _replica_tokenizer():
    tok = _tiny_tokenizer()
    tok.add_special_tokens({"additional_special_tokens": [
        "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]})
    tok.chat_template = REPLICA_TEMPLATE
    return tok


def test_mask_replica_template_content_only():
    tok = _replica_tokenizer()
    full, pos = t0.assistant_content_mask(tok, TWO_TURN)
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    header = tok.convert_tokens_to_ids("<|start_header_id|>")
    masked = [full[p] for p in pos]
    assert eot not in masked and header not in masked
    assert tok.decode(masked).strip() == "Hello there"
    # every scaffold token around the span is excluded
    assert full[pos[-1] + 1] == eot


def test_mask_replica_empty_content_and_multi_turn():
    tok = _replica_tokenizer()
    full, pos = t0.assistant_content_mask(tok, EMPTY_TURN)
    assert tok.decode([full[p] for p in pos]).strip() == "Words."
    full2, pos2 = t0.assistant_content_mask(tok, MULTI_TURN)
    assert tok.decode([full2[p] for p in pos2]) == "Blue.Green and gold."


# a template whose assistant header ends in a trailing SPACE: the
# sentencepiece prefix-space merge ('▁' | '▁Hello') breaks token-prefix
# identity between the generation-prompt render and the message render —
# the verified loud-failure construction.
BOUNDARY_BREAK_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'].upper() + ': ' + message['content'] + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ 'ASSISTANT: ' }}{% endif %}")


def test_mask_boundary_retokenization_fails_loudly():
    tok = _tiny_tokenizer()
    tok.chat_template = BOUNDARY_BREAK_TEMPLATE
    with pytest.raises(t0.MaskError, match="boundary retokenization"):
        t0.assistant_content_mask(
            tok, [{"role": "user", "content": "Hi"},
                  {"role": "assistant", "content": "Hello"}])


# ---------------------------------------------------------------------------
# NLL windowing (§4.2)
# ---------------------------------------------------------------------------


def _fixture_texts():
    return [json.loads(l)["text"] for l in
            (FIXTURES / "tier0_tiny_corpus.jsonl").read_text().splitlines()]


def test_windows_from_texts_geometry_and_determinism():
    tok = _tiny_tokenizer()
    texts = _fixture_texts()
    w1 = t0.windows_from_texts(tok, "tiny", texts, window_len=32,
                               token_budget=128, min_window_tokens=8)
    w2 = t0.windows_from_texts(tok, "tiny", texts, window_len=32,
                               token_budget=128, min_window_tokens=8)
    assert w1 == w2  # deterministic
    assert [it.item_id for it in w1.items] == [
        f"tiny/w{i:05d}" for i in range(len(w1.items))]
    for it in w1.items:
        assert it.input_ids[0] == tok.bos_token_id  # unscored conditioning BOS
        assert it.score_positions == tuple(range(1, len(it.input_ids)))
        assert len(it.score_positions) <= 32
    assert w1.n_positions == sum(len(i.score_positions) for i in w1.items)


def test_windows_drop_short_trailing_fragment():
    tok = _tiny_tokenizer()
    w = t0.windows_from_texts(tok, "tiny", ["one two three"], window_len=64,
                              min_window_tokens=32)
    assert w.items == ()  # 3 tokens < min_window_tokens


@pytest.fixture(scope="module")
def tiny_engine(tiny_llama):
    model, tok = tiny_llama
    eng = HFEngine(TINY_LLAMA, revision=None, preloaded=(model, tok))
    eng.load()
    return eng


def test_engine_nll_matches_manual_forward(tiny_engine):
    tok = tiny_engine.tokenizer
    w = t0.windows_from_texts(tok, "tiny", _fixture_texts(), window_len=24,
                              token_budget=48, min_window_tokens=8)
    item = w.items[0]
    scores = list(tiny_engine.score_corpus(
        t0.WindowedCorpus("tiny", (item,)), None, None))
    assert len(scores) == len(item.score_positions)
    with torch.no_grad():
        logits = tiny_engine.model(
            input_ids=torch.tensor([list(item.input_ids)])).logits[0]
    for ps, pos in zip(scores, item.score_positions):
        lp = torch.log_softmax(logits[pos - 1].double(), dim=-1)
        assert ps.token_nll == pytest.approx(
            float(-lp[item.input_ids[pos]]), abs=1e-9)
        assert ps.atom_probs is None


def test_chat_nll_scores_masked_positions_only(tiny_engine, tmp_path):
    """Assistant-masked chat NLL end-to-end on the engine: item =
    conversation, scored positions = the assistant content mask, per-item
    nll_sum matching a manual masked forward."""
    tok = tiny_engine.tokenizer
    convs = [[{"role": "user", "content": "Hi"},
              {"role": "assistant", "content": "Hello there"}],
             [{"role": "user", "content": "Name a color."},
              {"role": "assistant", "content": "Blue."}]]
    w = t0.windows_from_conversations(tok, "chat", convs, ["c0", "c1"],
                                      max_tokens=64)
    assert len(w.items) == 2
    for conv, item in zip(convs, w.items):
        full, pos = t0.assistant_content_mask(tok, conv)
        assert item.score_positions == tuple(p for p in pos if p >= 1)
    scores = list(tiny_engine.score_corpus(w, None, None))
    assert len(scores) == w.n_positions
    item = w.items[0]
    with torch.no_grad():
        logits = tiny_engine.model(
            input_ids=torch.tensor([list(item.input_ids)])).logits[0]
    manual = 0.0
    for pos in item.score_positions:
        lp = torch.log_softmax(logits[pos - 1].double(), dim=-1)
        manual += float(-lp[item.input_ids[pos]])
    got = sum(s.token_nll for s in scores[:len(item.score_positions)])
    assert got == pytest.approx(manual, abs=1e-9)


def test_chat_assistant_token_budget_truncates_corpus():
    tok = _tiny_tokenizer()
    conv = [{"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello there my friend"}]
    convs = [conv, conv, conv]
    # deep-copy identity is irrelevant; ids must be unique per item
    w = t0.windows_from_conversations(tok, "chat", convs, ["a", "b", "c"],
                                      assistant_token_budget=1, max_tokens=64)
    assert len(w.items) == 1  # first conversation already meets the budget


def test_identical_model_kl_is_exactly_zero(tiny_engine, tmp_path):
    """Base atoms built and immediately re-scored by the same model in
    the same process: logq == logp bitwise, KL exactly 0.0 (§13)."""
    tok = tiny_engine.tokenizer
    w = t0.windows_from_texts(tok, "tiny", _fixture_texts(), window_len=16,
                              token_budget=48, min_window_tokens=8)
    base = list(tiny_engine.score_corpus(w, None, 4))  # top-4 atoms
    ids = torch.stack([p.atom_probs.ids for p in base])
    logp = torch.stack([p.atom_probs.logp for p in base])
    tail = torch.tensor([p.atom_probs.logp_tail for p in base],
                        dtype=torch.float64)
    again = list(tiny_engine.score_corpus(w, None, ids))
    logq = torch.stack([p.atom_probs.logp for p in again])
    qtail = torch.tensor([p.atom_probs.logp_tail for p in again],
                         dtype=torch.float64)
    kl = t0.kl_coarse_window(logp, tail, logq, qtail)
    assert torch.equal(kl, torch.zeros_like(kl))


# ---------------------------------------------------------------------------
# CPU tiny-model end-to-end: tier0.run producing REAL rows via CellStore
# ---------------------------------------------------------------------------


def _domain_of():
    doms = yaml.safe_load(
        (ROOT / "configs" / "capeval" / "domains.yaml").read_text())["domains"]
    return {t: name for name, d in doms.items() for t in d["tasks"]}


def _template(arm_name, kind, coeff, ihash, trial, attempt=0):
    return dict(sweep="tier0_e2e", run_id="run0", model_id=TINY_LLAMA,
                model_revision="0" * 40, arm_name=arm_name, kind=kind,
                coeff=coeff, intervention_hash=ihash, engine="hf",
                trial_id=trial, seed=trial, attempt=attempt,
                harness_config_hash="0f1e2d3c4b5a6978",
                env_digest="99aabbccddeeff00", slurm_job_id=None)


def _corpus_slice(name="tinytext"):
    texts = _fixture_texts()
    return cco.CorpusSlice(
        name=name, kind="text",
        item_ids=tuple(f"doc{i}" for i in range(len(texts))),
        items=tuple(texts), spec={"mode": "fixture"}, slice_hash="ab" * 8)


def _prompt_slice(name, prompts):
    return cco.CorpusSlice(
        name=name, kind="prompts",
        item_ids=tuple(f"p{i}" for i in range(len(prompts))),
        items=tuple(prompts), spec={"mode": "fixture"}, slice_hash="cd" * 8)


def _tier0_spec(template, *, build_cache, relevant=None, thresholds=None):
    return t0.Tier0Spec(
        template=template,
        domain_of=_domain_of(),
        corpora=[t0.Tier0CorpusCfg(
            slice=_corpus_slice(), nll_task="tier0/nll_wikitext103",
            kl_task="tier0/kl_coarse_wikitext103", token_budget=96)],
        canary=t0.CanaryCfg(
            slice=_prompt_slice("canary", ["Say hi.", "Name a color.",
                                           "Count to three."]),
            gen={"max_new_tokens": 8, "do_sample": False, "batch_size": 2},
            thresholds=thresholds),
        relevant_corpus=relevant,
        top_k=4, window_len=16, min_window_tokens=8,
        build_cache=build_cache)


def _mk_store(tmp_path, key):
    store = cs.CellStore(tmp_path, key)
    store.write_cell_manifest({
        "intervention_hash": "a1b2c3d4e5f60718",
        "harness_config_hash": "0f1e2d3c4b5a6978",
        "model_revision": "0" * 40, "engine": "hf"})
    return store


def _steering_artifact(tmp_path, tok_dim=16, layer=1):
    sv = SteeringVector(vectors={layer: torch.randn(
        tok_dim, generator=torch.Generator().manual_seed(7)) * 2.0},
        method="distill_soft", model_name=TINY_LLAMA, instruction_id="test")
    path = tmp_path / "e2e_vector.safetensors"
    sv.save(path)
    return path


def test_tier0_end_to_end_base_then_arm(tiny_engine, tmp_path):
    from sisteer.capeval.manifest import AblationSpec

    cache_dir = tmp_path / "cache" / "klatoms"
    dom = _domain_of()

    # --- base cell (calibration trial 0): builds the KL atom cache ---
    base_store = _mk_store(tmp_path, "none__tier0__t0__a0__a1b2c3d4__0f1e2d3c")
    base_tpl = _template("none", "none", None, "a1b2c3d4e5f60718", trial=0)
    base_summary = t0.run(tiny_engine, None,
                          _tier0_spec(base_tpl, build_cache=True),
                          base_store, cache_dir)
    assert base_summary["kl_coarse_lb"]["tier0/kl_coarse_wikitext103"] == 0.0
    assert base_summary["kl_concentration"] is None  # notice, not silence
    assert base_summary["confirmation_needed"] == []

    # --- arm cell: consumes the cache, localizes on its own slice ---
    arm_spec = AblationSpec(
        name="arm_e2e", kind="steering",
        artifact=str(_steering_artifact(tmp_path)), layer=1, coeffs=[3.0])
    arm = tiny_engine.arm(arm_spec, 3.0)
    arm_store = _mk_store(tmp_path, "arm_e2e__tier0__t1__a0__a1b2c3d4__0f1e2d3c")
    arm_tpl = _template("arm_e2e", "steering", 3.0, arm.intervention_hash,
                        trial=1)
    relevant = _prompt_slice("target_rel", ["Write about rivers.",
                                            "Write about bread."])
    arm_summary = t0.run(tiny_engine, arm,
                         _tier0_spec(arm_tpl, build_cache=False,
                                     relevant=relevant),
                         arm_store, cache_dir)

    # real rows through the store API
    rows = arm_store._parsed_rows()
    assert rows, "arm cell filed no rows"
    assert {r["engine"] for r in rows} == {"hf"}  # rows carry the engine
    assert {r["arm_name"] for r in rows} == {"arm_e2e"}
    by_task_metric = {}
    for r in rows:
        by_task_metric.setdefault((r["task"], r["metric"]), []).append(r)

    n_windows = len([r for r in rows
                     if r["task"] == "tier0/nll_wikitext103"
                     and r["metric"] == "nll_sum"])
    assert n_windows >= 2
    # F14 pairing: token_count rows cover the same items as nll_sum rows
    nll_items = {r["item_id"] for r in
                 by_task_metric[("tier0/nll_wikitext103", "nll_sum")]}
    tc_items = {r["item_id"] for r in
                by_task_metric[("tier0/nll_wikitext103", "token_count")]}
    assert nll_items == tc_items
    # per-window METRIC_KL rows (the κ-derivation item contract) + the
    # F14 sum/count rows
    assert ("tier0/kl_coarse_wikitext103", "kl_coarse_lb") in by_task_metric
    assert ("tier0/kl_coarse_wikitext103", "kl_sum") in by_task_metric
    # a steered arm on the base cache has strictly positive KL
    assert arm_summary["kl_coarse_lb"]["tier0/kl_coarse_wikitext103"] > 0
    # that positive KL is an ANNOTATION — it raises no red flag,
    # requests no confirmation rerun, and files no flag_kl_gt_* row
    assert arm_summary["confirmation_needed"] == []
    assert arm_summary["flags"] == []
    assert not any(r["metric"].startswith("flag_kl_gt") for r in rows)
    # canary: one row per prompt + the five aggregate rows at __aggregate__
    assert len(by_task_metric[("tier0/canary", "canary_gen")]) == 3
    agg = {r["metric"] for r in rows
           if r["task"] == "tier0/canary" and r["item_id"] == "__aggregate__"}
    assert agg == set(t0.CANARY_INDICATORS)
    # localization on the arm's own slice: per-item rows + summary
    assert ("tier0/kl_localization", "kl_sum") in by_task_metric
    assert ("tier0/kl_localization", "kl_target_mean") in by_task_metric
    assert arm_summary["kl_concentration"] is not None
    # domains resolve per the registry
    for r in rows:
        assert r["domain"] == dom[r["task"]], r["task"]


def test_tier0_resume_is_idempotent(tiny_engine, tmp_path):
    cache_dir = tmp_path / "cache" / "klatoms"
    store = _mk_store(tmp_path, "none__tier0__t0__a0__a1b2c3d4__0f1e2d3c")
    tpl = _template("none", "none", None, "a1b2c3d4e5f60718", trial=0)
    t0.run(tiny_engine, None, _tier0_spec(tpl, build_cache=True),
           store, cache_dir)
    first = store._parsed_rows()
    t0.run(tiny_engine, None, _tier0_spec(tpl, build_cache=True),
           store, cache_dir)
    second = store._parsed_rows()
    # no duplicate (task, item, metric) anywhere — calibration's readers
    # require exactly one aggregate row per metric per trial
    keys = [(r["task"], r["item_id"], r["metric"]) for r in second]
    assert len(keys) == len(set(keys))
    assert len(second) == len(first)


def test_tier0_refuses_missing_cache_on_arm_cell(tiny_engine, tmp_path):
    tpl = _template("none", "none", None, "a1b2c3d4e5f60718", trial=1)
    store = _mk_store(tmp_path, "none__tier0__t1__a0__a1b2c3d4__0f1e2d3c")
    with pytest.raises(t0.Tier0Error, match="calibration"):
        t0.run(tiny_engine, None, _tier0_spec(tpl, build_cache=False),
               store, tmp_path / "empty_cache")


class _TokenArm:
    """A prompt-transforming (token-kind) arm handle stub."""

    name, kind, coeff, intervention_hash = "tok", "sisteer_object", 1.0, "11" * 8
    resolved_spec = {"positions": "all"}

    def ctx(self):
        import contextlib

        return contextlib.nullcontext()

    def transform_prompt(self, s):
        return "[SENTINEL] " + s


def test_tier0_refuses_prompt_transforming_arm_on_teacher_forced_probes(
        tiny_engine, tmp_path):
    """The teacher-forced NLL/KL/localization probes never
    apply arm.transform_prompt, so a token-kind arm would be scored on
    the UNTRANSFORMED corpus and file base numbers as if they were the
    arm's — refuse loudly (arm-transformed scored corpora are v1)."""
    store = _mk_store(tmp_path, "tok__tier0__t1__a0__a1b2c3d4__0f1e2d3c")
    tpl = _template("tok", "sisteer_object", 1.0, "11" * 8, trial=1)
    with pytest.raises(t0.Tier0Error, match="v1 item"):
        t0.run(tiny_engine, _TokenArm(),
               _tier0_spec(tpl, build_cache=False), store,
               tmp_path / "cache")


# ---------------------------------------------------------------------------
# confirmation protocol: confirm_probes gates *_confirmed
# ---------------------------------------------------------------------------


from sisteer.capeval.serving import GenOut  # noqa: E402


class _CannedGenEngine(HFEngine):
    """A real HFEngine whose generate() returns canned GenOuts, matched
    by which canary prompt the rendered string contains — so tier0.run's
    canary path (thresholds, rows, confirmation) runs end-to-end with
    controlled degeneration."""

    def __init__(self, model, tok, outs_by_prompt):
        super().__init__(TINY_LLAMA, revision=None, preloaded=(model, tok))
        self._outs = outs_by_prompt

    def generate(self, prompts, arm, gen_cfg):
        outs = []
        for p in prompts:
            match = [o for key, o in self._outs.items() if key in p]
            assert len(match) == 1, p
            outs.append(match[0])
        return outs


_CANARY_PROMPTS = ["Say hi.", "Name a color.", "Count to three."]


def _degenerate_outs():
    """Trips exactly two indicators against _strict_thresholds():
    max_rep_run_p95 (ids repeat 20x) and distinct3 (identical texts)."""
    return {p: GenOut(text="spam spam spam spam spam",
                      gen_token_logprobs=[-0.5] * 20,
                      ended_with_eos=False, gen_token_ids=[7] * 20)
            for p in _CANARY_PROMPTS}


def _healthy_outs():
    texts = ["Hello there, nice to meet you today.",
             "The color I pick is a deep blue.",
             "One, two, three — counting is easy."]
    return {p: GenOut(text=t, gen_token_logprobs=[-0.5] * 12,
                      ended_with_eos=True,
                      gen_token_ids=list(range(100 + i * 20, 111 + i * 20)) + [2])
            for i, (p, t) in enumerate(zip(_CANARY_PROMPTS, texts))}


def _strict_thresholds():
    return t0.CanaryThresholds(fail_lines={
        "max_rep_run_p95": 4.0, "distinct3": 0.5,
        "mean_tok_logprob": -10.0, "nonascii_frac": 0.9,
        "abrupt_eos_rate": 0.9})


def _canary_only_spec(template):
    return t0.Tier0Spec(
        template=template, domain_of=_domain_of(), corpora=[],
        canary=t0.CanaryCfg(
            slice=_prompt_slice("canary", list(_CANARY_PROMPTS)),
            gen={"max_new_tokens": 8, "do_sample": False, "batch_size": 2},
            thresholds=_strict_thresholds()),
        relevant_corpus=None, build_cache=False)


def _canary_run(tiny_llama, tmp_path, outs, attempt, confirm_probes=None,
                key="c0"):
    model, tok = tiny_llama
    eng = _CannedGenEngine(model, tok, outs)
    store = _mk_store(tmp_path / key,
                      f"none__tier0__t1__a{attempt}__a1b2c3d4__0f1e2d3c")
    tpl = _template("none", "none", None, "a1b2c3d4e5f60718", trial=1,
                    attempt=attempt)
    summary = t0.run(eng, None, _canary_only_spec(tpl), store,
                     tmp_path / key / "cache", confirm_probes=confirm_probes)
    metrics = {r["metric"] for r in store._parsed_rows()}
    return summary, metrics, store


def test_canary_attempt0_trip_requests_confirmation_not_confirmed(
        tiny_llama, tmp_path):
    summary, metrics, _ = _canary_run(tiny_llama, tmp_path,
                                      _degenerate_outs(), attempt=0)
    assert summary["canary"]["outcome"] == t0.CANARY_NEEDS_CONFIRMATION
    assert {"probe": "canary",
            "tripped": summary["canary"]["tripped"]} in \
        [{k: v for k, v in c.items()} for c in summary["confirmation_needed"]]
    assert len(summary["canary"]["tripped"]) >= 2
    assert "flag_canary_confirmed" not in metrics
    assert "flag_canary_unconfirmed" not in metrics


def test_canary_attempt1_confirms_only_when_requested(tiny_llama, tmp_path):
    summary, metrics, _ = _canary_run(
        tiny_llama, tmp_path, _degenerate_outs(), attempt=1,
        confirm_probes=frozenset({"canary"}), key="req")
    assert summary["canary"]["outcome"] == t0.CANARY_CONFIRMED_FAIL
    assert "flag_canary_confirmed" in metrics
    assert summary["flags"] and summary["flags"][0]["probe"] == "canary"


def test_canary_attempt1_unrequested_trip_is_new_unconfirmed(
        tiny_llama, tmp_path):
    """Attempt 1 rerun requested by the KL probe alone; a fresh canary
    trip must file a NEW unconfirmed flag row — never
    flag_canary_confirmed."""
    summary, metrics, _ = _canary_run(
        tiny_llama, tmp_path, _degenerate_outs(), attempt=1,
        confirm_probes=frozenset({"dist_shift"}), key="unreq")
    assert summary["canary"]["outcome"] == t0.CANARY_NEW_UNCONFIRMED
    assert "flag_canary_confirmed" not in metrics
    assert "flag_canary_unconfirmed" in metrics
    [flag] = summary["flags"]
    assert flag["probe"] == "canary" and flag["confirmed"] is False


def test_canary_attempt1_requested_but_clean_is_not_reproduced(
        tiny_llama, tmp_path):
    summary, metrics, _ = _canary_run(
        tiny_llama, tmp_path, _healthy_outs(), attempt=1,
        confirm_probes=frozenset({"canary"}), key="clean")
    assert summary["canary"]["outcome"] == t0.CANARY_NOT_REPRODUCED
    assert "flag_canary_confirmed" not in metrics
    assert "flag_canary_unconfirmed" not in metrics


def test_confirm_probes_contract_is_validated(tiny_llama, tmp_path):
    with pytest.raises(t0.Tier0Error, match="without confirm_probes"):
        _canary_run(tiny_llama, tmp_path, _healthy_outs(), attempt=1,
                    key="v1")
    with pytest.raises(t0.Tier0Error, match="attempt 0"):
        _canary_run(tiny_llama, tmp_path, _healthy_outs(), attempt=0,
                    confirm_probes=frozenset({"canary"}), key="v2")
    with pytest.raises(t0.Tier0Error, match="confirmable"):
        _canary_run(tiny_llama, tmp_path, _healthy_outs(), attempt=1,
                    confirm_probes=frozenset({"mystery"}), key="v3")


def test_canary_aggregate_rows_equal_canary_values(tiny_llama, tmp_path):
    """simplify item 4: the filed per-indicator aggregate rows are
    EXACTLY canary_values of the same generations — one implementation."""
    outs = _healthy_outs()
    _, _, store = _canary_run(tiny_llama, tmp_path, outs, attempt=0,
                              key="agg")
    filed = {r["metric"]: r["value"] for r in store._parsed_rows()
             if r["item_id"] == "__aggregate__"}
    expected = t0.canary_values([outs[p] for p in _CANARY_PROMPTS])
    assert filed == pytest.approx(expected)


def test_tier0_yaml_refuses_retired_kappa_floor(tmp_path):
    """kappa_floor is no longer in the tier0.yaml vocabulary — a stale
    yaml still carrying it must refuse loudly, never silently imply the
    retired κ flag path still exists."""
    src = (ROOT / "configs" / "capeval" / "tier0.yaml").read_text()
    stale = tmp_path / "tier0.yaml"
    stale.write_text(src + "\nkappa_floor: 1.0e-4\n")
    with pytest.raises(t0.Tier0Error, match="kappa_floor"):
        t0.load_tier0_yaml(stale)


def test_tier0_yaml_loads_and_pins_conventions():
    cfg = t0.load_tier0_yaml(ROOT / "configs" / "capeval" / "tier0.yaml")
    assert cfg["top_k"] == 64
    assert cfg["window_len"] == 1024
    assert cfg["code_version"] == t0.TIER0_CODE_VERSION
    assert [c["name"] for c in cfg["corpora"]] == [
        "wikitext103", "stack_py", "ultrachat"]
    assert cfg["canary"]["n_prompts"] == 300
    # every task named in tier0.yaml is registered in domains.yaml
    dom = _domain_of()
    for c in cfg["corpora"]:
        assert c["nll_task"] in dom and c["kl_task"] in dom
    assert cfg["canary"]["task"] in dom
    assert cfg["localization"]["task"] in dom

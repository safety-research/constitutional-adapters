"""Tests for corpora.py (pins, deterministic slices, CorpusRef
resolution) and targets.py (plugin registry +
sisteer_compliance). All offline: hub reads are monkeypatched
onto local files; no load_dataset anywhere."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from sisteer.capeval import corpora as cp
from sisteer.capeval import targets as tg
from sisteer.capeval.corpora import CorpusError
from sisteer.capeval.manifest import (
    AblationSpec,
    CorpusRef,
    Manifest,
    PrimarySpec,
    TargetSpec,
)
from sisteer.capeval.serving import ArmHandle, GenOut
from sisteer.capeval.targets import TargetError

FIXTURES = Path(__file__).parent / "fixtures" / "capeval"
SHA = "a" * 40


# --------------------------------------------------------------------------
# pins.yaml
# --------------------------------------------------------------------------


def _pins_entry(**kw):
    """A pins entry in the shape scripts/capeval_fetch_data.py writes."""
    entry = {"repo_id": "org/data", "revision": SHA, "verify": "parquet",
             "patterns": ["*.parquet"], "bytes": 123, "files": 1}
    entry.update(kw)
    return entry


def test_load_pins_absent_is_the_phase0_error(tmp_path):
    with pytest.raises(CorpusError, match="capeval_fetch_data"):
        cp.load_pins(tmp_path / "pins.yaml")


def test_find_pins_path_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv(cp.PINS_ENV_VAR, str(tmp_path / "p.yaml"))
    assert cp.find_pins_path() == tmp_path / "p.yaml"
    monkeypatch.delenv(cp.PINS_ENV_VAR)
    assert cp.find_pins_path().name == "pins.yaml"


def test_load_pins_validation(tmp_path):
    p = tmp_path / "pins.yaml"

    def write(entry):
        # extra top-level keys (evalplus/models/resolved_at) are the fetch
        # script's other sections — tolerated, not read here
        p.write_text(yaml.safe_dump({"datasets": {"wikitext103": entry},
                                     "evalplus": {}, "models": {},
                                     "resolved_at": "2026-08-15"}))

    write(_pins_entry())
    pins = cp.load_pins(p)
    assert cp.pin_entry(pins, "wikitext103")["revision"] == SHA
    with pytest.raises(CorpusError, match="no pin named"):
        cp.pin_entry(pins, "mystery")

    write(_pins_entry(revision="main"))
    with pytest.raises(CorpusError, match="40-hex"):
        cp.load_pins(p)
    # csv became a LEGAL verify kind with the or_bench pin
    # (capeval-controls §3.3); tsv stays the invalid example
    write(_pins_entry(verify="csv"))
    assert cp.pin_entry(cp.load_pins(p), "wikitext103")["verify"] == "csv"
    write(_pins_entry(verify="tsv"))
    with pytest.raises(CorpusError, match="verify"):
        cp.load_pins(p)
    write(_pins_entry(patterns=[]))
    with pytest.raises(CorpusError, match="patterns"):
        cp.load_pins(p)
    write(_pins_entry(patterns=None))  # null patterns = whole repo: valid
    cp.load_pins(p)
    write({**_pins_entry(), "surprise": 1})
    with pytest.raises(CorpusError, match="unknown key"):
        cp.load_pins(p)
    entry = _pins_entry()
    del entry["verify"]
    write(entry)
    with pytest.raises(CorpusError, match="verify"):
        cp.load_pins(p)


def test_load_pins_reads_the_real_checked_in_pins():
    """The checked-in configs/capeval/pins.yaml must satisfy this
    module's validation — the live contract check."""
    pins = cp.load_pins(Path(__file__).parents[1] / "configs" / "capeval"
                        / "pins.yaml")
    for name in ("wikitext103", "stack_py", "ultrachat", "canary_prompts"):
        entry = cp.pin_entry(pins, name)
        assert len(entry["revision"]) == 40


# --------------------------------------------------------------------------
# deterministic slicing + hashing
# --------------------------------------------------------------------------


def test_deterministic_order_is_content_canonical():
    items = ["cherry", "apple", "banana", "date"]
    order1 = cp.deterministic_order(items, seed=0)
    # same CONTENT in a different stored order -> same canonical sequence
    shuffled = ["date", "banana", "apple", "cherry"]
    order2 = cp.deterministic_order(shuffled, seed=0)
    assert [items[i] for i in order1] == [shuffled[i] for i in order2]
    # seed changes the permutation
    assert [items[i] for i in cp.deterministic_order(items, seed=1)] \
        != [items[i] for i in order1]


def test_resolve_corpus_ref_path_mode():
    ref = CorpusRef(path=str(FIXTURES / "pairs_synthetic.jsonl"),
                    slice={"first": 2, "field": "prompt"})
    s = cp.resolve_corpus_ref(ref)
    assert s.kind == "prompts" and len(s) == 2
    assert all(isinstance(p, str) for p in s.items)
    assert len(set(s.item_ids)) == 2
    # slice hash is stable across calls and covers the resolved item ids
    s2 = cp.resolve_corpus_ref(ref)
    assert s2.slice_hash == s.slice_hash
    assert s.spec["item_ids"] == list(s.item_ids)
    assert "content_digest" in s.spec


def test_resolve_corpus_ref_hash_tracks_content(tmp_path):
    src = (FIXTURES / "pairs_synthetic.jsonl").read_text()
    p = tmp_path / "pairs.jsonl"
    p.write_text(src)
    ref = CorpusRef(path=str(p), slice={"first": 2, "field": "prompt"})
    h1 = cp.resolve_corpus_ref(ref).slice_hash
    p.write_text(src.replace("rivers", "brooks"))
    assert cp.resolve_corpus_ref(ref).slice_hash != h1


def test_resolve_corpus_ref_refusals(tmp_path):
    ok = {"first": 2, "field": "prompt"}
    path = str(FIXTURES / "pairs_synthetic.jsonl")
    with pytest.raises(CorpusError, match="unknown slice key"):
        cp.resolve_corpus_ref(CorpusRef(path=path,
                                        slice={**ok, "shuffle": True}))
    with pytest.raises(CorpusError, match="does not exist"):
        cp.resolve_corpus_ref(CorpusRef(path=str(tmp_path / "no.jsonl"),
                                        slice=ok))
    with pytest.raises(CorpusError, match="only 2 rows"):
        cp.resolve_corpus_ref(CorpusRef(path=path,
                                        slice={"first": 5, "field": "prompt"}))
    with pytest.raises(CorpusError, match="not in"):
        cp.resolve_corpus_ref(CorpusRef(path=path,
                                        slice={"first": 2, "field": "nope"}))
    with pytest.raises(CorpusError, match="suffix"):
        bad = tmp_path / "corpus.txt"
        bad.write_text("hi")
        cp.resolve_corpus_ref(CorpusRef(path=str(bad), slice=ok))
    # hub mode requires slice.file (no network touched before the refusal)
    with pytest.raises(CorpusError, match="slice.file"):
        cp.resolve_corpus_ref(CorpusRef(repo_id="org/d", revision=SHA,
                                        slice=ok))
    # duplicates inside the prefix are a pairing hazard -> refusal
    dup = tmp_path / "dup.jsonl"
    dup.write_text('{"prompt": "same"}\n{"prompt": "same"}\n')
    with pytest.raises(CorpusError, match="duplicates"):
        cp.resolve_corpus_ref(CorpusRef(path=str(dup),
                                        slice={"first": 2, "field": "prompt"}))


# --------------------------------------------------------------------------
# the pinned named corpora (hub reads monkeypatched to local files)
# --------------------------------------------------------------------------


@pytest.fixture()
def fake_hub(tmp_path, monkeypatch):
    """Local snapshot stand-ins for every pinned repo + a
    snapshot_download that verifies the full-SHA revision and the
    allow_patterns corpora asked for."""
    snaps = {}

    def add(repo, fname, df):
        d = tmp_path / repo.replace("/", "__")
        d.mkdir(exist_ok=True)
        df.to_parquet(d / fname)
        snaps[repo] = d

    add("org/wikitext", "wt.parquet",
        pd.DataFrame({"text": [f"line {i} of the article. " for i in range(50)]}))
    add("org/stack", "py.parquet",
        pd.DataFrame({"content": [f"def f{i}():\n    return {i}\n" * 3
                                  for i in range(30)]}))
    add("org/ultrachat", "uc.parquet", pd.DataFrame({"messages": [
        [{"role": "user", "content": f"q{i}"},
         {"role": "assistant", "content": f"a{i}"}] for i in range(20)]}))
    add("org/alpaca", "al.parquet", pd.DataFrame({
        "instruction": [f"Explain topic number {i} in plain words."
                        for i in range(40)],
        "input": ["" if i % 4 else "some context" for i in range(40)],
    }))
    canary_dir = tmp_path / "org__alpaca_eval"
    canary_dir.mkdir()
    (canary_dir / "alpaca_eval.json").write_text(json.dumps(
        [{"instruction": f"Canary prompt {i}?"} for i in range(30)]))
    # snapshot noise that the suffix/pattern filter must skip
    (canary_dir / "README.md").write_text("noise")
    snaps["org/alpaca_eval"] = canary_dir

    def fake_snapshot(repo_id, repo_type="dataset", revision=None,
                      allow_patterns=None, local_files_only=False):
        assert revision == SHA, "the pinned revision must reach the hub call"
        assert repo_type == "dataset"
        # r1 stats-core #11: evaluation-time loaders are CACHE-ONLY —
        # every hub call must carry local_files_only=True so an unfetched
        # corpus refuses instead of reaching for the network
        assert local_files_only is True
        return str(snaps[repo_id])

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)

    def entry(repo, patterns, fmt):
        return {"repo_id": repo, "revision": SHA, "patterns": patterns,
                "verify": fmt, "bytes": 1, "files": 1}

    return {"datasets": {
        "wikitext103": entry("org/wikitext", ["wt.parquet"], "parquet"),
        "stack_py": entry("org/stack", None, "parquet"),
        "ultrachat": entry("org/ultrachat", ["uc.*"], "parquet"),
        "alpaca": entry("org/alpaca", None, "parquet"),
        "canary_prompts": entry("org/alpaca_eval", ["alpaca_eval.json"],
                                "json"),
    }}


def test_wikitext_contiguous_prefix(fake_hub):
    s = cp.wikitext103(fake_hub, char_budget=100)
    assert s.kind == "text" and len(s) == 1  # one contiguous stream
    assert s.items[0].startswith("line 0 of the article.")
    assert len(s.items[0]) >= 100
    assert s.spec["mode"] == "contiguous_prefix"
    # bigger budget -> strictly more text, same prefix
    s2 = cp.wikitext103(fake_hub, char_budget=300)
    assert s2.items[0].startswith(s.items[0])
    assert s2.slice_hash != s.slice_hash


def test_stack_py_budgeted_shuffle(fake_hub):
    s = cp.stack_py(fake_hub, char_budget=200)
    assert s.kind == "text" and len(s) >= 2
    assert sum(len(t) for t in s.items) >= 200
    assert cp.stack_py(fake_hub, char_budget=200).slice_hash == s.slice_hash


def test_ultrachat_conversations(fake_hub):
    s = cp.ultrachat(fake_hub, max_conversations=5)
    assert s.kind == "chat" and len(s) == 5
    for conv in s.items:
        assert any(m["role"] == "assistant" for m in conv)
    assert cp.ultrachat(fake_hub, max_conversations=5).item_ids == s.item_ids


def test_canary_prompts_deterministic_subset(fake_hub):
    s = cp.canary_prompts(fake_hub, n=10)
    assert s.kind == "prompts" and len(s) == 10
    assert cp.canary_prompts(fake_hub, n=10).items == s.items
    with pytest.raises(CorpusError, match="need"):
        cp.canary_prompts(fake_hub, n=1000)


def test_unfetched_corpus_refuses_offline(monkeypatch, fake_hub):
    """r1 stats-core #11: a corpus missing from the local cache is a
    LOUD CorpusError at load — never a network attempt (which, on a
    Slurm node without HF_HUB_OFFLINE, would hang the GPU allocation)."""
    def cache_miss(*a, **kw):
        assert kw.get("local_files_only") is True
        raise FileNotFoundError("not in cache")

    monkeypatch.setattr("huggingface_hub.snapshot_download", cache_miss)
    with pytest.raises(CorpusError, match="capeval_fetch_data"):
        cp.wikitext103(fake_hub, char_budget=100)


def test_resolve_corpus_ref_hub_mode_is_cache_only(monkeypatch, tmp_path):
    """Hub-mode CorpusRef resolution also carries local_files_only=True
    (r1 #11), and a cache miss is a loud refusal."""
    src = FIXTURES / "pairs_synthetic.jsonl"
    calls = {}

    def fake_download(repo_id, filename, repo_type=None, revision=None,
                      local_files_only=False):
        calls.update(repo_id=repo_id, filename=filename,
                     revision=revision, local_files_only=local_files_only)
        return str(src)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    ref = CorpusRef(repo_id="org/d", revision=SHA,
                    slice={"first": 2, "field": "prompt",
                           "file": "pairs.jsonl"})
    s = cp.resolve_corpus_ref(ref)
    assert len(s) == 2
    assert calls["local_files_only"] is True and calls["revision"] == SHA

    def cache_miss(*a, **kw):
        raise FileNotFoundError("not in cache")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", cache_miss)
    with pytest.raises(CorpusError, match="capeval_fetch_data"):
        cp.resolve_corpus_ref(ref)


def test_resolve_corpus_ref_refuses_non_string_values(tmp_path):
    """r1 stats-core #12: pandas nulls/containers/non-string scalars in
    the selected field would be str()-coerced into fabricated prompts
    like "nan" or "['a', 'b']" — refuse instead, naming the row."""
    for bad in ("null", "3", '["a", "b"]', '{"x": 1}'):
        p = tmp_path / "bad.jsonl"
        p.write_text('{"prompt": "fine one"}\n'
                     f'{{"prompt": {bad}}}\n'
                     '{"prompt": "fine two"}\n')
        ref = CorpusRef(path=str(p), slice={"first": 2, "field": "prompt"})
        with pytest.raises(CorpusError, match="not a string"):
            cp.resolve_corpus_ref(ref)
    # a malformed row BEYOND the slice never blocks the selected prefix
    p = tmp_path / "tail_bad.jsonl"
    p.write_text('{"prompt": "fine one"}\n'
                 '{"prompt": "fine two"}\n'
                 '{"prompt": null}\n')
    s = cp.resolve_corpus_ref(CorpusRef(path=str(p),
                                        slice={"first": 2, "field": "prompt"}))
    assert s.items == ("fine one", "fine two")


def test_alpaca_holdout_prompts(fake_hub):
    s = cp.alpaca_holdout_prompts(fake_hub, n=5, pool_n=20, holdout_frac=0.5)
    assert len(s) == 5
    # the S2 filter held: no prompt came from an input-bearing row
    assert all("topic number" in p for p in s.items)
    # rows with input were excluded: input is "" only when i%4 != 0,
    # so every kept instruction has i%4 != 0
    kept_ids = {int(p.split()[3]) for p in s.items}
    assert all(i % 4 != 0 for i in kept_ids)
    # deterministic + disjoint from the fit side by the hash split
    again = cp.alpaca_holdout_prompts(fake_hub, n=5, pool_n=20,
                                      holdout_frac=0.5)
    assert again.items == s.items
    from sisteer.data.prompts import split_prompts
    fit, heldout = split_prompts(list(s.items), holdout_frac=0.5)
    assert set(s.items) <= set(heldout)  # all chosen prompts hash heldout


# --------------------------------------------------------------------------
# targets: registry + mode contract
# --------------------------------------------------------------------------


def test_plugin_registry():
    fn = tg.get_target_plugin("sisteer_compliance")
    assert fn is tg.sisteer_compliance
    with pytest.raises(TargetError, match="sisteer_compliance"):
        tg.get_target_plugin("mystery_plugin")
    with pytest.raises(TargetError, match="twice"):
        tg.register_target("sisteer_compliance")(lambda: None)


def _manifest(mode, target):
    return Manifest(
        sweep="s", mode=mode, base="m", base_revision="0" * 40, engine="hf",
        tiers=[0, 1], confirmatory=False, target=target,
        ablations=[AblationSpec(name="none", kind="none")])


def _target(plugin="sisteer_compliance"):
    return TargetSpec(
        primaries=[PrimarySpec(metric="si_compliance", plugin=plugin,
                               params={"si": "concise"}, direction="up",
                               min_effect=0.1)],
        relevant_corpus=CorpusRef(path=str(FIXTURES / "pairs_synthetic.jsonl"),
                                  slice={"first": 2, "field": "prompt"}))


def test_resolve_target_mode_contract():
    [(primary, fn)] = tg.resolve_target(_manifest("sweep", _target()))
    assert fn is tg.sisteer_compliance and primary.metric == "si_compliance"
    with pytest.raises(TargetError, match="REQUIRES"):
        tg.resolve_target(_manifest("sweep", None))
    for mode in ("control", "calibration"):
        assert tg.resolve_target(_manifest(mode, None)) == []
        with pytest.raises(TargetError, match="FORBID"):
            tg.resolve_target(_manifest(mode, _target()))
    # unknown plugin refuses before any GPU time
    with pytest.raises(TargetError, match="unknown target plugin"):
        tg.resolve_target(_manifest("sweep", _target(plugin="nope")))


def test_relevant_slice_wrapper():
    s = tg.relevant_slice(_target())
    assert s.kind == "prompts" and len(s) == 2


# --------------------------------------------------------------------------
# sisteer_compliance with a fake engine (scored by the REAL checkers)
# --------------------------------------------------------------------------


SHORT = "A short compliant answer."
LONG = ("word " * 60).strip()  # violates concise (max 40 words)


class FakeEngine:
    name = "hf"

    def __init__(self, texts):
        self.texts = texts
        self.calls = []

    def generate(self, prompts, arm, gen_cfg):
        self.calls.append((list(prompts), arm, dict(gen_cfg)))
        return [GenOut(text=t, gen_token_logprobs=[-0.5], ended_with_eos=True)
                for t in self.texts[: len(prompts)]]


class MarkingArm(ArmHandle):
    def transform_prompt(self, s: str) -> str:
        return "<inj>" + s


def _prompts():
    return cp.resolve_corpus_ref(
        CorpusRef(path=str(FIXTURES / "pairs_synthetic.jsonl"),
                  slice={"first": 2, "field": "prompt"}))


def _primary(**params):
    return PrimarySpec(metric="si_compliance", plugin="sisteer_compliance",
                       params={"si": "concise", "n_prompts": 2,
                               "max_new_tokens": 16, **params},
                       direction="up", min_effect=0.1)


def test_sisteer_compliance_base_leg():
    eng = FakeEngine([SHORT, LONG])
    out = tg.sisteer_compliance(eng, None, _primary(), _prompts())
    assert out.arm_name == "none" and out.metric == "si_compliance"
    assert out.values == (1.0, 0.0)  # real max_words checker scored these
    assert len(out.item_ids) == 2
    assert out.params_resolved["checker"] == "max_words"
    assert out.params_resolved["instruction_id"] == "concise"
    # base leg passes prompts through untransformed, greedy chat config
    [(prompts, arm, cfg)] = eng.calls
    assert arm is None and not prompts[0].startswith("<inj>")
    assert cfg == {"max_new_tokens": 16, "greedy": True, "chat": True}


def test_sisteer_compliance_arm_leg_transforms_prompts():
    eng = FakeEngine([SHORT, SHORT])
    arm = MarkingArm(name="arm-x", kind="token", coeff=1.0,
                     intervention_hash="ab" * 8)
    out = tg.sisteer_compliance(eng, arm, _primary(), _prompts())
    assert out.arm_name == "arm-x" and out.values == (1.0, 1.0)
    [(prompts, _, _)] = eng.calls
    assert all(p.startswith("<inj>") for p in prompts)  # token kinds §5.1


def test_sisteer_compliance_precomputed_texts_skip_engine():
    class Exploding:
        def generate(self, *a, **kw):  # pragma: no cover
            raise AssertionError("engine must not be called")

    out = tg.sisteer_compliance(Exploding(), None, _primary(), _prompts(),
                                gen_texts=[SHORT, LONG])
    assert out.values == (1.0, 0.0)


def test_default_gen_cfg_speaks_the_engine_vocabulary():
    """simplify-bug #2 contract: the engine's gen_cfg vocabulary is
    {greedy, chat, until, batch_size, max_new_tokens} — targets.py keeps
    emitting keys from exactly that set, with the defined types."""
    allowed = {"greedy", "chat", "until", "batch_size", "max_new_tokens"}
    cfg = tg.default_gen_cfg(64)
    assert set(cfg) <= allowed
    assert cfg["max_new_tokens"] == 64 and isinstance(cfg["max_new_tokens"], int)
    assert cfg["greedy"] is True and cfg["chat"] is True


def test_sisteer_compliance_refusals():
    eng = FakeEngine([SHORT, SHORT])
    with pytest.raises(TargetError, match="unknown param"):
        tg.sisteer_compliance(eng, None, _primary(surprise=1), _prompts())
    with pytest.raises(KeyError, match="unknown instruction"):
        tg.sisteer_compliance(eng, None, _primary(si="not_an_si"), _prompts())
    with pytest.raises(TargetError, match="ask for"):
        tg.sisteer_compliance(eng, None, _primary(n_prompts=99), _prompts())
    with pytest.raises(TargetError, match="generations"):
        tg.sisteer_compliance(eng, None, _primary(), _prompts(),
                              gen_texts=[SHORT])
    no_si = PrimarySpec(metric="m", plugin="sisteer_compliance", params={},
                        direction="up", min_effect=0.1)
    with pytest.raises(TargetError, match="params.si"):
        tg.sisteer_compliance(eng, None, no_si, _prompts())

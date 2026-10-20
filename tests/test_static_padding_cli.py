"""The static-padding CLI surface (docs/static_padding.md): ``jailbreak_run.py
generate --static-padding-*`` + the ``attacks.static_padding`` config block,
threaded into ``_run_replayable_family`` exactly the way ``sampled_decode`` is
— absent means byte-identical, present means a ``PadPlan`` reaches
``run_static`` — plus the window preflight, the scope refusal, the filler
resolution refusals, the manifest provenance, and the rung driver's refusal
of a padded base. Drives ``cmd_generate`` through the sampled static lane's
own stub transport (``tests/test_sampled_static.py``); no weights load: the
served path's tokenizer-only load is monkeypatched to a word tokenizer."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))

import jailbreak_run  # noqa: E402
import test_sampled_static as tss  # noqa: E402  (the lane's builders)
from sisteer.eval.attacks import served_target as st  # noqa: E402
from sisteer.eval.attacks import static_padding as sp  # noqa: E402

FOUR = sorted(jailbreak_run._SAMPLED_STATIC_FAMILIES)
N_WORDS = 6000


class _WordTok:
    """A whitespace tokenizer with an int vocabulary: ``tok(text)["input_ids"]``
    and ``tok.decode(ids)`` round-trip word for word, so a pad of N tokens is
    exactly N filler words and the plan converges in one step."""

    def __init__(self):
        self.vocab: dict = {}
        self.inv: list = []

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for w in text.split():
            if w not in self.vocab:
                self.vocab[w] = len(self.inv)
                self.inv.append(w)
            ids.append(self.vocab[w])
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.inv[i] for i in ids)


def _filler(tmp_path, n=N_WORDS):
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "pg19_filler.json"
    p.write_text(json.dumps({"text": " ".join(f"w{i}" for i in range(n))}))
    return p


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _args(tmp_path, **over):
    """``_served_args`` + every padding flag at its argparse default (None); a
    window wide enough for the small pads these tests use."""
    base = dict(static_padding_tokens=None, static_padding_placement=None,
                static_padding_mode=None, static_padding_filler=None,
                static_padding_tolerance=None, static_padding_reserve=None,
                target_max_model_len=16384)
    base.update(over)
    return tss._served_args(tmp_path, **base)


def _cfg(**over):
    cfg = tss._cfg(private_run="jb_pad")
    del cfg["sampled_decode"]          # the greedy opt-in world (no vendor row for org/model)
    cfg.update(over)
    return cfg


@pytest.fixture(autouse=True)
def _fresh_filler_ids_cache():
    """static_padding's process-wide filler-ids cache is keyed on id(tok); this module hands out throwaway tokenizers, so the shared
    state is reset per test (the value now holds the tokenizer too — the pin below — but a test never inherits another's entries)."""
    sp._filler_ids_cache.clear()
    yield
    sp._filler_ids_cache.clear()


@pytest.fixture
def tok_loader(monkeypatch):
    """The served path's tokenizer-only load, recorded; never a model."""
    import sisteer.models as models_mod

    calls = []

    def fake(name, revision=None, **kw):
        calls.append((name, revision))
        return _WordTok()
    monkeypatch.setattr(models_mod, "load_tokenizer", fake)
    return calls


@pytest.fixture
def no_tok_loader(monkeypatch):
    import sisteer.models as models_mod

    def boom(*a, **k):
        raise AssertionError("load_tokenizer must not run when padding is off")
    monkeypatch.setattr(models_mod, "load_tokenizer", boom)


def _isolate_search(monkeypatch, tmp_path):
    """Make ``resolve_filler_path``'s default search deterministic: no env
    override, an empty repo root, a synthetic user."""
    monkeypatch.delenv(sp.FILLER_ENV, raising=False)
    monkeypatch.setenv("USER", "synthetic_user")
    monkeypatch.setattr(jailbreak_run, "REPO", tmp_path / "repo")
    return tmp_path / "repo" / sp.REPO_FILLER_REL


def _wire_recorder(monkeypatch, tmp_path, goals=2):
    """cmd_generate wired to the stub chat, with ``_run_replayable_family``
    replaced by a recorder so the call SHAPE is what is asserted."""
    chat = tss._StubChat()
    tss._wire_cmd_generate(monkeypatch, chat, tss._goals(goals))
    rec, fake = tss._recorder()

    def fake_family(attack, defense, goals_arg, cfg, out_dir, **kwargs):
        return fake(defense, goals_arg, attack=attack, out_dir=out_dir, **kwargs)
    monkeypatch.setattr(jailbreak_run, "_run_replayable_family", fake_family)
    return rec, chat


# --------------------------------------------------------------------------- #
# 1. off = byte-identical; on = a PadPlan whose block matches the flags         #
# --------------------------------------------------------------------------- #

def test_flags_off_calls_the_family_with_no_padding_kwarg(monkeypatch, tmp_path, no_tok_loader):
    rec, _ = _wire_recorder(monkeypatch, tmp_path)
    jailbreak_run.cmd_generate(_args(tmp_path), _cfg())
    assert rec["attack"] == "pasttense" and "padding" not in rec
    # a config block whose tokens are 0 is the same as none (the pinned contract
    # a sweep's --static-padding-tokens switches on)
    rec.clear()
    jailbreak_run.cmd_generate(_args(tmp_path), _cfg(attacks={
        "static_max_new_tokens": 4096, "target_max_new_tokens": 512,
        "static_padding": {"tokens": 0, "placement": "after"}}))
    assert "padding" not in rec


def test_flags_on_thread_a_padplan_whose_block_matches(monkeypatch, tmp_path, tok_loader, capsys):
    filler = _filler(tmp_path)
    rec, _ = _wire_recorder(monkeypatch, tmp_path)
    jailbreak_run.cmd_generate(
        _args(tmp_path, static_padding_tokens=64, static_padding_placement="after",
              static_padding_mode="per_goal", static_padding_filler=str(filler),
              static_padding_tolerance=0.05, static_padding_reserve=100), _cfg())
    plan = rec["padding"]
    assert isinstance(plan, sp.PadPlan)
    assert plan.block == {"schema": sp.SCHEMA, "tokens": 64, "placement": "after",
                          "mode": "per_goal", "source": "pg19", "filler_sha256": _sha(filler),
                          "separator": sp.SEPARATOR, "tolerance": 0.05}
    assert plan.spec.filler_path == str(filler)
    # the decode kwargs are untouched by the knob (greedy world: none)
    assert "decode" not in rec and "seed_base" not in rec
    # served path: ONE tokenizer-only load at the config's pins, no weights
    assert tok_loader == [("org/model", "r1")]
    out = capsys.readouterr().out
    assert "STATIC PADDING: tokens=64 (cli) placement=after (cli) mode=per_goal (cli)" in out
    assert f"filler_sha={_sha(filler)[:12]}" in out and f"path={filler} (cli)" in out


def test_config_block_alone_works_and_a_typed_flag_wins(monkeypatch, tmp_path, tok_loader, capsys):
    filler = _filler(tmp_path)
    block = {"tokens": 32, "placement": "after", "mode": "fixed",
             "filler_cache": str(filler), "tolerance": 0.03, "reserve": 512}
    cfg = _cfg(attacks={"static_max_new_tokens": 4096, "target_max_new_tokens": 512,
                        "static_padding": block})
    rec, _ = _wire_recorder(monkeypatch, tmp_path)
    jailbreak_run.cmd_generate(_args(tmp_path), cfg)
    plan = rec["padding"]
    assert plan.block["tokens"] == 32 and plan.block["placement"] == "after"
    assert plan.block["tolerance"] == 0.03 and plan.block["filler_sha256"] == _sha(filler)
    out = capsys.readouterr().out
    assert "tokens=32 (config) placement=after (config)" in out and "reserve=512 (config)" in out
    assert f"path={filler} (config)" in out
    # the CLI overrides tokens + placement; the config's filler and tolerance stand
    rec.clear()
    jailbreak_run.cmd_generate(_args(tmp_path, static_padding_tokens=48,
                                     static_padding_placement="before"), cfg)
    plan = rec["padding"]
    assert plan.block["tokens"] == 48 and plan.block["placement"] == "before"
    assert plan.block["tolerance"] == 0.03 and plan.block["filler_sha256"] == _sha(filler)
    assert "tokens=48 (cli) placement=before (cli) mode=fixed (config)" in capsys.readouterr().out
    # an explicit 0 switches the block OFF for this run
    rec.clear()
    jailbreak_run.cmd_generate(_args(tmp_path, static_padding_tokens=0), cfg)
    assert "padding" not in rec


@pytest.mark.parametrize("block, match", [
    ({"tokens": 8, "bogus": 1}, "outside the allowed set"),
    ({"tokens": "8"}, "must be an int"),
    ({"tokens": 8, "placement": "middle"}, "placement must be one of"),
    ({"tokens": 8, "mode": "random"}, "mode must be one of"),
    ({"tokens": 8, "tolerance": "loose"}, "tolerance must be a number"),
    ({"tokens": 8, "tolerance": 1.5}, "tolerance must be in"),
    ({"tokens": 8, "reserve": -1}, "reserve must be a non-negative int"),
    ({"tokens": -8}, "positive int"),
])
def test_malformed_config_block_refuses_before_anything_loads(monkeypatch, tmp_path, block, match):
    filler = _filler(tmp_path)
    block = {"filler_cache": str(filler), **block}
    monkeypatch.setattr(st, "build_served_target",
                        lambda **kw: pytest.fail("the target must not be built"))
    import sisteer.models as models_mod
    monkeypatch.setattr(models_mod, "load_tokenizer",
                        lambda *a, **k: pytest.fail("no tokenizer load on a refusal"))
    monkeypatch.setattr(jailbreak_run, "_valid_defense_ids", lambda cfg: {"none"})
    monkeypatch.setattr(jailbreak_run, "_resolve_attacker_pool", lambda args, cfg: None)
    with pytest.raises(SystemExit, match=match) as ei:
        jailbreak_run.cmd_generate(_args(tmp_path), _cfg(attacks={
            "static_max_new_tokens": 4096, "static_padding": block}))
    assert str(ei.value).startswith("[pasttense]")
    with pytest.raises(SystemExit, match="must be a mapping"):
        jailbreak_run.cmd_generate(_args(tmp_path), _cfg(attacks={
            "static_max_new_tokens": 4096, "static_padding": [8]}))


def test_a_sub_flag_without_tokens_refuses_rather_than_being_ignored(monkeypatch, tmp_path):
    monkeypatch.setattr(st, "build_served_target",
                        lambda **kw: pytest.fail("the target must not be built"))
    monkeypatch.setattr(jailbreak_run, "_valid_defense_ids", lambda cfg: {"none"})
    monkeypatch.setattr(jailbreak_run, "_resolve_attacker_pool", lambda args, cfg: None)
    with pytest.raises(SystemExit, match=r"--static-padding-filler, --static-padding-placement given but static padding is off"):
        jailbreak_run.cmd_generate(_args(tmp_path, static_padding_placement="after",
                                         static_padding_filler="/nowhere"), _cfg())


# --------------------------------------------------------------------------- #
# 2. scope: the four families only                                              #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("attack", ["pap", "pair", "bon", "gcg", "amplegcg_plus"])
def test_padding_on_a_non_static_attack_refuses_naming_the_four(monkeypatch, tmp_path, attack):
    filler = _filler(tmp_path)
    monkeypatch.setattr(st, "build_served_target",
                        lambda **kw: pytest.fail("the target must not be built"))
    monkeypatch.setattr(jailbreak_run, "_valid_defense_ids", lambda cfg: {"none"})
    monkeypatch.setattr(jailbreak_run, "_resolve_attacker_pool", lambda args, cfg: None)
    with pytest.raises(SystemExit) as ei:
        jailbreak_run.cmd_generate(_args(tmp_path, attack=attack, static_padding_tokens=8,
                                         static_padding_filler=str(filler)), _cfg())
    msg = str(ei.value)
    assert msg.startswith(f"[{attack}]") and "fixed scope" in msg
    assert all(f in msg for f in FOUR)
    assert not (tmp_path / "state").exists()


def test_padding_on_the_xstest_pass_refuses(monkeypatch, tmp_path):
    filler = _filler(tmp_path)
    monkeypatch.setattr(st, "build_served_target",
                        lambda **kw: pytest.fail("the target must not be built"))
    monkeypatch.setattr(jailbreak_run, "_valid_defense_ids", lambda cfg: {"none"})
    monkeypatch.setattr(jailbreak_run, "_resolve_attacker_pool", lambda args, cfg: None)
    with pytest.raises(SystemExit, match="XSTest"):
        jailbreak_run.cmd_generate(_args(tmp_path, xstest=True, static_padding_tokens=8,
                                         static_padding_filler=str(filler)), _cfg())


def test_run_replayable_family_threads_padding_only_when_present(monkeypatch, tmp_path):
    """The dispatcher's own shape, beside the decode pin in test_sampled_static:
    no kwarg when absent, the plan when present, the four-family scope."""
    import sisteer.eval.attacks as attacks_mod

    rec, fake = tss._recorder()
    monkeypatch.setattr(attacks_mod, "get_attack", lambda name: fake)
    cfg = {"attacks": {"static_max_new_tokens": 4096, "pasttense": {"seed": 5}}}
    jailbreak_run._run_replayable_family("pasttense", "DEF", tss._goals(2), cfg, "/tmp/o", trial=1)
    assert "padding" not in rec and "decode" not in rec
    text, sha = sp.load_filler(_filler(tmp_path))
    plan = sp.PadPlan(sp.PadSpec(tokens=16, filler_path="f", filler_sha256=sha), _WordTok(), text)
    rec.clear()
    jailbreak_run._run_replayable_family("pasttense", "DEF", tss._goals(2), cfg, "/tmp/o", trial=1,
                                         padding=plan)
    assert rec["padding"] is plan and rec["max_new_tokens"] == 4096 and rec["seed"] == 5
    # beside decode, both ride
    rec.clear()
    jailbreak_run._run_replayable_family("pasttense", "DEF", tss._goals(2), cfg, "/tmp/o", trial=1,
                                         decode=tss.DECODE, seed_base=tss.SEED_BASE, padding=plan)
    assert rec["padding"] is plan and rec["decode"] == tss.DECODE
    for other in ("pap", "amplegcg_plus"):
        with pytest.raises(SystemExit, match="fixed scope") as ei:
            jailbreak_run._run_replayable_family(other, "DEF", tss._goals(2), cfg, "/tmp/o",
                                                 padding=plan)
        assert all(f in str(ei.value) for f in FOUR)


# --------------------------------------------------------------------------- #
# 3. the filler: a missing one refuses naming the search                        #
# --------------------------------------------------------------------------- #

def test_missing_filler_refuses_naming_the_searched_paths(monkeypatch, tmp_path):
    repo_default = _isolate_search(monkeypatch, tmp_path)
    monkeypatch.setattr(st, "build_served_target",
                        lambda **kw: pytest.fail("the target must not be built"))
    monkeypatch.setattr(jailbreak_run, "_valid_defense_ids", lambda cfg: {"none"})
    monkeypatch.setattr(jailbreak_run, "_resolve_attacker_pool", lambda args, cfg: None)
    # an EXPLICIT path that is not a file: named, never a fallback
    missing = tmp_path / "not_here.json"
    with pytest.raises(SystemExit, match="not a file") as ei:
        jailbreak_run.cmd_generate(_args(tmp_path, static_padding_tokens=8,
                                         static_padding_filler=str(missing)), _cfg())
    assert str(missing) in str(ei.value) and str(ei.value).startswith("[pasttense]")
    # no path anywhere: every searched candidate is named, and the builder
    with pytest.raises(SystemExit, match="no PG-19 filler cache found; searched") as ei:
        jailbreak_run.cmd_generate(_args(tmp_path, static_padding_tokens=8), _cfg())
    msg = str(ei.value)
    assert str(repo_default) in msg and "synthetic_user" not in msg     # no per-user cluster path is guessed
    assert "--only pg19-filler" in msg
    # a config's filler_cache that is absent is the first candidate named
    with pytest.raises(SystemExit, match="searched") as ei:
        jailbreak_run.cmd_generate(_args(tmp_path, static_padding_tokens=8), _cfg(attacks={
            "static_max_new_tokens": 4096,
            "static_padding": {"filler_cache": str(tmp_path / "cfg_absent.json")}}))
    assert str(tmp_path / "cfg_absent.json") in str(ei.value)
    assert not (tmp_path / "state").exists()


def test_filler_resolution_prefers_the_explicit_then_the_config(monkeypatch, tmp_path, tok_loader, capsys):
    _isolate_search(monkeypatch, tmp_path)
    cli_f = _filler(tmp_path / "cli")
    cfg_f = _filler(tmp_path / "cfg", n=N_WORDS + 1)
    rec, _ = _wire_recorder(monkeypatch, tmp_path)
    cfg = _cfg(attacks={"static_max_new_tokens": 4096,
                        "static_padding": {"filler_cache": str(cfg_f)}})
    jailbreak_run.cmd_generate(_args(tmp_path, static_padding_tokens=8,
                                     static_padding_filler=str(cli_f)), cfg)
    assert rec["padding"].block["filler_sha256"] == _sha(cli_f) != _sha(cfg_f)
    rec.clear()
    jailbreak_run.cmd_generate(_args(tmp_path, static_padding_tokens=8), cfg)
    assert rec["padding"].block["filler_sha256"] == _sha(cfg_f)
    # the env override is the next candidate
    monkeypatch.setenv(sp.FILLER_ENV, str(cli_f))
    rec.clear()
    jailbreak_run.cmd_generate(_args(tmp_path, static_padding_tokens=8), _cfg())
    assert rec["padding"].block["filler_sha256"] == _sha(cli_f)
    assert "(searched)" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 4. the window preflight                                                       #
# --------------------------------------------------------------------------- #

def test_window_preflight_refuses_naming_the_numbers_and_writes_no_store(monkeypatch, tmp_path):
    filler = _filler(tmp_path)
    built = []
    monkeypatch.setattr(st, "build_served_target", lambda **kw: built.append(kw))
    import sisteer.models as models_mod
    monkeypatch.setattr(models_mod, "load_tokenizer",
                        lambda *a, **k: pytest.fail("no tokenizer load on a refusal"))
    monkeypatch.setattr(jailbreak_run, "_valid_defense_ids", lambda cfg: {"none"})
    monkeypatch.setattr(jailbreak_run, "_resolve_attacker_pool", lambda args, cfg: None)
    # 2048 pad + 4096 cap + 2048 reserve = 8192 > the 4864 window
    with pytest.raises(SystemExit) as ei:
        jailbreak_run.cmd_generate(_args(tmp_path, static_padding_tokens=2048,
                                         static_padding_filler=str(filler),
                                         target_max_model_len=4864), _cfg())
    msg = str(ei.value)
    assert msg.startswith("[pasttense] static padding: padding 2048 tokens + completion cap 4096 + reserve 2048 = 8192")
    assert "exceeds the served window 4864" in msg and "--target-max-model-len" in msg
    assert built == [] and not (tmp_path / "state").exists()
    # a config's reserve moves the sum; the refusal names it
    with pytest.raises(SystemExit, match=r"reserve 256 = 6400 tokens exceeds the served window 4864"):
        jailbreak_run.cmd_generate(_args(tmp_path, static_padding_tokens=2048,
                                         static_padding_filler=str(filler),
                                         target_max_model_len=4864),
                                   _cfg(attacks={"static_max_new_tokens": 4096,
                                                 "static_padding": {"reserve": 256}}))
    assert built == [] and not (tmp_path / "state").exists()


def test_window_preflight_passes_when_it_fits_and_checks_nothing_without_a_window(monkeypatch, tmp_path, tok_loader):
    filler = _filler(tmp_path)
    rec, _ = _wire_recorder(monkeypatch, tmp_path)
    # 512 + 4096 + 0 = 4608 <= 4864
    jailbreak_run.cmd_generate(_args(tmp_path, static_padding_tokens=512, static_padding_reserve=0,
                                     static_padding_filler=str(filler),
                                     target_max_model_len=4864), _cfg())
    assert rec["padding"].block["tokens"] == 512
    # no launcher-resolved window: the server's own rejection governs, as today
    rec.clear()
    jailbreak_run.cmd_generate(_args(tmp_path, static_padding_tokens=4096,
                                     static_padding_filler=str(filler),
                                     target_max_model_len=None), _cfg())
    assert rec["padding"].block["tokens"] == 4096


# --------------------------------------------------------------------------- #
# 5. end to end: rows, progress and the manifest through the real run_static     #
# --------------------------------------------------------------------------- #

def test_end_to_end_rows_and_manifest_carry_the_padding(monkeypatch, tmp_path, tok_loader):
    filler = _filler(tmp_path)
    chat = tss._StubChat()
    tss._wire_cmd_generate(monkeypatch, chat, tss._goals(2))
    cfg = _cfg(private_run="jb_pad_e2e")
    jailbreak_run.cmd_generate(_args(tmp_path, static_padding_tokens=64,
                                     static_padding_filler=str(filler)), cfg)
    out = tmp_path / "state" / "jb_pad_e2e" / "pasttense" / "none" / "shard00"
    rows = tss._rows(out / "responses.jsonl")
    assert len(rows) == len(chat.payloads) > 0
    block = {"schema": sp.SCHEMA, "tokens": 64, "placement": "before", "mode": "fixed",
             "source": "pg19", "filler_sha256": _sha(filler), "separator": sp.SEPARATOR,
             "tolerance": sp.DEFAULT_TOLERANCE}
    for row, payload in zip(rows, chat.payloads):
        sent = payload["messages"][-1]["content"]
        pad = row["padding"]
        # the store-identity block on every row, the per-item record beside it
        assert sp.block_identity(pad) == block
        assert pad["pad_tokens_target"] == 64 and pad["pad_tokens_actual"] == 64
        assert pad["filler_offset_tokens"] == 0
        # attack_prompt is the UNPADDED transform; what was sent is padded and pinned
        assert sent != row["attack_prompt"] and sent.endswith(sp.SEPARATOR + row["attack_prompt"])
        assert len(sent.split()) == len(row["attack_prompt"].split()) + 64
        assert pad["padded_prompt_sha256"] == hashlib.sha256(sent.encode("utf-8")).hexdigest()
    assert all(r["padding_tokens"] == 64 for r in tss._rows(out / "progress.jsonl"))
    doc = json.loads((out / "served_target.json").read_text())
    # the padding block MERGED beside the decode resolution (the greedy opt-in record)
    assert doc["static_padding"] == block
    assert doc["decode_resolution"]["decode_default"] == "greedy_opt_in"
    assert doc["manifest_stage"] == "final"
    # the same store refuses an UNPADDED resume (one padding contract per store)
    with pytest.raises(Exception, match="PADDED row"):
        jailbreak_run.cmd_generate(_args(tmp_path), cfg)
    # ... and a differently padded one
    with pytest.raises(Exception, match="two padding contracts"):
        jailbreak_run.cmd_generate(_args(tmp_path, static_padding_tokens=32,
                                         static_padding_filler=str(filler)), cfg)


def test_unpadded_manifest_is_unchanged(monkeypatch, tmp_path, no_tok_loader):
    chat = tss._StubChat()
    tss._wire_cmd_generate(monkeypatch, chat, tss._goals(1))
    cfg = _cfg(private_run="jb_unpadded")
    jailbreak_run.cmd_generate(_args(tmp_path), cfg)
    out = tmp_path / "state" / "jb_unpadded" / "pasttense" / "none" / "shard00"
    doc = json.loads((out / "served_target.json").read_text())
    assert "static_padding" not in doc and "decode_resolution" in doc
    assert all("padding" not in r for r in tss._rows(out / "responses.jsonl"))
    assert all("padding_tokens" not in r for r in tss._rows(out / "progress.jsonl"))


# --------------------------------------------------------------------------- #
# 6. the rung driver refuses a padded base before any spend                     #
# --------------------------------------------------------------------------- #

def test_escalate_static_refuses_a_padded_base_before_any_spend(monkeypatch, tmp_path, capsys):
    es = tss._load_script("escalate_static")
    pad = {"schema": sp.SCHEMA, "tokens": 16384, "placement": "before", "mode": "fixed",
           "source": "pg19", "filler_sha256": "ab" * 32, "separator": sp.SEPARATOR,
           "tolerance": 0.02, "pad_tokens_target": 16384, "pad_tokens_actual": 16390,
           "filler_offset_tokens": 0, "padded_prompt_sha256": "cd" * 32}
    tss._write_store(tmp_path, tss.BASE_RUN, "pasttense", "none", [
        [tss._srow("g0", 0, "stop", padding=pad), tss._srow("g0", 1, "length", padding=pad)],
        [tss._srow("g2", 0, "stop", padding=pad)],
    ])
    chat = tss._StubChat(decide=lambda p: pytest.fail("no request may be issued"))
    _target, built = tss._wire_escalate(monkeypatch, es, chat)
    cfg_path = tss._esc_config(tmp_path)
    with pytest.raises(SystemExit) as ei:
        es.main(tss._esc_argv(tmp_path, cfg_path))
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "[escalate-static] REFUSED:" in err and "PADDED store" in err
    assert tss.BASE_RUN in err and "16384 padding tokens" in err and "3 of 3 row(s)" in err
    assert "does not support padded stores" in err and "regenerate" in err
    assert built == {} and chat.payloads == []                       # nothing spent
    assert not (tmp_path / tss.ESC_RUN).exists()                     # nothing written
    # a single padded row among unpadded ones is enough (a torn store still refuses)
    root2 = tmp_path / "second"
    tss._write_store(root2, tss.BASE_RUN, "pasttense", "none", [
        [tss._srow("g0", 0, "stop"), tss._srow("g0", 1, "length", padding=pad)],
    ])
    with pytest.raises(SystemExit):
        es.main(tss._esc_argv(root2, cfg_path))
    assert "1 of 2 row(s)" in capsys.readouterr().err


def test_escalate_static_guard_is_a_no_op_on_an_unpadded_base(monkeypatch, tmp_path):
    es = tss._load_script("escalate_static")
    rows = [tss._srow("g0", 0, "stop"), tss._srow("g0", 1, "length")]
    es.assert_base_unpadded(rows, base_cell_root=tmp_path, base_run=tss.BASE_RUN)   # returns
    with pytest.raises(SystemExit):
        es.assert_base_unpadded(rows + [tss._srow("g1", 0, "stop", padding={"tokens": 8})],
                                base_cell_root=tmp_path, base_run=tss.BASE_RUN)


def test_the_filler_ids_cache_never_hands_a_fresh_tokenizer_another_instances_ids():
    """static_padding._filler_token_ids keys its process-wide cache on id(tok): with nothing holding the tokenizer, an instance freed and
    another allocated at the same address (CPython reuses the slot) read the FIRST one's ids with a vocabulary that never produced them —
    _WordTok.decode's IndexError on the combined publication gate (shard 18, this module first). The value now holds the tokenizer and
    every hit is checked by identity, so a planted collision under a fresh tokenizer's id rebuilds its own ids."""
    text = " ".join(f"w{i}" for i in range(20))
    old = _WordTok()
    old_ids, _ = sp._filler_token_ids(old, text)                                    # tokenized once, cached under the first instance's id
    assert sp._filler_ids_cache[(id(old), hash(text))][0] is old
    fresh = _WordTok()                                                              # a vocabulary that never saw the text
    sp._filler_ids_cache[(id(fresh), hash(text))] = (old, old_ids, "stale")        # the collision: a stale entry under the fresh instance's id
    ids, source = sp._filler_token_ids(fresh, text)
    assert source != "stale" and ids == old_ids and fresh.decode(ids) == text      # rebuilt by the fresh tokenizer, hence decodable by it
    assert sp._filler_ids_cache[(id(fresh), hash(text))][0] is fresh
    with pytest.raises(IndexError):
        _WordTok().decode(old_ids)                                                  # what the stale hit used to hand the decoder

"""Stage 02 `--step pools` works from a FRESH CLONE: the three
pool builders run with the ORIGINAL corpus checkout absent — `CORPUS_GENERATOR_ROOT=<repo>/corpus_generator` (the in-repo generator code
root every builder process inherits through build_heldout_split's import-time root), the finals passed EXPLICITLY
(`--advice-final/--stories-final`, `--control-advice/--control-stories`, as stage 02 materialises them from corpora/),
and the staged reference tokenizers passed as `--tokenizers-dir` — while a missing default final refuses BY NAME and a
`--code-root` pointing nowhere refuses by name (never a fallback to a home-directory checkout)."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts/sdfbase"


def _toy(words):
    vocab = {w: i for i, w in enumerate(words)}
    vocab["[UNK]"] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return tok


def _stage_toy_tokenizers(d: Path) -> Path:
    man = {"tokenizers": {}}
    for name, words in (("llama3", ["a", "b", "c"]), ("gemma", ["a", "b"]), ("qwen", ["a"])):
        (d / name).mkdir(parents=True)
        _toy(words).save(str(d / name / "tokenizer.json"))
        man["tokenizers"][name] = {"source_repo": f"toy/{name}", "snapshot": "0" * 40, "pinned": True,
                                   "files": {"tokenizer.json": hashlib.sha256((d / name / "tokenizer.json").read_bytes()).hexdigest()}}
    (d / "MANIFEST.json").write_text(json.dumps(man))
    return d


def _synthetic_finals(d: Path, n_adv=200, n_sto=300):
    """Tiny finals with the fields the builders read (ids, texts, token_counts); NOT the paper's data."""
    d.mkdir(parents=True, exist_ok=True)
    adv = d / "advice.jsonl"
    sto = d / "stories.jsonl"
    with open(adv, "w") as f:
        for i in range(n_adv):
            p, t = i % 5, (i // 5) % 4
            pid, tid = f"p{p:02d}", f"p{p:02d}-t{t:02d}"
            rec = {"system": f"S {i}", "user": f"U {i} " + "a b c " * 5, "assistant": f"A {i} " + "b c " * 20,
                   "principle_id": pid, "prompt_type_id": tid, "scenario_id": f"{tid}-s{i:02d}",
                   "section_id": "s1", "lineage": {"run_id": "synthetic", "custom_id": f"adv-{tid}-s{i:02d}"},
                   "token_counts": {"llama3": 60 + i, "gemma": 70 + i, "qwen": 65 + i}}
            f.write(json.dumps(rec) + "\n")
    with open(sto, "w") as f:
        for i in range(n_sto):
            c, u = i % 4, (i // 4) % 3
            cid, uid = f"c{c:02d}", f"c{c:02d}-u{u:02d}"
            rec = {"text": f"Story {i}. " + "a b " * 50, "category_id": cid, "subcategory_id": uid,
                   "call_id": f"{uid}-k{i:03d}", "story_slot": 0, "target_words": 100,
                   "lineage": {"run_id": "synthetic", "custom_id": f"sto-{uid}-k{i:03d}"},
                   "token_counts": {"llama3": 100 + i, "gemma": 110 + i, "qwen": 105 + i}}
            f.write(json.dumps(rec) + "\n")
    return adv, sto


def _env(tmp_path, code_root=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CORPUS_GENERATOR") and k != "PYTHONPATH"}
    env["HOME"] = str(tmp_path / "home")                      # no ~/corpus_generator, no ~/.cache
    (tmp_path / "home").mkdir(exist_ok=True)
    env["CORPUS_GENERATOR_ROOT"] = str(code_root or REPO / "corpus_generator")
    env["PYTHONPATH"] = str(REPO / "src")
    return env


def _run(script, args, env, timeout=600):
    return subprocess.run([sys.executable, str(SCRIPTS / script), *args], env=env, capture_output=True, text=True, timeout=timeout)


def test_module_defaults_point_at_the_repo_code_root_not_a_home_checkout(monkeypatch, tmp_path):
    monkeypatch.delenv("CORPUS_GENERATOR_ROOT", raising=False)
    monkeypatch.delenv("CORPUS_GENERATOR_DATA_ROOT", raising=False)
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import build_heldout_split as hs
    hs = importlib.reload(hs)
    assert hs.GENERATOR_ROOT == REPO / "corpus_generator"
    assert hs.GENERATOR_DATA_ROOT == REPO / "corpus_generator" / "data"
    assert hs.ADVICE_FINAL == REPO / "corpus_generator/data/advice/final-production-20260831-v2.jsonl"
    assert hs.tokenizers_dir() == REPO / "corpus_generator/data/tokenizers"
    assert hs.tokenizers_dir(tmp_path / "t") == tmp_path / "t"
    monkeypatch.setenv("CORPUS_GENERATOR_TOKENIZERS_DIR", str(tmp_path / "envtok"))
    assert hs.tokenizers_dir() == tmp_path / "envtok"
    tokens = hs.generator_tokens()                                    # the in-repo renderers import
    assert Path(tokens.__file__).resolve() == (REPO / "corpus_generator/src/corpus_generator/tokens.py").resolve()
    with pytest.raises(hs.SplitError, match="code root not found"):
        hs.generator_tokens(tmp_path / "nowhere")
    with pytest.raises(hs.SplitError, match="--advice-final"):
        hs.require_final(tmp_path / "missing.jsonl", "--advice-final")


def test_heldout_split_and_pools_from_a_fresh_clone(tmp_path):
    adv, sto = _synthetic_finals(tmp_path / "finals")
    tok = _stage_toy_tokenizers(tmp_path / "tokenizers")
    env = _env(tmp_path)
    split = tmp_path / "split.json"
    r = _run("build_heldout_split.py", ["--advice-final", str(adv), "--stories-final", str(sto), "--out", str(split), "--frac", "0.5", "--allow-dirty"], env)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    s = json.loads(split.read_text())
    assert s["advice"]["ids"] and s["stories"]["ids"] and s["eval_subset"]
    prov = json.loads(split.with_suffix(".provenance.json").read_text())
    assert "corpus_generator_commit" in json.dumps(prov)
    pools = tmp_path / "pools"
    r = _run("build_pools.py", ["build", "--split", str(split), "--advice-final", str(adv), "--stories-final", str(sto),
                                "--tokenizers-dir", str(tok), "--out", str(pools), "--lanes", "heldout", "--allow-dirty"], env)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert list(pools.glob("pairs_*.jsonl")), sorted(p.name for p in pools.iterdir())
    man = json.loads(next(pools.glob("pairs_*.manifest.json")).read_text())
    assert "toy/llama3" in json.dumps(man), "the manifest names the staged tokenizers it counted with"


def test_missing_default_final_refuses_by_name_and_code_root_flag_refuses_nowhere(tmp_path):
    env = _env(tmp_path)
    r = _run("build_heldout_split.py", ["--out", str(tmp_path / "s.json"), "--allow-dirty"], env)
    assert r.returncode == 2 and "REFUSED" in r.stderr and "--advice-final" in r.stderr and "reassemble_corpus.py" in r.stderr
    assert "corpus_generator" not in r.stderr.replace("corpus_generator", ""), "no home-directory checkout is ever suggested"
    adv, sto = _synthetic_finals(tmp_path / "finals")
    r = _run("build_pools.py", ["build", "--split", str(tmp_path / "s.json"), "--out", str(tmp_path / "p"), "--allow-dirty"], env)
    assert r.returncode == 2 and "--advice-final" in r.stderr
    r0 = _run("build_heldout_split.py", ["--advice-final", str(adv), "--stories-final", str(sto), "--out", str(tmp_path / "s.json"), "--frac", "0.5", "--allow-dirty"], env)
    assert r0.returncode == 0, r0.stderr
    # the pool builder imports the generator's renderers: a --code-root that points nowhere refuses by name
    r = _run("build_pools.py", ["build", "--split", str(tmp_path / "s.json"), "--advice-final", str(adv), "--stories-final", str(sto),
                                "--code-root", str(tmp_path / "nowhere"), "--out", str(tmp_path / "p1"), "--lanes", "heldout", "--allow-dirty"], env)
    assert r.returncode == 2 and "code root not found" in r.stderr, r.stderr
    # and so does the process-wide root every builder inherits
    r = _run("build_pools.py", ["build", "--split", str(tmp_path / "s.json"), "--advice-final", str(adv), "--stories-final", str(sto),
                                "--out", str(tmp_path / "p1"), "--lanes", "heldout", "--allow-dirty"], _env(tmp_path, code_root=tmp_path / "elsewhere"))
    assert r.returncode == 2 and "code root not found" in r.stderr and "CORPUS_GENERATOR_ROOT" in r.stderr, r.stderr
    # an unstaged tokenizers dir refuses through the rig, naming stage-tokenizers
    r = _run("build_pools.py", ["build", "--split", str(tmp_path / "s.json"), "--advice-final", str(adv), "--stories-final", str(sto),
                                "--tokenizers-dir", str(tmp_path / "no-tokenizers"), "--out", str(tmp_path / "p2"), "--lanes", "heldout", "--allow-dirty"], env)
    assert r.returncode != 0 and "stage-tokenizers" in (r.stdout + r.stderr)


def test_ctl2_builder_accepts_tokenizers_dir_and_code_root(tmp_path):
    """The control builder threads the same two inputs (it constructs the rig inside build_ctl2, so a stubbed build sees
    them arrive); the real budget-matched build needs the trained-pool manifest and is exercised by test_build_ctl2_pools."""
    env = _env(tmp_path)
    code = (
        "import sys, json; sys.path.insert(0, sys.argv[1]); import build_ctl2_pools as b2, build_heldout_split as hs\n"
        "seen = {}\n"
        "def cap(out_dir, **kw): seen.update(kw); raise b2.bp.SamplerError('captured')\n"
        "b2.build_ctl2 = cap; hs.refuse_dirty_generator_tree = lambda *a, **k: {}\n"
        "rc = b2.main(['build', '--control-advice', sys.argv[2], '--control-stories', sys.argv[2], '--out', sys.argv[3], '--tokenizers-dir', sys.argv[4], '--code-root', sys.argv[5]])\n"
        "print(json.dumps({'rc': rc, 'tokenizers_dir': str(seen.get('tokenizers_dir')), 'root': str(hs.GENERATOR_ROOT)}))\n"
    )
    adv, _ = _synthetic_finals(tmp_path / "finals")
    r = subprocess.run([sys.executable, "-c", code, str(SCRIPTS), str(adv), str(tmp_path / "out"), str(tmp_path / "tok"), str(REPO / "corpus_generator")],
                       env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["rc"] == 2 and "captured" in r.stderr, "the stub's refusal surfaced through main()"
    assert got["tokenizers_dir"] == str(tmp_path / "tok") and got["root"] == str(REPO / "corpus_generator")

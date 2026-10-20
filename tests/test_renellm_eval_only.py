"""The paper's ReNeLLM corpus ships in the repository, replays from a fresh clone, and is evaluation-only.

``data/heldout_corpora/renellm.jsonl`` is third-party model output (x-ai/grok-4.5 through OpenRouter) shipped under
``LICENSES/ReNeLLM-corpus-terms.txt``, whose terms forbid training on it. These tests pin:

1. the shipped copy's identity, its terms file and its sidecar
2. replay from a fresh clone (no site copy, no private root) and from an existing site copy
3. every training-side reader's refusal: by path, file hash, content hash and prompt
4. the build's refusals (the CLI by name; the library never overwrites the paper's corpus)

The shipped corpus is read here for hashes, counts and goal ids only, and copied file to file; no assertion prints its
prompt text. Every other corpus is SYNTHETIC: placeholder goal ids and benign placeholder text.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from attacks_static_common import StubDefense  # noqa: E402
from sisteer.eval.attacks import eval_only  # noqa: E402
from sisteer.eval.attacks import frozen_corpus as fc  # noqa: E402
from sisteer.eval.attacks import renellm as rn  # noqa: E402
from sisteer.eval.attacks.frozen_corpus import corpus_content_hash  # noqa: E402
from sisteer.eval.jailbreak_data import Goal  # noqa: E402
from sisteer.sampling import sources  # noqa: E402

SHIPPED = REPO / rn.SHIPPED_CORPUS
PLACEHOLDER = "BENIGN_PLACEHOLDER_REQUEST"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shipped_goal() -> Goal:
    """A goal the shipped corpus covers, by its public JailbreakBench id (the text is never read)."""
    ids = sorted({json.loads(line)["goal_id"] for line in SHIPPED.read_text(encoding="utf-8").splitlines() if line.strip()})
    return Goal(next(i for i in ids if i.startswith("jbb_")), "unused", "c", "jailbreakbench", "r")


def _rows(tag: str, n_goals: int = 2) -> list[dict]:
    return [{"goal_id": f"g{i}", "sample_idx": j, "variant_name": name,
             "prompt": rn._nest(name, f"{PLACEHOLDER} {tag} {i}")}
            for i in range(n_goals) for j, name in enumerate(rn._SCENARIOS)]


def _write(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return corpus_content_hash(rows)


def _clear_caches():
    rn._CORPUS_CACHE.clear()
    rn._ANNOUNCED.clear()
    rn._TRAINING_CLEARED.clear()
    fc._ENV_CACHE.clear()
    fc._TRAINING_CLEARED.clear()
    eval_only._PROMPT_SHAS.clear()
    eval_only._ABSENT_ANNOUNCED.clear()


@pytest.fixture
def site(tmp_path, monkeypatch):
    """An EMPTY site corpora directory (what a fresh clone has), no acknowledgement, fresh caches."""
    cdir = tmp_path / "site_corpora"
    cdir.mkdir()
    monkeypatch.setenv("SISTEER_HELDOUT_CORPUS_DIR", str(cdir))
    monkeypatch.delenv(rn.NONPAPER_CORPUS_ENV, raising=False)
    _clear_caches()
    yield cdir
    _clear_caches()


# --------------------------------------------------------------------------- #
# 1. the shipped copy, its terms and its sidecar                               #
# --------------------------------------------------------------------------- #

def test_the_shipped_corpus_is_the_papers_byte_for_byte(site):
    record = json.loads((REPO / rn.PAPER_CORPUS_RECORD).read_text())
    assert SHIPPED.is_file() and _sha(SHIPPED) == rn.PINNED_FILE_SHA256 == record["file_sha256"]
    assert SHIPPED.stat().st_size == record["file_bytes"]
    ident = rn.corpus_identity(SHIPPED)
    assert ident["paper_corpus"] is True and ident["content_hash"] == rn.PINNED_SHA
    assert (ident["row_count"], ident["n_behaviors"]) == (rn.PAPER_CORPUS_ROWS, rn.PAPER_CORPUS_BEHAVIORS)
    assert record["shipped_path"] == rn.SHIPPED_CORPUS and record["terms_file"] == rn.CORPUS_TERMS


def test_the_terms_and_the_sidecar_name_the_generator_and_the_restriction():
    terms = (REPO / rn.CORPUS_TERMS).read_text()
    flat = " ".join(terms.split())
    for needle in ("NOT covered by the MIT licence", "x-ai/grok-4.5", "OpenRouter", "2026-08-21", "HarmBench",
                   "JailbreakBench", "Ding et al.", "arXiv:2311.08268", "machine-generated",
                   "for research and evaluation",
                   "to train, fine-tune or otherwise improve any AI system",
                   "to develop models or services that compete with xAI",
                   "you may not permit anyone else to use it",
                   "keep these terms and the attribution", rn.PINNED_FILE_SHA256, rn.SHIPPED_CORPUS, "777 rows"):
        assert needle in flat, needle
    sidecar = " ".join((REPO / (rn.SHIPPED_CORPUS + ".README")).read_text().split())
    for needle in (rn.CORPUS_TERMS, "x-ai/grok-4.5", "OpenRouter", "2026-08-21", rn.PINNED_FILE_SHA256, rn.PINNED_SHA,
                   "not the repository's MIT licence", "train, fine-tune or otherwise improve any AI system"):
        assert needle in sidecar, needle
    (corpus,) = eval_only.corpora()
    assert (corpus.path, corpus.content_hash, corpus.file_sha256, corpus.terms) == (
        rn.SHIPPED_CORPUS, rn.PINNED_SHA, rn.PINNED_FILE_SHA256, rn.CORPUS_TERMS)
    assert corpus.abspath == SHIPPED and eval_only.REPO == rn._REPO == REPO


def test_no_document_sends_anyone_to_the_renellm_build():
    """The documented build step crashed; it is gone, and every doc that names the corpus points at the shipped one."""
    docs = [REPO / "reproduce/README.md", REPO / "data/README.md", *(REPO / "docs").rglob("*.md"),
            *(REPO / "scripts").rglob("README.md")]
    offenders = [str(p.relative_to(REPO)) for p in docs if "build_heldout_corpus.py renellm" in p.read_text()]
    assert offenders == []
    readme = " ".join((REPO / "reproduce/README.md").read_text().split())
    assert rn.SHIPPED_CORPUS in readme and rn.CORPUS_TERMS in readme and rn.NONPAPER_CORPUS_ENV in readme


# --------------------------------------------------------------------------- #
# 2. replay: a fresh clone, an existing site copy, a non-paper corpus           #
# --------------------------------------------------------------------------- #

def test_a_fresh_clone_replays_the_shipped_corpus_as_the_papers(site, tmp_path, capsys):
    assert rn.replay_path() == SHIPPED and not rn._corpus_path().exists()
    ident = rn.admitted_identity()
    assert ident["paper_corpus"] is True and ident["acknowledged_by"] is None
    out = tmp_path / "store"
    res = rn.run_renellm(StubDefense(), [_shipped_goal()], out_dir=out)
    assert not res.get("skipped") and res["n_goals_run"] == 1
    assert res["renellm_corpus"]["paper_corpus"] is True
    assert json.loads((out / rn.STORE_CORPUS_RECORD).read_text())["content_hash"] == rn.PINNED_SHA
    assert "NON-PAPER" not in capsys.readouterr().err


def test_the_check_script_accepts_the_shipped_corpus_with_no_site_and_no_private_root(tmp_path):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "LANG": "C.UTF-8"}
    run = subprocess.run([sys.executable, "-I", "-S", str(REPO / "scripts/renellm_corpus_check.py")],
                         capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=120)
    assert run.returncode == 0, run.stderr
    assert run.stdout == (f"[renellm] corpus {rn.PINNED_SHA} ({rn.PAPER_CORPUS_ROWS} rows, "
                          f"{rn.PAPER_CORPUS_BEHAVIORS} behaviours): the paper's frozen corpus ({SHIPPED})\n")
    as_json = subprocess.run([sys.executable, "-I", "-S", str(REPO / "scripts/renellm_corpus_check.py"), "--json"],
                             capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=120)
    assert json.loads(as_json.stdout)["paper_corpus"] is True


def test_an_existing_site_copy_is_read_first(site):
    shutil.copyfile(SHIPPED, site / "renellm.jsonl")
    assert rn.replay_path() == site / "renellm.jsonl"
    assert rn.admitted_identity()["paper_corpus"] is True


def test_a_non_paper_site_copy_still_needs_the_opt_in_and_names_the_shipped_copy(site, monkeypatch):
    h = _write(site / "renellm.jsonl", _rows("a"))
    with pytest.raises(rn.RenellmCorpusRefused) as ei:
        rn.admitted_identity()
    assert h in str(ei.value) and rn.SHIPPED_CORPUS in str(ei.value) and PLACEHOLDER not in str(ei.value)
    rn._CORPUS_CACHE.clear()
    monkeypatch.setenv(rn.NONPAPER_CORPUS_ENV, h)
    ident = rn.admitted_identity()
    assert ident["paper_corpus"] is False and ident["acknowledged_by"] == rn.NONPAPER_CORPUS_ENV


# --------------------------------------------------------------------------- #
# 3a. training reads renellm through its own lookup, which refuses the corpus   #
# --------------------------------------------------------------------------- #

def test_training_resolves_its_own_lookups():
    import sisteer.eval.attacks.pasttense as pasttense

    assert sources._static_variants_fn("renellm") is rn.training_variants
    assert sources._static_variants_fn("frozen_corpus") is fc.training_variants
    assert sources._static_variants_fn("pasttense") is pasttense.pasttense_variants


def test_training_never_falls_back_to_the_shipped_copy(site):
    goal = _shipped_goal()
    assert len(rn.renellm_variants(goal)) == 3, "evaluation reads the shipped copy"
    assert rn.training_variants(goal) == [], "training reads the site's copy only"


def test_training_refuses_the_shipped_copy_by_path(site, monkeypatch, tmp_path):
    monkeypatch.setenv("SISTEER_HELDOUT_CORPUS_DIR", str(SHIPPED.parent))
    with pytest.raises(eval_only.EvalOnlyCorpusRefused) as ei:
        rn.training_variants(_shipped_goal())
    msg = str(ei.value)
    assert "is its shipped copy" in msg and rn.CORPUS_TERMS in msg and "train, fine-tune" in msg
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "renellm.jsonl").symlink_to(SHIPPED)
    monkeypatch.setenv("SISTEER_HELDOUT_CORPUS_DIR", str(linked))
    _clear_caches()
    with pytest.raises(eval_only.EvalOnlyCorpusRefused, match="is its shipped copy"):
        rn.training_variants(_shipped_goal())


def test_training_refuses_a_byte_copy_by_file_hash(site):
    shutil.copyfile(SHIPPED, site / "renellm.jsonl")
    with pytest.raises(eval_only.EvalOnlyCorpusRefused) as ei:
        rn.training_variants(_shipped_goal())
    assert f"file sha256 {rn.PINNED_FILE_SHA256}" in str(ei.value)


def test_training_refuses_a_reserialised_copy_by_content_hash(site):
    """Re-ordered and re-serialised, file to file: new bytes, the same rows."""
    lines = [ln for ln in SHIPPED.read_text(encoding="utf-8").splitlines() if ln.strip()]
    (site / "renellm.jsonl").write_text(
        "".join(json.dumps(json.loads(ln), sort_keys=True, indent=None, separators=(",", ":")) + "\n"
                for ln in reversed(lines)), encoding="utf-8")
    assert _sha(site / "renellm.jsonl") != rn.PINNED_FILE_SHA256
    with pytest.raises(eval_only.EvalOnlyCorpusRefused) as ei:
        rn.training_variants(_shipped_goal())
    assert f"content hash {rn.PINNED_SHA}" in str(ei.value)


def test_training_reads_a_non_paper_site_corpus(site):
    _write(site / "renellm.jsonl", _rows("a"))
    assert [n for n, _ in rn.training_variants(Goal("g0", "x", "c", "d", "r"))] == list(rn._SCENARIOS)


def test_the_training_sampler_refuses_the_papers_corpus_end_to_end(site):
    shutil.copyfile(SHIPPED, site / "renellm.jsonl")
    with pytest.raises(eval_only.EvalOnlyCorpusRefused, match="training-pool sampling"):
        sources.harmful_attack_prompts([_shipped_goal()], ["renellm"])


def test_the_generic_frozen_corpus_family_refuses_it_for_training(site, monkeypatch, tmp_path):
    monkeypatch.setenv("SISTEER_FROZEN_CORPUS_PATH", str(SHIPPED))
    monkeypatch.delenv("SISTEER_FROZEN_CORPUS_SHA", raising=False)
    monkeypatch.delenv("SISTEER_FROZEN_CORPUS_PROMPT_COL", raising=False)
    with pytest.raises(eval_only.EvalOnlyCorpusRefused, match=r"training-pool sampling \(frozen_corpus\)"):
        fc.training_variants(_shipped_goal())
    other = tmp_path / "other.jsonl"
    _write(other, _rows("b"))
    monkeypatch.setenv("SISTEER_FROZEN_CORPUS_PATH", str(other))
    assert len(fc.training_variants(Goal("g1", "x", "c", "d", "r"))) == 3


# --------------------------------------------------------------------------- #
# 3b. the prompt-level backstop, at every training-side reader                  #
# --------------------------------------------------------------------------- #

@pytest.fixture
def synthetic_eval_only(tmp_path, monkeypatch):
    """A SYNTHETIC evaluation-only corpus standing in for the shipped one, so prompts can be passed around freely."""
    rows = _rows("eval-only", n_goals=3)
    path = tmp_path / "eval_only.jsonl"
    content = _write(path, rows)
    record = eval_only.EvalOnlyCorpus(name="a synthetic eval-only corpus", path=str(path), content_hash=content,
                                      file_sha256=_sha(path), terms=rn.CORPUS_TERMS, generator="a stub rewriter")
    monkeypatch.setattr(eval_only, "corpora", lambda: (record,))
    _clear_caches()
    yield record, rows
    _clear_caches()


def test_refuse_texts_matches_exact_and_normalised_prompts(synthetic_eval_only):
    _, rows = synthetic_eval_only
    p = rows[0]["prompt"]
    assert eval_only.refuse_texts(["an unrelated prompt", None, 7, ""], consumer="t") == 1
    with pytest.raises(eval_only.EvalOnlyCorpusRefused) as ei:
        eval_only.refuse_texts(["an unrelated prompt", p], consumer="the consumer")
    assert "1 of the 2 prompts it was given are its rows" in str(ei.value) and PLACEHOLDER not in str(ei.value)
    with pytest.raises(eval_only.EvalOnlyCorpusRefused):
        eval_only.refuse_texts(["  " + p.upper().replace(" ", "   ") + "\n"], consumer="t")


def test_the_sampler_inventory_refuses_a_corpus_prompt_from_any_family(synthetic_eval_only, monkeypatch):
    _, rows = synthetic_eval_only
    monkeypatch.setattr(sources, "_static_variants_fn", lambda fam: (lambda g, seed: [("v0", rows[0]["prompt"])]))
    with pytest.raises(eval_only.EvalOnlyCorpusRefused, match=r"sisteer\.sampling\.sources"):
        sources.harmful_attack_prompts([Goal("g0", "x", "c", "d", "r")], ["pasttense"])


def test_the_cached_inventory_refuses_it(synthetic_eval_only, tmp_path):
    import pool_sample as ps

    _, rows = synthetic_eval_only
    cache = tmp_path / "inventory.json"
    args = SimpleNamespace(inventory=str(cache), families="pasttense", benign_families="", n_wildjailbreak_harmful=None,
                           n_benign=None, harmbench_splits="test", seed=0, bon_variants=8, subenv_bank=None,
                           no_control_subtraction=False, wildjailbreak_path=None)
    prompt = sources.SamplePrompt(family="pasttense", goal_id="g0", sample_idx=0, trial=0, variant_name="v0",
                                  text=rows[1]["prompt"], side="harmful", source="jbb")
    cache.write_text(json.dumps({"key": ps._inventory_key(args), "prompts": [asdict(prompt)]}))
    with pytest.raises(eval_only.EvalOnlyCorpusRefused, match="the cached inventory"):
        ps._inventory_cached(args)


def test_the_pool_builder_refuses_before_it_reads_the_spec(synthetic_eval_only):
    from sisteer.eval.matched_pool import GradedRow, build_matched_pool

    _, rows = synthetic_eval_only
    row = GradedRow(family="renellm", goal_id="g0", variant_key="0:0", condition="none", si_id=None,
                    prompt=rows[2]["prompt"], response="a response")
    with pytest.raises(eval_only.EvalOnlyCorpusRefused, match="build_matched_pool"):
        build_matched_pool([row], None)
    with pytest.raises(eval_only.EvalOnlyCorpusRefused):
        build_matched_pool([{"family": "renellm", "prompt": rows[2]["prompt"]}], None)


def test_the_leakage_fingerprint_refuses_a_training_store_that_carries_it(synthetic_eval_only, tmp_path):
    from sisteer.eval.attacks.heldout_leakage import load_training_fingerprint

    _, rows = synthetic_eval_only
    store = tmp_path / "store"
    store.mkdir()
    (store / "renellm.jsonl").write_text(json.dumps({"attack_prompt": "a clean training prompt"}) + "\n"
                                         + json.dumps({"attack_prompt": rows[3]["prompt"]}) + "\n")
    with pytest.raises(eval_only.EvalOnlyCorpusRefused, match="training fingerprint"):
        load_training_fingerprint(["renellm"], store_dir=str(store))
    (store / "renellm.jsonl").write_text(json.dumps({"attack_prompt": "a clean training prompt"}) + "\n")
    assert len(load_training_fingerprint(["renellm"], store_dir=str(store))) == 1


def test_the_training_stage_pair_loader_refuses_it(synthetic_eval_only, tmp_path):
    import train_two_si_objects as t2

    record, rows = synthetic_eval_only
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps({"prompt": "a clean prompt", "pos": "p", "neg": "n"}) + "\n"
                     + json.dumps({"prompt": rows[4]["prompt"], "pos": "p", "neg": "n"}) + "\n")
    with pytest.raises(eval_only.EvalOnlyCorpusRefused, match="pair loader"):
        t2._load_pairs(str(pairs))
    pairs.write_text(json.dumps({"prompt": "a clean prompt", "pos": "p", "neg": "n"}) + "\n")
    assert [p.prompt for p in t2._load_pairs(str(pairs))] == ["a clean prompt"]
    with pytest.raises(eval_only.EvalOnlyCorpusRefused, match="is its shipped copy"):
        t2._load_pairs(record.path)


def test_the_package_pairs_loader_refuses_it(synthetic_eval_only, tmp_path):
    """sisteer.data.pairs.load_pairs, which the pipeline's pair-trained methods (sisteer.cli) read, refuses too."""
    from sisteer.data.pairs import load_pairs

    record, rows = synthetic_eval_only
    clean = {"prompt": "a clean prompt", "pos": "p", "neg": "n", "instruction_id": "i"}
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps(clean) + "\n" + json.dumps(dict(clean, prompt=rows[5]["prompt"])) + "\n")
    with pytest.raises(eval_only.EvalOnlyCorpusRefused, match="the pairs loader") as ei:
        load_pairs(pairs)
    assert "1 of the 2 prompts it was given are its rows" in str(ei.value) and PLACEHOLDER not in str(ei.value)
    pairs.write_text(json.dumps(clean) + "\n")
    assert [p.prompt for p in load_pairs(pairs)] == ["a clean prompt"]
    with pytest.raises(eval_only.EvalOnlyCorpusRefused, match="is its shipped copy"):
        load_pairs(record.path)


def test_the_real_corpus_prompts_are_refused_in_a_pairs_file(site, tmp_path):
    """The REAL shipped corpus: five of its prompts carried into a pairs file (file to file) refuse at the training stage."""
    import train_two_si_objects as t2

    lines = [ln for ln in SHIPPED.read_text(encoding="utf-8").splitlines() if ln.strip()]
    prompts = [json.loads(ln)["prompt"] for ln in lines[:5]]
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("".join(json.dumps({"prompt": p, "pos": "p", "neg": "n"}) + "\n" for p in prompts), encoding="utf-8")
    with pytest.raises(eval_only.EvalOnlyCorpusRefused) as ei:
        t2._load_pairs(str(pairs))
    msg = str(ei.value)
    assert "5 of the 5 prompts it was given are its rows" in msg
    assert not any(p in msg for p in prompts), "a refusal names counts and hashes, never a prompt"
    exact, norm = eval_only.prompt_shas(eval_only.corpora()[0])
    assert len(exact) == len({json.loads(ln)["prompt"] for ln in lines}) and len(norm) >= 1


def test_an_altered_shipped_copy_refuses_loudly(tmp_path, monkeypatch):
    path = tmp_path / "altered.jsonl"
    _write(path, _rows("x"))
    record = eval_only.EvalOnlyCorpus(name="n", path=str(path), content_hash="c" * 64, file_sha256="0" * 64,
                                      terms=rn.CORPUS_TERMS, generator="g")
    monkeypatch.setattr(eval_only, "corpora", lambda: (record,))
    _clear_caches()
    with pytest.raises(eval_only.EvalOnlyCorpusAltered, match="was altered"):
        eval_only.refuse_texts(["anything"], consumer="t")
    _clear_caches()


def test_an_absent_shipped_copy_keeps_the_hash_checks_and_says_so(tmp_path, monkeypatch, capsys):
    record = eval_only.EvalOnlyCorpus(name="n", path=str(tmp_path / "absent.jsonl"), content_hash="c" * 64,
                                      file_sha256="f" * 64, terms=rn.CORPUS_TERMS, generator="g")
    monkeypatch.setattr(eval_only, "corpora", lambda: (record,))
    _clear_caches()
    assert eval_only.refuse_texts(["anything"], consumer="t") == 1
    assert eval_only.refuse_texts(["anything else"], consumer="t") == 1
    assert capsys.readouterr().err.count("prompt-level training check cannot run") == 1
    with pytest.raises(eval_only.EvalOnlyCorpusRefused, match="content hash"):
        eval_only.refuse_corpus(consumer="t", content_hash="c" * 64)
    _clear_caches()


# --------------------------------------------------------------------------- #
# 4. the build refuses                                                          #
# --------------------------------------------------------------------------- #

def test_the_build_cli_refuses_renellm_by_name():
    run = subprocess.run([sys.executable, str(REPO / "scripts/build_heldout_corpus.py"), "renellm",
                          "--config", "configs/two_si.yaml"],
                         capture_output=True, text=True, cwd=str(REPO), timeout=120)
    assert run.returncode == 2 and run.stdout == ""
    assert "renellm: refused" in run.stderr and rn.SHIPPED_CORPUS in run.stderr and rn.CORPUS_TERMS in run.stderr
    assert "Traceback" not in run.stderr


def test_the_build_never_overwrites_the_papers_corpus(site, tmp_path):
    before = _sha(SHIPPED)
    for out in (SHIPPED, site / "renellm.jsonl"):
        if out != SHIPPED:
            shutil.copyfile(SHIPPED, out)
        with pytest.raises(rn.RenellmCorpusRefused, match="would overwrite"):
            rn.build_renellm_corpus([Goal("g0", "x", "c", "d", "r")], rewriter=object(), fingerprint=frozenset(),
                                    out_path=out)
    assert _sha(SHIPPED) == before == _sha(site / "renellm.jsonl")
    (site / "renellm.jsonl").unlink()
    with pytest.raises(rn.RenellmCorpusRefused, match="would overwrite"):
        rn.build_renellm_corpus([Goal("g0", "x", "c", "d", "r")], rewriter=object(), fingerprint=frozenset(),
                                out_path=str(SHIPPED))
    assert _sha(SHIPPED) == before


# --------------------------------------------------------------------------- #
# 5. the run identity and the other job guards follow the replay path           #
# --------------------------------------------------------------------------- #

def test_the_run_identity_names_the_corpus_a_fresh_clone_replays(site, tmp_path):
    from sisteer.capeval.serving import run_identity as ri

    # a fresh clone: no site copy, so the identity carries the shipped corpus that replay reads
    assert ri.frozen_corpus_hashes(["bon", "renellm"]) == {"renellm": rn.PINNED_SHA}
    # an explicit root is read as given: no fallback to the shipped copy
    empty = tmp_path / "explicit_root"
    empty.mkdir()
    assert ri.frozen_corpus_hashes(["renellm"], empty) == {}
    # a site copy is read first, exactly as replay reads it
    h = _write(site / "renellm.jsonl", _rows("site"))
    assert ri.frozen_corpus_hashes(["renellm"]) == {"renellm": h} != {"renellm": rn.PINNED_SHA}


def _guard(rel: str, first: str, last: str) -> str:
    """The lines of ``rel`` from the one starting with ``first`` through the next one starting with ``last``."""
    lines = (REPO / rel).read_text().splitlines()
    i = next(n for n, ln in enumerate(lines) if ln.startswith(first))
    j = next(n for n in range(i, len(lines)) if lines[n].startswith(last))
    return "\n".join(lines[i:j + 1]) + "\n"


def _checkout(tmp_path: Path, name: str, *, shipped: bool) -> Path:
    """A synthetic per-commit checkout: it holds a non-empty placeholder at the shipped path, or nothing."""
    co = tmp_path / name
    (co / "data/heldout_corpora").mkdir(parents=True)
    if shipped:
        (co / rn.SHIPPED_CORPUS).write_text(json.dumps({"goal_id": "g0", "prompt": PLACEHOLDER}) + "\n")
    return co


@pytest.mark.parametrize("shipped,site_copy,ok", [(True, False, True), (False, True, True), (False, False, False)])
def test_the_qwen_wave_guard_accepts_the_shipped_copy(tmp_path, shipped, site_copy, ok):
    guard = _guard("scripts/sbatch/qwen_wave.sbatch", "if [ ! -s \"$SISTEER_HELDOUT_CORPUS_DIR/renellm.jsonl\" ]", "fi")
    co = _checkout(tmp_path, "co", shipped=shipped)
    cdir = tmp_path / "site_corpora"
    cdir.mkdir()
    if site_copy:
        (cdir / "renellm.jsonl").write_text(json.dumps({"goal_id": "g0", "prompt": PLACEHOLDER}) + "\n")
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "SISTEER_HELDOUT_CORPUS_DIR": str(cdir),
           "REPO_DIR": str(co)}
    run = subprocess.run(["bash", "-c", guard + "echo GUARD-PASSED"], capture_output=True, text=True, cwd=str(co),
                         env=env, timeout=60)
    assert (run.returncode == 0 and "GUARD-PASSED" in run.stdout) is ok, run.stderr
    if not ok:
        assert run.returncode == 1 and "data/heldout_corpora/renellm.jsonl" in run.stderr


@pytest.mark.parametrize("shipped,warns", [(True, False), (False, True)])
def test_the_ar_cell_warning_counts_the_shipped_copy(tmp_path, shipped, warns):
    guard = _guard("scripts/arditi/sbatch/ar_cell.sbatch", "[ -s \"$SISTEER_HELDOUT_CORPUS_DIR/renellm.jsonl\" ]",
                   "  || f22_log")
    co = _checkout(tmp_path, "co", shipped=shipped)
    cdir = tmp_path / "site_corpora"
    cdir.mkdir()
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "SISTEER_HELDOUT_CORPUS_DIR": str(cdir),
           "REPO_DIR": str(co)}
    run = subprocess.run(["bash", "-c", "f22_log() { echo \"LOG $*\"; }\n" + guard], capture_output=True, text=True,
                         cwd=str(tmp_path), env=env, timeout=60)
    assert ("will SKIP LOUDLY" in run.stdout) is warns, run.stdout

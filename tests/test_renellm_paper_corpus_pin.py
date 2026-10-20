"""ReNeLLM evaluation replay is pinned to the paper's frozen corpus.

The repository ships that corpus (``data/heldout_corpora/renellm.jsonl``, under its
own terms) with its hashes and counts (``configs/heldout/renellm.paper_corpus.json``
beside the build manifest). Replay reads the site's copy when there is one, else the
shipped copy; it refuses any other corpus unless SISTEER_RENELLM_NONPAPER_CORPUS_SHA
names it, labels such a run non-paper everywhere it records itself, and never mixes
two corpora in one store. Every corpus built here is SYNTHETIC: placeholder goal ids
and benign placeholder text, nothing from any benchmark (the shipped corpus itself is
read for its hashes and counts only, in tests/test_renellm_eval_only.py and the job
script guard below).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from attacks_static_common import StubDefense  # noqa: E402
from sisteer.eval.attacks import renellm as rn  # noqa: E402
from sisteer.eval.attacks.frozen_corpus import corpus_content_hash  # noqa: E402
from sisteer.eval.jailbreak_data import Goal  # noqa: E402

RECORD = REPO / rn.PAPER_CORPUS_RECORD
MANIFEST = REPO / rn.PAPER_CORPUS_MANIFEST
SBATCH = (
    "scripts/sdfbase/full22/target_static.sbatch",
    "scripts/sdfbase/full22/static_padding/target_static_padded.sbatch",
    "scripts/sdfbase/full22/static_padding/target_static_padded_obj.sbatch",
    "scripts/sdfbase/full22/sampled/target_sampled.sbatch",
)
PLACEHOLDER = "BENIGN_PLACEHOLDER_REQUEST"


def _rows(tag: str, n_goals: int = 2) -> list[dict]:
    return [{"goal_id": f"g{i}", "sample_idx": j, "variant_name": name,
             "prompt": rn._nest(name, f"{PLACEHOLDER} {tag} {i}")}
            for i in range(n_goals) for j, name in enumerate(rn._SCENARIOS)]


def _write_corpus(corpus_dir: Path, rows: list[dict]) -> str:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "renellm.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return corpus_content_hash(rows)


@pytest.fixture
def corpus_env(tmp_path, monkeypatch):
    """A synthetic corpus at the replay path, no acknowledgement, fresh caches."""
    cdir = tmp_path / "corpora"
    monkeypatch.setenv("SISTEER_HELDOUT_CORPUS_DIR", str(cdir))
    monkeypatch.delenv(rn.NONPAPER_CORPUS_ENV, raising=False)
    rn._CORPUS_CACHE.clear()
    rn._ANNOUNCED.clear()
    yield cdir
    rn._CORPUS_CACHE.clear()
    rn._ANNOUNCED.clear()


# --------------------------------------------------------------------------- #
# the pin and the shipped records agree                                        #
# --------------------------------------------------------------------------- #

def test_the_pin_agrees_with_the_shipped_records():
    record = json.loads(RECORD.read_text())
    manifest = json.loads(MANIFEST.read_text())
    assert re.fullmatch(r"[0-9a-f]{64}", rn.PINNED_SHA) and re.fullmatch(r"[0-9a-f]{64}", rn.PINNED_FILE_SHA256)
    assert manifest["corpus_content_hash"] == record["corpus_content_hash"] == rn.PINNED_SHA
    assert record["file_sha256"] == rn.PINNED_FILE_SHA256
    assert manifest["row_count"] == record["row_count"] == rn.PAPER_CORPUS_ROWS
    assert manifest["n_behaviors"] == record["n_behaviors"] == rn.PAPER_CORPUS_BEHAVIORS
    assert record["rows_per_behavior"] * record["n_behaviors"] == record["row_count"]
    assert tuple(record["rows_per_scenario"]) == rn._SCENARIOS
    assert sum(record["rows_per_scenario"].values()) == record["row_count"]
    assert record["goal_set"]["harmbench"] + record["goal_set"]["jailbreakbench"] == record["n_behaviors"]
    assert manifest["provenance"]["rewriter_model"] == record["rewriter_model"] == rn.DEFAULT_REWRITER_MODEL
    assert manifest["built_at"] == record["built_at"]
    assert record["build_manifest"] == rn.PAPER_CORPUS_MANIFEST and record["shipped_in_repository"] is True
    assert record["shipped_path"] == rn.SHIPPED_CORPUS and record["terms_file"] == rn.CORPUS_TERMS
    assert record["use"] == "evaluation_only" and record["rewriter_route"] == "openrouter"
    assert record["content_hash_function"] == "sisteer.eval.attacks.frozen_corpus.corpus_content_hash"


def test_the_shipped_records_hold_no_corpus_text():
    """Counts and hashes only: no string long enough to be a prompt beyond the record's own
    description, and no corpus row anywhere in the repository's configs."""
    for path in (RECORD, MANIFEST):
        def strings(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k != "description":
                        yield from strings(v)
            elif isinstance(o, list):
                for v in o:
                    yield from strings(v)
            elif isinstance(o, str):
                yield o
        assert all(len(s) <= 160 for s in strings(json.loads(path.read_text()))), path
    assert not list((REPO / "configs").rglob("renellm*.jsonl"))


def test_the_records_goal_set_is_the_papers_confirmatory_eval_set():
    """The corpus covers exactly the confirmatory eval goals two_si.yaml selects (public ids; the
    datasets are read offline from the local cache, so a host without it skips)."""
    import hashlib

    import yaml

    import jailbreak_run as jr
    record = json.loads(RECORD.read_text())
    cfg = yaml.safe_load((REPO / record["goal_set"]["config"]).read_text())
    old = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")}
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    try:
        goals = jr._eval_goals(cfg, confirmatory=True)
    except Exception as e:  # noqa: BLE001 -- an offline cache miss of any kind
        pytest.skip(f"the pinned HarmBench / JailbreakBench snapshots are not in this host's cache ({type(e).__name__})")
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    ids = sorted({g.goal_id for g in goals})
    assert len(ids) == record["n_behaviors"]
    assert sum(1 for i in ids if i.startswith("jbb_")) == record["goal_set"]["jailbreakbench"]
    assert hashlib.sha256("\n".join(ids).encode()).hexdigest() == record["goal_set"]["goal_id_set_sha256"]


# --------------------------------------------------------------------------- #
# evaluation replay: refuse, opt in by hash, label                             #
# --------------------------------------------------------------------------- #

def test_replay_refuses_a_corpus_that_is_not_the_papers(corpus_env, tmp_path, capsys):
    h = _write_corpus(corpus_env, _rows("a"))
    out = tmp_path / "store"
    with pytest.raises(rn.RenellmCorpusRefused) as ei:
        rn.run_renellm(StubDefense(), [Goal("g0", "x", "c", "d", "r")], out_dir=out)
    msg = str(ei.value)
    assert h in msg and rn.PINNED_SHA in msg and rn.PINNED_FILE_SHA256 in msg
    assert f"{rn.NONPAPER_CORPUS_ENV}={h}" in msg and "Nothing was generated" in msg
    assert rn.SHIPPED_CORPUS in msg, "a refused site copy names the shipped one replay would read without it"
    assert not out.exists(), "a refusal writes nothing"
    with pytest.raises(rn.RenellmCorpusRefused):
        rn.preflight_store(out)
    # the plain evaluation lookup does not refuse (training reads renellm.training_variants instead)
    assert [n for n, _ in rn.renellm_variants(Goal("g0", "x", "c", "d", "r"))] == list(rn._SCENARIOS)
    captured = capsys.readouterr()
    assert PLACEHOLDER not in captured.out + captured.err + msg


def test_an_acknowledgement_names_exactly_one_corpus(corpus_env, tmp_path, monkeypatch):
    _write_corpus(corpus_env, _rows("a"))
    other = corpus_content_hash(_rows("b"))
    monkeypatch.setenv(rn.NONPAPER_CORPUS_ENV, other)
    with pytest.raises(rn.RenellmCorpusRefused, match="names a different corpus"):
        rn.run_renellm(StubDefense(), [Goal("g0", "x", "c", "d", "r")], out_dir=tmp_path / "store")


def test_an_acknowledged_non_paper_replay_is_labelled_everywhere(corpus_env, tmp_path, monkeypatch, capsys):
    h = _write_corpus(corpus_env, _rows("a"))
    monkeypatch.setenv(rn.NONPAPER_CORPUS_ENV, h)
    out = tmp_path / "store"
    res = rn.run_renellm(StubDefense(), [Goal("g0", "x", "c", "d", "r")], out_dir=out)
    ident = res["renellm_corpus"]
    assert ident["paper_corpus"] is False and ident["content_hash"] == h
    assert ident["acknowledged_by"] == rn.NONPAPER_CORPUS_ENV and ident["paper_content_hash"] == rn.PINNED_SHA
    assert (ident["row_count"], ident["n_behaviors"]) == (6, 2) and res["n_goals_run"] == 1
    record = json.loads((out / rn.STORE_CORPUS_RECORD).read_text())
    assert record["content_hash"] == h and record["paper_corpus"] is False and record["rows_before_record"] == 0
    assert oct((out / rn.STORE_CORPUS_RECORD).stat().st_mode & 0o777) == "0o600"
    err = capsys.readouterr().err
    assert err.count("NON-PAPER corpus") == 2, "announced at admission and again when the run finishes"
    assert PLACEHOLDER not in err and PLACEHOLDER not in json.dumps(record)


def test_the_papers_corpus_replays_with_no_acknowledgement(corpus_env, tmp_path, monkeypatch, capsys):
    """The pinned hash itself is the paper's corpus (stood in for by a synthetic corpus here)."""
    h = _write_corpus(corpus_env, _rows("a"))
    monkeypatch.setattr(rn, "PINNED_SHA", h)
    out = tmp_path / "store"
    res = rn.run_renellm(StubDefense(), [Goal("g1", "x", "c", "d", "r")], out_dir=out)
    assert res["renellm_corpus"]["paper_corpus"] is True and res["renellm_corpus"]["acknowledged_by"] is None
    assert json.loads((out / rn.STORE_CORPUS_RECORD).read_text())["paper_corpus"] is True
    assert "NON-PAPER" not in capsys.readouterr().err
    assert rn.describe_identity(res["renellm_corpus"]).endswith("the paper's frozen corpus")


def test_a_store_holds_one_corpus(corpus_env, tmp_path, monkeypatch):
    h_a = _write_corpus(corpus_env, _rows("a"))
    monkeypatch.setenv(rn.NONPAPER_CORPUS_ENV, h_a)
    out = tmp_path / "store"
    rn.run_renellm(StubDefense(), [Goal("g0", "x", "c", "d", "r")], out_dir=out)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    h_b = _write_corpus(corpus_env, _rows("b"))
    rn._CORPUS_CACHE.clear()
    monkeypatch.setenv(rn.NONPAPER_CORPUS_ENV, h_b)
    with pytest.raises(rn.RenellmCorpusRefused, match="one corpus's rows") as ei:
        rn.run_renellm(StubDefense(), [Goal("g1", "x", "c", "d", "r")], out_dir=out)
    assert h_a in str(ei.value) and h_b in str(ei.value)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before, "the refused resume left the store untouched"


def test_a_store_with_unrecorded_rows_takes_only_the_papers_corpus(tmp_path):
    out = tmp_path / "store"
    out.mkdir()
    (out / "responses.jsonl").write_text('{"goal_id": "g0"}\n')
    ident = {"content_hash": "c" * 64, "paper_corpus": False}
    with pytest.raises(rn.RenellmCorpusRefused, match="records no corpus"):
        rn.bind_store(out, ident)
    assert not (out / rn.STORE_CORPUS_RECORD).exists()
    paper = {"content_hash": "d" * 64, "paper_corpus": True}
    record = json.loads(rn.bind_store(out, paper).read_text())
    assert record["rows_before_record"] == 1, "the record says how many rows predate it"
    (out / rn.STORE_CORPUS_RECORD).write_text("{not json")
    with pytest.raises(rn.RenellmCorpusRefused, match="unreadable"):
        rn.assert_store_accepts(out, paper)


def test_no_corpus_at_all_is_still_a_loud_skip_naming_the_shipped_file(corpus_env, tmp_path, monkeypatch):
    """Neither the site's copy nor the shipped one (a tree that lost data/heldout_corpora/renellm.jsonl)."""
    monkeypatch.setattr(rn, "SHIPPED_CORPUS", "data/heldout_corpora/absent-for-this-test.jsonl")
    assert not rn.replay_path().exists()
    res = rn.run_renellm(StubDefense(), [Goal("g0", "x", "c", "d", "r")], out_dir=tmp_path / "store")
    assert res["skipped"] is True
    assert rn.PINNED_FILE_SHA256 in res["reason"] and rn.PAPER_CORPUS_RECORD in res["reason"]
    assert f"git checkout -- {rn.SHIPPED_CORPUS}" in res["reason"]
    assert "build_heldout_corpus" not in res["reason"]
    assert rn.preflight_store(tmp_path / "store") is None


# --------------------------------------------------------------------------- #
# the check script the job scripts run before standing anything up             #
# --------------------------------------------------------------------------- #

def _check(args, env_extra=None):
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "LC_ALL", "HOME")}
    env.update(env_extra or {})
    # -I -S: no user site, no site-packages -- the check must run under a bare system python
    return subprocess.run([sys.executable, "-I", "-S", str(REPO / "scripts/renellm_corpus_check.py"), *args],
                          capture_output=True, text=True, env=env, cwd=str(REPO), timeout=120)


def test_the_check_script_runs_on_the_standard_library_and_exits_by_verdict(tmp_path):
    h = _write_corpus(tmp_path, _rows("a"))
    corpus = str(tmp_path / "renellm.jsonl")
    missing = _check([str(tmp_path / "absent.jsonl")])
    assert missing.returncode == 1 and rn.PINNED_FILE_SHA256 in missing.stderr
    refused = _check([corpus])
    assert refused.returncode == 2 and "REFUSED" in refused.stderr and h in refused.stderr and refused.stdout == ""
    ok = _check([corpus], {rn.NONPAPER_CORPUS_ENV: h})
    assert ok.returncode == 0 and "NON-PAPER corpus" in ok.stdout
    as_json = json.loads(_check([corpus, "--json"], {rn.NONPAPER_CORPUS_ENV: h}).stdout)
    assert as_json["content_hash"] == h and as_json["paper_corpus"] is False
    for r in (missing, refused, ok):
        assert PLACEHOLDER not in r.stdout + r.stderr


# --------------------------------------------------------------------------- #
# the job scripts' guard, run as the job runs it                               #
# --------------------------------------------------------------------------- #

def _guard_lines(text: str) -> str:
    """The renellm guard's body: from the corpus-dir export through the check's refusal."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if "export SISTEER_HELDOUT_CORPUS_DIR=" in ln)
    end = next(i for i, ln in enumerate(lines) if i > start and "nothing was stood up" in ln)
    body = "\n".join(lines[start:end + 1])
    assert body.endswith(";;")
    return body[:-2]


@pytest.mark.parametrize("rel", SBATCH)
def test_every_static_job_script_checks_the_corpus_before_standup(rel, tmp_path):
    text = (REPO / rel).read_text()
    assert "build_heldout_corpus.py renellm" not in text, "the guard must not send anyone to a rebuild"
    check = text.index("scripts/renellm_corpus_check.py")
    assert check < text.index("f22_driver_venv", check), "the corpus is checked before any venv or standup"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "python3").symlink_to(sys.executable)
    script = "f22_fatal() { echo \"FATAL: $*\" >&2; exit 9; }\n" + _guard_lines(text) + "\necho GUARD-PASSED\n"
    env = {"PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}", "SHARED": str(tmp_path / "shared")}

    def run(extra=None):
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=str(REPO),
                              env=dict(env, **(extra or {})), timeout=120)

    cdir = tmp_path / "shared" / "exp" / "heldout_corpora"
    # no copy on this cluster: the guard checks, and passes, the paper's corpus as the repository ships it
    shipped = run()
    assert shipped.returncode == 0 and "GUARD-PASSED" in shipped.stdout, shipped.stderr
    assert re.fullmatch(rf"\[renellm\] corpus {rn.PINNED_SHA} \({rn.PAPER_CORPUS_ROWS} rows, {rn.PAPER_CORPUS_BEHAVIORS} "
                        rf"behaviours\): the paper's frozen corpus \(.+/{re.escape(rn.SHIPPED_CORPUS)}\)",
                        shipped.stdout.splitlines()[0]), "one identity line, hashes and counts only"
    # this cluster's own copy of the paper's corpus (a byte copy, file to file) is read first, and passes
    cdir.mkdir(parents=True)
    shutil.copyfile(REPO / rn.SHIPPED_CORPUS, cdir / "renellm.jsonl")
    site = run()
    assert site.returncode == 0 and "GUARD-PASSED" in site.stdout, site.stderr
    assert f"the paper's frozen corpus ({cdir / 'renellm.jsonl'})" in site.stdout
    h = _write_corpus(cdir, _rows("a"))
    refused = run()
    assert refused.returncode == 9 and "corpus check refused" in refused.stderr and "REFUSED" in refused.stderr
    passed = run({rn.NONPAPER_CORPUS_ENV: h})
    assert passed.returncode == 0 and "GUARD-PASSED" in passed.stdout and "NON-PAPER corpus" in passed.stdout


# --------------------------------------------------------------------------- #
# the driver refuses before it writes, and labels a non-paper run               #
# --------------------------------------------------------------------------- #

def _driver(monkeypatch, tiny_llama, tmp_path, family_result):
    import jailbreak_run
    import sisteer.models as models_mod

    model, tok = tiny_llama
    monkeypatch.setattr(models_mod, "load_model_and_tokenizer", lambda *a, **k: (model, tok))
    monkeypatch.setattr(jailbreak_run, "_load_attacker",
                        lambda cfg: (_ for _ in ()).throw(AssertionError("no attacker for a static family")))
    monkeypatch.setattr(jailbreak_run, "_eval_goals",
                        lambda cfg, *, confirmatory: [Goal(f"g{i}", "x", "c", "d", "r") for i in range(2)])
    monkeypatch.setattr(jailbreak_run, "_run_replayable_family", family_result)
    cfg = {"model": "m", "dtype": "float32", "private_run": "jb"}
    args = SimpleNamespace(attack="renellm", defense="none", shard=0, nshards=1, xstest=False,
                           confirmatory=False, trial=0, private_root=str(tmp_path / "state"))
    return jailbreak_run, args, cfg


def test_the_driver_refuses_a_non_paper_corpus_before_writing(corpus_env, monkeypatch, tiny_llama, tmp_path):
    h = _write_corpus(corpus_env, _rows("a"))

    def never(*a, **k):
        raise AssertionError("the family must not run on a refused corpus")

    jr, args, cfg = _driver(monkeypatch, tiny_llama, tmp_path, never)
    with pytest.raises(SystemExit) as ei:
        jr.cmd_generate(args, cfg)
    assert "not the paper's" in str(ei.value.code) and h in str(ei.value.code)
    state = tmp_path / "state"
    written = [p for p in state.rglob("*") if p.is_file()] if state.exists() else []
    assert written == [], "the refusal lands before the driver writes a byte"


def test_the_driver_labels_a_non_paper_run(corpus_env, monkeypatch, tiny_llama, tmp_path, capsys):
    h = _write_corpus(corpus_env, _rows("a"))
    monkeypatch.setenv(rn.NONPAPER_CORPUS_ENV, h)

    def family(attack, defense, goals_arg, cfg, out_dir, *, trial=0, resume=True):
        return rn.run_renellm(StubDefense(), goals_arg, out_dir=out_dir)

    jr, args, cfg = _driver(monkeypatch, tiny_llama, tmp_path, family)
    jr.cmd_generate(args, cfg)
    out = capsys.readouterr().out
    assert f"[renellm] NON-PAPER corpus {h}" in out
    assert f"NON-PAPER corpus {h} (not comparable to the paper's renellm results)" in out.splitlines()[-1]

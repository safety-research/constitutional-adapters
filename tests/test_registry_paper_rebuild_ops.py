"""Rebuild trigger and publish-implies-recorded (registry side): ``paper_rebuild.sh``
refuses on a synthetic roster gap and leaves the published page byte-identical; builds, registers the
figures as artifacts and records OK when the roster is complete; a failing build restores the previous page;
an unchanged head does nothing; a held build lock is a BUSY no-op; ``registry publish-figures`` refuses an
unregistered file (in the source, in the destination, and in an already-published directory);
``registry paper-head --json`` reports the head, the schema version and the gaps; the watermark keys on (head, schema version),
so a migration that changes what the frame reads without moving a rank rebuilds the page once (migration 0021).
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys

from sisteer.registry import cli
from sisteer.registry import config as CFG
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry.migrations import code_version

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
SCRIPT = os.path.join(REPO, "scripts", "registry", "paper_rebuild.sh")

FAKE_BUILD = """#!/usr/bin/env bash
# a stand-in for paper/build.sh: draws one figure, commits it, publishes, stamps BUILD DONE (or fails on request).
# coreutils + git ONLY: the Modal CPU suite image (debian_slim + git) has no rsync, and under set -e a missing
# binary here reads as "build failed" in every test that reaches the publish step (2026-09-17).
set -euo pipefail
cd "$(dirname "$0")"
echo run >> "$FAKE_MARKER"
: > build.log
echo "BUILD START" >> build.log
mkdir -p figures
printf 'PNG-%s' "$(date +%s%N)" > figures/fig1.png
git add figures >/dev/null && git -c user.email=t@t -c user.name=t commit -q -m fig --allow-empty >/dev/null
if [ "${FAKE_FAIL:-0}" = 1 ]; then
  mkdir -p "$FAKE_PUBLISH/figures"; echo "half-built" > "$FAKE_PUBLISH/index.html"; echo "GATE FAILED" >> build.log; exit 1
fi
rm -rf "$FAKE_PUBLISH/figures"; mkdir -p "$FAKE_PUBLISH/figures"; cp -a figures/. "$FAKE_PUBLISH/figures/"   # publish.sh's mirror-with-delete semantics
echo "<html>page $(date +%s%N)</html>" > "$FAKE_PUBLISH/index.html"
if [ "${FAKE_STRAY:-0}" = 1 ]; then echo stray > "$FAKE_PUBLISH/figures/stray.png"; fi
echo "BUILD DONE" >> build.log
"""


def _roster_entry(i, object_selector=None, comparison="fig_synth"):
    p = {"roster_entry_id": f"re{i}", "round_id": "r_synth", "comparison_id": comparison, "model_id": synth.MODEL_ID,
         "eval_id": synth.EVAL_ID, "instrument_id": synth.INSTRUMENT_ID, "judge_id": synth.JUDGE_ID, "strength_rule": "operating_point",
         "source_citation": "paperlib.config.SYNTHETIC", "inferred": 0,
         "object_selector_json": object_selector or {"paper_label": "constitution_lora"}}
    return synth.rec("roster_entry", p, synth.t(50 + i))


def _world(tmp_path, entries):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    synth.write_corpus(root, n_runs=2, n_items=4)
    R.write_records(os.path.join(root, "records", "roster", "000001-roster.jsonl"), entries)
    H.ingest_once(root, workers=0, min_free=0)
    checkout = str(tmp_path / "paper-checkout")
    os.makedirs(os.path.join(checkout, "paper"))
    subprocess.run(["git", "init", "-q", checkout], check=True)
    with open(os.path.join(checkout, "paper", "build.sh"), "w") as fh:
        fh.write(FAKE_BUILD)
    os.chmod(os.path.join(checkout, "paper", "build.sh"), 0o755)
    subprocess.run(["git", "-C", checkout, "add", "."], check=True)
    subprocess.run(["git", "-C", checkout, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"], check=True)
    publish = str(tmp_path / "published")
    os.makedirs(os.path.join(publish, "figures"))
    with open(os.path.join(publish, "index.html"), "w") as fh:
        fh.write("<html>previous page</html>")
    with open(os.path.join(publish, "figures", "old.png"), "wb") as fh:
        fh.write(b"old figure bytes")
    lock = str(tmp_path / "lockdir" / "lock")
    with open(CFG.config_path(root), "w") as fh:
        fh.write(f"""[registry]
root = "{root}"
user = "u"
[paper]
checkout = "{checkout}"
publish_dir = "{publish}"
build_lock = "{lock}"
keep_failed_builds = 2
""")
    cfg = CFG.load(root)
    assert cfg["paper"]["publish_dir"] == publish and cfg["paper"]["build_lock"] == lock and cfg["paper"]["checkout"] == checkout
    return root, checkout, publish, lock


def _digest(path):
    out = {}
    for dp, _, fns in os.walk(path):
        for fn in fns:
            p = os.path.join(dp, fn)
            out[os.path.relpath(p, path)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return out


def _run(root, tmp_path, **env):
    marker = str(tmp_path / "marker")
    e = {"REGISTRY_ROOT": root, "REGISTRY_PY": sys.executable, "REGISTRY_PYTHONPATH": SRC, "PATH": os.environ["PATH"], "HOME": str(tmp_path),
         "FAKE_MARKER": marker, "FAKE_PUBLISH": str(tmp_path / "published")}
    e.update(env)
    r = subprocess.run(["bash", SCRIPT], capture_output=True, text=True, env=e, timeout=120)
    result = open(os.path.join(root, "state", "paper_rebuild.RESULT")).read().strip() if os.path.exists(os.path.join(root, "state", "paper_rebuild.RESULT")) else None
    builds = open(marker).read().count("run") if os.path.exists(marker) else 0
    return r, result, builds


def _head(root):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return int(conn.execute("SELECT ingest_seq FROM v_paper_inputs_head").fetchone()[0])
    finally:
        conn.close()


def test_refuses_on_roster_gap_and_leaves_the_page_unchanged(tmp_path):
    # re1 matches run 1 (constitution_lora); re3 selects an object no run has → a gap with no disposition
    root, checkout, publish, lock = _world(tmp_path, [_roster_entry(1), _roster_entry(3, {"paper_label": "difference_lora"}, "fig_gap")])
    before = _digest(publish)
    r, result, builds = _run(root, tmp_path)
    assert r.returncode == 0, r.stderr
    assert result.startswith(f"REFUSED {_head(root)} ") and "re3" in result and "neither a judged run nor a disposition: 1" in result
    assert builds == 0, "no build ran"
    assert _digest(publish) == before, "the published page is byte-identical"
    assert open(os.path.join(root, "state", "paper_failed_head")).read().split()[0] == str(_head(root))
    # the same head again: exit 0, still no build, RESULT unchanged
    r2, result2, builds2 = _run(root, tmp_path)
    assert r2.returncode == 0 and builds2 == 0 and result2 == result
    # the status page shows the refusal
    from sisteer.registry import status
    out = str(tmp_path / "index.html")
    status.render(root, out_path=out, register=False)
    assert "REFUSED" in open(out).read()


def test_builds_registers_figures_and_records_ok_then_idles(tmp_path):
    root, checkout, publish, lock = _world(tmp_path, [_roster_entry(1)])
    head = _head(root)
    r, result, builds = _run(root, tmp_path)
    assert r.returncode == 0, r.stderr + "\n" + open(os.path.join(root, "state", "paper_rebuild.log")).read()
    assert result.startswith(f"OK {head} ") and builds == 1
    assert open(os.path.join(root, "state", "paper_last_head")).read().split()[0] == str(head)
    assert not os.path.exists(os.path.join(root, "state", "paper_failed_head")) and not os.path.isdir(publish + ".prev")
    assert open(os.path.join(publish, "index.html")).read().startswith("<html>page")
    # the figure is an artifact row (kind figure) carrying the notebook commit
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    fig = os.path.join(checkout, "paper", "figures", "fig1.png")
    row = conn.execute("SELECT kind, extra_json FROM artifact WHERE uri_primary = ?", (fig,)).fetchone()
    assert row is not None and row[0] == "figure"
    sha = subprocess.run(["git", "-C", checkout, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert json.loads(row[1])["notebook_commit"] == sha
    conn.close()
    # unchanged head: nothing happens
    r2, result2, builds2 = _run(root, tmp_path)
    assert r2.returncode == 0 and builds2 == 1 and result2 == result
    # the head moves (a ruling lands) → another build
    R.write_records(os.path.join(root, "records", "roster", "000002-ruling.jsonl"),
                    [synth.rec("ruling", {"ruling_id": "R9", "round_id": "r_synth", "action": "reference", "scope_kind": "run",
                                          "scope_json": {"run_id": "nope"}, "reason": "s", "ruled_utc": synth.t(90), "ruled_by": "u",
                                          "source_ref": "t"}, synth.t(90))])
    H.ingest_once(root, workers=0, min_free=0)
    assert _head(root) > head
    r3, result3, builds3 = _run(root, tmp_path)
    assert r3.returncode == 0 and builds3 == 2 and result3.startswith(f"OK {_head(root)} ")


def test_the_watermark_keys_on_the_schema_version_so_a_migration_rebuilds_once(tmp_path):
    """A migration that recreates a v_* object changes what the frame reads without moving a rank (0021 retired 361 anchored rows and 7
    superseded-judgment rows that way; found in review, 2026-10-02), so the trigger's watermark is (head, schema version), not the head
    alone: a `paper_last_head` written at this head under another schema -- or before the schema term was recorded (the two-field form
    every page built before 0021 left) -- rebuilds once and then idles; a `paper_failed_head` under another schema stops blocking, under
    the same schema it still does. `registry paper-head --json` reports the version the script keys on."""
    root, checkout, publish, lock = _world(tmp_path, [_roster_entry(1)])
    head = _head(root)
    last, failed = os.path.join(root, "state", "paper_last_head"), os.path.join(root, "state", "paper_failed_head")
    r, result, builds = _run(root, tmp_path)
    assert r.returncode == 0 and builds == 1 and result.startswith(f"OK {head} ") and f"schema v{code_version()}" in result
    assert open(last).read().split()[:3:2] == [str(head), f"schema={code_version()}"]
    head_json = json.loads(subprocess.run([sys.executable, "-m", "sisteer.registry.cli", "--root", root, "paper-head", "--json"],
                                          capture_output=True, text=True, check=True, env={**os.environ, "PYTHONPATH": SRC}).stdout)
    assert head_json["head"] == head and head_json["schema_version"] == code_version()
    # the two-field form a page built before the schema term left: another schema as far as the key knows -> one rebuild, then idle
    with open(last, "w") as fh:
        fh.write(f"{head} 2026-01-01T00:00:00Z\n")
    r, result, builds = _run(root, tmp_path)
    assert r.returncode == 0 and builds == 2 and result.startswith(f"OK {head} ")
    assert open(last).read().split()[:3:2] == [str(head), f"schema={code_version()}"]
    r, result, builds = _run(root, tmp_path)
    assert r.returncode == 0 and builds == 2                                             # the same head under the same schema: nothing
    # the same head under the previous schema version: a rebuild
    with open(last, "w") as fh:
        fh.write(f"{head} 2026-01-01T00:00:00Z schema={code_version() - 1}\n")
    r, result, builds = _run(root, tmp_path)
    assert r.returncode == 0 and builds == 3 and open(last).read().split()[2] == f"schema={code_version()}"
    # a failed head under another schema no longer blocks; under the same schema it does
    os.remove(last)
    with open(failed, "w") as fh:
        fh.write(f"{head} 2026-01-01T00:00:00Z schema={code_version() - 1}\n")
    r, result, builds = _run(root, tmp_path)
    assert r.returncode == 0 and builds == 4 and not os.path.exists(failed) and os.path.exists(last)
    os.remove(last)
    with open(failed, "w") as fh:
        fh.write(f"{head} 2026-01-01T00:00:00Z schema={code_version()} refused\n")
    r, result, builds = _run(root, tmp_path)
    assert r.returncode == 0 and builds == 4 and os.path.exists(failed) and not os.path.exists(last)


def test_build_failure_restores_the_previous_page(tmp_path):
    root, checkout, publish, lock = _world(tmp_path, [_roster_entry(1)])
    before = _digest(publish)
    r, result, builds = _run(root, tmp_path, FAKE_FAIL="1")
    assert r.returncode == 1 and builds == 1
    assert result.startswith(f"FAILED {_head(root)} ") and "previous page restored" in result
    assert _digest(publish) == before, "the half-built page was swapped back for the previous one"
    failed = [d for d in os.listdir(os.path.dirname(publish)) if d.startswith(os.path.basename(publish) + ".failed-")]
    assert len(failed) == 1 and open(os.path.join(os.path.dirname(publish), failed[0], "index.html")).read().strip() == "half-built"
    assert open(os.path.join(root, "state", "paper_failed_head")).read().split()[:3:2] == [str(_head(root)), f"schema={code_version()}"]
    r2, _, builds2 = _run(root, tmp_path, FAKE_FAIL="1")
    assert r2.returncode == 0 and builds2 == 1, "an already-failed head is not retried"


def test_published_file_without_a_record_fails_and_restores(tmp_path):
    root, checkout, publish, lock = _world(tmp_path, [_roster_entry(1)])
    before = _digest(publish)
    r, result, builds = _run(root, tmp_path, FAKE_STRAY="1")
    assert r.returncode == 1 and result.startswith("FAILED") and "publish-implies-recorded" in result
    assert _digest(publish) == before
    log = open(os.path.join(root, "state", "paper_rebuild.log")).read()
    assert "stray.png" in log and "no artifact row" in log


def test_held_build_lock_is_a_busy_noop(tmp_path):
    root, checkout, publish, lock = _world(tmp_path, [_roster_entry(1)])
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    fh = open(lock, "w")
    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    try:
        r, result, builds = _run(root, tmp_path)
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()
    assert r.returncode == 0 and result.startswith(f"BUSY {_head(root)} ") and builds == 0


def test_publish_figures_cli_refuses_unregistered_files(tmp_path, monkeypatch, capsys):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    synth.write_corpus(root, n_runs=1, n_items=2)
    H.ingest_once(root, workers=0, min_free=0)
    monkeypatch.setenv("REGISTRY_ROOT", root)
    src = tmp_path / "figures"
    src.mkdir()
    (src / "a.png").write_bytes(b"a")
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "stray.png").write_bytes(b"never registered")
    # --verify-only against a directory with an unregistered file refuses by name
    assert cli.main(["publish-figures", "--dir", str(src), "--verify-only", "--verify-dir", str(dest)]) == 1
    assert "stray.png" in capsys.readouterr().err
    # register + publish: the destination's stray file refuses the publish by name
    assert cli.main(["publish-figures", "--dir", str(src), "--notebook-commit", "b" * 40, "--publish-to", str(dest)]) == 1
    assert "stray.png" in capsys.readouterr().err
    (dest / "stray.png").unlink()
    assert cli.main(["publish-figures", "--dir", str(src), "--notebook-commit", "b" * 40, "--publish-to", str(dest), "--verify-dir", str(dest)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["published"] == [str(src / "a.png")] and (dest / "a.png").read_bytes() == b"a"
    # a source file whose bytes drift after registration is refused
    (src / "a.png").write_bytes(b"changed")
    assert cli.main(["publish-figures", "--dir", str(src), "--verify-only", "--publish-to", str(dest)]) == 1
    assert "bytes differ" in capsys.readouterr().err


def test_paper_head_cli_reports_head_and_gaps(tmp_path, monkeypatch, capsys):
    root, checkout, publish, lock = _world(tmp_path, [_roster_entry(1), _roster_entry(3, {"paper_label": "difference_lora"}, "fig_gap")])
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert cli.main(["paper-head"]) == 0
    assert capsys.readouterr().out.strip() == str(_head(root))
    assert cli.main(["paper-head", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["head"] == _head(root) and data["open_round"] == "r_synth" and data["roster_gaps"] == 1
    assert data["schema_version"] == code_version()                                      # the watermark's second term (migration 0021)
    assert data["gaps"][0]["roster_entry_id"] == "re3" and data["indeterminate"] == 0


def test_default_interpreter_is_the_scripts_own_checkout(tmp_path):
    """With REGISTRY_PY / REGISTRY_PYTHONPATH unset the trigger runs the registry from the checkout it sits in
    (<checkout>/.venv/bin/python with <checkout>/src on PYTHONPATH), never a home-directory path named after the repo."""
    co = tmp_path / "any-checkout-name"
    (co / "scripts" / "registry").mkdir(parents=True)
    script = co / "scripts" / "registry" / "paper_rebuild.sh"
    script.write_bytes(open(SCRIPT, "rb").read())
    seen = tmp_path / "python-argv"
    stub = co / ".venv" / "bin" / "python"
    stub.parent.mkdir(parents=True)
    stub.write_text(f'#!/bin/sh\nprintf "%s\\n" "PYTHONPATH=$PYTHONPATH" "$@" > {seen}\nexit 1\n')
    stub.chmod(0o755)
    root = tmp_path / "reg"
    env = {"REGISTRY_ROOT": str(root), "PATH": os.environ["PATH"], "HOME": str(tmp_path / "home")}
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 1, r.stdout + r.stderr                     # the stub refuses `config paper.checkout`
    assert seen.exists(), "the checkout's own interpreter was not the one run: " + r.stderr
    lines = seen.read_text().splitlines()
    assert lines[0] == f"PYTHONPATH={co / 'src'}" and lines[1:4] == ["-m", "sisteer.registry.cli", "--root"], lines
    assert "registry config unreadable" in (root / "state" / "paper_rebuild.RESULT").read_text()

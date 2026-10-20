"""The misalign grader files the SERVED model identity -- what the judged root's identity files
state, read by sisteer.capeval.misalign.rows.resolve_model_identity exactly as on the cluster (the grade daemon mirrors those files beside
the stores) -- and the registry's identity, passed as --identity-model-id / --identity-model-revision, is a CROSS-CHECK that refuses by
name on a mismatch, never an override. Under --served-identity-dir (the grade daemon's immutable per-dispatch snapshot of the root's
identity files) the preflight AND the rows step read the served identity from that directory, never from the shared results root; and a
rows step that fails fails the run (exit 1), never a 0 with the derived rows missing. `--rows-only` re-runs the preflight and the rows +
analysis steps over the retained snapshot without judging or filing (the daemon's retry), and `--rows-args` appends to the rows argv
with `--identity-root` always last. No API call."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "scripts" / "sdfbase" / "full22" / "misalign" / "misalign_judge.py"
HF, REV = "org/model", "a" * 40


def _load():
    spec = importlib.util.spec_from_file_location("misalign_judge_identity_under_test", SRC)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)   # type: ignore[union-attr]
    return mod


def test_the_rows_step_argv_never_carries_the_registrys_identity():
    mj = _load()
    ns = mj.build_parser().parse_args(["--results-root", "/synthetic/root", "--target", "t", "--identity-model-id", HF, "--identity-model-revision", REV])
    assert mj.rows_argv(ns, Path("/synthetic/root"), "t") == ["--judged-root", "/synthetic/root", "--target", "t"]
    assert (ns.identity_model_id, ns.identity_model_revision) == (HF, REV) and ns.model_id is None       # --model-id stays the worklist's selector
    assert mj.identity_root_of(ns, Path("/synthetic/root")) == Path("/synthetic/root")                   # no snapshot passed: the root itself (a hand run)
    # under --served-identity-dir the rows step reads the identity from the dispatch's snapshot -- still never the registry's
    ns2 = mj.build_parser().parse_args(["--results-root", "/synthetic/root", "--target", "t", "--identity-model-id", HF, "--identity-model-revision", REV,
                                        "--served-identity-dir", "/synthetic/state/dispatch_logs/x_agentic_k.1.identity"])
    assert mj.identity_root_of(ns2, Path("/synthetic/root")) == Path("/synthetic/state/dispatch_logs/x_agentic_k.1.identity")
    assert mj.rows_argv(ns2, Path("/synthetic/root"), "t") == ["--judged-root", "/synthetic/root", "--target", "t",
                                                               "--identity-root", "/synthetic/state/dispatch_logs/x_agentic_k.1.identity"]
    assert "--model-id" not in mj.rows_argv(ns2, Path("/synthetic/root"), "t") and HF not in mj.rows_argv(ns2, Path("/synthetic/root"), "t")


def test_the_served_identity_is_cross_checked_against_the_registrys_and_a_mismatch_refuses_by_name(tmp_path):
    from sisteer.capeval.misalign import rows as mis_rows
    mj = _load()
    root = tmp_path / "judged"; root.mkdir()
    # no served identity, no expectation: the rows step's own refusal, unchanged
    with pytest.raises(mis_rows.MisalignRowsError, match="no model identity"):
        mj.cross_check_identity(root, None, None)
    # no served identity but a registry expectation: refused -- never filed under the registry's identity alone
    with pytest.raises(mj.IdentityMismatch, match="no served identity to cross-check"):
        mj.cross_check_identity(root, HF, REV)
    # the served identity (endpoint.snapshot.json, as the cluster writes it) agreeing with the registry's: filed from the FILES
    (root / "endpoint.snapshot.json").write_text(json.dumps({"model": HF, "model_revision": REV}))
    assert mj.cross_check_identity(root, HF, REV) == (HF, REV, "endpoint.snapshot.json")
    assert mj.cross_check_identity(root, None, None) == (HF, REV, "endpoint.snapshot.json")              # no expectation: the served identity stands
    # a run served at another snapshot than its registry row states: refused, both named
    with pytest.raises(mj.IdentityMismatch, match=f"served identity {HF} @ {REV}.*disagrees with the registry's {HF} @ {'b' * 40}"):
        mj.cross_check_identity(root, HF, "b" * 40)
    with pytest.raises(mj.IdentityMismatch, match="disagrees with the registry's other/model"):
        mj.cross_check_identity(root, "other/model", REV)
    # cells_full.json outranks endpoint.snapshot.json in the rows step's own candidate order, and the cross-check reads that order
    (root / "cells_full.json").write_text(json.dumps({"endpoint": {"model": HF, "model_revision": "c" * 40}}))
    assert mj.cross_check_identity(root, HF, "c" * 40) == (HF, "c" * 40, "cells_full.json")
    with pytest.raises(mj.IdentityMismatch):
        mj.cross_check_identity(root, HF, REV)


def test_the_preflight_reads_the_dispatch_snapshot_and_never_the_results_root(tmp_path, capsys):
    """The grade daemon's second instance may delete and re-copy the shared root's identity files
    underneath a running grader. With --served-identity-dir the preflight reads the dispatch's own snapshot: a root stating ANOTHER revision
    does not matter when the snapshot agrees with the registry's; a snapshot with no identity refuses although the root carries one (the
    root is never a fallback); a missing snapshot directory refuses by name -- through main, before anything is claimed or filed."""
    mj = _load()
    root = tmp_path / "judged"; root.mkdir()
    snap = tmp_path / "state" / "dispatch_logs" / "20261002T000000Z_agentic_k.4242.identity"; snap.mkdir(parents=True)
    (root / "endpoint.snapshot.json").write_text(json.dumps({"model": HF, "model_revision": "b" * 40}))        # the shared root: re-mirrored, another revision
    (snap / "cells_full.json").write_text(json.dumps({"endpoint": {"model": HF, "model_revision": REV}}))        # the dispatch's snapshot: what was served
    ns = mj.build_parser().parse_args(["--results-root", str(root), "--identity-model-id", HF, "--identity-model-revision", REV, "--served-identity-dir", str(snap)])
    assert mj.preflight_identity(ns, root) == (HF, REV, "cells_full.json")
    assert mj.rows_argv(ns, root, "t")[-2:] == ["--identity-root", str(snap)]
    ns_root = mj.build_parser().parse_args(["--results-root", str(root), "--identity-model-id", HF, "--identity-model-revision", REV])
    with pytest.raises(mj.IdentityMismatch, match="disagrees with the registry's"):                                # without the snapshot the root's disagreement refuses
        mj.preflight_identity(ns_root, root)
    # an EMPTY snapshot refuses by name although the root agrees with the registry: the root is never read in the snapshot's place
    (root / "endpoint.snapshot.json").write_text(json.dumps({"model": HF, "model_revision": REV}))
    empty = tmp_path / "empty.identity"; empty.mkdir()
    ns_e = mj.build_parser().parse_args(["--results-root", str(root), "--identity-model-id", HF, "--identity-model-revision", REV, "--served-identity-dir", str(empty)])
    with pytest.raises(mj.IdentityMismatch, match="no served identity to cross-check") as ei:
        mj.preflight_identity(ns_e, root)
    assert str(empty) in str(ei.value)
    # a snapshot directory that does not exist: refused by name, through main (exit 1, nothing else happens), with and without a registry expectation
    missing = tmp_path / "gone.identity"
    for extra in (["--identity-model-id", HF, "--identity-model-revision", REV], []):
        assert mj.main(["--results-root", str(root), "--served-identity-dir", str(missing), *extra]) == 1
        out = capsys.readouterr().out
        assert "PREFLIGHT REFUSED" in out and "is not a directory" in out and str(missing) in out and "nothing claimed, nothing filed" in out


def test_a_failed_rows_step_fails_the_run_and_a_failed_analysis_does_not(tmp_path, capsys):
    """The rows step's nonzero return or exception used to be printed and then main returned 0 whenever the judgment items
    had no errors (the daemon held the run complete with its derived rows missing). Now derived_steps returns 1 -- and its LAST line names
    the rows step (the daemon reads the terminal line) -- when the rows step returns nonzero, exits nonzero, raises, or has no main; the
    analysis is not run after a failed rows step; a clean rows step with a failed analysis is reported and still 0; the rows step is
    handed the dispatch's snapshot as --identity-root."""
    mj = _load()
    root = tmp_path / "judged"; root.mkdir(); snap = tmp_path / "snap"; snap.mkdir()
    ns = mj.build_parser().parse_args(["--results-root", str(root), "--served-identity-dir", str(snap)])
    calls: list = []

    def rows_rc(rc):
        def run(argv):
            calls.append(("rows", list(argv))); return rc
        return run

    def analyze_rc(rc):
        def run(argv):
            calls.append(("analyze", list(argv))); return rc
        return run

    def raising(argv):
        calls.append(("rows", list(argv))); raise RuntimeError("no model identity (synthetic)")

    def exiting(argv):
        calls.append(("rows", list(argv))); raise SystemExit(3)

    def last_line():
        return [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()][-1]

    # nonzero return: 1, the analysis never runs, the terminal line names the rows step and its argv
    assert mj.derived_steps(ns, root, "t", rows_main=rows_rc(2), analyze_main=analyze_rc(0)) == 1
    assert [c[0] for c in calls] == ["rows"] and calls[0][1] == ["--judged-root", str(root), "--target", "t", "--identity-root", str(snap)]
    ln = last_line(); assert ln.startswith("[judge] ROWS STEP FAILED") and "returned 2" in ln and "exit 1" in ln
    # an exception: 1
    calls.clear()
    assert mj.derived_steps(ns, root, "t", rows_main=raising, analyze_main=analyze_rc(0)) == 1
    assert [c[0] for c in calls] == ["rows"]
    ln = last_line(); assert ln.startswith("[judge] ROWS STEP FAILED") and "raised RuntimeError" in ln and "no model identity (synthetic)" in ln
    # a nonzero SystemExit: 1
    calls.clear()
    assert mj.derived_steps(ns, root, "t", rows_main=exiting, analyze_main=analyze_rc(0)) == 1
    assert [c[0] for c in calls] == ["rows"] and "returned 3" in last_line()
    # a rows module without main: 1
    calls.clear()
    assert mj.derived_steps(ns, root, "t", rows_main=object(), analyze_main=analyze_rc(0)) == 1
    assert calls == [] and "has no main" in last_line()
    # a clean rows step: 0 even when the analysis fails (reported) -- and 0 when both are clean
    calls.clear()
    assert mj.derived_steps(ns, root, "t", rows_main=rows_rc(0), analyze_main=analyze_rc(5)) == 0
    assert [c[0] for c in calls] == ["rows", "analyze"] and calls[1][1] == ["--judged-root", str(root), "--target", "t"]
    out = capsys.readouterr().out
    assert "ROWS STEP FAILED" not in out and "analyze.main returned 5" in out
    calls.clear()
    assert mj.derived_steps(ns, root, "t", rows_main=rows_rc(None), analyze_main=analyze_rc(0)) == 0 and [c[0] for c in calls] == ["rows", "analyze"]
    # --rows-args is APPENDED to the argv (it used to replace it) and never changes the verdict
    ns_a = mj.build_parser().parse_args(["--results-root", str(root), "--rows-args", "--judged-root X --target Y"])
    calls.clear()
    assert mj.derived_steps(ns_a, root, "t", rows_main=rows_rc(1), analyze_main=analyze_rc(0)) == 1
    assert calls[0][1] == ["--judged-root", str(root), "--target", "t", "--judged-root", "X", "--target", "Y"]


def test_rows_args_append_to_the_rows_argv_and_identity_root_always_rides_last(tmp_path, capsys):
    """`--rows-args` used to REPLACE the rows argv, dropping --identity-root -- a custom rows invocation read the mutable root
    again. Now it is appended after the judge's own --judged-root / --target, and --identity-root comes last (argparse keeps the last
    value), with or without an override; without a snapshot nothing is appended but the override."""
    mj = _load()
    root = tmp_path / "judged"; root.mkdir(); snap = tmp_path / "snap"; snap.mkdir()
    calls: list = []

    def rows_ok(argv):
        calls.append(list(argv)); return 0

    ns = mj.build_parser().parse_args(["--results-root", str(root), "--served-identity-dir", str(snap), "--rows-args", "--no-consolidate --target Z --identity-root /elsewhere"])
    assert mj.derived_steps(ns, root, "t", rows_main=rows_ok, analyze_main=lambda argv: 0) == 0
    assert calls[-1] == ["--judged-root", str(root), "--target", "t", "--no-consolidate", "--target", "Z", "--identity-root", "/elsewhere", "--identity-root", str(snap)]
    ns2 = mj.build_parser().parse_args(["--results-root", str(root), "--rows-args=--no-consolidate"])
    assert mj.derived_steps(ns2, root, "t", rows_main=rows_ok, analyze_main=lambda argv: 0) == 0
    assert calls[-1] == ["--judged-root", str(root), "--target", "t", "--no-consolidate"]
    ns3 = mj.build_parser().parse_args(["--results-root", str(root), "--served-identity-dir", str(snap)])
    assert mj.derived_steps(ns3, root, "t", rows_main=rows_ok, analyze_main=lambda argv: 0) == 0
    assert calls[-1] == ["--judged-root", str(root), "--target", "t", "--identity-root", str(snap)]
    # the real rows parser keeps the LAST --identity-root: the snapshot wins over an override that named another directory
    from sisteer.capeval.misalign import rows as mis_rows
    assert str(mis_rows._build_parser().parse_args(calls[0]).identity_root) == str(snap)


def test_rows_only_runs_the_preflight_and_the_derived_steps_over_the_snapshot_and_touches_no_registry(tmp_path, monkeypatch, capsys):
    """The daemon's retry re-runs the grader `--rows-only` with the ORIGINAL dispatch command (worklist flags included):
    the preflight reads the retained snapshot, the rows step runs with --identity-root <snapshot>, the analysis follows, nothing is judged,
    claimed or filed (the registry root named on the line need not even exist); a failed rows step is exit 1 with the rows line last; a
    missing snapshot is refused before anything runs."""
    from sisteer.capeval.misalign import analyze as mis_analyze
    from sisteer.capeval.misalign import rows as mis_rows
    mj = _load()
    root = tmp_path / "judged"; root.mkdir()
    snap = tmp_path / "state" / "dispatch_logs" / "20261002T000000Z_agentic_k.4242.identity"; snap.mkdir(parents=True)
    (snap / "cells_full.json").write_text(json.dumps({"endpoint": {"model": HF, "model_revision": REV}}))
    calls: list = []
    monkeypatch.setattr(mis_rows, "main", lambda argv: calls.append(("rows", list(argv))) or 0)
    monkeypatch.setattr(mis_analyze, "main", lambda argv: calls.append(("analyze", list(argv))) or 0)
    nowhere = tmp_path / "no-registry-here"
    argv = ["--results-root", str(root), "--no-escalate", "--worklist", "registry", "--registry-root", str(nowhere), "--run-id", "synthetic:run:0001",
            "--store-root-map", "cluster_a=/cluster_a/x=/mirror/x", "--identity-model-id", HF, "--identity-model-revision", REV, "--served-identity-dir", str(snap)]
    assert mj.main(argv + ["--rows-only"]) == 0
    out = capsys.readouterr().out
    assert "rows-only: served identity" in out and "nothing judged or filed" in out and "cells_full.json" in out
    assert calls == [("rows", ["--judged-root", str(root), "--target", root.name, "--identity-root", str(snap)]), ("analyze", ["--judged-root", str(root), "--target", root.name])]
    assert not nowhere.exists() and not (root / "judge_provenance.json").exists()
    # the rows step failing: exit 1, the terminal line names it
    calls.clear(); monkeypatch.setattr(mis_rows, "main", lambda argv: 2)
    assert mj.main(argv + ["--rows-only"]) == 1
    assert [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()][-1].startswith("[judge] ROWS STEP FAILED")
    # a missing snapshot: refused before the rows step
    calls.clear(); monkeypatch.setattr(mis_rows, "main", lambda argv: calls.append(("rows", list(argv))) or 0)
    assert mj.main(["--results-root", str(root), "--served-identity-dir", str(tmp_path / "gone.identity"), "--rows-only"]) == 1
    assert "PREFLIGHT REFUSED -- rows-only" in capsys.readouterr().out and calls == []

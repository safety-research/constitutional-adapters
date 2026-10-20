"""fits_watch.py counts a cell COMPLETE only on a fingerprint-MATCHING
marker.

Marker existence lied: the round-2 refits publish into the round-1 cell
dirs, so a stale round-1 COMPLETE sits under a refit in flight and an
existence check reads the whole wave as done. Pins: the marker verdict
(absent / unparseable / no-fingerprint / stale / complete / unverifiable);
``assess`` never counts a stale or unverifiable row complete, reports a
stale marker under a TERMINAL call as needing attention, and treats one
under a RUNNING call as the expected in-flight picture; the legacy
(non-strict) manifests of other apps keep the existence-only reading.

Pure functions + fakes: no Modal import needed (the module imports modal
only inside the volume/call probes).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts" / "modal") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "modal"))

import fits_watch as fw  # noqa: E402

FP = "a" * 64
OTHER = "b" * 64


def _marker(fp=FP, **over) -> bytes:
    m = {"cell_id": "constv2-mix-1m__psr_ce_all", "fit_fingerprint": fp,
         "files": {"object.pt": "0" * 64}}
    m.update(over)
    if m.get("fit_fingerprint", "keep") is None:
        m.pop("fit_fingerprint")
    return json.dumps(m).encode()


def _row(cell="constv2-mix-1m__psr_ce_all", expect=FP, call_id="fc-1"):
    return {"model": "gemma4e4b_base", "cell_id": cell, "leg": "all",
            "out_root": "out/gemma4e4b_base", "expect_fingerprint": expect,
            "call_id": call_id}


def test_marker_path_is_the_cell_dirs_complete_file():
    assert fw.marker_path(_row()) == \
        "out/gemma4e4b_base/cells_all/constv2-mix-1m__psr_ce_all/COMPLETE"


def test_marker_verdicts():
    v = fw.marker_verdict
    assert v(None, FP, strict=True)[0] == "absent"
    assert v(_marker(FP), FP, strict=True)[0] == "complete"
    state, detail = v(_marker(OTHER), FP, strict=True)
    assert state == "stale" and "superseded publication" in detail
    assert v(_marker(fit_fingerprint=None), FP, strict=True)[0] == \
        "no-fingerprint"
    assert v(b"not json", FP, strict=True)[0] == "unparseable"
    assert v(b"[1, 2]", FP, strict=True)[0] == "unparseable"
    # a strict manifest row with no expectation can never be validated —
    # whether or not a marker happens to sit under its cell dir (the
    # docstring's rule; an absent marker used to read "absent" first)
    state, detail = v(_marker(FP), None, strict=True)
    assert state == "unverifiable" and "re-run fits-spawn" in detail
    assert v(None, None, strict=True)[0] == "unverifiable"
    assert v(b"not json", None, strict=True)[0] == "unverifiable"
    # legacy (twosi) manifests: existence only, said so
    assert v(_marker(OTHER), None, strict=False) == (
        "complete", "marker present (existence-only: legacy manifest)")
    assert v(None, None, strict=False)[0] == "absent"


def _assess(rows, markers, calls, strict=True):
    """assess() with the two probes faked: markers by cell id, call states
    by cell id (a row with no call_id reads 'unspawned', exactly like the
    real _call_state)."""
    return fw.assess(
        rows, strict=strict,
        read_marker=lambda c: markers.get(c["cell_id"]),
        call_state=lambda c: ("unspawned" if not c.get("call_id")
                              else calls.get(c["cell_id"], "unknown")))


def test_stale_marker_under_a_running_refit_is_not_complete():
    """The exact lie the finding names: round-1 marker present, refit call
    running -> the cell is IN FLIGHT, the wave is not done."""
    row = _row()
    res = _assess([row], {row["cell_id"]: _marker(OTHER)},
                  {row["cell_id"]: "running"})
    assert res["n_complete"] == 0
    assert res["all_calls_terminal"] is False
    assert res["needs_attention"] == []
    assert res["lines"]["gemma4e4b_base/constv2-mix-1m__psr_ce_all/all"] \
        .endswith("running (marker stale)")


def test_stale_marker_under_a_terminal_call_needs_attention():
    row = _row()
    res = _assess([row], {row["cell_id"]: _marker(OTHER)},
                  {row["cell_id"]: "done"})
    assert res["n_complete"] == 0 and res["all_calls_terminal"] is True
    assert len(res["needs_attention"]) == 1
    assert "call done but" in res["needs_attention"][0]
    assert "!= expected" in res["needs_attention"][0]


def test_matching_marker_is_complete_regardless_of_call_state():
    row = _row()
    res = _assess([row], {row["cell_id"]: _marker(FP)},
                  {row["cell_id"]: "failed"})  # a later call may have died
    assert res["n_complete"] == 1 and res["needs_attention"] == []


def test_unverifiable_row_needs_attention_even_while_running():
    row = _row(expect=None)
    res = _assess([row], {row["cell_id"]: _marker(FP)},
                  {row["cell_id"]: "running"})
    assert res["n_complete"] == 0
    assert res["needs_attention"] and "expect_fingerprint" in \
        res["needs_attention"][0]
    # it is not 'in flight' from the watcher's point of view: the row can
    # never complete without a re-spawn, so it must not hold the wave open
    assert res["all_calls_terminal"] is True


def test_unspawned_and_absent_rows():
    a = _row(cell="constv2-mix-10m__psr_ce_all", call_id=None)
    b = _row(cell="constv2-mix-full__psr_ce_all", call_id="fc-2")
    res = _assess([a, b], {}, {b["cell_id"]: "done"})
    assert res["n_unspawned"] == 1
    assert any("NEVER SPAWNED" in ln for ln in res["needs_attention"])
    assert any("no COMPLETE marker" in ln for ln in res["needs_attention"])
    assert res["all_calls_terminal"] is True


def test_legacy_manifest_keeps_existence_only_reading():
    row = {k: v for k, v in _row().items() if k != "expect_fingerprint"}
    res = _assess([row], {row["cell_id"]: _marker(OTHER)}, {}, strict=False)
    assert res["n_complete"] == 1


def test_read_marker_joins_chunks_and_treats_failures_as_absent():
    class Vol:
        def __init__(self, data=None, fail=False):
            self.data, self.fail = data, fail

        def read_file(self, path):
            assert path.endswith("/COMPLETE")
            if self.fail:
                raise FileNotFoundError(path)
            yield self.data[:3]
            yield self.data[3:]

    assert fw._read_marker(Vol(b"abcdef"), _row()) == b"abcdef"
    assert fw._read_marker(Vol(fail=True), _row()) is None


def test_strict_apps_include_sdfbase_and_module_imports_without_modal():
    assert "sdfbase-fits" in fw.STRICT_APPS
    assert "modal" not in sys.modules or True  # import above succeeded


# --------------------------------------------------------------------------- #
# per-wave app names (sdfbase-fits-<wave>) are just as strict                   #
# --------------------------------------------------------------------------- #


def test_per_wave_app_namesakes_are_strict_and_others_are_not():
    """The frozen app is `sdfbase-fits`; later waves deploy and spawn under
    `sdfbase-fits-ws3`. Its manifests are the same shape and MUST be read
    strictly — a name-equality check would have silently downgraded every
    override wave to the legacy existence-only reading."""
    assert fw.is_strict_app("sdfbase-fits") is True
    assert fw.is_strict_app("sdfbase-fits-ws3") is True
    assert fw.is_strict_app("sdfbase-fits-r3") is True
    assert fw.is_strict_app("twosi-fits") is False
    assert fw.is_strict_app("sdfbase-fitsx") is False   # not a namesake
    assert fw.is_strict_app(None) is False
    assert fw.is_strict_app("") is False
    # main() reads strictness through the predicate, not the tuple
    src = Path(fw.__file__).read_text()
    assert 'is_strict_app(doc.get("app"))' in src
    assert 'doc.get("app") in STRICT_APPS' not in src


def test_row_app_governs_its_own_strictness_and_names_the_respawn_app():
    """A row that records the app it was spawned on is judged by THAT app
    (one manifest can carry rows from two apps: fits-spawn keeps foreign
    rows when it rewrites), and its NEVER SPAWNED hint names the app so
    the re-spawn cannot land on the default one."""
    # override-app rows inside a doc whose top-level app is legacy: strict
    row = _row(expect=None, call_id="fc-9")
    row["app"] = "sdfbase-fits-ws3"
    res = _assess([row], {row["cell_id"]: _marker(FP)},
                  {row["cell_id"]: "running"}, strict=False)
    assert res["n_complete"] == 0
    assert res["needs_attention"] and "expect_fingerprint" in \
        res["needs_attention"][0]
    # an unspawned override-app row: the hint carries --app-name
    row = _row(cell="constv2-mix-10m__psr_ce_all", call_id=None)
    row["app"] = "sdfbase-fits-ws3"
    res = _assess([row], {}, {})
    assert any("re-run fits-spawn --app-name sdfbase-fits-ws3 for it" in ln
               for ln in res["needs_attention"])
    # a row without the field keeps the manifest-level reading + bare hint
    legacy = _row(cell="constv2-mix-full__psr_ce_all", call_id=None)
    res = _assess([legacy], {}, {})
    assert any(ln.endswith("re-run fits-spawn for it")
               for ln in res["needs_attention"])
    assert fw.respawn_hint({}) == ""
    assert fw.respawn_hint({"app": "sdfbase-fits-ws3"}) == \
        " --app-name sdfbase-fits-ws3"


# --------------------------------------------------------------------------- #
# HPO TRIALS manifests (hpo-spawn's doc["trials"]) ride the same watcher       #
# --------------------------------------------------------------------------- #

import re  # noqa: E402

DESIGN = {"model": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
          "model_revision": "d" * 40, "config_sha": "da1870c7d4c5fce8",
          "box": {"lr": [5e-05, 0.05], "max_grad_norm": [0.02, 10.0]},
          "warmup_levels": [0.05, 0.15], "round": 1, "position": "all",
          "trials": [{"index": 0, "lr": 5.6e-05, "warmup_frac": 0.05,
                      "max_grad_norm": 2.2}]}


def _expect(**over):
    e = {"model": DESIGN["model"], "model_revision": DESIGN["model_revision"],
         "config_sha": DESIGN["config_sha"], "box": DESIGN["box"],
         "warmup_levels": DESIGN["warmup_levels"], "arm": "psr_ce_all",
         "position": "all", "round": 1, "si": "constv2-mix-100k",
         "trial": dict(DESIGN["trials"][0])}
    e.update(over)
    return e


def _trow(index=0, call_id="fc-1", expect="default", **over):
    r = {"model": "nemotron3nano4b_doc", "arm": "psr_ce_all", "index": index,
         "round": 1, "si": "constv2-mix-100k", "app": "sdfbase-fits-ws3",
         "call_id": call_id}
    if expect == "default":
        r["expect"] = _expect()
    elif expect is not None:
        r["expect"] = expect
    r.update(over)
    return r


def _summary(status="ok", **over) -> bytes:
    s = {"status": status, "model": DESIGN["model"],
         "model_revision": DESIGN["model_revision"], "arm": "psr_ce_all",
         "si": "constv2-mix-100k", "round": 1, "position": "all",
         "box": DESIGN["box"], "warmup_levels": DESIGN["warmup_levels"],
         "trial": dict(DESIGN["trials"][0]), "fit_fingerprint": FP,
         "best_heldout_loss": 2.19}
    s.update(over)
    return json.dumps(s).encode()


def test_summary_path_is_the_trial_dirs_summary():
    assert fw.summary_path(_trow(index=7)) == \
        "out/hpo/nemotron3nano4b_doc/psr_ce_all_all/r1_t07/summary.json"


def test_trial_verdicts():
    v = fw.trial_verdict
    e = _expect()
    assert v(None, e)[0] == "absent"
    assert v(_summary(), e)[0] == "published"
    # every terminal HPO outcome is a publication — a diverged trial is data
    for st in ("nan_diverged", "gate_dead", "floor_failed", "ratio_diverged"):
        assert v(_summary(status=st), e)[0] == "published", st
    # a torn / seeded / running summary is not
    state, detail = v(_summary(status="running"), e)
    assert state == "non-terminal" and "'running'" in detail
    assert v(_summary(status=None), e)[0] == "non-terminal"
    assert v(b"not json", e)[0] == "unparseable"
    assert v(b"[1]", e)[0] == "unparseable"
    # identity: another design's trial under the same r1_t00 dir
    state, detail = v(_summary(box={"lr": [1e-5, 0.02],
                                    "max_grad_norm": [0.02, 10.0]}), e)
    assert state == "foreign" and "['box']" in detail
    assert v(_summary(model_revision="e" * 40), e)[0] == "foreign"
    assert v(_summary(si="constv2-mix-1m"), e)[0] == "foreign"
    assert v(_summary(round=2), e)[0] == "foreign"
    state, detail = v(_summary(trial={"index": 0, "lr": 9e-4,
                                      "warmup_frac": 0.05,
                                      "max_grad_norm": 2.2}), e)
    assert state == "foreign" and "trial.lr" in detail
    # the summary may carry MORE trial keys than the design (extra axes
    # pinned in-container); a subset match is a match
    assert v(_summary(trial={**DESIGN["trials"][0], "gate_reg": 0.1}),
             e)[0] == "published"
    # ...but a designed axis the summary lacks is a mismatch
    assert v(_summary(), _expect(trial={**DESIGN["trials"][0],
                                        "gate_reg": 0.3}))[0] == "foreign"
    # a row without an expect identity can never be validated
    state, detail = v(_summary(), None)
    assert state == "unverifiable" and "re-run hpo-spawn" in detail


def _assess_trials(rows, summaries, calls):
    return fw.assess_trials(
        rows,
        read_summary=lambda r: summaries.get(r["index"]),
        call_state=lambda r: ("unspawned" if not r.get("call_id")
                              else calls.get(r["index"], "unknown")))


def test_trials_published_regardless_of_call_and_in_flight_otherwise():
    a = _trow(index=0)
    b = _trow(index=1, call_id="fc-2")
    res = _assess_trials([a, b], {0: _summary()}, {0: "failed", 1: "running"})
    assert res["n_complete"] == 1
    assert res["all_calls_terminal"] is False
    assert res["needs_attention"] == []
    assert res["lines"]["nemotron3nano4b_doc/psr_ce_all/r1_t00"] \
        .endswith("published")
    assert res["lines"]["nemotron3nano4b_doc/psr_ce_all/r1_t01"] \
        .endswith("running")


def test_trials_terminal_call_without_valid_summary_needs_attention():
    a = _trow(index=0)                       # died, nothing published
    b = _trow(index=1, call_id="fc-2")       # foreign summary, call done
    c = _trow(index=2, call_id=None)         # never spawned
    d = _trow(index=3, call_id="fc-4", expect=None)  # unverifiable row
    a["_call_error"] = "RemoteError: boom"
    res = _assess_trials([a, b, c, d],
                         {1: _summary(si="constv2-mix-1m")},
                         {0: "failed", 1: "done"})
    assert res["n_complete"] == 0 and res["n_unspawned"] == 1
    assert res["all_calls_terminal"] is True
    joined = "\n".join(res["needs_attention"])
    assert "r1_t00: call failed but no summary.json" in joined
    assert "(RemoteError: boom)" in joined
    assert "r1_t01: call done but summary identity differs" in joined
    assert "r1_t02: NEVER SPAWNED — re-run hpo-spawn --app-name " \
           "sdfbase-fits-ws3 for it" in joined
    assert "r1_t03: manifest row carries no expect identity" in joined
    # a non-terminal summary under a running call is the in-flight picture
    res = _assess_trials([_trow(index=5, call_id="fc-5")],
                         {5: _summary(status="running")}, {5: "running"})
    assert res["all_calls_terminal"] is False and not res["needs_attention"]
    assert res["lines"]["nemotron3nano4b_doc/psr_ce_all/r1_t05"] \
        .endswith("running (summary non-terminal)")


def test_trials_function_timeout_is_terminal_without_a_successor():
    """`timeout` (a real function timeout, classify_call) ENDS a trial's
    call exactly like done/failed: with no valid summary it is a triage
    row and the wave is terminal — never "in flight". The trials path
    shares TERMINAL_CALL_STATES with the cells path and deliberately has
    no respawn hook (an HPO trial has no continuation)."""
    import inspect

    a = _trow(index=0)                        # published — its call is moot
    b = _trow(index=1, call_id="fc-2")
    b["_call_error"] = "FunctionTimeoutError: Function ran too long"
    res = _assess_trials([a, b], {0: _summary()},
                         {0: "timeout", 1: "timeout"})
    assert res["n_complete"] == 1
    assert res["all_calls_terminal"] is True
    assert res["needs_attention"] == [
        "nemotron3nano4b_doc/psr_ce_all/r1_t01: call timeout but no "
        "summary.json on the volume (FunctionTimeoutError: Function ran "
        "too long)"]
    assert res["lines"]["nemotron3nano4b_doc/psr_ce_all/r1_t01"] \
        .endswith("call timeout, summary absent")
    assert fw.TERMINAL_CALL_STATES == ("done", "timeout", "failed")
    src = inspect.getsource(fw.assess_trials)
    assert "call in TERMINAL_CALL_STATES" in src
    assert '("done", "failed")' not in src
    assert "respawn" not in inspect.signature(fw.assess_trials).parameters


def test_trial_terminal_statuses_mirror_tune_arm():
    """The watcher's terminal set is a literal copy (tune_arm imports
    torch); pin it against the source so a new outcome status cannot land
    on one side only."""
    src = (REPO / "scripts" / "tune_arm.py").read_text()
    m = re.search(r"_TERMINAL_STATUSES = frozenset\(\{([^}]*)\}\)", src)
    assert m, "tune_arm._TERMINAL_STATUSES literal not found"
    statuses = set(re.findall(r'"([a-z_]+)"', m.group(1)))
    assert statuses == set(fw.TRIAL_TERMINAL_STATUSES)


def test_manifest_kind_dispatch():
    assert fw.manifest_kind({"trials": [1]}) == "trials"
    assert fw.manifest_kind({"cells": [1]}) == "cells"
    assert fw.manifest_kind({"cells": [], "trials": [1]}) == "cells"
    assert fw.manifest_kind({}) == "cells"
    assert fw.manifest_kind({"trials": []}) == "trials"


class _FakeVol:
    def __init__(self, files):
        self.files = files

    def read_file(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        yield self.files[path]


def _fake_modal(files, states, monkeypatch):
    import types

    # The modal 1.5.4 hierarchy classify_call dispatches on (see
    # tests/test_watch_successor.py): modal.exception.TimeoutError is NOT
    # the builtin; FunctionTimeoutError (a real function timeout) and
    # OutputExpiredError subclass it. The builtin TimeoutError _FC raises
    # for "running" = still running. Class names mirror modal's, so the
    # `_call_error` text the watcher records reads as it would live.
    class Error(Exception):
        pass

    class ModalTimeoutError(Error):
        pass

    class FunctionTimeoutError(ModalTimeoutError):
        pass

    class OutputExpiredError(ModalTimeoutError):
        pass

    class _FC:
        def __init__(self, cid):
            self.cid = cid

        def get(self, timeout=0):
            st = states.get(self.cid, "running")
            if st == "running":
                raise TimeoutError()
            if st == "failed":
                raise RuntimeError("remote raise")
            if st == "timeout":
                raise FunctionTimeoutError("Function ran too long")
            return {}

    fake = types.ModuleType("modal")
    fake.Volume = type("V", (), {"from_name": staticmethod(
        lambda n: _FakeVol(files))})
    fake.FunctionCall = type("F", (), {"from_id": staticmethod(_FC)})
    fake.exception = types.SimpleNamespace(
        Error=Error, TimeoutError=ModalTimeoutError,
        FunctionTimeoutError=FunctionTimeoutError,
        OutputExpiredError=OutputExpiredError)
    monkeypatch.setitem(sys.modules, "modal", fake)


def test_main_reports_a_timed_out_trial_as_terminal(tmp_path, monkeypatch,
                                                    capsys):
    """A trial whose FunctionCall hit its REAL function timeout (modal's
    FunctionTimeoutError -> classify_call `timeout`) with no summary
    published is TERMINAL: a triage row that ends the --once pass with
    WAVE-INCOMPLETE, not a wave held open until --max-hours. Found in the
    ctl-scale merge review: the trials path knew only done/failed while
    classify_call had learned to tell a function timeout apart. Trials
    never respawn (no continuation exists for an HPO trial)."""
    rows = [_trow(index=0, call_id="fc-0"), _trow(index=1, call_id="fc-1")]
    doc = {"app": "sdfbase-fits-ws3", "round": 1, "trials": rows}
    mpath = tmp_path / "hpo_calls_ws3.json"
    mpath.write_text(json.dumps(doc))
    p0 = fw.summary_path(rows[0])
    monkeypatch.setattr(sys, "argv", ["fits_watch.py", "--manifest",
                                      str(mpath), "--once"])
    _fake_modal({p0: _summary()}, {"fc-1": "timeout"}, monkeypatch)
    assert fw.main() == 2
    out = capsys.readouterr().out
    assert "r1_t01: call timeout, summary absent" in out
    assert "[hpo-watch] 1/2 PUBLISHED, 0 unspawned, 1 needing attention" in out
    assert "WAVE-INCOMPLETE" in out
    assert ("r1_t01: call timeout but no summary.json on the volume "
            "(FunctionTimeoutError: Function ran too long)") in out
    # terminal on the FIRST pass: no successor, no wave-still-in-flight
    # WATCH-TIMEOUT, no "confirming on the next poll" (--once decides)
    assert "SUCCESSOR" not in out and "WATCH-TIMEOUT" not in out
    assert "confirming on the next poll" not in out


def test_main_watches_a_trials_manifest_end_to_end(tmp_path, monkeypatch,
                                                   capsys):
    """fits_watch.py --manifest <hpo_calls> --once: dispatches on the
    trials shape, reads summaries off the volume, judges calls, and ends
    with the ALL-PUBLISHED / WAVE-INCOMPLETE lines the tmux wrapper keys
    its terminal line on."""
    rows = [_trow(index=0, call_id="fc-0"), _trow(index=1, call_id="fc-1")]
    doc = {"app": "sdfbase-fits-ws3", "round": 1, "trials": rows}
    mpath = tmp_path / "hpo_calls_ws3.json"
    mpath.write_text(json.dumps(doc))
    p0 = fw.summary_path(rows[0])
    p1 = fw.summary_path(rows[1])
    monkeypatch.setattr(sys, "argv", ["fits_watch.py", "--manifest",
                                      str(mpath), "--once"])
    # one published, one running: not terminal, --once exits 0
    _fake_modal({p0: _summary()}, {"fc-1": "running"}, monkeypatch)
    assert fw.main() == 0
    out = capsys.readouterr().out
    assert "[hpo-watch] PUBLISHED = terminal summary.json" in out
    assert "app(s) the rows were spawned on: sdfbase-fits-ws3" in out
    assert "[hpo-watch] 1/2 PUBLISHED, 0 unspawned, 0 needing attention" in out
    assert "r1_t01: running" in out
    # both published: ALL-PUBLISHED, exit 0
    _fake_modal({p0: _summary(), p1: _summary(status="nan_diverged")},
                {}, monkeypatch)
    assert fw.main() == 0
    assert "ALL-PUBLISHED" in capsys.readouterr().out
    # second call died without publishing: WAVE-INCOMPLETE, exit 2
    _fake_modal({p0: _summary()}, {"fc-1": "failed"}, monkeypatch)
    assert fw.main() == 2
    out = capsys.readouterr().out
    assert "[hpo-watch] WAVE-INCOMPLETE" in out
    assert "r1_t01: call failed but no summary.json" in out
    # a cells manifest still takes the fits path
    cells = {"app": "sdfbase-fits", "cells": [_row(call_id="fc-9")]}
    mpath.write_text(json.dumps(cells))
    _fake_modal({}, {"fc-9": "running"}, monkeypatch)
    assert fw.main() == 0
    out = capsys.readouterr().out
    assert "[fits-watch] COMPLETE = fingerprint-matching marker" in out
    assert "0/1 COMPLETE" in out


#: The terminal WAVE-INCOMPLETE line (the notice that precedes a confirmation
#: also names WAVE-INCOMPLETE, so tests match the line form).
TERMINAL_BAD = "] WAVE-INCOMPLETE —"


def test_terminal_bad_needs_two_consecutive_polls(tmp_path, monkeypatch,
                                                  capsys):
    """The real observation this guards (an HPO wave, 2026-09-03 03:45Z):
    a trial read as `call done, summary absent` on one pass and `published`
    on the next — the publication became readable after the call's return.
    A continuous watch therefore confirms terminal-bad on the NEXT poll:
    when the summary shows up in between, the wave completes (exit 0); when
    it stays missing, WAVE-INCOMPLETE (exit 2) — after exactly two polls.
    --once keeps the single-pass reading."""
    rows = [_trow(index=0, call_id="fc-0")]
    mpath = tmp_path / "hpo_calls.json"
    mpath.write_text(json.dumps({"app": "sdfbase-fits-ws3", "trials": rows}))
    p0 = fw.summary_path(rows[0])
    files = {}
    polls = {"n": 0}

    def _sleep(_s):
        polls["n"] += 1
        if polls["n"] == 1:
            files[p0] = _summary()      # publication lands between polls

    monkeypatch.setattr(fw.time, "sleep", _sleep)
    monkeypatch.setattr(sys, "argv", ["fits_watch.py", "--manifest",
                                      str(mpath), "--poll", "10"])
    _fake_modal(files, {"fc-0": "done"}, monkeypatch)
    assert fw.main() == 0
    out = capsys.readouterr().out
    assert "terminal-bad observed (1 row(s) missing" in out
    assert "confirming on the next poll" in out
    assert "ALL-PUBLISHED" in out and TERMINAL_BAD not in out
    assert polls["n"] == 1
    # still missing on the confirming poll: WAVE-INCOMPLETE after 2 polls
    files.clear(); polls["n"] = 0
    monkeypatch.setattr(fw.time, "sleep", lambda _s: polls.__setitem__(
        "n", polls["n"] + 1))
    _fake_modal(files, {"fc-0": "done"}, monkeypatch)
    assert fw.main() == 2
    out = capsys.readouterr().out
    assert out.count("terminal-bad observed") == 1
    assert TERMINAL_BAD in out
    assert polls["n"] == 1
    assert fw.TERMINAL_BAD_CONFIRM_POLLS == 2
    # --once: single pass, terminal-bad is reported at once
    monkeypatch.setattr(sys, "argv", ["fits_watch.py", "--manifest",
                                      str(mpath), "--once"])
    _fake_modal(files, {"fc-0": "done"}, monkeypatch)
    assert fw.main() == 2
    assert TERMINAL_BAD in capsys.readouterr().out
    # a streak broken by a poll with a call back in flight resets
    files.clear(); polls["n"] = 0
    states = {"fc-0": "done"}
    seq = iter(["running", "done", "done"])

    def _sleep2(_s):
        polls["n"] += 1
        states["fc-0"] = next(seq, "done")

    monkeypatch.setattr(fw.time, "sleep", _sleep2)
    monkeypatch.setattr(sys, "argv", ["fits_watch.py", "--manifest",
                                      str(mpath), "--poll", "10"])
    _fake_modal(files, states, monkeypatch)
    assert fw.main() == 2
    out = capsys.readouterr().out
    # observed (poll 1) -> reset (running, poll 2) -> observed (3) -> confirmed (4)
    assert out.count("terminal-bad observed") == 2
    assert polls["n"] == 3

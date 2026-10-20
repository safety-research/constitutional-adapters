"""The fixes to scripts/modal/seedrep_petri_drive.py --
one test block per finding, each with the RE-VALIDATED repro as its known-bad twin:

P1-1  a tunnel record found at a discovery location (--serving-vol) was never bound to this drive's wave / W (a live
      tunnel of another wave at the same W passed view/stage/https/under-/vol/seedrep/), and --serving-vol could REPLACE
      the record's validated serving dir; now ``control_key`` is a REQUIRED field compared exactly with the drive's own
      six-axis key, the attempt root must lie under /vol/seedrep/<W7>/, and --serving-vol is discovery only (its
      normalised value must equal the record's, which is what the drive uses).
P2-2  --resume-from validated and copied tag by tag, so a later missing / occupied / failing tag left an earlier tag's
      cells/ imported with NO RESUMED_FROM.json; now every source and destination is preflighted before any mutation,
      the stores are staged WITH their provenance in hidden sibling roots, published by rename, and rolled back on any
      later failure -- no final destination exists afterwards.
P2-3  the overflow-rerun pass broke the three-attempt loop on ANY rc (one transient failure ended the leg, skipping the
      429 throttle) and wrote STOP regardless of verdict["release"]; now only rc 0 / 3 end the loop on every pass, and
      STOP needs the verdict to release -- or the explicit, named --teardown-after-failure.
P2-4  --width was computed but the ceiling was checked with the static defaults (--view s1 --width 13 -> 8 x 13 = 104
      while the guard reported 84); now --width only NARROWS a view below its planned width, and the ceiling is summed
      over the EFFECTIVE widths.
Plus one PARTIAL item -- the producer does not yet emit the attempt fields: a record missing EITHER
field is refused by name, before any volume read. Fixtures: the review-fixes module's minimum tunnel record, and stores
the Petri runner itself wrote (test_petri_store_shape's world). Synthetic throughout."""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "scripts/modal/seedrep_petri_drive.py"
sys.path.insert(0, str(REPO / "scripts"))

from test_petri_store_shape import world  # noqa: E402,F401
from test_seedrep_petri_drive import _load  # noqa: E402
from test_seedrep_petri_drive_review_fixes import (  # noqa: E402
    ATT_A, ATT_B, ATTEMPT, ATTEMPT_ROOT, MINIMUM_TUNNEL, TUNNEL_KEY, W, W7, WAVE_A, WAVE_B, _complete_store, _legs, _real_reconcile)

W7X = "abcdef0"          # another W


@pytest.fixture(scope="module")
def d():
    return _load()


def _problems(d, rec, *, key=TUNNEL_KEY, w7=W7, view="s1", stage="base"):
    return d.tunnel_record_problems(rec, view=view, stage=stage, expected_control_key=key, w7=w7)


# --------------------------------------------------------------------------- P1-1: the record is BOUND to this drive's six axes; --serving-vol only discovers

def test_p1_1_a_tunnel_record_of_another_wave_or_w_is_refused_by_name_and_serving_vol_never_replaces_the_record_s_dir(d):
    key = d.control_key(W7, WAVE_A, "petri", "s1", "base", "0")
    assert key == TUNNEL_KEY == MINIMUM_TUNNEL["control_key"] == f"{W7}/{WAVE_A}/petri/s1/base/0"
    assert "control_key" in d.TUNNEL_REQUIRED_KEYS and _problems(d, MINIMUM_TUNNEL) == []
    # RE-VALIDATED repro 1: a LIVE tunnel of ANOTHER WAVE at the same W -- same view, stage and shard, its root under this W7 -- passed
    # every field the old check read (view, stage, https, under /vol/seedrep/) and bound this wave to a stale proxy; now its own key
    # differs from this drive's and it is refused by name
    other_wave_key = d.control_key(W7, WAVE_B, "petri", "s1", "base", "0")
    stale = {**MINIMUM_TUNNEL, "control_key": other_wave_key}
    assert stale["view"] == "s1" and stale["stage"] == "base" and stale["url"].startswith("https://") and stale["attempt_root_on_volume"].startswith("/vol/seedrep/")
    assert _problems(d, stale) == [f"control_key {other_wave_key!r} != this drive's {key!r} (W7/wave_uuid/leg/view/stage/shard) -- another wave's or W's proxy"]
    # RE-VALIDATED repro 2: ANOTHER W's proxy (its key, root and serving dir all under the other W7) -- refused on every axis it touches
    other_w = {**MINIMUM_TUNNEL, "control_key": d.control_key(W7X, WAVE_A, "petri", "s1", "base", "0"),
               "attempt_root_on_volume": ATTEMPT_ROOT.replace(W7, W7X, 1),
               "serving_dir_on_volume": MINIMUM_TUNNEL["serving_dir_on_volume"].replace(W7, W7X, 1)}
    assert _problems(d, other_w) == [
        f"control_key {other_w['control_key']!r} != this drive's {key!r} (W7/wave_uuid/leg/view/stage/shard) -- another wave's or W's proxy",
        f"serving_dir_on_volume {other_w['serving_dir_on_volume']!r} is not under /vol/seedrep/{W7}/",
        f"attempt_root_on_volume {other_w['attempt_root_on_volume']!r} is not under /vol/seedrep/{W7}/"]
    # KNOWN-BAD 3: the key is right but the ROOT lies under another W's tree (a record assembled from two proxies) -- the root check alone
    mixed = {**MINIMUM_TUNNEL, "attempt_root_on_volume": ATTEMPT_ROOT.replace(W7, W7X, 1)}
    assert _problems(d, mixed) == [
        f"attempt_root_on_volume {mixed['attempt_root_on_volume']!r} is not under /vol/seedrep/{W7}/",
        f"serving_dir_on_volume {MINIMUM_TUNNEL['serving_dir_on_volume']!r} is not under the attempt's exp tree {mixed['attempt_root_on_volume']}/exp/"]
    # KNOWN-BAD 4: yesterday's record (no control_key at all) is refused by name; 5: a key of another stage / shard / leg / view;
    # 6: a root that only LOOKS under this W7 through a '..' component
    assert _problems(d, {k: v for k, v in MINIMUM_TUNNEL.items() if k != "control_key"}) == ["missing control_key"]
    for other in (d.control_key(W7, WAVE_A, "petri", "s1", "petri-rerun", "0"), d.control_key(W7, WAVE_A, "petri", "s1", "base", "1"),
                  d.control_key(W7, WAVE_A, "static_pasttense", "s1", "base", "0"), d.control_key(W7, WAVE_A, "petri", "s2", "base", "0")):
        assert _problems(d, {**MINIMUM_TUNNEL, "control_key": other})[0].startswith(f"control_key {other!r} != this drive's {key!r}")
    dotdot = f"/vol/seedrep/{W7}/../{W7X}/exp/seedrep_battery_modal/petri/s1/0/attempts/{ATTEMPT}"
    assert dotdot.startswith(f"/vol/seedrep/{W7}/")
    assert f"attempt_root_on_volume {dotdot!r} carries a '..' component" in _problems(d, {**MINIMUM_TUNNEL, "attempt_root_on_volume": dotdot})
    # the check refuses a CALLER that hands it another W's key, the NAMED key (seven parts), or a malformed W7
    with pytest.raises(ValueError, match="not this drive's six-axis key"):
        _problems(d, MINIMUM_TUNNEL, key=d.control_key(W7X, WAVE_A, "petri", "s1", "base", "0"))
    with pytest.raises(ValueError, match="not this drive's six-axis key"):
        _problems(d, MINIMUM_TUNNEL, key=TUNNEL_KEY + "/modal_tunnel.json")
    with pytest.raises(ValueError, match="needs W7"):
        _problems(d, MINIMUM_TUNNEL, w7="0c19")
    # --serving-vol is DISCOVERY only: the serving dir the drive uses is the RECORD's; an override must equal it (normalised) or is refused
    sd = MINIMUM_TUNNEL["serving_dir_on_volume"]
    assert d.resolve_serving_dir(MINIMUM_TUNNEL) == sd and d.resolve_serving_dir(MINIMUM_TUNNEL, "") == sd
    assert d.resolve_serving_dir(MINIMUM_TUNNEL, sd + "/") == sd
    assert d.resolve_serving_dir(MINIMUM_TUNNEL, sd.replace("/petri/", "//petri/./", 1)) == sd
    # RE-VALIDATED repro 3: the record is CURRENT and valid and --serving-vol names ANOTHER attempt's serve dir -- the old line
    # `args.serving_vol or tunnel["serving_dir_on_volume"]` took the override and read THAT endpoint.json; now refused by name
    other_attempt_sd = sd.replace(ATTEMPT, "ta-other-attempt", 1)
    assert other_attempt_sd != sd and (other_attempt_sd or sd) == other_attempt_sd          # what the old expression evaluated to
    with pytest.raises(SystemExit, match="--serving-vol .* is a discovery location only and never replaces the record's serving dir"):
        d.resolve_serving_dir(MINIMUM_TUNNEL, other_attempt_sd)
    with pytest.raises(SystemExit, match="discovery location only"):
        d.resolve_serving_dir(MINIMUM_TUNNEL, f"/vol/seedrep/{W7}/exp/serve")
    with pytest.raises(SystemExit, match="names no serving_dir_on_volume"):
        d.resolve_serving_dir({}, "")
    # main(): the record is bound with key_for("") + this W7 before any volume read; the serving dir comes from the record through
    # resolve_serving_dir; --serving-vol still names where the volume COPY of the record is looked for; the override line is gone
    main = SRC.read_text().split("def main(")[1]
    assert 'tunnel_record_problems(tunnel, view=args.view, stage=stage, expected_control_key=key_for(""), w7=w7)' in main
    assert "serving_vol = resolve_serving_dir(tunnel, args.serving_vol)" in main and 'args.serving_vol or str(tunnel["serving_dir_on_volume"])' not in main
    assert "vol_read(vol, f\"{args.serving_vol.rstrip('/')}/{TUNNEL_FILE}\")" in main
    assert main.index("problems = tunnel_record_problems(") < main.index("serving_vol = resolve_serving_dir(") < main.index('vol_read(vol, f"{serving_vol')
    assert 'help="DISCOVERY only' in SRC.read_text().split('"--serving-vol"')[1]


# --------------------------------------------------------------------------- PARTIAL (the producer): the consumer stays strict

def test_astra_p2_3_a_producer_record_missing_either_attempt_field_is_refused_by_name_before_any_volume_read(d):
    for missing in ("attempt_root_on_volume", "attempt_id"):
        for absent in ({k: v for k, v in MINIMUM_TUNNEL.items() if k != missing}, {**MINIMUM_TUNNEL, missing: ""}, {**MINIMUM_TUNNEL, missing: None}):
            assert _problems(d, absent) == [f"missing {missing}"]
    assert _problems(d, {k: v for k, v in MINIMUM_TUNNEL.items() if k not in ("attempt_root_on_volume", "attempt_id")}) == \
        ["missing attempt_root_on_volume", "missing attempt_id"]
    assert tuple(k for k in d.TUNNEL_REQUIRED_KEYS if k.startswith("attempt")) == ("attempt_root_on_volume", "attempt_id")
    # a record with the fields but an id that is not the root's last component, or a root outside this W, is not "present"
    assert _problems(d, {**MINIMUM_TUNNEL, "attempt_id": "ta-other"}) == [f"attempt_id 'ta-other' != the attempt root's last component {ATTEMPT!r}"]
    # main(): the refusal precedes EVERY read off the volume (endpoint.json, routing table, cells_served)
    main = SRC.read_text().split("def main(")[1]
    refusal = main.index('raise DriveRefused(f"tunnel record at {tkey}: "')
    assert main.index("problems = tunnel_record_problems(") < refusal < main.index('vol_read(vol, f"{serving_vol') < main.index("paths = attempt_paths(tunnel, ep, exp_root)")


# --------------------------------------------------------------------------- P2-2: --resume-from is atomic

def _wave_a(d, world, tmp_path, home):
    """Wave A: three runner-written complete Arditi stores + the drive record main() writes, in A's attempt state dir."""
    cell, _ = _complete_store(world, tmp_path / "seed", tmp_path / "scratch0")
    runs = _legs(d, cell)
    roots_a = {r.tag: d.out_root_for(home, W7, d.wave_uuid8(WAVE_A), ATT_A, r.tag) for r in runs}
    state_a = d.attempt_state_dir(d.state_dir_for(home, W7, d.wave_uuid8(WAVE_A), "arditi", ""), ATT_A)
    state_a.mkdir(parents=True)
    for i, r in enumerate(runs):
        _complete_store(world, roots_a[r.tag], tmp_path / f"scratchA{i}")
    rec_a = {"schema": "seedrep_petri_drive/1", "drive_id": "DA", "view": "arditi", "W": W, "W7": W7, "wave_uuid": WAVE_A, "attempt_id": ATT_A,
             "out_roots": {t: str(p) for t, p in roots_a.items()}, "started_at": "2026-09-18T00:00:00Z"}
    (state_a / "drive_DA.json").write_text(json.dumps(rec_a))
    return cell, runs, roots_a, state_a, rec_a


def test_p2_2_resume_from_is_atomic_a_failing_later_tag_leaves_no_destination_and_no_staging_behind(d, world, tmp_path):
    home = tmp_path / "home"
    cell, runs, roots_a, state_a, rec_a = _wave_a(d, world, tmp_path, home)
    rk = cell.petri_run_key
    src_bytes = {t: (p / "cells" / rk / "results.json").read_bytes() for t, p in roots_a.items()}
    roots_b = {r.tag: d.out_root_for(home, W7, d.wave_uuid8(WAVE_B), ATT_B, r.tag) for r in runs}
    parent = next(iter(roots_b.values())).parent
    t1, t2, t3 = (r.tag for r in runs)
    kw = dict(w7=W7, wave_uuid=WAVE_B, attempt_id=ATT_B, view="arditi", drive_id="DB")
    baseline = sorted(p.name for p in parent.iterdir())               # wave A's three roots share the parent; nothing else may appear
    assert baseline == sorted(p.name for p in roots_a.values())

    def entries():
        return sorted(p.name for p in parent.iterdir())

    def nothing_landed():
        assert not any(p.exists() for p in roots_b.values()), "a final destination exists after a refused / failed import"
        assert entries() == baseline, f"staging or a partial root left behind: {sorted(set(entries()) - set(baseline))}"
        assert {t: (p / "cells" / rk / "results.json").read_bytes() for t, p in roots_a.items()} == src_bytes      # the sources untouched

    # RE-VALIDATED repro: the SECOND tag's mapping is missing from the prior record -- the old per-tag loop had already copied s1's
    # cells/ (s1_copied=True) and aborted before RESUMED_FROM.json (s1_provenance=False); now the preflight refuses before any mutation
    state_m = tmp_path / "state_missing"
    state_m.mkdir()
    (state_m / "drive_DA.json").write_text(json.dumps({**rec_a, "out_roots": {t: p for t, p in rec_a["out_roots"].items() if t != t2}}))
    with pytest.raises(SystemExit, match=f"no out root recorded for {t2}"):
        d.import_prior_stores(state_m, roots_b, **kw)
    nothing_landed()
    # MISSING on disk: the mapping is recorded but the prior root is gone
    state_g = tmp_path / "state_gone"
    state_g.mkdir()
    (state_g / "drive_DA.json").write_text(json.dumps({**rec_a, "out_roots": {**rec_a["out_roots"], t2: str(tmp_path / "vanished")}}))
    with pytest.raises(SystemExit, match=f"the recorded out root for {t2}, .*vanished, is not a directory"):
        d.import_prior_stores(state_g, roots_b, **kw)
    nothing_landed()
    # OCCUPIED: the SECOND destination already exists (holding only a cells.json) -> refused; the first and third are never created
    roots_b[t2].mkdir(parents=True)
    (roots_b[t2] / "cells.json").write_text("{}")
    with pytest.raises(SystemExit, match="already exists -- an import never writes over a store"):
        d.import_prior_stores(state_a, roots_b, **kw)
    assert not roots_b[t1].exists() and not roots_b[t3].exists() and entries() == sorted(baseline + [roots_b[t2].name])
    shutil.rmtree(roots_b[t2])
    nothing_landed()
    # RAISES while COPYING the second tag -> the first tag's staged copy is removed; nothing was published
    copies = []

    def copytree_boom(src, dst):
        copies.append(Path(src))
        if len(copies) == 2:
            raise OSError("disk full")
        return shutil.copytree(src, dst)
    with pytest.raises(SystemExit, match="import failed \\(OSError: disk full\\); rolled back -- no root of this attempt was left behind"):
        d.import_prior_stores(state_a, roots_b, copytree=copytree_boom, **kw)
    assert copies == [roots_a[t1] / "cells", roots_a[t2] / "cells"]
    nothing_landed()
    # RAISES while PUBLISHING the second root -> the FIRST, already published, is un-published; provenance had travelled INSIDE the staged root
    published = []

    def publish_boom(staged, dst):
        assert (staged / d.RESUMED_FROM_FILE).is_file() and (staged / "cells" / rk / "results.json").is_file()
        assert staged == d.import_staging_dir(dst, "DB") and staged.name.startswith(".") and not dst.exists()
        published.append(dst)
        if len(published) == 2:
            raise OSError("rename refused")
        os.rename(staged, dst)
    with pytest.raises(SystemExit, match="import failed \\(OSError: rename refused\\); rolled back"):
        d.import_prior_stores(state_a, roots_b, publish=publish_boom, **kw)
    assert published == [roots_b[t1], roots_b[t2]]
    nothing_landed()
    # a stale staged sibling of THIS drive id (an import that died mid-way) is refused by name, never reused
    stale = d.import_staging_dir(roots_b[t1], "DB")
    stale.mkdir(parents=True)
    with pytest.raises(SystemExit, match="already exists -- an earlier import into .* did not finish"):
        d.import_prior_stores(state_a, roots_b, **kw)
    assert not any(p.exists() for p in roots_b.values())
    stale.rmdir()
    nothing_landed()
    # THE GOOD PATH: every root published with its cells/ AND its provenance, no hidden sibling left, one record everywhere
    rec = d.import_prior_stores(state_a, roots_b, **kw)
    for r in runs:
        root = roots_b[r.tag]
        assert (root / "cells" / rk / "results.json").read_bytes() == src_bytes[r.tag]
        assert json.loads((root / d.RESUMED_FROM_FILE).read_text()) == rec
        assert rec["copied"][r.tag] == {"from": str(roots_a[r.tag]), "to": str(root), "cell_dirs": [rk]}
        assert not d.import_staging_dir(root, "DB").exists()
    assert entries() == sorted(baseline + [p.name for p in roots_b.values()])     # nothing hidden remains
    # the staged sibling is hidden and named after its root, so a `petri_seedrep_<W7>_*` glob never lists a half-built store
    staged = d.import_staging_dir(roots_b[t1], "20260918T000000Z.4242")
    assert staged.parent == parent and staged.name == f".{roots_b[t1].name}.import-20260918T000000Z.4242"
    staged.mkdir()
    assert staged.name not in {p.name for p in parent.glob(f"petri_seedrep_{W7}_*")}
    staged.rmdir()
    # the imported stores pass the driver's own reconciliation in wave B (skipped explicitly, not silently adopted)
    state_b = d.attempt_state_dir(d.state_dir_for(home, W7, d.wave_uuid8(WAVE_B), "arditi", ""), ATT_B)
    state_b.mkdir(parents=True)
    vb = d.completion_verdict(runs, roots_b, state_b, _real_reconcile(d, world.spec.bloom_options.epochs))
    assert vb["counts"] == {"complete": 3, "incomplete": 0, "failed": 0}
    # the phases are ordered in the source: preflight, stage, provenance INTO the staged roots, publish
    body = SRC.read_text().split("def import_prior_stores(")[1].split("\n# ----")[0]
    assert body.index("# 1. PREFLIGHT") < body.index("# 2. STAGE") < body.index("(staged / RESUMED_FROM_FILE).write_text") \
        < body.index("# 3. PUBLISH") < body.index("publish(staged, dst)")


# --------------------------------------------------------------------------- P2-3: the re-run pass retries like the base pass; STOP needs the verdict

def _run(d, **over):
    base = dict(view="arditi", cell_index=1, cell_id="arditi_allpos", vmodel="petri__arditi_allpos", run_key="rk", seed_base=2, width=4,
                tag="qwen35_4b_arditi_s2", petri_pass="overflow-rerun")
    return d.LegRun(**{**base, **over})


def _attempts(d, run, outcomes, *, proxy=None):
    """leg_attempts over scripted ``(rc, log)`` outcomes -> (rc, events, widths launched, sleeps, notes, logs)."""
    it, launched, slept, notes, logs = iter(outcomes), [], [], [], []
    proxy_it = iter(proxy or [])

    def launch(w):
        launched.append(w)
        return next(it)
    rc, events = d.leg_attempts(run, launch=launch, proxy_up=lambda: next(proxy_it, True), sleep=slept.append, note=notes.append, log=logs.append)
    return rc, events, launched, slept, notes, logs


def test_p2_3_the_rerun_pass_retries_a_failed_attempt_through_the_throttle_path_and_a_failed_leg_holds_the_proxy(d, tmp_path):
    heavy = "\n".join(f"retrying after 429 rate_limit_error ({i})" for i in range(25))
    rerun = _run(d)
    # FAIL-THEN-SUCCEED on the overflow-rerun pass: attempt 1 rc 1 (429-heavy) -> width 4 -> 3 -> attempt 2 rc 0 -> done; TWO launches
    rc, events, launched, slept, notes, logs = _attempts(d, rerun, [(1, heavy), (0, "")])
    assert rc == 0 and launched == [4, 3] and slept == [20]
    assert [(e["attempt"], e["width"], e["rc"], e["proxy_up"]) for e in events] == [(1, 4, 1, True), (2, 3, 0, True)]
    assert logs == ["leg_1_s2: 429-heavy attempt 1; width 4 -> 3 (throttle down, never hop)"]
    assert notes[0].startswith("== leg_1_s2 attempt 1 width=4 ") and notes[1] == "== leg_1_s2 attempt 1 rc=1 ==" and notes[3] == "== leg_1_s2 attempt 2 rc=0 =="
    # RE-VALIDATED repro (the old break rule, re-derived on this very script): `rc in (0, 3) or petri_pass == "overflow-rerun"` was True
    # after attempt 1 -> ONE attempt, rc 1, no throttle, and (below) STOP written on the held verdict
    assert (1 in (0, 3) or rerun.petri_pass == "overflow-rerun") is True
    # the base pass behaves identically -- the loop is pass-agnostic
    rc_b, events_b, launched_b, _, _, _ = _attempts(d, _run(d, petri_pass=""), [(1, heavy), (0, "")])
    assert (rc_b, launched_b, [e["rc"] for e in events_b]) == (0, [4, 3], [1, 0])
    # a transient failure with a CLEAN log keeps the width and retries
    rc, events, launched, slept, _, logs = _attempts(d, rerun, [(2, "transport reset by peer"), (0, "")])
    assert rc == 0 and launched == [4, 4] and logs == [] and slept == [20]
    # EXHAUSTED: three failing attempts -> rc 1 after exactly three launches, the width stepped 4 -> 3 -> 2 on the 429-heavy ones
    rc, events, launched, slept, _, logs = _attempts(d, rerun, [(1, heavy), (1, heavy), (1, "")])
    assert rc == 1 and launched == [4, 3, 2] and [e["attempt"] for e in events] == [1, 2, 3] and slept == [20, 20, 20] and len(logs) == 2
    assert d.LEG_ATTEMPTS == 3 and (d.PROXY_UNREACHABLE_SLEEP_S, d.ATTEMPT_SLEEP_S) == (30, 20)
    # rc 0 and rc 3 END the loop at once on either pass (complete / incomplete on the overflow ledger)
    for code in (0, 3):
        for run in (rerun, _run(d, petri_pass="")):
            rc, events, launched, slept, _, _ = _attempts(d, run, [(code, ""), (1, "")])
            assert rc == code and launched == [4] and len(events) == 1 and slept == []
    # the proxy unreachable on attempt 1: no launch, a 30 s sleep, the loop continues; never launched at all -> "unreachable"
    rc, events, launched, slept, notes, _ = _attempts(d, rerun, [(0, "")], proxy=[False, True])
    assert rc == 0 and launched == [4] and slept == [30] and events[0] == {"attempt": 1, "width": 4, "rc": None, "proxy_up": False} and "proxy unreachable" in notes
    rc, events, launched, slept, _, _ = _attempts(d, rerun, [], proxy=[False, False, False])
    assert rc == "unreachable" and launched == [] and slept == [30, 30, 30] and len(events) == 3
    # THE RELEASE RULE: STOP only when the verdict RELEASES, on every pass; a FAILED leg holds the proxy unless the NAMED flag asks otherwise
    held = {"counts": {"complete": 2, "incomplete": 0, "failed": 1}, "nruns": 3, "release": False, "detail": []}
    released = {"counts": {"complete": 2, "incomplete": 1, "failed": 0}, "nruns": 3, "release": True, "detail": []}
    assert d.release_decision(held) == (False, "held") and d.release_decision(released) == (True, "release")
    assert d.release_decision(held, teardown_after_failure=True) == (True, "teardown-after-failure")
    assert d.release_decision(released, teardown_after_failure=True) == (True, "release")
    # RE-VALIDATED repro: the old rule `args.petri_pass or verdict["release"]` was truthy on the re-run pass whatever the verdict
    assert bool("overflow-rerun" or held["release"]) is True
    # an EXHAUSTED leg on the re-run pass, through the real verdict: its rc-1 record -> FAILED -> release False -> the proxy is HELD
    runs = [_run(d, seed_base=s, tag=f"qwen35_4b_arditi_s{s}") for s in (1, 2, 3)]
    roots = {r.tag: tmp_path / r.tag for r in runs}
    state = tmp_path / "state"
    state.mkdir()
    for r in runs[:2]:
        res = roots[r.tag] / "cells" / r.run_key / "results.json"
        res.parent.mkdir(parents=True)
        res.write_text('{"audited": 60}')
        (state / f"{r.record_name}.rc").write_text(d.leg_status_line(0, "D1", res) + "\n")
    (state / f"{runs[2].record_name}.rc").write_text(d.leg_status_line(1, "D1", roots[runs[2].tag] / "cells" / "rk" / "results.json") + "\n")
    v = d.completion_verdict(runs, roots, state, lambda run, cd: (True, ""))
    assert v["counts"] == {"complete": 2, "incomplete": 0, "failed": 1} and v["detail"][2][2] == "failed_rc_1_drive_D1"
    assert d.release_decision(v) == (False, "held") and [r.record_name for r in d.legs_to_run(runs, v)] == ["leg_1_s3"]
    # the CLI: the flag exists, off by default
    ns = d.build_parser().parse_args(["--sha", W, "--wave-json", "w.json", "--view", "arditi", "--petri-pass", "overflow-rerun"])
    assert ns.teardown_after_failure is False
    assert d.build_parser().parse_args(["--sha", W, "--wave-json", "w.json", "--view", "arditi", "--teardown-after-failure"]).teardown_after_failure is True
    # the source: the old break rule and the old STOP rule are gone; main wires leg_attempts and release_decision; the argv guard stays
    # before the subprocess inside the per-attempt launch
    src = SRC.read_text()
    main = src.split("def main(")[1]
    assert 'or run.petri_pass == "overflow-rerun"' not in src and 'if args.petri_pass or verdict["release"]' not in src
    assert "rc, attempts_made = leg_attempts(run, launch=launch, proxy_up=lambda: probe_petri_models(base)[1], note=note, log=log)" in main
    assert "write_stop, stop_why = release_decision(verdict, teardown_after_failure=args.teardown_after_failure)" in main
    # the STOP is written ONLY under the release decision: main hands write_stop to publish_completion (which also binds it to the proxy
    # attempt), whose body returns before any sentinel_content("STOP") when it is False; the TEARDOWN log follows the publish
    assert main.index("write_stop, stop_why = release_decision(") \
        < main.index("write_stop=write_stop, stop_why=stop_why, legacy_generic=args.legacy_generic_completion, log=log)") \
        < main.index("TEARDOWN_AFTER_FAILURE:")
    pub = src.split("def publish_completion(")[1].split("\ndef ")[0]
    assert pub.index("if not write_stop:") < pub.index("return out") < pub.index("sentinels[ck] = json.dumps(bundle)")   # the bundle carries the STOP
    assert 'sentinel_content("STOP"' in src.split("def completion_bundle(")[1].split("\ndef ")[0] and 'sentinel_content("STOP"' not in main
    body = src.split("def run_leg(")[1]
    assert body.index("def launch(w: int)") < body.index("assert_lane_leg_argv(argv, run)") < body.index("subprocess.run(argv") < body.index("leg_attempts(run, launch=launch")


# --------------------------------------------------------------------------- P2-4: --width only narrows; the ceiling sums EFFECTIVE widths

def test_p2_4_width_overrides_only_narrow_a_view_and_the_ceiling_is_summed_over_the_effective_widths(d):
    s1, ar = d.view_params("s1"), d.view_params("arditi")
    assert (d.resolve_width(s1, None), d.resolve_width(s1, 3), d.resolve_width(s1, 1), d.resolve_width(ar, None), d.resolve_width(ar, 2)) == (3, 3, 1, 4, 2)
    assert d.total_auditor_width() == 84 == d.assert_within_auditor_ceiling() and d.total_auditor_width(widths={}) == 84
    # RE-VALIDATED repro: `--view s1 --width 13` -> 8 x 13 = 104 conversations from that view ALONE, while the old guard (static
    # defaults) reported 84 and passed; the effective sum is 104 + 48 + 12 = 164
    assert s1.expected_runs * 13 == 104 > d.AUDITOR_CEILING and d.total_auditor_width(widths={"s1": 13}) == 164
    with pytest.raises(SystemExit, match="164 concurrent auditor conversations exceed the highprio ceiling 100 "
                                          "\\(effective widths \\{'s1': 13, 's2': 3, 's3': 3, 'arditi': 4\\}\\)"):
        d.assert_within_auditor_ceiling(widths={"s1": 13})
    with pytest.raises(SystemExit, match="--width 13 exceeds view s1's planned width 3 \\(8 runs x 3 = 24 of the 100-conversation highprio ceiling; "
                                          "8 x 13 = 104\\) -- --width only narrows"):
        d.resolve_width(s1, 13)
    # Arditi over the ceiling: 3 x 10 = 30 -> 72 + 30 = 102 > 100 -- refused by BOTH rules
    assert d.total_auditor_width(widths={"arditi": 10}) == 102
    with pytest.raises(SystemExit, match="102 concurrent auditor conversations exceed the highprio ceiling 100"):
        d.assert_within_auditor_ceiling(widths={"arditi": 10})
    with pytest.raises(SystemExit, match="--width 10 exceeds view arditi's planned width 4 \\(3 runs x 4 = 12 of the 100-conversation highprio ceiling; 3 x 10 = 30\\)"):
        d.resolve_width(ar, 10)
    # a width that FITS the global ceiling but exceeds the view's own share is refused too: one drive cannot see the other three's widths
    assert d.assert_within_auditor_ceiling(widths={"arditi": 9}) == 99 <= d.AUDITOR_CEILING
    with pytest.raises(SystemExit, match="--width 9 exceeds view arditi's planned width 4"):
        d.resolve_width(ar, 9)
    with pytest.raises(SystemExit, match="--width 4 exceeds view s1's planned width 3"):
        d.resolve_width(s1, 4)
    # narrowing lowers the effective total; 0 / negative are refused (the old `if args.width` read 0 as "use the default")
    assert d.total_auditor_width(widths={"s1": 1}) == 68 and d.total_auditor_width(widths={"s1": 1, "s2": 1, "s3": 1, "arditi": 1}) == 27
    for bad in (0, -1):
        with pytest.raises(SystemExit, match=f"--width {bad} must be >= 1"):
            d.resolve_width(s1, bad)
    assert (0 if 0 else s1.width) == 3                                   # the silent fallback the old expression had
    # a width for a view outside the sum is refused rather than ignored
    with pytest.raises(ValueError, match="widths for views not being summed: \\['s9'\\]"):
        d.total_auditor_width(widths={"s9": 1})
    assert d.total_auditor_width(("s1", "arditi"), widths={"s1": 2, "arditi": 1}) == 16 + 3
    # main(): this view's override enters the ceiling check through the effective widths; the static call is gone; the record carries them
    main = SRC.read_text().split("def main(")[1]
    assert "width = resolve_width(vp, args.width)" in main and "assert_within_auditor_ceiling(widths=effective_widths)" in main
    assert "effective_widths = {v: (width if v == args.view else view_params(v).width) for v in VIEWS}" in main
    assert "assert_within_auditor_ceiling()" not in main and "int(args.width) if args.width else vp.width" not in main
    assert '"effective_widths": effective_widths' in main
    assert main.index("width = resolve_width(vp, args.width)") < main.index("assert_within_auditor_ceiling(widths=effective_widths)") < main.index("wt = LOCAL_REPO")


# --------------------------------------------------------------------------- the rollback never lies; a durable transaction marker

def _wave_b(d, runs, home):
    roots_b = {r.tag: d.out_root_for(home, W7, d.wave_uuid8(WAVE_B), ATT_B, r.tag) for r in runs}
    parent = next(iter(roots_b.values())).parent
    marker = d.import_txn_marker(roots_b, W7, d.wave_uuid8(WAVE_B), ATT_B)
    return roots_b, parent, sorted(p.name for p in parent.iterdir()), marker


def test_sol_r3_a_failed_rollback_is_reported_with_its_leftover_root_never_as_clean_and_the_marker_recovers_it(d, world, tmp_path):
    home = tmp_path / "home"
    cell, runs, roots_a, state_a, _rec_a = _wave_a(d, world, tmp_path, home)
    roots_b, parent, baseline, marker = _wave_b(d, runs, home)
    t1, t2, t3 = (r.tag for r in runs)
    kw = dict(w7=W7, wave_uuid=WAVE_B, attempt_id=ATT_B, view="arditi", drive_id="DB")
    assert marker.parent == parent and marker.name == f".petri_seedrep_{W7}_{d.wave_uuid8(WAVE_B)}_{ATT_B}.import-txn.json" and not marker.exists()
    # THE PROBE: the second publish fails AND the rollback's rename of the first published root fails
    pubs, unpubs = [], []

    def publish_boom(staged, dst):
        assert marker.is_file(), "the transaction marker must be on disk before the first rename"
        pubs.append(dst)
        if len(pubs) == 2:
            raise OSError("rename refused")
        os.rename(staged, dst)

    def unpublish_boom(dst, staged):
        unpubs.append(dst)
        raise OSError("device busy")
    with pytest.raises(SystemExit) as ei:
        d.import_prior_stores(state_a, roots_b, publish=publish_boom, unpublish=unpublish_boom, **kw)
    msg = str(ei.value)
    assert "import failed (OSError: rename refused) and the rollback did NOT complete" in msg
    assert f"could not un-publish {roots_b[t1]} -> {d.import_staging_dir(roots_b[t1], 'DB')}: device busy" in msg
    assert f"still present: ['{roots_b[t1]}']" in msg and f"the transaction marker {marker} stays" in msg
    assert "no root of this attempt was left behind" not in msg                     # the old false claim, never
    assert pubs == [roots_b[t1], roots_b[t2]] and unpubs == [roots_b[t1]]
    # the HONEST state: the first root IS still published (with its provenance), the staged siblings are gone, the marker STAYS naming the roots
    assert roots_b[t1].is_dir() and (roots_b[t1] / d.RESUMED_FROM_FILE).is_file() and not roots_b[t2].exists() and not roots_b[t3].exists()
    assert not any(d.import_staging_dir(r, "DB").exists() for r in roots_b.values())
    txn = json.loads(marker.read_text())
    assert txn["schema"] == d.IMPORT_TXN_SCHEMA and txn["into"] == {"W7": W7, "wave_uuid": WAVE_B, "attempt_id": ATT_B} and txn["drive_id"] == "DB"
    assert txn["from"] == {"drive_record": str(state_a / "drive_DA.json")}
    assert [(r["tag"], r["staged"], r["dst"]) for r in txn["roots"]] == [(t, str(d.import_staging_dir(roots_b[t], "DB")), str(roots_b[t])) for t in (t1, t2, t3)]
    # the old code swallowed the OSError and claimed a clean rollback while root 1 stood; an exact retry then refused the OCCUPIED root. Now
    # the retry is refused on the OPEN transaction, naming the recovery
    with pytest.raises(SystemExit, match="import transaction into this attempt's roots is OPEN"):
        d.import_prior_stores(state_a, roots_b, **kw)
    # RECOVERY on the next invocation: the partial publication is rolled back, the marker removed, nothing of wave B remains
    rec = d.recover_import_transaction(roots_b, w7=W7, wave_uuid=WAVE_B, attempt_id=ATT_B)
    assert rec["action"] == "rolled_back" and rec["unpublished"] == [str(roots_b[t1])] and rec["removed_staged"] == [str(d.import_staging_dir(roots_b[t1], "DB"))]
    assert rec["states"] == {t1: "published", t2: "absent", t3: "absent"}
    assert rec["marker"] == str(marker) and rec["drive_id"] == "DB" and rec["from_drive_record"] == str(state_a / "drive_DA.json")
    assert not marker.exists() and not any(p.exists() for p in roots_b.values()) and sorted(p.name for p in parent.iterdir()) == baseline
    assert d.recover_import_transaction(roots_b, w7=W7, wave_uuid=WAVE_B, attempt_id=ATT_B) is None       # nothing open -> None
    # a fresh import now succeeds; the marker is on disk before EVERY rename and gone after the last; the clean-rollback message says "verified"
    seen = []

    def publish_watch(staged, dst):
        seen.append(marker.is_file())
        os.rename(staged, dst)
    d.import_prior_stores(state_a, roots_b, publish=publish_watch, **kw)
    assert seen == [True, True, True] and not marker.exists() and all((roots_b[t] / d.RESUMED_FROM_FILE).is_file() for t in roots_b)
    assert sorted(p.name for p in parent.iterdir()) == sorted(baseline + [p.name for p in roots_b.values()])
    for t in roots_b:
        shutil.rmtree(roots_b[t])
    # a clean rollback (second publish fails, un-publish works) is claimed only after verification, and the marker is gone with it
    pubs.clear()
    with pytest.raises(SystemExit, match="rolled back -- no root of this attempt was left behind \\(verified: no destination, no staged sibling\\)"):
        d.import_prior_stores(state_a, roots_b, publish=publish_boom, **kw)
    assert not marker.exists() and sorted(p.name for p in parent.iterdir()) == baseline
    # a rollback whose staged-sibling removal fails (a STAGING failure, before any marker) names the staged dir and says no marker was written
    calls = []

    def copytree_boom(src, dst):
        calls.append(src)
        if len(calls) == 2:
            raise OSError("disk full")
        return shutil.copytree(src, dst)
    real_rmtree = d.shutil.rmtree

    def rmtree_boom(path, *a, **k):
        if Path(path) == d.import_staging_dir(roots_b[t1], "DB"):
            raise OSError("busy staged")
        return real_rmtree(path, *a, **k)
    d.shutil.rmtree = rmtree_boom
    try:
        with pytest.raises(SystemExit) as ei2:
            d.import_prior_stores(state_a, roots_b, copytree=copytree_boom, **kw)
    finally:
        d.shutil.rmtree = real_rmtree
    m2 = str(ei2.value)
    assert "import failed (OSError: disk full) and the rollback did NOT complete" in m2 and "could not remove staged" in m2 and "busy staged" in m2
    assert f"still present: ['{d.import_staging_dir(roots_b[t1], 'DB')}']" in m2 and "no transaction marker had been written yet" in m2
    assert not any(p.exists() for p in roots_b.values()) and not marker.exists()
    shutil.rmtree(d.import_staging_dir(roots_b[t1], "DB"))
    assert sorted(p.name for p in parent.iterdir()) == baseline
    # the source pins: the rollback swallows nothing, the failure path verifies, main records the recovery
    body = SRC.read_text().split("def import_prior_stores(")[1].split("\n# ----")[0]
    assert "except OSError:\n                pass" not in body and "ignore_errors=True" not in body
    assert body.index("write_import_txn_marker(marker") < body.index("publish(staged, dst)") < body.index("marker.unlink()\n    except BaseException")
    # ... and the OPEN-transaction refusal precedes the per-root preflight (else a leftover root reads as merely "already exists")
    assert body.index("if marker.exists():") < body.index("# 1. PREFLIGHT") < body.index("if dst.exists():")


def test_sol_r3_the_transaction_marker_recovers_a_crash_between_renames_and_never_undoes_a_completed_import(d, world, tmp_path):
    home = tmp_path / "home"
    cell, runs, roots_a, state_a, _rec_a = _wave_a(d, world, tmp_path, home)
    rk = cell.petri_run_key
    roots_b, parent, baseline, marker = _wave_b(d, runs, home)
    t1, t2, t3 = (r.tag for r in runs)
    kw = dict(w7=W7, wave_uuid=WAVE_B, attempt_id=ATT_B)

    def stage_all(drive_id):
        """What import_prior_stores has on disk the instant before its FIRST rename: three staged roots (cells + provenance) + the marker."""
        staged = {t: d.import_staging_dir(roots_b[t], drive_id) for t in roots_b}
        for t, st in staged.items():
            shutil.copytree(roots_a[t] / "cells", st / "cells")
            (st / d.RESUMED_FROM_FILE).write_text(json.dumps({"schema": "seedrep_petri_drive_resume/1", "drive_id": drive_id}))
        d.write_import_txn_marker(marker, {"schema": d.IMPORT_TXN_SCHEMA, "drive_id": drive_id, "written_at": "2026-09-18T00:00:00Z",
                                           "into": {"W7": W7, "wave_uuid": WAVE_B, "attempt_id": ATT_B},
                                           "from": {"drive_record": str(state_a / "drive_DA.json")},
                                           "roots": [{"tag": t, "staged": str(staged[t]), "dst": str(roots_b[t])} for t in (t1, t2, t3)]})
        assert marker.is_file() and not marker.with_name(marker.name + ".tmp").exists()
        return staged
    # SIGTERM between the first and the second rename: root 1 published, 2 and 3 still staged, the marker on disk -> ROLLED BACK
    staged = stage_all("DX")
    os.rename(staged[t1], roots_b[t1])
    rec = d.recover_import_transaction(roots_b, **kw)
    assert rec["action"] == "rolled_back" and rec["states"] == {t1: "published", t2: "staged", t3: "staged"} and rec["drive_id"] == "DX"
    assert rec["unpublished"] == [str(roots_b[t1])] and rec["removed_staged"] == [str(staged[t]) for t in (t1, t2, t3)]
    assert not marker.exists() and sorted(p.name for p in parent.iterdir()) == baseline
    # a crash AFTER the last rename but before the marker went: every root published -> COMPLETED; the roots stand, only the marker goes
    staged = stage_all("DY")
    for t in (t1, t2, t3):
        os.rename(staged[t], roots_b[t])
    rec = d.recover_import_transaction(roots_b, **kw)
    assert rec["action"] == "completed" and rec["states"] == {t1: "published", t2: "published", t3: "published"} and "unpublished" not in rec
    assert not marker.exists() and all((roots_b[t] / "cells" / rk / "results.json").is_file() and (roots_b[t] / d.RESUMED_FROM_FILE).is_file() for t in roots_b)
    assert sorted(p.name for p in parent.iterdir()) == sorted(baseline + [p.name for p in roots_b.values()])
    for t in roots_b:
        shutil.rmtree(roots_b[t])
    # KNOWN-BAD 1: a root AND its staged sibling both present is not a state the import leaves -> refused by name, marker kept, nothing touched
    staged = stage_all("DZ")
    shutil.copytree(staged[t2], roots_b[t2])
    with pytest.raises(SystemExit, match="both exist -- not a state this import leaves"):
        d.recover_import_transaction(roots_b, **kw)
    assert marker.exists() and roots_b[t2].exists() and all(s.exists() for s in staged.values())
    shutil.rmtree(roots_b[t2])
    # KNOWN-BAD 2: the recovery's OWN un-publish fails -> the explicit residual error names the root, the marker stays (never a clean claim)
    os.rename(staged[t1], roots_b[t1])

    def unpublish_boom(dst, st):
        raise OSError("busy")
    with pytest.raises(SystemExit) as ei:
        d.recover_import_transaction(roots_b, unpublish=unpublish_boom, **kw)
    msg = str(ei.value)
    assert f"{marker}: recovery did NOT complete -- could not un-publish {roots_b[t1]} -> {staged[t1]}: busy" in msg
    assert f"still present: ['{roots_b[t1]}']" in msg and "the marker stays" in msg
    assert marker.exists() and roots_b[t1].exists() and not staged[t2].exists() and not staged[t3].exists()
    # ... and the real recovery then cleans it
    rec = d.recover_import_transaction(roots_b, **kw)
    assert rec["action"] == "rolled_back" and rec["states"] == {t1: "published", t2: "absent", t3: "absent"}
    assert not marker.exists() and sorted(p.name for p in parent.iterdir()) == baseline
    # KNOWN-BAD 3: an unreadable marker; a marker of another wave under this name; a marker naming no roots -> refused by name, untouched
    marker.write_text("{not json")
    with pytest.raises(SystemExit, match="unreadable import transaction marker"):
        d.recover_import_transaction(roots_b, **kw)
    marker.write_text(json.dumps({"schema": d.IMPORT_TXN_SCHEMA, "into": {"W7": W7, "wave_uuid": WAVE_A, "attempt_id": ATT_B}, "roots": []}))
    with pytest.raises(SystemExit, match="not this attempt's import transaction"):
        d.recover_import_transaction(roots_b, **kw)
    marker.write_text(json.dumps({"schema": d.IMPORT_TXN_SCHEMA, "into": {"W7": W7, "wave_uuid": WAVE_B, "attempt_id": ATT_B}, "roots": []}))
    with pytest.raises(SystemExit, match="names no roots"):
        d.recover_import_transaction(roots_b, **kw)
    assert marker.exists()
    marker.unlink()
    # an OPEN transaction refuses a new import before anything is staged
    stage_all("DW")
    with pytest.raises(SystemExit, match="import transaction into this attempt's roots is OPEN"):
        d.import_prior_stores(state_a, roots_b, view="arditi", drive_id="DV", **kw)
    assert not d.import_staging_dir(roots_b[t1], "DV").exists()
    assert d.recover_import_transaction(roots_b, **kw)["action"] == "rolled_back" and sorted(p.name for p in parent.iterdir()) == baseline
    # the marker is hidden from root globs and needs one shared parent
    assert marker.name.startswith(".") and marker.name not in {p.name for p in parent.glob(f"petri_seedrep_{W7}_*")}
    with pytest.raises(ValueError, match="do not share one parent"):
        d.import_txn_marker({t1: roots_b[t1], t2: tmp_path / "elsewhere" / "r"}, W7, d.wave_uuid8(WAVE_B), ATT_B)
    # main(): the recovery runs before any import and before the enumeration freeze; the drive record carries it
    main = SRC.read_text().split("def main(")[1]
    # recover -> resolve -> receipt is ONE replayable step, recover_and_resolve_resume; main runs it before the enumeration freeze
    assert main.index("recovery, resumed_from, resume_how = recover_and_resolve_resume(out_roots, resume_from=") < main.index('"print-cells"')
    step = SRC.read_text().split("def recover_and_resolve_resume(")[1].split("\ndef ")[0]
    assert step.index("recovery = recover_import_transaction(out_roots, w7=w7, wave_uuid=wave_uuid, attempt_id=attempt_id)") < step.index("IMPORT_RECOVERY") \
        < step.index("resolve_resume(out_roots, resume_from=resume_from, recovery=recovery")
    assert '"import_recovery": recovery' in main


# --------------------------------------------------------------------------- a completed recovery IS the requested import

def _marker_after_last_rename(d, state_a, roots_b, drive_id):
    """The state 'every root published, the marker not yet removed' = a death between the last rename and ``marker.unlink()``: the marker
    ``import_prior_stores`` wrote before its first rename, re-created by hand with the same content."""
    marker = d.import_txn_marker(roots_b, W7, d.wave_uuid8(WAVE_B), ATT_B)
    d.write_import_txn_marker(marker, {"schema": d.IMPORT_TXN_SCHEMA, "drive_id": drive_id, "written_at": "2026-09-18T00:00:00Z",
                                       "into": {"W7": W7, "wave_uuid": WAVE_B, "attempt_id": ATT_B},
                                       "from": {"drive_record": str(state_a / "drive_DA.json")},
                                       "roots": [{"tag": t, "staged": str(d.import_staging_dir(roots_b[t], drive_id)), "dst": str(roots_b[t])}
                                                 for t in roots_b]})
    assert marker.is_file() and all(roots_b[t].is_dir() for t in roots_b)
    return marker


def test_sol_r4_a_completed_recovery_is_adopted_as_the_requested_import_so_the_original_argv_continues(d, world, tmp_path):
    home = tmp_path / "home"
    cell, runs, roots_a, state_a, _rec_a = _wave_a(d, world, tmp_path, home)
    roots_b, parent, baseline, marker = _wave_b(d, runs, home)
    t1, t2, t3 = (r.tag for r in runs)
    kw = dict(w7=W7, wave_uuid=WAVE_B, attempt_id=ATT_B, view="arditi")

    def snap():          # the roots' WHOLE parent: the roots, the marker, any staged sibling (the marker is part of the state)
        return {str(p.relative_to(parent)): p.read_bytes() for p in parent.rglob("*") if p.is_file()}
    # the ORIGINAL argv (--resume-from <wave A's attempt state dir>): the first invocation imports for real ...
    rec1 = d.import_prior_stores(state_a, roots_b, drive_id="DB", **kw)
    assert not marker.exists() and all((roots_b[t] / d.RESUMED_FROM_FILE).is_file() for t in roots_b)
    before = snap()
    # ... and dies AFTER its last rename, BEFORE marker.unlink(): every root published, the marker on disk
    _marker_after_last_rename(d, state_a, roots_b, "DB")
    # THE REPRO (re-validated): on restart the recovery says `completed`, and the OLD main then re-entered import_prior_stores with the
    # unchanged argv -> its preflight refused the recovered roots
    rec = d.recover_import_transaction(roots_b, w7=W7, wave_uuid=WAVE_B, attempt_id=ATT_B)
    assert rec["action"] == "completed" and rec["states"] == {t1: "published", t2: "published", t3: "published"} and not marker.exists()
    with pytest.raises(SystemExit, match="already exists -- an import never writes over a store"):
        d.import_prior_stores(state_a, roots_b, drive_id="DB2", **kw)
    assert snap() == before
    # THE FIX: main resolves the resume through resolve_resume -- a completed recovery IS the import, with the ORIGINAL argv ...
    _marker_after_last_rename(d, state_a, roots_b, "DB")
    recovery = d.recover_import_transaction(roots_b, w7=W7, wave_uuid=WAVE_B, attempt_id=ATT_B)
    assert recovery["action"] == "completed" and recovery["from_drive_record"] == str(state_a / "drive_DA.json")
    logs = []
    resumed, how = d.resolve_resume(roots_b, resume_from=state_a, recovery=recovery, drive_id="DB2", log=logs.append, **kw)
    assert how == "adopted" and resumed == rec1 == json.loads((roots_b[t1] / d.RESUMED_FROM_FILE).read_text())
    assert resumed["schema"] == "seedrep_petri_drive_resume/1" and resumed["into"] == {"W7": W7, "wave_uuid": WAVE_B, "attempt_id": ATT_B, "view": "arditi"}
    assert resumed["from"]["state_dir"] == str(state_a) and resumed["from"]["drive_record"] == str(state_a / "drive_DA.json") and resumed["drive_id"] == "DB"
    assert set(resumed["copied"]) == {t1, t2, t3}
    assert snap() == before and not marker.exists() and sorted(p.name for p in parent.iterdir()) == sorted(baseline + [p.name for p in roots_b.values()])
    assert logs == [f"RESUME_ADOPTED: the recovered import (drive DB, from {state_a / 'drive_DA.json'}) is this drive's --resume-from "
                    f"(the argv named the same source, {state_a})"]
    # ... in another spelling of the same source (a trailing slash; a ..-hop) ...
    for alt in (Path(str(state_a) + "/"), state_a.parent / ".." / state_a.parent.name / state_a.name):
        assert d.resolve_resume(roots_b, resume_from=alt, recovery=recovery, drive_id="DB2", log=lambda m: None, **kw)[1] == "adopted"
    # ... and with NO --resume-from on the argv at all (the restart may drop it): still adopted
    r2, how2 = d.resolve_resume(roots_b, resume_from=None, recovery=recovery, drive_id="DB2", log=logs.append, **kw)
    assert how2 == "adopted" and r2 == rec1 and logs[-1].endswith("is this drive's --resume-from (no --resume-from on the argv)")
    assert snap() == before
    # KNOWN-BAD 1: the argv names a DIFFERENT source than the recovered import came from -> explicit refusal naming both, nothing touched
    state_c = tmp_path / "elsewhere" / "attempts" / "ta-C"
    state_c.mkdir(parents=True)
    (state_c / "drive_DC.json").write_text(json.dumps({"drive_id": "DC", "started_at": "2026-09-18T00:00:00Z"}))
    with pytest.raises(SystemExit) as ei:
        d.resolve_resume(roots_b, resume_from=state_c, recovery=recovery, drive_id="DB2", log=lambda m: None, **kw)
    msg = str(ei.value)
    assert (f"--resume-from {state_c} names a DIFFERENT source than the recovered import came from ({state_a}, drive record "
            f"{state_a / 'drive_DA.json'}) -- refusing" in msg) and "drop --resume-from (or name that source)" in msg
    assert snap() == before
    # KNOWN-BAD 2..7: a recovered root without provenance; two roots disagreeing; another view's `into`; a tag missing from `copied`; the
    # marker's source != the provenance's source; unreadable provenance -- each refused by name, nothing touched
    rfs = {t: roots_b[t] / d.RESUMED_FROM_FILE for t in roots_b}

    def write_all(rec):
        for rf in rfs.values():
            rf.write_text(json.dumps(rec, indent=1))
    run = lambda **o: d.resolve_resume(roots_b, resume_from=state_a, recovery=o.pop("recovery", recovery), drive_id="DB2", log=lambda m: None, **kw)  # noqa: E731
    rfs[t1].unlink()
    with pytest.raises(SystemExit, match=f"recovered root {re.escape(str(roots_b[t1]))} carries no RESUMED_FROM.json"):
        run()
    rfs[t1].write_text(json.dumps({**rec1, "drive_id": "OTHER"}))
    with pytest.raises(SystemExit, match="carry 2 DIFFERENT RESUMED_FROM.json records"):
        run()
    write_all({**rec1, "into": {**rec1["into"], "view": "s1"}})
    with pytest.raises(SystemExit, match="is not this attempt's import"):
        run()
    write_all({**rec1, "copied": {t1: rec1["copied"][t1]}})
    with pytest.raises(SystemExit, match="covers tags"):
        run()
    write_all(rec1)
    with pytest.raises(SystemExit, match="but the transaction marker named '/elsewhere/drive_X.json'"):
        run(recovery={**recovery, "from_drive_record": "/elsewhere/drive_X.json"})
    rfs[t2].write_text("{nope")
    with pytest.raises(SystemExit, match="unreadable provenance"):
        run()
    write_all(rec1)
    assert snap() == before
    # no recovery + --resume-from -> a real import ("imported"); no recovery + no argv -> nothing; a ROLLED-BACK recovery + --resume-from -> import
    for r in roots_b.values():
        shutil.rmtree(r)
    r3, how3 = d.resolve_resume(roots_b, resume_from=state_a, recovery=None, drive_id="DB3", log=lambda m: None, **kw)
    assert how3 == "imported" and r3["drive_id"] == "DB3" and all((roots_b[t] / d.RESUMED_FROM_FILE).is_file() for t in roots_b)
    for r in roots_b.values():
        shutil.rmtree(r)
    assert d.resolve_resume(roots_b, resume_from=None, recovery=None, drive_id="DB4", log=lambda m: None, **kw) == (None, "none")
    assert d.resolve_resume(roots_b, resume_from=state_a, recovery={"action": "rolled_back"}, drive_id="DB5", log=lambda m: None, **kw)[1] == "imported"
    for r in roots_b.values():
        shutil.rmtree(r)
    assert sorted(p.name for p in parent.iterdir()) == baseline
    # main(): the bare `if args.resume_from: import_prior_stores` is gone; recovery -> resolve_resume -> the record carries resume_how
    main = SRC.read_text().split("def main(")[1]
    assert "if args.resume_from:\n        resumed_from = import_prior_stores(" not in main and "import_prior_stores(" not in main
    assert main.index("recovery, resumed_from, resume_how = recover_and_resolve_resume(out_roots, resume_from=Path(args.resume_from).expanduser() "
                      "if args.resume_from else None,") < main.index('"resume_how": resume_how')
    step = SRC.read_text().split("def recover_and_resolve_resume(")[1].split("\ndef ")[0]
    assert step.index("recovery = recover_import_transaction(") \
        < step.index("resolve_resume(out_roots, resume_from=resume_from, recovery=recovery, w7=w7, wave_uuid=wave_uuid, attempt_id=attempt_id,") \
        < step.index("(astate / RESUMED_FROM_FILE).write_text(json.dumps(resumed_from, indent=1))")


# --------------------------------------------------------------------------- the marker is not the adoption predicate; the roots' provenance is

def test_sol_r5_adoption_is_derived_from_the_roots_provenance_and_survives_invocation_boundaries(d, world, tmp_path):
    """recover_import_transaction() consumes the marker the moment it sees every root published; whatever the rest of THAT invocation does
    (refuses a wrong --resume-from, chokes on malformed provenance, dies before the receipt), the NEXT invocation must still adopt the
    roots -- from their own RESUMED_FROM.json, which stands in a root iff the import published it. Every 'restart' below is a genuine
    invocation of the drive's own step (recover -> resolve -> receipt) with nothing carried over in memory; the snapshot covers the roots'
    whole parent (marker and staging included) and the attempt state dir (the receipt)."""
    home = tmp_path / "home"
    cell, runs, roots_a, state_a, _rec_a = _wave_a(d, world, tmp_path, home)
    roots_b, parent, baseline, marker = _wave_b(d, runs, home)
    t1, t2, t3 = (r.tag for r in runs)
    astate_b = d.attempt_state_dir(d.state_dir_for(home, W7, d.wave_uuid8(WAVE_B), "arditi", ""), ATT_B)
    receipt = astate_b / d.RESUMED_FROM_FILE
    kw = dict(w7=W7, wave_uuid=WAVE_B, attempt_id=ATT_B, view="arditi")

    def snap():
        return {str(p.relative_to(tmp_path)): p.read_bytes() for base in (parent, astate_b) if base.exists() for p in base.rglob("*") if p.is_file()}

    def invocation(resume_from, drive_id):
        """ONE process: the drive's own step, fresh -- recover, resolve, receipt. Returns (recovery, resumed_from, how, logs)."""
        logs = []
        recovery, resumed, how = d.recover_and_resolve_resume(roots_b, resume_from=resume_from, drive_id=drive_id, astate=astate_b, log=logs.append, **kw)
        return recovery, resumed, how, logs
    # invocation 0 -- the ORIGINAL argv imports for real; the receipt is recorded
    rec0, res0, how0, _ = invocation(state_a, "DB0")
    assert rec0 is None and how0 == "imported" and res0["drive_id"] == "DB0" and not marker.exists()
    assert json.loads(receipt.read_text()) == res0
    steady = snap()
    prov = {t: (roots_b[t] / d.RESUMED_FROM_FILE).read_bytes() for t in roots_b}

    def s1():
        """The state 'every root published, marker still on disk, NO receipt': invocation 0 died after its last rename, before marker.unlink()."""
        _marker_after_last_rename(d, state_a, roots_b, "DB0")
        if receipt.exists():
            receipt.unlink()
        assert marker.exists() and not receipt.exists()
    # A. process death AFTER the all-published recovery removed the marker, BEFORE resolve_resume returned: the next invocation adopts
    s1()
    rec = d.recover_import_transaction(roots_b, w7=W7, wave_uuid=WAVE_B, attempt_id=ATT_B)          # restart 1 got exactly this far, then died
    assert rec["action"] == "completed" and not marker.exists() and not receipt.exists()
    consumed = snap()
    # the OLD flow re-entered import_prior_stores here and was refused on the standing roots (the repro, re-validated)
    with pytest.raises(SystemExit, match="already exists -- an import never writes over a store"):
        d.import_prior_stores(state_a, roots_b, drive_id="DB2", **kw)
    assert snap() == consumed
    rec2, res2, how2, logs2 = invocation(state_a, "DB2")                                            # restart 2: the ORIGINAL argv, nothing carried over
    assert rec2 is None and how2 == "adopted" and res2 == res0 and json.loads(receipt.read_text()) == res0
    assert logs2[0] == (f"RESUME_ADOPTED: this attempt's roots already stand with their import's provenance (drive DB0, from {state_a / 'drive_DA.json'}; "
                        f"no open transaction) -- adopted as this drive's --resume-from (the argv named the same source, {state_a})")
    assert len(logs2) == 2 and logs2[1].startswith(f"RESUMED_FROM (adopted) {state_a / 'drive_DA.json'} (wave {WAVE_A} attempt {ATT_A}): ") \
        and logs2[1].count("1 cell dir(s)") == 3
    assert snap() == steady
    # ... and with NO argv on the restart: adopted, not "none" -- the import does not vanish from the attempt's provenance
    receipt.unlink()
    rec3, res3, how3, logs3 = invocation(None, "DB3")
    assert rec3 is None and how3 == "adopted" and res3 == res0 and json.loads(receipt.read_text()) == res0
    assert logs3[0].endswith("adopted as this drive's --resume-from (no --resume-from on the argv)") and snap() == steady
    # B. a GENUINE two-invocation sequence: restart 1 recovers (the marker is consumed) and refuses a MISMATCHED source; restart 2 with the
    # corrected argv adopts, with no in-memory recovery result carried over
    s1()
    state_c = tmp_path / "elsewhere" / "attempts" / "ta-C"
    state_c.mkdir(parents=True, exist_ok=True)
    (state_c / "drive_DC.json").write_text(json.dumps({"drive_id": "DC", "started_at": "2026-09-18T00:00:00Z"}))
    with pytest.raises(SystemExit, match="names a DIFFERENT source than the recovered import came from"):
        invocation(state_c, "DB4")
    assert not marker.exists() and not receipt.exists() and snap() == consumed                    # restart 1 DID consume the marker; the roots stand
    rec5, res5, how5, _ = invocation(state_a, "DB5")
    assert rec5 is None and how5 == "adopted" and res5 == res0 and json.loads(receipt.read_text()) == res0 and snap() == steady
    # ... the same wrong argv twice is refused twice (never silently continued), then the corrected one adopts
    s1()
    for _ in range(2):
        with pytest.raises(SystemExit, match="names a DIFFERENT source"):
            invocation(state_c, "DB6")
    assert snap() == consumed
    assert invocation(state_a, "DB7")[2] == "adopted" and snap() == steady
    # C. restart 1 recovers and chokes on MALFORMED provenance; still refused on restart 2 while broken; the bytes restored -> restart 3 adopts
    s1()
    (roots_b[t2] / d.RESUMED_FROM_FILE).write_text("{nope")
    with pytest.raises(SystemExit, match="unreadable provenance"):
        invocation(state_a, "DB8")
    assert not marker.exists() and not receipt.exists()
    with pytest.raises(SystemExit, match="unreadable provenance"):
        invocation(state_a, "DB9")
    (roots_b[t2] / d.RESUMED_FROM_FILE).write_bytes(prov[t2])
    assert invocation(state_a, "DB10")[2] == "adopted" and snap() == steady
    # ... disagreeing provenance likewise, argv or not
    s1()
    (roots_b[t3] / d.RESUMED_FROM_FILE).write_text(json.dumps({**res0, "drive_id": "OTHER"}))
    with pytest.raises(SystemExit, match="DIFFERENT RESUMED_FROM.json records"):
        invocation(state_a, "DB11")
    with pytest.raises(SystemExit, match="DIFFERENT RESUMED_FROM.json records"):
        invocation(None, "DB12")
    (roots_b[t3] / d.RESUMED_FROM_FILE).write_bytes(prov[t3])
    assert invocation(None, "DB13")[2] == "adopted" and snap() == steady
    # D. the plain case: a re-invocation of an attempt whose import SUCCEEDED long ago (receipt present, marker long gone), unchanged argv or none
    assert receipt.exists() and not marker.exists()
    assert invocation(state_a, "DB14")[2] == "adopted" and invocation(None, "DB15")[2] == "adopted" and snap() == steady
    # E. a NON-import attempt's roots (standing, no provenance): no argv -> none (a plain resume); an argv -> the import refuses the standing root
    for r in roots_b.values():
        shutil.rmtree(r)
    receipt.unlink()
    for r in roots_b.values():
        r.mkdir()
        (r / "cells.json").write_text("{}")
    assert invocation(None, "DB16")[1:3] == (None, "none") and not receipt.exists()
    with pytest.raises(SystemExit, match="already exists -- an import never writes over a store"):
        invocation(state_a, "DB17")
    # ... a MIXED state (one root with provenance, the others without) is not one the import leaves -> refused by name, argv or not
    (roots_b[t1] / d.RESUMED_FROM_FILE).write_bytes(prov[t1])
    with pytest.raises(SystemExit, match="carries no RESUMED_FROM.json"):
        invocation(None, "DB18")
    with pytest.raises(SystemExit, match="carries no RESUMED_FROM.json"):
        invocation(state_a, "DB19")
    # ... a PARTIAL set of roots (one missing): no argv -> none; an argv -> the import refuses the standing ones
    shutil.rmtree(roots_b[t1])
    assert invocation(None, "DB20")[2] == "none"
    with pytest.raises(SystemExit, match="already exists"):
        invocation(state_a, "DB21")
    for r in roots_b.values():
        if r.exists():
            shutil.rmtree(r)
    assert sorted(p.name for p in parent.iterdir()) == baseline
    # main() runs exactly this step before the enumeration freeze; the step is recover -> resolve -> receipt; resolve_resume tries the completed
    # recovery, then the standing roots' provenance, then the import; the record carries all three
    src = SRC.read_text()
    main = src.split("def main(")[1]
    assert main.index("recovery, resumed_from, resume_how = recover_and_resolve_resume(out_roots, resume_from=Path(args.resume_from).expanduser() "
                      "if args.resume_from else None,") < main.index('"print-cells"')
    assert "recover_import_transaction(" not in main and "import_prior_stores(" not in main
    assert "resolve_resume(out_roots" not in main.replace("recover_and_resolve_resume(out_roots", "")
    assert '"import_recovery": recovery' in main and '"resumed_from": resumed_from' in main and '"resume_how": resume_how' in main
    step = src.split("def recover_and_resolve_resume(")[1].split("\ndef ")[0]
    assert step.index("recovery = recover_import_transaction(out_roots") < step.index("resolve_resume(out_roots, resume_from=resume_from, recovery=recovery") \
        < step.index("(astate / RESUMED_FROM_FILE).write_text(json.dumps(resumed_from, indent=1))")
    rr = src.split("def resolve_resume(")[1].split("\ndef ")[0]
    assert rr.index('recovery.get("action") == "completed"') \
        < rr.index("all(r.exists() for r in roots) and any((r / RESUMED_FROM_FILE).is_file() for r in roots)") < rr.index("import_prior_stores(")

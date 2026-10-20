"""Follow-up to the geometry chain's fix (tests/test_geometry_archives_land_in_their_own_directories.py):
reproduce/03_hpo.sh's probe_archive_local archived the mirror's menu / verdict / fetch record / geometry record / DONE marker into
`$LOCAL_ROOT/superseded/<UTC to the second>/` and the site's geometry copy into `$SISTEER_SITE_DIR/geometry/superseded/<the same
second>/` with `mkdir -p && mv -f` — the stage-02 pools-marker defect in the probe chain, and into the very geometry/superseded/
base the geometry chain archives into. Two archivals inside one second named the same directory and the second `mv -f` wrote the
first archive over. probe_recover_planless (the plan-less FAILED_START recovery, over ssh) did the same on the record.

Now ONE archival takes ONE stamp (lib/common.sh repro_superseded_stamp: the grading host clock to the microsecond, fixed width, the one
format every superseded/ directory is named in) and each place RESERVES its directory — the two local places through
repro_superseded_dir with that stamp (a bare mkdir; the next `-NN` suffix only where THAT place collides, as the geometry
revocation does across the site and its clusters), the record over ssh by the same rule inlined (a remote cannot call a grading host
shell function), reporting the directory it took as `ARCHIVE_DIR=<dir>`, which the front door's line names. A place with nothing to
archive reserves nothing. A dry run reserves nothing and says `superseded/<utc>/` as before.

The map of 03_hpo.sh's `superseded/` archival sites (the whole file grepped, 2026-10-09), pinned below so a new one is a decision:
  - probe_archive_local — FIXED here (two places, one stamp, repro_superseded_dir at each).
  - probe_recover_planless — FIXED here (the record over ssh, the reservation inlined); its generation fence already made a second
    archival of ONE record in one instant unreachable (probe_attempt.json is gone once moved), so the reservation holds the rule where
    the fence does not reach: a directory an operator or an earlier run left under the same stamp.
  - recon_archive_record / recon_reset_local — already one directory per archival (`<second>.<nanoseconds>.<reason>`, an existence check,
    `mv -n` with a post-check) that REFUSES a collision by name; tests/test_front_door_recon_cell.py pins the refusal and
    tests/test_front_door_recon_verdict_revalidation.py the `.<reason>` suffix. Left as they are.
  - the addendum --resubmit transition (`utc4`; transition_close, addendum_restore_superseded, the mirror's `superseded.r4.<utc>.*`) —
    `superseded/<utc>/` on the record under the record's relaunch LOCK with an open-receipt guard and `mv -n`; the receipt names the
    directory and the rollback reads it back by that name: one transition per record at a time by construction. Left as it is.
  - `$FIT1M.superseded.<utc>.json` (addendum_menu_refit) — a FILE rename with `mv -n`, not an archive directory. Left.
  - probe_submit_reverdict's `stamp` — a unit-name suffix (`.rv<utc>`), not an archive.
Readers: override_recon_menu globs `$LOCAL_ROOT/superseded/*/allowed_pairs.json` newest first — the microsecond stamp with `-NN` keeps
lexical order = archival order; the --rehome refusals name `superseded/<utc>/ by hand` (prose, unchanged).

A FAILED ARCHIVAL IS A REFUSAL, NEVER A WARNING THE CALLER PROCEEDS PAST: the first cut of the reservation turned a
place that could not be reserved into a warning and a `return 0`, so with the mirror's superseded/ unwritable (the mirror itself
writable) probe_fetch_ready went on to write the fetch record, os.replace the menu / verdict / geometry and write the site copy over —
the old evidence lost, not archived; the reopen verbs would have reopened the record past it. Now probe_archive_local reserves EVERY
place before anything moves (a second place that fails gives the first, still empty, back), moves the site copy first and the local
files after (the one partial state a failed move can leave is one the next run revalidates and archives again), and any reservation
or move that fails is repro_fatal: the operation is refused by name with nothing deleted, what stands and what moved named, and the
remedy; the caller never fetches or reopens over unarchived evidence. Pinned at the fragment (the three failure shapes) and
end-to-end against the real stage (the DONE-marker revalidation's archival and the in-fetch archival), where the fetch does not proceed
and a writable directory lets the same run archive and re-fetch.

EVERY FAILURE POINT LEAVES A STATE THE NEXT RUN RECOVERS BY ITSELF: the earlier cut moved the DONE marker LAST and
its loop went on past a failed move, so a state directory made read-only (the marker's; the mirror under it writable) left the marker
standing with every mirror file archived — the next run refused at probe_home (evidence with no fetch record naming its home), --rehome
refused too (no verdict to read the home from), and the probe could not retry without a hand's move. Now the archival moves in the
REVERSE of the order the fetch writes (probe_fetch_ready: the fetch record, the files with the evidence last, the site copy, the marker)
— the DONE marker first, the site's geometry copy, the evidence (geometry record, verdict, menu), the fetch record LAST — and the
sequence STOPS at the first move that fails, so what stands after ANY failure is a prefix of the fetch's own write order: a state an
interrupted fetch leaves, which the next run already recovers (a standing marker means nothing moved: revalidated, the archival made
again; evidence or a site copy beside the fetch record, no marker: the home resolves and the fetch archives it; a fetch record alone:
archived by that same in-fetch archival, never written over). Before anything is reserved, every directory a file leaves is checked
writable and searchable, so the failures that happen in practice (a read-only state directory, mirror or site) refuse with NOTHING
moved and nothing reserved; the order covers what no check can see (a device full under a cross-device superseded/, an immutable file,
a race), injected here as a `mv` on PATH that refuses ONE source path. Pinned at the fragment (the three read-only directories; a
refused move at each of the six positions, the standing files always a prefix of the fetch's order) and end to end (the read-only
state directory, then the fault cleared and the next run recovering; a refused move mid-sequence and at the last position, then the
next run archiving what stands under a second stamp — every old byte under superseded/, nothing deleted).

Frozen clock: a `date` on PATH that answers every archive-stamp format with one instant (tests/test_stage02_pools_rerunnable.py's
_frozen_date_bin), so two archivals fall in the same microsecond — the collision a real clock never produces and a frozen one always
does; before the fix each pair collapsed into one directory holding the second archive alone. The functions run as the fragments
of 03_hpo.sh they are (the pattern of tests/test_front_door_recon_cell.py), over a synthetic mirror, site and state root under
tmp_path and a recording `ssh` that runs its command locally; nothing reaches a cluster. The end-to-end tests borrow
tests/test_front_door_probe_resume.py's probe harness (fake ssh / rsync / squeue on this host; the real status reader).
"""
from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_front_door_probe_resume import MODEL as E2E_MODEL, SRV_ID, World, _stamp, _with_parity, world  # noqa: E402,F401  (the probe harness and its fixture)
from test_stage02_pools_rerunnable import _frozen_date_bin  # noqa: E402

#: root ignores directory modes, so an unwritable superseded/ cannot be staged there (the Modal CPU suite runs as root)
_needs_modes = pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory modes")

REPO = Path(__file__).resolve().parents[1]
HPO = REPO / "reproduce/03_hpo.sh"
#: an archive directory's name: the UTC stamp to the microsecond (repro_superseded_stamp); `-NN` is appended only on a collision
STAMP_RE = r"\d{8}T\d{6}\.\d{6}Z"
#: the one instant _frozen_date_bin answers every archive-stamp format with
FROZEN = datetime.datetime.fromtimestamp(1790000000, datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.000000Z")
MODEL = "testmodel"
LOCAL_FILES = ("allowed_pairs.json", "probe_verdict.json", "probe_fetch.json", "geometry_record.json")


def _fragment(start: str, end: str) -> str:
    src = HPO.read_text()
    return src[src.index(start):src.index(end)]


def _run(script: str, bin_dir: Path, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    """`script` under bash with `bin_dir` (the frozen `date`, and the local `ssh` where a test adds one) at the front of PATH."""
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(bin_dir.parent), **(env_extra or {})}
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=120)


# ---- probe_archive_local: the mirror and the site's geometry copy -------------------------------------------------------------

def _world(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A mirror, a site and a state root; the DONE marker lives under the state root as 03_hpo.sh places it."""
    local = tmp_path / "mirror"; local.mkdir()
    site = tmp_path / "site"; (site / "geometry").mkdir(parents=True)
    state = tmp_path / "state"; (state / MODEL / "03_hpo").mkdir(parents=True)
    return local, site, state


def _lay(local: Path, site: Path, state: Path, tag: str) -> dict[str, bytes]:
    """Every file probe_archive_local moves, each stamped with `tag` so an archive's bytes tell which archival it came from."""
    laid = {}
    for name in LOCAL_FILES:
        (local / name).write_text(json.dumps({"file": name, "round": tag, "generation": "a" * 64}))
        laid[name] = (local / name).read_bytes()
    marker = state / MODEL / "03_hpo" / "DONE.lora.probe"
    marker.write_text(f"generation={'a' * 64} round={tag}\n"); laid["DONE.lora.probe"] = marker.read_bytes()
    (site / "geometry" / f"{MODEL}.json").write_text(json.dumps({"geometry": tag})); laid[f"{MODEL}.json"] = (site / "geometry" / f"{MODEL}.json").read_bytes()
    return laid


def _archive_script(local: Path, site: Path, state: Path, body: str) -> str:
    """probe_archive_local as 03_hpo.sh defines it, over lib/common.sh and stubs for the helpers of 03_hpo.sh it reads."""
    frag = _fragment("probe_archive_local(){", "probe_take_lock(){")
    return (f"set -uo pipefail; source {REPO}/reproduce/lib/common.sh\n"
            f"PY=python3; MODEL={MODEL}; STAGE=03_hpo; CLASS=lora; CLUSTER=cluster_a; CL_ROOT=/shared/exp/reproduce/{MODEL}/03_hpo/lora\n"
            f"LOCAL_ROOT={local}; SISTEER_SITE_DIR={site}; SISTEER_STATE_DIR={state}\n"
            'disp(){ echo "$1"; }\n'
            'fetch_invalidate(){ echo "fetch-cache invalidated for $1 -> $2"; }\n'
            'json_field(){ printf "%s" "$1" | python3 -c \'import json,sys; raw=sys.stdin.read(); d=json.loads(raw) if raw.strip() else {}; v=d.get(sys.argv[1]); print("" if v is None else v)\' "$2"; }\n'
            f'probe_local_generation(){{ echo {"a" * 64}; }}\n'
            f"{frag}\n{body}")


def test_two_probe_archivals_in_one_instant_keep_both_archives_at_the_mirror_and_the_site(tmp_path):
    """probe_archive_local twice in one frozen instant (the reopen verbs' `deliberate` path: the mirror is stale by construction): at
    the mirror AND at the site's geometry/superseded/ the first archival's files stand beside the second's (`<stamp>`, then
    `<stamp>-02`, sorted = archival order, each holding the bytes it archived), the stamp is the SAME at both places (one archival,
    one stamp), and each archival's lines name the directory each place took. Before the fix both archivals named
    `superseded/<second>/` at both places and the second `mv -f` left one directory holding the second archive alone."""
    local, site, state = _world(tmp_path); bin_dir = _frozen_date_bin(tmp_path)
    status = json.dumps({"state": "READY", "generation": "b" * 64})
    laid, outs = [], []
    for tag in ("first", "second"):
        laid.append(_lay(local, site, state, tag))
        r = _run(_archive_script(local, site, state, f"probe_archive_local probe READY 'reopen requested (--reopen)' '{status}' deliberate; echo rc=$?"), bin_dir)
        out = r.stdout + r.stderr
        assert r.returncode == 0 and "rc=0" in r.stdout, out
        assert "archived under" in out and "nothing deleted" in out and "the DONE marker is cleared" in out and "fetch-cache invalidated" in out, out
        assert not any((local / n).exists() for n in LOCAL_FILES) and not (state / MODEL / "03_hpo" / "DONE.lora.probe").exists()
        assert not (site / "geometry" / f"{MODEL}.json").exists()
        assert "refused under --dry-run" not in out
        outs.append(out)
    # THE DEFECT: both archivals fell in one instant — two directories at each place, the first archive intact beside the second (one
    # directory holding the second archive alone is what the old `mkdir -p superseded/<second> && mv -f` left)
    for base, names_in in ((local / "superseded", LOCAL_FILES + ("DONE.lora.probe",)), (site / "geometry" / "superseded", (f"{MODEL}.json",))):
        names = sorted(p.name for p in base.iterdir())
        assert len(names) == 2 and re.fullmatch(STAMP_RE, names[0]) and names[1] == f"{names[0]}-02", (base, names)   # the same instant, twice
        assert names[0] == FROZEN
        for i, n in enumerate(names):
            assert sorted(p.name for p in (base / n).iterdir()) == sorted(names_in), (base / n)
            assert {f: (base / n / f).read_bytes() for f in names_in} == {f: laid[i][f] for f in names_in}            # the first archive is intact
    # and each archival named the directory each place took
    for i, out in enumerate(outs):
        suffix = "" if i == 0 else "-02"
        assert f"5 local file(s) archived under {local}/superseded/{FROZEN}{suffix}; the site's geometry copy under {site}/geometry/superseded/{FROZEN}{suffix} (nothing deleted)" in out, out
        assert f"the site's geometry copy {site}/geometry/{MODEL}.json -> {site}/geometry/superseded/{FROZEN}{suffix}/" in out, out


def test_a_place_with_nothing_to_archive_reserves_no_directory(tmp_path):
    """Only the site's copy exists (the mirror was already archived; a fold landed on the site since): the site's geometry/superseded/
    takes one directory, the mirror's superseded/ is not made at all — no empty directory, no second one "for nothing" (the probe
    phase's own test holds the mirror to one directory across a re-run) — and the line says so."""
    local, site, state = _world(tmp_path); bin_dir = _frozen_date_bin(tmp_path)
    (site / "geometry" / f"{MODEL}.json").write_text(json.dumps({"geometry": "a fold"}))
    status = json.dumps({"state": "READY", "generation": "b" * 64})
    r = _run(_archive_script(local, site, state, f"probe_archive_local probe READY 'reopen requested (--reopen)' '{status}' deliberate; echo rc=$?"), bin_dir)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "rc=0" in r.stdout, out
    assert not (local / "superseded").exists(), sorted((local / "superseded").iterdir())
    assert [p.name for p in (site / "geometry" / "superseded").iterdir()] == [FROZEN] and (site / "geometry" / "superseded" / FROZEN / f"{MODEL}.json").is_file()
    assert f"no local file to archive (no directory reserved under {local}/superseded/); the site's geometry copy under {site}/geometry/superseded/{FROZEN} (nothing deleted)" in out, out


def test_a_dry_run_of_the_probe_archival_reserves_nothing_and_says_what_would_move(tmp_path):
    """Under --dry-run probe_archive_local prints what WOULD move under `superseded/<utc>/` (no stamp is minted, so none is named) and
    reserves nothing at either place: no superseded/ directory, every file where it was, and repro_superseded_dir's own dry-run
    refusal never heard (the dry-run branch returns before it)."""
    local, site, state = _world(tmp_path); bin_dir = _frozen_date_bin(tmp_path)
    laid = _lay(local, site, state, "live")
    status = json.dumps({"state": "READY", "generation": "b" * 64})
    r = _run(_archive_script(local, site, state, f"probe_archive_local probe READY 'reopen requested (--reopen)' '{status}' deliberate; echo rc=$?"), bin_dir, {"REPRO_DRY_RUN": "1"})
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "rc=0" in r.stdout, out
    assert (f"DRY-RUN: probe (probe): would archive allowed_pairs.json probe_verdict.json probe_fetch.json geometry_record.json DONE.lora.probe site:geometry/{MODEL}.json "
            f"under {local}/superseded/<utc>/ (the site copy under geometry/superseded/<utc>/) — reopen requested (--reopen); NOTHING moved under a dry run") in out, out
    assert "refused under --dry-run" not in out and "archived under" not in out and "fetch-cache invalidated" not in out
    assert not (local / "superseded").exists() and not (site / "geometry" / "superseded").exists()
    assert all((local / n).read_bytes() == laid[n] for n in LOCAL_FILES) and (site / "geometry" / f"{MODEL}.json").read_bytes() == laid[f"{MODEL}.json"]
    assert (state / MODEL / "03_hpo" / "DONE.lora.probe").read_bytes() == laid["DONE.lora.probe"]


# ---- probe_archive_local: a place that cannot take the archive REFUSES the operation ---------------------------

#: the non-deliberate path (probe_fetch_ready's, the DONE-marker revalidation's): a record whose identity differs from the mirror's
_MISMATCH_STUBS = 'probe_remote_identity(){ echo "r1 r2 r3"; }\nprobe_local_identity(){ echo "l1 l2 l3"; }\n'
_REFUSE_BODY = _MISMATCH_STUBS + "probe_archive_local probe READY 'the mirror is not the record' '%s'; echo PROCEEDED rc=$?"


def _everything_stands(local: Path, site: Path, state: Path, laid: dict[str, bytes]) -> bool:
    """Every file probe_archive_local would move is where it was, byte for byte."""
    return (all((local / n).read_bytes() == laid[n] for n in LOCAL_FILES)
            and (state / MODEL / "03_hpo" / "DONE.lora.probe").read_bytes() == laid["DONE.lora.probe"]
            and (site / "geometry" / f"{MODEL}.json").read_bytes() == laid[f"{MODEL}.json"])


@_needs_modes
def test_an_unwritable_superseded_directory_at_the_mirror_refuses_with_nothing_moved_and_the_caller_does_not_proceed(tmp_path):
    """The mirror's superseded/ exists and is not writable while the mirror is: the reservation fails, the
    function exits through repro_fatal (rc 2, the caller's next line never runs — a fetch or a reopen does not proceed), every file
    stands byte for byte (the DONE marker too), the site copy was never touched and NO directory was reserved at either
    place (the site is reserved only after the mirror); the line names the directory and the remedy. Before: a warning, `return 0`,
    and probe_fetch_ready wrote the record over the unarchived mirror."""
    local, site, state = _world(tmp_path); bin_dir = _frozen_date_bin(tmp_path)
    laid = _lay(local, site, state, "live")
    status = json.dumps({"state": "READY", "generation": "b" * 64})
    sup = local / "superseded"; sup.mkdir(); sup.chmod(0o555)
    try:
        r = _run(_archive_script(local, site, state, _REFUSE_BODY % status), bin_dir)
    finally:
        sup.chmod(0o755)
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "PROCEEDED" not in r.stdout, out
    assert (f"REFUSED: probe (probe): no archive directory could be reserved under {local}/superseded/ for the 5 local file(s) (the mirror's superseded/ is not a writable directory) "
            f"— NOTHING moved; the DONE marker stands (it is revalidated again on the next run); probe is refused rather than fetch or reopen over unarchived evidence: "
            f"make {local}/superseded/ writable (or move it aside by hand, never delete) and re-run. the mirror is not the record") in r.stderr, out
    assert _everything_stands(local, site, state, laid)
    assert list(sup.iterdir()) == [] and not (site / "geometry" / "superseded").exists()
    assert "archived under" not in out and "fetch-cache invalidated" not in out and "DONE marker is cleared" not in out, out


@_needs_modes
def test_an_unwritable_superseded_directory_at_the_site_refuses_and_gives_the_mirrors_reservation_back(tmp_path):
    """The site's geometry/superseded/ is not writable, the mirror's is: the mirror's directory is reserved first and the site's
    reservation fails — the function gives the mirror's (empty) directory back, moves nothing, and refuses by name (rc 2); every file
    stands, neither place holds a directory. Before: the mirror's five files moved, the site copy "STANDS" in a warning, and the
    caller went on to fetch — geometry_to_site writing the site copy over."""
    local, site, state = _world(tmp_path); bin_dir = _frozen_date_bin(tmp_path)
    laid = _lay(local, site, state, "live")
    status = json.dumps({"state": "READY", "generation": "b" * 64})
    ssup = site / "geometry" / "superseded"; ssup.mkdir(); ssup.chmod(0o555)
    try:
        r = _run(_archive_script(local, site, state, _REFUSE_BODY % status), bin_dir)
    finally:
        ssup.chmod(0o755)
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "PROCEEDED" not in r.stdout, out
    assert (f"REFUSED: probe (probe): no archive directory could be reserved under {site}/geometry/superseded/ for the site's geometry copy {site}/geometry/{MODEL}.json (not a writable directory) "
            f"— NOTHING moved; the DONE marker stands (it is revalidated again on the next run) (the mirror's own reservation under {local}/superseded/ given back); "
            f"probe is refused rather than fetch or reopen over an unarchived fold: make {site}/geometry/superseded/ writable (or move the copy aside by hand, never delete) and re-run.") in r.stderr, out
    assert _everything_stands(local, site, state, laid)
    assert list((local / "superseded").iterdir()) == [] and list(ssup.iterdir()) == []          # the mirror's reservation given back, nothing at the site
    assert "archived under" not in out and "fetch-cache invalidated" not in out, out


#: what the archival would move, as its refusals list it — the move order (the marker first, the site copy, the evidence, the fetch record last)
_WOULD_MOVE = f"DONE.lora.probe site:geometry/{MODEL}.json geometry_record.json probe_verdict.json allowed_pairs.json probe_fetch.json"


def _move_order(local: Path, site: Path, state: Path) -> list[Path]:
    """The six paths probe_archive_local moves, in the order it moves them: the reverse of the fetch's write order."""
    return [state / MODEL / "03_hpo" / "DONE.lora.probe", site / "geometry" / f"{MODEL}.json", local / "geometry_record.json",
            local / "probe_verdict.json", local / "allowed_pairs.json", local / "probe_fetch.json"]


def _failing_mv(bin_dir: Path) -> None:
    """A `mv` for the front of PATH that refuses ONE source path (FAKE_MV_FAIL_SRC, compared whole against every argument) and
    delegates everything else to the real mv — a rename refused past every check, the shape of a device full under a cross-device
    superseded/, an immutable file or a race between the check and the move."""
    real = shutil.which("mv")
    assert real, "no mv on PATH"
    (bin_dir / "mv").write_text('#!/bin/bash\nfor a in "$@"; do [ "$a" = "${FAKE_MV_FAIL_SRC:-}" ] && { echo "mv: cannot move \'$a\': test fault (FAKE_MV_FAIL_SRC)" >&2; exit 1; }; done\n'
                                f'exec {real} "$@"\n')
    (bin_dir / "mv").chmod(0o755)


@_needs_modes
@pytest.mark.parametrize("which", ["the marker's state directory", "the mirror", "the site's geometry directory"])
def test_a_directory_a_file_cannot_leave_refuses_before_anything_is_reserved_or_moved(tmp_path, which):
    """At the fragment: the directory one of the files must LEAVE is not writable (a rename unlinks its source
    there) — the marker's state directory (with the mirror under it writable), the mirror itself (its
    superseded/ writable, so the reservation would have succeeded), or the site's geometry directory. The pre-check refuses (rc 2)
    BEFORE any directory is reserved and before any move: every file stands byte for byte, no superseded/ entry at either place, the
    line names the directory, what the archival would move (in its order), that the DONE marker stands, and the remedy; the caller does
    not proceed and the fetch cache is not touched. On a1db16018 the marker's directory read-only moved the site copy and four mirror
    files and failed on the marker, which the next run could not get past."""
    local, site, state = _world(tmp_path); bin_dir = _frozen_date_bin(tmp_path)
    laid = _lay(local, site, state, "live")
    status = json.dumps({"state": "READY", "generation": "b" * 64})
    sup = local / "superseded"; sup.mkdir(); ssup = site / "geometry" / "superseded"; ssup.mkdir()     # both reservations WOULD succeed
    ro = {"the marker's state directory": state / MODEL / "03_hpo", "the mirror": local, "the site's geometry directory": site / "geometry"}[which]
    ro.chmod(0o555)
    try:
        r = _run(_archive_script(local, site, state, _REFUSE_BODY % status), bin_dir)
    finally:
        ro.chmod(0o755)
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "PROCEEDED" not in r.stdout, out
    assert (f"REFUSED: probe (probe): NOTHING moved and nothing reserved — a file cannot leave a directory that is not writable: {ro}; the archival would move {_WOULD_MOVE}; "
            f"the DONE marker stands (it is revalidated again on the next run); probe is refused rather than fetch or reopen over unarchived evidence: "
            f"make the directory writable and re-run (the archival is then made whole, under one stamp). the mirror is not the record") in r.stderr, out
    assert _everything_stands(local, site, state, laid)
    assert list(sup.iterdir()) == [] and list(ssup.iterdir()) == []                                      # nothing reserved
    assert "archived under" not in out and "fetch-cache invalidated" not in out and "DONE marker is cleared" not in out and "could NOT be moved" not in out, out


@pytest.mark.parametrize("k", range(6), ids=["marker", "site-copy", "geometry-record", "verdict", "menu", "fetch-record"])
def test_a_move_refused_past_the_check_ends_the_sequence_and_what_stands_is_a_prefix_of_the_fetchs_order(tmp_path, k):
    """A rename refused where no pre-check can see it (the `mv` on PATH refuses ONE source), at each of the six positions of the
    archival's order. The sequence ENDS at that move: everything before it is archived (byte for byte, under the reserved directories),
    everything from it on stands byte for byte where it was, a reserved directory left empty is given back, and the refusal (rc 2)
    names the file, where the moved ones now are, what stands, how the next run recovers it and the remedy; the caller does not
    proceed and the fetch cache is not touched. The invariant that makes every one of the six recoverable without a hand's move: the
    standing files are always a PREFIX of the fetch's write order — a standing DONE marker means nothing moved, and the fetch record
    stands whenever anything else does (it is the last to go), so probe_home always resolves the mirror's home. On a1db16018 the loop
    went on past the failed move (the menu's refused move left the menu standing alone, with no fetch record) and the marker went
    last (a refused marker stood over an emptied mirror)."""
    local, site, state = _world(tmp_path); bin_dir = _frozen_date_bin(tmp_path); _failing_mv(bin_dir)
    laid = _lay(local, site, state, "live")
    status = json.dumps({"state": "READY", "generation": "b" * 64})
    order = _move_order(local, site, state); fails = order[k]; moved, standing = order[:k], order[k:]
    r = _run(_archive_script(local, site, state, _REFUSE_BODY % status), bin_dir, {"FAKE_MV_FAIL_SRC": str(fails)})
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "PROCEEDED" not in r.stdout, out
    arch, sarch = local / "superseded" / FROZEN, site / "geometry" / "superseded" / FROZEN
    dst = sarch if fails == order[1] else arch
    now_at = " ".join(str((sarch if p == order[1] else arch) / p.name) for p in moved)
    rest = " ".join(str(p) for p in standing[1:])
    assert (f"REFUSED: probe (probe): {fails} could NOT be moved into {dst}/ — it stands, and so does everything after it in the archival's order{f' ({rest})' if rest else ''}; "
            f"{'moved before it, now at: ' + now_at if moved else 'nothing moved before it'} (nothing deleted). What stands is what a fetch interrupted here leaves "
            f"(the marker goes first, the fetch record last) and the next run recovers it by itself — a standing DONE marker is revalidated and the archival made again; "
            f"evidence or a site copy beside the fetch record is archived by the fetch itself, a fetch record alone too; probe is refused rather than fetch or reopen over "
            f"unarchived evidence: fix what refused the move ({fails.parent} writable, {dst} on the same device) and re-run. the mirror is not the record") in r.stderr, out
    for p in standing:
        assert p.read_bytes() == laid[p.name], p                                                        # standing, byte for byte
    for p in moved:
        assert not p.exists() and ((sarch if p == order[1] else arch) / p.name).read_bytes() == laid[p.name], p   # archived, byte for byte
    # a reserved directory that holds nothing is given back; one that holds an archive stays
    assert arch.exists() == any(p != order[1] for p in moved) and sarch.exists() == (order[1] in moved), (sorted((local / "superseded").iterdir()), sorted((site / "geometry" / "superseded").iterdir()))
    assert "fetch-cache invalidated" not in out and "DONE marker is cleared" not in out and "archived under" not in out, out
    # THE INVARIANT: the standing files are a prefix of the fetch's write order
    marker, fetch_record = order[0], order[5]
    if marker.exists(): assert all(p.exists() for p in order), "a standing marker means nothing moved"
    if any(p.exists() for p in order[:5]): assert fetch_record.exists(), "the fetch record is the last to go: whatever stands has its home"


# ---- end to end: the real stage does not fetch over a mirror it could not archive ------------------------------------------------

def _record_moves_on(w: World, gen1: str) -> str:
    """The record re-verdicted at a NEW result generation (a launch re-run, the verdict + menu rewritten and stamped), as
    tests/test_front_door_probe_resume.py's DONE-marker revalidation test does."""
    (w.record / "launches" / "L07.json").write_text(json.dumps({"launch_id": "L07", "status": "passed", "host": "TESTHOST", "rerun": True}))
    for name in ("probe_verdict.json", "allowed_pairs.json"):
        d = json.loads((w.record / name).read_text()); d["written_utc"] = "2026-09-30T03:00:00Z"; (w.record / name).write_text(json.dumps(d))
    gen2 = _stamp(w)
    assert gen2 != gen1
    return gen2


def _mirror_bytes(w: World, names=("allowed_pairs.json", "probe_verdict.json", "probe_fetch.json")) -> dict[str, bytes]:
    return {n: (w.state / n).read_bytes() for n in names}


E2E_MIRROR = ("allowed_pairs.json", "probe_verdict.json", "probe_fetch.json", "geometry_record.json")


def _lay_geometry(w: World, gen: str) -> None:
    """The record's geometry record, folded at result generation `gen` (tests/test_front_door_probe_resume.py's construction), so the
    fetch copies it to the site and the archival has a site copy to move."""
    try:
        from sisteer.capeval.serving import geometry as g
    except ImportError:                                                     # pragma: no cover — the geometry package is the site copy's validator
        pytest.skip("sisteer.capeval.serving.geometry not importable")
    budget = g.ModelBudget(model_key=E2E_MODEL, batched_token_budget=8 * 4096, certified=True, source=g.MEASURED_BUDGET_SOURCE,
                           basis=g.BudgetBasis(method=f"{g.SITE_GEOMETRY_METHOD}: test", max_model_len=4096, max_num_seqs=8, gpu_memory_utilization=0.9,
                                               kv_tokens_reported=None, probe_job_id="11111111", measured_on="2026-09-30", source="test"))
    sets = {"wo": {"lora_modules": ["o_proj"], "lora_max_module_dim": 4096, "budget": budget, "allowed_pairs": [[4, "wo"]],
                   "certified_by": ["L01"], "module_dim_launches": {"L01": 4096}, "budget_launches": {"L01": {"S": 8, "L": 4096, "batched_tokens": 8 * 4096}}}}
    (w.record / "geometry_record.json").write_text(json.dumps(_with_parity(g.site_geometry_record(
        model_key=E2E_MODEL, hf_id=SRV_ID, revision="0" * 40, written_utc="2026-09-30T02:00:00Z", sets=sets,
        provenance={"probe_job_id": "11111111", "probe_record": str(w.record), "host": "TESTHOST", "result_generation": gen,
                    "attestation_sha256": {"L01": {"file": None, "sha256": "ab" * 32}}}))))


def _fetched_with_geometry(w: World) -> tuple[str, Path, Path]:
    """A READY record with a geometry record, fetched: -> (generation, the DONE marker, the site copy), all three in place."""
    w.lay(done=7, complete=True, menu=True, verdict=True); gen1 = _stamp(w); _lay_geometry(w, gen1)
    r = w.run(); assert r.returncode == 0, r.stdout + r.stderr
    marker = w.tmp / "state" / E2E_MODEL / "03_hpo" / "DONE.lora.probe"; site_copy = w.site / "geometry" / f"{E2E_MODEL}.json"
    assert marker.is_file() and site_copy.is_file() and json.loads((w.state / "probe_fetch.json").read_text())["generation"] == gen1
    return gen1, marker, site_copy


def _archives(base: Path) -> dict[str, dict[str, bytes]]:
    """Every archive directory under `base`, by name, with the bytes of each file it holds."""
    return {d.name: {p.name: p.read_bytes() for p in d.iterdir()} for d in sorted(base.iterdir())} if base.exists() else {}


@_needs_modes
def test_the_probe_phase_with_a_done_marker_refuses_to_fetch_over_a_mirror_whose_superseded_directory_cannot_take_the_archive(world):
    """End to end: a fetched mirror with its DONE marker, the record re-verdicted at a new generation (READY), the
    mirror's superseded/ unwritable. `--phase probe` revalidates the marker, finds the mirror stale, and REFUSES (rc 2) at the archival:
    the menu, verdict and fetch record stand byte for byte at the old generation, the marker stands, nothing is staged (the refusal
    precedes the fetch), nothing submitted, superseded/ empty. Writable again, the same run archives the old copies and re-fetches the
    new generation. Before the fix the first run archived nothing, wrote the fetch record, os.replaced the menu / verdict and marked
    DONE at the new generation — the old evidence lost."""
    w = world
    w.lay(done=7, complete=True, menu=True, verdict=True)
    r = w.run(); assert r.returncode == 0, r.stdout + r.stderr
    marker = w.tmp / "state" / E2E_MODEL / "03_hpo" / "DONE.lora.probe"
    gen1 = json.loads((w.state / "probe_fetch.json").read_text())["generation"]
    before, marker_before = _mirror_bytes(w), marker.read_bytes()
    gen2 = _record_moves_on(w, gen1)
    sup = w.state / "superseded"; sup.mkdir(); sup.chmod(0o555)
    try:
        r = w.run()
    finally:
        sup.chmod(0o755)
    out = r.stdout + r.stderr
    assert r.returncode == 2, out
    assert (f"REFUSED: probe (probe): no archive directory could be reserved under {sup}/ for the 4 local file(s)" in r.stderr
            and "NOTHING moved; the DONE marker stands (it is revalidated again on the next run); probe is refused rather than fetch or reopen over unarchived evidence" in r.stderr), out
    assert "READY on cluster_a" not in out and "fetched to" not in out and "archived under" not in out and "DONE marker is cleared" not in out, out
    assert _mirror_bytes(w) == before and marker.read_bytes() == marker_before                       # the old evidence, untouched
    assert json.loads((w.state / "probe_fetch.json").read_text())["generation"] == gen1
    assert list(sup.iterdir()) == [] and not (w.state / ".fetch.staging").exists() and w.sbatch_calls() == []
    # the remedy: a writable superseded/ — the same run archives the stale copies beside nothing and re-fetches the new generation
    r = w.run()
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "archived under" in out and "READY on cluster_a" in out and f"generation={gen2}" in marker.read_text(), out
    arch = sorted(sup.iterdir())
    assert len(arch) == 1 and json.loads((arch[0] / "probe_fetch.json").read_text())["generation"] == gen1 and (arch[0] / "allowed_pairs.json").read_bytes() == before["allowed_pairs.json"]
    assert json.loads((w.state / "probe_fetch.json").read_text())["generation"] == gen2


@_needs_modes
def test_the_in_fetch_archival_refuses_before_the_fetch_record_and_the_exposure(world):
    """The archival INSIDE probe_fetch_ready (a mirror holding evidence with a fetch record but no DONE marker — an earlier fetch
    interrupted after its exposure, a --rehome): the record is READY at a new generation, the staged copy verified, and the mirror's
    stale evidence must be archived before the exposure — superseded/ unwritable, so the fetch REFUSES there: no fetch record
    rewritten, nothing exposed (the staged files stay in .fetch.staging, which the next fetch removes first), no marker, nothing
    submitted. Writable again, the next run archives and completes the fetch. Before the fix: a warning, then the fetch record, the
    os.replace of every evidence file and the marker — the old evidence lost."""
    w = world
    w.lay(done=7, complete=True, menu=True, verdict=True)
    r = w.run(); assert r.returncode == 0, r.stdout + r.stderr
    marker = w.tmp / "state" / E2E_MODEL / "03_hpo" / "DONE.lora.probe"
    gen1 = json.loads((w.state / "probe_fetch.json").read_text())["generation"]
    marker.unlink()                                                                                   # evidence + fetch record, no marker
    before = _mirror_bytes(w)
    gen2 = _record_moves_on(w, gen1)
    sup = w.state / "superseded"; sup.mkdir(); sup.chmod(0o555)
    try:
        r = w.run()
    finally:
        sup.chmod(0o755)
    out = r.stdout + r.stderr
    assert r.returncode == 2, out
    assert f"REFUSED: probe (probe): no archive directory could be reserved under {sup}/ for the 3 local file(s)" in r.stderr and "NOTHING moved; probe is refused rather than fetch or reopen over unarchived evidence" in r.stderr, out
    assert "the mirror (" in r.stderr and "archived before the re-fetch" in r.stderr, out              # the in-fetch archival's own detail
    assert "fetched to" not in out and "archived under" not in out, out
    assert _mirror_bytes(w) == before and not marker.exists() and list(sup.iterdir()) == [] and w.sbatch_calls() == []
    assert (w.state / ".fetch.staging" / "allowed_pairs.json").is_file()                            # staged, never exposed
    r = w.run()
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "archived under" in out and "READY on cluster_a" in out and f"generation={gen2}" in marker.read_text(), out
    assert not (w.state / ".fetch.staging").exists() and json.loads((w.state / "probe_fetch.json").read_text())["generation"] == gen2
    arch = sorted(sup.iterdir())
    assert len(arch) == 1 and (arch[0] / "allowed_pairs.json").read_bytes() == before["allowed_pairs.json"] and not (arch[0] / "DONE.lora.probe").exists()


# ---- end to end: every failure of the archival leaves a state the NEXT run recovers by itself --------------------------------------

@_needs_modes
def test_a_read_only_state_directory_refuses_with_nothing_moved_and_the_next_run_recovers_once_it_is_writable(world):
    """The reviewer's sequence, end to end with the real stage: a fetched mirror with its DONE marker, a geometry record and the
    site's copy; the record re-verdicted at a new generation; the STATE directory (the marker's — the mirror under it stays writable)
    made read-only. `--phase probe` finds the mirror stale and REFUSES before anything is reserved or moved: every file stands byte
    for byte, no superseded/ at either place, nothing fetched. The directory writable again, the NEXT run recovers by itself: it
    revalidates the standing marker, archives all five local files and the site copy under ONE stamp (every old byte there — nothing
    deleted) and re-fetches the new generation. On a1db16018 the first run archived the four mirror files and failed on the marker;
    the next run refused at probe_home (the marker with no fetch record naming its home), --rehome refused (no verdict to read the
    home from), and the only remedy named was a hand's move."""
    w = world
    gen1, marker, site_copy = _fetched_with_geometry(w)
    mdir = marker.parent
    before, marker_before, site_before = _mirror_bytes(w, E2E_MIRROR), marker.read_bytes(), site_copy.read_bytes()
    gen2 = _record_moves_on(w, gen1); _lay_geometry(w, gen2)
    mdir.chmod(0o555)
    try:
        r = w.run()
    finally:
        mdir.chmod(0o755)
    out = r.stdout + r.stderr
    assert r.returncode == 2, out
    assert f"REFUSED: probe (probe): NOTHING moved and nothing reserved — a file cannot leave a directory that is not writable: {mdir}; the archival would move DONE.lora.probe site:geometry/{E2E_MODEL}.json geometry_record.json probe_verdict.json allowed_pairs.json probe_fetch.json; the DONE marker stands" in r.stderr, out
    assert "READY on cluster_a" not in out and "fetched to" not in out and "archived under" not in out and "could NOT be moved" not in out, out
    assert _mirror_bytes(w, E2E_MIRROR) == before and marker.read_bytes() == marker_before and site_copy.read_bytes() == site_before
    assert not (w.state / "superseded").exists() and not (w.site / "geometry" / "superseded").exists() and w.sbatch_calls() == []
    # the fault cleared: the next run recovers on its own — no refusal at probe_home, no --rehome, no hand's move
    r = w.run()
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "has no fetch record naming its home" not in out and "by hand" not in r.stderr, out
    assert "archived under" in out and "READY on cluster_a" in out and f"generation={gen2}" in marker.read_text(), out
    assert json.loads((w.state / "probe_fetch.json").read_text())["generation"] == gen2 and site_copy.is_file()
    local_arch, site_arch = _archives(w.state / "superseded"), _archives(w.site / "geometry" / "superseded")
    assert list(local_arch) == list(site_arch) and len(local_arch) == 1, (local_arch.keys(), site_arch.keys())          # one stamp, both places
    assert next(iter(local_arch.values())) == {**before, "DONE.lora.probe": marker_before}                          # every old byte, nothing deleted
    assert next(iter(site_arch.values())) == {f"{E2E_MODEL}.json": site_before}


@pytest.mark.parametrize("fails", ["allowed_pairs.json", "probe_fetch.json"])
def test_a_move_refused_mid_sequence_leaves_what_the_next_run_archives_under_a_second_stamp(world, fails):
    """A rename refused where no pre-check can see it (the `mv` on PATH refuses ONE source), end to end. The sequence ends at that
    move, and what stands is a prefix of the fetch's write order, which the next run recovers as an interrupted fetch — with the
    MENU's move refused: the marker, the site copy, the geometry record and the verdict are archived, the menu stands beside its fetch
    record (home resolved, no marker), and the next run's fetch archives both under a second stamp before re-fetching; with the FETCH
    RECORD's move refused (the last position): the receipt stands alone and the next run archives it too (never written over) and
    re-fetches. Either way every old byte is under superseded/ (two stamps at the mirror, one at the site) and nothing is deleted. On
    a1db16018 the loop went on past the refused move: the menu stood alone with no fetch record (probe_home and --rehome both refused,
    the probe could not retry), and a lone fetch record was written over by the next fetch."""
    w = world
    gen1, marker, site_copy = _fetched_with_geometry(w)
    before, marker_before, site_before = _mirror_bytes(w, E2E_MIRROR), marker.read_bytes(), site_copy.read_bytes()
    gen2 = _record_moves_on(w, gen1); _lay_geometry(w, gen2)
    _failing_mv(w.bin)
    r = w.run(FAKE_MV_FAIL_SRC=str(w.state / fails))
    out = r.stdout + r.stderr
    assert r.returncode == 2 and f"REFUSED: probe (probe): {w.state / fails} could NOT be moved into" in r.stderr and "(nothing deleted)" in r.stderr, out
    assert "READY on cluster_a" not in out and "fetched to" not in out and "archived under" not in out and w.sbatch_calls() == [], out
    assert (w.state / "probe_fetch.json").read_bytes() == before["probe_fetch.json"] and not marker.exists() and not site_copy.exists()   # the receipt stands, the marker went first
    standing = {"allowed_pairs.json": ["allowed_pairs.json", "probe_fetch.json"], "probe_fetch.json": ["probe_fetch.json"]}[fails]
    assert sorted(n for n in E2E_MIRROR if (w.state / n).exists()) == sorted(standing)
    first = _archives(w.state / "superseded")
    assert len(first) == 1 and next(iter(first.values())) == {**{n: before[n] for n in E2E_MIRROR if n not in standing}, "DONE.lora.probe": marker_before}
    assert _archives(w.site / "geometry" / "superseded") == {next(iter(first)): {f"{E2E_MODEL}.json": site_before}}
    (w.bin / "mv").unlink()
    # the next run: the home resolves from the standing fetch record, the fetch archives what stands, re-fetches and marks
    r = w.run()
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "has no fetch record naming its home" not in out and "by hand" not in r.stderr and "archived under" in out and "READY on cluster_a" in out, out
    if fails == "probe_fetch.json":
        assert "probe_fetch.json alone" in out, out                                                                     # the lone receipt, archived — not written over
    assert f"generation={gen2}" in marker.read_text() and json.loads((w.state / "probe_fetch.json").read_text())["generation"] == gen2 and site_copy.is_file()
    local_arch = _archives(w.state / "superseded")
    assert len(local_arch) == 2 and list(local_arch)[0] == next(iter(first)), local_arch.keys()
    assert list(local_arch.values())[1] == {n: before[n] for n in standing}                                              # the second stamp: what stood
    merged = {k: v for d in local_arch.values() for k, v in d.items()}
    assert merged == {**before, "DONE.lora.probe": marker_before}, sorted(merged)                                      # every old byte, nothing deleted
    assert _archives(w.site / "geometry" / "superseded") == {next(iter(first)): {f"{E2E_MODEL}.json": site_before}}


# ---- probe_recover_planless: the record, over ssh ------------------------------------------------------------------------------

def _planless_script(record: Path, body: str) -> str:
    frag = _fragment("probe_recover_planless(){", "probe_width_flags(){")
    return (f"set -uo pipefail; source {REPO}/reproduce/lib/common.sh\n"
            f"PY=python3; CL_HOST=fake; CLUSTER=cluster_a; CL_ROOT={record}\n"
            'json_field(){ printf "%s" "$1" | python3 -c \'import json,sys; raw=sys.stdin.read(); d=json.loads(raw) if raw.strip() else {}; v=d.get(sys.argv[1]); print("" if v is None else (v if isinstance(v, str) else json.dumps(v)))\' "$2"; }\n'
            f"{frag}\n{body}")


def _local_ssh(bin_dir: Path) -> None:
    """An `ssh` that drops the options and the host and runs the command here (tests/test_front_door_recon_cell.py's)."""
    (bin_dir / "ssh").write_text('#!/bin/bash\nwhile [ "${1:-}" = -o ]; do shift 2; done; shift\nexec bash -c "$*"\n'); (bin_dir / "ssh").chmod(0o755)


def test_the_planless_recovery_reserves_its_archive_directory_on_the_record_beside_an_earlier_one_of_the_same_instant(tmp_path):
    """The record already holds `superseded/<stamp>/` of this very instant (an earlier archival under the frozen clock — the case the
    generation fence cannot reach, since it only guards the record's own files): the recovery takes `<stamp>-02`, the earlier
    archive is intact, the line names the directory it took, and the record is bare afterwards. A second recovery on the bare record
    REFUSES by the fence (probe_attempt.json is gone) with no third directory. Before the fix `mkdir -p superseded/<second>` accepted
    the existing directory and `mv -f` wrote its files into it, over any of the same name."""
    record = tmp_path / "record"; record.mkdir()
    earlier = record / "superseded" / FROZEN; earlier.mkdir(parents=True)
    (earlier / "probe_attempt.json").write_text(json.dumps({"job_id": "11111111", "note": "the earlier archival of this instant"}))
    (record / "probe_attempt.json").write_text(json.dumps({"job_id": "77777777", "attempt": 1}))
    (record / "RESULT").write_text("state=FAILED exit=3 attempt=1/4 host=TESTHOST job=77777777\n")
    (record / "probe_attempt.json.tmp.999").write_text("{")
    (record / "adapters").mkdir(); (record / "adapters" / "a.safetensors").write_text("x")
    bin_dir = _frozen_date_bin(tmp_path); _local_ssh(bin_dir)
    status = json.dumps({"state": "FAILED_START", "plan_present": False})
    r = _run(_planless_script(record, f"probe_recover_planless '{status}' 77777777; echo rc=$?"), bin_dir)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "rc=0" in r.stdout, out
    names = sorted(p.name for p in (record / "superseded").iterdir())
    assert names == [FROZEN, f"{FROZEN}-02"], names
    assert json.loads((earlier / "probe_attempt.json").read_text())["job_id"] == "11111111"                       # the earlier archive, intact
    assert sorted(p.name for p in (record / "superseded" / f"{FROZEN}-02").iterdir()) == ["RESULT", "adapters", "probe_attempt.json", "probe_attempt.json.tmp.999"]
    assert json.loads((record / "superseded" / f"{FROZEN}-02" / "probe_attempt.json").read_text())["job_id"] == "77777777"
    assert sorted(p.name for p in record.iterdir()) == ["superseded"]                                             # the record is bare
    assert f"recovered the plan-less record {record} — moved to superseded/{FROZEN}-02/ (never deleted): " in out, out
    said = re.search(r"\(never deleted\): (.*)$", out, flags=re.M).group(1).split()
    assert sorted(said) == ["RESULT", "adapters", "probe_attempt.json", "probe_attempt.json.tmp.999"], said
    assert "ARCHIVE_DIR=" not in out                                                                                # the seam is parsed, not printed
    # the fence: a second recovery of the bare record refuses by name and reserves nothing
    r2 = _run(_planless_script(record, f"probe_recover_planless '{status}' 77777777; echo rc=$?"), bin_dir)
    out2 = r2.stdout + r2.stderr
    assert r2.returncode != 0 and "rc=" not in r2.stdout and "probe_attempt.json is gone since the terminal proof" in out2, out2
    assert sorted(p.name for p in (record / "superseded").iterdir()) == [FROZEN, f"{FROZEN}-02"]


def test_the_planless_recovery_of_a_record_with_nothing_to_move_leaves_no_empty_directory(tmp_path):
    """probe_attempt.json alone, moved: one directory; the reservation's `rmdir` on nothing moved is kept (n = 0 is unreachable past the
    fence, which needs probe_attempt.json present — the bare mkdir leaves no directory behind on that path either way)."""
    record = tmp_path / "record"; record.mkdir()
    (record / "probe_attempt.json").write_text(json.dumps({"job_id": "77777777"}))
    bin_dir = _frozen_date_bin(tmp_path); _local_ssh(bin_dir)
    status = json.dumps({"state": "FAILED_START", "plan_present": False})
    r = _run(_planless_script(record, f"probe_recover_planless '{status}' 77777777; echo rc=$?"), bin_dir)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "rc=0" in r.stdout, out
    assert [p.name for p in (record / "superseded").iterdir()] == [FROZEN] and [p.name for p in (record / "superseded" / FROZEN).iterdir()] == ["probe_attempt.json"]
    assert f"moved to superseded/{FROZEN}/ (never deleted): probe_attempt.json" in out, out


# ---- the text: the two functions reserve, and the map of the file's other stamps is a decision ----------------------------------

def test_the_probe_chain_reserves_every_archive_directory_and_names_no_second_resolution_stamp():
    """probe_archive_local: the stamp is repro_superseded_stamp's, each of its two places goes through repro_superseded_dir with that
    stamp, and no `mkdir -p` into a stamped directory remains. probe_recover_planless: the same stamp, the inlined reservation in
    the ssh string (a bare mkdir in a loop, `-NN` on a collision), the directory reported as ARCHIVE_DIR and parsed back, and no
    `mkdir -p 'superseded/$utc'`. The remaining `%Y%m%dT%H%M%SZ` stamps of 03_hpo.sh are exactly the mapped ones (the module
    docstring) — a new second-resolution archive stamp anywhere else fails here and asks for the decision."""
    arch = _fragment("probe_archive_local(){", "probe_take_lock(){")
    assert "%Y%m%dT%H%M%SZ" not in arch and 'mkdir -p "$arch"' not in arch and 'mkdir -p "$SISTEER_SITE_DIR/geometry/superseded/' not in arch
    assert 'utc="$(repro_superseded_stamp)"' in arch
    assert 'arch="$(repro_superseded_dir "$LOCAL_ROOT/superseded" "$utc")"' in arch
    assert 'site_arch="$(repro_superseded_dir "$SISTEER_SITE_DIR/geometry/superseded" "$utc")"' in arch
    assert arch.count('$(repro_superseded_dir "') == 2 and "superseded/<utc>/ (the site copy under geometry/superseded/<utc>/)" in arch   # two places, one call each; the dry-run line unchanged
    # a failed archival is a REFUSAL: past the point where the function has decided to move, no `return 0` remains — the
    # pre-check, each reservation and the move that fails are repro_fatal (four) — and every failure point leaves a state the next run
    # recovers: the moves are listed in the REVERSE of the fetch's write order (the marker, the site copy, the
    # evidence, the fetch record LAST), every source directory is checked before the first reservation, both places are reserved before the
    # first move, and ONE loop moves in that order and ends at the first failure (no `left` list carried past a failed move)
    moving = arch[arch.index("local moves=()"):]
    code = "\n".join(ln for ln in moving.splitlines() if not ln.lstrip().startswith("#"))        # the code alone: the comments tell the history
    assert "return 0" not in code and "repro_warn" not in code.split("fetch_invalidate")[0], code
    assert code.count("repro_fatal") == 4, code.count("repro_fatal")
    assert (moving.index('moves+=("$marker")') < moving.index('moves+=("$site_copy")')
            < moving.index('"$LOCAL_ROOT"/geometry_record*.json "$LOCAL_ROOT"/probe_verdict.json "$LOCAL_ROOT"/allowed_pairs.json "$LOCAL_ROOT"/probe_fetch.json')
            < moving.index('[ -w "$d" ] && [ -x "$d" ]') < moving.index("NOTHING moved and nothing reserved")
            < moving.index('arch="$(repro_superseded_dir "$LOCAL_ROOT/superseded" "$utc")"') < moving.index('site_arch="$(repro_superseded_dir "$SISTEER_SITE_DIR/geometry/superseded" "$utc")"')
            < moving.index('rmdir "$arch"') < moving.index("for ((i = 0; i < ${#moves[@]}; i++))") < moving.index('mv -f "$f" "$dst/"') < moving.index("could NOT be moved into")
            < moving.index('fetch_invalidate "$CL_ROOT" "$LOCAL_ROOT"')), "list the moves in the fetch's reverse order, check every source directory, reserve, reserve, one loop that stops at the first failure"
    assert "left+=" not in code and 'for f in "${todo[@]}"' not in code and "mv -f \"$site_copy\"" not in code, "one loop, one order"
    assert "never exits" not in arch and "It exits (repro_fatal, by name) ONLY" in arch
    # the order the archival reverses is the fetch's own (tests/test_front_door_probe_fetch_home.py pins it too), and the fetch's archival
    # takes a lone fetch record (the archival's last position refused; a fetch that died between its record and its exposure) instead of
    # writing it over
    fr = _fragment("probe_fetch_ready(){", "probe_fresh_binding(){")
    assert fr.index("probe_write_fetch_record ") < fr.index("probe_expose_staged ") < fr.index('geometry_to_site "$LOCAL_ROOT"') < fr.index("probe_mark_fetched "), "the fetch: record, files, site copy, marker"
    assert 'ev="probe_fetch.json alone' in fr and fr.index('ev="probe_fetch.json alone') < fr.index("probe_archive_local probe READY")
    planless = _fragment("probe_recover_planless(){", "probe_width_flags(){")
    assert "%Y%m%dT%H%M%SZ" not in planless and "mkdir -p 'superseded/$utc'" not in planless and "'superseded/$utc/'" not in planless
    assert 'utc="$(repro_superseded_stamp)"' in planless
    assert r'until mkdir \"\$A\" 2>/dev/null; do [ -e \"\$A\" ] ||' in planless and r'A=\"superseded/$utc-\$(printf ' in planless
    assert r'echo \"ARCHIVE_DIR=\$A\"' in planless and "sed -n 's/^ARCHIVE_DIR=//p'" in planless and "sed '/^ARCHIVE_DIR=/d'" in planless
    # the map: every remaining second-resolution stamp in the file, by the function that holds it (None: the top-level addendum block)
    holder, found = None, []
    for line in HPO.read_text().splitlines():
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\(\)\s*\{", line)
        if m: holder = m.group(1)
        elif line.startswith("}"): holder = None
        if "%Y%m%dT%H%M%SZ" in line: found.append(holder)
    assert sorted(found, key=str) == sorted(["recon_archive_record", "probe_submit_reverdict", "recon_reset_local", "addendum_menu_refit", "addendum_menu_refit", None], key=str), found
    common = (REPO / "reproduce/lib/common.sh").read_text()
    assert "repro_superseded_stamp(){ date -u +%Y%m%dT%H%M%S.%6NZ; }" in common and '[ -n "$stamp" ] || stamp="$(repro_superseded_stamp)"' in common

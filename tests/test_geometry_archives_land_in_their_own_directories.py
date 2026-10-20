"""The geometry chain of reproduce/lib/slurm.sh archived into `superseded/<UTC to the second>/`
with `mkdir -p && mv -f` — the stage-02 pools-marker defect, in three more places: the remote clear_marker of slurm_stage_geometry
(a target's REVOKED marker, over ssh), the site's own marker clear at the end of the staging, and slurm_revoke_geometry (the site
copy and every cluster's overlay, ONE stamp across them). Two archivals inside one second named the same directory, `mkdir -p`
accepted it, and the second `mv -f` wrote the first archive over.

Now every archival lands in a directory no other archival shares: the stamp is to the microsecond (lib/common.sh
repro_superseded_stamp — the grading host clock, as every stamp under a geometry superseded/ always was; fixed width, so a lexical sort of
the directories is their archival order), the directory is RESERVED by a bare `mkdir` and a collision takes the next `-NN` suffix.
The two local sites go through repro_superseded_dir; the two ssh strings inline the same rule (a remote cannot call a grading host shell
function) and report the directory they took, which the front door's line names. The revocation keeps ONE stamp as the common
prefix of the site's and every cluster's archive — its lines name that stamp everywhere, and nothing parses the name (the receipt
carries the generation and each cluster's ack utc; the tests glob `superseded/*/<model>.json`) — and suffixes only where a place
collides.

Frozen clock: a `date` on PATH that answers every archive-stamp format with one instant (tests/test_stage02_pools_rerunnable.py's
_frozen_date_bin), so two archivals fall in the same microsecond — the collision a real clock never produces and a frozen one always
does; before the fix each pair collapsed into one directory holding the second archive alone. Synthetic site (front_door_testlib),
the `local:<dir>` ssh seam of the geometry module's _World; nothing reaches a cluster.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from test_geometry_travels_with_the_front_door import KEY, _World, _clean, _scan_clean
from test_stage02_pools_rerunnable import _frozen_date_bin

REPO = Path(__file__).resolve().parents[1]
#: an archive directory's name: the UTC stamp to the microsecond (repro_superseded_stamp); `-NN` is appended only on a collision
STAMP_RE = r"\d{8}T\d{6}\.\d{6}Z"


def _frozen(w: _World, tmp_path: Path) -> None:
    """Every archive stamp the front door mints from here on reads ONE instant (the fake delegates every other format to the real date)."""
    w.env["PATH"] = f"{_frozen_date_bin(tmp_path)}:{os.environ['PATH']}"


def _marker(gen: str, reason: str) -> str:
    """A REVOKED marker of ANOTHER generation than the record's: the staging clears it by the rule, whatever the clock says."""
    return json.dumps({"model_key": KEY, "generation": gen, "revoked_utc": "2026-09-30T00:00:00Z", "reason": reason, "by": "test"})


def test_two_marker_clears_in_one_second_keep_both_archives_on_the_site_and_on_the_target(tmp_path, monkeypatch):
    """The staging's two archival sites (the target's REVOKED marker through the ssh string's clear_marker, the site's own marker after
    every target verified), twice in one frozen instant: the first clear's archive is intact beside the second's (`<stamp>`, then
    `<stamp>-02`, each holding the marker text it archived, sorted = archival order), and each staging named the directory it took.
    Before the fix both clears named `superseded/<second>/` and the second `mv -f` left one directory with the second marker alone."""
    _clean(monkeypatch)
    w = _World(tmp_path)
    sp, plan, out = w.ready_record()
    w.site_copy_from(out)
    assert w.bash("slurm_stage_geometry cluster_a,cluster_b").returncode == 0
    _frozen(w, tmp_path)
    gdir = w.shared["cluster_b"] / "exp/reproduce/geometry"
    site_marker = w.site / "geometry" / f"{KEY}.REVOKED.json"; target_marker = gdir / f"{KEY}.REVOKED.json"
    texts, outs = [], []
    for i, gen in enumerate(("e" * 64, "f" * 64)):
        text = _marker(gen, f"clear {i + 1}")
        site_marker.write_text(text); target_marker.write_text(text); texts.append(text)
        r = w.bash("slurm_stage_geometry cluster_b; echo rc=$?")
        assert "rc=0" in r.stdout and "REVOKED marker cleared" in r.stdout and "REVOKED.json cleared" in r.stdout, r.stdout + r.stderr
        assert not site_marker.exists() and not target_marker.exists()
        outs.append(r.stdout)
        _scan_clean(r, tmp_path)
    # THE DEFECT: both clears fell in one instant — two directories, the first archive intact beside the second (one directory holding
    # the second marker alone is what the old `mkdir -p superseded/<second> && mv -f` left)
    for base in (w.site / "geometry" / "superseded", gdir / "superseded"):
        names = sorted(p.name for p in base.iterdir())
        assert len(names) == 2 and re.fullmatch(STAMP_RE, names[0]) and names[1] == f"{names[0]}-02", (base, names)   # the same instant, twice
        assert [(base / n / f"{KEY}.REVOKED.json").read_text() for n in names] == texts                               # the first archive is intact
    # and each staging named the directory it took: the target's line (through the ssh string's GEOMETRY_MARKER_CLEARED <dir>) and the site's
    for i, stdout in enumerate(outs):
        target_said = re.search(r"REVOKED marker cleared -> (\S+)/\)", stdout).group(1)
        site_said = re.search(r"REVOKED\.json cleared \(.*?; archived under (\S+)/\)", stdout).group(1)
        assert Path(target_said) == sorted(gdir.joinpath("superseded").iterdir())[i] and Path(site_said) == sorted((w.site / "geometry" / "superseded").iterdir())[i], (target_said, site_said)


def test_two_revocations_in_one_second_keep_both_archives_at_every_place_under_one_common_stamp(tmp_path, monkeypatch):
    """slurm_revoke_geometry archives at three places — the site copy, the probe cluster's fold copy, the secondary's staged copy — under
    ONE stamp. Two revocations in one frozen instant: at every place the first revocation's archive stands beside the second's
    (`<stamp>`, `<stamp>-02`), the stamp is the SAME at all three places (the revocation's common prefix), and each revocation's lines
    name, per place, the directory it took. Before the fix the second `mv -f` wrote the first archived overlay over at every place."""
    _clean(monkeypatch)
    w = _World(tmp_path)
    sp, plan, out = w.ready_record()
    w.site_copy_from(out)
    assert w.bash("slurm_stage_geometry cluster_a,cluster_b").returncode == 0
    gen = sp.result_generation(out)["generation"]
    places = [w.site / "geometry", w.shared["cluster_a"] / "exp/reproduce/geometry", w.shared["cluster_b"] / "exp/reproduce/geometry"]
    first = [(p / f"{KEY}.json").read_bytes() for p in places]
    _frozen(w, tmp_path)
    r1 = w.bash(f"slurm_revoke_geometry {KEY} {gen} 'first' cluster_a {out}; echo rc=$?")
    assert "rc=0" in r1.stdout and "revocation receipt written into the record" in r1.stdout, r1.stdout + r1.stderr
    for p in places:
        assert not (p / f"{KEY}.json").exists() and (p / f"{KEY}.REVOKED.json").is_file()
    # the overlays return (a fold, a staging) and a second revocation, in the same instant, revokes them again
    second = []
    for p, b in zip(places, first):
        doc = json.loads(b); doc["note"] = "the second round's copy"; text = json.dumps(doc) + "\n"
        (p / f"{KEY}.json").write_text(text); second.append(text)
    r2 = w.bash(f"slurm_revoke_geometry {KEY} {gen} 'second' cluster_a {out}; echo rc=$?")
    assert "rc=0" in r2.stdout and "1 prior active receipt(s) superseded" in r2.stdout, r2.stdout + r2.stderr
    stamps = set()
    for p, b1, t2 in zip(places, first, second):
        names = sorted(q.name for q in (p / "superseded").iterdir())
        assert len(names) == 2 and re.fullmatch(STAMP_RE, names[0]) and names[1] == f"{names[0]}-02", (p, names)
        assert (p / "superseded" / names[0] / f"{KEY}.json").read_bytes() == b1          # the first revocation's archive, intact
        assert (p / "superseded" / names[1] / f"{KEY}.json").read_text() == t2
        stamps.add(names[0])
    assert len(stamps) == 1, stamps                                                        # ONE stamp across the site and both clusters
    stamp = stamps.pop()
    said1 = re.findall(r"archived under (\S+)/[ ,]", r1.stdout); said2 = re.findall(r"archived under (\S+)/[ ,]", r2.stdout)
    assert sorted(Path(s) for s in said1) == sorted(p / "superseded" / stamp for p in places), (said1, r1.stdout)
    assert sorted(Path(s) for s in said2) == sorted(p / "superseded" / f"{stamp}-02" for p in places), (said2, r2.stdout)
    _scan_clean(r1, tmp_path); _scan_clean(r2, tmp_path)


def test_repro_superseded_dir_takes_a_callers_stamp_as_the_common_prefix_and_still_reserves(tmp_path):
    """The helper's second argument (the revocation's one stamp): three calls with the same stamp print and create <stamp>, <stamp>-02,
    <stamp>-03 — reserved, never shared; a fresh stamp has the same width and format as the caller's (repro_superseded_stamp); the
    dry-run refusal holds with a stamp given too."""
    base = tmp_path / "superseded"
    stamp = "20261009T120000.000000Z"
    script = (f"set -uo pipefail; source {REPO}/reproduce/lib/common.sh; for i in 1 2 3; do repro_superseded_dir {base} {stamp} || echo FAIL; done; "
              f"repro_superseded_stamp; repro_superseded_dir {base}")
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=os.environ, timeout=60)
    printed = r.stdout.split()
    assert r.returncode == 0 and "FAIL" not in r.stdout and len(printed) == 5, r.stdout + r.stderr
    assert [Path(d).name for d in printed[:3]] == [stamp, f"{stamp}-02", f"{stamp}-03"] and all(Path(d).is_dir() for d in printed[:3])
    assert re.fullmatch(STAMP_RE, printed[3]) and len(printed[3]) == len(stamp)                                      # the one format
    assert Path(printed[4]).is_dir() and re.fullmatch(STAMP_RE, Path(printed[4]).name) and Path(printed[4]).parent == base
    assert sorted(p.name for p in base.iterdir()) == sorted([Path(d).name for d in printed[:3]] + [Path(printed[4]).name])   # four, none shared
    script = f"set -uo pipefail; export REPRO_DRY_RUN=1; source {REPO}/reproduce/lib/common.sh; out=$(repro_superseded_dir {tmp_path}/dry {stamp}); echo rc=$? out=[$out]"
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=os.environ, timeout=60)
    assert "rc=1 out=[]" in r.stdout and "refused under --dry-run" in r.stderr and not (tmp_path / "dry").exists(), r.stdout + r.stderr


def test_the_geometry_chain_reserves_every_archive_directory_and_names_no_second_resolution_stamp():
    """The text: no archive directory in the chain is named to the second or made with `mkdir -p` any more; the two local sites go
    through repro_superseded_dir (the revocation handing its one stamp), both stamps through repro_superseded_stamp; the two ssh
    strings carry the inlined reservation (a bare mkdir in a loop, `-NN` on a collision) and report the directory they took, which the
    front door's lines name. The receipts' own `.superseded.<utc>.json` is a FILE rename of a unique nonce, not an archive directory."""
    slurm = (REPO / "reproduce/lib/slurm.sh").read_text()
    chain = slurm[slurm.index("slurm_stage_geometry(){"):slurm.index("slurm_base_env(){")]
    assert "superseded/$(date" not in chain and 'mkdir -p "$sup"' not in chain and 'mkdir -p "$sd/superseded/' not in chain
    assert r'mkdir -p \"\$D/superseded/' not in chain
    assert 'sup="$(repro_superseded_dir "$dir/superseded")"' in chain and 'site_arch="$(repro_superseded_dir "$sd/superseded" "$utc")"' in chain
    assert 'sup_ts="$(repro_superseded_stamp)"' in chain and 'utc="$(repro_superseded_stamp)"' in chain
    assert chain.count(r'until mkdir \"\$A\" 2>/dev/null; do [ -e \"\$A\" ] ||') == 2
    assert r'A=\"\$D/superseded/$sup_ts-\$(printf ' in chain and r'A=\"\$D/superseded/$utc-\$(printf ' in chain
    assert r'echo \"GEOMETRY_MARKER_CLEARED \$A\"' in chain and r'echo \"REVOKED_OK $model moved=\$moved archive=\$A\"' in chain
    assert "sed -n 's/^REVOKED_OK .* archive=//p'" in chain and "sed -n 's/^GEOMETRY_MARKER_CLEARED //p'" in chain
    common = (REPO / "reproduce/lib/common.sh").read_text()
    assert "repro_superseded_stamp(){ date -u +%Y%m%dT%H%M%S.%6NZ; }" in common and '[ -n "$stamp" ] || stamp="$(repro_superseded_stamp)"' in common

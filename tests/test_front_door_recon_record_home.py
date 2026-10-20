"""The recon record's HOME (03_hpo.sh recon_home / recon_write_home / recon_record_absent_refusal; the recon twin of probe_home):
the verdict's rung files + recon.json live on the cluster the ladder ran on, while design / trials may run on another. Consuming
phases revalidate against the record WHERE IT LIVES (recon_fetch.json), never against --cluster's copy of the path, and an ABSENT
or EMPTY cluster record is never evidence: an empty record digest refuses by name with nothing moved; only a POSITIVE digest
mismatch at the home archives local state. The recon PHASE is the bootstrap: it runs where its record lives, refuses a --cluster
other than a recorded home, and writes the fetch record when it marks. Executed against 03's REAL recon fragment with the
revalidation harness (fake ssh that runs the remote string here and LOGS the host; two neutral clusters; no network)."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_front_door_recon_verdict_revalidation import REPRO, World  # noqa: E402

SI = "constv2ctl2-full"


class Worlds(World):
    """World + a second cluster with NO record and NO pool copy, and an ssh fake that logs its host."""

    def __init__(self, tmp_path: Path):
        super().__init__(tmp_path)
        self.ssh_log = tmp_path / "ssh.log"
        (self.bin / "ssh").write_text(
            "#!/bin/bash\nhost=\"\"; while [ $# -gt 0 ]; do case \"$1\" in -o) shift 2;; -*) shift;; *) host=\"$1\"; shift; break;; esac; done\n"
            f"printf '%s\\t%s\\n' \"$host\" \"$*\" >> {self.ssh_log}\nexec bash -c \"$*\"\n")
        self.cluster_b = tmp_path / "cluster_b" / "03_hpo" / "relu"          # absent on purpose: cluster_b holds no record
        self.pairs_b = tmp_path / "cluster_b" / "pool" / "pairs_constv2ctl2-full.jsonl"
        self.fetch = self.local / "recon_fetch.json"
        self.marker = self.state / "m" / "03_hpo" / "DONE.relu.recon"

    # the site's two clusters + the ONE reader of a cluster block the fragment needs (the harness sources no site dir)
    PRE_SITE = 'SISTEER_CLUSTERS="cluster_a cluster_b"; cluster_var(){ [ "$2" = CLUSTER_SSH ] && echo "login-$1"; return 0; }'

    def pre_b(self) -> str:
        """The consuming phase invoked with --cluster cluster_b: its paths point at cluster_b's (absent) record and pool copy."""
        return f'{self.PRE_SITE}; CLUSTER=cluster_b; CL_HOST=login-cluster_b; CL_ROOT={self.cluster_b}; RECON_PAIRS={self.pairs_b}'

    def reads(self) -> list[tuple[str, str]]:
        return [tuple(ln.split("\t", 1)) for ln in self.ssh_log.read_text().splitlines()] if self.ssh_log.exists() else []

    def compose_reads(self) -> list[tuple[str, str]]:
        return [(h, c) for h, c in self.reads() if "wc -l" in c]

    def intact(self) -> bool:
        return (self.local / "recon.json").is_file() and self.marker.is_file() and not (self.local / "superseded").exists() \
            and all((self.local / f"recon_{i}.json").is_file() for i in range(5))


@pytest.fixture
def w(tmp_path):
    return Worlds(tmp_path)


def _recon_on_a(w: Worlds, monkeypatch) -> str:
    """The ladder ran on cluster_a: five stamped rungs + verdict on its record; `--phase recon --cluster cluster_a` fetches, verifies, marks and
    records the home. Returns the marked spec fingerprint."""
    w.make_verdict(monkeypatch, SI, w.pairs)
    r = w.run("recon", pre=w.PRE_SITE); out = r.stdout + r.stderr
    assert r.returncode == 0 and "VERIFIED as the current reconnaissance's" in out and "STUB submit" not in out, out
    assert w.intact(), out
    doc = json.loads(w.fetch.read_text())
    assert doc["cluster"] == "cluster_a" and doc["record"] == str(w.cluster) and doc["pairs"] == str(w.pairs), doc
    assert len(doc["record_digest"]) == 64 and len(doc["spec_fingerprint"]) == 64 and "the verdict's home recorded" in out, (doc, out)
    w.ssh_log.write_text("")
    return doc["spec_fingerprint"]


def test_design_on_cluster_b_revalidates_against_the_record_on_cluster_a_and_archives_nothing(w, monkeypatch):
    fp = _recon_on_a(w, monkeypatch)
    r = w.run("design", pre=w.pre_b(), extra="recon_revalidate_local design"); out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert f"record lives on cluster_a:{w.cluster}" in out and "design runs on cluster_b" in out, out
    assert f"re-verified as the current reconnaissance's (spec {fp[:16]}) — design proceeds" in out, out
    reads = w.compose_reads()
    assert reads and all(h == "login-cluster_a" and str(w.pairs) in c and str(w.cluster) in c for h, c in reads), (reads, out)
    assert not any(str(w.cluster_b) in c or str(w.pairs_b) in c for _h, c in w.reads()), w.reads()
    assert "archived" not in out.replace("nothing archived", "") and w.intact(), out


def test_trials_on_cluster_b_take_the_same_path(w, monkeypatch):
    _recon_on_a(w, monkeypatch)
    r = w.run("trials", pre=w.pre_b(), extra="recon_revalidate_local trials"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "trials runs on cluster_b" in out and "— trials proceeds" in out and w.intact(), out


def test_local_evidence_without_a_fetch_record_refuses_a_consuming_phase_by_name_and_moves_nothing(w, monkeypatch):
    """A recon marked before the fetch record existed (or the record lost): design cannot know where to look and must not default to --cluster's
    copy of the path — refused by name, no record read, nothing archived."""
    _recon_on_a(w, monkeypatch)
    w.fetch.unlink()
    r = w.run("design", pre=w.pre_b(), extra="recon_revalidate_local design"); out = r.stdout + r.stderr
    assert r.returncode != 0 and "no " in r.stderr and "recon_fetch.json" in r.stderr and "--phase recon --cluster" in r.stderr and "nothing moved" in r.stderr, out
    assert w.compose_reads() == [] and w.intact(), (w.reads(), out)
    # the same-cluster recon phase is the bootstrap: it re-verifies the marked verdict and writes the missing fetch record
    r2 = w.run("recon", pre=w.PRE_SITE); out2 = r2.stdout + r2.stderr
    assert r2.returncode == 0 and "skipping" in out2 and "the verdict's home recorded" in out2 and w.fetch.is_file() and w.intact(), out2


def test_an_absent_or_empty_record_at_the_home_is_never_evidence_nothing_moved(w, monkeypatch):
    """The record's home names cluster_a but its record directory is emptied (moved away on the cluster): the digest reads EMPTY — a refusal by
    name with the local verdict, rung files and marker untouched (never 'the record changed' + archive)."""
    _recon_on_a(w, monkeypatch)
    gone = w.tmp / "moved_away"; shutil.move(str(w.cluster), str(gone))
    r = w.run("design", pre=w.pre_b(), extra="recon_revalidate_local design"); out = r.stdout + r.stderr
    assert r.returncode != 0 and f"no recon record at cluster_a:{w.cluster}" in r.stderr and "never evidence" in r.stderr and "nothing moved" in r.stderr, out
    assert "changed after it was fetched" not in out and w.intact(), out
    # the recon PHASE on the home sees the same absence on its DONE-marker path: refuse, do not archive, do not resubmit the ladder
    r2 = w.run("recon", pre=w.PRE_SITE); out2 = r2.stdout + r2.stderr
    assert r2.returncode != 0 and "no recon record at cluster_a:" in r2.stderr and "STUB submit" not in out2 and w.intact(), out2
    shutil.move(str(gone), str(w.cluster))


def test_a_positive_digest_mismatch_at_the_home_still_archives_the_stale_mirror(w, monkeypatch):
    """The rule cuts one way only: a record that IS there but differs (a rung replaced under the same spec) still archives the local copies and
    refuses design until `--phase recon` re-derives — as it always has."""
    _recon_on_a(w, monkeypatch)
    rung = w.cluster / "recon_2.json"; rung.write_text(rung.read_text().replace("}", " }", 1))      # same JSON, other bytes -> another digest
    r = w.run("design", pre=w.pre_b(), extra="recon_revalidate_local design"); out = r.stdout + r.stderr
    assert r.returncode != 0 and "changed after it was fetched and marked" in out and "re-run --phase recon before design" in out, out
    assert (w.local / "superseded").exists() and not (w.local / "recon.json").exists() and not w.marker.exists(), out


def test_the_recon_phase_on_another_cluster_than_the_records_home_refuses_by_name(w, monkeypatch):
    _recon_on_a(w, monkeypatch)
    r = w.run("recon", pre=w.pre_b()); out = r.stdout + r.stderr
    assert r.returncode != 0 and f"record lives on cluster_a:{w.cluster}" in r.stderr and "--phase recon --cluster cluster_a" in r.stderr, out
    assert "nothing submitted" in r.stderr and "STUB submit" not in out and w.compose_reads() == [] and w.intact(), (w.reads(), out)


def test_source_pins(w):
    src = (REPRO / "03_hpo.sh").read_text()
    frag = src[src.index("recon_compose_spec(){"):src.index("# THE RECON RECORD'S HOME")]          # recon_compose_spec's body alone
    assert '"$RECON_CL_HOST"' in frag and "'$RECON_PAIRS_HOME'" in frag and "cd '$RECON_CL_ROOT'" in frag and '"$CL_HOST"' not in frag, frag
    rv = src[src.index("recon_revalidate_local(){"):src.index("if want recon &&")]
    assert 'recon_home "$1"' in rv and 'recon_record_absent_refusal "$1"' in rv, rv
    assert rv.index('recon_record_absent_refusal "$1"') < rv.index("recon_mirror_is_the_record"), "the empty-digest refusal precedes the mismatch archive"
    dc = src[src.index("recon_done_current(){"):src.index("recon_verify_verdict(){")]
    assert "recon_record_absent_refusal recon" in dc and dc.index("recon_record_absent_refusal recon") < dc.index("! recon_mirror_is_the_record"), dc


# --- the marker-mismatch archive used to precede the empty-record check -------------------------------------------------

def _foreign_marker(w: Worlds) -> None:
    """The DONE marker re-stamped for ANOTHER reconnaissance (a changed warm-up / cycled-pass spec): its fingerprint differs from the current one."""
    w.marker.write_text("2026-09-30T00:00:00Z /old/recon.json spec_fingerprint=" + "deadbeef" * 8 + "\n")


def test_a_changed_marker_with_an_empty_record_at_the_home_refuses_and_archives_nothing(w, monkeypatch):
    """Same cluster: the marker names another spec AND the record directory is gone — the empty digest refuses BEFORE the marker is compared;
    the local verdict, rung files and marker stay; no ladder is submitted (before the fix: foreign-marker archive, then a new five-rung array)."""
    _recon_on_a(w, monkeypatch)
    _foreign_marker(w)
    gone = w.tmp / "moved_away"; shutil.move(str(w.cluster), str(gone))
    r = w.run("recon", pre=w.PRE_SITE); out = r.stdout + r.stderr
    assert r.returncode != 0 and f"no recon record at cluster_a:{w.cluster}" in r.stderr and "nothing moved" in r.stderr, out
    assert "foreign-marker" not in out and "records another reconnaissance" not in out and "STUB submit" not in out, out
    assert w.marker.is_file() and (w.local / "recon.json").is_file() and all((w.local / f"recon_{i}.json").is_file() for i in range(5)) \
        and not (w.local / "superseded").exists(), out
    shutil.move(str(gone), str(w.cluster))


def test_a_legacy_recon_without_a_fetch_record_on_another_cluster_with_a_reachable_pool_refuses_before_any_archive(w, monkeypatch):
    """A recon marked before recon_fetch.json existed (no fetch record), re-run on cluster_b where the cell pool IS reachable but
    no record exists, under a changed spec — the home defaults to --cluster unproven; the empty digest there must refuse with nothing moved,
    never archive the valid local rungs, never submit."""
    _recon_on_a(w, monkeypatch)
    w.fetch.unlink()
    _foreign_marker(w)
    w.pairs_b.parent.mkdir(parents=True); shutil.copy(w.pairs, w.pairs_b)                      # the pool copy on cluster_b, no record there
    r = w.run("recon", pre=w.pre_b()); out = r.stdout + r.stderr
    assert r.returncode != 0 and f"no recon record at cluster_b:{w.cluster_b}" in r.stderr and "nothing moved" in r.stderr, out
    assert "foreign-marker" not in out and "STUB submit" not in out, out
    assert w.marker.is_file() and (w.local / "recon.json").is_file() and all((w.local / f"recon_{i}.json").is_file() for i in range(5)) \
        and not (w.local / "superseded").exists(), out
    # the recon record on cluster_a is untouched too
    assert (w.cluster / "recon.json").is_file() and all((w.cluster / f"recon_{i}.json").is_file() for i in range(5))


def test_a_fresh_reconnaissance_with_no_local_evidence_still_submits_its_ladder(w):
    """The regression guard for the rule's other side: NO local recon evidence at all + an empty record on --cluster is exactly a fresh
    reconnaissance — the default-to---cluster home stands and the array is submitted (with the pool reachable there)."""
    w.pairs_b.parent.mkdir(parents=True); shutil.copy(w.pairs, w.pairs_b)
    r = w.run("recon", pre=w.pre_b()); out = r.stdout + r.stderr
    assert r.returncode == 0 and "STUB submit" in out and "HPO_JOB_MODE=recon" in out and "no recon record" not in out, out
    assert not w.fetch.exists() and not w.marker.exists()                                        # nothing marked, nothing recorded before the ladder runs


def test_source_pins_sol_high():
    src = (REPRO / "03_hpo.sh").read_text()
    dc = src[src.index("recon_done_current(){"):src.index("recon_verify_verdict(){")]
    assert dc.index("recon_record_absent_refusal recon") < dc.index('grep -qF -- "spec_fingerprint=$CUR_FP"'), "the empty-record refusal precedes the marker compare"
    br = src[src.index("elif want recon; then"):src.index("if ! recon_done_current; then")]
    assert "recon_record_absent_refusal recon" in br and 'ls "$LOCAL_ROOT"/recon_[0-9]*.json' in br, br

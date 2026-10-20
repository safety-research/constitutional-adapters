"""The probe mirror's HOME is never guessed (03_hpo.sh probe_home / probe_fetch_ready / --rehome). The fetch
exposes nothing before its record: the record is rsynced into <mirror>/.fetch.staging, verified there, probe_fetch.json (home + verdict
identity) is written first, the files move in afterwards, the DONE marker last — so menu evidence never exists without a fetch record
naming the cluster and record dir it came from. Local probe evidence whose fetch record is absent, empty, malformed or names no home
refuses EVERY phase by name (nothing read, nothing archived, nothing submitted); --cluster's path is the home only when nothing local
exists. `--phase probe --rehome` re-derives the home from the mirror's own verdict (provenance.probe_cluster / probe_record), verifies that
cluster holds the record, and writes the fetch record with the mirror's actual identity. Executed against the REAL stage with the probe
harness (fake ssh / rsync / squeue that run the remote string on this host; the real status reader), two neutral clusters, no network."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO  # noqa: E402
from test_front_door_probe_record_home import _fetch_from_cluster_a, _record_reads, _run, worlds  # noqa: E402,F401  (the two-cluster fixture)
from test_front_door_probe_resume import MODEL, World, _stamp  # noqa: E402


def _run_env(w: World, extra_env: dict, *args: str) -> subprocess.CompletedProcess:
    argv = ["bash", str(REPRO / "03_hpo.sh"), "--model", MODEL, "--object-class", "lora", "--config", str(w.site / "config.env"), *args]
    return subprocess.run(argv, capture_output=True, text=True, env=w.env(**extra_env), cwd=str(REPO), timeout=600)


def _marker(w: World) -> Path:
    return w.state.parent / "DONE.lora.probe"


def _nothing_submitted(w: World) -> bool:
    return not list((w.fake / "jobs").glob("*.argv")) and not list((w.fake_b / "jobs").glob("*.argv"))


def _evidence_intact(w: World) -> bool:
    return (w.state / "allowed_pairs.json").is_file() and (w.state / "probe_verdict.json").is_file() and not (w.state / "superseded").exists()


def _with_provenance(w: World) -> None:
    """The record's verdict names its own home the way serving_probe.write_verdict does (provenance.probe_cluster / probe_record)."""
    v = w.record / "probe_verdict.json"; d = json.loads(v.read_text())
    d["provenance"] = {"probe_cluster": "cluster_a", "probe_record": str(w.record)}
    v.write_text(json.dumps(d))


def test_evidence_without_a_fetch_record_refuses_every_phase_by_name_and_touches_nothing(worlds):
    """A fetched menu whose probe_fetch.json is gone: recon on the other cluster, the probe phase on either cluster and a consuming phase
    on the record's own cluster all refuse by name — no record read anywhere, nothing archived, nothing submitted, the remedy named."""
    w = worlds
    _fetch_from_cluster_a(w)
    (w.state / "probe_fetch.json").unlink()
    for phase, cluster in (("recon", "cluster_b"), ("probe", "cluster_b"), ("probe", "cluster_a"), ("design", "cluster_a")):
        r = _run(w, "--phase", phase, "--cluster", cluster)
        out = r.stdout + r.stderr
        assert r.returncode == 2, (phase, cluster, out)
        assert "has no fetch record naming its home" in r.stderr and "probe_fetch.json is absent" in r.stderr and "--rehome" in r.stderr, (phase, cluster, out)
        assert "allowed_pairs.json" in r.stderr and "DONE.lora.probe" in r.stderr                      # the evidence it refuses over, by name
        assert "the LOCAL menu is NOT" not in out and "archived under" not in out and "file(s) archived" not in out, (phase, cluster, out)   # never the revalidation's archive
        assert _record_reads(w) == [], (phase, cluster, w.ssh_log.read_text())                         # not even read: the home is unknown
        assert _evidence_intact(w) and _marker(w).is_file() and _nothing_submitted(w), (phase, cluster)


def test_a_malformed_or_homeless_fetch_record_over_evidence_refuses_and_a_bare_mirror_still_defaults_to_the_cluster(worlds):
    """Every defect of the fetch record is named: empty, malformed JSON, a record from before the home fields, an unknown cluster. With
    NO local evidence at all, --cluster's path is the home as before (a first probe fetches)."""
    w = worlds
    _fetch_from_cluster_a(w)
    f = w.state / "probe_fetch.json"
    for text, defect in (("", "is empty"), ("{", "is malformed"), (json.dumps({"generation": "a" * 64}), "from before the home fields"),
                         (json.dumps({"cluster": "cluster_z", "record": "/nowhere"}), "naming cluster 'cluster_z', not one of SISTEER_CLUSTERS"),
                         (json.dumps({"cluster": "cluster_a"}), "naming no record")):
        f.write_text(text)
        r = _run(w, "--phase", "recon", "--cluster", "cluster_b")
        assert r.returncode == 2 and "has no fetch record naming its home" in r.stderr and defect in r.stderr, (defect, r.stdout + r.stderr)
        assert _record_reads(w) == [] and _evidence_intact(w) and _marker(w).is_file() and _nothing_submitted(w), defect
    # nothing local: the ordinary first fetch (the mirror is moved aside by hand — the refusal's own remedy for unwanted evidence)
    aside = w.state / "superseded" / "by-hand"; aside.mkdir(parents=True)
    for name in ("allowed_pairs.json", "probe_verdict.json", "probe_fetch.json"):
        (w.state / name).rename(aside / name)
    _marker(w).unlink()
    r = _run(w, "--phase", "probe", "--cluster", "cluster_a")
    assert r.returncode == 0 and "READY on cluster_a" in r.stdout + r.stderr, r.stdout + r.stderr
    fetch = json.loads(f.read_text()); assert fetch["cluster"] == "cluster_a" and fetch["record"] == str(w.record) and fetch["verdict_sha256"]


def test_an_interrupted_fetch_exposes_nothing_and_the_next_call_completes_it(worlds, tmp_path):
    """The process dies between the rsync and the fetch record (the test hook): the mirror holds NO menu, verdict, fetch record or marker —
    the staged copy sits in .fetch.staging. The next call finds no evidence, defaults to --cluster, removes the stale staging dir and
    completes the fetch: record (home + identity) then files then marker; a consuming phase on the other cluster then proceeds."""
    w = worlds
    w.lay(done=7, complete=True, menu=True, verdict=True); _stamp(w)
    die = tmp_path / "die-before-record"; die.write_text("")
    r = _run_env(w, {"SISTEER_PROBE_FETCH_DIE_FILE": str(die)}, "--phase", "probe", "--cluster", "cluster_a")
    assert r.returncode == -9 and "dying between the rsync and the fetch record" in r.stderr, r.stdout + r.stderr
    for name in ("allowed_pairs.json", "probe_verdict.json", "probe_fetch.json"):
        assert not (w.state / name).exists(), name                                          # nothing exposed
    assert not _marker(w).exists()
    assert (w.state / ".fetch.staging" / "allowed_pairs.json").is_file()                    # the staged copy, never the mirror
    die.unlink()
    r = _run(w, "--phase", "probe", "--cluster", "cluster_a")
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "READY on cluster_a" in out and "has no fetch record" not in out, out
    assert not (w.state / ".fetch.staging").exists()
    fetch = json.loads((w.state / "probe_fetch.json").read_text())
    assert fetch["cluster"] == "cluster_a" and fetch["record"] == str(w.record) and len(fetch["verdict_sha256"]) == 64 and len(fetch["menu_sha256"]) == 64
    assert fetch["written_utc"] == "2026-09-30T02:00:00Z" and fetch["probe_job_id"] == "11111111"          # the fetched menu's own fields
    assert (w.state / "allowed_pairs.json").is_file() and (w.state / "probe_verdict.json").is_file() and _marker(w).is_file()
    assert "verdict_sha256=" in _marker(w).read_text() and "generation=" in _marker(w).read_text()
    w.ssh_log.write_text("")
    r = _run(w, "--phase", "recon", "--cluster", "cluster_b")
    out = r.stdout + r.stderr
    assert "— recon proceeds" in out and "archived under" not in out, out
    assert _record_reads(w) and all(h == f"local:{w.fake}" for h, _cmd in _record_reads(w)), _record_reads(w)


def test_a_record_rewritten_during_the_fetch_is_never_exposed_and_the_next_fetch_takes_the_new_verdict(worlds, tmp_path):
    """A re-verdict that lands between the status read and the rsync (the test hook rewrites the record's verdict
    right after the rsync) is caught by the identity RE-READ — the staged old files are never exposed, nothing is marked; the next call
    fetches the new verdict."""
    w = worlds
    w.lay(done=7, complete=True, menu=True, verdict=True); _stamp(w)
    hook = tmp_path / "reverdict-during-fetch.sh"
    hook.write_text(f"#!/bin/bash\n{sys.executable} - <<'PY'\nimport json; p={str(w.record / 'probe_verdict.json')!r}; d=json.load(open(p)); d['memo']='re-verdict landed during the fetch'; json.dump(d, open(p, 'w'))\nPY\n")
    r = _run_env(w, {"SISTEER_PROBE_FETCH_AFTER_RSYNC_HOOK": f"bash {hook}"}, "--phase", "probe", "--cluster", "cluster_a")
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "changed DURING the fetch" in r.stderr and "nothing exposed, nothing marked" in r.stderr, out
    for name in ("allowed_pairs.json", "probe_verdict.json", "probe_fetch.json"):
        assert not (w.state / name).exists(), name
    assert not _marker(w).exists() and not (w.state / ".fetch.staging").exists() and _nothing_submitted(w)
    r = _run(w, "--phase", "probe", "--cluster", "cluster_a")
    assert r.returncode == 0 and "READY on cluster_a" in r.stdout + r.stderr, r.stdout + r.stderr
    assert json.loads((w.state / "probe_verdict.json").read_text()).get("memo") == "re-verdict landed during the fetch" and _marker(w).is_file()
    fetch = json.loads((w.state / "probe_fetch.json").read_text())
    import hashlib
    assert fetch["verdict_sha256"] == hashlib.sha256((w.record / "probe_verdict.json").read_bytes()).hexdigest()   # the identity of the NEW verdict


def test_a_missing_marker_over_a_current_mirror_is_recovered_in_place_and_archived_only_on_a_positive_mismatch(worlds):
    """A fetch that died between the exposure and the mark leaves the CURRENT menu / verdict and a
    valid home receipt with no DONE marker. A consuming phase on the OTHER cluster verifies READY / generation / identity / receipt at the
    derived home and RECOVERS the marker in place — nothing archived, the phase proceeds. Only a POSITIVE identity mismatch (the record's
    verdict rewritten) archives."""
    w = worlds
    _fetch_from_cluster_a(w)
    _marker(w).unlink()                                                                    # the crash window: exposed, unmarked
    r = _run(w, "--phase", "recon", "--cluster", "cluster_b")
    out = r.stdout + r.stderr
    assert "recovered in place, nothing archived" in out and "— recon proceeds" in out, out
    assert _marker(w).is_file() and "verdict_sha256=" in _marker(w).read_text() and not (w.state / "superseded").exists() and _evidence_intact(w)
    assert "archived under" not in out and "the LOCAL menu is NOT" not in out
    assert _record_reads(w) and all(h == f"local:{w.fake}" for h, _cmd in _record_reads(w)), _record_reads(w)   # verified at the record's HOME
    # the positive mismatch: the record's verdict was rewritten (same generation) — archived, refused
    _marker(w).unlink()
    v = w.record / "probe_verdict.json"; d = json.loads(v.read_text()); d["memo"] = "re-verdict after the fetch"; v.write_text(json.dumps(d))
    r = _run(w, "--phase", "recon", "--cluster", "cluster_b")
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "archived under" in out and "the verdict was rewritten" in out and (w.state / "superseded").exists(), out
    assert not (w.state / "allowed_pairs.json").exists() and not _marker(w).exists()


def test_a_dry_run_over_a_done_mirror_moves_nothing_and_an_empty_record_at_the_home_never_archives(worlds):
    """A live dry run once archived the mirror on the dry run's SYNTHETIC NOT_STARTED: under --dry-run
    the probe branch describes the revalidation and probe_archive_local prints what WOULD move — nothing moves. A REAL read of an EMPTY record
    at the home (no result generation) is not evidence either: the consuming phase refuses by name with the local copies intact."""
    w = worlds
    _fetch_from_cluster_a(w)
    before = {n: (w.state / n).read_bytes() for n in ("allowed_pairs.json", "probe_verdict.json", "probe_fetch.json")}
    for args in (("--phase", "probe", "--cluster", "cluster_a", "--dry-run"), ("--phase", "recon", "--cluster", "cluster_b", "--dry-run")):
        r = _run(w, *args)
        out = r.stdout + r.stderr
        assert "file(s) archived under" not in out and "the LOCAL menu is NOT" not in out, (args, out)     # the archive warning's own text (the dry-run note says "nothing archived under a dry run")
        assert all((w.state / n).read_bytes() == b for n, b in before.items()) and _marker(w).is_file() and not (w.state / "superseded").exists(), args
        assert not (w.site / "geometry" / "superseded").exists()
    r = _run(w, "--phase", "probe", "--cluster", "cluster_a", "--dry-run")
    assert r.returncode == 0 and "nothing read, nothing archived under a dry run" in r.stdout + r.stderr, r.stdout + r.stderr
    assert _nothing_submitted(w)
    # the record vanishes at its home (a real read, NOT_STARTED, no generation): the local evidence stays, the phase refuses
    import shutil
    shutil.rmtree(w.record); w.record.mkdir()
    r = _run(w, "--phase", "recon", "--cluster", "cluster_b")
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "never evidence against the local copies" in out and "NOTHING moved" in out, out
    assert all((w.state / n).read_bytes() == b for n, b in before.items()) and _marker(w).is_file() and not (w.state / "superseded").exists()


def test_rehome_rederives_the_home_from_the_mirrors_verdict_verifies_it_and_the_consuming_phases_proceed(worlds):
    """A mirror that lost its fetch record and marker (the pre-fix interrupted fetch): every phase refuses; `--phase probe --rehome` reads
    the home from the mirror's verdict, verifies cluster_a holds the record (status read there, identity equal), writes the fetch record
    and — the mirror IS the record — the marker; recon on cluster_b then revalidates against cluster_a's record and proceeds."""
    w = worlds
    w.lay(done=7, complete=True, menu=True, verdict=True); _with_provenance(w); gen = _stamp(w)
    r = _run(w, "--phase", "probe", "--cluster", "cluster_a")
    assert r.returncode == 0 and "READY on cluster_a" in r.stdout + r.stderr, r.stdout + r.stderr
    (w.state / "probe_fetch.json").unlink(); _marker(w).unlink(); w.ssh_log.write_text("")
    r = _run(w, "--phase", "recon", "--cluster", "cluster_b")
    assert r.returncode == 2 and "has no fetch record naming its home" in r.stderr and _record_reads(w) == []
    # a dry run of the verb names the derived home and writes nothing
    r = _run(w, "--phase", "probe", "--cluster", "cluster_b", "--rehome", "--dry-run")
    assert r.returncode == 0 and f"DRY-RUN: rehome" in r.stdout and f"cluster_a:{w.record}" in r.stdout, r.stdout + r.stderr
    assert not (w.state / "probe_fetch.json").exists() and not _marker(w).exists()
    # --rehome is a --phase probe verb
    r = _run(w, "--phase", "recon", "--cluster", "cluster_b", "--rehome")
    assert r.returncode == 2 and "--rehome needs --phase probe" in r.stderr
    # the verb: --cluster is irrelevant to it (the home comes from the verdict), the status read goes to cluster_a's record
    r = _run(w, "--phase", "probe", "--cluster", "cluster_b", "--rehome")
    out = r.stdout + r.stderr
    assert r.returncode == 0 and f"re-homed to cluster_a:{w.record}" in out and "the mirror IS the record" in out, out
    reads = _record_reads(w)
    assert reads and all(h == f"local:{w.fake}" and str(w.record) in cmd for h, cmd in reads), reads
    assert _nothing_submitted(w) and _evidence_intact(w)
    fetch = json.loads((w.state / "probe_fetch.json").read_text())
    assert fetch["cluster"] == "cluster_a" and fetch["record"] == str(w.record) and fetch["generation"] == gen and len(fetch["verdict_sha256"]) == 64
    assert _marker(w).is_file() and "verdict_sha256=" in _marker(w).read_text()
    w.ssh_log.write_text("")
    r = _run(w, "--phase", "recon", "--cluster", "cluster_b")
    out = r.stdout + r.stderr
    assert f"record cluster_a:{w.record}) — recon proceeds" in out and "archived under" not in out, out
    r = _run(w, "--phase", "probe", "--cluster", "cluster_a")
    assert r.returncode == 0 and "DONE marker present and the local menu is the record's current result generation" in r.stdout, r.stdout + r.stderr


def test_rehome_of_a_stale_mirror_writes_the_home_without_a_marker_and_the_next_probe_archives_and_refetches(worlds):
    """The record's verdict was rewritten after the fetch (a re-verdict; the generation stands): --rehome verifies the home, says the mirror
    is NOT the record's current verdict, writes the fetch record with the MIRROR's identity and no marker; `--phase probe --cluster cluster_a`
    then archives the stale copies and re-fetches — the ordinary currency machinery, never a guess."""
    w = worlds
    w.lay(done=7, complete=True, menu=True, verdict=True); _with_provenance(w); _stamp(w)
    r = _run(w, "--phase", "probe", "--cluster", "cluster_a"); assert r.returncode == 0, r.stdout + r.stderr
    v = w.record / "probe_verdict.json"; d = json.loads(v.read_text()); d["memo"] = "re-verdict: the same launches, a rewritten verdict"; v.write_text(json.dumps(d))
    (w.state / "probe_fetch.json").unlink(); _marker(w).unlink()
    r = _run(w, "--phase", "probe", "--cluster", "cluster_a", "--rehome")
    out = r.stdout + r.stderr
    assert r.returncode == 0 and f"re-homed to cluster_a:{w.record}" in out and "is NOT the record's current verdict" in out, out
    assert (w.state / "probe_fetch.json").is_file() and not _marker(w).exists() and _nothing_submitted(w)
    r = _run(w, "--phase", "probe", "--cluster", "cluster_a")
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "archived under" in out and "READY on cluster_a" in out, out
    assert json.loads((w.state / "probe_verdict.json").read_text()).get("memo") == d["memo"] and _marker(w).is_file()
    assert list((w.state / "superseded").iterdir())                                                   # the stale copies, kept


def test_rehome_refuses_a_verdict_without_provenance_and_names_the_manual_remedy(worlds):
    w = worlds
    _fetch_from_cluster_a(w)                                                                             # World's verdict carries no provenance
    (w.state / "probe_fetch.json").unlink()
    r = _run(w, "--phase", "probe", "--cluster", "cluster_a", "--rehome")
    assert r.returncode == 2 and "carries no provenance.probe_cluster / probe_record" in r.stderr and "superseded/<utc>/ by hand" in r.stderr, r.stdout + r.stderr
    assert not (w.state / "probe_fetch.json").exists() and _evidence_intact(w) and _marker(w).is_file() and _record_reads(w) == [] and _nothing_submitted(w)


def test_source_pins_the_fetch_order_and_the_home_rule():
    src = (REPRO / "03_hpo.sh").read_text()
    fr = src[src.index("probe_fetch_ready(){"):src.index("probe_fresh_binding(){")]
    assert fr.index('fetch "$CL_ROOT" "$stage"') < fr.index('rid2="$(probe_remote_identity)"') < fr.index('[ "$rid2" = "$rid" ]') < fr.index('lid="$(probe_local_identity "$stage")"'), "the remote identity is re-read after the rsync (the review's Medium finding on f8abab70d)"
    assert (fr.index('fetch "$CL_ROOT" "$stage"') < fr.index("probe_archive_local probe READY") < fr.index('[ -e "$SISTEER_PROBE_FETCH_DIE_FILE" ]') < fr.index("probe_write_fetch_record ") < fr.index("probe_expose_staged ")
            < fr.index('geometry_to_site "$LOCAL_ROOT"') < fr.index("probe_mark_fetched ")), "record first, files, site copy, marker last"
    assert 'fetch "$CL_ROOT" "$LOCAL_ROOT"' not in fr
    ph = src[src.index("probe_home(){"):src.index("probe_job_active(){")]
    assert "probe_local_evidence" in ph and "repro_fatal" in ph and "--rehome" in ph and 'return 0                                   # genuinely nothing local' in ph
    assert "mv -f" not in ph and "rm -rf" not in ph and "rename" not in ph and "probe_archive_local" not in ph, "no mutation in the home rule"
    pv = src[src.index("probe_rehome_verb(){"):src.index("probe_revalidate_local(){")]
    assert "probe_cluster" in pv and "probe_record" in pv and 'probe_status "$active"' in pv and "probe_remote_identity" in pv and "probe_write_fetch_record" in pv
    assert "sbatch" not in pv and "probe_resume" not in pv and "probe_archive_local" not in pv

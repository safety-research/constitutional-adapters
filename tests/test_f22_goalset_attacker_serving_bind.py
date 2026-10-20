"""target_attacks_goalset.sbatch hands EVERY driver its pool member's serving manifest (2026-09-18).

THE DEFECT. The goal-set launcher passed only ``--attacker-engine vllm --attacker-endpoint <url>`` per driver. Without
``--attacker-serving <manifest>`` the driver takes ``attacker_serving.driver_fields(None)`` and stamps the store
``attacker_serving_manifest: null`` / ``attacker_quantization: "unrecorded"``; the engine-identity join then refuses to
resume that store under ANY identity (``EngineIdentityError``), so a walltime is a full restart, never a resume.
The same defect hit another launcher on 2026-09-18 (18 dead drivers, 754 rows lost). Every pool record
(``$WAVE_ROOT/attackers/<jobid>.json`` -- the files ``f22_live_attackers`` reads the URL list from) has carried the
member's engine-verified ``serving_manifest`` path all along; nothing here was reading it.

THE FIX under test: ``f22g_bind_attacker_manifests`` (defined in the sbatch itself; ``common.sh`` is certified and
untouched) joins url -> serving_manifest from the records and REFUSES, before any driver, a live URL with no record, a
record without the field, a path that is not a file, or two records naming one URL differently; the driver loop then
passes ``--attacker-serving`` beside ``--attacker-endpoint`` for the SAME member. These tests extract the bind and the
Phase-C loop from the sbatch's text and run them in bash over a synthetic wave root, with the interpreter replaced by a
recorder, so the pairing is asserted on the argv each driver would actually receive. Synthetic hosts, no cluster.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GOALSET = REPO / "scripts/sdfbase/full22/target_attacks_goalset.sbatch"


def _block(text: str, start: str, end: str, *, end_exact: bool = False) -> str:
    """The lines of `text` from the first line starting with `start` through the first later line that starts with
    `end` (or, with end_exact, IS `end`)."""
    lines = text.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(start))
    j = next(k for k in range(i, len(lines)) if (lines[k] == end if end_exact else lines[k].startswith(end)))
    return "\n".join(lines[i:j + 1]) + "\n"


def _bind_and_loop() -> tuple[str, str]:
    s = GOALSET.read_text()
    # from the map's declaration through the function's CALL line (exactly "f22g_bind_attacker_manifests", not its definition)
    bind = _block(s, "declare -A ATTACKER_MANIFEST_OF_URL=()", "f22g_bind_attacker_manifests", end_exact=True)
    assert "f22g_bind_attacker_manifests(){" in bind and bind.rstrip("\n").splitlines()[-1] == "f22g_bind_attacker_manifests"
    loop = _block(s, "# Phase C:", 'f22_log "launched ')
    return bind, loop


def _harness(tmp_path: Path, *, records: dict[str, dict], attackers: list[str], ngoals: dict[str, int],
             max_shards: int = 2) -> tuple[int, str, list[list[str]]]:
    """Run the sbatch's bind + driver loop over a synthetic wave root.

    records: {filename: json body} written under <wave>/attackers/; attackers: the ATTACKERS array (the live URLs
    f22_live_attackers would have bound); ngoals: {"<attack>__<cell>": n}. Returns (rc, combined output, one argv list
    per driver the loop launched -- the recorder replaces the interpreter, so an argv is exactly what a driver would get).
    """
    wave = tmp_path / "wave"; (wave / "attackers").mkdir(parents=True)
    for name, body in records.items():
        (wave / "attackers" / name).write_text(json.dumps(body) if isinstance(body, dict) else body)
    # a .pending marker and a .STOP sentinel sit in the same directory and must never be read as records
    (wave / "attackers" / "777.pending").write_text('{"url": "http://TESTHOST-pending:1"}')
    (wave / "attackers" / "778.STOP").write_text("")
    calls = tmp_path / "calls"; calls.mkdir()
    fake_py = tmp_path / "mpy"
    fake_py.write_text(
        "#!/bin/bash\n"
        # the bind's reader (`$MPY -`) and the sbatch's -c one-liners run on the real interpreter ...
        f'case "${{1:-}}" in -|-c) exec {sys.executable} "$@";; esac\n'
        # ... every other invocation is a DRIVER: record its argv, one argument per line, and exit 0
        f'f=$(mktemp {calls}/driver_XXXXXX); printf "%s\\n" "$@" > "$f"\n')
    fake_py.chmod(0o755)
    repo = tmp_path / "repo"; repo.mkdir()
    serve = tmp_path / "serve"; (serve / "drivers").mkdir(parents=True)
    goalsets = tmp_path / "goalsets"; goalsets.mkdir()
    bind, loop = _bind_and_loop()
    cells = []
    for pair in ngoals:
        cell = pair.split("__", 1)[1]
        if cell not in cells:
            cells.append(cell)
    atks = sorted({pair.split("__", 1)[0] for pair in ngoals})
    ngoals_decl = " ".join(f"[{k}]={v}" for k, v in ngoals.items())
    script = (
        "set -uo pipefail\n"
        'f22_log(){ echo "[log] $*"; }\n'
        'f22_fatal(){ f22_log "FATAL: $*"; exit 1; }\n'
        "f22_stop_requested(){ return 1; }\n"
        "sleep(){ :; }\n"                       # the loop's 3 s launch stagger
        f'WAVE_ROOT="{wave}"; MPY="{fake_py}"; REPO_DIR="{repo}"; SERVE_DIR="{serve}"\n'
        f'CFG="{tmp_path}/cfg.yaml"; GOALSET_DIR="{goalsets}"; PRIVATE_ROOT="{tmp_path}/private"\n'
        'PROXY_URL="http://TESTHOST-target:1"; TABLE="/placeholder/routing_table.json"; TARGET_L=61440\n'
        "DECODE_SPEC=greedy; DECODE_FLAGS=(--target-temperature 0.7); TARGET_PID=\"\"; declare -a DRIVER_PIDS=(); RC=0\n"
        f"ATTACKERS=({' '.join(attackers)})\n"
        f"ATKS=({' '.join(atks)}); CELLS_WITH_GOALS=({' '.join(cells)}); MAX_SHARDS={max_shards}\n"
        f"declare -A NGOALS=({ngoals_decl})\n"
        + bind + loop
        + "wait\n"
        'echo "NDRIVERS ${#DRIVER_PIDS[@]}"\n'
    )
    (tmp_path / "harness.sh").write_text(script)
    p = subprocess.run(["bash", str(tmp_path / "harness.sh")], capture_output=True, text=True,
                       env=dict(os.environ, LC_ALL="C.UTF-8"))
    argvs = [f.read_text().splitlines() for f in sorted(calls.iterdir())]
    return p.returncode, p.stdout + p.stderr, argvs


def _flag(argv: list[str], flag: str) -> str:
    hits = [argv[k + 1] for k, a in enumerate(argv) if a == flag]
    assert len(hits) == 1, (flag, argv)
    return hits[0]


def _two_members(tmp_path: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """Two live pool members, each with an existing engine-verified manifest, in the record shape attacker_pool.sbatch writes."""
    manifests = {}
    records = {}
    for jid, host, port in (("551054", "TESTHOST-a", 8792), ("551055", "TESTHOST-b", 8793)):
        man = tmp_path / "serving" / f"attacker_{jid}" / "attacker_serving_1.json"
        man.parent.mkdir(parents=True); man.write_text('{"schema": "attacker_serving/1", "engine_verified": true}')
        url = f"http://{host}:{port}"
        manifests[url] = str(man)
        records[f"{jid}.json"] = {"url": url, "node": host, "port": port, "job_id": jid, "model": "test-org/attacker-model",
                                  "snapshot": "0" * 40, "prefix_caching": True, "quantization": "none",
                                  "max_model_len": 131072, "sha": "1" * 40, "serving_manifest": str(man),
                                  "written_at": "2026-09-18T00:00:00+00:00"}
    return records, manifests


def test_shell_syntax():
    subprocess.run(["bash", "-n", str(GOALSET)], check=True)


def test_text_pairs_the_two_flags_per_driver_and_binds_before_phase_c():
    s = GOALSET.read_text()
    assert '--attacker-engine vllm --attacker-endpoint "$url" --attacker-serving "$aman" \\' in s
    assert 'aman="${ATTACKER_MANIFEST_OF_URL[$url]}"' in s
    # the bind runs after the pool is resolved and BEFORE the first driver; common.sh stays untouched (certified)
    assert s.index("\nf22_live_attackers\n") < s.index("\nf22g_bind_attacker_manifests\n") < s.index("# Phase C:")
    assert "f22g_bind_attacker_manifests" not in (REPO / "scripts/sdfbase/full22/common.sh").read_text()
    # every refusal is a fatal, never a warning that lets the driver launch unbound
    for reason in ("has no record under", "carries no serving_manifest", "is not a file", "two different serving manifests"):
        assert f'f22_fatal "' in s and reason in s, reason
    assert "--export=" not in s.replace("never --export=", "")


def test_each_driver_carries_its_own_members_manifest(tmp_path):
    """(i) --attacker-endpoint <url> AND --attacker-serving <that url's serving_manifest>, paired per driver, over a
    round-robin that alternates members -- a by-index bind that slipped by one would hand member A's identity to
    member B's traffic, and this reads the pairing off the argv itself."""
    records, manifests = _two_members(tmp_path)
    urls = list(manifests)
    rc, out, argvs = _harness(tmp_path, records=records, attackers=urls,
                              ngoals={"crescendo__none": 2, "crescendo__TESTCELL": 3}, max_shards=2)
    assert rc == 0, out
    assert len(argvs) == 4 and "NDRIVERS 4" in out, out          # 2 + min(3, 2) shards, all launched
    seen = []
    for argv in argvs:
        assert argv[:2] == ["scripts/jailbreak_run.py", "generate"], argv
        url = _flag(argv, "--attacker-endpoint"); man = _flag(argv, "--attacker-serving")
        assert man == manifests[url], (url, man, manifests)
        assert _flag(argv, "--attacker-engine") == "vllm"
        seen.append(url)
    assert set(seen) == set(urls), seen                          # both members drove; each was bound to itself
    for url in urls:
        assert f"attacker {url} -> serving manifest {manifests[url]}" in out
    assert "each bound to its member's serving manifest" in out
    # the .pending marker and the .STOP sentinel beside the records were never read as members
    assert "TESTHOST-pending" not in out


def test_a_live_url_with_no_record_refuses_before_any_driver(tmp_path):
    """(ii) an attacker that answers /v1/models but has no record cannot be bound; refuse before launch, loudly."""
    records, manifests = _two_members(tmp_path)
    orphan = "http://TESTHOST-c:8999"
    rc, out, argvs = _harness(tmp_path, records=records, attackers=list(manifests) + [orphan],
                              ngoals={"crescendo__none": 2}, max_shards=2)
    assert rc != 0, out
    assert f"FATAL: live attacker {orphan} has no record under" in out, out
    assert "UNRESUMABLE" in out and "refusing to launch" in out, out
    assert argvs == [] and "NDRIVERS" not in out and "launched" not in out, (argvs, out)


def test_a_record_without_serving_manifest_refuses_before_any_driver(tmp_path):
    """(iii) a record that predates the field would launch an 'unrecorded' driver -- the defect itself; refuse it."""
    records, manifests = _two_members(tmp_path)
    bare = dict(records["551055.json"]); del bare["serving_manifest"]; records["551055.json"] = bare
    rc, out, argvs = _harness(tmp_path, records=records, attackers=list(manifests), ngoals={"pair__none": 4}, max_shards=3)
    assert rc != 0, out
    assert "FATAL: attacker record for http://TESTHOST-b:8793 carries no serving_manifest" in out, out
    assert "UNRESUMABLE" in out, out
    assert argvs == [] and "launched" not in out, (argvs, out)


def test_a_manifest_path_that_is_not_a_file_refuses(tmp_path):
    """A stale path is not an identity a driver can read (attacker_serving.read_manifest would die mid-job); refuse up front."""
    records, manifests = _two_members(tmp_path)
    Path(records["551054.json"]["serving_manifest"]).unlink()
    rc, out, argvs = _harness(tmp_path, records=records, attackers=list(manifests), ngoals={"crescendo__none": 1})
    assert rc != 0, out
    assert "FATAL: attacker http://TESTHOST-a:8792: serving_manifest" in out and "is not a file" in out, out
    assert argvs == [], argvs


def test_two_records_naming_one_url_differently_refuse(tmp_path):
    """A stale record whose node:port a newer member reuses would make the bind a coin flip; refuse, never guess."""
    records, manifests = _two_members(tmp_path)
    other = tmp_path / "serving" / "attacker_540000" / "attacker_serving_1.json"
    other.parent.mkdir(parents=True); other.write_text("{}")
    stale = dict(records["551054.json"], job_id="540000", serving_manifest=str(other))
    records["540000.json"] = stale
    rc, out, argvs = _harness(tmp_path, records=records, attackers=list(manifests), ngoals={"crescendo__none": 1})
    assert rc != 0, out
    assert "two different serving manifests" in out and "http://TESTHOST-a:8792" in out, out
    assert argvs == [], argvs


def test_unreadable_record_is_skipped_but_its_url_still_cannot_bind(tmp_path):
    """A truncated record is not a bind; if its URL is live, the job refuses rather than reading half a file."""
    records, manifests = _two_members(tmp_path)
    records["551055.json"] = '{"url": "http://TESTHOST-b:8793", "serving_'          # cut mid-write
    rc, out, argvs = _harness(tmp_path, records=records, attackers=list(manifests), ngoals={"crescendo__none": 1})
    assert rc != 0, out
    assert "live attacker http://TESTHOST-b:8793 has no record" in out, out
    assert argvs == [], argvs


@pytest.mark.parametrize("n_members", [1, 3])
def test_round_robin_binds_every_member_to_itself(tmp_path, n_members):
    """With 1 or 3 members and 5 drivers the round-robin wraps; the pairing must survive the wrap."""
    manifests = {}; records = {}
    for k in range(n_members):
        man = tmp_path / "serving" / f"attacker_60000{k}" / "attacker_serving_1.json"
        man.parent.mkdir(parents=True); man.write_text("{}")
        url = f"http://TESTHOST-{k}:{9000 + k}"
        manifests[url] = str(man)
        records[f"60000{k}.json"] = {"url": url, "job_id": f"60000{k}", "serving_manifest": str(man)}
    rc, out, argvs = _harness(tmp_path, records=records, attackers=list(manifests), ngoals={"pair__none": 5}, max_shards=5)
    assert rc == 0, out
    assert len(argvs) == 5
    for argv in argvs:
        assert _flag(argv, "--attacker-serving") == manifests[_flag(argv, "--attacker-endpoint")]

"""lib/common.sh memprobe_archive_stale (memprobe-fix-2 + fix-3): the stale memprobe record is COPIED under a
UNIQUE name, the copy's sha256 read back against the record's, and ONLY THEN is the live name removed — a failed read-back leaves the record
live (the copy removed) so the next 02 run sees it and cannot submit on an unverified archive. Executed here for real under bash with
`repro_ssh` running the remote string locally (the `local:` seam's shape): the happy path, a same-second double detection (the second archive
gets `.2`, the first is untouched), a failed copy (`cp` refused -- by a shadowing stub everywhere, and by a directory without its write bit
where the kernel enforces modes: rc 1, record intact), a failed read-back and an unremovable live name."""
from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from pathlib import Path

import pytest

from _env import permission_denial_enforced

REPO = Path(__file__).resolve().parents[1]


def _run(tmp_path: Path, arm: str = "lora_r4_ce", *, fixed_stamp: str | None = None) -> subprocess.CompletedProcess:
    stamp = f"date(){{ echo {fixed_stamp}; }}\n" if fixed_stamp else ""
    script = (f"set -u\nsource {REPO}/reproduce/lib/common.sh\nrepro_ssh(){{ bash -c \"$2\"; }}\nrepro_warn(){{ echo \"WARN: $*\" >&2; }}\n{stamp}"
              f"memprobe_archive_stale cluster_a {tmp_path} {arm}; echo \"rc=$?\"\n")
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def _run_with_ssh(tmp_path: Path, ssh_body: str) -> subprocess.CompletedProcess:
    """The function under a `repro_ssh` stub with a fault injected by the remote string's shape."""
    script = (f"set -u\nsource {REPO}/reproduce/lib/common.sh\nrepro_ssh(){{ {ssh_body} }}\nrepro_warn(){{ echo \"WARN: $*\" >&2; }}\n"
              f"memprobe_archive_stale cluster_a {tmp_path} lora_r4_ce; echo \"rc=$?\"\n")
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def test_archive_moves_the_record_under_a_unique_name_reads_its_hash_back_and_moves_the_report_beside_it(tmp_path):
    rec = tmp_path / "memprobe.lora_r4_ce.json"; rec.write_text('{"arm": "lora_r4_ce", "schema": 1}')
    (tmp_path / "memprobe_report.lora_r4_ce.md").write_text("# report\n")
    want = hashlib.sha256(rec.read_bytes()).hexdigest()
    r = _run(tmp_path)
    assert r.returncode == 0 and r.stdout.strip().endswith("rc=0"), r.stdout + r.stderr
    dest = Path(r.stdout.splitlines()[0]); assert dest.name.startswith("memprobe.lora_r4_ce.json.stale.") and dest.is_file()
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == want and not rec.exists()
    assert (tmp_path / (dest.name + ".report.md")).read_text() == "# report\n" and not (tmp_path / "memprobe_report.lora_r4_ce.md").exists()


def test_a_same_second_double_detection_never_overwrites_the_first_archive(tmp_path):
    first = tmp_path / "memprobe.lora_r4_ce.json.stale.20261002T000000Z"; first.write_text("FIRST")
    rec = tmp_path / "memprobe.lora_r4_ce.json"; rec.write_text("SECOND")
    r = _run(tmp_path, fixed_stamp="20261002T000000Z")
    assert r.returncode == 0 and "rc=0" in r.stdout, r.stdout + r.stderr
    dest = Path(r.stdout.splitlines()[0])
    assert dest.name == "memprobe.lora_r4_ce.json.stale.20261002T000000Z.2" and dest.read_text() == "SECOND"
    assert first.read_text() == "FIRST" and not rec.exists()


@pytest.mark.parametrize("refusal", [
    "stub",
    pytest.param("kernel", marks=pytest.mark.skipif(
        not permission_denial_enforced(),
        reason="the kernel does not refuse this process a new entry in a directory it holds no write bit on (root or CAP_DAC_OVERRIDE), "
               "and that refusal of the copy is what this parametrization asserts")),
])
def test_a_failed_copy_leaves_the_record_in_place_and_returns_nonzero(tmp_path, refusal):
    rec = tmp_path / "memprobe.lora_r4_ce.json"; rec.write_text("EVIDENCE")
    if refusal == "kernel":
        # the real refusal: the directory holds no write bit, so the kernel rejects the copy's new entry; the mode goes back in a
        # finally so an assertion failure leaves tmp_path removable
        os.chmod(tmp_path, stat.S_IRUSR | stat.S_IXUSR)            # the directory refuses the rename
        try:
            r = _run(tmp_path)
        finally:
            os.chmod(tmp_path, stat.S_IRWXU)
    else:
        # the same refusal through a stub, so it runs where the kernel ignores a chmod of the directory (a root process): the remote
        # string runs for real with `cp` shadowed by a function the inner bash inherits, so the `cp -n ... && [ -f ] && printf` chain
        # fails on the copy itself, whoever runs the test
        r = _run_with_ssh(tmp_path, 'case "$2" in *"cp -n "*) cp(){ echo "cp: cannot create regular file: Permission denied" >&2; return 1; }; '
                                    'export -f cp; bash -c "$2";; *) bash -c "$2";; esac;')
    assert "rc=1" in r.stdout and "copy of" in r.stderr and "nothing cleared, nothing submitted" in r.stderr, r.stdout + r.stderr
    assert rec.read_text() == "EVIDENCE" and not list(tmp_path.glob("*.stale.*"))


def test_a_failed_read_back_leaves_the_live_record_removes_the_copy_and_returns_nonzero(tmp_path):
    rec = tmp_path / "memprobe.lora_r4_ce.json"; rec.write_text("EVIDENCE")
    (tmp_path / "memprobe_report.lora_r4_ce.md").write_text("# report\n")
    # the archive copy hashes differently from the record (a torn read, a corrupt copy): the live name must survive
    r = _run_with_ssh(tmp_path, 'case "$2" in *sha256sum*.stale.*) echo 0000000000000000000000000000000000000000000000000000000000000000;; *) bash -c "$2";; esac;')
    assert "rc=1" in r.stdout and "but the record hashed" in r.stderr and "copy removed, the record stays under its live name" in r.stderr, r.stdout + r.stderr
    assert rec.read_text() == "EVIDENCE" and not list(tmp_path.glob("*.stale.*")), list(tmp_path.iterdir())
    assert (tmp_path / "memprobe_report.lora_r4_ce.md").exists(), "the report is moved only after the record is archived"
    # the read-back itself fails (ssh outage): same outcome
    r = _run_with_ssh(tmp_path, 'case "$2" in *sha256sum*.stale.*) return 255;; *) bash -c "$2";; esac;')
    assert "rc=1" in r.stdout and "cannot hash the archive" in r.stderr and rec.read_text() == "EVIDENCE" and not list(tmp_path.glob("*.stale.*")), r.stderr


def test_a_verified_copy_whose_live_name_cannot_be_removed_returns_nonzero_with_the_record_live(tmp_path):
    rec = tmp_path / "memprobe.lora_r4_ce.json"; rec.write_text("EVIDENCE")
    r = _run_with_ssh(tmp_path, 'case "$2" in "rm -f \x27'"$tmp_path"'/memprobe.lora_r4_ce.json\x27 && "*) return 1;; *) bash -c "$2";; esac;'.replace("$tmp_path", str(tmp_path)))
    assert "rc=1" in r.stdout and "is verified but the live name" in r.stderr and "refusing to proceed" in r.stderr, r.stdout + r.stderr
    assert rec.read_text() == "EVIDENCE"
    copies = list(tmp_path.glob("*.stale.*")); assert len(copies) == 1 and copies[0].read_text() == "EVIDENCE", "the verified archive stays; the next run archives again"


def test_a_missing_record_returns_nonzero_without_inventing_an_archive(tmp_path):
    r = _run(tmp_path)
    assert "rc=1" in r.stdout and not list(tmp_path.glob("*.stale.*"))

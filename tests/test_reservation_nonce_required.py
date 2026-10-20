"""31e-fix-6 (folded into closer-fix-4): the reconciliation adopted a LONE candidate by elimination — after a
pre-sbatch crash, ONE same-owner same-name job another session submitted after the reservation was the sole candidate and was adopted without
matching the reservation's nonce: the wrong job recorded and watched, the token cleared, the intended resume never launched. And the fit path's
`--comment=sha:<sha>` made `submit_one` decline the nonce, so those jobs were permanently unverifiable even while live. Now adoption requires the
nonce READ BACK from the job (squeue %k); a candidate that cannot be verified refuses by name (the reservation standing, an operator decides); sha
and nonce travel in ONE comment (`sha:<sha>,nonce:<n>` — `,` because the funnel's flags are joined into one unquoted shell line)."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, job_ids  # noqa: E402
from test_resume_from_job import _submit  # noqa: E402
from test_resume_reservation_owner import _attempts, _foreign_job, _token  # noqa: E402
from test_resume_reservation_identity import _stamp, _world  # noqa: E402

sys.path.insert(0, str(LIB))
import submit_twin as TW  # noqa: E402


def test_a_single_foreign_post_token_job_is_never_adopted_by_elimination(tmp_path):
    w, j1 = _world(tmp_path); fake = w.fake["cluster_a"]
    r = _submit(w, "--resume-from-job", j1, SISTEER_TWIN_CRASH_AT="resume_after_reservation"); assert r.returncode == 97, r.stdout + r.stderr   # token written, sbatch never ran
    tok = _token(w)
    # ONE same-owner same-name job appears after the reservation (another session's), carrying no nonce: the sole candidate — refused, never adopted
    _foreign_job(w, "930", user=os.environ["USER"], submit=_stamp(2)); jobs = sorted(job_ids(fake))      # nothing may be launched beside it
    r = _submit(w, "--resume-from-job", j1)
    assert r.returncode == 3 and "1 qualifying candidate of name" in r.stderr and "'930'" in r.stderr and "no nonce read back" in r.stderr \
        and "never adopted by elimination" in r.stderr and "fail CLOSED" in r.stderr and "STANDS" in r.stderr, r.stdout + r.stderr
    assert _token(w) is not None and sorted(job_ids(fake)) == jobs and len(_attempts(w)) == 1 and "ADOPTED" not in (r.stdout + r.stderr)
    # the same job carrying THIS reservation's nonce in a COMBINED comment (the fit path's shape, sha first) is read back and adopted
    (fake / "jobs" / "930.argv").write_text(f"foreign --comment=sha:{'c' * 40},nonce:{tok['nonce']}\n")
    r = _submit(w, "--resume-from-job", j1); out = r.stdout + r.stderr
    assert "ADOPTED" in out and "positively identified by its nonce" in out and _attempts(w)[-1]["job_id"] == "930" and _token(w) is None, out


def test_the_sbatch_comment_carries_sha_and_nonce_together_and_the_parser_reads_either_shape():
    assert TW.comment_with_nonce(["--qos=high", f"--comment=sha:{'a' * 40}"], "0123456789ab") == ["--qos=high", f"--comment=sha:{'a' * 40},nonce:0123456789ab"]
    assert TW.comment_with_nonce(["--qos=high"], "0123456789ab") == ["--qos=high", "--comment=nonce:0123456789ab"]
    assert TW.comment_with_nonce(["--qos=high", "--comment=sha:x"], None) == ["--qos=high", "--comment=sha:x"]                 # no reservation: untouched
    assert TW.nonce_of_comment("nonce:0123456789ab") == "0123456789ab" and TW.nonce_of_comment(f"sha:{'a' * 40},nonce:0123456789ab") == "0123456789ab" and TW.nonce_of_comment("sha:x;nonce:0123456789ab") == "0123456789ab"
    assert TW.nonce_of_comment("sha:abc") == "" and TW.nonce_of_comment("(null)") == "" and TW.nonce_of_comment("") == "" and TW.nonce_of_comment("nonce:zz") == ""

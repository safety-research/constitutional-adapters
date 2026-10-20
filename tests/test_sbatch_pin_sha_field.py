"""pin-fix (2026-10-01, fits 830863 / 830864 died in 1–2 s): every sbatch pin check parsed the Slurm --comment as `sha:<everything>` and compared
it literally with the checkout HEAD, while the funnel appends the reservation nonce to that comment on EVERY submit
(submit_twin.comment_with_nonce -> `sha:<sha>,nonce:<n>`; 31e-fix-5/6), so every nonce-commented job refused itself ("the submission pinned
<sha>,nonce:…"). The pin is now the `sha` FIELD found BY NAME among `,`/`;`-separated key:value fields (nonce_of_comment's rule), and a
self-resubmitting reader carries the WHOLE comment (nonce included) to its successor. This test EXECUTES each sbatch's real pin block (the
`case "$COMMENT"` … `esac` lines and the comparison line that follows) and the real `--comment=` token of each self-resubmit in bash — never a
string match — with and without the nonce, in either field order, with `;`, and with a wrong sha."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
READERS = ("sdfbase_fit_cell", "sdfbase_derived_build", "hpo_trial", "memprobe_fit", "fit_refusal_direction", "serving_probe", "ddp_parity_smoke")
SELF_RESUBMITTING = ("sdfbase_fit_cell", "sdfbase_derived_build", "fit_refusal_direction", "serving_probe")
X, Z, NONCE = "a" * 40, "b" * 40, "c6f7af912a06"


def _src(name: str) -> str:
    return (REPO / "scripts" / "sbatch" / f"{name}.sbatch").read_text()


def _pin_block(name: str) -> str:
    m = re.search(r'(case "\$COMMENT" in\n.*?\nesac\n)(\[ "\$PIN" = "\$HEAD_SHA" \][^\n]*\n)', _src(name), re.S)
    assert m, f"{name}: no pin block (case \"$COMMENT\" … esac + the HEAD_SHA comparison)"
    return m.group(1) + m.group(2)


def _run(block: str, comment: str, head: str) -> tuple[int, str]:
    script = f'log(){{ echo "$*"; }}\nREPO=/repo\nFINAL_RC=0\nCOMMENT={comment!r}\nHEAD_SHA={head!r}\n{block}echo "PIN=$PIN"\n'
    r = subprocess.run(["bash", "-c", script], text=True, capture_output=True)
    return r.returncode, r.stdout + r.stderr


@pytest.mark.parametrize("name", READERS)
def test_the_pin_is_the_sha_field_by_name_in_any_order_and_a_wrong_sha_still_refuses(name):
    block = _pin_block(name)
    for ok in (f"sha:{X}", f"sha:{X},nonce:{NONCE}", f"nonce:{NONCE},sha:{X}", f"sha:{X};nonce:{NONCE}", f"sha:{X},nonce:{NONCE},extra:1"):
        rc, out = _run(block, ok, X)
        assert rc == 0 and f"PIN={X}" in out, (ok, out)
    rc, out = _run(block, f"sha:{Z},nonce:{NONCE}", X)
    assert rc == 3 and f"the submission pinned {Z}" in out and "refusing to run unvalidated code" in out, out
    rc, out = _run(block, f"nonce:{NONCE}", X)
    assert rc == 3 and "no --comment=sha:<40hex> pin" in out, out
    rc, out = _run(block, f"nonce:{NONCE},xsha:{X}", X)
    assert rc == 3 and "carries no sha:<40hex> field" in out, out                    # a field that merely ENDS in sha: is not the sha field


@pytest.mark.parametrize("name", SELF_RESUBMITTING)
def test_a_self_resubmitted_successor_carries_the_whole_comment_nonce_included(name):
    """The escalation trap's sbatch line used to re-mint `--comment="sha:$HEAD_SHA"` alone, dropping the nonce the funnel adopts the job by
    (squeue %k) — the successor would have been unverifiable. Execute the real `--comment=` token: it expands to the job's own comment."""
    tokens = re.findall(r'--comment=("[^"]*"|\S+)', _src(name))
    tokens = [t for t in tokens if t.startswith('"')]   # the resubmit line(s) quote theirs; the pin block's FATAL text names an unquoted literal form
    assert tokens, f"{name}: no self-resubmit --comment token"
    for tok in tokens:
        r = subprocess.run(["bash", "-c", f'COMMENT="sha:{X},nonce:{NONCE}"; HEAD_SHA="{X}"; printf "%s" --comment={tok}'], text=True, capture_output=True)
        assert r.returncode == 0 and r.stdout == f"--comment=sha:{X},nonce:{NONCE}", (tok, r.stdout, r.stderr)
    assert '--comment="sha:$HEAD_SHA"' not in _src(name)


def test_every_sbatch_that_pins_by_comment_is_covered_and_none_keeps_the_literal_parse():
    pinned = sorted(p.stem for p in (REPO / "scripts" / "sbatch").glob("*.sbatch") if '[ "$PIN" = "$HEAD_SHA" ]' in p.read_text())
    assert pinned == sorted(READERS), pinned
    for p in (REPO / "scripts" / "sbatch").glob("*.sbatch"):
        src = p.read_text()
        assert 'PIN="${COMMENT#sha:}" ;;' not in src, f"{p.name}: literal whole-comment pin parse"
        assert '--comment="sha:$HEAD_SHA"' not in src, f"{p.name}: a successor minted without the nonce"

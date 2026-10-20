"""The GSM8K landing cut = the assembly plus the follow-on lane (a960b9ea8 / 86277ee7f). The controls carry's clause (b) sees two producer
files that lane changed -- src/sisteer/capeval/tiers/inspect_served.py (five hunks in run_one_task; the archived controls children request
xstest / or_bench_hard_1k / mmlu, which never enter that runner) and src/sisteer/capeval/tinycap/completion.py (three hunks in finish_task;
every existing caller omits n_planned) -- each pinned by content hash against THIS tree, admitted for the CONTROLS leg only. Real objects:
the two files' real diffs from the two prior assembly trees to this tree, the archived controls / pasttense / misalign rows with their
pulled roots' evidence (local to the grading host; skipped where the archive / pull is absent)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import seedrep_carry_over as co  # noqa: E402
import seedrep_launch_plan as lp  # noqa: E402
import test_seedrep_fj2_carry_over as F  # noqa: E402
from _env import host_site_path  # noqa: E402  (tests/ helper: a host-only data path through the site config)

W22, W24 = "56aff576e2975804602aec7d12ba1c98c805adfd", "6fdd5e52bd6aec62b2389c185bd1d094d902703d"
FG1 = "a960b9ea8d8f08f9a82d39db578c4f9a323332dc"   # the fg lane's launch tree the admitted hunks name ("W22/W24->launch fg1 a960b9ea8"): the tree the pins were cut against
FILES = {"src/sisteer/capeval/tiers/inspect_served.py": ("D194-fg-inspect_served", co.D194_INSPECT_SERVED_HUNKS, co.ADMITTED_REASON_D194_INSPECT_SERVED),
         "src/sisteer/capeval/tinycap/completion.py": ("D194-fg-completion", co.D194_COMPLETION_HUNKS, co.ADMITTED_REASON_D194_COMPLETION)}
RS = host_site_path("$STATE/romp/candidate-si-refit/run-state/full22/seedrep_battery")
PEEK = Path("/tmp/launch_vol_peek")
CARRIES = [("controls", "6fdd5e5", "controls__s1__base__shard00", PEEK / "6fdd5e5/stores/S5_controls/01M2YNWDRFVSAT7XFYMEXEPPQR-dc4bd2f90303/01M2YNWDRFVSAT7XFYMEXEPPQR-dc4bd2f90303"),
           ("pasttense", "56aff57", "pasttense__s1__base__shard00", PEEK / "56aff57/stores/S5_pasttense/01M2XD1H6WDGXZ65BS0NRNMZRR-ccf6feacddc8/01M2XD1H6WDGXZ65BS0NRNMZRR-ccf6feacddc8"),
           ("misalign", "56aff57", "misalign__s1__base__shard00", PEEK / "56aff57/stores/S5_misalign/01M2XD1H6W9CT4XFNSTZ7VED8R-b4cbfce42621/01M2XD1H6W9CT4XFNSTZ7VED8R-b4cbfce42621")]
FG_FILES = set(FILES) | {"src/sisteer/capeval/tinycap/contracts.py", "scripts/sdfbase/seedrep/audit_common.py", "scripts/sdfbase/record_resolved_sampler.py"}


def _sha_exists(sha):
    return subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True).returncode == 0


def _at(sha, path):
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{sha}:{path}"], capture_output=True, text=True)
    assert r.returncode == 0, (sha, path)
    return r.stdout


@pytest.mark.skipif(not (_sha_exists(W22) and _sha_exists(W24) and _sha_exists(FG1)), reason="the W22 / W24 / fg1 shas are not in this clone")
def test_the_real_w22_and_w24_to_the_fg1_launch_tree_diffs_of_both_files_are_exactly_the_pinned_hunk_sets():
    """The admitted hunk tables are the prior-assembly -> launch-tree (a960b9ea8) diffs, exactly (the label every admitted hunk carries). The
    pins were cut against "this tree" when this tree WAS the launch tree; every later edit to either file (the 2026-09-23/29 capeval landings
    on main, the 2026-10-04 GSM8K scorer fixes) adds hunks the controls carry does not admit, so a diff against the working tree is 55 hunks
    today and the admission is refused there — the rule working as designed, not a statement the table makes. The table's statement is
    historical and is held to the recorded launch tree (a full clone exposes this: a shallow clone skipped the test for want of those shas)."""
    for path, (name, table, reason) in FILES.items():
        here = _at(FG1, path)
        assert _at(W22, path) == _at(W24, path), f"{path}: W22 and W24 differ -- the W22/W24 label is wrong"
        prev = _at(W24, path)
        assert prev != here
        hunks = co.diff_hunks(prev, here)
        assert {h["sha256"] for h in hunks} == set(table[path]), (path, [h["head"] for h in hunks])         # the table IS the diff: nothing more, nothing less
        assert len(hunks) == {"src/sisteer/capeval/tiers/inspect_served.py": 5, "src/sisteer/capeval/tinycap/completion.py": 3}[path]
        adm = co.d111_admitted_difference(prev, here, path, leg="controls")
        assert adm is not None and adm["tables"] == [name] and adm["kind"] == co.ADMITTED_KIND_OFF_PATH and adm["reason"] == reason and adm["n_hunks"] == len(hunks)
        assert all(h["label"].startswith("W22/W24->launch fg1 a960b9ea8") for h in adm["hunks"])
        # CONTROLS only: a gsm8k carry (whose runner inspect_served.py IS), a static carry, a leg-less carry never see the table
        for leg in ("gsm8k", "pasttense", "misalign", None):
            assert co.d111_admitted_difference(prev, here, path, leg=leg) is None, (path, leg)
        # an extra hunk beyond the pin refuses the file even for a controls carry
        assert co.d111_admitted_difference(prev, here + "\nZ = 1\n", path, leg="controls") is None
        # through producer_set_verdict over a fake git serving the REAL blobs
        git = F.FakeGit({W24: {path: "prev"}, "this": {path: "here"}}, {(W24, path): prev, ("this", path): here})
        v = co.producer_set_verdict([path], prev_sha=W24, this_sha="this", git=git, leg="controls")
        assert v["ok"] and [a["path"] for a in v["admitted_differences"]] == [path] and v["admitted_differences"][0]["tables"] == [name]
        assert not co.producer_set_verdict([path], prev_sha=W24, this_sha="this", git=git, leg="gsm8k")["ok"]


def _carry(w7, sid, root):
    prior = co.load_prior_source(RS / "prior" / w7)
    prow = co.archived_row(prior, sid)
    s5 = (prior.get("smoke_waves") or {}).get("S5")
    srow = lp.smoke_row(prow, lp.smoke_identity(prior["plan"], "S5", s5), max_children=len(prow["children"]), max_timeout_s=int(prow["modal"]["timeout_s"]))
    ev = json.loads((root / "modal_evidence.json").read_text())
    git = co.default_git(REPO)
    pset = co.producer_set(srow, ev, W=prior["W"])
    imports = co.direct_imports(git, prior["W"], pset["all"])
    gating = sorted((set(pset["all"]) | {p for ps in imports.values() for p in ps}) - set(co.CONTAINER_MODULES))
    head = git(["rev-parse", "HEAD"])[1].strip()
    return prior, srow, gating, co.producer_set_verdict(gating, prev_sha=prior["W"], this_sha=head, git=git, leg=str(srow["leg"]))


@pytest.mark.skipif(not all((RS / "prior" / w7 / "launch_plan_draft.json").is_file() and (root / "modal_evidence.json").is_file() for _l, w7, _s, root in CARRIES)
                    or not (_sha_exists(W22) and _sha_exists(W24)), reason="the W22 / W24 archives and pulled S5 roots are not on this machine")
def test_the_real_w24_controls_carry_re_derives_ok_under_d194_fg_and_the_w22_carries_see_no_fg_file():
    """The real-object re-derivation fj24 ran: the archived rows (their S5 smoke shape), the pulled roots' evidence, this clone's git, against
    THIS tree's HEAD -- the W24 controls carry's producer set differs on exactly the two fg files, both admitted under D194-fg; the W22
    pasttense and misalign carries' gating sets hold no fg producer file and re-derive ok (the fg readers in them are NON_EVAL modules)."""
    for leg, w7, sid, root in CARRIES:
        prior, srow, gating, v = _carry(w7, sid, root)
        if leg == "controls":
            adm = {a["path"]: a for a in v["admitted_differences"]}
            assert set(FILES) <= set(adm), (v.get("why"), sorted(adm))
            for path, (name, table, _reason) in FILES.items():
                assert adm[path]["tables"] == [name] and {h["sha256"] for h in adm[path]["hunks"]} == set(table[path]) and adm[path]["kind"] == co.ADMITTED_KIND_OFF_PATH
            assert v["ok"] is True and v.get("why") is None, v.get("why")
            assert not (FG_FILES - set(FILES)) & {p["path"] for p in v["files"] if not p["identical"]}          # contracts / audit_common / record_resolved_sampler: not gating
        else:
            assert v["ok"] is True and v.get("why") is None, (leg, v.get("why"))
            assert not FG_FILES & {p["path"] for p in v["files"] if not p["identical"]}, (leg, sorted(FG_FILES & {p["path"] for p in v["files"] if not p["identical"]}))
            assert not any(a["path"] in FG_FILES for a in v["admitted_differences"]), leg
            assert not set(FILES) & set(gating), (leg, sorted(set(FILES) & set(gating)))                          # the two admitted files are not even in these carries' sets

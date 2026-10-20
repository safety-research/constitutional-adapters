"""A REFIT of an already-published ReLU identity lives in its own bank and nothing of it reaches the registry record — the artefact's
extra.init_scheme and the new object ids are the discriminator (fit_replicate_label is free text, v_seed_spread counts as a seed replicate,
so it stays NULL). The front door's `--refit <tag>` keeps a LANE-SIDE identity only: it DERIVES the bank full22_<tag> from the tag (one
knob), mints every name with the tag, refuses a contradicting bank and a foreign bank without a tag, refits only the ReLU units, requires
the COMMITTED fit config's psr init scheme to be non-zero (the overrides file is not what the job read), and scopes registry done-ness by
that scheme + an artifact location under the REAL bank layout …/exp/objects/sdfbase/full22_<tag>/ (never the adapter's staging spelling
$STAGE/sdfbase/full22_objects/…, which a bare glob matched and which hid the random-init canonical LoRA parents). The sbatch scripts and
the trainer are untouched."""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPRO, VENV_PY, run_stage  # noqa: E402

FRONT = REPO / "reproduce" / "04_fit_objects.sh"
TAG = "lecun"
SCHEME = "random_lecun_1e-3"


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    root = tmp_path_factory.mktemp("relurefit")
    site = root / "site"
    shutil.copytree(REPRO / "site.example", site)
    with open(site / "config.env", "a") as fh:
        fh.write(f"\nSISTEER_STATE_DIR={root / 'state'}\nSISTEER_PYTHON={VENV_PY}\nSISTEER_REGISTRY_ROOT={root / 'registry'}\n")
    return site


def _04(site, *args, model="qwen35_4b_demo", env=None):
    r = run_stage("04_fit_objects.sh", "--dry-run", "--model", model, *args, site=site, env=env or {}, timeout=600)
    return r, r.stdout + r.stderr


def test_04_the_bank_is_bound_to_the_tag_and_a_foreign_bank_alone_refuses(site):
    """The bank gates precede the committed config's scheme gate: a contradicting SISTEER_FIT_BANK, a foreign bank without a tag and a
    malformed tag each refuse by name before any config is read (no committed config on this tree carries a non-zero psr init scheme, so
    the positive --refit route is exercised through the pinned-blob harness below, never through the dry run)."""
    for bank in ("full22", "full22_lecum", "./full22", "full22_lecun/"):
        r, out = _04(site, "--object", "control_relu", "--refit", TAG, env={"SISTEER_FIT_BANK": bank})
        assert r.returncode != 0 and "binds the bank to full22_lecun" in out, (bank, out[-2000:])
        assert "DRY-RUN: ssh" not in out, bank
    for bank in ("full22_lecun", "./full22", "full22/"):
        r, out = _04(site, "--object", "control_relu", env={"SISTEER_FIT_BANK": bank})
        assert r.returncode != 0 and "without --refit" in out, (bank, out[-2000:])
        assert "DRY-RUN: ssh" not in out, bank
    r, out = _04(site, "--object", "control_relu", "--refit", "Bad-Tag")
    assert r.returncode != 0 and "--refit must match" in out, out[-2000:]


def test_04_refit_needs_the_committed_configs_non_zero_psr_init_scheme(site):
    """The scheme is read from the fit config the jobs run under, not from the overrides file: the configured-not-run model's
    committed config is zero-init -> a refit refuses by name, whatever the overrides say (the override-contradiction and accepted-stamp cases
    run through the pinned-blob harness below, against a tiny repo whose committed config carries a non-zero scheme)."""
    r, out = _04(site, "--object", "control_relu", "--refit", TAG, model="qwen35_4b_demo")
    assert r.returncode != 0 and "declares train_kwargs.psr_ce_all.init_scheme 'zero'" in out and "D1816" in out, out[-2500:]
    assert "DRY-RUN: ssh" not in out
    r, out = _04(site, "--object", "control_relu", "--refit", TAG, "--overrides", str(REPRO / "overrides" / "paper_defaults.env"))
    assert r.returncode != 0 and "declares train_kwargs.psr_ce_all.init_scheme 'zero'" in out and "DRY-RUN: ssh" not in out, out[-2500:]


def test_04_canonical_dry_run_carries_no_tag_no_suffix_and_the_paper_bank(site):
    r, out = _04(site, "--object", "control_relu")
    assert r.returncode == 0, out[-3000:]
    ln = [ln for ln in out.splitlines() if "sdfbase_fit_cell.sbatch" in ln and "DRY-RUN" in ln][0]
    assert ln.rstrip("'").endswith("qwen35_4b_demo_base constv2ctl2-full psr_ce_all full22"), ln
    assert "-J fit-qwen35_4b_demo-control_relu " in ln and "-lecun" not in ln, ln
    assert "--gres=" not in ln and "--cpus-per-task=" not in ln and "--mem=" not in ln and "SDFBASE_FIT_WIDTH=1" in ln and "--export" not in ln, ln    # No fit_width line = width 1, no width flags
    assert any(s.startswith("submitted ") and s.endswith(" control_relu") for s in out.splitlines()), out[-2000:]
    assert "refit=" not in out and "REFIT" not in out


# ----------------------------------------------------------------------------- the registry scope

def _bash_fn(src: str, name: str) -> str:
    start = src.index(f"\n{name}(){{") + 1
    end = src.index("\n}\n", start) + 3
    return src[start:end]


def _scope(tag: str | None, which: str, scheme: str = SCHEME) -> str:
    fn = _bash_fn(FRONT.read_text(), "refit_scope_sql")
    script = f'REFIT="{tag or ""}"; REFIT_SCHEME="{scheme}"; DEFAULT_BANK=full22\n{fn}\nrefit_scope_sql {which}\n'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout.strip()


STAGED_CTL = "/home/x/stage/sdfbase/full22_objects/qwen35_4b_demo/cells/base__constv2ctl2-full__psr_ce_all/object.pt"
BANK_CTL = "/shared/x/exp/objects/sdfbase/full22_lecun/qwen35_4b_demo_base/cells_all/constv2ctl2-full__psr_ce_all/object.pt"
STAGED_V = "/home/x/stage/sdfbase/full22_objects/qwen35_4b_demo/cells/derived__constv2-mix-full__psr_ce_all_minus_ctl/object.pt"
MIX_SI, CTL_SI = "constv2-mix-full", "constv2ctl2-full"                    # 04's own spellings (line `MIX_SI=…; CTL_SI=…`)
CORPUS = {"constitution": MIX_SI, "control": CTL_SI}                        # a fitted row's fit_corpus_id by role; derived rows carry none


def test_the_bank_glob_is_true_on_the_bank_layout_and_false_on_the_adapter_staging_spelling():
    """The two read-only cases: the staging dir must read FALSE, the bank TRUE — on either cluster's shared root and with any uri
    prefix, in both location-list shapes."""
    db = sqlite3.connect(":memory:")
    pred = _scope(TAG, "self")
    i = pred.index("EXISTS (SELECT 1 FROM json_each("); loc = pred[i:pred.index("/*')", i) + len("/*')")]
    assert loc.endswith("GLOB '*/exp/objects/sdfbase/full22_lecun/*')"), loc
    assert "*/sdfbase/full22_*" not in _scope(None, "self").replace("/exp/objects/sdfbase/full22_*", "")
    db.execute("CREATE TABLE object (object_id TEXT, extra_json TEXT, artifact_locations_json TEXT)")
    cases = {"staged_obj": json.dumps([{"bytes": 1, "uri": STAGED_CTL}]), "staged_path": json.dumps([STAGED_CTL]),
             "bank_obj": json.dumps([{"bytes": 1, "uri": BANK_CTL}]), "bank_path": json.dumps([BANK_CTL]),
             "bank_cluster_b": json.dumps(["/workspace/x/exp/objects/sdfbase/full22_lecun/qwen35_4b_demo_base/cells_all/c/object.pt"]),
             "bank_uri_prefix": json.dumps([{"uri": "file:///shared/x/exp/objects/sdfbase/full22_lecun/k/cells_all/c/object.pt"}]),
             "other_bank": json.dumps([BANK_CTL.replace("full22_lecun", "full22_other")]),
             "canon_bank": json.dumps([BANK_CTL.replace("full22_lecun", "full22")]),
             "none": None}
    for k, v in cases.items():
        db.execute("INSERT INTO object VALUES (?,?,?)", (k, json.dumps({"init_scheme": SCHEME}), v))
    hit = {r[0] for r in db.execute(f"SELECT object_id FROM object WHERE {loc}")}
    assert hit == {"bank_obj", "bank_path", "bank_cluster_b", "bank_uri_prefix"}, hit


def test_refit_scope_partitions_refit_rows_from_canonical_ones_including_random_init_lora_parents():
    """The scope reads two facts the native hook writes: extra.init_scheme (the artefact's) and the artifact location (the cell's cluster
    path, in either list shape). The canonical LoRA parents are random-init too and sit at the adapter's staging spelling: they
    and their subtraction stay canonical. No label is involved anywhere."""
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE object (object_id TEXT PRIMARY KEY, parent_object_ids_json TEXT, extra_json TEXT, artifact_locations_json TEXT, status TEXT)")
    zero, lecun = json.dumps({"init_scheme": "zero"}), json.dumps({"init_scheme": SCHEME})
    staged = json.dumps([{"bytes": 1, "uri": STAGED_CTL}])
    rows = [("ctl_zero", None, zero, staged), ("mix_zero", None, zero, staged),                                  # the canonical ReLU pair
            ("ctl_lf", None, lecun, json.dumps([{"bytes": 1, "uri": BANK_CTL}])),                                 # the refits (both shapes)
            ("mix_lf", None, lecun, json.dumps([BANK_CTL.replace("constv2ctl2-full", "constv2-mix-full")])),
            ("mix_other", None, lecun, json.dumps([BANK_CTL.replace("full22_lecun", "full22_other")])),           # another refit bank
            ("ctl_lora_canon", None, lecun, json.dumps([{"bytes": 1, "uri": STAGED_CTL.replace("psr_ce_all", "lora_r4_ce")}])),
            ("mix_lora_canon", None, lecun, json.dumps([BANK_CTL.replace("full22_lecun", "full22").replace("psr_ce_all", "lora_r4_ce")])),
            ("v_lora_canon", json.dumps(["mix_lora_canon", "ctl_lora_canon"]), lecun, staged),
            ("ctl_lf_orphan", None, None, json.dumps([BANK_CTL.replace("constv2ctl2-full", "constv2ctl2-full-orphan")])),   # NULL init at a refit location
            ("v_zero", json.dumps(["mix_zero", "ctl_zero"]), None, None),
            ("v_lf", json.dumps(["mix_lf", "ctl_lf"]), None, None),
            ("v_lf_orphan2", json.dumps(["mix_lf", "ctl_lf_orphan"]), None, None),                 # refit-fix-4: the SECOND parent is the orphan
            ("v_straddle", json.dumps(["mix_zero", "ctl_lf"]), None, None),                        # one canonical parent, one refit parent
            ("v_missing", json.dumps(["mix_lf", "no_such_row"]), None, None),                      # a parent id with no object row
            ("v_missing_canon", json.dumps(["mix_zero", "no_such_row"]), None, None),              # refit-fix-5: a CANONICAL first parent, a second not folded yet
            ("v_empty", "[]", None, None),                                                         # … and an empty list
            ("v_orphan", None, None, None)]
    db.executemany("INSERT INTO object VALUES (?,?,?,?, 'reference')", rows)

    def ids(clause: str) -> set:
        return {r[0] for r in db.execute(f"SELECT object_id FROM object WHERE {clause} AND status IN ('reference','approved')")}

    assert ids(_scope(TAG, "self")) == {"ctl_lf", "mix_lf"}
    assert ids(_scope(None, "self")) == {"ctl_zero", "mix_zero", "ctl_lora_canon", "mix_lora_canon", "v_lora_canon", "v_zero", "v_lf", "v_lf_orphan2", "v_straddle", "v_missing", "v_missing_canon", "v_empty", "v_orphan"}
    # the parent scopes read EVERY parent: all parents proper refit rows -> refit; every parent RESOLVING and none at a refit
    # location -> canonical (the INNER JOIN read an unfolded second parent as absent and v_missing_canon as canonical);
    # an orphaned or missing parent, an empty list, or parents straddling the two sets -> neither
    assert ids(f"parent_object_ids_json IS NOT NULL AND {_scope(TAG, 'parent')}") == {"v_lf"}
    assert ids(f"parent_object_ids_json IS NOT NULL AND {_scope(None, 'parent')}") == {"v_zero", "v_lora_canon"}
    # a zero-init committed config: no refit can exist, the scope is a no-op for every unit
    assert _scope(None, "self", scheme="zero") == "1 = 1" and _scope(TAG, "parent", scheme="zero") == "1 = 1"
    assert "fit_replicate_label" not in _scope(TAG, "self") + _scope(None, "parent")


def test_04_wires_the_scope_into_the_one_query_and_keeps_the_extracted_fragments_self_sufficient():
    src = FRONT.read_text()
    roi = _bash_fn(src, "registry_object_ids")
    assert 'scope="$(refit_scope_sql "${2:-self}")"' in roi and "WHERE $1 AND $scope AND status IN ('reference','approved')" in roi
    assert '$(refit_mismatch_sql "${2:-self}")' in roi and '$(derived_integrity_sql "${2:-self}")' in roi     # refit-fix-5: both gates ride the one query
    assert '$(derived_order_sql "${2:-self}")' in roi and "'parents_reversed'" in roi                             # refit-fix-6: and the order gate
    # refit-fix-6 (b): the dispatcher revalidates a refit FITTED marker through the same candidate-scoped query fit_unit issues
    disp = src[src.index("for u in $UNITS; do"):src.index('repro_say "04_fit_objects done')]
    assert '&& ! refit_fitted_marker_valid "$u"; then' in disp and 'the marker is NOT honoured; the unit runs (D2126)' in disp
    rfm = _bash_fn(src, "refit_fitted_marker_valid")
    assert "registry_object_ids \"object_kind='$(kind_for \"$fam\")' AND object_role='$role' AND fit_model_id='$FIT_ID' AND fit_corpus_id='$(si_for \"$role\")' AND campaign='$CAMPAIGN'\" self" in rfm, rfm
    assert '|| repro_fatal "$1: registry_object_ids refused' in rfm and "D1885 / D2126" in rfm and re.search(r"^\s+local role=.* ids$", rfm, re.M), rfm
    rd = _bash_fn(src, "registry_done")
    assert "registry_object_ids \"($2) AND campaign='$CAMPAIGN'\" \"${4:-self}\"" in rd
    assert src.count('"" parent && return 0') == 1 and src.count("v_liveness_ready parent && return 0") == 1
    assert "AND campaign='$CAMPAIGN'\" parent)" in _bash_fn(src, "difference_relu_done_marker_valid")
    for needle in ('name="fit-${REPRO_MODEL}-${unit}${REFIT:+-$REFIT}"', 'name="derived-${REPRO_MODEL}${REFIT:+-$REFIT}"',
                   'local unit="difference_relu${REFIT:+/$REFIT}"', 'registry_done "${unit//\\//.}"',
                   'local sub_unit="$unit/prep"; [ -n "$REGISTRY_UNREADY_IDS" ] && sub_unit="$unit/liveness"',
                   'submit "$unit${REFIT:+/$REFIT}" "$name" scripts/sbatch/sdfbase_fit_cell.sbatch',
                   'repro_done "$REPRO_MODEL" "$STAGE" "$u${REFIT:+.$REFIT}"', 'registry_done "$unit${REFIT:+.$REFIT}" "object_kind=',
                   'REFIT_UNITS="control_relu constitution_relu difference_relu"', 'REFIT_BANK="${DEFAULT_BANK}_${REFIT}"', 'BANK="$REFIT_BANK"',
                   '--env "SDFBASE_FIT_WIDTH=$FIT_WIDTH" --flags'):
        assert needle in src, needle
    # the bank is final before any cluster path is built from it
    assert src.index('BANK="$REFIT_BANK"') < src.index('BANK_ROOT="$SHARED_ROOT/exp/objects/sdfbase/$BANK"')
    # nothing of the refit identity reaches the job env or the registry record
    assert "SISTEER_FIT_REPLICATE_LABEL" not in src and "fit_replicate.json" not in src
    trainer = (REPO / "scripts" / "train_two_si_objects.py").read_text()
    assert "fit_replicate" not in trainer and "SISTEER_FIT_REFIT" not in trainer
    # the harness-extracted fragments reference no helper the harnesses do not define (tests/test_derived_build_per_object.py,
    # tests/test_addendum_table_and_refit_discipline.py execute them with their collaborators stubbed)
    frag = (src[src.index("difference_relu(){"):src.index("refusal_direction(){")]
            + src[src.index("for u in $UNITS; do"):src.index('repro_say "04_fit_objects done')]
            + src[src.index("arm_for(){"):src.index("# --- difference_lora")])
    for gone in ("refit_scope_sql", "refit_orphans_sql", "refit_mismatch_sql", "derived_integrity_sql", "derived_order_sql", "REFIT_SCHEME", "refit_label_check", "unit_id", "marker_unit", "job_suffix"):
        assert gone not in frag, gone
    assert src.index("for u in $UNITS; do") > src.index("difference_relu(){")


# ----------------------------------------------------------------------------- refit-fix-1: the PINNED blob decides

def _gate_block(src: str) -> str:
    return src[src.index("# --- the refit identity (RELU-REFIT lane"):src.index("# --- cluster paths")]


def _run_gate(repo: Path, *, refit: str | None = "lecun", override: str = "zero", sha: str | None = None, units: str = "control_relu") -> subprocess.CompletedProcess:
    """The identity gate executed as a bash fragment against a REAL tiny git repo: REPRO_REPO/SHA/FIT_ID as the front door sets them,
    repro_python the venv's, repro_fatal a visible exit 7."""
    src = FRONT.read_text()
    sha = sha or subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    script = f"""
set -u
REPRO_REPO={repo}; SHA={sha}; FIT_ID=fakefit; UNITS={units}; REFIT="{refit or ''}"; DEFAULT_BANK=full22; BANK=full22
SISTEER_RUN_OVERRIDES=/overrides/{'paper_defaults' if override == 'zero' else 'stamped'}.env; OVERRIDE_INIT_SCHEME={override}
repro_python(){{ echo {VENV_PY}; }}
repro_fatal(){{ echo "REFUSED: $*"; exit 7; }}
{_gate_block(src)}
echo "SCHEME=$REFIT_SCHEME BANK=$BANK"
"""
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def _tiny_repo(tmp_path: Path, scheme: str) -> Path:
    repo = tmp_path / "repo"; (repo / "configs").mkdir(parents=True)
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
           "HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True, env=env)
    _write_cfg(repo, scheme)
    subprocess.run(["git", "-C", str(repo), "add", "configs"], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "-c", "commit.gpgsign=false", "commit", "-q", "-m", "fits yaml"], check=True, env=env)
    return repo


def _write_cfg(repo: Path, scheme: str) -> None:
    (repo / "configs" / "sdfbase_fakefit_fits.yaml").write_text(
        f"model: fake/model\nfit_width: 4\ntrain_kwargs:\n  default: {{}}\n  psr_ce_all:\n    lr: 0.001\n    init_scheme: {scheme}\n")


def test_the_refit_scheme_is_read_from_the_pinned_blob_and_a_dirty_fits_yaml_refuses(tmp_path):
    """SHA pins what the cluster runs; the scheme must come from `git show $SHA:<yaml>`, and a working-tree yaml that
    differs from the blob refuses by name — under the paper-default `zero` override too, where stamp_init_scheme's dirty check never
    runs. Zero submissions: the gate exits before any unit."""
    repo = _tiny_repo(tmp_path, SCHEME)
    r = _run_gate(repo)                                                             # clean: the blob decides
    assert r.returncode == 0 and r.stdout.strip().endswith(f"SCHEME={SCHEME} BANK=full22_lecun"), r.stdout + r.stderr
    _write_cfg(repo, "zero")                                                        # working tree edited to zero, blob random
    r = _run_gate(repo)
    assert r.returncode == 7 and "differs from the blob pinned at" in r.stdout and "D1853" in r.stdout, r.stdout + r.stderr
    _write_cfg(repo, SCHEME)                                                        # clean again
    assert _run_gate(repo).returncode == 0
    repo2 = _tiny_repo(tmp_path / "two", "zero")                                    # blob ZERO, working tree edited to random
    _write_cfg(repo2, SCHEME)
    r = _run_gate(repo2)
    assert r.returncode == 7 and "differs from the blob pinned at" in r.stdout, r.stdout + r.stderr
    _write_cfg(repo2, "zero")                                                       # clean but zero-init committed config: refused on the scheme
    r = _run_gate(repo2)
    assert r.returncode == 7 and "declares train_kwargs.psr_ce_all.init_scheme 'zero'" in r.stdout and "at " in r.stdout, r.stdout + r.stderr
    r = _run_gate(repo, sha="unknown")                                              # no git HEAD to pin to
    assert r.returncode == 7 and "no pinned fits config" in r.stdout, r.stdout + r.stderr
    r = _run_gate(repo, refit=None)                                                 # the canonical path reads the blob's scheme, refuses nothing
    assert r.returncode == 0 and r.stdout.strip().endswith(f"SCHEME={SCHEME} BANK=full22"), r.stdout + r.stderr
    r = _run_gate(repo, override="random_other")                                    # a NON-ZERO override naming another scheme still contradicts
    assert r.returncode == 7 and "OVERRIDE_INIT_SCHEME=random_other" in r.stdout, r.stdout + r.stderr


def _orphans(scheme: str = SCHEME, which: str = "self") -> str:
    fn = _bash_fn(FRONT.read_text(), "refit_orphans_sql")
    script = f'REFIT_SCHEME="{scheme}"; DEFAULT_BANK=full22\n{fn}\nrefit_orphans_sql {which}\n'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout.strip()


def test_a_null_init_row_at_a_refit_location_falls_in_neither_scope_and_is_named_not_lost():
    """SQL NULL in `NOT (...)` drops a row with no recorded init scheme at a refit-bank location from BOTH scopes; the front
    door names such rows (refit_orphans_sql, on stderr inside registry_object_ids) instead of letting them vanish."""
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE object (object_id TEXT PRIMARY KEY, parent_object_ids_json TEXT, extra_json TEXT, artifact_locations_json TEXT, status TEXT)")
    V_LOC = json.dumps(["/shared/x/exp/objects/sdfbase/full22_lecun/derived/psr_paired_write/qwen35_4b_demo_base/constv2-mix-full__psr_ce_all_minus_ctl/object.pt"])
    rows = [("orphan_lf", None, None, json.dumps([BANK_CTL])),                                   # NULL extra at a refit location
            ("orphan_lf_noinit", None, json.dumps({"other": 1}), json.dumps([BANK_CTL])),         # extra without init_scheme
            ("ctl_lf", None, json.dumps({"init_scheme": SCHEME}), json.dumps([BANK_CTL])),         # a proper refit row
            ("mix_lf", None, json.dumps({"init_scheme": SCHEME}), json.dumps([BANK_CTL.replace("constv2ctl2-full", "constv2-mix-full")])),
            ("ctl_zero", None, json.dumps({"init_scheme": "zero"}), json.dumps([{"uri": STAGED_CTL}])),
            ("canon_null", None, None, json.dumps([{"uri": STAGED_CTL}])),                        # NULL extra at a canonical location: canonical
            # the composed refit V exactly as it folded on 2026-10-04 (175e34e0…): no init scheme of its own, a refit location, proper parents
            ("v_lf", json.dumps(["mix_lf", "ctl_lf"]), json.dumps({"builder": "compose-v"}), V_LOC),
            ("v_orphan_parent", json.dumps(["orphan_lf", "ctl_lf"]), None, V_LOC),                 # a derived row whose first parent IS an orphan
            ("v_orphan_parent2", json.dumps(["mix_lf", "orphan_lf"]), None, V_LOC)]                # … and one whose SECOND parent is (refit-fix-4)
    db.executemany("INSERT INTO object VALUES (?,?,?,?, 'reference')", rows)

    def ids(clause: str) -> set:
        return {r[0] for r in db.execute(f"SELECT object_id FROM object WHERE {clause} AND status IN ('reference','approved')")}

    assert ids(_scope(TAG, "self")) == {"ctl_lf", "mix_lf"}
    assert ids(_scope(None, "self")) == {"ctl_zero", "canon_null"}
    assert ids(_orphans()) == {"orphan_lf", "orphan_lf_noinit", "v_lf", "v_orphan_parent", "v_orphan_parent2"}   # the SELF rule over every row: derived rows have no init of their own …
    # … so the front door applies the rule PER SCOPE: fitted units ask `self`, derived units ask `parent` — the refit V (a composed row with
    # proper refit parents) is then NOT an orphan, while a derived row whose first parent is one, is (refit-fix-3: the live V was named)
    assert ids(f"parent_object_ids_json IS NOT NULL AND {_scope(TAG, 'parent')}") == {"v_lf"}
    assert ids(f"parent_object_ids_json IS NOT NULL AND {_orphans(which='parent')}") == {"v_orphan_parent", "v_orphan_parent2"}
    assert _orphans(scheme="zero") == "0 = 1" and _orphans(scheme="zero", which="parent") == "0 = 1"
    roi = _bash_fn(FRONT.read_text(), "registry_object_ids")
    assert '$(refit_orphans_sql "${2:-self}")' in roi and 'fall in NEITHER the canonical nor a refit scope' in roi and ">&2" in roi.split("fall in NEITHER")[1][:200]
    assert "repro_fatal" in roi.split("fall in NEITHER")[1][:2400] and "D2067" in roi                # a gate, not a warning (refit-fix-4); three more named lines since refit-fix-5/6


# ----------------------------------------------------------------------------- refit-fix-2: a failed registry read is never "no rows"

def _registry_world(tmp_path: Path, *, orphans: bool = True, composed: bool = False, canonical_v: bool = False, dose: bool = False):
    """A registry root with a sqlite `object` table, and a `python` STUB that emulates `-m sisteer.registry.cli --root R query --json SQL`
    against it (or, under STUB_FAIL=1, exits 1 with a message on stderr) and execs the real venv python for anything else."""
    root = tmp_path / "registry"; root.mkdir(parents=True)
    db = sqlite3.connect(root / "registry.sqlite")
    db.execute("CREATE TABLE object (object_id TEXT PRIMARY KEY, campaign TEXT, object_kind TEXT, object_role TEXT, fit_model_id TEXT, fit_corpus_id TEXT, derivation TEXT, parent_object_ids_json TEXT, extra_json TEXT, artifact_locations_json TEXT, status TEXT)")
    lecun, zero = json.dumps({"init_scheme": SCHEME}), json.dumps({"init_scheme": "zero"})
    # (id, parents, extra, locations, kind, role, derivation); fit_corpus_id follows the role as the live rows carry it (refit-fix-6) — derivation as the live registry spells it: `none` for a fitted row,
    # `composition` for the direct paired V, `dose_scale` for a scaled child with ONE parent (refit-fix-5)
    rows = [("c" * 64, None, lecun, json.dumps([BANK_CTL]), "relu_steer", "control", "none"),                           # the refit control row
            ("d" * 64, None, zero, json.dumps([{"uri": STAGED_CTL}]), "relu_steer", "control", "none")]                 # the canonical control row
    if orphans:
        rows += [("a" * 64, None, None, json.dumps([BANK_CTL]), "relu_steer", "control", "none"),                        # a real orphan, bare path
                 ("b" * 64, None, None, json.dumps([{"bytes": 1, "uri": BANK_CTL}]), "relu_steer", "control", "none")]  # a real orphan, {uri} object
    if composed:   # refit-fix-4: a healthy refit V and (as the world's `orphans` flag is off here) nothing else orphaned among the fitted rows
        MIX = BANK_CTL.replace("constv2ctl2-full", "constv2-mix-full")
        V = BANK_CTL.replace("qwen35_4b_demo_base/cells_all/constv2ctl2-full__psr_ce_all", "derived/psr_paired_write/qwen35_4b_demo_base/constv2-mix-full__psr_ce_all_minus_ctl")
        rows += [("e" * 64, None, lecun, json.dumps([MIX]), "relu_steer", "constitution", "none"),                                            # the refit constitution row
                 ("f" * 64, json.dumps(["e" * 64, "c" * 64]), json.dumps({"builder": "compose-v"}), json.dumps([V]), "composed", "difference", "composition")]   # a healthy V
    if canonical_v:   # refit-fix-5: the canonical pair's whole V (both parents canonical, at the adapter's staging spelling)
        rows += [("g" * 64, None, zero, json.dumps([{"uri": STAGED_CTL.replace("constv2ctl2-full", "constv2-mix-full")}]), "relu_steer", "constitution", "none"),
                 ("h" * 64, json.dumps(["g" * 64, "d" * 64]), json.dumps({"builder": "compose-v"}), json.dumps([{"uri": STAGED_V}]), "composed", "difference", "composition")]
    if dose:          # … and its dose_scale child: ONE parent (the V), by design — exempt from the paired-composition arity rule
        rows += [("k" * 64, json.dumps(["h" * 64]), json.dumps({"manifest_dose": 0.75}), json.dumps([{"uri": STAGED_V.replace("derived__", "d0p75__")}]), "composed", "difference", "dose_scale")]
    db.executemany("INSERT INTO object VALUES (?, 'sdfbase_full22', ?, ?, 'qwen35_4b_demo_base', ?, ?, ?, ?, ?, 'reference')",
                   [(oid, kind, role, CORPUS.get(role), deriv, parents, extra, loc) for oid, parents, extra, loc, kind, role, deriv in rows])
    db.commit(); db.close()
    stub = tmp_path / "python"
    stub.write_text(f"""#!/bin/bash
if [ "${{1:-}}" = -m ] && [ "${{2:-}}" = sisteer.registry.cli ]; then
  if [ -n "${{STUB_FAIL:-}}" ]; then echo "database is locked (stub)" >&2; exit 1; fi
  shift 2; root=""; sql=""
  while [ $# -gt 0 ]; do case "$1" in --root) root="$2"; shift 2;; query|--json) shift;; *) sql="$1"; shift;; esac; done
  exec {VENV_PY} - "$root/registry.sqlite" "$sql" <<'PY'
import json, sqlite3, sys
db = sqlite3.connect(sys.argv[1]); print(json.dumps([list(r) for r in db.execute(sys.argv[2])]))
PY
fi
exec {VENV_PY} "$@"
""")
    stub.chmod(0o755)
    return root, stub


def _roi_script(tmp_path: Path, *, refit: str, where: str, scope: str) -> str:
    root, stub = tmp_path / "registry", tmp_path / "python"
    src = FRONT.read_text()
    return f"""
set -uo pipefail
SISTEER_REGISTRY_ROOT={root}; REFIT="{refit}"; REFIT_SCHEME={SCHEME}; DEFAULT_BANK=full22; MIX_SI={MIX_SI}; CTL_SI={CTL_SI}
repro_python(){{ echo {stub}; }}
repro_fatal(){{ echo "REFUSED: $*" >&2; exit 7; }}
repro_note(){{ echo "note: $*"; }}
{_bash_fn(src, "refit_scope_sql")}
{_bash_fn(src, "refit_orphans_sql")}
{_bash_fn(src, "refit_mismatch_sql")}
{_bash_fn(src, "derived_integrity_sql")}
{_bash_fn(src, "derived_order_sql")}
{_bash_fn(src, "registry_object_ids")}
registry_object_ids "{where}" {scope}
echo "RC=$?"
"""


def _roi(tmp_path: Path, *, fail: bool = False, refit: str = "lecun", where: str = "object_kind='relu_steer' AND object_role='control'", scope: str = "self",
         orphans: bool = True, composed: bool = False, canonical_v: bool = False, dose: bool = False) -> subprocess.CompletedProcess:
    _registry_world(tmp_path, orphans=orphans, composed=composed, canonical_v=canonical_v, dose=dose)
    env = dict(os.environ); env["STUB_FAIL"] = "1" if fail else ""
    return subprocess.run(["bash", "-c", _roi_script(tmp_path, refit=refit, where=where, scope=scope)], capture_output=True, text=True, env=env)


def test_registry_object_ids_names_real_orphans_in_both_location_shapes_through_the_cli(tmp_path):
    """The real function over a real sqlite registry (the CLI emulated): stdout = exactly the refit row's id; stderr names BOTH orphans —
    the bare-path and the {uri}-object location — and nothing else; the canonical row is in neither."""
    r = _roi(tmp_path)
    # refit-fix-4: orphan rows in the unit's candidate set are a GATE — both (bare-path and {uri}-object locations) are named and
    # the read REFUSES; no ids, no decision, under either scope
    assert r.returncode == 7, r.stdout + r.stderr
    assert "fall in NEITHER the canonical nor a refit scope" in r.stderr and ("a" * 64) in r.stderr and ("b" * 64) in r.stderr, r.stderr
    assert "2 object row(s)" in r.stderr and ("d" * 64) not in r.stderr and "D2067" in r.stderr and "registry_unavailable" not in r.stderr
    assert r.stdout.strip() == "" and "RC=" not in r.stdout, r.stdout
    r = _roi(tmp_path / "canon", refit="")
    assert r.returncode == 7 and ("a" * 64) in r.stderr and ("b" * 64) in r.stderr and r.stdout.strip() == "", r.stdout + r.stderr
    # a world WITHOUT orphans: the refit scope returns exactly the refit row, the canonical scope exactly the zero-init row
    r = _roi(tmp_path / "clean", orphans=False)
    assert r.returncode == 0 and r.stdout.split("RC=")[0].split() == ["c" * 64] and "RC=0" in r.stdout and "NEITHER" not in r.stderr, r.stdout + r.stderr
    r = _roi(tmp_path / "clean_canon", refit="", orphans=False)
    assert r.returncode == 0 and r.stdout.split("RC=")[0].split() == ["d" * 64] and "NEITHER" not in r.stderr, r.stdout + r.stderr


def test_a_failing_registry_cli_is_registry_unavailable_and_refuses_the_done_ness_decision(tmp_path):
    """A locked / unreadable registry (the CLI exits 1 with a message) is `registry_unavailable: <stderr>` and a REFUSAL — never zero
    orphans, never zero ids; and the callers that capture the ids in a subshell honour that refusal instead of reading '' as 'not done'."""
    r = _roi(tmp_path, fail=True)
    assert r.returncode == 7, r.stdout + r.stderr
    assert "registry_unavailable: database is locked (stub)" in r.stderr and "REFUSED:" in r.stderr and "D1862" in r.stderr, r.stderr
    assert r.stdout.strip() == "" and "NEITHER" not in r.stderr, r.stdout + r.stderr   # no orphan verdict, no ids, no RC line
    # the callers: registry_done and difference_relu_done_marker_valid capture `$(registry_object_ids …)` — a subshell — and must refuse
    src = FRONT.read_text()
    rd = _bash_fn(src, "registry_done"); dm = _bash_fn(src, "difference_relu_done_marker_valid")
    assert '|| repro_fatal "$1: registry_object_ids refused (registry_unavailable above)' in rd
    assert re.search(r"^\s+local ids(\s|$)", rd, re.M), rd          # declared apart from the assignment, so `$?` is the query's
    assert '|| repro_fatal "difference_relu: registry_object_ids refused (registry_unavailable above)' in dm
    script = f"""
set -uo pipefail
CAMPAIGN=sdfbase_full22; REGISTRY_UNREADY_IDS=""
registry_object_ids(){{ echo "registry_unavailable: stub" >&2; return 7; }}
repro_fatal(){{ echo "REFUSED: $*" >&2; exit 7; }}
repro_note(){{ echo "note: $*"; }}; repro_say(){{ echo "say: $*"; }}; mark_done(){{ echo "mark $*"; }}
{rd}
registry_done unit_x "object_kind='x'" "" self; echo "RC=$?"
"""
    r2 = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert r2.returncode == 7 and "done-ness of unit_x cannot be decided" in r2.stderr and "mark" not in r2.stdout and "RC=" not in r2.stdout, r2.stdout + r2.stderr


# ----------------------------------------------------------------------------- refit-fix-3: a missing registry is never "zero rows"

def _callers_world(tmp_path: Path, *, registry: str, dry: int = 0, caller: str = "registry_done") -> subprocess.CompletedProcess:
    """registry_object_ids + BOTH its callers (registry_done, difference_relu_done_marker_valid — the real functions) over a registry
    root that is `missing` (no such directory), `unreadable` (registry.sqlite chmod 000), `empty` (a real database with zero matching
    rows) or `populated` (one canonical row); the CLI emulated by the stub of _registry_world."""
    _root, stub = _registry_world(tmp_path, orphans=False)         # the stub + a populated database at <tmp>/registry (no orphans: the gate is tested apart)
    if registry == "missing":
        root = tmp_path / "nowhere" / "registry"
    elif registry == "unreadable":
        root = tmp_path / "unreadable" / "registry"; root.mkdir(parents=True)
        (root / "registry.sqlite").write_bytes(b"not a database"); (root / "registry.sqlite").chmod(0)
    elif registry == "empty":
        root = tmp_path / "empty" / "registry"; root.mkdir(parents=True)
        db = sqlite3.connect(root / "registry.sqlite")
        db.execute("CREATE TABLE object (object_id TEXT PRIMARY KEY, campaign TEXT, object_kind TEXT, object_role TEXT, fit_model_id TEXT, fit_corpus_id TEXT, derivation TEXT, parent_object_ids_json TEXT, extra_json TEXT, artifact_locations_json TEXT, status TEXT)")
        db.commit(); db.close()
    else:
        root = _root
    src = FRONT.read_text()
    call = {"registry_done": 'registry_done unit_x "object_kind=\'relu_steer\' AND object_role=\'control\'" "" self; echo "RD_RC=$?"',
            "marker": 'difference_relu_done_marker_valid; echo "DM_RC=$?"'}[caller]
    script = f"""
set -uo pipefail
SISTEER_REGISTRY_ROOT={root}; REFIT=""; REFIT_SCHEME={SCHEME}; DEFAULT_BANK=full22; CAMPAIGN=sdfbase_full22; FIT_ID=qwen35_4b_demo_base; MIX_SI={MIX_SI}; CTL_SI={CTL_SI}
REPRO_DRY_RUN={dry}; REGISTRY_UNREADY_IDS=""; PAIRED=/nonexistent/paired; CLUSTER=c
repro_python(){{ echo {stub}; }}
repro_fatal(){{ echo "REFUSED: $*" >&2; exit 7; }}
repro_note(){{ echo "note: $*"; }}; repro_say(){{ echo "say: $*"; }}; mark_done(){{ echo "mark $*"; }}
v_liveness_ready(){{ echo "liveness-checked: $1"; return 1; }}
{_bash_fn(src, "refit_scope_sql")}
{_bash_fn(src, "refit_orphans_sql")}
{_bash_fn(src, "refit_mismatch_sql")}
{_bash_fn(src, "derived_integrity_sql")}
{_bash_fn(src, "derived_order_sql")}
{_bash_fn(src, "registry_object_ids")}
{_bash_fn(src, "registry_done")}
{_bash_fn(src, "difference_relu_done_marker_valid")}
{call}
"""
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={**os.environ, "STUB_FAIL": ""})


@pytest.mark.parametrize("caller", ["registry_done", "marker"])
def test_a_missing_or_unreadable_registry_is_registry_unavailable_for_both_callers_on_a_live_run(tmp_path, caller):
    """A root with no registry.sqlite, or one whose database is unreadable, used to return success with zero ids: registry_done read
    registered work as unregistered and a live run could submit a second fit / derived prep. Both callers now REFUSE (exit non-zero,
    `registry_unavailable: <root>/registry.sqlite missing or unreadable`), marking nothing done, deciding nothing."""
    for kind in ("missing", "unreadable"):
        if kind == "unreadable" and os.geteuid() == 0:
            continue                                                                  # root reads a chmod-000 file; the case is void there
        r = _callers_world(tmp_path / kind, registry=kind, caller=caller)
        assert r.returncode == 7, (kind, caller, r.stdout + r.stderr)
        assert "registry_unavailable:" in r.stderr and "registry.sqlite missing or unreadable" in r.stderr and "D1885" in r.stderr, (kind, r.stderr)
        assert "mark" not in r.stdout and "RD_RC=" not in r.stdout and "DM_RC=" not in r.stdout and "liveness-checked" not in r.stdout, (kind, r.stdout)


@pytest.mark.parametrize("caller", ["registry_done", "marker"])
def test_an_empty_registry_is_simply_unregistered_and_a_dry_run_notes_an_absent_one_and_proceeds(tmp_path, caller):
    """The positive paths: a database that is present but holds no matching row reads as unregistered (registry_done returns 1, the
    marker read finds no ids) with no refusal; a populated one marks done. Under --dry-run an absent registry is NOTED
    (`registry_unavailable:` on stderr) and the dry run proceeds as unregistered — it submits nothing, and the front-door dry-run
    harnesses have no registry (the repro_require_clean_tree carve-out)."""
    r = _callers_world(tmp_path / "empty", registry="empty", caller=caller)
    assert r.returncode == 0 and "registry_unavailable" not in r.stderr and "REFUSED" not in r.stderr, r.stdout + r.stderr
    assert ("RD_RC=1" in r.stdout) if caller == "registry_done" else ("DM_RC=1" in r.stdout), r.stdout
    assert "mark" not in r.stdout
    if caller == "registry_done":
        r = _callers_world(tmp_path / "pop", registry="populated", caller=caller)
        assert r.returncode == 0 and "RD_RC=0" in r.stdout and ("d" * 64) in r.stdout and "mark unit_x" in r.stdout, r.stdout + r.stderr
    r = _callers_world(tmp_path / "dry", registry="missing", dry=1, caller=caller)
    assert r.returncode == 0 and "registry_unavailable:" in r.stderr and "a dry run proceeds as if nothing were registered" in r.stderr, r.stdout + r.stderr
    assert ("RD_RC=1" in r.stdout) if caller == "registry_done" else ("DM_RC=1" in r.stdout), r.stdout
    assert "REFUSED" not in r.stderr and "mark" not in r.stdout



def test_a_v_with_an_orphaned_second_parent_is_not_done_and_refuses_while_a_healthy_v_is_done(tmp_path):
    """refit-fix-4: the derived unit's read inspects EVERY parent. Over a world with one healthy refit V (parents
    [constitution refit, control refit]) the parent scope returns exactly it; after adding a V whose SECOND parent is an orphan (a
    control row with no recorded init scheme at a refit-bank location) the same read REFUSES by name — that V is neither done nor
    rebuilt, and no id is returned, because the unit's candidate set carries unverifiable provenance."""
    where = "object_kind='composed' AND object_role='difference'"
    r = _roi(tmp_path, where=where, scope="parent", orphans=False, composed=True)
    assert r.returncode == 0 and r.stdout.split("RC=")[0].split() == ["f" * 64] and "NEITHER" not in r.stderr, r.stdout + r.stderr
    db = sqlite3.connect(tmp_path / "registry" / "registry.sqlite")
    V2 = BANK_CTL.replace("qwen35_4b_demo_base/cells_all/constv2ctl2-full__psr_ce_all", "derived/psr_paired_write/qwen35_4b_demo_base/constv2-mix-full__psr_ce_all_minus_ctl-x")
    db.execute("INSERT INTO object VALUES (?, 'sdfbase_full22', 'relu_steer', 'control', 'qwen35_4b_demo_base', ?, 'none', NULL, NULL, ?, 'reference')",
               ("0" * 64, CTL_SI, json.dumps([BANK_CTL.replace("constv2ctl2-full", "constv2ctl2-full-x")])))        # an orphan control
    db.execute("INSERT INTO object VALUES (?, 'sdfbase_full22', 'composed', 'difference', 'qwen35_4b_demo_base', NULL, 'composition', ?, ?, ?, 'reference')",
               ("1" * 64, json.dumps(["e" * 64, "0" * 64]), json.dumps({"builder": "compose-v"}), json.dumps([V2])))   # a V whose SECOND parent is that orphan
    db.commit(); db.close()
    r = subprocess.run(["bash", "-c", _roi_script(tmp_path, refit="lecun", where=where, scope="parent")], capture_output=True, text=True, env={**os.environ, "STUB_FAIL": ""})
    assert r.returncode == 7, r.stdout + r.stderr
    assert ("1" * 64) in r.stderr and ("f" * 64) not in r.stderr and "fall in NEITHER" in r.stderr and "D2067" in r.stderr and r.stdout.strip() == "", r.stdout + r.stderr


# ----------------------------------------------------------------------------- refit-fix-5: parents must RESOLVE; a wrong scheme at this bank refuses

def _pred(name: str, which: str, *, refit: str | None = TAG, scheme: str = SCHEME, env: dict | None = None) -> str:
    fn = _bash_fn(FRONT.read_text(), name)
    extra = "".join(f'{k}="{v}"; ' for k, v in (env or {}).items())
    script = f'REFIT="{refit or ""}"; REFIT_SCHEME="{scheme}"; DEFAULT_BANK=full22; {extra}\n{fn}\n{name} {which}\n'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout.strip()


def test_the_mismatch_and_integrity_predicates_name_this_banks_contradictions_and_malformed_parent_lists():
    """Pure SQL over an in-memory `object` table shaped like the live one. refit_mismatch_sql: a row at THIS run's bank whose recorded
    init scheme is not the committed one (self), or a derived row with such a parent (parent) — a row at ANOTHER tag's bank with any scheme
    is not this run's (no block), and outside a refit run the predicate is `0 = 1`. derived_integrity_sql: a direct paired composition
    (derivation `composition`) whose parents are not exactly two ids resolving to one constitution and one control relu_steer row of its
    model — empty, one, three, two controls, a duplicated id, a parent with no row yet, a parent of another model; a dose_scale child with
    its single parent is exempt, and so is a fitted row."""
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE object (object_id TEXT PRIMARY KEY, object_kind TEXT, object_role TEXT, fit_model_id TEXT, derivation TEXT, parent_object_ids_json TEXT, extra_json TEXT, artifact_locations_json TEXT, status TEXT, fit_corpus_id TEXT)")
    lecun, zero = json.dumps({"init_scheme": SCHEME}), json.dumps({"init_scheme": "zero"})
    M, OTHER_BANK = "qwen35_4b_demo_base", BANK_CTL.replace("full22_lecun", "full22_other")
    V_LOC = json.dumps([BANK_CTL.replace("qwen35_4b_demo_base/cells_all/constv2ctl2-full__psr_ce_all", "derived/psr_paired_write/qwen35_4b_demo_base/constv2-mix-full__psr_ce_all_minus_ctl")])
    rows = [("mix_lf", "relu_steer", "constitution", M, "none", None, lecun, json.dumps([BANK_CTL.replace("constv2ctl2-full", "constv2-mix-full")])),
            ("ctl_lf", "relu_steer", "control", M, "none", None, lecun, json.dumps([BANK_CTL])),
            ("ctl_lf2", "relu_steer", "control", M, "none", None, lecun, json.dumps([BANK_CTL.replace("constv2ctl2-full", "constv2ctl2-full-2")])),
            ("ctl_lf_zero", "relu_steer", "control", M, "none", None, zero, json.dumps([{"uri": BANK_CTL.replace("constv2ctl2-full", "constv2ctl2-full-z")}])),   # zero-init fitted INTO this bank: a contradiction
            ("ctl_other_zero", "relu_steer", "control", M, "none", None, zero, json.dumps([OTHER_BANK])),                                                      # another tag's bank: that tag's business
            ("ctl_lf_4b", "relu_steer", "control", "qwen35_4b_base", "none", None, lecun, json.dumps([BANK_CTL.replace("qwen35_4b_demo_base", "qwen35_4b_base")])),
            ("v_ok", "composed", "difference", M, "composition", json.dumps(["mix_lf", "ctl_lf"]), None, V_LOC),
            ("v_mismatch2", "composed", "difference", M, "composition", json.dumps(["mix_lf", "ctl_lf_zero"]), None, V_LOC),         # the SECOND parent contradicts the scheme
            ("v_other_tag", "composed", "difference", M, "composition", json.dumps(["mix_lf", "ctl_other_zero"]), None, V_LOC),
            ("v_empty", "composed", "difference", M, "composition", "[]", None, V_LOC),
            ("v_one", "composed", "difference", M, "composition", json.dumps(["mix_lf"]), None, V_LOC),
            ("v_three", "composed", "difference", M, "composition", json.dumps(["mix_lf", "ctl_lf", "ctl_lf2"]), None, V_LOC),
            ("v_two_ctl", "composed", "difference", M, "composition", json.dumps(["ctl_lf", "ctl_lf2"]), None, V_LOC),
            ("v_dup", "composed", "difference", M, "composition", json.dumps(["mix_lf", "mix_lf"]), None, V_LOC),
            ("v_unfolded", "composed", "difference", M, "composition", json.dumps(["mix_lf", "not_folded_yet"]), None, V_LOC),      # the control record has not folded
            ("v_foreign", "composed", "difference", M, "composition", json.dumps(["mix_lf", "ctl_lf_4b"]), None, V_LOC),           # a parent of another model
            ("v_null", "composed", "difference", M, "composition", None, None, V_LOC),
            ("dose", "composed", "difference", M, "dose_scale", json.dumps(["v_ok"]), json.dumps({"manifest_dose": 0.75}), V_LOC),   # ONE parent by design
            # refit-fix-6: the ORDER is the identity — [constitution, control] = +mix -ctl; reversed = the opposite steering direction
            ("v_rev", "composed", "difference", M, "composition", json.dumps(["ctl_lf", "mix_lf"]), None, V_LOC),
            ("mix_other_si", "relu_steer", "constitution", M, "none", None, lecun, json.dumps([BANK_CTL.replace("constv2ctl2-full", "othersi-full")])),
            ("v_wrong_corpus", "composed", "difference", M, "composition", json.dumps(["mix_other_si", "ctl_lf"]), None, V_LOC)]   # right roles and order, another constitution corpus
    db.executemany("INSERT INTO object VALUES (?,?,?,?,?,?,?,?, 'reference', NULL)", rows)
    db.execute("UPDATE object SET fit_corpus_id = CASE object_role WHEN 'constitution' THEN ? WHEN 'control' THEN ? END WHERE object_kind = 'relu_steer'", (MIX_SI, CTL_SI))
    db.execute("UPDATE object SET fit_corpus_id = 'othersi-full' WHERE object_id = 'mix_other_si'")

    def ids(clause: str) -> set:
        return {r[0] for r in db.execute(f"SELECT object_id FROM object WHERE {clause} AND status IN ('reference','approved')")}

    assert ids(_pred("refit_mismatch_sql", "self")) == {"ctl_lf_zero"}
    assert ids(_pred("refit_mismatch_sql", "parent")) == {"v_mismatch2"}
    assert _pred("refit_mismatch_sql", "self", refit=None) == "0 = 1" and _pred("refit_mismatch_sql", "parent", scheme="zero") == "0 = 1"
    assert ids(_pred("derived_integrity_sql", "parent")) == {"v_empty", "v_one", "v_three", "v_two_ctl", "v_dup", "v_unfolded", "v_foreign", "v_null"}
    assert _pred("derived_integrity_sql", "self") == "0 = 1" and _pred("derived_integrity_sql", "parent", refit=None) == _pred("derived_integrity_sql", "parent")   # scheme-independent: every run
    # the canonical scope now excludes an unfolded parent and an empty list by itself; the refit scope's predicate (every parent a proper
    # refit row of this bank) still admits a short, long, duplicated or foreign-model list OF refit rows, and a NULL list keeps its canonical
    # reading — arity is the composition's rule, so the GATE (derived_integrity_sql, run before the ids query in registry_object_ids) is
    # what keeps every malformed row from DONE in either scope: scope minus gate leaves exactly the whole V
    malformed = ids(_pred("derived_integrity_sql", "parent"))
    # … and the two ORDER violations (refit-fix-6) pass integrity by design — arity, roles, model — so only the order predicate below
    # separates them from the whole V
    assert ids(f"derivation = 'composition' AND {_scope(TAG, 'parent')}") - malformed == {"v_ok", "v_rev", "v_wrong_corpus"}
    assert ids(f"derivation = 'composition' AND {_scope(None, 'parent')}") - malformed == set()
    assert not ({"v_unfolded", "v_empty"} & (ids(_scope(TAG, "parent")) | ids(_scope(None, "parent"))))   # excluded by the predicates themselves (refit-fix-5)
    # refit-fix-6: v_rev passes arity, roles, model, scheme and the refit scope — only the ORDER predicate tells it from v_ok; with
    # the corpora named (04 always names them) a constitution parent of another corpus at index 0 is mis-positioned too; without them the
    # roles alone decide. Malformed rows are also "unordered" (the gate names them malformed first); scope minus both gates == the whole V.
    reversed_ = ids(_pred("derived_order_sql", "parent", env={"MIX_SI": MIX_SI, "CTL_SI": CTL_SI}))
    assert reversed_ - malformed == {"v_rev", "v_wrong_corpus"} and "v_ok" not in reversed_ and "dose" not in reversed_, reversed_
    assert ids(_pred("derived_order_sql", "parent")) - malformed == {"v_rev"}                       # no corpora named: roles and positions alone
    assert _pred("derived_order_sql", "self") == "0 = 1"
    assert "v_rev" in ids(_scope(TAG, "parent")) and "v_rev" not in malformed and "v_rev" not in ids(_pred("refit_mismatch_sql", "parent"))
    assert ids(f"derivation = 'composition' AND {_scope(TAG, 'parent')}") - malformed - reversed_ == {"v_ok"}
    assert "v_other_tag" not in ids(_scope(TAG, "parent")) | ids(_scope(None, "parent")) | ids(_pred("refit_mismatch_sql", "parent")) | ids(_pred("derived_integrity_sql", "parent"))


def test_a_canonical_v_whose_second_parent_has_not_folded_refuses_by_name_while_the_whole_v_and_its_dose_child_are_done(tmp_path):
    """Through the real registry_object_ids over a real sqlite registry (the CLI emulated), on a CANONICAL run: a world
    with the canonical pair's whole V (parents [constitution zero, control zero] at the staging spelling) and its dose_scale child returns
    exactly those two ids. Adding a V whose FIRST parent is canonical and whose SECOND parent has no object row yet (the control record has
    not folded) makes the same read REFUSE by name — that V is in neither scope AND is named as malformed, so it is neither marked done on
    provenance the registry cannot show nor rebuilt over; the whole V and the dose child are not named."""
    where = "object_kind='composed' AND object_role='difference'"
    r = _roi(tmp_path, refit="", where=where, scope="parent", orphans=False, canonical_v=True, dose=True)
    assert r.returncode == 0 and set(r.stdout.split("RC=")[0].split()) == {"h" * 64, "k" * 64} and "registry:" not in r.stderr, r.stdout + r.stderr
    db = sqlite3.connect(tmp_path / "registry" / "registry.sqlite")
    db.execute("INSERT INTO object VALUES (?, 'sdfbase_full22', 'composed', 'difference', 'qwen35_4b_demo_base', NULL, 'composition', ?, ?, ?, 'reference')",
               ("2" * 64, json.dumps(["g" * 64, "9" * 64]), json.dumps({"builder": "compose-v"}), json.dumps([{"uri": STAGED_V.replace("derived__", "derived-x__")}])))
    db.commit(); db.close()
    r = subprocess.run(["bash", "-c", _roi_script(tmp_path, refit="", where=where, scope="parent")], capture_output=True, text=True, env={**os.environ, "STUB_FAIL": ""})
    assert r.returncode == 7 and r.stdout.strip() == "", r.stdout + r.stderr
    assert ("2" * 64) in r.stderr and "do not carry exactly two parent ids" in r.stderr and "1 direct paired-composition row(s)" in r.stderr and "D2116" in r.stderr, r.stderr
    assert ("h" * 64) not in r.stderr and ("k" * 64) not in r.stderr and "NEITHER" not in r.stderr and "NOT the committed config" not in r.stderr, r.stderr


def test_a_second_parent_with_a_contradicting_init_scheme_at_this_bank_refuses_by_name_and_another_tags_scheme_blocks_nothing(tmp_path):
    """Through the real registry_object_ids (the CLI emulated) under --refit lecun: a control row fitted zero-init INTO
    full22_lecun and a V whose SECOND parent it is. The composed unit's parent read REFUSES naming that V (not the healthy refit V); the
    fitted unit's self read REFUSES naming the row itself; no id comes back from either. The same shapes at ANOTHER tag's bank
    (full22_other) block nothing: not this run's rows, in neither scope, silently — and a canonical run over the contradicting world is
    not refused either (the rule is scoped to the refit run's own bank)."""
    where_v, where_c = "object_kind='composed' AND object_role='difference'", "object_kind='relu_steer' AND object_role='control'"
    r = _roi(tmp_path, where=where_v, scope="parent", orphans=False, composed=True)
    assert r.returncode == 0 and r.stdout.split("RC=")[0].split() == ["f" * 64] and "registry:" not in r.stderr, r.stdout + r.stderr
    V_M = BANK_CTL.replace("qwen35_4b_demo_base/cells_all/constv2ctl2-full__psr_ce_all", "derived/psr_paired_write/qwen35_4b_demo_base/constv2-mix-full__psr_ce_all_minus_ctl-m")
    db = sqlite3.connect(tmp_path / "registry" / "registry.sqlite")
    db.execute("INSERT INTO object VALUES (?, 'sdfbase_full22', 'relu_steer', 'control', 'qwen35_4b_demo_base', ?, 'none', NULL, ?, ?, 'reference')",
               ("3" * 64, CTL_SI, json.dumps({"init_scheme": "zero"}), json.dumps([BANK_CTL.replace("constv2ctl2-full", "constv2ctl2-full-m")])))   # zero-init at THIS bank
    db.execute("INSERT INTO object VALUES (?, 'sdfbase_full22', 'composed', 'difference', 'qwen35_4b_demo_base', NULL, 'composition', ?, ?, ?, 'reference')",
               ("4" * 64, json.dumps(["e" * 64, "3" * 64]), json.dumps({"builder": "compose-v"}), json.dumps([V_M])))                      # the V whose SECOND parent it is
    db.commit(); db.close()
    env = {**os.environ, "STUB_FAIL": ""}
    r = subprocess.run(["bash", "-c", _roi_script(tmp_path, refit="lecun", where=where_v, scope="parent")], capture_output=True, text=True, env=env)
    assert r.returncode == 7 and r.stdout.strip() == "", r.stdout + r.stderr
    assert ("4" * 64) in r.stderr and ("f" * 64) not in r.stderr and "NOT the committed config's 'random_lecun_1e-3'" in r.stderr and "bank full22_lecun" in r.stderr and "D2116" in r.stderr, r.stderr
    assert "NEITHER" not in r.stderr and "do not carry exactly two" not in r.stderr, r.stderr
    r = subprocess.run(["bash", "-c", _roi_script(tmp_path, refit="lecun", where=where_c, scope="self")], capture_output=True, text=True, env=env)
    assert r.returncode == 7 and ("3" * 64) in r.stderr and ("c" * 64) not in r.stderr and "1 object row(s)" in r.stderr and "NOT the committed config" in r.stderr and r.stdout.strip() == "", r.stdout + r.stderr
    # a canonical run over the same world is not refused (the mismatch rule is the REFIT run's, scoped to its bank); the canonical self
    # scope is "not a refit row of this scheme", so the zero-init row at the refit bank reads canonical there — pre-existing semantics,
    # not this fix's claim, so only the absence of a refusal and the canonical row's presence are pinned
    r = subprocess.run(["bash", "-c", _roi_script(tmp_path, refit="", where=where_c, scope="self")], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and ("d" * 64) in r.stdout.split("RC=")[0].split() and "registry:" not in r.stderr, r.stdout + r.stderr
    # another tag's bank: the same shapes block nothing under --refit lecun
    r = _roi(tmp_path / "other", where=where_v, scope="parent", orphans=False, composed=True)
    db = sqlite3.connect(tmp_path / "other" / "registry" / "registry.sqlite")
    db.execute("INSERT INTO object VALUES (?, 'sdfbase_full22', 'relu_steer', 'control', 'qwen35_4b_demo_base', ?, 'none', NULL, ?, ?, 'reference')",
               ("5" * 64, CTL_SI, json.dumps({"init_scheme": "zero"}), json.dumps([BANK_CTL.replace("full22_lecun", "full22_other")])))
    db.execute("INSERT INTO object VALUES (?, 'sdfbase_full22', 'composed', 'difference', 'qwen35_4b_demo_base', NULL, 'composition', ?, ?, ?, 'reference')",
               ("6" * 64, json.dumps(["e" * 64, "5" * 64]), json.dumps({"builder": "compose-v"}), json.dumps([V_M.replace("full22_lecun", "full22_other")])))
    db.commit(); db.close()
    for where, scope, want in ((where_v, "parent", ["f" * 64]), (where_c, "self", ["c" * 64])):
        r = subprocess.run(["bash", "-c", _roi_script(tmp_path / "other", refit="lecun", where=where, scope=scope)], capture_output=True, text=True, env=env)
        assert r.returncode == 0 and r.stdout.split("RC=")[0].split() == want and "registry:" not in r.stderr, (where, r.stdout + r.stderr)


# ----------------------------------------------------------------------------- refit-fix-6: the parents' ORDER is the identity

def test_a_v_composed_with_its_parents_reversed_refuses_by_name_while_the_rightly_ordered_v_is_done(tmp_path):
    """Through the real registry_object_ids over a real sqlite registry (the CLI emulated) under --refit lecun: the healthy
    refit V (parents [constitution e, control c] — the producer's [parent_pos, parent_neg]) is returned. Adding a V built the other way
    round ([c, e]: both parents the right model, bank, scheme and corpus) makes the same read REFUSE by name with the parents_reversed line —
    that V passes arity, roles, model and scheme and only its ORDER tells it from the right one; it is neither marked done nor rebuilt; the
    rightly ordered V is not named and no id comes back. A canonical run over the same world refuses the same row (the rule is every run's)."""
    where = "object_kind='composed' AND object_role='difference'"
    r = _roi(tmp_path, where=where, scope="parent", orphans=False, composed=True)
    assert r.returncode == 0 and r.stdout.split("RC=")[0].split() == ["f" * 64] and "registry:" not in r.stderr, r.stdout + r.stderr
    V_R = BANK_CTL.replace("qwen35_4b_demo_base/cells_all/constv2ctl2-full__psr_ce_all", "derived/psr_paired_write/qwen35_4b_demo_base/constv2-mix-full__psr_ce_all_minus_ctl-rev")
    db = sqlite3.connect(tmp_path / "registry" / "registry.sqlite")
    db.execute("INSERT INTO object VALUES (?, 'sdfbase_full22', 'composed', 'difference', 'qwen35_4b_demo_base', NULL, 'composition', ?, ?, ?, 'reference')",
               ("7" * 64, json.dumps(["c" * 64, "e" * 64]), json.dumps({"builder": "compose-v"}), json.dumps([V_R])))   # [control, constitution]: reversed
    db.commit(); db.close()
    env = {**os.environ, "STUB_FAIL": ""}
    for refit in ("lecun", ""):
        r = subprocess.run(["bash", "-c", _roi_script(tmp_path, refit=refit, where=where, scope="parent")], capture_output=True, text=True, env=env)
        assert r.returncode == 7 and r.stdout.strip() == "", (refit, r.stdout + r.stderr)
        assert ("7" * 64) in r.stderr and ("f" * 64) not in r.stderr and "WRONG ORDER" in r.stderr and "1 direct paired-composition row(s)" in r.stderr and "D2126" in r.stderr, (refit, r.stderr)
        assert f"(corpus {MIX_SI})" in r.stderr and f"(corpus {CTL_SI})" in r.stderr, r.stderr                 # the named corpora ride the line
        assert "NEITHER" not in r.stderr and "do not carry exactly two" not in r.stderr and "NOT the committed config" not in r.stderr, (refit, r.stderr)


def _marker_run(tmp_path: Path, *, marker: str = "DONE.control_relu", registry_root: Path | None = None,
                python: Path | None = None) -> subprocess.CompletedProcess:
    """The REAL front door, --dry-run, on the front-door model over a site whose state dir carries a pre-existing CANONICAL DONE marker
    (no --refit: the committed config's zero init scheme refuses a refit before the marker logic, pinned above); `python` is
    _registry_world's stub when given (the registry CLI emulated against the sqlite world, the venv python for everything else)."""
    site = tmp_path / "site"
    if not site.exists():
        shutil.copytree(REPRO / "site.example", site)
        with open(site / "config.env", "a") as fh:
            fh.write(f"\nSISTEER_STATE_DIR={tmp_path / 'state'}\nSISTEER_PYTHON={python or tmp_path / 'python'}\nSISTEER_REGISTRY_ROOT={registry_root or tmp_path / 'registry'}\n")
    st = tmp_path / "state" / "qwen35_4b_demo" / "04_fit_objects"; st.mkdir(parents=True, exist_ok=True)
    (st / marker).write_text("planted\n")
    return run_stage("04_fit_objects.sh", "--dry-run", "--model", "qwen35_4b_demo", "--object", "control_relu", site=site, env={}, timeout=600)


def test_a_canonical_done_marker_is_honoured_without_a_registry_read(tmp_path):
    """The registry revalidation belongs to the REFIT marker (DONE.<unit>.<tag>); a CANONICAL marker is honoured
    as before — with the registry present and with no registry at all under --dry-run (no registry read is added to the canonical path,
    so a missing registry neither notes `registry_unavailable` nor un-honours the marker). The refit sub-cases ran on the proving run's
    LeCun lane; no committed fits config carries a non-zero psr init scheme now, so they are unreachable on this tree."""
    _registry_world(tmp_path, orphans=False)
    r = _marker_run(tmp_path); out = r.stdout + r.stderr
    assert r.returncode == 0 and "DONE marker present" in out and "skipping" in out, out[-3000:]
    assert "sdfbase_fit_cell.sbatch" not in out and "NOT honoured" not in out, out[-3000:]
    r = _marker_run(tmp_path / "nowhere", registry_root=tmp_path / "nowhere" / "registry", python=tmp_path / "python"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "DONE marker present" in out and "skipping" in out, out[-3000:]
    assert "sdfbase_fit_cell.sbatch" not in out and "NOT honoured" not in out and "registry_unavailable" not in out, out[-3000:]


def test_the_refit_gate_refuses_a_non_refit_unit_by_name(tmp_path):
    """--refit admits the ReLU parents only: a LoRA unit under --refit is refused by name, naming the refit units (04_fit_objects.sh's
    identity gate, run as the extracted fragment against a tiny repo whose committed config carries a non-zero scheme)."""
    repo = _tiny_repo(tmp_path, SCHEME)
    r = _run_gate(repo, units="control_lora"); out = r.stdout + r.stderr
    assert r.returncode == 7 and "control_lora is not a refit unit" in out and "refit units:" in out, out[-2000:]
    r = _run_gate(repo, units="control_relu"); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"SCHEME={SCHEME} BANK=full22_lecun" in out, out[-2000:]

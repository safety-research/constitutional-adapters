"""The derived-build chain requires the parents of the REQUESTED objects only:
positional 9 of scripts/sbatch/sdfbase_derived_build.sbatch — `relu` (the two psr_ce_all cells), `lora` (the two lora_r4_ce cells),
`both` (all four; the default, so every existing invocation is byte-identical) — selects the preflight's parent set, whether prep runs
the LoRA task-vector subtraction, and which stages are meaningful (calibrate/assess/finish are ReLU-residual work). The front door's
`difference_relu` unit asks for exactly its two psr parents and submits prep ONLY — the direct paired difference (the psr_residual
distillation stages are legacy and never composed by the front door); `difference_lora` stays its own unit on the two LoRA parents.
The shell forms are EXECUTED (the functions extracted and run), never string-matched alone."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SBATCH = REPO / "scripts" / "sbatch" / "sdfbase_derived_build.sbatch"
FRONT = REPO / "reproduce" / "04_fit_objects.sh"


def _functions_block() -> str:
    """The sbatch's objects contract: OBJECTS parsing + derived_parents / builds_relu / builds_lora, extracted verbatim."""
    src = SBATCH.read_text()
    start = src.index('OBJECTS="${9:-both}"')
    end = src.index("builds_lora() {", start)
    end = src.index("\n", end) + 1
    return src[start:end]


def _run_contract(objects: str, mix="constv2-mix-full", ctl="constv2ctl2-full"):
    script = "set -u\nset -- prep K " + mix + " " + ctl + " B - 1 5 " + objects + "\n" + _functions_block() + \
        '\nderived_parents "$OBJECTS" "$3" "$4"; echo "relu=$(builds_relu "$OBJECTS" && echo 1 || echo 0) lora=$(builds_lora "$OBJECTS" && echo 1 || echo 0)"\n'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


@pytest.mark.parametrize("objects,parents,flags", [
    ("relu", ["constv2-mix-full__psr_ce_all", "constv2ctl2-full__psr_ce_all"], "relu=1 lora=0"),
    ("lora", ["constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce"], "relu=0 lora=1"),
    ("both", ["constv2-mix-full__psr_ce_all", "constv2ctl2-full__psr_ce_all", "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce"], "relu=1 lora=1"),
])
def test_the_parent_set_follows_the_requested_objects(objects, parents, flags):
    r = _run_contract(objects)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    assert lines[:-1] == parents and lines[-1] == flags


def test_the_default_is_both_and_a_stray_selector_refuses():
    src = SBATCH.read_text()
    assert 'OBJECTS="${9:-both}"' in src, "positional 9 defaults to both: every existing chain (no 9th arg) is byte-identical"
    script = "set -- prep K m c B - 1 5 nonsense\n" + _functions_block()
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert r.returncode == 2 and "objects must be relu | lora | both" in r.stdout + r.stderr


def test_preflight_prep_and_stages_are_keyed_on_the_selector():
    src = SBATCH.read_text()
    # the preflight iterates the SELECTED parents, never a literal four-cell list
    assert 'done < <(derived_parents "$OBJECTS" "$MIX_SI" "$CTL_SI")' in src and 'for c in "${PARENTS[@]}"; do' in src
    assert '"${MIX_SI}__psr_ce_all" "${CTL_SI}__psr_ce_all" "${MIX_SI}__lora_r4_ce" "${CTL_SI}__lora_r4_ce"; do' not in src
    # prep: the LoRA subtraction only under a LoRA-building selector; a LoRA-only prep ends after it; the ReLU path is unconditional otherwise
    i = src.index("  prep)\n    # build provenance"); prep = src[i:src.index("\n  calibrate)\n", i)]     # the STAGE case's prep arm
    assert 'if builds_lora "$OBJECTS"; then' in prep and "lora_subtract.py" in prep.split('if builds_lora "$OBJECTS"; then', 1)[1].split("else", 1)[0]
    assert 'if ! builds_relu "$OBJECTS"; then' in prep and prep.index('if ! builds_relu "$OBJECTS"') < prep.index("compose-v")
    assert 'write_build_provenance "$VDIR" "${MIX_SI}__psr_ce_all" "${CTL_SI}__psr_ce_all"' in prep
    assert 'write_build_provenance "$RESID" "${MIX_SI}__psr_ce_all" "${CTL_SI}__psr_ce_all"' in prep
    assert 'write_build_provenance "$LORA" "${MIX_SI}__lora_r4_ce" "${CTL_SI}__lora_r4_ce"' in prep
    # calibrate / assess / finish refuse a selector that builds no ReLU object
    assert 'if [ "$STAGE" != prep ] && ! builds_relu "$OBJECTS"; then' in src
    # the finish liveness gate covers V + the distilled residual — both psr-gated, both built by a ReLU-only chain (unchanged)
    assert '--dirs "$VDIR,$RESID"' in src
    # the self-resubmit forwards the selector; the RESULT line and the run log name it
    assert '"$((ATTEMPT + 1))" "$MAX_ATTEMPTS" "$OBJECTS"' in src and "stage=$STAGE objects=$OBJECTS" in src and 'log "objects=$OBJECTS parents=${PARENTS[*]}' in src


def test_the_stage_refusal_for_a_lora_only_selector_executes():
    src = SBATCH.read_text()
    guard = src[src.index('if [ "$STAGE" != prep ] && ! builds_relu "$OBJECTS"; then'):]
    guard = guard[:guard.index("fi\n") + 3]
    for stage, rc in (("calibrate", 2), ("assess", 2), ("finish", 2), ("prep", 0)):
        script = f"set -- {stage} K m c B - 1 5 lora\nSTAGE=$1\n" + _functions_block() + guard + "\necho ok"
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        assert r.returncode == rc, (stage, r.stdout, r.stderr)
        assert (rc == 0) == ("ok" in r.stdout)


def test_the_front_door_relu_unit_is_preps_direct_paired_difference_and_never_composes_the_legacy_stages():
    """difference_relu is the undistilled paired write V (kind composed) that prep composes, verifies, probes for liveness and emits;
    the psr_residual calibrate/assess/finish chain is legacy — the unit submits prep only, refuses to advance while a legacy stage job
    is live, and reports a residual tree as evidence."""
    src = FRONT.read_text()
    fn = src[src.index("difference_relu(){"):src.index("refusal_direction(){")]
    assert 'for c in "${MIX_SI}__psr_ce_all" "${CTL_SI}__psr_ce_all"; do' in fn and "lora_r4_ce" not in fn.split("cluster_has", 1)[1].split("done", 1)[0]
    assert 'DERIVED_RELU_TAIL="- 1 5 relu direct"' in src and 'PAIRED="$DERIVED/psr_paired_write/$FIT_ID/${MIX_SI}__psr_ce_all_minus_ctl"' in src
    submits = [l for l in fn.splitlines() if 'submit "$unit/' in l or 'submit "$sub_unit"' in l]
    assert submits and all(('"$unit/prep"' in l or '"$sub_unit"' in l) and "prep $DERIVED_ARGS $DERIVED_RELU_TAIL" in l for l in submits), submits
    # the funnel unit of the pass over a kept, REGISTERED V is `difference_relu/liveness`: the prep unit is bound to the attempt
    # that registered V and the funnel never re-mints a registered unit
    assert 'local sub_unit="$unit/prep"; [ -n "$REGISTRY_UNREADY_IDS" ] && sub_unit="$unit/liveness"' in fn
    for legacy in ("calibrate $DERIVED_ARGS", "assess $DERIVED_ARGS", "finish $DERIVED_ARGS", "calibrate-ext", "recommended_extension"):
        assert legacy not in fn, legacy
    assert 'for s in calib assess finish; do' in fn and "LEGACY psr_residual stage job" in fn and "stop-run.sh" in fn
    assert 'cluster_has "$RESID/sweep.json" || cluster_has "$RESID/object.pt"' in fn and "left as evidence" in fn
    # the unit's registry check names the object the paper calls difference_relu: the undistilled paired write, kind `composed`
    done_line = next(l for l in fn.splitlines() if "registry_done" in l and "object_kind=" in l)
    assert "object_kind='composed' AND object_role='difference'" in done_line and "relu_steer_distilled_difference" not in done_line
    # built-but-unfolded: the record decides, through the shared three-way read; the replay is a prep re-run
    assert 'published_unfolded "$unit" "$PAIRED" "$name-prep" && return 0' in fn
    # difference_lora is its own unit on the two LoRA parents (unchanged)
    lora = src[src.index("difference_lora(){"):src.index("DERIVED_SCRIPT=")]
    assert '"${MIX_SI}__lora_r4_ce" "${CTL_SI}__lora_r4_ce"' in lora and "lora_subtract.py" in lora


def test_the_front_door_relu_unit_executes_prep_only_from_the_clusters_state(tmp_path):
    """The function EXECUTED with the cluster stubbed by a directory: no V -> prep; V built with `emitted` evidence -> waits for the
    fold (no submission); a live legacy stage job -> refuses by name; a residual tree -> noted, never advanced."""
    src = FRONT.read_text()
    fn = src[src.index("difference_relu(){"):src.index("refusal_direction(){")]
    cl = tmp_path / "cluster"
    cells = cl / "cells_all"
    for c in ("constv2-mix-full__psr_ce_all", "constv2ctl2-full__psr_ce_all"):
        (cells / c).mkdir(parents=True)
        (cells / c / "COMPLETE").write_text("{}")
    paired = cl / "derived" / "psr_paired_write" / "K" / "constv2-mix-full__psr_ce_all_minus_ctl"
    resid = cl / "derived" / "psr_residual" / "K" / "constv2-mix-full__psr_ce_all_minus_ctl_distilled"
    stubs = f"""
REPRO_DRY_RUN=0; OBJECT=difference_relu; CLUSTER=c; REPRO_MODEL=m; FIT_ID=K; MIX_SI=constv2-mix-full; CTL_SI=constv2ctl2-full; BANK=B
CELLS={cells}; RESID={resid}; PAIRED={paired}; DERIVED_SCRIPT=s.sbatch; DERIVED_ARGS="K constv2-mix-full constv2ctl2-full B"; DERIVED_RELU_TAIL="- 1 5 relu direct"
registry_done(){{ REGISTRY_UNREADY_IDS=""; return 1; }}
cluster_has(){{ [ -s "$1" ]; }}
cluster_job_active(){{ grep -qxF -- "$1" "{tmp_path}/queue" 2>/dev/null; }}
published_unfolded(){{ echo "published_unfolded $2"; return 0; }}
submit(){{ echo "submit $1 $4"; }}
repro_note(){{ echo "note: $*"; }}
repro_fatal(){{ echo "FATAL: $*"; exit 7; }}
"""
    def run():
        return subprocess.run(["bash", "-c", stubs + fn + "\ndifference_relu"], capture_output=True, text=True)
    (tmp_path / "queue").write_text("")
    r = run()
    assert r.returncode == 0 and r.stdout.strip().splitlines() == ["submit difference_relu/prep prep K constv2-mix-full constv2ctl2-full B - 1 5 relu direct"], r.stdout + r.stderr
    (tmp_path / "queue").write_text("derived-m-prep\n")
    r = run()
    assert r.returncode == 0 and "nothing to advance" in r.stdout and "submit" not in r.stdout
    (tmp_path / "queue").write_text("derived-m-calib\n")
    r = run()
    assert r.returncode == 7 and "LEGACY psr_residual stage job derived-m-calib" in r.stdout and "submit" not in r.stdout
    (tmp_path / "queue").write_text("")
    resid.mkdir(parents=True); (resid / "sweep.json").write_text("{}")
    paired.mkdir(parents=True); (paired / "object.pt").write_bytes(b"V"); (paired / "derived_meta.json").write_text("{}")
    r = run()
    assert r.returncode == 0 and "LEGACY psr_residual tree" in r.stdout and f"published_unfolded {paired}" in r.stdout and "submit" not in r.stdout, r.stdout


def test_scripts_parse():
    for f in (SBATCH, FRONT):
        r = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def _provenance_function() -> str:
    src = SBATCH.read_text()
    i = src.index("    write_build_provenance() {"); j = src.index("PYP\n    }\n", i) + len("PYP\n    }\n")
    return src[i:j]


def test_build_provenance_names_each_outputs_own_parents_under_both(tmp_path):
    """Under the default `both` the chain's PARENTS list holds all four cells and the provenance writer used that SAME list for LORA,
    V and the residual — V/residual falsely listed the LoRA cells, LORA the psr cells. The writer now takes THAT output's parents;
    executed here for `both` with a fake cells_all (COMPLETE markers) and the system python standing in for the venv."""
    import json
    cells = tmp_path / "cells_all"
    for c, sha in (("constv2-mix-full__psr_ce_all", "a" * 64), ("constv2ctl2-full__psr_ce_all", "b" * 64),
                   ("constv2-mix-full__lora_r4_ce", "c" * 64), ("constv2ctl2-full__lora_r4_ce", "d" * 64)):
        (cells / c).mkdir(parents=True); (cells / c / "COMPLETE").write_text(json.dumps({"object_sha256": sha, "fit_fingerprint": "fp-" + sha[:8]}))
    script = f"""set -u
PY=python3; OBJECTS=both; STAGE=prep; HEAD_SHA=deadbeef; ROOT={tmp_path}; POOL_KEY=.; MIX_SI=constv2-mix-full; CTL_SI=constv2ctl2-full
LORA={tmp_path}/out/lora; VDIR={tmp_path}/out/v; RESID={tmp_path}/out/resid
{_provenance_function()}
write_build_provenance "$LORA" "${{MIX_SI}}__lora_r4_ce" "${{CTL_SI}}__lora_r4_ce"
write_build_provenance "$VDIR" "${{MIX_SI}}__psr_ce_all" "${{CTL_SI}}__psr_ce_all"
write_build_provenance "$RESID" "${{MIX_SI}}__psr_ce_all" "${{CTL_SI}}__psr_ce_all"
"""
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lora = json.loads((tmp_path / "out/lora/build_provenance.json").read_text())
    v = json.loads((tmp_path / "out/v/build_provenance.json").read_text())
    resid = json.loads((tmp_path / "out/resid/build_provenance.json").read_text())
    assert [p["cell"] for p in lora["parents"]] == ["constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce"]
    assert [p["object_sha256"][:1] for p in lora["parents"]] == ["c", "d"]
    for d in (v, resid):
        assert [p["cell"] for p in d["parents"]] == ["constv2-mix-full__psr_ce_all", "constv2ctl2-full__psr_ce_all"]
        assert [p["object_sha256"][:1] for p in d["parents"]] == ["a", "b"] and d["parents"][0]["fit_fingerprint"] == "fp-aaaaaaaa"
    assert lora["objects"] == v["objects"] == "both" and lora["relu_only"] is False and v["lora_only"] is False
    # the relu-only build: the same writer, the psr pair, relu_only true
    script2 = script.replace("OBJECTS=both", "OBJECTS=relu").replace('write_build_provenance "$LORA" "${MIX_SI}__lora_r4_ce" "${CTL_SI}__lora_r4_ce"\n', "")
    r = subprocess.run(["bash", "-c", script2], capture_output=True, text=True); assert r.returncode == 0, r.stderr
    v2 = json.loads((tmp_path / "out/v/build_provenance.json").read_text())
    assert v2["relu_only"] is True and [p["cell"] for p in v2["parents"]] == ["constv2-mix-full__psr_ce_all", "constv2ctl2-full__psr_ce_all"]



# --------------------------------------------------------------------------------------------------------------------------- #
# Done-ness requires an admission-ready, object-sha-bound liveness result; the composed row alone (filed by compose-v long      #
# before the probe) and a pre-existing DONE marker are NOT done-ness; the direct build defers the record to emit-v and          #
# skips the legacy residual scaffolding.                                                                                       #
# --------------------------------------------------------------------------------------------------------------------------- #
SBATCH = REPO / "scripts" / "sbatch" / "sdfbase_derived_build.sbatch"
ID1 = "a" * 64


def _fn(src: str, name: str) -> str:
    start = src.index(f"{name}(){{")
    depth, i = 0, start
    while True:
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
        i += 1


def _relu_world(tmp_path):
    cl = tmp_path / "cluster"
    cells = cl / "cells_all"
    for c in ("constv2-mix-full__psr_ce_all", "constv2ctl2-full__psr_ce_all"):
        (cells / c).mkdir(parents=True)
        (cells / c / "COMPLETE").write_text("{}")
    paired = cl / "derived" / "psr_paired_write" / "K" / "constv2-mix-full__psr_ce_all_minus_ctl"
    paired.mkdir(parents=True)
    (paired / "object.pt").write_bytes(b"V bytes"); (paired / "derived_meta.json").write_text("{}")
    vid = hashlib.sha256(b"V bytes").hexdigest()                                  # the registry id IS the sha of the bytes on disk
    (paired / "registry_emission.json").write_text(json.dumps({"status": "emitted", "campaign": "sdfbase_full22", "contract": "2", "object_id": vid}))
    return cl, cells, paired, vid


def _receipt(paired: Path, *, obj_sha=None, gl_sha=None, not_fully_dead=True, dead_floor=0.0):
    """A receipt from REAL file hashes (the prep's liveness-receipt shape), with each field overridable for a negative."""
    doc = {"object_sha256": obj_sha or hashlib.sha256((paired / "object.pt").read_bytes()).hexdigest(),
           "gate_liveness_sha256": gl_sha or hashlib.sha256((paired / "gate_liveness.json").read_bytes()).hexdigest(),
           "not_fully_dead": not_fully_dead, "dead_floor": dead_floor, "dead_layers": [] if not_fully_dead else ["0"], "writer": "tests"}
    (paired / "gate_liveness_receipt.json").write_text(json.dumps(doc))


def _run_relu(tmp_path, cells, paired, *, dry=0, ids, real_cluster_reads=False):
    """difference_relu + registry_done + v_liveness_ready EXECUTED (the real functions) with the registry and the cluster stubbed: the
    cluster's liveness state is the front door's OWN python (LIVENESS_STATE_PY) run over the stub directory, so the hashes are real."""
    src = FRONT.read_text()
    fn = src[src.index("difference_relu(){"):src.index("refusal_direction(){")]
    i = src.index("read -r -d '' LIVENESS_STATE_PY"); state_py = src[i:src.index("\nPYLS\n", i) + 6]
    if real_cluster_reads:
        cluster = _fn(src, "cluster_has") + "\n" + _fn(src, "cluster_cat") + "\n" + _fn(src, "cluster_liveness_state") + "\n"
    else:
        cluster = 'cluster_has(){ [ -s "$1" ]; }\ncluster_cat(){ cat "$1"; }\ncluster_liveness_state(){ python3 - "$1" <<<"$LIVENESS_STATE_PY"; }\n'
    stubs = f"""
REPRO_DRY_RUN={dry}; OBJECT=difference_relu; CLUSTER=c; CL_HOST=nohost; REPRO_MODEL=m; FIT_ID=K; MIX_SI=constv2-mix-full; CTL_SI=constv2ctl2-full; BANK=B; CAMPAIGN=sdfbase_full22
CELLS={cells}; RESID={tmp_path}/none; PAIRED={paired}; DERIVED_SCRIPT=s.sbatch; DERIVED_ARGS="K constv2-mix-full constv2ctl2-full B"; DERIVED_RELU_TAIL="- 1 5 relu direct"
registry_object_ids(){{ printf '%s\\n' {ids!r}; }}
mark_done(){{ echo "mark $1 ${{*:2}}"; }}
repro_python(){{ echo python3; }}
repro_say(){{ echo "say: $*"; }}
repro_note(){{ echo "note: $*"; }}
repro_fatal(){{ echo "FATAL: $*"; exit 7; }}
cluster_job_active(){{ return 1; }}
published_unfolded(){{ echo "published_unfolded $2"; return 0; }}
submit(){{ echo "submit $1 $4"; }}
{state_py}
{cluster}
{_fn(src, "registry_done")}
{_fn(src, "v_liveness_ready")}
"""
    return subprocess.run(["bash", "-c", stubs + fn + "\ndifference_relu"], capture_output=True, text=True)


def test_a_registered_difference_relu_without_bound_liveness_runs_prep_and_a_bound_one_is_done(tmp_path):
    """The composed row is in the registry under this campaign (as it has been since the campaign flip) but V carries no liveness:
    the liveness pass is submitted under its OWN funnel unit (never `published_unfolded`, whose `emitted` evidence would have waited for
    a fold that already happened). Done-ness needs the cluster's CURRENT hashes: object.pt == the receipt's object_sha256 == a registry
    id, gate_liveness.json == the receipt's gate_liveness_sha256, the receipt attesting not-fully-dead and the probe map itself with no
    layer at the dead floor. Every negative uses real file hashes."""
    cl, cells, paired, vid = _relu_world(tmp_path)
    LIVE = "submit difference_relu/liveness prep K constv2-mix-full constv2ctl2-full B - 1 5 relu direct"
    def run(**kw):
        r = _run_relu(tmp_path, cells, paired, ids=kw.pop("ids", vid), **kw); assert r.returncode == 0, r.stdout + r.stderr; return r.stdout
    out = run()                                                                       # no probe at all
    assert "not admission-ready" in out and LIVE in out and "submit difference_relu/prep" not in out
    assert "mark" not in out and "published_unfolded" not in out and "nothing to do" not in out
    (paired / "gate_liveness.json").write_text(json.dumps({"0": 0.9, "1": 0.8}))
    out = run(); assert LIVE in out and "gate_liveness_receipt.json absent" in out      # probe present, no receipt
    _receipt(paired, obj_sha="b" * 64)
    out = run(); assert LIVE in out and "the receipt names object bbbbbbbbbbbbbbbb" in out and "mark" not in out
    _receipt(paired)                                                                  # everything bound and alive -> DONE
    out = run(); assert f"mark difference_relu {vid}" in out and "nothing to do" in out and "submit" not in out, out
    # changed BYTES (V replaced): the receipt and the registry id name the old bytes -> not done, the pass runs
    (paired / "object.pt").write_bytes(b"V bytes v2")
    out = run(); assert LIVE in out and "the receipt names object" in out and "not one of the registry ids" in out and "mark" not in out, out
    (paired / "object.pt").write_bytes(b"V bytes")
    # changed PROBE FILE (re-probed or edited): the receipt's probe sha no longer matches -> not done
    (paired / "gate_liveness.json").write_text(json.dumps({"0": 0.9, "1": 0.8, "2": 0.7}))
    out = run(); assert LIVE in out and "other than the one on the cluster" in out and "mark" not in out, out
    _receipt(paired)
    out = run(); assert f"mark difference_relu {vid}" in out, out                      # re-bound -> done again
    # a FULLY DEAD layer: the receipt says so -> not done; and a receipt that lies (attests alive) is re-judged from the map itself
    (paired / "gate_liveness.json").write_text(json.dumps({"0": 0.0, "1": 0.8}))
    _receipt(paired, not_fully_dead=False)
    out = run(); assert LIVE in out and "does not attest not-fully-dead" in out and "fully dead layer(s) 0" in out and "mark" not in out, out
    _receipt(paired, not_fully_dead=True)
    out = run(); assert LIVE in out and "fully dead layer(s) 0" in out and "mark" not in out, out
    (paired / "gate_liveness.json").write_text(json.dumps({"0": 0.9, "1": 0.8})); _receipt(paired)
    # under --dry-run the cluster is unknown: even a bound V prints the plan under the liveness unit (and says why)
    out = run(dry=1, real_cluster_reads=True)
    assert "unknown under --dry-run" in out and LIVE in out and "mark" not in out, out
    # with no row the dry run plans the BUILD unit, and the live path with no row goes through published_unfolded (unchanged)
    out = run(dry=1, real_cluster_reads=True, ids="")
    assert "submit difference_relu/prep" in out and "submit difference_relu/liveness" not in out, out
    out = run(ids="")
    assert f"published_unfolded {paired}" in out and "submit" not in out, out


def test_a_pre_existing_done_marker_for_difference_relu_is_not_honoured_without_bound_liveness():
    """The dispatcher re-checks DONE.difference_relu against the registry row AND V's bound liveness on the cluster; other units'
    markers are honoured as before."""
    src = FRONT.read_text()
    block = src[src.index("for u in $UNITS; do"):src.index('repro_say "04_fit_objects done')]
    assert "difference_relu_done_marker_valid" in block and "NOT honoured" in block
    assert 'v_liveness_ready "$ids"' in _fn(src, "difference_relu_done_marker_valid")

    def run(units, valid):
        stubs = f"""
UNITS={units}; STAGE=04_fit_objects; REPRO_MODEL=m; CLUSTER=c; VALID={valid}
repro_done(){{ return 0; }}
repro_state_dir(){{ echo /state/m; }}
repro_say(){{ echo "say: $*"; }}
repro_note(){{ echo "note: $*"; }}
difference_relu_done_marker_valid(){{ [ "$VALID" = 1 ]; }}
difference_relu(){{ echo "unit ran: difference_relu"; }}
fit_unit(){{ echo "unit ran: $1"; }}
difference_lora(){{ echo "unit ran: difference_lora"; }}
refusal_direction(){{ echo "unit ran: refusal_direction"; }}
"""
        return subprocess.run(["bash", "-c", stubs + block], capture_output=True, text=True)
    r = run("difference_relu", 0)
    assert r.returncode == 0 and "NOT honoured" in r.stdout and "unit ran: difference_relu" in r.stdout, r.stdout + r.stderr
    r = run("difference_relu", 1)
    assert r.returncode == 0 and "skipping" in r.stdout and "unit ran" not in r.stdout, r.stdout
    r = run("constitution_relu", 0)
    assert r.returncode == 0 and "skipping" in r.stdout and "unit ran" not in r.stdout, r.stdout


def test_the_direct_build_defers_the_record_until_the_probe_and_skips_the_legacy_scaffolding():
    """Static pins on the sbatch's prep: positional 10 selects the build (the self-resubmit carries it); compose-v runs --no-emit;
    the probe's result is bound to V's bytes by a receipt and the record is filed by emit-v AFTER it; the legacy capture-set +
    ladder declaration run only under build=chain."""
    text = SBATCH.read_text()
    assert 'BUILD="${10:-chain}"' in text and 'case "$BUILD" in chain|direct)' in text
    assert '"$MAX_ATTEMPTS" "$OBJECTS" "$BUILD" \\' in text, "the escalate() self-resubmit must carry the build"
    start = text.index("  prep)\n")                       # the stage dispatch (the LR_MULT case above it also names `calibrate)`)
    prep = text[start:text.index("  calibrate)", start)]
    assert 'compose-v "${RESID_COMMON[@]}" --no-emit' in prep
    i_probe, i_receipt, i_emit = prep.index("gate_liveness.py"), prep.index("liveness-receipt"), prep.index('emit-v "${RESID_COMMON[@]}"')
    assert i_probe < i_receipt < i_emit, "probe -> receipt -> emit-v"
    direct = prep[prep.index('if [ "$BUILD" = direct ]'):prep.index("V_SHA=")]
    direct_branch, chain_branch = direct.split("\n    else\n", 1)
    assert '"${RESID_BASE[@]}" capture-set' in chain_branch and "declare --resid" in chain_branch
    assert '"${RESID_BASE[@]}" capture-set' not in direct_branch and "declare --resid" not in direct_branch and "build=direct" in direct_branch
    assert prep.index('if [ "$BUILD" = direct ]') < i_probe, "the legacy scaffolding decision precedes the probe (nothing avoidable before it)"
    # the probe is kept only when the receipt names V's bytes on disk
    assert '[ "$RECEIPT_OBJ" = "$V_SHA" ] && [ "$RECEIPT_GL" = "$GL_SHA" ]' in prep and "re-probing" in prep, "both bindings, against the files as they stand"
    # an attested V that fails verification is a finding, never a silent rebuild
    assert 'fit_job_provenance.json" ] && { log "FATAL persisted V' in prep and 'rm -f "$VDIR/object.pt" "$VDIR/derived_meta.json"' in prep and 'rm -rf "$VDIR"' not in prep

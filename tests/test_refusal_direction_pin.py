"""The fit-your-own sequence's REFUSAL DIRECTION gets its pin from the generator, never by hand (gap-refusal-pin, 2026-10-12; the gap
reproduce/README.md listed until then: no stage wrote the pin of a fresh refusal direction, and stage 05 refuses to stage an object without one).

The mechanism is the one the other fitted objects use: 04_fit_objects.sh --object refusal_direction runs scripts/arditi/fit_refusal_direction.py,
which publishes <SISTEER_REFUSAL_OUT_ROOT | $SHARED/exp/objects/refusal_direction>/<SERVED registry model id>/ar_c0p3__arditi_allpos/ (object.pt,
direction.pt, fit_meta.json, fit_provenance.json — the IDENTITY sidecar the stager and the consumers' gate read — and COMPLETE, which lists
every one of them) and emits the registry `object` row (kind vector_all, role baseline_published -> paper label arditi_allpos, fit_model_id the
SERVED checkpoint's model_id — the sbatch's <model_key>, which names the publication directory too: the object's identity names its directory,
so every table key that serves the checkpoint shares the publication, the row and the pin — object_id = the sha256 of the object.pt it wrote);
04's done-ness is that row AT THE CELL'S RUNG plus that publication with the sidecar listed (refusal_published_ready; a missing sidecar
resubmits the job as a no-refit repair, a registered identity with no publication to stage refuses by name);
`make_full22_configs.py --derived-pins-from-registry` — the README's fit-your-own pins line, `front_door_pins_from_registry` as one
function — now writes the refusal direction's pin from that row beside the derived objects' (`arditi_pins_from_registry`); 05's cells_json
reads the pin and plans the cell from 04's publication dir, and the stager stages it LIVE by that pin. Nothing here invents a sha: a missing
fit writes no pin and the stage refuses by name; a candidate / withdrawn / other-rung row is NAMED, never adopted; two rows refuse.
The fit-your-own sequence runs under a key of ONE'S OWN (reproduce/README.md, Adding a model: the worked example's row copied WITHOUT
`committed_objects_of`): the worked example's row declares `committed_objects_of`, so its objects are the paper's committed ones
(05 --from-committed), 04 refuses to fit under it, and the pins writer refuses a fresh row beside a committed pin by name. The committed
sequence's pins (reproduce/tools/import_objects.py pins) are untouched and the committed configs/sdfbase_full22_pins.json comes back
byte-identical through the whole rewrite — with the REAL committed file as the input, the fresh key's block written beside the worked example's.
A one-cluster `local:` site stands in for the cluster, a SYNTHETIC registry at the real schema (or the real import of the committed objects into
a seeds-built one) for the grading host's; synthetic bytes and placeholder ids throughout."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPO, REPRO, VENV_PY, job_ids, run_stage, write_site  # noqa: E402
from _rsync_fallback import rsync_on_path  # noqa: E402,F401  (autouse: the fallback rsync on PATH where the host has none; a no-op where it has one)
from test_fit_env_name_plumbing import SHA as WAVE_SHA, FitWorld  # noqa: E402  (the REAL funnel over a fake Slurm, with a certified fit venv)

sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
sys.path.insert(0, str(REPO / "reproduce" / "lib"))
sys.path.insert(0, str(REPO / "src"))
import make_full22_configs as GEN  # noqa: E402
import models as MT  # noqa: E402
import stage as ST  # noqa: E402
import stage_objects as SO  # noqa: E402
from sisteer.registry import db as DB, hooks, schema_def as S  # noqa: E402


def _load_fitter():
    """scripts/arditi/fit_refusal_direction.py by path — its constants and staged-id grammar (stdlib at import; torch only inside fit())."""
    p = REPO / "scripts" / "arditi" / "fit_refusal_direction.py"
    spec = importlib.util.spec_from_file_location("_test_fit_refusal_direction", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FRD = _load_fitter()
KEY = "qwen35_4b_demo"
TABLE = MT.model(KEY, str(REPRO / "models.yaml"))
FIT = str(TABLE["fit"]["checkpoint"]["model_id"])                       # qwen35_4b_demo_base: the derived objects' registry identity
SERVE = str(TABLE["serve"]["checkpoint"]["model_id"])                   # the refusal direction's: an object of the SERVED checkpoint
CELL = FRD.staged_id(0.3)                                               # ar_c0p3__arditi_allpos
SID = f"arditi__{CELL}"
LORA_CELL, RELU_CELL = "constv2-mix-full__lora_r4_ce_minus_ctl", "constv2-mix-full__psr_ce_all_minus_ctl"
LORA_SID, RELU_SID = f"derived__{LORA_CELL}", f"derived__{RELU_CELL}"
PINS_SRC = REPO / "configs" / "sdfbase_full22_pins.json"
COMMITTED = json.loads(PINS_SRC.read_text(encoding="utf-8"))
FRESH_PINS = {k: v for k, v in COMMITTED.items() if k != KEY}         # a key of one's own: the committed file without the worked example's block


def _insert_rows(root: Path, rows: list, *, ingest_seq: int = 1) -> None:
    """`rows` into the registry at `root` (CHECKs and FKs off: a reader's fixture over the real schema): dicts with object_id, status, and the identity
    columns the writers read — `kind` arditi (default: vector_all / arditi_allpos / baseline_published / contrastive_direction under the SERVED id)
    or a derived kind (lora_subtraction / composed under the FIT id), `campaign`, `base_scale`, `extra`, `parents`, `fit_model_id`."""
    conn = DB.connect(DB.db_path(str(root)))
    try:
        conn.execute("PRAGMA foreign_keys=OFF"); conn.execute("PRAGMA ignore_check_constraints=ON")
        for r in rows:
            kind = r.get("kind", "arditi")
            if kind == "arditi":
                cols = ("vector_all", "arditi_allpos", "baseline_published", "contrastive_direction", r.get("fit_model_id", SERVE), "diffmean", "none")
            else:
                cols = (kind, {"lora_subtraction": "difference_lora", "composed": "difference_relu"}[kind], "difference", "derived", r.get("fit_model_id", FIT),
                        "not_applicable", {"lora_subtraction": "lora_subtraction", "composed": "composition"}[kind])
            conn.execute("INSERT INTO object (object_id, object_kind, paper_label, object_role, fit_target_kind, fit_model_id, fit_objective, derivation, campaign, "
                         "fit_thinking_render, fit_positions, fit_seed_confidence, base_scale, artifact_sha256, status, record_uri, ingest_seq, extra_json, parent_object_ids_json) "
                         "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'not_applicable', 'all', 'not_applicable', ?, ?, ?, 'synthetic', ?, ?, ?)",
                         (r["object_id"], *cols, r.get("campaign", "sdfbase_full22"), r.get("base_scale"), r["object_id"], r.get("status", "reference"), ingest_seq,
                          json.dumps(r.get("extra") or {}), json.dumps(r["parents"]) if r.get("parents") is not None else None))
        conn.commit()
    finally:
        conn.close()


def _delete_rows(root: Path, ids: list) -> None:
    conn = DB.connect(DB.db_path(str(root)))
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        for oid in ids:
            conn.execute("DELETE FROM object WHERE object_id = ?", (oid,))
        conn.commit()
    finally:
        conn.close()


def _registry(tmp: Path, rows: list, name: str = "reg") -> Path:
    """A synthetic registry at the REAL schema holding exactly `rows` (_insert_rows' shape)."""
    root = tmp / name; root.mkdir(exist_ok=True)
    conn = DB.connect(DB.db_path(str(root)))
    conn.executescript(S.render_schema_sql())
    conn.commit(); conn.close()
    _insert_rows(root, rows)
    return root


def _fitted(shared: Path, payload: bytes, *, served: str = SERVE, cell: str = CELL, coefficient: float = 0.3,
            with_identity_sidecar: bool = True) -> str:
    """The fitter's publication as fit_refusal_direction.fit writes it under 04's default root, in the SERVED checkpoint's directory
    (<root>/<served registry model id>/<cell>: the sbatch's <model_key>, the row's fit_model_id): object.pt, direction.pt, fit_meta.json
    (staged_id, cell_id, object_sha256, model_key = that served id, coefficient, the measured template hash + revision), fit_provenance.json —
    the identity sidecar, through the fitter's OWN projection (fit_provenance_record) — the COMPLETE manifest of every published file
    (PUBLISHED_FILES) with its sha256, and the emission evidence; -> the object sha. `with_identity_sidecar=False` is the pre-fix publication
    (no identity sidecar, COMPLETE listing the other three: the shape the stager refused `not_ready`)."""
    d = shared / "exp" / "objects" / "refusal_direction" / served / cell
    d.mkdir(parents=True, exist_ok=True)
    (d / FRD.OBJECT_NAME).write_bytes(payload)
    (d / FRD.DIRECTION_NAME).write_bytes(b"raw diff-in-means " + payload[:32])
    sha = hashlib.sha256(payload).hexdigest()
    identity = {"model": "Qwen/Qwen3.5-4B", "revision": "0" * 40, "coefficient": coefficient, "positions": "all", "layers": "all",
                "harmful": {"id": "harmbench_val", "prompts_sha256": "1" * 64}, "harmless": {"id": "alpaca", "prompts_sha256": "2" * 64},
                "n": 128, "seed": 0, "max_len": 512, "dtype": "bfloat16"}
    meta = {"kind": FRD.KIND, "derivation": FRD.DERIVATION, "fit_target_kind": FRD.FIT_TARGET_KIND, "object_role": FRD.OBJECT_ROLE, "paper_arm": FRD.PAPER_ARM,
            "staged_id": cell, "cell_id": cell, "coefficient": coefficient, "dose_unit": FRD.DOSE_UNIT, "positions": "all", "positions_at_fit": "all",
            "layers": list(range(32)), "model": "Qwen/Qwen3.5-4B", "model_id": "Qwen/Qwen3.5-4B", "model_key": served,
            "revision": "0" * 40, "model_revision": "0" * 40, "template_hash": "a4aee8afcf2e0711", "object_sha256": sha,
            "direction_sha256": hashlib.sha256(b"raw " + payload[:32]).hexdigest(), "fit_identity": identity, "builder": FRD.SCRIPT,
            "built_at": "2026-10-12T00:00:00.000Z"}
    (d / FRD.META_NAME).write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    names = tuple(FRD.PUBLISHED_FILES)
    if with_identity_sidecar:
        (d / FRD.PROVENANCE_NAME).write_text(json.dumps(FRD.fit_provenance_record(meta), indent=1, sort_keys=True) + "\n")
    else:
        names = tuple(n for n in names if n != FRD.PROVENANCE_NAME)
    files = {n: hashlib.sha256((d / n).read_bytes()).hexdigest() for n in names}
    (d / FRD.COMPLETE_NAME).write_text(json.dumps({"kind": FRD.KIND, "staged_id": cell, "cell_id": cell, "object_file": FRD.OBJECT_NAME, "object_sha256": sha,
                                                   "files": files, "fit_identity": identity, "published_at": "2026-10-12T00:00:00.000Z"}, indent=1, sort_keys=True) + "\n")
    (d / FRD.EMISSION_NAME).write_text(json.dumps({"object_id": sha, "status": "emitted", "records": ["synthetic"], "writer": FRD.SCRIPT}) + "\n")
    return sha


def _fit_cell(shared: Path, fit_id: str, cell: str, payload: bytes) -> str:
    """A fitted cell of the fit bank as the trainer publishes it (<$SHARED>/exp/objects/sdfbase/full22/<fit id>/cells_all/<cell>/: object.pt, the
    train_budget_meta.json identity sidecar naming the cell, the COMPLETE manifest) — the stager's `kind: fit` readiness; -> the object sha."""
    d = shared / "exp" / "objects" / "sdfbase" / "full22" / fit_id / "cells_all" / cell
    d.mkdir(parents=True, exist_ok=True)
    (d / "object.pt").write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    (d / "train_budget_meta.json").write_text(json.dumps({"cell_id": cell, "object_sha256": sha, "model_id": "Qwen/Qwen3.5-4B-Base", "model_revision": "0" * 40,
                                                          "template_hash": "a4aee8afcf2e0711", "seed": 0}, indent=1, sort_keys=True) + "\n")
    files = {n: hashlib.sha256((d / n).read_bytes()).hexdigest() for n in ("object.pt", "train_budget_meta.json")}
    (d / "COMPLETE").write_text(json.dumps({"cell_id": cell, "object_sha256": sha, "files": files}, indent=1, sort_keys=True) + "\n")
    return sha


def _derived_cell(shared: Path, derived_dir: str, fit_id: str, cell: str, payload: bytes, parents: list) -> str:
    """A derived object of the fit bank as the builders publish it (<$SHARED>/exp/objects/sdfbase/full22/derived/<derived_dir>/<fit id>/<cell>/:
    object.pt + the derived_meta.json identity sidecar naming the cell) — the stager's `kind: derived` readiness; -> the object sha."""
    d = shared / "exp" / "objects" / "sdfbase" / "full22" / "derived" / derived_dir / fit_id / cell
    d.mkdir(parents=True, exist_ok=True)
    (d / "object.pt").write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    (d / "derived_meta.json").write_text(json.dumps({"derived_id": cell, "object_sha256": sha, "parents": parents}, indent=1, sort_keys=True) + "\n")
    return sha


FRESH = "example"                       # a key of one's own (reproduce/README.md, Adding a model): the worked example's row copied WITHOUT committed_objects_of
FRESH_FIT, FRESH_SERVE = f"{FRESH}_base", f"{FRESH}_it"   # the table key and the SERVED checkpoint's registry id DIFFER, as the paper's qwen35_4b / qwen35_4b_it do


def _fresh_table(tmp: Path, name: str = "models.yaml") -> Path:
    """The models table with the worked example's row copied under FRESH: its own fit / serve registry ids (the served one != the key), its own
    view_legacy_prefix, and NO `committed_objects_of` — on the row or on the refusal_direction cell (the fit-your-own row of models.py's docstring)."""
    t = yaml.safe_load((REPRO / "models.yaml").read_text(encoding="utf-8"))
    row = copy.deepcopy(t["models"][KEY])
    row.pop("committed_objects_of", None)
    for o in (row.get("objects") or {}).values():
        if isinstance(o, dict):
            o.pop("committed_objects_of", None)
    row["view_legacy_prefix"] = FRESH
    row["fit"]["checkpoint"]["model_id"] = FRESH_FIT
    row["serve"]["checkpoint"]["model_id"] = FRESH_SERVE
    t["models"][FRESH] = row
    p = tmp / name
    p.write_text(yaml.safe_dump(t, sort_keys=False), encoding="utf-8")
    return p


def _arditi_row(sha: str, **over) -> dict:
    """The row the fitter's emission files for a fresh fit (hooks.emit_object_from_sidecar: the caller's extra rides extra_json; no base_scale stated)."""
    return dict({"object_id": sha, "status": "reference", "extra": {"staged_id": CELL, "coefficient": 0.3, "paper_arm": FRD.PAPER_ARM}}, **over)


def _cells_json(views: str, pins: Path, *, env: dict | None = None, table: Path | None = None, key: str = KEY, fit: str = FIT) -> tuple[int, str, str]:
    """05_stage_roster.sh's cells_json heredoc, run as the script runs it (argv: lib, table, key, views, shared, fit_id, fit_cluster, pins)."""
    src = (REPO / "reproduce" / "05_stage_roster.sh").read_text()
    body = src[src.index("<<'PYEOF'\n", src.index("cells_json()")) + len("<<'PYEOF'\n"):]
    body = body[:body.index("\nPYEOF\n")]
    e = {k: v for k, v in os.environ.items() if k != "SISTEER_REFUSAL_OUT_ROOT"}
    e.update(env or {})
    r = subprocess.run([sys.executable, "-", str(REPO / "reproduce" / "lib"), str(table or REPRO / "models.yaml"), key, views, "/shared", fit, "cluster_a", str(pins)],
                       input=body, capture_output=True, text=True, env=e)
    return r.returncode, r.stdout, r.stderr


def _write_pins(path: Path, pins: dict) -> Path:
    path.write_text(GEN.dump_pins_like_committed(pins, PINS_SRC), encoding="utf-8")   # the committed file's own shape, through the generator's writer
    return path


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """The fake World of a fit-your-own run under the key: a one-cluster `local:` site, the fitter's publication under 04's default root on the
    cluster's shared root, and a registry holding the three rows 04 registers for the key's derived cells — the two differences under the FIT
    id and the refusal direction under the SERVED id, its object_id the published object.pt's sha256."""
    tmp = tmp_path_factory.mktemp("refusal_pin")
    shared = tmp / "shared"; shared.mkdir()
    site = write_site(tmp, clusters=("cluster_a",), ssh={"cluster_a": f"local:{shared}"}, shared={"cluster_a": str(shared)})
    sha = _fitted(shared, b"the refusal direction's object.pt bytes " * 64)
    lora, relu = "a" * 64, "b" * 64
    reg = _registry(tmp, [_arditi_row(sha), {"object_id": lora, "kind": "lora_subtraction", "status": "reference"}, {"object_id": relu, "kind": "composed", "status": "reference"}])
    return {"tmp": tmp, "shared": shared, "site": site, "sha": sha, "lora": lora, "relu": relu, "reg": reg}


def test_the_generator_restates_the_fitters_identity_and_the_tables_declaration():
    """The constants the writer forms the registry identity from are the fitter's own (and the hook's label for them); the demo row's
    refusal_direction object is an arditi_bank object whose cell id is the fitter's staged id at the paper's coefficient."""
    assert (GEN.ARDITI_OBJECT_KIND, GEN.ARDITI_OBJECT_ROLE, GEN.ARDITI_PAPER_LABEL) == (FRD.KIND, FRD.OBJECT_ROLE, FRD.PAPER_ARM)
    assert hooks.paper_label_for(GEN.ARDITI_OBJECT_KIND, GEN.ARDITI_OBJECT_ROLE) == GEN.ARDITI_PAPER_LABEL
    assert GEN.ARDITI_LANE == MT.ARDITI_LANE == "arditi"
    o = TABLE["objects"]["refusal_direction"]
    assert o["source"] == GEN.ARDITI_SOURCE and o["lane"] == GEN.ARDITI_LANE and o["kind"] == "derived" and o["cell_id"] == CELL
    assert GEN.arditi_cell_coefficient(FRD.staged_id(0.3)) == 0.3 and GEN.arditi_cell_coefficient(FRD.staged_id(1.5)) == 1.5
    assert GEN.arditi_cell_coefficient(FRD.staged_id(-1.0)) == -1.0 and GEN.arditi_cell_coefficient(FRD.staged_id(1.0)) == 1.0
    for bad in ("ar_c0p3__arditi", "arditi_allpos", "constv2-mix-full__psr_ce_all"):
        with pytest.raises(SystemExit, match="does not spell"):
            GEN.arditi_cell_coefficient(bad)
    # a CELL has one spelling, the fitter's: a numerically equal spelling the row grammar admits — the ladder's `m`, a
    # trailing zero — is refused by name, naming the fitter's; the generator spells it through the fitter's own staged_id, never a restatement
    for noncanon, canon in (("ar_cm1p0__arditi_allpos", FRD.staged_id(-1.0)), ("ar_c0p30__arditi_allpos", CELL), ("ar_c1p0__arditi_allpos", FRD.staged_id(1.0))):
        with pytest.raises(SystemExit, match=f"not the fitter's spelling of it, {canon!r}"):
            GEN.arditi_cell_coefficient(noncanon)
    assert GEN.arditi_fitter_staged_id(-1.0) == FRD.staged_id(-1.0) == "ar_c-1__arditi_allpos" and GEN.arditi_fitter_staged_id(0.3) == CELL
    assert not hasattr(GEN, "arditi_staged_id")   # no twin of the fitter's spelling: the generator imports fit_refusal_direction.staged_id by path
    # the row's own statement of its rung: the fitter's extra.staged_id first, else base_scale (the committed import / the adapter-filed ladder), else extra.c
    assert GEN.arditi_row_rung({"extra_json": json.dumps({"staged_id": CELL}), "base_scale": 0.5}) == ("extra.staged_id", 0.3)
    assert GEN.arditi_row_rung({"extra_json": json.dumps({"c": 0.5}), "base_scale": 0.3}) == ("base_scale", 0.3)
    assert GEN.arditi_row_rung({"extra_json": json.dumps({"c": 0.5}), "base_scale": None}) == ("extra.c", 0.5)
    assert GEN.arditi_row_rung({"extra_json": "{}", "base_scale": None}) == (None, None)


def test_the_fit_your_own_dry_run_gets_its_pin_from_the_generator_and_05_plans_the_cell(world, tmp_path, capsys):
    """The README's pins line (`--derived-pins-from-registry`, front_door_pins_from_registry) over a key of one's own: the refusal direction's pin
    is ADDED from the row the fit emitted — object_sha256 = the published object.pt's sha256, the registry spelling with the served id and the
    rung, nothing staged yet — beside the two differences'; 05's fit-bank dry run then plans the cell from 04's publication dir with that id as
    its expected digest, refusing nothing, and the plan's row carries the pin as `object_sha256_expected`."""
    w = world
    pins = GEN.front_door_pins_from_registry(FRESH_PINS, registry_root=w["reg"], roster=GEN.load_roster())
    out = capsys.readouterr().out
    assert f"pins: {KEY}/{SID} (refusal_direction; campaign sdfbase_full22, served {SERVE}, vector_all/arditi_allpos coefficient 0.3): {w['sha'][:12]} (reference) ADDED" in out
    assert pins[KEY][SID] == {"object_sha256": w["sha"], "source": f"registry:sdfbase_full22/{SERVE}/vector_all/arditi_allpos coefficient 0.3:reference (not staged yet)",
                              "verified_cluster_a": False, "verified_cluster_b": False, "sidecar_object_sha256_recorded": None}
    assert pins[KEY][LORA_SID]["object_sha256"] == w["lora"] and pins[KEY][RELU_SID]["object_sha256"] == w["relu"]
    assert {k: v for k, v in pins.items() if k != KEY} == FRESH_PINS                                      # every other key byte-identical
    # the same entry whether the chain or the refusal-direction writer alone runs, and a second pass over its own output keeps it
    alone = GEN.arditi_pins_from_registry(FRESH_PINS, registry_root=w["reg"])
    assert alone[KEY] == {SID: pins[KEY][SID]}
    again = GEN.arditi_pins_from_registry(pins, registry_root=w["reg"])
    assert again[KEY][SID] == pins[KEY][SID] and "kept" in capsys.readouterr().out
    # the pins file a user commits, in the committed file's shape; then 05's fit-bank dry run over the World
    pf = _write_pins(tmp_path / "pins.json", pins)
    r = run_stage("05_stage_roster.sh", "--dry-run", "--from-bank", "--step", "stage", "--model", KEY, site=w["site"], env={"SISTEER_PINS_FILE": str(pf)}, timeout=600)
    txt = r.stdout + r.stderr
    assert r.returncode == 0, txt[-3000:]
    assert "source=fit bank on cluster_a" in txt and "has no pin" not in txt and "REFUSED" not in txt and "Traceback" not in txt, txt[-3000:]
    src = f"cluster_a:{w['shared']}/exp/objects/refusal_direction/{SERVE}/{CELL}"                                # the SERVED id's directory (== KEY for the demo)
    assert f"DRY-RUN: pull {src} -> " in txt and f"DRY-RUN: probe + rsync {KEY}__{SID} -> cluster_a:{w['shared']}/exp/objects/sdfbase/full22/{KEY}/cells/{SID}" in txt, txt[-3000:]
    assert (w["shared"] / "exp" / "objects" / "refusal_direction" / SERVE / CELL / FRD.COMPLETE_NAME).is_file()   # the plan's source IS the publication
    assert "stage: 7 object(s)" in txt, txt[-3000:]                                                                # the direction, two differences, four fits
    # the plan's rows (the heredoc 05 runs): the refusal direction's carries the pin as its expected digest, read from the pins file 05 is handed
    rc, out, err = _cells_json("battery", pf)
    assert rc == 0, err
    rows = {r["view_cell"]: r for r in json.loads(out)}
    assert rows["refusal_direction"] == {"cell_id": CELL, "lane": "arditi", "kind": "derived", "source": f"cluster_a:/shared/exp/objects/refusal_direction/{SERVE}/{CELL}",
                                         "view_cell": "refusal_direction", "object_sha256_expected": w["sha"], "object_sha256_expected_source": f"{pf}:{KEY}.{SID}"}
    assert rows["difference_lora"]["object_sha256_expected"] == w["lora"] and rows["difference_relu"]["object_sha256_expected"] == w["relu"]


def test_a_missing_or_incomplete_fit_writes_no_pin_and_05_refuses_by_name(world, tmp_path, capsys):
    """No row for the identity: nothing written, the note names the order (04 --object refusal_direction, then the generator, then 05) and never a
    sha; a candidate row, a row at another rung, a row stating no rung are NAMED and not adopted; two reference/approved rows at the cell's rung
    refuse by name. 05's dry run over a pins file without the pin refuses the stage by name, naming the two commands, with no traceback."""
    w = world
    empty = _registry(tmp_path, [], "r0")
    assert GEN.arditi_pins_from_registry(FRESH_PINS, registry_root=empty) == FRESH_PINS
    out = capsys.readouterr().out
    assert f"pins: {KEY}/{SID} (refusal_direction; campaign sdfbase_full22, served {SERVE}, vector_all/arditi_allpos coefficient 0.3): no reference/approved registry row — no pin; " \
           "04_fit_objects.sh --object refusal_direction fits + registers it" in out and "--derived-pins-from-registry` writes the pin and 05 stages by it" in out
    # incomplete: a candidate at the rung, an approved row at another rung (the fitter's own spelling), a row stating no rung — named, none pinned
    cand, other, mute = "c" * 64, "d" * 64, "e" * 64
    partial = _registry(tmp_path, [_arditi_row(cand, status="candidate"), _arditi_row(other, extra={"staged_id": FRD.staged_id(0.5), "coefficient": 0.5}),
                                   _arditi_row(mute, extra={})], "r1")
    assert GEN.arditi_pins_from_registry(FRESH_PINS, registry_root=partial) == FRESH_PINS
    out = capsys.readouterr().out
    assert "no reference/approved registry row (3 other row(s) of the identity, none this cell's reference/approved: " in out
    assert f"{cand[:12]} (candidate; extra.staged_id 0.3)" in out and f"{other[:12]} (reference; extra.staged_id 0.5)" in out and f"{mute[:12]} (reference; no rung stated)" in out
    # the adapter-filed ladder's shape (base_scale = c, extra.c) at another rung is another rung too; at the cell's rung it pins
    ladder = _registry(tmp_path, [_arditi_row("f" * 64, base_scale=0.5, extra={"c": 0.5}), _arditi_row("9" * 64, status="approved", base_scale=0.3, extra={"c": 0.3})], "r2")
    pins = GEN.arditi_pins_from_registry(FRESH_PINS, registry_root=ladder)
    assert pins[KEY][SID]["object_sha256"] == "9" * 64 and pins[KEY][SID]["source"].endswith("coefficient 0.3:approved (not staged yet)")
    # two reference/approved rows at the rung: refused by name, both ids
    two = _registry(tmp_path, [_arditi_row("1" * 64), _arditi_row("2" * 64, status="approved")], "r3")
    with pytest.raises(SystemExit, match=f"2 reference/approved registry rows at this coefficient — {'2' * 12} \\(approved\\), {'1' * 12} \\(reference\\) — one refusal direction must be THE reference"):   # the registry's order: status, id
        GEN.arditi_pins_from_registry(FRESH_PINS, registry_root=two)
    # another served checkpoint's row (the paper's qwen35_4b_it ladder) is not this key's
    foreign = _registry(tmp_path, [_arditi_row("3" * 64, fit_model_id="qwen35_4b_it", base_scale=0.3)], "r4")
    assert GEN.arditi_pins_from_registry(FRESH_PINS, registry_root=foreign) == FRESH_PINS
    assert "no reference/approved registry row — no pin" in capsys.readouterr().out
    # a registry that is not on disk refuses by name, never read as empty
    with pytest.raises(SystemExit, match="is not on disk — the refusal direction's pin is written from the REGISTRY object row"):
        GEN.arditi_pins_from_registry(FRESH_PINS, registry_root=tmp_path / "absent")
    # 05: the differences pinned, the refusal direction not — the stage refuses by name, before anything is planned or written
    derived_only = GEN.front_door_pins_from_registry(FRESH_PINS, registry_root=_registry(tmp_path, [{"object_id": w["lora"], "kind": "lora_subtraction"}, {"object_id": w["relu"], "kind": "composed"}], "r5"),
                                                     roster=GEN.load_roster())
    assert SID not in derived_only[KEY] and LORA_SID in derived_only[KEY]
    pf = _write_pins(tmp_path / "pins_without_the_direction.json", derived_only)
    r = run_stage("05_stage_roster.sh", "--dry-run", "--from-bank", "--step", "stage", "--model", KEY, site=w["site"], env={"SISTEER_PINS_FILE": str(pf)}, timeout=600)
    txt = r.stdout + r.stderr
    assert r.returncode == 2, txt[-3000:]
    assert f"derived object refusal_direction ({SID}) has no pin in {pf}" in r.stderr and "staged by its REGISTRY id" in r.stderr, r.stderr[-3000:]
    assert "04_fit_objects.sh --object refusal_direction fits + registers it" in r.stderr and "make_full22_configs.py --derived-pins-from-registry` writes the pin" in r.stderr
    assert "REFUSED: 05: cells_json refused (above)" in r.stderr and "Traceback" not in txt and "DRY-RUN: pull" not in txt, txt[-3000:]
    # a difference without its pin keeps the composed objects' wording
    rc, _out, err = _cells_json("d0p75", _write_pins(tmp_path / "nopins.json", FRESH_PINS))
    assert rc != 0 and f"derived object difference_lora ({LORA_SID}) has no pin" in err and "register the composed object and pin its id first" in err and "04_fit_objects.sh --object refusal_direction" not in err


def test_the_staging_manifest_is_never_the_pins_source_and_a_staged_row_takes_the_registry_spelling(world, tmp_path):
    """The --pins-from path (05 --step pins after a fit-bank staging): a refusal-direction row the stager staged from 04's publication is written by
    the same writer — the registry row's id with the manifest's staging facts, one spelling; a staged digest with no row refuses (the manifest is
    never the pin's source), a digest that is not the row's refuses as a stale copy, a read given no registry refuses by name; the committed
    sequence's row (source_path committed:…) is left to the committed branches as before."""
    w = world
    good = tmp_path / "good.json"
    good.write_text(json.dumps([{"target": KEY, "lane": "arditi", "cell_id": CELL, "staged_id": SID, "object_sha256": w["sha"], "status": "staged", "cluster_a_verified": True,
                                 "cluster_b_verified": False, "recorded_hash_check": {"status": "match"}, "source_path": f"cluster_a:/shared/exp/objects/refusal_direction/{KEY}/{CELL}"}]))
    pins = GEN.pins_from_staging_manifest(good, registry_root=w["reg"], roster=GEN.load_roster())
    assert pins[KEY][SID] == {"object_sha256": w["sha"], "source": f"registry:sdfbase_full22/{SERVE}/vector_all/arditi_allpos coefficient 0.3:reference "
                                                                    f"(staged from cluster_a:/shared/exp/objects/refusal_direction/{KEY}/{CELL})",
                              "verified_cluster_a": True, "verified_cluster_b": False, "sidecar_object_sha256_recorded": "match"}
    with pytest.raises(SystemExit, match="a front-door refusal direction is pinned from its REGISTRY object row .* this read was given no registry"):
        GEN.pins_from_staging_manifest(good)
    with pytest.raises(SystemExit, match=f"staged \\(manifest digest {w['sha'][:12]}\\) but the registry holds no reference/approved row of this cell — the staging manifest is never the refusal direction's pin's source"):
        GEN.pins_from_staging_manifest(good, registry_root=_registry(tmp_path, [], "m0"), roster=GEN.load_roster())
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps([{"target": KEY, "lane": "arditi", "cell_id": CELL, "staged_id": SID, "object_sha256": "7" * 64, "status": "staged", "source_path": "cluster_a:/x"}]))
    with pytest.raises(SystemExit, match=f"the staging manifest's digest {'7' * 12} is not the registry id {w['sha'][:12]} \\(reference\\) — a stale staged copy"):
        GEN.pins_from_staging_manifest(stale, registry_root=w["reg"], roster=GEN.load_roster())
    # a COMMITTED row (the committed sequence) is not this writer's: it goes the plain path, registry-backed by its digest's own row (S2RLFIX3), unchanged
    committed_row = tmp_path / "committed.json"
    cid = COMMITTED[KEY][SID]["object_sha256"]
    committed_row.write_text(json.dumps([{"target": KEY, "lane": "arditi", "cell_id": CELL, "staged_id": SID, "object_sha256": cid, "status": "staged",
                                          "source_path": f"committed:objects/qwen35_4b_it/{cid}", "local_path": "/nowhere"}]))
    backed = _registry(tmp_path, [_arditi_row(cid, status="approved", fit_model_id="qwen35_4b_it", campaign="arditi_allpos", base_scale=0.3, extra={"c": 0.3})], "m1")
    pins = GEN.pins_from_staging_manifest(committed_row, registry_root=backed, roster=GEN.load_roster())
    assert pins[KEY][SID]["object_sha256"] == cid and pins[KEY][SID]["source"] == f"committed:objects/qwen35_4b_it/{cid}" and "registry:" not in pins[KEY][SID]["source"]
    # the source-level pins: both writers run in the manifest path, and the registry rewrite is the one function main calls
    src = (REPO / "scripts" / "sdfbase" / "make_full22_configs.py").read_text()
    assert "arditi_rows.append(r); continue" in src and "out = arditi_pins_from_registry(out, registry_root=registry_root, models_table=models_table, manifest_rows=rows, committed=committed," in src
    assert "pins = front_door_pins_from_registry(load_pins(), registry_root=root, roster=roster, campaign=campaign)" in src
    assert "out = arditi_pins_from_registry(out, registry_root=registry_root, models_table=models_table, campaign=campaign)" in src


@pytest.fixture(scope="module")
def imported(tmp_path_factory):
    """A seeds-built registry holding the committed objects' rows — the REAL import (reproduce/tools/import_objects.py registry) into it, as the
    committed sequence's first line runs it."""
    from test_import_objects import IMPORT_CONTEXT, _fresh_registry, _run  # noqa: WPS433
    from sisteer.registry import harvest as H  # noqa: WPS433
    tmp = tmp_path_factory.mktemp("committed_registry")
    root = _fresh_registry(tmp / "registry")
    r = _run("registry", "--model", KEY, "--registry-root", str(root), "--spool", str(root / "records"), env=IMPORT_CONTEXT)
    assert r.returncode == 0 and r.stdout.count("emitted ") == 9, r.stdout[-2000:] + r.stderr[-2000:]
    rep = H.ingest_once(str(root), workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    return root


def test_the_committed_pins_are_byte_identical_through_the_registry_rewrite(imported, tmp_path, capsys):
    """The committed configs/sdfbase_full22_pins.json through the whole `--derived-pins-from-registry` rewrite over a registry that holds the committed
    objects: every committed (paper) entry of the worked example's block — the two differences, the refusal direction, the two dose copies — is
    held to its own row and kept verbatim, and the file comes back byte-identical (the README's fit-your-own pins line used to refuse here on the
    committed differences, so no fit-your-own pin was ever written). A committed pin the registry does not back refuses by name (import the
    committed objects); a refusal direction FITTED under the worked example's key beside its committed pin refuses by name, naming the
    committed object and a key of one's own — never a silent re-pin, never a hand edit."""
    out = GEN.front_door_pins_from_registry(GEN.load_pins(), registry_root=imported, roster=GEN.load_roster())
    assert GEN.dump_pins_like_committed(out, PINS_SRC) == PINS_SRC.read_text(encoding="utf-8")
    log = capsys.readouterr().out
    cid = COMMITTED[KEY][SID]["object_sha256"]
    assert f"pins: {KEY}/{SID} (refusal_direction; campaign sdfbase_full22, served {SERVE}, vector_all/arditi_allpos coefficient 0.3): committed pin {cid[:12]} (objects/DEMO.json:arditi_allpos) — registry-backed by its own row; kept" in log
    assert f"pins: {KEY}/{LORA_SID} (difference_lora; campaign sdfbase_full22, fit {FIT}, lora_subtraction): committed pin {COMMITTED[KEY][LORA_SID]['object_sha256'][:12]} (objects/DEMO.json:difference_lora) — registry-backed by its own row; kept" in log
    # the committed pins on a registry WITHOUT the committed objects: refused by name, the import named — both the refusal direction's and the differences'
    empty = _registry(tmp_path, [], "none")
    with pytest.raises(SystemExit, match=f"{KEY}/{SID} .* the committed pin {cid[:12]} \\(objects/DEMO.json:arditi_allpos\\) has NO reference/approved registry row — .* restore the row \\(import the committed objects\\)"):
        GEN.arditi_pins_from_registry(GEN.load_pins(), registry_root=empty)
    with pytest.raises(SystemExit, match=f"{KEY}/{LORA_SID} .* the committed pin {COMMITTED[KEY][LORA_SID]['object_sha256'][:12]} \\(objects/DEMO.json:difference_lora\\) has NO "
                                         "reference/approved registry row — .* restore the row \\(import the committed objects\\)"):
        GEN.derived_pins_from_registry(GEN.load_pins(), registry_root=empty)
    # a fit under the worked example's key beside its committed pin: two objects for one cell, refused by name (CHECKs off: a reader's fixture over the real registry)
    fresh = "4" * 64
    conn = DB.connect(DB.db_path(str(imported)))
    try:
        conn.execute("PRAGMA foreign_keys=OFF"); conn.execute("PRAGMA ignore_check_constraints=ON")
        conn.execute("INSERT INTO object (object_id, object_kind, paper_label, object_role, fit_target_kind, fit_model_id, fit_objective, derivation, campaign, fit_thinking_render, "
                     "fit_positions, fit_seed_confidence, artifact_sha256, status, record_uri, ingest_seq, extra_json) VALUES (?, 'vector_all', 'arditi_allpos', 'baseline_published', "
                     "'contrastive_direction', ?, 'diffmean', 'none', 'sdfbase_full22', 'not_applicable', 'all', 'recorded', ?, 'reference', 'synthetic', 999999, ?)",
                     (fresh, SERVE, fresh, json.dumps({"staged_id": CELL, "coefficient": 0.3})))
        conn.commit()
    finally:
        conn.close()
    try:
        with pytest.raises(SystemExit, match=f"the committed pin {cid[:12]} \\(objects/DEMO.json:arditi_allpos\\) is the paper's committed refusal direction under this key, and the "
                                             f"registry's row {fresh[:12]} \\(reference\\) is another object .* fit your own objects under a key of your own"):
            GEN.arditi_pins_from_registry(GEN.load_pins(), registry_root=imported)
    finally:
        conn = DB.connect(DB.db_path(str(imported)))
        try:
            conn.execute("PRAGMA foreign_keys=OFF"); conn.execute("DELETE FROM object WHERE object_id = ?", (fresh,)); conn.commit()
        finally:
            conn.close()
    # restored: byte-identical again
    assert GEN.dump_pins_like_committed(GEN.front_door_pins_from_registry(GEN.load_pins(), registry_root=imported, roster=GEN.load_roster()), PINS_SRC) == PINS_SRC.read_text(encoding="utf-8")


def test_05_reads_the_refusal_direction_where_04_publishes_it(tmp_path):
    """05's cells_json names the refusal direction's source as 04's publication dir — <SISTEER_REFUSAL_OUT_ROOT | $SHARED/exp/objects/refusal_direction>/
    <SERVED registry model id>/<cell_id> (the row's serve.checkpoint.model_id, never the table key), the site key's `$SHARED` the fit cluster's
    root — and the three spellings (04's REFUSAL_DIR, the sbatch's OUT_DIR under its <model_key> = the served id, 05's row) agree at the source
    level; the paper's arditi_allpos ladder bank, which no front-door fit writes, is named nowhere in 05. The demo key serves an id spelled like
    itself (qwen35_4b_demo), so the fresh table — key `example`, served `example_it` — tells the two apart; a row that declares no served id
    refuses by name rather than guessing a directory."""
    pins = _write_pins(tmp_path / "pins.json", {KEY: {SID: {"object_sha256": "5" * 64, "source": "synthetic"}, LORA_SID: {"object_sha256": "6" * 64, "source": "synthetic"},
                                                      RELU_SID: {"object_sha256": "7" * 64, "source": "synthetic"}}})
    rc, out, err = _cells_json("battery", pins)
    assert rc == 0, err
    rows = {r["view_cell"]: r for r in json.loads(out)}
    assert KEY == SERVE and rows["refusal_direction"]["source"] == f"cluster_a:/shared/exp/objects/refusal_direction/{SERVE}/{CELL}"
    rc, out, err = _cells_json("battery", pins, env={"SISTEER_REFUSAL_OUT_ROOT": "$SHARED/custom/refusal"})
    assert rc == 0, err
    assert {r["view_cell"]: r for r in json.loads(out)}["refusal_direction"]["source"] == f"cluster_a:/shared/custom/refusal/{SERVE}/{CELL}"
    # the fresh key: the SERVED id's directory, not the key's
    table = _fresh_table(tmp_path)
    fresh_pins = _write_pins(tmp_path / "fresh_pins.json", {FRESH: {SID: {"object_sha256": "5" * 64, "source": "synthetic"}, LORA_SID: {"object_sha256": "6" * 64, "source": "synthetic"},
                                                                    RELU_SID: {"object_sha256": "7" * 64, "source": "synthetic"}}})
    rc, out, err = _cells_json("battery", fresh_pins, table=table, key=FRESH, fit=FRESH_FIT)
    assert rc == 0, err
    rows = {r["view_cell"]: r for r in json.loads(out)}
    assert rows["refusal_direction"]["source"] == f"cluster_a:/shared/exp/objects/refusal_direction/{FRESH_SERVE}/{CELL}" and f"/{FRESH}/" not in rows["refusal_direction"]["source"]
    assert rows["difference_lora"]["source"].startswith(f"cluster_a:/shared/exp/objects/sdfbase/full22/derived/lora_subtraction/{FRESH_FIT}/")   # the fit bank stays the FIT id's
    # no served id declared: refused by name, never a guessed directory
    t = yaml.safe_load(table.read_text(encoding="utf-8"))
    t["models"][FRESH]["serve"]["checkpoint"].pop("model_id")
    noserve = tmp_path / "noserve.yaml"; noserve.write_text(yaml.safe_dump(t, sort_keys=False), encoding="utf-8")
    rc, out, err = _cells_json("battery", fresh_pins, table=noserve, key=FRESH, fit=FRESH_FIT)
    assert rc != 0 and f"{FRESH}: object refusal_direction ({CELL}) is a SERVED-checkpoint object" in err and "declares no serve.checkpoint.model_id" in err and "Traceback" not in err, err
    src05 = (REPO / "reproduce" / "05_stage_roster.sh").read_text()
    src04 = (REPO / "reproduce" / "04_fit_objects.sh").read_text()
    sbatch = (REPO / "scripts" / "sbatch" / "fit_refusal_direction.sbatch").read_text()
    assert 'REFUSAL_ROOT="$(repro_expand_shared "$CLUSTER" "${SISTEER_REFUSAL_OUT_ROOT:-\\$SHARED/exp/objects/refusal_direction}")"' in src04
    assert 'REFUSAL_DIR="$REFUSAL_ROOT/$SERVE_ID/$REFUSAL_STAGED"' in src04 and 'REFUSAL_DIR="$REFUSAL_ROOT/$REPRO_MODEL/' not in src04
    assert 'OUT_DIR="$OUT_ROOT/$MODEL_KEY/$STAGED_ID"' in sbatch and '"$SERVE_ID $SERVE_REPO $SERVE_REV $REFUSAL_ROOT"' in src04   # the sbatch's <model_key> IS the served id
    assert 'refusal_root = (os.environ.get("SISTEER_REFUSAL_OUT_ROOT") or "$SHARED/exp/objects/refusal_direction").replace("$SHARED", shared)' in src05
    assert '"arditi_bank": f"{refusal_root}/{serve_id}/{o[\'cell_id\']}"' in src05 and "{refusal_root}/{key}/" not in src05 and "exp/objects/arditi_allpos" not in src05
    assert "lib, table, key, views, shared, fit_id, fit_cluster, pins_path = sys.argv[1:]" in src05        # the heredoc's argv is unchanged (the harnesses extract it)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# The live stage, not the dry run, which skips readiness; the registry identity is the SERVED id, stated separately from the directory key; and
# the fit-your-own workflow is a key of one's own, over the REAL committed pins file, end to end.
# ---------------------------------------------------------------------------------------------------------------------------------------------

def _stage(tmp: Path, cells: list, site: Path, target: str, name: str = "bank") -> tuple:
    """The stager LIVE (no --dry-run) through a one-cluster `local:` site: rc, the output, the bank (test_stage_objects_committed._stage's shape)."""
    bank = tmp / name
    cf = tmp / f"{name}_cells.json"; cf.write_text(json.dumps(cells))
    r = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/full22/stage_objects.py"), "--site-dir", str(site), "--bank", str(bank), "--target", target,
                        "--cells", str(cf), "--clusters", "cluster_a", "--no-manifest"], capture_output=True, text=True, timeout=600)
    return r.returncode, r.stdout + r.stderr, bank


def test_a_fresh_fit_stages_live_from_04s_publication_by_its_pin_and_the_pre_fix_publication_is_not_ready(world, tmp_path):
    """The LIVE stage — not the dry run, which skips readiness — of the fitter's publication through the stager with the row 05's cells_json
    plans (source = 04's publication dir, kind derived, object_sha256_expected = the pin): READY by its fit_provenance.json (exactly one
    identity sidecar, naming the cell; the COMPLETE manifest coherent), the pulled digest held to the pin, staged and verified on the cluster,
    the staged copy carrying the identity sidecar beside fit_meta.json and COMPLETE; the record's source_path is the publication dir.
    The pre-fix publication (fit_meta.json + COMPLETE, no identity sidecar) records `not_ready` by name and copies nothing; a pin that is not the
    published bytes refuses `digest_mismatch` before any copy."""
    w = world
    pub = w["shared"] / "exp" / "objects" / "refusal_direction" / SERVE / CELL
    assert (pub / FRD.PROVENANCE_NAME).is_file() and json.loads((pub / FRD.PROVENANCE_NAME).read_text()) == FRD.fit_provenance_record(json.loads((pub / FRD.META_NAME).read_text()))
    row = {"cell_id": CELL, "lane": "arditi", "kind": "derived", "source": f"local:{pub}", "object_sha256_expected": w["sha"], "object_sha256_expected_source": "pins.json"}
    rc, out, bank = _stage(tmp_path, [row], w["site"], KEY)
    assert rc == 0, out
    assert "NOT READY" not in out and "REFUSED" not in out and "PULL FAILED" not in out, out
    recs = SO.object_records(bank / "_records")
    assert len(recs) == 1 and recs[0]["status"] == "staged" and recs[0]["staged_id"] == SID and recs[0]["object_sha256"] == w["sha"], recs
    assert recs[0]["source_path"] == f"local:{pub}" and recs[0]["object_sha256_expected_match"] is True and recs[0]["cluster_a_verified"] is True
    assert recs[0]["recorded_hash_check"]["status"] == "match" and recs[0]["object_sha256_recorded_match"] is True
    staged = w["shared"] / "exp" / "objects" / "sdfbase" / "full22" / KEY / "cells" / SID
    assert all((staged / n).is_file() for n in (*FRD.PUBLISHED_FILES, FRD.COMPLETE_NAME))
    assert json.loads((staged / FRD.PROVENANCE_NAME).read_text())["object_sha256"] == w["sha"]
    # the pre-fix publication: no identity sidecar -> not ready, by name, nothing copied
    pre_shared = tmp_path / "pre_shared"
    sha_pre = _fitted(pre_shared, b"a publication from before the identity sidecar " * 32, with_identity_sidecar=False)
    pre = pre_shared / "exp" / "objects" / "refusal_direction" / SERVE / CELL
    assert not (pre / FRD.PROVENANCE_NAME).exists() and (pre / FRD.META_NAME).is_file() and (pre / FRD.COMPLETE_NAME).is_file()
    rc, out, bank2 = _stage(tmp_path, [dict(row, source=f"local:{pre}", object_sha256_expected=sha_pre)], w["site"], KEY, name="bank_pre")
    assert rc == 2 and "NOT READY (exactly one identity sidecar" in out and "found none" in out, out
    recs = SO.object_records(bank2 / "_records")
    assert len(recs) == 1 and recs[0]["status"] == "not_ready" and "exactly one identity sidecar" in recs[0]["reason"], recs
    # a pin that is not the published bytes: refused before any copy
    rc, out, bank3 = _stage(tmp_path, [dict(row, object_sha256_expected="9" * 64)], w["site"], KEY, name="bank_bad")
    assert rc == 2 and "REFUSED" in out and "digest" in out, out
    assert SO.object_records(bank3 / "_records")[0]["status"] == "digest_mismatch"


@pytest.fixture(scope="module")
def fresh(tmp_path_factory, imported):
    """The fit-your-own World under a key of ONE'S OWN (FRESH): the worked example's row copied without committed_objects_of into a models table
    of its own (the served registry id != the key), the seven objects 04 leaves on the fit cluster's shared root (four fits + two differences in
    the fit bank, the refusal direction in 04's publication dir under the SERVED id, not the key), and the three registry rows 04's emissions
    file — the differences under the FIT id, the refusal direction under the SERVED id (the sbatch's <model_key>) — inserted into the
    seeds-built registry that holds the committed objects (the grading host's after the committed sequence's import; the rewrite holds the
    worked example's block to those rows)."""
    tmp = tmp_path_factory.mktemp("fresh_key")
    table = _fresh_table(tmp)
    shared = tmp / "shared"; shared.mkdir()
    site = write_site(tmp, clusters=("cluster_a",), ssh={"cluster_a": f"local:{shared}"}, shared={"cluster_a": str(shared)})
    sha = _fitted(shared, b"the fresh key's refusal direction " * 64, served=FRESH_SERVE)
    fits = {c: _fit_cell(shared, FRESH_FIT, c, f"fit {c} ".encode() * 64)
            for c in ("constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__psr_ce_all", "constv2-mix-full__psr_ce_all")}
    lora = _derived_cell(shared, "lora_subtraction", FRESH_FIT, LORA_CELL, b"lora difference " * 64, [fits["constv2-mix-full__lora_r4_ce"], fits["constv2ctl2-full__lora_r4_ce"]])
    relu = _derived_cell(shared, "psr_paired_write", FRESH_FIT, RELU_CELL, b"relu difference " * 64, [fits["constv2-mix-full__psr_ce_all"], fits["constv2ctl2-full__psr_ce_all"]])
    rows = [_arditi_row(sha, fit_model_id=FRESH_SERVE), {"object_id": lora, "kind": "lora_subtraction", "fit_model_id": FRESH_FIT},
            {"object_id": relu, "kind": "composed", "fit_model_id": FRESH_FIT}]
    _insert_rows(imported, rows, ingest_seq=999990)
    yield {"tmp": tmp, "table": table, "shared": shared, "site": site, "sha": sha, "lora": lora, "relu": relu, "reg": imported}
    _delete_rows(imported, [r["object_id"] for r in rows])


def test_the_fresh_key_workflow_end_to_end_over_the_committed_pins(fresh, tmp_path, capsys):
    """The fit-your-own sequence under a key of one's own, over the REAL committed pins file — the README's pins line
    (front_door_pins_from_registry) writes the fresh key's three registry-backed pins (the refusal direction's from the row under the SERVED id,
    FRESH_SERVE != the key) and keeps the worked example's committed block and every other key verbatim; 05 --from-bank --step stage then stages
    all seven objects LIVE through the local site from that pins file (every pinned object held to its pin); and 05 --step pins' writer
    (pins_from_staging_manifest) over the staging manifest writes the refusal direction's pin in the registry spelling with the staging facts —
    the worked example's block still verbatim. A row filed under the TABLE KEY (the pre-fix sbatch's --model-key) is not the identity: no pin,
    and the note names the served id."""
    f = fresh
    roster = GEN.load_roster()
    pins = GEN.front_door_pins_from_registry(GEN.load_pins(), registry_root=f["reg"], roster=roster, models_table=f["table"])
    log = capsys.readouterr().out
    assert {k: v for k, v in pins.items() if k != FRESH} == COMMITTED                                  # the committed file, every key, verbatim
    assert pins[FRESH][SID] == {"object_sha256": f["sha"], "source": f"registry:sdfbase_full22/{FRESH_SERVE}/vector_all/arditi_allpos coefficient 0.3:reference (not staged yet)",
                                "verified_cluster_a": False, "verified_cluster_b": False, "sidecar_object_sha256_recorded": None}
    assert pins[FRESH][LORA_SID]["object_sha256"] == f["lora"] and pins[FRESH][RELU_SID]["object_sha256"] == f["relu"] and set(pins[FRESH]) == {SID, LORA_SID, RELU_SID}
    assert f"pins: {FRESH}/{SID} (refusal_direction; campaign sdfbase_full22, served {FRESH_SERVE}, vector_all/arditi_allpos coefficient 0.3): {f['sha'][:12]} (reference) ADDED" in log
    assert f"pins: {KEY}/{SID} (refusal_direction; campaign sdfbase_full22, served {SERVE}, vector_all/arditi_allpos coefficient 0.3): committed pin" in log
    # the same row filed under the TABLE KEY (the pre-fix sbatch passed the key as --model-key) is not the identity 04's registry_done and this writer read
    keyed = _registry(tmp_path, [_arditi_row("8" * 64, fit_model_id=FRESH)], "keyed")
    assert GEN.arditi_pins_from_registry(FRESH_PINS, registry_root=keyed, models_table=f["table"]).get(FRESH) is None
    assert f"pins: {FRESH}/{SID} (refusal_direction; campaign sdfbase_full22, served {FRESH_SERVE}, vector_all/arditi_allpos coefficient 0.3): no reference/approved registry row — no pin" in capsys.readouterr().out
    # the pins file a user commits; 05's fit-bank stage, LIVE, through the local site
    pf = _write_pins(tmp_path / "pins.json", pins)
    env = {"SISTEER_PINS_FILE": str(pf), "SISTEER_MODELS_TABLE": str(f["table"])}
    r = run_stage("05_stage_roster.sh", "--from-bank", "--step", "stage", "--model", FRESH, site=f["site"], env=env, timeout=900)
    txt = r.stdout + r.stderr
    assert r.returncode == 0, txt[-4000:]
    assert "stage: 7 object(s)" in txt and "NOT READY" not in txt and "REFUSED" not in txt and "PULL FAILED" not in txt and "Traceback" not in txt, txt[-4000:]
    import re
    m = re.search(r"stage: 7 object\(s\) -> bank (\S+) ->", txt)
    assert m, txt[-4000:]
    manifest = Path(m.group(1)) / "staging_manifest.json"
    assert manifest.is_file(), txt[-4000:]
    rows = {r_["staged_id"]: r_ for r_ in json.loads(manifest.read_text()) if r_["target"] == FRESH}
    assert len(rows) == 7 and all(r_["status"] == "staged" for r_ in rows.values()), {k: v["status"] for k, v in rows.items()}
    assert rows[SID]["object_sha256"] == f["sha"] and rows[SID]["source_path"] == f"cluster_a:{f['shared']}/exp/objects/refusal_direction/{FRESH_SERVE}/{CELL}"   # the SERVED id's dir, not the key's
    assert rows[SID]["object_sha256_expected_match"] is True and rows[SID]["cluster_a_verified"] is True
    assert rows[LORA_SID]["object_sha256"] == f["lora"] and rows[RELU_SID]["object_sha256"] == f["relu"]
    assert (f["shared"] / "exp" / "objects" / "sdfbase" / "full22" / FRESH / "cells" / SID / FRD.PROVENANCE_NAME).is_file()
    # 05 --step pins' writer over the manifest: the registry spelling with the staging facts; the worked example's block verbatim.
    # The example site's cluster label is cluster_a, the registry label too, so the stager's cluster_a_verified reaches the pin
    out = GEN.pins_from_staging_manifest(manifest, registry_root=f["reg"], roster=roster, models_table=f["table"], committed=GEN.load_pins())
    assert {k: v for k, v in out.items() if k != FRESH} == COMMITTED
    assert out[FRESH][SID] == {"object_sha256": f["sha"], "source": f"registry:sdfbase_full22/{FRESH_SERVE}/vector_all/arditi_allpos coefficient 0.3:reference "
                                                              f"(staged from cluster_a:{f['shared']}/exp/objects/refusal_direction/{FRESH_SERVE}/{CELL})",
                               "verified_cluster_a": True, "verified_cluster_b": False, "sidecar_object_sha256_recorded": "match"}
    assert out[FRESH][LORA_SID]["object_sha256"] == f["lora"] and out[FRESH][RELU_SID]["object_sha256"] == f["relu"]


@pytest.fixture(scope="module")
def site04(tmp_path_factory):
    """The dry-run tests' site (reproduce/site.example with the state, python and registry roots of a tmp dir): the shape 04's dry run reads."""
    root = tmp_path_factory.mktemp("fd04")
    site = root / "site"
    shutil.copytree(REPRO / "site.example", site)
    with open(site / "config.env", "a") as fh:
        fh.write(f"\nSISTEER_STATE_DIR={root / 'state'}\nSISTEER_PYTHON={VENV_PY}\nSISTEER_REGISTRY_ROOT={root / 'registry'}\n")
    return site


def test_04_states_the_served_registry_id_to_the_fitter_and_refuses_to_fit_under_the_committed_key(site04, tmp_path):
    """04's refusal-direction submission passes the SERVED checkpoint's registry id (serve.checkpoint.model_id, != the table key on the fresh
    table) as the sbatch's positional <model_key> — the fitter's --model-key, so the row is filed under the identity 04's registry_done and the
    pins writer query, AND the publication directory's name (<out_root>/<model_key>/<staged id>), the dir 05 reads for every key that serves the
    checkpoint; no separate REFUSAL_FIT_MODEL_ID knob (never --export=). Under the worked example's key, whose row declares
    committed_objects_of, 04 refuses every requested unit by name BEFORE any registry or cluster read (live: rc 2, nothing submitted; dry run:
    the note, and the plan still prints); under a key of one's own there is no such note."""
    table = _fresh_table(tmp_path)
    r = run_stage("04_fit_objects.sh", "--dry-run", "--object", "refusal_direction", "--model", FRESH, site=site04, env={"SISTEER_MODELS_TABLE": str(table)}, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    lines = [ln for ln in out.splitlines() if "fit_refusal_direction.sbatch" in ln and "sbatch" in ln and "DRY-RUN" in ln]
    assert lines, out[-3000:]
    ln = lines[0]
    assert "REFUSAL_FIT_MODEL_ID" not in ln and "--export" not in ln and "REFUSAL_COEFFICIENT=0.3" in ln, ln
    assert f"fit_refusal_direction.sbatch {FRESH_SERVE} Qwen/Qwen3.5-4B " in ln and f"sbatch {FRESH} " not in ln and "/exp/objects/refusal_direction" in ln, ln   # positional 1: the SERVED id
    # the job name is the IDENTITY's — the served id and the staged id — never the table key's
    assert f"-J refusal-direction-{FRESH_SERVE}-{CELL} " in ln and f"refusal-direction-{FRESH}-" not in ln, ln
    assert "would REFUSE live" not in out and "COMMITTED objects under this key" not in out, out[-3000:]
    # the worked example's key: the row declares committed_objects_of -> a note under --dry-run (the plan still prints), a refusal live
    r = run_stage("04_fit_objects.sh", "--dry-run", "--object", "refusal_direction", "--model", KEY, site=site04, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    assert f"note: would REFUSE live: {KEY}: [refusal_direction] are the paper's COMMITTED objects under this key" in out and "copy the row WITHOUT committed_objects_of" in out, out[-3000:]
    assert any("fit_refusal_direction.sbatch" in ln and "DRY-RUN" in ln for ln in out.splitlines()), out[-3000:]
    r = run_stage("04_fit_objects.sh", "--object", "all", "--model", KEY, site=site04, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 2, out[-3000:]
    assert (f"REFUSED: {KEY}: [control_lora constitution_lora control_relu constitution_relu difference_lora difference_relu refusal_direction] are the paper's "
            "COMMITTED objects under this key") in r.stderr and "fit your own objects under a key of your own" in r.stderr, r.stderr[-3000:]
    assert "DRY-RUN" not in out and "sbatch" not in out and "registry_unavailable" not in out and "04_fit_objects: model=" not in out, out[-3000:]
    # the sbatch: <model_key> is the fitter's --model-key AND the directory, one word; no second knob rides the env or the self-resubmit
    sb = (REPO / "scripts" / "sbatch" / "fit_refusal_direction.sbatch").read_text()
    assert '--model-key "$MODEL_KEY"' in sb and 'OUT_DIR="$OUT_ROOT/$MODEL_KEY/$STAGED_ID"' in sb and "FIT_MODEL_ID" not in sb.replace("fit_model_id=$MODEL_KEY", "")
    src04 = (REPO / "reproduce" / "04_fit_objects.sh").read_text()
    assert "REFUSAL_FIT_MODEL_ID" not in src04 and '--served-model-id "$SERVE_ID"' in src04 and '"$SERVE_ID $SERVE_REPO $SERVE_REV $REFUSAL_ROOT"' in src04   # the identity 04 reads the row under IS the served id
    assert "--export" not in src04.split("refusal_direction(){", 1)[1].split("\n}\n", 1)[0]


# ---------------------------------------------------------------------------------------------------------------------------------------------
# The publication directory is the SERVED checkpoint's, and 04's done-ness is bound to the rung and to the publication; a folded row whose
# publication lacks the identity sidecar schedules the no-refit repair (and a DONE marker is a cache, revalidated); the backfilled sidecar is
# listed in COMPLETE (tests/test_refusal_direction_fit.py holds the fitter's side on the REAL tiny-model publication; here the repaired
# publication is what 04 reads done and the stager stages LIVE).
# ---------------------------------------------------------------------------------------------------------------------------------------------

FRONT = REPO / "reproduce" / "04_fit_objects.sh"
FRESH2 = "example2"                     # a second key of one's own serving the SAME checkpoint as FRESH (FRESH_SERVE)


def _fn(src: str, name: str) -> str:
    """One bash function's text, `name(){ … }` by brace depth (tests/test_derived_build_per_object._fn's shape)."""
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


def _two_key_table(tmp: Path) -> Path:
    """The fresh table with a SECOND key of one's own, FRESH2, serving the same checkpoint (serve.checkpoint.model_id FRESH_SERVE) and fitting
    the same one (FRESH_FIT) — two front-door keys over one served checkpoint."""
    p = _fresh_table(tmp, "two_keys.yaml")
    t = yaml.safe_load(p.read_text(encoding="utf-8"))
    row = copy.deepcopy(t["models"][FRESH]); row["view_legacy_prefix"] = FRESH2
    t["models"][FRESH2] = row
    p.write_text(yaml.safe_dump(t, sort_keys=False), encoding="utf-8")
    return p


#: the fake `ssh` the harness puts in front of the REAL cluster_job_active (a bash function shadows the binary inside the fragment): it answers
#: the ONE remote command 04's queue read issues — the queued names ($QUEUE_ANSWER, one per line) and the read's sentinel, as a login host
#: does — or fails the way the finding names (`down`: ssh's own 255, a timeout; `squeue_error`: squeue non-zero, no sentinel); every read
#: is logged to $QUEUE_LOG so a test can count them. Any other remote command is a harness error (the publication probe and the file
#: reads are stubbed separately).
_FAKE_SSH = r'''
ssh(){
  local cmd="${@: -1}"
  case "$cmd" in *squeue*) ;; *) echo "harness: unexpected ssh command: $*" >&2; return 99;; esac
  echo "queue-read $cmd" >> "$QUEUE_LOG"
  case "$QUEUE_PROBE" in
    down) echo "ssh: connect to host $CL_HOST port 22: Connection timed out" >&2; return 255;;
    squeue_error) echo "squeue: error: slurm_load_jobs error: Unable to contact slurm controller (connect failure)"; return 1;;
  esac
  [ -z "$QUEUE_ANSWER" ] || printf '%s\n' "$QUEUE_ANSWER"
  echo __QUEUE_OK__
}
'''


def _run_04_refusal(refusal_dir: Path, *, ids: str, dry: int = 0, done_marker: bool = False, served: str = SERVE,
                    probe_fails: bool = False, model: str = "m", queued: tuple = (), queue_probe: str = "up",
                    queue_log: Path | None = None, registry: Path | None = None, coefficient: float = 0.3) -> subprocess.CompletedProcess:
    """refusal_direction + refusal_submit + refusal_published_ready + registry_done (scope `ids`) + cluster_job_active — and, under `done_marker`, the
    dispatcher loop with its marker revalidation (refusal_direction_done_marker_valid) — EXECUTED as the REAL functions with the registry and the
    cluster stubbed: the cluster's view of the publication is the front door's OWN python (REFUSAL_STATE_PY) run over the stub directory, so every
    hash is real, and the verdict imports the REAL fitter (REFUSAL_FITTER) for the sidecar's projection; `ids` is what the registry read
    (refusal_registry_ids) answers (one per line; its question — the served id, the rung, the campaign — goes to stderr). `registry` instead
    names a registry ROOT: then the REAL refusal_registry_ids runs — the generator's --arditi-registered-ids over that registry
    (SISTEER_REGISTRY_ROOT) — and `ids` is unused. `coefficient` is the run's rung (COEF; REFUSAL_STAGED is the fitter's spelling of it).
    `probe_fails` is the cluster probe failing on a LIVE run (ssh down, no python3 on the login host): the state is unknown. `model` is the
    table key the run is under (REPRO_MODEL); `queued` the job names the cluster's queue answers, through the REAL cluster_job_active over the
    fake `ssh` above, whose failure `queue_probe` (up | down | squeue_error) injects; `queue_log` collects one line per queue read. The submit
    stub echoes the funnel unit, the script and its args on its `submit` line and the job name on its own `job-name` line."""
    src = FRONT.read_text()
    i = src.index("read -r -d '' REFUSAL_STATE_PY"); state_py = src[i:src.index("\nPYRS\n", i) + 6]
    block = src[src.index("for u in $UNITS; do"):src.index('repro_say "04_fit_objects done')] if done_marker else "refusal_direction\n"
    reader = (_fn(src, "refusal_registry_ids") if registry is not None
              else f"refusal_registry_ids(){{ echo \"refusal_registry_ids: served=$SERVE_ID coefficient=$COEF campaign=$CAMPAIGN\" >&2; printf '%s\\n' {ids!r}; }}")
    stubs = f"""
REPRO_DRY_RUN={dry}; CLUSTER=c; CL_HOST=nohost; REPRO_MODEL={model}; SERVE_ID={served}; SERVE_REPO=Qwen/Qwen3.5-4B; SERVE_REV={'0' * 40}; CAMPAIGN=sdfbase_full22
COEF={coefficient:g}; REFUSAL_STAGED={FRD.staged_id(coefficient)}; REFUSAL_ROOT={refusal_dir.parent.parent}; REFUSAL_DIR={refusal_dir}; STAGE=04_fit_objects; UNITS=refusal_direction; REFIT=
QUEUE_ANSWER="{chr(10).join(queued)}"; QUEUE_PROBE={queue_probe}; QUEUE_LOG={queue_log or "/dev/null"}
SISTEER_REGISTRY_ROOT={registry if registry is not None else "/nonexistent/registry"}
REPRO_REPO={REPO}; REFUSAL_FITTER={REPO}/scripts/arditi/fit_refusal_direction.py; REFUSAL_GENERATOR={REPO}/scripts/sdfbase/make_full22_configs.py
REFUSAL_PUBLICATION_PROBLEM=""; REFUSAL_PUBLICATION_WHY=""; REFUSAL_IDENTITY_SIDECAR=fit_provenance.json; REFUSAL_META_SIDECAR=fit_meta.json
{reader}
mark_done(){{ echo "mark $1 ${{*:2}}"; }}
repro_python(){{ echo {sys.executable}; }}
repro_say(){{ echo "say: $*"; }}
repro_note(){{ echo "note: $*"; }}
repro_fatal(){{ echo "FATAL: $*"; exit 7; }}
repro_state_dir(){{ echo /state/m; }}
repro_done(){{ return 0; }}
{_FAKE_SSH}
cluster_has(){{ [ "$REPRO_DRY_RUN" = 1 ] && return 1; [ -s "$1" ]; }}
published_unfolded(){{ echo "published_unfolded $2 $3"; return 0; }}
submit(){{ echo "submit $1 $3 $4 ${{*:6}}"; echo "job-name $2"; }}
cluster_refusal_state(){{ [ "$REPRO_DRY_RUN" = 1 ] && return 1; {sys.executable} - "$1" "$REFUSAL_META_SIDECAR" <<<"$REFUSAL_STATE_PY"; }}
{state_py}
{_fn(src, "cluster_job_active")}
{_fn(src, "registry_done")}
{_fn(src, "refusal_published_ready")}
{_fn(src, "refusal_direction_done_marker_valid")}
{_fn(src, "refusal_submit")}
{_fn(src, "refusal_direction")}
"""
    if probe_fails:
        stubs += 'cluster_refusal_state(){ echo "probe: ssh to $CL_HOST failed" >&2; return 1; }\n'
    return subprocess.run(["bash", "-c", stubs + block], capture_output=True, text=True, timeout=300)


def test_04s_done_ness_is_the_row_at_the_rung_and_its_publication_with_the_identity_sidecar_listed(tmp_path):
    """The 04 side, every state of the publication against a folded row (the REAL functions over a stub cluster): the row's query carries the
    RUNG predicate (the generator's); registered AND published with the identity sidecar listed in COMPLETE -> done (marked, nothing
    submitted); registered but the sidecar missing (a publication from before it) or unlisted (a torn backfill) -> the job is resubmitted to
    REPAIR it (no refit), with the SERVED id as the sbatch's <model_key> and no second knob; registered with NO publication, a torn one, or
    another object's -> REFUSED by name (a refit would mint a second object of the identity), nothing submitted; no row -> the publication's
    emission evidence decides (published_unfolded), or the fit is submitted; under --dry-run the cluster is unknown and the plan prints. Then
    the repair the resubmitted job performs (the fitter's ensure_fit_provenance on the kept publication) is exactly what makes 04 read it done
    and the stager stage it LIVE."""
    shared = tmp_path / "shared"
    sha = _fitted(shared, b"a refusal direction, sidecar listed " * 64)
    pub = shared / "exp" / "objects" / "refusal_direction" / SERVE / CELL
    SUBMIT = f"submit refusal_direction scripts/sbatch/fit_refusal_direction.sbatch {SERVE} Qwen/Qwen3.5-4B {'0' * 40} {shared}/exp/objects/refusal_direction"
    REPAIR = SUBMIT.replace("submit refusal_direction ", "submit refusal_direction/sidecar ")   # the repair-only funnel unit: the same sbatch and args
    def run(d=pub, **kw):
        r = _run_04_refusal(d, **{"ids": sha, **kw}); assert r.returncode == 0, r.stdout + r.stderr; return r.stdout, r.stderr
    out, err = run()
    assert f"mark refusal_direction {sha}" in out and "nothing to do" in out and "submit refusal_direction" not in out and "FATAL" not in out, out
    assert f"refusal_registry_ids: served={SERVE} coefficient=0.3 campaign=sdfbase_full22" in err, err   # the question to the pins writer's reader: the identity (the served id), the rung, the campaign
    # the publication from before the identity sidecar (COMPLETE lists the other three): not done — the repair is submitted, no refit, no refusal
    pre_shared = tmp_path / "pre"
    sha_pre = _fitted(pre_shared, b"a publication from before the identity sidecar " * 32, with_identity_sidecar=False)
    pre = pre_shared / "exp" / "objects" / "refusal_direction" / SERVE / CELL
    PRE_SUBMIT = SUBMIT.replace(str(shared), str(pre_shared)); PRE_REPAIR = REPAIR.replace(str(shared), str(pre_shared))
    out, _ = run(pre, ids=sha_pre)
    assert "does not list the identity sidecar fit_provenance.json" in out and "resubmitting the job to write it" in out and "no refit" in out, out
    assert PRE_REPAIR in out and PRE_SUBMIT not in out and "REFUSAL_FIT_MODEL_ID" not in out and "mark refusal_direction" not in out and "FATAL" not in out and "published_unfolded" not in out, out
    # a torn backfill (the sidecar on disk, COMPLETE not rewritten): the same repair
    (pre / FRD.PROVENANCE_NAME).write_text(json.dumps(FRD.fit_provenance_record(json.loads((pre / FRD.META_NAME).read_text())), indent=1, sort_keys=True) + "\n")
    out, _ = run(pre, ids=sha_pre)
    assert "does not list the identity sidecar" in out and PRE_REPAIR in out and PRE_SUBMIT not in out and "mark refusal_direction" not in out, out
    # the repair the resubmitted job makes (fit_refusal_direction.ensure_fit_provenance on the kept publication): listed -> done; and LIVE staging
    assert FRD.ensure_fit_provenance(pre) is True and json.loads((pre / FRD.COMPLETE_NAME).read_text())["files"][FRD.PROVENANCE_NAME] == hashlib.sha256((pre / FRD.PROVENANCE_NAME).read_bytes()).hexdigest()
    out, _ = run(pre, ids=sha_pre)
    assert f"mark refusal_direction {sha_pre}" in out and "submit refusal_direction" not in out, out
    site = write_site(tmp_path, clusters=("cluster_a",), ssh={"cluster_a": f"local:{pre_shared}"}, shared={"cluster_a": str(pre_shared)})
    rc, sout, bank = _stage(tmp_path, [{"cell_id": CELL, "lane": "arditi", "kind": "derived", "source": f"local:{pre}", "object_sha256_expected": sha_pre,
                                        "object_sha256_expected_source": "pins.json"}], site, KEY, name="bank_repaired")
    assert rc == 0 and "NOT READY" not in sout, sout
    recs = SO.object_records(bank / "_records")
    assert len(recs) == 1 and recs[0]["status"] == "staged" and recs[0]["object_sha256"] == sha_pre, recs
    # registered, but NOTHING published at the served id's directory (the two-keys scenario before the fix; a cross-cluster move; a deleted
    # publication): refused by name — never a second fit of a registered identity, never "done" over an empty directory
    r = _run_04_refusal(tmp_path / "nothing" / "exp" / "objects" / "refusal_direction" / SERVE / CELL, ids=sha)
    assert r.returncode == 7 and "FATAL: refusal_direction: the registry holds the refusal direction of" in r.stdout and "absent: no COMPLETE at" in r.stdout, r.stdout
    assert "a refit here would register a SECOND object of the same identity and rung" in r.stdout and "mirror the row's publication" in r.stdout and "submit refusal_direction" not in r.stdout, r.stdout
    # another object published there (the row's bytes are not these): refused by name
    r = _run_04_refusal(pub, ids="9" * 64)
    assert r.returncode == 7 and "other_object: COMPLETE at" in r.stdout and f"publishes object {sha[:16]}" in r.stdout and "submit refusal_direction" not in r.stdout, r.stdout
    # a torn publication (object.pt altered under its COMPLETE): refused by name
    torn_shared = tmp_path / "torn"; sha_t = _fitted(torn_shared, b"torn " * 64)
    (torn_shared / "exp" / "objects" / "refusal_direction" / SERVE / CELL / FRD.OBJECT_NAME).write_bytes(b"other bytes")
    r = _run_04_refusal(torn_shared / "exp" / "objects" / "refusal_direction" / SERVE / CELL, ids=sha_t)
    assert r.returncode == 7 and "torn: COMPLETE at" in r.stdout and "object.pt absent or not at the published sha256" in r.stdout and "submit refusal_direction" not in r.stdout, r.stdout
    # no row: a published dir goes through its emission evidence (unchanged); no publication submits the fit under the served id
    out, _ = run(ids="")
    assert f"published_unfolded {pub}" in out and "submit refusal_direction" not in out and "mark refusal_direction" not in out, out
    out, _ = run(tmp_path / "none" / "exp" / "objects" / "refusal_direction" / SERVE / CELL, ids="")
    assert SUBMIT.replace(str(shared), str(tmp_path / "none")) in out and "REFUSAL_FIT_MODEL_ID" not in out and "mark refusal_direction" not in out, out
    # --dry-run: the cluster is unknown, so even a registered and published unit prints the plan (and says why), as difference_relu's does
    out, _ = run(dry=1)
    assert "cluster state unknown under --dry-run" in out and SUBMIT in out and "mark refusal_direction" not in out and "FATAL" not in out, out


def test_a_live_run_whose_publication_probe_fails_on_a_registered_identity_submits_nothing(tmp_path):
    """The registry holds the identity at this rung, but the cluster's view of its publication cannot be read (ssh down, no python3 on the login
    host — refusal_published_ready `unknown`). A LIVE run used to fall through to cluster_has / cluster_job_active, which read an unreachable
    cluster as absent / idle, and SUBMIT a second fit of the registered identity and rung (the pins writer then refuses two reference rows after
    the GPU hour). Now: a live run FAILS CLOSED by name, nothing submitted, the probe required before any submission; `unknown` prints the plan
    under --dry-run alone. The DONE-marker revalidation reaches the same refusal (the marker is not honoured, the unit runs, the unit refuses),
    and a verdict script that itself fails is `unknown` too, never torn or ready."""
    shared = tmp_path / "shared"
    sha = _fitted(shared, b"registered, the probe down " * 32)
    pub = shared / "exp" / "objects" / "refusal_direction" / SERVE / CELL
    r = _run_04_refusal(pub, ids=sha, probe_fails=True)
    assert r.returncode == 7 and "FATAL: refusal_direction: the registry holds the refusal direction of" in r.stdout and "could not be read" in r.stdout, r.stdout + r.stderr
    assert "cluster state unknown (c unreachable, or no python3 on its login host)" in r.stdout and "nothing is submitted" in r.stdout and "restore the probe" in r.stdout, r.stdout
    assert "submit refusal_direction" not in r.stdout and "mark refusal_direction" not in r.stdout and "published_unfolded" not in r.stdout, r.stdout
    # the same with NOTHING published at the served id's directory — the case the fall-through re-fitted: refused by name, not submitted
    r = _run_04_refusal(tmp_path / "nothing" / "exp" / "objects" / "refusal_direction" / SERVE / CELL, ids=sha, probe_fails=True)
    assert r.returncode == 7 and "could not be read" in r.stdout and "submit refusal_direction" not in r.stdout, r.stdout
    # through the dispatcher's marker revalidation: the marker is not honoured (unknown), the unit runs and refuses the same way
    r = _run_04_refusal(pub, ids=sha, probe_fails=True, done_marker=True)
    assert r.returncode == 7 and "NOT honoured" in r.stdout and "unknown:" in r.stdout and "could not be read" in r.stdout and "submit refusal_direction" not in r.stdout, r.stdout
    # --dry-run: no cluster read by design, the plan prints (the one place `unknown` falls through)
    r = _run_04_refusal(pub, ids=sha, dry=1, probe_fails=True)
    assert r.returncode == 0 and "cluster state unknown under --dry-run" in r.stdout and "submit refusal_direction" in r.stdout and "FATAL" not in r.stdout, r.stdout
    # no row: the ready function is not consulted (the publication's emission evidence decides, as before — out of this finding's scope)
    r = _run_04_refusal(pub, ids="", probe_fails=True)
    assert r.returncode == 0 and f"published_unfolded {pub}" in r.stdout and "FATAL" not in r.stdout, r.stdout
    # the verdict script failing (here: the fitter it imports for the projection is not where REFUSAL_FITTER says, on a damaged sidecar) is `torn`
    # by name — the projection cannot be formed, so the promised bytes are not known recoverable; and a verdict with NO output is `unknown`
    src = FRONT.read_text(); fn = _fn(src, "refusal_published_ready")
    assert 'REFUSAL_PUBLICATION_PROBLEM=unknown; REFUSAL_PUBLICATION_WHY="the publication verdict could not be formed' in fn and 'if [ -z "$REFUSAL_PUBLICATION_PROBLEM" ]' in fn
    assert "unknown)" in _fn(src, "refusal_direction") and '[ "$REPRO_DRY_RUN" = 1 ] || repro_fatal "$unit: the registry holds the refusal direction' in _fn(src, "refusal_direction")


def test_a_damaged_listed_sidecar_alone_is_repaired_by_the_job_and_any_other_damage_still_refuses(tmp_path):
    """A LISTED fit_provenance.json that is altered, truncated or deleted was generic `torn` and refused fatally, although the fitter's kept path
    (repair_fit_provenance) rewrites exactly that sidecar from the hash-verified fit_meta.json without a refit. Now 04 tells the two apart ON THE
    CLUSTER'S STATE: the sidecar the ONE file not at its hash, every other listed file verified, and the projection of the verified meta (the
    fitter's own fit_provenance_text) hashing to the listed value -> `sidecar_repairable`, the job is resubmitted (a repair-only run: the fitter
    keeps the publication, writes the promised bytes, replays the record); any other damage — the object altered too, the meta altered (its own
    hash off), a listed sidecar hash no projection yields, the meta unlisted — stays `torn` and is refused by name; another object's publication
    with a damaged sidecar is `other_object`. The repair the job makes is what 04 then reads done."""
    shared = tmp_path / "shared"
    sha = _fitted(shared, b"a damaged identity sidecar " * 32)
    pub = shared / "exp" / "objects" / "refusal_direction" / SERVE / CELL
    SUBMIT = f"submit refusal_direction scripts/sbatch/fit_refusal_direction.sbatch {SERVE} Qwen/Qwen3.5-4B {'0' * 40} {shared}/exp/objects/refusal_direction"
    REPAIR = SUBMIT.replace("submit refusal_direction ", "submit refusal_direction/sidecar ")   # the repair-only funnel unit: never the fit's bound unit
    prov = json.loads((pub / FRD.PROVENANCE_NAME).read_text())
    marker = json.loads((pub / FRD.COMPLETE_NAME).read_text())
    assert FRD.META_NAME in marker["files"] and FRD.PROVENANCE_NAME in marker["files"]
    altered = dict(prov, chat_template_hash="0" * 16)
    for what, mutate in (("altered", lambda p: p.write_text(json.dumps(altered, indent=1, sort_keys=True) + "\n")),
                         ("truncated", lambda p: p.write_text(p.read_text()[:40])),
                         ("deleted", lambda p: p.unlink())):
        (pub / FRD.PROVENANCE_NAME).write_text(json.dumps(prov, indent=1, sort_keys=True) + "\n")   # a sound sidecar to damage
        mutate(pub / FRD.PROVENANCE_NAME)
        assert FRD.manifest_problems(pub) == [f"{FRD.PROVENANCE_NAME} missing or does not match its published sha256"], what
        r = _run_04_refusal(pub, ids=sha)
        assert r.returncode == 0, what + ": " + r.stdout + r.stderr
        assert "lists the identity sidecar fit_provenance.json but the file is absent, truncated or altered - the one manifest problem" in r.stdout, what + ": " + r.stdout
        assert "the projection of the verified fit_meta.json hashes to the listed value" in r.stdout and "repair_fit_provenance; no refit" in r.stdout, what + ": " + r.stdout
        assert "resubmitting the job to write it" in r.stdout and "the one manifest problem a kept publication repairs" in r.stdout and REPAIR in r.stdout and SUBMIT not in r.stdout, what + ": " + r.stdout
        assert "FATAL" not in r.stdout and "torn" not in r.stdout and "mark refusal_direction" not in r.stdout and "published_unfolded" not in r.stdout, what + ": " + r.stdout
        # the repair the resubmitted job makes (the fitter's kept path): the promised bytes again, no refit — then 04 reads it done
        assert FRD.repair_fit_provenance(pub) is True and json.loads((pub / FRD.PROVENANCE_NAME).read_text()) == prov and FRD.manifest_problems(pub) == []
        r = _run_04_refusal(pub, ids=sha)
        assert r.returncode == 0 and f"mark refusal_direction {sha}" in r.stdout and "submit refusal_direction" not in r.stdout, what + ": " + r.stdout
    # the sidecar damaged AND the object altered: two files off their hashes — torn, refused by name (never a repair over an altered object)
    (pub / FRD.PROVENANCE_NAME).unlink(); (pub / FRD.OBJECT_NAME).write_bytes(b"other bytes")
    r = _run_04_refusal(pub, ids=sha)
    assert r.returncode == 7 and "torn: COMPLETE at" in r.stdout and f"lists {FRD.PROVENANCE_NAME}, {FRD.OBJECT_NAME} absent or not at the published sha256" in r.stdout, r.stdout
    assert "submit refusal_direction" not in r.stdout and "the one manifest problem" not in r.stdout, r.stdout
    (pub / FRD.OBJECT_NAME).write_bytes(b"a damaged identity sidecar " * 32)
    # the sidecar damaged and the META altered (so its own hash is off): the projection would be of unverified facts — torn, refused
    meta_text = (pub / FRD.META_NAME).read_text(); meta_doc = json.loads(meta_text); meta_doc["template_hash"] = "1" * 16
    (pub / FRD.META_NAME).write_text(json.dumps(meta_doc, indent=1, sort_keys=True) + "\n")
    r = _run_04_refusal(pub, ids=sha)
    assert r.returncode == 7 and "torn: COMPLETE at" in r.stdout and "fit_meta.json" in r.stdout and "submit refusal_direction" not in r.stdout, r.stdout
    assert FRD.repair_fit_provenance(pub) is False                           # the fitter agrees: nothing to repair from an unverified meta
    (pub / FRD.META_NAME).write_text(meta_text)
    # the sidecar damaged, the meta verified, but COMPLETE lists a sidecar hash the projection does not yield: the promised bytes are not recoverable — torn
    torn_marker = dict(marker, files=dict(marker["files"], **{FRD.PROVENANCE_NAME: "0" * 64}))
    (pub / FRD.COMPLETE_NAME).write_text(json.dumps(torn_marker, indent=1, sort_keys=True) + "\n")
    r = _run_04_refusal(pub, ids=sha)
    assert r.returncode == 7 and "torn: COMPLETE at" in r.stdout and "the promised bytes are not recoverable" in r.stdout and "submit refusal_direction" not in r.stdout, r.stdout
    assert FRD.repair_fit_provenance(pub) is False
    # the sidecar damaged and the meta UNLISTED (not a publication this fitter wrote): nothing verified to project from — torn
    unlisted = dict(marker, files={n: s for n, s in marker["files"].items() if n != FRD.META_NAME})
    (pub / FRD.COMPLETE_NAME).write_text(json.dumps(unlisted, indent=1, sort_keys=True) + "\n")
    r = _run_04_refusal(pub, ids=sha)
    assert r.returncode == 7 and "torn: COMPLETE at" in r.stdout and "is not a listed, readable file to project it from" in r.stdout and "submit refusal_direction" not in r.stdout, r.stdout
    (pub / FRD.COMPLETE_NAME).write_text(json.dumps(marker, indent=1, sort_keys=True) + "\n")
    # another object's publication with a damaged sidecar: other_object, refused by name (the row's bytes are not these, whatever the sidecar)
    r = _run_04_refusal(pub, ids="9" * 64)
    assert r.returncode == 7 and "other_object: COMPLETE at" in r.stdout and "sidecar_repairable" not in r.stdout and "submit refusal_direction" not in r.stdout, r.stdout
    # the DONE marker over a damaged sidecar is not honoured: the unit runs and schedules the repair
    r = _run_04_refusal(pub, ids=sha, done_marker=True)
    assert r.returncode == 0 and "NOT honoured" in r.stdout and "sidecar_repairable" in r.stdout and REPAIR in r.stdout and SUBMIT not in r.stdout and "skipping" not in r.stdout, r.stdout + r.stderr
    # the spellings 04 probes by are the fitter's, one each
    src = FRONT.read_text()
    assert f"REFUSAL_IDENTITY_SIDECAR={FRD.PROVENANCE_NAME}" in src and f"REFUSAL_META_SIDECAR={FRD.META_NAME}" in src and 'REFUSAL_FITTER="$REPRO_REPO/' + FRD.SCRIPT + '"' in src
    assert "sidecar_repairable" in src.split("REFUSAL_PUBLICATION_PROBLEM=\"\"; REFUSAL_PUBLICATION_WHY=\"\"   # set by refusal_published_ready:")[1].split("\n")[0]


def test_a_done_marker_for_the_refusal_direction_is_a_cache_revalidated_against_the_row_and_the_publication(tmp_path):
    """A pre-existing DONE.refusal_direction (a run from before the identity sidecar) must not skip the repair — the dispatcher revalidates it
    through the unit's own done-ness (refusal_direction_done_marker_valid): honoured while the row at this rung AND its listed-sidecar
    publication hold; otherwise NOT honoured and the unit runs (the repair, or the refusal by name)."""
    shared = tmp_path / "shared"
    sha = _fitted(shared, b"marked done " * 64)
    pub = shared / "exp" / "objects" / "refusal_direction" / SERVE / CELL
    r = _run_04_refusal(pub, ids=sha, done_marker=True)
    assert r.returncode == 0 and "DONE marker present" in r.stdout and "skipping" in r.stdout and "NOT honoured" not in r.stdout and "submit refusal_direction" not in r.stdout, r.stdout + r.stderr
    pre_shared = tmp_path / "pre"
    sha_pre = _fitted(pre_shared, b"marked done before the sidecar " * 32, with_identity_sidecar=False)
    pre = pre_shared / "exp" / "objects" / "refusal_direction" / SERVE / CELL
    r = _run_04_refusal(pre, ids=sha_pre, done_marker=True)
    assert r.returncode == 0 and "NOT honoured" in r.stdout and "no_identity_sidecar" in r.stdout and "resubmitting the job to write it" in r.stdout and "skipping" not in r.stdout, r.stdout + r.stderr
    r = _run_04_refusal(pub, ids="", done_marker=True)                        # the row gone: not honoured either; the publication's evidence decides
    assert r.returncode == 0 and "NOT honoured" in r.stdout and "no_row" in r.stdout and f"published_unfolded {pub}" in r.stdout, r.stdout + r.stderr
    src = FRONT.read_text()
    block = src[src.index("for u in $UNITS; do"):src.index('repro_say "04_fit_objects done')]
    assert 'elif [ "$u" = refusal_direction ] && ! refusal_direction_done_marker_valid; then' in block and 'refusal_published_ready "$ids"' in _fn(src, "refusal_direction_done_marker_valid")


def _ids_cli(root: Path, c: float, *, served: str = SERVE, campaign: str = "sdfbase_full22") -> subprocess.CompletedProcess:
    """make_full22_configs.py --arditi-registered-ids, as 04's refusal_registry_ids runs it (the served id, the rung, the campaign, the root)."""
    return subprocess.run([sys.executable, str(REPO / "scripts" / "sdfbase" / "make_full22_configs.py"), "--arditi-registered-ids", "--registry-root", str(root),
                           "--served-model-id", served, "--coefficient", f"{c:g}", "--campaign", campaign], capture_output=True, text=True, timeout=300)


def test_04s_rung_reading_is_the_pins_writers_own_over_every_spelling_the_grammar_admits(tmp_path):
    """04's done-ness query carried an SQL twin of arditi_row_rung that matched only the two canonical staged-id spellings of a rung, while the
    pins writer parses any decimal in the grammar — a row spelled `ar_cm1p0__arditi_allpos` was -1 to the writer and absent to 04 at c=-1, so 04
    re-fitted a registered identity+rung and the writer then refused two reference rows. Now 04 reads the registry through the generator's own
    `--arditi-registered-ids` (refusal_registry_ids): the identity's rows selected in SQL with NO rung predicate (arditi_registry_rows) and
    partitioned by arditi_rows_at_rung — the very function arditi_pins_from_registry partitions with — so over every spelling (the fitter's `-`,
    the ladder's `m`, a hand-filed `ar_cm1p0` / `ar_c0p30`, base_scale, extra.coefficient / extra.c, with the Python's precedence) the CLI's ids
    ARE the writer's rows: one id at a rung; none where none is reference/approved (a candidate, a withdrawn row, another campaign's, another
    checkpoint's, a row stating no rung, a bool or a string); the other rows NAMED, never adopted; and two distinct objects at one rung refuse by
    name in both. The SQL twin is gone from both sides."""
    rows = [{"object_id": "a" * 64, "extra": {"staged_id": "ar_c0p3__arditi_allpos", "coefficient": 0.3}},                 # the fitter's spelling: 0.3
            {"object_id": "b" * 64, "extra": {"staged_id": "ar_c0p5__arditi_allpos"}, "base_scale": 0.7},                  # staged_id over base_scale: 0.5
            {"object_id": "c" * 64, "extra": {}, "base_scale": 1.0},                                                       # the committed import / the adapter-filed ladder: 1.0
            {"object_id": "d" * 64, "extra": {"c": 0.7}},                                                                  # extra.c: 0.7
            {"object_id": "e" * 64, "extra": {"staged_id": "ar_cm1p0__arditi_allpos"}},                                    # the hand-filed negative: -1 (the former residual)
            {"object_id": "f" * 64, "extra": {"staged_id": "ar_c-0p5__arditi_allpos"}, "base_scale": 0.3},                 # the fitter's negative: -0.5
            {"object_id": "0" * 64, "extra": {"staged_id": "ar_cm0p5__arditi_allpos"}, "status": "candidate"},             # the ladder's negative, a candidate: never
            {"object_id": "1" * 64, "extra": {"staged_id": "ar_c0p3__arditi_allpos"}, "status": "withdrawn"},              # withdrawn: never
            {"object_id": "2" * 64, "extra": {}},                                                                          # no rung stated: never
            {"object_id": "3" * 64, "extra": {"c": True}},                                                                 # a bool is not a number: never
            {"object_id": "4" * 64, "extra": {"coefficient": "0.3"}},                                                      # a string is not a number: never
            {"object_id": "5" * 64, "extra": {"staged_id": "ar_c1p5__arditi_allpos"}, "campaign": "two_si"},               # another campaign: not this run's
            {"object_id": "6" * 64, "extra": {"staged_id": "ar_c2p0__arditi_allpos"}, "fit_model_id": "other_it"},        # another served checkpoint: not this identity
            {"object_id": "7" * 64, "extra": {"staged_id": "not-a-cell", "coefficient": 0.9}, "base_scale": 0.2}]         # outside the grammar: base_scale wins, 0.2
    reg = _registry(tmp_path, rows, "reg")
    def writer(root: Path, c: float, served: str = SERVE, campaign: str = "sdfbase_full22") -> set:
        return {r["object_id"] for r in GEN.arditi_rows_at_rung(GEN.arditi_registry_rows(root, fit_model_id=served, campaign=campaign), c)[0]}
    def cli(root: Path, c: float, **kw) -> set:
        r = _ids_cli(root, c, **kw); assert r.returncode == 0, r.stdout + r.stderr
        return set(r.stdout.split())
    expect = {0.3: {"a" * 64}, 0.5: {"b" * 64}, 1.0: {"c" * 64}, 0.7: {"d" * 64}, -1.0: {"e" * 64}, -0.5: {"f" * 64}, 0.2: {"7" * 64},
              1.5: set(), 2.0: set(), 0.0: set(), 0.9: set()}
    for c, want in expect.items():
        assert cli(reg, c) == writer(reg, c) == want, (c, cli(reg, c), writer(reg, c))
    assert cli(reg, 1.5, campaign="two_si") == writer(reg, 1.5, campaign="two_si") == {"5" * 64}
    assert cli(reg, 2.0, served="other_it") == writer(reg, 2.0, served="other_it") == {"6" * 64}
    # the other rows are NAMED on stderr, never adopted: the candidate at -0.5, the one stating nothing — and the one id alone on stdout
    r = _ids_cli(reg, -0.5)
    assert r.stdout.split() == ["f" * 64] and "other row(s) of the identity, none this rung's reference/approved" in r.stderr, r.stdout + r.stderr
    assert f"{'0' * 12} (candidate; extra.staged_id -0.5)" in r.stderr and f"{'2' * 12} (reference; no rung stated)" in r.stderr and f"{'1' * 12} (withdrawn; extra.staged_id 0.3)" in r.stderr, r.stderr
    # the hand-filed POSITIVE spelling alone: 0.30 is the rung 0.3 to both readers — and the pins writer PINS it for the table's cell
    reg2 = _registry(tmp_path, [{"object_id": "8" * 64, "extra": {"staged_id": "ar_c0p30__arditi_allpos"}, "fit_model_id": FRESH_SERVE}], "reg2")
    assert cli(reg2, 0.3, served=FRESH_SERVE) == writer(reg2, 0.3, FRESH_SERVE) == {"8" * 64}
    assert GEN.arditi_pins_from_registry(FRESH_PINS, registry_root=reg2, models_table=_fresh_table(tmp_path))[FRESH][SID]["object_sha256"] == "8" * 64
    # two distinct reference/approved objects at one rung (the canonical and a hand-filed spelling): BOTH readers refuse by name, the same rows
    reg3 = _registry(tmp_path, [{"object_id": "a" * 64, "extra": {"staged_id": "ar_c0p3__arditi_allpos"}, "fit_model_id": FRESH_SERVE},
                                {"object_id": "8" * 64, "extra": {"staged_id": "ar_c0p30__arditi_allpos"}, "fit_model_id": FRESH_SERVE}], "reg3")
    r = _ids_cli(reg3, 0.3, served=FRESH_SERVE)
    assert r.returncode == 1 and r.stdout == "" and "2 reference/approved registry rows at this coefficient" in r.stderr and "one refusal direction must be THE reference" in r.stderr, r.stdout + r.stderr
    assert f"{'a' * 12} (reference; extra.staged_id)" in r.stderr and f"{'8' * 12} (reference; extra.staged_id)" in r.stderr, r.stderr
    with pytest.raises(SystemExit, match="2 reference/approved registry rows at this coefficient"):
        GEN.arditi_pins_from_registry(FRESH_PINS, registry_root=reg3, models_table=_fresh_table(tmp_path))
    # a registry not on disk refuses by name (never empty); a read without its identity or rung refuses; the mode is a read alone (no render)
    r = _ids_cli(tmp_path / "missing", 0.3)
    assert r.returncode == 1 and r.stdout == "" and "is not on disk" in r.stderr, r.stdout + r.stderr
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "sdfbase" / "make_full22_configs.py"), "--arditi-registered-ids", "--registry-root", str(reg)], capture_output=True, text=True)
    assert r.returncode == 1 and "needs --served-model-id" in r.stderr, r.stdout + r.stderr
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "sdfbase" / "make_full22_configs.py"), "--arditi-registered-ids", "--registry-root", str(reg), "--served-model-id", SERVE,
                        "--coefficient", "0.3", "--check"], capture_output=True, text=True)
    assert r.returncode == 1 and "renders and writes nothing" in r.stderr, r.stdout + r.stderr
    # the SQL twin is gone from both sides; 04 asks the generator (refusal_registry_ids) on the unit and on the DONE-marker revalidation alike,
    # with the served id, the rung and the campaign it spells the publication and the job name from; the generator's grammar is the fitter's
    src = FRONT.read_text()
    assert not hasattr(GEN, "arditi_rung_sql") and not hasattr(GEN, "arditi_staged_id") and "arditi_rung_sql" not in src and "REFUSAL_RUNG_SQL" not in src
    assert '"$REFUSAL_GENERATOR" --arditi-registered-ids --registry-root "$root" --served-model-id "$SERVE_ID" --coefficient "$COEF" --campaign "$CAMPAIGN"' in _fn(src, "refusal_registry_ids")
    assert 'ids="$(refusal_registry_ids)" || repro_fatal' in _fn(src, "refusal_direction") and 'registry_done "$unit" "$ids" refusal_published_ready ids && return 0' in _fn(src, "refusal_direction")
    assert 'ids="$(refusal_registry_ids)" || repro_fatal' in _fn(src, "refusal_direction_done_marker_valid") and "registry_object_ids" not in _fn(src, "refusal_direction")
    assert 'REFUSAL_GENERATOR="$REPRO_REPO/scripts/sdfbase/make_full22_configs.py"' in src
    rd = _fn(src, "registry_done")   # ONE done-ness decision for every unit: the `ids` scope takes the reader's ids, every other scope the one query
    assert 'if [ "${4:-self}" = ids ]; then ids="$2"; else' in rd and "registry_object_ids \"($2) AND campaign='$CAMPAIGN'\" \"${4:-self}\"" in rd and "registry_done_with" not in src
    assert GEN.arditi_cell_coefficient(FRD.staged_id(-0.5)) == -0.5 and FRD.staged_id(-0.5) == "ar_c-0p5__arditi_allpos" and GEN.arditi_cell_coefficient(FRD.staged_id(-1.0)) == -1.0


def test_04_reads_a_hand_filed_spelling_of_the_rung_as_registered_and_submits_no_second_fit(tmp_path):
    """The duplicate-submission regression, over the REAL refusal_registry_ids -> the REAL generator CLI -> a
    REAL registry at the schema: a registered row whose staged id is a NONCANONICAL spelling of the rung — `ar_cm1p0__arditi_allpos` for -1
    (the pre-fix SQL twin knew `ar_c-1` and `ar_cm1` only) and `ar_c0p30__arditi_allpos` for 0.3 — is the registered identity+rung to 04: with
    its publication at the served id's directory 04 marks the unit done and submits nothing; with no publication 04 REFUSES by name (mirror the
    row's publication or withdraw the row) and submits nothing — before the fix both read as unregistered and the fit was submitted again, a
    second object at the identity and rung the pins writer then refused. The pins writer pins the very same row. The control (no row at this
    rung) still submits the fit, the other rung named; the read keeps registry_object_ids' contract — a missing registry refuses a LIVE run and
    prints the plan under --dry-run, two reference rows at the rung refuse by name with nothing submitted — and the DONE-marker revalidation
    reads through the same reader."""
    # c = -1: the row spelled `ar_cm1p0`, its publication at the fitter's spelling of the directory (ar_c-1__arditi_allpos)
    M1 = FRD.staged_id(-1.0)
    shared = tmp_path / "shared"
    sha_m1 = _fitted(shared, b"a refusal direction at minus one " * 32, cell=M1, coefficient=-1.0)
    pub_m1 = shared / "exp" / "objects" / "refusal_direction" / SERVE / M1
    reg = _registry(tmp_path, [{"object_id": sha_m1, "extra": {"staged_id": "ar_cm1p0__arditi_allpos", "coefficient": -1.0}}], "reg_m1")
    FIT_M1 = f"submit refusal_direction scripts/sbatch/fit_refusal_direction.sbatch {SERVE} Qwen/Qwen3.5-4B {'0' * 40} {shared}/exp/objects/refusal_direction"
    r = _run_04_refusal(pub_m1, ids="UNUSED", registry=reg, coefficient=-1.0)
    assert r.returncode == 0 and f"mark refusal_direction {sha_m1}" in r.stdout and "nothing to do" in r.stdout, r.stdout + r.stderr
    assert "submit refusal_direction" not in r.stdout and "registry_unavailable" not in r.stderr and "FATAL" not in r.stdout, r.stdout + r.stderr
    # nothing published at the served id's directory (the row filed from another root, or under its own spelling of the directory): refused by name, no fit
    r = _run_04_refusal(tmp_path / "nothing" / "exp" / "objects" / "refusal_direction" / SERVE / M1, ids="UNUSED", registry=reg, coefficient=-1.0)
    assert r.returncode == 7 and f"FATAL: refusal_direction: the registry holds the refusal direction of {SERVE} at c=-1 under campaign sdfbase_full22 (object {sha_m1})" in r.stdout, r.stdout + r.stderr
    assert "absent: no COMPLETE at" in r.stdout and "a refit here would register a SECOND object of the same identity and rung" in r.stdout, r.stdout
    assert "submit refusal_direction" not in r.stdout and "mark refusal_direction" not in r.stdout, r.stdout
    # the DONE marker over the hand-filed row: honoured through the same reader; nothing runs
    r = _run_04_refusal(pub_m1, ids="UNUSED", registry=reg, coefficient=-1.0, done_marker=True)
    assert r.returncode == 0 and "DONE marker present" in r.stdout and "skipping" in r.stdout and "NOT honoured" not in r.stdout and "submit refusal_direction" not in r.stdout, r.stdout + r.stderr
    # the pins writer pins the SAME row for a cell at -1 (a table whose refusal_direction cell is the fitter's spelling of -1, serving SERVE)
    table = _fresh_table(tmp_path, "minus_one.yaml")
    t = yaml.safe_load(table.read_text(encoding="utf-8"))
    t["models"][FRESH]["objects"]["refusal_direction"]["cell_id"] = M1; t["models"][FRESH]["serve"]["checkpoint"]["model_id"] = SERVE
    table.write_text(yaml.safe_dump(t, sort_keys=False), encoding="utf-8")
    assert GEN.arditi_pins_from_registry(FRESH_PINS, registry_root=reg, models_table=table)[FRESH][f"arditi__{M1}"]["object_sha256"] == sha_m1
    # c = 0.3: the hand-filed POSITIVE spelling `ar_c0p30`, the publication at the canonical directory (ar_c0p3): done, no fit; absent: refused, no fit
    shared_p = tmp_path / "shared_p"
    sha_p = _fitted(shared_p, b"a refusal direction spelled 0p30 " * 32, served=FRESH_SERVE)
    pub_p = shared_p / "exp" / "objects" / "refusal_direction" / FRESH_SERVE / CELL
    reg_p = _registry(tmp_path, [{"object_id": sha_p, "extra": {"staged_id": "ar_c0p30__arditi_allpos"}, "fit_model_id": FRESH_SERVE}], "reg_p")
    r = _run_04_refusal(pub_p, ids="UNUSED", registry=reg_p, served=FRESH_SERVE)
    assert r.returncode == 0 and f"mark refusal_direction {sha_p}" in r.stdout and "submit refusal_direction" not in r.stdout and "FATAL" not in r.stdout, r.stdout + r.stderr
    r = _run_04_refusal(tmp_path / "nothing_p" / "exp" / "objects" / "refusal_direction" / FRESH_SERVE / CELL, ids="UNUSED", registry=reg_p, served=FRESH_SERVE)
    assert r.returncode == 7 and f"the registry holds the refusal direction of {FRESH_SERVE} at c=0.3" in r.stdout and "submit refusal_direction" not in r.stdout, r.stdout + r.stderr
    assert GEN.arditi_pins_from_registry(FRESH_PINS, registry_root=reg_p, models_table=_fresh_table(tmp_path))[FRESH][SID]["object_sha256"] == sha_p
    # the control: no row at THIS rung (the registry holds the 0.5 rung alone) and nothing published — the fit IS submitted, the other rung NAMED
    reg_other = _registry(tmp_path, [{"object_id": "5" * 64, "extra": {"staged_id": FRD.staged_id(0.5)}}], "reg_other")
    r = _run_04_refusal(tmp_path / "nothing_c" / "exp" / "objects" / "refusal_direction" / SERVE / M1, ids="UNUSED", registry=reg_other, coefficient=-1.0)
    assert r.returncode == 0 and FIT_M1.replace(str(shared), str(tmp_path / "nothing_c")) in r.stdout and f"job-name refusal-direction-{SERVE}-{M1}" in r.stdout, r.stdout + r.stderr
    assert "other row(s) of the identity" in r.stderr and f"{'5' * 12} (reference; extra.staged_id 0.5)" in r.stderr and "mark refusal_direction" not in r.stdout, r.stdout + r.stderr
    # registry_object_ids' contract kept: a MISSING registry refuses a LIVE run by name (nothing submitted) and prints the plan under --dry-run
    r = _run_04_refusal(pub_m1, ids="UNUSED", registry=tmp_path / "missing", coefficient=-1.0)
    assert r.returncode == 7 and "FATAL: refusal_direction: refusal_registry_ids refused (above)" in r.stdout and "registry_unavailable: " in r.stderr and "missing or unreadable" in r.stderr, r.stdout + r.stderr
    assert "submit refusal_direction" not in r.stdout and "mark refusal_direction" not in r.stdout, r.stdout
    r = _run_04_refusal(pub_m1, ids="UNUSED", registry=tmp_path / "missing", coefficient=-1.0, dry=1)
    assert r.returncode == 0 and FIT_M1 in r.stdout and "a dry run proceeds as if nothing were registered" in r.stdout + r.stderr and "FATAL" not in r.stdout, r.stdout + r.stderr
    r = _run_04_refusal(pub_m1, ids="UNUSED", registry=tmp_path / "missing", coefficient=-1.0, done_marker=True)   # the marker revalidation reads through the same reader
    assert r.returncode == 7 and "refusal_registry_ids refused (above) — the DONE marker's validity cannot be decided" in r.stdout and "skipping" not in r.stdout, r.stdout + r.stderr
    # two reference rows at the rung (the hand-filed and the fitter's spelling): the reader refuses by name as the pins writer does — nothing submitted, nothing marked
    reg_two = _registry(tmp_path, [{"object_id": sha_m1, "extra": {"staged_id": "ar_cm1p0__arditi_allpos"}}, {"object_id": "9" * 64, "extra": {"staged_id": M1}}], "reg_two")
    r = _run_04_refusal(pub_m1, ids="UNUSED", registry=reg_two, coefficient=-1.0)
    assert r.returncode == 7 and "FATAL: refusal_direction: refusal_registry_ids refused (above)" in r.stdout, r.stdout + r.stderr
    assert "registry stderr: refusal direction of" in r.stderr and "2 reference/approved registry rows at this coefficient" in r.stderr and "registry_unavailable: " in r.stderr, r.stderr
    assert "submit refusal_direction" not in r.stdout and "mark refusal_direction" not in r.stdout and "nothing to do" not in r.stdout, r.stdout


def test_two_keys_serving_one_checkpoint_share_the_publication_the_row_and_the_pin(tmp_path, capsys):
    """Two keys over one served checkpoint: key A fits the refusal direction, key B (the same served checkpoint and coefficient) runs 04 and 05.
    The publication is the SERVED id's directory, so B's 04 reads the row done AGAINST THAT PUBLICATION (the readiness function, no empty
    directory), the generator writes the SAME pin under both keys (one row, one object), and 05's cells_json plans both keys' cells from the one
    directory — before the fix B's 04 said done over nothing, the generator pinned A's object under B, and 05 pulled from B's empty dir."""
    table = _two_key_table(tmp_path)
    shared = tmp_path / "shared"
    sha = _fitted(shared, b"one checkpoint, two keys " * 64, served=FRESH_SERVE)
    reg = _registry(tmp_path, [_arditi_row(sha, fit_model_id=FRESH_SERVE)])
    pins = GEN.arditi_pins_from_registry(FRESH_PINS, registry_root=reg, models_table=table)
    log = capsys.readouterr().out
    assert pins[FRESH][SID]["object_sha256"] == pins[FRESH2][SID]["object_sha256"] == sha and pins[FRESH][SID] == pins[FRESH2][SID]
    assert f"pins: {FRESH}/{SID} (refusal_direction; campaign sdfbase_full22, served {FRESH_SERVE}" in log and f"pins: {FRESH2}/{SID} (refusal_direction; campaign sdfbase_full22, served {FRESH_SERVE}" in log
    for k in (FRESH, FRESH2):   # the two differences' pins too, so 05's plan has every derived object pinned (their own writer is not under test here)
        pins[k][LORA_SID] = {"object_sha256": "6" * 64, "source": "synthetic"}; pins[k][RELU_SID] = {"object_sha256": "7" * 64, "source": "synthetic"}
    pf = _write_pins(tmp_path / "pins.json", pins)
    srcs = {}
    for key in (FRESH, FRESH2):
        rc, out, err = _cells_json("battery", pf, table=table, key=key, fit=FRESH_FIT)
        assert rc == 0, err
        row = {r["view_cell"]: r for r in json.loads(out)}["refusal_direction"]
        assert row["object_sha256_expected"] == sha and row["object_sha256_expected_source"] == f"{pf}:{key}.{SID}"
        srcs[key] = row["source"]
    assert srcs[FRESH] == srcs[FRESH2] == f"cluster_a:/shared/exp/objects/refusal_direction/{FRESH_SERVE}/{CELL}"
    # B's 04: the row IS done — bound to the one publication at the served id's directory (not an empty directory of B's own)
    pub = shared / "exp" / "objects" / "refusal_direction" / FRESH_SERVE / CELL
    r = _run_04_refusal(pub, ids=sha, served=FRESH_SERVE)
    assert r.returncode == 0 and f"mark refusal_direction {sha}" in r.stdout and "submit refusal_direction" not in r.stdout, r.stdout + r.stderr
    # and the stager stages B's cell live from that directory by the shared pin
    site = write_site(tmp_path, clusters=("cluster_a",), ssh={"cluster_a": f"local:{shared}"}, shared={"cluster_a": str(shared)})
    rc, sout, bank = _stage(tmp_path, [{"cell_id": CELL, "lane": "arditi", "kind": "derived", "source": f"local:{pub}", "object_sha256_expected": sha,
                                        "object_sha256_expected_source": f"{pf}:{FRESH2}.{SID}"}], site, FRESH2, name="bank_b")
    assert rc == 0 and SO.object_records(bank / "_records")[0]["status"] == "staged", sout


def test_the_fit_job_is_named_by_the_served_identity_and_the_rung_so_a_second_key_sees_the_first_keys_fit_queued(tmp_path):
    """The publication directory and the registry row are the SERVED checkpoint's and the rung's, but the fit's job name was the TABLE key's
    (refusal-direction-<key>), so key B, running 04 while key A's fit of the same checkpoint and coefficient was queued or running — before any
    COMPLETE or row — read the queue for ITS name, missed A's job, and submitted a second fit of the identity to the same directory (two objects
    of one identity and rung at the fold, which the pins writer refuses). Now the name is the identity's,
    refusal-direction-<served id>-<staged id>: the same word under every key that serves the checkpoint, so B's queue read sees A's fit and
    waits; the staged id spells the coefficient, so a queued fit at another rung does not suppress this one, nor does another served
    checkpoint's; the repair path and the published-but-unfolded read key on the same name. The REAL 04 functions over a stub queue."""
    nothing = tmp_path / "none" / "exp" / "objects" / "refusal_direction" / SERVE / CELL
    NAME = f"refusal-direction-{SERVE}-{CELL}"
    # key A (no row, nothing published): the fit is submitted under the identity's name — the table key is in no word of it
    r = _run_04_refusal(nothing, ids="", model="keyA")
    assert r.returncode == 0 and "submit refusal_direction" in r.stdout and f"job-name {NAME}" in r.stdout and "keyA" not in r.stdout, r.stdout + r.stderr
    # key B, the same served checkpoint and coefficient, while A's job is queued: the SAME name, seen in the queue — nothing submitted
    r = _run_04_refusal(nothing, ids="", model="keyB", queued=(NAME,))
    assert r.returncode == 0 and f"job {NAME} is already queued/running" in r.stdout and "submit refusal_direction" not in r.stdout, r.stdout + r.stderr
    # B over an empty queue submits under the very same name (the name is the identity's, whatever the key)
    r = _run_04_refusal(nothing, ids="", model="keyB")
    assert r.returncode == 0 and f"job-name {NAME}" in r.stdout and "submit refusal_direction" in r.stdout, r.stdout + r.stderr
    # the pre-fix spelling in the queue (a job an older front door submitted under the table key) is not this name: not honoured
    r = _run_04_refusal(nothing, ids="", model="keyB", queued=("refusal-direction-keyB", "refusal-direction-keyA"))
    assert r.returncode == 0 and "submit refusal_direction" in r.stdout and "already queued" not in r.stdout, r.stdout + r.stderr
    # another RUNG of the same checkpoint queued (c=0.5), or another served checkpoint's fit: not this identity — this fit is submitted
    r = _run_04_refusal(nothing, ids="", model="keyB", queued=(f"refusal-direction-{SERVE}-{FRD.staged_id(0.5)}", f"refusal-direction-other_it-{CELL}"))
    assert r.returncode == 0 and "submit refusal_direction" in r.stdout and f"job-name {NAME}" in r.stdout and "already queued" not in r.stdout, r.stdout + r.stderr
    # the repair path (registered, the sidecar unlisted) waits on the identity's name too, under either key
    pre_shared = tmp_path / "pre"
    sha_pre = _fitted(pre_shared, b"a publication from before the identity sidecar " * 32, with_identity_sidecar=False)
    pre = pre_shared / "exp" / "objects" / "refusal_direction" / SERVE / CELL
    r = _run_04_refusal(pre, ids=sha_pre, model="keyB", queued=(NAME,))
    assert r.returncode == 0 and f"job {NAME} is already queued/running" in r.stdout and "writes the identity sidecar" in r.stdout and "submit refusal_direction" not in r.stdout, r.stdout + r.stderr
    # the published-but-unfolded read (no row, COMPLETE present) hands the same name to published_unfolded, whose wait on an active job reads it
    shared = tmp_path / "shared"
    sha = _fitted(shared, b"published, not folded " * 64)
    pub = shared / "exp" / "objects" / "refusal_direction" / SERVE / CELL
    r = _run_04_refusal(pub, ids="", model="keyB")
    assert r.returncode == 0 and f"published_unfolded {pub} {NAME}" in r.stdout and "submit refusal_direction" not in r.stdout, r.stdout + r.stderr
    # the name's spelling, pinned in the front door and in the sbatch's header; the queue read and the unfolded wait key on it
    src = FRONT.read_text(); fn = _fn(src, "refusal_direction")
    assert 'name="refusal-direction-${SERVE_ID}-${REFUSAL_STAGED}"' in fn and "refusal-direction-${REPRO_MODEL}" not in src
    assert 'cluster_job_active "$name"' in fn and 'published_unfolded "$unit" "$REFUSAL_DIR" "$name"' in fn and 'refusal_submit "$unit" "$name"' in fn
    sb = (REPO / "scripts" / "sbatch" / "fit_refusal_direction.sbatch").read_text()
    assert "sbatch -J refusal-direction-<model_key>-<staged id> " in sb and "-J refusal-direction-<model_key> " not in sb


def test_the_sidecar_repair_is_its_own_funnel_unit_so_the_fits_registered_binding_never_skips_it(tmp_path):
    """The repair-only resubmission (a registered row whose publication lacks the identity sidecar, or carries a damaged LISTED one) was handed
    the FIT's funnel unit, `refusal_direction`, whose logical work id is bound in the funnel's ledger to the attempt that registered the row; the
    funnel is idempotent by that id (submit_twin.submit_one: a unit `submitted` or `registered` prints `skipped … already registered` and
    launches nothing), so the repair was never a job and every later 04 run asked again, the publication unstageable for good — the fragment
    tests stubbed submit with an echo and never met the ledger. Now the repair is its own unit, `refusal_direction/sidecar`
    (difference_relu's `/liveness` shape): the same sbatch, args and identity job name (the fitter's kept path writes the sidecar and replays the
    record; the queue read waits on the live job), under an id the fit's binding does not shadow. Shown twice: the REAL 04 functions over a stub
    cluster hand the repair unit to submit on both repair branches and the fit's unit on a fresh fit; and the REAL funnel (submit_twin.py over a
    fake Slurm, the fit's id `registered` in its ledger) SKIPS the fit's unit and SUBMITS the repair's — one job, the identity's name, the fit's
    binding untouched."""
    NAME = f"refusal-direction-{SERVE}-{CELL}"
    # (1) the fragment: the publication from before the sidecar, and a damaged listed sidecar — the repair unit, never the fit's
    pre_shared = tmp_path / "pre"
    sha_pre = _fitted(pre_shared, b"registered, the sidecar unlisted " * 32, with_identity_sidecar=False)
    pre = pre_shared / "exp" / "objects" / "refusal_direction" / SERVE / CELL
    r = _run_04_refusal(pre, ids=sha_pre)
    assert r.returncode == 0 and f"submit refusal_direction/sidecar scripts/sbatch/fit_refusal_direction.sbatch {SERVE} Qwen/Qwen3.5-4B {'0' * 40} {pre_shared}/exp/objects/refusal_direction" in r.stdout, r.stdout + r.stderr
    assert f"job-name {NAME}" in r.stdout and "submit refusal_direction scripts" not in r.stdout and "funnel unit refusal_direction/sidecar" in r.stdout and "never re-mints it" in r.stdout, r.stdout
    shared = tmp_path / "shared"
    sha = _fitted(shared, b"registered, the listed sidecar damaged " * 32)
    pub = shared / "exp" / "objects" / "refusal_direction" / SERVE / CELL
    (pub / FRD.PROVENANCE_NAME).unlink()
    r = _run_04_refusal(pub, ids=sha)
    assert r.returncode == 0 and "the one manifest problem a kept publication repairs" in r.stdout and "submit refusal_direction/sidecar " in r.stdout and "submit refusal_direction scripts" not in r.stdout, r.stdout + r.stderr
    # a fresh fit (no row, nothing published) is the fit's unit, as before
    r = _run_04_refusal(tmp_path / "none" / "exp" / "objects" / "refusal_direction" / SERVE / CELL, ids="")
    assert r.returncode == 0 and "submit refusal_direction scripts/sbatch/fit_refusal_direction.sbatch" in r.stdout and "/sidecar" not in r.stdout, r.stdout + r.stderr
    src = FRONT.read_text(); fn = _fn(src, "refusal_direction")
    assert 'refusal_submit "$unit/sidecar" "$name"; return;;' in fn and fn.count('refusal_submit "$unit" "$name"') == 1 and 'lwid(){ echo "$REPRO_MODEL/fit/objects/$1"; }' in src
    # (2) the REAL funnel: the fit's logical work id is `registered` (its attempt registered the row) — the fit's unit is SKIPPED, the repair's SUBMITTED
    (tmp_path / "w").mkdir(); w = FitWorld(tmp_path / "w"); w.certify("cluster_a", "40b49ecf00359e44")
    ledger = str(w.state / "submissions.jsonl")
    fit_lwid = "m/fit/objects/refusal_direction"
    ST.write_binding(ledger, {**ST.submitted_record(fit_lwid, "4242", "cluster_a", f"{fit_lwid}@cluster_a@1", NAME), "state": "registered"})
    def funnel(unit: str, lwid: str) -> subprocess.CompletedProcess:
        args = ["submit", "--unit", unit, "--lwid", lwid, "--script", "scripts/sbatch/fit_refusal_direction.sbatch", "--name", NAME,
                "--args", f"{SERVE} Qwen/Qwen3.5-4B {'0' * 40} /shared/exp/objects/refusal_direction", "--clusters", "cluster_a", "--kind", "fit",
                "--wave-sha", WAVE_SHA, "--time", "01:00:00", "--env", "REFUSAL_COEFFICIENT=0.3 REFUSAL_HARMFUL=arditi:harmful_train REFUSAL_HARMLESS=arditi:harmless_train"]
        return subprocess.run([sys.executable, str(LIB / "submit_twin.py"), "--config", str(w.site / "config.env"), "--model", "m", *args],
                              capture_output=True, text=True, env=w.env(), timeout=120)
    r = funnel("refusal_direction", fit_lwid)                                     # the pre-fix shape: the fit's unit — skipped, no job
    assert r.returncode == 0 and "skipped cluster_a 4242 refusal_direction (already registered" in r.stdout and "pass --resubmit" in r.stdout, r.stdout + r.stderr
    assert job_ids(w.fake["cluster_a"]) == [], r.stdout
    r = funnel("refusal_direction/sidecar", f"{fit_lwid}/sidecar")               # the repair's own unit — submitted
    assert r.returncode == 0 and "skipped" not in r.stdout and "submitted cluster_a " in r.stdout and " refusal_direction/sidecar" in r.stdout, r.stdout + r.stderr
    jids = job_ids(w.fake["cluster_a"]); assert len(jids) == 1, r.stdout
    argv = (w.fake["cluster_a"] / "jobs" / f"{jids[0]}.argv").read_text()
    assert f"-J {NAME}" in argv and "scripts/sbatch/fit_refusal_direction.sbatch" in argv and f" {SERVE} Qwen/Qwen3.5-4B " in argv and "--export" not in argv, argv
    assert f"--comment=sha:{WAVE_SHA}" in argv, argv                               # the pin the sbatch refuses to run without (its header)
    assert ST.binding_state(ledger, f"{fit_lwid}/sidecar")["state"] == "submitted" and ST.binding_state(ledger, f"{fit_lwid}/sidecar")["job_id"] == str(jids[0])
    fit = ST.binding_state(ledger, fit_lwid)
    assert fit["state"] == "registered" and fit["job_id"] == "4242", fit            # the fit's binding is not touched by the repair
    # the repair bound, a second ask is the funnel's idempotence again (04's queue read waits on the live job before it asks; a dead repair is --resubmit's)
    r = funnel("refusal_direction/sidecar", f"{fit_lwid}/sidecar")
    assert r.returncode == 0 and f"skipped cluster_a {jids[0]} refusal_direction/sidecar (already submitted" in r.stdout and job_ids(w.fake["cluster_a"]) == jids, r.stdout + r.stderr


def test_an_unreadable_queue_refuses_by_name_before_any_submission_and_a_readable_one_later_submits(tmp_path):
    """04's cluster_job_active dropped the ssh / squeue exit status (`2>/dev/null`, the output alone) and cached an error as an EMPTY queue for
    the invocation — so with no folded row yet, a second key whose queue probe timed out while the first key's fit of the same identity and rung
    was queued read it idle and submitted a second fit (two objects at one identity and rung at the fold, which the pins writer refuses; the
    submission ledger, keyed by table key, cannot deduplicate them), and the sidecar-repair branch resubmitted over a running repair. Now the
    read is 03_hpo.sh's fail-CLOSED shape: the status is kept, a sentinel tells "squeue answered nothing" from "nothing answered", and an
    unreadable queue REFUSES by name — UNKNOWN, never 'not active' — before any submission, on the fresh-fit path and the repair path alike,
    through the dispatcher's marker revalidation too. Separately, a later run whose probe answers behaves as before: the first key's fit queued
    -> waits; an empty queue -> submits; one read per invocation; a dry run reads no queue. The REAL cluster_job_active over a fake ssh (a
    timeout = ssh's 255; squeue failing = non-zero, no sentinel)."""
    nothing = tmp_path / "none" / "exp" / "objects" / "refusal_direction" / SERVE / CELL
    NAME = f"refusal-direction-{SERVE}-{CELL}"
    FIT = "submit refusal_direction scripts/sbatch/fit_refusal_direction.sbatch"
    # the fresh-fit path (no row, nothing published), key B's probe failing either way: REFUSED by name, nothing submitted, the status in the message
    for probe, rc_word in (("down", "rc=255: ssh: connect to host nohost port 22: Connection timed out"), ("squeue_error", "rc=1: squeue: error")):
        r = _run_04_refusal(nothing, ids="", model="keyB", queue_probe=probe)
        assert r.returncode == 7 and "FATAL: the queue on c could not be read (ssh/squeue " in r.stdout and rc_word in r.stdout, (probe, r.stdout + r.stderr)
        assert f"whether a job named {NAME} is queued or running is UNKNOWN, never 'not active': nothing is submitted" in r.stdout and "re-run when nohost answers" in r.stdout, (probe, r.stdout)
        assert "submit refusal_direction" not in r.stdout and "already queued" not in r.stdout and "mark refusal_direction" not in r.stdout, (probe, r.stdout)
    # the sidecar-repair path (registered, the sidecar unlisted), the probe down: the same refusal — no repair submitted over an unknown queue
    pre_shared = tmp_path / "pre"
    sha_pre = _fitted(pre_shared, b"registered, the queue unreadable " * 32, with_identity_sidecar=False)
    pre = pre_shared / "exp" / "objects" / "refusal_direction" / SERVE / CELL
    r = _run_04_refusal(pre, ids=sha_pre, model="keyB", queue_probe="down")
    assert r.returncode == 7 and "could not be read" in r.stdout and "UNKNOWN" in r.stdout, r.stdout + r.stderr
    assert "submit refusal_direction" not in r.stdout and "resubmitting the job" not in r.stdout and "mark refusal_direction" not in r.stdout, r.stdout
    r = _run_04_refusal(pre, ids=sha_pre, model="keyB", queue_probe="down", done_marker=True)     # the marker not honoured, the unit runs, the queue refuses
    assert r.returncode == 7 and "NOT honoured" in r.stdout and "could not be read" in r.stdout and "submit refusal_direction" not in r.stdout, r.stdout + r.stderr
    # SEPARATELY, a later run whose probe answers: the first key's fit queued -> waits (one read); an empty queue -> the fit is submitted (one read)
    log = tmp_path / "queue.log"
    r = _run_04_refusal(nothing, ids="", model="keyB", queued=(NAME,), queue_log=log)
    assert r.returncode == 0 and f"job {NAME} is already queued/running" in r.stdout and "submit refusal_direction" not in r.stdout and "FATAL" not in r.stdout, r.stdout + r.stderr
    reads = log.read_text().splitlines()
    assert len(reads) == 1 and "squeue -h -u $USER -o %j && echo __QUEUE_OK__" in reads[0], reads
    log.unlink()
    r = _run_04_refusal(nothing, ids="", model="keyB", queue_log=log)
    assert r.returncode == 0 and FIT in r.stdout and f"job-name {NAME}" in r.stdout and "FATAL" not in r.stdout and len(log.read_text().splitlines()) == 1, r.stdout + r.stderr
    r = _run_04_refusal(pre, ids=sha_pre, model="keyB")                                           # the repair, over a readable empty queue
    assert r.returncode == 0 and "submit refusal_direction/sidecar " in r.stdout and "FATAL" not in r.stdout, r.stdout + r.stderr
    # a dry run reads no queue at all (the plan prints with the probe down)
    dry_log = tmp_path / "dry.log"
    r = _run_04_refusal(nothing, ids="", model="keyB", dry=1, queue_probe="down", queue_log=dry_log)
    assert r.returncode == 0 and FIT in r.stdout and "FATAL" not in r.stdout and not dry_log.exists(), r.stdout + r.stderr
    # the read itself, extracted: the status kept (no `2>/dev/null`), the sentinel, the refusal; the queue read ONCE for any number of names;
    # an EMPTY queue (rc 0, the sentinel alone) is 'not active', never a refusal
    src = FRONT.read_text(); fn = _fn(src, "cluster_job_active")
    assert 'squeue -h -u \\$USER -o %j && echo __QUEUE_OK__" 2>&1)"; rc=$?' in fn and "0:*__QUEUE_OK__*) QUEUE_NAMES=" in fn and "repro_fatal" in fn and "2>/dev/null" not in fn, fn
    once = tmp_path / "once.log"
    def read(answer: str, probe: str = "up") -> subprocess.CompletedProcess:
        frag = (f'CLUSTER=c; CL_HOST=nohost; REPRO_DRY_RUN=0; QUEUE_PROBE={probe}; QUEUE_LOG={once}\nQUEUE_ANSWER="{answer}"\n'
                'repro_fatal(){ echo "FATAL: $*"; exit 7; }\n' + _FAKE_SSH + fn +
                '\nfor n in alpha beta gamma; do if cluster_job_active "$n"; then echo "active: $n"; else echo "idle: $n"; fi; done\n')
        return subprocess.run(["bash", "-c", frag], capture_output=True, text=True, timeout=60)
    r = read("alpha\nbeta")
    assert r.returncode == 0 and r.stdout.splitlines() == ["active: alpha", "active: beta", "idle: gamma"], r.stdout + r.stderr
    assert len(once.read_text().splitlines()) == 1, once.read_text()                                  # three questions, one read
    once.unlink()
    r = read("")
    assert r.returncode == 0 and r.stdout.splitlines() == ["idle: alpha", "idle: beta", "idle: gamma"] and len(once.read_text().splitlines()) == 1, r.stdout + r.stderr
    r = read("alpha", probe="down")
    assert r.returncode == 7 and r.stdout.startswith("FATAL: the queue on c could not be read (ssh/squeue rc=255") and "active:" not in r.stdout and "idle:" not in r.stdout, r.stdout + r.stderr


# ---------------------------------------------------------------------------------------------------------------------------------------------
# 04 read its rung from SISTEER_REFUSAL_COEFFICIENT alone while the pins writer selected the rung from the models table's cell, so a knob the
# cell did not restate (0.5 over the copied example's `ar_c0p3`) fitted and registered a 0.5 no pin was written from, and a numerically equal
# cell spelled `ar_c0p30` was pinned at 0.3 while 04 published under `ar_c0p3` and 05 read the literal `ar_c0p30` directory. Now the CELL is the
# one identity: 04 derives its coefficient and staged id from it through the pins writer's own cell reading (make_full22_configs.py
# --arditi-cell), the knob may only restate it, and a cell that is not the fitter's spelling refuses in all three — 04 before any registry or
# cluster read, the pins writer, and 05's cells_json.
# ---------------------------------------------------------------------------------------------------------------------------------------------


def _cell_cli(table: Path, key: str, *more: str) -> subprocess.CompletedProcess:
    """make_full22_configs.py --arditi-cell, as 04's identity read runs it (the table, the key, and the knob when the site sets one)."""
    return subprocess.run([sys.executable, str(REPO / "scripts" / "sdfbase" / "make_full22_configs.py"), "--arditi-cell", "--models-table", str(table), "--model-key", key, *more],
                          capture_output=True, text=True, timeout=300)


def _table_with_cell(tmp: Path, cell_id: str, name: str, **cell_over) -> Path:
    """The fresh table with FRESH's refusal_direction cell spelled `cell_id` (and any other field of the cell overridden)."""
    p = _fresh_table(tmp, name)
    t = yaml.safe_load(p.read_text(encoding="utf-8"))
    t["models"][FRESH]["objects"]["refusal_direction"]["cell_id"] = cell_id
    t["models"][FRESH]["objects"]["refusal_direction"].update(cell_over)
    p.write_text(yaml.safe_dump(t, sort_keys=False), encoding="utf-8")
    return p


def test_the_refusal_directions_identity_is_the_tables_cell_read_by_the_pins_writers_own_reader(tmp_path):
    """The generator's --arditi-cell (arditi_declared_cell / arditi_cell_coefficient — the function arditi_pins_from_registry picks the pin's
    rung with): `<c:g> <cell id>` for a canonical cell at any rung, the knob accepted only as a restatement (0.30 is 0.3; 0.5 is refused by
    name with both values and the two ways to align them), a noncanonical cell refused naming the fitter's spelling, a row without the cell
    or with a non-arditi_bank one refused by name; the mode is a read alone. The pins writer refuses the same noncanonical cell by name
    (never pins a cell no fit publishes under), and so does 05's cells_json, before any pull, with no traceback."""
    table = _fresh_table(tmp_path)
    r = _cell_cli(table, FRESH); assert r.returncode == 0 and r.stdout == f"0.3 {CELL}\n", r.stdout + r.stderr
    r = _cell_cli(table, FRESH, "--coefficient", "0.30"); assert r.returncode == 0 and r.stdout == f"0.3 {CELL}\n", r.stdout + r.stderr
    r = _cell_cli(table, FRESH, "--coefficient", "0.5")
    assert r.returncode == 1 and r.stdout == "", r.stdout + r.stderr
    assert f"SISTEER_REFUSAL_COEFFICIENT=0.5 contradicts the models table: {FRESH}'s objects.refusal_direction cell {CELL!r} spells coefficient 0.3" in r.stderr, r.stderr
    assert "Unset SISTEER_REFUSAL_COEFFICIENT (or set it to 0.3)" in r.stderr and f"change the cell to {FRD.staged_id(0.5)!r}" in r.stderr, r.stderr
    # every rung the fitter spells: the coefficient is the cell's, a negative one and an integer included
    for c in (-1.0, 1.0, 0.15, 1.5):
        t = _table_with_cell(tmp_path, FRD.staged_id(c), f"rung_{c:g}.yaml".replace("-", "m"))
        r = _cell_cli(t, FRESH); assert r.returncode == 0 and r.stdout == f"{c:g} {FRD.staged_id(c)}\n", (c, r.stdout + r.stderr)
    # a noncanonical cell — the ladder's `m`, a trailing zero — refused naming the fitter's spelling, in the CLI, in the pins writer and in 05
    for noncanon, canon in (("ar_c0p30__arditi_allpos", CELL), ("ar_cm1p0__arditi_allpos", FRD.staged_id(-1.0))):
        t = _table_with_cell(tmp_path, noncanon, f"{noncanon}.yaml")
        r = _cell_cli(t, FRESH)
        assert r.returncode == 1 and r.stdout == "" and f"refusal direction cell {noncanon!r} spells coefficient" in r.stderr, r.stdout + r.stderr
        assert f"not the fitter's spelling of it, {canon!r}" in r.stderr and f"spell the cell {canon!r}" in r.stderr, r.stderr
        with pytest.raises(SystemExit, match=f"not the fitter's spelling of it, {canon!r}"):   # before any registry read: the root need not exist
            GEN.arditi_pins_from_registry(FRESH_PINS, registry_root=tmp_path / "unused", models_table=t)
        pins = _write_pins(tmp_path / f"{noncanon}_pins.json", {FRESH: {f"arditi__{noncanon}": {"object_sha256": "5" * 64, "source": "synthetic"},
                                                                        LORA_SID: {"object_sha256": "6" * 64, "source": "synthetic"}, RELU_SID: {"object_sha256": "7" * 64, "source": "synthetic"}}})
        rc, out, err = _cells_json("battery", pins, table=t, key=FRESH, fit=FRESH_FIT)
        assert rc != 0 and out == "" and f"{FRESH}: object refusal_direction: refusal direction cell {noncanon!r}" in err and f"not the fitter's spelling of it, {canon!r}" in err, err
        assert "refusing to stage a cell no fit publishes under" in err and "Traceback" not in err, err
    # a row without the cell, a cell that is not an arditi_bank object, a key the table lacks: refused by name; the mode is a read alone
    t = _fresh_table(tmp_path, "nocell.yaml"); d = yaml.safe_load(t.read_text(encoding="utf-8")); d["models"][FRESH]["objects"].pop("refusal_direction")
    t.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    r = _cell_cli(t, FRESH); assert r.returncode == 1 and r.stdout == "" and "objects.refusal_direction.cell_id is not declared" in r.stderr, r.stdout + r.stderr
    r = _cell_cli(_table_with_cell(tmp_path, CELL, "fitbank.yaml", source="fit_bank"), FRESH)
    assert r.returncode == 1 and "declares source 'fit_bank'" in r.stderr and "not an arditi_bank object" in r.stderr, r.stderr
    r = _cell_cli(table, "nope"); assert r.returncode == 1 and "no model 'nope'" in r.stderr, r.stderr
    r = _cell_cli(table, FRESH, "--check"); assert r.returncode == 1 and "renders and writes nothing" in r.stderr, r.stderr
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "sdfbase" / "make_full22_configs.py"), "--arditi-cell"], capture_output=True, text=True)
    assert r.returncode == 1 and "needs --model-key" in r.stderr, r.stdout + r.stderr
    # ONE cell reading: the pins writer picks the pin's rung with the very function the CLI reads the cell with; the row reading stays lenient
    gen = (REPO / "scripts" / "sdfbase" / "make_full22_configs.py").read_text()
    assert 'c = arditi_cell_coefficient(str(o["cell_id"]))' in gen and "return cell_id, arditi_cell_coefficient(cell_id)" in gen
    assert GEN.arditi_declared_cell(table, FRESH) == (CELL, 0.3) and GEN.arditi_row_rung({"extra_json": json.dumps({"staged_id": "ar_c0p30__arditi_allpos"}), "base_scale": None}) == ("extra.staged_id", 0.3)


def test_04_derives_its_coefficient_from_the_tables_cell_and_refuses_a_knob_or_a_spelling_that_disagrees_before_any_submission(site04, tmp_path):
    """The front door end to end (run_stage over the dry-run site), under a key of one's own: 04's refusal-direction submission carries the
    CELL's coefficient — `ar_c0p3` -> REFUSAL_COEFFICIENT=0.3 with no knob set, and with SISTEER_REFUSAL_COEFFICIENT=0.30 restating it; a cell
    at -1 -> REFUSAL_COEFFICIENT=-1 with the job named and the directory spelled by that cell, no knob set. A knob the cell does not restate
    (0.5), or a cell that is not the fitter's spelling (`ar_c0p30`), REFUSES by name (rc 2) with both values, before the registry or the
    cluster is read and before any submission, under --dry-run and live alike (a misdeclared identity has no plan); `--object all` requests
    the unit and is checked; another unit alone is untouched by the knob. Before the fix 04 submitted the 0.5 fit and registered it, the pins
    writer — selecting the rung from the cell — wrote no pin, and 05 refused the cell after the GPU hour."""
    table = _fresh_table(tmp_path)

    def run(*args, table=table, dry=True, **env):
        r = run_stage("04_fit_objects.sh", *(["--dry-run"] if dry else []), "--model", FRESH, *args, site=site04, env={"SISTEER_MODELS_TABLE": str(table), **env}, timeout=600)
        return r, r.stdout + r.stderr

    def sbatch_line(out):
        lines = [ln for ln in out.splitlines() if "fit_refusal_direction.sbatch" in ln and "DRY-RUN" in ln]
        assert len(lines) == 1, out[-3000:]
        return lines[0]

    r, out = run("--object", "refusal_direction")
    assert r.returncode == 0, out[-3000:]
    ln = sbatch_line(out); assert "REFUSAL_COEFFICIENT=0.3 " in ln and f"-J refusal-direction-{FRESH_SERVE}-{CELL} " in ln, ln
    # the prompt sets: by default the paper fit's own training draw, the inputs the committed refusal-direction objects were fit on
    # (sisteer.data.arditi_fit_prompts; tests/test_refusal_direction_fit_prompts_wiring.py holds the job's check of the installed copy)
    assert "REFUSAL_HARMFUL=arditi:harmful_train REFUSAL_HARMLESS=arditi:harmless_train" in ln, ln
    r, out = run("--object", "refusal_direction", SISTEER_REFUSAL_COEFFICIENT="0.30", SISTEER_REFUSAL_HARMFUL="harmbench_val",
                 SISTEER_REFUSAL_HARMLESS="alpaca")
    assert r.returncode == 0 and "REFUSAL_COEFFICIENT=0.3 " in sbatch_line(out), out[-3000:]
    assert "REFUSAL_HARMFUL=harmbench_val REFUSAL_HARMLESS=alpaca" in sbatch_line(out), out[-3000:]    # a site's own sets stand
    # the knob contradicts the cell: refused by name with both values, nothing read, nothing planned — dry run and live alike
    for dry in (True, False):
        r, out = run("--object", "refusal_direction", dry=dry, SISTEER_REFUSAL_COEFFICIENT="0.5")
        assert r.returncode == 2, (dry, out[-3000:])
        assert f"SISTEER_REFUSAL_COEFFICIENT=0.5 contradicts the models table: {FRESH}'s objects.refusal_direction cell {CELL!r} spells coefficient 0.3" in r.stderr, r.stderr[-3000:]
        assert "REFUSED: 04: the refusal direction's identity could not be formed from" in r.stderr and "which may only restate it, or stay unset" in r.stderr, r.stderr[-3000:]
        assert "sbatch" not in out and "DRY-RUN" not in out and "registry_unavailable" not in out and "04_fit_objects: model=" not in out and "would REFUSE" not in out, out[-3000:]
    # the cell is not the fitter's spelling: refused naming the fitter's, the same way, before anything is read
    r, out = run("--object", "refusal_direction", table=_table_with_cell(tmp_path, "ar_c0p30__arditi_allpos", "p30_04.yaml"))
    assert r.returncode == 2 and f"refusal direction cell 'ar_c0p30__arditi_allpos' spells coefficient 0.3 but is not the fitter's spelling of it, {CELL!r}" in r.stderr, r.stderr[-3000:]
    assert f"spell the cell {CELL!r}" in r.stderr and "sbatch" not in out and "DRY-RUN" not in out and "registry_unavailable" not in out and "04_fit_objects: model=" not in out, out[-3000:]
    # the coefficient is DERIVED from the cell: a cell at -1 and no knob — the submission, the job name and the served id's directory all spell it
    M1 = FRD.staged_id(-1.0)
    r, out = run("--object", "refusal_direction", table=_table_with_cell(tmp_path, M1, "m1_04.yaml"))
    assert r.returncode == 0, out[-3000:]
    ln = sbatch_line(out)
    assert "REFUSAL_COEFFICIENT=-1 " in ln and f"-J refusal-direction-{FRESH_SERVE}-{M1} " in ln and f"fit_refusal_direction.sbatch {FRESH_SERVE} Qwen/Qwen3.5-4B " in ln and "REFUSAL_COEFFICIENT=0.3" not in ln, ln
    # --object all requests the unit (the knob is checked); a unit that is not the refusal direction is untouched by the knob
    r, out = run("--object", "all", SISTEER_REFUSAL_COEFFICIENT="0.5")
    assert r.returncode == 2 and "contradicts the models table" in r.stderr and "sbatch" not in out and "DRY-RUN" not in out, out[-3000:]
    r, out = run("--object", "difference_lora", SISTEER_REFUSAL_COEFFICIENT="0.5")
    assert r.returncode == 0 and "contradicts" not in out and "REFUSED" not in out, out[-3000:]
    # source pins: no knob-derived identity remains in 04 — the cell is read through the generator, and the job is handed the cell's coefficient
    src = FRONT.read_text()
    assert 'COEF="$(printf' not in src and 'REFUSAL_STAGED="ar_c${COEF' not in src and "SISTEER_REFUSAL_COEFFICIENT:-0.3" not in src
    assert '--arditi-cell --models-table "$(repro_models_table)" --model-key "$REPRO_MODEL"' in src and 'REFUSAL_KNOB=(--coefficient "$SISTEER_REFUSAL_COEFFICIENT")' in src
    assert 'REFUSAL_DIR="$REFUSAL_ROOT/$SERVE_ID/$REFUSAL_STAGED"' in src and '"REFUSAL_COEFFICIENT=$COEF ' in src
    assert 'case " $UNITS " in *" refusal_direction "*)' in src   # read only when the unit is requested


# ---- the pin is read under the campaign 04 REGISTERED under, never the default alone ------------------------------

def test_the_pin_is_read_under_the_campaign_04_registered_under_never_the_default_alone(tmp_path, capsys, monkeypatch):
    """04 resolves CAMPAIGN="${SISTEER_REGISTRY_CAMPAIGN:-${REGISTRY_CAMPAIGN:-sdfbase_full22}}", hands it to every fit job (job_env
    REGISTRY_CAMPAIGN -> the fitter's --campaign default) and reads each unit's done-ness under it (registry_done's campaign predicate;
    refusal_registry_ids' --arditi-registered-ids --campaign "$CAMPAIGN"), while arditi_pins_from_registry called arditi_registry_rows with no
    campaign and so read sdfbase_full22 alone — a site naming another campaign fitted and registered its refusal direction,
    `--derived-pins-from-registry` and 05's `--pins-from` found no row and wrote no pin, and 05 refused to stage the completed fit. Now every
    registry-backed pins writer and both entry points take `campaign`, resolving None through registry_campaign_default — 04's own resolution —
    main hands ONE resolved value to every mode, 05 names it beside the root, and the campaign a pin was read under is spelled in its source and
    in every `pins:` line."""
    for k in ("SISTEER_REGISTRY_CAMPAIGN", "REGISTRY_CAMPAIGN"):
        monkeypatch.delenv(k, raising=False)
    # the resolution IS 04's (tests/test_hpo_registry_campaign.py's table): the site's, else the job-env spelling, else the paper frame's; empty is unset
    assert GEN.registry_campaign_default() == "sdfbase_full22" == GEN.REGISTRY_CAMPAIGN
    monkeypatch.setenv("REGISTRY_CAMPAIGN", "alpha")
    assert GEN.registry_campaign_default() == "alpha"
    monkeypatch.setenv("SISTEER_REGISTRY_CAMPAIGN", "beta")
    assert GEN.registry_campaign_default() == "beta", "the site's value wins"
    monkeypatch.setenv("SISTEER_REGISTRY_CAMPAIGN", "")
    assert GEN.registry_campaign_default() == "alpha", "an empty value is unset, as bash's ${X:-} reads it"
    monkeypatch.delenv("REGISTRY_CAMPAIGN")
    assert GEN.registry_campaign_default() == "sdfbase_full22"
    # the fit a site registered under ITS campaign: the fitter's row (hooks.emit_object_from_sidecar) filed under paper_frame by 04's job env
    sha = hashlib.sha256(b"a site's own refusal direction, registered under its campaign").hexdigest()
    reg = _registry(tmp_path, [_arditi_row(sha, campaign="paper_frame")], "site_reg")
    ident = f"pins: {KEY}/{SID} (refusal_direction; campaign paper_frame, served {SERVE}, vector_all/arditi_allpos coefficient 0.3)"
    entry = {"object_sha256": sha, "source": f"registry:paper_frame/{SERVE}/vector_all/arditi_allpos coefficient 0.3:reference (not staged yet)",
             "verified_cluster_a": False, "verified_cluster_b": False, "sidecar_object_sha256_recorded": None}
    # named: the pin is written from the row, and the source spells the campaign it was read under
    pins = GEN.arditi_pins_from_registry(FRESH_PINS, registry_root=reg, campaign="paper_frame")
    assert pins[KEY][SID] == entry and f"{ident}: {sha[:12]} (reference) ADDED" in capsys.readouterr().out
    # the default alone — the pre-fix reading: no row under sdfbase_full22, so no pin; right ONLY for a site that names no campaign
    assert GEN.arditi_pins_from_registry(FRESH_PINS, registry_root=reg) == FRESH_PINS
    assert f"pins: {KEY}/{SID} (refusal_direction; campaign sdfbase_full22, served {SERVE}, vector_all/arditi_allpos coefficient 0.3): no reference/approved registry row — no pin" \
        in capsys.readouterr().out
    # the site's environment (repro_load_config exports config.env into every stage's process): BOTH entry points read 04's campaign unprompted,
    # and every writer of the chain reads the same one
    monkeypatch.setenv("SISTEER_REGISTRY_CAMPAIGN", "paper_frame")
    chain = GEN.front_door_pins_from_registry(FRESH_PINS, registry_root=reg, roster=GEN.load_roster())
    out = capsys.readouterr().out
    assert chain[KEY][SID] == entry and f"{ident}: {sha[:12]} (reference) ADDED" in out
    assert "campaign paper_frame" in out and "campaign sdfbase_full22" not in out, out
    assert {k: v for k, v in chain.items() if k != KEY} == FRESH_PINS
    staged = tmp_path / "staged.json"
    staged_from = f"cluster_a:/shared/exp/objects/refusal_direction/{SERVE}/{CELL}"
    staged.write_text(json.dumps([{"target": KEY, "lane": "arditi", "cell_id": CELL, "staged_id": SID, "object_sha256": sha, "status": "staged", "cluster_a_verified": True,
                                   "cluster_b_verified": False, "recorded_hash_check": {"status": "match"}, "source_path": staged_from}]))
    m = GEN.pins_from_staging_manifest(staged, registry_root=reg, roster=GEN.load_roster())
    assert m[KEY][SID] == {"object_sha256": sha, "source": f"registry:paper_frame/{SERVE}/vector_all/arditi_allpos coefficient 0.3:reference (staged from {staged_from})",
                           "verified_cluster_a": True, "verified_cluster_b": False, "sidecar_object_sha256_recorded": "match"}
    # --campaign names it over the environment, for every mode; read under the WRONG campaign, the staged, registered fit refuses as unregistered —
    # the refusal 05's --pins-from met on a site that named a campaign
    assert GEN.pins_from_staging_manifest(staged, registry_root=reg, roster=GEN.load_roster(), campaign="paper_frame") == m
    with pytest.raises(SystemExit, match=f"campaign sdfbase_full22, .*staged \\(manifest digest {sha[:12]}\\) but the registry holds no reference/approved row of this cell"):
        GEN.pins_from_staging_manifest(staged, registry_root=reg, roster=GEN.load_roster(), campaign="sdfbase_full22")
    # 04's done-ness read (--arditi-registered-ids, which 04 runs WITH --campaign "$CAMPAIGN") resolves the same default when none is named, so a
    # by-hand read and the pins writer cannot disagree on where to look; renders nothing, writes nothing
    args = ["--arditi-registered-ids", "--registry-root", str(reg), "--served-model-id", SERVE, "--coefficient", "0.3"]
    capsys.readouterr()   # drain the writers' `pins:` lines above: the CLI's stdout is the one answer 04 acts on, nothing else
    assert GEN.main(args) == 0 and capsys.readouterr().out.strip() == sha
    monkeypatch.delenv("SISTEER_REGISTRY_CAMPAIGN")
    assert GEN.main(args) == 0 and capsys.readouterr().out.strip() == ""
    assert GEN.main(args + ["--campaign", "paper_frame"]) == 0 and capsys.readouterr().out.strip() == sha
    # the source-level pins: main hands ONE resolved campaign to both entry points and the done-ness read; every registry-backed writer takes it and
    # resolves None the same way; the refusal direction's rows are read under it; 05 names it on the generator as it names the root, by 04's line
    src = (REPO / "scripts" / "sdfbase" / "make_full22_configs.py").read_text()
    assert "campaign = args.campaign if args.campaign is not None else registry_campaign_default()" in src
    assert "allow_drop=args.allow_dropping_committed_pins, roster=roster, campaign=campaign)" in src
    assert "pins = front_door_pins_from_registry(load_pins(), registry_root=root, roster=roster, campaign=campaign)" in src
    assert "arditi_registered_ids(root, fit_model_id=args.served_model_id, coefficient=args.coefficient, campaign=campaign)" in src
    writers = ("pins_from_staging_manifest", "derived_pins_from_registry", "base_pins_from_registry", "check_derived_parents", "arditi_pins_from_registry",
               "dose_pins_from_registry", "front_door_pins_from_registry")
    for fn in writers:
        assert re.search(rf"def {fn}\([^)]*?campaign: \"str \| None\" = None[^)]*\)", src), fn
    assert src.count("campaign = campaign if campaign is not None else registry_campaign_default()") == len(writers)
    assert "rows = arditi_registry_rows(registry_root, fit_model_id=serve_id, campaign=campaign)" in src
    assert 'return os.environ.get("SISTEER_REGISTRY_CAMPAIGN") or os.environ.get("REGISTRY_CAMPAIGN") or REGISTRY_CAMPAIGN' in src
    line = 'CAMPAIGN="${SISTEER_REGISTRY_CAMPAIGN:-${REGISTRY_CAMPAIGN:-sdfbase_full22}}"'
    s05 = (REPRO / "05_stage_roster.sh").read_text()
    assert line in s05 and '--pins-from "$MANIFEST" --registry-root "$REG_ROOT" --campaign "$CAMPAIGN"' in s05
    assert line in (REPRO / "04_fit_objects.sh").read_text(), "one resolution, 04's, on both stages"

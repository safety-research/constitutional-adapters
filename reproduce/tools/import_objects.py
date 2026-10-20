#!/usr/bin/env python3
"""reproduce/tools/import_objects.py — the worked example's committed objects (objects/DEMO.json) into a model key's bank, pins and registry.

    python reproduce/tools/import_objects.py cells    --model K [--views v1,v2]              # stage 05's cells.json rows, source committed:<dir>
    python reproduce/tools/import_objects.py pins     --model K --manifest staging_manifest.json [--registry-root R] [--pins-file F] [--dry-run]   # the key's block
    python reproduce/tools/import_objects.py registry --model K [--spool S] [--registry-root R] [--dry-run]     # object rows into a registry

A table row that declares `committed_objects_of` (reproduce/models.yaml; reproduce/lib/models.py committed_model_id) imports the paper's
committed Qwen3.5-4B objects in place of stage 04's fits: for every object cell of the requested views the DEMO.json entry of the cell's
paper label (one training seed, 3, for every fitted intervention; the refusal direction at its operating coefficient), and for a DOSE view the committed
`dose_scale` copy of that entry at the view's dose (objects/MANIFEST.json; the copy's own `dose_scaled_from` stamp names the parent and the
scale). Rows are deduplicated by the STAGED id, so a parent and its dose copy are two rows. Every address is cross-checked through
`sisteer.objects.resolve` (the eb-64 fits are ambiguous without `seed=`).

`cells`: {cell_id, lane, kind, source: committed:objects/<dir>, view_cell, object_sha256_expected (= the object id), object_sha256_expected_source,
seed, fit_model_id, identity_sidecar} — what scripts/sdfbase/full22/stage_objects.py stages from the checkout (no ssh pull) and holds to the id.
`pins`: the key's block of configs/sdfbase_full22_pins.json, written by the ONE pins writer — make_full22_configs.pins_from_staging_manifest
over the stager's manifest and the registry, scoped to this key (`targets`) — so every refusal the spec keeps for a committed row fires here
too (Q12 NARROW: only the campaign='sdfbase_full22' / demo-fit-id lookup is bypassed): a digest with no reference|approved row ("import the
committed objects first"), a withdrawn or superseded row, a difference whose live parents are not its derived_meta.json's, a copy at another
dose or of another parent, a manifest digest that is not the committed id, a committed entry whose identity a row would change. Every planned
staged id must come back pinned to its committed id, else the write refuses by name; every other key of the file is byte-identical
(dump_pins_like_committed). Nothing is written without the staging manifest and a registry on disk.
`registry`: one `object` statement per object through `sisteer.registry.hooks.emit_object_from_sidecar` (the fit writers' own hook), with the
sidecar-validated mapping below and the paper's OWN campaign / object_kind / paper_label / object_role / derivation / status restated from
objects/MANIFEST.json (byte-identical copies keep their object_id; the paper's row stays canonical; never the demo key as fit_model_id, never a
demo campaign). Per object: the hook's dry-run preview first; then, when the registry already holds the row, its IDENTITY columns are read
back (read-only) and compared to the preview — an exact match is `skip (already stated)`, a difference REFUSES naming the column, and no row
means the statement is written for real (a `None` from the guarded hook is a failure). Identity columns: object_id, campaign, object_kind,
paper_label, object_role, fit_target_kind, fit_objective, derivation, fit_model_id, status, parent_object_ids_json, base_scale,
fit_layers_json, fit_layer_index_convention, fit_code_sha.

The mapping (demo-import M1; the owners' rulings Q7 / Q8 / Q12 / Q14): fits are `corpus_document_fit` / `ce` / derivation `none` under the
paper's fit checkpoint (the sidecar's model_id is the HF repo, not the registry id), `fit_corpus_id` the cell's corpus prefix,
`fit_thinking_render not_applicable` (the trainer bypassed the chat template), the layers the hook reads from train_budget_meta.json; the
differences are `derived` / `not_applicable` with `derivation` lora_subtraction | composition, the parents `[parent_pos, parent_neg]` and
`fit_layers_json` from derived_meta.json with `fit_layer_index_convention zero_based_decoder_block` stated beside it (the hook never guesses the
convention of an explicit list); the refusal direction mirrors the adapter-filed row — `contrastive_direction` / `arditi_search`, the SERVED
checkpoint as fit_model_id, `base_scale` = the operating coefficient `fit_provenance.json:arditi_allpos.c`, `fit_layers_json` = [the SERVED
layer] (`served_layer`, which must equal `stored_layer`; `fit_layer` is the fit-time index in another convention and is never converted),
`zero_based_decoder_block`, the capture sets stated as unverified; the dose copies are `derived` / `dose_scale` with their one parent from the
stamp (the hook takes base_scale from the stamp). `status` is `approved` for every one (the identity column the live rows carry; the
statement's provenance names objects/MANIFEST.json), campaigns are the committed ones exactly (never harmonised), `arditi_allpos` is kept as
the distinct live campaign it is. A fit's `fit_effective_batch` is the sidecar's recorded `effective_batch` (64; the hook states it, this
tool cross-checks it against objects/DEMO.json and the preview).

Fit-time provenance: the hook derives `fit_code_sha` / `fit_substrate` / `fit_job_id` / `fit_node_hostname` /
`fit_producer_session` from the EMITTING context unless the caller states them, and the committed directories carry no fit_job_provenance.json
— so every statement passes an explicit `fit_provenance` that states what the committed sidecars record and UNSTATES the rest: the fit job's
id, node, substrate and session are not recorded (None — the column is omitted as the objects_bank adapter omits it, so a registry that holds
the row keeps its value); `fit_code_sha` is STATED `unverified` for the eight fits, differences and dose copies (the schema's spelling for a
fit whose code sha no record carries — the hook writes the field_confidence_json entry the validator requires beside it; the live rows read
`unverified`) and left UNSTATED for the refusal direction (its row reads NULL, as the arditi_allpos adapter files it, and the column is part
of the live cross-check); `fit_finished_utc` is a difference's `derived_meta.json:built_at`,
a copy's own `dose_scaled_from.written_at` (never the copied parent's built_at), the refusal direction's
`fit_provenance.json:arditi_allpos.written_at`, and nothing for a fit (its sidecar records no time). The IMPORT's context stays where it belongs:
the record's `producer` block, and `extra.fit_provenance` names the sources and the rule.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
SCRIPT = "reproduce/tools/import_objects.py"
DEMO_REL = os.path.join("objects", "DEMO.json")
MANIFEST_REL = os.path.join("objects", "MANIFEST.json")
#: the paper objects this repository does not ship (sisteer.objects.WITHHELD_REL): a request for one is refused with its recorded reason
WITHHELD_REL = os.path.join("objects", "WITHHELD.json")
PINS_REL = os.path.join("configs", "sdfbase_full22_pins.json")
#: the one identity sidecar a committed object directory carries (sisteer.eval.train_budget.IDENTITY_SIDECARS, spelled here so the
#: `cells` / `pins` paths stay PyYAML + stdlib; tests hold the two equal)
IDENTITY_SIDECARS = ("train_budget_meta.json", "derived_meta.json", "fit_provenance.json")
#: the columns a committed statement must agree on with a live row (plan M1; Q14's corollary added the two layer columns; fit_code_sha
#: is required too — the eight stated 'unverified' must read the live 'unverified', the unstated refusal direction the live NULL)
IDENTITY_COLUMNS = ("object_id", "campaign", "object_kind", "paper_label", "object_role", "fit_target_kind", "fit_objective", "derivation",
                    "fit_model_id", "status", "parent_object_ids_json", "base_scale", "fit_layers_json", "fit_layer_index_convention", "fit_code_sha")
JSON_COLUMNS = ("parent_object_ids_json", "fit_layers_json")
LAYER_CONVENTION = "zero_based_decoder_block"
#: Q7: the imported statements' status — the identity column every live row carries (the hook's default `reference` would refuse against them)
STATUS = "approved"
#: the derived sidecar's `kind` per registry object_kind (derived_meta.json names the builder's kind)
DERIVED_META_KIND = {"lora_subtraction": "lora_subtraction", "composed": "psr_paired_write"}
#: the refusal direction's capture sets as the adapter files them (the sidecar records no set ids or shas)
ARDITI_CONTRAST_SETS = {"unverified": "the battery record names no capture-set ids or shas"}


class ImportRefused(SystemExit):
    def __init__(self, msg: str):
        super().__init__(f"REFUSED: {msg}")


def _load_models_module(repo: Path):
    reader = repo / "reproduce" / "lib" / "models.py"
    if not reader.is_file():
        raise ImportRefused(f"{reader}: the models table's reader is not in this checkout")
    spec = importlib.util.spec_from_file_location("_import_objects_models", reader)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except OSError as exc:
        raise ImportRefused(f"{path}: cannot read ({exc})")
    except ValueError as exc:
        raise ImportRefused(f"{path}: not JSON ({exc})")


def _sidecar_of(obj_dir: Path) -> str:
    present = [s for s in IDENTITY_SIDECARS if (obj_dir / s).is_file()]
    if len(present) != 1:
        raise ImportRefused(f"{obj_dir}: a committed object directory carries exactly ONE identity sidecar of {list(IDENTITY_SIDECARS)}; "
                            f"found {present or 'none'}")
    return present[0]


class Plan:
    """The committed objects a model key's views import: one row per STAGED id."""

    def __init__(self, key: str, *, views: Optional[List[str]] = None, table: Optional[str] = None, repo: Path = REPO):
        self.repo = Path(repo)
        self.M = _load_models_module(self.repo)
        self.key = key
        self.table = table
        self.m = self.M.model(key, table)
        self.views = list(views) if views else list(self.M.views(key, table))
        for v in self.views:
            if v not in (self.m.get("views") or {}):
                raise ImportRefused(f"{key}: no view {v!r} (views: {sorted(self.m.get('views') or {})})")
        if self.M.committed_objects_of(self.m) is None and not any(self.M.committed_objects_of(self.m, c) for c in (self.m.get("objects") or {})):
            raise ImportRefused(f"{key}: the table row declares no `{self.M.COMMITTED_OBJECTS_KEY}` (neither the row nor a cell) — nothing to import; "
                                "the fit-your-own path stages from the cluster bank (05 without --from-committed)")
        self.demo = _read_json(self.repo / DEMO_REL)
        self.manifest = _read_json(self.repo / MANIFEST_REL)
        if not isinstance(self.demo, dict) or not isinstance(self.demo.get("objects"), list):
            raise ImportRefused(f"{DEMO_REL}: expected {{objects: [...]}}")
        if not isinstance(self.manifest, dict) or not isinstance(self.manifest.get("objects"), list):
            raise ImportRefused(f"{MANIFEST_REL}: expected {{objects: [...]}}")
        self.demo_by_label: Dict[str, dict] = {}
        for e in self.demo["objects"]:
            if e["paper_label"] in self.demo_by_label:
                raise ImportRefused(f"{DEMO_REL}: two entries carry paper_label {e['paper_label']!r} — one fixed object per intervention")
            self.demo_by_label[str(e["paper_label"])] = e
        self.manifest_by_id: Dict[str, dict] = {str(r["object_id"]): r for r in self.manifest["objects"]}
        wpath = self.repo / WITHHELD_REL
        wdoc = _read_json(wpath) if wpath.is_file() else {"objects": []}
        self.withheld: Dict[str, dict] = {str(w["object_id"]): w for w in (wdoc.get("objects") or [])}
        self.rows: List[dict] = self._plan()

    # -- resolution ---------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _withheld_text(w: dict) -> str:
        return (f"object {w['object_id']} ({w.get('model_id')} {w.get('paper_label')}) is withheld from this repository: "
                f"{w.get('reason')} ({WITHHELD_REL})")

    def _manifest_row(self, object_id: str, what: str) -> dict:
        r = self.manifest_by_id.get(object_id)
        if r is None and object_id in self.withheld:
            raise ImportRefused(f"{what}: {self._withheld_text(self.withheld[object_id])}")
        if r is None:
            raise ImportRefused(f"{what}: object {object_id[:12]} is not in {MANIFEST_REL}")
        return r

    def _entry(self, cell: str) -> Tuple[dict, dict, str]:
        """(the DEMO entry, its MANIFEST row, the registry model id) of a view cell's committed object."""
        o = self.M.object_of(self.m, cell)
        label = str(o.get("paper_label") or cell)
        spelling = self.M.committed_objects_of(self.m, cell)
        if spelling is None:
            raise ImportRefused(f"{self.key}/{cell}: neither the cell nor the row declares `{self.M.COMMITTED_OBJECTS_KEY}`")
        model_id = self.M.committed_model_id(spelling, self.table)
        e = self.demo_by_label.get(label)
        if e is None:
            held = [w for w in self.withheld.values() if str(w.get("model_id")) == model_id and str(w.get("paper_label")) == label]
            if held:
                raise ImportRefused(f"{self.key}/{cell}: " + "; ".join(self._withheld_text(w) for w in held))
            raise ImportRefused(f"{self.key}/{cell}: {DEMO_REL} carries no entry with paper_label {label!r} (entries: {sorted(self.demo_by_label)})")
        if str(e.get("model_id")) != model_id:
            raise ImportRefused(f"{self.key}/{cell}: {DEMO_REL}'s {label} is a {e.get('model_id')!r} object but the table's "
                                f"{self.M.COMMITTED_OBJECTS_KEY} {spelling!r} names {model_id!r}")
        row = self._manifest_row(str(e["object_id"]), f"{DEMO_REL}:{label}")
        for k in ("model_id", "paper_label", "object_kind", "derivation"):
            if str(e.get(k)) != str(row.get(k)):
                raise ImportRefused(f"{DEMO_REL}:{label}: {k} {e.get(k)!r} is not {MANIFEST_REL}'s {row.get(k)!r} for {e['object_id'][:12]}")
        return e, row, model_id

    def _dose_copy(self, cell: str, view: str, entry: dict, model_id: str) -> Tuple[dict, dict]:
        """(the MANIFEST row, its stamp) of the committed dose_scale copy of `entry` at the view's dose."""
        vd = (self.m.get("views") or {}).get(view) or {}
        dose = float(vd["dose"])
        label = str(entry["paper_label"])
        hits = []
        for r in self.manifest["objects"]:
            if str(r.get("model_id")) != model_id or str(r.get("paper_label")) != label or str(r.get("derivation")) != "dose_scale" or not r.get("dir"):
                continue
            d = self.repo / "objects" / str(r["dir"])
            if not d.is_dir():
                continue
            stamp = None
            for name in ("derived_meta.json", "train_budget_meta.json"):
                p = d / name
                if p.is_file():
                    doc = _read_json(p)
                    if isinstance(doc, dict) and isinstance(doc.get("dose_scaled_from"), dict):
                        stamp = doc["dose_scaled_from"]; break
            if stamp is None:
                raise ImportRefused(f"{MANIFEST_REL}: {r['object_id'][:12]} ({label}, dose_scale) carries no dose_scaled_from stamp in objects/{r['dir']}")
            if str(stamp.get("parent_object_sha256")) == str(entry["object_id"]) and abs(float(stamp.get("scale")) - dose) < 1e-9:
                hits.append((r, stamp))
        if len(hits) != 1:
            raise ImportRefused(f"{self.key}/{view}/{cell}: {len(hits)} committed dose_scale cop{'y' if len(hits) == 1 else 'ies'} of "
                                f"{DEMO_REL}:{label} ({str(entry['object_id'])[:12]}) at scale {dose:g} in {MANIFEST_REL}"
                                + (": " + ", ".join(r['object_id'][:12] for r, _ in hits) if hits else "") + " — exactly one is the view's copy")
        return hits[0]

    def _plan(self) -> List[dict]:
        out: Dict[str, dict] = {}
        for view in self.views:
            vd = (self.m.get("views") or {}).get(view) or {}
            for cell in [str(c) for c in (vd.get("cells") or [])]:
                if cell == "none" or cell.startswith(self.M.NO_OBJECT_PREFIX):
                    continue
                entry, row, model_id = self._entry(cell)
                o = self.M.object_of(self.m, cell)
                staged = self.M.staged_id(self.m, cell, view)
                if staged in out:
                    continue                                   # dedup by the resolved STAGED id (a parent and its copy are two rows)
                if "dose" in vd:
                    crow, stamp = self._dose_copy(cell, view, entry, model_id)
                    oid, rdir = str(crow["object_id"]), str(crow["dir"])
                    source_note = f"{MANIFEST_REL}:{oid[:12]} (dose {float(vd['dose']):g} copy of {DEMO_REL}:{entry['paper_label']})"
                    parent = str(stamp["parent_object_sha256"])
                    kind = "derived"; lane = staged.split("__", 1)[0]; mrow = crow
                else:
                    oid, rdir = str(entry["object_id"]), str(entry["dir"])
                    source_note = f"{DEMO_REL}:{entry['paper_label']}"
                    parent = None
                    kind = "fit" if str(o.get("kind")) == "fit" else "derived"
                    lane = str(o["lane"]); mrow = row
                obj_dir = self.repo / "objects" / rdir
                if not obj_dir.is_dir():
                    raise ImportRefused(f"{self.key}/{view}/{cell}: objects/{rdir} is not in this checkout (a sparse checkout: "
                                        f"`git sparse-checkout add objects/{rdir}`)")
                sidecar = _sidecar_of(obj_dir)
                out[staged] = {"cell_id": str(o["cell_id"]), "lane": lane, "kind": kind, "source": f"committed:objects/{rdir}", "view_cell": cell,
                               "view": view, "staged_id": staged, "object_sha256_expected": oid, "object_sha256_expected_source": source_note,
                               "seed": entry.get("seed"), "fit_model_id": model_id, "identity_sidecar": sidecar, "paper_label": str(entry["paper_label"]),
                               "dose_parent": parent, "dose": float(vd["dose"]) if "dose" in vd else None, "manifest": mrow, "demo": entry, "dir": rdir}
        if not out:
            raise ImportRefused(f"{self.key}: the views {self.views} name no object cell")
        return list(out.values())

    # -- the three outputs -----------------------------------------------------------------------------------------------------------
    def cells(self) -> List[dict]:
        keys = ("cell_id", "lane", "kind", "source", "view_cell", "object_sha256_expected", "object_sha256_expected_source", "seed", "fit_model_id", "identity_sidecar")
        return [{k: r[k] for k in keys} for r in self.rows]

    def expected_pins(self) -> Dict[str, str]:
        """{staged id: the committed object id} — what the key's pins block must come back as from the validated writer (cmd_pins)."""
        return {r["staged_id"]: r["object_sha256_expected"] for r in self.rows}

    # -- the registry statements --------------------------------------------------------------------------------------------------------
    def statement(self, r: dict) -> Dict[str, Any]:
        """The keyword arguments of hooks.emit_object_from_sidecar for one planned row — sidecar-validated, the paper's columns restated."""
        obj_dir = self.repo / "objects" / r["dir"]
        row = r["manifest"]; entry = r["demo"]
        common = dict(campaign=str(row["campaign"]), object_kind=str(row["object_kind"]), paper_label=str(row["paper_label"]),
                      object_role=str(row["object_role"]), derivation=str(row["derivation"]), fit_model_id=r["fit_model_id"], status=STATUS)
        extra: Dict[str, Any] = {"committed_source": {"manifest": MANIFEST_REL, "demo": DEMO_REL, "dir": f"objects/{r['dir']}",
                                                      "paper_label": r["paper_label"], "view_cell": r["view_cell"], "staged_id": r["staged_id"]}}
        side = r["identity_sidecar"]
        doc = _read_json(obj_dir / side)
        if r["dose"] is not None:
            # a committed dose_scale copy: the stamp is the statement (base_scale, producer); the parent is the DEMO object
            if side != "derived_meta.json" and side != "train_budget_meta.json":
                raise ImportRefused(f"{obj_dir}: a dose copy carries derived_meta.json / train_budget_meta.json, found {side}")
            stamp = doc.get("dose_scaled_from") or {}
            if str(stamp.get("parent_object_sha256")) != str(r["dose_parent"]) or str(r["dose_parent"]) != str(entry["object_id"]):
                raise ImportRefused(f"{obj_dir}: dose_scaled_from.parent_object_sha256 {str(stamp.get('parent_object_sha256'))[:12]} is not the DEMO parent {str(entry['object_id'])[:12]}")
            if common["derivation"] != "dose_scale":
                raise ImportRefused(f"{obj_dir}: {MANIFEST_REL} says derivation {common['derivation']!r} for a dose copy")
            layers = doc.get("layers")
            if not isinstance(layers, list) or not layers:
                raise ImportRefused(f"{obj_dir}: {side} records no `layers` list — the copy's layers are the parent's and must be stated")
            kw = dict(common, fit_target_kind="derived", fit_objective="not_applicable", fit_thinking_render="not_applicable",
                      parent_object_ids_json=[str(r["dose_parent"])], fit_layers_json=[int(x) for x in layers], fit_layer_index_convention=LAYER_CONVENTION,
                      extra=dict(extra, dose_scaled_from_parent=str(r["dose_parent"]), dose=r["dose"], training_seed=entry.get("seed")),
                      fit_provenance=fit_provenance_statement(obj_dir, side, doc, kind="dose_copy"))
            return kw
        if side == "train_budget_meta.json":
            # a corpus-document fit of the paper's fit lane
            if common["derivation"] != "none":
                raise ImportRefused(f"{obj_dir}: a fit with derivation {common['derivation']!r} in {MANIFEST_REL}")
            if str(doc.get("cell_id")) != r["cell_id"]:
                raise ImportRefused(f"{obj_dir}: train_budget_meta.json names cell {doc.get('cell_id')!r}, the table's cell is {r['cell_id']!r}")
            corpus = r["cell_id"].split("__", 1)[0]
            if str(doc.get("si_id")) != corpus:
                raise ImportRefused(f"{obj_dir}: train_budget_meta.json si_id {doc.get('si_id')!r} is not the cell's corpus {corpus!r}")
            if doc.get("chat_template_bypassed") is not True:
                raise ImportRefused(f"{obj_dir}: train_budget_meta.json chat_template_bypassed is {doc.get('chat_template_bypassed')!r}; the fit's thinking render "
                                    "cannot be stated not_applicable")
            seed = doc.get("seed")
            if entry.get("seed") is not None and seed != entry.get("seed"):
                raise ImportRefused(f"{obj_dir}: train_budget_meta.json seed {seed!r} is not {DEMO_REL}'s {entry.get('seed')!r}")
            if not isinstance(doc.get("target_layers"), list):
                raise ImportRefused(f"{obj_dir}: train_budget_meta.json records no target_layers list")
            # the effective batch the fit was run at: the sidecar's recorded `effective_batch` (micro-batch x accum_steps), which the hook states
            # as fit_effective_batch and objects/DEMO.json records per entry — the two must agree (the hook used to state the micro-batch, 2,
            # for these batch-64 fits)
            eb = doc.get("effective_batch")
            if not isinstance(eb, int) or isinstance(eb, bool):
                raise ImportRefused(f"{obj_dir}: train_budget_meta.json records no integer effective_batch ({eb!r}) — the fit's batch cannot be stated")
            if entry.get("effective_batch") is not None and eb != entry.get("effective_batch"):
                raise ImportRefused(f"{obj_dir}: train_budget_meta.json effective_batch {eb} is not {DEMO_REL}'s {entry.get('effective_batch')!r}")
            r["effective_batch"] = eb                                  # the preview's fit_effective_batch is held to it (check_statement_provenance)
            return dict(common, fit_target_kind="corpus_document_fit", fit_objective="ce", fit_thinking_render="not_applicable", fit_corpus_id=corpus,
                        extra=dict(extra, training_seed=seed, fit_replicate_label=(f"s{seed}" if isinstance(seed, int) else None)),
                        fit_provenance=fit_provenance_statement(obj_dir, side, doc, kind="fit"))
        if side == "derived_meta.json":
            # a difference: the composed paired write or the LoRA subtraction, parents and layers from the sidecar
            if str(doc.get("derived_id")) != r["cell_id"]:
                raise ImportRefused(f"{obj_dir}: derived_meta.json derived_id {doc.get('derived_id')!r} is not the table's cell {r['cell_id']!r}")
            want_kind = DERIVED_META_KIND.get(common["object_kind"])
            if want_kind is None or str(doc.get("kind")) != want_kind:
                raise ImportRefused(f"{obj_dir}: derived_meta.json kind {doc.get('kind')!r} is not the {common['object_kind']!r} builder's ({want_kind!r})")
            if common["derivation"] not in ("lora_subtraction", "composition"):
                raise ImportRefused(f"{obj_dir}: {MANIFEST_REL} derivation {common['derivation']!r} is not a difference's")
            parents = []
            for role in ("parent_pos", "parent_neg"):
                pp = doc.get(role)
                if not isinstance(pp, dict) or not pp.get("object_sha256"):
                    raise ImportRefused(f"{obj_dir}: derived_meta.json {role} names no object_sha256")
                parents.append(str(pp["object_sha256"]))
            layers = doc.get("layers")
            if not isinstance(layers, list) or not layers:
                raise ImportRefused(f"{obj_dir}: derived_meta.json records no `layers` list")
            for role in ("parent_pos", "parent_neg"):
                tl = (doc.get(role) or {}).get("target_layers")
                if isinstance(tl, list) and [int(x) for x in tl] != [int(x) for x in layers]:
                    raise ImportRefused(f"{obj_dir}: derived_meta.json layers {layers} are not {role}'s target_layers {tl}")
            return dict(common, fit_target_kind="derived", fit_objective="not_applicable", fit_thinking_render="not_applicable",
                        parent_object_ids_json=parents, fit_layers_json=[int(x) for x in layers], fit_layer_index_convention=LAYER_CONVENTION,
                        extra=dict(extra, training_seed=entry.get("seed"), fit_replicate_label=(f"s{entry['seed']}" if isinstance(entry.get("seed"), int) else None)),
                        fit_provenance=fit_provenance_statement(obj_dir, side, doc, kind="difference"))
        # fit_provenance.json: the refusal direction (the arditi_allpos adapter's statement, mirrored)
        blk = doc.get("arditi_allpos") if isinstance(doc.get("arditi_allpos"), dict) else None
        if blk is None:
            raise ImportRefused(f"{obj_dir}: fit_provenance.json carries no `arditi_allpos` block")
        return dict(common, **arditi_statement(obj_dir, doc, blk, entry, extra), fit_provenance=fit_provenance_statement(obj_dir, side, doc, kind="direction"))

    def previews(self, *, spool: Optional[str], registry_root: Optional[str]):
        """[(row, the hook's dry-run object payload)] — the exact statements a write would make, through the hook."""
        import sisteer.registry.hooks as hooks  # noqa: WPS433 — the registry's own writer
        out = []
        for r in self.rows:
            kw = self.statement(r)
            recs = hooks.emit_object_from_sidecar(str(self.repo / "objects" / r["dir"]), script=SCRIPT, dry_run=True, spool=spool,
                                                  registry_root=registry_root, registry_emit=True, **kw)
            if recs is None:
                raise ImportRefused(f"{self.key}/{r['staged_id']}: the hook's dry run refused (its [registry] notice above names why)")
            objs = [p for t, p in recs if t == "object"]
            if not objs:
                raise ImportRefused(f"{self.key}/{r['staged_id']}: the hook's dry run previewed no object record")
            payload = objs[0]
            check_against_manifest(payload, r["manifest"], f"{self.key}/{r['staged_id']}")
            check_statement_provenance(payload, kw, f"{self.key}/{r['staged_id']}", effective_batch=r.get("effective_batch"))
            out.append((r, kw, payload))
        return out


def arditi_statement(obj_dir: Path, prov: dict, blk: dict, entry: dict, extra: dict) -> Dict[str, Any]:
    """The refusal direction's columns from its fit_provenance.json (Q14 and its corollary): base_scale = the operating coefficient `c`,
    fit_layers_json = [served_layer] where served_layer == stored_layer (the stored block-output index IS the zero-based decoder block the
    adapter files from cell_facts.json:served_layer), never a conversion of `fit_layer` (another convention; no stated rule converts it)."""
    c = blk.get("c", prov.get("c"))
    try:
        c = float(c)
    except (TypeError, ValueError):
        raise ImportRefused(f"{obj_dir}: fit_provenance.json:arditi_allpos.c {c!r} is not a number")
    if entry.get("dose") is not None and abs(float(entry["dose"]) - c) > 1e-9:
        raise ImportRefused(f"{obj_dir}: the operating coefficient c={c:g} is not {DEMO_REL}'s dose {entry['dose']!r}")
    served, stored = blk.get("served_layer"), blk.get("stored_layer")
    for name, v in (("served_layer", served), ("stored_layer", stored)):
        if not isinstance(v, int) or isinstance(v, bool):
            raise ImportRefused(f"{obj_dir}: fit_provenance.json:arditi_allpos.{name} is {v!r}, not an integer — no stated served layer and no stated "
                                "convention to convert fit_layer by; the layer list is never guessed")
    if served != stored:
        raise ImportRefused(f"{obj_dir}: fit_provenance.json:arditi_allpos.served_layer {served} != stored_layer {stored} — the blob's stored layer and the "
                            "served layer disagree, so no single zero-based decoder block can be stated")
    return dict(fit_target_kind="contrastive_direction", fit_objective="arditi_search", base_scale=c, fit_layers_json=[int(served)],
                fit_layer_index_convention=LAYER_CONVENTION, fit_contrast_sets_json=dict(ARDITI_CONTRAST_SETS),
                extra=dict(extra, c=c, fit_layer=blk.get("fit_layer"), served_layer=int(served), stored_layer=int(stored), position_mode=blk.get("positions"),
                           dose_unit_declared=blk.get("dose_unit") or prov.get("dose_unit"), producer=blk.get("producer"), producer_commit=blk.get("producer_commit")))


#: the fit-time provenance columns the hook derives from the EMITTING context when no one states them (hooks.FIT_PROVENANCE_KEYS's first five;
#: tests hold the two in step): the import's host, checkout, job and session are never the fit's. The last four are UNSTATED (the committed
#: sidecars record none; the column is omitted exactly as the objects_bank adapter omits it, so a registry that holds the row keeps its value
#: and a fresh one leaves it NULL). `fit_code_sha` is STATED — the schema's 'unverified', the spelling for a fit whose code sha no record
#: carries, and the hook writes the field_confidence_json entry validate.py requires beside it (hooks.FIT_CODE_SHA_UNVERIFIED_ENTRY) — for
#: the eight fits, differences and dose copies, whose live rows read 'unverified'; the refusal direction's is left unstated (its live row reads
#: NULL, as the arditi_allpos adapter files it). The live cross-check holds every row to its value.
CONTEXT_PROVENANCE_COLUMNS = ("fit_code_sha", "fit_job_id", "fit_node_hostname", "fit_substrate", "fit_producer_session")
#: the `fit_provenance_statement` kinds whose fit_code_sha is stated 'unverified'; every other kind (the refusal direction) leaves it unstated
FIT_CODE_SHA_UNVERIFIED_KINDS = ("fit", "difference", "dose_copy")


def _utc_ms(ts: Any, where: str) -> str:
    """A sidecar's UTC timestamp (`…Z` or `…+00:00`, with or without fractional seconds) in the schema's `YYYY-MM-DDTHH:MM:SS.mmmZ` form
    (the fit_*_utc CHECKs); another offset, or no timestamp, refuses by name — never a local-time guess."""
    from datetime import datetime, timezone
    if not isinstance(ts, str) or not ts:
        raise ImportRefused(f"{where}: no UTC timestamp to state ({ts!r})")
    s = ts.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        raise ImportRefused(f"{where}: {ts!r} is not an ISO-8601 timestamp")
    if dt.tzinfo is None or dt.utcoffset() != timezone.utc.utcoffset(None):
        raise ImportRefused(f"{where}: {ts!r} is not a UTC timestamp (offset {dt.utcoffset()!r}); the registry's fit_*_utc columns are UTC")
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def fit_provenance_statement(obj_dir: Path, side: str, doc: dict, *, kind: str) -> Dict[str, Any]:
    """The explicit `fit_provenance` of one committed object (hooks.apply_fit_provenance's mapping): what the committed sidecar records, and None
    for every fit-time column it does not — the hook would otherwise fill `fit_code_sha` / `fit_substrate` / `fit_job_id` / `fit_node_hostname` /
    `fit_producer_session` from the IMPORTING process (its checkout, host, Slurm job and session), which is the import's provenance, not the
    fit's. Per kind: a fit (train_budget_meta.json) records none of the five and no time; a difference records `built_at`
    (its derivation's finish); a dose copy's stamp records its own `written_at` (never the copied parent's built_at beside it); the
    refusal direction records the rung producer's `written_at`. `fit_code_sha`: a fit, a difference and a
    dose copy STATE the schema's 'unverified' — no committed sidecar records the fit's own code sha, and the hook writes the
    field_confidence_json entry the validator requires beside that spelling (hooks.FIT_CODE_SHA_UNVERIFIED_ENTRY), so the stated row reads as
    the live rows do; the refusal direction leaves it UNSTATED (None) — its sidecar records the rung producer's commit (`producer_commit`,
    kept in extra), never the fit's own code sha, and its live row reads NULL as the arditi_allpos adapter files it. `sources` names each
    stated column's sidecar key, or why a column is stated 'unverified' / left unstated."""
    prov: Dict[str, Any] = {c: None for c in CONTEXT_PROVENANCE_COLUMNS}
    if kind in FIT_CODE_SHA_UNVERIFIED_KINDS:
        prov["fit_code_sha"] = "unverified"
        sources: Dict[str, str] = {"fit_code_sha": f"{side} records no fit code sha; stated 'unverified' (the schema's spelling for a fit whose code sha no "
                                                   "record carries; the hook writes the field_confidence_json entry beside it) — the live rows' value"}
        unstated = [c for c in CONTEXT_PROVENANCE_COLUMNS if c != "fit_code_sha"]
    else:
        sources = {"fit_code_sha": f"{side} records the rung producer's commit (producer_commit, kept in extra), never the fit's own code sha; the "
                                   "column is left unstated (NULL, as the arditi_allpos adapter files the row; the registry keeps a row's value)"}
        unstated = list(CONTEXT_PROVENANCE_COLUMNS)
    finished = None
    if kind == "difference":
        finished = (doc.get("built_at"), "derived_meta.json:built_at")
    elif kind == "dose_copy":
        stamp = doc.get("dose_scaled_from") if isinstance(doc.get("dose_scaled_from"), dict) else {}
        if stamp.get("written_at"):
            finished = (stamp["written_at"], f"{side}:dose_scaled_from.written_at (the copy's own production, never the copied parent's built_at)")
    elif kind == "direction":
        blk = doc.get("arditi_allpos") if isinstance(doc.get("arditi_allpos"), dict) else {}
        if blk.get("written_at"):
            finished = (blk["written_at"], "fit_provenance.json:arditi_allpos.written_at (the rung producer's write)")
    elif kind != "fit":
        raise ImportRefused(f"{obj_dir}: fit_provenance_statement knows no kind {kind!r}")
    if finished is not None:
        ts = _utc_ms(finished[0], f"{obj_dir}: {finished[1]}")
        prov["fit_finished_utc"] = ts
        prov["artifact_created_utc"] = ts
        sources["fit_finished_utc"] = finished[1]
        sources["artifact_created_utc"] = finished[1]
    prov["sources"] = sources
    prov["written_by"] = SCRIPT
    prov["note"] = (f"imported from the paper's committed objects ({MANIFEST_REL}); the fit job's {', '.join(unstated)} are not recorded in the committed "
                    f"sidecars and are unstated — the record's producer block is the IMPORT's context, not the fit's")
    return prov


def check_statement_provenance(payload: dict, kw: dict, where: str, *, effective_batch: Optional[int] = None) -> None:
    """The previewed statement carries no import-context provenance and the fit's batch the sidecar records: the five context columns are
    absent (unstated) or exactly what the explicit fit_provenance states, and a fit's fit_effective_batch is its train_budget_meta.json
    `effective_batch` — a hook that stated the micro-batch again, or the importing host, refuses here before any write."""
    stated = kw.get("fit_provenance") or {}
    for c in CONTEXT_PROVENANCE_COLUMNS:
        want = stated.get(c)
        if want is None and c in payload:
            raise ImportRefused(f"{where}: the hook's preview states {c}={payload[c]!r} from the importing context, which the statement unstates — not written")
        if want is not None and payload.get(c) != want:
            raise ImportRefused(f"{where}: the hook's preview states {c}={payload.get(c)!r}, the statement {want!r} — not written")
        if want == "unverified":
            # the schema's spelling needs its field_confidence_json entry (validate.py refuses the bare literal at harvest): the hook that states
            # the value writes it; a preview without it would be refused at the grading host, so it is refused here, before any write
            fc = payload.get("field_confidence_json")
            entry = fc.get(c) if isinstance(fc, dict) else None
            if not (isinstance(entry, dict) and entry.get("confidence") == "unverified" and str(entry.get("reason", "")).strip()):
                raise ImportRefused(f"{where}: the hook's preview states {c}='unverified' with no field_confidence_json[{c!r}] entry "
                                    f"({{'confidence': 'unverified', 'reason': <non-empty>}}), which the registry would refuse at harvest — not written")
    if effective_batch is not None and payload.get("fit_effective_batch") != effective_batch:
        raise ImportRefused(f"{where}: the hook's preview states fit_effective_batch={payload.get('fit_effective_batch')!r} but train_budget_meta.json records "
                            f"effective_batch {effective_batch} (the micro-batch is resolved_batch_size, not the fit's batch) — not written")


def check_against_manifest(payload: dict, row: dict, where: str) -> None:
    """The previewed statement restates the paper's own columns (objects/MANIFEST.json): a difference refuses by name."""
    for k in ("object_id", "campaign", "object_kind", "paper_label", "object_role", "derivation"):
        if str(payload.get(k)) != str(row.get(k)):
            raise ImportRefused(f"{where}: the statement's {k} {payload.get(k)!r} is not {MANIFEST_REL}'s {row.get(k)!r}")


# -- the live cross-check ----------------------------------------------------------------------------------------------------------------
def _norm(col: str, v: Any) -> Any:
    if v is None:
        return None
    if col in JSON_COLUMNS:
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except ValueError:
                return v
        return [str(x) for x in v] if col == "parent_object_ids_json" else [int(x) for x in v]
    if col == "base_scale":
        return float(v)
    return str(v)


def identity_discrepancies(payload: dict, live: dict) -> List[Tuple[str, Any, Any]]:
    """[(column, live value, committed value)] over IDENTITY_COLUMNS; empty when the statement and the row agree exactly."""
    out = []
    for col in IDENTITY_COLUMNS:
        a, b = _norm(col, live.get(col)), _norm(col, payload.get(col))
        if a != b:
            out.append((col, live.get(col), payload.get(col)))
    return out


def live_identity_row(registry_root: Optional[str], object_id: str) -> Optional[dict]:
    """The registry's object row IDENTITY columns, read-only (mode=ro); None when the registry or the row is absent."""
    if not registry_root:
        return None
    from sisteer.registry import db as DB
    path = DB.db_path(os.path.expanduser(registry_root))
    if not os.path.isfile(path):
        return None
    conn = DB.connect(path, readonly=True)
    try:
        r = conn.execute(f"SELECT {', '.join(IDENTITY_COLUMNS)} FROM object WHERE object_id = ?", (object_id,)).fetchone()
        return dict(zip(IDENTITY_COLUMNS, tuple(r))) if r is not None else None
    finally:
        conn.close()


def registry_root_default() -> str:
    env = os.environ.get("SISTEER_REGISTRY_ROOT")
    if env:
        return os.path.expanduser(env)
    from sisteer.registry import emit
    return os.path.expanduser(emit.HOME_REGISTRY)


def cmd_registry(plan: Plan, *, spool: Optional[str], registry_root: str, dry_run: bool) -> int:
    import sisteer.registry.hooks as hooks
    spool = spool or os.path.join(registry_root, "records")
    rc = 0
    n_skip = n_emit = 0
    for r, kw, payload in plan.previews(spool=spool, registry_root=registry_root):
        oid = str(payload["object_id"])
        layers = payload.get("fit_layers_json"); conv = payload.get("fit_layer_index_convention")
        print(f"preview {plan.key}/{r['staged_id']}: {oid[:12]} campaign={payload.get('campaign')} kind={payload.get('object_kind')} label={payload.get('paper_label')} "
              f"derivation={payload.get('derivation')} fit_model_id={payload.get('fit_model_id')} status={payload.get('status')} base_scale={payload.get('base_scale')} "
              f"fit_layers_json={layers} fit_layer_index_convention={conv} fit_code_sha={payload.get('fit_code_sha')}")
        if layers is None or conv is None:
            raise ImportRefused(f"{plan.key}/{r['staged_id']}: the hook's preview states fit_layers_json={layers!r} / fit_layer_index_convention={conv!r} — "
                                "every committed statement carries both (Q14's corollary)")
        live = live_identity_row(registry_root, oid)
        if live is not None:
            diffs = identity_discrepancies(payload, live)
            if diffs:
                print(f"REFUSED: {plan.key}/{r['staged_id']}: discrepancy " + "; ".join(f"{c}: live {a!r} vs committed {b!r}" for c, a, b in diffs), file=sys.stderr)
                rc = 2
                continue
            print(f"skip (already stated) {plan.key}/{r['staged_id']}: {oid[:12]} — the registry row agrees on every identity column")
            n_skip += 1
            continue
        if dry_run:
            print(f"would emit {plan.key}/{r['staged_id']}: {oid[:12]} (no registry row at {registry_root}; dry run writes nothing)")
            n_emit += 1
            continue
        paths = hooks.emit_object_from_sidecar(str(plan.repo / "objects" / r["dir"]), script=SCRIPT, spool=spool, registry_root=registry_root,
                                               registry_emit=True, **kw)
        if not paths:
            print(f"REFUSED: {plan.key}/{r['staged_id']}: emit_object_from_sidecar wrote nothing (the [registry] notice above names why)", file=sys.stderr)
            rc = 2
            continue
        print(f"emitted {plan.key}/{r['staged_id']}: {oid[:12]} -> {paths[0]} (+{len(paths) - 1})")
        n_emit += 1
    print(f"registry: {len(plan.rows)} committed object(s): {n_skip} already stated, {n_emit} {'to emit' if dry_run else 'emitted'}, "
          f"{len(plan.rows) - n_skip - n_emit} refused; spool {spool}; registry {registry_root}")
    return rc


def _load_gen(repo: Path):
    p = repo / "scripts" / "sdfbase" / "make_full22_configs.py"
    spec = importlib.util.spec_from_file_location("_import_objects_make_full22_configs", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _roster_for(plan: Plan, gen, roster_path: Optional[Path]) -> dict:
    """The roster the dose-pin writer reads the key's dose views from (make_full22_configs.dose_view_cells): the run's roster when it carries
    the key, else the key's block rendered from the models table (lib/models.py roster-block — the block 05 --step roster prints for the
    owner to paste), so a committed dose copy is held to its view's dose BEFORE the roster carries the target, not only after."""
    roster = gen.load_roster(roster_path) if roster_path is not None else gen.load_roster()
    if plan.key in (roster.get("targets") or {}):
        return roster
    import yaml
    full22 = str(plan.repo / "scripts" / "sdfbase" / "full22")
    if full22 not in sys.path:
        sys.path.insert(0, full22)
    import f22_roster as fr  # noqa: WPS433 — the roster's ONE loader (duplicate keys refused)
    block = yaml.load("targets:\n" + plan.M.roster_block(plan.key, plan.views, plan.table), Loader=fr._StrictRosterLoader)["targets"]
    print(f"pins: the roster carries no {plan.key} block yet — its dose views are read from the models table's block (lib/models.py roster-block {plan.key})")
    return dict(roster, targets={**(roster.get("targets") or {}), **block})


def cmd_pins(plan: Plan, *, manifest: Optional[Path], pins_file: Path, registry_root: str, roster: Optional[Path], dry_run: bool) -> int:
    """The key's block through the ONE registry-backed pins writer (make_full22_configs.pins_from_staging_manifest, scoped to the key): the
    stager's manifest rows of the key, the registry's rows for their digests, every Q12 refusal; every planned staged id must come back pinned
    to its committed id; every other key of the file byte-identical. No manifest or no registry on disk: nothing written, said so."""
    if manifest is None:
        raise ImportRefused(f"{plan.key}: the pins are written from the staging manifest (05 --step stage writes <bank>/staging_manifest.json) and the "
                            "registry's rows for its digests — pass --manifest; nothing to hold the pins to, nothing written")
    if not manifest.is_file():
        raise ImportRefused(f"{manifest}: no staging manifest here (05 --step stage writes it) — nothing written")
    gen = _load_gen(plan.repo)
    pins = json.loads(pins_file.read_text(encoding="utf-8")) if pins_file.is_file() else {}
    if not isinstance(pins, dict):
        raise ImportRefused(f"{pins_file}: a pins file is a JSON object keyed by target")
    try:
        full = gen.pins_from_staging_manifest(manifest, registry_root=registry_root, models_table=plan.table, roster=_roster_for(plan, gen, roster),
                                              committed=pins, targets={plan.key})
    except SystemExit as exc:                       # the writer's refusals are SystemExit(<reason>): the same words, this tool's exit code
        if isinstance(exc.code, str) and exc.code:
            raise ImportRefused(f"{plan.key}: the pins writer refused — {exc.code}")
        raise
    block = dict(full.get(plan.key) or {})
    want = plan.expected_pins()
    off = {sid: oid for sid, oid in want.items() if str((block.get(sid) or {}).get("object_sha256") or "") != oid}
    if off:
        raise ImportRefused(f"{plan.key}: {len(off)} of {len(want)} committed object(s) came back unpinned or pinned to another id — "
                            + ", ".join(f"{sid}: {oid[:12]} -> {str((block.get(sid) or {}).get('object_sha256') or 'no pin')[:12]}" for sid, oid in sorted(off.items()))
                            + f" — a pin is registry-backed or not written: import the committed objects first ({SCRIPT} registry --model {plan.key}), "
                            "stage them (05 --step stage), then write the pins; the pins file is unchanged")
    kept = sorted(set(block) - set(want))
    pins[plan.key] = block
    text = gen.dump_pins_like_committed(pins, pins_file)
    tail = (f"{len(want)} committed object(s) pinned under {plan.key}, registry-backed at {registry_root}"
            + (f"; {len(kept)} other committed entr{'y' if len(kept) == 1 else 'ies'} of the key kept: {', '.join(kept)}" if kept else "")
            + "; every other key byte-identical")
    if dry_run:
        print(json.dumps({plan.key: block}, indent=1, sort_keys=True))
        print(f"DRY-RUN: would write {pins_file} ({tail})")
        return 0
    pins_file.write_text(text, encoding="utf-8")
    print(f"wrote {pins_file} ({tail})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python reproduce/tools/import_objects.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("cells", "pins", "registry"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, help="the models-table key (reproduce/models.yaml) whose row declares committed_objects_of")
        p.add_argument("--views", default=None, help="comma list of the model's views (default: every view)")
        p.add_argument("--table", default=None, help="the models table (default reproduce/models.yaml / $SISTEER_MODELS_TABLE)")
        p.add_argument("--repo", default=str(REPO), help="the checkout holding objects/ (default: this file's)")
        if name == "pins":
            p.add_argument("--manifest", type=Path, default=None, help="the stager's staging_manifest.json (05 --step stage) whose rows of the model are pinned — REQUIRED to write")
            p.add_argument("--pins-file", type=Path, default=None, help=f"the pins file to rewrite (default {PINS_REL})")
            p.add_argument("--roster", type=Path, default=None, help="the run's roster the dose views are read from (default configs/sdfbase_full22_roster.yaml; the "
                                                                   "model's block is rendered from the models table while the roster lacks it)")
            p.add_argument("--dry-run", action="store_true", help="validate and print the block, write nothing")
        if name in ("pins", "registry"):
            p.add_argument("--registry-root", default=None, help="the registry root read (default $SISTEER_REGISTRY_ROOT, else the grading host default): the live cross-check "
                                                               "for `registry`, the rows a pin is held to for `pins`")
        if name == "registry":
            p.add_argument("--spool", default=None, help="the record spool (default <registry-root>/records)")
            p.add_argument("--dry-run", action="store_true", help="preview and cross-check every statement; write nothing")
    a = ap.parse_args(argv)
    repo = Path(a.repo).resolve()
    src = str(repo / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    views = [v for v in (a.views or "").split(",") if v] or None
    plan = Plan(a.model, views=views, table=a.table, repo=repo)
    if a.cmd == "cells":
        print(json.dumps(plan.cells(), indent=2))
        return 0
    root = os.path.expanduser(a.registry_root) if a.registry_root else registry_root_default()
    if a.cmd == "pins":
        return cmd_pins(plan, manifest=a.manifest, pins_file=a.pins_file or (repo / PINS_REL), registry_root=root, roster=a.roster, dry_run=a.dry_run)
    return cmd_registry(plan, spool=a.spool, registry_root=root, dry_run=a.dry_run)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ImportRefused as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)

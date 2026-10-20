"""`multiturn_grades` adapter: the graded roots the lane files name → runs with per-goal items.

Roots are enumerated ONLY from the lane files (`transfer_stats_full22_cap4096.json` / `_llama8b.json` `meta.roots`,
the 512 wave files' `meta.namespaces.<ns>.dir`), never by walking `run-state/`. In every root
`<root>/<view>/two_si_<attack>_aggregate.json` (multi-turn, static and BoN aggregates alike), `grading_provenance.json`
and `completion.json` are the inputs; one root holds up to nine objects under one eval and instrument, so the run
identity is (root, aggregate cell, attack, instrument) through :func:`identity.run_id` (S1).

Per (root, cell, attack): a `run` (`generated`) + `run_state judged` (evidence: the root's completion.json), a
`judgment` (the grader's own record: judge model + rubric → the seed judge, `api_pool` from `org_final`,
`grading_code_sha` = the grading worktree head, config sha), `run_metric` asr @0.5 / 0.25 / 0.75 (+ held-in /
held-out) as the aggregate files them (no intervals — the aggregate carries none), `item_result` per goal
(`best_score`, `queries_to_first_success`) and `item_flags` (the goal's at-cap flag). The cap of a 4,096 leg comes
from the lane file's PROVED cap, never the grader's config-derived stamp (its `decode_cap_stamp_note`).
Greedy-512 roots get `status_ruling_id` = the superseding ruling of `multiturn_greedy_512_full22` (emitted here too;
the rulings seed adopts the id). Judge caches (`live_two_si_grade_cache.*.jsonl`) are never read.

Item pointers: the per-goal transcripts live in the store the lane file declares for the root (`meta.roots.<root>.transcripts_root`
+ `private_run`, or the llama lane's `transcripts_run`), whose grading-host copies were mirrored to the clusters and deleted. For every
multi-turn cell this adapter resolves `<private_run>/<attack>/<cell>` through :mod:`transcript_mirrors` — the grading-host copy when one
still exists (hashed), else `transcripts_mirrors/index.json` + `grading_host_file_sha256.tsv` (both inputs of the `_lane_files`
group) — and files ONE `store_root` artifact per cell (both clusters in `locations_json`, every constituent file's cluster relpath
and sha in `extra.files`) that each goal's `item_flags` names as `generation_artifact_id` (`generation_offset` null: the goal's shard
is recorded only inside the files). A declared store with neither a grading-host copy nor an index entry REFUSES the cell's items by name;
a root whose lane names no store (the 512 wave namespaces) keeps its items with a null pointer and the reason on the row;
static / BoN items are outside this step. The run's `store_uri` stays the grades root; `extra.transcript_store_uri` names the cell's
cluster directory.

Declared roots (EMISSION_VERSION 4): a graded root no lane file names is
invisible to the enumeration above, so the registry declares such roots itself in `configs/registry/multiturn_declared_roots.yaml`
(:data:`DECLARED_ROOTS_FILE`, read as a sha'd input of the `_lane_files` group -- a change to it republishes the declared groups; never an
edit on the mirror). A declaration names the root, its object keys and attacks, the PROVED cap (with its proof), the served cluster, the
transcript store, and an `attestation` block. A root a lane file ALSO names is refused by name (one owner per root). The store's
`grading_provenance.json` carries the grader's `attacker_identity_check.<attack>`: a `harness_check` (REFUSED when every shard's sidecar
records `attacker_quantization 'unrecorded'`, a pre-cutover attacker pool) and an `independent_check` (UNIFORM when the sidecars agree and
the config's attacker block attests the value elsewhere). A lane-file root's rows never looked at this block, and still do not (the
producer's lane file is their attestation). A DECLARED root with a REFUSED harness check is filed ONLY when its declaration carries an
`attestation` of kind `independent` whose `ruling_id` is a ruling ON THE RECORD LOG (`<registry root>/rulings/`, :func:`ruling_on_record_log`)
and whose `attacker_quantization` is what the store's UNIFORM independent check attests; otherwise every row of that (root, attack) is
refused by name with the reason. Every row filed under the ruling carries `extra.attacker_attestation` (source `independent`, the ruling
id, the harness refusal, the declaration's evidence) and the sidecar-unrecorded attacker fields' `field_confidence_json` entries marked
`attestation: attested_independently` (confidence `derived`: the value is read off documented evidence, not the producer's sidecar).
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import re
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from .. import identity as I
from .. import site_aliases as SA
from . import declared_common as DC
from . import lineage_common as LC
from .base import AdapterContext, AdapterRefusal, Input, RecordBatch, SourceStateRow, check_input, register, store_glob, store_walk
from .runner import source_id_for
from .transcript_mirrors import TranscriptMirrors, local_run_dir_pointer

NAME = "multiturn_grades"
#: the registry's own declaration of graded roots no lane file names (module docstring); absent = no declared roots
DECLARED_ROOTS_FILE = DC.repo_path("configs", "registry", "multiturn_declared_roots.yaml")
DECLARED_ROOTS_SCHEMA = "multiturn_declared_roots/1"
#: the attacker fields a pre-cutover sidecar leaves `unrecorded` and an independent attestation supplies (strength_mtdiff.ATTESTED_FIELDS)
ATTESTED_ATTACKER_FIELDS = ("attacker_quantization", "attacker_prefix_caching", "attacker_kv_cache_dtype")
ATTESTED_INDEPENDENTLY = "attested_independently"
TRANSFER_STATS_FILES = ("transfer_stats_full22_cap4096.json", "transfer_stats_full22_llama8b.json", "transfer_stats_full22_all.json",
              "transfer_stats_full22_prompted.json", "transfer_stats_full22_glm.json", "transfer_stats_full22_halfdose.json",
              "transfer_stats_nemotron_complete.json")
MT = ("crescendo", "pair", "actorattack")
STATIC = ("pasttense", "codeattack", "renellm", "grok420")
MT_INSTRUMENT_4096 = {"crescendo": "multiturn_crescendo_cap4096_generation_config", "pair": "multiturn_pair_cap4096_generation_config",
                      "actorattack": "multiturn_actorattack_cap4096_generation_config"}
_AGG_RE = re.compile(r"^two_si_(?P<attack>[a-z0-9_]+)_aggregate\.json$")


def _root_entry(view: Optional[str], lane: Optional[str], cap_default: Optional[int]) -> Dict[str, Any]:
    return {"view": view, "caps": {}, "served_cluster": None, "lane": lane, "cap_default": cap_default,
            "transcripts_root": None, "private_run": None, "declaration": None}


def ruling_on_record_log(registry_root: str, ruling_id: str) -> Optional[str]:
    """The record-log file under ``<registry_root>/rulings/`` that carries a ``ruling`` record with this id (the rulings the CLI added
    under ``rulings/added/`` and the round seeds under ``rulings/r2_v*/``, plain or gzipped), or None when the log has none. The log is
    the authoritative source (a ruling reaches the ``ruling`` table only through it), so the database is never consulted here."""
    base = os.path.join(registry_root, "rulings")
    if not os.path.isdir(base):
        return None
    for dirpath, dirnames, filenames in store_walk(base):
        dirnames.sort()
        for fn in sorted(filenames):
            if not (fn.endswith(".jsonl") or fn.endswith(".jsonl.gz")):
                continue
            p = os.path.join(dirpath, fn)
            opener = gzip.open if fn.endswith(".gz") else open
            try:
                with opener(p, "rt", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(rec, dict) or rec.get("_header"):
                            continue
                        payload = rec.get("payload") if isinstance(rec.get("payload"), dict) else rec
                        if rec.get("record_type", "ruling") == "ruling" and payload.get("ruling_id") == ruling_id:
                            return p
            except OSError:
                continue
    return None


def load_declared_roots(path: str) -> Tuple[Input, List[Dict[str, Any]]]:
    """:data:`DECLARED_ROOTS_FILE` -> (its hashed Input, the root declarations); refused by name when it is not the schema this reader
    knows or a declaration lacks a field the read needs (root, view_legacy, attacks, object_keys, decode_cap_tokens)."""
    import yaml
    inp = check_input(path, role="report")
    with open(inp.path, "r", encoding="utf-8") as fh:
        try:
            doc = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise AdapterRefusal(path, f"not YAML: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema") != DECLARED_ROOTS_SCHEMA:
        raise AdapterRefusal(path, f"schema {(doc or {}).get('schema') if isinstance(doc, dict) else None!r} is not {DECLARED_ROOTS_SCHEMA!r}")
    out: List[Dict[str, Any]] = []
    for i, d in enumerate(doc.get("roots") or []):
        if not isinstance(d, dict):
            raise AdapterRefusal(path, f"roots[{i}] is not a mapping")
        missing = [k for k in ("root", "view_legacy", "attacks", "object_keys", "decode_cap_tokens") if d.get(k) in (None, "", [])]
        if missing:
            raise AdapterRefusal(path, f"roots[{i}] ({d.get('root')!r}) lacks {', '.join(missing)}")
        out.append(d)
    return inp, out


def attestation_for(decl: Mapping[str, Any], attack: str, prov: Optional[Mapping[str, Any]], registry_root: str,
                    rulings_seen: Dict[str, Optional[str]]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """The attacker-identity gate of a DECLARED root's (attack) aggregate (module docstring): ``(stamp, None)`` when the rows may be filed --
    the stamp is what every row carries as ``extra.attacker_attestation`` (``{"source": "harness", ...}`` when the grader's own check passed,
    ``{"source": "independent", ...}`` under the ruling) -- or ``(None, why)`` when they are refused. ``rulings_seen`` caches
    :func:`ruling_on_record_log` per ruling id across one read."""
    check = ((prov or {}).get("attacker_identity_check") or {}).get(attack) or {}
    harness = check.get("harness_check") or {}
    status = harness.get("status")
    if status != "REFUSED":
        if not check:
            return None, "the store's grading_provenance.json carries no attacker_identity_check for this attack (a declared root needs the grader's check)"
        return {"source": "harness", "harness_check": status}, None
    att = decl.get("attestation") or {}
    if att.get("kind") != "independent":
        return None, ("the harness attacker-identity check is REFUSED and the declaration carries no `attestation` of kind `independent` "
                      f"(has {att.get('kind')!r})")
    rid = str(att.get("ruling_id") or "")
    if not rid:
        return None, "the harness attacker-identity check is REFUSED and the declaration's attestation names no ruling_id"
    if rid not in rulings_seen:
        rulings_seen[rid] = ruling_on_record_log(registry_root, rid)
    where = rulings_seen[rid]
    if where is None:
        return None, f"the harness attacker-identity check is REFUSED and ruling {rid!r} is not on the record log ({os.path.join(registry_root, 'rulings')})"
    independent = check.get("independent_check") or {}
    if independent.get("status") != "UNIFORM":
        return None, f"the store's independent attacker check is {independent.get('status')!r}, not UNIFORM; the ruling accepts a uniform identity only"
    elsewhere = independent.get("attested_elsewhere") or {}
    attested_q = elsewhere.get("config.attacker.quantization")
    if str(attested_q) != str(att.get("attacker_quantization")):
        return None, (f"the declaration attests attacker_quantization {att.get('attacker_quantization')!r} but the store's independent check "
                      f"attests {attested_q!r}")
    reason = str(harness.get("error") or "")
    short = "REFUSED: attacker_quantization unrecorded" if "attacker_quantization='unrecorded'" in reason else f"REFUSED: {reason[:160]}"
    stamp = {"source": "independent", "ruling_id": rid, "ruling_on_record_log": LC.rel_home(where),
             "attacker_quantization": att.get("attacker_quantization"), "attacker_prefix_caching": att.get("attacker_prefix_caching"),
             "harness_check": short, "harness_error": reason or None, "independent_check": independent.get("status"),
             "n_shards": independent.get("n_shards"), "distinct_recorded_tuples": independent.get("distinct_recorded_tuples"),
             "fields_unrecorded_in_sidecar": list(independent.get("fields_unrecorded_in_sidecar") or []),
             "recorded_identity": independent.get("recorded_identity"), "attested_elsewhere": elsewhere,
             "evidence": list(att.get("evidence") or []), "attestation_note": att.get("harness_check_note")}
    return stamp, None


def _declared_store(rs: str, r: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """(transcript store dir relative to the run-state root, private_run) the lane declares for a grades root: `transcripts_root` +
    `private_run`, or `transcripts_run` = `<transcripts_root>/<private_run>` (the llama lane's spelling). (None, None) when it
    declares no store; (store, None) when it names the store but not the private run (the caller says so on the row)."""
    tr, priv = r.get("transcripts_root"), r.get("private_run")
    if not tr and r.get("transcripts_run"):
        run = LC.expand(str(r["transcripts_run"])).rstrip("/")
        tr, priv = os.path.dirname(run), os.path.basename(run)
    if not tr:
        return None, None
    return os.path.relpath(LC.under(rs, str(tr)), rs), (str(priv) if priv else None)


class MultiturnGradesAdapter:
    name = NAME
    covers_sources = ("multiturn_grades",)
    #: 4 = the registry's declared roots (`configs/registry/multiturn_declared_roots.yaml`) join the enumeration, and a declared root
    #: whose harness attacker-identity check is REFUSED is filed only under the ruling its declaration names, every such row stamped
    #: `extra.attacker_attestation` + `attested_independently` confidence entries (module docstring; the two si-constitution run-2
    #: stores); lane-file roots' bytes do not move. 3 = a fitted object's run
    #: files `is_operating_point` = (nominal == 1.0) and `strength_unit` nominal_only beside its filed nominal (`lineage_common.run_payload_base`,
    #: 2026-09-20; every fitted run had carried NULL there); 2 = the distilled cell's paper label (difference_relu_distilled)
    EMISSION_VERSION = 4
    rng_scope = "none"
    allow_symlinks = False

    def __init__(self, rs: str = LC.RS_DEFAULT, seeds: Optional[LC.Seeds] = None, objects: Optional[LC.ObjectResolver] = None,
                 transfer_stats_files: Sequence[str] = TRANSFER_STATS_FILES, mirrors: Optional[TranscriptMirrors] = None,
                 declared_roots_file: Optional[str] = DECLARED_ROOTS_FILE):
        self.rs = LC.expand(rs)
        self.roots = (self.rs,)
        self._seeds = seeds
        self._objects = objects
        self.transfer_stats_files = tuple(transfer_stats_files)
        self.declared_roots_file = declared_roots_file         # the registry's own declaration of roots no lane file names (None = none)
        self.mirrors = mirrors or TranscriptMirrors(self.rs)   # Former the grading host transcript stores -> cluster copies
        self.refusals = LC.RowRefusals(NAME)
        self.stats: Dict[str, int] = {}
        self._root_meta: Dict[str, Dict[str, Any]] = {}        # root (relative to RS) -> {view, caps, served_cluster, lane, transcripts_root, private_run, declaration}
        self._lane_inputs: List[Input] = []
        self._artifacts_emitted: set = set()
        self._registry_root: Optional[str] = None              # ctx.root of the current read: where the record log (rulings/) lives
        self._rulings_seen: Dict[str, Optional[str]] = {}

    @property
    def seeds(self) -> LC.Seeds:
        if self._seeds is None:
            self._seeds = LC.Seeds.locate()
        return self._seeds

    @property
    def objects(self) -> LC.ObjectResolver:
        if self._objects is None:
            self._objects = LC.ObjectResolver(self.seeds)
        return self._objects

    # ---- roots from the lane files ------------------------------------------------------------
    def _enumerate_roots(self) -> Dict[str, Dict[str, Any]]:
        roots: Dict[str, Dict[str, Any]] = {}
        self._lane_inputs = []
        for name in self.transfer_stats_files:
            p = os.path.join(self.rs, name)
            if not os.path.lexists(p):
                continue
            inp, lane = LC.checked_json(p)
            self._lane_inputs.append(inp)
            meta = lane.get("meta") or {}
            for rname, r in (meta.get("roots") or {}).items():
                root_path = r.get("root") or os.path.join(self.rs, rname)
                rel = os.path.relpath(LC.under(self.rs, root_path), self.rs)
                entry = roots.setdefault(rel, _root_entry(r.get("view"), name, None))
                entry["served_cluster"] = (meta.get("served_cluster_by_root") or {}).get(rname) or r.get("served_cluster") or entry["served_cluster"]
                store, priv = _declared_store(self.rs, r)
                if store:
                    entry["transcripts_root"], entry["private_run"] = store, priv
                if name.endswith("_llama8b.json"):
                    entry["cap_default"] = 4096   # the lane's instrument: BoN and multi-turn both at 4,096 (its meta.decode_caps)
            for ns in (meta.get("namespaces") or {}).values():
                d = ns.get("dir") if isinstance(ns, dict) else None
                if d:
                    roots.setdefault(d, _root_entry(None, name, 512))
            # the PROVED caps per (root, view, cell, attack) from the lane's cell blocks
            model_sections = [lane.get("models") or {}] + [(ns or {}).get("models") or {} for ns in (lane.get("namespaces") or {}).values()]
            for models in model_sections:
                for view, mv in models.items():
                    cells = mv.get("cells") if isinstance(mv, dict) else None
                    if not isinstance(cells, dict):
                        continue
                    for cid, cell in cells.items():
                        if not isinstance(cell, dict):
                            continue
                        for attack, blk in cell.items():
                            if isinstance(blk, dict) and blk.get("root") and blk.get("decode_cap_tokens") is not None:
                                rel = os.path.relpath(LC.under(self.rs, str(blk["root"])), self.rs)
                                e = roots.setdefault(rel, _root_entry(view, name, None))
                                e["caps"][(view, cid, attack)] = int(blk["decode_cap_tokens"])
                                if blk.get("served_cluster"):
                                    e["served_cluster"] = blk["served_cluster"]
        self._add_declared_roots(roots)
        return roots

    def _add_declared_roots(self, roots: Dict[str, Dict[str, Any]]) -> None:
        """The registry's own declaration (module docstring, EMISSION_VERSION 4): every root it names joins the enumeration with its
        declared view, cap (proved by the declaration's evidence, so it is the (view, object key, attack) cap the read uses), served
        cluster and transcript store, and the declaration itself on the entry for the attestation gate. A declared root a lane file also
        names is refused by name: one owner per root."""
        path = self.declared_roots_file
        if not path or not os.path.lexists(path):
            return
        inp, declared = load_declared_roots(path)
        self._lane_inputs.append(inp)
        spelling = f"declared:{os.path.basename(path)}"
        for d in declared:
            rel = os.path.relpath(LC.under(self.rs, str(d["root"])), self.rs)
            if rel in roots:
                raise AdapterRefusal(path, f"declared root {rel!r} is already named by lane file {roots[rel].get('lane')!r}; a declaration is for roots no lane file names")
            cap = int(d["decode_cap_tokens"])
            entry = _root_entry(str(d["view_legacy"]), spelling, cap)
            entry["served_cluster"] = d.get("served_cluster")
            entry["declaration"] = dict(d, _declaration_file=LC.rel_home(path), _declaration_sha256=inp.sha256)
            store, priv = _declared_store(self.rs, d)
            if store:
                entry["transcripts_root"], entry["private_run"] = store, priv
            for cid in d["object_keys"]:
                for attack in d["attacks"]:
                    entry["caps"][(str(d["view_legacy"]), str(cid), str(attack))] = cap
            roots[rel] = entry

    def discover(self) -> List[Input]:
        self._root_meta = self._enumerate_roots()
        if not self._root_meta:
            raise AdapterRefusal(self.rs, "no lane file names a grades root")
        out: List[Input] = list(self._lane_inputs)
        for rel in sorted(self._root_meta):
            root = os.path.join(self.rs, rel)
            if not os.path.isdir(root):
                continue
            for view_dir in sorted(store_glob(os.path.join(root, "*"))):
                if not os.path.isdir(view_dir):
                    continue
                aggs = sorted(store_glob(os.path.join(view_dir, "two_si_*_aggregate.json")))
                if not aggs:
                    continue
                for a in aggs:
                    out.append(check_input(a, role="report"))
                for n in ("grading_provenance.json", "completion.json"):
                    p = os.path.join(view_dir, n)
                    if os.path.lexists(p):
                        out.append(check_input(p, role="completion" if n == "completion.json" else "report"))
        out += self.mirrors.inputs()                                   # The pointer material rides the `_lane_files` group
        return out

    def comparison_group(self, inp: Input) -> str:
        rel = os.path.relpath(inp.uri, self.rs)
        if os.path.dirname(rel) == "" or rel.startswith(".."):
            return "_lane_files"
        parts = rel.split(os.sep)
        if parts[0] == "transcripts_mirrors":
            return "_lane_files"
        return parts[0] if parts[0] != "relu_native" else os.path.join(parts[0], parts[1])

    def source_states(self, ctx: AdapterContext, inputs: Sequence[Input]) -> List[SourceStateRow]:
        rows: List[SourceStateRow] = []
        for inp in inputs:
            if os.path.basename(inp.uri).startswith("two_si_") and inp.uri.endswith("_aggregate.json"):
                with open(inp.path, "r", encoding="utf-8") as fh:
                    cells = sorted(((json.load(fh)).get("cells") or {}).keys())
                rows.append(SourceStateRow(source_id_for(NAME, self.comparison_group(inp)), inp.uri, "$.cells", 1, 1, cells, inp.sha256))
        if inputs and self.refusals.items:
            rows += self.refusals.source_state_rows(source_id_for(NAME, self.comparison_group(inputs[0])), inputs[0].uri, inputs[0].sha256)
        return rows

    # ---- read ---------------------------------------------------------------------------------
    def read(self, ctx: AdapterContext, group: str, inputs: Sequence[Input]) -> Iterator[RecordBatch]:
        if group == "_lane_files":
            yield RecordBatch(source_id_for(NAME, group), group)
            return
        if not self._root_meta:
            self._root_meta = self._enumerate_roots()
        self.refusals = LC.RowRefusals(NAME)
        self._artifacts_emitted = set()
        self._registry_root = ctx.root                 # the record log a declared root's ruling must be on (module docstring)
        self._rulings_seen = {}
        seeds = self.seeds
        producer, now = ctx.producer, ctx.now
        batch = RecordBatch(source_id_for(NAME, group), group)
        meta = self._root_meta.get(group) or _root_entry(None, None, None)
        by_dir: Dict[str, Dict[str, Input]] = {}
        for inp in inputs:
            by_dir.setdefault(os.path.dirname(inp.uri), {})[os.path.basename(inp.uri)] = inp
        n_runs = n_items = n_metrics = 0
        rulings: set = set()
        for view_dir, files in sorted(by_dir.items()):
            view_legacy = os.path.basename(view_dir)
            prov = comp = None
            if "grading_provenance.json" in files:
                with open(files["grading_provenance.json"].path, "r", encoding="utf-8") as fh:
                    prov = json.load(fh)
            if "completion.json" in files:
                with open(files["completion.json"].path, "r", encoding="utf-8") as fh:
                    comp = json.load(fh)
            for fname, inp in sorted(files.items()):
                m = _AGG_RE.match(fname)
                if not m:
                    continue
                attack = m.group("attack")
                with open(inp.path, "r", encoding="utf-8") as fh:
                    agg = json.load(fh)
                r = self._emit_aggregate(batch, group, view_legacy, attack, agg, inp, prov, comp, files.get("completion.json"), meta, producer, now, rulings)
                n_runs += r[0]
                n_items += r[1]
                n_metrics += r[2]
        for rul in sorted(rulings):
            seed = rul.rsplit(":", 1)[1]
            inst = seeds.instruments.get(seed) or {}
            batch.add("ruling", {"ruling_id": rul, "round_id": LC.ROUND_ID, "action": "supersede", "scope_kind": "key",
                                 "scope_json": {"instrument_seed_id": seed}, "target_json": {"superseded_by": "the canonical instrument the seed names"},
                                 "reason": str(inst.get("paper_status_ruling_id") or "superseded instrument (instruments.yaml paper_status)"),
                                 "ruled_utc": now, "ruled_utc_precision": "date", "ruled_by": "the user (instruments.yaml paper_status_ruling_id)",
                                 "source_ref": "configs/registry/instruments.yaml; plan §2.6 / acceptance 4; PR-7's rulings seed adopts this id",
                                 "extra": {"placeholder_for_pr7": True}}, producer, now)
        self.stats[f"runs:{group}"] = n_runs
        self.stats["runs"] = self.stats.get("runs", 0) + n_runs
        self.stats["items"] = self.stats.get("items", 0) + n_items
        self.stats["metrics"] = self.stats.get("metrics", 0) + n_metrics
        self.stats["row_refusals"] = len(self.refusals)
        yield batch

    def _transcript_pointer(self, batch, cid: str, attack: str, meta: Mapping[str, Any], inp: Input, key: str, producer, now):
        """The cell's transcript pointer as ``(pointer, reason, items_refused)``. A pointer means the cell's `store_root` artifact
        is in the batch and every goal's `item_flags` names it; ``(None, reason, False)`` keeps the items with a null pointer and the
        reason on the row (no store declared, not a multi-turn item); ``(None, reason, True)`` is a REFUSAL of the cell's items — the
        lane declares a store that has neither a grading-host copy nor a mirror-index entry."""
        if attack not in MT:
            return None, "not a multi-turn item: static / BoN generations are not in the transcript stores this step relocates", False
        store, priv = meta.get("transcripts_root"), meta.get("private_run")
        if not store:
            return None, "the lane file names no transcript store (transcripts_root / transcripts_run) for this grades root", False
        if not priv:
            return None, f"the lane file names transcript store {store!r} but no private_run, so the cell directory cannot be named", False
        run_dir_rel = f"{priv}/{attack}/{cid}"
        try:
            ptr, why_local = local_run_dir_pointer(self.rs, store, run_dir_rel)
            if ptr is None:
                ptr, why_index = self.mirrors.locate_run_dir(store, run_dir_rel)
                if ptr is None:
                    why = f"{why_local}; {why_index}"
                    self.refusals.add(f"{key}|items", f"per-goal items refused (D12 / D47): {why}", inp.uri)
                    return None, why, True
        except AdapterRefusal as exc:
            self.refusals.add(f"{key}|items", f"per-goal items refused (D12 / D47): {exc.path}: {exc.why}", inp.uri)
            return None, f"{exc.path}: {exc.why}", True
        if ptr.artifact_id not in self._artifacts_emitted:
            batch.add("artifact", ptr.artifact_payload(), producer, now, observe=False)
            self._artifacts_emitted.add(ptr.artifact_id)
        return ptr, None, False

    def _instrument_for(self, attack: str, cap: Optional[int]) -> Tuple[Optional[str], Optional[str]]:
        """(eval_id, instrument seed) for an aggregate's attack at the proved cap."""
        seeds = self.seeds
        if attack in MT:
            return seeds.eval_for_spelling(attack), (MT_INSTRUMENT_4096[attack] if cap == 4096 else "multiturn_greedy_512_full22")
        if attack in STATIC:
            return seeds.eval_for_spelling(attack), ("static_sampled_vendor_preset_4096_8192" if cap == 8192 else "static_greedy_4096_full22")
        if attack == "bon":
            return seeds.eval_for_spelling("bon"), ("bon_512store_4096climb_T1" if (cap or 512) >= 4096 else "bon_512_only_T1")
        return None, None

    def _emit_aggregate(self, batch, group, view, attack, agg, inp, prov, comp, comp_inp, meta, producer, now, rulings) -> Tuple[int, int, int]:
        seeds = self.seeds
        n_runs = n_items = n_metrics = 0
        judge_block = (prov or {}).get("judge") or agg.get("grader_provenance") or {}
        judge_seed = seeds.judge_for_string(f"{judge_block.get('judge_model', '')} / {judge_block.get('rubric', '')}") or "strongreject_v1_haiku45"
        judge_id = seeds.judge_id(judge_seed) if judge_seed in seeds.judges else None
        finished = LC.iso_ms((prov or {}).get("finished_at") or agg.get("written_at"), now)
        written = LC.iso_ms(agg.get("written_at"), finished)
        worktree = ((prov or {}).get("worktree") or {}).get("head")
        pool = str((prov or {}).get("org_final") or judge_block.get("org_initial") or "")
        api_pool = pool if pool in ("lowprio", "highprio", "batch") else None
        cfg = (prov or {}).get("config") or {}
        for cid, cell in (agg.get("cells") or {}).items():
            am = cell.get("attack_metrics") or {}
            cap = meta["caps"].get((view, cid, attack)) or meta.get("cap_default") or LC.int_or_none(cell.get("decode_cap_tokens"))
            if attack in STATIC and cap == 512:
                cap = 4096
            eval_id, inst_seed = self._instrument_for(attack, cap)
            key = f"{group}|{view}|{cid}|{attack}"
            decl, attest_stamp = meta.get("declaration"), None
            if decl is not None:
                # a DECLARED root (module docstring): only the declared (object key, attack) pairs, and the attacker-identity gate
                if str(cid) not in [str(c) for c in decl.get("object_keys") or []] or attack not in [str(a) for a in decl.get("attacks") or []]:
                    self.refusals.add(key, f"declared root {group!r} declares object keys {list(decl.get('object_keys') or [])} x attacks "
                                           f"{list(decl.get('attacks') or [])}; this aggregate's ({cid!r}, {attack!r}) is not declared", inp.uri)
                    continue
                attest_stamp, why = attestation_for(decl, attack, prov, self._registry_root or "", self._rulings_seen)
                if why:
                    self.refusals.add(key, f"attacker identity: {why}", inp.uri)
                    continue
            if eval_id is None or inst_seed is None or inst_seed not in seeds.instruments:
                self.refusals.add(key, f"no eval / instrument for attack {attack!r} at cap {cap}", inp.uri)
                continue
            try:
                inst_row = seeds.instrument_row(inst_seed, eval_id, model_id=seeds.model_id_for_view_legacy(view))
            except KeyError as exc:
                self.refusals.add(key, f"instrument seed unusable: {exc}", inp.uri)
                continue
            obj = self.objects.resolve(view, cid)
            if obj.object_id is None:
                self.refusals.add(key, f"object unresolvable: {obj.source}", inp.uri)
                continue
            pk = LC.producer_key_for_grades_root(group)
            rid = I.run_id(LC.CAMPAIGN, pk, cid, eval_id, str(inst_row["instrument_id"]))
            cluster = SA.canonical(meta.get("served_cluster"))      # a lane file a site wrote with its own spelling
            substrate = ((cluster, "recorded", LC.confidence_entry("recorded", "the lane file's served_cluster for this root")) if cluster in ("cluster_a", "cluster_b")
                         else LC.substrate_from_grades_root(group, inp.uri))
            comp_cell_legacy = cell.get("completion") if isinstance(cell.get("completion"), dict) else {}
            bon_comp = comp_cell_legacy.get("bon") if isinstance(comp_cell_legacy.get("bon"), dict) else {}
            comp_block = (((comp or {}).get("cells") or {}).get(cid) or {})
            gated = (comp_block.get("gated") or {}).get(attack) if isinstance(comp_block.get("gated"), dict) else None
            gated = gated or comp_cell_legacy.get("gated") or comp_cell_legacy or {}
            if not isinstance(gated, dict):
                gated = {}
            attacker_hf = ((gated.get("attacker_engine") or {}).get("attacker_model")) if isinstance(gated, dict) else None
            attacker_model = seeds.model_for_hf_id(attacker_hf) if attacker_hf else None
            n_graded = LC.int_or_none(am.get("n_graded")) or LC.int_or_none(am.get("n_goals")) or LC.int_or_none(bon_comp.get("n_goals"))
            try:
                p = LC.run_payload_base(seeds, run_id=rid, producer_key=pk, object_key=cid, view_legacy=view, cell_id_legacy=cid, obj=obj, eval_id=eval_id,
                                        instrument_row=inst_row, instrument_seed_id=inst_seed, lifecycle_state="generated", when_utc=written,
                                        when_source="the aggregate's written_at (grading time; generation time is in the store, not this root)",
                                        token_cap_tokens=cap, judge_id=judge_id, judge_seed_id=judge_seed, substrate=substrate, lane_source_legacy=NAME,
                                        store_uri=os.path.join(self.rs, group, view), instrument_match="unverified",
                                        extra={"grades_root": group, "attack": attack, "arm": cell.get("arm"), "si_id": cell.get("si_id"),
                                               "grader_stamp_cap": cell.get("decode_cap_tokens"), "lens_reduction_mode": cell.get("lens_reduction_mode"),
                                               "served_alpha": cell.get("served_alpha"), "lane_file": meta.get("lane"), "attacker_hf_id": attacker_hf})
            except KeyError as exc:
                self.refusals.add(key, f"run row: {exc}", inp.uri)
                continue
            if attacker_hf and attacker_model is None:
                p["field_confidence_json"]["attacker_model_id"] = LC.unverified(f"attacker {attacker_hf!r} has no model seed")
            elif attacker_model:
                p["attacker_model_id"] = attacker_model
            if decl is not None:
                # the provenance on the row (module docstring): declared, and how the attacker's identity was established
                p["extra"]["declared_root"] = True
                p["extra"]["declaration_file"] = decl.get("_declaration_file")
                p["extra"]["declaration_sha256"] = decl.get("_declaration_sha256")
                p["extra"]["attacker_attestation"] = attest_stamp
                if attest_stamp and attest_stamp.get("source") == "independent":
                    ruling_id = str(attest_stamp["ruling_id"])          # NOT `rid`: that is the run id every later record names
                    for fld in ATTESTED_ATTACKER_FIELDS:
                        if fld not in attest_stamp.get("fields_unrecorded_in_sidecar", []):
                            continue
                        value = attest_stamp.get(fld)
                        if value is None:
                            p["field_confidence_json"][fld] = LC.unverified(f"the sidecars record no {fld} and the declaration's attestation supplies none")
                            continue
                        p["field_confidence_json"][fld] = dict(LC.confidence_entry(
                            "derived", f"{ATTESTED_INDEPENDENTLY} (ruling {ruling_id}): the sidecars record attacker_quantization 'unrecorded' (a pre-cutover "
                                       f"attacker pool) and the harness check is refused; the value is the declaration's evidence -- the config attacker "
                                       f"block and the attacker serving manifests -- accepted by the ruling"),
                            attestation=ATTESTED_INDEPENDENTLY, ruling_id=ruling_id, value=value)
            if isinstance(gated, dict) and gated.get("n_goals_with_progress") is not None:
                p["n_items_generated"] = LC.int_or_none(gated.get("n_goals_with_progress"))
            neg = agg.get("n_expected_goals")
            p["n_items_planned"] = LC.int_or_none(am.get("n_expected")) or LC.int_or_none(neg.get(cid) if isinstance(neg, dict) else neg)
            p["n_items_judged"] = n_graded
            p["config_path"] = cfg.get("path")
            p["config_sha256"] = cfg.get("sha256") if re.fullmatch(r"[0-9a-f]{64}", str(cfg.get("sha256") or "")) else None
            if comp_inp is not None:
                p["store_completion_artifact_uri"] = comp_inp.uri
                p["store_completion_sha256"] = comp_inp.sha256
            if inst_seed in ("multiturn_greedy_512_full22", "bon_512_only_T1", "static_greedy_4096_full22"):
                rul = LC.supersede_instrument_ruling_id(inst_seed)
                p["status_ruling_id"] = rul
                rulings.add(rul)
            if p["model_id"] == "glm46vflash":
                p["extra"]["superseded_instrument_ruling_id"] = p.get("status_ruling_id")
                p["status_ruling_id"] = LC.DESCOPE_GLM_RULING_ID
            ptr, ptr_reason, items_refused = self._transcript_pointer(batch, cid, attack, meta, inp, key, producer, now)
            p["extra"]["transcript_store"] = meta.get("transcripts_root")
            p["extra"]["transcript_store_uri"] = ptr.uri_primary if ptr else None
            if ptr_reason:
                p["extra"]["transcript_pointer_reason"] = ptr_reason
            if items_refused:
                p["extra"]["items_refused"] = True
            batch.add("run", p, producer, now)
            n_runs += 1
            jid = None
            if judge_id:
                jid = I.judgment_id(rid, judge_id)
                jp = LC.judgment_payload(rid, judge_id, n_graded, finished, os.path.join(self.rs, group, view), api_pool=api_pool,
                                         extra={"judge_seed_id": judge_seed, "pass_id": judge_block.get("pass_id"), "escalation_rule": judge_block.get("escalation_rule"),
                                                "n_429": (prov or {}).get("n_429"), "cache_rows": judge_block.get("cache_rows")})
                jp["judge_cache_kind"] = judge_block.get("cache_kind")
                jp["grading_code_sha"] = worktree if re.fullmatch(r"[0-9a-f]{40}", str(worktree or "")) else "unverified"
                if jp["grading_code_sha"] == "unverified":
                    jp["extra"]["grading_code_sha_reason"] = "grading_provenance.json carries no worktree head"
                jp["grading_config_sha256"] = p["config_sha256"] or "unverified"
                jp["rubric_sha256"] = "unverified"
                jp["started_utc"] = LC.iso_ms((prov or {}).get("started_at"), finished)
                batch.add("judgment", jp, producer, now)
            batch.add("run_state", {"run_id": rid, "state": "judged", "event_utc": finished, "evidence_kind": "completion_artifact",
                                    "evidence_uri": comp_inp.uri if comp_inp is not None else inp.uri}, producer, now, observe=False)
            # metrics
            src = LC.rel_home(inp.uri) + f" ›cells.{cid}.attack_metrics"
            asr = am.get("asr") or {}
            n = n_graded
            if attack == "bon" and am.get("asr_at_n"):
                for N, pval in sorted(am["asr_at_n"].items(), key=lambda kv: int(kv[0])):
                    mid = "asr" if int(N) == 75 else f"asr_at_n{int(N)}"
                    batch.add("run_metric", {"run_id": rid, "judgment_id": jid or "", "metric_id": mid, "threshold": 0.5, "computation_id": "",
                                             "producer_metric_spelling": "asr_at_n", "value": pval, "k": round(pval * n) if n else None, "n": n,
                                             "computed_by": "producer", "computed_utc": written, "source_uri": src, "extra": {"bon_n": int(N)}}, producer, now)
                    n_metrics += 1
            else:
                for t_str, pval in asr.items():
                    t = float(t_str)
                    k = LC.int_or_none(am.get("n_jailbroken")) if t == 0.5 else (round(pval * n) if n else None)
                    batch.add("run_metric", {"run_id": rid, "judgment_id": jid or "", "metric_id": "asr", "threshold": t, "computation_id": "",
                                             "producer_metric_spelling": "asr", "value": pval, "k": k, "n": n,
                                             "n_error_outcomes": LC.int_or_none(am.get("n_errored")), "computed_by": "producer", "computed_utc": written,
                                             "source_uri": src, "extra": {"mean_best_score": am.get("mean_best_score"), "n_capped": am.get("n_capped"),
                                                                          "errored_by_type": am.get("errored_by_type")}}, producer, now)
                    n_metrics += 1
                for grp in ("held_in", "held_out"):
                    g = (am.get("by_category_group") or {}).get(grp) or {}
                    if g.get("n_goals") and g.get("n_jailbroken") is not None:
                        batch.add("run_metric", {"run_id": rid, "judgment_id": jid or "", "metric_id": f"asr_{grp}", "threshold": 0.5, "computation_id": "",
                                                 "producer_metric_spelling": f"by_category_group.{grp}", "value": g["n_jailbroken"] / g["n_goals"],
                                                 "k": int(g["n_jailbroken"]), "n": int(g["n_goals"]), "computed_by": "producer", "computed_utc": written,
                                                 "source_uri": src}, producer, now)
                        n_metrics += 1
            # per-goal items (none when the cell's transcript store resolves nowhere — the refusal above names it)
            per_goal = am.get("per_goal") or {}
            if items_refused:
                self.stats["items_refused"] = self.stats.get("items_refused", 0) + len(per_goal)
                per_goal = {}
            flag_ptr = {"generation_pointer": "cell_store_dir"} if ptr else {"generation_pointer_reason": ptr_reason}
            for goal, gm in sorted(per_goal.items()):
                if not isinstance(gm, dict):
                    continue
                base = {"run_id": rid, "item_id": str(goal), "trial_index": 0, "sample_index": 0, "turn_index": -1, "judgment_id": jid or "",
                        "api_pool": api_pool, "created_utc": written, "judged_utc": finished, "withdrawn_by_revision_id": None}
                batch.add("item_result", dict(base, metric_id="best_score", value=gm.get("best_score"), label=None), producer, now)
                if gm.get("queries_to_first_success") is not None:
                    batch.add("item_result", dict(base, metric_id="queries_to_first_success", value=gm.get("queries_to_first_success"), label=None), producer, now)
                batch.add("item_flags", {"run_id": rid, "item_id": str(goal), "trial_index": 0, "sample_index": 0, "turn_index": -1, "judgment_id": jid or "",
                                         "at_token_cap": 1 if gm.get("capped") else 0, "finish_reason": None, "response_tokens": None, "response_chars": None,
                                         "degenerate_classifier": None, "degenerate_contentless": None, "degenerate_rep_ratio_loop": None,
                                         "degenerate_zlib_collapse": None, "degenerate_union5": None, "loop_first512": None, "rep_ratio": None, "zlib_ratio": None,
                                         "alnum_count": None, "tail_loop": None, "line_loop": None, "decode_seed": None,
                                         "generation_artifact_id": ptr.artifact_id if ptr else None,
                                         "generation_offset": None, "rationale_artifact_id": None, "rationale_offset": None,
                                         "extra": dict({"n_target_forwards": gm.get("n_target_forwards")}, **flag_ptr)}, producer, now)
                n_items += 1
        return n_runs, n_items, n_metrics


def factory() -> MultiturnGradesAdapter:
    return MultiturnGradesAdapter()


register(NAME, factory)

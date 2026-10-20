#!/usr/bin/env python
"""Fold the native-multi-LoRA certification records into two figures, a
markdown table, and W&B runs (plan native-multilora-serving §2.F report
paragraph). Runs on the grading host (matplotlib + wandb); the cluster job
writes JSON only and never holds a W&B key.

  multilora_cert_report.py --records $SISTEER_STATE_DIR/multilora-cert/ \\
      --figs $FIGS_DIR/native-multilora-serving/ [--wandb-project PROJECT | --no-wandb]

Per target (the newest record dir ``<target>_<jobid>`` by default; ``--all``
adds every dir to the TABLE — the figures always draw the newest COMPLETE
record per target and skip incomplete ones, which stay in the table):

* ``head_to_head_tok_s.png`` — lens_stacked vs lora_native decode tok/s at
  B in {8, 32, S}, the ratio labelled on the native bar, lens per_row (B = S)
  as a marker; palette and conventions of ``bench_eager_vs_graphs_plot.py``.
* ``admission_disagreement.png`` — "Greedy-token disagreement (diagnostic)":
  ONE ROW PER TARGET, two panels: left, Δ = native-vs-stacked minus the
  PRIMARY floor (per_row vs stacked) as a point with its 90% prompt-resampling
  band (the 5th and 95th percentiles; the band measures sensitivity to
  resampling the fixed probe set only — no shared-scheduling or between-run
  uncertainty, nominal coverage unvalidated, stated under every table and in
  the figure's footnote) against the m = 0 line, the diagnostic pairs' own
  Δ-vs-primary as lighter bars with thin bands; right, the raw pooled
  disagreement per pair with the primary dashed. The VERDICT word inside the
  axes, and its hue, are the record's OWN rule's verdict — the functional
  rule's for a new record, the earlier three-way rule's for a historical
  readmit — and the footnote states the rule. The long per-record facts live
  in the markdown table, not the figure. Titles are short; nothing squeezes a
  plot.

THE FUNCTIONAL RULE (native-serving-default §1.4, ``functional-2026-09-13``):
a record's verdict is ``admitted`` iff its readiness is attested, its five
positive controls pass, its gate canary passes and it is not slower than the
lens at S; the greedy-token disagreement is a RECORDED DIAGNOSTIC and never an
admission input. The table therefore has NO column that reads as a numerics
verdict: ``verdict`` (the record's own rule's), the four gates, and one
``diagnostics`` column carrying Δ [band], the primary floor, the zero
adapter's concurrent n_div/n, the solo certificate's N × T and the first-token
ulps. Historical schema-4 sidecars render under their own rule beside
schema-5 records (``adm.validate_admission_record`` validates each against
ITS schema), so a MIXED directory renders every record.

PULLED RECORDS (P2c): a record copied off the cluster lacks the server
logs (``*.log``, deliberately not pulled) and its documents name paths under
the ORIGINAL ``<out>/unit/``. The report reads it with the readmit's
semantics: an inventoried log that is absent is ``absent_diagnostic_logs``
(reported in the detail line, never a problem and never a pass), and a path
under the original unit is remapped to the local ``unit/`` (``resolve_path``).

READMITTED RECORDS ARE PREFERRED: when ``unit/admission.readmitted.json`` is
present — the offline re-evaluation ``multilora_admission.py readmit`` writes
beside the job's own ``admission.json``, never over it — and its
``readmitted_from.sha256_of_original_admission_json`` is the sha256 of the
``admission.json`` on disk (same attempt id, valid schema), the table and the
figure read IT and say so, rendering the provenance as ``<rule> readmitted
(module <sha12> @ tree <sha8>[, dirty])`` — never a bare tree sha for a
dirty tree; the job's own verdict and its rule stay in the ``in-job verdict
(rule)`` column, and the ``RESULT.readmitted`` sidecar (when it agrees with
the bound readmit) is preferred over the job's ``RESULT`` line. A readmitted
file that does not bind (the original changed under it, another attempt, a
broken schema) is named under PROBLEMS and never read.

A unit reused from a prior job is drawn under its ORIGINAL job id
with a "measured <date>, job <id>" label — never as this job's measurement.
Every number here is READ from ``unit/admission.json`` / ``unit/head_to_head.json``;
this script computes no verdict of its own.

ONE ATTEMPT PER RECORD: the manifest, the unit's ``identity.json``,
``DONE.json`` and ``admission.json`` each carry the driver attempt's
``attempt_id``, and ``DONE.json`` the sha256 of the admission bytes.
A record whose unit does not bind to its manifest's attempt (or, for a reused
unit, whose records do not agree among themselves under the prior attempt) is
presented as ``unbound`` with the mismatch named — never as its admission's
verdict. Before this a refused rerun into a successful out dir read
``admitted`` here off the earlier attempt's unit.

A REUSED UNIT IS READ AT THE RECORD'S OWN COPY: ``unit_reused_from`` is
provenance only. The accepting attempt copied the source unit verbatim into
``reused_unit/`` and recorded the accepted source ``attempt_id`` and per-file
sha256s in ``unit_reuse.json``; this report binds the copy to those
(``adm.reused_unit_binding_problems``) and never follows the source path,
which a later rerun of the source replaces (a new unit at the old path — a
different verdict, code or probe settings). Before this the report followed
the path and presented the source's NEW verdict under the reusing record.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

HERE = Path(__file__).resolve()
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import multilora_admission as adm  # noqa: E402

# categorical slots 1 (blue) and 2 (orange) of the validated default palette
# (bench_eager_vs_graphs_plot.py): identity = lens path vs native LoRA; the
# per_row context marker rides slot 3 (aqua); status colours only beside a label
HUE = {"lens": "#2a78d6", "native": "#eb6834", "per_row": "#1baf7a"}
#: the W&B purpose: the project is sisteer.wandb_entity.wandb_project(WANDB_PURPOSE) (the site's
#: SISTEER_WANDB_PROJECT_MULTILORA_CERT first) unless --wandb-project names one
WANDB_PURPOSE = "multilora-cert"
SURFACE, TEXT, TEXT2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
TARGET_ORDER = ("gemma4e4b", "nemotron3nano4b", "qwen35_4b", "llama8b", "llama8b_subtraction")


class TargetRecord:
    """One record dir: manifest + the unit's admission and head-to-head (the
    unit's own, or the reused prior's under its provenance)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.manifest: Dict[str, Any] = _read(self.path / "manifest.json") or {}
        self.target: str = str(self.manifest.get("target") or self.path.name.rsplit("_", 1)[0])
        self.job_id: Optional[str] = self.manifest.get("slurm_job_id") or _job_from_name(self.path.name)
        self.reused_from: Optional[str] = self.manifest.get("unit_reused_from")
        # a reused unit lives at THIS record's verbatim copy (adm.REUSED_UNIT_DIR),
        # bound by unit_reuse.json; the source path is provenance, never followed
        unit = self.path / adm.REUSED_UNIT_DIR if self.reused_from else self.path / "unit"
        self.unit_dir = unit
        self.attempt_id: Optional[str] = self.manifest.get(adm.ATTEMPT_ID_KEY)
        admission: Optional[Dict[str, Any]] = _read(unit / "admission.json")
        self.head_to_head: Optional[Dict[str, Any]] = _read(unit / "head_to_head.json")
        self.identity: Dict[str, Any] = _read(unit / "identity.json") or {}
        #: the fp32 oracle reference the unit consumed (its gate prompt count and cap: the canary's scope)
        self.oracle_reference: Dict[str, Any] = _read(unit / "oracle_reference.json") or {}
        self.done = (unit / "DONE.json").is_file()
        self.result = (self.path / "RESULT").read_text().strip() if (self.path / "RESULT").is_file() else None
        #: the readmit's one-line sidecar beside RESULT (P2b), raw
        self.result_readmitted_raw: Optional[str] = ((self.path / adm.RESULT_SIDECAR).read_text().strip()
                                                     if (self.path / adm.RESULT_SIDECAR).is_file() else None)
        #: inventoried server logs absent from this (pulled) record: reported, never a problem (P2c)
        self.absent_diagnostic_logs: List[str] = adm.absent_diagnostic_logs(unit)
        # F1: the unit's records bind to ONE attempt — this record's own for a
        # unit measured here (the manifest's attempt_id), the PRIOR attempt's
        # own for a reused unit (its records agree among themselves and this
        # record's unit_reuse.json names the reuse). An unbound admission is
        # kept for diagnosis but never presented as the verdict.
        self.binding_problems: List[str] = self._bind(unit, admission)
        self.admission_unbound: Optional[Dict[str, Any]] = admission if self.binding_problems else None
        #: the JOB's own admission (bound), whatever rule it was computed under
        self.admission_in_job: Optional[Dict[str, Any]] = None if self.binding_problems else admission
        #: an offline re-evaluation BOUND to the in-job admission's bytes, else None
        self.readmitted, self.readmit_problems = self._load_readmitted(unit, self.admission_in_job)
        #: what the table and the figures read: the bound readmit when present, else the job's own
        self.admission: Optional[Dict[str, Any]] = (self.readmitted if self.readmitted is not None
                                                    else self.admission_in_job)

    @staticmethod
    def _load_readmitted(unit: Path, in_job: Optional[Dict[str, Any]]
                         ) -> "tuple[Optional[Dict[str, Any]], List[str]]":
        """``(readmitted, problems)``: the ``admission.readmitted.json`` under
        ``unit`` when it binds to the in-job admission — its recorded
        ``sha256_of_original_admission_json`` is the sha256 of the
        ``admission.json`` bytes on disk, the attempt ids agree, and the
        record validates — else ``(None, [why])``; ``(None, [])`` when absent
        or when there is no bound in-job admission to bind to."""
        p = unit / adm.READMITTED_ADMISSION_FILE
        if not p.is_file():
            return None, []
        if in_job is None:
            return None, [f"{adm.READMITTED_ADMISSION_FILE} present but the in-job admission is unbound or missing"]
        doc = _read(p)
        if not isinstance(doc, Mapping):
            return None, [f"{adm.READMITTED_ADMISSION_FILE} unreadable"]
        prov = doc.get("readmitted_from")
        if not isinstance(prov, Mapping):
            return None, [f"{adm.READMITTED_ADMISSION_FILE} carries no readmitted_from provenance"]
        try:
            on_disk = hashlib.sha256((unit / "admission.json").read_bytes()).hexdigest()
        except OSError as e:
            return None, [f"admission.json unreadable for the readmit binding: {e}"]
        problems: List[str] = []
        if prov.get("sha256_of_original_admission_json") != on_disk:
            problems.append(f"{adm.READMITTED_ADMISSION_FILE} was computed from admission.json "
                            f"{str(prov.get('sha256_of_original_admission_json'))[:12]}, on disk {on_disk[:12]} — stale")
        if doc.get(adm.ATTEMPT_ID_KEY) != in_job.get(adm.ATTEMPT_ID_KEY):
            problems.append(f"{adm.READMITTED_ADMISSION_FILE} attempt {str(doc.get(adm.ATTEMPT_ID_KEY))[:8]} != "
                            f"the in-job admission's {str(in_job.get(adm.ATTEMPT_ID_KEY))[:8]}")
        problems += [f"{adm.READMITTED_ADMISSION_FILE}: {q}" for q in adm.validate_admission_record(doc)]
        if problems:
            return None, problems
        return dict(doc), []

    def _bind(self, unit: Path, admission: Optional[Dict[str, Any]]) -> List[str]:
        if self.reused_from:
            # the ONE rule for a reused record (a missing copy is a binding problem too)
            return adm.reused_unit_binding_problems(self.path, record_attempt_id=self.attempt_id,
                                                    unit_reused_from=self.reused_from)
        if admission is None:
            return []
        if not self.attempt_id:
            return [f"manifest.json carries no {adm.ATTEMPT_ID_KEY} (a record from before the attempt binding)"]
        return adm.unit_binding_problems(unit, expected_attempt_id=self.attempt_id)

    @property
    def reuse_source(self) -> Dict[str, Any]:
        """``unit_reuse.json``'s accepted ``source`` block (a reused record), else ``{}``."""
        if not self.reused_from:
            return {}
        reuse = _read(self.path / adm.UNIT_REUSE_FILE) or {}
        src = reuse.get("source")
        return dict(src) if isinstance(src, Mapping) else {}

    def resolve_path(self, path: Any) -> Optional[Path]:
        """A path one of this record's unit documents names (the admission's
        ``gate_canary.record``, …) as a LOCAL path: for a reused record, a
        path under the accepted source unit maps into ``reused_unit/`` (A2:
        the copies are byte-verbatim and digest-bound, so references resolve
        at read time instead of being rewritten); otherwise the path itself."""
        if not isinstance(path, str) or not path:
            return None
        if self.reused_from:
            return adm.resolve_reused_path(path, source_unit_dir=self.reuse_source.get("unit_dir"),
                                           copy_dir=self.unit_dir)
        p = Path(path)
        if p.is_file():
            return p
        # a PULLED record (P2c): the document names the ORIGINAL <out>/unit/...; the same relative
        # path under the local unit is the file it means
        return _remap_under_unit(path, self.unit_dir)

    @property
    def gate_record_path(self) -> Optional[Path]:
        """The gate canary record the admission names, resolved locally."""
        return self.resolve_path(((self.admission or {}).get("gate_canary") or {}).get("record"))

    @property
    def measured_job(self) -> Optional[str]:
        """The job that MEASURED the unit: the unit identity's Slurm id, else
        (a unit reused from a prior record) the prior record dir's own
        ``<target>_<jobid>`` name, else this record's job."""
        sid = self.identity.get("slurm_job_id") if self.identity else None
        if sid:
            return str(sid)
        if self.reused_from:
            prior = _job_from_name(Path(self.reused_from).parent.name)
            if prior:
                return prior
        return self.job_id

    @property
    def measured_date(self) -> Optional[str]:
        t = self.identity.get("started_at") or self.manifest.get("started_at")
        if isinstance(t, (int, float)):
            return _dt.datetime.fromtimestamp(float(t), _dt.timezone.utc).strftime("%Y-%m-%d")
        return None

    @property
    def provenance(self) -> str:
        if self.reused_from:
            base = f"measured {self.measured_date or '?'}, job {self.measured_job or '?'} (reused by job {self.job_id})"
        else:
            base = f"job {self.job_id or '?'}" + (f", {self.measured_date}" if self.measured_date else "")
        return base + (" — UNBOUND: the unit is not this record's attempt's" if self.binding_problems else "")

    @property
    def verdict(self) -> str:
        """The verdict the table headlines: the bound readmit's when present, else the job's own."""
        if self.binding_problems:
            return "unbound"
        return str((self.admission or {}).get("verdict") or ("incomplete" if not self.done else "unknown"))

    @property
    def in_job_verdict(self) -> str:
        """The JOB's own verdict, whatever rule it ran under."""
        if self.binding_problems:
            return "unbound"
        return str((self.admission_in_job or {}).get("verdict") or ("incomplete" if not self.done else "unknown"))

    @staticmethod
    def _rule_of(a: Mapping[str, Any]) -> str:
        """The rule a record was computed under: its own ``rule`` field, else the
        schema-2 point rule."""
        return str(a.get("rule") or f"point rule (schema {a.get('schema_version')})")

    @property
    def in_job_rule(self) -> str:
        a = self.admission_in_job or {}
        return self._rule_of(a) if a else "—"

    @property
    def rule(self) -> str:
        """Which rule the headline verdict was computed under, and where: a
        readmit renders ``<rule> readmitted (module <sha12> @ tree <sha8>[, dirty])``
        (P2a) — the module hash is the code that ran; the tree sha only
        names it when the tree was clean."""
        if self.readmitted is not None:
            prov = self.readmitted.get("readmitted_from") or {}
            tree = str(prov.get("tree_sha_of_readmit") or "unknown")[:8]
            module = str(prov.get("module_sha256") or "unknown")[:12]
            dirty = ", dirty" if prov.get("tree_dirty") else ""
            return (f"{self.readmitted.get('rule') or prov.get('rule', adm.ADMISSION_RULE)} readmitted "
                    f"(module {module} @ tree {tree}{dirty})")
        a = self.admission_in_job or {}
        if not a:
            return "—"
        return f"{self._rule_of(a)} in-job"

    @property
    def is_functional(self) -> bool:
        """The headline admission was computed under the functional rule (schema 5)."""
        return bool((self.admission or {}).get("schema_version") == 5)

    @property
    def rule_name(self) -> str:
        """The bare rule the headline verdict was computed under (``functional-2026-09-13`` / ``D111`` / …)."""
        a = self.admission or {}
        return self._rule_of(a) if a else "—"

    @property
    def zero_verdict(self) -> str:
        """The zero adapter's own verdict under the record's rule: functional —
        the SOLO certificate (``exact (solo certificate)`` / ``INEXACT (solo
        certificate)``); the earlier three-way rule — the concurrent legs with
        the failed instrument(s) named and its consequence."""
        a = self.admission or {}
        if not a:
            return "—"
        za = a.get("zero_adapter") or {}
        if not za:
            return "exact" if a.get("zero_adapter_exact") else "INEXACT (schema 2 concurrent legs) -> not admitted"
        if self.is_functional:
            return f"{'exact' if za.get('exact') else 'INEXACT'} (solo certificate)"
        if za.get("exact"):
            return "exact"
        failed = ",".join(za.get("failed_instruments") or ["unmeasured"])
        return f"INEXACT ({failed}) -> {'not admitted' if za.get('definite_refusal', True) else 'inconclusive'}"

    @property
    def readiness_txt(self) -> str:
        """The readiness gate: ``attested`` / ``UNATTESTED (<problem>)`` for a
        functional record; ``— (not an input under <rule>)`` for a historical one."""
        a = self.admission or {}
        if not a:
            return "—"
        rd = a.get("readiness")
        if not isinstance(rd, Mapping):
            return f"— (not an input under {self.rule_name})"
        if rd.get("attested") is True:
            return "attested"
        return f"UNATTESTED ({rd.get('problem') or 'no attestation'})"

    @property
    def controls_txt(self) -> str:
        a = self.admission or {}
        c = a.get("controls") or {}
        if not a:
            return "—"
        return f"{c.get('passed', '—')}/{c.get('of', 5)}" + ("" if c.get("all_passed") else " FAILED")

    @property
    def gate_canary_txt(self) -> str:
        """The gate canary as a gate: passed / FAILED / MISSING, with its scope."""
        a = self.admission or {}
        if not a:
            return "—"
        gc = a.get("gate_canary") or {}
        word = "passed" if gc.get("passed") else str(gc.get("status") or "missing").upper()
        return f"{word} ({self.gate_scope})"

    @property
    def not_slower_txt(self) -> str:
        """``yes (1.40x @ S=107; B=32 1.31x, B=8 1.20x)`` / ``NO (…)`` / ``INVALID …``."""
        a, h = self.admission or {}, self.head_to_head or {}
        if not a:
            return "—"
        per_B = h.get("per_B") or {}
        S = h.get("headline_B")
        ns = (a.get("head_to_head") or {}).get("not_slower_at_S")
        ratio = _fmt((a.get("head_to_head") or {}).get("ratio_at_S", h.get("ratio_at_S")), 2)
        extra = f"; B=32 {_fmt((per_B.get('32') or {}).get('ratio'), 2)}x, B=8 {_fmt((per_B.get('8') or {}).get('ratio'), 2)}x"
        head = "yes" if ns else ("INVALID" if h.get("headline_invalid") else "NO")
        return f"{head} ({ratio}x @ S={S}{extra})"

    @property
    def diagnostics_txt(self) -> str:
        """ONE cell of recorded diagnostics — nothing here is a verdict: Δ vs
        the primary [band], the primary floor, the zero adapter's concurrent
        n_div/n, the solo certificate's N × T, the first-token ulps."""
        a = self.admission or {}
        if not a:
            return "—"
        fl = a.get("floor") or {}
        prs = (a.get("numerics") or {}).get("prompt_resampling_sensitivity") or {}
        solo, conc = _zero_txt(a)
        ul = (a.get("first_token_ulps") or {}).get("native_vs_stacked") or {}
        floor_txt = f"{_fmt(fl.get('disagreement'), 4)} (n_div {fl.get('n_diverging', '—')})"
        if fl.get("degenerate"):
            floor_txt += f" DEGENERATE -> {fl.get('source')}"
        return (f"native vs stacked Δ {_band_txt(prs)} · primary floor {floor_txt} · zero concurrent {conc} · "
                f"solo certificate {solo} · first-token ulps max {_fmt(ul.get('max'))} / p50 {_fmt(ul.get('p50'))}")

    @property
    def result_line(self) -> Optional[str]:
        """The one-line RESULT a reader should quote: the readmit's sidecar
        when it names the SAME rule and verdict as the bound readmit, else the
        job's own RESULT (a stale or unmatched sidecar is a problem)."""
        if self.readmitted is not None and self.result_readmitted_raw:
            fields = dict(kv.split("=", 1) for kv in self.result_readmitted_raw.split() if "=" in kv)
            if fields.get("rule") == self.readmitted.get("rule") and fields.get("verdict") == self.readmitted.get("verdict"):
                return self.result_readmitted_raw
        return self.result

    @property
    def rule_short(self) -> str:
        return "readmitted" if self.readmitted is not None else "in-job"

    @property
    def primary_point_inequality(self) -> Optional[bool]:
        """The point inequality ``native.disagreement <= primary floor``:
        ``numerics.primary_point_inequality`` in a current record; an earlier
        generation called it ``spec_literal_within_floor``; a schema-2 record's
        ``within_floor`` WAS that comparison."""
        a = self.admission or {}
        nm = a.get("numerics") or {}
        for key in ("primary_point_inequality", "spec_literal_within_floor"):
            if key in nm:
                return nm.get(key)
        if a.get("schema_version") == 2:
            return nm.get("within_floor")
        return None

    @property
    def outcome(self) -> str:
        """The three-way comparison outcome (``numerics.outcome``), or the
        verdict for a record that predates it."""
        a = self.admission or {}
        return str((a.get("numerics") or {}).get("outcome") or self.verdict)

    @property
    def gate_scope(self) -> str:
        """The §10 gate canary's SCOPE, read from the record: the oracle
        reference's prompt count and cap, the comparator's margin (the first
        divergent position at a served top-2 margin ≤ the bound)."""
        ref = self.oracle_reference
        n = len(ref.get("prompt_ids") or []) if isinstance(ref.get("prompt_ids"), list) else None
        cap = ref.get("gate_max_tokens")
        gc = (self.admission or {}).get("gate_canary") or {}
        mg = gc.get("margin_atol_nats")
        if n is None and cap is None and mg is None:
            return "—"
        return (f"{n if n is not None else '?'} frozen prompts, {cap if cap is not None else '?'} tokens, "
                f"first fork, {_fmt(mg, 2) if mg is not None else '?'} nats; {gc.get('status') or 'missing'}")

    @property
    def problems(self) -> List[str]:
        out: List[str] = [f"unbound: {p}" for p in self.binding_problems]
        out += [f"readmit: {p}" for p in self.readmit_problems]
        if self.admission is None:
            out.append("no admission.json" if self.admission_unbound is None
                       else "admission.json present but NOT this record's attempt's (see unbound)")
        else:
            out += [f"admission: {p}" for p in adm.validate_admission_record(self.admission)]
        if self.head_to_head is None:
            out.append("no head_to_head.json")
        if not self.done:
            out.append("unit not complete (no DONE.json)")
        if self.admission is not None:
            # the evidence the unit claims: its inventory binds (A2) — with the readmit's semantics
            # for a pulled record (an absent server log is reported, not a problem) — and the gate
            # record the admission names resolves to a LOCAL file (remapped under unit/ when pulled)
            out += [f"inventory: {p}" for p in adm.unit_inventory_problems(self.unit_dir, allow_absent_logs=True)]
            rec = ((self.admission.get("gate_canary") or {}).get("record"))
            if rec and (self.gate_record_path is None or not self.gate_record_path.is_file()):
                out.append(f"gate_canary.record {rec!r} does not resolve to a file under {self.unit_dir}")
        if self.readmitted is not None and self.result_readmitted_raw and self.result_line == self.result:
            out.append(f"{adm.RESULT_SIDECAR} does not name the bound readmit's rule and verdict: "
                       f"{self.result_readmitted_raw!r}")
        return out

    @property
    def complete(self) -> bool:
        """A record with a bound admission and a head-to-head: what the figures draw."""
        return self.admission is not None and self.head_to_head is not None and not self.binding_problems


def _read(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _remap_under_unit(path: str, unit_dir: Path) -> Optional[Path]:
    """A path a pulled record's document names under its ORIGINAL ``…/unit/``
    (or ``…/reused_unit/``), as the same relative path under the local
    ``unit_dir``; ``None`` when the path names no unit."""
    parts = Path(path).as_posix().split("/")
    for marker in ("unit", adm.REUSED_UNIT_DIR):
        if marker in parts:
            idx = len(parts) - 1 - parts[::-1].index(marker)      # the LAST occurrence
            tail = parts[idx + 1:]
            if tail:
                return Path(unit_dir).joinpath(*tail)
    return None


def newest_complete_per_target(records: List["TargetRecord"]) -> List["TargetRecord"]:
    """What the figures draw: the newest COMPLETE record per target (an
    incomplete one — a failed standup with no admission — stays in the table)."""
    newest: Dict[str, TargetRecord] = {}
    for r in records:
        if not r.complete:
            continue
        cur = newest.get(r.target)
        if cur is None or _job_key(r) > _job_key(cur):
            newest[r.target] = r
    order = {k: i for i, k in enumerate(TARGET_ORDER)}
    return sorted(newest.values(), key=lambda r: (order.get(r.target, 99), r.target))


def _job_from_name(name: str) -> Optional[str]:
    tail = name.rsplit("_", 1)[-1]
    return tail if tail.isdigit() else None


def load_records(records_dir: Path, *, all_dirs: bool = False) -> List[TargetRecord]:
    """Every ``<target>_<jobid>`` dir under ``records_dir`` (newest job per
    target unless ``all_dirs``), ordered by :data:`TARGET_ORDER`."""
    recs: List[TargetRecord] = []
    for p in sorted(Path(records_dir).iterdir()):
        if p.is_dir() and (p / "manifest.json").is_file():
            recs.append(TargetRecord(p))
    if not all_dirs:
        newest: Dict[str, TargetRecord] = {}
        for r in recs:
            cur = newest.get(r.target)
            if cur is None or _job_key(r) > _job_key(cur):
                newest[r.target] = r
        recs = list(newest.values())
    order = {k: i for i, k in enumerate(TARGET_ORDER)}
    return sorted(recs, key=lambda r: (order.get(r.target, 99), r.target, _job_key(r)))


def _job_key(r: TargetRecord) -> int:
    try:
        return int(r.job_id or 0)
    except ValueError:
        return 0


# --------------------------------------------------------------------------- #
# table                                                                         #
# --------------------------------------------------------------------------- #

def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _band_txt(prs: Mapping[str, Any]) -> str:
    """``+8.26 pp [+1.26, +16.33]`` — Δ and its two one-sided 95% bounds, in percentage points."""
    if not prs or prs.get("delta_point") is None:
        return "—" + (f" ({prs.get('skipped')})" if prs and prs.get("skipped") else "")
    lo, hi = prs.get("delta_ci95_low"), prs.get("delta_ci95_high")
    return (f"{100 * prs['delta_point']:+.2f} pp [{100 * lo:+.2f}, {100 * hi:+.2f}]"
            if lo is not None and hi is not None else f"{100 * prs['delta_point']:+.2f} pp")


def _zero_txt(a: Mapping[str, Any]) -> "tuple[str, str]":
    """``(solo certificate cell, concurrent cell)`` for the zero adapter."""
    if not a:
        return "—", "—"
    za = a.get("zero_adapter") or {}
    sc = za.get("solo_certificate") or {}
    cc = za.get("concurrent") or {}
    if sc or cc:
        q = lambda v: "?" if v is None else v  # noqa: E731 — an unrecorded scope reads "?", never "None"
        solo = (f"{_fmt(sc.get('passed'))} ({q(sc.get('n_prompts'))} × {q(sc.get('max_tokens'))})" if sc else "—")
        conc = (f"{'exact' if za.get('exact') else 'INEXACT'} ({q(cc.get('n_diverging'))}/{q(cc.get('n'))} diverge, "
                f"{_fmt(cc.get('disagreement'), 4)})" if cc else "—")
        return solo, conc
    # schema 2 (concurrent legs decided; no solo scope recorded) / schema 3 (certificate decided)
    if a.get("schema_version") == 3:
        return f"{_fmt(a.get('zero_adapter_exact'))} (D102 certificate)", "—"
    return "—", f"{'exact' if a.get('zero_adapter_exact') else 'INEXACT'} (schema 2, concurrent legs)"


#: the table's header (native-serving-default §1.4): the verdict, the rule it was computed under,
#: the four functional gates, ONE diagnostics column — no column reads as a numerics verdict
TABLE_HEADER = ("| target | rule | verdict | readiness | controls | gate canary | not slower at S (ratio; B=32, B=8) | "
                "diagnostics: native vs stacked Δ [90% band] · primary floor · zero concurrent n_div/n · "
                "solo certificate N × T · first-token ulps | provenance |")


def table(records: List[TargetRecord]) -> str:
    """One row per record: the RULE the headline verdict was computed under
    (the bound readmit's when present, else the job's) and that rule's
    VERDICT; the four functional gates (readiness, controls, gate canary, not
    slower at S with the ratios); one DIAGNOSTICS column (Δ vs the PRIMARY
    floor with its 90% band, the primary floor, the zero adapter's concurrent
    n_div/n, the solo certificate's N × T, the first-token ulps). A historical
    record's verdict is the three-way word of the rule it was computed under;
    its readiness reads "not an input". The detail lines below carry the
    recorded outcome, reasons and in-job verdict as diagnostics."""
    lines = [TABLE_HEADER, "|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        lines.append(
            f"| {r.target} | {r.rule} | **{r.verdict}** | {r.readiness_txt} | {r.controls_txt} | {r.gate_canary_txt} | "
            f"{r.not_slower_txt} | {r.diagnostics_txt} | {r.provenance} |")
    lines.append("")
    for r in records:
        a, h = r.admission or {}, r.head_to_head or {}
        low = ((h.get("geometry") or {}).get("lowered")) or {}
        fl = a.get("floor") or {}
        dg = fl.get("diagnostics") or {}
        pairs = dg.get("pairs") or {}
        diag_txt = ""
        if pairs:
            cells = []
            for k, v in pairs.items():
                d = v.get("delta_vs_primary") or {}
                cells.append(f"{k}: {_fmt(v.get('disagreement'), 4)}"
                             + (f" (Δ {100 * d['delta_point']:+.2f} pp [{100 * d['delta_ci95_low']:+.2f}, "
                                f"{100 * d['delta_ci95_high']:+.2f}])" if d.get("delta_point") is not None else ""))
            diag_txt = (f" diagnostics={{{'; '.join(cells)}}} native_variability_exceeds_lens="
                        f"{_fmt(dg.get('native_variability_exceeds_lens'))}")
        prov = (r.readmitted or {}).get("readmitted_from") or {}
        ul = (a.get("first_token_ulps") or {}).get("native_vs_stacked") or {}
        za = a.get("zero_adapter") or {}
        diag27 = a.get("diagnostics") or {}
        pl = ((a.get("numerics") or {}).get("prompt_resampling_sensitivity") or {}).get("planning") or {}
        plan_txt = ""
        if pl.get("n_prompts_for_margin"):
            nfm = pl["n_prompts_for_margin"]
            plan_txt = (f" planning (80% power to DETECT an excess of m, one-sided 0.05, conditional approximation; "
                        f"not noninferiority, not the zero adapter): n(5 pp)={nfm.get('0.05')} n(10 pp)={nfm.get('0.1')} "
                        f"sd_influence={_fmt(pl.get('sd_influence'), 3)}")
        gates = a.get("gates") or {}
        gates_txt = (" gates={" + ", ".join(f"{k}={_fmt(gates.get(k))}" for k in adm.FUNCTIONAL_GATES) + "}"
                     if gates else "")
        lines.append(f"{r.target} ({r.path.name}): rule={r.rule} verdict={r.verdict} reasons={a.get('reasons')}"
                     + gates_txt
                     + (f" in-job={r.in_job_verdict} ({r.in_job_rule}) in-job reasons={prov.get('original_reasons')}" if prov else "")
                     + f" | DIAGNOSTIC numerics{' (gate=false)' if r.is_functional else ''}: outcome={r.outcome}"
                     + (f" numerics.reasons={(a.get('numerics') or {}).get('reasons')}" if r.is_functional else "")
                     + f" primary_point_inequality={_fmt(r.primary_point_inequality)}"
                     + (f" zero: {(za.get('solo_certificate') or {}).get('label')}; {(za.get('concurrent') or {}).get('label')};"
                        + (f" instruments_disagree={_fmt(za.get('instruments_disagree'))}" if r.is_functional
                           else f" contract_question_open={_fmt(za.get('contract_question_open'))}") if za else "")
                     + f" gate canary: {r.gate_scope}"
                     + f" ulps native_vs_stacked max={_fmt(ul.get('max'))} p50={_fmt(ul.get('p50'))} (D11)"
                     + f" deterministic native/stacked={_fmt(diag27.get('native_deterministic'))}/"
                       f"{_fmt(diag27.get('stacked_deterministic'))} (D27) geometry={h.get('geometry')}"
                     + (f" LOWERED S {low.get('from')} -> {low.get('to')} ({low.get('reason')})" if low else "")
                     + diag_txt + plan_txt
                     + f" ready_s={h.get('ready_s')} record={r.path}"
                     + (f" RESULT={r.result_line!r}" if r.result_line else "")
                     + (f" absent_diagnostic_logs={r.absent_diagnostic_logs}" if r.absent_diagnostic_logs else "")
                     + (f" PROBLEMS={r.problems}" if r.problems else ""))
    lines.append("")
    lines.append(FUNCTIONAL_NOTE)
    lines.append(D102_NOTE)
    lines.append(adm.BOOTSTRAP_COVERAGE_NOTE)
    lines.append(adm.PLANNING_NOTE)
    return "\n".join(lines)


def figure_footnote() -> str:
    """The admission figure's footnote: the functional rule, the admission
    history and the band's coverage limitation, one after the other."""
    return "\n".join(textwrap.fill(s, width=175) for s in (FUNCTIONAL_NOTE, D102_NOTE, adm.BOOTSTRAP_COVERAGE_NOTE))


#: the ruling every table and figure states (native-serving-default §1.4)
FUNCTIONAL_NOTE = ("Admission is FUNCTIONAL (user ruling 2026-09-13; rule functional-2026-09-13): a standup is admitted "
                   "iff its readiness is attested (graphs on, plugin out, adapters bound by name), its five positive "
                   "controls pass (the solo zero-adapter certificate among them), its fp32-oracle gate canary passes "
                   "and it is not slower than the lens at S. Production decodes by sampling, so the greedy-token "
                   "disagreement shown here is a recorded DIAGNOSTIC and never an admission input; zero-adapter "
                   "exactness is the solo certificate, the concurrent legs' disagreement recorded beside it. "
                   "Historical records keep the verdict of the rule they were computed under (D111's three-way); "
                   "they are not rewritten.")
#: the history every table and figure states
D102_NOTE = ("History: D102 (the floor raised to the native engine's own noise, admission on the lower bound) was "
             "formulated after observing these records and was reversed by the analysis review (D111), whose "
             "three-way rule against the PRIMARY floor at m = 0 is what a 'D111' verdict names; under the "
             "functional rule that comparison is the recorded numerics.outcome.")


# --------------------------------------------------------------------------- #
# figures                                                                       #
# --------------------------------------------------------------------------- #

def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="y", colors=TEXT2, labelsize=8.5, length=0)
    ax.tick_params(axis="x", colors=TEXT2, labelsize=8.5, length=0)


def _mpl():
    """matplotlib on the Agg backend plus the pieces both figures draw with
    (``plt``, ``Patch``, ``Line2D``), or None — the report then stays a table."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError:
        print("matplotlib unavailable; table only")
        return None
    return plt, Patch, Line2D


def figure_head_to_head(records: List[TargetRecord], out_dir: Path) -> Optional[Path]:
    mpl = _mpl()
    if mpl is None:
        return None
    plt, Patch, Line2D = mpl
    records = [r for r in records if r.complete]          # an incomplete record stays in the table only
    if not records:
        return None
    n = len(records)
    fig, axes = plt.subplots(1, n, figsize=(4.9 * n, 4.0), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, r in zip(axes[0], records):
        _style(ax)
        h = r.head_to_head or {}
        per_B = h.get("per_B") or {}
        bs = sorted(int(b) for b in per_B)
        S = h.get("headline_B")
        xs = list(range(len(bs)))
        w = 0.36
        lens = [(per_B[str(b)] or {}).get("lens_stacked_tok_s") or 0.0 for b in bs]
        nat = [(per_B[str(b)] or {}).get("lora_native_tok_s") or 0.0 for b in bs]
        ax.bar([x - w / 2 for x in xs], lens, width=w, color=HUE["lens"], edgecolor=SURFACE, linewidth=1.5,
               label="lens (stacked)")
        ax.bar([x + w / 2 for x in xs], nat, width=w, color=HUE["native"], edgecolor=SURFACE, linewidth=1.5,
               label="lora-native (graphs on)")
        ymax = max(lens + nat + [1.0])
        for x, b in zip(xs, bs):
            ratio = (per_B[str(b)] or {}).get("ratio")
            if ratio is not None:
                ax.text(x + w / 2, nat[x] + ymax * 0.02, f"{ratio:.2f}x", ha="center", va="bottom",
                        fontsize=9, color=TEXT, fontweight=("bold" if b == S else "normal"))
            elif not nat[x]:
                ax.text(x + w / 2, ymax * 0.02, "not run", ha="center", va="bottom", fontsize=8, color=MUTED)
        pr = h.get("lens_per_row_tok_s_at_S")
        if pr is not None and S in bs:
            ax.plot([bs.index(S) - w / 2], [pr], marker="D", markersize=7, color=HUE["per_row"],
                    markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"B = {b}" + ("  (S)" if b == S else "") for b in bs])
        ax.set_ylim(0, ymax * 1.2)
        geom = h.get("geometry") or {}
        low = geom.get("lowered")
        sub = f"L={geom.get('L')} S={geom.get('S')}" + (f" (lowered from {low.get('from')})" if low else "")
        ax.set_title(r.target, fontsize=11, color=TEXT, loc="left", fontweight="bold")
        ax.set_xlabel(f"{sub}  ·  {r.provenance}", fontsize=8, color=TEXT2)
        ax.set_ylabel("decode tok/s (engine counter)", fontsize=9, color=TEXT2)
    handles = [Patch(facecolor=HUE["lens"], label="lens path, stacked reductions (production)"),
               Patch(facecolor=HUE["native"], label="lora-native, CUDA graphs on (label = native / lens)"),
               Line2D([0], [0], marker="D", color=HUE["per_row"], linestyle="none", markersize=7,
                      label="lens per_row at B = S (context)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Decode throughput", fontsize=12, color=TEXT, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "head_to_head_tok_s.png"
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


#: the agreement pairs in drawing order: native pairs first, then the lens pairs
PAIR_ORDER = ("native_vs_stacked", "native_rep_vs_native", "native_zero_vs_none",
              "per_row_vs_stacked", "stacked_rep_vs_stacked", "stacked_compA_vs_compB")
PAIR_LABEL = {"native_vs_stacked": "native\nvs stacked", "native_rep_vs_native": "native rep\nvs native",
              "native_zero_vs_none": "zero\nvs none", "per_row_vs_stacked": "per_row\nvs stacked",
              "stacked_rep_vs_stacked": "stacked rep\nvs stacked", "stacked_compA_vs_compB": "compA\nvs compB"}
NATIVE_PAIRS = {"native_vs_stacked", "native_rep_vs_native", "native_zero_vs_none"}
#: the diagnostic pairs drawn beside the native Δ (their own Δ vs the primary; recorded, never a gate)
DIAG_ORDER = ("native_rep_vs_native", "native_zero_vs_none", "stacked_rep_vs_stacked", "stacked_compA_vs_compB")
VERDICT_HUE = {"admitted": "#1a8f5c", "not_admitted": "#b3261e", "inconclusive": "#b8860b"}
MARGIN_HUE = "#7b3fbf"


def figure_admission(records: List[TargetRecord], out_dir: Path) -> Optional[Path]:
    """One row per COMPLETE target record: left, Δ vs the primary floor with
    its band and the diagnostic pairs; right, the raw pooled disagreement per
    pair. The facts live in the markdown table."""
    mpl = _mpl()
    if mpl is None:
        return None
    plt, Patch, Line2D = mpl
    records = [r for r in records if r.complete]
    if not records:
        return None
    n = len(records)
    fig, axes = plt.subplots(n, 2, figsize=(13.0, 3.9 * n + 1.6), squeeze=False,
                             gridspec_kw={"width_ratios": [1.15, 1.0], "hspace": 0.55, "wspace": 0.28})
    fig.patch.set_facecolor(SURFACE)
    to_pp = lambda v: (100.0 * v) if v is not None else None  # noqa: E731
    for row, r in enumerate(records):
        a = r.admission or {}
        ag = a.get("agreement") or {}
        fl = a.get("floor") or {}
        nm = a.get("numerics") or {}
        prs = nm.get("prompt_resampling_sensitivity") or {}
        dg = (fl.get("diagnostics") or {}).get("pairs") or {}
        m = ((fl.get("margin") or {}).get("m")) or 0.0
        # ---- left: Δ vs the primary floor ----------------------------------------------
        ax = axes[row][0]
        _style(ax)
        xs = list(range(1 + len(DIAG_ORDER)))
        labels = ["native\nvs stacked"] + [PAIR_LABEL[k] for k in DIAG_ORDER]
        pts: List[Optional[float]] = [prs.get("delta_point")]
        los: List[Optional[float]] = [prs.get("delta_ci95_low")]
        his: List[Optional[float]] = [prs.get("delta_ci95_high")]
        for k in DIAG_ORDER:
            d = (dg.get(k) or {}).get("delta_vs_primary") or {}
            pts.append(d.get("delta_point"))
            los.append(d.get("delta_ci95_low"))
            his.append(d.get("delta_ci95_high"))
        for i, k in enumerate(DIAG_ORDER, start=1):
            v = to_pp(pts[i])
            if v is None:
                ax.text(i, 0.0, "not run", ha="center", va="bottom", fontsize=7.5, color=MUTED)
                continue
            hue = HUE["native"] if k in NATIVE_PAIRS else HUE["lens"]
            ax.bar([i], [v], width=0.55, color=hue, alpha=0.35, edgecolor=SURFACE, linewidth=1.2, zorder=2)
            if los[i] is not None and his[i] is not None:
                ax.errorbar([i], [v], yerr=[[max(0.0, v - to_pp(los[i]))], [max(0.0, to_pp(his[i]) - v)]],
                            fmt="none", ecolor=TEXT2, elinewidth=0.9, capsize=3, alpha=0.8, zorder=3)
            ax.annotate(f"{v:+.1f}", (i, v), xytext=(0, 4 if v >= 0 else -11), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.6, color=TEXT2)
        v0 = to_pp(pts[0])
        if v0 is not None and los[0] is not None and his[0] is not None:
            lo0, hi0 = to_pp(los[0]), to_pp(his[0])
            ax.errorbar([0], [v0], yerr=[[max(0.0, v0 - lo0)], [max(0.0, hi0 - v0)]], fmt="o", color=HUE["native"],
                        ecolor=HUE["native"], elinewidth=2.2, capsize=7, markersize=7, zorder=5)
            ax.annotate(f"{v0:+.2f} pp  [{lo0:+.2f}, {hi0:+.2f}]", (0, hi0), xytext=(9, 2), textcoords="offset points",
                        ha="left", va="bottom", fontsize=8.5, color=TEXT, fontweight="bold")
        elif fl.get("degenerate"):
            ax.text(0, 0.0, "primary degenerate\n-> gate canary", ha="center", va="bottom", fontsize=7.8, color=TEXT2)
        else:
            ax.text(0, 0.0, "not run", ha="center", va="bottom", fontsize=7.8, color=MUTED)
        ax.axhline(100.0 * m, color=MARGIN_HUE, linewidth=1.4, linestyle="--", zorder=1)
        vals = [x for x in [to_pp(p) for p in pts] + [to_pp(l) for l in los] + [to_pp(h) for h in his] if x is not None]
        top = max(vals + [100.0 * m, 1.0])
        bot = min(vals + [100.0 * m, 0.0])
        span = max(top - bot, 1.0)
        ax.set_ylim(bot - 0.12 * span, top + 0.5 * span)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8.2)
        ax.set_ylabel("Δ token disagreement vs primary floor (pp)", fontsize=9, color=TEXT2)
        ax.set_title(r.target, fontsize=12, loc="left", color=TEXT, fontweight="bold")
        # the verdict word and its hue are the record's OWN rule's (functional for a new
        # record, the three-way rule for a historical readmit), the rule named beside it
        verdict = r.verdict
        word = {"admitted": "ADMITTED", "not_admitted": "NOT ADMITTED", "inconclusive": "INCONCLUSIVE"}.get(
            verdict, verdict.upper())
        ax.text(0.99, 0.97, f"{word}  ({r.rule_name}, {r.rule_short})", transform=ax.transAxes, ha="right", va="top",
                fontsize=10, fontweight="bold", color=VERDICT_HUE.get(verdict, TEXT))
        if r.is_functional:
            sub = (f"gates: readiness {r.readiness_txt}, controls {r.controls_txt}, canary "
                   f"{'passed' if ((a.get('gate_canary') or {}).get('passed')) else 'FAILED/MISSING'}, "
                   f"not slower {'yes' if (a.get('head_to_head') or {}).get('not_slower_at_S') else 'NO'}\n"
                   f"diagnostic: three-way outcome {r.outcome};  zero {r.zero_verdict}")
        else:
            sub = f"primary: {r.outcome};  zero: {r.zero_verdict}"
        ax.text(0.99, 0.885, sub, transform=ax.transAxes, ha="right", va="top", fontsize=8.2, color=TEXT2)
        if r.readmitted is not None:
            ax.text(0.99, 0.76 if r.is_functional else 0.80, f"in-job: {r.in_job_verdict} ({r.in_job_rule})",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8.2, color=TEXT2)
        # ---- right: the raw pooled disagreement per pair -----------------------------------
        ax2 = axes[row][1]
        _style(ax2)
        keys = list(PAIR_ORDER)
        vals2 = [((ag.get(k) or {}).get("disagreement")) for k in keys]
        hs = [v if v is not None else 0.0 for v in vals2]
        cols = [HUE["native"] if k in NATIVE_PAIRS else HUE["lens"] for k in keys]
        bars = ax2.bar(range(len(keys)), hs, width=0.62, color=cols, edgecolor=SURFACE, linewidth=1.5)
        bars[keys.index("stacked_compA_vs_compB")].set_hatch("//")
        bars[keys.index("per_row_vs_stacked")].set_edgecolor(TEXT2)
        bars[keys.index("per_row_vs_stacked")].set_linewidth(2.0)
        prim = fl.get("disagreement")
        if prim is not None:
            ax2.axhline(prim, color=TEXT2, linewidth=1.0, linestyle="--", zorder=1)
        ymax = max(hs + [prim or 0.0, 1e-3])
        for i, (v, k) in enumerate(zip(vals2, keys)):
            nd = (ag.get(k) or {}).get("n_diverging")
            nnn = (ag.get(k) or {}).get("n")
            lab = "not run" if v is None else f"{v:.3f}\n{nd}/{nnn}"
            ax2.annotate(lab, (i, hs[i]), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                         fontsize=7.8, color=TEXT)
        ax2.set_ylim(0, ymax * 1.4)
        ax2.set_xticks(range(len(keys)))
        ax2.set_xticklabels([PAIR_LABEL[k] for k in keys], fontsize=8.0)
        ax2.set_ylabel("greedy token disagreement\n(1 − pooled agreement)", fontsize=9, color=TEXT2)
        ax2.set_title("Pooled disagreement (diagnostic)", fontsize=10, loc="left", color=TEXT, pad=8)
        ax2.set_xlabel(r.provenance, fontsize=8, color=TEXT2)
    handles = [Line2D([0], [0], color=HUE["native"], marker="o", markersize=7, linewidth=2.2,
                      label="native vs stacked: Δ vs primary, 90% prompt-resampling band (the verdict reads its bounds; "
                            "coverage unvalidated — see the note)"),
               Patch(facecolor=HUE["native"], alpha=0.35, label="native-engine diagnostic pair: its own Δ vs primary (recorded)"),
               Patch(facecolor=HUE["lens"], alpha=0.35, label="lens diagnostic pair: its own Δ vs primary (recorded)"),
               Line2D([0], [0], color=MARGIN_HUE, linestyle="--", label="m = 0 (the recorded outcome's threshold; not a gate)"),
               Patch(facecolor=HUE["native"], label="native-engine pair (pooled)"),
               Patch(facecolor=HUE["lens"], label="lens pair (pooled)"),
               Patch(facecolor=SURFACE, edgecolor=TEXT2, linewidth=2.0, label="the primary floor (per_row vs stacked)"),
               Patch(facecolor=HUE["lens"], hatch="//", edgecolor=SURFACE, label="lens batch-composition pair")]
    legend_h = 1.25 / (3.9 * n + 1.6)
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.36 * legend_h))
    fig.text(0.01, 0.004, figure_footnote(), fontsize=7.4, color=TEXT2, ha="left", va="bottom")
    height_in = 3.9 * n + 1.6
    fig.suptitle("Greedy-token disagreement (diagnostic)", fontsize=13, color=TEXT, x=0.01, y=1 - 0.18 / height_in,
                 ha="left", va="top")
    # the first row's title sits a full title-height below the suptitle: nothing overlaps at any n
    fig.subplots_adjust(left=0.07, right=0.985, top=1 - 0.95 / height_in, bottom=legend_h + 0.045)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "admission_disagreement.png"
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# W&B                                                                           #
# --------------------------------------------------------------------------- #

def _flatten(prefix: str, obj: Any, out: Dict[str, Any]) -> None:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            _flatten(f"{prefix}/{k}" if prefix else str(k), v, out)
    elif isinstance(obj, (int, float, bool, str)) or obj is None:
        out[prefix] = obj


def wandb_log(records: List[TargetRecord], figs: List[Path], *, project: str, entity: Optional[str]) -> List[str]:
    """One W&B run per target (``<target>_<jobid>``): every scalar of
    head_to_head.json and admission.json, both figures as images, and a
    ``figures-<run>`` artifact. Returns the run urls."""
    import wandb
    urls: List[str] = []
    for r in records:
        name = f"{r.target}_{r.job_id or 'nojob'}"
        run = wandb.init(project=project, entity=entity, name=name, reinit=True,
                         config={"target": r.target, "job_id": r.job_id, "unit_reused_from": r.reused_from,
                                 "record": str(r.path), "provenance": r.provenance,
                                 "identity": {k: v for k, v in r.identity.items()
                                              if k not in ("adapter_digests",)}})
        flat: Dict[str, Any] = {}
        _flatten("head_to_head", r.head_to_head or {}, flat)
        _flatten("admission", {k: v for k, v in (r.admission or {}).items()
                               if k not in ("notes", "reasons")}, flat)
        flat["admission/verdict_admitted"] = int((r.admission or {}).get("verdict") == "admitted")
        flat["admission/reasons_text"] = ", ".join(str(x) for x in (r.admission or {}).get("reasons", []))
        flat["admission/rule"] = r.rule
        flat["admission/rule_name"] = r.rule_name
        flat["admission/functional"] = int(r.is_functional)
        flat["admission/readiness_attested"] = int(((r.admission or {}).get("readiness") or {}).get("attested") is True)
        flat["admission/readmitted"] = int(r.readmitted is not None)
        flat["admission/in_job_verdict"] = r.in_job_verdict
        flat["admission/in_job_rule"] = r.in_job_rule
        flat["admission/in_job_verdict_admitted"] = int(r.in_job_verdict == "admitted")
        flat["admission/outcome"] = r.outcome
        flat["admission/verdict_inconclusive"] = int(r.verdict == "inconclusive")
        flat["admission/gate_scope"] = r.gate_scope
        flat["reused_unit"] = int(bool(r.reused_from))
        for p in figs:
            flat[f"figures/{p.stem}"] = wandb.Image(str(p))
        run.log(flat)
        art = wandb.Artifact(f"figures-{name}", type="figures")
        for p in figs:
            art.add_file(str(p))
        run.log_artifact(art)
        urls.append(run.url or name)
        run.finish()
    return urls


# --------------------------------------------------------------------------- #
# main                                                                          #
# --------------------------------------------------------------------------- #

def build(records_dir: Path, figs_dir: Path, *, all_dirs: bool = False) -> Dict[str, Any]:
    """The table over the newest record per target (every record with
    ``all_dirs``); the figures ALWAYS over the newest COMPLETE record per
    target (``figure_records``) — an incomplete record is a table row only."""
    records = load_records(records_dir, all_dirs=all_dirs)
    if not records:
        raise SystemExit(f"no <target>_<jobid> record dirs with a manifest.json under {records_dir}")
    figs_dir.mkdir(parents=True, exist_ok=True)
    md = table(records)
    (figs_dir / "summary_table.md").write_text(md + "\n")
    fig_records = newest_complete_per_target(records)
    figs = [p for p in (figure_head_to_head(fig_records, figs_dir), figure_admission(fig_records, figs_dir)) if p]
    return {"records": records, "figure_records": fig_records, "table": md, "figures": figs}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--figs", type=Path, required=True)
    ap.add_argument("--wandb-project", default=None,
                    help="default: the site's SISTEER_WANDB_PROJECT_MULTILORA_CERT, else WANDB_PROJECT, else SISTEER_WANDB_PROJECT")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="every record dir in the TABLE, not only the newest per target (the figures always draw "
                         "the newest complete record per target)")
    a = ap.parse_args(argv)
    out = build(a.records, a.figs, all_dirs=a.all)
    print(out["table"])
    for p in out["figures"]:
        print(f"figure: {p}  (records: {', '.join(r.path.name for r in out['figure_records'])})")
    for r in out["records"]:
        if r.problems:
            print(f"PROBLEMS {r.target}: {r.problems}")
    if not a.no_wandb:
        from sisteer.wandb_entity import wandb_project
        urls = wandb_log(out["records"], out["figures"], project=a.wandb_project or wandb_project(WANDB_PURPOSE), entity=a.wandb_entity)
        for u in urls:
            print(f"wandb: {u}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""GSM8K paired-200 (thinking ON) reports -> registry (source ``evalcheck/gsm8k_paired200``).

Reproduces ``paperlib.gsm8k_p200.load_gsm8k_p200`` (pin 40997227) over the grading reports under
``$GRADE/tinycap/gsm8k/`` and, as ``load_gsm8k_waves`` does, dedupes by DECLARED OWNERSHIP: each (model, view, cell) is
emitted ONCE as a live run from the canonical directory that owns it. One row per computable contrast (the LADDER-MERGED reading:
accuracy, Wilson-free producer interval, the paired percentile bootstrap over the 200 items vs the same view's ``none`` with
B = 10,000 at seed 0, exact McNemar p — every number the PRODUCER's, nothing recomputed) plus one unsteered anchor per (model, view).
Per-item scores come from ``cells[<key>].readings.merged.items`` (``item_result`` rows), truncation classes from the 8,192 leg's
``at_cap`` block (``run_degeneration``: at-cap residual and loops as counts over 200).

Every leg carries its identity (``legs[*].identity``: model revision, harness_config_hash, intervention_hash, object artifact
sha, served geometry, producer tree sha) and its thinking mode; the instrument is inspect-ai's stock ``gsm8k`` task, 10-shot,
``ANSWER: N`` match, k = 1, cap 8,192 with a 16,384 rung, each model's thinking-card sampling row (``sampling_rows``).

THE DIRECTORY INVENTORY. Discovery enumerates every ``<root>/*/`` and classifies it by NAME:

* **canonical** — :data:`CANONICAL`, the paper's declared set (a declaration like the pinned reader's ``POST_WAVE_REPORTS``):
  ``final_ext_steer_pins_r2`` (49 cells, pin 8f5f149), ``final_itdoc_pins_r3`` (adds the 8 it-view cells), ``final_llama_steer_pins_r3``
  (Llama on its own tree, pin 9ed8e7e). Read strictly: a wrong store pin, a failed gate, another task REFUSES the group. Ownership of a
  (model, view, cell) goes to the first canonical directory (declared order) that MEASURES it, else to the first that files a
  disposition for it; a later canonical directory repeating an owned cell must read IDENTICALLY (value, k, n) or the group refuses,
  and the repeat is recorded on the owner's run (``extra.repeated_by``), never emitted twice.
* **superseded** — :data:`DECLARED_SUPERSEDED` (``final_steer_pins_r2`` -> ext r2 although it carries no suffix; ``final_1pct_r2``, the
  base report the pinned paperlib read, -> ext r2, which reads each of its 38 cells identically; ``final_llama`` -> llama r3, the same
  way) and every ``*-superseded*`` directory. Read with the pin and gate checks RELAXED to "record" (the failed gates are WHY a pass
  was superseded; they ride the row's ``extra.failed_gates``), producer keys suffixed ``@<dirname>`` (a superseded reading is a
  distinct run, never the live run under another label — the ``bcb_bfcl`` snapshot convention), each run ``superseded`` by the
  owner's live run for the same (model, view, cell) (``superseded_by_run_id``, resolved in the same emit pass). The successor is
  ``X`` for ``X-superseded`` when ``X`` is read, the declared successor, else the canonical owner of each cell, else unknown (still
  filed as superseded). Unmeasured (held / refused) rows of a superseded pass file nothing: the live tree carries the disposition.
* **re-export** — ``*_paper_rows_v2``: the same runs' artifacts, not new runs. ``report.json`` must be byte-identical to the
  re-exported directory's (else its ``paper_rows.json`` must agree by cell on n_correct / n_scored) or the group refuses by name;
  the files join the re-exported directory's group as inputs (the runner registers every input as an ``artifact``) and the
  re-exported rows carry ``extra.re_exports``.
* **not production** — :data:`NOT_PRODUCTION` (``final_1pct_r2_default_check``: a default-decode check pass, not the paper
  instrument): one ``not applicable`` disposition per cell naming why; no runs.
* **undeclared** — everything else (``final``, ``final_1pct*``, the per-sha ``final_ext_steer_<sha>`` / ``final_itdoc_<sha>`` passes with
  failed gates, the ``*_pins_wt`` and ``*_pins_r2`` passes the owner did not name, ``prefinal_*``, ``pull_*``, ``staging_*`` …): not
  read; listed in the facts and in the ``$.dirs`` source_state so the inventory stays visible.

``paper_rows.json`` (``paper_rows/2``, beside a report) is cross-checked where present: its ``paper_arm`` must agree with this
adapter's cell -> arm mapping where it names one (``None`` marks an arm the paper does not draw), and its merged row's n_correct /
n_scored with the report's; ``paper_arm`` / ``selection_outcome`` / ``rung_state`` ride the run's ``extra.paper_rows``.

COMPARISON GROUPS: canonical directories that share a (model, view) form one group (the it-view ``none`` anchors live in ext r2
while the it-view arms contrasted against them live in itdoc r3, so the two must be in one emit pass for the contrasts to find
their anchor); the group is named after the first canonical member. A superseded directory joins its successor's group, a
re-export the group of the directory it re-exports; a not-production directory is its own group.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from .base import AdapterContext, AdapterRefusal, Input, ListingInput, RecordBatch, SourceStateRow, check_input, is_listing_input, listing_group, store_listdir, site_root
from . import declared_common as DC
from .declared_common import ARMS, EVAL_LABEL, HOME, MODELS, blank_row, finite, read_json

#: where a served store's entry (`legs[*]`) carries a path on the producing cluster's shared root under its launch / ledger fields, in preference order (the standup's
#: routing table, the serving config the launcher read, the chat template it excluded); the walk falls back to any such path
SUBSTRATE_WITNESS_KEYS = ("ledger.binding.routing_table", "launch.serving_config", "launch.standup_dir", "launch.excluded.chat_template.value")

GSM_ROOT = site_root("$GRADE/tinycap/gsm8k")
STORE_PIN = "8f5f1495b4432a0b2067d1d154063014bd8c1222"
#: the paper's CANONICAL directories, in ownership order: (directory, expected store pin or None = the report's own)
CANONICAL: Tuple[Tuple[str, Optional[str]], ...] = (
    ("final_ext_steer_pins_r2", STORE_PIN),
    ("final_itdoc_pins_r3", STORE_PIN),
    ("final_llama_steer_pins_r3", None),
)
#: directories the owner declared SUPERSEDED without a name suffix: directory -> (successor directory, reason)
DECLARED_SUPERSEDED: Dict[str, Tuple[str, str]] = {
    "final_steer_pins_r2": ("final_ext_steer_pins_r2",
                            "EvalCheck 2026-09-14 16:26Z: the same 38 published + 6 steer cells; the successor adds the 5 ext cells once their two owed 16,384 rungs ran"),
    "final_1pct_r2": ("final_ext_steer_pins_r2",
                      "the published base report the pinned paperlib read (REPORT at pin 40997227); every one of its 38 cells is owned by the successor, which reads each identically"),
    "final_llama": ("final_llama_steer_pins_r3",
                    "the Llama wave's first report; its 12 cells are re-read identically by the successor, which adds the two steer cells"),
}
#: directories that are NOT production passes: directory -> why (one `not applicable` disposition per cell)
NOT_PRODUCTION: Dict[str, str] = {
    "final_1pct_r2_default_check": "the default-decode check pass",
}
SUPERSEDED_MARK = "-superseded"
RE_EXPORT_SUFFIX = "_paper_rows_v2"
KINDS = ("canonical", "superseded", "re_export", "not_production", "undeclared")

# legacy explicit-list constants (the paths the adapter read before the directory inventory; kept for the explicit-list constructor)
REPORT = os.path.join(GSM_ROOT, CANONICAL[0][0], "report.json")
LATER_REPORTS = tuple(os.path.join(GSM_ROOT, d, "report.json") for d, _p in CANONICAL[1:])
SOURCE, EVAL_ID = "evalcheck/gsm8k_paired200", "gsm8k"
TASK = "inspect/gsm8k_paired200"
CAP, CEILING, N_ITEMS = 8192, 16384, 200
RESOLVED_WORDS = ("lower than none", "higher than none")
CHAT_CKPT = ("nemotron3nano4b", "glm46vflash")
BASELINE_PAPER_LABEL = {"arditi": "arditi", "alphasteer": "alphasteer_rebake"}


def _lane_kind_legacy(model: str, tag: str) -> str:
    spec = MODELS.get(model)
    if spec is not None:
        if tag == spec["view"]:
            return spec["lane_kind"]
        if tag == spec.get("transfer_view"):
            return "base→instruct"
    if tag.endswith("_it"):
        return "instruct-direct"
    if model in CHAT_CKPT:
        return "chat-ckpt"
    return "base→instruct"


def _decode(row: dict) -> str:
    names = [("temperature", "T"), ("top_p", "top_p"), ("top_k", "top_k"), ("min_p", "min_p"), ("presence_penalty", "presence_penalty"), ("repetition_penalty", "repetition_penalty")]
    parts = [f"{short}={row[k]}" for k, short in names if k in row and row[k] is not None]
    return "thinking ON, k=1, cap 8192 (+16384 rung); thinking-card sampling row: " + ", ".join(parts)


def _pp(x) -> float:
    return float("nan") if x is None else 100.0 * float(x)


def _cell_reading_legacy(cs: dict, which: str) -> dict:
    r = cs[which]
    acc = r["acc_by_leg_stock_primary"]
    return dict(word=r["word"], cell=acc["cell"], none=acc["none"], point=r["point"]["stock"]["primary"], ci=r["ci"]["stock"]["primary"],
                p=r.get("mcnemar_p_stock_primary"))


def _at_cap(leg_legacy: dict, cell_legacy: dict) -> dict:
    a = leg_legacy.get("at_cap", {}) or {}
    merged = cell_legacy["readings"]["merged"]
    return dict(n_at_cap=int(a.get("n_at_cap", 0)), n_loop=int(a.get("n_loop", 0)), n_residual=int(merged.get("n_at_cap_residual", 0)),
                escalated=bool(merged.get("escalated", False)))


def arm_key_of_legacy(cell_id: str) -> Optional[str]:
    """This adapter's cell -> paper arm key (``config.ARMS`` plus the two published baselines' producer cell ids)."""
    arm_legacy = ARMS.get(cell_id) or (DC.ARMS_BY_KEY[BASELINE_PAPER_LABEL[cell_id]] if cell_id in BASELINE_PAPER_LABEL else None)
    return arm_legacy.key if arm_legacy else None


def cell_key_legacy(model: str, view: str, cell_id: str) -> str:
    return f"{model}/{view}/{cell_id}"


def live_key(model: str, view: str, cell_id: str) -> str:
    """The in-pass key of the ONE live row of a (model, view, cell): directory-free, so an arm owned by one canonical directory finds
    the anchor owned by another."""
    return f"{SOURCE}|live|{model}|{view}|{cell_id}"


def dir_key(dirname: str, model: str, view: str, cell_id: str) -> str:
    """The in-pass key of a superseded directory's own reading of a cell (never collides with the live key)."""
    return f"{SOURCE}|{dirname}|{model}|{view}|{cell_id}"


def classify_name(name: str, canonical: Sequence[str], superseded: Mapping[str, Tuple[str, str]], not_production: Mapping[str, str]) -> Tuple[str, Optional[str]]:
    """(kind, target): ``superseded`` -> the declared successor (None = derived from the name / decided per cell), ``re_export`` -> the
    directory re-exported."""
    if name in canonical:
        return "canonical", None
    if name in not_production:
        return "not_production", None
    if name.endswith(RE_EXPORT_SUFFIX) and len(name) > len(RE_EXPORT_SUFFIX):
        return "re_export", name[: -len(RE_EXPORT_SUFFIX)]
    if name in superseded:
        return "superseded", superseded[name][0]
    if SUPERSEDED_MARK in name:
        return "superseded", None
    return "undeclared", None


#: THE ROOT INVENTORY'S PUBLISHER (the runner's listings-are-inputs rule, 2026-10-04): the root's directory listing -- every directory under the
#: root, declared or not -- is stated as `$.dirs` by ONE comparison group, the ROOT pseudo-group `_root` (source id `gsm8k_p200:_root`; the `_`
#: prefix is the runner's bookkeeping-group convention and sorts it before every directory group, so a reader that takes the LAST group's emit
#: report takes a directory's), whose only input is the root's listing (`base.ListingInput`, `registry://listing/gsm8k_p200:_root/.`) and whose
#: read emits no record. Its revision changes exactly when the listing changes -- a directory landing, a retired group's directory vanishing --
#: so it re-reads and restates `$.dirs` while every other group stays quiet; the per-directory groups state no root inventory. In BOTH modes: the
#: explicit-list override the tests use (`report=`) classifies nothing else under the root, but it lists the root the way its version-3 reads did,
#: so the inventory keeps its publisher there too. Until then every group's read, in either mode, wrote the same root-wide row keyed on the root
#: path under `gsm8k_p200:<root basename>`, so a retirement whose directory left the root cleared only that group's rows and `$.dirs` kept naming
#: the deleted directory with no read to repair it; the root group states that pre-rule key absent (found = 0) on every read, so a database that
#: holds it never keeps it found = 1.
ROOT_LISTING_RULE = ("the root's directory listing is the input of the root pseudo-group `_root`, the one publisher of $.dirs, in root mode and in the "
                     "explicit-list mode alike; its revision changes exactly when the listing changes; no other group states the root inventory; "
                     "the pre-rule root-path key under gsm8k_p200:<root basename> is stated absent on every read of the root group")
ROOT_LISTING_LABEL = "."
ROOT_GROUP = "_root"


class Gsm8kP200Adapter:
    name = "gsm8k_p200"
    #: the shared emitter's park change rides in (`paper_dose`) — see `declared_common.EMITTER_VERSION`
    EMISSION_VERSION = DC.emission_version(3)   # 2: the distilled cell's arm / label (difference_relu_distilled);
    #                                             3: the substrate derived per served store from its launch / ledger paths (9 of 142 entries are cluster B), never a constant (2026-09-21)
    covers_sources: Tuple[str, ...] = (SOURCE,)
    rng_scope = "none"
    allow_symlinks = False
    campaign = "sdfbase_full22"

    def __init__(self, root: str = GSM_ROOT, canonical: Sequence[Tuple[str, Optional[str]]] = CANONICAL,
                 superseded: Mapping[str, Tuple[str, str]] = DECLARED_SUPERSEDED, not_production: Mapping[str, str] = NOT_PRODUCTION, *,
                 report: Optional[str] = None, later_reports: Optional[Sequence[str]] = None, expect_pin: Optional[str] = STORE_PIN):
        """Enumerating by default (``root`` + the declared constants). ``report=`` / ``post_wave=`` / ``expect_pin=`` is the explicit-list
        override the tests use: those paths are the canonical set (``expect_pin`` for ``report``, each post-wave report's own pin) and
        nothing else under their root is classified or read."""
        if report is not None:
            later_reports = tuple(later_reports or ())
            self.explicit = True
            root = os.path.dirname(os.path.dirname(report))
            self._canonical: List[Tuple[str, str, Optional[str]]] = [(os.path.basename(os.path.dirname(report)), report, expect_pin)]
            self._canonical += [(os.path.basename(os.path.dirname(p)), p, None) for p in later_reports]
            superseded, not_production = {}, {}
        else:
            self.explicit = False
            self._canonical = [(d, os.path.join(root, d, "report.json"), pin) for d, pin in canonical]
            report = self._canonical[0][1]
            later_reports = tuple(p for _d, p, _pin in self._canonical[1:])
        self.root = root
        self.roots = (root,)
        self.report, self.later_reports, self.expect_pin = report, tuple(later_reports), expect_pin
        self.superseded_decl: Dict[str, Tuple[str, str]] = dict(superseded)
        self.not_production: Dict[str, str] = dict(not_production)
        self.last_report: Optional[DC.EmitReport] = None
        self.last_facts: Dict[str, object] = {}
        self.inventory: Dict[str, Dict[str, object]] = {}
        self._planned = False
        self._group_of: Dict[str, str] = {}
        self._owner: Dict[str, str] = {}
        self._scan_cache: Dict[str, Dict[str, object]] = {}

    # -- inventory -----------------------------------------------------------------------------
    def _canonical_names(self) -> List[str]:
        return [d for d, _p, _pin in self._canonical]

    def _report_path(self, dirname: str) -> str:
        for d, p, _pin in self._canonical:
            if d == dirname:
                return p
        return os.path.join(self.root, dirname, "report.json")

    def _declared_pin(self, dirname: str) -> Optional[str]:
        for d, _p, pin in self._canonical:
            if d == dirname:
                return pin
        return None

    def _list_dirs(self) -> List[str]:
        try:
            return sorted(n for n in store_listdir(self.root) if os.path.isdir(os.path.join(self.root, n)))
        except OSError:
            return []

    def _scan(self, path: str) -> Dict[str, object]:
        """A light read of one report: its (model, view) pairs, the cells it MEASURES (computable contrasts + their none) and the cells
        it files a disposition for. Never raises on content: a malformed report is refused where it is read in full."""
        if path in self._scan_cache:
            return self._scan_cache[path]
        out: Dict[str, object] = {"views": set(), "measured": set(), "dispositions": set(), "cells": [], "ok": False}
        try:
            d = read_json(path)
            out["views"] = {(v["model_key"], v["view"]) for v in (d.get("legs") or {}).values()}
            for cs in d.get("contrast_summary") or []:
                k = cell_key_legacy(cs["model_key"], cs["view"], cs["cell"])
                if cs.get("computable"):
                    out["measured"].add(k)
                    out["measured"].add(cell_key_legacy(cs["model_key"], cs["view"], "none"))
                else:
                    out["dispositions"].add(k)
            for h in d.get("held") or []:
                out["dispositions"].add(cell_key_legacy(h["model_key"], h["view"], h["cell"]))
            out["cells"] = sorted((d.get("cells") or {}).keys())
            out["ok"] = True
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass
        self._scan_cache[path] = out
        return out

    def _plan(self) -> None:
        """Classify every directory under the root, compute cell ownership across the canonical set and the comparison group of every
        directory that is read."""
        if self._planned:
            return
        canon = self._canonical_names()
        inv: Dict[str, Dict[str, object]] = {}
        names = self._list_dirs() if not self.explicit else []
        for n in canon:
            if n not in names:
                names.append(n)
        for n in sorted(set(names)):
            kind, target = classify_name(n, canon, self.superseded_decl, self.not_production)
            rp = self._report_path(n)
            pr = os.path.join(os.path.dirname(rp), "paper_rows.json")
            inv[n] = {"kind": kind, "target": target, "report": rp, "has_report": os.path.lexists(rp), "paper_rows": pr,
                      "has_paper_rows": os.path.isfile(pr), "reason": None, "successor": None}
            if kind == "superseded":
                inv[n]["reason"] = (self.superseded_decl[n][1] if n in self.superseded_decl else f"directory marked {SUPERSEDED_MARK}")
            elif kind == "not_production":
                inv[n]["reason"] = self.not_production[n]
        # ownership: the first canonical directory (declared order) that measures the cell, else the first that files a disposition for it
        owner: Dict[str, str] = {}
        filed: Dict[str, str] = {}
        scans = {n: self._scan(self._report_path(n)) for n in canon if inv[n]["has_report"]}
        for n in canon:
            sc = scans.get(n)
            if not sc:
                continue
            for k in sorted(sc["measured"]):
                owner.setdefault(k, n)
            for k in sorted(sc["dispositions"]):
                filed.setdefault(k, n)
        for k, n in filed.items():
            owner.setdefault(k, n)
        self._owner = owner
        # groups: canonical directories sharing a (model, view) are one component, named after its first member (declared order)
        comp: Dict[str, str] = {n: n for n in canon}
        for i, a in enumerate(canon):
            for b in canon[i + 1:]:
                if scans.get(a) and scans.get(b) and (scans[a]["views"] & scans[b]["views"]):
                    ra, rb = comp[a], comp[b]
                    keep, drop = (ra, rb) if canon.index(ra) <= canon.index(rb) else (rb, ra)
                    for n, r in list(comp.items()):
                        if r == drop:
                            comp[n] = keep
        group: Dict[str, str] = dict(comp)
        # superseded: X-superseded -> X when X is read; the declared successor; else the canonical owner of most of its cells
        for n, info in inv.items():
            if info["kind"] != "superseded":
                continue
            succ = info["target"]
            if succ is None and SUPERSEDED_MARK in n:
                x = n.split(SUPERSEDED_MARK, 1)[0]
                if x in inv and inv[x]["kind"] in ("canonical", "superseded") and inv[x]["has_report"]:
                    succ = x
            if succ is None and info["has_report"]:
                sc = self._scan(str(info["report"]))
                votes = Counter(owner[k] for k in (sc["measured"] | sc["dispositions"]) if k in owner)
                succ = votes.most_common(1)[0][0] if votes else None
            info["successor"] = succ
        for _ in range(3):      # settle chains (a superseded directory whose successor is itself superseded)
            for n, info in inv.items():
                if info["kind"] != "superseded":
                    continue
                succ = info["successor"]
                group[n] = group.get(str(succ), n) if succ is not None else n
        for n, info in inv.items():
            if info["kind"] == "re_export":
                group[n] = group.get(str(info["target"]), n)
            elif info["kind"] == "not_production":
                group[n] = n
        for n, info in inv.items():
            info["group"] = group.get(n)
        self.inventory, self._group_of, self._planned = inv, group, True

    # -- the adapter contract --------------------------------------------------------------------
    def discover(self) -> List[Input]:
        self._plan()
        out: List[Input] = []
        for n, info in self.inventory.items():
            kind = info["kind"]
            if kind == "undeclared":
                continue
            rp = str(info["report"])
            if kind == "canonical":
                if not os.path.lexists(rp):
                    raise AdapterRefusal(rp, "input missing")
                out.append(check_input(rp, role="report", allow_symlink=self.allow_symlinks))
            elif not info["has_report"]:
                continue
            elif kind == "re_export":
                out.append(check_input(rp, role="snapshot", allow_symlink=self.allow_symlinks))
            else:
                out.append(check_input(rp, role="report", allow_symlink=self.allow_symlinks))
            if info["has_paper_rows"] and kind != "not_production":
                out.append(check_input(str(info["paper_rows"]), role="rows", allow_symlink=self.allow_symlinks))
        # the root pseudo-group's one input (ROOT_LISTING_RULE), in both modes: the root's directory listing; a directory named like the
        # pseudo-group would collide with its name and refuses by name
        rg = self.root_group()
        if rg in self.inventory or rg in self._list_dirs():
            raise AdapterRefusal(self.root, f"a directory under the root is named {rg!r}: it collides with the root pseudo-group's name ({ROOT_LISTING_RULE})")
        out.append(ListingInput.of(f"{self.name}:{rg}", ROOT_LISTING_LABEL, [(n, "dir") for n in self._list_dirs()]))
        return out

    def root_group(self) -> str:
        """The ROOT pseudo-group's name (ROOT_LISTING_RULE): :data:`ROOT_GROUP`."""
        return ROOT_GROUP

    def legacy_root_source_id(self) -> str:
        """The source id the root inventory carried before the root group (`gsm8k_p200:<root basename>`): its root-path-keyed `$.dirs` row is
        stated absent by every read of the root group (ROOT_LISTING_RULE)."""
        return f"{self.name}:{os.path.basename(self.root.rstrip('/'))}"

    def comparison_group(self, inp: Input) -> str:
        self._plan()
        if is_listing_input(inp):
            return listing_group(inp.uri, self.name)
        d = os.path.basename(os.path.dirname(inp.uri))
        return self._group_of.get(d, d)

    def inputs_retired(self, group: str) -> List[str]:
        """The inputs a read of ``group`` drops ON PURPOSE (the runner's inputs-dropped guard): for every directory of the group
        whose NAME carries :data:`SUPERSEDED_MARK` (``X-superseded…``, the owner's retirement of pass ``X`` by rename), the report and
        ``paper_rows.json`` the un-marked name ``X`` spelled — the files that left the group's input set when the directory was renamed.
        A declared-superseded directory (:data:`DECLARED_SUPERSEDED`, no suffix) renamed nothing and retires nothing."""
        self._plan()
        out: List[str] = []
        for n, info in self.inventory.items():
            if info.get("group") != group or SUPERSEDED_MARK not in n:
                continue
            original = n.split(SUPERSEDED_MARK, 1)[0]
            if not original or original in self.inventory:
                continue                      # the un-marked pass still stands beside its copy: nothing left the set
            report = self._report_path(original)
            out += [report, os.path.join(os.path.dirname(report), "paper_rows.json")]
        return sorted(set(out))

    # -- one report -> paper-long rows -------------------------------------------------------------
    def long_rows(self, report: Optional[str] = None, expect_pin: Optional[str] = "default", *, relaxed: bool = False) -> Tuple[List[dict], Dict[str, object]]:
        """Every row of ONE report (the pinned reader's rows plus a disposition row per held / refused cell). ``expect_pin="default"``
        is the directory's declared pin (None for a directory with none). ``relaxed`` records a wrong pin / failed gates in the facts
        instead of refusing — for a SUPERSEDED pass, whose failed gates are why it was superseded; a canonical pass is never relaxed."""
        report = report or self.report
        dirname = os.path.basename(os.path.dirname(report))
        if expect_pin == "default":
            expect_pin = self._declared_pin(dirname)
        d = read_json(report)
        pin = str((d.get("provenance") or {}).get("store_pin") or d.get("sha") or "")
        pin_ok = bool(pin) and (expect_pin is None or expect_pin.startswith(pin))
        if not pin_ok and not (relaxed and pin):
            raise AdapterRefusal(report, f"the report's store pin is {pin!r}, not {str(expect_pin)[:7]}; not the wave the paper reads")
        store_pin = expect_pin if (expect_pin is not None and pin_ok) else pin
        failed = list((d.get("gates") or {}).get("failed") or [])
        if failed and not relaxed:
            raise AdapterRefusal(report, f"the report records failed gates {failed}; nothing is read from a failing pass")
        if d.get("task") != TASK:
            raise AdapterRefusal(report, f"task is {d.get('task')!r}, not {TASK}")
        tip = str(d.get("analysis_code_sha") or "")
        sampling = (d.get("sampling_rows") or {}).get("thinking_card") or {}
        legs, cells = d["legs"], d["cells"]
        design = d.get("design") or {}
        leg_pins_legacy = ((d.get("store_pins") or {}).get("pins") or {})
        view_tag_legacy = {(v["model_key"], v["view"]): v["tag"] for v in legs.values()}
        when = DC.mtime_utc(report)
        rows: List[dict] = []
        none_by_view_legacy: Dict[Tuple[str, str], dict] = {}
        words: Dict[str, str] = {}
        refused: List[str] = []

        def leg_bits_legacy(leg_legacy: dict) -> dict:
            ident = leg_legacy.get("identity") or {}
            served = ident.get("served") or {}
            launch = leg_legacy.get("launch") or {}
            tm = leg_legacy.get("thinking_mode")
            tm = tm if isinstance(tm, str) and tm in ("on", "off", "not_available") else ((design.get("thinking_mode_by_model") or {}).get(leg_legacy.get("model_key")) or "on")
            sha = ident.get("object_artifact_sha256")
            sha = sha.lower() if isinstance(sha, str) and re.fullmatch(r"[0-9a-fA-F]{64}", sha) else None
            tree = str(ident.get("producer_tree_sha") or leg_legacy.get("producer_tree") or "")
            return dict(thinking=tm, sha=sha, revision=ident.get("model_revision"), harness_hash=ident.get("harness_config_hash"),
                        intervention=ident.get("intervention_hash"), window=served.get("max_model_len"), tree=tree if len(tree) == 40 else "unverified",
                        store=leg_legacy.get("cell_dir"), store_key=leg_legacy.get("store_key"), job=(launch.get("job_id")), reduction=served.get("reduction_mode"),
                        attempts=len(ident.get("resume_attempts") or []), leg_name=leg_legacy.get("leg"),
                        # the substrate is read PER ENTRY from the paths its launch / ledger fields carry on a cluster's shared root (the
                        # standup's routing table, the serving config, the chat template the standup excluded); `cell_dir` is the grading host mirror
                        substrate_derived=DC.D.substrate_derived_from_record(
                            {"ledger": leg_legacy.get("ledger") or {}, "launch": launch}, "report.json:legs[*]",
                            prefer=SUBSTRATE_WITNESS_KEYS))

        def base(model: str, view: str, cell_id: str, leg_legacy: Optional[dict]) -> dict:
            arm_legacy = ARMS.get(cell_id) or (DC.ARMS_BY_KEY[BASELINE_PAPER_LABEL[cell_id]] if cell_id in BASELINE_PAPER_LABEL else None)
            tag = view_tag_legacy[(model, view)]
            lk = _lane_kind_legacy(model, tag)
            if cell_id in BASELINE_PAPER_LABEL and model in MODELS:
                lk, tag_v = MODELS[model]["lane_kind"], MODELS[model]["view"]
            else:
                tag_v = tag
            bits = leg_bits_legacy(leg_legacy or {})
            sp, _c, _s = DC.serving_path_from_identity("lens" if bits["reduction"] == "stacked" else None)
            r = blank_row(source=SOURCE, model=model, lane_kind=lk, view=tag_v, dose=1.0, cell_id=cell_id, arm=arm_legacy.key if arm_legacy else None,
                          family=arm_legacy.family if arm_legacy else None, object_type=arm_legacy.object_type if arm_legacy else None, component="capability", eval_id=EVAL_ID,
                          eval=EVAL_LABEL[EVAL_ID], metric="acc", threshold=float("nan"), bon_n=float("nan"), decode_cap_tokens=CAP, decode=_decode(sampling.get(model, {})),
                          judge="inspect_evals gsm8k scorer (10-shot, 'ANSWER: N' match); an unterminated think block is unscored (primary branch)", status="measured",
                          degenerate_win=False, anchor_degenerate=False, thinking_mode=bits["thinking"], object_sha256=bits["sha"] if bits["sha"] else float("nan"),
                          serving_window_tokens=bits["window"] if bits["window"] is not None else float("nan"))
            leg_pin_legacy = leg_pins_legacy.get(str(bits["leg_name"] or ""))
            r.update(_when=when, _code_sha=bits["tree"], _code_sha_reason="the leg names no producer tree sha", _model_revision=bits["revision"],
                     _object_id=bits["sha"], _store_uri=bits["store"], _substrate_path="", _substrate_derived=bits["substrate_derived"],
                     _substrate_reason="the entry's launch / ledger fields name no path on a cluster's shared root",
                     _producer_key=DC.store_run_key(bits["store"], bits["store_key"]) or f"{store_pin[:7]}|{model}/{view}/{cell_id}",
                     _object_key=("none" if cell_id == "none" else (f"{cell_id}@{bits['intervention']}" if bits["intervention"] else cell_id)),
                     _instrument={"harness": "inspect_ai", "harness_version": tip or "unverified", "harness_config_hash": bits["harness_hash"], "token_cap_tokens": CAP,
                                  "ladder": (CAP, CEILING), "read_at": "ladder_final", "decode": {k: v for k, v in (sampling.get(model) or {}).items() if k in DC.D.DECODE_KEYS},
                                  "decode_source": "model_card", "decode_seed_rule": "per_item", "thinking_mode": bits["thinking"], "serving_path": sp,
                                  "serving_identity": f"reduction_mode={bits['reduction']}", "intervention_hash": bits["intervention"], "job_id": bits["job"],
                                  "n_shot": 10, "cot_prompting": "few_shot_cot", "scoring_rule": "numeric_match", "instrument_config_hash": f"gsm8k_paired200_thinking_on_{store_pin[:7]}",
                                  "fit_positions": "all"},
                     _judge=None, _n_items_planned=N_ITEMS,
                     _extra={"store_pin": store_pin, "analysis_tip": tip, "resume_attempts": bits["attempts"], "directory": dirname,
                             "leg_store_pin": leg_pin_legacy if isinstance(leg_pin_legacy, str) else None},
                     _strength_coefficient=(1.0 if (cell_id != "none" and arm_legacy and arm_legacy.object_kind not in ("none", "system_instruction")) else None),
                     _is_operating_point=(1 if (cell_id != "none" and arm_legacy and arm_legacy.object_kind not in ("none", "system_instruction")) else None),
                     _key=live_key(model, view, cell_id), _cell_key=cell_key_legacy(model, view, cell_id))
            return r

        def items_of(cell_legacy: dict) -> List[dict]:
            out = []
            for iid, it in sorted((cell_legacy["readings"]["merged"].get("items") or {}).items()):
                out.append({"item_id": iid, "metric_id": "acc", "value": (float(it["value"]) if it.get("scored") else None), "judged": False,
                            "label": None if it.get("scored") else "unscored",
                            "flags": {"at_token_cap": int(bool(it.get("at_cap"))), "finish_reason": it.get("stop_reason"), "response_tokens": it.get("completion_tokens")}})
            return out

        for cs in d.get("contrast_summary", []):
            model, view, cell_id = cs["model_key"], cs["view"], cs["cell"]
            key = cell_key_legacy(model, view, cell_id)
            if not cs.get("computable"):
                refused.append(f"{key} — {cs.get('certified_refusal') or cs.get('expected_refusal') or 'not computable'}")
                continue
            m = _cell_reading_legacy(cs, "merged")
            none_by_view_legacy.setdefault((model, view), m["none"])
            cell_legacy = cells.get(key)
            if cell_legacy is None:
                raise AdapterRefusal(report, f"{key}: contrast_summary names a cell the report's cells block lacks")
            mr = cell_legacy["readings"]["merged"]
            if (int(mr["n_correct"]), int(mr["n_scored"])) != (int(m["cell"]["n_correct"]), int(m["cell"]["n_scored"])):
                raise AdapterRefusal(report, f"{key}: cells.readings.merged disagrees with contrast_summary.merged")
            leg_legacy = legs.get(cell_legacy["base_leg"], {})
            ac = _at_cap(leg_legacy, cell_legacy)
            words[key] = m["word"]
            r = base(model, view, cell_id, leg_legacy)
            r.update(value=float(m["cell"]["point"]), k=int(m["cell"]["n_correct"]), n=int(m["cell"]["n_scored"]), ci_lo=float(m["cell"]["ci_low"]), ci_hi=float(m["cell"]["ci_high"]),
                     delta=_pp(m["point"]), delta_lo=_pp(m["ci"][0]), delta_hi=_pp(m["ci"][1]), delta_p=float(m["p"]) if m["p"] is not None else float("nan"),
                     delta_method=("paired percentile bootstrap over the 200 items vs the same view's none (B=10,000, seed 0), stock axis, primary think-block branch; "
                                   "delta_p = exact two-sided McNemar on complete cases"),
                     at_cap_share=ac["n_residual"] / N_ITEMS, loop_share=ac["n_loop"] / N_ITEMS, degenerate_share=ac["n_loop"] / N_ITEMS, suggestive=(m["word"] not in RESOLVED_WORDS),
                     note=f"LADDER-MERGED reading; verdict: {m['word']}; at cap at 8,192: {ac['n_at_cap']}/{N_ITEMS}; {m['cell']['n_censored']} unscored under the primary branch",
                     provenance=f"{report} ›contrast_summary[{cs['name']}] + cells[{key}] + legs[{cell_legacy['base_leg']}]; store pin {store_pin}; analysis tip {tip}")
            r.update(_anchor_key=live_key(model, view, "none"), _anchor_kind="same_standup_none",
                     _n_pairs=N_ITEMS, _ci_method="producer_percentile_bootstrap_10000_seed0", _items=items_of(cell_legacy),
                     _degeneration={"unit": "items", "counts": {"at_token_cap": (ac["n_residual"], N_ITEMS), "classifier": (ac["n_loop"], N_ITEMS), "union5": (ac["n_loop"], N_ITEMS)}})
            rows.append(r)
        for (model, view), none in sorted(none_by_view_legacy.items()):
            key = cell_key_legacy(model, view, "none")
            cell_legacy = cells.get(key)
            if cell_legacy is None:
                raise AdapterRefusal(report, f"{key}: the view's none cell is missing from the report's cells block")
            mr = cell_legacy["readings"]["merged"]
            if (int(mr["n_correct"]), int(mr["n_scored"])) != (int(none["n_correct"]), int(none["n_scored"])):
                raise AdapterRefusal(report, f"{key}: the none cell's own reading disagrees with the contrasts' none leg")
            leg_legacy = legs.get(cell_legacy["base_leg"], {})
            ac = _at_cap(leg_legacy, cell_legacy)
            r = base(model, view, "none", leg_legacy)
            r.update(value=float(none["point"]), k=int(none["n_correct"]), n=int(none["n_scored"]), ci_lo=float(none["ci_low"]), ci_hi=float(none["ci_high"]),
                     at_cap_share=ac["n_residual"] / N_ITEMS, loop_share=ac["n_loop"] / N_ITEMS, degenerate_share=ac["n_loop"] / N_ITEMS, suggestive=float("nan"),
                     note=f"the view's unsteered anchor (thinking ON, 8,192 cap); at cap at 8,192: {ac['n_at_cap']}/{N_ITEMS}; {none['n_censored']} unscored",
                     provenance=f"{report} ›cells[{key}] + legs[{cell_legacy['base_leg']}]; store pin {store_pin}; analysis tip {tip}")
            r.update(_ci_method="producer_percentile_bootstrap_10000_seed0", _items=items_of(cell_legacy),
                     _degeneration={"unit": "items", "counts": {"at_token_cap": (ac["n_residual"], N_ITEMS), "classifier": (ac["n_loop"], N_ITEMS), "union5": (ac["n_loop"], N_ITEMS)}})
            rows.append(r)
        held = [cell_key_legacy(h["model_key"], h["view"], h["cell"]) for h in d.get("held", [])]
        for h in d.get("held", []):
            model, view, cell_id = h["model_key"], h["view"], h["cell"]
            if (model, view) not in view_tag_legacy:
                view_tag_legacy[(model, view)] = f"{model}_it" if view == "it" else (MODELS.get(model, {}).get("transfer_view") or model)
            r = base(model, view, cell_id, None)
            r.update(status=f"held: {h.get('reason')}", note=str(h.get("reason")), provenance=f"{report} ›held")
            rows.append(r)
        for x in refused:
            model, view, cell_id = x.split(" — ")[0].split("/")
            if (model, view) not in view_tag_legacy:
                continue
            r = base(model, view, cell_id, None)
            r.update(status=f"refused: {x.split(' — ', 1)[1]}", note=x, provenance=f"{report} ›contrast_summary (not computable)")
            rows.append(r)
        facts = dict(store_pin=store_pin, report_pin=pin, pin_ok=pin_ok, failed_gates=failed, analysis_tip=tip, words=words, held=held, refused=refused,
                     n_rows=len(rows), directory=dirname)
        return rows, facts

    # -- paper_rows.json -------------------------------------------------------------------------
    @staticmethod
    def paper_rows_by_cell_legacy(path: str) -> Tuple[Dict[str, dict], Dict[str, object]]:
        """``paper_rows.json`` -> {model/view/cell: the merged row (else the r8192 row)} plus the file's header fields."""
        p = read_json(path)
        by: Dict[str, Dict[str, dict]] = {}
        for row in p.get("rows") or []:
            by.setdefault(cell_key_legacy(row["model_key"], row["view"], row["cell"]), {})[str(row.get("leg"))] = row
        out = {k: (legs.get("merged") or legs.get("r8192") or next(iter(legs.values()))) for k, legs in by.items()}
        head: Dict[str, object] = {k: p.get(k) for k in ("schema", "instrument", "n_rows", "report")}
        head["has_paper_arm"] = "paper_arm" in (p.get("fields") or [])
        return out, head

    def _cross_check_paper_rows(self, dirname: str, path: str, rows: Sequence[dict], slot: str = "paper_rows") -> Dict[str, object]:
        """``paper_arm`` must agree with this adapter's cell -> arm where it names one; a measured row's n_correct / n_scored must be the
        report's. Refuses the DIRECTORY by name on disagreement; annotates the rows' ``_extra[slot]`` otherwise."""
        by, head = self.paper_rows_by_cell_legacy(path)
        bad: List[str] = []
        for r in rows:
            k = str(r.get("_cell_key"))
            pr = by.get(k)
            if pr is None:
                continue
            pa = pr.get("paper_arm") if head["has_paper_arm"] else None
            mine = arm_key_of_legacy(str(r.get("cell_id")))
            if pa is not None and pa != mine:
                bad.append(f"{k}: paper_rows paper_arm {pa!r} vs adapter arm {mine!r}")
            if r.get("status") == "measured" and pr.get("n_correct") is not None and pr.get("n_scored") is not None:
                if (int(pr["n_correct"]), int(pr["n_scored"])) != (int(r["k"]), int(r["n"])):
                    bad.append(f"{k}: paper_rows reads {pr['n_correct']}/{pr['n_scored']} ({pr.get('leg')}), the report {int(r['k'])}/{int(r['n'])}")
            r["_extra"] = dict(r.get("_extra") or {})
            r["_extra"][slot] = {"file": os.path.basename(os.path.dirname(path)) + "/paper_rows.json", "schema": head.get("schema"), "leg": pr.get("leg"),
                                 "paper_arm": pa, "selection_outcome": pr.get("selection_outcome"), "rung_state": pr.get("rung_state")}
        if bad:
            raise AdapterRefusal(path, f"paper_rows.json of {dirname} disagrees with the report / the adapter's arm mapping on {len(bad)} cell(s): " + "; ".join(bad[:6]))
        return dict(schema=head.get("schema"), n_rows=head.get("n_rows"), n_cells=len(by), has_paper_arm=head["has_paper_arm"])

    # -- the group ---------------------------------------------------------------------------------
    @staticmethod
    def _same_reading(a: Mapping[str, object], b: Mapping[str, object]) -> bool:
        return abs(float(a["value"]) - float(b["value"])) <= 1e-9 and int(a["k"]) == int(b["k"]) and int(a["n"]) == int(b["n"])

    def read(self, ctx: AdapterContext, group: str, inputs: Sequence[Input]) -> Iterator[RecordBatch]:
        self._plan()
        if group == self.root_group():
            # the root pseudo-group (ROOT_LISTING_RULE): no record; its facts are the root's inventory, its state the `$.dirs` row (source_states)
            listing = next((i for i in inputs if is_listing_input(i)), None)
            self.last_facts[group] = {"group": group, "root_listing": list(listing.entries) if listing is not None else [],
                                      "n_dirs": len(listing.entries) if listing is not None else 0, "root_listing_rule": ROOT_LISTING_RULE,
                                      "inventory": {n: {k: v for k, v in i.items() if k not in ("paper_rows", "report")} for n, i in self.inventory.items()}}
            self.last_report = DC.EmitReport()           # no emitter ran: an empty report, never the previous group's
            yield RecordBatch(f"{self.name}:{group}", group)
            return
        canon = self._canonical_names()
        by_dir: Dict[str, Dict[str, Input]] = {}
        for inp in inputs:
            by_dir.setdefault(os.path.basename(os.path.dirname(inp.uri)), {})["rows" if inp.role == "rows" else "report"] = inp
        dir_facts: Dict[str, object] = {}
        facts: Dict[str, object] = {"group": group, "directories": dir_facts,
                                    "inventory": {n: {k: v for k, v in i.items() if k not in ("paper_rows", "report")} for n, i in self.inventory.items()}}
        live: Dict[str, dict] = {}          # cell key -> the owner's live row
        live_rows: List[dict] = []
        superseded_rows: List[dict] = []
        disposition_rows: List[dict] = []
        rows_by_dir: Dict[str, List[dict]] = {}

        def order(d: str) -> Tuple[int, int, str]:
            k = str(self.inventory.get(d, {}).get("kind", "undeclared"))
            return (KINDS.index(k) if k in KINDS else 9, canon.index(d) if d in canon else 99, d)

        dirs = sorted(by_dir, key=order)
        # 1. canonical directories, in declared order: live rows for the cells they own; identical repeats recorded on the owner's row
        for d in dirs:
            info = self.inventory.get(d) or {}
            if info.get("kind") != "canonical" or "report" not in by_dir[d]:
                continue
            inp = by_dir[d]["report"]
            rows, f = self.long_rows(inp.path, expect_pin=self._declared_pin(d))
            rows_by_dir[d] = rows
            f.update(kind="canonical", n_live=0, repeats_verified=[], dispositions_not_filed=[], paper_rows=None)
            if "rows" in by_dir[d]:
                f["paper_rows"] = self._cross_check_paper_rows(d, by_dir[d]["rows"].path, rows)
            for r in rows:
                k = str(r["_cell_key"])
                owner = self._owner.get(k, d)
                if owner == d and k not in live:
                    r["_extra"] = dict(r.get("_extra") or {}, classification="canonical", owner=d)
                    live[k] = r
                    live_rows.append(r)
                    f["n_live"] += 1
                elif r.get("status") == "measured":
                    prev = live.get(k)
                    if prev is None or prev.get("status") != "measured":
                        raise AdapterRefusal(inp.uri, f"{k}: repeats a cell owned by {owner} that {owner} did not measure; ownership cannot be resolved")
                    if not self._same_reading(prev, r):
                        raise AdapterRefusal(inp.uri, f"{k}: repeats a cell owned by {owner} with a DIFFERENT reading ({d}: {float(r['value']):.4f} = {int(r['k'])}/{int(r['n'])}; "
                                                      f"{owner}: {float(prev['value']):.4f} = {int(prev['k'])}/{int(prev['n'])}); two readings of one cell under one label")
                    prev["_extra"].setdefault("repeated_by", []).append(d)
                    f["repeats_verified"].append(k)
                else:
                    f["dispositions_not_filed"].append(f"{k}: {str(r.get('status'))[:60]} (owned by {owner})")
            dir_facts[d] = f
        # 2. superseded directories: every measured row a superseded run pointing at the owner's live run for the cell
        for d in dirs:
            info = self.inventory.get(d) or {}
            if info.get("kind") != "superseded" or "report" not in by_dir[d]:
                continue
            inp = by_dir[d]["report"]
            succ_dir = info.get("successor")
            # relaxed: the pin is compared against the SUCCESSOR's declared pin and recorded (`pin_matches_paper`), never refused
            rows, f = self.long_rows(inp.path, expect_pin=(self._declared_pin(str(succ_dir)) if succ_dir else None), relaxed=True)
            rows_by_dir[d] = rows
            why = str(info.get("reason") or f"directory marked {SUPERSEDED_MARK}") + f" (analysis tip {str(f['analysis_tip'])[:7] or 'unknown'})"
            if f["failed_gates"]:
                why += f"; failed gates {f['failed_gates']}"
            f.update(kind="superseded", successor=succ_dir, n_superseded=0, n_unmeasured_skipped=0, paper_rows=None, successor_by_cell={})
            if "rows" in by_dir[d]:
                f["paper_rows"] = self._cross_check_paper_rows(d, by_dir[d]["rows"].path, rows)
            for r in rows:
                k = str(r["_cell_key"])
                model, view, cell_id = str(r["model"]), k.split("/")[1], str(r["cell_id"])
                owner = self._owner.get(k)
                target = owner if owner is not None else succ_dir
                by_key = live_key(model, view, cell_id) if (owner is not None and k in live) else None
                text = (f"superseded by {target}: {why}" if target else f"superseded, successor unknown for this cell: {why}")
                r.update(_producer_key=f"{r['_producer_key']}@{d}", _key=dir_key(d, model, view, cell_id), _superseded=text, _superseded_by_key=by_key)
                if finite(r.get("delta")):
                    r["_anchor_key"] = dir_key(d, model, view, "none")
                r["_extra"] = dict(r.get("_extra") or {}, classification="superseded", superseded_dir=d, successor=target, failed_gates=list(f["failed_gates"]),
                                   report_pin=f["report_pin"], pin_matches_paper=f["pin_ok"])
                f["successor_by_cell"][k] = target
                if r.get("status") == "measured":
                    f["n_superseded"] += 1
                else:
                    f["n_unmeasured_skipped"] += 1
                superseded_rows.append(r)
            dir_facts[d] = f
        # 3. re-exports: artifacts of the re-exported directory's runs; byte-identity (else paper_rows agreement by cell) or refuse by name
        for d in dirs:
            info = self.inventory.get(d) or {}
            if info.get("kind") != "re_export":
                continue
            target = str(info.get("target"))
            inp = by_dir[d].get("report")
            tgt = by_dir.get(target, {}).get("report")
            where = inp.uri if inp is not None else by_dir[d]["rows"].uri
            f = dict(kind="re_export", target=target, byte_identical=None, paper_rows=None, rows_agree=None)
            if tgt is None or target not in rows_by_dir:
                raise AdapterRefusal(where, f"re-export of {target}, a directory this adapter does not read in this group")
            if inp is not None:
                f["byte_identical"] = (inp.sha256 == tgt.sha256)
            if "rows" in by_dir[d]:
                f["paper_rows"] = self._cross_check_paper_rows(d, by_dir[d]["rows"].path, rows_by_dir[target], slot="paper_rows_re_export")
                f["rows_agree"] = True
            if not f["byte_identical"] and not f["rows_agree"]:
                raise AdapterRefusal(where, f"re-export {d} of {target}: report.json is not byte-identical ({(inp.sha256 if inp else '-')[:12]}… vs {tgt.sha256[:12]}…) "
                                            "and no paper_rows.json agrees by cell")
            for r in rows_by_dir[target]:
                r["_extra"] = dict(r.get("_extra") or {})
                r["_extra"].setdefault("re_exports", []).append(d)
            dir_facts[d] = f
        # 4. not-production passes: one disposition per cell, nothing else
        for d in dirs:
            info = self.inventory.get(d) or {}
            if info.get("kind") != "not_production" or "report" not in by_dir[d]:
                continue
            inp = by_dir[d]["report"]
            rows, f = self.long_rows(inp.path, expect_pin=None, relaxed=True)
            status = f"not applicable: not a production run — {info.get('reason')}; the paper reads " + " / ".join(canon)
            f.update(kind="not_production", n_dispositions=0)
            for r in rows:
                was = (f"the check pass read {int(r['k'])}/{int(r['n'])} = {float(r['value']):.4f}" if r.get("status") == "measured" else f"the check pass filed: {r.get('status')}")
                r.update(status=status, value=float("nan"), k=float("nan"), n=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"), delta=float("nan"),
                         delta_lo=float("nan"), delta_hi=float("nan"), delta_p=float("nan"), note=f"{status}; {was}", _items=[], _superseded=None, _anchor_key=None,
                         _extra=dict(r.get("_extra") or {}, classification="not_production"))
                f["n_dispositions"] += 1
                disposition_rows.append(r)
            dir_facts[d] = f
        self.last_facts[group] = facts
        batch = RecordBatch(f"{self.name}:{group}", group)
        rep = DC.emit_rows(batch, ctx, live_rows + superseded_rows + disposition_rows, adapter=self.name, campaign=self.campaign,
                           reader="gsm8k_p200.load_gsm8k_waves",
                           seed_provenance="producer statistics read (B=10000 seed 0 per the report), none recomputed; cells deduped by declared ownership",
                           code_sha="unverified", code_sha_reason="the producer tree sha rides each leg's identity")
        self.last_report = rep
        yield batch

    def source_states(self, ctx: AdapterContext, inputs: Sequence[Input]) -> List[SourceStateRow]:
        self._plan()
        out = []
        for inp in inputs:
            if is_listing_input(inp):
                # the root inventory, the root pseudo-group's own row (ROOT_LISTING_RULE): keyed on the listing input, its fingerprint the sha --
                # and the pre-rule root-path key under `gsm8k_p200:<root basename>` stated absent, so a database that holds it never keeps it found = 1
                out.append(SourceStateRow(f"{self.name}:{self.root_group()}", inp.uri, "$.dirs", 1 if inp.entries else 0, 1, list(inp.entries), inp.sha256))
                out.append(SourceStateRow(self.legacy_root_source_id(), self.root, "$.dirs", 0, 0, [], None))
                continue
            if inp.role == "rows":
                continue
            d = read_json(inp.path)
            out.append(SourceStateRow(f"{self.name}:{os.path.basename(os.path.dirname(inp.uri))}", inp.uri, "$.cells", 1, 1, sorted((d.get("cells") or {}).keys()), inp.sha256))
        return out


def register() -> None:
    from .base import register as _register
    _register(Gsm8kP200Adapter.name, lambda: Gsm8kP200Adapter())

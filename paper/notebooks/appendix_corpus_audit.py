"""appendix_corpus_audit: the training-corpus audit's appendix table (Table 3, `tab:corpus-audit`, with a fuller notebook-only form) and
figure (`figA_corpus_audit`, notebook only), read from
a committed sidecar of aggregate counts, `data/appendix_corpus_audit.json`.

    import appendix_corpus_audit as CA
    ca = CA.load_corpus_audit()                   # the sidecar, refused by name unless its stamp, provenance and counts check out
    print(CA.provenance_line(ca))
    CA.corpus_audit_figure(ca, "figA_corpus_audit")
    print(CA.table_corpus_audit(ca))             # the LaTeX of tab:corpus-audit, the paper's Table 3
    print(CA.table_corpus_audit_full(ca))        # every claim and state the audit records (tab:corpus-audit-full, notebook only)
    print(CA.overlap_line(ca))                   # the deterministic overlap scan's maxima, as the Results paragraph quotes them

THE AUDIT read the trainable text of all 36,570 records of the two training corpora (constitution and control, advice and stories,
training and held-out sides) against three claims: C1 (templates and evaluation prompts; a deterministic eval-text half and a labelled
jailbreak half), C2 (requests of the kinds HarmBench, JailbreakBench or StrongREJECT test, or seeking serious harm) and C3 (requests to
help with a wrongful act short of serious harm). Each record ends in one state per claim: positive, negative, or not cleared (contested,
partially read, unresolved, or declined on input by the labeller's own safety classifier). The labels are by a corpus labeller, not an
evaluation judge, so the sidecar's `judge` field says so in those words (`JUDGE`); the C1 eval-text counts and the overlap maxima come
from a deterministic n-gram scan with no labeller.

THE SIDECAR holds aggregate counts only: no record text and no record id. `numbers` is a subset of the audit summary's own `numbers`
map, every value copied under its summary key (`required_keys()` names the subset), and `overlap` carries the scan's thresholds and the
role and status of each scanned set, less the sets of an evaluation family the paper omits (`omitted_family_set`), of which
`overlap.sets_left_out` keeps only the count, the longest shared run and the records over a threshold.
`paper/notebooks/data/build_appendix_corpus_audit.py` writes it from the audit's summary file and scan metadata (which stay outside the
repository) and cross-checks it against the manuscript's number ledger; its provenance block names those files by sha256.
`corpus_audit_problems` re-checks what the file alone can show: the stamp and schema, the provenance, that no string looks like a record
id, that no listed set is of an omitted evaluation family and none left out had a record over a threshold or a shared run longer than
the next-longest listed one (the text quotes the two longest), and that the counts agree with one another (each claim's states fit
within the part's records, every share equals its counts, the parts sum to the summary's totals, the blind-reread bound is the
Clopper-Pearson bound of its counts).

THE FIGURE keeps the content of the manuscript's interim figure (records per claim and corpus part; positive and not-cleared records,
negatives not drawn) in the notebook's style: `R.style()` fonts, `R.legend_stack` above the panels, titles bold top-left, saved as
png, pdf and svg by `appendix_common.save_figure`. Positive records take the corpus's colour in the paper's palette (the constitution
objects' green, the control objects' blue, as the cosine figure colours the two corpora); the two not-cleared states are greys, the
notebook's colour for a reading that is not there.
"""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

import registry_nb as R
import appendix_common as AC

HERE = os.path.dirname(os.path.abspath(__file__))
SIDECAR = os.path.join(HERE, "data", "appendix_corpus_audit.json")
SCHEMA = "appendix-corpus-audit-v1"
JUDGE = "labeller: Claude Opus 5.5 (corpus audit, not an evaluation judge)"
LABELLER_MODEL = "claude-opus-5-5"
SUMMARY_VERSION, RUBRIC_VERSION = "hrc-summary-v2", "hrc-v2"
BUILDER = "paper/notebooks/data/build_appendix_corpus_audit.py"
FIGURE_NAME = "figA_corpus_audit"
TABLE_LABEL = "tab:corpus-audit"
FULL_TABLE_LABEL = "tab:corpus-audit-full"
# Table 3's caption as the paper sets it; the counts are the sidecar's
TABLE_CAPTION = ("Records per corpus part with each kind of content (labels by Claude Opus~5.5 as a corpus auditor). Counts are lower "
                 "bounds, as the labeler declined to read some records.")

PARTS = ("constitution_advice", "constitution_stories", "control_advice", "control_stories")
PART_LABEL = {"constitution_advice": "Constitution advice", "constitution_stories": "Constitution stories",
              "control_advice": "Control advice", "control_stories": "Control stories"}
CORPUS = {p: p.split("_")[0] for p in PARTS}
# THE CORPUS COLOURS are the paper's: the constitution objects' green and the control objects' blue (the LoRA shades, dark enough to
# stand apart from the two greys in luminance as well as hue).
CORPUS_COLOR = {"constitution": R.COLOR["constitution_lora"], "control": R.COLOR["control_lora"]}
# THE NOT-CLEARED STATES ARE GREY, the notebook's colour for a reading that is not there (its gap glyphs): the small contested,
# partially read or unresolved states in the dark grey, the larger declined-on-input state in the lighter one.
OTHER_COLOR, DECLINED_COLOR = R.INK2, R.MUTED
# (claim, summary key prefix, share-of-read key stem, panel title)
CLAIMS = (("c2", "c2.headline", "c2_headline", "C2: benchmark-kind or\nserious-harm requests"),
          ("c1_jailbreak", "c1.c1_jailbreak", "c1_jailbreak", "C1 (jailbreak half): depicted\njailbreak attempts"),
          ("c3", "c3.any_wrongful", "any_wrongful", "C3: requests to help with\na wrongful act"))
STATES = ("positive", "contested", "partial", "unresolved", "declined_on_input")
OTHER_NOT_CLEARED = ("contested", "partial", "unresolved")
SEGMENT_LABEL = {"positive": "positive",
                 "other": "not cleared: contested, partially read or unresolved",
                 "declined": "not cleared: declined on input by the labeller's safety classifier"}
PETRI_SET = "petri_auditor_turns"            # the one attack-corpus set: auditor turns generated during each Petri audit, not fixed
C2_TREATMENT = ("requests_with_treatment", "declined_any", "complied_or_partial")
# A record id in the audit reads `<part>/adv-p39-t00-s01` or `<part>/sto-c47-u02-k11-y1`: no string in the sidecar may look like one.
RECORD_ID = re.compile(r"(?:\b(?:adv|sto)-[a-z]\d+-)|(?:\b(?:constitution|control)_(?:advice|stories)/)")
# no field of the sidecar may be named like the scan's per-record columns or a labelled span
FORBIDDEN_FIELDS = re.compile(r"(record_id|_record$|_item$|^quote|^span|^text$|^excerpt|^ids?$)")
TOP_FIELDS = {"schema", "about", "judge", "parts", "numbers", "overlap", "uplift_levels", "provenance"}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class CorpusAuditSidecarError(ValueError):
    """The corpus-audit sidecar failed its checks; the message lists every problem found."""


# ---- the sidecar ---------------------------------------------------------------------------------------------------------------------
def required_keys() -> list:
    """The summary keys every sidecar carries, in a fixed order: per part and in total the records, the records declined on input
    and the declined share; per claim every state's count per part and in total, and the share of read records that is positive; the
    C1 eval-text records; the records with uplift; the C2 agreement kappa; the blind reread's counts and bound; and the C2 requests
    judged partial uplift. The overlap maxima (`overlap.*` keys, one per scanned set) are required by set, from `overlap.sets`."""
    keys = []
    for p in PARTS + ("all",):
        keys += [f"coverage.records.{p}", f"declined_on_input.records.{p}"]
    keys += [f"pct.share.declined_on_input.records.{p}" for p in PARTS]
    for _c, pre, stem, _t in CLAIMS:
        keys += [f"{pre}.{s}.{p}" for s in STATES for p in PARTS + ("all",)]
        keys += [f"pct.share_read.{stem}.positive.{p}" for p in PARTS]
    keys += [f"c1.eval_text.records.{p}" for p in PARTS + ("all",)]
    keys += [f"response.any_uplift.positive.{p}" for p in PARTS + ("all",)]
    keys += [f"agreement.c2_headline.kappa.{p}" for p in PARTS]
    keys += [f"fn.advice_census.{k}.{p}" for k in ("n", "misses_confirmed") for p in ("constitution_advice", "control_advice")]
    keys += [f"fn.headline.{k}.{p}" for k in ("n", "x", "misses_confirmed", "cp_upper95") for p in ("constitution_stories", "control_stories")]
    keys += ["c2.requests.uplift.partial.constitution_advice", "c2.requests.uplift.partial.all"]
    return keys


def optional_keys() -> list:
    """Summary keys a sidecar carries when the summary has them: the C2 request treatment counts, which exist only for a part with C2
    requests (the table prints `--` for a part without them, and refuses a part with C2 positives but no treatment counts)."""
    return [f"c2.requests.total.{k}.{p}" for k in C2_TREATMENT for p in PARTS + ("all",)]


def overlap_keys(set_ids, goal_sets, expected_ids) -> list:
    """The overlap keys for the scanned sets: every set's longest shared run and records over a threshold, every goal set's highest
    goal coverage, and every expected-by-construction set's longest run."""
    return ([f"overlap.max_run.{s}" for s in set_ids] + [f"overlap.records_eval_text_positive.{s}" for s in set_ids]
            + [f"overlap.max_coverage.{s}" for s in goal_sets] + [f"overlap_expected.max_run.{s}" for s in expected_ids])


def omitted_family_set(set_id) -> bool:
    """True for a scanned prompt set of an evaluation family the paper omits (`registry_nb.OMITTED_EVALS`): a set whose id opens with
    such an evaluation's family stem, the part of the eval id before its first underscore. The builder leaves these sets out of the
    sidecar: their ids are not listed and their numbers not copied; `overlap.sets_left_out` keeps how many there were, their longest
    shared run and their records over a threshold, so leaving them out can be seen to change no count the table or text quotes."""
    stems = {str(e).split("_", 1)[0] for e in R.OMITTED_EVALS}
    return str(set_id).split("_", 1)[0] in stems


LEFT_OUT_FIELDS = ("n_sets", "max_run", "records_over_threshold", "why")
LEFT_OUT_WHY = ("scanned prompt sets of an evaluation family the paper omits (the notebook's OMITTED_EVALS): their ids are not listed "
                "and their per-set numbers not copied; only their count, their longest shared run in words and their records over a "
                "threshold are kept")


def left_out_problems(ov: dict) -> list:
    """`overlap.sets_left_out` against its contract: the four fields, counts that are non-negative integers, no listed set of an
    omitted family, and no record over a threshold among the sets left out (footnote a of the fuller table counts those records)."""
    lo = ov.get("sets_left_out")
    if not isinstance(lo, dict) or tuple(lo) != LEFT_OUT_FIELDS:
        return [f"overlap.sets_left_out {lo!r}: expected the fields {list(LEFT_OUT_FIELDS)}"]
    out = []
    for k in LEFT_OUT_FIELDS[:3]:
        v = lo[k]
        if not (isinstance(v, int) and not isinstance(v, bool) and v >= 0):
            out.append(f"overlap.sets_left_out.{k} {v!r} is not a non-negative integer")
    if lo["why"] != LEFT_OUT_WHY:
        out.append("overlap.sets_left_out.why is not the reader's wording")
    if not out and lo["n_sets"] == 0 and (lo["max_run"] or lo["records_over_threshold"]):
        out.append("overlap.sets_left_out: no set left out, yet a longest run or records over a threshold")
    if not out and lo["records_over_threshold"]:
        out.append(f"overlap.sets_left_out: {lo['records_over_threshold']} record(s) over a threshold in a set left out")
    listed = sorted(s.get("set_id") for s in ov.get("sets") or [] if omitted_family_set(s.get("set_id")))
    if listed:
        out.append(f"overlap sets list {len(listed)} set(s) of an evaluation family the paper omits")
    return out


def left_out_run_problems(n: dict, ov: dict) -> list:
    """The sets left out against the runs the text quotes: `overlap_facts` names the longest and the next-longest shared run over the
    listed fixed sets, so a set left out may not hold a run longer than that next-longest one (were it longer, the two quoted runs
    would understate the scan as it ran). A tie with the next-longest leaves both quoted values true."""
    lo = ov.get("sets_left_out") or {}
    runs = sorted((int(n[f"overlap.max_run.{s['set_id']}"]) for s in ov.get("sets") or []
                   if s.get("role") != "attack_corpus" and s.get("status") == "materialised"), reverse=True)
    if isinstance(lo.get("max_run"), int) and len(runs) > 1 and lo["max_run"] > runs[1]:
        return [f"overlap.sets_left_out: a longest shared run of {lo['max_run']} words, longer than the next-longest listed "
                f"({runs[1]}), so the quoted longest runs would understate the scan"]
    return []


def load_corpus_audit(path: str = SIDECAR) -> dict:
    """The sidecar at `path`, after `corpus_audit_problems`; raises CorpusAuditSidecarError naming every problem."""
    with open(path, encoding="utf-8") as fh:
        ca = json.load(fh)
    problems = corpus_audit_problems(ca)
    if problems:
        raise CorpusAuditSidecarError(f"{os.path.basename(path)}: " + "; ".join(problems))
    return ca


def _strings(obj, where=""):
    """Every (address, string) in a JSON value, mapping keys included."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{where}{{{k}}}", str(k)
            yield from _strings(v, f"{where}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _strings(v, f"{where}[{i}]")
    elif isinstance(obj, str):
        yield where, obj


def _fields(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _fields(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _fields(v)


def _close(a: float, b: float, tol: float) -> bool:
    return abs(float(a) - float(b)) <= tol


def corpus_audit_problems(ca: dict) -> list:
    """Every reason the sidecar must not be drawn, as short phrases (empty when it checks out): the schema and the judge stamp, the
    provenance (summary and rubric version, the builder, each source with a sha256), no record text or id anywhere, every required
    key present with a number, and the counts' agreement with one another."""
    out = []
    if not isinstance(ca, dict):
        return ["the sidecar is not a JSON object"]
    if ca.get("schema") != SCHEMA:
        out.append(f"schema {ca.get('schema')!r}, expected {SCHEMA!r}")
    if ca.get("judge") != JUDGE:
        out.append(f"judge stamp {ca.get('judge')!r}, expected {JUDGE!r}")
    extra = sorted(set(ca) - TOP_FIELDS)
    if extra:
        out.append(f"unexpected top-level field(s) {extra}")
    if tuple(ca.get("parts") or ()) != PARTS:
        out.append(f"parts {ca.get('parts')!r}, expected {list(PARTS)}")
    prov = ca.get("provenance") or {}
    if prov.get("summary_version") != SUMMARY_VERSION or prov.get("rubric_version") != RUBRIC_VERSION:
        out.append(f"provenance names summary {prov.get('summary_version')!r} / rubric {prov.get('rubric_version')!r}, expected "
                   f"{SUMMARY_VERSION!r} / {RUBRIC_VERSION!r}")
    if prov.get("builder") != BUILDER or not _HEX64.match(str(prov.get("builder_sha256", ""))):
        out.append(f"provenance does not name the builder {BUILDER} with a sha256")
    judge = prov.get("judge") or {}
    if judge.get("labeller_model") != LABELLER_MODEL or judge.get("evaluation_judge") is not False:
        out.append(f"provenance judge block {judge!r}: expected labeller_model {LABELLER_MODEL!r}, evaluation_judge false")
    sources = (prov.get("sources") or []) + (prov.get("cross_checked") or [])
    roots = prov.get("roots") or {}
    if not any(s.get("root") == "audit" and s.get("file") == "summary.json" for s in prov.get("sources") or []):
        out.append("provenance names no audit summary.json source")
    for s in sources:
        if s.get("root") not in roots or not s.get("file") or not _HEX64.match(str(s.get("sha256", ""))):
            out.append(f"provenance source {s!r} lacks a declared root, a file or a sha256")
        elif os.path.isabs(str(s["file"])) or str(s["file"]).startswith(("~", "..")):
            out.append(f"provenance source {s['file']!r} is not a path inside its root")
    for where, s in _strings(ca):
        if RECORD_ID.search(s):
            out.append(f"a string that looks like a record id at {where or 'top'}")
    bad = sorted({f for f in _fields(ca) if FORBIDDEN_FIELDS.search(str(f))})
    if bad:
        out.append(f"field(s) named like per-record data: {bad}")
    num = ca.get("numbers")
    ov = ca.get("overlap") or {}
    if not isinstance(num, dict):
        return out + ["no `numbers` map"]
    sets = ov.get("sets") or []
    set_ids = [s.get("set_id") for s in sets]
    goal_sets = [s.get("set_id") for s in sets if s.get("goal_set")]
    expected = [s.get("set_id") for s in ov.get("expected_sets") or []]
    if PETRI_SET not in set_ids:
        out.append(f"overlap sets lack {PETRI_SET}")
    th = ov.get("thresholds") or {}
    if not isinstance(th.get("route_run"), int) or not isinstance(th.get("goal_frac"), (int, float)):
        out.append(f"overlap thresholds {th!r} lack route_run (words) and goal_frac")
    out += left_out_problems(ov)
    need = required_keys() + overlap_keys(set_ids, goal_sets, expected)
    nullable = {f"agreement.c2_headline.kappa.{p}" for p in PARTS}
    missing = [k for k in need if k not in num]
    if missing:
        out.append(f"{len(missing)} required key(s) missing, e.g. {missing[:4]}")
    allowed = set(need) | set(optional_keys())
    stray = sorted(set(num) - allowed)
    if stray:
        out.append(f"{len(stray)} key(s) outside the declared subset, e.g. {stray[:4]}")
    nonnum = [k for k, v in num.items() if not ((v is None and k in nullable) or (isinstance(v, (int, float)) and not isinstance(v, bool)
                                                                                    and math.isfinite(v)))]
    if nonnum:
        out.append(f"non-numeric value(s) at {nonnum[:4]}")
    if missing or nonnum:
        return out
    out += _consistency(num, ov, ca.get("uplift_levels"))
    out += left_out_run_problems(num, ov)
    return out


def _consistency(n: dict, ov: dict, uplift_levels) -> list:
    """The counts against one another: what a copied subset of a self-consistent summary must satisfy."""
    out = []
    g = n.get
    for p in PARTS:
        rec, dec = g(f"coverage.records.{p}"), g(f"declined_on_input.records.{p}")
        if not rec or rec <= 0 or dec < 0 or dec > rec:
            out.append(f"{p}: records {rec}, declined on input {dec}")
            continue
        if not _close(g(f"pct.share.declined_on_input.records.{p}"), 100.0 * dec / rec, 1e-4):
            out.append(f"{p}: declined share {g(f'pct.share.declined_on_input.records.{p}')} is not 100 x {dec} / {rec}")
        read = rec - dec
        for c, pre, stem, _t in CLAIMS:
            st = {s: g(f"{pre}.{s}.{p}") for s in STATES}
            if any(v < 0 or v != int(v) for v in st.values()):
                out.append(f"{c} {p}: a state count is negative or fractional ({st})")
            if st["declined_on_input"] != dec:
                out.append(f"{c} {p}: declined on input {st['declined_on_input']}, but the part's record-level count is {dec}")
            if sum(st.values()) > rec:
                out.append(f"{c} {p}: the drawn states sum to {sum(st.values())}, more than the part's {rec} records")
            if not _close(g(f"pct.share_read.{stem}.positive.{p}"), 100.0 * st["positive"] / read, 1e-4):
                out.append(f"{c} {p}: share of read records {g(f'pct.share_read.{stem}.positive.{p}')} is not 100 x {st['positive']} / {read}")
    sums = ["coverage.records", "declined_on_input.records", "c1.eval_text.records", "response.any_uplift.positive"]
    sums += [f"{pre}.{s}" for _c, pre, _st, _t in CLAIMS for s in STATES]
    for k in sums:
        if sum(g(f"{k}.{p}") for p in PARTS) != g(f"{k}.all"):
            out.append(f"{k}: the parts sum to {sum(g(f'{k}.{p}') for p in PARTS)}, the summary's total is {g(f'{k}.all')}")
    for p in PARTS:
        have = [k for k in C2_TREATMENT if f"c2.requests.total.{k}.{p}" in n]
        if have and len(have) != len(C2_TREATMENT):
            out.append(f"{p}: C2 treatment counts {have} only, expected all of {list(C2_TREATMENT)}")
        elif have:
            t = {k: g(f"c2.requests.total.{k}.{p}") for k in C2_TREATMENT}
            if t["declined_any"] + t["complied_or_partial"] > t["requests_with_treatment"]:
                out.append(f"{p}: C2 requests declined {t['declined_any']} + complied {t['complied_or_partial']} exceed {t['requests_with_treatment']}")
        elif g(f"c2.headline.positive.{p}"):
            out.append(f"{p}: {g(f'c2.headline.positive.{p}')} C2-positive records but no C2 treatment counts")
    for p in ("constitution_stories", "control_stories"):
        x, nn, ub = g(f"fn.headline.x.{p}"), g(f"fn.headline.n.{p}"), g(f"fn.headline.cp_upper95.{p}")
        if g(f"fn.headline.misses_confirmed.{p}") != x:
            out.append(f"{p}: blind reread misses {g(f'fn.headline.misses_confirmed.{p}')} differ from the bound's count {x}")
        if x == 0 and nn > 0 and not _close(ub, 1.0 - 0.05 ** (1.0 / nn), 1e-6):
            out.append(f"{p}: one-sided 95% upper bound {ub} is not 1 - 0.05^(1/{nn})")
    if not isinstance(uplift_levels, list) or not uplift_levels:
        out.append("no `uplift_levels` list (the uplift levels the summary names)")
    return out


# ---- frames ------------------------------------------------------------------------------------------------------------------------
def claim_states(ca: dict) -> pd.DataFrame:
    """One row per (claim, part): every drawn state's count, `other` (contested + partial + unresolved), `not_cleared` (other +
    declined on input), `records` and the summary's share of read records that is positive (`share_read_pct`)."""
    n = ca["numbers"]
    rows = []
    for c, pre, stem, title in CLAIMS:
        for p in PARTS:
            st = {s: int(n[f"{pre}.{s}.{p}"]) for s in STATES}
            other = sum(st[s] for s in OTHER_NOT_CLEARED)
            rows.append(dict(claim=c, part=p, title=title, **st, other=other, not_cleared=other + st["declined_on_input"],
                             records=int(n[f"coverage.records.{p}"]), share_read_pct=float(n[f"pct.share_read.{stem}.positive.{p}"])))
    return pd.DataFrame(rows)


def figure_values(ca: dict) -> pd.DataFrame:
    """The three segments the figure draws per (claim, part), in drawing order: positive, other not cleared, declined on input."""
    s = claim_states(ca)
    return s[["claim", "part", "positive", "other", "declined_on_input"]].rename(columns={"declined_on_input": "declined"})


def bar_label(positive: int, not_cleared: int) -> str:
    """The text at a bar's end: the positive count, then the not-cleared count when there is one."""
    return f"{positive:,}" + (f" + {not_cleared:,} not cleared" if not_cleared else "")


def overlap_maxima(ca: dict) -> pd.DataFrame:
    """One row per scanned set (and per expected-by-construction set): its tier, role and status, whether it is a goal set, its
    longest shared run in words, its highest goal coverage (goal sets only), and its records over a threshold. `kind` is `fixed` for
    a fixed evaluation prompt set or template, `attack_corpus` for the Petri auditor's turns (generated during each audit), and
    `expected` for a set that contains corpus text by construction (the prompted baselines' system prompts, the selection probes)."""
    n, ov = ca["numbers"], ca["overlap"]
    rows = []
    for s in ov["sets"]:
        sid = s["set_id"]
        rows.append(dict(set_id=sid, tier=s.get("tier", ""), role=s.get("role", ""), status=s.get("status", ""),
                         goal_set=bool(s.get("goal_set")), standin=bool(s.get("standin")),
                         kind="attack_corpus" if s.get("role") == "attack_corpus" else "fixed",
                         max_run=int(n[f"overlap.max_run.{sid}"]), max_coverage=n.get(f"overlap.max_coverage.{sid}", np.nan),
                         records_over_threshold=int(n[f"overlap.records_eval_text_positive.{sid}"])))
    for s in ov.get("expected_sets") or []:
        sid = s["set_id"]
        rows.append(dict(set_id=sid, tier=s.get("tier", ""), role=s.get("role", ""), status=s.get("status", ""), goal_set=False,
                         standin=False, kind="expected", max_run=int(n[f"overlap_expected.max_run.{sid}"]), max_coverage=np.nan,
                         records_over_threshold=np.nan))
    return pd.DataFrame(rows)


def overlap_facts(ca: dict) -> dict:
    """The overlap numbers the Results paragraph and the table's footnote quote: over the scanned fixed sets, the longest and the
    second-longest shared run (with their sets) and the highest goal coverage; the attack-corpus set's records over a threshold and its
    longest run; the expected sets' longest runs; the sets not scanned (status other than `materialised`); the thresholds; and how many
    scanned sets of an omitted evaluation family the sidecar leaves out."""
    m = overlap_maxima(ca)
    th = ca["overlap"]["thresholds"]
    fixed = m[(m.kind == "fixed") & (m.status == "materialised")].sort_values(["max_run", "set_id"], ascending=[False, True])
    cov = fixed[fixed.goal_set & fixed.max_coverage.notna()]
    petri = m[m.set_id == PETRI_SET].iloc[0]
    return dict(route_run=int(th["route_run"]), goal_frac=float(th["goal_frac"]),
                longest=(fixed.iloc[0].set_id, int(fixed.iloc[0].max_run)), second=(fixed.iloc[1].set_id, int(fixed.iloc[1].max_run)),
                max_coverage=float(cov.max_coverage.max()), max_coverage_sets=sorted(cov[cov.max_coverage == cov.max_coverage.max()].set_id),
                fixed_over_threshold=int(m.loc[m.kind == "fixed", "records_over_threshold"].sum()),
                petri_records=int(petri.records_over_threshold), petri_max_run=int(petri.max_run),
                expected={r.set_id: int(r.max_run) for r in m[m.kind == "expected"].itertuples()},
                not_scanned=sorted(m.loc[(m.kind != "expected") & (m.status != "materialised"), "set_id"]),
                standins=sorted(m.loc[m.standin, "set_id"]), n_sets=int((m.kind != "expected").sum()),
                n_left_out=int(ca["overlap"]["sets_left_out"]["n_sets"]))


def overlap_line(ca: dict) -> str:
    """The overlap scan in one printed line, from `overlap_facts`."""
    f = overlap_facts(ca)
    exp = ", ".join(f"{k} {v}" for k, v in f["expected"].items())
    n, left = f["n_sets"] + f["n_left_out"], f["n_left_out"]
    sets = f"{n} sets" + (f", {left} of them from an evaluation family the paper omits and not listed" if left else "")
    return (f"overlap scan (deterministic, {sets}; a record is over a threshold at a shared run of {f['route_run']} words or "
            f"{f['goal_frac']:.0%} of a goal's words): fixed sets over a threshold {f['fixed_over_threshold']} records; longest shared run "
            f"{f['longest'][1]} words ({f['longest'][0]}), next {f['second'][1]} ({f['second'][0]}); highest goal coverage "
            f"{f['max_coverage']:.2f} ({', '.join(f['max_coverage_sets'])}) · {PETRI_SET} (generated during each audit): "
            f"{f['petri_records']} records over a threshold, longest run {f['petri_max_run']} words · excluded as corpus text by "
            f"construction: {exp} · not scanned: {', '.join(f['not_scanned']) or 'none'}"
            + (f" · scanned at a stand-in revision: {', '.join(f['standins'])}" if f["standins"] else ""))


def provenance_line(ca: dict) -> str:
    """The sidecar's provenance in one printed line."""
    p = ca["provenance"]
    src = "; ".join(f"{s['file']} {s['sha256'][:12]}" for s in p["sources"])
    return (f"corpus-audit sidecar {R.display_path(SIDECAR)}: {p['summary_version']} (rubric {p['rubric_version']}), sources {src}; "
            f"built by {p['builder']} ({p['builder_sha256'][:12]}) · {ca['judge']} · the C1 eval-text row and the overlap maxima are a "
            "deterministic scan · aggregate counts only, no record text or id")


# ---- the table -----------------------------------------------------------------------------------------------------------------------
def _n(v) -> str:
    """A count with LaTeX thousands separators: 16{,}321."""
    return f"{int(v):,}".replace(",", "{,}")


def _pct(v) -> str:
    """A share in percent at one decimal, or two where a non-zero share is under 0.1 (so 0.05 does not print as 0.1)."""
    return f"{v:.2f}" if 0 < v < 0.1 else f"{v:.1f}"


def paper_table_rows(ca: dict) -> list:
    """(row label, [one cell per part]) for the paper's Table 3 (tab:corpus-audit): the records, the depicted jailbreak attempts (the C1
    jailbreak half), the harmful requests (C2) and the requests for a lesser wrongful act (C3), positive records each, and the records
    the labeller declined on input; the cells as LaTeX, in the paper's row order."""
    n = ca["numbers"]
    s = claim_states(ca).set_index(["claim", "part"])
    return [("Records", [_n(n[f"coverage.records.{p}"]) for p in PARTS]),
            ("Depicted jailbreak attempt", [_n(s.loc[("c1_jailbreak", p), "positive"]) for p in PARTS]),
            ("Harmful request", [_n(s.loc[("c2", p), "positive"]) for p in PARTS]),
            ("Request for a lesser wrongful act", [_n(s.loc[("c3", p), "positive"]) for p in PARTS]),
            ("Declined by the labeler", [_n(n[f"declined_on_input.records.{p}"]) for p in PARTS])]


def table_rows(ca: dict) -> list:
    """(row label, [one cell per part]) for the fuller table (tab:corpus-audit-full), every claim and state the audit records, the
    cells as LaTeX; None marks a spacer."""
    n = ca["numbers"]
    s = claim_states(ca).set_index(["claim", "part"])
    rows = [("Records", [_n(n[f"coverage.records.{p}"]) for p in PARTS]),
            ("\\quad declined on input (not cleared)", [_n(n[f"declined_on_input.records.{p}"]) for p in PARTS]),
            None,
            ("C1 eval-text: over a threshold$^a$", [_n(n[f"c1.eval_text.records.{p}"]) for p in PARTS]),
            ("C1 jailbreak half: positive", [_n(s.loc[("c1_jailbreak", p), "positive"]) for p in PARTS]),
            None,
            ("C2: positive", [_n(s.loc[("c2", p), "positive"]) for p in PARTS]),
            ("\\quad share of records read (\\%)", [_pct(s.loc[("c2", p), "share_read_pct"]) for p in PARTS]),
            ("\\quad other not cleared$^b$", [_n(s.loc[("c2", p), "other"]) for p in PARTS]),
            ("\\quad C2 requests declined", [(f"{_n(n[f'c2.requests.total.declined_any.{p}'])}/{_n(n[f'c2.requests.total.requests_with_treatment.{p}'])}"
                                              if f"c2.requests.total.requests_with_treatment.{p}" in n else "--") for p in PARTS]),
            ("\\quad with uplift$^c$", [_n(n[f"response.any_uplift.positive.{p}"]) for p in PARTS]),
            None,
            ("C3: positive", [_n(s.loc[("c3", p), "positive"]) for p in PARTS]),
            ("\\quad share of records read (\\%)", [_pct(s.loc[("c3", p), "share_read_pct"]) for p in PARTS]),
            None,
            ("C2 agreement, Cohen's $\\kappa$", [("--" if n[f"agreement.c2_headline.kappa.{p}"] is None else f"{n[f'agreement.c2_headline.kappa.{p}']:.2f}")
                                                for p in PARTS]),
            ("Blind reread: misses$^d$", [f"{_n(n[_reread(p, 'misses_confirmed')])}/{_n(n[_reread(p, 'n')])}" for p in PARTS])]
    return rows


def _reread(part: str, what: str) -> str:
    """The blind reread's key for a part: the advice census for advice, the random-story sample for stories."""
    return f"fn.advice_census.{what}.{part}" if part.endswith("_advice") else f"fn.headline.{what}.{part}"


def _footnote_checks(ca: dict) -> list:
    """What the caption's footnotes assert, checked against the counts: (a) every C1 eval-text record is over the threshold through a
    shared run with the Petri auditor's turns, none through a fixed set; (c) the uplift is partial (requests judged partial uplift exist
    wherever records with uplift do) and no uplift level above partial exists in the summary."""
    f = overlap_facts(ca)
    n = ca["numbers"]
    out = []
    if f["fixed_over_threshold"] or f["petri_records"] != n["c1.eval_text.records.all"]:
        out.append(f"footnote a: {f['fixed_over_threshold']} records over a threshold through a fixed set and {f['petri_records']} through "
                   f"{PETRI_SET}, of {n['c1.eval_text.records.all']} C1 eval-text records")
    if set(ca["uplift_levels"]) - {"none", "partial"}:
        out.append(f"footnote c: uplift levels {ca['uplift_levels']} include one above partial")
    if n["response.any_uplift.positive.all"] and not n["c2.requests.uplift.partial.all"]:
        out.append(f"footnote c: {n['response.any_uplift.positive.all']} records with uplift but no C2 request judged partial uplift")
    return out


def table_caption(ca: dict) -> str:
    """The caption of the fuller table (tab:corpus-audit-full), its numbers read from the sidecar."""
    n = ca["numbers"]
    f = overlap_facts(ca)
    ub = {p: 100.0 * n[f"fn.headline.cp_upper95.{p}"] for p in ("constitution_stories", "control_stories")}
    bound = (f"{ub['constitution_stories']:.2f}\\%" if f"{ub['constitution_stories']:.2f}" == f"{ub['control_stories']:.2f}" else
             f"{ub['constitution_stories']:.2f}\\% (constitution) and {ub['control_stories']:.2f}\\% (control)")
    return ("What the training corpora contain: records per claim and corpus part, from the corpus audit (labels by Claude Opus 5.5, "
            "not an evaluation judge; the C1 eval-text row is a deterministic scan). The labeller's safety classifier declined "
            f"{n['pct.share.declined_on_input.records.constitution_advice']:.1f}\\% of the constitution advice and "
            f"{n['pct.share.declined_on_input.records.constitution_stories']:.1f}\\% of the constitution stories; shares are over the "
            "records it read. Records that are not cleared are neither zeros nor positives, so positive counts are lower bounds. "
            f"$^a$All through a {f['route_run']}-word run shared with a Petri auditor turn. $^b$Contested, partially read or unresolved. "
            "$^c$Partial uplift; none substantial. $^d$Confirmed misses on C2, the C1 jailbreak half or uplift when unflagged records "
            f"were reread blind (one-sided 95\\% upper bound for stories, {bound}).")


def _latex(rows: list, caption: str, label: str) -> str:
    body = ["\\addlinespace" if r is None else f"{r[0]:<38} & " + " & ".join(f"{c:<9}" for c in r[1]).rstrip() + " \\\\" for r in rows]
    return ("\\begin{table}[ht]\n\\caption{" + caption + "}\n\\label{" + label + "}\n\\begin{center}\n\\small\n"
            "\\begin{tabular}{lrrrr}\n\\toprule\n"
            " & \\multicolumn{2}{c}{\\bf CONSTITUTION} & \\multicolumn{2}{c}{\\bf CONTROL} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
            " & \\multicolumn{1}{c}{advice} & \\multicolumn{1}{c}{stories} & \\multicolumn{1}{c}{advice} & \\multicolumn{1}{c}{stories} \\\\\n"
            "\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{center}\n\\end{table}")


def table_corpus_audit(ca: dict, caption: str = TABLE_CAPTION, label: str = TABLE_LABEL) -> str:
    """The paper's Table 3 (tab:corpus-audit) as the manuscript sets it (booktabs, the two corpora as column groups over their advice
    and story parts): five rows (`paper_table_rows`), every cell from the sidecar."""
    return _latex(paper_table_rows(ca), caption, label)


def table_corpus_audit_full(ca: dict, caption: str | None = None, label: str = FULL_TABLE_LABEL) -> str:
    """The fuller table (tab:corpus-audit-full, drawn in the notebook only): every claim and state the audit records, every cell and the
    caption's numbers from the sidecar. Refuses when a footnote's claim does not hold for the counts."""
    bad = _footnote_checks(ca)
    if bad:
        raise CorpusAuditSidecarError("the table's footnotes would be false: " + "; ".join(bad))
    return _latex(table_rows(ca), caption or table_caption(ca), label)


# ---- the figure ----------------------------------------------------------------------------------------------------------------------
def _fit_xlim(fig, panels: list, pad: float = 0.02, room: float = 1.03) -> None:
    """Set each panel's x range so its bar-end labels fit inside it: with a label `w` pixels wide on an axis `W` pixels wide, a bar of
    total t needs xmax >= t / (1 - pad - w / W). Measured on the drawn canvas, twice (a new range can move the layout a little)."""
    for _ in range(2):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        for ax, items in panels:
            W = ax.get_window_extent(r).width
            need = 1.0
            for total, txt in items:
                frac = pad + txt.get_window_extent(r).width / W
                if frac >= 0.95:
                    raise ValueError(f"bar label {txt.get_text()!r} is wider than its panel: widen the figure")
                need = max(need, total / (1.0 - frac))
            xmax = need * room
            ax.set_xlim(0, xmax)
            for total, txt in items:
                txt.set_x(total + pad * xmax)


def corpus_audit_figure(ca: dict, name: str | None = FIGURE_NAME, *, out_dirs=None, extra_dirs=(), width: float = AC.MODEL_WIDTH,
                        height: float = 5.0, bar_height: float = 0.6):
    """Records per claim and corpus part: one panel per claim (C2, the C1 jailbreak half, C3), one bar per corpus part, stacked as
    positive records (the corpus's colour), then the contested, partially read or unresolved records (dark grey), then the records the
    labeller's safety classifier declined on input (light grey); negatives are not drawn. Each bar ends in its positive count and, where
    there are any, its not-cleared count. Saved as `name` (png, pdf, svg) by `appendix_common.save_figure` into `out_dirs` (default
    `R.FIGS`) plus the mirrors and `extra_dirs` when a name is given, and returned. Every bar carries a gid
    `corpus-audit|<claim>|<part>|<positive|other|declined>`, every label `corpus-audit-label|<claim>|<part>`."""
    v = claim_states(ca)
    y = np.arange(len(PARTS))[::-1].astype(float)
    with AC.style_context():
        fig, axes = plt.subplots(1, len(CLAIMS), figsize=(width, height), layout="constrained")
        fs = plt.rcParams["font.size"]
        panels = []
        for ax, (c, _pre, _stem, title) in zip(axes, CLAIMS):
            sub = v[v.claim == c].set_index("part").loc[list(PARTS)]
            left = np.zeros(len(PARTS))
            for seg, vals, colours in (("positive", sub.positive.values, [CORPUS_COLOR[CORPUS[p]] for p in PARTS]),
                                       ("other", sub.other.values, [OTHER_COLOR] * len(PARTS)),
                                       ("declined", sub.declined_on_input.values, [DECLINED_COLOR] * len(PARTS))):
                bars = ax.barh(y, vals, left=left, height=bar_height, color=colours, edgecolor="white", linewidth=1.0, zorder=3)
                for b, p in zip(bars, PARTS):
                    b.set_gid(f"corpus-audit|{c}|{p}|{seg}")
                left = left + vals
            items = []
            for yi, p, total in zip(y, PARTS, left):
                txt = ax.text(total, yi, bar_label(int(sub.loc[p, "positive"]), int(sub.loc[p, "not_cleared"])), va="center", ha="left",
                              color=R.INK, fontsize=fs, gid=f"corpus-audit-label|{c}|{p}", zorder=4)
                txt.set_in_layout(False)          # placed inside the panel by _fit_xlim; it must not shrink the panel meanwhile
                items.append((float(total), txt))
            panels.append((ax, items))
            ax.set_yticks(y)
            ax.set_yticklabels([PART_LABEL[p] for p in PARTS] if ax is axes[0] else [])
            ax.set_ylim(-0.6, len(PARTS) - 0.4)
            ax.set_title(title)
            ax.set_xlabel("records")
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10], integer=True))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:,.0f}"))
            ax.grid(True, axis="x", color=R.RULE, lw=0.9, zorder=0)
            ax.set_axisbelow(True)
        _fit_xlim(fig, panels)
        handles = [[Patch(facecolor=CORPUS_COLOR["constitution"], edgecolor="none", label="positive: constitution corpus"),
                    Patch(facecolor=CORPUS_COLOR["control"], edgecolor="none", label="positive: control corpus")],
                   [Patch(facecolor=OTHER_COLOR, edgecolor="none", label=SEGMENT_LABEL["other"]),
                    Patch(facecolor=DECLINED_COLOR, edgecolor="none", label=SEGMENT_LABEL["declined"])]]
        R.legend_stack(fig, [(None, [h for g in handles for h in g], len(handles))])
    if name:
        AC.save_figure(fig, name, dirs=out_dirs, extra_dirs=extra_dirs)
    return fig

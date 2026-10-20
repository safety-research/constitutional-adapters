"""Offline filter chain -> final.jsonl. Deterministic, reproducible from
stored artifacts with zero API calls. Every drop is logged with reason +
stage + record id.

Chain order (each rule sees only survivors of the previous one):
1. schema validation (pydantic, full lineage)
2. persona guard — HARD gate on TRAINABLE text only (advice
   system/user/assistant; stories text). Mentions in generated non-trainable
   metadata (rewrite_problems, critiques, rationales, draft/diagnostic
   lineage) are counted + sampled in the report, never dropped — the pilot
   lost 31% of stories to constitution-quoting critique notes.
3. hard flag gates: ai_named=true drops regardless of scores; stories
   factual_claims=true likewise
4. sanity filters: empty/near-empty; story length bounds; placeholder text;
   stray pipeline tags (a "</story>" echo — 2026-09-03); degenerate
   repetition; advice refusal=true
5. score thresholds — calibrated on the pilot's stratified inspection to a
   precision objective, NEVER to a target pass rate. A FINAL corpus REFUSES
   null thresholds (ThresholdsNotCalibrated); pre-calibration pilots run in
   explicit calibration mode (`corpusgen filter --calibrate`), which writes a
   clearly non-final calibration artifact and records the skip in the report.
   A record with NO score verdict at all (its scoring wave was lost through
   the repair budget) drops here with the counted reason "unscored" — in
   calibration mode too — so an unscored record can never pass silently.
6. near-dedup (MinHash LSH; keep lexicographically-first lineage id)

7. register caps (post-review Fix 2, 2026-09-01): config-exposed regex
   families (``filter.register_families``) whose members may make up at most
   ``max_frac`` (strictly less) of the final corpus — the "not the only/last
   safeguard" story beat and the "Deployment context:" / "Internal
   deployment/tool" advice system-prompt openings. Excess members are dropped
   deterministically (seeded hash order, round-robin across diversity groups
   so survivors keep category/length/ordinary or principle/type spread).

7b. seeded stratified downsample:
   when ``filter.downsample.<corpus>`` is set and the survivor set overshoots
   it, the corpus is trimmed to the target — each (lane) stratum keeps its
   proportional share, survivors chosen in seeded hash order — and the
   register caps are re-applied to the trimmed set so every family still ends
   strictly under its cap of the RELEASED corpus. Counted drops ("downsample"),
   before/after mix histograms in the report. Never pads on the downside.

8. length gate: response word-length
   statistics of the FINAL set are always reported; when
   ``filter.length_gates.<corpus>`` is set they are judged against the
   frozen v2 reference (mean ±10% / p90 ±15% by default) — a miss is a
   visible acceptance failure here and, when blocking, a release refusal in
   the mirror gate. Never a per-record drop: the gate measures the corpus.

Token accounting runs on survivors (min across the three reference
tokenizers) and is stored per record.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from . import persona
from .config import RegisterFamily, RunConfig, filter_report_path
from .dedup import dedup
from .io import atomic_write, write_jsonl_atomic
from .lengths import evaluate_length_gate, length_stats
from .schema import AdviceRecord, StoryRecord
from .tokens import TokenCounter, render_advice_text, render_story_text

PLACEHOLDER_RE = re.compile(
    r"\[CITATION|\[insert|\[INSERT|TODO:|\[placeholder|\[PLACEHOLDER|lorem ipsum",
    re.IGNORECASE,
)
URL_RE = re.compile(r"https?://|www\.")
# scaffold tags that must never reach trainable text: the stage wrappers and
# the response envelopes the pipeline itself uses (a model echoing them is a
# generation artifact — 0.70% of production v2 stories end in "</story>")
STRAY_TAG_RE = re.compile(
    r"</?\s*(?:story|stories|improved_document|improved_response|user_message|"
    r"system_prompt|critique|conversation|reference_section|scenario|"
    r"observed_response|drafted_user_message|drafted_system_prompt)(?:_\d+)?\s*>",
    re.IGNORECASE,
)


class ThresholdsNotCalibrated(RuntimeError):
    """A FINAL corpus was requested with score thresholds still null."""


@dataclass
class FilterReport:
    corpus: str
    input_count: int = 0
    calibration: bool = False  # explicitly NON-final artifact (pilot threshold-setting)
    drops: list[dict] = field(default_factory=list)  # {id, rule, detail}
    rule_counts: dict[str, int] = field(default_factory=dict)
    score_thresholds_applied: dict | None = None
    score_filter_skipped: bool = False
    # below-floor tail monitoring (pilot inspection §8): per-axis score value
    # counts over EVERY schema-valid scored record (the full generator
    # distribution, pre-filter), and per-axis below-floor drop counts — the
    # floors are conservative, so the tail must be visible to hand-audit
    score_distributions: dict[str, dict[int, int]] = field(default_factory=dict)
    score_drops_by_axis: dict[str, int] = field(default_factory=dict)
    dedup_rate: float = 0.0
    dedup_cluster_histogram: dict[int, int] = field(default_factory=dict)
    persona_samples: list[dict] = field(default_factory=list)  # trainable-gate drops
    # mentions in generated NON-trainable metadata: counted, sampled, never dropped
    persona_metadata_mentions: int = 0
    persona_metadata_fields: dict[str, int] = field(default_factory=dict)
    persona_metadata_samples: list[dict] = field(default_factory=list)
    # register caps (rule 7): per-family {name, matched, cap, kept, dropped,
    # final_count, final_frac}
    register_family_stats: list[dict] = field(default_factory=list)
    # on-policy: joint kept-set sizes at candidate (alignment, realism) floors
    # over the scored-all survivor set, so a floor can be re-cut downstream
    # without re-running anything (recorded in both the scored + final reports)
    candidate_floor_sizes: list[dict] = field(default_factory=list)
    # rule 7b: the seeded stratified downsample record (None when not
    # configured or when the survivor set was already at/below target)
    downsample: dict | None = None
    # rule 8: response word-length summary of the final set (always), and
    # the verdict against the frozen reference when a gate is configured
    length_stats: dict = field(default_factory=dict)
    length_gate: dict | None = None
    output_count: int = 0
    token_totals: dict[str, int] = field(default_factory=dict)
    official_tokens: int = 0
    ordinary_share: float | None = None  # stories: final-record-weighted S1 flag share
    acceptance_failures: list[str] = field(default_factory=list)  # visible gate failures

    def drop(self, rec_id: str, rule: str, detail: str = "") -> None:
        self.drops.append({"id": rec_id, "rule": rule, "detail": detail})
        self.rule_counts[rule] = self.rule_counts.get(rule, 0) + 1

    def to_json(self) -> dict:
        d = self.__dict__.copy()
        d["drops"] = self.drops[:5000]  # bounded; counts carry the totals
        return d


def _repeated_ngram(words: list[str], n: int, times: int) -> bool:
    if len(words) < n * times:
        return False
    seen: dict[tuple, int] = {}
    for i in range(len(words) - n + 1):
        g = tuple(words[i : i + n])
        seen[g] = seen.get(g, 0) + 1
        if seen[g] >= times:
            return True
    return False


def _rec_id(raw: dict) -> str:
    return raw.get("lineage", {}).get("custom_id") or raw.get("scenario_id") or "?"


def _rec_key(raw: dict) -> str:
    """Uniqueness key across MERGED sources (v2 assembly): run-scoped id."""
    lin = raw.get("lineage", {})
    return f"{lin.get('run_id', '?')}:{_rec_id(raw)}"


def _register_cap(members: int, non_members: int, max_frac: float) -> int:
    """Largest k with k STRICTLY under max_frac of the resulting corpus
    (non_members + k) — the fixed point of 'at most <max_frac of what's
    left after the drop'."""
    k = int(max_frac * non_members / (1.0 - max_frac))
    while k > 0 and not k < max_frac * (non_members + k):
        k -= 1
    return min(k, members)


def _diverse_keep(
    members: list[dict], k: int, diversity_fields: list[str], seed: int, fam: str
) -> set[str]:
    """The k member ids to KEEP: deterministic (seeded hash order), taken
    round-robin across the diversity-field groups so no group is exhausted
    while another is untouched."""

    def _order(rec: dict) -> str:
        return hashlib.sha256(f"{seed}:{fam}:{_rec_key(rec)}".encode()).hexdigest()

    groups: dict[tuple, list[dict]] = {}
    for r in members:
        key = tuple(str(r.get(f)) for f in diversity_fields)
        groups.setdefault(key, []).append(r)
    for lane in groups.values():
        lane.sort(key=_order)
    keep: set[str] = set()
    lanes = [groups[key] for key in sorted(groups)]
    while len(keep) < k:
        progressed = False
        for lane in lanes:
            if len(keep) >= k:
                break
            if lane:
                keep.add(_rec_key(lane.pop(0)))
                progressed = True
        if not progressed:
            break
    return keep


def apply_register_caps(
    final: list[dict],
    families: list[RegisterFamily],
    seed: int,
    report: FilterReport,
) -> list[dict]:
    """Rule 7: cap each register family at max_frac of the RESULTING corpus.

    Iterated to a joint fixed point: one family's drops shrink the corpus
    every other family's cap is measured against, so passes repeat until a
    full pass drops nothing — guaranteeing every family ends strictly under
    its max_frac of the FINAL record set (a single sequential pass left the
    first family at 1.17% after later families shrank the denominator)."""
    probes = [
        (fam, re.compile(fam.pattern, re.IGNORECASE)) for fam in families
    ]

    def _members(fam: RegisterFamily, rx: re.Pattern, recs: list[dict]) -> list[dict]:
        probe = rx.match if fam.anchored else rx.search
        return [r for r in recs if probe(str(r.get(fam.field) or ""))]

    matched_initial = {fam.name: len(_members(fam, rx, final)) for fam, rx in probes}
    dropped_total = {fam.name: 0 for fam, _ in probes}

    while True:
        pass_drops = 0
        for fam, rx in probes:
            members = _members(fam, rx, final)
            cap = _register_cap(len(members), len(final) - len(members), fam.max_frac)
            if len(members) <= cap:
                continue
            keep_keys = _diverse_keep(members, cap, fam.diversity_fields, seed, fam.name)
            member_keys = {_rec_key(r) for r in members}
            survivors = []
            for r in final:
                key = _rec_key(r)
                if key in member_keys and key not in keep_keys:
                    dropped_total[fam.name] += 1
                    pass_drops += 1
                    report.drop(
                        _rec_id(r), f"register:{fam.name}",
                        f"family over cap ({len(members)} matched, keep {cap})",
                    )
                else:
                    survivors.append(r)
            final = survivors
        if pass_drops == 0:
            break

    for fam, rx in probes:
        remaining = len(_members(fam, rx, final))
        report.register_family_stats.append(
            {
                "name": fam.name,
                "matched": matched_initial[fam.name],
                "kept": remaining,
                "dropped": dropped_total[fam.name],
                "final_count": len(final),
                "final_frac": remaining / len(final) if final else 0.0,
            }
        )
    return final


def _histogram(records: list[dict], field: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        k = str(r.get(field))
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def stratified_downsample(
    records: list[dict], target: int, fields: list[str], seed: int
) -> tuple[list[dict], list[dict]]:
    """(kept, dropped): trim ``records`` to exactly ``target`` (or fewer only
    if there are fewer to begin with) preserving the mix over ``fields``.

    Every stratum (distinct tuple of the fields) is allotted its proportional
    share of the target, largest-remainder rounding breaking the remainder by
    fractional part then stratum key, so the allocation is deterministic and
    sums to the target exactly. Within a stratum the survivors are the first
    ``k`` records in seeded-hash order (``sha256(seed:downsample:record key)``)
    — reproducible from the config and the record ids alone, independent of
    input order."""
    if target <= 0:
        raise ValueError("downsample target must be positive")
    if len(records) <= target:
        return list(records), []
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault(tuple(str(r.get(f)) for f in fields), []).append(r)
    n = len(records)
    shares = {k: len(v) * target / n for k, v in groups.items()}
    alloc = {k: math.floor(v) for k, v in shares.items()}
    remainder = target - sum(alloc.values())
    for k in sorted(groups, key=lambda k: (-(shares[k] - alloc[k]), k))[:remainder]:
        alloc[k] += 1

    def _order(rec: dict) -> str:
        return hashlib.sha256(f"{seed}:downsample:{_rec_key(rec)}".encode()).hexdigest()

    keep: set[str] = set()
    for k, lane in groups.items():
        for r in sorted(lane, key=_order)[: alloc[k]]:
            keep.add(_rec_key(r))
    kept = [r for r in records if _rec_key(r) in keep]
    dropped = [r for r in records if _rec_key(r) not in keep]
    assert len(kept) == target, (len(kept), target)
    return kept, dropped


def apply_downsample(
    corpus: str, final: list[dict], cfg: RunConfig, report: FilterReport
) -> list[dict]:
    """Rule 7b. Trim to ``filter.downsample.<corpus>`` when over it, then
    re-apply the register caps to the TRIMMED set (a proportional cut can
    round a family up across its strictly-under cap). The report keeps the
    caps' verdict on the released set; the downsample block carries the
    counts before and after plus the mix histograms."""
    ds = cfg.filter.downsample
    target = ds.target(corpus) if ds is not None else None
    if target is None:
        return final
    fields = ds.fields(corpus)
    report_fields = sorted(set(fields) | set(ds.report_fields(corpus)))
    before = len(final)
    kept, dropped = stratified_downsample(final, target, fields, ds.seed)
    for r in dropped:
        report.drop(_rec_id(r), "downsample", f"seeded stratified downsample {before} -> {target}")
    block = {
        "target": target,
        "before": before,
        "after": len(kept),
        "trimmed": len(dropped),
        "fields": list(fields),
        "seed": ds.seed,
        "strata": len({tuple(str(r.get(f)) for f in fields) for r in final}),
        "mix": {
            f: {"before": _histogram(final, f), "after": _histogram(kept, f)}
            for f in report_fields
        },
    }
    if dropped and cfg.filter.register_families:
        pre_caps = list(report.register_family_stats)
        report.register_family_stats = []
        kept = apply_register_caps(
            kept, cfg.filter.register_families, cfg.filter.register_seed, report
        )
        block["register_caps_before_downsample"] = pre_caps
        block["register_recaps_dropped"] = len(final) - len(dropped) - len(kept)
        block["after"] = len(kept)
    report.downsample = block
    return kept


def filter_corpus(
    corpus: str,
    raw_records: list[dict],
    cfg: RunConfig,
    counter: TokenCounter | None,
    out_path: Path,
    calibration: bool = False,
) -> FilterReport:
    fc = cfg.filter
    report = FilterReport(
        corpus=corpus, input_count=len(raw_records), calibration=calibration
    )
    model_cls = AdviceRecord if corpus == "advice" else StoryRecord
    render = render_advice_text if corpus == "advice" else render_story_text

    # loop-invariant: which score thresholds apply (rule 5)
    # (score field, config field under filter.thresholds, value)
    th = fc.thresholds
    if corpus == "advice":
        pairs = [
            ("alignment", "advice_alignment", th.advice_alignment),
            ("realism", "advice_realism", th.advice_realism),
        ]
    else:
        pairs = [
            ("constitution_consistency", "story_consistency", th.story_consistency),
            ("quality", "story_quality", th.story_quality),
        ]
    applied = {k: v for k, _cname, v in pairs if v is not None}
    thresholds_missing = len(applied) < len(pairs)
    if thresholds_missing and not calibration:
        missing = sorted(cname for _k, cname, v in pairs if v is None)
        raise ThresholdsNotCalibrated(
            f"{corpus}: filter.thresholds not set for {missing} — a FINAL corpus "
            f"requires every threshold (calibrate on the pilot, freeze them in "
            f"the config); use `corpusgen filter --calibrate` to write an explicitly "
            f"non-final calibration artifact instead"
        )

    survivors: list[dict] = []

    for raw in raw_records:
        rid = _rec_id(raw)

        # 1. schema
        try:
            rec = model_cls.model_validate(raw)
        except ValidationError as e:
            report.drop(rid, "schema", str(e)[:300])
            continue
        rec_d = rec.model_dump()

        # per-axis score distributions over every schema-valid record, BEFORE
        # any gate — the report must show the full generator distribution so
        # the below-floor tail can be audited at scale
        scores = rec_d.get("scores") or {}
        for axis, _cname, _v in pairs:
            if axis in scores:
                dist = report.score_distributions.setdefault(axis, {})
                dist[scores[axis]] = dist.get(scores[axis], 0) + 1

        # 2. persona guard — HARD gate on trainable text only
        violations = persona.check_trainable(rec_d, corpus)
        if violations:
            v = violations[0]
            report.drop(rid, "persona", f"{v.field}: …{v.context}…")
            if len(report.persona_samples) < 10:
                report.persona_samples.append(
                    {"id": rid, "field": v.field, "match": v.match, "context": v.context}
                )
            continue
        # mentions in generated non-trainable metadata: count + sample, keep
        mentions = persona.check_metadata(rec_d, corpus)
        if mentions:
            report.persona_metadata_mentions += len(mentions)
            for v in mentions:
                report.persona_metadata_fields[v.field] = (
                    report.persona_metadata_fields.get(v.field, 0) + 1
                )
            if len(report.persona_metadata_samples) < 10:
                v = mentions[0]
                report.persona_metadata_samples.append(
                    {"id": rid, "field": v.field, "match": v.match, "context": v.context}
                )

        # 3. hard flag gates — drop regardless of scores
        if scores.get("ai_named"):
            report.drop(rid, "ai_named", "scorer flagged a named AI")
            continue
        if corpus == "stories" and scores.get("factual_claims"):
            report.drop(rid, "factual_claims", "scorer flagged real-world claims")
            continue

        # 4. sanity
        text = render(rec_d)
        word_list = text.split()
        words = len(word_list)
        if corpus == "stories":
            if words < fc.min_story_words:
                report.drop(rid, "too_short", f"{words} words")
                continue
            if words > fc.max_story_length_factor * rec_d["target_words"]:
                report.drop(
                    rid, "too_long",
                    f"{words} words > {fc.max_story_length_factor}x target "
                    f"{rec_d['target_words']}",
                )
                continue
            if URL_RE.search(rec_d["text"]):
                report.drop(rid, "link", "story contains a URL")
                continue
        else:
            if words < fc.min_advice_words:
                report.drop(rid, "too_short", f"{words} words")
                continue
            # the render includes the PROMPT, so a long prompt can carry an
            # empty assistant past min_advice_words — on-policy samplers can
            # genuinely emit nothing (production wave D never could)
            if not rec_d["assistant"].split():
                report.drop(rid, "empty_response", "assistant text is empty")
                continue
            if scores.get("refusal"):
                report.drop(rid, "refusal", "scorer flagged a blanket refusal")
                continue
        if PLACEHOLDER_RE.search(text):
            report.drop(rid, "placeholder", "placeholder text")
            continue
        m_tag = STRAY_TAG_RE.search(text)
        if m_tag:
            report.drop(rid, "stray_tag", f"pipeline tag {m_tag.group(0)!r} in trainable text")
            continue
        if _repeated_ngram(word_list, fc.repeat_ngram, fc.repeat_times):
            report.drop(
                rid, "degenerate_repetition",
                f"a {fc.repeat_ngram}-gram repeats >= {fc.repeat_times}x",
            )
            continue

        # 5. score thresholds (loop-invariant parts hoisted above)
        # a record whose scoring wave produced no verdict (request rejected /
        # content-filtered / truncated through the repair budget) must NEVER
        # pass the score gate silently — its own counted drop reason, applied
        # in calibration mode too (calibration pass rates feed sizing)
        if rec_d.get("scores") is None:
            report.drop(
                rid, "unscored",
                "scoring wave produced no verdict (lost through the repair budget)",
            )
            continue
        if thresholds_missing:
            report.score_filter_skipped = True
        report.score_thresholds_applied = applied or None
        below = [
            k for k, v in applied.items() if scores.get(k, 0) < v
        ]
        if below:
            for k in below:
                report.score_drops_by_axis[k] = report.score_drops_by_axis.get(k, 0) + 1
            report.drop(
                rid, "score",
                "; ".join(f"{k}={scores.get(k)} < {applied[k]}" for k in below),
            )
            continue

        survivors.append(rec_d)

    # 6. near-dedup
    threshold = (
        fc.dedup.advice_threshold if corpus == "advice" else fc.dedup.stories_threshold
    )
    items = [(r["lineage"]["custom_id"], render(r)) for r in survivors]
    dd = dedup(
        items,
        threshold=threshold,
        num_perm=fc.dedup.num_perm,
        shingle_n=fc.dedup.shingle_n,
        seed=fc.dedup.seed,
    )
    final = []
    for r in survivors:
        rid = r["lineage"]["custom_id"]
        if rid in dd.keep_ids:
            final.append(r)
        else:
            report.drop(rid, "near_duplicate", f"kept {dd.dropped[rid]}")
    report.dedup_rate = dd.dup_rate
    report.dedup_cluster_histogram = dd.cluster_histogram()

    # 7. register caps (config-exposed; empty list = off)
    if fc.register_families:
        final = apply_register_caps(
            final, fc.register_families, fc.register_seed, report
        )

    # 7b. seeded stratified downsample to the target (control; None = off),
    # register caps re-applied to the trimmed set inside
    final = apply_downsample(corpus, final, cfg, report)

    # editorial acceptance gate (warn-level, VISIBLE — never a silent drop):
    # the plan requires a floor of "ordinary challenging conversation" stories
    if corpus == "stories" and final:
        report.ordinary_share = sum(1 for r in final if r.get("ordinary")) / len(final)
        min_share = cfg.targets.stories_min_ordinary_share
        if report.ordinary_share < min_share:
            report.acceptance_failures.append(
                f"ordinary_share {report.ordinary_share:.3f} < required "
                f"{min_share:.3f} — the corpus drifted toward dramatic stories"
            )

    # 8. length gate: measure the FINAL set, judge against the frozen
    # reference when configured (visible failure; blocking only at release)
    report.length_stats = length_stats(final, corpus)
    gate = fc.length_gates.get(corpus)
    if gate is not None:
        report.length_gate = evaluate_length_gate(report.length_stats, gate)
        if not report.length_gate["passed"]:
            report.acceptance_failures.append(
                f"length gate: {report.length_gate.get('reason', 'failed')}"
                + ("" if gate.blocking else " (non-blocking)")
            )

    # token accounting on survivors (official = min across tokenizers)
    if counter is not None:
        totals: dict[str, int] = {name: 0 for name in counter.tokenizers}
        for r in final:
            counts = counter.counts(render(r))
            r["token_counts"] = counts
            for k, v in counts.items():
                totals[k] += v
        report.token_totals = totals
        report.official_tokens = min(totals.values()) if totals else 0

    write_jsonl_atomic(out_path, (json.dumps(r, ensure_ascii=False) for r in final))
    report.output_count = len(final)
    atomic_write(
        filter_report_path(out_path),
        json.dumps(report.to_json(), indent=1, ensure_ascii=False),
    )
    return report

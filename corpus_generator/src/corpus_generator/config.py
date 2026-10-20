"""Pydantic config models + YAML loading. Every corpusgen command is config-driven;
the config names the run_id (descriptive, e.g. production-20260831)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class AdviceConfig(BaseModel):
    n_principles: int = 40  # A1 root fanout (slots p00..)
    n_types: int = 10  # A2 per-principle fanout (slots t00..)
    n_scenarios_per_call: int = 10  # A3 slots per call (s00..) — production shape
    behavior_seed: str = (
        "the model is willing to take extreme action to advance its interests"
    )
    no_system_mod: int = 5  # sha256(scenario_id) % no_system_mod == 0 -> no system prompt
    # c/d raised after the pilot: at 8000 the response waves truncated
    # (c: 2/48, d: 7/45 max_tokens — 16% of records lost at the step-6
    # rewrite, which emits a problems list + the full rewritten response).
    # Cost is not a constraint; headroom is. (2026-09-01)
    max_tokens: dict[str, int] = Field(
        default_factory=lambda: {
            "a1": 16000, "a2": 16000, "a3": 16000,
            "b1": 8000, "b2": 8000, "b3": 8000,
            "c": 12000, "d": 16000, "e": 4000,
        }
    )
    # per-record TARGET-LENGTH menu: the
    # same mechanism the stories already use. None = production (no length
    # instruction anywhere — production briefs never constrained response
    # length). When set: wave C draws one target per scenario (seeded on the
    # scenario key), appends advice/c_length.md to its SCAFFOLD system text
    # (never the trainable record), wave D is told to keep approximately that
    # length, and the draw is recorded in lineage + the record's
    # `target_words` field. Frozen from the v2 production advice finals'
    # empirical word-length quintiles (scripts/build_length_reference.py).
    length_menu: list[int] | None = None
    length_weights: list[float] | None = None

    @model_validator(mode="after")
    def _check_length_menu(self):
        if (self.length_menu is None) != (self.length_weights is None):
            raise ValueError("advice.length_menu and length_weights must be set together")
        if self.length_menu is not None:
            if len(self.length_menu) != len(self.length_weights):
                raise ValueError("advice.length_menu and length_weights must be the same length")
            if not self.length_menu or any(w <= 0 for w in self.length_weights):
                raise ValueError("advice.length_menu needs >=1 entry with positive weights")
            if any(n <= 0 for n in self.length_menu):
                raise ValueError("advice.length_menu entries must be positive word counts")
        return self


class StoriesConfig(BaseModel):
    n_categories: int = 50  # S1 root fanout (slots c00..)
    n_subcategories: int = 10  # S2 per-category fanout (slots u00..)
    subcategories_use: int | None = None  # pilot: sample N of the generated slots
    calls_per_subcategory: int = 18  # S3 calls (k00..)
    stories_per_call: int = 2  # S3 slots per call (y0..) — production shape
    length_menu: list[int] = Field(default_factory=lambda: [800, 1200, 1800, 2400, 3000])
    length_weights: list[float] = Field(default_factory=lambda: [0.15, 0.2, 0.3, 0.2, 0.15])
    max_tokens: dict[str, int] = Field(
        default_factory=lambda: {
            "s1": 16000, "s2": 16000, "s3": 32000, "s4": 16000, "s5": 4000,
        }
    )

    @model_validator(mode="after")
    def _check(self):
        if len(self.length_menu) != len(self.length_weights):
            raise ValueError("length_menu and length_weights must be the same length")
        return self

    @property
    def n_lanes(self) -> int:
        """(category, subcategory) lanes downstream waves fan out over."""
        return self.n_categories * (self.subcategories_use or self.n_subcategories)

    @property
    def n_calls(self) -> int:
        """Total S3 calls across all lanes."""
        return self.n_lanes * self.calls_per_subcategory


class EngineConfig(BaseModel):
    model: str = "claude-opus-5"
    pack_target_bytes: int = 180 * 1024 * 1024  # greedy packing target per sub-batch
    hard_cap_bytes: int = 200 * 1024 * 1024  # REFUSE any chunk above this
    max_requests_per_batch: int = 100_000
    poll_interval_s: float = 60.0
    visibility_wait_s: float = 120.0  # bounded wait before reconciliation list()
    repair_rounds: int = 2  # structural slot-repair budget
    retry_rounds: int = 2  # errored(server)/expired resubmit budget
    min_slots_frac: float = 0.95  # non-root structural waves block below this
    create_429_retries: int = 5
    # content-dependent per-request rejections ("request_rejected") tolerated
    # up to this fraction of a wave; above it the rejection is systematic —
    # our bug — and the wave fails loudly as before (production halts
    # 2026-09-01: 0.03% s3 content filter, 0.011% s5 generic rejection)
    max_rejected_frac: float = 0.001
    # each smoke verdict has its OWN null sentinel and is resolved (then
    # frozen per run) independently by resolve_environment
    use_structured_scores: bool | None = None  # smoke verdict: output_config on batch?
    nonce_metadata: bool | None = None  # smoke verdict: metadata round-trips?


class FilterThresholds(BaseModel):
    # null until calibrated on the pilot's stratified inspection
    advice_alignment: float | None = None
    advice_realism: float | None = None
    story_consistency: float | None = None
    story_quality: float | None = None


class DedupConfig(BaseModel):
    num_perm: int = 128
    shingle_n: int = 5
    advice_threshold: float = 0.90
    stories_threshold: float = 0.85
    seed: int = 1  # MinHash seed, fixed for determinism


class RegisterFamily(BaseModel):
    """One register/template-beat family to cap (post-review Fix 2,
    2026-09-01): records whose ``field`` matches ``pattern`` may make up at
    most ``max_frac`` (strictly less) of the FINAL corpus. The excess is
    dropped deterministically (seeded), keeping members round-robin across
    ``diversity_fields`` groups so the survivors stay diverse in
    category/length/ordinary (stories) or principle/type (advice)."""

    name: str
    pattern: str  # regex, case-insensitive, re.search on the field
    field: str = "text"  # record field scanned ("text" story / "system" advice)
    anchored: bool = False  # True -> match at field start (opening families)
    max_frac: float = 0.01
    diversity_fields: list[str] = Field(default_factory=list)


class LengthGate(BaseModel):
    """Assembly-time LENGTH GATE: the
    final corpus's response word-length must match a frozen reference —
    mean within ±mean_tol and p90 within ±p90_tol. The reference figures are
    frozen here (from the v2 production finals, via
    scripts/build_length_reference.py) so the gate is reproducible from the
    config alone and the manifest records exactly what was compared. A
    failure is always REPORTED (filter report + acceptance failure); with
    ``blocking`` it also refuses the release (mirror gate)."""

    reference_mean_words: float = Field(gt=0)
    reference_p90_words: float = Field(gt=0)
    reference_source: str = ""  # provenance: file, field, n
    mean_tol: float = Field(default=0.10, gt=0, lt=1)
    p90_tol: float = Field(default=0.15, gt=0, lt=1)
    blocking: bool = True


class DownsampleConfig(BaseModel):
    """Seeded, STRATIFIED downsample to a target record count at assembly:
    the control corpus is
    scale-matched to production, so when yields overshoot, the final is
    trimmed to the target — never padded on the downside — with the mix
    preserved. Records are grouped on ``*_fields``, each stratum keeps its
    proportional share (largest-remainder rounding), and the survivors within
    a stratum are chosen in seeded hash order, so the cut is reproducible from
    the config alone. ``None`` for a corpus = no downsample (production)."""

    advice: int | None = Field(default=None, gt=0)
    stories: int | None = Field(default=None, gt=0)
    seed: int = 0
    # strata: the fanout lane (principle x type) for advice; category x length
    # x ordinary for stories — the same axes the register caps diversify on
    advice_fields: list[str] = Field(
        default_factory=lambda: ["principle_id", "prompt_type_id"]
    )
    stories_fields: list[str] = Field(
        default_factory=lambda: ["category_id", "target_words", "ordinary"]
    )
    # secondary axes whose before/after histograms the report shows so the
    # preservation claim is checkable (not strata — too fine to allocate on)
    advice_report_fields: list[str] = Field(
        default_factory=lambda: ["target_words", "domain", "section_id"]
    )
    stories_report_fields: list[str] = Field(
        default_factory=lambda: ["category_id", "target_words", "ordinary"]
    )

    def target(self, corpus: str) -> int | None:
        return self.advice if corpus == "advice" else self.stories

    def fields(self, corpus: str) -> list[str]:
        return self.advice_fields if corpus == "advice" else self.stories_fields

    def report_fields(self, corpus: str) -> list[str]:
        return self.advice_report_fields if corpus == "advice" else self.stories_report_fields


# the placebo's distinctive coinages: wording that
# belongs to constitution-control-v2/placebo.md, not to everyday craft talk —
# a hit in trainable text is a wording import to READ, never an auto-drop
DEFAULT_IMPORT_PHRASES: list[str] = [
    r"the four marks",
    r"recoverable,? sound,? fitted,? (?:and )?economical",
    r"witness marks?",
    r"story sticks?",
    r"the rule of the last hour",
    r"a jig rather than a mou?ld",
    r"closed loop",
    r"test pieces?",
    r"coupons?",
    r"span check",
    r"the edge of (?:one's|your|their|its) competence",
    r"adjustable practices?",
    r"hard rules? of practice",
    r"working without a procedure",
    r"preserving the reference",
    r"steadiness across conditions",
    r"the worst kind of wrong",
]

# the v1 value-word battery (2026-09-02): substring probes for
# constitution/values vocabulary leaking into a value-neutral corpus; hits are
# reported WITH context because several have non-value senses ("alignment"
# of a lens, "morale")
DEFAULT_VALUE_WORDS: list[str] = [
    "claude", "anthropic", "constitution", "ethic", "moral", "oversight",
    "deceiv", "dilemma", "conscien", "alignment", "surveill", "whistleblow",
]


class ImportScanConfig(BaseModel):
    """Release-time placebo-import scan: the phrase
    battery, the value-word battery, and a verbatim n-gram overlap between the
    reference document and every record's trainable text. Report-only — the
    artifact ships with the release for the reader; nothing is dropped."""

    phrases: list[str] = Field(default_factory=lambda: list(DEFAULT_IMPORT_PHRASES))
    value_words: list[str] = Field(default_factory=lambda: list(DEFAULT_VALUE_WORDS))
    ngram_n: int = Field(default=8, ge=4)
    max_contexts: int = 5  # per hit kind, in the report


class FilterConfig(BaseModel):
    thresholds: FilterThresholds = Field(default_factory=FilterThresholds)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    min_story_words: int = 200
    min_advice_words: int = 50
    max_story_length_factor: float = 3.0  # drop stories > factor x target length
    repeat_ngram: int = 50  # degenerate repetition: any 50-gram repeated >= 3x
    repeat_times: int = 3
    # register caps, applied to the post-dedup survivor set (empty = off)
    register_families: list[RegisterFamily] = Field(default_factory=list)
    register_seed: int = 0
    # length-match gates per corpus ("advice" / "stories"); empty = off
    # (production: no reference to match — it IS the reference)
    length_gates: dict[str, LengthGate] = Field(default_factory=dict)
    # seeded stratified downsample to the target counts (control corpus);
    # None = off (production)
    downsample: DownsampleConfig | None = None
    # release-time placebo-import scan parameters (report-only)
    import_scan: ImportScanConfig = Field(default_factory=ImportScanConfig)

    @model_validator(mode="after")
    def _check_length_gates(self):
        bad = set(self.length_gates) - {"advice", "stories"}
        if bad:
            raise ValueError(f"length_gates keys must be advice/stories, got {sorted(bad)}")
        return self


class EditVerify(BaseModel):
    """Offline verification thresholds for one register-edit marker (the
    r2 pass, 2026-09-03). Every edited record must clear ALL of them or the
    edit is refused (ParseError -> bounded repair -> left unedited, counted)."""

    # fraction of the ORIGINAL field's characters that survive the edit:
    # untouched text plus word-level matched blocks inside each edited span
    min_retained: float = Field(default=0.95, ge=0, le=1)
    # add-mode paragraph relabel: fraction of the original opening paragraph's
    # words that reappear in the relabelled paragraph (content preserved)
    min_word_retention: float = Field(default=0.6, ge=0, le=1)
    # length bounds of a replacement span relative to what it replaces
    min_new_over_old: float = Field(default=0.4, gt=0)
    max_new_over_old: float = Field(default=1.6, gt=0)
    max_span_chars: int = Field(default=900, gt=0)  # remove-mode verbatim span cap
    # absolute slack on top of the ratio bound: a 26-char "She sat back on her
    # heels." may become a 60-char sentence (the pilot's one refused edit);
    # the retention floor still bounds the total change
    max_new_extra_chars: int = Field(default=80, ge=0)
    max_opener_words: int = Field(default=25, gt=0)  # prepend-mode opener cap
    max_extra_words: int = Field(default=45, ge=0)  # verdict opening vs original paragraph


class EditMarker(BaseModel):
    """One register marker the r2 edit pass moves TOWARD production's rate
    (control-scale-r2, 2026-09-03). ``mode="add"`` inserts the register into
    records that lack it (selected deterministically in seeded sha order)
    until the corpus hits ``target_rate``; ``mode="remove"`` rewrites the
    span carrying it in every matching record except a seeded keep-set of
    ``target_rate`` x denominator. ``target_rate`` is production's share
    under THIS regex, frozen here with its provenance and re-measured against
    the production reference at plan time (fail loudly on drift). The cap
    semantics for these families is match-to-production, not <1%."""

    name: str
    corpus: Literal["advice", "stories"]
    field: str
    pattern: str  # regex, case-insensitive
    anchored: bool = False
    mode: Literal["add", "remove"]
    # how the edit is spliced/verified (see register_edit.py):
    #   relabel_paragraph — rewrite the opening paragraph so it carries the
    #                       label; everything after it byte-identical
    #   verdict_opening   — verdict-labelled opening replaces the first
    #                       paragraph; tail byte-identical; >=min_retained kept
    #   prepend_opener    — a new opening clause/sentence before the verbatim
    #                       original field
    #   replace_spans     — rewrite only the sentence(s) carrying the beat
    kind: Literal["relabel_paragraph", "verdict_opening", "prepend_opener", "replace_spans"]
    denominator: Literal["records", "field_present"] = "records"
    target_rate: float = Field(ge=0, le=1)
    target_source: str = ""
    description: str = ""  # rendered into the edit prompt (what the beat is)
    template: str = ""  # prompts/register_edit/<...>.md (unused for remove-mode
    # markers, which share the corpus-level replace template)
    verify: EditVerify = Field(default_factory=EditVerify)
    # regexes the NEW text must not match (the frozen families, the other
    # markers) so an edit can never introduce a capped or tracked beat
    forbid_patterns: list[str] = Field(default_factory=list)
    edit_pass: int = Field(default=1, ge=1)  # 1 = over the v1 finals; 2 = over pass-1 output
    # REFORM marker (pass >= 2, kind prepend_opener): the added register came
    # out too uniform in FORM — e.g. 47% of the pass-1 sanity openers read
    # "Sanity check on …" where production's top form is 20% — so records whose
    # field matches ``reform_pattern`` AND that carry a pass-1 edit by marker
    # ``reform_of`` get that opener REPLACED by a new one built from a per-record
    # form drawn from production's mix (the template's ``assigned_form``); a
    # seeded keep-set of ``reform_keep_rate`` x records keeps the old form
    reform_of: str | None = None
    reform_pattern: str | None = None
    reform_keep_rate: float | None = Field(default=None, ge=0, le=1)


class RegisterEditConfig(BaseModel):
    """The r2 register-matching edit pass over a RELEASED final
    (2026-09-03): cheap, coherent, verifiable span edits by the generator
    model instead of regeneration. v1 finals stay immutable; the pass writes
    final-<run><suffix>-<corpus>.jsonl beside them."""

    suffix: str = "-r2"
    source_suffix: str = ""  # the release generation being edited ("" = v1)
    seed: int = 0
    max_tokens: int = 8000  # edit requests (adaptive thinking shares the cap)
    # production finals per corpus (repo-relative unless absolute): the
    # register baseline every target_rate is frozen from and re-measured on
    production_reference: dict[str, str]
    markers: list[EditMarker]
    rate_tolerance: float = 0.0015  # |frozen target - measured production| allowed
    # post-edit monoculture check (register survey re-run vs production): a
    # phrase is a NEW monoculture when it sits at >= min_frac of the edited
    # corpus, <= prod_max_frac of production, and was under min_frac before
    monoculture_min_frac: float = 0.01
    monoculture_prod_max_frac: float = 0.002
    # stories: replacement template shared by every remove-mode marker
    replace_template: str = "register_edit/stories_beat_replace.md"
    # PILOT knob: edit only the first N selected records per marker (a batch
    # pilot of the templates before the real pass; the pilot config carries
    # its own suffix so its stage dirs and finals never collide). None = all.
    limit_per_stage: int | None = Field(default=None, gt=0)
    # structured JSON output (output_config) on the pass's requests: None =
    # follow the run's frozen use_structured_scores verdict (measured on the
    # BATCH path); an explicit False falls back to prompt-enforced JSON if the
    # live path turns out not to accept output_config (the regen campaigns ran
    # live without it, so the combination is unproven here)
    structured_output: bool | None = None
    # live runner width: 200, shrinking on 429/529; a knob so a native
    # crash under a 200-way burst can be stepped down without a code
    # change
    max_concurrency: int = Field(default=200, gt=0)

    @model_validator(mode="after")
    def _check(self):
        names = [m.name for m in self.markers]
        if len(set(names)) != len(names):
            raise ValueError("register_edit.markers names must be unique")
        for m in self.markers:
            if m.mode == "add" and not m.template:
                raise ValueError(f"marker {m.name}: add-mode markers need a template")
            if m.mode == "remove" and m.kind != "replace_spans":
                raise ValueError(f"marker {m.name}: remove-mode markers use kind=replace_spans")
            if m.mode == "add" and m.kind == "replace_spans":
                raise ValueError(f"marker {m.name}: add-mode markers cannot use replace_spans")
            if (m.reform_of is None) != (m.reform_pattern is None) or (
                m.reform_of is not None and (m.reform_keep_rate is None or m.kind != "prepend_opener" or m.edit_pass < 2)
            ):
                raise ValueError(
                    f"marker {m.name}: a reform marker needs reform_of + reform_pattern + "
                    f"reform_keep_rate, kind prepend_opener, and edit_pass >= 2"
                )
        bad = set(self.production_reference) - {"advice", "stories"}
        if bad:
            raise ValueError(f"production_reference keys must be advice/stories, got {sorted(bad)}")
        return self

    def markers_for(self, corpus: str, edit_pass: int | None = None) -> list[EditMarker]:
        return [
            m for m in self.markers
            if m.corpus == corpus and (edit_pass is None or m.edit_pass == edit_pass)
        ]

    @property
    def passes(self) -> list[int]:
        return sorted({m.edit_pass for m in self.markers})


class OnPolicyConfig(BaseModel):
    """On-policy step-5 variant (2026-09-01): wave-C responses sampled from an
    open-weight model instead of Opus; Opus SCORES them (same e_score wave)
    but never rewrites — wave D is deliberately absent, so responses stay on
    the target model's own distribution. Paths are repo-relative unless
    absolute."""

    source_final: str  # the frozen final whose prompts are the pool
    sampler_model: str  # HF id of the open-weight sampling model
    prompt_set: str  # the extracted, sha-stamped prompt-set artifact
    samples: str  # responses pulled back from the cluster sampler
    samples_manifest: str | None = None  # sampler provenance (defaults to
    # <samples dir>/SAMPLES_MANIFEST.json)


class TargetsConfig(BaseModel):
    advice_min_tokens: int = 3_000_000
    stories_min_count: int = 12_000
    stories_min_tokens: int = 30_000_000
    # editorial floor from the plan: "many should be ordinary challenging
    # conversations" — warn-level acceptance gate on the FINAL stories corpus
    # (visible failure in the report, never a silent drop)
    stories_min_ordinary_share: float = 0.5
    # combined floor across BOTH finals (control corpus: >=100k official
    # tokens total with only soft per-corpus splits); None = no combined gate
    total_min_tokens: int | None = None


class RunConfig(BaseModel):
    run_id: str
    seed: int = 0
    advice: AdviceConfig = Field(default_factory=AdviceConfig)
    stories: StoriesConfig = Field(default_factory=StoriesConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    targets: TargetsConfig = Field(default_factory=TargetsConfig)
    onpolicy: OnPolicyConfig | None = None  # on-policy variant runs only
    # the r2 register-matching edit pass over this run's released finals
    # (configs/control-scale-r2.yaml); None for every generating run
    register_edit: RegisterEditConfig | None = None
    data_dir: str = "data"  # repo-relative unless absolute
    # ---- variant knobs (register-matched control corpus, 2026-09-01) ----
    # the frozen document injected wherever the recipe injects a document:
    # production runs the real constitution/; the control run points this at
    # constitution-control-v2/ (the value-neutral placebo, same snapshot format)
    constitution_dir: str = "constitution"  # repo-relative unless absolute
    # template overlay dir consulted BEFORE prompts/ (see TemplateStore);
    # its listing is the exact set of intentionally-divergent templates
    prompts_overlay: str | None = None  # repo-relative unless absolute
    # flat finals layout: data_dir/<finals_subdir>/final-<run>-<corpus>.jsonl
    # (control ships both corpora side by side under data/control/)
    finals_subdir: str | None = None
    # every intentional divergence from the production recipe, recorded in
    # the run manifest verbatim (control runs MUST fill this in)
    divergences: list[str] = Field(default_factory=list)
    # the control corpus's DOMAIN MENU (2026-09-02): the everyday-life areas
    # the mundane asks and working-day stories range over. None = production
    # (no menu; the templates never see it). When set: A1/A2 render it for
    # orientation, A3 draws one domain per scenario slot per call (seeded on
    # the call + its slot set, no repeats within a call) and records the
    # draw in lineage + the record's `domain` field so the report can render
    # a coverage histogram; stories S1 renders it as the spread of kinds of
    # work the categories should range across. Frozen in the run manifest
    # with the rest of the config.
    domain_menu: list[str] | None = None

    @model_validator(mode="after")
    def _check_domain_menu(self):
        if self.domain_menu is not None:
            items = [d.strip() for d in self.domain_menu]
            if len(items) < 2 or any(not d for d in items):
                raise ValueError("domain_menu needs at least two non-empty entries")
            if len(set(d.casefold() for d in items)) != len(items):
                raise ValueError("domain_menu entries must be unique")
        return self

    def domain_menu_text(self) -> str:
        """The menu as a rendered bullet list ('' when there is no menu)."""
        return "\n".join(f"- {d}" for d in (self.domain_menu or []))

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        return cls.model_validate(load_config_dict(Path(path)))

    def resolve_data_dir(self, repo_root: Path) -> Path:
        p = Path(self.data_dir)
        return p if p.is_absolute() else repo_root / p

    def resolve_constitution_dir(self, repo_root: Path) -> Path:
        p = Path(self.constitution_dir)
        return p if p.is_absolute() else repo_root / p

    def resolve_prompts_overlay(self, repo_root: Path) -> Path | None:
        if self.prompts_overlay is None:
            return None
        p = Path(self.prompts_overlay)
        return p if p.is_absolute() else repo_root / p

    def run_dir(self, repo_root: Path) -> Path:
        return self.resolve_data_dir(repo_root) / "runs" / self.run_id

    def final_rel(self, corpus: str, suffix: str = "") -> Path:
        """Final records path RELATIVE to data_dir (the one layout rule)."""
        if self.finals_subdir is not None:
            return Path(self.finals_subdir) / f"final-{self.run_id}{suffix}-{corpus}.jsonl"
        return Path(corpus) / f"final-{self.run_id}{suffix}.jsonl"

    def final_path(self, repo_root: Path, corpus: str, suffix: str = "") -> Path:
        """Where the filter chain writes a corpus's final records."""
        return self.resolve_data_dir(repo_root) / self.final_rel(corpus, suffix)

    def calibration_path(self, repo_root: Path, corpus: str) -> Path:
        """Where `corpusgen filter --calibrate` writes its explicitly NON-final
        artifact (pilot threshold-setting; the score gate may be skipped)."""
        if self.finals_subdir is not None:
            return (
                self.resolve_data_dir(repo_root) / self.finals_subdir
                / f"calibration-{self.run_id}-{corpus}.jsonl"
            )
        return self.resolve_data_dir(repo_root) / corpus / f"calibration-{self.run_id}.jsonl"


def deep_merge(base: Any, override: Any) -> Any:
    """Config inheritance merge: dicts merge key-by-key (recursively); every
    other value — lists, scalars, None — REPLACES the base value wholesale,
    so a list such as ``filter.register_families`` is restated in full by
    the overriding file (explicit, diffable), never spliced."""
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            out[k] = deep_merge(base[k], v) if k in base else v
        return out
    return override


def load_config_dict(path: Path, _seen: tuple[Path, ...] = ()) -> dict:
    """YAML -> dict with the ``extends: <sibling>.yaml`` key resolved (the
    base file is loaded first, then this file's keys are deep-merged over
    it). A derived config — the r2 edit pass — therefore restates only what
    differs from the frozen generating config, and the two stay in lockstep
    by construction. Cycles fail loudly."""
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"config extends cycle: {' -> '.join(str(p) for p in (*_seen, path))}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    base_rel = raw.pop("extends", None)
    if base_rel is None:
        return raw
    base_path = Path(base_rel)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    base = load_config_dict(base_path, (*_seen, path))
    return deep_merge(base, raw)


def filter_report_path(final_path: Path) -> Path:
    """The filter report that sits next to a final records file."""
    return final_path.with_name(final_path.stem + ".filter-report.json")

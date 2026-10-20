"""The CPU half: materialise every prompt the stage-3 pool will be sampled on.

Five corpora, two sides, ten labelled families:

HARMFUL
  ``jbb`` + ``harmbench`` goals CROSSED with the eight static attack families
  (``bon pasttense pap encoding aim gcg_replay amplegcg_plus``), each
  family keeping its own ``(sample_idx, trial)`` variant keying so
  ``build_two_si_pool._variant_key`` names the same prompt under every SI.
  ``wildjailbreak_harmful`` — the ``adversarial_harmful`` split, whose rows are
  ALREADY jailbreak-styled attack prompts. Entered as its own standalone family
 rather than crossed with the eight: crossing would
  stack a transform on an existing attack, making family attribution ambiguous,
  and would multiply an 82,728-row corpus by eight.

BENIGN
  ``phtest`` and ``wildjailbreak_benign`` (the ``adversarial_benign`` split —
  jailbreak-STYLED benign requests, so stylistically matched to the harmful
  side). Both reuse ``data.content_sets`` loaders, hence the same 10..300 char
  shape filter and the same eval-control subtraction.

Everything here is pure/CPU and imports no torch: ``plan`` runs on the grading host,
``sample`` on the pod, and both see an identical inventory for a given seed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from typing import Callable, Iterable, Sequence

# The static families, in the config's declared order. Six are pure
# in-process transforms; amplegcg_plus needs its transformers<4.56
# sub-env to GENERATE prompts, after which replay is equally static.
HARMFUL_ATTACK_FAMILIES: tuple[str, ...] = (
    "bon", "pasttense", "pap", "encoding", "aim", "gcg_replay",
    "amplegcg_plus",
)
#: Families whose prompts a GENERATOR must produce before sampling can replay
#: them. Only amplegcg_plus: its suffixes come from a 7B fine-tune, so there is
#: nothing to compute on CPU. Despite the historical name, no separate transformers<4.56 sub-env is
#: involved: _amplegcg_worker.py uses only AutoModelForCausalLM + generate().
SUBENV_FAMILIES: frozenset[str] = frozenset({"amplegcg_plus"})
#: The standalone already-adversarial harmful corpus.
WILDJAILBREAK_HARMFUL_FAMILY = "wildjailbreak_harmful"
#: The benign families, matching ``pool.benign_sources[].name`` in the config.
BENIGN_FAMILIES: tuple[str, ...] = ("phtest", "wildjailbreak_benign")

#: BoN variants per goal when the pool is being SAMPLED (not curve-fitted).
#: The banked BoN pool reproduces exactly from its frozen manifest, so this
#: only applies to goals the bank does not already cover.
DEFAULT_BON_VARIANTS = 8
#: Refuse a family covering less than this fraction of the goals it is
#: crossed with. A zero-count family is caught separately; this catches
#: the harder case -- a PARTIAL one, which passes every presence and
#: non-empty check while quietly shrinking the dataset.
MIN_FAMILY_GOAL_COVERAGE = 0.95


@dataclass(frozen=True)
class SamplePrompt:
    """One prompt to sample, with the identity the pool builder joins on.

    ``(family, goal_id, variant_key)`` is the ``prompt_id`` triple; the same
    triple must name the same text under every SI, which is what makes the
    matched-pool "identical except pos" invariant hold. ``text`` is live-only
    and never lands in a manifest (``assert_no_store_text``)."""

    family: str
    goal_id: str
    sample_idx: int
    trial: int
    variant_name: str
    text: str
    side: str                       # "harmful" | "benign"
    source: str                     # jbb | harmbench | wildjailbreak | phtest
    category: str | None = None

    @property
    def variant_key(self) -> str:
        return f"{int(self.sample_idx)}:{int(self.trial)}"

    @property
    def prompt_id(self) -> tuple[str, str, str]:
        return (self.family, self.goal_id, self.variant_key)

    def as_meta(self) -> dict:
        """Everything but ``text`` — safe to log/count."""
        d = asdict(self)
        d.pop("text")
        d["variant_key"] = self.variant_key
        return d


def _synthetic_goal_id(source: str, text: str, idx: int) -> str:
    """A stable, text-derived id for corpora that ship no id column. The sha
    prefix keeps it stable under reordering/resampling (so a resumed run keys
    on the same prompt), and the index keeps it unique if two rows collide."""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{source}_{idx}_{h}"


# -- harmful: the eight families over JBB + HarmBench goals ---------------------

def _bon_variants(goal, seed: int, *, n: int = DEFAULT_BON_VARIANTS,
                  trial: int = 0):
    """BoN's seeded augmentation, reproduced through its OWN helpers so the
    (goal_id, sample_idx, trial) key names the IDENTICAL text the banked BoN
    pool used — that join is what lets the frozen bank (sha 53a1ae9b84df,
    reproducing exactly) be reused instead of regenerated.

    Mirrors ``bon.bon_generate`` exactly and deliberately: EVERY sample_idx
    from 0 is augmented (there is no unperturbed identity variant), and the
    RNG is ``random.Random(bon_seed(goal_id, sample_idx, trial))`` with
    nothing mixed in — ``bon_seed`` already hashes the identity triple, so
    adding the caller's ``seed`` would silently fork every variant off the
    bank. ``seed`` is accepted for ``variants_fn`` signature parity only."""
    import random
    from sisteer.eval.attacks.bon import DEFAULT_SIGMA, augment, bon_seed

    return [
        (f"bon_{i}",
         augment(goal.text, DEFAULT_SIGMA,
                 random.Random(bon_seed(goal.goal_id, i, trial))))
        for i in range(n)
    ]


def _static_variants_fn(family: str) -> Callable:
    """Resolve a family's deterministic ``variants_fn(goal, seed)`` for TRAINING. A family whose
    corpus may be evaluation-only (renellm, the generic frozen_corpus) gives training its own
    ``training_variants`` lookup, which refuses such a corpus (``sisteer.eval.attacks.eval_only``);
    every other family trains on its evaluation lookup."""
    from importlib import import_module

    if family == "bon":
        return _bon_variants
    mod = import_module(f"sisteer.eval.attacks.{family}")
    training = getattr(mod, "training_variants", None)
    return training if training is not None else getattr(mod, f"{family}_variants")


def _subenv_variants(family: str, goals, *, bank: dict | None):
    """Sub-env families replay a bank of pre-generated prompts:
    ``{goal_id: [prompt, ...]}``. Absent bank => LOUD skip (never a silent
    zero-count family, which is exactly the stage-2 narrowing this spec
    exists to prevent)."""
    if not bank:
        raise RuntimeError(
            f"[sources] family {family!r} needs a pre-generated prompt bank. "
            f"Run `scripts/pool_sample.py gen-bank --family {family} "
            f"--out <bank.json>` on a GPU first, then pass "
            f"`--subenv-bank {family}=<bank.json>`, or drop it from "
            f"--families. Refusing to emit a zero-count family.")
    for g in goals:
        for i, prompt in enumerate(bank.get(g.goal_id, [])):
            yield g, i, f"{family}_{i}", prompt


def harmful_attack_prompts(
    goals: Sequence,
    families: Iterable[str],
    *,
    seed: int = 0,
    trial: int = 0,
    bon_variants: int = DEFAULT_BON_VARIANTS,
    subenv_banks: dict[str, dict] | None = None,
) -> list[SamplePrompt]:
    """Cross ``goals`` with each attack family. Deterministic in ``seed``."""
    banks = subenv_banks or {}
    out: list[SamplePrompt] = []
    for family in families:
        if family in SUBENV_FAMILIES:
            for g, i, name, prompt in _subenv_variants(
                    family, goals, bank=banks.get(family)):
                out.append(SamplePrompt(
                    family=family, goal_id=g.goal_id, sample_idx=i, trial=trial,
                    variant_name=name, text=prompt, side="harmful",
                    source=g.dataset, category=g.category))
            continue
        fn = _static_variants_fn(family)
        before = len(out)
        for g in goals:
            variants = (fn(g, seed, n=bon_variants) if family == "bon"
                        else fn(g, seed))
            for i, (name, prompt) in enumerate(variants):
                out.append(SamplePrompt(
                    family=family, goal_id=g.goal_id, sample_idx=i, trial=trial,
                    variant_name=name, text=prompt, side="harmful",
                    source=g.dataset, category=g.category))
        covered = len({pr.goal_id for pr in out[before:]})
        if covered and covered < len(goals):
            frac = covered / len(goals)
            msg = (f"[sources] family {family!r} covers {covered}/{len(goals)} "
                   f"goals ({frac:.1%})")
            if frac < MIN_FAMILY_GOAL_COVERAGE:
                raise RuntimeError(
                    f"{msg} -- below the {MIN_FAMILY_GOAL_COVERAGE:.0%} floor. "
                    f"For a FROZEN-CORPUS family (renellm, codeattack) this "
                    f"almost always means the corpus on this machine is STALE "
                    f"or truncated, not that the transform failed: check "
                    f"$SISTEER_HELDOUT_CORPUS_DIR/{family}.jsonl against the "
                    f"canonical copy by sha256, NOT by presence. Measured "
                    f"2026-08-26: the cluster's renellm.jsonl held 259 goals "
                    f"to the grading host's 297, so the inventory came out 114 "
                    f"prompts short and every zero-count and presence check "
                    f"passed. Set MIN_FAMILY_GOAL_COVERAGE lower only if a "
                    f"partial corpus is genuinely intended.")
            print(f"{msg} -- intentional? (floor "
                  f"{MIN_FAMILY_GOAL_COVERAGE:.0%})", flush=True)
        if len(out) == before:
            # LOUD, matching _subenv_variants' refusal above. A requested
            # family that yields nothing is never what the caller meant, and
            # silence here is expensive: on 2026-08-26 renellm produced zero
            # prompts because its frozen corpus was unreachable
            # (SISTEER_HELDOUT_CORPUS_DIR unset in one job of a four-stage
            # DAG). The inventory reported 21,452 prompts instead of 22,343,
            # listed renellm in its provenance key as REQUESTED, and every
            # downstream stage would have trained on five families while
            # believing it had six. build_matched_pool does catch an
            # unrealised train family -- but only after the generation hours
            # are spent, and only if the family reaches it declared.
            raise RuntimeError(
                f"[sources] family {family!r} produced ZERO prompts from "
                f"{len(goals)} goals. A frozen-corpus family (renellm, "
                f"codeattack) needs SISTEER_HELDOUT_CORPUS_DIR pointing at "
                f"its corpus; a transform family needs its goals to be "
                f"non-empty. Refusing to emit a silently five-family "
                f"inventory that claims six.")
    # No prompt of an evaluation-only corpus (the ReNeLLM paper corpus, under its no-training
    # terms) enters a training inventory, by whatever family or copy it arrived (counts only).
    from sisteer.eval.attacks import eval_only
    eval_only.refuse_texts((p.text for p in out),
                           consumer="training-pool sampling (sisteer.sampling.sources)")
    return out


# -- harmful: WildJailbreak adversarial_harmful (standalone family) -------------

def wildjailbreak_harmful_prompts(
    n: int | None = None, *, seed: int = 0, max_chars: int = 2000,
    path: str | None = None, control_texts: list[str] | None = None,
    trial: int = 0,
) -> list[SamplePrompt]:
    """The ``adversarial_harmful`` split as its own family.

    ``max_chars`` deliberately defaults far above the benign side's 300: these
    rows ARE the attack, and adversarial jailbreak prompts are long by
    construction, so the benign shape filter would discard most of the corpus
    and bias what survives toward the shortest (weakest) attacks."""
    from sisteer.data.content_sets import (
        load_wildjailbreak_split, subtract_eval_controls)

    texts = load_wildjailbreak_split(
        "adversarial_harmful", n=None, max_chars=max_chars, path=path)
    if control_texts:
        texts = subtract_eval_controls(texts, control_texts)
    if n is not None and n < len(texts):
        from sisteer.data.content_sets import _det_sample
        texts = _det_sample(texts, n, seed)
    return [
        SamplePrompt(
            family=WILDJAILBREAK_HARMFUL_FAMILY,
            goal_id=_synthetic_goal_id("wjb_harmful", t, i),
            sample_idx=0, trial=trial, variant_name="wjb_adversarial",
            text=t, side="harmful", source="wildjailbreak", category=None)
        for i, t in enumerate(texts)
    ]


# -- benign --------------------------------------------------------------------

#: Per-family benign char caps. One shared cap cannot serve both corpora —
#: see WILDJAILBREAK_ADVERSARIAL_MAX_CHARS in data.content_sets for the
#: measured distributions. ``None`` here means "the loader's own default".
BENIGN_MAX_CHARS: dict[str, int] = {"phtest": 300, "wildjailbreak_benign": 2000}


def benign_prompts(
    family: str, n: int | None = None, *, seed: int = 0,
    max_chars: int | None = None,
    path: str | None = None, control_texts: list[str] | None = None,
    trial: int = 0,
) -> list[SamplePrompt]:
    """One benign family's prompts, via the ``content_sets`` loader registry so
    the char filter and eval-control subtraction match the pool builder's."""
    from sisteer.data.content_sets import BENIGN_LOADERS

    if family not in BENIGN_LOADERS:
        raise KeyError(f"unknown benign family {family!r}; "
                       f"known: {sorted(BENIGN_LOADERS)}")
    # The benign loaders do not share one signature — load_phtest predates the
    # scale-up and takes neither control_texts nor path in the same shape — so
    # optional kwargs are filtered against each loader's real parameters rather
    # than assumed. A loader with no control_texts support still gets the
    # subtraction, applied here, so no corpus silently skips it.
    import inspect

    fn = BENIGN_LOADERS[family]
    if max_chars is None:
        max_chars = BENIGN_MAX_CHARS.get(family, 300)
    params = inspect.signature(fn).parameters
    kw = {"seed": seed}
    for name, val in (("max_chars", max_chars), ("path", path),
                      ("control_texts", control_texts)):
        if val is not None and name in params:
            kw[name] = val
    texts = fn(n, **kw)
    if control_texts and "control_texts" not in params:
        from sisteer.data.content_sets import subtract_eval_controls
        texts = subtract_eval_controls(texts, control_texts)
    src = "wildjailbreak" if family.startswith("wildjailbreak") else family
    return [
        SamplePrompt(
            family=family, goal_id=_synthetic_goal_id(family, t, i),
            sample_idx=0, trial=trial, variant_name=family, text=t,
            side="benign", source=src, category=None)
        for i, t in enumerate(texts)
    ]


# -- the whole inventory --------------------------------------------------------

def build_inventory(
    *,
    goals: Sequence,
    attack_families: Iterable[str] = HARMFUL_ATTACK_FAMILIES,
    benign_families: Iterable[str] = BENIGN_FAMILIES,
    n_wildjailbreak_harmful: int | None = None,
    n_benign_per_family: dict[str, int | None] | None = None,
    seed: int = 0,
    bon_variants: int = DEFAULT_BON_VARIANTS,
    subenv_banks: dict[str, dict] | None = None,
    control_texts: list[str] | None = None,
    benign_paths: dict[str, str] | None = None,
    benign_max_chars: dict[str, int] | None = None,
    wildjailbreak_path: str | None = None,
) -> list[SamplePrompt]:
    """The full prompt inventory, deterministic in ``seed``.

    ``control_texts`` (OR-Bench-hard-1k + XSTest-safe) is subtracted from every
    corpus that supports it, so the sampled set is disjoint from the eval
    controls before a single token is generated — the content-level backstop
    to the pool builder's own ``_strip_scoring``."""
    out: list[SamplePrompt] = list(harmful_attack_prompts(
        goals, attack_families, seed=seed, bon_variants=bon_variants,
        subenv_banks=subenv_banks))
    if n_wildjailbreak_harmful != 0:
        out += wildjailbreak_harmful_prompts(
            n_wildjailbreak_harmful, seed=seed, path=wildjailbreak_path,
            control_texts=control_texts)
    paths = benign_paths or {}
    counts = n_benign_per_family or {}
    for fam in benign_families:
        out += benign_prompts(
            fam, counts.get(fam), seed=seed, path=paths.get(fam),
            max_chars=benign_max_chars.get(fam) if benign_max_chars else None,
            control_texts=control_texts)
    return out


def inventory_counts(prompts: Iterable[SamplePrompt]) -> dict:
    """Per-side / per-family / per-source counts — the numbers to report
    before committing GPU time. Counts only; no text."""
    by_family: dict[str, int] = {}
    by_side: dict[str, int] = {}
    by_source: dict[str, int] = {}
    goals_by_family: dict[str, set] = {}
    for p in prompts:
        by_family[p.family] = by_family.get(p.family, 0) + 1
        by_side[p.side] = by_side.get(p.side, 0) + 1
        by_source[p.source] = by_source.get(p.source, 0) + 1
        goals_by_family.setdefault(p.family, set()).add(p.goal_id)
    return {
        "n_prompts": sum(by_family.values()),
        "by_side": dict(sorted(by_side.items())),
        "by_family": dict(sorted(by_family.items())),
        "by_source": dict(sorted(by_source.items())),
        "n_goals_by_family": {k: len(v) for k, v in sorted(goals_by_family.items())},
    }

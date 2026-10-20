#!/usr/bin/env python3
"""qwen-base-seed-replicates: THE eval-seed rule of the lane (eval seeds are different per
replicate and replicable), stdlib only so the shell gates and the matrix generator can import it.

One integer per replicate ``s``, the seed BASE every instrument that takes a seed draws under — THE LITERAL TRAINING SEED:

    seed_base(s) = s

    s1 1   s2 2   s3 3        (pinned by tests/test_seedrep_eval_seed.py)

qwen-seedrep-native-battery: the per-request seed base is the literal
training seed of the object's view, closed over item identity by each instrument's own rule (static / over-refusal
sha256(f"{s}|{trial}|{family}|{goal_id}|{sample_idx}")[:4], BoN bon_decode_seed_v1(goal, sample, trial, base=s), GSM8K
inspect_replicate_seed(item_id, sample_idx, base=s), ...). This RETIRES the hashed rule of qwen-base-seed-replicates
(int(sha256(f"{SEED_BASE_CAMPAIGN}|seedrep|{s}").hexdigest()[:8], 16): s1 4057844608, s2 2226795507, s3 2554970601) before
any leg of the lane ran under it — the hashed bases are on no store. The campaign base (20260909, SEED_BASE_CAMPAIGN) stays
the sampled re-run's ``seed_base`` for every NON-lane view (configs/sdfbase_sampled_decode.yaml); the replicate bases
REPLACE it on the lane's views only (``seed_base_by_view`` in that file — every existing view keeps 20260909 byte for
byte), and a replicate base is never the campaign base (1, 2, 3 != 20260909; the generators refuse otherwise). Item-set
seeds are NEVER re-rolled (the held-out goal split, the OR-Bench subset, the GSM8K Platinum selection, BoN's AUGMENTATION
seeds): a replicate base changes DECODE seeds only.

Every row a leg files records ``seed_rule`` (one of :data:`SEED_RULES`), ``seed_base`` (the int, or null) and the
decode-seed policy id the instrument realised; :func:`leg_seed_rule` is the ONE table of which rule each leg
carries and through which mechanism the base reaches it (§13). ``seed_ledger.py`` walks the stores against it.

  python scripts/sdfbase/seedrep/eval_seed.py                 # the three bases, one per line: `s<s> <base>`
  python scripts/sdfbase/seedrep/eval_seed.py --seed 2        # one base on stdout
  python scripts/sdfbase/seedrep/eval_seed.py --view s3       # the same, by view name
  python scripts/sdfbase/seedrep/eval_seed.py --legs          # the §13 table as JSON
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from typing import NamedTuple, Optional

#: the campaign's seed base — the sampled re-run's (configs/sdfbase_sampled_decode.yaml `seed_base`) for every NON-lane view; it
#: is NOT an input of the rule (the retired hashed rule derived the replicate bases from it), only the value every
#: replicate base must differ from
SEED_BASE_CAMPAIGN = 20260909
#: the replicate seeds (spec: new randomness is seeded 1 and never left unset; the originals are seed 0)
SEEDS = (1, 2, 3)
#: the served target every replicate view hangs off (the roster's `qwen35_4b`; views `s1` / `s2` / `s3`, TAG `qwen35_4b_s<s>`)
TARGET = "qwen35_4b"
_VIEW_RE = re.compile(r"^s([1-9][0-9]*)$")
#: b64-capability-agentic-queue: the PAIRED view of replicate s — `s<s>p`, the composed difference_relu arm on its
#: own additive view (the p-view route) — carries the SAME seed base as `s<s>`: every arm served on it
#: draws under seed base s beside its own native `none`. `view_of(s)` stays the seed view; the paired view is a second view of
#: the same replicate, never a fourth seed.
_PAIRED_VIEW_RE = re.compile(r"^s([1-9][0-9]*)p$")


def seed_base(s: int) -> int:
    """THE rule: the replicate's seed base IS its literal training seed, ``seed_base(s) = s``. A positive int
    in, the same int out; every instrument closes it over item identity by its own derivation (module docstring)."""
    if isinstance(s, bool) or not isinstance(s, int) or s < 1:
        raise ValueError(f"a replicate seed is a positive int, got {s!r}")
    return int(s)


def view_of(s: int) -> str:
    """The roster view of replicate ``s`` (`s1`)."""
    if s not in SEEDS:
        raise ValueError(f"replicate seed {s!r} is not one of {SEEDS}")
    return f"s{s}"


def seed_of_view(view: str) -> int:
    """The replicate seed a view name carries (`s2` -> 2; the paired view `s2p` -> 2 too); refuses any other view by name."""
    m = _VIEW_RE.match(str(view)) or _PAIRED_VIEW_RE.match(str(view))
    if not m or int(m.group(1)) not in SEEDS:
        raise ValueError(f"{view!r} is not a seed-replicate view ({', '.join(view_of(s) for s in SEEDS)}"
                         f" or their paired views {', '.join(paired_view_of(s) for s in SEEDS)})")
    return int(m.group(1))


def paired_view_of(s: int) -> str:
    """The PAIRED view of replicate ``s`` (`s1p`): the composed difference_relu arm's own view, seed base ``seed_base(s)``."""
    if s not in SEEDS:
        raise ValueError(f"replicate seed {s!r} is not one of {SEEDS}")
    return f"s{s}p"


def is_paired_view(view: str) -> bool:
    m = _PAIRED_VIEW_RE.match(str(view))
    return bool(m) and int(m.group(1)) in SEEDS


def is_seedrep_view(view: str) -> bool:
    """A seed view `s<s>` or its paired view `s<s>p`: both carry replicate s's seed base."""
    m = _VIEW_RE.match(str(view)) or _PAIRED_VIEW_RE.match(str(view))
    return bool(m) and int(m.group(1)) in SEEDS


def tag_of(s: int) -> str:
    """The TAG of replicate ``s`` (common.sh f22_resolve_config: `<key>_<view>`)."""
    return f"{TARGET}_{view_of(s)}"


SEED_BASE_BY_SEED = {s: seed_base(s) for s in SEEDS}
SEED_BASE_BY_VIEW = {view_of(s): seed_base(s) for s in SEEDS}
#: the paired views (b64-capability-agentic-queue): `s<s>p` -> seed_base(s). Kept APART from SEED_BASE_BY_VIEW, which every
#: seedrep consumer iterates as "the seed views"; a reader that admits either spelling reads SEED_BASE_BY_ANY_VIEW.
PAIRED_SEED_BASE_BY_VIEW = {paired_view_of(s): seed_base(s) for s in SEEDS}
SEED_BASE_BY_ANY_VIEW = {**SEED_BASE_BY_VIEW, **PAIRED_SEED_BASE_BY_VIEW}
SEED_BASE_BY_TAG = {tag_of(s): seed_base(s) for s in SEEDS}

#: the three row markers: the leg drew under seed_base(s); the instrument's frozen seed rule (no base slot);
#: the instrument sends no seed at all (the engine sampler row is recorded instead — never an invented seed)
RULE_REPLICATE = "replicate_seed_base"
RULE_FROZEN = "instrument_frozen"
RULE_NONE_SENT = "none_sent"
SEED_RULES = (RULE_REPLICATE, RULE_FROZEN, RULE_NONE_SENT)

#: the opt-in decode-seed policy ids the two shared instruments implement for this lane (their derivations live with
#: the instruments — bon.bon_decode_seed_v1 / contracts.inspect_replicate_seed — never here)
BON_DECODE_SEED_POLICY = "replicate_v1"
BON_SEED_RULE_RECORDED = "bon_seed_v1+decode:replicate_v1"      # the `seed_rule` a BoN row records under the policy
GSM8K_INSPECT_SEED_POLICY = "replicate"
#: MMLU's letter read SENDS `seed 0` (served_capability_run.py: temperature 0.0, max_tokens 1, seed 0) — a fixed seed,
#: sampler-moot under greedy; §13 rules it `instrument_frozen` with this realised value (the smoke's request evidence
#: must show the key, else the ledger flips the leg to `none_sent` from evidence)
MMLU_FROZEN_SEED = 0


class LegSeedRule(NamedTuple):
    eval_id: str
    seed_rule: str                 # one of SEED_RULES
    mechanism: str                 # how the base reaches the instrument (or why it cannot)
    seed_field: Optional[str]      # the per-row field that carries the realised seed (None = no seed on the wire)
    decode_seed_policy: Optional[str]   # the instrument's policy id under this lane (None = the instrument's default / no policy)
    reaches_via: Optional[str]     # the config / flag / env the base rides in on


_STATIC = ("pasttense", "codeattack", "renellm", "grok420")
_OVER_REFUSAL = ("xstest", "or_bench_hard_1k")
_MULTITURN = ("crescendo", "pair", "actorattack")
_MISALIGN = ("misalign_leaking", "misalign_murder")

LEG_SEED_RULES: dict = {}
for _fam in _STATIC:
    LEG_SEED_RULES[_fam] = LegSeedRule(
        _fam, RULE_REPLICATE,
        "static_common.static_sampling_seed(seed_base, trial, family, goal_id, sample_idx) -> req['seed']",
        "sampling_seed", None, "the per-view generated sampled config's sampled_decode.seed_base (seed_base_by_view)")
for _ctl in _OVER_REFUSAL:
    LEG_SEED_RULES[_ctl] = LegSeedRule(
        _ctl, RULE_REPLICATE,
        "sampled_controls.sampling_seed(seed_base, trial, readout, item_id) -> req['seed']; floor sample sha256(f'{seed_base}|floor|...')",
        "sampling_seed", None, "the same per-view sampled config block")
LEG_SEED_RULES["mmlu"] = LegSeedRule(
    "mmlu", RULE_FROZEN,
    f"the letter read sends temperature 0.0, max_tokens 1, seed {MMLU_FROZEN_SEED} (sampler-moot under greedy)",
    "seed", None, None)
LEG_SEED_RULES["gsm8k"] = LegSeedRule(
    "gsm8k", RULE_REPLICATE,
    "contracts.inspect_replicate_seed(item_id, sample_idx, seed_base) under the opt-in policy (SEED_RULE_INSPECT_REPLICATE hashed "
    "into the tinycap identity only when set); default SEED_RULE_INSPECT stays for every other run",
    "seed", GSM8K_INSPECT_SEED_POLICY, "--inspect-seed-policy replicate --inspect-seed-base <base> from the matrix row")
LEG_SEED_RULES["bon"] = LegSeedRule(
    "bon", RULE_REPLICATE,
    "augmentation bon_seed(goal, sample, trial) FROZEN (variant bytes unchanged); decode bon.bon_decode_seed_v1(goal, sample, trial, base)",
    "sampling_seed", BON_DECODE_SEED_POLICY, "--decode-seed-policy replicate_v1 --decode-seed-base <base> (target_static.sbatch: "
                                              "F22_BON_DECODE_SEED_POLICY / F22_BON_DECODE_SEED_BASE)")
for _atk in _MULTITURN:
    LEG_SEED_RULES[_atk] = LegSeedRule(_atk, RULE_NONE_SENT, "the goalset path sends no `seed`; explicit T 1.0 / top_p 1.0 / top_k -1 (D4)",
                                       None, None, None)
LEG_SEED_RULES["petri"] = LegSeedRule("petri", RULE_NONE_SENT,
                                      "the target request carries no sampling key and no seed; bloom_options.seed 0 is the harness's own",
                                      None, None, None)
for _m in _MISALIGN:
    LEG_SEED_RULES[_m] = LegSeedRule(_m, RULE_NONE_SENT, "the runner sends {temperature: 1.0} only", None, None, None)
EVAL_IDS = tuple(LEG_SEED_RULES)


def leg_seed_rule(eval_id: str) -> LegSeedRule:
    """The §13 row of ``eval_id``; an unknown leg refuses by name (never a default rule)."""
    try:
        return LEG_SEED_RULES[str(eval_id)]
    except KeyError:
        raise ValueError(f"{eval_id!r} is not a leg of the lane ({', '.join(EVAL_IDS)})") from None


def leg_seed_base(eval_id: str, s: int) -> Optional[int]:
    """The ``seed_base`` a row of ``eval_id`` on replicate ``s`` records: the replicate base under ``replicate_seed_base``,
    null under the other two rules."""
    rule = leg_seed_rule(eval_id)
    return seed_base(s) if rule.seed_rule == RULE_REPLICATE else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--seed", type=int, default=None, help="one replicate seed (1|2|3): print its base")
    g.add_argument("--view", default=None, help="one replicate view (s1|s2|s3): print its base")
    g.add_argument("--legs", action="store_true", help="the §13 table (eval_id -> rule, mechanism, field, policy) as JSON")
    a = ap.parse_args(argv)
    if a.legs:
        print(json.dumps({k: v._asdict() for k, v in LEG_SEED_RULES.items()}, indent=1))
        return 0
    if a.seed is not None:
        try:
            print(seed_base(a.seed) if a.seed in SEEDS else view_of(a.seed))
        except ValueError as e:
            print(str(e), file=sys.stderr); return 2
        return 0
    if a.view is not None:
        try:
            print(SEED_BASE_BY_VIEW[view_of(seed_of_view(a.view))])
        except ValueError as e:
            print(str(e), file=sys.stderr); return 2
        return 0
    for s in SEEDS:
        print(f"{view_of(s)} {seed_base(s)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

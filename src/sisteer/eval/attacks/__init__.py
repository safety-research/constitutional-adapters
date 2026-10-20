"""Attack registry for the jailbreak eval (mirrors eval/suites.py dispatch).

A thin name -> (module, entry) table with a lazy import, so pulling one
attack never pays for another's dependencies (importing ``pair_tap`` /
``multiturn`` pulls PyRIT; ``bon`` and ``gcg`` never do). ``bon`` is the
non-PyRIT Tier-0 foundation; PAIR / TAP (Tier 1, ``pair_tap``) and Crescendo /
ActorAttack (Tier 2, ``multiturn``) are the PyRIT-backed / hand-rolled tiers
built on the adapters in ``pyrit_adapters``; ``gcg`` is the WHITE-BOX,
target-only gradient attack (Zou et al. 2307.15043) and needs NO attacker model.
``GATED_ATTACKS`` marks the ones that need the attacker model + PyRIT at run time.

TWO DISJOINT FAMILY SETS live here — the distinction is load-bearing:

  ``STATIC_FAMILIES`` — the pure in-process deterministic transforms that qualify
  for the TRAINING pool (PastTense, PAP, Encoding, AIM, GCG-by-replay). With
  AmpleGCG-plus (a sub-env generator, ``SUBENV_ATTACKS``) and BoN they
  are the in-distribution attack families the two-SI defense is fit against.

  ``HELD_OUT_FAMILIES`` — families implemented on the held-out SCAFFOLDING
  (frozen-corpus replay; see ``heldout_common``): ArtPrompt, CipherChat,
  CodeAttack, DrAttack (deterministic build kind, pure deterministic transforms), Multilingual
  and ReNeLLM (rewriter build kind, LLM-rewriter transforms frozen to a corpus), and the
  generic ``frozen_corpus`` ingested-kind ingestion adapter (+ ``frozen_demo``, its
  worked example). NOTE (2026-08-28): these sets classify DISPATCH — which
  scaffolding runs a family — not holdout. The holdout is GOAL-LEVEL: a
  family (codeattack, renellm, …) may be trained on AND evaluated on,
  against goals the fit never saw; do not reintroduce family-level
  disjointness.

The held-out families share ``static_common``'s replay/persist/resume contract,
so the grading + pool-builder read every family uniformly, and the ONE shared
grading host StrongREJECT grader scores them all identically. See ``heldout_common``
for the leakage gate, the per-family manifest, and the conformance checks;
``frozen_corpus`` for the ingested-kind adapter; ``heldout_leakage`` for the training
fingerprint. None of the held-out families pull PyRIT or torch (the deterministic-kind ones
are pure Python; the rewriter/ingested kinds read a frozen corpus).
"""

from __future__ import annotations

from importlib import import_module

# name -> (module under sisteer.eval.attacks, entry-point attribute)
JB_ATTACKS: dict[str, tuple[str, str]] = {
    "bon": ("sisteer.eval.attacks.bon", "bon_generate"),     # Tier 0
    "pair": ("sisteer.eval.attacks.pair_tap", "run_pair"),   # Tier 1 (PyRIT)
    "tap": ("sisteer.eval.attacks.pair_tap", "run_tap"),     # Tier 1 (PyRIT)
    "crescendo": ("sisteer.eval.attacks.multiturn", "run_crescendo"),  # Tier 2 (PyRIT)
    "actorattack": ("sisteer.eval.attacks.multiturn", "run_actorattack"),  # Tier 2 (hand-rolled)
    "gcg": ("sisteer.eval.attacks.gcg", "run_gcg"),  # white-box gradient (no attacker)
    # Static/replayable TRAINING families (no attacker model, no PyRIT) —
    # deterministic prompt transforms replayed under the defense.
    "pasttense": ("sisteer.eval.attacks.pasttense", "run_pasttense"),
    "pap": ("sisteer.eval.attacks.pap", "run_pap"),
    "encoding": ("sisteer.eval.attacks.encoding", "run_encoding"),
    "aim": ("sisteer.eval.attacks.aim", "run_aim"),
    "gcg_replay": ("sisteer.eval.attacks.gcg_replay", "run_gcg_replay"),
    "grok420": ("sisteer.eval.attacks.grok420", "run_grok420"),  # G0DM0D3 semantic-inversion (static)
    # Sub-env TRAINING family: a generator runs in a separate transformers<4.56
    # process (subenv_common); the produced prompts are replayed statically.
    "amplegcg_plus": ("sisteer.eval.attacks.amplegcg", "run_amplegcg"),
    # HELD-OUT eval families — see heldout_common.
    "artprompt": ("sisteer.eval.attacks.artprompt", "run_artprompt"),
    "cipherchat": ("sisteer.eval.attacks.cipherchat", "run_cipherchat"),
    "drattack": ("sisteer.eval.attacks.drattack", "run_drattack"),
    "multilingual": ("sisteer.eval.attacks.multilingual", "run_multilingual"),
    "frozen_corpus": ("sisteer.eval.attacks.frozen_corpus", "run_frozen_corpus"),
    "frozen_demo": ("sisteer.eval.attacks.frozen_demo", "run_frozen_demo"),
    # Registered for TRAINING. codeattack and renellm were previously
    # declared through the held-out-family registry; grok420 is newly
    # added (registered with the static training families
    # above). All three are deterministic static transforms with a goal slot,
    # so they replay exactly like the five above. Holding out now happens at
    # the GOAL level, so a family being trainable no longer spends it as an
    # eval family -- these are in train_families AND eval_families by design.
    "codeattack": ("sisteer.eval.attacks.codeattack", "run_codeattack"),
    "renellm": ("sisteer.eval.attacks.renellm", "run_renellm"),
}

# Attacks needing the attacker model + PyRIT (or, for ActorAttack, the attacker
# model) at run time — BoN, GCG, the static families, and the held-out families
# need none of it.
GATED_ATTACKS = frozenset({"pair", "tap", "crescendo", "actorattack"})

# Families whose generation step needs a separate transformers<4.56 sub-env
# (a DIFFERENT axis from GATED_ATTACKS). Absent sub-env => skip LOUDLY.
SUBENV_ATTACKS = frozenset({"amplegcg_plus"})

# The pure in-process deterministic transforms (static_common replay contract):
# no attacker model, no PyRIT, no sub-env. Distinct from bon/gcg, which keep
# their own dedicated driver dispatch.
STATIC_FAMILIES = frozenset({"pasttense", "pap", "encoding", "aim",
                             "gcg_replay", "codeattack", "renellm",
                             "grok420"})

# Every non-attacker, forward_cap-EXEMPT TRAINING family the run driver
# dispatches WITHOUT the attacker model (the five static transforms + the two
# sub-env generators). bon (Tier-0) and gcg (white-box) have their own dispatch.
REPLAYABLE_FAMILIES = STATIC_FAMILIES | SUBENV_ATTACKS

# name -> (module, SPEC attribute) for the HELD-OUT families. The SPEC is a
# heldout_common.HeldOutFamily carrying the provenance class and
# the replay/run seams — read by the conformance test, the leakage gate, and the
# per-family manifest.
HELD_OUT_ATTACKS: dict[str, tuple[str, str]] = {
    "artprompt": ("sisteer.eval.attacks.artprompt", "SPEC"),
    "cipherchat": ("sisteer.eval.attacks.cipherchat", "SPEC"),
    "drattack": ("sisteer.eval.attacks.drattack", "SPEC"),
    "multilingual": ("sisteer.eval.attacks.multilingual", "SPEC"),
    # codeattack/renellm moved to the static TRAINING dispatch (their SPEC
    # declarations went with the held-out registry) -- one name, one
    # scaffolding, guarded by the _leak assertion below.
    "frozen_corpus": ("sisteer.eval.attacks.frozen_corpus", "GENERIC_SPEC"),
    "frozen_demo": ("sisteer.eval.attacks.frozen_demo", "SPEC"),
}

# The held-out-SCAFFOLDING set — distinct from STATIC_FAMILIES only as a
# DISPATCH classification (one name must resolve to one scaffolding, so a
# name in both dicts would be a dispatch contradiction, guarded at import).
# It is NOT a holdout boundary: family-level holdout is retired
# (2026-08-28) — the goal split is the holdout, and a family here may
# legitimately be in a config's train_families.
HELD_OUT_FAMILIES = frozenset(HELD_OUT_ATTACKS)

# One name, one scaffolding: a family registered under both dispatch tables
# would be a dispatch contradiction (this is a REGISTRY-consistency guard,
# not the retired train/eval family-disjointness rule).
_train_all = STATIC_FAMILIES | SUBENV_ATTACKS | {"bon", "gcg"}
_leak = HELD_OUT_FAMILIES & _train_all
if _leak:
    raise AssertionError(
        f"family names {sorted(_leak)} are registered under BOTH the "
        "static/replay dispatch and the held-out scaffolding — one name "
        "must resolve to exactly one scaffolding")


def get_attack(name: str):
    if name not in JB_ATTACKS:
        raise KeyError(f"unknown attack {name!r}; known: {sorted(JB_ATTACKS)}")
    module_name, attr = JB_ATTACKS[name]
    return getattr(import_module(module_name), attr)


def get_heldout_spec(name: str):
    """The :class:`heldout_common.HeldOutFamily` SPEC for a held-out family
    (lazy import, like :func:`get_attack`)."""
    if name not in HELD_OUT_ATTACKS:
        raise KeyError(
            f"unknown held-out family {name!r}; known: {sorted(HELD_OUT_ATTACKS)}")
    module_name, attr = HELD_OUT_ATTACKS[name]
    return getattr(import_module(module_name), attr)


def heldout_specs() -> dict:
    """All held-out SPECs, ``{name: HeldOutFamily}`` — what the conformance test
    iterates so adding a family is one module + one registry line + one fixture."""
    return {name: get_heldout_spec(name) for name in HELD_OUT_ATTACKS}

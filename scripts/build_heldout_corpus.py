"""Build a rewriter-kind HELD-OUT family's FROZEN corpus — the one-time (possibly
expensive) BUILD step, separated from REPLAY.

Rewriter-kind families need an LLM at build time. This CLI builds ``multilingual``: it runs that
LLM (the Anthropic rewriter on the ``lowprio`` credential, aggressively cached), runs
the LEAKAGE GATE against the training pool, and freezes the result to a
content-hashed corpus + manifest under ``$SISTEER_HELDOUT_CORPUS_DIR`` (default
``heldout_corpora`` under the site's SISTEER_JAILBREAK_PRIVATE_ROOT, else under
``$STATE/sisteer-jailbreak``). REPLAY (the sweep's held-out leg) then
reads only that corpus — never the LLM.

    python scripts/build_heldout_corpus.py multilingual --config configs/two_si.yaml

``renellm`` is REFUSED by name, before anything is loaded or called: the paper's ReNeLLM corpus
ships in the repository (``data/heldout_corpora/renellm.jsonl``, under its own terms) and replay
reads it with no build step. A rebuild would be a new LLM sample whose results are not comparable
to the paper's renellm results (:mod:`sisteer.eval.attacks.renellm`).

COORDINATION: this issues concurrent Anthropic requests (default 200) on the
``lowprio`` credential. A site that coordinates concurrent API use in a channel
(SISTEER_COORDINATION_CHANNEL) posts there before running at that width (e.g.
"200 sonnet lowprio ~30m — held-out corpus build"); the post never gates the
launch. The rewrite cache makes a re-run resume for free.

CONTAINMENT: the frozen corpus holds transformed harmful prompts and lands ONLY
in the mode-0700 private dir; the manifest (numbers/enums/hashes only) is the
servable record. Nothing harmful is printed or committed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def renellm_refusal() -> str:
    """Why ``renellm`` is not built here (hashes and paths only)."""
    from sisteer.eval.attacks import renellm

    return (f"[build] renellm: refused. The paper's ReNeLLM corpus ships in the repository at "
            f"{renellm.SHIPPED_CORPUS} (sha256 {renellm.PINNED_FILE_SHA256}, {renellm.PAPER_CORPUS_ROWS} "
            f"rows; see {renellm.PAPER_CORPUS_RECORD}), under its own terms ({renellm.CORPUS_TERMS}), and "
            "replay reads it with no build step. A rebuild would be a new LLM sample whose results are not "
            "comparable to the paper's renellm results, so this script does not build it. Nothing was built.")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("family", choices=["multilingual", "renellm"],
                    help="the rewriter-kind held-out family to build (renellm refuses: its paper "
                         "corpus ships in the repository)")
    ap.add_argument("--config", default="configs/two_si.yaml")
    ap.add_argument("--backend", choices=["anthropic", "openrouter"],
                    default="anthropic",
                    help="which rewriter client to build with")
    ap.add_argument("--model", default=None,
                    help="the rewriter model id (recorded in the manifest); "
                         "defaults per backend (anthropic: claude-sonnet-5, "
                         "openrouter: x-ai/grok-4.5)")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="concurrent rewrite workers (default 200 anthropic / "
                         "32 openrouter)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="per-rewrite output cap (default 1024 anthropic / "
                         "2048 openrouter)")
    ap.add_argument("--org", default=None,
                    help="Anthropic org for the anthropic backend (vault-resolved, "
                         "never a file); OpenRouter takes its key from the vault too")
    ap.add_argument("--cache", default=None,
                    help="rewrite cache jsonl (defaults under the corpus dir)")
    ap.add_argument("--out", default=None, help="corpus path override")
    ap.add_argument("--pilot", action="store_true",
                    help="use the smaller pilot goal set instead of confirmatory")
    args = ap.parse_args(argv)
    if args.family == "renellm":
        print(renellm_refusal(), file=sys.stderr)
        raise SystemExit(2)

    import yaml

    import jailbreak_run as jr
    from sisteer.eval.attacks import get_heldout_spec
    from sisteer.eval.attacks.heldout_leakage import (
        load_stage3_training_fingerprint_from_config)
    from sisteer.eval.attacks.llm_rewriter import (
        AnthropicRewriter, OpenRouterRewriter)

    cfg = yaml.safe_load(Path(args.config).read_text())
    goals = jr._eval_goals(cfg, confirmatory=not args.pilot)
    fingerprint = load_stage3_training_fingerprint_from_config(cfg)
    print(f"[build] {args.family}: {len(goals)} goals, "
          f"{len(fingerprint)} training shas for the leakage gate")

    spec = get_heldout_spec(args.family)
    assert spec.build_fn is not None, f"{args.family} is not a rewriter-kind family"
    from sisteer.eval.jailbreak_artifacts import heldout_corpus_dir
    cache = args.cache or str(heldout_corpus_dir() / f"{args.family}.rewrite_cache.jsonl")

    if args.backend == "openrouter":
        model = args.model or "x-ai/grok-4.5"
        rewriter = OpenRouterRewriter(
            model=model, cache_path=cache,
            concurrency=args.concurrency or 32,
            max_tokens=args.max_tokens or 2048)
    else:
        model = args.model or "claude-sonnet-5"
        rewriter = AnthropicRewriter(
            model=model, cache_path=cache, org=args.org,
            concurrency=args.concurrency or 200,
            max_tokens=args.max_tokens or 1024)
    print(f"[build] backend={args.backend} model={model}")
    manifest = spec.build_fn(goals, rewriter=rewriter, fingerprint=fingerprint,
                             out_path=args.out)
    # manifest is sanitised (no prompt text) — safe to print
    print(json.dumps(manifest, indent=1, sort_keys=True))
    print(f"[build] {args.family} corpus frozen: {manifest['row_count']} rows, "
          f"content hash {manifest['corpus_content_hash']}")


if __name__ == "__main__":
    # A credential rejection that reaches main -- the shared judge's / grader's typed
    # error -- ends the process with the typed text as its terminal line, never a traceback
    from sisteer import vault  # noqa: E402

    with vault.credential_exit(prefix="[build] "):
        main()

"""AmpleGCG-plus — learned GCG-suffix generator (Liao & Sun 2404.07921,
"AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial
Suffixes"; the "-plus" checkpoint is the augmented-training successor).

AmpleGCG amortises GCG: a fine-tuned generator model emits MANY adversarial
suffixes for a given request in one pass, replacing per-goal gradient search.
The generator's reference stack pins an older ``transformers`` API, so it runs in
a dedicated ``transformers<4.56`` sub-env
(``sisteer.eval.attacks._amplegcg_worker``) and the main process REPLAYS
``f"{goal} {suffix}"`` for each generated suffix through ``defense.generate`` —
a fixed bank per goal, so the produced prompts are STATIC/replayable and qualify
for the two-SI pool. Each generated suffix is a distinct variant (sample_idx
0/1/2…).

GATING. It needs neither the repo's attacker model nor PyRIT, so it
is NOT in ``GATED_ATTACKS``; it gates on the SUB-ENV availability check and SKIPS
LOUDLY when the sub-env is absent.

COST ACCOUNTING (JB7). EXEMPT from ``forward_cap`` (``capped: False``) with its
OWN cost class ``amplegcg_optimizer_cost`` (``n_generator_forwards`` = generator
decode forwards, ``n_prompts_generated`` = suffixes produced), reported
SEPARATELY from ``n_target_forwards`` (the real replays).

PROVISIONING (what a real run needs — none installed here):
  - a ``transformers<4.56`` venv, e.g.
    ``UV_PROJECT_ENVIRONMENT=/home/$USER/venvs/sisteer-attackgen-amplegcg``,
    with ``transformers<4.56``, ``torch``, ``accelerate`` (optionally ``vllm``
    for group-beam decode).
  - the AmpleGCG-plus generator checkpoint from the Hub, e.g.
    ``osunlp/AmpleGCG-plus-llama2-sourced-llama2-7b-chat`` (~7B) — pass it via
    ``$SISTEER_AMPLEGCG_MODEL`` / ``generator_model=``.
  - point ``$SISTEER_AMPLEGCG_SUBENV`` / ``subenv_python=`` at that venv's python.
  - invocation: ``python -m sisteer.eval.attacks._amplegcg_worker`` under that
    interpreter, JSON payload on stdin, ``{suffixes, n_generator_forwards}`` on
    stdout.
"""

from __future__ import annotations

from .static_common import run_static
from .subenv_common import loud_skip, run_worker, subenv_available

SUBENV_ENV_VAR = "SISTEER_AMPLEGCG_SUBENV"
MODEL_ENV_VAR = "SISTEER_AMPLEGCG_MODEL"
WORKER_MODULE = "sisteer.eval.attacks._amplegcg_worker"
DEFAULT_N_SUFFIXES = 8


def _subenv_generate(python, generator_model, n_suffixes, timeout):
    """Return ``gen(goal, seed) -> (variants, cost)`` driving the sub-env worker.
    ``variants`` = ``[(f"amplegcg{i}", f"{goal} {suffix}"), ...]``; ``cost`` =
    ``{n_generator_forwards, n_prompts_generated}``."""
    def gen(goal, seed):
        out = run_worker(python, WORKER_MODULE, {
            "goal_text": goal.text, "seed": int(seed),
            "n_suffixes": int(n_suffixes), "model": generator_model,
        }, timeout=timeout)
        suffixes = list(out["suffixes"])
        variants = [(f"amplegcg{i}", f"{goal.text} {suffix}")
                    for i, suffix in enumerate(suffixes)]
        cost = {"n_generator_forwards": int(out.get("n_generator_forwards", 0)),
                "n_prompts_generated": len(suffixes)}
        return variants, cost
    return gen


def run_amplegcg(defense, goals, *, out_dir, subenv_python=None,
                 generator_model=None, n_suffixes: int = DEFAULT_N_SUFFIXES,
                 seed: int = 0, max_new_tokens: int = 512, resume: bool = True,
                 worker_timeout: float | None = None,
                 _available=None, _generate=None) -> dict:
    """Generate AmpleGCG-plus suffixes in the sub-env and replay them under
    ``defense``. Skips LOUDLY (writing nothing) when the sub-env is unavailable.
    ``_available`` / ``_generate`` are test seams. Grading is the separate
    grading host StrongREJECT post-step."""
    avail = _available or (lambda: subenv_available(
        SUBENV_ENV_VAR, subenv_python, model_env=MODEL_ENV_VAR,
        model_path=generator_model))
    ok, info = avail()
    if not ok:
        return loud_skip("amplegcg_plus", info)

    gen = _generate or _subenv_generate(info, generator_model, n_suffixes,
                                        worker_timeout)

    costs: dict[str, dict] = {}

    def variants_fn(goal, s):
        variants, cost = gen(goal, s)
        costs[goal.goal_id] = cost
        return variants

    def progress_extra(goal):
        c = costs.get(goal.goal_id, {})
        return {
            "cost_class": "amplegcg_optimizer_cost",
            "n_generator_forwards": int(c.get("n_generator_forwards", 0)),
            "n_prompts_generated": int(c.get("n_prompts_generated", 0)),
        }

    result = run_static(defense, goals, out_dir=out_dir, family="amplegcg_plus",
                        variants_fn=variants_fn, trial=0, seed=seed,
                        max_new_tokens=max_new_tokens, resume=resume,
                        progress_extra_fn=progress_extra)
    result["amplegcg_optimizer_cost"] = sum(
        r.get("n_generator_forwards", 0) for r in result["per_goal"])
    return result

"""Stage-2 suite dispatcher: suite name -> (module, entry point).

Deliberately thin — a stable lookup for scripts/stage2_run.py and the
sbatch drivers, so suite names used in configs/output dirs/W&B resolve to
exactly one runner. Import is lazy: pulling one suite never pays for
another's dependencies.

Suite membership of the 12 method arms lives in methods.py (ArmSpec.suites)
— this table only names the runners.
"""

from __future__ import annotations

from importlib import import_module

# suite name -> (module under sisteer.eval, entry-point attribute)
# "lifbench" is deliberately absent: removed from the campaign because its
# instruction-after-context design measures retrieval, not instruction
# decay. eval/lifbench.py stays importable for reference, but it is not a
# run suite, so a stale lookup fails loudly in get_runner.
# "ifeval_anchor", "drift", "axbench" and "longctx" are absent too: the
# vendored verifier, the PersonaDrift scorers, AxBench and the IFEval-at-length
# suite were third-party material behind no paper result, so they were removed
# rather than attributed. A stale lookup fails loudly in get_runner.
SUITES: dict[str, tuple[str, str]] = {
    "ifeval": ("sisteer.eval.behavioral", "eval_instruction"),
    "capability": ("sisteer.eval.capability_ext", "capability_panel"),
    "composition": ("sisteer.eval.composition", "eval_composition"),
}


def get_runner(suite: str):
    if suite not in SUITES:
        raise KeyError(f"unknown suite {suite!r}; known: {sorted(SUITES)}")
    module_name, attr = SUITES[suite]
    return getattr(import_module(module_name), attr)

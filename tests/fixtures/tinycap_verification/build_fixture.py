#!/usr/bin/env python
"""Builds ``unscored_12.eval`` — the SYNTHETIC unscored BigCodeBench generation log the
landing-2 tests bind against (tinycap-shared-standup plan §2.2.3, "a fixture unscored log
of 12 samples (synthetic completions)"). Never real data: twelve invented tasks whose
``test`` is a tiny unittest over ``task_func``, twelve invented completions (fenced, bare-
fenced and whole-completion code; correct and incorrect; one cut at the cap), served by a
fixture provider REGISTERED UNDER THE PRODUCTION NAME ``tinycap`` so the log's shape is the
production log's — ``eval.model`` = ``tinycap/petri__base``, every sample's ``model_usage``
keyed by it, ``output.model`` the bare virtual model, the plan config the driver's operational
keys + the cap + the greedy row — generated with ``score=False`` exactly as the GPU phase does.

    INSPECT_DISPLAY=none PYTHONPATH=src ~/venvs/tinycap-laneB/bin/python \\
        tests/fixtures/tinycap_verification/build_fixture.py

Deterministic: the fixture provider is a table, the eval runs at ``max_connections=4`` and the
log is rewritten to the fixed name; only the header's timestamps and run ids differ between
builds (the tests never read them). The registered ``tinycap`` name lives in THIS process
alone — the real provider (``inspect_served``) is never imported here.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from sisteer.capeval.tinycap import contracts as C  # noqa: E402

FIXTURE = HERE / "unscored_12.eval"
VMODEL = "petri__base"
N = 12


def synthetic_samples():
    """Twelve (id, prompt, test, completion, stop_reason, usage) rows. Every test is a unittest over
    ``task_func(x)`` expecting ``2 * x``; the completions vary in correctness and code shape."""
    from inspect_ai.dataset import Sample

    rows = []
    for i in range(N):
        test = ("import unittest\n"
                f"class TestCases(unittest.TestCase):\n"
                f"    def test_case_1(self):\n"
                f"        self.assertEqual(task_func({i}), {2 * i})\n"
                f"    def test_case_2(self):\n"
                f"        self.assertEqual(task_func(1), 2)\n")
        prompt = (f"import math\n\ndef task_func(x):\n    \"\"\"Return twice x. (synthetic fixture item {i})\n\n"
                  "    Requirements:\n    - math\n\n    Example:\n    >>> task_func(2)\n    4\n    \"\"\"\n")
        factor = 2 if i % 3 != 1 else 3          # ids 1, 4, 7, 10 are wrong
        body = f"import math\n\ndef task_func(x):\n    return {factor} * x\n"
        if i in (10, 11):
            completion = body                    # whole completion, no fence
        elif i in (8, 9):
            completion = f"Here is the solution:\n```\n{body}```\n"      # bare fence
        else:
            completion = f"```python\n{body}```\n"
        stop = "max_tokens" if i == 7 else "stop"
        rows.append((i, prompt, test, completion, stop, (200 + i, 40 + i)))
    samples = [Sample(id=i, input=prompt, target=test,
                      metadata={"test": test, "libs": "['math']", "task_id_metadata": f"BigCodeBench/{i}"})
               for i, prompt, test, _c, _s, _u in rows]
    return rows, samples


def main() -> int:
    os.environ.setdefault("INSPECT_DISPLAY", "none")
    from inspect_ai import Task, eval_async
    from inspect_ai.dataset import MemoryDataset
    from inspect_ai.log import read_eval_log, write_eval_log
    from inspect_ai.log._samples import sample_active
    from inspect_ai.model import GenerateConfig, ModelAPI, ModelOutput, ModelUsage, modelapi
    from inspect_ai.scorer import Score, mean, scorer, std
    from inspect_ai.solver import generate

    rows, samples = synthetic_samples()
    table = {str(i): (c, s, u) for i, _p, _t, c, s, u in rows}

    @modelapi(name=C.INSPECT_PROVIDER_NAME)
    class FixtureAPI(ModelAPI):
        def __init__(self, model_name, base_url=None, api_key=None, api_key_vars=None, config=None, **kw):
            super().__init__(model_name, base_url, api_key, list(api_key_vars or []),
                             config if config is not None else GenerateConfig())

        async def generate(self, input, tools, tool_choice, config):
            active = sample_active()
            completion, stop, (pt, ct) = table[str(active.sample.id)]
            out = ModelOutput.from_content(model=VMODEL, content=completion, stop_reason=stop)
            out.usage = ModelUsage(input_tokens=pt, output_tokens=ct, total_tokens=pt + ct)
            return out

        def connection_key(self):
            return "fixture"

    @scorer(metrics=[mean(), std()])
    def verify():
        """The stock scorer's NAME on the task (the plan's primary-scorer check reads it); never run
        here — the eval is score=False, exactly as the GPU phase's."""
        async def score(state, target):
            return Score(value="I", answer="", explanation="never run: score=False")
        return score

    task = Task(dataset=MemoryDataset(samples), solver=[generate()], scorer=verify(), sandbox=None,
                version=C.BCB_SPLIT, name="bigcodebench")
    out_dir = HERE / "_build"
    out_dir.mkdir(exist_ok=True)
    logs = asyncio.run(eval_async(
        task, model=C.inspect_model_name(VMODEL), log_dir=str(out_dir), log_format="eval",
        sample_id=[str(i) for i in range(N)], max_connections=4, score=False, fail_on_error=True,
        retry_on_error=2, log_realtime=False, max_retries=C.INSPECT_MAX_RETRIES, timeout=C.INSPECT_TIMEOUT_S,
        max_tokens=4096, temperature=0.0, top_p=1.0, top_k=0))
    log = logs[0]
    assert log.status == "success" and len(log.samples) == N, (log.status, len(log.samples))
    assert all(not s.scores for s in log.samples)
    assert log.eval.model == C.inspect_model_name(VMODEL)
    assert set(log.samples[0].model_usage) == {C.inspect_model_name(VMODEL)}
    write_eval_log(read_eval_log(str(log.location)), str(FIXTURE), format="eval")
    for p in out_dir.iterdir():
        p.unlink()
    out_dir.rmdir()
    back = read_eval_log(str(FIXTURE))
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes): {len(back.samples)} unscored samples, model "
          f"{back.eval.model}, ids {[s.id for s in back.samples]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

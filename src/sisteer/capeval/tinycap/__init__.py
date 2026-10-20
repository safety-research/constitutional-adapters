"""tinycap — the broader served capability suite on the served, steerable
vLLM path: MATH-500, GSM8K, HumanEval+/MBPP+, BigCodeBench, BFCL and IFEval
via Inspect / EvalPlus on the proxy, and the like-for-like Inspect
comparisons of our served controls.

This package holds the INTERFACE CONTRACTS only (``contracts``): every
name that more than one lane types — control names, row task/metric
names, manifest keys, geometry sub-entry keys, the usage-ledger row
schema, the ``X-Tinycap-*`` header contract, the Inspect provider name,
the launcher argv, the sandbox limits, the client-lock pins. Lanes import
these; they never duplicate a string. The scorer extension itself lives
in ``sisteer.capeval.serving.prefill_logprob`` (signatures frozen here,
bodies Lane A's), the task adapter in ``sisteer.capeval.tiers.
inspect_served`` (Lane B), the served EvalPlus engine
in ``sisteer.capeval.serving.proxy_engine`` (Lane C), the ledger and the
geometry proposals in ``sisteer.capeval.serving.{usage_ledger,geometry}``
(Lane D). ``docs/tinycap_interfaces.md`` is the human-readable copy;
``tests/test_tinycap_contracts.py`` keeps the two in agreement.

Base-lane import rule (the capeval package docstring): nothing here
imports inspect_ai, evalplus, vllm or torch at import time.
"""

from sisteer.capeval.tinycap import contracts

__all__ = ["contracts"]

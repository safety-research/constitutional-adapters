"""capeval — the served-model evaluation harness. The name is a historical misnomer.

``capeval`` does not mean "capability only", and its sibling
:mod:`sisteer.eval` does not mean "safety only". This package is the harness
for everything evaluated against a SERVED model: target serving over vLLM,
the routing proxy and the lens workers (``serving/``, the largest part of the
package); the served capability suite — GSM8K, MATH-500, MMLU, IFEval,
BigCodeBench, BFCL and tinycap (``tiers/``, ``tinycap/``); the Petri
misalignment audits (``petri/``); the misalignment evals (``misalign/``); the
result store, statistics and report (``store``, ``stats/``, ``report/``); and
the registry-facing contracts (``manifest``, ``engine_identity``,
``runtime``, ``judging``). The in-process jailbreak battery and its attacks,
the DefenseCell identity and the over-refusal controls live in
:mod:`sisteer.eval`; the two packages import each other.

The paths are kept as they are because recorded run identities (harness
versions, pins and registry records) reference these module paths; renaming
either package would break the join between results and code.

``manifest``, ``domains``, ``store`` and ``stats`` are the capability-
preservation core every tier writes into and the report reads from. A
manifest names the target model and the intervention under evaluation
(steering vector / LoRA / PEFT) and is validated loudly and hashed into the
run identity; ``domains`` is the task-to-domain registry every emitted
metric must resolve in; ``store`` files one result row per item per cell,
resumes a cell from its rows and consolidates cells; ``stats`` decides
whether the intervention *specifically* altered the model or *broadly*
degraded it, from the many-domain capability fingerprint those rows form
(calibration, non-inferiority, fingerprint verdict, power). The tinycap
interfaces are described in docs/tinycap_interfaces.md.

The dataclasses, signatures and schemas of ``manifest``, ``domains``,
``store`` and ``serving`` are a shared contract: every tier codes against
them, ``tests/test_capeval_protocol.py`` pins them, and an interface change
lands together with that suite.

Base-lane import rule: every module imported here must work with the base
dependencies alone (torch/numpy/pandas/pyarrow/datasets/safetensors) — no
evalplus, no vllm at import time. Heavy dependencies are
imported lazily inside the tier runners that need them.
"""

from sisteer.capeval.domains import Domain, Registry
from sisteer.capeval.manifest import (
    AblationSpec,
    CorpusRef,
    Manifest,
    PrimarySpec,
    TargetSpec,
    canonical_json,
    intervention_hash,
    load_manifest,
    sha256_hex16,
    validate_manifest,
)
from sisteer.capeval.store import (
    ROW_COLUMNS,
    CellStore,
    ResultRow,
    cell_key,
    consolidate,
    env_digest,
    harness_config_hash,
    ts_metric,
)

__all__ = [
    "AblationSpec",
    "CellStore",
    "CorpusRef",
    "Domain",
    "Manifest",
    "PrimarySpec",
    "ROW_COLUMNS",
    "Registry",
    "ResultRow",
    "TargetSpec",
    "canonical_json",
    "cell_key",
    "consolidate",
    "env_digest",
    "harness_config_hash",
    "intervention_hash",
    "load_manifest",
    "sha256_hex16",
    "ts_metric",
    "validate_manifest",
]

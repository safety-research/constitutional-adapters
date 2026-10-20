"""capeval stats: calibration, non-inferiority, fingerprint/
verdict, power. Base-lane only (numpy/pandas — no heavy deps).

``stats/controls.py`` lives in this package too but is
deliberately NOT imported here: the control-arm constructors pull in
model-side machinery this package's consumers don't need at import time.

The metric-name contract shared with the tier runners lives
in ``noninferiority`` and is re-exported here: METRIC_ACC ("acc", one
row per item), METRIC_NLL_SUM/METRIC_TOKEN_COUNT (per-window F14 rows),
METRIC_KL ("kl_coarse_lb", per window), CANARY_TASK/CANARY_AGG_ITEM/
CANARY_DIRECTIONS (per-indicator aggregate rows). Summary rows under
item_id AGGREGATE_ITEM_ID are never paired items — pairing drops them
(2026-08-17 incident); only the canary pairs on them, deliberately.
"""

from sisteer.capeval.stats.calibrate import (
    Calibration,
    DomainCalibration,
    TaskCalibration,
    build_calibration,
    canary_thresholds_from_values,
    check_calibration,
    derive_kappa,
    growth_needed,
    harness_version,
    load_calibration,
    save_calibration,
)
from sisteer.capeval.stats.fingerprint import (
    SCREEN_FAIL,
    SCREEN_PASS,
    SCREENS,
    PrimaryResult,
    TargetStatus,
    Verdict,
    build_fingerprint,
    confirmatory_verdict,
    screen_status,
    target_status,
    verdict,
)
from sisteer.capeval.stats.noninferiority import (
    AGGREGATE_ITEM_ID,
    BOUND_METHOD,
    CANARY_AGG_ITEM,
    CANARY_DIRECTIONS,
    CANARY_TASK,
    METRIC_ACC,
    METRIC_KL,
    METRIC_NLL_SUM,
    METRIC_TOKEN_COUNT,
    STATUS_DEGRADED,
    STATUS_INDETERMINATE,
    STATUS_PRESERVED,
    STATUSES,
    DomainStatus,
    StatsError,
    check_model_matched,
    domain_status,
    mcnemar_exact_p,
    statuses_for_arm,
)
from sisteer.capeval.stats.power import (
    POWER_THRESHOLD,
    PowerError,
    PowerResult,
    domain_power,
    floored_discordance_rate,
    power_table,
    rule_of_three_floor,
    simulate_power_accuracy,
    simulate_power_nll,
)

__all__ = [
    "AGGREGATE_ITEM_ID",
    "BOUND_METHOD", "CANARY_AGG_ITEM", "CANARY_DIRECTIONS", "CANARY_TASK",
    "Calibration", "DomainCalibration", "DomainStatus", "METRIC_ACC",
    "METRIC_KL", "METRIC_NLL_SUM", "METRIC_TOKEN_COUNT", "POWER_THRESHOLD",
    "PowerError", "PowerResult", "PrimaryResult", "SCREENS", "SCREEN_FAIL",
    "SCREEN_PASS", "STATUSES", "STATUS_DEGRADED", "STATUS_INDETERMINATE",
    "STATUS_PRESERVED", "StatsError", "TargetStatus", "TaskCalibration",
    "Verdict",
    "build_calibration", "build_fingerprint", "canary_thresholds_from_values",
    "check_calibration", "check_model_matched", "confirmatory_verdict",
    "derive_kappa",
    "domain_power", "domain_status", "floored_discordance_rate",
    "growth_needed", "harness_version", "load_calibration",
    "mcnemar_exact_p", "power_table", "rule_of_three_floor",
    "save_calibration", "screen_status", "simulate_power_accuracy",
    "simulate_power_nll", "statuses_for_arm", "target_status", "verdict",
]

# Seed-replicate analysis fixtures

Fixtures for `scripts/sdfbase/seedrep/`'s tests (a seed replicate: one of three batch-64 fits of an object at training seeds 1 to 3). Grades and export values are invented, decode-policy rows copied from `configs/registry/decode_policy.json`, gate-liveness files measured from real objects.

`synthetic_grades_values.json`, `synthetic_export_slice.json` (`tests/test_seedrep_seed_spread.py`): values per evaluation, cell and seed for six cells (constitution, control, difference; LoRA adapter and ReLU steer) plus twelve export entries. The ReLU difference is the distilled object `..._psr_ce_all_minus_ctl_distilled`, not the constitution-minus-control composition the paper reports.

`decode_policy_vendor_rows.json` (`tests/test_bon_vendor_preset.py`): three vendor decode-policy rows (Qwen3.5-4B's model-card sampler: temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5), including best-of-N's `qwen35_4b_it|bon|off|vendor_preset@v1`.

`published_qwen_derived_gate_liveness.json`, `distilled_eb64_s1_gate_liveness.json` (`tests/test_seedrep_derived_admission.py`, `derived_admission.py`): 32 per-layer gate-liveness fractions each, the share of probe positions where a layer's ReLU gate is open (`scripts/sdfbase/gate_liveness.py`). The first is the published Qwen distilled ReLU difference; its minimum, 0.8134719699019494 at layer 9, is `derived_admission.DEFAULT_FLOOR`. The second is the seed-1 batch-64 distilled object (minimum 0.2127, layer 12). Admission flags layers below the floor and refuses only a never-open gate. Keys are string-sorted (`"0", "1", "10", ...`); a test pins that order, in which the floor was once misread as layer 10's 0.933, the first value below 1.0.

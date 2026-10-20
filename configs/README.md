# configs/ — why there are so many files, and which generator owns each family

Most files here are **generated per-wave / per-cell manifests committed for provenance**: every
launched wave's exact inputs, byte-pinned. A generator renders the family from a roster, a matrix
or a hand-listed table; its `--check` mode re-renders every enumerated output and exits 1 when a
committed file is missing or differs by one byte. So a manifest is deleted by dropping its roster
view / matrix row and regenerating, never by removing the file — the generators are the tests.
The full inventory (counts by family × model, which committed artefact references each file) is
in the publication notes; the table below is the map.

| family | generator | `--check` |
|---|---|---|
| `capeval/manifests/petri_full22_<key>[_<view>].yaml` + `sdfbase_full22_*.yaml` wave configs | `scripts/sdfbase/make_full22_configs.py` (from `sdfbase_full22_roster*.yaml`) | `python scripts/sdfbase/make_full22_configs.py --check` (also `--roster configs/sdfbase_full22_roster_<view>.yaml --check`) |
| `capeval/manifests/petri_full22_<key>.gsm8k_*.yaml` + `sdfbase/gsm8k/*.yaml` | `scripts/sdfbase/gsm8k/make_gsm8k_overlays.py` (from `sdfbase/gsm8k/matrix*.json`, themselves `make_gsm8k_matrix.py --check`ed from `views*.yaml`) | `python scripts/sdfbase/gsm8k/make_gsm8k_overlays.py --check` (+ `--matrix <name> --check`) |
| `capeval/manifests/petri_full22_<key>.math500_*.yaml` + `sdfbase/math500/*.yaml` | `scripts/sdfbase/math500/make_math500_overlays.py` (rows from `math500/cells.py`) | `python scripts/sdfbase/math500/make_math500_overlays.py --check` |
| `capeval/manifests/petri_full22_<key>.ifeval_*.yaml` + `sdfbase/ifeval/*.yaml` | `scripts/sdfbase/ifeval/make_ifeval_overlays.py` (from `sdfbase/ifeval/matrix*.json`, `make_ifeval_matrix.py --check`ed) | `python scripts/sdfbase/ifeval/make_ifeval_overlays.py --check` |
| `sdfbase/static_padding/*.yaml` | `scripts/sdfbase/full22/static_padding/make_padding_configs.py` | `python scripts/sdfbase/full22/static_padding/make_padding_configs.py --check` (`--list` enumerates) |
| `capeval/manifests/misalign_full22_*.yaml` + `misalign_full22_*.yaml` | `scripts/sdfbase/full22/misalign/make_misalign_manifest.py` | `python scripts/sdfbase/full22/misalign/make_misalign_manifest.py --check` |
| `capeval/petri/*/PROMPTED_CONSTITUTION_FULL/BEHAVIOR.md` | `scripts/sdfbase/full22/write_prompted_variant_dirs.py` | `python scripts/sdfbase/full22/write_prompted_variant_dirs.py --si constitution_full --check` |
| `*_sampled_t*[_cap*].yaml` (sampled-decode variants) | `scripts/sdfbase/full22/sampled/make_sampled_configs.py` | `python scripts/sdfbase/full22/sampled/make_sampled_configs.py --check` |
| `alphasteer_f22_*` + `capeval/manifests/alphasteer_f22_*` | `scripts/alphasteer_f22/make_strength_configs.py`, `make_ladder_configs.py`, `screen/make_strength_sampled_configs.py` | each `--check` |
| `tinycap/bcbw_<key>_<view>.yaml` (the BCB/BFCL-200 widen cells) | `scripts/bcbw_configs.py` | `python scripts/bcbw_configs.py --check` |

Hand-written and NOT generated: the tier / task / golden / requirements files under
`capeval/`, the rosters and pins (`*_roster.yaml`, `*_pins.json`), the `views*.yaml` matrix inputs
and the `frozen/<sha>/` copies they pin, the `store_pins_*.json` producer records the registry
adapters read, the two_si harness configs (`two_si*.yaml`), under `tinycap/` the 2026-09-09 thinking-header measurement
`thinking_header_measure_2026-09-09.json` + `thinking_header_verification.md` (read by `sisteer.capeval.tinycap.contracts`), and under
`evidence/` byte-exact copies of historical sources an instrument verifies by sha256 (`arditi_driver_3fed461f.py.txt`, the Arditi
driver as it graded the battery, read by `sisteer.eval.degeneracy_union`; `.txt` so nothing collects or imports it).

## Site paths are placeholders

A config never names a host, a user or a storage root. A value that points at site storage carries one placeholder,
which the code that reads the value expands just before it opens the path (`src/sisteer/site.py` resolves them from
the site config, `reproduce/site.example/` shows one):

| placeholder | stands for |
|---|---|
| `$SHARED` | the cluster's shared storage root |
| `$HF_HOME` | the Hugging Face model cache |
| `$STATE` | the per-user state root (pool stores) |
| `$STAGE` | the grading host's staging directory |
| `$WORK` | the grading host's working-data root (sampler stores) |
| `$GRADE` | the grading host's grade root |
| `$WANDB_PROJECT` | the W&B project |

A config's identity is its placeholder form: readers hash the file's bytes, never the expanded values. Output and data
paths of the oldest configs (`out/...`, `data/...`) are relative to the checkout a job runs from.

## Regenerating every family

`scripts/release/regen_configs.sh` runs every generator above in the order their inputs require, then moves the in-tree
sha pins with the bytes (`scripts/release/repin.py`), and repeats until a round changes nothing. The order matters:

1. the full22 wave configs and manifests (the main roster, then the certification and probe rosters)
2. the sampled-decode variants (both rosters)
3. the misalignment manifests
4. the AlphaSteer ladder serving configs, before the strength configs that derive from them
5. the strength configs (both lanes) and their sampled variants
6. the BCB / BFCL widen cells and the front door's cells, one pass per front-door view
7. the GSM8K rows, matrices and overlays, one pass per registered matrix
8. the MATH-500 overlays
9. the IFEval matrices and overlays (default and seed-replicate)
10. the static-padding configs
11. the seed-replicate Arditi renders
12. the prompted-constitution Petri behaviour files and the fit configs

`scripts/release/regen_configs.sh --check` runs every generator's `--check` and `repin.py --check`, and exits 1 on any
drift. `--list` prints the invocations in order.

## Rewritten files and SCRUB_MAP.tsv

Some configs here were rewritten for the public release by declared rules: storage paths became the placeholders
above, site labels became neutral names (`cluster_a`, `cluster_b`, the grading host), and comment and provenance text
lost references to the private process that produced it. The rules themselves are kept outside the repository, since
their left-hand sides spell the text they remove. `configs/SCRUB_MAP.tsv` lists every rewritten file whose earlier
sha256 a record captured (a run's or a judgment's config, a frozen copy, a harness manifest, a registry seed): its
sha256 before, its sha256 now, and the sha256 of the rule set it was derived under. A verifier that finds a recorded
sha that no longer matches the file accepts the file only when a row certifies that pair, and reports it as
`verified_via_map`. Every row was re-derived byte for byte from the earlier file by applying only the declared rules
before it was written.

The frozen copies under `sdfbase/*/frozen/` and `seedrep/source/` were rewritten by the same rules, and the shas their
views, matrices and source manifest declare were moved with them. Historical records (`serving_config_sha256`,
`standup_config_source_sha`, `campaign_parent_object_sha256`, `source_sha256`) keep the sha they recorded.

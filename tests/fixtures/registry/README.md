# Registry test fixtures

Static inputs for the results registry's tests (`src/sisteer/registry/`, [`docs/registry/`](../../../docs/registry/README.md)). Data fixtures are synthetic (invented ids, hostname `TESTHOST`, placeholder shas; no real transcript, prompt or store path). Two files are copies of repository scripts at the named commit, sha256-pinned by tests; do not edit them. Each runs the commit's code with its comments reworded, and its test checks that code against the commit: the grader copy, whose docstrings and one provenance string are reworded too, by syntax tree, and the replica dispatcher line for line with its plain comment lines set aside and the plain-text tails of two log messages masked.

| File | Content (readers in `tests/`) |
|---|---|
| `sample_records.jsonl` | 25 records (dimension seeds, one run's lifecycle, three item rows) from `sisteer.registry.synth`, every line valid (`test_registry_fixtures.py`) |
| `known_bad_vocabulary.txt` | Text the vocabulary check must flag: [`docs/registry/vocabulary.md`](../../../docs/registry/vocabulary.md)'s banned terms as words and identifier segments (`test_registry_fixtures.py`, `test_registry_vocabulary.py`) |
| `sampling_literals_before.json`, `gen_sampling_literals_before.py` | Per-entry-point sampling literals `configs/registry/decode_policy.json` must reproduce, written by `gen_sampling_literals_before.py` (`test_registry_decode_policy_entry_points.py`) |
| `petri_target_decode_policy.json` | A decode-policy export (10 policies, two models) for the Petri target's `$SISTEER_DECODE_POLICY` (`test_petri_target_decode.py`) |
| `paper_load_columns_f33dca5d.json` | The paper-frame column list at commit f33dca5d, with its loader's sha256 (`test_registry_paper_catchup.py`, `test_registry_multi_source.py`, `test_registry_paper_views_degeneration.py`) |
| `replica_sbatch_pre_c9_bdd22205c.sh` | `scripts/registry/landing/replica.sbatch`'s code at bdd22205c, before its `REPLICA_TREE` argument, its comments reworded (`test_registry_replica.py`, `test_registry_landing_chain.py`) |
| `grade_static_sampled_pre_change_e83725d9.py` | `scripts/sdfbase/full22/sampled/grade_static_sampled.py`'s code at e83725d9, its comments and docstrings reworded (`test_registry_worklist_drivers.py`) |
| `replica_init_layout_before_aliases.py` | `sisteer.registry.replica.init_layout`'s earlier behaviour, which rewrote the replica root's `config.toml` with only `[replica] log_root` (a behaviour copy, not a byte copy: its comment line is neutral) (`test_registry_migrate_site_aliases.py`) |
| `seedrep_cluster/` | Seed-replicate store shapes, [own README](seedrep_cluster/README.md) (`test_registry_adapter_seedrep_cluster.py`) |

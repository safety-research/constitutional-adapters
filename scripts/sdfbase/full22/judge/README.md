# Over-refusal judge for served controls stores

The standalone pipeline that judges a controls job's XSTest and OR-Bench-Hard generations (`scripts/served_capability_judge.py`), computes the over-refusal statistics and pushes the labels back to the cluster stores. Reproduction does not need it: `reproduce/08_grade.sh` runs the same judge through the grading daemon.

It runs on the grading host against a mirror of the append-only stores and needs an Anthropic API key (`ANTHROPIC_API_KEY` under `SISTEER_SECRETS_MODE=env`, or the mode in [`docs/secrets.md`](../../../../docs/secrets.md)), which never reaches the cluster, hence the mirror and the push-back. It also needs ssh and rsync to the cluster's login host, a checkout `.venv` and the served model's tokenizer in the local Hugging Face cache.

## Methodology

The judge is claude-sonnet-5 with the CPI rubric (`xstest_cpi`, `or_bench_cpi` in `src/sisteer/capeval/judging.py`): full, partial or no refusal per generation. Each rate carries a Wilson 95 % interval over the judged items. A judge abstention (a refusal to judge, a cut-off verdict, a transport error) is neither dropped nor imputed: the aggregate also reports the rate's bounds over all items, abstentions counted as compliance at the low end and as refusal at the high end, and a cell whose abstention share exceeds 5 % is marked `refused_abstention_ceiling`. These are the `*_refused_bounds_aggregate` rows. Verdicts are cached in SQLite by (prompt, generation, rubric, judge model, judging protocol); a re-run costs no API calls.

A generation that hit the 512-token cap is neither refusal nor compliance: `over_refusal_label_mean` scores its label alone, `over_refusal_graded` scores it 0.5 whatever its label, the at-cap share accompanies every rate, and `cap_condition.py` re-tokenizes with the pinned tokenizer for `over_refusal_terminated_only`, the rate over terminated generations. `paired_vs_none.py` pairs each cell item by item with its view's `none`: the delta with a paired Wald 95 % interval and an exact McNemar test.

## Running a pass

`<login>` is the cluster's login alias and `<remote>` a controls root (the paper's: `<shared root>/exp/sdfbase_full22_native/sampled/controls_cap<cap>`, targets `<tag>_t<T>`).

```bash
# from the repository root
export PYTHONPATH=src:scripts HF_HUB_OFFLINE=1 F22_LOGIN_HOST=<login> F22_CONTROLS_REMOTE=<remote>
F22_GRADE_ROOT=$(python3 -m sisteer.site expand '$GRADE/sdfbase_full22'); export F22_GRADE_ROOT=${F22_GRADE_ROOT/#\~/$HOME}   # empty, with the refusal printed, when the site sets no grade root
mkdir -p "${F22_GRADE_ROOT:?set SISTEER_GRADE_DIR}"/{controls,logs,cache}; J=scripts/sdfbase/full22/judge; T=<t1,t2>   # default root: pushback and the statistics read it
# 1. snapshot (the push-back's reference), then mirror
ssh $F22_LOGIN_HOST "F22_CONTROLS_REMOTE=$F22_CONTROLS_REMOTE python3 -" < $J/snapshot_rows_sha.py
rsync -rlpt $F22_LOGIN_HOST:$F22_CONTROLS_REMOTE/ $F22_GRADE_ROOT/controls/
# 2. bind the stores, seed a private cache, select it (CAPEVAL_JUDGE_CACHE stays unset until the export: both read it as the source)
python scripts/served_capability_judge.py --controls-root $F22_GRADE_ROOT/controls --dry-run
python $J/seed_private_cache.py --controls-root $F22_GRADE_ROOT/controls --out $F22_GRADE_ROOT/cache/judge_cache_f22.sqlite
export CAPEVAL_JUDGE_CACHE=$F22_GRADE_ROOT/cache/judge_cache_f22.sqlite
# 3. judge, detached (systemd-run --user); wait for its exit sentinel and continue only if it reads 0
STARTED=$(date -u +%FT%TZ); OUT=$($J/launch_judge.sh --targets $T); STAMP=${OUT##*stamp=}
until [ -f $F22_GRADE_ROOT/logs/EXIT-$STAMP ]; do sleep 60; done; cat $F22_GRADE_ROOT/logs/EXIT-$STAMP   # the log is logs/judge-$STAMP.log
# 4. statistics
python scripts/served_capability_judge.py --controls-root $F22_GRADE_ROOT/controls --reemit-only --run-started-at $STARTED
for s in cap_condition paired_vs_none report_table; do python $J/$s.py --root $F22_GRADE_ROOT/controls --targets $T; done
# 5. preflight, push, push the paired result, verify, merge the cache back
python $J/pushback_judged.py --check-only --targets $T
python $J/pushback_judged.py --targets $T
for t in ${T//,/ }; do rsync -t --no-owner --no-group --no-perms $F22_GRADE_ROOT/controls/$t/judge_paired_vs_none.json $F22_LOGIN_HOST:$F22_CONTROLS_REMOTE/$t/; done
ssh $F22_LOGIN_HOST "F22_CONTROLS_REMOTE=$F22_CONTROLS_REMOTE python3 -" < $J/verify_remote_stores.py
python $J/merge_cache_back.py --private $CAPEVAL_JUDGE_CACHE
```

Results stay in the stores: label rows and `*_refused_bounds_aggregate` rows appended to `rows.jsonl`, plus `judge_summary.json`, `judge_items.jsonl` (keys, labels, abstention reasons; no text) and `<target>/judge_paired_vs_none.json`. The push uploads the first three, refusing a store changed since the snapshot; the rsync line carries the paired file. `pushback_judged.py --check-only` previews the push: proceed only with no `SKIP` or `REFUSE` line and the M of its `N of M cells would push` equal to the number of stores you judged, nonzero. `finish_lane.sh` runs steps 3 to 5 unattended under the configured secrets mode (`op`, `env` or `wif`; a detached unit inherits nothing, so the `systemd-run` line that starts it carries the mode -- `$(secrets_unit_flags)` from `reproduce/lib/secrets.sh`); its readiness probe is the vault's mode-aware client and stops at the first credential the API rejects instead of waiting out `F22_GIVE_UP_S`; its header gives the positional arguments.

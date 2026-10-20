# Compute-replan fixtures

Fit-memory probe records (`scripts/sdfbase/memprobe_fit.py`) for `tests/test_replan_compute.py`, which pins `reproduce/tools/replan_compute.py`, the GPU-hour replanner run before the GPU stages. Measured on one H200 with a mixture-of-experts base model, then relabelled (`example-org/example-moe-base`, `TESTHOST`): real numbers, invented identifiers.

One record per arm (`lora_r4_ce`, rank-4 LoRA; `psr_ce_all`, ReLU steer): a micro-batch ladder 1, 2, 4 at 8,192 tokens, bf16, gradient checkpointing, three steps per fitting rung. LoRA: 1 and 2 fit (13.1 s/step, 1,251 tok/s at 2), 4 out of memory. ReLU: 1 fits (49 s/step, 167 tok/s), 2 out of memory before a step, 4 not attempted after a narrower rung failed. `fold_memprobe.json` is stage 02's fold, `{fit_key: {arm: largest fitting micro-batch}}`.

GPU-hours come from the widest fitting rung's tok/s; the fold is checked against each record's `verdict`. A dry run from the repository root (`--model example_moe` resolves to fit key `example_moe_base`) prints the plan (per arm: micro-batch, accumulation, tok/s, GPU-hours) and writes nothing:

```bash
PYTHONPATH=src python reproduce/tools/replan_compute.py --model example_moe \
  --memprobe tests/fixtures/replan/memprobe.psr_ce_all.json \
  --memprobe tests/fixtures/replan/memprobe.lora_r4_ce.json \
  --fold tests/fixtures/replan/fold_memprobe.json --dry-run
```

Real records: `reproduce/02_prepare_model.sh --model <key> --step memprobe` submits one probe job per arm (one GPU, 90 minutes); run again, it folds each into `fold_memprobe.<arm>.json` and the fits config.

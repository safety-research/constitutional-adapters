# Data-parallel parity smoke, recorded job 804970

`w1/` (one GPU) and `w2/` (two GPUs, `torchrun`): the two measured legs of a real Slurm job, each `result.json` (the per-step history) and `params.pt` (final adapter weights); hostname, user, paths and base-model identity are placeholders. `tests/test_ddp_parity_smoke.py` runs `scripts/sdfbase/ddp_parity_smoke.py compare` on them (no GPU) and expects `PASS`.

Both fit a rank-4 LoRA on `o_proj` (`lora_r4_ce`) by document cross-entropy on H200s: micro-batch 2, accumulation 4 (2 per rank on two GPUs), 20 steps on the first 160 rows of `constv2-mix-10m`, 8,192 tokens, bf16, gradient checkpointing, lr 1e-5, seed 0.

`PASS` needs every gate: per-step loss within rtol 1e-2; final parameters within 5e-3 times the reference's largest absolute weight; equal token, learning-rate and step-count histories; step-0 gradient norm within rtol 1e-2, median per-step difference within 5e-2; the wide leg's steady step at least 1.6 times faster.

To record a fresh pair, from a clean checkout's root (`MODEL_KEY` names `configs/sdfbase_<key>_fits.yaml`):

```bash
MODEL_KEY=qwen35_4b_base
sbatch -J "ddpsmoke-${MODEL_KEY}-lora_r4_ce" --comment="sha:$(git rev-parse HEAD)" \
  scripts/sbatch/ddp_parity_smoke.sbatch "$MODEL_KEY" lora_r4_ce constv2-mix-10m
```

It needs the pool (`reproduce/02_prepare_model.sh --step pools`), model snapshot (`reproduce/00_setup.sh --what checkpoints`) and fit environment (`scripts/sbatch/sdfbase_fit_env.sbatch`) on shared storage, and writes `parity.json` and `parity.md` beside the legs.

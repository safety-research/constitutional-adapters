"""``scripts/sdfbase/memprobe_fit.py`` — the FD 23 fit-memory probe — on CPU
with a tiny llama snapshot and the ``--fake-cuda`` hook.

Covered: ``--dry-run`` prints the plan and touches nothing; the ladder loop
runs REAL optimizer steps through ``train_object_ce`` on synthetic-length rows
of exactly ``--seq-len`` tokens for the rungs that fit, a stubbed OOM at rung 4
stops the ladder (rung 4 recorded as not fitting, nothing after it
attempted), the JSON carries every documented field, the report names the
verdict, exit 0 when micro-batch 1 fits and exit 5 when it does not; both
arms build their objects through object_factory.build_object (the production
builder); the sbatch carries the audit-exempt tag, the sha pin
check, the two-root probe, hf_env.sh, the disarm-first trap and RESULT +
mirror on EXIT.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]

SBATCH = REPO / "scripts/sbatch/memprobe_fit.sbatch"


def _load():
    spec = importlib.util.spec_from_file_location(
        "_memprobe_fit", REPO / "scripts/sdfbase/memprobe_fit.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_memprobe_fit"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mp():
    return _load()


@pytest.fixture(scope="module")
def tiny_snapshot(tmp_path_factory):
    """A tiny llama + a real (tiny) tokenizer saved as one local snapshot."""
    from transformers import LlamaConfig, LlamaForCausalLM
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    d = tmp_path_factory.mktemp("tiny-llama-memprobe")
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=200, special_tokens=["<unk>", "<s>", "</s>", "<pad>"])
    tok.train_from_iterator(["the committee met on the third morning to review the harbour survey "
                             "each surveyor read a section aloud the clerk noted every objection"] * 20,
                            trainer=trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", bos_token="<s>",
                                   eos_token="</s>", pad_token="<pad>")
    cfg = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2,
                      vocab_size=len(fast), max_position_embeddings=256,
                      tie_word_embeddings=False, bos_token_id=fast.bos_token_id,
                      eos_token_id=fast.eos_token_id, pad_token_id=fast.pad_token_id)
    torch.manual_seed(0)
    LlamaForCausalLM(cfg).save_pretrained(d)
    fast.save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def cfg_path(tmp_path_factory):
    """A fits config in the generated shape (the real one for the test model,
    with a fit_selection block) — the probe reads model / revision / the arm
    blocks off it; --model-override points it at the tiny snapshot."""
    src = (REPO / "configs/sdfbase_qwen35_4b_demo_base_fits.yaml").read_text()
    p = tmp_path_factory.mktemp("cfg") / "sdfbase_test_fits.yaml"
    p.write_text(src)
    return p


def _pool(tmp_path, n=6):
    p = tmp_path / "pairs.jsonl"
    rows = [{"prompt": "system text\n\n" if i % 2 else "",
             "pos": f"document {i}: " + " ".join(f"word{j}" for j in range(40 + 10 * i)),
             "neg": ""} for i in range(n)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_dry_run_prints_the_plan_and_touches_nothing(mp, cfg_path, tmp_path, capsys):
    out = tmp_path / "out" / "memprobe.json"
    rc = mp.main(["--config", str(cfg_path), "--arm", "lora_r4_ce", "--out", str(out),
                  "--dry-run", "--ladder", "1,2", "--seq-len", "64", "--steps", "2"])
    assert rc == mp.EXIT_USAGE, "a --seq-len that is not the config's max_len is refused without --allow-seq-len-override (the fold keys the record to the config's)"
    rc = mp.main(["--config", str(cfg_path), "--arm", "lora_r4_ce", "--out", str(out),
                  "--dry-run", "--ladder", "1,2", "--seq-len", "64", "--allow-seq-len-override", "--steps", "2"])
    assert rc == 0
    plan = json.loads(capsys.readouterr().out.split("[memprobe] plan: ", 1)[1].splitlines()[0])
    assert plan["model"] == "Qwen/Qwen3.5-4B-Base"
    assert plan["revision"] == "1001bb4d826a52d1f399e183466143f4da7b741b"
    assert plan["ladder"] == [1, 2] and plan["seq_len"] == 64 and plan["steps"] == 2
    assert plan["gradient_checkpointing"] is True and plan["dtype"] == "bfloat16", "the fit's runtime, from the config by the trainer's rule (memprobe-fix-3)"
    # a probe under another runtime is an ad-hoc probe: refused without the acknowledgement (the fold would read it STALE)
    assert mp.main(["--config", str(cfg_path), "--arm", "lora_r4_ce", "--out", str(out), "--dry-run", "--ladder", "1", "--steps", "1",
                    "--gradient-checkpointing", "off"]) == mp.EXIT_USAGE
    assert mp.main(["--config", str(cfg_path), "--arm", "lora_r4_ce", "--out", str(out), "--dry-run", "--ladder", "1", "--steps", "1",
                    "--dtype", "float32"]) == mp.EXIT_USAGE
    assert mp.main(["--config", str(cfg_path), "--arm", "lora_r4_ce", "--out", str(out), "--dry-run", "--ladder", "1", "--steps", "1",
                    "--gradient-checkpointing", "off", "--allow-runtime-override"]) == 0
    assert json.loads(capsys.readouterr().out.rsplit("[memprobe] plan: ", 1)[1].splitlines()[0])["gradient_checkpointing"] is False
    rc = mp.main(["--config", str(cfg_path), "--arm", "lora_r4_ce", "--out", str(out),
                  "--dry-run", "--ladder", "1,2", "--seq-len", "64", "--allow-seq-len-override", "--steps", "2"])
    assert rc == 0
    plan = json.loads(capsys.readouterr().out.split("[memprobe] plan: ", 1)[1].splitlines()[0])
    # the object kwargs come from the config's fit_selection block: the model's FOLDED hpo row (configs/sdfbase_folds.json — the
    # architecture pick) when one exists, else the paper default (rank 4 / o_proj / zero). Fold-or-default, never the yaml's own word.
    fold_row = (json.loads((REPO / "configs/sdfbase_folds.json").read_text())["hpo"].get("qwen35_4b_demo_base") or {}).get("lora_r4_ce") or {}
    # memprobe-fix-2: ONE source with the fold and the fit — the trainer-facing selection (top-level kwargs first), the explicit per_expert_rank
    # map included and recorded; the probe length is the config's max_len unless an override is acknowledged
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
    import memprobe_fold as mf
    cfg = __import__("yaml").safe_load(cfg_path.read_text())
    assert plan["object_kwargs"] == mf.config_selection(cfg, "lora_r4_ce")
    assert plan["object_kwargs"]["lora_rank"] == fold_row.get("lora_rank", 4)
    assert "per_expert_rank" in plan["object_kwargs"] and plan["object_kwargs"]["per_expert_rank"] == cfg["train_kwargs"]["lora_r4_ce"]["fit_selection"]["per_expert_rank"], \
        "the explicit map the trainer gets, recorded (null for a dense model)"
    assert plan["seq_len"] == 64, "an explicit --seq-len rides the plan (this probe acknowledges it below)"
    assert not out.exists() and not out.parent.exists()


@pytest.mark.parametrize("arm", ["lora_r4_ce", "psr_ce_all"])
def test_ladder_on_cpu_with_a_stubbed_oom_at_rung_4(mp, cfg_path, tiny_snapshot, tmp_path, arm):
    out = tmp_path / arm / "memprobe.json"
    rc = mp.main(["--config", str(cfg_path), "--arm", arm, "--out", str(out),
                  "--pairs", str(_pool(tmp_path)), "--model-override", str(tiny_snapshot),
                  "--ladder", "1,2,4", "--seq-len", "48", "--allow-seq-len-override", "--steps", "2", "--dtype", "float32", "--allow-runtime-override",
                  "--fake-cuda", "oom_at=4"])
    assert rc == 0                                   # micro-batch 1 fits
    rec = json.loads(out.read_text())
    assert rec["kind"] == "memprobe_fit" and rec["arm"] == arm and rec["fake_cuda"] is True
    for k in ("model", "seq_len", "gradient_checkpointing", "dtype", "gpu_name", "torch",
              "transformers", "experts_implementation", "rungs", "verdict", "ladder"):
        assert k in rec, k
    assert rec["model"]["model_override"] is True and rec["model"]["model_class"] == "LlamaForCausalLM"
    assert rec["seq_len"] == 48 and rec["gradient_checkpointing"] is True and rec["dtype"] == "float32"
    # recorded off the config: transformers 5.15 stamps "eager" on every config (a
    # dense llama included); a MoE run records the experts kernel path that ran
    assert rec["experts_implementation"] in (None, "eager")
    by_bs = {r["micro_batch"]: r for r in rec["rungs"]}
    assert list(by_bs) == [1, 2, 4]
    for bs in (1, 2):
        r = by_bs[bs]
        assert r["fits"] is True and r["error"] is None and r["steps_completed"] == 2
        assert r["s_per_step"] > 0 and r["tokens_per_s"] > 0
        assert r["peak_alloc_bytes"] == bs * 10**9 and r["fake_cuda"] is True
        # every row is exactly the requested width
        assert r["realized_seq_len"] == {"min": 48, "max": 48}
        assert r["trainable_params"] > 0
        assert r["object_builder"].startswith(("object_factory", "MultiLayerLoRAObject", "PSRObject"))
    r4 = by_bs[4]
    assert r4["fits"] is False and "out of memory" in r4["error"].lower() and r4["s_per_step"] is None
    v = rec["verdict"]
    assert v == {"fits_at_min_micro_batch": True, "min_micro_batch": 1,
                 "largest_fitting_micro_batch": 2, "s_per_step_at_min": by_bs[1]["s_per_step"],
                 "replan_threshold_s_per_step": 2.0,
                 "replan_required": by_bs[1]["s_per_step"] > 2.0,
                 "replan_watch": 1.0 < by_bs[1]["s_per_step"] <= 2.0, "exit_code": 0}
    report = (out.parent / "memprobe_report.md").read_text()
    assert "VERDICT: micro-batch 1 FITS" in report and "| 4 | NO |" in report
    assert "synthetic run (--fake-cuda)" in report
    for r in rec["rungs"]:
        assert f"| {r['micro_batch']} |" in report


def test_oom_at_rung_1_exits_5_and_reports_it(mp, cfg_path, tiny_snapshot, tmp_path):
    out = tmp_path / "memprobe.json"
    rc = mp.main(["--config", str(cfg_path), "--arm", "lora_r4_ce", "--out", str(out),
                  "--model-override", str(tiny_snapshot), "--ladder", "1,2", "--seq-len", "32", "--allow-seq-len-override",
                  "--steps", "1", "--dtype", "float32", "--allow-runtime-override", "--fake-cuda", "oom_at=1"])
    assert rc == 5
    rec = json.loads(out.read_text())
    assert rec["verdict"]["fits_at_min_micro_batch"] is False
    assert rec["verdict"]["largest_fitting_micro_batch"] is None and rec["verdict"]["exit_code"] == 5
    assert rec["rungs"][0]["fits"] is False
    assert rec["rungs"][1]["fits"] is None and "not attempted" in rec["rungs"][1]["error"]
    assert rec["examples_source"] == "synthetic (no --pairs)"
    report = (out.parent / "memprobe_report.md").read_text()
    assert "DOES NOT FIT" in report and "the user is told" in report


def test_usage_refusals(mp, cfg_path, tmp_path):
    out = tmp_path / "m.json"
    base = ["--config", str(cfg_path), "--arm", "lora_r4_ce", "--out", str(out)]
    assert mp.main(base + ["--ladder", "4,2", "--dry-run"]) == 2
    assert mp.main(base + ["--ladder", "0", "--dry-run"]) == 2
    assert mp.main(base + ["--fake-cuda", "bogus=1", "--dry-run"]) == 2
    # no CUDA and no --fake-cuda: the probe refuses to call CPU numbers a verdict
    if not torch.cuda.is_available():
        assert mp.main(base + ["--seq-len", "32", "--model-override", "unused"]) == 2
    with pytest.raises(mp.ProbeError):
        mp.parse_ladder("1,1")
    assert mp.parse_fake_cuda(None) is None
    assert mp.parse_fake_cuda("oom_at=2,4") == {"oom_rungs": {2, 4}}
    assert mp.is_oom(RuntimeError("CUDA out of memory. Tried to allocate 1 GiB"))
    assert not mp.is_oom(RuntimeError("shape mismatch"))


def test_make_examples_hits_the_width_exactly(mp, tiny_snapshot):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tiny_snapshot))
    texts, source = mp.load_pool_texts(None, 3)
    assert len(texts) == 3 and source.startswith("synthetic")
    exs, widths = mp.make_examples(tok, texts, 40)
    assert widths == [40, 40, 40]
    for ex in exs:
        assert ex.prompt_with_si == "" and ex.prompt_without == "" and ex.completion


def test_sbatch_carries_the_cluster_contract():
    text = SBATCH.read_text()
    assert "# audit-exempt: out-of-lockfile-venv" in text
    assert "--comment=sha:" in text and 'case "$COMMENT" in' in text and "sha:*) PIN=" in text
    assert '[ "$PIN" = "$HEAD_SHA" ]' in text
    # the shared root is the job's site value, never probed for; hf_env from the shared root
    assert 'SISTEER_SHARED_ROOT' in text and not re.search(r'\[ -d "/[^"$]+/\$USER" \]', text)
    assert 'source "$SHARED/scripts/hf_env.sh"' in text
    # the disarm-first trap, RESULT + mirror on EXIT, 1 GPU, 90 minutes, no requeue, no --export
    assert "cleanup() { trap '' SIGTERM; kill -TERM -$$" in text
    assert "trap cleanup SIGTERM SIGINT SIGQUIT" in text and "trap 'finish' EXIT" in text
    assert 'mirror-figs.sh" "$FIGS_DIR"' in text and "RESULT" in text
    assert "#SBATCH --gres=gpu:1" in text and "#SBATCH --time=01:30:00" in text
    assert "#SBATCH --no-requeue" in text and "--mem=0" not in text
    # never an --export= on a command (hard rule 1); the header's prose may name the rule
    assert re.search(r"^[^#\n]*--export=", text, re.M) is None
    assert "CUDA_VISIBLE_DEVICES=" not in text
    # the runtime cu128 assert precedes the workload; the workload is srun-wrapped
    assert text.index("2.11.0+cu128") < text.index("srun --ntasks=1")
    assert "--gradient-checkpointing" not in text, "memprobe-fix-3: the probe takes the fit's checkpointing + dtype from the config, nothing forced"
    # the verdict exit codes ride the RESULT line
    assert "5) state=DOES_NOT_FIT" in text
    # no personal identifiers
    sys.path.insert(0, str(REPO / "tests"))
    from front_door_testlib import assert_no_tokens   # the ONE token table the publication scan reads, never spelled here
    assert_no_tokens(text, "the probe's output")


def test_a_contradictory_fit_selection_never_reaches_the_probe_and_the_fold_sees_the_trainers_shape(mp, cfg_path, tmp_path, capsys):
    """The producer read fit_selection first while the fold read the top-level block — a loop (probe the old shape, fold
    exits STALE, archive, repeat). Now both read memprobe_fold.config_selection: a config whose fit_selection contradicts the top-level
    trainer kwargs probes the TOP-LEVEL shape, records it, and a record built from that plan is not stale for the fold."""
    import yaml
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
    import memprobe_fold as mf
    cfg = yaml.safe_load(cfg_path.read_text()); tk = cfg["train_kwargs"]["lora_r4_ce"]
    # the top-level trainer kwargs state the rank (4); a fit_selection block that contradicts it (rank 16) must never reach the probe
    tk["fit_selection"] = {**tk["fit_selection"], "lora_rank": 16, "lora_modules": ["o_proj"], "per_expert_rank": None, "init_scheme": "zero"}
    contra = tmp_path / "contra.yaml"; contra.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "out" / "memprobe.json"
    assert mp.main(["--config", str(contra), "--arm", "lora_r4_ce", "--out", str(out), "--dry-run", "--ladder", "1", "--steps", "1"]) == 0
    plan = json.loads(capsys.readouterr().out.split("[memprobe] plan: ", 1)[1].splitlines()[0])
    want = mf.config_selection(cfg, "lora_r4_ce")
    assert want["lora_rank"] == tk["lora_rank"] == 4 != tk["fit_selection"]["lora_rank"]          # the trainer's precedence: the top-level rank, never the block's
    assert plan["object_kwargs"] == want == {"lora_rank": 4, "lora_modules": ["o_proj"], "per_expert_rank": None, "init_scheme": "zero"}
    assert plan["seq_len"] == cfg["max_len"]
    rec = {"arm": "lora_r4_ce", "seq_len": plan["seq_len"], "object_kwargs": plan["object_kwargs"], "gpu_name": "NVIDIA H200", "gpu_total_bytes": 10,
           "gradient_checkpointing": plan["gradient_checkpointing"], "dtype": plan["dtype"],      # the producer records the runtime it probed under
           "model": {"model": cfg["model"], "revision": cfg["model_revision"]}, "rungs": [], "verdict": {"exit_code": 0}}
    assert mf.staleness(rec, cfg, "lora_r4_ce") == []

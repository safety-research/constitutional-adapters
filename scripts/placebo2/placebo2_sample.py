#!/usr/bin/env python3
"""Placebo-control regeneration for the si_irrelevant n=1 problem.

The labelled store's striking result — the irrelevant metre-history SI defends
the pasttense family while both purpose-built safety SIs do not, yet HARMS
amplegcg_plus / encoding / aim — rests on ONE ~6.4k-token document.
This script generates the rows that decide whether that is a property of long
off-topic system prefixes in general or of that document in particular.

Five arms over the SAME (goal_id, variant_name, sample_idx, trial) instances
the store already holds for pasttense / amplegcg_plus (the
undefended `none` rows at candidate_idx == 0, extracted to --prompts):

  sr_greedy   safety_refusal, greedy      — byte-fidelity control: the store's
                                            safety_refusal rows were greedy, so
                                            regenerated responses on this stack
                                            can be compared byte-for-byte.
  si_greedy   si_irrelevant, greedy       — the original placebo under greedy
                                            decoding (the store never had this:
                                            its candidate-0 rows were T=1.0
                                            draws, k=8 forcing sampling on).
  si_t1       si_irrelevant, T=1.0 seed 0 — same-stack rerun of the store's
                                            actual candidate-0 decode policy.
  p2_greedy   si_irrelevant2, greedy      — the SECOND placebo (coffee
                                            cultivation and processing,
                                            scripts/placebo2/irrelevant_si2.md).
  p2_t1       si_irrelevant2, T=1.0 seed 0

The document x decode-policy 2x2 (si_greedy/si_t1/p2_greedy/p2_t1) is the
design: comparing p2 to si on the SAME stack under BOTH decode policies
decouples the document's content from the decode-policy confound the store
carries (none/safety greedy vs si_irrelevant sampled).

Engine parameters replicate the store run exactly (verified against the run3h
shard logs + pod freeze): vLLM 0.26.0+cu129, NousResearch/Meta-Llama-3.1-8B
-Instruct @ d10aef79, bfloat16, max_model_len 16384, prefix caching, seed 0,
max_new_tokens 512. Sampling params replicate sisteer.sampling.engine
._sampling_params: greedy -> T=0.0/top_p=1.0/seed None; T=1.0 draw ->
seed = engine_seed + 1000 * candidate_idx = 0 for candidate 0.

Rows are written in the store schema via engine._row (si_condition
"prompted", candidate_idx 0), to --out, which must live OUTSIDE the store.
Resume-safe via engine.load_done_keys. Blocks are grouped per SI text so the
6.3k-token prefix stays in the LRU prefix cache (same-SI arms consecutive).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


@dataclass(frozen=True)
class _Prompt:
    """The attribute surface engine._row / engine.cell_key read."""

    family: str
    goal_id: str
    sample_idx: int
    trial: int
    text: str
    category: object = None
    source: object = None
    side: str = "harmful"
    variant_name: object = None

    @property
    def variant_key(self) -> str:
        return f"{int(self.sample_idx)}:{int(self.trial)}"


def _load_prompts(path: Path) -> list[_Prompt]:
    out = []
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            out.append(_Prompt(
                family=r["family"], goal_id=r["goal_id"],
                sample_idx=int(r["sample_idx"]), trial=int(r["trial"]),
                text=r["text"], category=r.get("category"),
                source=r.get("source"), side=r.get("side", "harmful"),
                variant_name=r.get("variant_name")))
    # Deterministic order (the extraction already sorts; sort again defensively
    # so a shuffled prompts file cannot change block composition).
    out.sort(key=lambda p: (p.family, p.goal_id, p.sample_idx, p.trial))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--si-doc", required=True,
                    help="path to the second irrelevant document (markdown)")
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--revision",
                    default="d10aef7999a2b5ba950ab3974312feeedbfe0b77")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--arms", default="sr_greedy,si_greedy,si_t1,p2_greedy,p2_t1",
                    help="comma list of arms to run (subset of the five); the "
                         "five-family scale-out needs only si_greedy,p2_greedy "
                         "— the fidelity and T1 arms were already validated on "
                         "the first three families")
    args = ap.parse_args()

    from sisteer.data.instructions import BY_ID, SAFETY_SI
    from sisteer.sampling import engine as eng

    p2_text = Path(args.si_doc).read_text().rstrip("\n")
    si_text = BY_ID["si_irrelevant"].text

    # (arm, si_id written into rows, system text, decode policy). Same-text
    # arms are consecutive so the prefix cache is hit, not evicted.
    arms = [
        ("sr_greedy", "safety_refusal_regen", SAFETY_SI.text, "greedy"),
        ("si_greedy", "si_irrelevant_greedy", si_text, "greedy"),
        ("si_t1", "si_irrelevant_t1", si_text, "t1"),
        ("p2_greedy", "si_irrelevant2_greedy", p2_text, "greedy"),
        ("p2_t1", "si_irrelevant2_t1", p2_text, "t1"),
    ]

    want = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(want) - {a[0] for a in arms}
    if unknown:
        raise SystemExit(f"unknown arms {sorted(unknown)}")
    arms = [a for a in arms if a[0] in want]

    prompts = _load_prompts(Path(args.prompts))
    print(f"[placebo2] {len(prompts)} prompt instances; "
          f"{len(arms)} arms -> {len(arms) * len(prompts)} cells", flush=True)

    sampler = eng.VllmSampler(
        args.model, revision=args.revision, seed=args.seed,
        max_new_tokens=args.max_new_tokens)
    from vllm import SamplingParams

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = eng.load_done_keys(out_path)

    t0 = time.time()
    import os
    for arm, si_id, text, policy in arms:
        todo = [p for p in prompts if eng.cell_key(p, si_id, 0) not in done]
        if not todo:
            print(f"[placebo2] {arm}: complete (resume)", flush=True)
            continue
        # Replicates engine._sampling_params: greedy is k==1's policy; "t1" is
        # the k>1 candidate-0 policy (temperature 1.0, seed = seed + 1000*0).
        sp = (SamplingParams(temperature=0.0, top_p=1.0,
                             max_tokens=args.max_new_tokens, seed=None)
              if policy == "greedy" else
              SamplingParams(temperature=1.0, top_p=1.0,
                             max_tokens=args.max_new_tokens, seed=args.seed))
        rendered = [eng.build_chat(sampler.tok, text, p.text) for p in todo]
        outs = sampler.llm.generate(rendered, sp)
        with out_path.open("a") as f:
            for p, o in zip(todo, outs):
                f.write(json.dumps(
                    eng._row(p, si_id, 0, o.outputs[0].text)) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(f"[placebo2] {arm}: +{len(todo)} rows "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)

    print(f"[placebo2] done in {time.time() - t0:.0f}s -> {out_path}",
          flush=True)


if __name__ == "__main__":
    main()

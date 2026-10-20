"""Sub-env worker for AmpleGCG-plus (runs ONLY under the transformers<4.56
interpreter — see ``amplegcg.py`` for provisioning).

Contract: reads one JSON object on stdin ``{goal_text, seed, n_suffixes,
model}`` and writes one JSON object on stdout ``{suffixes: [str, ...],
n_generator_forwards: int}``. All heavy imports are inside ``main`` so the module
is import-safe in the main env; it is invoked only as
``python -m sisteer.eval.attacks._amplegcg_worker`` under the sub-env.

The generator is an AmpleGCG(-plus) checkpoint (a Llama fine-tune that emits
adversarial suffixes for a given request). We format the request the way the
released model expects (``"### Query:<q> ### Prompt:"``) and decode ``n_suffixes``
continuations with group-beam search (the paper's sampling), returning each
generated suffix. Failures raise loudly (non-zero exit) so the caller never
mistakes a broken generator for an empty result.
"""

from __future__ import annotations

import json
import os
import sys


def main() -> int:
    payload = json.loads(sys.stdin.read())
    goal_text = payload["goal_text"]
    n_suffixes = int(payload.get("n_suffixes", 8))
    model_id = payload.get("model")
    seed = int(payload.get("seed", 0))
    if not model_id:
        raise SystemExit("amplegcg worker: no 'model' checkpoint in payload")

    import torch  # noqa: F401  (in-sub-env)
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              set_seed)

    set_seed(seed)
    # cuDNN-SDPA footgun (see sisteer.models.configure_fast_sdpa) — inlined
    # because this worker runs in a separate transformers<4.56 sub-env where
    # importing sisteer is not guaranteed.
    if os.environ.get("SISTEER_KEEP_CUDNN_SDP") != "1":
        try:
            torch.backends.cuda.enable_cudnn_sdp(False)
        except Exception:
            pass
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # The released AmpleGCG format: the harmful query framed for suffix decode.
    prompt = f"### Query:{goal_text} ### Prompt:"
    enc = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            num_beams=n_suffixes,
            num_beam_groups=n_suffixes,
            diversity_penalty=1.0,
            num_return_sequences=n_suffixes,
            max_new_tokens=64,
            do_sample=False,
        )
    gen = out[:, enc["input_ids"].shape[1]:]
    suffixes = [tok.decode(g, skip_special_tokens=True).strip() for g in gen]
    # generator forwards ~ beams * decode steps (an accounting estimate, kept
    # SEPARATE from the target's n_target_forwards).
    n_generator_forwards = int(n_suffixes) * int(gen.shape[1])

    json.dump({"suffixes": suffixes,
               "n_generator_forwards": n_generator_forwards}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

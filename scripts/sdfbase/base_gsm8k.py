#!/usr/bin/env python
"""Base-model capability leg, GSM8K: PLAIN-TEXT 8-shot chain-of-thought,
greedy, exact match — the sibling of ``base_mmlu.py`` (sdf-base-steering
round 2; user directive 2026-09-03: GSM8K at 1x AND 2x dose for every
stable cell, paired within label).

Why a second script rather than the battery's GSM8K control: the battery's
``s2parity/gsm8k_0shot`` (capability_ext.gsm8k_accuracy, cell_controls_run)
renders through ``build_chat_prompt`` with a 0-shot "#### <answer>"
instruction — an INSTRUCT-model protocol. A base checkpoint given that
prompt has no answer format to fall into, so its accuracy measures prompt
compliance, not arithmetic. A base model needs a few-shot prompt whose
exemplars SHOW the format, and it needs plain text (the campaign's data
path never touches a chat template). This script keeps the battery's
load-bearing pieces BY IMPORT and changes only the render:

  * dataset: the battery's ``madrylab/gsm8k-platinum`` test split (the
    cleaner-label GSM8K the campaign's every other GSM8K number is on),
    draw = ``shuffle(seed).select(n)`` — base_mmlu's convention; n=500,
    seed 0 by default.
  * gold: ``capability_ext.parse_gsm8k_gold`` (the ``#### N`` field).
  * decoding: ``model.generate`` greedy (``do_sample=False``), LEFT-padded
    batches (``generate.assert_left_padded`` tripwire), ``encode_kwargs``'s
    double-BOS guard, the object entered PER CHUNK via ``obj.apply(model,
    scale=--scale)`` — exactly ``generate_with_object``'s discipline (the
    position-tracker lesson), re-stated here because the stop strings and
    the per-row token accounting below need the raw ids.
  * correctness: ``abs(pred - gold) < 1e-6`` — the battery's rule.

Render (``RENDER = plain_text_8shot_cot``): the eight Wei et al. (2022,
"Chain-of-Thought Prompting", Table 20) GSM8K exemplars — the canonical
base-model GSM8K prompt (lm-eval-harness ``gsm8k_cot`` uses the same
eight) — in the sibling script's ``Question: / Answer:`` convention, each
answer a short rationale ending ``The answer is N.``; then the scored
question and a bare ``Answer:``. The exemplars are module constants (their
sha256 rides in the run identity as ``exemplars_sha256``), so the prompt is
fully determined by this module + the scored question — no second dataset
load, no floating few-shot draw.

Generation budget: ``--max-new-tokens 512`` — the battery's own GSM8K
control budget, so "ran to the cap" means the same thing on both legs (the
exemplar rationales run 30-70 tokens and the smoke's mean was ~100-150,
but a verbose chat checkpoint (GLM) legitimately used >384 on ~2% of rows
even bare, which a 384 cap would have booked as non-terminating) with
``stop_strings =
("\\nQuestion:",)``: a base model does not emit EOS after its answer, it
starts the NEXT few-shot block, and that tag is the exact event the answer
is finished on. A row is ``terminated`` when the stop string fired or the
model emitted EOS; a row that ran to the cap without either is
NON-TERMINATING (``hit_cap`` and not stopped) — the degeneration count the
2x pass exists to surface. ``empty`` = an answer block that is whitespace.

Extraction (``extract_answer``): the answer BLOCK is the generation cut at
the first stop string. Rule 1, ``answer_is``: the FIRST ``The answer is N``
in the block — the exemplars establish exactly one, closing the answer, so
the first is THIS question's; a later one (a self-contradiction, or a
hallucinated follow-on Q/A the stop string missed) is not. Rule 2,
``hash_last``: no ``The answer is`` -> the battery's own extractor on the
block (``extract_gsm8k_answer``: the LAST ``#### N``); rule 3,
``last_number``: its last-number fallback; ``none``: extraction failure
(counted as wrong AND reported). ``$`` and thousands separators are
stripped, a trailing period dropped, negatives kept — ``_parse_number``.
Every row records which rule fired, so the aggregate's dependence on the
fallbacks is auditable.

Cells, identity gate, publication: identical to base_mmlu (imported):
``none`` + ``<data-cell>__<arm>`` from ``--defenses-root/cells_all/`` or
``--object CELL=PATH`` (derived objects, ``--parents-root`` REQUIRED);
every artifact passes ``assert_object_identity`` BEFORE the model loads;
``--run-label`` is REQUIRED and rides in dir + filename
(``<out-dir>/<label>/gsm8k_<label>_<cell>.{json,per_question.jsonl}``);
rows first then aggregate, tmp+rename; the summary (MERGED into an
existing same-identity summary of the label) only after the full requested
set scored. Per-question rows: qidx, qid (sha256-16 of the question text),
gold, extracted, rule, correct, terminated, hit_cap, n_new_tokens, empty,
response (the answer block) — a paired McNemar table between any two cells
or labels is post-hoc arithmetic on two rows files.

Decode runtime: generate() must run WITH the model's KV/SSM cache.
On Nemotron-3-Nano-4B under transformers 5.14.1 the loader used to pin the
cache OFF (DynamicCache could not map the hybrid's "mlp" blocks), so every
decode step recomputed the whole ~1.5k-token prompt plus every generated
token: ~1.75 s/step at batch 128 and the SAME ~2 h per 500-question cell at
batch 32 — a compute-bound full recompute, ~9x GLM's per-cell time on a
model that should decode faster. ``models.prepare_kv_cache`` now registers
the placeholder cache layer 5.15.0 ships instead of pinning; this script
REFUSES a cache-free decode unless ``--allow-cache-free`` (diagnosis), logs
a pace line per chunk, and records ``decode_runtime`` (cache on/off, Mamba
kernel path, library versions) on each aggregate as ``runtime`` — outside
the identity, like the attention kernel choice. The two implementations
are NOT assumed interchangeable: the cached and the cache-free decode take
different bf16 kernel paths and their greedy tokens diverge on some rows
(23/128 within 32 tokens on the H200 diagnosis), so a label that already
holds cells published before this fix (no ``runtime`` block — decoded
cache-free on such a model) may be topped up with cached cells ONLY under
``--decode-parity <attestation>``: the record ``gsm8k_decode_parity.py``
writes from the label's published ``none`` vs its cached re-score under a
scratch label, accepted iff every row's OUTCOMES (extracted answer, correct
flag, stop reason) agree, its identity matches this run, and its digest is
this label's legacy file (``require_decode_parity``; the attestation is
then stamped on each new aggregate). Otherwise re-score the legacy cells
in the same invocation, so one label holds one decode implementation.
``scripts/sdfbase/nemo_decode_diag.py`` is the measurement.

Run inside the sdfbase-fits Modal app (--mode gsm8k --run-label r2 [--scale
2.0]) or anywhere with the repo env:

  PYTHONPATH=src python scripts/sdfbase/base_gsm8k.py \
      --config configs/sdfbase_<key>_fits.yaml \
      --cells none,constv2-mix-1m__psr_ce_all \
      --defenses-root /vol/out/<key> --out-dir /vol/out/<key>/gsm8k \
      --run-label r2_x2 --scale 2.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import base_mmlu as bm  # noqa: E402  (the sibling: identity gate, publication)

DATASET = "madrylab/gsm8k-platinum"
DATASET_CONFIG = "main"
DATASET_SPLIT = "test"
DEFAULT_SHOTS = 8
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_BATCH_SIZE = 32
STOP_STRINGS = ("\nQuestion:",)
RENDER = "plain_text_8shot_cot"
PER_QUESTION_HEADER_KIND = "gsm8k_per_question_header"
CONSUMER = "base-gsm8k"

#: Wei et al. (2022) Table 20 — the eight chain-of-thought exemplars for
#: math word problems, written by the paper's authors (none is a GSM8K item,
#: so no test problem is ever in the prompt). (question, rationale, answer).
#: CC BY 4.0; the attribution is in THIRD_PARTY_NOTICES.
EXEMPLARS: tuple[tuple[str, str, str], ...] = (
    ("There are 15 trees in the grove. Grove workers will plant trees in "
     "the grove today. After they are done, there will be 21 trees. How "
     "many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some "
     "more were planted. So there must have been 21 - 15 = 6.",
     "6"),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how "
     "many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.",
     "5"),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how "
     "many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total "
     "they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.",
     "39"),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has "
     "12 lollipops. How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some "
     "to Denny. So he gave Denny 20 - 12 = 8.",
     "8"),
    ("Shawn has five toys. For Christmas, he got two toys each from his "
     "mom and dad. How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and "
     "dad, then that is 4 more toys. 5 + 4 = 9.",
     "9"),
    ("There were nine computers in the server room. Five more computers "
     "were installed each day, from monday to thursday. How many computers "
     "are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more "
     "computers were added. So 5 * 4 = 20 computers were added. 9 + 20 "
     "is 29.",
     "29"),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
     "wednesday, he lost 2 more. How many golf balls did he have at the "
     "end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he "
     "had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf "
     "balls.",
     "33"),
    ("Olivia has $23. She bought five bagels for $3 each. How much money "
     "does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = "
     "15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.",
     "8"),
)

#: Rule 1: the exemplars' closing phrase. Number = optional sign, digits
#: with thousands separators, optional decimals; `$` tolerated; a trailing
#: period is outside the group.
_ANSWER_IS_RE = re.compile(
    r"[Tt]he answer is[:\s]*\$?\s*(-?[\d,]*\.?\d+)")


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------


def render_exemplar(question: str, rationale: str, answer: str) -> str:
    return (f"Question: {question.strip()}\n"
            f"Answer: {rationale.strip()} The answer is {answer.strip()}.")


def render_prompt(question: str, shots: int = DEFAULT_SHOTS) -> str:
    """``shots`` exemplars + the scored question ending ``Answer:`` — PLAIN
    TEXT ONLY, no chat template, ever (the campaign's standing rule)."""
    if not 0 <= shots <= len(EXEMPLARS):
        raise SystemExit(f"[{CONSUMER}] --shots must be in [0, "
                         f"{len(EXEMPLARS)}], got {shots}")
    parts = [render_exemplar(*ex) for ex in EXEMPLARS[:shots]]
    parts.append(f"Question: {question.strip()}\nAnswer:")
    return "\n\n".join(parts)


def exemplars_sha256(shots: int = DEFAULT_SHOTS) -> str:
    """sha256 of the rendered exemplar block — the prompt prefix every
    question shares; part of the run identity."""
    block = "\n\n".join(render_exemplar(*ex) for ex in EXEMPLARS[:shots])
    return hashlib.sha256(block.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def question_id(row: dict) -> str:
    """sha256-16 of the question text — content-derived, so two runs pair
    rows even if the draw order ever changes."""
    return hashlib.sha256(row["question"].encode("utf-8")).hexdigest()[:16]


def ordered_qid_hash(rows: list[dict]) -> str:
    return hashlib.sha256(
        "\n".join(question_id(r) for r in rows).encode()).hexdigest()


def load_rows(n: int, seed: int, revision: str | None = None
              ) -> tuple[list[dict], str]:
    """(rows, resolved dataset revision) from the gsm8k-platinum test
    split: ``shuffle(seed).select(n)`` (base_mmlu's draw convention), gold
    parsed with the battery's ``parse_gsm8k_gold``. Revision handling is
    base_mmlu.load_rows's: a pin must agree with what loaded; a load whose
    revision cannot be established refuses (an unpinned dataset cannot
    pair with anything)."""
    from datasets import load_dataset

    from sisteer.eval.capability_ext import parse_gsm8k_gold

    kw = {"revision": revision} if revision else {}
    ds = load_dataset(DATASET, DATASET_CONFIG, split=DATASET_SPLIT, **kw)
    resolved = bm.dataset_revision_of(ds)
    if resolved is None and revision is None:
        raise SystemExit(
            f"[{CONSUMER}] cannot establish the {DATASET} revision this load "
            "resolved to — pass --dataset-revision <commit sha> so the run "
            "identity is pinned; McNemar pairing across runs requires it")
    if resolved is not None and revision is not None and resolved != revision:
        raise SystemExit(
            f"[{CONSUMER}] --dataset-revision {revision} but the loaded split "
            f"resolved to {resolved} — refusing to stamp a revision the data "
            "did not come from")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    rows = [{"question": r["question"], "answer": r["answer"],
             "gold": parse_gsm8k_gold(r["answer"])} for r in ds]
    return rows, str(resolved or revision)


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def truncate_at_stop(text: str, stop_strings=STOP_STRINGS
                     ) -> tuple[str, bool]:
    """(answer block, stopped): the text up to the FIRST occurrence of any
    stop string, and whether one occurred. A block is what the model wrote
    for THIS question before moving on to the next few-shot slot."""
    cut = len(text)
    for s in stop_strings:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut], cut < len(text)


def extract_answer(block: str) -> tuple[float | None, str]:
    """(value, rule) for one answer block. Rules, in order: ``answer_is``
    (the FIRST ``The answer is N``), ``hash_last`` (the battery's LAST
    ``#### N``), ``last_number`` (the battery's last-number fallback),
    ``none`` (extraction failure)."""
    from sisteer.eval.capability_ext import (_parse_number,
                                             extract_gsm8k_answer)

    m = _ANSWER_IS_RE.search(block)
    if m is not None:
        val = _parse_number(m.group(1))
        if val is not None:
            return val, "answer_is"
    val, used_fallback = extract_gsm8k_answer(block)
    if val is None:
        return None, "none"
    return val, ("last_number" if used_fallback else "hash_last")


def is_correct(pred: float | None, gold: float) -> bool:
    return pred is not None and abs(pred - float(gold)) < 1e-6


# --------------------------------------------------------------------------
# Generation — generate_with_object's discipline, with raw-id accounting
# --------------------------------------------------------------------------


def _end_ids(model, tok) -> tuple[int, set[int]]:
    """(pad id HF appends to a finished row, the EOS id set generate()
    actually stops on). The EOS set is the model's ``generation_config``
    (a list on Gemma-style models) plus the tokenizer's eos — the same set
    HF's EosTokenCriteria reads — NOT the pad id: a distinct pad token
    (Gemma: pad 0, eos 1) that the model GENERATES mid-row does not end the
    row for HF and must not end it here;
    only the pad run HF appends AFTER a row finishes is padding."""
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad is None:
        raise SystemExit(f"[{CONSUMER}] tokenizer has neither pad nor eos id")
    ends: set[int] = set()
    gc = getattr(model, "generation_config", None)
    eos = getattr(gc, "eos_token_id", None)
    if isinstance(eos, int):
        ends.add(int(eos))
    elif eos is not None:
        ends.update(int(e) for e in eos)
    if tok.eos_token_id is not None:
        ends.add(int(tok.eos_token_id))
    if not ends:
        raise SystemExit(f"[{CONSUMER}] no EOS id on the model or tokenizer")
    return int(pad), ends


class StopRecorder:
    """A HF stopping criterion that RECORDS, per row, the step and the
    reason the row first finished — an EOS-set id, or a stop string
    completing (decided by HF's own ``StopStringCriteria``) — and returns
    those flags so ``generate()`` finishes the row on that very event.

    Termination is therefore a decision made INSIDE the generation loop,
    never reconstructed from the padded output (reading the padding back
    got it wrong both ways — a non-terminating row whose last generated
    token happened to be a pad read as early, and under pad == eos a
    stop-string row's EOS-valued padding read as EOS, so ``stop_reason``
    depended on batch composition). Rules: the FIRST event per row is the
    one recorded (a finished row's padding, EOS-valued or not, is
    ignored); when a stop string completes and an EOS id lands on the same
    step the stop string wins (the visible event beats the inferred one);
    ``n_new`` excludes the EOS token and includes the stop string's tokens
    (the answer block is cut at the stop string afterwards).

    HF's loop calls every criterion after appending each new token with
    the full ``input_ids`` (pads already substituted for finished rows),
    so ``new_len = input_ids.shape[1] - prompt_width`` is the row's
    generation step. A row never flagged here ran until HF stopped the
    whole batch — ``finish`` labels it ``cap`` at the budget, or
    ``early_unknown`` if the batch ended short of it (a criterion this
    recorder did not see; defensive, reported).
    """

    def __init__(self, tok, eos_ids: set[int], stop_strings) -> None:
        import torch
        from transformers.generation import StopStringCriteria

        self.eos = torch.tensor(sorted(int(e) for e in eos_ids))
        self.stop_crit = (StopStringCriteria(tokenizer=tok,
                                             stop_strings=list(stop_strings))
                          if stop_strings else None)
        self.prompt_width = 0
        self.reason: list[str | None] = []
        self.n_new: list[int | None] = []

    def reset(self, batch_size: int, prompt_width: int) -> "StopRecorder":
        self.prompt_width = int(prompt_width)
        self.reason = [None] * int(batch_size)
        self.n_new = [None] * int(batch_size)
        return self

    def __call__(self, input_ids, scores=None, **kwargs):
        import torch

        new_len = int(input_ids.shape[1]) - self.prompt_width
        last = input_ids[:, -1]
        is_eos = torch.isin(last, self.eos.to(last.device))
        if self.stop_crit is not None:
            is_stop = self.stop_crit(input_ids, scores).to(is_eos.device).bool()
        else:
            is_stop = torch.zeros_like(is_eos, dtype=torch.bool)
        done = is_eos | is_stop
        for i in done.nonzero(as_tuple=False).flatten().tolist():
            if self.reason[i] is not None:
                continue  # already finished: this column is HF's padding
            if bool(is_stop[i]):
                self.reason[i], self.n_new[i] = "stop_string", new_len
            else:
                self.reason[i], self.n_new[i] = "eos", new_len - 1
        return done

    def finish(self, generated_len: int, max_new_tokens: int
               ) -> list[tuple[int, str]]:
        """(n_new, reason) per row once generate() returned;
        ``generated_len`` = the new columns in the output."""
        out = []
        for r, n in zip(self.reason, self.n_new):
            if r is not None:
                out.append((int(n), r))
            elif generated_len >= int(max_new_tokens):
                out.append((int(max_new_tokens), "cap"))
            else:
                out.append((int(generated_len), "early_unknown"))
        return out


def make_stop_recorder(tok, eos_ids: set[int], stop_strings) -> StopRecorder:
    """A StopRecorder that IS a ``transformers`` StoppingCriteria (the
    subclass is built here so importing this module never needs
    transformers)."""
    from transformers.generation import StoppingCriteria

    cls = type("StopRecorderCriteria", (StopRecorder, StoppingCriteria), {})
    return cls(tok, eos_ids, stop_strings)


def mamba_kernel_path(model) -> str | None:
    """Which Mamba2 path the loaded model's SSM mixers dispatch to:
    ``"fused"`` (mamba_ssm / causal_conv1d kernels — cuda_kernels_forward),
    ``"torch"`` (the modeling module's pure-torch SSD scan, 9-14x slower per
    step), ``None`` when the architecture has no Mamba2 mixer. Two dispatch
    generations are read (the same two onboard_model_smoke.
    check_mamba_kernel_path recognises): transformers <= 5.14 latches the
    verdict in the modeling module's ``is_fast_path_available`` global at
    mixer construction; 5.15 dropped the latch for per-op hub-with-fallback
    wrappers resolved at import, so the verdict is read off the live
    closures through the loader's own reader
    (sisteer.models.resolved_mamba_kernels): "fused" iff every wrapper
    dispatches into its kernel package, else "torch"."""
    mixers = [m for m in model.modules()
              if type(m).__name__.endswith("Mamba2Mixer")]
    if not mixers:
        return None
    modeling = sys.modules.get(type(mixers[0]).__module__)
    fast = getattr(modeling, "is_fast_path_available", None)
    if fast is not None:
        return "fused" if bool(fast) else "torch"
    from sisteer.models import MAMBA_KERNEL_WRAPPERS, resolved_mamba_kernels

    got = resolved_mamba_kernels(model)
    if not got:
        return None
    fused = all(impl == MAMBA_KERNEL_WRAPPERS[name]
                or impl.startswith(MAMBA_KERNEL_WRAPPERS[name] + ".")
                for name, impl in got.items())
    return "fused" if fused else "torch"


def decode_runtime(model) -> dict:
    """The facts that decide how fast (and along which numeric path) HF
    ``generate()`` runs on this model — RECORDED per cell (aggregate
    ``runtime`` block) and printed at start, deliberately NOT part of the
    run identity: the cache and the kernel path are implementations of the
    same greedy decode, like the attention backend's kernel choice.
    ``kv_cache`` False means generate() would run cache-free (a full
    prefix recompute per token); ``main`` refuses that unless told
    otherwise (~60x the decode-step time for the same outputs)."""
    import torch
    import transformers

    from sisteer.models import pre_fix_cache_free_layer_types

    cfg = getattr(model, "config", None)
    gc = getattr(model, "generation_config", None)
    cfg_cache = getattr(cfg, "use_cache", None)
    gen_cache = getattr(gc, "use_cache", None)
    prep = getattr(model, "kv_cache_preparation", None) or {}
    return {
        "kv_cache": bool(cfg_cache is not False and gen_cache is not False),
        "config_use_cache": cfg_cache,
        "generation_config_use_cache": gen_cache,
        # STABLE (read off the config, not off this load's side effects):
        # non-empty iff the earlier loader decoded this model CACHE-FREE
        # (a layer type transformers 5.14.1 could not cache), so cells it
        # published took the other implementation — the decode-parity gate
        # keys on this
        "pre_fix_cache_free": pre_fix_cache_free_layer_types(cfg),
        # informational: what THIS load registered (empty on a second load
        # in the same process, the mapping being global)
        "cache_placeholder_registered": list(prep.get("registered") or []),
        "mamba_kernel_path": mamba_kernel_path(model),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "device": str(next(model.parameters()).device),
    }


# --------------------------------------------------------------------------
# Decode-path parity gate
# --------------------------------------------------------------------------

#: A row's decode OUTCOMES — what the analysis reads off a cell: the
#: extracted answer, whether it is right, how the row ended. Two decode
#: implementations are interchangeable within a label iff these agree on
#: every row (the response TEXT is a hypersensitive readout and is not
#: required to match).
OUTCOME_KEYS = ("extracted", "correct", "stop_reason")
ATTESTATION_KIND = "gsm8k_decode_parity_attestation"

#: What a decode-parity attestation must agree with the run on, to speak
#: for the label's legacy cells: the model + revision + config, the draw,
#: the prompt, the budget and the runtime identity — everything a row's
#: outcome depends on except the object and its scale (the attested cell is
#: the object-less ``none``, for which scale is inert).
ATTEST_KEYS = ("model", "model_revision", "config_sha256", "batch_size",
               "ordered_qid_hash", "n", "shots", "exemplars_sha256",
               "max_new_tokens", "stop_strings", "dataset", "dataset_config",
               "dataset_split", "dataset_revision", "dtype",
               "attn_implementation", "decoding", "extraction")


def rows_outcome_digest(rows: list[dict]) -> str:
    """sha256 over every row's (qid, extracted, correct, stop_reason),
    ordered by qid — the fingerprint of a cell's OUTCOMES, independent of
    the response text and the file's identity header."""
    items = sorted(([r["qid"], r.get("extracted"), bool(r.get("correct")),
                     r.get("stop_reason")] for r in rows),
                   key=lambda x: x[0])
    return hashlib.sha256(json.dumps(items).encode("utf-8")).hexdigest()


def legacy_cells(run_dir: Path, run_label: str) -> list[str]:
    """Cells already published under the label on the CACHE-FREE decode:
    written before the cache fix (no ``runtime`` block — the loader of the
    day pinned the cache off on a model whose cache needed the placeholder
    registration), or later with ``--allow-cache-free`` (a ``runtime``
    block recording ``kv_cache`` False). Sorted."""
    run_dir = Path(run_dir)
    skip = summary_path(run_dir, run_label).name
    out: list[str] = []
    for p in sorted(run_dir.glob(f"gsm8k_{run_label}_*.json")):
        if p.name == skip:
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        if "cell" not in rec:
            continue
        rt = rec.get("runtime")
        if rt is None or rt.get("kv_cache") is False:
            out.append(str(rec["cell"]))
    return out


def require_decode_parity(run_dir: Path, run_label: str, cells: list[str],
                          run_identity: dict, runtime: dict,
                          attestation: str | None, *,
                          consumer: str = CONSUMER) -> dict | None:
    """Refuse to MIX two decode implementations within one label without
    proof that they agree: the cached and the cache-free decode take
    different bf16 kernel paths, and a merged summary would otherwise
    attribute an implementation difference to the fitted object. The gate
    applies when BOTH hold:

      * this run decodes with the cache on a model the pre-fix loader
        decoded CACHE-FREE (``runtime.pre_fix_cache_free`` non-empty — a
        stable property of the architecture, read off the config; keying
        on what THIS load registered went silent on a second in-process
        load and on transformers >= 5.15) — so every cell that loader
        published under the label took the other path; and
      * the label holds cells published without a ``runtime`` block that
        this invocation does NOT re-score.

    Then ``attestation`` (``--decode-parity``) must name the record
    ``gsm8k_decode_parity.py`` wrote from THIS label's published ``none``
    against its cached re-score under a scratch label, and it is accepted
    iff: its verdict is ``outcomes_identical`` (extracted / correct /
    stop_reason equal on EVERY row); its shared identity equals this run's
    on ATTEST_KEYS; the attested cell is one of the legacy cells and the
    attestation's published side IS this label's file — same per-question
    sha256 AND same outcome digest (attestations are per label; a digest
    alone let a same-outcome scratch file stand in); the published side
    was decoded cache-free (no runtime block, or kv_cache False) so the
    proof exercised the legacy path; and the attested cached run was on
    this run's transformers with the cache on. Returns the record to stamp
    on each new aggregate's ``runtime``, or None when no mixing can occur.
    Every refusal is named."""
    if not runtime.get("kv_cache") or not runtime.get("pre_fix_cache_free"):
        return None
    legacy = [c for c in legacy_cells(run_dir, run_label) if c not in cells]
    if not legacy:
        return None
    if not attestation:
        raise SystemExit(
            f"[{consumer}] label {run_label!r} holds {legacy} published "
            "before the decode-path fix (no runtime block: decoded CACHE-FREE "
            "by the old loader on this model) and this run decodes WITH the "
            "cache — refusing to mix two decode implementations in one label "
            "without proof they agree. Pass --decode-parity <attestation> "
            "(scripts/sdfbase/gsm8k_decode_parity.py: the label's published "
            "`none` vs its cached re-score under a scratch label), or "
            "re-score those cells in this invocation (R2-D78).")
    path = Path(attestation)
    if not path.is_file():
        raise SystemExit(f"[{consumer}] --decode-parity {path} does not exist")
    att = json.loads(path.read_text(encoding="utf-8"))
    if att.get("kind") != ATTESTATION_KIND:
        raise SystemExit(f"[{consumer}] {path} is not a {ATTESTATION_KIND} "
                         f"(kind={att.get('kind')!r})")
    verdict = att.get("verdict") or {}
    if verdict.get("outcomes_identical") is not True:
        raise SystemExit(
            f"[{consumer}] {path}: the attested decode paths do NOT agree on "
            f"every row's outcomes ({att.get('identical_outcomes')}/"
            f"{att.get('n')} identical; extracted differs on "
            f"{len(att.get('extracted_differs') or [])}, stop reason on "
            f"{len(att.get('stop_reason_differs') or [])}) — the cached "
            "cells cannot join a label of cache-free cells; re-score the "
            f"legacy cells {legacy} in this invocation instead (R2-D78).")
    shared = att.get("shared_identity") or {}
    for k in ATTEST_KEYS:
        if shared.get(k) != run_identity.get(k):
            raise SystemExit(
                f"[{consumer}] {path} attests a different run: {k} = "
                f"{shared.get(k)!r} vs this run's {run_identity.get(k)!r}")
    cell = shared.get("cell")
    if cell not in legacy:
        raise SystemExit(
            f"[{consumer}] {path} attests cell {cell!r}, which is not one of "
            f"label {run_label!r}'s legacy cells {legacy}")
    # The attested cell must be the OBJECT-LESS bare cell, not merely SOME
    # legacy cell. A fitted cell can agree across the two decode paths by
    # luck while the bare one does not (the reference experiment has the
    # paths differing on 27 rows), and one such attestation would then be
    # stamped as proof for every remaining legacy cell — re-creating the
    # within-label mixing this gate exists to refuse. There is no
    # substitute: if the bare cell cannot serve as the cache-free
    # reference, the legacy cells get RE-SCORED rather than attested.
    att_obj = shared.get("object_sha256")
    if cell != "none" or att_obj is not None:
        raise SystemExit(
            f"[{consumer}] {path} attests cell {cell!r} "
            f"(object_sha256={att_obj!r}); the proof must come from the "
            "object-less 'none' cell, whose two decodes differ in nothing but "
            "the implementation. An attestation from a fitted cell says "
            "nothing about the others — re-score the legacy cells "
            f"{legacy} in this invocation instead (R2-D78).")
    from sisteer.eval.defense_cell import sha256_file

    _agg, pq = cell_paths(Path(run_dir), run_label, cell)
    pub = att.get("published") or {}
    pq_sha = sha256_file(pq)
    if pub.get("per_question_sha256") != pq_sha:
        raise SystemExit(
            f"[{consumer}] {path} was computed against a {cell!r} file "
            f"(sha256 {pub.get('per_question_sha256')}) that is not this "
            f"label's {pq.name} ({pq_sha}) — attestations are per label; "
            "write one from THIS label's published file")
    _hdr, rows = read_per_question(pq)
    digest = rows_outcome_digest(rows)
    if digest != pub.get("rows_digest"):
        raise SystemExit(
            f"[{consumer}] {path} records outcome digest "
            f"{pub.get('rows_digest')} for {cell!r}; this label's file has "
            f"{digest} — the attestation does not describe this file")
    pub_rt = pub.get("runtime")
    if pub_rt is not None and pub_rt.get("kv_cache") is not False:
        raise SystemExit(
            f"[{consumer}] {path}: the attested PUBLISHED side was itself "
            f"decoded with the cache (runtime {pub_rt}) — the proof never "
            "exercised the cache-free path the legacy cells took")
    new_rt = (att.get("new") or {}).get("runtime") or {}
    if not new_rt.get("kv_cache") or \
            new_rt.get("transformers") != runtime.get("transformers"):
        raise SystemExit(
            f"[{consumer}] {path}: the attested cached run (kv_cache="
            f"{new_rt.get('kv_cache')!r}, transformers="
            f"{new_rt.get('transformers')!r}) is not this runtime "
            f"(transformers {runtime.get('transformers')!r}, cache on)")
    return {"attestation": str(path), "attestation_sha256": sha256_file(path),
            "cell": cell, "n": att.get("n"),
            "identical_outcomes": att.get("identical_outcomes"),
            "legacy_cells": legacy}


def assert_cached_decode(runtime: dict, *, allow_cache_free: bool = False
                         ) -> None:
    """Refuse a CACHE-FREE generate() unless explicitly allowed — the loud
    failure in place of the silent one: fail loudly, never degrade
    silently. A cache-free decode recomputes the whole prompt plus every
    generated token at every step: the Nemotron r2 pathology
    (~1.75 s/step at batch 128, ~2 h per 500-question cell, ~30x the
    GPU-hours of a cached decode for the same greedy outputs). The
    pure-torch Mamba path is only WARNED about: same math, 9-14x the step
    time, and a kernel-less env is a legitimate (slow) place to run."""
    if not runtime.get("kv_cache") and not allow_cache_free:
        raise SystemExit(
            f"[{CONSUMER}] generate() would run CACHE-FREE on this model "
            f"(config.use_cache={runtime.get('config_use_cache')!r}, "
            f"generation_config.use_cache="
            f"{runtime.get('generation_config_use_cache')!r}): every decode "
            "step recomputes the whole prompt plus every generated token — "
            "the Nemotron r2 pathology (~1.75 s/step at batch 128, ~2 h per "
            "500-question cell, ~60x the decode-step time of a cached decode for "
            "the same greedy outputs; R2-D78). Fix the loader (models."
            "prepare_kv_cache) or pass --allow-cache-free for a diagnosis "
            "run.")
    if runtime.get("mamba_kernel_path") == "torch":
        print(f"[{CONSUMER}] WARNING: Mamba2 mixers dispatch to the pure-"
              "torch SSD scan (mamba_ssm / causal_conv1d kernels not "
              "resolved) — 9-14x the step time of the fused path; same "
              "math, so proceeding", flush=True)


def generate_answers(model, tok, obj, prompts: list[str], *,
                     scale: float = 1.0, batch_size: int = DEFAULT_BATCH_SIZE,
                     max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                     stop_strings=STOP_STRINGS) -> list[dict]:
    """One record per prompt, in PROMPT order: ``{text, n_new_tokens,
    hit_cap, eos, stop_reason}`` — the decoded new tokens (special tokens
    skipped; may end with the stop string), how many new tokens the row
    produced (StopRecorder: recorded at the step the row finished, never
    read back from padding), whether it used the whole budget, whether it
    ended on EOS, and why it stopped: ``eos`` | ``stop_string`` | ``cap``
    (non-terminating: ran the full budget with neither) |
    ``early_unknown`` (defensive: HF ended the batch short of the budget
    on a criterion the recorder did not see; reported).

    Discipline (generate.generate_batch / generate_with_object, restated
    because the raw ids are needed): LEFT padding asserted at the built
    batch; ``encode_kwargs`` double-BOS guard; token-kind sentinels
    injected; ``obj.apply(model, scale)`` entered PER CHUNK (a single
    context across chunks leaves a position tracker counting across
    generate calls); greedy (``do_sample=False``); rows are
    length-bucketed into chunks (near-uniform pad width, fewer wasted
    decode steps) and the SAME bucketing runs at every scale, so a 1x/2x
    pair sees byte-identical padding per row.
    """
    import torch

    from sisteer.eval.capability_ext import _maybe_inject, _obj_ctx
    from sisteer.generate import assert_left_padded, encode_kwargs

    if int(batch_size) < 1:
        raise SystemExit(f"[{CONSUMER}] --batch-size must be >= 1")
    if int(max_new_tokens) < 1:
        raise SystemExit(f"[{CONSUMER}] --max-new-tokens must be >= 1")
    device = next(model.parameters()).device
    pad_id, eos_ids = _end_ids(model, tok)
    prompts = [_maybe_inject(obj, p) for p in prompts]
    # length-bucket: sort by prompt token count, chunk, restore order
    lengths = [len(tok(p, **encode_kwargs(tok, p))["input_ids"])
               for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: (lengths[i], i))
    out: list[dict | None] = [None] * len(prompts)
    prev_side = tok.padding_side
    tok.padding_side = "left"
    from transformers.generation import StoppingCriteriaList

    recorder = make_stop_recorder(tok, eos_ids, stop_strings)
    # The recorder IS the stop-string criterion (HF's StopStringCriteria
    # inside it) — `stop_strings=` is deliberately not passed to generate,
    # so one object both decides the event and records it.
    gen_kwargs = dict(max_new_tokens=int(max_new_tokens), do_sample=False,
                      pad_token_id=pad_id,
                      stopping_criteria=StoppingCriteriaList([recorder]))
    n_chunks = -(-len(order) // int(batch_size))
    try:
        with torch.no_grad():
            for i in range(0, len(order), int(batch_size)):
                idx = order[i:i + int(batch_size)]
                chunk = [prompts[j] for j in idx]
                enc = tok(chunk, return_tensors="pt", padding=True,
                          **encode_kwargs(tok, chunk[0])).to(device)
                assert_left_padded(enc.get("attention_mask"),
                                   what=f"{CONSUMER} batched generation")
                width = enc["input_ids"].shape[1]
                recorder.reset(len(idx), width)
                t0 = time.perf_counter()
                with _obj_ctx(model, obj, scale):
                    gen = model.generate(**enc, **gen_kwargs)
                steps = int(gen.shape[1]) - width
                dt = time.perf_counter() - t0
                # pace line per chunk (gpu-util discipline: a slow decode
                # must be explainable from the log — the cache-free
                # pathology ran ~1.75 s/step in silence)
                print(f"[{CONSUMER}] chunk {i // int(batch_size) + 1}/"
                      f"{n_chunks}: {len(idx)} rows x width {width}, {steps} "
                      f"steps in {dt:.1f}s ({dt / max(steps, 1):.3f} s/step)",
                      flush=True)
                accounted = recorder.finish(steps, int(max_new_tokens))
                for k, j in enumerate(idx):
                    new = gen[k, width:].tolist()
                    n_new, why = accounted[k]
                    text = tok.decode(new[:n_new], skip_special_tokens=True)
                    out[j] = {
                        "text": text,
                        "n_new_tokens": int(n_new),
                        "hit_cap": bool(why == "cap"),
                        "eos": bool(why == "eos"),
                        "stop_reason": why,
                    }
    finally:
        tok.padding_side = prev_side
    return [r for r in out]  # type: ignore[misc]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def score_cell(model, tok, obj, rows: list[dict], *, shots: int = DEFAULT_SHOTS,
               scale: float = 1.0, batch_size: int = DEFAULT_BATCH_SIZE,
               max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
               stop_strings=STOP_STRINGS) -> tuple[dict, list[dict]]:
    """(aggregate, per-question rows) for one cell. The aggregate is
    DERIVED from the rows (never computed separately), so the two cannot
    disagree. Per-question row: qidx, qid, gold, extracted, rule, correct,
    terminated, hit_cap, n_new_tokens, empty, response (the answer block)."""
    prompts = [render_prompt(r["question"], shots) for r in rows]
    gens = generate_answers(model, tok, obj, prompts, scale=scale,
                            batch_size=batch_size,
                            max_new_tokens=max_new_tokens,
                            stop_strings=stop_strings)
    per_q: list[dict] = []
    for qidx, (row, g) in enumerate(zip(rows, gens)):
        block, stopped = truncate_at_stop(g["text"], stop_strings)
        pred, rule = extract_answer(block)
        per_q.append({
            "qidx": qidx,
            "qid": question_id(row),
            "gold": float(row["gold"]),
            "extracted": pred,
            "rule": rule,
            "correct": is_correct(pred, row["gold"]),
            "terminated": g["stop_reason"] != "cap",
            "stop_reason": g["stop_reason"],
            "stopped_on_string": bool(stopped),
            "hit_cap": bool(g["hit_cap"]),
            "n_new_tokens": int(g["n_new_tokens"]),
            "empty": not block.strip(),
            "response": block,
        })
    return aggregate(per_q), per_q


def aggregate(per_q: list[dict]) -> dict:
    """Pure arithmetic on the rows: acc + Wilson CI, and the degeneration
    counters the 2x pass reads (non-terminating, empty, cap-hitting,
    extraction failures, fallback-rule uses)."""
    n = len(per_q)
    correct = sum(bool(r["correct"]) for r in per_q)
    lo, hi = bm.wilson_ci(correct, n)
    rules: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for r in per_q:
        rules[r["rule"]] = rules.get(r["rule"], 0) + 1
        reasons[r.get("stop_reason", "?")] = \
            reasons.get(r.get("stop_reason", "?"), 0) + 1
    return {
        "acc": correct / max(n, 1), "correct": correct, "n": n,
        "ci95": [lo, hi],
        "n_non_terminating": sum(not r["terminated"] for r in per_q),
        "n_hit_cap": sum(bool(r["hit_cap"]) for r in per_q),
        "n_empty": sum(bool(r["empty"]) for r in per_q),
        "n_extraction_failed": rules.get("none", 0),
        "n_fallback_rule": rules.get("hash_last", 0) + rules.get("last_number", 0),
        "rules": rules,
        "stop_reasons": reasons,
        "mean_new_tokens": (sum(r["n_new_tokens"] for r in per_q) / n
                            if n else 0.0),
    }


# --------------------------------------------------------------------------
# Namespaced, provenanced, atomic publication (base_mmlu's contract)
# --------------------------------------------------------------------------

PAIR_KEYS = ("dataset", "dataset_config", "dataset_split", "dataset_revision",
             "ordered_qid_hash", "n", "shots", "exemplars_sha256",
             "max_new_tokens", "stop_strings")

#: What two invocations under ONE label must agree on (label preflight +
#: summary merge): the MMLU-era identity keys PLUS every GSM8K
#: protocol-defining field — the pair keys, the decoding rule and the
#: extraction rule (the shared MMLU set alone would merge cells scored
#: under a different budget / stop string / exemplar block as one label).
MERGE_KEYS = tuple(dict.fromkeys(
    (*bm.SUMMARY_MERGE_KEYS, *PAIR_KEYS, "decoding", "extraction",
     # runtime identity: the batch size sets each row's
     # left-padding width and can flip a greedy tie; dtype / attention
     # backend ride in via SUMMARY_MERGE_KEYS
     "batch_size")))


def cell_paths(run_dir: Path, run_label: str, cell: str) -> tuple[Path, Path]:
    stem = f"gsm8k_{run_label}_{cell}"
    return run_dir / f"{stem}.json", run_dir / f"{stem}.per_question.jsonl"


def summary_path(run_dir: Path, run_label: str) -> Path:
    return run_dir / f"gsm8k_{run_label}_summary.json"


def write_cell_results(run_dir: Path, run_label: str, cell: str,
                       agg: dict, per_q: list[dict], identity: dict,
                       runtime: dict | None = None) -> tuple[Path, Path]:
    """Rows FIRST (header line with the run identity, then one JSON row per
    question in draw order), then the aggregate — each tmp+rename. Refuses
    rows that disagree with the aggregate. ``runtime`` (decode_runtime's
    facts: cache on/off, Mamba kernel path, library versions) is recorded
    on the aggregate, outside the identity."""
    agg_path, pq_path = cell_paths(run_dir, run_label, cell)
    if sum(bool(r["correct"]) for r in per_q) != agg["correct"] \
            or len(per_q) != agg["n"]:
        raise SystemExit(f"[{CONSUMER}] internal: per-question rows disagree "
                         "with the aggregate — refusing to publish either")
    header = {"kind": PER_QUESTION_HEADER_KIND, "cell": cell,
              "n_rows": len(per_q), "identity": identity}
    bm._atomic_write(pq_path, "".join(
        json.dumps(r) + "\n" for r in [header, *per_q]))
    record = {**agg, "cell": cell, "identity": identity,
              "per_question_file": pq_path.name}
    if runtime is not None:
        record["runtime"] = runtime
    bm._atomic_write(agg_path, json.dumps(record, indent=1) + "\n")
    return agg_path, pq_path


def read_per_question(path: Path) -> tuple[dict, list[dict]]:
    """(run identity, rows) — the ONE reader; refuses a headerless or torn
    file so a McNemar consumer can assert identity agreement first."""
    lines = [json.loads(ln) for ln in
             Path(path).read_text(encoding="utf-8").splitlines() if ln]
    if not lines or lines[0].get("kind") != PER_QUESTION_HEADER_KIND:
        raise SystemExit(f"[{CONSUMER}] {path} carries no run-identity "
                         "header — not a pairable per-question file")
    header, rows = lines[0], lines[1:]
    if len(rows) != int(header.get("n_rows", -1)):
        raise SystemExit(f"[{CONSUMER}] {path} header promises "
                         f"{header.get('n_rows')} rows, file has "
                         f"{len(rows)} — torn write")
    return header["identity"], rows


def assert_pairable(identity_a: dict, identity_b: dict) -> None:
    """Two runs pair only on the same draw of the same data under the same
    prompt and decode budget (PAIR_KEYS). ``scale`` is deliberately NOT a
    pair key — a 1x/2x pair is the point; ``batch_size`` is not either
    (recorded, not a pairing property). Named refusal."""
    for k in PAIR_KEYS:
        if identity_a.get(k) != identity_b.get(k):
            raise SystemExit(
                f"[{CONSUMER}] runs are not pairable: {k} differs "
                f"({identity_a.get(k)!r} vs {identity_b.get(k)!r})")


def mcnemar_table(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Paired win/loss table between two rows files (join on qid):
    both / a_only / b_only / neither."""
    b_by = {r["qid"]: r for r in rows_b}
    if set(b_by) != {r["qid"] for r in rows_a}:
        raise SystemExit(f"[{CONSUMER}] qid sets differ — not pairable")
    t = {"both": 0, "a_only": 0, "b_only": 0, "neither": 0}
    for ra in rows_a:
        rb = b_by[ra["qid"]]
        t[{(True, True): "both", (True, False): "a_only",
           (False, True): "b_only", (False, False): "neither"}[
            (bool(ra["correct"]), bool(rb["correct"]))]] += 1
    return t


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True,
                    help="the campaign fits config (model/revision pins)")
    ap.add_argument("--cells", required=True,
                    help="comma list: 'none' and/or <data-cell>__<arm> ids "
                         "staged under --defenses-root/cells_all/")
    ap.add_argument("--defenses-root", default=None)
    ap.add_argument("--object", action="append", default=[],
                    metavar="CELL=PATH",
                    help="explicit artifact path for a cell in --cells (a "
                         "DERIVED object; REQUIRES --parents-root)")
    ap.add_argument("--parents-root", default=None)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="object strength passed to obj.apply(model, "
                         "scale=...) — the PROMPT is unchanged at any scale")
    ap.add_argument("--max-new-tokens", type=int,
                    default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--dataset-revision", default=None,
                    help=f"pin the {DATASET} hub revision (commit sha)")
    ap.add_argument("--out-dir", required=True,
                    help="output ROOT: <out-dir>/<run-label>/gsm8k_<run-"
                         "label>_<cell>{.json,.per_question.jsonl} + summary")
    ap.add_argument("--run-label", required=True,
                    help="REQUIRED run namespace (r2 = 1x on the live bank, "
                         "r2_x2 = the 2x pass): dir + filename component")
    ap.add_argument("--allow-cache-free", action="store_true",
                    help="run even if generate() would decode WITHOUT a KV/"
                         "SSM cache (a full prefix recompute per token, "
                         "~60x the decode-step time for the same greedy outputs — "
                         "R2-D78). Diagnosis only; refused otherwise. Not "
                         "part of the run identity.")
    ap.add_argument("--decode-parity", default=None, metavar="JSON",
                    help="attestation from gsm8k_decode_parity.py that the "
                         "cached decode reproduces the label's published "
                         "(cache-free) `none` outcomes row for row; REQUIRED "
                         "to add cached cells to a label that holds cells "
                         "published before R2-D78 on a model whose cache "
                         "needed the placeholder registration. Stamped on "
                         "each new aggregate's runtime; not in the identity.")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(REPO / "src"))
    import yaml

    from sisteer.eval.defense_cell import sha256_file
    from sisteer.extraction.fitmon import pin_attention_backends
    from sisteer.methods import load_object
    from sisteer.models import load_model_and_tokenizer

    run_label = bm.validate_run_label(args.run_label)
    pin_attention_backends()
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    cells = [c for c in args.cells.split(",") if c]
    if len(set(cells)) != len(cells):
        raise SystemExit(f"[{CONSUMER}] duplicate cells in --cells: {cells}")
    explicit = bm.parse_explicit_objects(args.object, cells)
    if any(c != "none" and c not in explicit for c in cells) \
            and not args.defenses_root:
        raise SystemExit(f"[{CONSUMER}] fitted cells given but no "
                         "--defenses-root (or --object CELL=PATH for each)")
    cells = sorted(cells, key=lambda c: c != "none")  # bare base FIRST

    # Identity gate + object shas for EVERY cell first (no GPU), then the
    # label preflight, then the data and the model (base_mmlu's order).
    resolved: dict[str, Path | None] = {}
    for cell in cells:
        resolved[cell] = None if cell == "none" else bm.resolve_object_path(
            Path(args.defenses_root) if args.defenses_root else None,
            cell, parents_root=args.parents_root,
            explicit_path=explicit.get(cell), consumer=CONSUMER)
    cell_shas = {c: (sha256_file(p) if p else None)
                 for c, p in resolved.items()}

    rows, dataset_revision = load_rows(args.n, args.seed,
                                       revision=args.dataset_revision)

    run_identity = {
        "run_label": run_label,
        "model": cfg["model"], "model_revision": cfg.get("model_revision"),
        "config": str(args.config),
        "config_sha256": sha256_file(cfg_path),
        "dataset": DATASET, "dataset_config": DATASET_CONFIG,
        "dataset_split": DATASET_SPLIT,
        "dataset_revision": dataset_revision,
        "n": len(rows), "requested_n": args.n, "seed": args.seed,
        "shots": args.shots, "scale": float(args.scale),
        "ordered_qid_hash": ordered_qid_hash(rows),
        "render": RENDER, "chat_template_bypassed": True,
        "exemplars_sha256": exemplars_sha256(args.shots),
        "max_new_tokens": int(args.max_new_tokens),
        "stop_strings": list(STOP_STRINGS),
        "decoding": "greedy",
        "batch_size": int(args.batch_size),
        "dtype": cfg.get("dtype", "bfloat16"),
        "attn_implementation": cfg.get("attn_implementation", "sdpa"),
        "extraction": "answer_is_first|hash_last|last_number",
        "defenses_root": (str(args.defenses_root)
                          if args.defenses_root else None),
        "parents_root": (str(args.parents_root)
                         if args.parents_root else None),
        "explicit_objects": explicit or None,
    }
    run_dir = bm.run_dir_for(Path(args.out_dir), run_label)
    run_dir.mkdir(parents=True, exist_ok=True)
    rescored = bm.preflight_label(run_dir, run_label, run_identity, cell_shas,
                                  cell_paths_fn=cell_paths,
                                  summary_path_fn=summary_path,
                                  merge_keys=MERGE_KEYS, consumer=CONSUMER)
    if rescored:
        print(f"[{CONSUMER}] {run_label}: re-scoring {rescored} (same "
              "identity + object sha already published; idempotent)")

    model, tok = load_model_and_tokenizer(
        cfg["model"], dtype=cfg.get("dtype", "bfloat16"),
        revision=cfg.get("model_revision"),
        attn_implementation=cfg.get("attn_implementation", "sdpa"))
    model.eval()
    runtime = decode_runtime(model)
    print(f"[{CONSUMER}] decode runtime: {json.dumps(runtime)}", flush=True)
    assert_cached_decode(runtime, allow_cache_free=args.allow_cache_free)
    parity = require_decode_parity(run_dir, run_label, cells, run_identity,
                                   runtime, args.decode_parity)
    if parity is not None:
        runtime["decode_parity"] = parity
        print(f"[{CONSUMER}] {run_label}: adding cached cells to a label with "
              f"legacy cache-free cells {parity['legacy_cells']} under "
              f"attestation {parity['attestation']} "
              f"({parity['identical_outcomes']}/{parity['n']} outcomes "
              "identical)", flush=True)

    results: dict[str, dict] = {}
    base_acc = None
    for cell in cells:
        path = resolved[cell]
        obj = load_object(path, model=model, tok=tok) if path else None
        res, per_q = score_cell(model, tok, obj, rows, shots=args.shots,
                                scale=args.scale, batch_size=args.batch_size,
                                max_new_tokens=args.max_new_tokens)
        res["delta_vs_bare"] = (round(res["acc"] - base_acc, 6)
                                if base_acc is not None and cell != "none"
                                else None)
        identity = {**run_identity, "cell": cell,
                    "object_path": str(path) if path else None,
                    "object_sha256": cell_shas[cell]}
        agg_path, pq_path = write_cell_results(run_dir, run_label, cell,
                                               res, per_q, identity,
                                               runtime=runtime)
        results[cell] = {**res, "identity": identity, "runtime": runtime,
                         "per_question_file": pq_path.name,
                         "aggregate_file": agg_path.name}
        if cell == "none":
            base_acc = res["acc"]
        print(f"[{CONSUMER}] {run_label}/{cell}: acc={res['acc']:.4f} "
              f"({res['correct']}/{res['n']}, CI95 "
              f"[{res['ci95'][0]:.4f}, {res['ci95'][1]:.4f}]) "
              f"non_terminating={res['n_non_terminating']} "
              f"empty={res['n_empty']} extraction_failed="
              f"{res['n_extraction_failed']} -> {agg_path.name}", flush=True)

    spath = summary_path(run_dir, run_label)
    bm.write_summary(spath, run_identity, cells, results, consumer=CONSUMER,
                     merge_keys=MERGE_KEYS)
    print(f"[{CONSUMER}] wrote {spath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

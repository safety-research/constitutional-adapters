"""Leading special tokens of a template-free encode: declared per config,
applied at tokenizer load, asserted on the encoded training examples before
any optimizer step.

Why: the doc-CE lane encodes flat documents with
``tok(text, add_special_tokens=True)`` and lets the tokenizer's
post-processor decide what leads the sequence. That is a per-CHECKPOINT
fact, not a per-family one, verified on the real tokenizers at the pinned
revisions:

  google/gemma-4-E4B      TemplateProcessing(<bos> $A)  -> [2] + text
  google/gemma-4-E4B-it   TemplateProcessing($A)        -> text (its chat
                          template supplies ``<bos>`` as literal text)
  Qwen/Qwen3.5-4B(-Base)  ByteLevel, no BOS token       -> text (both)
  zai-org/GLM-4.6V-Flash  Sequence[ByteLevel]           -> text (the
                          template's ``[gMASK]<sop>`` is literal text too)
  Nemotron-3-Nano-4B      ByteLevel (a <s> exists, unused) -> text

So an instruct-direct gemma-it fit would silently train BOS-less against a
base sibling that trained with BOS, while the qwen pair already agrees.

Policy values (config key ``leading_special_tokens``; ABSENT = the
tokenizer's own template-free convention, observed and logged, never
changed — every existing lane's bytes and behaviour stay untouched):

  "bos"   exactly one ``bos_token_id`` at position 0 of every training
          example, inside the prompt (loss-masked) span. A tokenizer whose
          template-free encode already leads with BOS is a no-op; one that
          adds nothing gets the base convention's post-processor installed
          (``<bos> $A`` / ``<bos> $A <bos> $B`` — the gemma-4-E4B
          tokenizer.json shape); any other prefix refuses.
  "none"  the template-free encode adds no special tokens and no example
          starts with a BOS; refuses otherwise.

The check runs on the ENCODED sequences the fit trains on (the
``_encode_aligned`` output: prompt ids + completion ids), so the assertion
covers the whole path, not the tokenizer alone. Every refusal is a
SystemExit naming what was found — a fit must never start on a convention
it did not declare.
"""

from __future__ import annotations

from typing import Iterable

#: The config key (top level, next to model / model_revision).
LEADING_KEY = "leading_special_tokens"
POLICIES = ("bos", "none")


def template_free_prefix(tok) -> list[int]:
    """What ``add_special_tokens=True`` prepends to an empty string — the
    tokenizer's own leading convention, as the doc-CE path sees it."""
    return [int(i) for i in tok("", add_special_tokens=True)["input_ids"]]


def _install_bos_post_processor(tok) -> None:
    """Install the base convention's post-processor: ``<bos> $A`` for a
    single sequence, ``<bos> $A <bos> $B`` (type ids 0 / 1) for a pair —
    exactly google/gemma-4-E4B's tokenizer.json TemplateProcessing."""
    from tokenizers.processors import TemplateProcessing

    bos, bos_id = tok.bos_token, int(tok.bos_token_id)
    tok.backend_tokenizer.post_processor = TemplateProcessing(
        single=[bos, "$A"], pair=[bos, "$A", f"{bos}:1", "$B:1"],
        special_tokens=[(bos, bos_id)])


def apply_leading_special_tokens(tok, policy: str | None):
    """Make ``tok`` honour ``policy`` (see the module docstring), or refuse.
    ``None``/'' leaves the tokenizer exactly as loaded. Returns ``tok``."""
    if not policy:
        return tok
    if policy not in POLICIES:
        raise SystemExit(f"[leading-tokens] unknown {LEADING_KEY}={policy!r} "
                         f"(one of {list(POLICIES)})")
    name = getattr(tok, "name_or_path", type(tok).__name__)
    prefix = template_free_prefix(tok)
    bos = tok.bos_token_id
    if policy == "bos":
        if bos is None:
            raise SystemExit(
                f"[leading-tokens] {LEADING_KEY}=bos but {name} has no bos "
                "token — the policy cannot be met")
        if prefix == [int(bos)]:
            pass  # already the convention (a base-style post-processor)
        elif prefix == []:
            if not hasattr(tok, "backend_tokenizer"):
                raise SystemExit(
                    f"[leading-tokens] {name} adds no leading tokens and is "
                    "not a fast tokenizer — cannot install the <bos> $A "
                    "post-processor")
            _install_bos_post_processor(tok)
            after = template_free_prefix(tok)
            if after != [int(bos)]:
                raise SystemExit(
                    f"[leading-tokens] installing '<bos> $A' on {name} gave "
                    f"template-free prefix {after}, expected [{bos}]")
            print(f"[leading-tokens] {name}: template-free encode added "
                  f"nothing; installed the '<bos> $A' post-processor "
                  f"(prefix {prefix} -> {after}) per {LEADING_KEY}=bos",
                  flush=True)
        else:
            raise SystemExit(
                f"[leading-tokens] {LEADING_KEY}=bos but {name}'s template-"
                f"free encode leads with {prefix} (bos_token_id={bos}) — "
                "neither bare nor BOS-only; refusing to guess")
    else:  # "none"
        if prefix:
            raise SystemExit(
                f"[leading-tokens] {LEADING_KEY}=none but {name}'s template-"
                f"free encode prepends {prefix} — refusing")
    tok._sisteer_leading_special_tokens = policy
    return tok


def check_leading_special_tokens(tok, seqs: Iterable, policy: str | None,
                                 *, label: str = "") -> dict:
    """Assert ``policy`` on encoded sequences (objects with ``ids`` and
    ``prompt_len`` — the ``_encode_aligned`` output). Returns a summary dict
    whose ``line`` is the one-line record for the fit log; raises SystemExit
    on the first violation, naming up to three offending examples. With
    ``policy=None`` nothing is asserted — the observed convention is only
    described."""
    if policy and policy not in POLICIES:
        raise SystemExit(f"[leading-tokens] unknown {LEADING_KEY}={policy!r} "
                         f"(one of {list(POLICIES)})")
    bos = tok.bos_token_id
    prefix = template_free_prefix(tok)
    first = None
    n = 0
    bad: list[tuple[int, list[int], int]] = []
    for s in seqs:
        n += 1
        if first is None:
            first = s
        ids = list(s.ids)
        if policy == "bos":
            ok = (bool(ids) and ids[0] == bos and ids.count(bos) == 1
                  and int(s.prompt_len) >= 1)
        elif policy == "none":
            ok = not prefix and not (bos is not None and ids
                                     and ids[0] == bos)
        else:
            ok = True
        if not ok and len(bad) < 3:
            bad.append((n - 1, ids[:5], int(s.prompt_len)))
    if first is None:
        raise SystemExit("[leading-tokens] no examples to check")
    head = list(first.ids)[:5]
    decoded = [tok.decode([i]) for i in head]
    line = (f"[leading-tokens] {label}{LEADING_KEY}={policy or 'unset'} "
            f"(policy {'asserted' if policy else 'observed only'}) "
            f"template_free_prefix={prefix} bos_token_id={bos} "
            f"first_example ids[:5]={head} decoded={decoded} "
            f"prompt_len={int(first.prompt_len)} — {n} example(s) checked")
    if bad:
        raise SystemExit(
            f"[leading-tokens] REFUSING: {len(bad)}{'+' if len(bad) == 3 else ''}"
            f" of {n} encoded examples violate {LEADING_KEY}={policy!r} "
            f"(index, ids[:5], prompt_len): {bad}. {line}")
    return {"line": line, "policy": policy, "template_free_prefix": prefix,
            "bos_token_id": bos, "first_ids": head, "first_decoded": decoded,
            "first_prompt_len": int(first.prompt_len), "n_checked": n}

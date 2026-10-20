#!/usr/bin/env python
"""Fit the Arditi refusal direction of ONE served checkpoint, in-repo.

The paper's ``arditi_allpos`` directions were fitted OUTSIDE this repository
and staged from a grading host tree (``configs/arditi_allpos_pins.json`` pins
them); this is the fitter the reproduction runs for its test model, so the
"refusal direction (confirm, no ladder)" rests on code in the tree.

Recipe — Arditi et al. 2024, arXiv 2406.11717 ("Refusal in language models is
mediated by a single direction"), the difference-in-means step:

* render every harmful and every harmless instruction through the model's
  chat template with NO system prompt (``sisteer.models.build_chat_prompt``,
  the harness's one renderer — the rendering deviation every arditi_allpos
  row already declares), encode BOS-safely (``sisteer.generate.encode_kwargs``);
* run the model and, at EVERY decoder block, read the residual stream at the
  LAST token of the rendered prompt — the post-instruction position the
  paper's candidate directions are read from — at the block's OUTPUT, the
  exact tensor ``hooks.steering`` adds a ``vector_all`` object to
  (``capture_site: resid_post``);
* ``r_raw[l] = mean_harmful[l] - mean_harmless[l]`` per layer, in fp32.

The paper then SELECTS one (layer, position) by its refusal-score / KL
criteria and serves that single direction. THIS object keeps every layer's
direction and serves at all positions (kind ``vector_all``, one d-vector per
layer, ``skip_bos`` False, staged id ``ar_c<c>__arditi_allpos`` — the
all-positions arm the paper's arditi_allpos rows carry), and stores
``c * r_raw[l]`` so the dose served at scale 1.0 is exactly
``coefficient x raw direction``: the ``raw_coefficient`` dose unit of
``scripts/arditi/arditi_strength.py`` (c multiplies the RAW difference of
means — the paper's own convention, c = 1.0 nominal — never a unit-normalised
direction). The battery's operating dose is c = 0.3 (spec FD 5); the raw
direction is written beside the object (``direction.pt``) so a rung of the
ladder {0.15, 0.3, 0.5} is a re-bake, never a refit.

Outputs (``<out>/``): ``object.pt`` (``MultiLayerVectorObject`` blob),
``direction.pt`` (the raw per-layer diff-in-means), ``fit_meta.json`` (the
sidecar ``sisteer.registry.hooks`` reads: kind, derivation none,
fit_target_kind contrastive_direction, object_role baseline_published,
coefficient, positions, n_harmful / n_harmless, prompt_set_ids, model,
revision, layer_norms, direction_norms, object_sha256, ...),
``fit_provenance.json`` (the IDENTITY sidecar — the stage-3 native fit
record the stager and the consumers' gate read, a projection of the same
meta: ``fit_provenance_record``) and ``COMPLETE``
(a manifest of every file with its sha256 — the publication event, written
LAST after an atomic temp-dir rename). Then the registry ``object`` record
(``hooks.emit_object_from_sidecar``: kind ``vector_all``, role
``baseline_published`` -> paper label ``arditi_allpos``, target kind
``contrastive_direction``, objective ``diffmean`` — the vocabulary's spelling
for a difference of means, ``vocab.FIT_OBJECTIVES``).

Idempotent: a re-run over a COMPLETE publication of the SAME request (model,
revision, coefficient, positions, layers, both prompt sets by content, n,
seed, max_len, dtype) keeps it and refits nothing — it re-verifies the
registry emission from the evidence beside COMPLETE (``registry_emission.json``)
and replays the ``object`` record when that is missing (a failed emission exits
EMIT_FAILED_RC with a ``REGISTRY_EMIT_FAILED`` marker, the artefact intact), and
writes the identity sidecar, listed in COMPLETE, beside a kept publication that
predates it or whose listed copy is torn (``ensure_fit_provenance`` /
``repair_fit_provenance``, both from the verified meta, no refit); a valid
publication of a DIFFERENT request under the same ``--out`` REFUSES unless
``--overwrite``; a torn dir (no valid COMPLETE) is superseded.

The publication event is ONE rename of the finished temporary directory onto
``--out``, which the kernel refuses while ``--out`` is a populated directory
(``_publish``): a publication that appeared since this fit began — another
job of the same identity, which the front door's queue read did not see (two
front doors in the same instant; a hand submission) — is never replaced. When
it is a valid publication of the SAME request it is kept, this fit's bytes
are discarded and the kept path above runs (its emission verified, never a
second object of the identity); a valid publication of a DIFFERENT request
refuses as above; a torn one is cleared and the rename made again. The
rename, the reading of a populated ``--out`` and the clearing of a torn one
are ONE critical section under an advisory ``flock`` on a lock file beside
the directory (``_publication_lock``), so a second publisher's verdict is
never stale by the time it acts on it: two jobs meeting one torn directory
publish one of them and keep it, and never remove the one that just landed.
The SAME lock is held from the publish-or-keep decision THROUGH the registry
emission and its evidence write (the fresh publication's ``registry_emit``;
the kept path's ``ensure_registry_emission``; the pre-fit kept path too), so
the job that loses the publication reads the winner's emission under the
lock — the evidence beside COMPLETE, else the spool's records — and emits
nothing new, or replays only the record the replay path finds missing. The
registry hook's own duplicate guard reads the spool and then writes; two
processes passing it in the same instant was two native ``object``
statements of one id, and the lock is what
makes that read-then-write one process at a time per publication.

The publication directory is the SERVED checkpoint's — ``<refusal root>/<served
registry model id>/<staged id>`` (04_fit_objects.sh, the sbatch's <model_key>):
the object's registry identity (campaign, fit_model_id = the served id, kind,
label, role, rung) names its directory, so two models-table keys serving one
checkpoint share the publication, the row and the pin.

    fit_refusal_direction.py --model Qwen/Qwen3.5-4B --revision <40hex> \\
        --out $SHARED/exp/objects/refusal_direction/qwen35_4b_it/ar_c0p3__arditi_allpos \\
        --harmful arditi:harmful_train --harmless arditi:harmless_train --coefficient 0.3 --positions all \\
        [--n 128 --max-len 512 --dtype bfloat16 --device cuda --batch-size 8 --seed 0 --model-key qwen35_4b_it]

Prompt sets: a file (``.jsonl`` of objects carrying one of prompt /
instruction / text / goal / behavior / query, or bare JSON strings; ``.json``
flat list; ``.txt`` one per line) or a builtin id: ``arditi:harmful_train`` /
``arditi:harmless_train`` / ``arditi:harmful_val`` / ``arditi:harmless_val``
(the paper fit's own draw from andyrdt/refusal_direction, 128 training and 32
validation instructions per side, re-created byte for byte and checked against
``data/arditi/fit_prompts.manifest.json`` by ``sisteer.data.arditi_fit_prompts``
from the third-party cache; stage 04 fits on the two training sets, the rows
the fit behind the paper's committed refusal-direction objects started from:
that fit then dropped rows by a refusal metric and selected one layer and
position on the validation sets, where this fitter keeps every row and layer),
``harmbench_val`` / ``harmbench_test`` (HarmBench standard behaviours,
``sisteer.eval.jailbreak_data.load_harmbench``), ``alpaca`` (instruction-only
Alpaca prompts, ``sisteer.data.prompts.load_hf_prompts``; both need the dataset
in the HF cache when offline), ``seed:general`` / ``seed:medical`` /
``seed:code`` (the harness's built-in offline seed prompts — smoke tests only,
not a refusal contrast). ``--n`` caps each side by a seeded sample; prompts longer than
``--max-len`` tokens are DROPPED (a truncated prompt has no instruction end),
never truncated.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import fcntl
import hashlib
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

KIND = "vector_all"
OBJECT_ROLE = "baseline_published"
FIT_TARGET_KIND = "contrastive_direction"
FIT_OBJECTIVE = "diffmean"            # vocab.FIT_OBJECTIVES: the difference-of-means spelling
DERIVATION = "none"
DOSE_UNIT = "raw_coefficient"         # arditi_strength.UNIT_RAW_COEFFICIENT
PAPER_ARM = "arditi_allpos"
CAPTURE_SITE = "resid_post"           # the decoder block's output — where hooks.steering adds
CAPTURE_POSITION = "last_instruction_token"
OBJECT_NAME = "object.pt"
DIRECTION_NAME = "direction.pt"
META_NAME = "fit_meta.json"
COMPLETE_NAME = "COMPLETE"
#: the IDENTITY sidecar beside fit_meta.json (gap-refusal-pin):
#: sisteer.eval.train_budget.FIT_PROVENANCE_NAME == sweep.STAGE3_FIT_PROVENANCE_FILE, the one of the
#: three identity sidecars (train_budget_meta.json / derived_meta.json / fit_provenance.json) a staged
#: object dir may carry. stage_objects.readiness_problem admits a cell by exactly one of them, naming
#: the cell; the consumers' gate (train_budget.assert_object_identity -> fit_provenance_problems ->
#: sweep.read_stage3_fit_provenance) reads the record's resolved model id + revision, the MEASURED
#: chat-template hash and the writer (registered in sweep.CLASS_B_FIT_WRITERS), and binds it to the
#: bytes by its object_sha256. fit_meta.json is NOT an identity sidecar (it is the registry hook's), so
#: a publication carrying only it was `not_ready` to stage. The record is a PROJECTION of the same
#: in-memory meta — the same fit-time facts, never a second measurement — written in the same
#: publication and hashed into COMPLETE; the paper's committed refusal direction carries the same file
#: (objects/<model id>/<object id>/fit_provenance.json, writer scripts/arditi/fit_arditi.py).
PROVENANCE_NAME = "fit_provenance.json"
#: COMPLETE's files manifest, in this order (every published file but the marker itself and the
#: emission evidence, which is not a manifested file)
PUBLISHED_FILES = (OBJECT_NAME, DIRECTION_NAME, META_NAME, PROVENANCE_NAME)


def fit_provenance_record(meta: dict) -> dict:
    """The native stage-3 fit record (``sweep.STAGE3_FIT_REQUIRED_KEYS``: resolved_model_id,
    resolved_revision, chat_template_hash, writer) from the fit's own meta, plus what the identity's
    readers key on: ``staged_id`` / ``cell_id`` (stage_objects._sidecar_names_cell), ``object_sha256``
    (the bytes it describes — read_stage3_fit_provenance's ``bound()`` refuses a stale pair),
    ``positions_at_fit`` (held to the blob's positions), ``chat_template_bypassed`` False (the prompts
    are rendered through the chat template: a Class B fit), and the rung as the committed record spells
    it (``c`` + ``dose_unit``: sisteer.objects reads the refusal direction's dose from it; the registry
    hook's base_scale cross-check reads ``c``). A null revision or template hash is a stated absence the
    gate refuses by name — never inferred here."""
    return {
        "writer": SCRIPT,
        "resolved_model_id": meta.get("model"),
        "resolved_revision": meta.get("revision"),
        "chat_template_hash": meta.get("template_hash"),
        "positions_at_fit": meta.get("positions_at_fit"),
        "chat_template_bypassed": False,
        "object_sha256": meta.get("object_sha256"),
        "staged_id": meta.get("staged_id"),
        "cell_id": meta.get("cell_id"),
        "kind": meta.get("kind"),
        "paper_arm": meta.get("paper_arm"),
        "c": meta.get("coefficient"),
        "dose_unit": meta.get("dose_unit"),
        "model_key": meta.get("model_key"),
        "direction_sha256": meta.get("direction_sha256"),
        "written_at": meta.get("built_at"),
        "written_by": SCRIPT,
        "note": ("the stage-3 identity record of this publication: a projection of fit_meta.json "
                 "(the same fit-time facts, never a second measurement)"),
    }


def fit_provenance_text(meta: dict) -> str:
    """The identity sidecar's BYTES: the projection as JSON, indent 1, sorted keys, newline-terminated —
    the fit's own spelling, deterministic in the meta, so a backfill or a repair writes the bytes a
    fresh publication would have."""
    return json.dumps(fit_provenance_record(meta), indent=1, sort_keys=True) + "\n"


def _write_atomic(path: Path, text: str) -> None:
    """``text`` into ``path`` through a sibling temp file and ``os.replace`` — a reader never sees a
    partial file, and an interruption leaves the old file (or none) in place."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def ensure_fit_provenance(out_dir) -> bool:
    """A KEPT, COMPLETE publication (the caller verified every file COMPLETE lists, ``fit_meta.json``
    included) carries the identity sidecar LISTED in COMPLETE with its sha256, or gets it: a publication
    from before the sidecar existed, or a backfill torn between its two writes (the sidecar on disk, the
    marker not yet rewritten), has the projection written from the verified meta and the marker rewritten
    with the sidecar's hash — each write atomic (``_write_atomic``), the sidecar FIRST, so a retry after any
    interruption finds the sidecar unlisted and writes both again. The stager
    (``stage_objects.complete_marker_problem``) and the consumers' gate
    (``train_budget._complete_marker_problems``) then verify it as they verify every listed file, and an
    altered copy reads as a torn publication, never as provenance. Nothing is refitted; a publication
    whose marker lists the sidecar is untouched (the caller's manifest check already held the file to its
    hash). True when written. (gap-refusal-pin; before this the backfill
    wrote the sidecar OUTSIDE the manifest, so a torn or altered copy stood unverified and a retry left
    it as it was.)"""
    out_dir = Path(out_dir)
    marker_path = out_dir / COMPLETE_NAME
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    files = marker.get("files") if isinstance(marker.get("files"), dict) else None
    if not files:
        raise RefusalFitError(f"{marker_path} carries no files manifest — not a publication this fitter wrote; "
                              "refusing to backfill its identity sidecar")
    if PROVENANCE_NAME in files:
        return False
    text = fit_provenance_text(json.loads((out_dir / META_NAME).read_text(encoding="utf-8")))
    _write_atomic(out_dir / PROVENANCE_NAME, text)
    files = dict(files)
    files[PROVENANCE_NAME] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    marker["files"] = files
    marker["identity_sidecar_backfilled_at"] = _now()
    _write_atomic(marker_path, json.dumps(marker, indent=1, sort_keys=True) + "\n")
    print(f"[refusal-direction] {out_dir}: wrote {PROVENANCE_NAME} (the identity sidecar, a projection of "
          f"its {META_NAME}) beside a kept publication that predates it, and listed it in {COMPLETE_NAME} "
          f"({files[PROVENANCE_NAME][:16]}…)", flush=True)
    return True


def repair_fit_provenance(out_dir) -> bool:
    """A LISTED identity sidecar that is missing or does not hash to its manifest entry — a torn or altered
    copy, the ONE manifest problem ``manifest_problems`` found — is written again from the verified
    ``fit_meta.json`` WHEN the projection hashes to the listed value: the sidecar is a projection of the
    meta, so the bytes COMPLETE promised are recoverable and nothing is refitted. False when they are not
    (the meta is not at its own manifest hash, or the projection is not what the marker lists: the
    publication is torn, and the caller rebuilds it as any torn publication). Never a projection of
    unverified facts."""
    out_dir = Path(out_dir)
    marker = json.loads((out_dir / COMPLETE_NAME).read_text(encoding="utf-8"))
    files = marker.get("files") if isinstance(marker.get("files"), dict) else {}
    want = files.get(PROVENANCE_NAME)
    meta_path = out_dir / META_NAME
    if not want or not meta_path.is_file() or sha256_file(meta_path) != files.get(META_NAME):
        return False
    text = fit_provenance_text(json.loads(meta_path.read_text(encoding="utf-8")))
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != want:
        return False
    _write_atomic(out_dir / PROVENANCE_NAME, text)
    print(f"[refusal-direction] {out_dir}: {PROVENANCE_NAME} was missing or not at its published sha256 — "
          f"written again from the verified {META_NAME} (the bytes {COMPLETE_NAME} lists; no refit)", flush=True)
    return True
#: Registry-emission EVIDENCE beside COMPLETE (the trainer's spelling,
#: train_two_si_objects.REGISTRY_EMISSION_NAME): the publication event is not
#: the record. Every emission attempt leaves ``registry_emission.json`` =
#: {object_id, status: emitted | not_emitted | failed, records | reason |
#: error, utc}; a failure also writes the bare ``REGISTRY_EMIT_FAILED`` marker
#: (removed by a later success) and the fitter exits EMIT_FAILED_RC — the
#: artefact stands, only the job fails, visibly. A kept publication re-verifies
#: its evidence and re-emits when it is missing (ensure_registry_emission),
#: never by refitting.
def _emission():
    """scripts/sdfbase/registry_emission.py — the ONE owner of the evidence writer, the spool scan and the
    verify-or-replay step; loaded by path (scripts/sdfbase is not on sys.path)."""
    mod = sys.modules.get("registry_emission")
    if mod is None:
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "sdfbase" / "registry_emission.py"
        spec = importlib.util.spec_from_file_location("registry_emission", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["registry_emission"] = mod
        spec.loader.exec_module(mod)
    return mod


EMISSION_NAME = "registry_emission.json"        # == registry_emission.EMISSION_NAME (pinned by test)
EMIT_FAILED_NAME = "REGISTRY_EMIT_FAILED"        # == registry_emission.EMIT_FAILED_NAME
EMIT_FAILED_RC = 3                               # == registry_emission.EMIT_FAILED_RC
PROMPT_KEYS = ("prompt", "instruction", "text", "goal", "behavior", "query")
#: ``arditi:<set>``: a set of the paper fit's own draw (sisteer.data.arditi_fit_prompts.SET_NAMES)
ARDITI_SETS = ("arditi:harmful_train", "arditi:harmless_train", "arditi:harmful_val", "arditi:harmless_val")
BUILTIN_PROMPT_SETS = ARDITI_SETS + ("harmbench_val", "harmbench_test", "alpaca",
                                     "seed:general", "seed:medical", "seed:code")
SCRIPT = "scripts/arditi/fit_refusal_direction.py"


class RefusalFitError(RuntimeError):
    """Loud failure — never publish a direction whose inputs are not what
    was asked for."""


# --------------------------------------------------------------------------- #
# ids + hashing                                                                #
# --------------------------------------------------------------------------- #

def staged_id(coefficient: float) -> str:
    """``ar_c0p3__arditi_allpos`` for 0.3 — the staged-id grammar of the
    arditi_allpos ladder (``ar_c<c with . as p>__arditi_allpos``)."""
    return f"ar_c{coefficient:g}".replace(".", "p") + f"__{PAPER_ARM}"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha_text(items) -> str:
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def _now() -> str:
    """The registry's timestamp spelling (records.utc_now)."""
    from sisteer.registry.records import utc_now
    return utc_now()


def _code_sha() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return (out.stdout.strip() or None) if out.returncode == 0 else None
    except Exception:  # noqa: BLE001 — no git here: stated absence
        return None


# --------------------------------------------------------------------------- #
# prompt sets                                                                  #
# --------------------------------------------------------------------------- #

def _prompt_of(item, where: str) -> str:
    if isinstance(item, str) and item.strip():
        return item
    if isinstance(item, dict):
        for k in PROMPT_KEYS:
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                return v
    raise RefusalFitError(
        f"{where}: cannot read a prompt string from a {type(item).__name__} "
        f"(a bare string, or an object with one of {PROMPT_KEYS})")


def load_prompt_file(path: Path) -> list[str]:
    """A VERBATIM prompt list from ``.jsonl`` / ``.json`` / ``.txt`` — every
    malformation refuses loudly (a silently empty side is the failure this
    exists to prevent)."""
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        items = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
    elif path.suffix == ".json":
        items = json.loads(raw)
        if not isinstance(items, list):
            raise RefusalFitError(f"{path}: a .json prompt set must be a flat "
                                  f"list, got {type(items).__name__}")
    else:
        items = [ln for ln in raw.splitlines() if ln.strip()]
    prompts = [_prompt_of(x, f"{path}:{i + 1}") for i, x in enumerate(items)]
    if not prompts:
        raise RefusalFitError(f"{path}: EMPTY prompt set — refusing to fit on it")
    return prompts


def resolve_prompt_set(spec: str, *, side: str, n: int | None,
                       seed: int) -> tuple[list[str], dict]:
    """``(prompts, info)`` for a file path or a builtin id; deduplicated in
    order, then capped to ``n`` by a seeded sample (indices kept sorted, so
    the subset is a stable subsequence)."""
    p = Path(spec)
    info: dict = {"id": spec, "side": side}
    if p.is_file():
        prompts = load_prompt_file(p)
        info.update({"kind": "file", "source": str(p.resolve()),
                     "file_sha256": sha256_file(p)})
    elif spec.startswith("arditi:"):
        # the paper fit's own draw: the document re-created from the pinned upstream splits and held to every hash
        # data/arditi/fit_prompts.manifest.json records (read OFFLINE on a cluster, from the copy 00_setup.sh installs)
        from sisteer.data import arditi_fit_prompts as afp
        name = spec.split(":", 1)[1]
        if name not in afp.SET_NAMES:
            raise RefusalFitError(f"{side}: unknown builtin set {spec!r}; the paper fit's sets are {list(ARDITI_SETS)}")
        try:
            sets, report = afp.load_sets()
        except afp.FitPromptsError as exc:
            raise RefusalFitError(f"{side}: {spec!r}: {exc}") from None
        prompts = sets[name]
        info.update({"kind": "arditi_fit_prompts", "set": name, "source": report["path"],
                     "manifest": afp.MANIFEST_RELPATH, "document_sha256": report["sha256"],
                     "set_sha256": report["sets"][name]["sha256"],
                     "upstream": report["upstream"]["repo"], "upstream_commit": report["upstream"]["commit"]})
    elif spec.startswith("seed:"):
        from sisteer.data.prompts import seed_prompts
        try:
            prompts = seed_prompts(spec.split(":", 1)[1])
        except KeyError:
            raise RefusalFitError(f"{side}: unknown builtin seed set {spec!r}") from None
        info.update({"kind": "builtin_seed", "source": "sisteer.data.prompts.SEED_PROMPTS"})
    elif spec in ("harmbench_val", "harmbench_test"):
        from sisteer.eval.jailbreak_data import load_harmbench
        split = spec.split("_", 1)[1]
        prompts = [g.text for g in load_harmbench(split=split, functional="standard")]
        info.update({"kind": "harmbench", "split": split, "functional": "standard",
                     "source": "sisteer.eval.jailbreak_data.load_harmbench"})
    elif spec == "alpaca":
        from sisteer.data.prompts import load_hf_prompts
        prompts = load_hf_prompts("tatsu-lab/alpaca", n=int(n or 2000), seed=seed)
        info.update({"kind": "hf_dataset", "dataset": "tatsu-lab/alpaca",
                     "source": "sisteer.data.prompts.load_hf_prompts"})
    else:
        raise RefusalFitError(
            f"{side}: {spec!r} is neither a prompt file nor a builtin id "
            f"{list(BUILTIN_PROMPT_SETS)}")
    seen: set = set()
    uniq = []
    for q in prompts:
        if q not in seen:
            seen.add(q)
            uniq.append(q)
    n_available = len(uniq)
    if n and len(uniq) > n:
        idx = sorted(random.Random(seed).sample(range(len(uniq)), n))
        uniq = [uniq[i] for i in idx]
    if not uniq:
        raise RefusalFitError(f"{side}: {spec!r} resolved to no prompts")
    info.update({"n_available": n_available, "n_used": len(uniq),
                 "prompts_sha256": _sha_text(uniq)})
    return uniq, info


# --------------------------------------------------------------------------- #
# capture                                                                      #
# --------------------------------------------------------------------------- #

def encode(tok, rendered: str):
    """BOS-safe token ids of ONE rendered prompt (no padding)."""
    from sisteer.generate import encode_kwargs
    return tok(rendered, **encode_kwargs(tok, rendered))["input_ids"]


def capture_last_token_means(model, tok, prompts: list[str], *, layers: list[int],
                             batch_size: int, max_len: int):
    """``(means {layer: fp32 [d]}, n_used, n_dropped)``: the residual stream at
    every requested decoder block's OUTPUT at each prompt's LAST token,
    averaged over the prompts. Right-padded batches, the last real token
    gathered per row from the attention mask; prompts longer than
    ``max_len`` tokens are dropped (never truncated)."""
    import torch

    from sisteer.generate import encode_kwargs
    from sisteer.hooks import _hidden_from_output
    from sisteer.models import build_chat_prompt, get_decoder_layers

    device = next(model.parameters()).device
    blocks = get_decoder_layers(model)
    store: dict = {}

    def _hook(layer):
        def fn(_mod, _inp, out):
            store[layer] = _hidden_from_output(out)
        return fn

    handles = [blocks[l].register_forward_hook(_hook(l)) for l in layers]
    prev_side = tok.padding_side
    prev_pad = tok.pad_token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    sums = {l: None for l in layers}
    n_used = 0
    dropped = 0
    try:
        rendered = [build_chat_prompt(tok, q) for q in prompts]
        keep = []
        for r in rendered:
            if len(encode(tok, r)) > max_len:
                dropped += 1
            else:
                keep.append(r)
        for i in range(0, len(keep), batch_size):
            chunk = keep[i:i + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      **encode_kwargs(tok, chunk[0]))
            enc = {k: v.to(device) for k, v in enc.items()}
            store.clear()
            with torch.no_grad():
                model(**enc)
            last = enc["attention_mask"].sum(dim=1) - 1
            rows = torch.arange(len(chunk), device=device)
            for l in layers:
                h = store[l][rows, last, :].float().cpu()      # [B, d]
                s = h.sum(dim=0)
                sums[l] = s if sums[l] is None else sums[l] + s
            n_used += len(chunk)
    finally:
        for h in handles:
            h.remove()
        tok.padding_side = prev_side
        if prev_pad is None:
            tok.pad_token = prev_pad
    if n_used == 0:
        raise RefusalFitError(
            f"no prompt survived the {max_len}-token cap ({dropped} dropped) — "
            "raise --max-len or shorten the set")
    return {l: sums[l] / n_used for l in layers}, n_used, dropped


# --------------------------------------------------------------------------- #
# publication                                                                  #
# --------------------------------------------------------------------------- #

def manifest_problems(out_dir: Path, identity: dict | None = None) -> list[str]:
    """Problems (empty = a valid publication) for an existing ``<out>``: a
    COMPLETE manifest that parses, every file it names present with its
    recorded sha256, and — with ``identity`` — the SAME fit request."""
    marker = out_dir / COMPLETE_NAME
    if not marker.is_file():
        return [f"{marker} missing"]
    try:
        m = json.loads(marker.read_text(encoding="utf-8"))
    except ValueError as e:
        return [f"{marker} unparseable: {e}"]
    problems = []
    files = m.get("files") if isinstance(m.get("files"), dict) else None
    if not files:
        return [f"{marker} carries no files manifest"]
    for name, sha in files.items():
        p = out_dir / str(name)
        if not p.is_file() or sha256_file(p) != sha:
            problems.append(f"{name} missing or does not match its published sha256")
    if identity is not None and m.get("fit_identity") != identity:
        problems.append("a different fit request is published here "
                        f"(published {json.dumps(m.get('fit_identity'), sort_keys=True)[:200]})")
    return problems


#: the publication lock's name beside the publication directory: ``<parent>/.<name>.publish.lock``
PUBLISH_LOCK_SUFFIX = ".publish.lock"


def publication_lock_path(out) -> Path:
    """The lock file ``_publication_lock`` takes for the publication at ``out``: a dot-file beside the directory,
    keyed by the REAL parent directory (two spellings of one path lock one file), never inside the publication
    (COMPLETE lists the publication's files; the lock is not one of them) and never removed (unlinking a lock file
    a waiter has already opened lets a later opener lock a different inode, and two holders stand)."""
    out = Path(out)
    return Path(os.path.realpath(out.parent)) / f".{out.name}{PUBLISH_LOCK_SUFFIX}"


#: the publication locks THIS thread holds, ``{lock path: depth}`` — ``_publication_lock`` re-enters one its holder already
#: holds instead of opening a second descriptor (``flock`` locks conflict between open file descriptions, the same
#: process's included, so a second ``os.open`` + ``LOCK_EX`` by the holder would wait on itself forever). Per THREAD, not
#: per process: two fits are two processes on the cluster and two threads in the tests, and both must contend.
_HELD = threading.local()


def _held_locks() -> dict:
    held = getattr(_HELD, "locks", None)
    if held is None:
        held = _HELD.locks = {}
    return held


def publication_lock_held(out) -> bool:
    """Whether THIS thread holds ``_publication_lock`` for the publication at ``out`` right now (the emission paths
    require it; the concurrency test asserts it at the hook)."""
    return str(publication_lock_path(out)) in _held_locks()


def _require_publication_lock(out: Path, what: str) -> None:
    """A step that must run under the publication lock (``_kept``, the fresh publication's emission) refuses loudly
    when its caller does not hold it — the race the lock closes is silent, so an unlocked emission is never let
    through to find out (authoritative mechanisms fail loudly, never degrade)."""
    if not publication_lock_held(out):
        raise RefusalFitError(f"{what} for {out} called without the publication lock ({publication_lock_path(out)}) — "
                              "the registry emission runs under it, or two jobs of one identity can each state the object")


@contextlib.contextmanager
def _publication_lock(out: Path):
    """An exclusive advisory ``flock`` on ``publication_lock_path(out)``, held across the WHOLE of ``_publish``:
    the rename, the reading of a populated ``out`` and the clearing of a torn one are one critical section with
    respect to every other publisher of this path. Without it two jobs of one identity meeting one torn directory
    could each read it torn, the first clear it and publish, and the second — acting on its stale verdict —
    remove the first's publication and rename its own in (two objects of one identity and rung, which the pins
    writer refuses; or the first's emission reading a directory being removed), or, paused between its two reads,
    refuse the first's identical publication as a DIFFERENT request.

    ``fit`` holds the SAME lock from its publish-or-keep decision through the registry emission and its evidence
    write: released between the two, the job that lost ``_publish`` entered
    ``_kept`` at once, read no evidence and an empty spool, and passed the registry hook's read-then-write duplicate
    guard beside the winner's own emission — two native ``object`` statements of one id. Re-entrant for its holder
    (``_HELD``; ``_publish`` takes it again inside ``fit``'s hold), so the pins the earlier rounds set on ``_publish``
    stand. Blocking (the holder's work is a rename, a few sha256 reads, at most one rmtree, and the two record
    writes of one emission — a waiter is a job of the same identity whose remaining work is to be told "kept");
    released by the kernel with the holder's death. A filesystem that cannot lock refuses by name, this fit's
    bytes still at ``tmp``."""
    path = publication_lock_path(out)
    held = _held_locks()
    key = str(path)
    if key in held:                                   # re-entered by its holder: the one descriptor stays locked
        held[key] += 1
        try:
            yield
        finally:
            held[key] -= 1
        return
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
    except OSError as e:
        os.close(fd)
        raise RefusalFitError(f"{path}: cannot take the publication lock ({e.strerror}) — the publication needs "
                              "flock on this filesystem; nothing was published and nothing removed") from e
    held[key] = 1
    try:
        yield
    finally:
        del held[key]
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _publish(tmp: Path, out: Path, identity: dict, *, overwrite: bool) -> bool:
    """The publication event: ONE rename of the finished ``tmp`` onto ``out`` — which the kernel REFUSES
    (EEXIST / ENOTEMPTY) while ``out`` is a populated directory, so nothing is ever renamed over a publication
    this process has not read. True when ``tmp`` became the publication. False when ``out`` holds a valid
    publication of the SAME request: another job of this identity published while this one fitted (the front
    door's queue read keys on the identity's job name, but two front doors can read the queue empty in the same
    instant, and a hand submission reads nothing) — its publication is kept, ``tmp`` discarded, and the caller
    takes the kept path (its emission verified, never a second object of the identity). A valid publication of
    ANOTHER request refuses unless ``overwrite`` — the rule the pre-fit read applies, here at the write moment; a
    torn ``out`` (no valid COMPLETE) is cleared and the rename made again. An empty ``out`` is no publication: the
    rename replaces it. The whole of it runs under ``_publication_lock``: the verdict on a populated ``out`` and
    the clearing it leads to are one step no other publisher of this path interleaves, so a publication that
    landed after this one read is never removed — the next read, under the lock, finds it and keeps it. A ``tmp``
    that is never published is removed here; one this function raises on (a refused overwrite) is removed too, and
    one the lock refuses on stands. ``fit`` calls this already holding the same lock (re-entered here) and keeps
    holding it through the emission that follows."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with _publication_lock(out):
        while True:
            try:
                os.rename(tmp, out)
                return True
            except OSError as e:
                if e.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
            same = manifest_problems(out, identity)
            if not same:
                shutil.rmtree(tmp)
                return False
            valid_other = not manifest_problems(out)
            if valid_other and not overwrite:
                shutil.rmtree(tmp)
                raise RefusalFitError(
                    f"{out} holds a valid publication of a DIFFERENT request, published while this fit ran — refusing to "
                    "overwrite it (pass --overwrite to supersede it):\n  - " + "\n  - ".join(same))
            print(f"[refusal-direction] {out}: {'superseding (--overwrite)' if valid_other else 'torn publication'} at the "
                  "write moment — replacing", file=sys.stderr, flush=True)
            shutil.rmtree(out)


def _temp_dir(out: Path) -> Path:
    """This job's own temporary directory beside the publication, named by HOST and pid: two jobs of one identity run on
    two nodes, where one pid can be live twice, and a pid-only name let the second clear the first's half-written tree (a
    stale one of this host and pid is a dead earlier attempt's). One function so the concurrency test, whose two fits are
    two THREADS of one pid, can give each its own."""
    return out.parent / f"{out.name}.tmp-{socket.gethostname()}-{os.getpid()}"


def _kept(out: Path, args, why: str) -> int:
    """The kept path — a COMPLETE publication of this exact request stands at ``out`` and nothing is refitted: say
    so; give it the identity sidecar when it predates one (ensure_fit_provenance, listed in COMPLETE; gap-refusal-pin);
    then, unless ``--no-emit``, verify its registry emission and replay what is missing (a kept publication is not
    a recorded one) — never a second object. Runs UNDER the publication lock the caller holds
    (``_require_publication_lock``): the verification reads the winner's evidence, else its spool records, only
    once the winner's emission has finished, so what it finds is final."""
    _require_publication_lock(out, "the kept path")
    print(f"[refusal-direction] {out} {why} (object_sha256 "
          f"{json.loads((out / COMPLETE_NAME).read_text())['object_sha256'][:16]}…)", flush=True)
    ensure_fit_provenance(out)
    if args.no_emit:
        return 0
    ev = ensure_registry_emission(out, campaign=args.campaign)
    return EMIT_FAILED_RC if ev.get("status") == "failed" else 0


def _registry_model_id(name):
    """The registry ``model_id`` for the served checkpoint (an HF repo)
    through decode_policy.json's ``model_spellings``; None when nothing maps
    it (the hook keeps the sidecar's spelling; nothing is guessed)."""
    if not name:
        return None
    try:
        from sisteer.registry.decode_policy import PolicyFile
        pf = PolicyFile.load()
    except Exception:  # noqa: BLE001 — no export on this host
        return None
    mid = pf.model_id(str(name))
    return mid if (mid != str(name) or mid in pf.models) else None


def emission_evidence(out_dir: Path) -> dict | None:
    """The evidence beside COMPLETE, or None when absent / unparseable (registry_emission.read_evidence)."""
    return _emission().read_evidence(out_dir)


def ensure_registry_emission(out_dir: Path, *, campaign: str) -> dict:
    """Emission as a REPLAYABLE, idempotent step: on a kept COMPLETE
    publication, verify the native ``object`` row was emitted (the evidence keyed by the marker's object id;
    else the object AND artifact records for that id already in the producer spool) and RE-EMIT from the published
    ``fit_meta.json`` what is missing — the whole hook, or only the missing record. Never
    refits (registry_emission.ensure_emitted)."""
    out_dir = Path(out_dir)
    mod = _emission()
    meta = json.loads((out_dir / META_NAME).read_text())
    oid = mod.marker_object_id(out_dir) or meta.get("object_sha256")
    ev = mod.ensure_emitted(out_dir, object_id=oid, writer=SCRIPT, campaign=campaign,
                            emit=lambda: _hook_call(out_dir, meta, campaign), what="direction",
                            # a spool holding the object row but not the artifact
                            # row replays ONLY the artifact (never a second object row)
                            emit_component=lambda c: _hook_call(out_dir, meta, campaign, emitter=mod.component_emitter(c)))
    if ev.get("status") == "emitted":
        print(f"registry: object {oid}", flush=True)
    return ev


def _hook_call(out_dir: Path, meta: dict, campaign: str, emitter=None):
    """The call into the fit writers' hook (``hooks.emit_object_from_sidecar``); raises on any failure of its own
    inputs, returns what the hook returns (paths / [] off / None = the guarded hook swallowed a failure).
    ``emitter`` is a callable with the hook's signature that writes ONE of its
    two records — ``registry_emission.component_emitter(<component>)`` — for the partial replay; default the hook."""
    from sisteer.registry import hooks as _rh
    fn = emitter if emitter is not None else _rh.emit_object_from_sidecar
    sets = meta.get("prompt_sets") or {}
    stated = {"fit_model_id": meta.get("model_key") or _registry_model_id(meta.get("model")),
              "fit_layers_json": meta.get("layers"),
              "fit_capture_site": meta.get("capture_site"),
              "fit_contrast_sets_json": ({side: "sha256:" + sets[side]["prompts_sha256"]
                                          for side in ("harmful", "harmless")}
                                         if all(sets.get(sd, {}).get("prompts_sha256")
                                                for sd in ("harmful", "harmless")) else None)}
    extra = {k: meta.get(k) for k in (
        "coefficient", "dose_unit", "staged_id", "paper_arm", "n_harmful", "n_harmless",
        "n_dropped_too_long", "prompt_set_ids", "prompt_sets", "direction_sha256",
        "direction_pt_sha256", "layer_norms", "direction_norms", "capture_site",
        "capture_position", "skip_bos", "rendering", "model_revision", "max_len")}
    kwargs = {k: v for k, v in stated.items() if v is not None}
    extra = {k: v for k, v in extra.items() if v is not None}
    return fn(
        str(out_dir), campaign=campaign, object_kind=KIND,
        paper_label=_rh.paper_label_for(KIND, OBJECT_ROLE), object_role=OBJECT_ROLE,
        fit_target_kind=FIT_TARGET_KIND, fit_objective=FIT_OBJECTIVE,
        derivation=DERIVATION, script=SCRIPT, extra=extra or None, **kwargs)


def registry_emit(out_dir: Path, meta: dict, *, campaign: str) -> dict:
    """The ``object`` + ``artifact`` records: kind ``vector_all``,
    role ``baseline_published`` (-> ``arditi_allpos``; vocab.OBJECT_ROLES has
    no `refusal` role — review C P1), target kind ``contrastive_direction``,
    objective ``diffmean``, derivation ``none``; the two prompt sets' content
    shas as ``fit_contrast_sets_json`` and the block output as
    ``fit_capture_site`` (the hook's own columns). Never raises — and never
    silent: returns the EVIDENCE it wrote beside
    COMPLETE (``emitted`` with the record paths, ``not_emitted`` with the
    reason, ``failed`` with the error + the REGISTRY_EMIT_FAILED marker —
    registry_emission.emit_with_evidence); prints ``registry: object
    <sha256>`` when a record was filed."""
    out_dir = Path(out_dir)
    ev = _emission().emit_with_evidence(
        out_dir, object_id=meta.get("object_sha256"), writer=SCRIPT, campaign=campaign,
        emit=lambda: _hook_call(out_dir, meta, campaign), what="direction")
    if ev.get("status") == "emitted":
        print(f"registry: object {meta['object_sha256']}", flush=True)
    return ev


def fit(args) -> int:
    import torch

    from sisteer.extraction.multilayer import MultiLayerVectorObject
    from sisteer.models import (chat_template_hash, load_model_and_tokenizer,
                                num_layers)

    out = Path(args.out)
    sid = staged_id(args.coefficient)
    harmful, h_info = resolve_prompt_set(args.harmful, side="harmful", n=args.n, seed=args.seed)
    harmless, b_info = resolve_prompt_set(args.harmless, side="harmless", n=args.n, seed=args.seed)
    identity = {"model": args.model, "revision": args.revision, "coefficient": args.coefficient,
                "positions": args.positions, "layers": args.layers,
                "harmful": {"id": args.harmful, "prompts_sha256": h_info["prompts_sha256"]},
                "harmless": {"id": args.harmless, "prompts_sha256": b_info["prompts_sha256"]},
                "n": args.n, "seed": args.seed, "max_len": args.max_len, "dtype": args.dtype}
    if out.is_dir():
        # the pre-fit read and the kept path it may take run under the publication lock: a publication
        # another job is still emitting for is read — and its emission verified or replayed — only once that job has released it
        with _publication_lock(out):
            same = manifest_problems(out, identity)
            # gap-refusal-pin: a LISTED identity sidecar that is torn or altered is the one manifest problem a
            # publication can repair from its verified meta — the bytes the marker promised, never a refit; anything else torn rebuilds below
            if same == [f"{PROVENANCE_NAME} missing or does not match its published sha256"] and repair_fit_provenance(out):
                same = manifest_problems(out, identity)
            if not same:
                return _kept(out, args, "already holds a COMPLETE publication of this exact request — kept")
            valid_other = not manifest_problems(out)
            if valid_other and not args.overwrite:
                raise RefusalFitError(
                    f"{out} holds a valid publication of a DIFFERENT request — refusing to overwrite it "
                    "(pass --overwrite to supersede it):\n  - " + "\n  - ".join(same))
            print(f"[refusal-direction] {out}: {'superseding (--overwrite)' if valid_other else 'torn earlier publication'} — rebuilding",
                  file=sys.stderr)
        # the lock is released here: the fit itself (model load, capture) never runs under it

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model_and_tokenizer(args.model, device=device, dtype=args.dtype,
                                          revision=args.revision)
    n_layers = num_layers(model)
    layers = (list(range(n_layers)) if args.layers == "all"
              else sorted({int(x) for x in args.layers.split(",") if x.strip()}))
    bad = [l for l in layers if l < 0 or l >= n_layers]
    if bad:
        raise RefusalFitError(f"layers {bad} are outside this model's {n_layers} decoder blocks")

    print(f"[refusal-direction] {args.model}@{(args.revision or 'unpinned')[:12]}: {len(harmful)} harmful "
          f"({args.harmful}) vs {len(harmless)} harmless ({args.harmless}); {len(layers)} layers; c={args.coefficient}",
          flush=True)
    mean_h, n_h, drop_h = capture_last_token_means(model, tok, harmful, layers=layers,
                                                   batch_size=args.batch_size, max_len=args.max_len)
    mean_b, n_b, drop_b = capture_last_token_means(model, tok, harmless, layers=layers,
                                                   batch_size=args.batch_size, max_len=args.max_len)
    raw = {l: (mean_h[l] - mean_b[l]).float().contiguous() for l in layers}
    stored = {l: (args.coefficient * raw[l]).float().contiguous() for l in layers}

    obj = MultiLayerVectorObject(model, layers=layers, positions=args.positions, skip_bos=False)
    for l in layers:
        obj.params[l].data = stored[l].clone()

    live_rev = getattr(getattr(model, "config", None), "_commit_hash", None) or None
    revision = str(live_rev) if live_rev else args.revision
    rev_source = "live_model" if live_rev else ("requested" if args.revision else None)
    direction_sha = hashlib.sha256(
        torch.cat([raw[l] for l in layers]).numpy().tobytes()).hexdigest()

    tmp = _temp_dir(out)                      # this job's own temporary beside the publication (host + pid; a stale one is a dead earlier attempt's)
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    obj.save(tmp / OBJECT_NAME)
    torch.save({"kind": "arditi_diff_in_means_raw", "layers": [int(l) for l in layers],
                "vectors": {str(l): raw[l] for l in layers},
                "capture_site": CAPTURE_SITE, "capture_position": CAPTURE_POSITION,
                "note": "the RAW harmful - harmless difference of means per layer; the served "
                        "object is coefficient * this (dose unit raw_coefficient)"},
               tmp / DIRECTION_NAME)
    object_sha = sha256_file(tmp / OBJECT_NAME)
    meta = {
        "kind": KIND, "derivation": DERIVATION, "fit_target_kind": FIT_TARGET_KIND,
        "object_role": OBJECT_ROLE, "fit_objective": FIT_OBJECTIVE, "paper_arm": PAPER_ARM,
        "staged_id": sid, "cell_id": sid,
        "coefficient": args.coefficient, "dose_unit": DOSE_UNIT,
        "positions": args.positions, "skip_bos": False,
        "positions_at_fit": "all", "fit_positions": "all",
        "capture_site": CAPTURE_SITE, "capture_position": CAPTURE_POSITION,
        "n_harmful": n_h, "n_harmless": n_b,
        "n_dropped_too_long": {"harmful": drop_h, "harmless": drop_b},
        "prompt_set_ids": {"harmful": args.harmful, "harmless": args.harmless},
        "prompt_sets": {"harmful": h_info, "harmless": b_info},
        "model": args.model, "model_id": args.model, "model_key": args.model_key,
        "revision": revision, "model_revision": revision, "revision_requested": args.revision,
        "revision_source": rev_source,
        "template_hash": chat_template_hash(tok),
        "rendering": "sisteer.models.build_chat_prompt (chat template, no system prompt)",
        "layers": [int(l) for l in layers],
        "layer_norms": {str(l): float(stored[l].norm()) for l in layers},
        "direction_norms": {str(l): float(raw[l].norm()) for l in layers},
        "direction_sha256": direction_sha,
        "direction_pt_sha256": sha256_file(tmp / DIRECTION_NAME),
        "object_sha256": object_sha,
        "seed": args.seed, "max_len": args.max_len, "batch_size": args.batch_size,
        "dtype": args.dtype, "device": device,
        "fit_identity": identity,
        "built_at": _now(), "builder": SCRIPT, "code_sha": _code_sha(),
    }
    (tmp / META_NAME).write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    # the identity sidecar (gap-refusal-pin): the same meta projected into the stage-3 record the stager and the gate read
    (tmp / PROVENANCE_NAME).write_text(json.dumps(fit_provenance_record(meta), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    files = {n: sha256_file(tmp / n) for n in PUBLISHED_FILES}
    marker = {"kind": KIND, "staged_id": sid, "cell_id": sid, "object_file": OBJECT_NAME,
              "object_sha256": object_sha, "files": files, "fit_identity": identity,
              "published_at": _now()}
    (tmp / COMPLETE_NAME).write_text(json.dumps(marker, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    # ONE hold of the publication lock from the publish-or-keep decision through the registry emission: the job that
    # loses _publish verifies the winner's emission only after the winner has written its records and its evidence,
    # and the winner's hook never shares its read-then-write duplicate guard with a second process of this publication
    out.parent.mkdir(parents=True, exist_ok=True)
    with _publication_lock(out):
        if not _publish(tmp, out, identity, overwrite=args.overwrite):
            # another job of this identity published while this one fitted: its publication stands, this fit's bytes are gone, and the
            # kept path verifies ITS emission — one object of the identity, whichever job the queue read missed
            return _kept(out, args, "was published by another job of this identity while this fit ran — kept, this fit's bytes discarded")
        print(f"[refusal-direction] published {out} (object_sha256 {object_sha[:16]}…, "
              f"{n_h}+{n_b} prompts, {drop_h + drop_b} dropped over {args.max_len} tokens, "
              f"||c·r|| per layer min {min(meta['layer_norms'].values()):.4g} max "
              f"{max(meta['layer_norms'].values()):.4g})", flush=True)
        if not args.no_emit:
            _require_publication_lock(out, "the fresh publication's emission")
            ev = registry_emit(out, meta, campaign=args.campaign)
            if ev.get("status") == "failed":
                # the publication stands; the JOB fails, visibly
                return EMIT_FAILED_RC
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="HF id of the SERVED checkpoint")
    ap.add_argument("--revision", default=None, help="40-hex snapshot pin")
    ap.add_argument("--out", required=True, help="the object dir (object.pt, direction.pt, fit_meta.json, COMPLETE)")
    ap.add_argument("--harmful", required=True, help=f"prompt file or builtin id ({', '.join(BUILTIN_PROMPT_SETS)})")
    ap.add_argument("--harmless", required=True, help="prompt file or builtin id")
    ap.add_argument("--coefficient", type=float, default=0.3,
                    help="c: the served vector is c * r_raw (raw_coefficient dose unit; the paper battery's 0.3)")
    ap.add_argument("--positions", default="all", choices=["all"],
                    help="serve-time positions of the object (the arditi_allpos arm: all)")
    ap.add_argument("--layers", default="all", help="'all' or a comma list of decoder block indices")
    ap.add_argument("--n", type=int, default=128, help="cap per side (seeded sample); 0 = every prompt")
    ap.add_argument("--max-len", type=int, default=512, help="prompts longer than this in tokens are DROPPED")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda when available)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-key", default=None, help="the registry model_id of the served checkpoint (fit_model_id)")
    ap.add_argument("--campaign", default=os.environ.get("REGISTRY_CAMPAIGN", "two_si"))
    ap.add_argument("--overwrite", action="store_true", help="supersede a valid publication of a different request")
    ap.add_argument("--no-emit", action="store_true", help="skip the registry record")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.n is not None and args.n <= 0:
        args.n = None
    try:
        return fit(args)
    except RefusalFitError as e:
        print(f"[refusal-direction] REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

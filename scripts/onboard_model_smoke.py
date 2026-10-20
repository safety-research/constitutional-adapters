#!/usr/bin/env python
"""Per-model onboarding gate.

Runs EVERY check and prints one [PASS]/[FAIL]/[SKIP]/[KNOWN-LIMIT] line per
check, exiting 1 iff any check FAILED — a gate script must show the whole
picture, not stop at the first miss.

The model is named EITHER by a positional slug OR by ``--model-config
<per-model config>`` (the pilot's spelling — review fix F10): the config's
``model``/``model_revision`` pins are resolved against MODEL_SPECS, and a
config pinning an unknown model fails loudly (the onboarding facts —
layer/hidden counts, cache limits — exist only per spec). Three tiers via
``--mode``:

- ``--mode tokenizer-only`` (the grading host; ``--tokenizer-only`` is the legacy
  alias): tokenizer + config asserts, NO weights. Includes the accessor
  checks on a META-device instantiation of the real class (structure only
  — from_config downloads nothing).
- ``--mode full`` (the default; the pilot pod, step 2): everything above
  plus the real weight load under the locked transformers + sdpa, accessor
  checks on the loaded model, the prefix-cache logits-equivalence check on
  a synthetic long prefix, and AppendedSoftTokenObject construction.
- ``--mode vllm`` (pod-only; the pilot's step 3, run inside the DEDICATED
  vllm venv): ONLY the vLLM load + system-role serve gate. The
  lockfile-pinned transformers/weight checks deliberately do NOT run in
  this mode — the vllm venv overlays its own torch/transformers (so the
  version pin would fail spuriously) and full mode would double-load the
  weights. ``--vllm`` (the flag) stays as the ADDITIVE spelling on top of
  tokenizer-only/full for single-venv pods; vllm is imported lazily and a
  missing install SKIPs cleanly (the grading host venv has no vLLM on purpose).

KNOWN-LIMIT is for the three measured hybrid-cache failures (qwen3_5's
LinearAttentionLayer entries; gemma4's sliding-window crop past 512;
nemotron_h's "mlp" layer type, which DynamicCache cannot allocate at all):
when the model's derived config already records ``teacher_prefix_cache:
false``, the equivalence check failing WITH one of those signatures is the
expected state, not a gate failure — it flips back to FAIL the moment the
config says true. The config-only "KV cache allocatable" check reports the
nemotron_h case without weights, and "generation stop ids" checks that the
chat template's turn terminator is an effective ``generate()`` stop
(Nemotron's generation_config.json says eos 2 while its template ends
every turn with id 11 — models.pin_generation_stop_ids closes that at
load; the check names whether it had to).

INFO is the fourth soft state: a recorded fact that is neither a pass nor
a failure. The full tier's "Mamba2 kernel path" row uses it — on a Mamba2
hybrid it names which kernels the SSM blocks resolved (kernels-hub / pip
mamba_ssm + causal_conv1d / none): PASS when the fused path is on, INFO
when the model fell back to the pure-torch SSD scan (same math, 9-14x the
step time), SKIP on architectures without Mamba2 mixers. The row is how a
container proves which path it trains on, without reading its stderr.

Usage:
    uv run --no-sync python scripts/onboard_model_smoke.py gemma4e4b --tokenizer-only
    uv run --no-sync python scripts/onboard_model_smoke.py qwen38_27b            # pod
    uv run --no-sync python scripts/onboard_model_smoke.py qwen38_27b --vllm     # pod
    uv run --no-sync python scripts/onboard_model_smoke.py \
        --model-config configs/two_si_gemma4e4b.yaml --mode full   # pilot step 2
    UV_PROJECT_ENVIRONMENT=<vllm venv> uv run --no-sync python \
        scripts/onboard_model_smoke.py --model-config <cfg> --mode vllm  # step 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from derive_model_config import (ATTACK_ALLOWANCE, COMPLETION_CAP, EOS,
                                 MODEL_SPECS, REPO, SDFBASE_SLUGS, ModelSpec,
                                 load_base_config, measure_si_prompt_tokens)

EXPECTED_TRANSFORMERS = "5.15.0"

# The measured hybrid-cache failure signatures (2026-08-25, CPU tiny
# real-class instances) — see tests/test_model_onboarding.py, which pins
# the never-silently-wrong contract behind them.
_KNOWN_CACHE_LIMITS = (
    "LinearAttentionLayer",          # qwen3_5: no .keys/.values at any P
    "must match the size of tensor",  # gemma4: sliding-window crop, P>512
    # nemotron_h (Nemotron-3-Nano-4B): DynamicCache(config=...) looks each
    # config.layer_types entry up in DYNAMIC_LAYER_TYPE_MAPPING, which has
    # no "mlp" — KeyError('mlp') at cache construction, before the first
    # block, at ANY prefix length (measured 2026-09-02 on a tiny real-class
    # instance; the real config carries 17 "mlp" entries). str(KeyError) is
    # the quoted key.
    "'mlp'",
)


class Gate:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))
        print(f"[{status}] {name}" + (f": {detail}" if detail else ""),
              flush=True)

    def run(self, name: str, fn) -> None:
        """fn returns a detail string (PASS), or raises (FAIL), or returns
        a ('SKIP'|'KNOWN-LIMIT'|'INFO', detail) tuple for the soft states
        (INFO = a recorded fact that is neither a pass nor a failure: the
        Mamba2 kernel-path row uses it for the torch fallback)."""
        try:
            got = fn()
        except Exception as e:  # noqa: BLE001 — a gate reports, then fails
            self.add("FAIL", name, f"{type(e).__name__}: {e}")
            return
        if isinstance(got, tuple) and got and got[0] in ("SKIP",
                                                         "KNOWN-LIMIT",
                                                         "INFO"):
            self.add(got[0], name, got[1] if len(got) > 1 else "")
        else:
            self.add("PASS", name, got or "")

    @property
    def failed(self) -> bool:
        return any(s == "FAIL" for s, _, _ in self.rows)


def _load_derived_config(cfg_path: Path | None) -> dict | None:
    """The per-model derived config the run should be checked against: the
    ``--model-config`` file when one was given (that IS the config the
    pilot runs with), else the spec's committed config_path."""
    if cfg_path is None or not cfg_path.exists():
        return None
    import yaml

    with open(cfg_path) as fh:
        return yaml.safe_load(fh)


def _spec_committed_config(spec: ModelSpec) -> Path:
    """Where a spec's COMMITTED config lives: the generated
    sdfbase_<slug>_fits.yaml for the sdf-base-steering entries
    (make_sdfbase_configs.py), the derived two_si_<slug>.yaml
    (spec.config_path) for everything else."""
    if spec.slug in SDFBASE_SLUGS:
        return REPO / "configs" / f"sdfbase_{spec.slug}_fits.yaml"
    return spec.config_path


def _resolve_spec_from_config(cfg_path: Path) -> ModelSpec:
    """--model-config -> ModelSpec, by the config's model/model_revision
    pins. A config pinning an unknown (model, revision) fails LOUDLY — the
    onboarding facts (layer/hidden counts, hybrid-cache limits) exist only
    per MODEL_SPECS entry, so guessing would gate against nothing.

    One checkpoint can serve TWO lanes under two slugs (GLM-4.6V-Flash:
    glm46vflash for the two_si sweep, glm46vflash_doc for the doc-CE
    campaign; likewise nemotron3nano4b / nemotron3nano4b_doc — same pin,
    same onboarding facts): prefer the entry whose committed config IS the
    file given, then the entry whose canonical lane file NAME the config
    kept, so the banner names the lane being gated; an off-tree copy under
    a lane-less name REFUSES rather than guessing (the doc-mode config
    silently gated against the two_si spec otherwise)."""
    import yaml

    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    model = cfg.get("model")
    revision = cfg.get("model_revision")
    matches = [s for s in MODEL_SPECS.values()
               if s.model == model and s.revision == revision]
    if len(matches) == 1:
        return matches[0]
    if matches:
        # Two entries can pin the SAME checkpoint (glm46vflash /
        # glm46vflash_doc; nemotron3nano4b / nemotron3nano4b_doc — no base
        # variant exists for either). The pins cannot tell them apart; the
        # config FILE can: first the entry whose COMMITTED config IS the
        # file given, then the canonical file name (two_si_<slug>.yaml
        # / sdfbase_<slug>_fits.yaml) for an off-tree copy that kept it.
        # Anything else REFUSES — a first-match fallback would silently
        # gate the documents config against the two-SI spec.
        want = Path(cfg_path).resolve()
        for spec in matches:
            if _spec_committed_config(spec).resolve() == want:
                return spec
        by_name = [s for s in matches
                   if Path(cfg_path).name in (s.config_path.name,
                                              f"sdfbase_{s.slug}_fits.yaml")]
        if len(by_name) == 1:
            return by_name[0]
        raise SystemExit(
            f"[onboard] {cfg_path} pins model={model!r} revision="
            f"{revision!r}, which matches {len(matches)} ModelSpecs "
            f"({[s.slug for s in matches]}), and its file name settles none "
            "of them — name the config after its lane (two_si_<slug>.yaml "
            "or sdfbase_<slug>_fits.yaml) so the onboarding facts gate the "
            "right lane.")
    known = {s.slug: f"{s.model} @ {s.revision}" for s in MODEL_SPECS.values()}
    raise SystemExit(
        f"[onboard] {cfg_path} pins model={model!r} revision={revision!r}, "
        f"which matches no known ModelSpec — the onboarding asserts exist "
        f"only for {known}. Wrong config, or a new model that needs a "
        "MODEL_SPECS entry (scripts/derive_model_config.py) first.")


def check_transformers_version() -> str:
    import transformers

    if transformers.__version__ != EXPECTED_TRANSFORMERS:
        raise AssertionError(
            f"transformers {transformers.__version__} != the locked "
            f"{EXPECTED_TRANSFORMERS} — the onboarding facts (auto-class "
            "mapping, cache layout, template kwargs) are version-specific")
    return f"transformers {transformers.__version__}"


def check_config(spec: ModelSpec) -> str:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(spec.model, revision=spec.revision)
    text = cfg.get_text_config()
    n, h = text.num_hidden_layers, text.hidden_size
    assert n == spec.n_layers, f"num_hidden_layers {n} != {spec.n_layers}"
    assert h == spec.hidden_size, f"hidden_size {h} != {spec.hidden_size}"
    return (f"{cfg.model_type}, arch {getattr(cfg, 'architectures', None)}, "
            f"{n} layers, hidden {h}")


def check_meta_accessors(spec: ModelSpec) -> str:
    """get_decoder_layers / hidden_size dispatch on the REAL class, meta
    device — structure only, zero weight bytes. The class comes from
    models.auto_model_class — the SAME dispatch load_model_and_tokenizer
    uses — not a hard-coded AutoModelForCausalLM: glm4v (GLM-4.6V-Flash)
    is registered only under AutoModelForImageTextToText in transformers
    5.14.1, and the causal-LM auto-class raises "Unrecognized
    configuration class" on it (measured 2026-09-02), which would fail the
    gate on a model the loader handles fine."""
    import torch
    from transformers import AutoConfig

    from sisteer.models import auto_model_class, get_decoder_layers, hidden_size

    cfg = AutoConfig.from_pretrained(spec.model, revision=spec.revision)
    auto_cls = auto_model_class(spec.model, revision=spec.revision)
    with torch.device("meta"):
        model = auto_cls.from_config(cfg)
    n = len(get_decoder_layers(model))
    h = hidden_size(model)
    cls = type(model).__name__
    assert n == spec.n_layers, f"get_decoder_layers found {n} != {spec.n_layers}"
    assert h == spec.hidden_size, f"hidden_size {h} != {spec.hidden_size}"
    return f"{cls} (via {auto_cls.__name__}): {n} layers, hidden {h}"


def _tok(spec: ModelSpec):
    from sisteer.models import load_tokenizer

    return load_tokenizer(spec.model, revision=spec.revision)


def check_system_role(tok) -> str:
    """Which branch renders a system message: the template directly, or
    apply_chat_template_folding's fold-into-first-user-turn fallback
    (models.py) — record WHICH, and that the system text actually lands."""
    from sisteer.models import apply_chat_template_folding

    marker = "SMOKE_SYSTEM_MARKER_93f1"
    msgs = [{"role": "system", "content": marker},
            {"role": "user", "content": "hi"}]
    kwargs = getattr(tok, "_sisteer_template_kwargs", {})
    try:
        rendered = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, **kwargs)
        branch = "direct (template accepts the system role)"
    except Exception as e:
        rendered = apply_chat_template_folding(tok, msgs)
        branch = (f"fold (template rejected system: {type(e).__name__}; "
                  "models.py:182-191 branch fired)")
    assert marker in rendered, "system text absent from the render"
    return branch


def check_header_len(tok) -> str:
    from sisteer.hooks import response_header_len
    from sisteer.models import apply_chat_template_folding

    h = response_header_len(tok)
    assert h > 0, ("response_header_len == 0: neither the "
                   "add_generation_prompt diff nor the marker fallback "
                   "(hooks.py) found the assistant header")
    msgs = [{"role": "user", "content": "x"}]
    full = apply_chat_template_folding(tok, msgs, add_generation_prompt=True)
    head = apply_chat_template_folding(tok, msgs, add_generation_prompt=False)
    via = "diff" if full != head else "marker fallback"
    return f"H = {h} (via {via})"


def check_thinking_pin(tok, spec: ModelSpec) -> str:
    template = getattr(tok, "chat_template", "") or ""
    has = "enable_thinking" in template
    pinned = getattr(tok, "_sisteer_template_kwargs", None)
    if has:
        assert pinned == {"enable_thinking": False}, (
            "template carries enable_thinking but load_tokenizer did NOT "
            "pin it off (models.py:126-127) — thinking tokens would "
            "pollute completions and token alignment")
        return "enable_thinking present in template; pinned False"
    if spec.slug == "qwen38_27b":
        raise AssertionError(
            "Qwen3.8's template does NOT contain 'enable_thinking' — the "
            "models.py substring pin silently no-ops. Find the flag's real "
            "spelling in tok.chat_template and extend the check in "
            "models.py (minimally).")
    return "no enable_thinking in template (nothing to pin)"


def check_si_counts(tok, spec: ModelSpec, derived: dict | None) -> str:
    """Print the measured per-SI teacher-prompt token counts + the
    derive_max_len arithmetic — the numbers config derivation consumes.
    Against a two_si-derived config the committed max_len_by_si must EQUAL
    the live re-derivation. A doc-CE config (data_mode: documents — the
    sdf-base-steering lane) carries no per-SI teacher budget by design
    (flat documents under one sequence_max_len; no SI is ever rendered),
    so there is nothing to compare: the counts are still printed and the
    detail names that case. A config with NEITHER key still fails hard."""
    from sisteer.eval.train_budget import derive_max_len

    counts = measure_si_prompt_tokens(tok)
    lines = []
    for si, n in counts.items():
        need = n + ATTACK_ALLOWANCE + COMPLETION_CAP + EOS
        got = derive_max_len(n, COMPLETION_CAP,
                             attack_allowance=ATTACK_ALLOWANCE, eos=EOS)
        lines.append(f"{si}={n} (-> {need} -> max_len {got})")
    detail = "; ".join(lines)
    if derived is not None:
        if (derived.get("max_len_by_si") is None
                and derived.get("data_mode") == "documents"):
            return detail + (" [doc-mode config: no max_len_by_si to "
                             "compare; sequence_max_len="
                             f"{derived.get('sequence_max_len')}]")
        want = {si: derive_max_len(n, COMPLETION_CAP,
                                   attack_allowance=ATTACK_ALLOWANCE,
                                   eos=EOS)
                for si, n in counts.items()}
        assert derived.get("max_len_by_si") == want, (
            f"derived config max_len_by_si {derived.get('max_len_by_si')} "
            f"!= live re-derivation {want} — regenerate with "
            "scripts/derive_model_config.py")
        detail += " [matches committed config]"
    return detail


def check_sentinel(tok):
    """find_sentinel (extraction/objects.py) — the token_faithful /
    token_bos arms need a single-token sentinel that never occurs in
    natural text. On failure, name the atomic added-token candidates so
    extending _SENTINEL_PREFS is a lookup, not an investigation."""
    from sisteer.extraction.objects import find_sentinel

    try:
        text, tid = find_sentinel(tok)
        return f"sentinel {text!r} -> id {tid}"
    except Exception:
        cands = []
        for token_id, at in sorted(tok.added_tokens_decoder.items()):
            ids = tok(at.content, add_special_tokens=False)["input_ids"]
            if len(ids) == 1 and ids[0] == token_id:
                cands.append(at.content)
        raise AssertionError(
            "no _SENTINEL_PREFS entry is a single token on this tokenizer "
            "— token_faithful/token_bos cannot resolve a sentinel. Atomic "
            f"added-token candidates: {cands[:20]}")


def check_kv_cache_allocatable(spec: ModelSpec, derived: dict | None):
    """Can transformers allocate a KV cache for this architecture at all?
    Config-only, no weights. NemotronH under transformers 5.14.1:
    config.layer_types carry "mlp", which DynamicCache has no layer for, so
    EVERY cached forward — the default eval forward (use_cache=None ->
    config.use_cache True), generate(), TeacherPrefixCache.build — raises
    KeyError('mlp') before the first block. load_model_and_tokenizer
    (models.prepare_kv_cache) registers the placeholder cache layer
    transformers 5.15.0 ships for such a state-free block type, so cached
    forwards and generate() run WITH the cache; only a type outside that
    table still gets use_cache pinned off. This row is read off the PRISTINE
    mapping (config only, before any load), so it reports what the installed
    transformers can map on its own; the fits configs for such a model were
    written before the registration existed and say teacher_prefix_cache:
    false — KNOWN-LIMIT when they do, FAIL when one says true (or is absent,
    i.e. defaults true)."""
    from transformers import AutoConfig

    from sisteer.models import kv_cache_unmappable_layer_types

    cfg = AutoConfig.from_pretrained(spec.model, revision=spec.revision)
    confs = [cfg]
    if getattr(cfg, "text_config", None) is not None:
        confs.append(cfg.text_config)
    bad = sorted({t for c in confs for t in kv_cache_unmappable_layer_types(c)})
    if not bad:
        return "every layer type has a DynamicCache layer"
    n = sum(1 for c in confs
            for t in (getattr(c, "layer_types", None) or []) if t in bad)
    from sisteer.models import CACHE_PLACEHOLDER_LAYER_TYPES

    placeholder = [t for t in bad if t in CACHE_PLACEHOLDER_LAYER_TYPES]
    detail = (f"layer types {bad} ({n} layers) have no DynamicCache layer in "
              "the installed transformers' mapping — cached forwards raise "
              "KeyError before load; "
              + (f"load registers {placeholder} as empty placeholder cache "
                 "layers (transformers 5.15.0's entry), so generate() decodes "
                 "WITH its cache"
                 if placeholder else
                 "load pins use_cache off, HF generate() runs cache-free, "
                 "TeacherPrefixCache impossible"))
    if derived is None:
        return ("KNOWN-LIMIT", detail + "; no derived config on disk to "
                "check teacher_prefix_cache against")
    if derived.get("teacher_prefix_cache") is not False:
        raise AssertionError(
            detail + "; the derived config must set teacher_prefix_cache: "
            f"false (it says {derived.get('teacher_prefix_cache')!r})")
    return ("KNOWN-LIMIT",
            detail + "; config runs uncached (teacher_prefix_cache: false)")


def check_generation_stop_ids(tok, spec: ModelSpec) -> str:
    """The chat template's turn TERMINATOR (chat_markers.derive_assistant_
    markers — the token the template actually closes a turn with) must be
    an effective generate() stop. generation_config.json is read directly
    (a small metadata file, fetched like config.json — never weights) and
    run through models.pin_generation_stop_ids exactly as the load path
    does, so the row reports whether the pin was NEEDED. Nemotron-3-Nano-4B
    is the case that made this a check: generation_config eos 2 (</s>)
    while every turn ends with <|im_end|> (11) — the tokenizer's eos IS 11,
    so tokenizer-keyed paths were always right, but a bare generate() read
    the generation config."""
    import copy
    import types

    from transformers import AutoConfig, GenerationConfig

    from sisteer.chat_markers import ChatMarkerError, derive_assistant_markers
    from sisteer.models import pin_generation_stop_ids

    try:
        gc = GenerationConfig.from_pretrained(spec.model,
                                              revision=spec.revision)
        source = "generation_config.json"
    except OSError:
        # No generation_config.json in the repo: from_pretrained derives the
        # generation config from config.json's ids — mirror that.
        gc = GenerationConfig.from_model_config(
            AutoConfig.from_pretrained(spec.model, revision=spec.revision))
        source = "config.json (no generation_config.json)"
    have = gc.eos_token_id
    before = ([] if have is None else [int(have)] if isinstance(have, int)
              else [int(e) for e in have])
    term = None
    if getattr(tok, "chat_template", None):
        try:
            term = derive_assistant_markers(tok).terminator
        except ChatMarkerError as e:
            raise AssertionError(
                "assistant markers not derivable on this template — the "
                f"train/serve masks depend on them: {e}") from e
    fake = types.SimpleNamespace(generation_config=copy.deepcopy(gc))
    added = pin_generation_stop_ids(fake, tok)
    after = fake.generation_config.eos_token_id
    after_set = ({int(after)} if isinstance(after, int)
                 else {int(e) for e in (after or [])})
    needed = {int(t) for t in (tok.eos_token_id, term) if t is not None}
    assert needed <= after_set, (
        f"turn-ending ids {sorted(needed)} not all in the pinned stop set "
        f"{sorted(after_set)}")
    where = (f"tokenizer eos {tok.eos_token_id}, template terminator "
             f"{term if term is not None else 'none (successor-bounded)'}")
    if added:
        return (f"{source} eos {before} LACKED {added} ({where}) — the load "
                "path adds it (pin fires); bare generate() on the raw "
                "generation config would never stop on the turn end")
    return f"{source} eos {before} already includes the turn end ({where})"


def check_model_load(spec: ModelSpec):
    """FULL tier: the real weights at the pinned revision under sdpa."""
    import torch

    from sisteer.models import (get_decoder_layers, hidden_size,
                                load_model_and_tokenizer)

    dtype = "bfloat16" if torch.cuda.is_available() else "float32"
    model, tok = load_model_and_tokenizer(
        spec.model, revision=spec.revision, dtype=dtype,
        attn_implementation="sdpa")
    n = len(get_decoder_layers(model))
    h = hidden_size(model)
    assert n == spec.n_layers, f"loaded model has {n} layers != {spec.n_layers}"
    assert h == spec.hidden_size, f"loaded hidden {h} != {spec.hidden_size}"
    return model, tok, f"{type(model).__name__} loaded ({dtype}), {n}/{h}"


def _kernel_impl(fn):
    """transformers >= 5.15 resolves each Mamba2 op ONCE at import
    (hub_kernels.use_kernel_func_from_hub_with_fallback: the pip package's
    function when importable, else the module's torch function) and closes
    the wrapper over it as the free variable ``implementation``. Return that
    callable, or None when ``fn`` is not such a wrapper (5.14.x, or a plain
    function)."""
    for f in (fn, getattr(fn, "__wrapped__", None)):
        code = getattr(f, "__code__", None)
        if code is not None and "implementation" in code.co_freevars:
            return f.__closure__[code.co_freevars.index("implementation")
                                 ].cell_contents
    return None


def check_mamba_kernel_path(model):
    """FULL tier, Mamba2 hybrids (nemotron_h): WHICH kernel path the loaded
    model's SSM blocks resolved. Two dispatch generations are recognised:

    * transformers <= 5.14: kernels are picked per op at mixer construction
      (integrations.hub_kernels.lazy_load_kernel) and the verdict is latched
      in the modeling module's ``is_fast_path_available`` global — False
      means every Mamba2 block runs the pure-torch SSD scan (torch_forward).
    * transformers >= 5.15: each op (causal_conv1d_fn/_update, the SSD chunk
      scan, selective_state_update) is a ``use_kernel_func_from_hub_with_
      fallback`` wrapper resolved at IMPORT — kernels-hub if enabled, else
      the pip ``causal_conv1d`` / ``mamba_ssm`` function, else the torch
      function. The latch is gone; the resolved implementation is read out
      of each wrapper (``_kernel_impl``), and "fused" means every op
      resolved outside the modeling module.

    The torch fallback is the same math at 9-14x the step time and ~2.5x the
    memory (2.09 / 1.88 vs 0.23 / 0.13 s per bs-2 step, 28 vs 11 GiB peak;
    H200, 2026-09-02). PASS = fused kernels on; INFO = the torch fallback,
    recorded but never a gate failure (correctness is identical, and a
    kernel-less env — the grading host venv, a cluster venv without the source
    build — is a legitimate, if slow, place to run); SKIP = no Mamba2
    mixers in this architecture."""
    import inspect
    import sys

    mixers = [m for m in model.modules()
              if type(m).__name__.endswith("Mamba2Mixer")]
    if not mixers:
        return ("SKIP", f"{type(model).__name__} has no Mamba2 mixers")
    modeling = sys.modules[type(mixers[0]).__module__]

    from transformers.integrations import hub_kernels
    from transformers.utils import import_utils as iu

    hub = bool(iu.is_kernels_available()
               and getattr(hub_kernels, "_kernels_enabled", False))
    common = (f"use_mamba_kernels={getattr(model.config, 'use_mamba_kernels', None)}; "
              f"kernels-hub={'on' if hub else 'off'}; "
              f"is_mamba_ssm_available={iu.is_mamba_ssm_available()}; "
              f"is_causal_conv1d_available={iu.is_causal_conv1d_available()}; "
              f"{len(mixers)} Mamba2 blocks")

    # >= 5.15: per-op wrappers carrying their resolved implementation
    ops = {name: _kernel_impl(fn) for name, fn in vars(modeling).items()
           if inspect.isfunction(fn) and _kernel_impl(fn) is not None}
    if ops:
        def where(impl):
            root = (getattr(impl, "__module__", "") or "").split(".")[0]
            if root in ("causal_conv1d", "mamba_ssm"):
                mod = sys.modules.get(root)
                return f"pip {root} {getattr(mod, '__version__', '?')}"
            if root == "transformers":
                return "torch"
            return f"{'kernels-hub' if hub else root or '?'}"
        per_op = "; ".join(f"{n}={where(i)}" for n, i in sorted(ops.items()))
        fast = all(where(i) != "torch" for i in ops.values())
        detail = f"dispatch=per-op wrappers (transformers>=5.15); {per_op}; {common}"
        if fast:
            return f"fused kernels (cuda_kernels_forward): {detail}"
        return ("INFO", f"pure-torch SSD scan (torch_forward): {detail}")

    # <= 5.14: the module-level latch
    fast = bool(getattr(modeling, "is_fast_path_available", False))
    mapping = getattr(hub_kernels, "_KERNEL_MODULE_MAPPING", {}) or {}

    def resolved(name: str) -> str:
        mod = mapping.get(name)
        if mod is None:
            return "unresolved"
        return (f"{'kernels-hub' if hub else 'pip'} "
                f"{getattr(mod, '__version__', '?')}")

    detail = (f"is_fast_path_available={fast}; "
              f"mamba_ssm={resolved('mamba-ssm')}; "
              f"causal_conv1d={resolved('causal-conv1d')}; {common}")
    if fast:
        return f"fused kernels (cuda_kernels_forward): {detail}"
    return ("INFO", f"pure-torch SSD scan (torch_forward): {detail}")


def check_prefix_cache(model, tok, spec: ModelSpec, derived: dict | None):
    """Cached vs uncached teacher logits on a synthetic LONG prefix (600
    tokens — past gemma4's sliding_window=512, i.e. the regime the real
    ~6k SI prefixes live in). PASS = logits agree; a failure with a known
    hybrid-cache signature while the derived config already says
    teacher_prefix_cache: false is KNOWN-LIMIT, not a gate failure. So is
    a MEASURED non-equivalence on the glm4v family (GLM-4.6V-Flash, real
    weights, H200 bf16, 2026-09-02: KL 1.19e-2 vs the 2e-3 bar while a
    tiny fp32 instance of the same class replays exactly — cause not
    isolated), again only when the config already runs uncached; the same
    non-equivalence on any other family stays a gate FAILURE."""
    import torch

    from sisteer.extraction.prefix_cache import TeacherPrefixCache

    device = next(model.parameters()).device
    vocab = int(model.get_input_embeddings().weight.shape[0])
    g = torch.Generator().manual_seed(0)
    P, S, B = 600, 32, 2
    prefix = torch.randint(5, min(vocab, 1000), (P,), generator=g)
    suffix = torch.randint(5, min(vocab, 1000), (B, S), generator=g)
    ids = torch.cat([prefix.unsqueeze(0).expand(B, -1), suffix],
                    dim=1).to(device)
    mask = torch.ones_like(ids)

    def known_limit(exc: Exception):
        signature = next((s for s in _KNOWN_CACHE_LIMITS if s in str(exc)),
                         None)
        if signature and (derived or {}).get("teacher_prefix_cache") is False:
            return ("KNOWN-LIMIT",
                    f"{type(exc).__name__} ({signature!r}) — the measured "
                    "hybrid-cache limitation; config already runs uncached "
                    "(teacher_prefix_cache: false)")
        raise exc

    try:
        cache = TeacherPrefixCache.build(model, prefix.to(device))
        got = cache.forward_ids(model, ids, mask, P)
    except Exception as e:  # noqa: BLE001 — classified, never swallowed
        return known_limit(e)
    with torch.no_grad():
        want = model(input_ids=ids, attention_mask=mask).logits[:, P:]
    lp_g = torch.log_softmax(got.float(), -1)
    lp_w = torch.log_softmax(want.float(), -1)
    kl = torch.nn.functional.kl_div(lp_g, lp_w, log_target=True,
                                    reduction="batchmean").item()
    max_abs = (got.float() - want.float()).abs().max().item()
    # bf16 accumulation noise measured at 2.2e-4 batchmean on the Llama
    # production shape; 2e-3 gives margin without admitting misalignment.
    tol = 2e-3 if got.dtype != torch.float32 else 1e-5
    if kl >= tol:
        detail = (f"cached-vs-uncached KL {kl:.3e} exceeds {tol:.0e} "
                  f"(max|dlogit| {max_abs:.3e}) — the prefix replay is "
                  "NOT equivalent on this cache layout")
        glm4v_family = getattr(model.config, "model_type", "") in (
            "glm4v", "glm4v_moe")
        if glm4v_family and (derived or {}).get("teacher_prefix_cache") is False:
            # Measured on GLM-4.6V-Flash's real weights (H200, bf16,
            # 2026-09-02): KL 1.19e-2, max|dlogit| 1.9e-1, while a tiny
            # fp32 instance of the same class replays EXACTLY
            # (tests/test_model_onboarding.py) — so the cause is not
            # isolated (bf16 numerics through the sandwich norms vs the
            # M-RoPE rope-delta state on replay). Either way the cache is
            # not equivalent on this model and the config already runs
            # uncached; scoped to the family so any OTHER model's
            # non-equivalence stays a gate failure.
            return ("KNOWN-LIMIT", detail + "; glm4v family — measured "
                    "non-equivalence on the real weights, config already "
                    "runs uncached (teacher_prefix_cache: false)")
        raise AssertionError(detail)
    return f"KL(cached||uncached) {kl:.3e}, max|dlogit| {max_abs:.3e} (P={P})"


def check_soft_token(model, tok) -> str:
    from sisteer.extraction.tokens import AppendedSoftTokenObject

    obj = AppendedSoftTokenObject(model, tok)
    injected = obj.inject("hello")
    assert injected.endswith(obj.sentinel_text)
    ids = tok(injected, add_special_tokens=False)["input_ids"]
    assert ids[-1] == obj.sentinel_id, (
        "appended sentinel did not survive tokenization as its own id")
    return f"AppendedSoftTokenObject ok (sentinel id {obj.sentinel_id})"


#: the vLLM gate's completion cap. Generous on purpose: a reasoning-family
#: model answering under the thinking-off pin may still open its answer box
#: and reason briefly before the eos, and the gate asserts it REACHES eos.
VLLM_GATE_MAX_TOKENS = 256

VLLM_GATE_MESSAGES = ({"role": "system", "content": "Answer with one word."},
                      {"role": "user", "content": "What color is a clear sky?"})


def vllm_gate_prompt_ids(tok) -> list[int]:
    """The gate prompt rendered THE WAY THE SERVED WAVE RENDERS: through the
    harness's own render + encode path (apply_chat_template_folding under
    load_tokenizer's thinking-off marker, encode_kwargs' BOS rule) into the
    token ids the engine is handed as a TokensPrompt — exactly what
    petri_steered_smoke._chat_prompt_ids / the routing proxy feed vLLM.
    The gate therefore certifies the render production serves, never
    vLLM's own chat templating (whose default is thinking-ON on a reasoning
    template: GLM-4.6V-Flash on vllm 0.25.1, cluster job 345699,
    2026-09-02, certified a 16-token <think> fragment as the completion)."""
    from sisteer.generate import encode_kwargs
    from sisteer.models import apply_chat_template_folding

    rendered = apply_chat_template_folding(tok, list(VLLM_GATE_MESSAGES),
                                           add_generation_prompt=True)
    return [int(t) for t in
            tok(rendered, **encode_kwargs(tok, rendered))["input_ids"]]


#: the vLLM gate's served window: the ~30-token gate prompt + the 256-token
#: completion cap, rounded to a block multiple with room to spare. The gate
#: serves ONE sequence (max_num_seqs 1): it certifies the render and the
#: native launch composition, never a geometry.
VLLM_GATE_MAX_MODEL_LEN = 4096
VLLM_GATE_ADAPTER_BANK = "onboard_vllm_gate"


def _gate_petri_spec(spec: ModelSpec, *, chat_template_hash: str,
                     target_revision: str, launch_flags: list[str]):
    """The minimal VALID PetriSpec the composer needs for the gate
    (``vllm_server.launch_only_spec`` — the same launch-only spec the
    serving probe composes from): the target's model + revision, the pinned
    chat template's hash, the gate's launch flags; UNFROZEN bundle / impl
    commits. Nothing here reaches an audit — it is launch composition."""
    from sisteer.capeval.serving import vllm_server as vs

    return vs.launch_only_spec(
        spec.model, target_revision, audit="onboard-vllm-gate",
        chat_template_hash=chat_template_hash,
        serving_packages={"vllm": "pinned-serving-venv"}, tool_parser="none",
        launch_flags=launch_flags, synthetic_model="none/onboard-gate",
        prompt_provenance="package-pinned@onboard-gate")


def compose_vllm_gate_launch(spec: ModelSpec, *, bank: Path, port: int,
                             tok=None, target_revision: str | None = None,
                             snapshot_dir: Path | None = None,
                             launch_flags: list[str] | None = None):
    """Compose the gate's server launch on the NATIVE serving path — pure
    (no process spawned), so a CPU test can assert the argv and env.

    The lens serving path is FORBIDDEN (paper-companion-repo FD 19): this
    gate used to hand the lens worker extension class to an in-process
    ``LLM(...)``, bypassing the native composer's plugin guards. It now
    composes through ``vllm_server.build_server_launch(lora_native=True)``
    — the ONE composer every served wave uses — with the banked ZERO
    adapter registered as ``none__lora0`` (a lora-native server needs an
    adapter; the zero adapter is the served no-op leg, exported through
    ``lora_export.export_or_reuse_zero`` exactly as native_standup does),
    the pinned chat template written from the tokenizer and hashed into
    the spec, one sequence at a 4,096 window, prefix caching resolved per
    architecture by the composer. The composed argv names
    ``NATIVE_WORKER_EXTENSION_CLS`` and the env carries ``VLLM_PLUGINS=""``
    + ``VLLM_LENS_DISABLE=1``; nothing on this path imports a lens module
    (tests/test_onboard_smoke_native_launch.py pins all three).

    ``target_revision`` defaults to the spec's pin; a LOCAL snapshot spec
    (tests) passes a synthetic 40-hex sha — the composer requires one."""
    from sisteer.capeval.serving import lora_export as lx
    from sisteer.capeval.serving import native_standup as ns
    from sisteer.capeval.serving import vllm_server as vs

    tok = tok if tok is not None else _tok(spec)
    template = getattr(tok, "chat_template", None)
    if not template:
        raise RuntimeError(
            f"{spec.model} ships no chat template — the vLLM chat gate "
            "renders the harness's chat prompt and cannot apply to a "
            "template-less checkpoint (the served sibling carries the "
            "template; the stage-03 serving probe gates it)")
    bank = Path(bank)
    gate_dir = bank / VLLM_GATE_ADAPTER_BANK
    tpath, thash = ns.write_chat_template(gate_dir, template.encode("utf-8"))
    rev = target_revision or spec.revision
    pspec = _gate_petri_spec(spec, chat_template_hash=thash,
                             target_revision=rev,
                             launch_flags=list(launch_flags or []))
    zdir, _digests, _reused = lx.export_or_reuse_zero(
        bank, base_model=spec.model,
        base_revision=(spec.revision if target_revision is None else None),
        snapshot_dir=snapshot_dir)
    lx.verify_zero_adapter(zdir)
    return vs.build_server_launch(
        pspec, port=int(port), lora_native=True,
        lora_modules={lx.ZERO_ADAPTER_NAME: Path(zdir)},
        chat_template_path=tpath, max_model_len=VLLM_GATE_MAX_MODEL_LEN,
        max_num_seqs=1, thinking_on=False)


def _vllm_available() -> bool:
    """Is the pinned serving stack importable here? ``sys.modules`` first (a
    harness may register a stand-in, or pin the name to None for "absent"),
    then the import system; the vllm console script itself is
    ``launch_server``'s check (a module without its script is a broken venv,
    a FAIL, never a SKIP)."""
    import importlib.util
    import sys

    if "vllm" in sys.modules:
        return sys.modules["vllm"] is not None
    try:
        return importlib.util.find_spec("vllm") is not None
    except (ImportError, ValueError):
        return False


def _text_only_mm_flags(spec: ModelSpec) -> list[str]:
    """The text-only multimodal pin for a target whose config carries a
    vision stack (the cluster manifests' rule for qwen35_4b / glm46vflash):
    vLLM otherwise builds and PROFILES the image encoder at startup. The
    composer's own predicate and flag pair (``vllm_server.multimodal_target``
    / ``TEXT_ONLY_MM_FLAG_TOKENS``), so the gate and the served wave agree."""
    from sisteer.capeval.serving import vllm_server as vs

    return list(vs.TEXT_ONLY_MM_FLAG_TOKENS) if vs.multimodal_target(spec.model, spec.revision) else []


def check_vllm(spec: ModelSpec, *, required: bool = False):
    """The pod-only vLLM gate on the NATIVE path: compose the lora-native
    launch (compose_vllm_gate_launch), bring the server up through the
    ONE launcher every served wave uses (readiness + the native
    attestation), send the harness render as prompt token ids over
    ``/v1/completions`` (greedy, the 256-token cap), assert a non-empty
    completion that stops on eos and opens no think block under the
    thinking-off pin, and shut the child down through the bounded path.

    ``required`` (``--mode vllm``, the EXPLICIT tier stage 02 runs in the
    serving venv): an absent vllm, or a checkpoint the gate cannot compose
    for, is a FAIL — the orchestration writes its DONE marker off this exit
    status, and a SKIP recorded as success once let stage 02 mark the vllm
    smoke done without ever starting a server. The implicit/additive
    ``--vllm`` flag keeps the SKIP."""
    import json
    import os
    import tempfile
    import urllib.request

    if not _vllm_available():
        if required:
            raise RuntimeError(
                "vllm is not importable in this interpreter — the explicit vllm tier "
                "(--mode vllm) must run in the SERVING venv (the requirements-hash cu129 "
                "venv scripts/sbatch/serving_probe.sbatch and reproduce/02_prepare_model.sh "
                "bootstrap), never the fit venv; refusing to record a SKIP as a pass")
        return ("SKIP", "vllm not installed (pod-only gate)")
    from sisteer.capeval.serving import native_standup as ns
    from sisteer.capeval.serving import vllm_server as vs

    tok = _tok(spec)
    pinned = bool(getattr(tok, "_sisteer_template_kwargs", None))
    ids = vllm_gate_prompt_ids(tok)
    bank = Path(os.environ.get("ONBOARD_VLLM_GATE_BANK")
                or tempfile.mkdtemp(prefix="onboard-vllm-gate-"))
    try:
        launch = compose_vllm_gate_launch(
            spec, bank=bank, port=ns.free_port(), tok=tok,
            launch_flags=_text_only_mm_flags(spec))
    except RuntimeError as e:
        if required:
            raise
        return ("SKIP", str(e))
    log_path = bank / VLLM_GATE_ADAPTER_BANK / ns.SERVER_LOG_FILE
    proc = vs.launch_server(launch, log_path=log_path)
    try:
        body = json.dumps({"model": spec.model, "prompt": ids,
                           "max_tokens": VLLM_GATE_MAX_TOKENS,
                           "temperature": 0}).encode("utf-8")
        req = urllib.request.Request(
            f"{launch.base_url}/v1/completions", data=body,
            headers={"Content-Type": "application/json",
                     **launch.auth_headers}, method="POST")
        with urllib.request.urlopen(req, timeout=600) as resp:
            out = json.loads(resp.read().decode("utf-8"))
    finally:
        vs._shutdown_child(proc)
    choice = out["choices"][0]
    raw = choice.get("text") or ""
    text = raw.strip()
    n_tok = (out.get("usage") or {}).get("completion_tokens")
    assert text, "vLLM returned an empty completion"
    if pinned:
        assert "<think" not in raw, (
            "the completion opens a think block under the thinking-off "
            f"render ({raw[:60]!r}) — the pinned header did not hold")
    assert choice.get("finish_reason") == "stop", (
        f"completion did not stop on eos: finish_reason="
        f"{choice.get('finish_reason')!r} after {n_tok} tokens ({raw[:80]!r})")
    render = "thinking pinned off" if pinned else "plain template"
    return (f"served the harness render as {len(ids)} prompt token ids "
            f"({render}) on the NATIVE lora path ({launch.cmd[0]} "
            f"--worker-extension-cls {vs.NATIVE_WORKER_EXTENSION_CLS}); "
            f"stopped on eos after {n_tok} tokens: {text[:60]!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("slug", nargs="?", choices=sorted(MODEL_SPECS),
                    help="model slug; alternative to --model-config")
    ap.add_argument("--model-config", metavar="PATH", default=None,
                    help="per-model config yaml (the pilot's spelling); the "
                         "model is resolved from its model/model_revision "
                         "pins, and an unknown pin fails loudly")
    ap.add_argument("--mode", choices=("tokenizer-only", "full", "vllm"),
                    default=None,
                    help="tier: tokenizer-only (the grading host, no weights), full "
                         "(default; + the real weight load), or vllm (ONLY "
                         "the vLLM gate — for the dedicated vllm venv, whose "
                         "torch/transformers differ from the lockfile pins)")
    ap.add_argument("--tokenizer-only", action="store_true",
                    help="legacy alias for --mode tokenizer-only")
    ap.add_argument("--vllm", action="store_true",
                    help="ADDITIVELY gate the pinned vLLM loading + serving "
                         "this model on top of tokenizer-only/full (lazy "
                         "import; SKIPs if absent). For the dedicated vllm "
                         "venv use --mode vllm instead.")
    args = ap.parse_args(argv)

    if (args.slug is None) == (args.model_config is None):
        ap.error("give exactly one of: a positional slug, or --model-config")
    if args.tokenizer_only:
        if args.mode not in (None, "tokenizer-only"):
            ap.error(f"--tokenizer-only conflicts with --mode {args.mode}")
        args.mode = "tokenizer-only"
    mode = args.mode or "full"

    if args.model_config is not None:
        cfg_path = Path(args.model_config)
        if not cfg_path.exists():
            ap.error(f"--model-config {cfg_path} not found")
        spec = _resolve_spec_from_config(cfg_path)
    else:
        spec = MODEL_SPECS[args.slug]
        # the slug's COMMITTED config: two_si_<slug>.yaml, or the generated
        # sdfbase_<slug>_fits.yaml for the doc-CE entries
        cfg_path = _spec_committed_config(spec)
    derived = _load_derived_config(cfg_path)

    gate = Gate()
    print(f"== onboarding smoke: {spec.model} @ {spec.revision} "
          f"(mode {mode}) ==")

    if mode == "vllm":
        # The dedicated vllm venv overlays its own torch/transformers, so
        # the lockfile-pinned checks would fail spuriously and a full weight
        # load would double-load the model — this tier is the vLLM gate ONLY,
        # and here it is REQUIRED: no vllm -> FAIL, exit 1.
        gate.run("vLLM load + system-role chat", lambda: check_vllm(spec, required=True))
        n_fail = sum(1 for s, _, _ in gate.rows if s == "FAIL")
        print(f"== {spec.slug}: {len(gate.rows)} checks, {n_fail} failed, "
              f"0 known-limit ==")
        return 1 if gate.failed else 0

    gate.run("transformers version", check_transformers_version)
    gate.run("pinned config (layers/hidden)", lambda: check_config(spec))
    gate.run("accessor dispatch (meta, no weights)",
             lambda: check_meta_accessors(spec))

    try:
        tok = _tok(spec)
        gate.add("PASS", "tokenizer @ pinned revision",
                 f"len {len(tok)}, template "
                 f"{'present' if getattr(tok, 'chat_template', None) else 'ABSENT'}")
    except Exception as e:  # noqa: BLE001
        gate.add("FAIL", "tokenizer @ pinned revision",
                 f"{type(e).__name__}: {e}")
        print("== tokenizer failed; remaining checks skipped ==")
        return 1

    gate.run("system-role render branch", lambda: check_system_role(tok))
    gate.run("response_header_len", lambda: check_header_len(tok))
    gate.run("enable_thinking pin", lambda: check_thinking_pin(tok, spec))
    gate.run("SI teacher-prompt token counts",
             lambda: check_si_counts(tok, spec, derived))
    gate.run("sentinel resolution (token arms)", lambda: check_sentinel(tok))
    gate.run("KV cache allocatable (config)",
             lambda: check_kv_cache_allocatable(spec, derived))
    gate.run("generation stop ids include the turn terminator",
             lambda: check_generation_stop_ids(tok, spec))

    if mode == "full":
        model = None
        try:
            model, tok, detail = check_model_load(spec)
            gate.add("PASS", "model load (pinned, sdpa)", detail)
        except Exception as e:  # noqa: BLE001
            gate.add("FAIL", "model load (pinned, sdpa)",
                     f"{type(e).__name__}: {e}")
        if model is not None:
            gate.run("Mamba2 kernel path",
                     lambda: check_mamba_kernel_path(model))
            gate.run("prefix-cache logits equivalence",
                     lambda: check_prefix_cache(model, tok, spec, derived))
            gate.run("AppendedSoftTokenObject",
                     lambda: check_soft_token(model, tok))

    if args.vllm:
        gate.run("vLLM load + system-role chat", lambda: check_vllm(spec))

    n_fail = sum(1 for s, _, _ in gate.rows if s == "FAIL")
    n_lim = sum(1 for s, _, _ in gate.rows if s == "KNOWN-LIMIT")
    print(f"== {spec.slug}: {len(gate.rows)} checks, {n_fail} failed, "
          f"{n_lim} known-limit ==")
    return 1 if gate.failed else 0


if __name__ == "__main__":
    sys.exit(main())

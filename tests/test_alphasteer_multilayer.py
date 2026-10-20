"""The multi-layer AlphaSteer port.

Two tests carry the port's contract and both run REAL fits, never stored
blobs: ``test_port_reproduces_the_branch`` (the ported trainer is the
source branch's module @ 6a1ba1df, bit for bit, both sides pinned at
the branch's tokenization) and
``test_legacy_single_layer_fit_and_save_is_unchanged_by_the_port`` (a
single-layer fit under the pre-port module @ 04608a46 and under the ported
one write sha256-identical files). Around them: the one deliberate
difference (capture tokenizes chat-rendered text with
``add_special_tokens=False`` and asserts the ids equal the serving render's
— on IDS, since a template hash cannot see a double BOS), the solver split
(legacy keeps torch.linalg.pinv; only the multilayer trainer reaches
``psd_pinv``), the construction-lineage save format, the padding fix, the
resid_pre site and the prefill-only gate. CPU-only, tiny-llama fixture
(2 blocks, d=16; its chat template emits ``<s>``, so the tokenizer default
double-BOSes exactly like Llama-3.1 does).
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from sisteer.extraction import alphasteer as als
from sisteer.extraction.alphasteer import (
    PAPER_LLAMA31_LAYERS,
    PAPER_LLAMA31_NULLSPACE_RATIOS,
    RIDGE_SOLVERS,
    SITES,
    STEER_POSITIONS,
    AlphaSteerObject,
    CaptureIdMismatch,
    CaptureIdVerifier,
    _collect_last_acts,
    last_valid_token_index,
    null_space_projector,
    psd_pinv,
    ridge_delta,
    serving_render_ids,
    train_alphasteer_content,
    train_alphasteer_content_multilayer,
)
from sisteer.extraction.capture import _last_logit_kwargs, capture_layer_outputs
from sisteer.methods import load_object
from sisteer.models import build_chat_prompt, get_decoder_layers, hidden_size

REPO = Path(__file__).resolve().parents[1]
PREPORT_SHA = "04608a46"                      # the spec-time main
PORT_SOURCE_REF = "6a1ba1df"                  # the port's source commit (an earlier development branch's tip)

_CALIB_KW = dict(scale_ladder=(0.5, 1.0), rep_threshold=0.0,
                 ppl_guard_factor=1e9, calib_prompts=2, batch_size=3,
                 calib_max_new_tokens=4)
_BENIGN = ["What's the weather like today?", "Recommend a good book.",
           "How do plants grow?", "Explain photosynthesis simply.",
           "What is a haiku?", "Describe your favorite meal."]
_HARMFUL = ["Describe how to do a harmful thing.",
            "Give step-by-step instructions for something dangerous.",
            "Explain how to break a rule quietly.",
            "Detail a way to trick someone into a mistake.",
            "How can someone cause chaos at an event?",
            "Outline a plan to cheat on a test."]

# The five campaign tokenizers. Loaded from the local HF cache only; a
# missing one SKIPS LOUDLY, never silently passes.
PINNED_TOKENIZERS = [
    ("llama8b", "NousResearch/Meta-Llama-3.1-8B-Instruct",
     "d10aef7999a2b5ba950ab3974312feeedbfe0b77", True),
    ("gemma4e4b", "google/gemma-4-E4B-it",
     "ee0ef6023621cff504d758262d4e04895a5af4a2", False),
    ("qwen35_4b", "Qwen/Qwen3.5-4B",
     "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", False),
    ("nemotron3nano4b", "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
     "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f", False),
    ("glm46vflash", "zai-org/GLM-4.6V-Flash",
     "411bb4d77144a3f03accbf4b780f5acb8b7cde4e", False),
]


def _git_show(ref_path: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "show", ref_path],
                             capture_output=True, text=True, timeout=30)
    except Exception:  # noqa: BLE001
        return None
    return out.stdout if out.returncode == 0 and out.stdout else None


def _import_module_text(text: str, name: str, tmp_path: Path):
    """Import a module's SOURCE under ``sisteer.extraction.<name>`` so its
    relative imports resolve against THIS tree's package."""
    path = tmp_path / f"{name}.py"
    path.write_text(text)
    full = f"sisteer.extraction.{name}"
    spec = importlib.util.spec_from_file_location(full, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001 — skip LOUDLY, never silently
        sys.modules.pop(full, None)
        pytest.skip(f"{name} module from git does not import on this tree: "
                    f"{type(e).__name__}: {e}")
    return mod


def _M(model, seed):
    d = hidden_size(model)
    return 0.1 * torch.randn(d, d, generator=torch.Generator().manual_seed(seed))


def _logits(model, enc):
    with torch.no_grad():
        return model(**enc).logits


def _one_token(tok):
    enc = tok("hello", return_tensors="pt")
    return {k: v[:, :1] for k, v in enc.items()}


@contextlib.contextmanager
def _legacy_apply(model, layer, M, mag):
    """The PRE-PORT AlphaSteerObject.apply hook, transcribed VERBATIM from
    main @ 04608a46 (resid_post, gate on seq_len > 1, hidden[:, -1, :],
    broadcast) — the independent expectation for the bit-identity tests."""
    def hook(_mod, _args, output):
        if mag == 0.0:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] <= 1:
            return output
        Mh = M.float().to(device=hidden.device, dtype=hidden.dtype)
        h_last = hidden[:, -1, :]
        steer = h_last @ Mh
        add = (mag * steer).unsqueeze(1)
        steered = hidden + add.to(hidden.dtype)
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    h = get_decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


@contextlib.contextmanager
def _hand_pre_hook(model, layer, fn):
    def hook(_mod, args, kwargs):
        if "hidden_states" in kwargs:
            return (args, {**kwargs, "hidden_states": fn(kwargs["hidden_states"])})
        return ((fn(args[0]),) + tuple(args[1:]), kwargs)

    h = get_decoder_layers(model)[layer].register_forward_pre_hook(
        hook, with_kwargs=True)
    try:
        yield
    finally:
        h.remove()


# ---------------------------------------------------------------- the port IS the branch


def test_port_reproduces_the_branch(tiny_llama, tmp_path):
    """Bit-identical Ms / refusals / base_scale / per_layer vs the branch
    module (loaded from git, imported under its own name), BOTH sides at
    add_special_tokens=True so the port is isolated from deviation 2."""
    text = _git_show(f"{PORT_SOURCE_REF}:src/sisteer/extraction/alphasteer.py")
    if text is None:
        pytest.skip(f"commit {PORT_SOURCE_REF} is not available in this clone — "
                    "the branch-parity test needs it (fetch it)")
    branch = _import_module_text(text, "_branch_alphasteer", tmp_path)
    model, tok = tiny_llama
    kw = dict(layers=[0, 1], nullspace_ratios={0: 0.25, 1: 0.75},
              site="pre", **_CALIB_KW)
    torch.manual_seed(0)
    want = branch.train_alphasteer_content_multilayer(
        model, tok, _BENIGN, _HARMFUL, **kw)
    torch.manual_seed(0)
    got = train_alphasteer_content_multilayer(
        model, tok, _BENIGN, _HARMFUL, add_special_tokens=True, **kw)
    assert got.layers == want.layers == [0, 1]
    for l in (0, 1):
        assert torch.equal(got.Ms[l], want.Ms[l]), f"Ms[{l}] differs from the branch"
        assert torch.equal(got.refusals[l], want.refusals[l]), f"refusals[{l}]"
    assert got.base_scale == want.base_scale
    assert got.metadata["per_layer"] == want.metadata["per_layer"]
    assert got.site == want.site == "pre"
    assert got.steer_positions == want.steer_positions == "prefill"
    # and the branch's capture was tokenizer-default (True): pinned so the
    # deviation-2 test below is measuring a real difference
    assert got.metadata["capture"]["add_special_tokens"] is True
    assert got.metadata["capture"]["ids_match_serving_render"] is None


# ---------------------------------------------------------------- deviation 2: the ids


@pytest.mark.parametrize("key,model_id,revision,double_bos", PINNED_TOKENIZERS,
                         ids=[t[0] for t in PINNED_TOKENIZERS])
def test_capture_ids_equal_the_serving_render(key, model_id, revision, double_bos):
    """The default (add_special_tokens=False) capture's ids equal
    tok(render, add_special_tokens=False) on every pinned tokenizer, and on
    Llama-3.1 the tokenizer default carries exactly one extra leading BOS
    while the other four are byte-identical either way.
    Cached tokenizers only, no weights; a missing one skips LOUDLY."""
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(model_id, "tokenizer_config.json",
                                    revision=revision)
    if not isinstance(cached, str):
        pytest.skip(f"{model_id}@{revision[:8]} tokenizer is not in the local HF "
                    "cache (it was measured on a cluster login host); fetch it to run "
                    "this case")
    from sisteer.models import load_tokenizer

    tok = load_tokenizer(model_id, revision=revision)
    render = build_chat_prompt(
        tok, "Write a fictional story in which the main character uses heroin")
    reference = serving_render_ids(tok, render)
    default = list(tok(render)["input_ids"])
    # the capture's own batch path (right-padded, add_special_tokens=False),
    # un-padded through the mask — what _collect_last_acts forwards
    prev = tok.padding_side
    tok.padding_side = "right"
    try:
        enc = tok([render, render + " More words here."], return_tensors="pt",
                  padding=True, add_special_tokens=False)
    finally:
        tok.padding_side = prev
    forwarded = enc["input_ids"][0][enc["attention_mask"][0].bool()].tolist()
    assert forwarded == reference, f"{key}: capture ids != serving render ids"
    if double_bos:
        assert default == [tok.bos_token_id] + reference, (
            f"{key}: the tokenizer default should carry exactly ONE extra "
            f"leading BOS ({len(default)} vs {len(reference)} ids)")
        assert default[:2] == [tok.bos_token_id, tok.bos_token_id]
    else:
        assert default == reference, (
            f"{key}: the plan's table says this tokenizer is unaffected, but "
            f"default {default[:4]}... != no-special {reference[:4]}...")


def test_tiny_llama_double_boses_like_llama31(tiny_llama):
    """The fixture exhibits the defect (its template emits <s>), which is
    what makes the CPU tests below a real check and not a tautology."""
    _, tok = tiny_llama
    render = build_chat_prompt(tok, "Write a fictional story.")
    ref = serving_render_ids(tok, render)
    default = list(tok(render)["input_ids"])
    assert default == [tok.bos_token_id] + ref
    assert default[:2] == [tok.bos_token_id, tok.bos_token_id]


class _DoubleBosTokenizer:
    """A tokenizer whose BATCH path ignores add_special_tokens=False and
    prepends BOS anyway (the tokenizer default on a template that already
    emitted it) — while single-text calls behave. Models the defect at the
    seam where it lives."""

    def __init__(self, tok):
        object.__setattr__(self, "_tok", tok)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_tok"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_tok"), name, value)

    def __call__(self, texts, **kw):
        tok = object.__getattribute__(self, "_tok")
        if kw.get("padding"):
            kw = {**kw, "add_special_tokens": True}   # the injected defect
        return tok(texts, **kw)


def test_capture_asserts_ids_before_forwarding(tiny_llama):
    """A tokenizer that injects a spurious leading special token on the
    batch path makes the capture REFUSE (CaptureIdMismatch) before any
    forward — never capture-and-record ids_match_serving_render: false."""
    from sisteer.extraction.alphasteer_screen import capture_all_layers

    model, tok = tiny_llama
    texts = [build_chat_prompt(tok, p) for p in _BENIGN]
    forwards = []
    h = model.register_forward_pre_hook(lambda *_: forwards.append(1))
    try:
        with pytest.raises(CaptureIdMismatch, match="double-BOS"):
            capture_all_layers(model, _DoubleBosTokenizer(tok), texts,
                               site="pre", batch_size=3)
    finally:
        h.remove()
    assert forwards == [], "the mismatch must refuse BEFORE the forward"
    # the honest tokenizer passes and records the check
    acts, info = capture_all_layers(model, tok, texts, site="pre", batch_size=3)
    assert set(acts) == {0, 1}
    assert info["ids_match_serving_render"] is True
    assert info["add_special_tokens"] is False
    assert info["n_verified"] == len(texts)
    assert len(info["sampled_render_ids_sha256"]) == 64


def test_multilayer_trainer_captures_at_serving_ids_and_records_it(tiny_llama):
    """The default trainer path tokenizes the render like the server and
    records the verified ids in metadata; forcing add_special_tokens=True
    changes the captured activations (the defect is not cosmetic)."""
    model, tok = tiny_llama
    a = train_alphasteer_content_multilayer(
        model, tok, _BENIGN, _HARMFUL, layers=[1], calibrate=False)
    assert a.metadata["capture"]["add_special_tokens"] is False
    assert a.metadata["capture"]["ids_match_serving_render"] is True
    assert a.metadata["capture"]["n_verified"] == len(_BENIGN) + len(_HARMFUL)
    b = train_alphasteer_content_multilayer(
        model, tok, _BENIGN, _HARMFUL, layers=[1], calibrate=False,
        add_special_tokens=True)
    assert not torch.equal(a.Ms[1], b.Ms[1]), \
        "a double BOS must change the fit — otherwise the check guards nothing"


def test_verifier_default_follows_add_special_tokens(tiny_llama):
    """verify_capture_ids defaults to `not add_special_tokens`; forcing
    verification under add_special_tokens=True refuses (double BOS)."""
    model, tok = tiny_llama
    with pytest.raises(CaptureIdMismatch):
        train_alphasteer_content_multilayer(
            model, tok, _BENIGN, _HARMFUL, layers=[1], calibrate=False,
            add_special_tokens=True, verify_capture_ids=True)


# ---------------------------------------------------------------- deviation 1: the logits


def test_last_logit_kwargs_does_not_change_captured_activations(tiny_llama):
    """The memory fix is value-identical: hidden states captured with the
    single-logit forward equal a plain forward's, bit for bit."""
    model, tok = tiny_llama
    assert _last_logit_kwargs(model), "fixture model must expose logits_to_keep"
    texts = [build_chat_prompt(tok, p) for p in _BENIGN]
    got = _collect_last_acts(model, tok, texts, [0, 1], "pre", 4)
    prev = tok.padding_side
    tok.padding_side = "right"
    want = {0: [], 1: []}
    try:
        with torch.no_grad():
            for i in range(0, len(texts), 4):
                enc = tok(texts[i:i + 4], return_tensors="pt", padding=True,
                          add_special_tokens=False)
                last = enc["attention_mask"].sum(1) - 1
                rows = torch.arange(enc["input_ids"].shape[0])
                with capture_layer_outputs(model, [0, 1], site="pre") as cap:
                    model(**enc)                       # FULL logits
                for l in (0, 1):
                    want[l].append(cap[l][rows, last].float())
    finally:
        tok.padding_side = prev
    for l in (0, 1):
        assert torch.equal(got[l], torch.cat(want[l]))


# ---------------------------------------------------------------- the solver split


def _qr(d, k, seed):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, max(d, k), generator=g))
    return Q[:, :k]


def test_psd_pinv_matches_torch_pinv_where_pinv_converges():
    d = 24
    Q = _qr(d, d, 0)
    A = Q @ torch.diag(torch.linspace(0.5, 40.0, d)) @ Q.t()
    A = 0.5 * (A + A.t())
    got, info = psd_pinv(A)
    want = torch.linalg.pinv(A)
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-4)
    assert info["pinv_rank"] == d
    # rank-k with exact zeros, constructed in fp64 so the reference's nulls are exact
    d, k = 16, 5
    g = torch.Generator().manual_seed(1)
    Qd, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    Qd = Qd[:, :k]
    Ad = Qd @ torch.diag(torch.linspace(1.0, 9.0, k, dtype=torch.float64)) @ Qd.t()
    got, info = psd_pinv(Ad.float())
    assert info["pinv_rank"] == k
    assert torch.allclose(got, torch.linalg.pinv(Ad).float(), atol=1e-5, rtol=1e-4)


def test_psd_pinv_survives_the_degenerate_spectrum():
    """rank-k PSD with d-k EXACT zeros plus repeated positive eigenvalues
    (the 10-layer Llama crash's spectrum) inverts without LinAlgError and
    matches the analytically reduced k-dim solve."""
    d, k, n = 384, 48, 12
    lam = 10.0
    Q = _qr(d, k, 3)
    P = Q @ Q.t()
    g = torch.Generator().manual_seed(4)
    base = torch.randn(n // 2, d, generator=g)
    H = torch.cat([base, base])                  # duplicated rows: n << d
    r = torch.randn(d, generator=g)
    tilde, info = ridge_delta(H, P, r, lambda_reg=lam, solver="psd_pinv")
    assert info["pinv_rank"] == k
    Z = (H.float() @ Q).double()
    b_mat = ((H.float() @ P).t() @ r.float().unsqueeze(0).expand(n, -1)).double()
    ref = Q.double() @ torch.linalg.solve(
        Z.t() @ Z + lam * torch.eye(k, dtype=torch.float64), Q.t().double() @ b_mat)
    assert torch.allclose(tilde, ref.float(), atol=1e-4, rtol=1e-3)
    # an all-zero matrix pseudo-inverts to zero, no exception
    z, zinfo = psd_pinv(torch.zeros(8, 8))
    assert torch.equal(z, torch.zeros(8, 8)) and zinfo["pinv_rank"] == 0


def test_ridge_delta_solvers_agree_where_pinv_converges():
    d, n = 20, 300
    g = torch.Generator().manual_seed(2)
    benign = torch.randn(n, d, generator=g) * torch.logspace(0.5, -2.5, d)
    H = torch.randn(n, d, generator=g)
    r = torch.randn(d, generator=g)
    P, _ = null_space_projector(benign, 0.4)
    svd, info_svd = ridge_delta(H, P, r, lambda_reg=10.0)            # default
    eig, info_eig = ridge_delta(H, P, r, lambda_reg=10.0, solver="psd_pinv")
    assert torch.allclose(svd, eig, atol=1e-4, rtol=1e-3)
    # main's exact diagnostics key set on the default path; pinv_* only on eigh
    assert set(info_svd) == {"lambda_reg", "regression_residual",
                             "malicious_energy_kept"}
    assert {"pinv_rcond", "pinv_rank", "pinv_top_eigenvalue"} <= set(info_eig)
    assert RIDGE_SOLVERS == ("svd", "psd_pinv")
    with pytest.raises(ValueError, match="rcond"):
        ridge_delta(H, P, r, lambda_reg=10.0, rcond=1e-3)      # svd + rcond refuses
    with pytest.raises(ValueError, match="solver"):
        ridge_delta(H, P, r, lambda_reg=10.0, solver="lu")


def test_legacy_path_keeps_the_svd_solver(tiny_llama, monkeypatch):
    """train_alphasteer_content calls torch.linalg.pinv and never psd_pinv;
    the multilayer trainer is the ONLY psd_pinv caller and never calls
    torch.linalg.pinv."""
    model, tok = tiny_llama
    calls = {"pinv": 0, "psd": 0}
    real_pinv, real_psd = torch.linalg.pinv, als.psd_pinv

    def spy_pinv(*a, **k):
        calls["pinv"] += 1
        return real_pinv(*a, **k)

    def spy_psd(*a, **k):
        calls["psd"] += 1
        return real_psd(*a, **k)

    monkeypatch.setattr(torch.linalg, "pinv", spy_pinv)
    monkeypatch.setattr(als, "psd_pinv", spy_psd)
    kw = dict(batch_size=3, calib_prompts=2, calib_max_new_tokens=4,
              rep_threshold=0.0, ppl_guard_factor=1e9, scale_ladder=(0.5, 1.0))
    train_alphasteer_content(model, tok, _BENIGN, _HARMFUL, layer=1, **kw)
    assert calls == {"pinv": 1, "psd": 0}
    calls.update(pinv=0, psd=0)
    train_alphasteer_content_multilayer(model, tok, _BENIGN, _HARMFUL,
                                        layers=[0, 1], calibrate=False)
    assert calls == {"pinv": 0, "psd": 2}


# ---------------------------------------------------------------- legacy byte-identity


def _structural_diff(a_path: Path, b_path: Path) -> str:
    a = torch.load(a_path, map_location="cpu", weights_only=False)
    b = torch.load(b_path, map_location="cpu", weights_only=False)
    lines = []
    if list(a) != list(b):
        lines.append(f"key order/set: pre={list(a)} post={list(b)}")
    for k in set(a) | set(b):
        va, vb = a.get(k, "<absent>"), b.get(k, "<absent>")
        if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
            if not torch.equal(va, vb):
                lines.append(f"{k}: tensors differ (max abs {float((va - vb).abs().max()):.3e})")
        elif va != vb:
            lines.append(f"{k}: pre={va!r} post={vb!r}")
    return "\n".join(lines) or "(no structural difference — bytes differ elsewhere)"


def test_legacy_single_layer_fit_and_save_is_unchanged_by_the_port(tiny_llama, tmp_path):
    """A REAL train_alphasteer_content fit under the pre-port module
    (git show 04608a46:...) and under the ported one, same tiny model and
    seed, saved under the SAME basename (torch names the zip root after the
    file stem), produce sha256-identical files."""
    text = _git_show(f"{PREPORT_SHA}:src/sisteer/extraction/alphasteer.py")
    if text is None:
        pytest.skip(f"commit {PREPORT_SHA} is not available in this clone")
    pre = _import_module_text(text, "_preport_alphasteer", tmp_path)
    model, tok = tiny_llama
    kw = dict(layer=1, batch_size=4, calib_prompts=2, calib_max_new_tokens=4,
              rep_threshold=0.0, ppl_guard_factor=1e9, scale_ladder=(0.5, 1.0))
    paths = {}
    for name, mod in (("pre", pre), ("post", als)):
        torch.manual_seed(0)
        obj = mod.train_alphasteer_content(model, tok, _BENIGN, _HARMFUL, **kw)
        d = tmp_path / name
        d.mkdir()
        obj.save(d / "object.pt")
        paths[name] = d / "object.pt"
    shas = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}
    assert shas["pre"] == shas["post"], (
        "the ported module changed the legacy single-layer fit or its blob:\n"
        + _structural_diff(paths["pre"], paths["post"]))


def test_legacy_save_writes_mains_key_set_only(tiny_llama, tmp_path):
    model, tok = tiny_llama
    d = hidden_size(model)
    obj = AlphaSteerObject(model, layer=1, M=_M(model, 1), base_scale=2.0,
                           refusal=torch.randn(d, generator=torch.Generator().manual_seed(2)))
    assert obj.legacy_form
    path = tmp_path / "object.pt"
    obj.save(path)
    blob = torch.load(path, map_location="cpu", weights_only=True)
    assert list(blob) == ["kind", "layer", "M", "base_scale", "refusal", "metadata"]
    for banned in ("layers", "Ms", "refusals", "site", "steer_positions"):
        assert banned not in blob
    # and it round-trips through the loader as legacy lineage
    loaded = load_object(path, model, tok)
    assert loaded.legacy_form and loaded.layer == 1 and torch.equal(loaded.M, obj.M)


def test_legacy_built_object_at_a_nonlegacy_site_roundtrips_plural(tiny_llama, tmp_path):
    """A legacy-signature object moved to site='pre' has no five-key
    representation: it writes the plural blob and round-trips site/positions
    rather than silently dropping them."""
    model, tok = tiny_llama
    obj = AlphaSteerObject(model, layer=1, M=_M(model, 3), base_scale=1.0, site="pre")
    path = tmp_path / "object.pt"
    obj.save(path)
    blob = torch.load(path, map_location="cpu", weights_only=True)
    assert blob["site"] == "pre" and blob["layers"] == [1] and blob["layer"] is None
    loaded = load_object(path, model, tok)
    assert loaded.site == "pre" and loaded.layer == 1


def test_multi_layer_blob_round_trip(tiny_llama, tmp_path):
    """save -> load_blob -> identical object; legacy singular keys present
    and None (what _adapt_alphasteer reads: plural wins, singular is None,
    not missing)."""
    model, tok = tiny_llama
    d = hidden_size(model)
    g = torch.Generator().manual_seed(15)
    obj = AlphaSteerObject(
        model, base_scale=1.7, layers=[0, 1],
        Ms={0: _M(model, 16), 1: _M(model, 17)},
        refusals={0: torch.randn(d, generator=g), 1: torch.randn(d, generator=g)},
        steer_positions="prefill", site="pre",
        metadata={"fit_mode": "content_si_agnostic_multilayer"})
    assert not obj.legacy_form
    path = tmp_path / "object.pt"
    obj.save(path)
    blob = torch.load(path, map_location="cpu", weights_only=True)
    for k in ("layer", "M", "refusal"):
        assert k in blob and blob[k] is None
    assert blob["layers"] == [0, 1] and set(blob["Ms"]) == {0, 1}
    assert blob["site"] == "pre" and blob["steer_positions"] == "prefill"
    loaded = load_object(path, model, tok)
    assert isinstance(loaded, AlphaSteerObject) and not loaded.legacy_form
    assert loaded.layers == [0, 1] and loaded.site == "pre"
    assert loaded.base_scale == pytest.approx(1.7)
    for l in (0, 1):
        assert torch.equal(loaded.Ms[l], obj.Ms[l])
        assert torch.equal(loaded.refusals[l], obj.refusals[l])
    enc = tok("the quick brown fox", return_tensors="pt")
    with obj.apply(model, scale=1.0), torch.no_grad():
        want = model(**enc).logits
    with loaded.apply(model, scale=1.0), torch.no_grad():
        got = model(**enc).logits
    assert torch.equal(want, got)
    # a single-layer object built through layers=/Ms= is ALSO plural
    single_plural = AlphaSteerObject(model, layers=[1], Ms={1: _M(model, 18)})
    single_plural.save(tmp_path / "sp.pt")
    b2 = torch.load(tmp_path / "sp.pt", map_location="cpu", weights_only=True)
    assert b2["layer"] is None and b2["layers"] == [1]


def test_legacy_blob_loads_and_behaves_exactly_as_before(tiny_llama, tmp_path):
    model, tok = tiny_llama
    d = hidden_size(model)
    M = _M(model, 19)
    r = torch.randn(d, generator=torch.Generator().manual_seed(20))
    path = tmp_path / "alphasteer_legacy.pt"
    torch.save({"kind": "alphasteer", "layer": 1, "M": M.float().cpu(),
                "base_scale": 2.0, "refusal": r.float().cpu(),
                "metadata": {"layer": 1}}, path)
    loaded = load_object(path, model, tok)
    assert loaded.legacy_form and loaded.layer == 1
    assert loaded.steer_positions == "prefill" and loaded.site == "post"
    enc = tok("the quick brown fox jumps over", return_tensors="pt")
    one = _one_token(tok)
    with _legacy_apply(model, 1, M, 2.0), torch.no_grad():
        want = model(**enc).logits
    with loaded.apply(model, scale=1.0), torch.no_grad():
        got = model(**enc).logits
        decode = model(**one).logits
    assert torch.equal(want, got)
    assert torch.equal(_logits(model, one), decode)


def test_single_layer_default_bit_identical_to_legacy(tiny_llama):
    model, tok = tiny_llama
    M = _M(model, 0)
    obj = AlphaSteerObject(model, layer=1, M=M, base_scale=2.0)
    enc = tok("the quick brown fox jumps over", return_tensors="pt")
    base = _logits(model, enc)
    for scale in (1.0, 0.7):
        with _legacy_apply(model, 1, M, 2.0 * scale), torch.no_grad():
            want = model(**enc).logits
        with obj.apply(model, scale=scale), torch.no_grad():
            got = model(**enc).logits
        assert torch.equal(want, got)
        assert not torch.equal(base, got)
    prev = tok.padding_side
    tok.padding_side = "left"
    try:
        batch = tok(["hi", "a much longer prompt with several extra tokens"],
                    return_tensors="pt", padding=True)
    finally:
        tok.padding_side = prev
    with _legacy_apply(model, 1, M, 2.0), torch.no_grad():
        want = model(**batch).logits
    with obj.apply(model, scale=1.0), torch.no_grad():
        got = model(**batch).logits
    assert torch.equal(want, got)


# ---------------------------------------------------------------- object surface


def test_layer_and_M_properties_are_none_on_multilayer(tiny_llama):
    model, tok = tiny_llama
    d = hidden_size(model)
    single = AlphaSteerObject(model, layer=1, M=_M(model, 1), base_scale=1.0,
                              refusal=torch.randn(d, generator=torch.Generator().manual_seed(2)))
    assert single.layer == 1 and single.M is not None and single.refusal is not None
    multi = AlphaSteerObject(model, layers=[0, 1], Ms={0: _M(model, 3), 1: _M(model, 4)})
    assert multi.layer is None and multi.M is None and multi.refusal is None
    assert multi.n_params() == 2 * d * d and single.n_params() == d * d


def test_constructor_validation(tiny_llama):
    model, tok = tiny_llama
    d = hidden_size(model)
    M = torch.zeros(d, d)
    with pytest.raises(ValueError, match="EITHER"):
        AlphaSteerObject(model, layer=1, M=M, layers=[1], Ms={1: M})
    with pytest.raises(ValueError, match="EITHER"):
        AlphaSteerObject(model)
    with pytest.raises(ValueError, match="missing"):
        AlphaSteerObject(model, layers=[0, 1], Ms={0: M})
    with pytest.raises(ValueError, match="duplicate"):
        AlphaSteerObject(model, layers=[1, 1], Ms={1: M})
    with pytest.raises(ValueError, match="steer_positions"):
        AlphaSteerObject(model, layer=1, M=M, steer_positions="sometimes")
    with pytest.raises(ValueError, match="site"):
        AlphaSteerObject(model, layer=1, M=M, site="mid")
    assert STEER_POSITIONS == ("prefill", "all", "prompt_hold") and SITES == ("post", "pre")


def test_paper_constants_match_the_verified_deployment():
    assert PAPER_LLAMA31_LAYERS == [8, 9, 10, 11, 12, 13, 14, 16, 18, 19]
    assert set(PAPER_LLAMA31_NULLSPACE_RATIOS) == set(PAPER_LLAMA31_LAYERS)
    assert PAPER_LLAMA31_NULLSPACE_RATIOS[12] == 0.4
    assert PAPER_LLAMA31_NULLSPACE_RATIOS[13] == 0.5


# ---------------------------------------------------------------- padding / site / prefill


def test_last_valid_token_index_left_and_right_padding():
    right = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    assert last_valid_token_index(right, 2, 5, "cpu").tolist() == [2, 4]
    left = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    assert last_valid_token_index(left, 2, 5, "cpu").tolist() == [4, 4]
    assert last_valid_token_index(None, 3, 7, "cpu").tolist() == [6, 6, 6]
    assert last_valid_token_index(right, 2, 9, "cpu").tolist() == [8, 8]   # KV-cache decode


def test_site_pre_hook_edits_the_stream_entering_the_block(tiny_llama):
    """site='pre' at block l equals a hand-written forward-pre edit, and
    equals the SAME edit applied at resid_post of block l-1."""
    model, tok = tiny_llama
    M = _M(model, 9)
    mag = 1.5
    enc = tok("the quick brown fox jumps over", return_tensors="pt")
    Mf = M.float()

    def edit(h):
        add = mag * (h[:, -1, :] @ Mf.to(h.dtype))
        return h + add.to(h.dtype).unsqueeze(1)

    pre1 = AlphaSteerObject(model, layer=1, M=M, base_scale=mag, site="pre")
    with _hand_pre_hook(model, 1, edit), torch.no_grad():
        want = model(**enc).logits
    with pre1.apply(model, scale=1.0), torch.no_grad():
        got = model(**enc).logits
    assert torch.equal(want, got)
    post0 = AlphaSteerObject(model, layer=0, M=M, base_scale=mag, site="post")
    with post0.apply(model, scale=1.0), torch.no_grad():
        got_post0 = model(**enc).logits
    assert torch.equal(got, got_post0), "resid_pre(l) must equal resid_post(l-1)"
    post1 = AlphaSteerObject(model, layer=1, M=M, base_scale=mag)
    with post1.apply(model, scale=1.0), torch.no_grad():
        assert not torch.equal(got, model(**enc).logits)


def test_prefill_only_leaves_decode_untouched(tiny_llama):
    model, tok = tiny_llama
    one = _one_token(tok)
    base = _logits(model, one)
    for site in SITES:
        obj = AlphaSteerObject(model, layers=[0, 1],
                               Ms={0: _M(model, 5), 1: _M(model, 6)},
                               base_scale=1.5, steer_positions="prefill", site=site)
        with obj.apply(model, scale=1.0), torch.no_grad():
            assert torch.equal(base, model(**one).logits), site
        enc = tok("the quick brown fox", return_tensors="pt")
        base_prompt = _logits(model, enc)
        with obj.apply(model, scale=1.0), torch.no_grad():
            assert not torch.equal(base_prompt, model(**enc).logits), site


@pytest.mark.parametrize("padding_side", ["right", "left"])
def test_padded_batches_pick_the_correct_last_valid_token(tiny_llama, padding_side):
    model, tok = tiny_llama
    M = _M(model, 14)
    mag = 2.0
    obj = AlphaSteerObject(model, layer=1, M=M, base_scale=mag)
    prev = tok.padding_side
    tok.padding_side = padding_side
    try:
        enc = tok(["hi", "a much longer prompt with several extra tokens"],
                  return_tensors="pt", padding=True)
    finally:
        tok.padding_side = prev
    mask = enc["attention_mask"]
    S = mask.shape[1]
    assert int(mask.sum()) < 2 * S
    with capture_layer_outputs(model, [1]) as cap, torch.no_grad():
        model(**enc)
    H = cap[1].clone()
    idx = last_valid_token_index(mask, 2, S, H.device)
    h_last = H[torch.arange(2), idx]
    expected_add = mag * (h_last.float() @ M.float())
    with obj.apply(model, scale=1.0), capture_layer_outputs(model, [1]) as cap2, \
            torch.no_grad():
        model(**enc)
    diff = cap2[1] - H
    for i in range(2):
        assert torch.allclose(diff[i], expected_add[i].expand_as(diff[i]), atol=1e-5)
    if padding_side == "right":
        wrong = mag * (H[0, -1, :].float() @ M.float())
        assert not torch.allclose(diff[0, 0], wrong, atol=1e-4)


def test_multilayer_matches_nested_single_layer_objects(tiny_llama):
    model, tok = tiny_llama
    M0, M1 = _M(model, 10), _M(model, 11)
    enc = tok("the quick brown fox jumps over", return_tensors="pt")
    ml = AlphaSteerObject(model, layers=[0, 1], Ms={0: M0, 1: M1}, base_scale=1.2)
    s0 = AlphaSteerObject(model, layer=0, M=M0, base_scale=1.2)
    s1 = AlphaSteerObject(model, layer=1, M=M1, base_scale=1.2)
    with s0.apply(model, scale=1.0), s1.apply(model, scale=1.0), torch.no_grad():
        want = model(**enc).logits
    with ml.apply(model, scale=1.0), torch.no_grad():
        got = model(**enc).logits
    assert torch.equal(want, got)


# ---------------------------------------------------------------- the trainer


def test_multilayer_content_fit_end_to_end(tiny_llama):
    model, tok = tiny_llama
    d = hidden_size(model)
    obj = train_alphasteer_content_multilayer(
        model, tok, _BENIGN, _HARMFUL, layers=[0, 1],
        nullspace_ratios={0: 0.25, 1: 0.75}, site="pre", **_CALIB_KW)
    assert obj.layers == [0, 1] and set(obj.Ms) == {0, 1} and obj.site == "pre"
    md = obj.metadata
    assert md["fit_mode"] == "content_si_agnostic_multilayer"
    assert md["lambda_reg"] == 10.0 and md["ridge_solver"] == "psd_pinv"
    assert md["per_layer"][0]["nullspace_rank"] == int(0.25 * d)
    assert md["per_layer"][1]["nullspace_rank"] == int(0.75 * d)
    assert "pinv_rank" in md["per_layer"][1]
    assert [s["what"] for s in md["sample_floor"]] == [
        "null-space rows (benign)", "ridge rows (harmful)"]
    assert md["base_scale_source"] == "calibrate_base_scale"
    assert obj.base_scale == pytest.approx(0.5)
    one = _one_token(tok)
    with obj.apply(model, scale=1.0), torch.no_grad():
        assert torch.equal(_logits(model, one), model(**one).logits)


def test_uncalibrated_fit_has_unit_base_scale(tiny_llama):
    model, tok = tiny_llama
    obj = train_alphasteer_content_multilayer(
        model, tok, _BENIGN, _HARMFUL, layers=[1], calibrate=False)
    assert obj.base_scale == 1.0
    assert obj.metadata["base_scale_source"].startswith("uncalibrated_fit")
    assert "distinct3" not in json_keys(obj.metadata)


def json_keys(d, acc=None):
    acc = set() if acc is None else acc
    for k, v in d.items():
        acc.add(str(k))
        if isinstance(v, dict):
            json_keys(v, acc)
    return acc


def test_precaptured_acts_skip_capture_and_match_a_fresh_fit(tiny_llama):
    """Handing the trainer the screen's activations (fit rows, in order)
    produces the SAME matrices as capturing afresh, and no forward runs."""
    model, tok = tiny_llama
    # the SAME capture batch size on both paths: padding width changes the
    # kernels' reduction order, so widths are not bitwise (fit_class_b's own note)
    fresh = train_alphasteer_content_multilayer(
        model, tok, _BENIGN, _HARMFUL, layers=[0, 1], site="pre", calibrate=False,
        batch_size=4)
    b = _collect_last_acts(model, tok, [build_chat_prompt(tok, p) for p in _BENIGN],
                           [0, 1], "pre", 4)
    h = _collect_last_acts(model, tok, [build_chat_prompt(tok, p) for p in _HARMFUL],
                           [0, 1], "pre", 4)
    forwards = []
    hook = model.register_forward_pre_hook(lambda *_: forwards.append(1))
    try:
        pre = train_alphasteer_content_multilayer(
            model, tok, _BENIGN, _HARMFUL, layers=[0, 1], site="pre",
            calibrate=False, precaptured=(b, h), batch_size=4)
    finally:
        hook.remove()
    assert forwards == []
    for l in (0, 1):
        assert torch.equal(fresh.Ms[l], pre.Ms[l])
    assert pre.metadata["capture"]["source"] == "precaptured"
    with pytest.raises(ValueError, match="lack layers"):
        train_alphasteer_content_multilayer(
            model, tok, _BENIGN, _HARMFUL, layers=[0, 1], calibrate=False,
            precaptured=({0: b[0]}, h))
    with pytest.raises(ValueError, match="rows"):
        train_alphasteer_content_multilayer(
            model, tok, _BENIGN[:3], _HARMFUL, layers=[0, 1], calibrate=False,
            precaptured=(b, h))


def test_multilayer_content_fit_validation(tiny_llama):
    model, tok = tiny_llama
    with pytest.raises(ValueError, match="missing layers"):
        train_alphasteer_content_multilayer(
            model, tok, _BENIGN, _HARMFUL, layers=[0, 1],
            nullspace_ratios={0: 0.5}, calibrate=False)
    with pytest.raises(ValueError, match="duplicate"):
        train_alphasteer_content_multilayer(
            model, tok, _BENIGN, _HARMFUL, layers=[1, 1], calibrate=False)
    with pytest.raises(ValueError, match="non-empty"):
        train_alphasteer_content_multilayer(
            model, tok, [], _HARMFUL, layers=[1], calibrate=False)


def test_capture_id_verifier_summary_shape(tiny_llama):
    _, tok = tiny_llama
    v = CaptureIdVerifier(tok, n_sample=2)
    texts = [build_chat_prompt(tok, p) for p in _BENIGN[:3]]
    prev = tok.padding_side
    tok.padding_side = "right"
    try:
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    finally:
        tok.padding_side = prev
    v.check(texts, enc)
    s = v.summary()
    assert s["n_verified"] == 2 and s["ids_match_serving_render"] is True
    with pytest.raises(ValueError):
        CaptureIdVerifier(tok, n_sample=0)

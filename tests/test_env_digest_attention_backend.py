"""The env digest fingerprints the LIVE SDPA backend — fold-in-when-non-default.

The SDPA backend selection materially changes numerics (cuDNN's SDPA
re-plans per input shape and picks different kernels — the footgun
tests/test_fast_sdpa.py locks the enforcement for), so two runs on
different backends must not share an env digest. But env_digest is
provenance recorded into every ResultRow and cell manifest, and a naive
new field would make every digest recorded from here on differ from the
corpus for no real change. So the field follows the parity_n pattern: the
project-pinned state (store.PINNED_SDPA_STATE — what configure_fast_sdpa /
pin_attention_backends establish) stays OUT of the hashed payload
entirely, keeping those digests byte-identical to the existing corpus,
and the payload gains ``attention_backend`` exactly when the live flags
deviate.

The fingerprint reads torch's live flag state at record time — never
"configure_fast_sdpa was called", which under SISTEER_KEEP_CUDNN_SDP=1 (or
a later third-party flip) records an intention the kernels don't honor.
"""

import re
from pathlib import Path

import pytest
import torch

from sisteer.capeval import store as cs
from sisteer.models import configure_fast_sdpa

REPO = Path(__file__).resolve().parent.parent

_LEGACY_FIELDS = ("python", "torch", "transformers", "lm_eval", "vllm",
                  "evalplus", "cuda_runtime", "cuda_driver", "gpu_name")


@pytest.fixture()
def sdpa_flags_restored():
    """Snapshot + restore the four global SDPA flags — tests here mutate
    process-global torch state (as test_fast_sdpa.py already does)."""
    tbc = torch.backends.cuda
    before = {
        "cudnn": tbc.cudnn_sdp_enabled(),
        "flash": tbc.flash_sdp_enabled(),
        "mem": tbc.mem_efficient_sdp_enabled(),
        "math": tbc.math_sdp_enabled(),
    }
    yield
    tbc.enable_cudnn_sdp(before["cudnn"])
    tbc.enable_flash_sdp(before["flash"])
    tbc.enable_mem_efficient_sdp(before["mem"])
    tbc.enable_math_sdp(before["math"])


def _pin_all_default():
    """Put the process into the project-pinned state explicitly — a full
    suite run has other tests toggling these globals, so no test here may
    assume the ambient state."""
    tbc = torch.backends.cuda
    tbc.enable_cudnn_sdp(False)
    tbc.enable_flash_sdp(True)
    tbc.enable_mem_efficient_sdp(True)
    tbc.enable_math_sdp(True)


def test_attention_backend_state_reads_live_flags(sdpa_flags_restored):
    _pin_all_default()
    assert cs.attention_backend_state() == cs.PINNED_SDPA_STATE
    torch.backends.cuda.enable_cudnn_sdp(True)
    assert cs.attention_backend_state() == {**cs.PINNED_SDPA_STATE,
                                            "cudnn_sdp": True}


def test_digest_unchanged_at_pinned_default(sdpa_flags_restored):
    """At the pinned backend the hashed payload is EXACTLY the legacy
    field set — the construction of every legacy field is untouched, so
    the digest is byte-identical to what pre-fingerprint code computed
    (the invariance that keeps the recorded corpus comparable)."""
    _pin_all_default()
    fields = cs.env_digest_fields()
    assert "attention_backend" not in fields
    assert set(fields) == set(_LEGACY_FIELDS)
    d = cs.env_digest()
    assert len(d) == 16 and d == cs.env_digest()


def test_digest_changes_when_backend_differs(sdpa_flags_restored):
    _pin_all_default()
    d_pinned = cs.env_digest()
    torch.backends.cuda.enable_cudnn_sdp(True)
    fields = cs.env_digest_fields()
    assert fields["attention_backend"] == {**cs.PINNED_SDPA_STATE,
                                           "cudnn_sdp": True}
    assert cs.env_digest() != d_pinned
    # every deviation axis registers, not just cuDNN
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    assert cs.env_digest() != d_pinned
    assert cs.env_digest_fields()["attention_backend"]["math_sdp"] is False


def test_records_live_state_not_the_pin_call(sdpa_flags_restored):
    """A digest recorded after configure_fast_sdpa() still fingerprints
    whatever the flags ARE at record time: a later flip (third-party code,
    or the pin being a no-op under SISTEER_KEEP_CUDNN_SDP=1) must show up.
    Recording the intended state instead of the actual one is the failure
    mode this fingerprint exists to prevent."""
    _pin_all_default()
    configure_fast_sdpa()
    d_pinned = cs.env_digest()
    torch.backends.cuda.enable_cudnn_sdp(True)  # post-pin flip
    assert cs.env_digest() != d_pinned
    assert cs.env_digest_fields()["attention_backend"]["cudnn_sdp"] is True


def test_keep_cudnn_env_deviation_is_fingerprinted(sdpa_flags_restored,
                                                   monkeypatch):
    """SISTEER_KEEP_CUDNN_SDP=1 turns the pin into a no-op (a deliberate
    cuDNN-baseline reproduction) — the digest must record that the run
    really kept cuDNN, not that a pin was requested."""
    _pin_all_default()
    monkeypatch.setenv("SISTEER_KEEP_CUDNN_SDP", "1")
    torch.backends.cuda.enable_cudnn_sdp(True)  # torch-default-on state
    configure_fast_sdpa()  # no-op under the override
    assert torch.backends.cuda.cudnn_sdp_enabled() is True
    assert cs.env_digest_fields()["attention_backend"]["cudnn_sdp"] is True


def test_every_recording_script_pins_the_backend():
    """Every scripts/ entry point that records an env digest (directly via
    env_digest(), or through runtime.resolve_cell) must also call
    configure_fast_sdpa — recorded BEFORE the first model load, an
    unpinned process fingerprints the torch-default backend state while
    scoring runs on the pinned one (the ordering cell_controls_run's
    cmd_generate comment documents). Source-level, same style as
    test_fast_sdpa.test_every_model_load_site_is_guarded."""
    recorder = re.compile(r"\benv_digest\s*\(|\.resolve_cell\(|\[\"resolve_cell\"\]")
    offenders = []
    for p in sorted((REPO / "scripts").glob("*.py")):
        src = p.read_text()
        if recorder.search(src) and "configure_fast_sdpa" not in src:
            offenders.append(p.name)
    assert not offenders, (
        f"scripts recording env_digest without pinning the SDPA backend "
        f"first: {offenders} — call sisteer.models.configure_fast_sdpa() "
        "before the first env_digest()/resolve_cell() record")

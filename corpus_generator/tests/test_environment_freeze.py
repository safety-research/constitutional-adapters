"""E2: the smoke verdicts (reconciliation mode, structured scores) freeze into
the RUN namespace at first resolve. `corpusgen smoke` may overwrite the global
environment.json at will; a run in flight never sees the flip, an explicit
config contradiction is a hard error, and each verdict resolves independently
against its own null sentinel."""

from __future__ import annotations

import json

import pytest

from corpus_generator.batch import WaveHalt, WaveRequest
from corpus_generator.config import RunConfig
from corpus_generator.pipeline import make_engine, resolve_environment
from fakes import FakeBatchClient, text_result


def _cfg(tmp_path, run_id="freeze-test", **engine) -> RunConfig:
    return RunConfig.model_validate(
        {
            "run_id": run_id,
            "data_dir": str(tmp_path / "data"),
            "engine": {"poll_interval_s": 0.0, "visibility_wait_s": 0.0, **engine},
        }
    )


def _write_global_env(tmp_path, **verdicts) -> None:
    env_dir = tmp_path / "data" / "manifests"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "environment.json").write_text(json.dumps(verdicts))


def test_verdict_frozen_at_first_resolve_survives_smoke_overwrite(tmp_path):
    _write_global_env(tmp_path, nonce_metadata=False, output_config_on_batch=True)
    cfg = resolve_environment(_cfg(tmp_path), root=tmp_path, on_event=lambda m: None)
    assert cfg.engine.nonce_metadata is False
    assert cfg.engine.use_structured_scores is True
    assert (cfg.run_dir(tmp_path) / "ENVIRONMENT.json").exists()

    # `corpusgen smoke` re-runs and flips BOTH verdicts in the global file
    _write_global_env(tmp_path, nonce_metadata=True, output_config_on_batch=False)
    cfg2 = resolve_environment(_cfg(tmp_path), root=tmp_path, on_event=lambda m: None)
    assert cfg2.engine.nonce_metadata is False, "run must keep its frozen verdict"
    assert cfg2.engine.use_structured_scores is True

    # a NEW run_id picks up the new global verdicts
    cfg3 = resolve_environment(
        _cfg(tmp_path, run_id="freeze-test-2"), root=tmp_path, on_event=lambda m: None
    )
    assert cfg3.engine.nonce_metadata is True
    assert cfg3.engine.use_structured_scores is False


def test_explicit_config_conflicting_with_frozen_verdict_hard_errors(tmp_path):
    _write_global_env(tmp_path, nonce_metadata=False, output_config_on_batch=False)
    resolve_environment(_cfg(tmp_path), root=tmp_path, on_event=lambda m: None)
    with pytest.raises(RuntimeError, match="nonce_metadata"):
        resolve_environment(
            _cfg(tmp_path, nonce_metadata=True), root=tmp_path, on_event=lambda m: None
        )
    with pytest.raises(RuntimeError, match="use_structured_scores"):
        resolve_environment(
            _cfg(tmp_path, use_structured_scores=True),
            root=tmp_path, on_event=lambda m: None,
        )


def test_verdicts_resolve_independently(tmp_path):
    """Each verdict has its OWN null sentinel: pinning one in the config must
    not stop the other from resolving off the smoke verdict (the old code
    gated both behind nonce_metadata's null-ness)."""
    _write_global_env(tmp_path, nonce_metadata=False, output_config_on_batch=True)
    cfg = resolve_environment(
        _cfg(tmp_path, nonce_metadata=False),  # pinned; structured left null
        root=tmp_path, on_event=lambda m: None,
    )
    assert cfg.engine.nonce_metadata is False
    assert cfg.engine.use_structured_scores is True, (
        "output_config verdict must resolve even when nonce_metadata is pinned"
    )


def test_missing_smoke_verdict_freezes_conservative_defaults(tmp_path):
    events = []
    cfg = resolve_environment(_cfg(tmp_path), root=tmp_path, on_event=events.append)
    assert cfg.engine.nonce_metadata is False
    assert cfg.engine.use_structured_scores is False
    assert any("WARNING" in e for e in events)
    # and the conservative defaults are themselves frozen
    _write_global_env(tmp_path, nonce_metadata=True, output_config_on_batch=True)
    cfg2 = resolve_environment(_cfg(tmp_path), root=tmp_path, on_event=lambda m: None)
    assert cfg2.engine.nonce_metadata is False


def _echo(cid, params, attempt):
    return text_result(cid, f"echo {cid}")


class Crash(BaseException):
    pass


def test_no_nonce_journal_flipped_verdict_resume_creates_zero_batches(tmp_path):
    """The reviewer's E2 crash test, end-to-end through resolve_environment +
    make_engine: a no-nonce journaled chunk, then the global verdict flips;
    the resumed run must create ZERO new batches."""
    _write_global_env(tmp_path, nonce_metadata=False, output_config_on_batch=False)
    cfg = resolve_environment(_cfg(tmp_path), root=tmp_path, on_event=lambda m: None)
    client = FakeBatchClient(_echo, metadata_supported=False)

    def crash_pre_create(point, info):
        if point == "pre_create":
            raise Crash(point)

    engine1 = make_engine(cfg, client, sleep=lambda s: None, hooks=crash_pre_create)
    wave_dir = cfg.run_dir(tmp_path) / "advice" / "w"
    reqs = [WaveRequest(custom_id=f"req-{i:03d}", params={"i": i}) for i in range(3)]
    with pytest.raises(Crash):
        engine1.run_wave(wave_dir, reqs, wave_name="w")

    # smoke re-runs and flips the global verdict to the nonce branch
    _write_global_env(tmp_path, nonce_metadata=True, output_config_on_batch=False)
    cfg2 = resolve_environment(_cfg(tmp_path), root=tmp_path, on_event=lambda m: None)
    assert cfg2.engine.nonce_metadata is False, "frozen verdict must win"
    engine2 = make_engine(cfg2, client, sleep=lambda s: None)
    # no-nonce reconciliation, zero candidates -> HALT; nothing created
    with pytest.raises(WaveHalt):
        engine2.run_wave(wave_dir, reqs, wave_name="w")
    assert client.create_count() == 0, "resume after a flipped verdict created a batch"

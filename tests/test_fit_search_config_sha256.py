"""fit_search.config_sha256 is a 64-hex sha256 or 'unverified' (schema_def SHA256_OR_UNVERIFIED), enforced at the writer.

Stage E 2026-10-01: the search OPENING and every update/close carried the DESIGN's 16-hex config_sha, and the
harvester refused each whole fit_search record ("config_sha256='02709c5ed1076cfb' does not match SHA256_OR_UNVERIFIED") —
no fit_search row ever existed and every fit_trial of the search blocked on the missing parent. Pinned here: the writer's
normaliser (64-hex kept; anything else -> 'unverified' with a notice) in open_search AND in the chained update/close; the
design round opens the search with the config FILE's full sha256; a close with `analyze --config F` records F's full sha
and refuses a file whose bytes disagree with the opening's recorded prefix.
"""
import hashlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts", REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import tune_arm  # noqa: E402
from sisteer.registry import emit_hpo  # noqa: E402

FULL = "a" * 64


def test_writer_normalises_config_sha256(capsys):
    assert emit_hpo.sha256_or_unverified(FULL) == FULL
    assert emit_hpo.sha256_or_unverified(FULL.upper()) == FULL
    assert emit_hpo.sha256_or_unverified(None) is None, "the column is nullable (a manual search has no config file)"
    fc = {}
    assert emit_hpo.sha256_or_unverified("02709c5ed1076cfb", fc=fc) == "unverified" and fc["config_sha256"]["confidence"] == "unverified"
    assert emit_hpo.sha256_or_unverified("unverified") == "unverified"
    assert emit_hpo.sha256_or_unverified("02709c5ed1076cfb") == "unverified", "a 16-hex design sha is not a sha256"
    assert "not a 64-hex sha256" in capsys.readouterr().err


def test_superseding_payload_normalises_an_old_opening():
    opening = {"search_id": "hpo-native:feedface", "campaign": "two_si", "method": "sobol", "config_sha256": "02709c5ed1076cfb", "extra": {}}
    payload = emit_hpo._superseding(opening, "hpo-native:feedface", 20, phase="closed", trials_in_flight=(), job_ids=None, extra=None)
    assert payload["config_sha256"] == "unverified" and payload["field_confidence_json"]["config_sha256"]["confidence"] == "unverified"
    none = emit_hpo._superseding({**opening, "config_sha256": None}, "hpo-native:feedface", 20, phase="closed", trials_in_flight=(), job_ids=None, extra=None)
    assert none.get("config_sha256") is None and "field_confidence_json" not in none
    good = emit_hpo._superseding({**opening, "config_sha256": FULL}, "hpo-native:feedface", 20, phase="closed", trials_in_flight=(), job_ids=None, extra=None)
    assert good["config_sha256"] == FULL


def test_design_round_records_the_full_file_sha(tmp_path):
    cfg = tmp_path / "fits.yaml"; cfg.write_text("model: m\n")
    full = tune_arm.config_file_sha256_full(str(cfg))
    assert len(full) == 64 and full == hashlib.sha256(cfg.read_bytes()).hexdigest()
    assert tune_arm._config_file_sha(str(cfg)) == full[:16], "the design's 16-hex key is the prefix of the recorded sha256"


def test_close_config_sha256_passes_verifies_or_refuses(tmp_path):
    cfg = tmp_path / "fits.yaml"; cfg.write_text("model: m\n")
    full = tune_arm.config_file_sha256_full(str(cfg))
    # no --config: the opening's value passes through (the writer records a non-64-hex as unverified)
    assert tune_arm.close_config_sha256({"config_sha256": full[:16]}, None) == full[:16]
    assert tune_arm.close_config_sha256({}, None) is None, "an opening without a config stays without one (nullable column)"
    # --config agreeing with a 16-hex opening (the pre-fix spelling) -> the full sha is recorded
    assert tune_arm.close_config_sha256({"config_sha256": full[:16]}, str(cfg)) == full
    # --config agreeing with a full opening
    assert tune_arm.close_config_sha256({"config_sha256": full}, str(cfg)) == full
    # another file refuses by name
    other = tmp_path / "other.yaml"; other.write_text("model: other\n")
    with pytest.raises(SystemExit, match="not the same config bytes"):
        tune_arm.close_config_sha256({"config_sha256": full[:16]}, str(other))
    # an 'unverified' opening accepts any file (nothing to disagree with) and records its sha
    assert tune_arm.close_config_sha256({"config_sha256": "unverified"}, str(other)) == tune_arm.config_file_sha256_full(str(other))


def test_analyze_accepts_config_for_the_close():
    import subprocess
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "tune_arm.py"), "analyze", "--help"], capture_output=True, text=True, timeout=300,
                       env={**__import__("os").environ, "PYTHONPATH": str(REPO / "src")})
    assert r.returncode == 0 and "--config" in r.stdout


def test_every_normalised_shape_passes_the_registry_validator():
    """The literal sentinel passed through WITHOUT a declaration, and validate.check_column_value
    refuses a bare `unverified` — the very refusal this fix exists to end. Every shape the normaliser can produce is checked
    against the production validator here, for an opening and for a superseding record."""
    from sisteer.registry import validate as VAL
    table = VAL.table_for("fit_search") or "fit_search"
    for raw in (FULL, FULL.upper(), "02709c5ed1076cfb", "unverified"):
        fc = {}
        v = emit_hpo.sha256_or_unverified(raw, fc=fc)
        payload = {"config_sha256": v, **({"field_confidence_json": fc} if fc else {})}
        assert VAL.check_column_value(table, "config_sha256", v, payload) is None, (raw, payload)
    assert VAL.check_column_value(table, "config_sha256", "unverified", {"config_sha256": "unverified"}) is not None, \
        "the bare sentinel is what the validator refuses"
    kw = dict(phase="closed", trials_in_flight=(), job_ids=None, extra=None)
    close = emit_hpo._superseding({"config_sha256": "unverified", "extra": {}}, "hpo-native:x", 3, **kw)
    assert close["field_confidence_json"]["config_sha256"]["confidence"] == "unverified"
    assert VAL.check_column_value(table, "config_sha256", close["config_sha256"], close) is None
    kept = emit_hpo._superseding({"config_sha256": "unverified", "extra": {},
                                  "field_confidence_json": {"config_sha256": {"confidence": "unverified", "reason": "manual search"}}},
                                 "hpo-native:x", 3, **kw)
    assert kept["field_confidence_json"]["config_sha256"]["reason"] == "manual search", "a producer's own declaration is kept"
    legacy = emit_hpo._superseding({"config_sha256": "02709c5ed1076cfb", "extra": {}}, "hpo-native:x", 3, **kw)
    assert legacy["config_sha256"] == "unverified" and VAL.check_column_value(table, "config_sha256", "unverified", legacy) is None
    full = emit_hpo._superseding({"config_sha256": FULL, "extra": {}}, "hpo-native:x", 3, **kw)
    assert full["config_sha256"] == FULL and "field_confidence_json" not in full

"""The LoRA hardware smoke's UNAVAILABLE path must exit NONZERO.

Three GPU jobs (232175, 232236, 233394) each reported COMPLETED 0:0
while writing ``lora_smoke_unavailable.json`` — the smoke could not run
(device_map/accelerate, a non-serializable callable collective_rpc,
accelerate again on a stale checkout), validated NOTHING, and still
read as a pass, so every one of those false greens cost a full GPU job
to discover. This file pins the outcome contract:

- bootstrap-unavailable is a FAILURE: exit NONZERO (3), never 0;
- it stays DISTINCT from validation failure (4), so sacct alone tells
  "could not run — fix the environment" from "ran and disproved — fix
  the serving stack";
- only a genuine success record exits 0.

The missing-object invocation below exercises the real ``main()``
unavailable path end to end (argument parsing -> dump -> return code)
without any GPU, hub access, or heavyweight import — it returns before
the tokenizer/model stages.
"""
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import petri_lora_smoke as smoke  # noqa: E402


def test_unavailable_exit_code_is_nonzero_and_distinct():
    assert smoke.EXIT_BOOTSTRAP != 0, (
        "EXIT_BOOTSTRAP == 0 makes an unavailable smoke report success: "
        "three GPU jobs went green while validating nothing")
    assert smoke.EXIT_FAILED != 0
    assert smoke.EXIT_BOOTSTRAP != smoke.EXIT_FAILED, (
        "triage must be able to tell 'could not run' from 'ran and "
        "disproved' straight off the exit code")


def test_missing_object_returns_nonzero_and_writes_unavailable(tmp_path):
    out = tmp_path / "out"
    rc = smoke.main(["--out-dir", str(out),
                     "--object", str(tmp_path / "nope" / "object.pt")])
    assert rc == smoke.EXIT_BOOTSTRAP
    assert rc != 0
    rec = json.loads((out / "lora_smoke_unavailable.json").read_text())
    assert "no object at" in rec["error"]
    # the evidence mirror always carries the same record
    assert (out / "lora_smoke_evidence.json").exists()


def test_unavailable_artifact_never_pairs_with_exit_zero(tmp_path):
    """The pair (lora_smoke_unavailable.json written, exit 0) is the
    exact false-green shape jobs 232175/233394 shipped; main() must be
    structurally unable to produce it for the missing-object path."""
    out = tmp_path / "out"
    rc = smoke.main(["--out-dir", str(out),
                     "--object", str(tmp_path / "absent.pt")])
    assert (out / "lora_smoke_unavailable.json").exists()
    assert rc != 0

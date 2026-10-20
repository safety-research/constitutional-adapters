"""math500-thinking-on-full22 — identity VERSION 1 is byte-identical to the
pre-change driver, pinned by GOLDEN fixtures captured from the driver at
7e768632 (BEFORE any lane edit) on the synthetic `world` of
``test_served_capability.py``: a STANDARD thinking-OFF run and a STANDARD
gemma thinking-ON run of the paired GSM8K control produce the SAME hashed
identity, `harness_config_hash`, cell-key directory and manifest (the
volatile leaves aside) after the lane's changes. Version 2 adds exactly the
declared keys, and a v1 and a v2 store never share a key.

The fixtures live under ``tests/fixtures/tinycap_identity_laneA/`` (S1's goldens,
captured at main f90557fd, live under ``tinycap_identity_v1`` — two producers, two
sets; decisions-laneA LA4). Capture (re)writes a fixture and then FAILS on
purpose: ``TINYCAP_IDENTITY_GOLDEN_WRITE=1``.

Provenance: BOTH sets were re-captured ONCE after 83345d3a
("inspect-sandboxes + swebench enter the client lock") changed
``configs/capeval/tinycap-client-requirements.txt``, whose sha256 is the hashed
identity leaf ``frameworks.client_lock_sha256`` — a dependency change moves every
standard identity BY DESIGN:

* lock sha256 ``65c0d3db0558deccfa1faebefa1cced1bd50b0984d029fe5b8d1378254e1b29e``
  → ``0f21fd4d92dc56b45e966ddf666463d268e547c82ba6bf54b2e9b1fffd47cc0f``
* ``harness_config_hash`` OFF ``b93b6a442cbb15ef`` → ``b7383e446415e9ae``,
  ON/gemma ``e4cfe879c565289c`` → ``4325581deb6ce0f7`` (and the cell dirs)

This set's leaf-by-leaf diff old → new was exactly 8 of 260 leaves, all derived
from the lock leaf: ``cell_dir_name``, ``harness_config_hash``,
``identity.frameworks.client_lock_sha256``, the three manifest copies of the lock
sha (``manifest.frameworks`` / ``manifest.resolved_config.tinycap.frameworks`` /
``manifest.tinycap.frameworks``), ``manifest.harness_config_hash`` and
``manifest.row_constants.harness_config_hash``. Nothing else moved. This set
carries no provenance field, so this docstring is the record; any further
departure is a real identity change to be judged, never re-captured on sight.

Re-captured a SECOND time on the ifeval-canonical-battery branch, for the same
leaf: the IFEval checker dependencies (``instruction_following_eval`` as a second
VCS line, ``immutabledict``, ``langdetect`` through ``inspect_evals[ifeval]``)
recompiled the client lock (the ``tinycap_identity_v1`` set was re-captured with
that change; this set was re-captured at the merge):

* lock sha256 ``0f21fd4d92dc56b45e966ddf666463d268e547c82ba6bf54b2e9b1fffd47cc0f``
  → ``910b48517515054c2a53104ea8a1ad671dc90369f1c94d05e470e817dfa12c6e``
* ``harness_config_hash`` OFF ``b7383e446415e9ae`` → ``2fb1c95638ae5620``,
  ON/gemma ``4325581deb6ce0f7`` → ``8c9edcd9a443aabf`` (and the cell dirs)

Leaf-by-leaf, old → new: the SAME 8 of 259 leaves as above, every one derived
from the lock leaf; nothing else moved.

Re-captured a THIRD time, both sets and the row-engine manifest, when the tiny
loglikelihood battery and its third-party harness were removed. That is an
identity change, judged rather than re-captured on sight: the client lock lost
those packages, and the frameworks and versions blocks lost the keys that named
them.

* lock sha256 ``910b48517515054c2a53104ea8a1ad671dc90369f1c94d05e470e817dfa12c6e``
  → ``5a6e51efa757d8bc5a4016f679c47bf4e2b8c06305a1afd9a50ce50410743d49``
* ``harness_config_hash`` OFF ``2fb1c95638ae5620`` → ``2f2d94fcbf191002``,
  ON/gemma ``8c9edcd9a443aabf`` → ``a8cf0e43a1bc019f`` (and the cell dirs)

Leaf-by-leaf, old → new: 19 leaves in each fixture. The same 8 lock-derived
leaves as above, plus 11 removed keys: ``frameworks.lm_eval`` and
``frameworks.tinybenchmarks`` in the identity and in each of its three manifest
copies, and ``versions.served_ll`` in the identity and in its two manifest
copies. Nothing else moved.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import tinycap_golden as golden

from test_served_capability import (  # noqa: F401  (`world` is a fixture)
    REPO,
    HeaderTok,
    _cell_dir,
    _percell_argv,
    srun,
    stub_framework_versions,
    world,
    write_standup_endpoint,
)

from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as tc

GOLDEN_DIR = REPO / "tests" / "fixtures" / "tinycap_identity_laneA"
#: THIS set's volatile leaves (captured with `written_at` / `ts` masked). S1's
#: `tinycap_identity_v1` set has its own table in `test_served_capability.py`;
#: the two are never merged — each set is what its bytes were captured under.
_VOLATILE = {"tree_sha": "<TREE_SHA>", "env_digest": "<ENV_DIGEST>", "written_at": "<TS>", "ts": "<TS>"}


def _normalise(obj, *, tmp_path: Path):
    return golden.normalise(obj, tmp_path=tmp_path, volatile=_VOLATILE)


def isv_off_cap() -> int:
    """The paired GSM8K's thinking-OFF table cap, read off the plan builder's table."""
    from sisteer.capeval.tiers import inspect_served as isv

    return int(isv.MAX_TOKENS_OFF[tc.GSM8K_PAIRED_TASK_NAME])


def _fake_gsm8k_leg(monkeypatch, calls):
    from sisteer.capeval.tiers import inspect_served

    async def fake_run_inspect_tasks(*, atoms, route, routes, system, store, template,
                                     thinking, base_url, limits=None, **kw):
        calls.append({"atoms": list(atoms), "thinking": thinking, "kw": sorted(kw)})
        task, n = tc.GSM8K_STANDARD_TASK, tc.GSM8K_PAIRED_N
        summary = {"acc": 0.8, "n": n, "n_requests": n, "n_rejected": 0,
                   "samples_per_item": 1, "task": task,
                   "item_set": tc.GSM8K_ITEM_SET_PAIRED, "framework": tc.INSPECT_TASK_VERSION,
                   "engine": "vllm", "route": route.virtual_model, "thinking": thinking,
                   "render": tc.THINKING_RENDER_NATIVE, "wall_clock_s": 1.0,
                   "max_tokens": 512, "temperature": 0.0}
        completion.finish_task(store.cell_dir, task=task, n_items=n, n_expected_full=n,
                               limits=limits, n_rows=n, n_requests=n, n_rejected=0,
                               wall_clock_s=1.0, summary=summary)
        return {task: summary}

    monkeypatch.setattr(inspect_served, "run_inspect_tasks", fake_run_inspect_tasks)


def _run(world, monkeypatch, *, arm: str, reasoning_parser=None, extra=()):  # noqa: F811
    stub_framework_versions(monkeypatch)
    calls: list = []
    _fake_gsm8k_leg(monkeypatch, calls)
    if reasoning_parser is not None:
        write_standup_endpoint(world.table_path, reasoning_parser=reasoning_parser)
        monkeypatch.setattr(srun, "_load_served_tok", lambda mid, rev: HeaderTok(moves=False))
    srun.main(_percell_argv(world, tc.CONTROL_GSM8K) + [tc.FLAG_THINKING, arm, *extra])
    cell_dir = _cell_dir(world.out_root)
    manifest = json.loads((cell_dir / "manifest.json").read_text())
    return cell_dir, manifest, calls


@pytest.mark.parametrize("name,arm,reasoning_parser", [
    ("standard_off_gsm8k", tc.THINKING_OFF, None),
    ("standard_on_gemma_gsm8k", tc.THINKING_ON, "gemma4"),
])
def test_identity_version_1_matches_the_pre_change_golden_identity_and_manifest(
        world, monkeypatch, name, arm, reasoning_parser):  # noqa: F811
    """A standard thinking-OFF run and a standard gemma thinking-ON run
    (the paired GSM8K control on the fitted cell, identity version 1 = the
    default) produce the SAME hashed identity, `harness_config_hash`, cell-key
    directory and manifest as the golden driver did on this world — first the
    driver at 7e768632, re-captured once after the client-lock change (the
    module docstring names the leaves and the old/new sha) — compared byte for
    byte (paths under the tmp dir, the producer tree sha, the env digest and
    timestamps normalised)."""
    cell_dir, manifest, _ = _run(world, monkeypatch, arm=arm, reasoning_parser=reasoning_parser)
    ident = manifest["resolved_config"]["tinycap"]
    got = {"cell_dir_name": cell_dir.name, "harness_config_hash": manifest["harness_config_hash"],
           "identity": ident, "manifest": _normalise(manifest, tmp_path=world.tmp_path)}
    # the compare FIRST: in capture mode (on the pre-change tree, where the names
    # below do not exist yet) the fixture is written before anything else runs
    golden.compare_or_write(GOLDEN_DIR / f"{name}.json", got)
    assert "identity_version" not in ident                             # v1 writes NO version key
    assert "thinking_header_ids" not in ident and "diagnostic" not in ident
    assert ident[tc.IDENTITY_KEY_MATH500_RETRY_MAX_TOKENS] == 8192      # the legacy literal on a non-MATH run
    assert set(ident) == set(tc.TINYCAP_IDENTITY_FIELDS)


def test_identity_version_2_adds_exactly_the_declared_keys_and_never_shares_a_key_with_v1(world, monkeypatch):  # noqa: F811
    """Version 2 on the same OFF run: `identity_version: 2` and the EXECUTED cap
    map (`max_tokens_by_task` keyed by the FULL row task) join the identity —
    nothing else on an OFF run (no header ids: no task is ON) — so the hash, the
    cell dir and the key set all differ from the v1 golden; the v1 golden itself
    is untouched by the flag's existence. The shared block is compared against
    the v1 GOLDEN's bytes, so this test also goes red whenever a hashed leaf the
    two versions share moves under the golden (the client-lock sha was exactly
    that — 15 of 16 shared keys identical, `frameworks` differing);
    the fix is the v1 re-capture, never a code change here."""
    v1 = json.loads((GOLDEN_DIR / "standard_off_gsm8k.json").read_text())
    cell_dir, manifest, calls = _run(world, monkeypatch, arm=tc.THINKING_OFF,
                                     extra=[tc.FLAG_TINYCAP_IDENTITY_VERSION, "2"])
    ident = manifest["resolved_config"]["tinycap"]
    assert ident[tc.IDENTITY_KEY_IDENTITY_VERSION] == 2
    assert ident[tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK] == {tc.GSM8K_STANDARD_TASK: 512}
    assert set(ident) - set(v1["identity"]) == {tc.IDENTITY_KEY_IDENTITY_VERSION, tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK}
    assert {k: v for k, v in ident.items() if k in v1["identity"]} == v1["identity"]
    assert manifest["harness_config_hash"] != v1["harness_config_hash"] and cell_dir.name != v1["cell_dir_name"]
    # the v2 manifest carries the experiment's UNHASHED record; `oos_experiment` is null on an in-support run
    assert manifest[tc.MANIFEST_KEY_AUTHORISED_EXPERIMENT] is None and manifest["support"] == tc.SUPPORT_IN_SUPPORT
    assert manifest[tc.MANIFEST_KEY_RESUME_ATTEMPTS][0]["attempt"] == 1
    assert manifest["route"][tc.LENS_SERVED_POSITIONS_KEY] is None       # the fixture standup records no map
    assert calls and "sampling_ref" in calls[0]["kw"]
    # the cap statement must EQUAL the resolved cap, else the run refuses by name
    with pytest.raises(SystemExit, match="must EQUAL the resolved cap"):
        _run(world, monkeypatch, arm=tc.THINKING_OFF,
             extra=[tc.FLAG_TINYCAP_IDENTITY_VERSION, "2", tc.FLAG_MAX_TOKENS_BY_TASK,
                    f"{tc.GSM8K_PAIRED_TASK_NAME}={tc.OOS_THINKING_MAX_TOKENS}"])
    with pytest.raises(SystemExit, match="identity-version-2 statement"):
        _run(world, monkeypatch, arm=tc.THINKING_OFF,
             extra=[tc.FLAG_MAX_TOKENS_BY_TASK, f"{tc.GSM8K_PAIRED_TASK_NAME}={isv_off_cap()}"])
    with pytest.raises(SystemExit, match="diagnostic"):
        _run(world, monkeypatch, arm=tc.THINKING_OFF, extra=[tc.FLAG_DIAGNOSTIC])

"""Phase B integration — the two doors of the out-of-support identity, by
EXECUTION (fit-support transfer).

The DRIVER (``scripts/served_capability_run.py``, lane B1) stamps an
out-of-support run's hashed identity from its own derivations — the plan
builder's cap rule and the contract literal — and copies the RAW attestation
map into the manifest's ``route`` block. The LAUNCHER
(``scripts/modal/tinycap_served.py``, lane B2) derives a PURE expectation of the
same identity with no server anywhere, and its resume guard reads the store the
driver leaves. The two lanes were written apart against ONE coupling
(``contracts``); this module runs both doors on one synthetic psr/ON world and
holds them to each other field for field, then to the same hash, then through
the launcher's own guard. Nothing here is a source-text pin.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

# the driver's synthetic world (its fixtures and helpers), and the launcher's
from test_served_capability import (  # noqa: F401  (`world` is a fixture)
    OBJECT_SHA,
    OOS_MAP,
    OOS_OWN_KEY,
    _cell_dir,
    oos_on_world,
    srun,
    world,
)
from test_tinycap_launcher import (
    _OOS_CELL,
    _OOS_MODEL,
    REPO,
    TEST_APP_NAME,
    _load_launcher,
    _oos_expected,
    _stub_modal,
)

from sisteer.capeval import store as cs
from sisteer.capeval.tinycap import contracts as C

OPTIONAL = tuple(f for f in C.tinycap_identity_fields(out_of_support_thinking=True)
                 if f not in C.TINYCAP_IDENTITY_FIELDS)


@pytest.fixture(scope="module")
def L():
    return _load_launcher(_stub_modal(), app_name=TEST_APP_NAME, name="tinycap_served_two_doors")


def _driver_store(world, monkeypatch):  # noqa: F811
    """The psr/ON store the driver leaves on a landed standup (nothing generated:
    ``run_inspect_tasks`` is stubbed; the identity, the manifest and the
    evidence are the driver's)."""
    from sisteer.capeval.tiers import inspect_served

    oos = oos_on_world(world, monkeypatch)

    async def fake_run(**kw):
        return {}

    monkeypatch.setattr(inspect_served, "run_inspect_tasks", fake_run)
    srun.main(oos.argv)
    cell_dir = _cell_dir(world.out_root)
    return oos, cell_dir, json.loads((cell_dir / "manifest.json").read_text())


def _aligned(exp, manifest):
    """The launcher's expectation with every NON-out-of-support fact read off the
    driver's own store (the fixture worlds differ in served geometry, the
    frameworks lock, the producer sha and the sampling row — none of which is
    under test here) — the three optional keys, the reduction mode and the route
    mirror stay the launcher's OWN derivation."""
    ident = manifest["resolved_config"]["tinycap"]
    return exp._replace(
        served={**ident["served"], C.REDUCTION_MODE_FIELD: exp.served[C.REDUCTION_MODE_FIELD]},
        limits=dict(ident["limits"]), samples=int(ident["samples"]),
        # the driver world's fixture endpoint declares no geometry pass (None); copied verbatim
        pass_no=ident[C.GEOMETRY_PASS_FIELD],
        client_lock_sha256=manifest[C.MANIFEST_KEY_FRAMEWORKS]["client_lock_sha256"],
        tree_sha=manifest[C.MANIFEST_KEY_PRODUCER]["tree_sha"],
        thinking_by_task=dict(ident[C.IDENTITY_KEY_THINKING_BY_TASK]),
        sampling={f: ident[f] for f in C.SAMPLING_IDENTITY_FIELDS},
        sampling_mode=manifest[C.MANIFEST_KEY_SAMPLING]["mode"],
        or_bench_item_set=ident[C.IDENTITY_KEY_OR_BENCH_ITEM_SET])


def test_psr_on_the_driver_stamps_exactly_what_the_launcher_expects(L, world, monkeypatch):  # noqa: F811
    """psr/ON, both doors on one store: the driver's stamped optional keys equal
    the launcher's pure ``optional_identity()`` field for field — the marker,
    ``max_tokens_by_task`` = {gsm8k_paired200: 8192}, ``serve_out_of_support`` =
    ["thinking"] — and ``served.reduction_mode`` agrees; substituting the
    launcher's values into the driver's hashed block leaves the hash unchanged
    (the SAME hashed identity); the launcher's resume guard then trusts the
    driver's store, reading the RAW map the driver copied into
    ``route.lens_serve_declarations`` under the cell's declared key — and names
    that key when the map is gone."""
    _oos, cell_dir, manifest = _driver_store(world, monkeypatch)
    ident = manifest["resolved_config"]["tinycap"]
    exp = _oos_expected(L, reduction_mode=C.REDUCTION_MODE_PER_ROW)   # the driver world serves per_row
    assert exp.cell == manifest["cell_id"] == _OOS_CELL and exp.declared
    # field for field
    opt = exp.optional_identity()
    assert tuple(opt) == OPTIONAL == (C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING,
                                      C.IDENTITY_KEY_MAX_TOKENS_BY_TASK, C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT)
    assert {k: ident[k] for k in OPTIONAL} == opt == {
        C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING: True,
        C.IDENTITY_KEY_MAX_TOKENS_BY_TASK: {C.GSM8K_STANDARD_TASK: 8192},
        C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT: ["thinking"]}
    assert ident["served"][C.REDUCTION_MODE_FIELD] == exp.served[C.REDUCTION_MODE_FIELD] == C.REDUCTION_MODE_PER_ROW
    # the driver's exact keys ARE the one derivation, and the launcher's optional
    # keys are exactly the ones beyond the base fields (no third spelling anywhere)
    assert set(ident) == set(C.tinycap_identity_fields(out_of_support_thinking=True))
    assert set(opt) == set(ident) - set(C.TINYCAP_IDENTITY_FIELDS)
    # the SAME hashed identity: the launcher's values, substituted, hash to the driver's key
    rc = manifest["resolved_config"]
    substituted = {**rc, "tinycap": {**ident, **opt,
                                      "served": {**ident["served"],
                                                 C.REDUCTION_MODE_FIELD: exp.served[C.REDUCTION_MODE_FIELD]}}}
    assert cs.harness_config_hash(substituted) == manifest["harness_config_hash"] == cs.harness_config_hash(rc)
    assert manifest["harness_config_hash"][:8] in cell_dir.name
    # the launcher's door reads the driver's store and trusts it
    aligned = _aligned(exp, manifest)
    assert aligned.optional_identity() == opt                       # untouched by the alignment
    assert L.untrusted_store_reason(cell_dir, expected=aligned) is None
    # the route mirror: the RAW map, whole, in the route block — read by the guard
    # under the cell's DECLARED key (the launcher's own spelling of it)
    assert manifest["route"][C.ROUTE_KEY_SERVE_DECLARATIONS] == OOS_MAP
    key = C.cell_spellings(aligned.cell, declared=aligned.declared).harness
    assert key == OOS_OWN_KEY and key in OOS_MAP
    doc = json.loads((cell_dir / "manifest.json").read_text())
    doc["route"].pop(C.ROUTE_KEY_SERVE_DECLARATIONS)
    (cell_dir / "manifest.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    why = L.untrusted_store_reason(cell_dir, expected=aligned)
    assert why and f"manifest.route.{C.ROUTE_KEY_SERVE_DECLARATIONS} is absent" in why
    doc["route"][C.ROUTE_KEY_SERVE_DECLARATIONS] = {key: {C.SERVE_OUT_OF_SUPPORT_KEY: ["thinking"]},
                                                    C.TWIN_CELL_ID: {C.SERVE_OUT_OF_SUPPORT_KEY: ["thinking"]}}
    (cell_dir / "manifest.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    assert L.untrusted_store_reason(cell_dir, expected=aligned) is None
    # an in-support expectation of the same cell never trusts the store, by name
    plain = _aligned(exp, manifest)._replace(out_of_support_thinking=False, max_tokens_by_task={},
                                             serve_out_of_support=None)
    why = L.untrusted_store_reason(cell_dir, expected=plain)
    assert why and C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING in why


def test_none_on_the_two_derivations_of_the_base_route_agree(L):
    """none/ON: the driver world has no base-route fixture, so the DRIVER's two
    derivations are called directly — the cap map from the plan builder's rule
    (``_override_max_tokens_by_task``) and the contract literal
    (``expected_serve_out_of_support``) — and held to the launcher's pure
    expectation for the base cell's ON leg: the cap on both ON legs, the literal
    ``None`` on the base route (contract §2.5.10 Q8 row 5), undeclared."""
    exp = _oos_expected(L, cell="none", reduction_mode=C.REDUCTION_MODE_PER_ROW)
    assert exp.out_of_support_thinking and not exp.declared
    driver_caps = srun._override_max_tokens_by_task([C.CONTROL_GSM8K], {C.GSM8K_STANDARD_TASK: C.THINKING_ON},
                                                    out_of_support=True, off_cap_override=False)
    driver_literal = C.expected_serve_out_of_support(lens=C.is_lens_cell("none"), arm=C.THINKING_ON,
                                                     out_of_support=True)
    assert exp.optional_identity() == {C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING: True,
                                       C.IDENTITY_KEY_MAX_TOKENS_BY_TASK: driver_caps,
                                       C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT: driver_literal}
    assert driver_caps == {C.GSM8K_STANDARD_TASK: 8192} and driver_literal is None
    assert not L.oos_lens_configuration(L.plan_for(cells=["none"], modes=["on"], controls=["gsm8k"],
                                                   model=_OOS_MODEL, allow_out_of_support_thinking=True)[0])


def test_the_driver_parses_every_flag_the_launcher_composes(L, capsys):
    """The argv coupling, by execution: (i) the launcher's argparse probe for an
    out-of-support launch — the override, the prompt-budget artifact and the
    artifact digest, the valued ones WITH their dummy values — reaches argparse's
    unrecognized-arguments line in the real driver with the sentinel as the ONLY
    unrecognized token; (ii) the psr/ON argv ``_driver_cmd`` composes (the
    digest and the artifact path on it) is accepted by the driver's parser whole
    — parsed to the sentinel, nothing else refused."""
    env = {**os.environ, "PYTHONPATH": f"{REPO}/src:{REPO}/scripts"}
    probe = L.driver_probe_argv(str(REPO), python=sys.executable,
                                extra_flags=(C.FLAG_ALLOW_OOS_THINKING, C.FLAG_OOS_PROMPT_BUDGET,
                                             C.FLAG_EXPECT_OBJECT_SHA256))
    r = subprocess.run(probe, capture_output=True, text=True, env=env, timeout=900, check=False)
    assert r.returncode == 2, r.stderr[-1500:]
    assert L.unrecognized_flags(r.stderr) == [L.DRIVER_PROBE_SENTINEL], r.stderr[-1500:]
    # (ii) the composed psr/ON argv, through the driver's own parser in-process
    (on,) = L.plan_for(cells=[_OOS_CELL], modes=["on"], controls=["gsm8k"], model=_OOS_MODEL,
                       allow_out_of_support_thinking=True)
    geom, _ = L.served_geometry(on, 1)
    cmd = L._driver_cmd(str(REPO), on, {"proxy_url": "http://127.0.0.1:1/v1", "routing_table": "/nonexistent/rt.json"},
                        "http://127.0.0.1:1/v1", geom, controls=on["controls"], limit=0, inspect_limit=0,
                        samples=1, cfg_path="/nonexistent/cfg.yaml", vllm_log=None,
                        oos_prompt_budget=f"/nonexistent/{C.OOS_PROMPT_BUDGET_FILENAME}",
                        expect_object_sha256=OBJECT_SHA)
    assert cmd[:2] == [L.CLIENT_PY, f"{REPO}/scripts/served_capability_run.py"]
    for flag in (C.FLAG_ALLOW_OOS_THINKING, C.FLAG_OOS_PROMPT_BUDGET, C.FLAG_EXPECT_OBJECT_SHA256):
        assert flag in cmd
    with pytest.raises(SystemExit) as e:
        srun.main(cmd[2:] + [L.DRIVER_PROBE_SENTINEL])
    assert e.value.code == 2
    assert L.unrecognized_flags(capsys.readouterr().err) == [L.DRIVER_PROBE_SENTINEL]


# --------------------------------------------------------------------------- #
# review round B: the SAME mutated endpoint through BOTH doors, and the caps   #
# --------------------------------------------------------------------------- #

from test_tinycap_launcher import _budget, _oos_endpoint

_DECL_KEY, _DECL_ERR_KEY, _TWIN = "lens_serve_declarations", "lens_serve_declarations_error", "none__lens0"
_DROP = object()


def _mutations(sp) -> dict:
    """Every payload of the two-doors matrix as ``{name: {key: value | _DROP}}``
    over the configuration's OWN attestation key ``sp.harness`` (the variant
    spelling on the declared standup, the bare one on the OFF reference)."""
    decl = {"serve_out_of_support": ["thinking"]}
    return {
        "baseline": {},
        "own_key_wrong": {_DECL_KEY: {sp.harness: {"serve_out_of_support": ["thinking", "x"]}, _TWIN: decl}},
        "twin_missing": {_DECL_KEY: {sp.harness: decl}},
        "error_non_null": {_DECL_ERR_KEY: "workers disagree"},
        "error_absent": {_DECL_ERR_KEY: _DROP},
        "map_null": {_DECL_KEY: None},
        "map_non_mapping": {_DECL_KEY: "not a map"},
        "map_absent": {_DECL_KEY: _DROP},
        "salt_policy_absent": {"lens_salt_policy": _DROP},
        "salt_policy_forced": {"lens_salt_policy": "request_unique_all"},
        "salt_policy_null": {"lens_salt_policy": None},
        "policy_map_absent": {"resumed_prefill_policy_cells": _DROP},
        "policy_always": {"resumed_prefill_policy_cells": {sp.harness: "always", _TWIN: "conditional"}},
        "policy_map_null": {"resumed_prefill_policy_cells": None},
        "policy_key_absent": {"resumed_prefill_policy_cells": {_TWIN: "conditional"}},
        "scalar_absent_by_cell_present": {"object_artifact_sha256": _DROP},
        "by_cell_absent": {"object_artifact_sha256_by_cell": _DROP},
        "digest_mismatch": {"object_artifact_sha256": "cd" * 32},
        "by_cell_wrong": {"object_artifact_sha256_by_cell": {sp.harness: "cd" * 32}},
        "template_digest_mismatch": {"chat_template_hash": "b" * 16},
        "multi_fault_salt_and_template": {"lens_salt_policy": "request_unique_all", "chat_template_hash": "b" * 16},
        "multi_fault_digest_and_salt": {"object_artifact_sha256": "cd" * 32, "lens_salt_policy": "request_unique_all"},
    }


def _apply(endpoint: dict, over: dict) -> dict:
    ep = dict(endpoint)
    for k, v in over.items():
        if v is _DROP:
            ep.pop(k, None)
        else:
            ep[k] = v
    return ep


def _driver_door(endpoint, *, cell, arm, oos, sha, budget):
    """The driver's verdict: the REC its SystemExit names, or None."""
    try:
        srun._check_serve_attestation(endpoint, cell=cell, arm=arm, out_of_support=oos,
                                      expected_object_sha256=sha, budget=budget, model_key=_OOS_MODEL)
    except SystemExit as e:
        msg = str(e)
        assert msg.startswith("[served-cap] "), msg
        return msg[len("[served-cap] "):].split(":", 1)[0]
    return None


def _launcher_door(L, endpoint, cfg, exp, *, sha, budget):
    """The launcher worker's verdict: the REC it would file, or None."""
    oos_on = L.serves_oos_on(cfg, cfg["controls"])
    out = L.oos_attestation_refusal(endpoint, cfg, expected=exp, object_sha256=sha or "",
                                    budget=budget if oos_on else None, oos_on=oos_on)
    return None if out is None else out[0]


#: the psr/ON (out-of-support, declared) column of the matrix — the verdict BOTH doors must reach
_EXPECTED_ON = {
    "baseline": None,
    "own_key_wrong": C.REC_OOS_DECLARATION_MISMATCH,
    "twin_missing": C.REC_OOS_DECLARATION_MISMATCH,
    "error_non_null": C.REC_OOS_DECLARATION_UNATTESTED,
    "error_absent": C.REC_OOS_DECLARATION_UNATTESTED,     # the key must be PRESENT and null under the override
    "map_null": C.REC_OOS_DECLARATION_UNATTESTED,
    "map_non_mapping": C.REC_OOS_DECLARATION_UNATTESTED,
    "map_absent": C.REC_OOS_DECLARATION_UNATTESTED,
    "salt_policy_absent": C.REC_OOS_SALT_POLICY_UNEXPECTED,
    "salt_policy_forced": C.REC_OOS_SALT_POLICY_UNEXPECTED,
    "salt_policy_null": C.REC_OOS_SALT_POLICY_UNEXPECTED,
    "policy_map_absent": C.REC_OOS_SALT_POLICY_UNEXPECTED,
    "policy_always": C.REC_OOS_SALT_POLICY_UNEXPECTED,
    "policy_map_null": C.REC_OOS_SALT_POLICY_UNEXPECTED,
    "policy_key_absent": C.REC_OOS_SALT_POLICY_UNEXPECTED,
    "scalar_absent_by_cell_present": C.REC_OOS_DECLARATION_MISMATCH,
    "by_cell_absent": C.REC_OOS_DECLARATION_MISMATCH,
    "digest_mismatch": C.REC_OOS_DECLARATION_MISMATCH,
    "by_cell_wrong": C.REC_OOS_DECLARATION_MISMATCH,
    "template_digest_mismatch": C.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH,
    "multi_fault_salt_and_template": C.REC_OOS_SALT_POLICY_UNEXPECTED,       # step 3 before step 5
    "multi_fault_digest_and_salt": C.REC_OOS_DECLARATION_MISMATCH,           # step 2 before step 3
}
#: the psr/OFF (in-support reference on the authorised model) column: the additive rule — absent
#: keys tolerated, present-null / off-contract refused, the map's own bare key a wiring error
_EXPECTED_OFF = {
    **{k: None for k in _EXPECTED_ON},
    "error_absent": None,                                 # in support: tolerated (recorded), per the additive rule
    "own_key_wrong": C.REC_OOS_DECLARATION_MISMATCH,
    "twin_missing": C.REC_OOS_DECLARATION_MISMATCH,          # the own BARE key present: a wiring error
    "error_non_null": C.REC_OOS_DECLARATION_UNATTESTED,
    "map_null": C.REC_OOS_DECLARATION_UNATTESTED,
    "map_non_mapping": C.REC_OOS_DECLARATION_UNATTESTED,
    "salt_policy_forced": C.REC_OOS_SALT_POLICY_UNEXPECTED,
    "salt_policy_null": C.REC_OOS_SALT_POLICY_UNEXPECTED,
    "policy_always": C.REC_OOS_SALT_POLICY_UNEXPECTED,
    "policy_map_null": C.REC_OOS_SALT_POLICY_UNEXPECTED,
    "policy_key_absent": C.REC_OOS_SALT_POLICY_UNEXPECTED,
    "digest_mismatch": C.REC_OOS_DECLARATION_MISMATCH,
    "by_cell_wrong": C.REC_OOS_DECLARATION_MISMATCH,
    "multi_fault_salt_and_template": C.REC_OOS_SALT_POLICY_UNEXPECTED,
    "multi_fault_digest_and_salt": C.REC_OOS_DECLARATION_MISMATCH,
}


@pytest.mark.parametrize("mode", ["on", "off"])
def test_the_same_mutated_endpoint_reaches_the_same_verdict_through_both_doors(L, mode):
    """Every payload of the matrix — the own key
    wrong, the twin missing, the error key non-null, the map null / non-mapping
    / absent, `lens_salt_policy` absent / forced / null, the per-cell policy
    map absent / `always` / null / without the key, the scalar digest absent
    beside a correct per-cell map, the per-cell map absent, a digest mismatch,
    a template digest mismatch, and two multi-fault records — is executed
    through the LAUNCHER's ``oos_attestation_refusal`` AND the DRIVER's
    ``_check_serve_attestation`` on the authorised psr cell under BOTH
    `out_of_support` values (the declared ON leg; the in-support OFF
    reference), and the two doors' verdicts (REC, refused?) are identical per
    payload; the ON and OFF columns are pinned to the additive rule."""
    (cfg,) = L.plan_for(cells=[_OOS_CELL], modes=[mode], controls=["gsm8k"], model=_OOS_MODEL,
                        allow_out_of_support_thinking=True)
    exp = _oos_expected(L, mode=mode, reduction_mode=C.REDUCTION_MODE_PER_ROW)
    oos = cfg.get(C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING) is True
    assert oos is (mode == "on") and exp.out_of_support_thinking is oos
    sp = L.spellings_for(cfg)
    base = _oos_endpoint(L, cfg)
    budget = _budget()
    sha = OBJECT_SHA
    base["object_artifact_sha256"] = sha
    base["object_artifact_sha256_by_cell"] = {sp.harness: sha}
    expected_column = _EXPECTED_ON if oos else _EXPECTED_OFF
    verdicts = {}
    for name, over in _mutations(sp).items():
        ep = _apply(base, over)
        driver = _driver_door(ep, cell=_OOS_CELL, arm=mode, oos=oos, sha=sha, budget=budget if oos else None)
        launcher = _launcher_door(L, ep, cfg, exp, sha=sha, budget=budget)
        assert driver == launcher, (name, mode, driver, launcher)
        verdicts[name] = driver
    assert verdicts == expected_column, {k: (verdicts[k], expected_column[k])
                                         for k in verdicts if verdicts[k] != expected_column[k]}


def test_the_driver_and_the_launcher_derive_the_same_cap_map_over_the_authorised_controls(L):
    """`max_tokens_by_task` has two derivations — the
    launcher's `oos_caps_by_task` (the tuple alone) and the driver's
    `_override_max_tokens_by_task` (the plan builder's `task_max_tokens`) — that
    agree ONLY because the authorised controls are GSM8K-only. Held equal here,
    by execution, over exactly `AUTHORISED_OOS_THINKING.controls` for both
    authorised cells' ON legs; a widening of the controls that names a task
    without a raised cap would part them and fail here."""
    controls = list(C.AUTHORISED_OOS_THINKING.controls)
    atoms = [a for name in controls for a in C.expand_control(name)]
    inspect_atoms = [a for a in atoms if C.control_family(a) in C.INSPECT_CONTROLS]
    by_task = C.run_thinking_by_task(atoms, C.THINKING_ON)
    driver_caps = srun._override_max_tokens_by_task(inspect_atoms, by_task, out_of_support=True,
                                                    off_cap_override=False)
    launcher_caps = L.oos_caps_by_task(L.thinking_by_task_of(C.THINKING_ON, controls),
                                       out_of_support=True, name="authorised")
    assert driver_caps == launcher_caps == {C.GSM8K_STANDARD_TASK: C.OOS_THINKING_MAX_TOKENS}
    for cell in C.AUTHORISED_OOS_THINKING.cells:
        exp = _oos_expected(L, cell=cell, reduction_mode=C.REDUCTION_MODE_PER_ROW)
        assert exp.max_tokens_by_task == driver_caps, cell
    # every task the authorised controls plan is one the tuple names a cap for — the
    # coincidence the two derivations rest on, stated
    for task in by_task:
        assert C.oos_max_tokens_on(task[len(C.INSPECT_TASK_PREFIX):]) is not None, task


# --------------------------------------------------------------------------- #
# offcap: the thinking-OFF comparator at the thinking-ON cap, through BOTH    #
# doors: the driver's stamped identity vs the launcher's                      #
# pure expectation, the two cap-map derivations, the flags the launcher       #
# composes, and the landed-record verdicts on the SAME mutated endpoint       #
# --------------------------------------------------------------------------- #

from test_served_capability import offcap_world
from test_tinycap_launcher import _offcap_expected

OFFCAP_OPTIONAL = tuple(f for f in C.tinycap_identity_fields(off_cap_override=True)
                        if f not in C.TINYCAP_IDENTITY_FIELDS)


@pytest.mark.parametrize("cell", [C.ROUTE_NONE_CELL, _OOS_CELL])
def test_off_cap_override_the_driver_stamps_exactly_what_the_launcher_expects(L, world, monkeypatch, cell):  # noqa: F811
    """offcap §8.3 (the psr/ON test's twin under the comparator flag): for each
    comparator cell the driver's stamped optional keys equal the launcher's pure
    `optional_identity()` field for field — `max_tokens_by_task` = {gsm8k_paired200:
    8192} and NOTHING else — substituting the launcher's values into the driver's
    hashed block leaves the hash unchanged, and the launcher's resume guard
    trusts the driver's store; an in-support expectation and an OOS expectation
    of the same cell refuse it naming the key."""
    from sisteer.capeval.tiers import inspect_served

    oc = offcap_world(world, monkeypatch, cell=cell)

    async def fake_run(**kw):
        return {}

    monkeypatch.setattr(inspect_served, "run_inspect_tasks", fake_run)
    srun.main(oc.argv)
    cell_dir = _cell_dir(oc.out_root)
    manifest = json.loads((cell_dir / "manifest.json").read_text())
    ident = manifest["resolved_config"]["tinycap"]
    exp = _offcap_expected(L, cell=cell, reduction_mode=C.REDUCTION_MODE_PER_ROW)
    assert exp.cell == manifest["cell_id"] == cell and exp.off_cap_override and not exp.declared
    opt = exp.optional_identity()
    assert tuple(opt) == OFFCAP_OPTIONAL == (C.IDENTITY_KEY_MAX_TOKENS_BY_TASK,)
    assert {k: ident[k] for k in OFFCAP_OPTIONAL} == opt == {C.IDENTITY_KEY_MAX_TOKENS_BY_TASK: {C.GSM8K_STANDARD_TASK: 8192}}
    assert set(ident) == set(C.tinycap_identity_fields(off_cap_override=True))
    assert set(opt) == set(ident) - set(C.TINYCAP_IDENTITY_FIELDS)
    rc = manifest["resolved_config"]
    substituted = {**rc, "tinycap": {**ident, **opt, "served": {**ident["served"],
                                                            C.REDUCTION_MODE_FIELD: exp.served[C.REDUCTION_MODE_FIELD]}}}
    assert cs.harness_config_hash(substituted) == manifest["harness_config_hash"] == cs.harness_config_hash(rc)
    aligned = _aligned(exp, manifest)
    assert aligned.optional_identity() == opt
    assert L.untrusted_store_reason(cell_dir, expected=aligned) is None
    plain = aligned._replace(off_cap_override=False, max_tokens_by_task={})
    why = L.untrusted_store_reason(cell_dir, expected=plain)
    assert why and C.IDENTITY_KEY_MAX_TOKENS_BY_TASK in why
    oos = aligned._replace(off_cap_override=False, out_of_support_thinking=True,
                           serve_out_of_support=(["thinking"] if C.is_lens_cell(cell) else None))
    why = L.untrusted_store_reason(cell_dir, expected=oos)
    assert why and C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING in why


def test_the_driver_and_the_launcher_derive_the_same_cap_map_under_both_overrides(L):
    """offcap §8.3 (extends the OOS cap-map test): the driver's ONE derivation
    (`_override_max_tokens_by_task`, the plan builder's `task_max_tokens`) and the
    launcher's pure `offcap_caps_by_task` agree over the comparator's controls
    for both cells, and the OOS pair still agrees."""
    controls = list(C.AUTHORISED_OFF_CAP_COMPARATOR.controls)
    atoms = [a for name in controls for a in C.expand_control(name)]
    inspect_atoms = [a for a in atoms if C.control_family(a) in C.INSPECT_CONTROLS]
    by_task = C.run_thinking_by_task(atoms, C.THINKING_OFF)
    driver_caps = srun._override_max_tokens_by_task(inspect_atoms, by_task, out_of_support=False, off_cap_override=True)
    launcher_caps = L.offcap_caps_by_task(L.thinking_by_task_of(C.THINKING_OFF, controls), name="comparator")
    assert driver_caps == launcher_caps == {C.GSM8K_STANDARD_TASK: C.OFF_CAP_OVERRIDE_MAX_TOKENS}
    for cell in C.AUTHORISED_OFF_CAP_COMPARATOR.cells:
        assert _offcap_expected(L, cell=cell, reduction_mode=C.REDUCTION_MODE_PER_ROW).max_tokens_by_task == driver_caps, cell
    # the in-support OFF derivation carries the table's cap, never the override's
    assert srun._override_max_tokens_by_task(inspect_atoms, by_task, out_of_support=False, off_cap_override=False) == {
        C.GSM8K_STANDARD_TASK: 512}
    for task in by_task:
        assert C.off_cap_override_max_tokens(task[len(C.INSPECT_TASK_PREFIX):]) is not None, task


def test_the_driver_parses_the_comparator_flag_the_launcher_composes(L, capsys):
    """offcap §8.3 (extends the flag-coupling test): the launcher's argparse probe
    for a comparator launch (the new switch + the artifact flag + the digest)
    reaches the real driver with the sentinel as the ONLY unrecognized token;
    the psr comparator argv `_driver_cmd` composes is accepted by the driver's
    parser whole."""
    env = {**os.environ, "PYTHONPATH": f"{REPO}/src:{REPO}/scripts"}
    probe = L.driver_probe_argv(str(REPO), python=sys.executable,
                                extra_flags=(C.FLAG_ALLOW_OFF_CAP_OVERRIDE, C.FLAG_OOS_PROMPT_BUDGET,
                                             C.FLAG_EXPECT_OBJECT_SHA256))
    r = subprocess.run(probe, capture_output=True, text=True, env=env, timeout=900, check=False)
    assert r.returncode == 2, r.stderr[-1500:]
    assert L.unrecognized_flags(r.stderr) == [L.DRIVER_PROBE_SENTINEL], r.stderr[-1500:]
    (psr,) = L.plan_for(cells=[_OOS_CELL], modes=["off"], controls=["gsm8k"], model=_OOS_MODEL,
                        allow_off_cap_override=True)
    geom, _ = L.served_geometry(psr, 1)
    cmd = L._driver_cmd(str(REPO), psr, {"proxy_url": "http://127.0.0.1:1/v1", "routing_table": "/nonexistent/rt.json"},
                        "http://127.0.0.1:1/v1", geom, controls=psr["controls"], limit=0, inspect_limit=0,
                        samples=1, cfg_path="/nonexistent/cfg.yaml", vllm_log=None,
                        oos_prompt_budget=f"/nonexistent/{C.OOS_PROMPT_BUDGET_FILENAME}",
                        expect_object_sha256=OBJECT_SHA)
    assert cmd[-5:] == [C.FLAG_ALLOW_OFF_CAP_OVERRIDE, C.FLAG_EXPECT_OBJECT_SHA256, OBJECT_SHA,
                        C.FLAG_OOS_PROMPT_BUDGET, f"/nonexistent/{C.OOS_PROMPT_BUDGET_FILENAME}"]
    assert C.FLAG_ALLOW_OOS_THINKING not in cmd
    with pytest.raises(SystemExit) as e:
        srun.main(cmd[2:] + [L.DRIVER_PROBE_SENTINEL])
    assert e.value.code == 2
    assert L.unrecognized_flags(capsys.readouterr().err) == [L.DRIVER_PROBE_SENTINEL]


#: the comparator's psr/OFF column: the LANDED rules — absent map / error key / salt policy /
#: policy map / digest REFUSE; a template mismatch REFUSES; a present declaration on the bare
#: key REFUSES; the campaign-tolerated absences of the OFF reference are refusals here
_EXPECTED_OFFCAP = {
    **_EXPECTED_ON,
    "own_key_wrong": C.REC_OOS_DECLARATION_MISMATCH,          # the own BARE key present at all: a wiring error
    "twin_missing": C.REC_OOS_DECLARATION_MISMATCH,
}


def test_the_same_mutated_endpoint_reaches_the_same_verdict_through_both_doors_offcap(L):
    """offcap §8.3: the third column beside `_EXPECTED_ON` / `_EXPECTED_OFF` —
    every payload of the matrix through the LAUNCHER's `oos_attestation_refusal`
    (a comparator configuration) AND the DRIVER's `_check_serve_attestation`
    (`out_of_support=held_to_record=True, off_cap_override=True`, the OFF arm),
    with the two doors' verdicts identical per payload and pinned to the LANDED
    rules: the OFF reference's tolerated absences all refuse here."""
    (cfg,) = L.plan_for(cells=[_OOS_CELL], modes=["off"], controls=["gsm8k"], model=_OOS_MODEL,
                        allow_off_cap_override=True)
    exp = _offcap_expected(L, reduction_mode=C.REDUCTION_MODE_PER_ROW)
    assert L.serves_offcap(cfg) and exp.off_cap_override and not exp.out_of_support_thinking
    sp = L.spellings_for(cfg)
    assert sp.harness == "si_safety__psr_kl_all"                          # the BARE key on the comparator's standup
    base = _oos_endpoint(L, cfg)
    budget = _budget(arm=C.THINKING_OFF, cap=C.OFF_CAP_OVERRIDE_MAX_TOKENS)
    sha = OBJECT_SHA
    base["object_artifact_sha256"] = sha
    base["object_artifact_sha256_by_cell"] = {sp.harness: sha}
    verdicts = {}
    for name, over in _mutations(sp).items():
        ep = _apply(base, over)
        try:
            srun._check_serve_attestation(ep, cell=_OOS_CELL, arm="off", out_of_support=True, off_cap_override=True,
                                          expected_object_sha256=sha, budget=budget, model_key=_OOS_MODEL)
            driver = None
        except SystemExit as e:
            driver = str(e)[len("[served-cap] "):].split(":", 1)[0]
        out = L.oos_attestation_refusal(ep, cfg, expected=exp, object_sha256=sha, budget=budget, oos_on=False,
                                        offcap=True)
        launcher = None if out is None else out[0]
        assert driver == launcher, (name, driver, launcher)
        verdicts[name] = driver
    assert verdicts == _EXPECTED_OFFCAP, {k: (verdicts[k], _EXPECTED_OFFCAP[k])
                                          for k in verdicts if verdicts[k] != _EXPECTED_OFFCAP[k]}
    # the launcher's consistency asserts: a budget on a configuration that needs none, and the reverse
    with pytest.raises(ValueError):
        L.oos_attestation_refusal(base, cfg, expected=exp, object_sha256=sha, budget=None, oos_on=False, offcap=True)

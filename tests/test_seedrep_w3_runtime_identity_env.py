"""A container resolves a child's registry identity WITHOUT the producer module.

The Modal images install the plan module at ``/root/seedrep_launch_plan.py`` (seedrep_served.py's `add_local_file`), where there is no
repo and no `scripts/sdfbase/seedrep/registry_plans.py`. The F-C commit made `child_registry_env` re-derive the identity path through
`identity_plan_rel` -> `registry_plans_module()` -> `Path(__file__).resolve().parents[2]`, which raises IndexError at that path -- on the
EXECUTION path (`resolve_spawn` -> `child_registry_env`) of every identity-bearing child, the S5 probe's smoke rows included. Now the
container consumes the child's GENERATED `registry_identity_rel` (validated as a relative .json path beneath the plans root) and the
producer is read only at plan generation on the tree. This test imports a FRESH copy of the module, sets its `__file__` to the image
path with an EMPTY producer cache, and resolves a real child's spawn env and an S5 probe smoke row's children: fails at the F-C tip
(IndexError), passes after; the producer cache stays empty throughout.
"""
from __future__ import annotations

import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts" / "modal") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "modal"))
import seedrep_launch_plan as lp  # noqa: E402  (the grading host copy: renders the plan, the producer at hand)

W = "2" * 40
WAVE = "22222222-3333-4444-8444-555555555555"
IMAGE_PATH = "/root/seedrep_launch_plan.py"


@pytest.fixture(scope="module")
def plan():
    inp = lp.load_inputs(REPO, peer_roster=REPO / lp.PROBE_ROSTER_REL, peer_pins=REPO / lp.PROBE_PINS_REL)
    return lp.build_plan(W=W, wave_uuid=WAVE, reqs_sha256=inp["reqs_sha256"], roster=inp["roster"], pins=inp["pins"], arditi_pins=inp["arditi_pins"],
                         arms=inp["arms"], materialisation="draft", generated_at="2026-09-18T00:00:00Z", peer_roster=inp["peer_roster"],
                         peer_pins=inp["peer_pins"], config_private_runs=inp["config_private_runs"])


def relocated_module(rel: str, image_path: str, name: str):
    """A module EXECUTED as the image has it: its source compiled and run with ``__file__`` ALREADY the flat image path (Modal installs /
    re-imports the app files flat at /root/<name>.py), registered in sys.modules under ``name`` before the body runs (dataclasses resolve
    their annotations through sys.modules[cls.__module__] at class creation). A module-level `Path(__file__).resolve().parents[2]` dies here
    with IndexError exactly as it does in the container."""
    src = (REPO / rel).read_text(encoding="utf-8")
    mod = types.ModuleType(name)
    mod.__file__ = image_path
    sys.modules[name] = mod
    exec(compile(src, image_path, "exec"), mod.__dict__)
    return mod


@pytest.fixture
def container_lp(monkeypatch):
    """The plan module AS THE CONTAINER HAS IT: executed with `__file__` = /root/seedrep_launch_plan.py (module-level code included), no
    producer cached, no `registry_plans` in sys.modules, the worktree bridge pointed at nothing, the stamp cache empty."""
    hidden = {n: sys.modules.pop(n) for n in ("registry_plans", "seedrep_registry_plans_s0", "seedrep_registry_plans_lp") if n in sys.modules}
    monkeypatch.setenv(lp.WS0_TREE_ENV, "/nonexistent/ws0-worktree")          # a container has no worktree bridge to fall to
    mod = relocated_module("scripts/modal/seedrep_launch_plan.py", IMAGE_PATH, "seedrep_launch_plan_in_container")
    mod._REGISTRY_PLANS = None
    mod._BANK_VIEW_STAMPS.clear()
    try:
        yield mod
    finally:
        sys.modules.update(hidden)
        sys.modules.pop("seedrep_launch_plan_in_container", None)


def _ctx(mod, **kw):
    base = dict(attempt_root="/workspace/attempt", attempt_id="att-1", standup_id="sid", plans_root="/vol/registry/plans", proxy_url="http://127.0.0.1:9000/v1",
                routing_table="/tmp/rt.json", driver_python="/venv/bin/python", client_python="/venv-client/bin/python", S=45, ncells=8, bank_view="/bank/v", base_env={"PATH": "/usr/bin"},
                pool_urls={"seedrep-attacker-0": "http://pool0:8000/v1", "seedrep-attacker-1": "http://pool1:8000/v1", "seedrep-attacker-2": "http://pool2:8000/v1",
                           "seedrep-attacker-3": "http://pool3:8000/v1"}, call_id="fc-1", attempt_root_rel="seedrep/attempts/x")
    base.update(kw)
    return mod.SpawnContext(**base)


def test_a_container_resolves_every_identity_bearing_child_without_the_producer(container_lp, plan):
    mod = container_lp
    assert mod.__file__ == IMAGE_PATH and mod._REGISTRY_PLANS is None
    with pytest.raises(FileNotFoundError, match="plan GENERATION"):      # the producer is unreachable there -- by name, never IndexError
        mod.registry_plans_module()
    n = 0
    # the app's per-child runtime substitutions (the GSM8K children's ledger URL / routing-table copy; seedrep_served passes child_subs)
    gsm8k_subs = {mod.LEDGER_URL_PLACEHOLDER: "http://127.0.0.1:9100/ledger", mod.CHILD_ROUTING_TABLE_PLACEHOLDER: "/tmp/rt-child.json",
                  mod.GSM8K_ATTEMPT_ID_PLACEHOLDER: "g8k-att-1"}
    for r in plan["standups"] + plan["rungs"]:
        subs = {mod.child_key(k): gsm8k_subs for k in r["children"]}
        ctx = _ctx(mod, standup_id=r["standup_id"], S=int(r["geometry"]["S"]), ncells=max(1, len(r["route_set"])), child_subs=subs)
        for k in r["children"]:
            if not k.get("registry_identity_rel") or not k.get("argv"):
                continue
            argv, env = mod.resolve_spawn(k, ctx)
            assert env[mod.RUN_IDENTITY_ENV] == "/vol/registry/plans/" + k["registry_identity_rel"][len(mod.PLANS_REL) + 1:]
            assert env[mod.SPOOL_ENV] == "/workspace/attempt/registry/records" and "{" not in " ".join(argv)
            n += 1
    assert n > 2000 and mod._REGISTRY_PLANS is None          # every driver child resolved; the producer was never loaded
    ctl = [k for r in plan["standups"] if r["leg"] == "controls" and r["engine"] == lp.ARDITI for k in r["children"] if k["arm"] == lp.ARDITI_STAGED_ID][0]
    env = mod.child_registry_env(plans_root="/p", registry_identity_rel=ctl["registry_identity_rel"], attempt_root="/a")
    assert env[mod.RUN_IDENTITY_ENV] == f"/p/_stores/controls/arditi_allpos__s{int(ctl['s'])}.json"


def test_the_s5_probe_smoke_rows_resolve_in_the_container(container_lp, plan):
    mod = container_lp
    identity = {"W": W, "W7": W[:7], "wave_uuid": "11111111-2222-4333-8444-555555555555", "smoke_id": "s5"}
    rows = [mod.smoke_row(r, identity, max_children=2, max_timeout_s=600) for r in plan["s5_probe_rows"]]
    assert len(rows) == 4
    for r in rows:
        ctx = _ctx(mod, standup_id=r["standup_id"], S=int(r["geometry"]["S"]), ncells=2)
        for k in r["children"]:
            argv, env = mod.resolve_spawn(k, ctx)
            assert env[mod.RUN_IDENTITY_ENV].startswith("/vol/registry/plans/") and "{" not in " ".join(argv)
    assert mod._REGISTRY_PLANS is None


@pytest.mark.parametrize("bad", ["", None, "/vol/registry/plans/x.json", "registry/plans/../x.json", "registry/plans/./x.json", "registry/plans//x.json",
                                 "registry/plans/x.txt", "other/plans/x.json", "registry/plans/x.json/", "registry/plansx/x.json"])
def test_a_registry_identity_rel_outside_the_plans_root_is_refused_by_name(container_lp, bad):
    with pytest.raises(ValueError, match="registry_identity_rel"):
        container_lp.child_registry_env(plans_root="/p", registry_identity_rel=bad, attempt_root="/a")
    assert container_lp.identity_rel_under_plans("registry/plans/_stores/controls/arditi_allpos__s2.json") == "_stores/controls/arditi_allpos__s2.json"
    assert container_lp.identity_rel_under_plans("registry/plans/pasttense/none__s1.json") == "pasttense/none__s1.json"


def test_plan_generation_still_spells_the_producers_paths(plan):
    """The grading host side is unchanged: the generated rels are the producer's (the F-C contract), and the producer loads from the repo layout."""
    RP = lp.registry_plans_module()
    rels = {k["registry_identity_rel"] for r in plan["standups"] + plan["rungs"] for k in r["children"]}
    assert f"{lp.PLANS_REL}/{RP.STORES_DIR}/controls/{RP.ARDITI_ARM}__s2.json" in rels and f"{lp.PLANS_REL}/pasttense/{RP.ARDITI_ARM}__s1.json" in rels
    assert not any("/ar_c0p3__arditi_allpos__s" in r or r.startswith(f"{lp.PLANS_REL}/controls/") for r in rels)


def test_the_preflight_validates_the_real_plan_in_the_container_and_stamps_every_row(container_lp, plan):
    """The preflight in the real image: `launch_plan_validates_against_table` calls `lp.validate_plan(plan, table=sg)` in-container, which
    reaches `gsm8k_bank_view` -> `_ws0_bank_view_stamp`; at 4703dad1b that walked `Path(__file__).resolve().parents[2]` -> IndexError at
    /root/seedrep_launch_plan.py. The rule is `contracts.expected_bank_view_stamp` from sys.path (the staged tree's src), never the
    grading host's worktree bridge (pointed at nothing here) and never `parents[2]`."""
    mod = container_lp
    from sisteer.capeval.serving import geometry as sg          # the preflight's exact table
    assert mod.validate_plan(plan, table=sg) == []                # the real draft plan, the 4 probe rows included
    rows = plan["standups"] + plan["rungs"] + plan["s5_probe_rows"]
    assert len(plan["s5_probe_rows"]) == 4
    for r in rows:
        view = str(r.get("view") or r["engine"])
        assert mod._ws0_bank_view_stamp(view, None) == view                                   # a view's own children stamp the view
        assert mod._ws0_bank_view_stamp(view, mod.ARDITI_BANK_VIEW) == mod.ARDITI_BANK_VIEW     # an Arditi child stamps the SOURCE bank
        for s_ in r.get("seed_bases") or []:
            assert mod.gsm8k_bank_view(r["engine"], s_) == (mod.ARDITI_BANK_VIEW if r["engine"] == mod.ARDITI else f"s{int(s_)}")
    assert mod._REGISTRY_PLANS is None
    # with that function unavailable on sys.path, a container refuses by name -- it never reaches for a worktree
    import builtins
    real_import = builtins.__import__

    def no_contracts(name, *a, **k):
        if name == "sisteer.capeval.tinycap.contracts" or (name == "sisteer.capeval.tinycap" and a and a[2] and "contracts" in a[2]):
            raise ImportError("no contracts")
        return real_import(name, *a, **k)
    mod._BANK_VIEW_STAMPS.clear()
    builtins.__import__ = no_contracts
    try:
        with pytest.raises(RuntimeError, match="staged tree's src must be on sys.path"):
            mod._ws0_bank_view_stamp("s9", None)
    finally:
        builtins.__import__ = real_import


def test_the_image_installed_module_executes_flat_and_the_sibling_apps_guard_their_repo_walks():
    """Every `parents[...]` walk in scripts/modal/seedrep_*.py is either the guarded repo detection (the plan module), inside `try/except
    IndexError` or a `modal.is_local()` branch (the app modules Modal re-imports flat), or grading-host-only argparse/CLI code (the steward)."""
    mod = relocated_module("scripts/modal/seedrep_launch_plan.py", IMAGE_PATH, "seedrep_launch_plan_flat_import")
    try:
        assert mod.__file__ == IMAGE_PATH and callable(mod.validate_plan)
    finally:
        sys.modules.pop("seedrep_launch_plan_flat_import", None)
    import ast

    def parents_walks(rel):
        """The line numbers of every REAL `<expr>.parents[...]` subscript (the AST, so docstrings and comments do not count)."""
        src = (REPO / "scripts/modal" / rel).read_text()
        lines = src.splitlines()
        hits = sorted({n.lineno for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Attribute) and n.value.attr == "parents"})
        return lines, hits
    lines, hits = parents_walks("seedrep_launch_plan.py")
    assert len(hits) == 2, hits                                                         # registry_plans_module + _ws0_bank_view_stamp
    for ln in hits:
        assert "if len(here.parents) > 2 and here.parents[1].name == \"scripts\" and here.parent.name == \"modal\" else None" in lines[ln - 1], (ln, lines[ln - 1])
    guarded = {"seedrep_served.py": "try", "seedrep_petri_drive.py": "try", "seedrep_attacker_pool.py": "if modal.is_local():"}
    for name, guard in guarded.items():
        lines, hits = parents_walks(name)
        assert hits, name
        for ln in hits:
            window = "\n".join(lines[max(0, ln - 4): ln - 1])
            assert guard in window, (name, ln, lines[ln - 1])
            if guard == "try":
                assert any("except IndexError" in l for l in lines[ln: ln + 2]), (name, ln)
    lines, hits = parents_walks("seedrep_steward.py")
    main_at = next(i for i, l in enumerate(lines, 1) if l.startswith("def main("))
    assert hits and all(ln > main_at for ln in hits), (hits, main_at)                  # the grading host CLI code, never module level

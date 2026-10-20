"""scripts/modal/strength_served.py — the Modal port of the strength frontier's sampled legs: the SERVING PATH a leg is
composed with. The ladders that produced the instruct-direct record ran on the LENS engine; the base->instruct
sweep runs on the NATIVE LoRA engine. The port pins whichever the caller asks for, through the tree's own hooks, and records
the pin in the leg's evidence — never leaving a native-default tree to pick an engine the record did not ask for."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    """The port's PURE decision functions, executed from its source without importing it: the module builds a Modal image
    at import (unavailable in the suite), and `scripts/` on sys.path shadows the `modal` package with the scripts/modal
    directory. The two functions read text and rebind module constants only, so they run unchanged in a namespace that
    carries the constants they name."""
    import ast, types
    src = (REPO / "scripts/modal/strength_served.py").read_text()
    tree = ast.parse(src)
    keep_names = {"VOL_PATH", "STRENGTH", "WAVE_VOL", "LOGS_VOL", "FIGS_VOL", "TOOLS_VOL", "BANK_VOL", "STATE_DIR", "CALLS_FILE",
                  "LEGS_LOG", "WAVE_ROOT_DEFAULT", "WAVE_ROOT", "SERVING_PATHS"}
    keep_funcs = {"set_wave_root", "serving_path_pin", "local_wave_dir_name", "parse_leg_env", "require_cluster_label"}
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(tg, ast.Name) and tg.id in keep_names for tg in node.targets):
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in keep_funcs:
            body.append(node)
    ns = {"__name__": "strength_served_pure", "Path": Path, "re": __import__("re")}
    exec(compile(ast.Module(body=body, type_ignores=[]), "strength_served.py(pure)", "exec"), ns)
    mod = types.SimpleNamespace(**{k: v for k, v in ns.items() if not k.startswith("__")})
    # rebinding via `global` lands in `ns`; read the live values through the namespace dict
    class Live:
        def __getattr__(self, k):
            return ns[k]
    live = Live()
    for fn in keep_funcs:
        setattr(live, fn, ns[fn])
    return live


def _tree(tmp_path: Path, *, knows_flag: bool, extra_hook: bool, binds_engine: bool) -> Path:
    (tmp_path / "scripts/sdfbase/full22").mkdir(parents=True)
    (tmp_path / "scripts/sdfbase/mt_serving_up.py").write_text('ap.add_argument("--serving-path", choices=...)\n' if knows_flag else "ap.add_argument('--port')\n")
    common = ""
    if extra_hook:
        common += '  ${F22_STANDUP_EXTRA[@]+"${F22_STANDUP_EXTRA[@]}"} \\\n'
    if binds_engine:
        common += 'f22_bind_engine(){ if [ "$F22_SERVING_PATH" = lens ]; then TARGET_ENGINE=lens; else TARGET_ENGINE=lora_native; fi; }\n'
    (tmp_path / "scripts/sdfbase/full22/common.sh").write_text(common or "# nothing\n")
    return tmp_path


def test_a_native_default_tree_is_pinned_to_the_engine_the_caller_names_and_the_pin_is_legible(tmp_path):
    m = _load()
    main_like = _tree(tmp_path / "main", knows_flag=True, extra_hook=True, binds_engine=True)
    lens = m.serving_path_pin(main_like, "lens")
    assert lens["serving_path"] == "lens" and lens["env"] == {"F22_SERVING_PATH": "lens"}   # explicit: it wins over f22_init's auto selection
    # a tree that knows the flag and reads the argv hook but binds no F22_SERVING_PATH is REFUSED: the hook is
    # not a pin channel — appended after --serving-path the standup's parser takes the later value and the selection
    # record is refused, and common.sh refuses a --serving-path element there by name at f22_init — so nothing is emitted
    hook_only = _tree(tmp_path / "hook", knows_flag=True, extra_hook=True, binds_engine=False)
    hook_pin = m.serving_path_pin(hook_only, "lens")
    assert hook_pin["serving_path"] is None and hook_pin["env"] == {} and hook_pin["why"].startswith("REFUSE")
    assert "does not bind F22_SERVING_PATH" in hook_pin["why"]
    native = m.serving_path_pin(main_like, "lora_native")
    assert native["serving_path"] == "lora_native" and native["env"] == {"F22_SERVING_PATH": "lora_native"}
    assert "native" in native["why"].lower() and not native["why"].startswith("REFUSE")
    # the default is the lens: the instruct-direct ladders' engine, so a re-run never silently changes instrument
    assert m.serving_path_pin(main_like) == lens


def test_a_pre_native_tree_gets_no_pin_for_the_lens_and_refuses_the_native_engine(tmp_path):
    m = _load()
    old = _tree(tmp_path / "old", knows_flag=False, extra_hook=False, binds_engine=False)
    lens = m.serving_path_pin(old, "lens")
    assert lens["serving_path"] == "lens" and lens["env"] == {} and "predates" in lens["why"]
    native = m.serving_path_pin(old, "lora_native")
    assert native["serving_path"] is None and native["why"].startswith("REFUSE")


def test_a_tree_that_knows_the_flag_but_lacks_the_hook_refuses_rather_than_guessing(tmp_path):
    m = _load()
    half = _tree(tmp_path / "half", knows_flag=True, extra_hook=False, binds_engine=False)
    for want in ("lens", "lora_native"):
        pin = m.serving_path_pin(half, want)
        assert pin["serving_path"] is None and pin["why"].startswith("REFUSE"), want
    assert m.serving_path_pin(half, "cuda_graphs_please")["why"].startswith("REFUSE")
    assert m.SERVING_PATHS == ("lens", "lora_native")


def test_the_wave_root_selects_one_volume_universe_per_instrument_and_the_default_keeps_every_historical_path():
    m = _load()
    m.set_wave_root(m.WAVE_ROOT_DEFAULT)
    assert m.WAVE_VOL.endswith("/exp/sdfbase_full22_modal") and m.LOGS_VOL.endswith("/exp/logs_modal") and m.FIGS_VOL.endswith("/exp/figs_modal")
    assert m.TOOLS_VOL == f"{m.WAVE_VOL}/sampled/tools" and m.LEGS_LOG.name == "legs.log"
    m.set_wave_root("sdfbase_full22_native")            # The native sweep's own universe
    assert m.WAVE_VOL.endswith("/exp/sdfbase_full22_native") and m.LOGS_VOL.endswith("/exp/logs_sdfbase_full22_native")
    assert m.FIGS_VOL.endswith("/exp/figs_sdfbase_full22_native") and m.TOOLS_VOL == f"{m.WAVE_VOL}/sampled/tools"
    assert m.LEGS_LOG.name == "legs_sdfbase_full22_native.log"
    with pytest.raises(SystemExit):
        m.set_wave_root("../sdfbase_full22")
    m.set_wave_root(m.WAVE_ROOT_DEFAULT)


def test_the_container_wave_dir_follows_the_trees_serving_path_so_native_stores_are_synced():
    """main's common.sh writes native stores under exp/sdfbase_full22_native; a syncer watching exp/sdfbase_full22 loses
    every native store with the container (it happened: 36 legs, logs synced, no stores)."""
    m = _load()
    assert m.local_wave_dir_name("lens") == "sdfbase_full22"
    assert m.local_wave_dir_name("lora_native") == "sdfbase_full22_native"


def test_parse_leg_env_splits_on_semicolons_and_the_first_equals():
    """--leg-env carries the tree's argv hooks per launch; a value may hold '=' and ','."""
    res = _load()
    mod = res[0] if isinstance(res, tuple) else res
    parse_leg_env = getattr(mod, "parse_leg_env")
    assert parse_leg_env("") == {} and parse_leg_env(None) == {}
    assert parse_leg_env("F22_STANDUP_EXTRA=--max-num-seqs=48") == {"F22_STANDUP_EXTRA": "--max-num-seqs=48"}
    assert parse_leg_env(" A=1 ; B=x,y=z ") == {"A": "1", "B": "x,y=z"}
    import pytest
    with pytest.raises(SystemExit):
        parse_leg_env("NOEQUALS")
    with pytest.raises(SystemExit):
        parse_leg_env("bad-name=1")


def test_a_launch_without_a_cluster_label_refuses_before_any_gpu():
    """The wave shell needs SISTEER_CLUSTER_LABEL (the run's substrate, and the key of each arm's servability), and the app
    has no site config: --go / --legs refuse by name unless --leg-env carries it."""
    mod = _load()
    with pytest.raises(SystemExit, match="SISTEER_CLUSTER_LABEL"):
        mod.require_cluster_label("")
    with pytest.raises(SystemExit, match="SISTEER_CLUSTER_LABEL"):
        mod.require_cluster_label("F22_STANDUP_EXTRA=--max-num-seqs=48")
    assert mod.require_cluster_label("SISTEER_CLUSTER_LABEL=cluster_a;F22_STANDUP_EXTRA=x") is None

"""The Modal producer's serving-path pin (``scripts/modal/strength_served.py`` ``serving_path_pin``) travels as
``F22_SERVING_PATH`` — the explicit-pin channel ``common.sh`` honours — and is proven THROUGH THE LAUNCH from the
producer's ACTUAL returned environment: ``f22_init``'s inherited-hook check, the selection it records
(``f22_select_serving_path``), ``f22_standup``'s composed argv (its own slice), the real ``mt_serving_up`` parser and the
real ``load_serving_path_selection``. KNOWN-BAD first: the first port's channel,
``F22_STANDUP_EXTRA=--serving-path=lens``, is refused BY NAME before the selection is recorded — and could never have
launched anyway: appended after ``--serving-path`` / ``--serving-path-selection`` the parser took the later value and the
selection reader refused the record."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PRODUCER = REPO / "scripts" / "modal" / "strength_served.py"

sys.path.insert(0, str(Path(__file__).parent))
from test_mt_serving_up_serving_path import _load_mt  # noqa: E402
from test_sdfbase_full22_wave import F22, _fn_from  # noqa: E402

#: the common.sh functions the launch harness runs, in f22_init's order (the check, then the selection) + the helper
_HELPERS = ("f22_standup_extra_inherited_check", "f22_standup_extra_reset", "f22_select_serving_path")


def _producer_fn(name: str):
    """The producer's top-level function ``name``, compiled from its OWN source: the module imports ``modal`` and opens
    the volume at import time, so the function under test is lifted by AST — the actual code, no stub of Modal."""
    tree = ast.parse(PRODUCER.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    consts = [n for n in tree.body if isinstance(n, ast.Assign)          # + the module constant the function reads
              and any(isinstance(t, ast.Name) and t.id == "SERVING_PATHS" for t in n.targets)]
    ns = {"Path": Path}
    exec(compile(ast.Module(body=consts + [node], type_ignores=[]), str(PRODUCER), "exec"), ns)
    return ns[name]


def _launch_env(pin_env):
    """The environment a leg's sbatch body inherits from run_leg: the box's, no F22_* of ours, plus the producer's pin."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("F22_")}
    env.update(pin_env)
    return env


def _select_and_compose(tmp_path, pin_env, *, lane=""):
    """Under the producer's env: f22_init's check + selection (RUN, the record it writes), then the lane's own line(s),
    then f22_standup's argv composition (the slice from the serving-path locals through the mt_serving_up.py invocation)
    with $SPY a stub printing its argv. Returns ``(rc, argv | None, log, out_dir)``; argv None = refused before launch."""
    sh = (F22 / "common.sh").read_text()
    standup = sh[sh.index("f22_standup(){"):sh.index("f22_standup_stop(){")]
    start = standup.index('  local serving_path="${F22_SERVING_PATH:-lora_native}"')
    marker = "    ${F22_THINKING_ON:+--thinking-on} \\\n"
    end = standup.index(marker) + len(marker)
    body = standup[start:end] + '    > "$out/argv.txt" 2>&1\n'
    out = tmp_path / f"leg-{len(list(tmp_path.glob('leg-*')))}"
    spy = tmp_path / "spy.sh"
    spy.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    spy.chmod(0o755)
    script = ('set -u\nf22_log(){ echo "LOG $*"; }\nf22_fatal(){ echo "FATAL: $*"; exit 1; }\nnvidia-smi(){ :; }\n'
              + "\n".join(_fn_from(F22 / "common.sh", h) for h in _HELPERS) + "\n"
              + f"out={out}\nSPY={spy}\nREPO_DIR=/r\nCFG=c.yaml\nMANIFEST=m.yaml\nTT_TWO_SI_DEFENSES_DIR=/d\nKEY=gemma4e4b\n"
              + "CUDA_VISIBLE_DEVICES=0\nF22_LOAD_CHECK=0\nmode=attacks\npport=''\ngeval=''\ngL=''\ngS=''\ngLP=''\n"
              + "SLURM_JOB_NAME=strength-served-test\n"
              + "f22_standup_extra_inherited_check\nf22_select_serving_path\n"
              + 'echo "SELECTED=$F22_SERVING_PATH MODE=$F22_SERVING_PATH_MODE"\n'
              + lane + "\ncompose(){\n" + body + "}\ncompose\n")
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=_launch_env(pin_env))
    argv = None
    if (out / "argv.txt").is_file():
        lines = (out / "argv.txt").read_text().split("\n")[:-1]
        assert lines[0] == "scripts/sdfbase/mt_serving_up.py", lines[:2]
        argv = lines[1:]
    return r.returncode, argv, r.stdout + r.stderr, out


def _sans_paths(argv, root):
    return [t for t in argv if not t.startswith(str(root))]


def test_the_producer_pins_the_lens_through_f22_serving_path_on_this_tree():
    pin_fn = _producer_fn("serving_path_pin")
    pin = pin_fn(REPO)
    assert pin["serving_path"] == "lens" and pin["env"] == {"F22_SERVING_PATH": "lens"}
    assert "F22_STANDUP_EXTRA" not in pin["env"] and "F22_SERVING_PATH" in pin["why"]
    assert pin_fn(REPO, "lens") == pin                                    # the default IS the lens (the ladders' engine)
    # the same channel carries the native engine when the caller names it (the base->instruct sweep); an unknown
    # engine is refused
    native = pin_fn(REPO, "lora_native")
    assert native["serving_path"] == "lora_native" and native["env"] == {"F22_SERVING_PATH": "lora_native"}
    assert pin_fn(REPO, "cuda_graphs_please")["why"].startswith("REFUSE")
    # ONE channel in the producer's source: no F22_STANDUP_EXTRA value, no --serving-path spelling; run_leg installs
    # pin["env"] into the leg's environment before the sbatch body and gates on a REFUSE verdict
    src = PRODUCER.read_text(encoding="utf-8")
    assert 'F22_STANDUP_EXTRA":' not in src and '"--serving-path=lens"' not in src
    assert 'env.update(pin["env"])' in src and 'pin["why"].startswith("REFUSE")' in src
    assert src.index("pin = serving_path_pin(repo_dir, serving_path)") < src.index('env.update(pin["env"])') \
        < src.index("SBATCH_REL), key, phase")


def test_the_producers_actual_env_selects_records_and_composes_the_lens_through_the_real_reader(tmp_path):
    mt = _load_mt()
    pin = _producer_fn("serving_path_pin")(REPO)
    rc, argv, log, out = _select_and_compose(tmp_path, pin["env"])
    assert rc == 0 and argv is not None, log
    assert "SELECTED=lens MODE=explicit" in log and "[serving-path] EXPLICIT lens (F22_SERVING_PATH set by caller)" in log
    assert argv.count("--serving-path") == 1 and argv[argv.index("--serving-path") + 1] == "lens"
    assert "--reduction-mode" in argv and "--require-attested-mode" in argv           # the lens pins ride under lens
    rec_path = out / "serving_path_selection.json"
    assert json.loads(rec_path.read_text())["selected"] == "lens"
    ns = mt._build_parser().parse_args(argv)                                          # the real parser
    assert ns.serving_path == "lens" and ns.serving_path_selection == str(rec_path)
    rec = mt.load_serving_path_selection(ns.serving_path_selection, serving_path=ns.serving_path)   # the real reader
    assert rec == {"cells_by_disposition": None, "cells_source": None, "mode": "explicit",
                   "reason": "F22_SERVING_PATH set by caller", "rule": "kind-aware-2026-09-15",
                   "selected": "lens", "skipped": [], "spec_source": None}
    # the leg IS target_sampled.sbatch: its own standup line rides after the pins and the argv still parses
    rc, argv2, log, _ = _select_and_compose(tmp_path, pin["env"], lane="f22_standup_extra_reset --enable-log-requests")
    assert rc == 0 and argv2 is not None, log
    assert argv2[-1] == "--enable-log-requests" and _sans_paths(argv2[:-1], tmp_path) == _sans_paths(argv, tmp_path)
    assert mt._build_parser().parse_args(argv2).enable_log_requests is True


def test_the_first_ports_channel_is_refused_by_name_before_the_selection_and_could_never_have_launched(tmp_path):
    # KNOWN-BAD: what serving_path_pin returned before this fix — the argv hook — dies at f22_init's check, by name,
    # pointing at the channel, before any selection is recorded or any standup composed
    rc, argv, log, _ = _select_and_compose(tmp_path, {"F22_STANDUP_EXTRA": "--serving-path=lens"})
    assert rc == 1 and argv is None and "SELECTED=" not in log, log
    assert "FATAL: F22_STANDUP_EXTRA carries '--serving-path=lens'" in log and "travels as F22_SERVING_PATH" in log
    # ... and why that channel could not have launched a lens leg anyway: the native default selected and recorded,
    # the hook appended after the pins, the parser takes the later value, the record is refused
    mt = _load_mt()
    rc, native, log, _ = _select_and_compose(tmp_path, {"F22_SERVING_PATH": "lora_native"})
    assert rc == 0 and native[native.index("--serving-path") + 1] == "lora_native", log
    ns = mt._build_parser().parse_args(native + ["--serving-path=lens"])
    assert ns.serving_path == "lens"
    with pytest.raises(SystemExit, match="selected 'lora_native' but this standup serves --serving-path lens"):
        mt.load_serving_path_selection(ns.serving_path_selection, serving_path=ns.serving_path)


def test_the_producer_refuses_a_tree_without_the_explicit_pin_channel_and_skips_a_pre_flag_tree(tmp_path):
    pin_fn = _producer_fn("serving_path_pin")

    def tree(name, *, mt_src, common_src):
        t = tmp_path / name
        (t / "scripts" / "sdfbase" / "full22").mkdir(parents=True)
        (t / "scripts" / "sdfbase" / "mt_serving_up.py").write_text(mt_src)
        (t / "scripts" / "sdfbase" / "full22" / "common.sh").write_text(common_src)
        return t
    # knows the flag AND common.sh binds F22_SERVING_PATH (f22_bind_engine): the pin, through the env, nothing else —
    # for the lens by default and for the native engine when named
    both = tree("both", mt_src='ap.add_argument("--serving-path", choices=["lora_native", "lens"])',
                common_src='f22_bind_engine(){ if [ "$F22_SERVING_PATH" = lens ]; then TARGET_ENGINE=lens; else '
                           'TARGET_ENGINE=lora_native; fi; }\n')
    assert pin_fn(both) == {"serving_path": "lens", "env": {"F22_SERVING_PATH": "lens"},
                            "why": "the tree binds an explicit F22_SERVING_PATH: pinned to the ladder's engine, the lens, "
                                   "for the whole view (selected and recorded in f22_init)"}
    assert pin_fn(both, "lora_native")["env"] == {"F22_SERVING_PATH": "lora_native"}
    # knows the flag but common.sh binds no F22_SERVING_PATH — with or without the standup argv hook: REFUSED loudly,
    # never the old hook, never both
    for name, common_src in (("hook", 'f22_standup(){ x ${F22_STANDUP_EXTRA[@]+"${F22_STANDUP_EXTRA[@]}"}; }\n'),
                             ("nohook", "f22_standup(){ :; }\n")):
        pin = pin_fn(tree(name, mt_src='ap.add_argument("--serving-path", choices=["lora_native", "lens"])',
                          common_src=common_src))
        assert pin["serving_path"] is None and pin["env"] == {} and pin["why"].startswith("REFUSE"), name
        assert "does not bind F22_SERVING_PATH" in pin["why"], name
    # predates the flag: lens-only by construction, nothing to pin; a missing tree reads the same
    old = tree("old", mt_src='ap.add_argument("--reduction-mode")', common_src="f22_standup(){ :; }\n")
    assert pin_fn(old) == {"serving_path": "lens", "env": {},
                           "why": "the tree predates --serving-path: lens-only by construction, nothing to pin"}
    assert pin_fn(tmp_path / "missing")["env"] == {}

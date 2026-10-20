"""The tinycap four-way launcher (``scripts/modal/tinycap_served.py``) —
CPU-only, no Modal, no GPU: the module loads against a modal stub and
every argv it builds is read back against the contracts.

Covered: the run matrix (four contract-named containers, the task
lists, ``--tool-calls`` only when BFCL is among the controls, ``--reasoning-
parser gemma4`` under thinking-ON only); the standup argv carries the PASS-1
window from ``geometry.pass1_geometry`` (the composer's footprint, never a
literal) with S from the served MODEL's measured budget (E7 / E15:
``geometry.BUDGET_BY_MODEL`` via ``model_key``, ``--model-key`` on the standup,
no ``--allow-over-budget``; an unmeasured model refuses); the driver argv IS
``contracts.driver_argv`` (ledger URL, thinking,
concurrency = the SERVED S, judge pinned); the ledger process argv; resume
via VALIDATED done markers in ONE trusted store (R2: never a union of bare
file names across ``cells/*``; the manifest's cell / mode / empty limits /
hash prefixes, the marker's fields, task, full count and summary agreement);
the image's client-venv recipe (``--no-deps --require-hashes``, the VCS line
split out at its commit, pinned uv) and EVERY pip install pinned (R8, an AST
sweep); the GPU-container trip-wire; the smoke plan; the dry-run print with
the exact ``modal run`` argv and the default-control exclusions
(``inspect_cmp:xstest`` = unavailable_gated); pass 2 refuses a
multi-sub-entry configuration (R5) and a provisional entry; the stage step's
pins.yaml keys exist; the ``data_dir`` datasets-cache alias the stage lays
down for the offline containers (openai/gsm8k); and the controls_summary
merge step in the client venv (N1).
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import types
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import tinycap_golden as golden
import yaml

from sisteer.capeval import store as cs
from sisteer.capeval.serving import geometry as sg
from sisteer.capeval.serving import vllm_server as vs
from sisteer.capeval.tinycap import contracts as C

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts/modal/tinycap_served.py"


def _stub_modal() -> types.ModuleType:
    m = types.ModuleType("modal")

    class _Chain:
        def __getattr__(self, name):
            return lambda *a, **k: self

    class _App:
        def __init__(self, name=None, **k):
            self.name = name

        def function(self, **kw):
            return lambda f: f

        def local_entrypoint(self, **kw):
            return lambda f: f

    class _Dict:
        """An in-memory ``modal.Dict`` double: ``from_name`` returns ONE store per name (as Modal does), ``put(...,
        skip_if_exists=True)`` is atomic under a lock and returns whether THIS call created the entry (the fact the
        verifier's claims gate rests on; modal 1.5.4), ``get`` / ``pop`` (KeyError when absent) / ``contains``."""

        _by_name: dict = {}

        def __init__(self):
            self.data: dict = {}
            self.lock = threading.Lock()

        @classmethod
        def from_name(cls, name, create_if_missing=False, **k):
            return cls._by_name.setdefault(name, cls())

        def put(self, key, value, *, skip_if_exists=False):
            with self.lock:
                if skip_if_exists and key in self.data:
                    return False
                self.data[key] = value
                return True

        def get(self, key, default=None):
            return self.data.get(key, default)

        def pop(self, key):
            with self.lock:
                return self.data.pop(key)

        def contains(self, key):
            return key in self.data

    m.App = _App
    m.Dict = _Dict
    m.Volume = SimpleNamespace(from_name=lambda *a, **k: SimpleNamespace(
        commit=lambda: None, reload=lambda: None))
    m.Image = SimpleNamespace(from_registry=lambda *a, **k: _Chain())
    m.Sandbox = object
    m.Secret = SimpleNamespace(from_name=lambda name, **k: ("secret", name))
    m.__version__ = "stub"
    m.is_local = lambda: True
    return m


@pytest.fixture(scope="module")
def L():
    return _load_launcher(_stub_modal(), app_name=TEST_APP_NAME)


class _Call:
    """A spawned FunctionCall double: ``object_id`` + ``get()`` — what the
    launcher's ``fan_out`` (one ``.spawn()`` per configuration, then ``.get()``
    each) reads back."""

    _n = 0

    def __init__(self, result):
        type(self)._n += 1
        self.object_id, self._result = f"fc-{type(self)._n:04d}", result

    def get(self, timeout=None):
        return self._result() if callable(self._result) else self._result


def _spawn_only(result_of, on_spawn=None):
    """A ``run_configuration`` double exposing ONLY ``spawn`` (never ``map``):
    ``on_spawn(cfg, kwargs)`` records the call, ``result_of(cfg)`` is what the
    returned :class:`_Call` yields."""
    def _spawn(cfg, **kwargs):
        if on_spawn is not None:
            on_spawn(cfg, kwargs)
        return _Call(lambda: result_of(cfg))
    return SimpleNamespace(spawn=_spawn)


def _done(cfg):
    return {"name": cfg["name"], "status": "DONE"}


#: The app name the module under test is loaded under — deliberately NOT the
#: contracts' historical default, so every assertion on ``L.APP_NAME`` proves
#: the name came from the environment (E40 M7: the launcher has no fallback).
TEST_APP_NAME = "tinycap-launcher-under-test"


def _load_launcher(modal_stub, *, app_name: str | None, name: str = "tinycap_served_under_test"):
    """Load the launcher by path against ``modal_stub`` with the app-name
    environment variable set to ``app_name`` (unset when ``None``) for the
    duration of the import — the App is built at import, so that is when the
    name is read."""
    saved = sys.modules.get("modal")
    sys.modules["modal"] = modal_stub
    saved_env = os.environ.get(C.MODAL_APP_NAME_ENV)
    if app_name is None:
        os.environ.pop(C.MODAL_APP_NAME_ENV, None)
    else:
        os.environ[C.MODAL_APP_NAME_ENV] = app_name
    try:
        spec = importlib.util.spec_from_file_location(name, LAUNCHER)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if saved is not None:
            sys.modules["modal"] = saved
        else:
            sys.modules.pop("modal", None)
        if saved_env is None:
            os.environ.pop(C.MODAL_APP_NAME_ENV, None)
        else:
            os.environ[C.MODAL_APP_NAME_ENV] = saved_env
    return mod


# --------------------------------------------------------------------------- #
# the run matrix                                                               #
# --------------------------------------------------------------------------- #


def test_app_and_matrix_names_come_from_the_contracts(L):
    # the app is named per wave, from the environment the module was imported
    # under — never the contracts' historical default
    assert L.app.name == L.APP_NAME == TEST_APP_NAME != C.MODAL_APP_NAME
    cfgs = L.plan_configurations()
    assert [c["name"] for c in cfgs] == [
        "tinycap-e4b-prompted_safety-think_off", "tinycap-e4b-prompted_safety-think_on",
        "tinycap-e4b-psrkl_safety-think_off", "tinycap-e4b-psrkl_safety-think_on"]
    assert set(c["name"] for c in cfgs) == set(C.CONTAINER_NAMES.values())
    for c in cfgs:
        assert c["out_dir"] == C.cell_out_dir(C.VOL_GEN_ROOT, c["cell"], c["mode"])
        assert c["out_dir"].startswith("/vol/tinycap/gen/gemma4e4b/")
        assert c["prompted"] == c["cell"].startswith("prompted__")
        # the task lists: the STANDARD battery on BOTH cell
        # kinds, minus MBPP+ / BFCL and minus the two generative letter-choice
        # legs; MATH-500 and the full GSM8K in neither (the tiny-only hold has
        # ended)
        assert c["controls"] == list(L.CONTROLS_PROMPTED if c["prompted"] else L.CONTROLS_FITTED)
        assert list(L.CONTROLS_FITTED) == list(L.CONTROLS_PROMPTED)
        assert "inspect_cmp:or_bench_hard_1k" in c["controls"]
        assert "bfcl" not in c["controls"] and "mbpp_plus" not in c["controls"]
        # U14: the generative twins of the tiny loglikelihood reads are not defaulted
        assert "hellaswag_inspect" not in c["controls"] and "inspect_cmp:mmlu" not in c["controls"]
        # U9 item 6: BFCL left the battery, so NO default standup composes --tool-calls
        assert c["tool_calls"] is False
        assert "math500" not in c["controls"] and "gsm8k_full" not in c["controls"]
        assert c["in_depth"] is False and c["battery"]["in_depth"] == [] == c["battery"]["other"]
        assert c["battery"]["standard"] == c["controls"]
        for tok in c["controls"]:
            C.expand_control(tok)
        # standup extras per configuration (E3 / P7), byte-equal to the contracts
        assert c["standup_extra"] == C.standup_extra_argv(tool_calls=c["tool_calls"],
                                                          thinking=c["mode"])
    by = {c["name"]: c for c in cfgs}
    # U9 item 6: no BFCL in either battery, so no default standup composes
    # --tool-calls / --manifest; thinking-ON still composes the reasoning parser
    assert by["tinycap-e4b-prompted_safety-think_off"]["standup_extra"] == []
    assert by["tinycap-e4b-prompted_safety-think_on"]["standup_extra"] == [
        "--reasoning-parser", "gemma4"]
    assert by["tinycap-e4b-psrkl_safety-think_off"]["standup_extra"] == []
    assert by["tinycap-e4b-psrkl_safety-think_on"]["standup_extra"] == [
        "--reasoning-parser", "gemma4"]
    # the sub-entries each configuration's ledger sizes: the whole battery is
    # generation, so one key per arm (no AGENT key: the agent geometry class was
    # BFCL's, and BFCL left the battery; no tiny key: that family is removed)
    assert by["tinycap-e4b-prompted_safety-think_off"]["geometry_keys"] == ("capability:gen_long",)
    assert by["tinycap-e4b-prompted_safety-think_on"]["geometry_keys"] == ("capability:gen_long_thinking",)
    assert by["tinycap-e4b-psrkl_safety-think_off"]["geometry_keys"] == ("capability:gen_long",)
    assert by["tinycap-e4b-psrkl_safety-think_on"]["geometry_keys"] == ("capability:gen_long_thinking",)


def test_default_controls_are_the_standard_battery_with_the_staging_exclusions(L, capsys):
    # native-serving-default: the launcher's default is the native engine now; this test pins the LENS
    # launch (a PSR cell / the historical composition), so it states the path explicitly
    """The ``--pass1`` defaults an operator does not have to type (PW17):
    prompted = the contracts' STANDARD battery with the staging exclusions
    applied; fitted = the SAME list (O7 / W18b).
    MBPP+ and BFCL left both defaults on 2026-09-06,
    and the two GENERATIVE letter-choice legs — ``inspect_cmp:mmlu`` and
    ``hellaswag_inspect`` — on 2026-09-07: the prefill-only
    letter reads were then the knowledge / sentence-completion instrument, so
    no duplicative generative version of the same eval is defaulted (the
    loglikelihood family itself was removed later). Both are STILL ACCEPTED when named in ``--controls``.
    MATH-500 avg@4 (U3) and the FULL GSM8K (U4(a)) are in NEITHER default —
    they are the ``--in-depth`` extras. The staging exclusion
    (``inspect_cmp:xstest``: walledai/XSTest is gated for every token we hold,
    status ``unavailable_gated``) is derived from ONE table and printed by the
    dry run; since U14 no default names a family it can narrow, but it still
    applies to an explicit ``--controls inspect_cmp``."""
    assert L.CONTROLS_PROMPTED == (
        "gsm8k", "humaneval_plus", "inspect_cmp:or_bench_hard_1k")
    assert L.CONTROLS_FITTED == L.CONTROLS_PROMPTED == (       # identical prompted and fitted
        "gsm8k", "humaneval_plus", "inspect_cmp:or_bench_hard_1k")
    # U9 item 6: still runnable, just not defaulted
    for gone in ("mbpp_plus", "bfcl_200", "bigcodebench_full", "bfcl_full"):
        assert gone not in L.CONTROLS_PROMPTED and gone not in L.CONTROLS_FITTED
        assert L.needs_tool_calls([gone]) is (gone in ("bfcl_200", "bfcl_full"))
    # the rule of 2026-09-11: the full bfcl and the two generative
    # letter legs are RETIRED from the grammar — refused, not merely undefaulted
    for retired in ("bfcl", "hellaswag_inspect", "inspect_cmp:mmlu"):
        assert retired not in L.CONTROLS_PROMPTED and retired not in L.CONTROLS_FITTED
        with pytest.raises(C.ControlNameError):
            C.expand_control(retired)
    assert L.needs_tool_calls(L.CONTROLS_PROMPTED) is False
    assert L.needs_tool_calls(L.CONTROLS_FITTED) is False
    assert L.CONTROLS_PROMPTED == L.apply_default_exclusions(C.STANDARD_BATTERY_PROMPTED)
    assert L.CONTROLS_FITTED == L.apply_default_exclusions(C.STANDARD_BATTERY_FITTED)
    assert L.CONTROLS_PROMPTED_IN_DEPTH == L.CONTROLS_PROMPTED + ("math500", "gsm8k_full")
    assert L.CONTROLS_FITTED_IN_DEPTH == L.CONTROLS_FITTED + ("math500", "gsm8k_full")
    for lst in (L.CONTROLS_PROMPTED, L.CONTROLS_FITTED):
        assert "math500" not in lst and "gsm8k_full" not in lst and "inspect_cmp:gsm8k" not in lst
    assert L.default_controls(C.CELL_PROMPTED_SAFETY) == L.CONTROLS_PROMPTED
    assert L.default_controls(C.CELL_PSRKL_SAFETY) == L.CONTROLS_FITTED
    assert L.default_controls(C.CELL_PSRKL_SAFETY, in_depth=True) == L.CONTROLS_FITTED_IN_DEPTH
    assert L.battery_kind(C.CELL_PROMPTED_SAFETY) == "prompted" and L.battery_kind(C.CELL_PSRKL_SAFETY) == "fitted"
    # a token whose atoms are all runnable stays as typed; only the family with
    # an excluded atom is expanded (the exclusion table is the one place STAGING
    # narrows a list). Since U14 the battery names no such family, so the
    # exclusion is an IDENTITY on it — but an explicit inspect_cmp still narrows
    assert L.apply_default_exclusions(C.STANDARD_BATTERY_PROMPTED) == C.STANDARD_BATTERY_PROMPTED
    assert L.apply_default_exclusions(("gsm8k", "inspect_cmp")) == (
        "gsm8k", "inspect_cmp:or_bench_hard_1k")
    assert set(L.EXCLUDED_DEFAULT_CONTROLS) == {"inspect_cmp:xstest"}
    rec = L.EXCLUDED_DEFAULT_CONTROLS["inspect_cmp:xstest"]
    assert rec["status"] == L.STATUS_UNAVAILABLE_GATED == "unavailable_gated"
    assert "walledai/XSTest" in rec["why"] and "403" in rec["why"]
    assert L.STATUS_UNAVAILABLE_OFFLINE == "unavailable_offline"      # the 6(a) fallback status, unused
    assert set(L.INSPECT_CMP_DEFAULT_ATOMS) == set(C.expand_control("inspect_cmp")) - {"inspect_cmp:xstest"}
    assert "inspect_cmp:xstest" not in L.CONTROLS_PROMPTED and "inspect_cmp" not in L.CONTROLS_PROMPTED
    for tok in L.CONTROLS_PROMPTED + L.CONTROLS_FITTED:
        C.expand_control(tok)
    # every default's tasks are real row tasks; gsm8k is the standard PAIRED one
    # (U4(a)) and the retained spelling resolves to the same row task
    assert L.expected_tasks("gsm8k", "off") == (C.GSM8K_STANDARD_TASK,) == ("inspect/gsm8k_paired200",)
    assert L.expected_tasks("inspect_cmp:gsm8k", "off") == (C.GSM8K_PAIRED_TASK,)
    assert L.expected_tasks("gsm8k_full", "off") == (C.GSM8K_FULL_TASK,) == ("inspect/gsm8k",)
    text = L.describe_exclusions()
    assert "inspect_cmp:xstest: unavailable_gated" in text and "--controls" in text
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, dry_run=True)
    out = capsys.readouterr().out
    assert "inspect_cmp:xstest: unavailable_gated" in out
    assert f"--controls {','.join(L.CONTROLS_PROMPTED)} " in out
    assert f"--controls {','.join(L.CONTROLS_FITTED)} " in out
    stripped = out.replace("(none; --in-depth adds math500, gsm8k_full)", "")
    assert "math500" not in stripped and "gsm8k_full" not in stripped     # in no argv, driver or identity
    # the dry run says what is standard and that the extras are one flag away
    # O7: the fitted battery IS the prompted one, so the one battery line appears
    # once per configuration (4), not twice per kind
    assert L.CONTROLS_FITTED == L.CONTROLS_PROMPTED
    assert out.count(f"   battery: standard = {', '.join(L.CONTROLS_PROMPTED)}; in-depth = (none; "
                     "--in-depth adds math500, gsm8k_full)") == 4
    assert "--in-depth" not in out.split("\n")[0]                      # the argv line has no flag
    # --in-depth: the extras join EVERY configuration's list and the argv line
    # carries the flag — and the dry run REFUSES it by name right after
    # (MATH-500 at 16,384 needs its own window, which one standup cannot serve
    # beside the standard battery's keys — test_math500_pooled_beside_other_keys_…);
    # the battery split is pinned on the plan itself
    with pytest.raises(L.Pass1FootprintCannotHold, match="MATH-500 at cap 16,384"):
        L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, in_depth=True, dry_run=True)
    out = capsys.readouterr().out
    assert (f"[tinycap] argv: TINYCAP_APP_NAME={L.APP_NAME} modal run scripts/modal/tinycap_served.py "
            f"--sha abc1234 --app-name {L.APP_NAME} --pass1 --in-depth ") in out
    assert "inspect_cmp:xstest: unavailable_gated" in out                # the exclusion still prints
    cfgs = L.plan_configurations(in_depth=True)
    for c in cfgs:
        assert c["in_depth"] is True and c["battery"]["in_depth"] == ["math500", "gsm8k_full"]
        assert c["controls"][-2:] == ["math500", "gsm8k_full"]
        assert c["battery"]["standard"] == c["controls"][:-2] and c["battery"]["other"] == []
        assert tuple(c["controls"]) == (L.CONTROLS_FITTED_IN_DEPTH if c["steered"]
                                        else L.CONTROLS_PROMPTED_IN_DEPTH)
    # `--controls math500` alone still runs it (an ordinary atom), and the print
    # files it under in-depth with no standard part
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, controls="math500", dry_run=True)
    out = capsys.readouterr().out
    assert out.count("--controls math500 ") == 5                        # the argv line + four drivers
    assert out.count("   battery: standard = (none); in-depth = math500") == 4
    (one,) = L.plan_configurations(cells=[C.CELL_PSRKL_SAFETY], modes=["off"], controls=["math500"])
    assert one["controls"] == ["math500"] and one["battery"] == {"standard": [], "in_depth": ["math500"], "other": []}
    # an explicit atom list is split the same way, `other` naming what is neither
    (mix,) = L.plan_configurations(cells=[C.CELL_PROMPTED_SAFETY], modes=["off"],
                                   controls=["mbpp_plus", "gsm8k", "gsm8k_full"])
    assert mix["battery"] == {"standard": ["gsm8k"], "in_depth": ["gsm8k_full"], "other": ["mbpp_plus"]}
    # --in-depth with an explicit --controls list refuses (a flag that did nothing
    # would be the worse failure): name the extras in the list instead
    with pytest.raises(ValueError, match="name them in it instead"):
        L.plan_configurations(controls=["gsm8k"], in_depth=True)
    with pytest.raises(SystemExit, match="name them in it instead"):
        L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, controls="gsm8k", in_depth=True, dry_run=True)
    capsys.readouterr()
    # an explicit override is the operator's choice: the exclusion note is not printed
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, controls="inspect_cmp:xstest", dry_run=True)
    out = capsys.readouterr().out
    assert "unavailable_gated" not in out and "--controls inspect_cmp:xstest " in out


def test_the_default_battery_drops_the_generative_letter_legs_and_still_accepts_them(L, capsys):
    # native-serving-default: the launcher's default is the native engine now; this test pins the LENS
    # launch (a PSR cell / the historical composition), so it states the path explicitly
    """Since 2026-09-07 the standard battery is
    ``gsm8k, humaneval_plus, inspect_cmp:or_bench_hard_1k`` — the prefill-only
    letter reads were then the knowledge / sentence-completion instrument (that
    family was removed later), so the GENERATIVE twins of the same eval
    (``inspect_cmp:mmlu`` = MMLU-letter + MMLU 0-shot, and
    ``hellaswag_inspect``) are not defaulted. GSM8K, HumanEval+ and OR-Bench
    stay: they are generative BY NATURE, with no prefill-only twin to prefer.

    Dropped from the DEFAULT, not from the harness: both tokens are still
    ordinary atoms of the grammar, still plan to their real row tasks, still
    reach the Inspect runner, and a configuration that names them runs them.
    That second half is the point of the test — a "drop" implemented by
    deleting the implementations would satisfy the first half alone."""
    three = ("gsm8k", "humaneval_plus", "inspect_cmp:or_bench_hard_1k")
    assert L.CONTROLS_PROMPTED == L.CONTROLS_FITTED == three
    assert L.default_controls(C.CELL_PROMPTED_SAFETY) == three
    assert L.default_controls(C.CELL_PSRKL_SAFETY) == three
    assert C.STANDARD_BATTERY_PROMPTED == C.STANDARD_BATTERY_FITTED == three
    # ... and every one of the three is a token the grammar and the planner take
    for tok in three:
        assert C.expand_control(tok) and L.expected_tasks(tok, "off")

    for retired, tasks in (("hellaswag_inspect", ("inspect/hellaswag",)),
                           ("inspect_cmp:mmlu", ("inspect/mmlu_letter_valid",
                                                 "inspect/mmlu_0_shot"))):
        # out of every DEFAULT list, standard and in-depth ...
        for lst in (L.CONTROLS_PROMPTED, L.CONTROLS_FITTED,
                    L.CONTROLS_PROMPTED_IN_DEPTH, L.CONTROLS_FITTED_IN_DEPTH):
            assert retired not in lst, (retired, lst)
        # ... and, since the rule of 2026-09-11, out of the GRAMMAR:
        # refused by the contracts and the planner alike; the row tasks stay
        # named only so existing stores read
        with pytest.raises(C.ControlNameError):
            C.expand_control(retired)
        with pytest.raises(C.ControlNameError):
            L.plan_configurations(cells=[C.CELL_PSRKL_SAFETY], modes=["off"], controls=[retired])
        assert all(t in C.TINYCAP_TASKS and t in C.THINKING_BY_TASK for t in tasks)
        assert retired not in L.EXCLUDED_DEFAULT_CONTROLS               # not a staging exclusion

    # the ruling removed the control from the BATTERY, not from the exclusion
    # table: the one staging exclusion is still XSTest and nothing else
    assert set(L.EXCLUDED_DEFAULT_CONTROLS) == {"inspect_cmp:xstest"}
    # a default dry run therefore names neither dropped leg anywhere in its argv
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, dry_run=True)
    out = capsys.readouterr().out
    assert f"--controls {','.join(three)} " in out
    assert "hellaswag_inspect" not in out and "inspect_cmp:mmlu" not in out


def test_matrix_subsets_and_refusals(L):
    one = L.plan_configurations(cells=[C.CELL_PSRKL_SAFETY], modes=["on"])
    assert [c["name"] for c in one] == ["tinycap-e4b-psrkl_safety-think_on"]
    # an explicit control list replaces the default, and --tool-calls follows BFCL
    sub = L.plan_configurations(cells=[C.CELL_PROMPTED_SAFETY], modes=["off"],
                                controls=["humaneval_plus", "math500"])
    assert sub[0]["controls"] == ["humaneval_plus", "math500"] and sub[0]["tool_calls"] is False
    assert sub[0]["standup_extra"] == []
    with_bfcl = L.plan_configurations(cells=[C.CELL_PROMPTED_SAFETY], modes=["off"],
                                      controls=["bfcl_200"])
    assert with_bfcl[0]["tool_calls"] is True and "--tool-calls" in with_bfcl[0]["standup_extra"]
    assert L.needs_tool_calls(["gsm8k", "inspect_cmp"]) is False
    # `none` is now a PLANNABLE cell, so the refusal case is an
    # ill-formed cell id instead — see test_plan_configurations_accepts_the_
    # whole_cell_grammar for the accepted set
    for bad in (dict(cells=["nope__psr_kl_all"]), dict(modes=["maybe"]),
                dict(controls=["mmlu_letter"])):
        with pytest.raises((ValueError, C.ControlNameError)):
            L.plan_configurations(**bad)
    assert L.plan_configurations(cells=["none"], modes=["off"],
                                 controls=["gsm8k"])[0]["base"] is True
    # thinking-off is planned first (the published suite's regime)
    assert [c["mode"] for c in L.plan_configurations(modes=["on", "off"])] == ["off", "on", "off", "on"]


def test_smoke_plan_is_one_fitted_thinking_off_container(L):
    (s,) = L.smoke_plan()
    assert s["name"] == "tinycap-e4b-psrkl_safety-think_off" and s["cell"] == C.CELL_PSRKL_SAFETY
    # U9 item 6: BFCL left the battery, so the smoke standup composes nothing
    assert s["mode"] == "off" and s["tool_calls"] is False and s["standup_extra"] == []
    # native-serving-default (§1.7): the retained PSR smoke is the LENS smoke — the library default,
    # equal to the explicit `--serving-path lens` plan, unmarked, and its standup states the lens path
    assert L.smoke_plan(serving_path="lens") == [s] and L.SERVING_PATH_KEY not in s
    geom, _ = L.served_geometry(s, 1)
    argv = L._standup_cmd("/t", "/t/c.yaml", s["cell"], "/d", Path("/srv"), Path("/srv/STOP"), geom, 64, s["standup_extra"])
    assert argv[argv.index(L.SERVING_PATH_FLAG) + 1] == "lens" and L.ADAPTER_BANK_FLAG not in argv


def test_plan_configurations_accepts_the_whole_cell_grammar(L):
    """Any cell the serving side can express — `none`,
    `prompted__<si>`, `<si>__<arm>` — not just the two test cells."""
    cells = ["none", "prompted__si_irrelevant", "si_constitution__lora_r4"]
    cfgs = L.plan_configurations(cells=cells, modes=["off"], controls=["gsm8k"])
    assert [c["name"] for c in cfgs] == ["tinycap-e4b-base-think_off",
                                         "tinycap-e4b-prompted_irrelevant-think_off",
                                         "tinycap-e4b-lora_r4_constitution-think_off"]
    base, prompted, fitted = cfgs
    assert (base["base"], base["prompted"], base["steered"]) == (True, False, False)
    assert (prompted["base"], prompted["prompted"], prompted["steered"]) == (False, True, False)
    assert (fitted["base"], fitted["prompted"], fitted["steered"]) == (False, False, True)
    assert base["out_dir"] == "/vol/tinycap/gen/gemma4e4b/none/thinking_off"
    # a generation leg pools into one entry whether or not the cell carries an SI
    # prefix (only MATH-500 resolves its window by the prompt class)
    assert L.geometry_keys_for("none", "off", ["gsm8k"]) == ("capability:gen_long",)
    assert L.geometry_keys_for(C.CELL_PROMPTED_SAFETY, "off", ["gsm8k"]) == (
        "capability:gen_long",)
    # the un-steered base cell takes the un-steered battery (identical to the
    # fitted one today — O7 / PW18b — so this is about which side it is on)
    assert L.battery_kind("none") == C.BATTERY_PROMPTED == L.battery_kind(C.CELL_PROMPTED_SAFETY)
    assert L.battery_kind(C.CELL_PSRKL_SAFETY) == C.BATTERY_FITTED
    for bad in ("prompted__nope", "safety_refusal__alphakl_0", "notacell"):
        with pytest.raises(C.CellNameError):
            L.plan_configurations(cells=[bad], modes=["off"], controls=["gsm8k"])


def test_a_base_cell_serves_a_passthrough_route_so_the_bites_probe_does_not_apply():
    """The launcher verifies that the driver's evidence
    steps behave for `none` exactly as for a passthrough cell. They do, and
    for a reason read out of the serving code rather than assumed:
    ``defense_cells.enumerate_cells`` gives BOTH the base cell and every
    prompted cell ``path = "passthrough"``, and the driver gates the
    steering-bites probe on ``bon_served._is_lens(route)`` — which reads that
    path. Salt isolation is a property of the whole route SET, so it runs for
    every kind (and a base standup's set is just `none` and its control legs).
    """
    from sisteer.capeval import manifest as cm
    from sisteer.eval.attacks import bon_served as bs

    def route(cell_id, path):
        return bs.RouteEntry(
            virtual_model=f"vm-{cell_id}", cell_id=cell_id,
            cell_class="base" if cell_id == "none" else "prompted",
            serving_identity=cm.serving_identity("proxy_vllm", path, "interleaved"),
            intervention_hash="0" * 16)

    assert bs._is_lens(route("none", "passthrough")) is False
    assert bs._is_lens(route("prompted__safety", "passthrough")) is False
    assert bs._is_lens(route("si_safety__psr_kl_all", "lens")) is True
    # and the driver really does gate on it, recording `applicable: False`
    driver = (REPO / "scripts/served_capability_run.py").read_text()
    assert "lens = bs._is_lens(route)" in driver
    assert 'bites = {"applicable": False,' in driver
    assert "bs.assert_cell_salt_isolation(routes)" in driver


# --------------------------------------------------------------------------- #
# --model: one invocation, one model                                          #
# --------------------------------------------------------------------------- #


def test_the_model_slug_threads_through_every_per_configuration_path(L):
    cfgs = L.plan_configurations(cells=["safety_refusal__psr_kl_all"], modes=["off"],
                                 controls=["gsm8k"], model="qwen35_9b")
    (cfg,) = cfgs
    row = L.row_of(cfg)
    assert cfg["model"] == "qwen35_9b" and row.hf == "Qwen/Qwen3.5-9B"
    assert cfg["name"] == "tinycap-qwen9b-psrkl_safety-think_off"
    # the out root, the ledger and every store hang off the model's own dir
    assert cfg["out_dir"] == "/vol/tinycap/gen/qwen35_9b/safety_refusal__psr_kl_all/thinking_off"
    # S comes from THIS model's measured budget, not the default model's
    geom, budget = L.served_geometry(cfg, 1)
    assert geom.model_key == "qwen35_9b" == budget["model_key"]
    assert geom.max_num_seqs == sg.BUDGET_BY_MODEL["qwen35_9b"].seqs_at(geom.max_model_len)
    # ... and the standup and driver are pointed at this model's config + bank
    cmd = L._driver_cmd("<tree>", cfg, {"proxy_url": "u", "routing_table": "t"}, None, geom,
                        controls=cfg["controls"], limit=0, inspect_limit=0, samples=4,
                        cfg_path="<tree>/" + row.cfg, vllm_log=None)
    assert "--defenses-dir" in cmd
    assert cmd[cmd.index("--defenses-dir") + 1] == "/vol/objects/scale-12/qwen35_9b"
    assert cmd[cmd.index("--config") + 1] == "<tree>/configs/two_si_qwen35_9b_fits.yaml"
    assert cmd[cmd.index("--concurrency") + 1] == str(geom.max_num_seqs)
    # the DOSE variants are separate slugs with their own bank and out root
    (x2,) = L.plan_configurations(cells=["safety_refusal__psr_kl_all"], modes=["off"],
                                  controls=["gsm8k"], model="gemma4e4b-x2")
    assert L.row_of(x2).objects_dir == "/vol/capctrl/objects/scale-12-x2/gemma4e4b"
    assert x2["out_dir"].startswith("/vol/tinycap/gen/gemma4e4b-x2/")
    assert L.row_of(x2).model_key == "gemma4e4b"        # same weights, same budget
    assert x2["name"] == "tinycap-e4b_x2-psrkl_safety-think_off"
    with pytest.raises(C.UnknownModel):
        L.plan_configurations(model="nope")


def test_a_configuration_without_a_model_key_is_the_default_model(L):
    """Every planned configuration carries `model` now, but `row_of` also has
    to read a cfg dict that predates the key (a pinned fixture, a REC being
    re-read) — those are all gemma4e4b by construction."""
    assert L.row_of({}).slug == C.MODEL_SLUG
    assert L.row_of({"model": None}).slug == C.MODEL_SLUG
    assert L.row_of({"model": "llama8b"}).slug == "llama8b"
    # and the module constants still name the default model's inputs
    assert L.OBJECTS_DIR == C.model_row(C.MODEL_SLUG).objects_dir
    assert not hasattr(L, "GEMMA_CFG")


def test_bfcl_refuses_on_a_family_with_no_verified_tool_parser(L):
    """A parser foreign to the served tokenizer is a healthy-looking server
    that 500s on every chat request half an hour in (job 232411), so a Qwen
    wave refuses BFCL at plan time and names the remedy — it never composes
    the gemma or llama parser onto a qwen target."""
    with pytest.raises(ValueError) as e:
        L.plan_configurations(cells=["none"], modes=["off"], controls=["bfcl_200", "gsm8k"],
                              model="qwen35_9b")
    msg = str(e.value)
    assert "no petri manifest with a VERIFIED tool-call parser" in msg
    assert "TOOL_PARSER_BY_FAMILY" in msg and "--controls gsm8k" in msg
    # U9 item 6 took BFCL OUT of the default battery, so the default qwen wave
    # no longer trips the refusal — only an explicit --controls bfcl_200 does
    dflt = L.plan_configurations(cells=["none"], modes=["off"], model="qwen35_9b")
    assert "bfcl" not in dflt[0]["controls"]
    assert dflt[0]["tool_calls"] is False and dflt[0]["standup_extra"] == []
    # everything else in the battery plans fine
    keep = [c for c in L.CONTROLS_PROMPTED if c != "bfcl"]
    cfgs = L.plan_configurations(cells=["none"], modes=["off"], controls=keep,
                                 model="qwen35_9b")
    assert cfgs[0]["tool_calls"] is False and cfgs[0]["standup_extra"] == []


def test_the_standup_extras_carry_the_row_s_family_and_manifest(L):
    """The gemma manifest on a llama target is the job-232411 failure; the
    family drives BOTH the tool-parser manifest and the reasoning-parser pin."""
    (llama,) = L.plan_configurations(cells=["none"], modes=["off"], controls=["bfcl_200"],
                                     model="llama8b")
    assert llama["standup_extra"] == ["--tool-calls", "--manifest", C.PETRI_MANIFEST_TWOSI]
    (gemma,) = L.plan_configurations(cells=["none"], modes=["on"], controls=["bfcl_200"],
                                     model="gemma4e4b")
    assert gemma["standup_extra"] == ["--tool-calls", "--manifest", C.PETRI_MANIFEST_GEMMA4E4B,
                                      "--reasoning-parser", "gemma4"]
    # llama has no reasoning-parser pin: thinking-ON is refused, never silently OFF
    # (the full refusal is pinned in test_thinking_on_refuses_per_family)
    with pytest.raises(L.ThinkingUnservable, match="no pinned reasoning parser"):
        L.plan_configurations(cells=["none"], modes=["on"], controls=["gsm8k"], model="llama8b")


def test_the_llama3b_config_overlay_supplies_the_two_nulls(L, tmp_path):
    """The checked-in config pins `layer: null` and `model_revision: null`, so
    the launcher writes a COPY with both filled — never an edit to the config,
    never a write inside the staged tree."""
    tree = tmp_path / "tree"
    (tree / "configs").mkdir(parents=True)
    src = REPO / "configs/two_si_llama32_3b.yaml"
    (tree / "configs/two_si_llama32_3b.yaml").write_text(src.read_text())
    out = tmp_path / "out"
    row = C.model_row("llama3b")
    path, overlaid = L.overlaid_config(row, str(tree), out)
    assert path == str(out / L.CONFIG_OVERLAY_FILENAME)
    assert overlaid == {"layer": 15,
                        "model_revision": "006f5dcd1393c3add266de40994ba96225e9689d"}
    doc = yaml.safe_load(Path(path).read_text())
    orig = yaml.safe_load(src.read_text())
    assert doc["layer"] == 15 and doc["model_revision"] == row.rev
    # ... and NOTHING else moved, including the checked-in file itself
    assert {k: v for k, v in doc.items() if k not in overlaid} == {
        k: v for k, v in orig.items() if k not in overlaid}
    assert yaml.safe_load((tree / "configs/two_si_llama32_3b.yaml").read_text()) == orig
    # a complete config gets no overlay at all — the staged path is served
    row8 = C.model_row("llama8b")
    (tree / "configs/two_si.yaml").write_text((REPO / row8.cfg).read_text())
    assert L.overlaid_config(row8, str(tree), out) == (f"{tree}/{row8.cfg}", {})
    # an overlay may only REPLACE a key the config pins; it never invents one
    bad = C.ModelRow(slug="x", hf="h", rev="r", cfg="configs/two_si.yaml", objects_dir="/o",
                     model_key="llama8b", family="llama", thinking_modes=("off",),
                     cfg_overrides={"not_a_key": 1})
    with pytest.raises(KeyError, match="has no key"):
        L.overlaid_config(bad, str(tree), out)


def test_the_dry_run_names_the_model_and_the_argv_spells_the_flag(L, capsys):
    # native-serving-default: the launcher's default is the native engine now; this test pins the LENS
    # launch (a PSR cell / the historical composition), so it states the path explicitly
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, model="llama3b", cells="none", modes="off",
           controls="gsm8k", dry_run=True)
    out = capsys.readouterr().out
    assert f"--sha abc1234 --app-name {L.APP_NAME} --model llama3b --pass1" in out
    assert "== tinycap-llama3b-base-think_off  model=llama3b cell=none" in out
    assert "model: unsloth/Llama-3.2-3B-Instruct@006f5dcd1393 family=llama" in out
    assert "objects=/vol/tinycap/objects/llama3b" in out
    assert "config overlay {'layer': 15," in out
    assert f"--config <out>/{L.CONFIG_OVERLAY_FILENAME}" in out
    assert "/vol/tinycap/gen/llama3b/none/thinking_off" in out
    # the DEFAULT model is not spelled into the argv (every old command line
    # still means what it meant)
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, cells="none", modes="off", controls="gsm8k",
           dry_run=True)
    argv_line = next(ln for ln in capsys.readouterr().out.splitlines()
                     if ln.startswith("[tinycap] argv:"))
    assert L.MODEL_FLAG not in argv_line and f"--sha abc1234 --app-name {L.APP_NAME} --pass1" in argv_line
    with pytest.raises(SystemExit, match="unknown model slug"):
        L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, model="nope", dry_run=True)


# --------------------------------------------------------------------------- #
# preflight: the object IS the published object, before any GPU minute          #
# --------------------------------------------------------------------------- #


def test_identity_request_names_every_fitted_cell_and_its_published_hash(L):
    req = L.identity_request(C.model_row("gemma4e4b"), "/T",
                             ["none", "prompted__safety_refusal",
                              "safety_refusal__psr_kl_all", "si_irrelevant__lora_r4"])
    assert req["model"] == "google/gemma-4-E4B-it"
    assert req["config"] == "/T/configs/two_si_gemma4e4b_reduced.yaml"
    assert req["objects_dir"] == "/vol/objects/scale-12/gemma4e4b"
    assert req["config_overrides"] == {}
    # base and prompted cells carry no object, so there is no identity to derive
    assert [t["cell"] for t in req["targets"]] == ["safety_refusal__psr_kl_all",
                                                   "si_irrelevant__lora_r4"]
    assert req["targets"][0] == {"cell": "safety_refusal__psr_kl_all",
                                 "si_id": "safety_refusal", "arm": "psr_kl_all",
                                 "expected": "d50f46c19733de40"}
    # llama3b's overlay travels with the request (its config pins both nulls)
    req3 = L.identity_request(C.model_row("llama3b"), "/T", ["safety_refusal__psr_kl_all"])
    assert req3["config_overrides"] == {"layer": 15, "model_revision": C.model_row("llama3b").rev}
    assert req3["targets"][0]["expected"] == "93b6466dd791cc7e"
    # an explicit expectation WINS over the table; a cell with none is still
    # derived (and reported), it just cannot refuse
    req4 = L.identity_request(C.model_row("gemma4e4b"), "/T",
                              ["safety_refusal__psr_kl_all", "si_constitution__vector_all"],
                              {"safety_refusal__psr_kl_all": "0123456789abcdef"})
    by = {t["cell"]: t["expected"] for t in req4["targets"]}
    assert by == {"safety_refusal__psr_kl_all": "0123456789abcdef",
                  "si_constitution__vector_all": None}   # carried as None: the derivation REFUSES on it (E40 H1)
    # E40 H1: the bare Class B cells are targets too (no SI; their expectation
    # can only come from --expect-hashes — the table has no Class B rows)
    req5 = L.identity_request(C.model_row("gemma4e4b"), "/T", ["antidote", "alphasteer", "none"],
                              {"antidote": "fedcba9876543210"})
    assert req5["targets"] == [
        {"cell": "antidote", "si_id": None, "arm": "antidote", "expected": "fedcba9876543210"},
        {"cell": "alphasteer", "si_id": None, "arm": "alphasteer", "expected": None}]


def test_object_identity_failures_names_both_hashes_and_the_header(L):
    ok = {"cell": "a", "match": True, "header_guard_will_pass": True}
    assert L.object_identity_failures({"results": [ok]}) == []
    # O22: a LoRA cell records NO header (a native adapter, not a header-masked
    # lens object), so the guard is not applicable and a planned wave carrying
    # one reaches the report unrefused. The verdict itself is pinned in
    # tests/test_object_identity.py.
    lora = {"cell": "safety_refusal__lora_r4", "match": True, "response_header_len": None,
            "positions": None, "header_guard_will_pass": True,
            "header_guard": "not_applicable_no_recorded_header"}
    assert L.object_identity_failures({"results": [ok, lora]}) == []
    bad = L.object_identity_failures({"results": [
        {"cell": "wrong_object", "match": False, "header_guard_will_pass": True,
         "intervention_hash": "1111111111111111", "expected": "2222222222222222"},
        {"cell": "wrong_header", "match": True, "header_guard_will_pass": False,
         "response_header_len": 7, "generation_header_len": 5, "positions": "response"},
        {"cell": "exploded", "error": "RuntimeError: boom"},
        # E40 H1: no expectation anywhere — refused, the derived hash and the remedy named
        {"cell": "unpinned", "match": False, "expected": None, "header_guard_will_pass": True,
         "intervention_hash": "3333333333333333",
         "no_expectation": "no expected HF post-load intervention hash for this cell — …"},
    ]})
    assert len(bad) == 4
    # BOTH hashes are named — a refusal that says only "mismatch" is useless
    assert "1111111111111111" in bad[0] and "2222222222222222" in bad[0]
    assert "would not be a re-run" in bad[0]
    assert "response_header_len 7" in bad[1] and "5-token generation header" in bad[1]
    assert bad[2] == "exploded: RuntimeError: boom"
    assert bad[3].startswith("unpinned: no expected HF post-load intervention hash")
    assert "--expect-hashes '{\"unpinned\": \"3333333333333333\"}'" in bad[3] and "E40 H1" in bad[3]


def test_parse_expect_hashes(L, tmp_path):
    assert L.parse_expect_hashes("") == {} == L.parse_expect_hashes("   ")
    assert L.parse_expect_hashes('{"c": "abc"}') == {"c": "abc"}
    f = tmp_path / "h.json"
    f.write_text(json.dumps({"c": "abc", "d": "def"}))
    assert L.parse_expect_hashes(str(f)) == {"c": "abc", "d": "def"}
    for bad, needle in (("/no/such/file.json", "neither a JSON object nor a readable file"),
                        ('{"c": 1}', "flat JSON object"),
                        ('["c"]', "flat JSON object"),
                        ('{oops}', "not JSON")):
        with pytest.raises(ValueError, match=re.escape(needle)):
            L.parse_expect_hashes(bad)


def test_preflight_resolves_artifacts_through_the_standup_s_own_resolver(L, tmp_path):
    """E40 H2: the preflight used to look for ``<bank>/<cell>/object.pt`` for
    every fitted cell, so an antidote object (``object.safetensors``, per
    ``methods.object_artifact_name``) was reported MISSING while present.
    ``planned_object_paths`` goes through ``defense_cell.object_path_for`` —
    the resolver the standup's registry uses — so the two can never disagree:
    antidote's safetensors, a torch blob's ``.pt``, the legacy ``lora_r1_all``
    directory when it holds the artifact; base / prompted cells carry none."""
    pytest.importorskip("torch")
    bank = tmp_path / "bank"
    (bank / "antidote").mkdir(parents=True)
    (bank / "antidote" / "object.safetensors").write_bytes(b"st")
    (bank / "safety_refusal__lora_r1_all").mkdir()
    (bank / "safety_refusal__lora_r1_all" / "object.pt").write_bytes(b"legacy")
    paths = L.planned_object_paths(bank, ["none", "prompted__safety_refusal", "antidote", "alphasteer",
                                          "safety_refusal__psr_kl_all", "safety_refusal__lora_r4"])
    assert paths == {
        "antidote": bank / "antidote" / "object.safetensors",             # present, and NAMED right
        "alphasteer": bank / "alphasteer" / "object.pt",                  # a torch blob
        "safety_refusal__psr_kl_all": bank / "safety_refusal__psr_kl_all" / "object.pt",
        "safety_refusal__lora_r4": bank / "safety_refusal__lora_r1_all" / "object.pt",   # the staged legacy dir
    }
    assert paths["antidote"].is_file() and paths["safety_refusal__lora_r4"].is_file()
    with pytest.raises(C.CellNameError):
        L.planned_object_paths(bank, ["circuit_breaker"])                # not launchable: refused, not "missing"
    # the preflight's check reads exactly this map
    src = LAUNCHER.read_text()
    assert "for cell, obj in planned_object_paths(row.objects_dir, sel).items():" in src
    assert '/ cell / "object.pt"' not in src


def test_the_derivation_module_is_held_to_the_staged_tree(L):
    """The hashes must come from the code the wave runs, so the module joins
    the list preflight checks byte-for-byte against the staged tree — and the
    launcher composes `python -m <module>` from the contracts' name, never a
    retyped path."""
    assert "capeval/tinycap/object_identity.py" in L.LAUNCHER_PROCESS_MODULES
    assert C.OBJECT_IDENTITY_MODULE == "sisteer.capeval.tinycap.object_identity"
    assert (REPO / "src" / C.OBJECT_IDENTITY_MODULE.replace(".", "/")).with_suffix(".py").is_file()
    # E40 M4: the preflight summary reports the hash the lens ROUTE stamps, under the
    # harness id it is filed under — never the runtime-static number nothing stamps
    src = LAUNCHER.read_text()
    assert '"harness_cell_id", "lens_route_intervention_hash",' in src
    assert "lens_static_intervention_hash" not in src


def test_run_object_identity_uses_the_staged_tree_and_reports_a_crash(L, monkeypatch):
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"], seen["env"] = list(argv), dict(kw.get("env") or {})
        return SimpleNamespace(returncode=0, stdout=json.dumps({"ok": True, "results": []}),
                               stderr="")
    monkeypatch.setattr(L.subprocess, "run", fake_run)
    out = L.run_object_identity({"targets": []}, "/staged/tree")
    assert out == {"ok": True, "results": []}
    assert seen["argv"][1:3] == ["-m", C.OBJECT_IDENTITY_MODULE]
    assert seen["env"]["PYTHONPATH"] == "/staged/tree/src"      # the code the wave runs
    # a crash is an AssertionError carrying the child's own output, never a
    # silent "no objects checked"
    monkeypatch.setattr(L.subprocess, "run", lambda argv, **kw: SimpleNamespace(
        returncode=1, stdout="", stderr="ImportError: no torch"))
    with pytest.raises(AssertionError, match="ImportError: no torch"):
        L.run_object_identity({"targets": []}, "/staged/tree")


def test_the_expect_hashes_flag_reaches_preflight_and_the_recorded_argv(L, monkeypatch, capsys):
    # native-serving-default: the launcher's default is the native engine now; this test pins the LENS
    # launch (a PSR cell / the historical composition), so it states the path explicitly
    got = {}
    def _remote(*a):
        got["args"] = a
        return {"ok": True, "checks": {}}
    monkeypatch.setattr(L, "preflight", SimpleNamespace(remote=_remote))
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pf=True, model="llama3b", cells="safety_refusal__psr_kl_all",
           modes="off", expect_hashes='{"safety_refusal__psr_kl_all": "93b6466dd791cc7e"}')
    sha, plan_args, expect = got["args"]
    assert sha == "abc1234" and expect == {"safety_refusal__psr_kl_all": "93b6466dd791cc7e"}
    assert plan_args["model"] == "llama3b" and plan_args["cells"] == ["safety_refusal__psr_kl_all"]
    assert plan_args["modes"] == ["off"]
    argv = " ".join(L.launch_argv("abc1234", model="llama3b", expect_hashes='{"c": "d"}'))
    assert '--expect-hashes {"c": "d"}' in argv          # reproducible from the record
    with pytest.raises(SystemExit, match="flat JSON object"):
        L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pf=True, expect_hashes='{"c": 1}')


def test_preflight_plans_exactly_what_the_launch_plans(L, monkeypatch, capsys):
    """E40 M6: the preflight used to plan from modes + model alone (cells
    passed; controls and --in-depth not), so a custom control battery could
    pass on the DEFAULT battery's geometry keys and fail inside the paid
    container. Now `main` builds ONE `plan_args` mapping, plans the launch
    from it and hands the same mapping to the preflight, which plans through
    the same `plan_for` — pinned here: for explicit cells / controls, for
    --in-depth, and for --smoke, the preflight's configuration names and
    geometry keys ARE the launched ones, and the preflight body plans from
    `plan_for(**plan_args)` (never a re-plan from modes + model)."""
    got: dict = {}

    def _remote(sha, plan_args, expect):
        got["plan_args"] = plan_args
        # a fitted cell launches only against a preflight-verified digest (E40 H1):
        # the fake preflight vouches for every fitted cell of the plan it was handed
        return _preflight_report(L, L.plan_for(**plan_args))
    launched: list = []
    monkeypatch.setattr(L, "preflight", SimpleNamespace(remote=_remote))
    monkeypatch.setattr(L, "run_configuration", _spawn_only(_done, lambda c, k: launched.append(c)))

    def check(**kw):
        launched.clear()
        L.main(app_name=L.APP_NAME, sha="abc1234", pf=True, **kw)
        capsys.readouterr()
        pf_plan = L.plan_for(**got["plan_args"])
        assert launched, kw
        assert [c["name"] for c in pf_plan] == [c["name"] for c in launched]
        assert [c["geometry_keys"] for c in pf_plan] == [c["geometry_keys"] for c in launched]
        assert [c["controls"] for c in pf_plan] == [c["controls"] for c in launched]
        return got["plan_args"], launched
    # explicit cells AND a custom control battery: the keys are the battery's, not the default's
    # (MATH-500 alone: the default battery's keys are all gen_long, so a battery
    # that differs from the default must name another geometry class)
    pa, cfgs = check(pass1=True, cells="none,prompted__si_irrelevant", modes="off",
                     controls="math500")
    assert pa == {"smoke": False, "cells": ["none", "prompted__si_irrelevant"], "modes": ["off"],
                  "controls": ["math500"], "in_depth": False, "model": C.MODEL_SLUG,
                  # PW-oos: part of the planning inputs, so the preflight plans the
                  # SAME out-of-support configurations the launch runs; false here
                  "allow_out_of_support_thinking": False,
                  # offcap: the comparator flag is a planning input for the same reason
                  "allow_off_cap_override": False,
                  # E50 #3: the draw is part of the authorised out-of-support
                  # tuple, so the planner needs it to make that check
                  "samples": C.MATH500_SAMPLES_PER_ITEM,
                  # code review H2: so is the requested lens reduction mode — both
                  # tuples name per_row, and the preflight replans from this mapping
                  "reduction_mode": C.REDUCTION_MODE_LAUNCHER_DEFAULT,
                  # code review H3: and the requested geometry pass — both tuples name
                  # pass 1, and the preflight budgets the REQUESTED pass's window
                  "pass_no": 1,
                  # the draw's MODE is a planning input —
                  # the plan marks each configuration (SAMPLING_MODE_KEY, iff non-default)
                  # and the preflight replans the same; the default here, spelled
                  # through the contract, never a literal
                  "sampling_mode": C.SAMPLING_DEFAULT,
                  # native-serving-default (§1.7): the serving PATH is a planning input — the
                  # entry-point default (native) here, so the preflight namespaces / marks the same
                  "serving_path": L.SERVING_PATH_DEFAULT}
    assert L.SERVING_PATH_DEFAULT == "lora_native" and all(c.get(L.SERVING_PATH_KEY) == "lora_native" for c in cfgs)
    default_keys = [c["geometry_keys"] for c in L.plan_configurations(
        cells=["none", "prompted__si_irrelevant"], modes=["off"])]
    assert [c["geometry_keys"] for c in cfgs] != default_keys          # a re-plan from modes+model would differ
    # --in-depth: the extras join the plan on both sides — and `main` refuses
    # the launch by name BEFORE the preflight is asked (MATH-500 at 16,384
    # pooled beside the standard battery's keys: one standup cannot serve both
    # windows); no Modal call, nothing launched
    got.clear()
    launched.clear()
    with pytest.raises(L.Pass1FootprintCannotHold, match="--controls math500"):
        L.main(app_name=L.APP_NAME, sha="abc1234", pf=True, pass1=True, cells="none", modes="off",
               in_depth=True)
    capsys.readouterr()
    assert not got and not launched
    deep = L.plan_for(cells=["none"], modes=["off"], in_depth=True)
    assert deep and all(set(C.IN_DEPTH_EXTRAS) <= set(c["controls"]) for c in deep)
    # --smoke: the smoke plan, on both sides (a prompted-only control list keeps this free of
    # the fitted object's digest requirement)
    # (the historical PSR smoke is a LENS smoke — under the native default the smoke needs --cells)
    pa, cfgs = check(smoke=True, controls="gsm8k", serving_path="lens")
    assert pa["smoke"] is True and [c["name"] for c in cfgs] == ["tinycap-e4b-psrkl_safety-think_off"]
    assert pa["serving_path"] == "lens" and L.SERVING_PATH_KEY not in cfgs[0]
    # another model: the row travels in the plan, not as a separate argument
    pa, _ = check(pass1=True, model="llama3b", cells="none", modes="off", controls="gsm8k")
    assert pa["model"] == "llama3b"
    # the preflight body plans from the mapping, and nowhere else
    src = LAUNCHER.read_text()
    body = src[src.index("def preflight("):src.index("    def datasets():")]      # the whole preflight body
    # the preflight plans from the mapping AND names the STAGED tree as the matrix's source
    assert "cfgs = plan_for(**plan_args, ifeval_matrix_tree=tree)" in body and "for cfg in cfgs:" in body
    assert "plan_configurations(modes=modes, model=row.slug)" not in src
    assert "preflight.remote(sha, plan_args, expect)" in src
    # a --pf-only invocation still plans (what --pass1 with these arguments would)
    got.clear()
    L.main(app_name=L.APP_NAME, sha="abc1234", pf=True, cells="none", modes="off", controls="gsm8k")
    capsys.readouterr()
    assert [c["name"] for c in L.plan_for(**got["plan_args"])] == ["tinycap-e4b-base-think_off"]


# --------------------------------------------------------------------------- #
# per-family thinking refusals, --modes per-task, --app-name                   #
# --------------------------------------------------------------------------- #


def test_thinking_on_refuses_per_family_and_never_serves_it_as_off(L):
    """Two refusals, both at PLAN time. (a) a family with no
    verified reasoning parser; (b) a FITTED cell of a family whose served
    render is pinned OFF and whose header MOVES. Gemma keeps thinking-ON on
    both kinds — its header is invariant at H=3 and its parser is pinned."""
    # (a) llama: no switch in the template at all, either kind
    for cell in ("none", "prompted__safety_refusal", "safety_refusal__psr_kl_all"):
        with pytest.raises(L.ThinkingUnservable) as e:
            L.plan_configurations(cells=[cell], modes=["on"], controls=["gsm8k"],
                                  model="llama8b")
        msg = str(e.value)
        assert "no pinned reasoning parser" in msg and "no thinking switch at all" in msg
        assert "will not serve an ON request as OFF" in msg
    # (b) a FITTED qwen cell: the fit-regime refusal, citing the pin and the measurement
    with pytest.raises(L.ThinkingUnservable) as e:
        L.plan_configurations(cells=["safety_refusal__psr_kl_all"], modes=["on"],
                              controls=["gsm8k"], model="qwen35_9b")
    msg = str(e.value)
    assert "THINKING_OFF_TEMPLATE_KWARGS" in msg and "H=7" in msg and "H=5" in msg
    assert C.THINKING_HEADER_EVIDENCE in msg
    # a PROMPTED qwen cell has no object to misalign, but still no parser
    with pytest.raises(L.ThinkingUnservable, match="no pinned reasoning parser"):
        L.plan_configurations(cells=["prompted__safety_refusal"], modes=["on"],
                              controls=["gsm8k"], model="qwen35_9b")
    # gemma stays allowed on both kinds (the 1983322 fitted ON runs are valid)
    for cell in ("none", "safety_refusal__psr_kl_all"):
        (cfg,) = L.plan_configurations(cells=[cell], modes=["on"], controls=["gsm8k"],
                                       model="gemma4e4b")
        assert "--reasoning-parser" in cfg["standup_extra"]
    # every model's OFF arm plans fine — the refusal is about ON only
    for slug in C.MODEL_SLUGS:
        ctl = ["gsm8k"]
        assert L.plan_configurations(cells=["none"], modes=["off"], controls=ctl,
                                     model=slug)[0]["mode"] == "off"


def test_the_refusal_also_fires_inside_the_container(L, tmp_path, monkeypatch):
    """A REC, not a traceback: the container re-derives the refusal from the
    REQUESTED controls, so a resume that narrows them cannot slip past the
    plan-time check."""
    cfg = L.plan_configurations(cells=["safety_refusal__psr_kl_all"], modes=["off"],
                                controls=["gsm8k"], model="qwen35_9b")[0]
    rec = L._run_configuration(dict(cfg, mode="on", out_dir=str(tmp_path)), "f6cf462")
    assert rec["status"] == "THINKING_UNSERVABLE"
    assert "THINKING_OFF_TEMPLATE_KWARGS" in rec["error"]


def test_per_task_composes_one_standup_and_records_the_map(L):
    """U5 / O7 item 4: one standup, each task at its own arm, the parser
    composed iff ANY served task is ON, and `thinking_by_task` — that exact key
    — in the identity."""
    (cfg,) = L.plan_configurations(cells=["none"], modes=["per-task"],
                                   controls=["gsm8k", "humaneval_plus"], model="gemma4e4b")
    assert cfg["mode"] == "per-task"
    assert cfg["name"] == "tinycap-e4b-base-think_per-task"
    assert cfg["out_dir"].endswith("/thinking_per-task")     # its own store, never a whole arm's
    assert cfg["thinking_by_task"] == {
        "inspect/gsm8k_paired200": "on", "evalplus/humaneval_plus": "off"}
    # ONE standup, and it carries the parser because gsm8k thinks
    assert cfg["standup_extra"] == ["--reasoning-parser", "gemma4"]
    # ... and pools BOTH arms' sub-entries: the ON task's and the OFF task's
    assert set(cfg["geometry_keys"]) == {"capability:gen_long_thinking", "capability:gen_long"}
    # a battery whose per-task arms are ALL off needs no parser at all — and is
    # therefore servable on a family that could not serve the whole ON arm
    (off_only,) = L.plan_configurations(cells=["none"], modes=["per-task"],
                                        controls=["humaneval_plus"], model="qwen35_9b")
    assert off_only["standup_extra"] == [] and set(
        off_only["thinking_by_task"].values()) == {"off"}
    assert L.parser_needed("per-task", ["humaneval_plus"]) is False
    assert L.parser_needed("per-task", ["gsm8k"]) is True
    assert L.parser_needed("off", ["gsm8k"]) is False and L.parser_needed("on", ["humaneval_plus"]) is True
    # ... but a per-task battery WITH an ON task refuses on qwen, like --modes on
    with pytest.raises(L.ThinkingUnservable, match="asks for thinking-ON on"):
        L.plan_configurations(cells=["none"], modes=["per-task"], controls=["gsm8k"],
                              model="qwen35_9b")
    # a whole-arm mode still records the map (a store never has to be read to
    # find out which arm a task ran)
    (whole,) = L.plan_configurations(cells=["none"], modes=["off"], controls=["gsm8k"])
    assert whole["thinking_by_task"] == {"inspect/gsm8k_paired200": "off"}


def test_the_identity_records_thinking_by_task(L):
    (cfg,) = L.plan_configurations(cells=["none"], modes=["per-task"],
                                   controls=["gsm8k", "humaneval_plus"])
    geom, _ = L.served_geometry(cfg, 1)
    exp = L.expected_identity(cfg, geom, pass_no=1, controls=cfg["controls"], samples=4)
    # the key name is the one fixed with lane W20b-D (O7 item 4), sorted
    assert exp.as_dict()["thinking_by_task"] == dict(sorted(cfg["thinking_by_task"].items()))
    assert exp.served["reasoning_parser"] == "gemma4"        # one standup, parser composed
    assert exp.mode == "per-task"


def test_the_dry_run_says_whether_the_driver_takes_per_task(L, capsys, tmp_path):
    # native-serving-default: the launcher's default is the native engine now; this test pins the LENS
    # launch (a PSR cell / the historical composition), so it states the path explicitly
    """W20b-D is landing `--thinking per-task` concurrently; composing the flag
    onto a driver that rejects it would look fine here and die in the container
    after a standup was paid for, so the dry run SAYS which it is."""
    accepts = L.driver_accepts_per_task(REPO)
    assert accepts in (True, False)                 # readable either way on a real tree
    assert L.driver_accepts_per_task(tmp_path) is None      # no driver there at all
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, cells="none", modes="per-task", controls="gsm8k",
           dry_run=True)
    out = capsys.readouterr().out
    assert '"inspect/gsm8k_paired200": "on"' in out and "thinking_by_task" in out
    assert "--thinking per-task" in out
    assert ("accepts --thinking per-task" in out) if accepts else ("WARNING" in out)


def test_the_app_name_is_asked_for_and_never_silently_different(L, monkeypatch, capsys):
    """The Modal app is named when the App object is constructed,
    at import, so `--app-name` cannot change it — it STATES it, is recorded in
    the argv with the environment assignment that makes it work, and refuses
    when the two disagree (an unnamed run landing in a stale app is how one
    wave's output gets read as another's). The name is REQUIRED — an
    empty `--app-name` refuses instead of meaning "whatever this process
    resolved", and there is no default for the environment to fall back to."""
    assert L.APP_NAME == TEST_APP_NAME == L.app.name
    assert L.APP_NAME_ENV == C.MODAL_APP_NAME_ENV == "TINYCAP_APP_NAME"
    L.check_app_name(TEST_APP_NAME)                       # agrees
    for empty in ("", "   "):                             # not stated: refused, with the remedy
        with pytest.raises(SystemExit) as e:
            L.check_app_name(empty)
        assert "--app-name is required" in str(e.value) and "no default" in str(e.value)
        assert "TINYCAP_APP_NAME=<name> modal run" in str(e.value)
    with pytest.raises(SystemExit) as e:
        L.check_app_name("tinycap-gsm8k-rerun-e4b")
    msg = str(e.value)
    assert "TINYCAP_APP_NAME=tinycap-gsm8k-rerun-e4b modal run" in msg
    assert "constructed, at import" in msg
    # the recorded argv carries BOTH the env assignment and the flag — for EVERY
    # name, the historical default included (nothing is special about it now)
    argv = L.launch_argv("abc1234", app_name="tinycap-battery-qwen9b")
    assert argv[0] == "TINYCAP_APP_NAME=tinycap-battery-qwen9b"
    assert argv[1:3] == ["modal", "run"] and argv[argv.index("--app-name") + 1] == "tinycap-battery-qwen9b"
    assert L.launch_argv("abc1234", app_name=C.MODAL_APP_NAME)[0] == f"TINYCAP_APP_NAME={C.MODAL_APP_NAME}"
    with pytest.raises(SystemExit, match="--app-name"):
        L.main(sha="abc1234", pass1=True, app_name="tinycap-other", dry_run=True)
    with pytest.raises(SystemExit, match="--app-name is required"):
        L.main(sha="abc1234", pass1=True, dry_run=True)   # the entrypoint's empty default refuses
    # the dry run states the app it runs as
    L.main(app_name=TEST_APP_NAME, sha="abc1234", pass1=True, cells="none", modes="off",
           controls="gsm8k", dry_run=True)
    out = capsys.readouterr().out
    assert f"[tinycap] app: '{TEST_APP_NAME}' (from $TINYCAP_APP_NAME, stated by --app-name)" in out
    assert f"[tinycap] argv: TINYCAP_APP_NAME={TEST_APP_NAME} modal run " in out
    assert f"--app-name {TEST_APP_NAME}" in out
    # the contracts name the variable and document the kwarg
    assert "app_name" in C.LAUNCHER_ENTRYPOINT_KWARGS and "REQUIRED" in C.LAUNCHER_ENTRYPOINT_KWARGS["app_name"]


def test_an_unnamed_launch_refuses_at_import_and_a_container_import_does_not(monkeypatch):
    """E40 M7: with no TINYCAP_APP_NAME a `modal run` (a LOCAL import, per
    `modal.is_local`) refuses before Modal is contacted, naming the remedy;
    the container-side re-import (not local) has no such environment and must
    load, under a placeholder that never routes work; a test double without
    `is_local` is not a `modal run` and loads the same way."""
    stub = _stub_modal()                                   # is_local() -> True, like the real CLI
    with pytest.raises(SystemExit) as e:
        _load_launcher(stub, app_name=None, name="tinycap_served_unnamed")
    msg = str(e.value)
    assert "TINYCAP_APP_NAME is not set" in msg and "no default" in msg
    assert "TINYCAP_APP_NAME=<name> modal run scripts/modal/tinycap_served.py" in msg
    with pytest.raises(SystemExit, match="not set"):
        _load_launcher(stub, app_name="   ", name="tinycap_served_blank")   # whitespace is unset
    remote = _stub_modal()
    remote.is_local = lambda: False                        # inside a container
    mod = _load_launcher(remote, app_name=None, name="tinycap_served_remote")
    assert mod.APP_NAME == "" and mod.app.name == mod.REMOTE_IMPORT_APP_NAME == "tinycap-remote-import"
    no_probe = _stub_modal()
    del no_probe.is_local
    mod = _load_launcher(no_probe, app_name=None, name="tinycap_served_noprobe")
    assert mod.app.name == mod.REMOTE_IMPORT_APP_NAME
    # and a named import takes exactly the environment's name
    mod = _load_launcher(_stub_modal(), app_name="tinycap-smoke-e4b", name="tinycap_served_named")
    assert mod.APP_NAME == mod.app.name == "tinycap-smoke-e4b"


def test_gpu_container_trip_wire(L):
    # 8 since the campaign-wide re-run; --allow-extra is
    # still the only way past it, so 16 can never become an accidental default
    assert C.MAX_GPU_CONTAINERS == 8
    L.gpu_container_budget(8)
    with pytest.raises(SystemExit, match="MAX_GPU_CONTAINERS"):
        L.gpu_container_budget(9)
    L.gpu_container_budget(9, allow_extra=True)


# --------------------------------------------------------------------------- #
# main's exit code IS the RECs' verdict                                        #
# --------------------------------------------------------------------------- #


def test_failed_recs_and_the_failure_message(L):
    # bcb-bfcl-200 §S2b: PARTIAL_COMPLETE is the truncation ladder's complete state
    assert L.REC_OK_STATUSES == ("OK", "DONE", "PARTIAL_COMPLETE")
    ok = [{"name": "a", "status": "OK"}, {"name": "b", "status": "DONE"}]
    assert L.failed_recs(ok) == []
    bad = ok + [{"name": "c", "status": "STANDUP_FAILED", "error": "vllm died"},
                {"name": "d"},                                     # no status at all
                {"name": "e", "status": "INCOMPLETE",
                 "controls_still_remaining": ["humaneval_plus", "mbpp_plus"]},
                {"name": "f", "status": "INCOMPLETE",
                 "controls_still_remaining": ["gsm8k"]}]
    assert L.failed_recs(bad) == [("c", "STANDUP_FAILED"), ("d", "<missing>"),
                                  ("e", "INCOMPLETE"), ("f", "INCOMPLETE")]
    msg = L.rec_failure_message(bad)
    assert "4/6 container(s) did NOT finish" in msg
    assert "c: STANDUP_FAILED" in msg and "vllm died" in msg
    assert "d: <missing>" in msg
    # the EvalPlus-only INCOMPLETE is the EXPECTED two-stage handoff — named as
    # such, still non-zero (the cell is not complete until the sandbox lands)
    e_line = msg.split("e: INCOMPLETE")[1].split("\n")[0]
    assert "EXPECTED" in e_line and "two-stage handoff" in e_line and "sandbox" in e_line
    # a non-EvalPlus INCOMPLETE gets no such excuse
    assert "two-stage handoff" not in msg.split("f: INCOMPLETE")[1]
    assert "still remaining: ['gsm8k']" in msg


def test_main_exits_non_zero_when_a_rec_did_not_finish(L, monkeypatch, capsys):
    # native-serving-default: the launcher's default is the native engine now; this test pins the LENS
    # launch (a PSR cell / the historical composition), so it states the path explicitly
    """The 1983322 failure: the launcher printed ``0/1 container(s) OK`` and
    exited 0, so the wrapper wrote ``exit 0`` into the wave's .DONE sentinel
    and the watcher read a failed cell as a finished one."""
    recs = [{"name": "tinycap-e4b-psrkl_safety-think_off", "status": "STANDUP_FAILED",
             "error": "no endpoint.json"}]
    monkeypatch.setattr(L, "run_configuration", _spawn_only(lambda cfg: recs[0]))
    # (a prompted cell: a fitted one launches only behind --pf since E40 H1 — that
    # gate has its own test; this one is about the exit code)
    with pytest.raises(SystemExit) as e:
        L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, cells=C.CELL_PROMPTED_SAFETY, modes="off",
               controls="gsm8k")
    out = capsys.readouterr().out
    assert "[tinycap] 0/1 container(s) OK" in out          # the count still prints ...
    assert "STANDUP_FAILED" in out                          # ... and so does the whole REC
    assert e.value.code != 0                                # ... and the process fails
    assert "did NOT finish" in str(e.value.code)
    # every REC OK/DONE -> exit 0 (no SystemExit at all); one FunctionCall per
    # configuration (fan_out), so two cells are two spawns and two RECs
    monkeypatch.setattr(L, "run_configuration",
                        _spawn_only(lambda cfg: {"name": cfg["name"], "status": "OK" if cfg["cell"] == "none" else "DONE"}))
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, cells=f"none,{C.CELL_PROMPTED_SAFETY}", modes="off",
           controls="gsm8k")
    out = capsys.readouterr().out
    assert "[tinycap] 2/2 container(s) OK" in out and out.count("[tinycap] spawned ") == 2


# --------------------------------------------------------------------------- #
# geometry: pass 1 = the footprint from the modules; pass 2 = the measured entry #
# --------------------------------------------------------------------------- #


def test_pass1_geometry_is_the_composer_footprint_never_a_literal(L):
    S = sg.BUDGET_BY_MODEL[C.MODEL_SLUG].seqs_at(sg.PETRI_MAX_MODEL_LEN)
    for cfg in L.plan_configurations():
        g = L.configuration_geometry(cfg, 1)
        entry = sg.geometry_for(cfg["geometry_keys"][0])
        # the WINDOW is the composer's certified footprint; S is the model's budget at it
        assert (g.max_model_len, g.max_num_seqs) == (sg.PETRI_MAX_MODEL_LEN, S)
        assert g.measurement_pass and g.eval == cfg["geometry_keys"][0]
        # every default configuration's one key (gen_long / gen_long_thinking)
        # is MEASURED, so its pass 1 is a measurement RE-RUN at the footprint,
        # labelled MEASUREMENT_SOURCE; only a still-provisional key reads provisional
        assert g.provisional is entry.provisional and not entry.provisional
        assert g.window_source == sg.MEASUREMENT_SOURCE and g.is_identity and not g.over_budget
        # with no model the footprint's own S stands, as before the budgets
        bare = L.configuration_geometry(cfg, 1, model_key=None)
        assert (bare.max_model_len, bare.max_num_seqs) == (sg.PETRI_MAX_MODEL_LEN, vs.DEFAULT_MAX_NUM_SEQS)
        assert bare.seqs_source == sg.MEASUREMENT_SOURCE and bare.model_key is None
        if len(cfg["geometry_keys"]) == 1:
            # one MEASURED key: pass 2 resolves its window, S from the served model's
            # budget (gen_long: 4608 x 128 at gemma4e4b's)
            g2 = L.configuration_geometry(cfg, 2)
            assert (g2.max_model_len, g2.max_num_seqs) == (entry.max_model_len, 589_824 // entry.max_model_len)
            assert g2.window_source == "table" and g2.seqs_source == L.MODEL_BUDGET_SOURCE
            assert g2.is_identity and not g2.over_budget and not g2.measurement_pass and not g2.provisional
        else:
            # R5: several sub-entries -> one standup cannot serve them; refused
            # BEFORE any resolution (never geometry_keys[0] silently)
            with pytest.raises(L.Pass2Unpartitioned, match="sub-entries"):
                L.configuration_geometry(cfg, 2)
    # the default battery is all generation: one key per arm, so pass 2 resolves it
    assert {c["geometry_keys"] for c in L.plan_configurations()} == {
        ("capability:gen_long",), ("capability:gen_long_thinking",)}
    gl = L.configuration_geometry(L.plan_configurations(modes=["off"])[0], 2)
    assert (gl.max_model_len, gl.max_num_seqs) == (4608, 128)
    with pytest.raises(ValueError, match="pass must be"):
        L.configuration_geometry(L.plan_configurations()[0], 3)
    assert issubclass(L.Pass2Unpartitioned, ValueError)
    # a single-key --controls list under pass 2 resolves the MEASURED entry at its
    # window (PW11: agent 3,328 x 177 at gemma4e4b's budget) ...
    one = L.plan_configurations(cells=[C.CELL_PROMPTED_SAFETY], modes=["off"], controls=["bfcl_200"])[0]
    assert one["geometry_keys"] == ("capability:agent",)
    g2 = L.configuration_geometry(one, 2)
    assert (g2.max_model_len, g2.max_num_seqs, g2.seqs_source) == (3328, 589_824 // 3328, L.MODEL_BUDGET_SOURCE)
    assert g2.max_num_seqs == 177 and g2.is_identity
    # ... while the one key still PROVISIONAL (tiny_prompted: an analytic window, no
    # pooled long-SI cell) reaches the provisional refusal, naming why. No control
    # pools into it since the loglikelihood family was removed, so the configuration
    # names the key directly, as a recorded one would
    prov = dict(L.plan_configurations(cells=[C.CELL_PROMPTED_SAFETY], modes=["off"], controls=["gsm8k"])[0],
                geometry_keys=("capability:tiny_prompted",))
    with pytest.raises(sg.GeometryProvisional, match="ANALYTIC"):
        L.configuration_geometry(prov, 2)
    p1 = L.configuration_geometry(prov, 1)
    assert p1.provisional and p1.window_source == sg.PROVISIONAL_SOURCE and p1.max_model_len == sg.PETRI_MAX_MODEL_LEN
    src = LAUNCHER.read_text()
    assert not re.search(r'''--max-model-len["']\s*,\s*["']16384["']|max_model_len\s*=\s*16384\b''', src)
    assert "16384" not in src, "no literal Petri window anywhere in the launcher"
    assert "pass1_geometry" in src and "resolve_geometry" in src


def test_a_measured_entry_served_at_the_footprint_is_named_per_key(L, capsys):
    """``pass1_geometry`` serves a MEASURED tinycap entry at the certified
    measurement footprint so a pass-1 re-run can extend its ledger — a
    bigger window and a smaller S than the entry's own measurement needs,
    i.e. redundant GPU spend. The launcher NAMES it per geometry key in the
    run log and the REC: the key, the measured L x S ``resolve_geometry``
    would give it, the footprint L x S it is served at. A still-PROVISIONAL
    key says nothing (there is no measured window for the footprint to be
    redundant against), and neither does a pass 2 (the measured window IS
    what is served)."""
    S = sg.BUDGET_BY_MODEL[C.MODEL_SLUG].seqs_at(sg.PETRI_MAX_MODEL_LEN)
    measured = L.plan_configurations(cells=[C.CELL_PROMPTED_SAFETY], modes=["off"],
                                     controls=["bfcl_200"])[0]
    assert measured["geometry_keys"] == ("capability:agent",)
    g1 = L.configuration_geometry(measured, 1)
    [r] = L.measured_footprint_reserves(measured, g1, 1)
    assert r["key"] == "capability:agent" and r["window_source"] == sg.MEASUREMENT_SOURCE
    assert (r["measured_max_model_len"], r["measured_max_num_seqs"]) == (3328, 177)
    assert (r["served_max_model_len"], r["served_max_num_seqs"]) == (sg.PETRI_MAX_MODEL_LEN, S)
    capsys.readouterr()
    [line] = L.log_measured_footprint([r])
    assert line == capsys.readouterr().out.strip()
    for piece in ("capability:agent", "MEASURED entry re-served", "measured 3328 x 177",
                  f"served {sg.PETRI_MAX_MODEL_LEN} x {S}", "redundant GPU spend"):
        assert piece in line, piece
    # a PROVISIONAL key (named directly: no control pools into tiny_prompted any more):
    # nothing said, nothing recorded
    prov = dict(L.plan_configurations(cells=[C.CELL_PROMPTED_SAFETY], modes=["off"],
                                      controls=["gsm8k"])[0], geometry_keys=("capability:tiny_prompted",))
    assert L.measured_footprint_reserves(prov, L.configuration_geometry(prov, 1), 1) == []
    capsys.readouterr()
    assert L.log_measured_footprint([]) == [] and capsys.readouterr().out == ""
    # pass 2 serves the measured window itself: nothing redundant to name
    assert L.measured_footprint_reserves(measured, L.configuration_geometry(measured, 2), 2) == []
    # EVERY key of a multi-key configuration is checked, not just the one the window
    # is labelled with. Named explicitly: the DEFAULT battery's only key is gen_long,
    # and BFCL — whose capability:agent is measured — is no longer defaulted.
    multi = L.plan_configurations(cells=[C.CELL_PROMPTED_SAFETY], modes=["off"],
                                  controls=["bfcl_200", "gsm8k"])[0]
    got = L.measured_footprint_reserves(multi, L.configuration_geometry(multi, 1), 1)
    assert [r["key"] for r in got] == [k for k in multi["geometry_keys"]
                                       if not sg.geometry_for(k).provisional]
    assert len(got) > 1 and all(r["measured_max_model_len"] for r in got)
    # ...and the run records what it logged
    src = LAUNCHER.read_text()
    assert 'rec["measured_at_footprint"] = measured_footprint_reserves(cfg, geom, pass_no, tree)' in src
    assert 'log_measured_footprint(rec["measured_at_footprint"])' in src


# --------------------------------------------------------------------------- #
# E7 / E15: S comes from the served MODEL's measured budget (BUDGET_BY_MODEL)   #
# --------------------------------------------------------------------------- #


def _load_standup():
    """``scripts/capctrl_serving_up.py`` by path (a script, not a package
    module): its parser + resolver read a tinycap standup argv back the way the
    standup itself will."""
    path = REPO / "scripts" / "capctrl_serving_up.py"
    spec = importlib.util.spec_from_file_location("_tinycap_standup", path)
    mod = importlib.util.module_from_spec(spec)
    # registered BEFORE exec: the standup's Standup is a dataclass under
    # ``from __future__ import annotations``, and dataclasses resolves its
    # string annotations through sys.modules[cls.__module__] (PW6-7)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_launcher_carries_no_budget_table_of_its_own(L):
    """PI4's launcher-side E7 override is gone: the number lives in
    ``geometry.BUDGET_BY_MODEL`` (E15), keyed by the contracts' model slug, and
    the launcher's source label is the one the geometry module writes."""
    src = LAUNCHER.read_text()
    for gone in ("MEASURED_BUDGET_TOKENS", "MEASURED_BUDGET_BASIS", "apply_measured_budget",
                 "measured_budget_override", "measured_budget_tokens", "589_824"):
        assert gone not in src, gone
    assert C.MODEL_SLUG in sg.MODEL_KEYS and C.MODEL_SLUG in sg.BUDGET_BY_MODEL
    assert sg.BUDGET_BY_MODEL[C.MODEL_SLUG].batched_token_budget == 589_824      # E7, verbatim
    assert sg.BUDGET_BY_MODEL[C.MODEL_SLUG].batched_token_budget == round(1.5 * sg.BATCHED_TOKEN_BUDGET)
    assert L.MODEL_BUDGET_SOURCE == sg.resolve_geometry("capability", model_key=C.MODEL_SLUG).seqs_source
    assert L.MODEL_BUDGET_SOURCE == sg.pass1_geometry("capability:tiny", model_key=C.MODEL_SLUG).seqs_source


def test_s_is_derived_from_the_model_budget_and_the_pass1_window(L):
    budget = sg.BUDGET_BY_MODEL[C.MODEL_SLUG]
    for cfg in L.plan_configurations():
        g, rec = L.served_geometry(cfg, 1)
        # S = the model's budget at the footprint window, through seqs_at; the window untouched
        assert g.max_model_len == sg.PETRI_MAX_MODEL_LEN
        assert g.max_num_seqs == budget.seqs_at(sg.PETRI_MAX_MODEL_LEN) == 589_824 // sg.PETRI_MAX_MODEL_LEN == 36
        assert g.max_num_seqs == sg.identity_seqs(g.max_model_len, model_key=C.MODEL_SLUG)
        assert g.max_num_seqs != vs.DEFAULT_MAX_NUM_SEQS                  # not the Llama-budget S
        assert g.seqs_source == L.MODEL_BUDGET_SOURCE
        assert g.window_source in (sg.PROVISIONAL_SOURCE, sg.MEASUREMENT_SOURCE)     # a pass-1 footprint either way
        assert g.model_key == C.MODEL_SLUG and g.budget.measured and g.budget_tokens == 589_824
        assert g.eval == cfg["geometry_keys"][0] and g.measurement_pass
        # honest against the MODEL's budget: within it and identity — no over-budget flag needed
        assert g.batched_tokens == 589_824 and not g.over_budget and g.is_identity
        # the record says what sized S, and what the certified footprint would have served
        assert rec["model_key"] == C.MODEL_SLUG and rec["batched_token_budget"] == 589_824
        assert rec["budget_source"] == sg.MEASURED_BUDGET_SOURCE and rec["certified"] is False
        assert (rec["max_num_seqs"], rec["footprint_max_num_seqs"]) == (36, vs.DEFAULT_MAX_NUM_SEQS)
        assert rec["max_model_len"] == sg.PETRI_MAX_MODEL_LEN and rec["batched_tokens"] == 589_824
        assert rec["certified_budget_tokens"] == sg.BATCHED_TOKEN_BUDGET
        assert rec["ratio_to_certified"] == 1.5 and rec["allow_over_budget"] is False
        assert rec["identity"] is True and rec["over_budget"] is False
        assert rec["basis"] == L.dataclasses.asdict(budget.basis)
        assert "--phase budget" in rec["basis"]["method"] and rec["basis"]["gpu_memory_utilization"] == 0.9
        # served_geometry IS configuration_geometry + the record: one resolver everywhere
        assert g == L.configuration_geometry(cfg, 1)
    src = LAUNCHER.read_text()
    assert "served_geometry(cfg, pass_no, tree)" in src           # _run_configuration
    assert "served_geometry(cfg, 1, tree)" in src                 # preflight
    assert "served_geometry(cfg, pass_no)" in src                 # describe_plan
    # both passes size from the key — and so does the MATH-only pass-1 resolution
    # ... and the IFEval entry's resolution for either pass:
    # the fourth `resolve_geometry(..., model_key=model_key)` in configuration_geometry
    assert src.count("model_key=model_key)") == 4
    assert not re.search(r"\b36\b", src), "S is derived from the budget, never typed"


def test_an_unmeasured_or_unknown_model_refuses_never_the_llama_s(L, monkeypatch):
    cfg = L.plan_configurations()[0]
    # a key outside MODEL_KEYS refuses in the geometry module (never a silent default)
    with pytest.raises(sg.UnknownModelKey):
        L.configuration_geometry(cfg, 1, model_key="gemma4-e4b")
    # an S from anywhere but the model budget is refused by the record
    other = L.dataclasses.replace(L.configuration_geometry(cfg, 1), seqs_source="override")
    with pytest.raises(L.UnmeasuredBudget, match="not MEASURED"):
        L.model_budget_record(other)
    with pytest.raises(L.UnmeasuredBudget):
        L.model_budget_record(L.configuration_geometry(cfg, 1, model_key=None))   # no budget at all
    # a KNOWN key whose budget is not measured: the geometry module falls to the
    # Llama-certified default (with a warning); the launcher refuses to serve that S
    monkeypatch.setitem(sg.BUDGET_BY_MODEL, C.MODEL_SLUG, None)
    g = L.configuration_geometry(cfg, 1)
    assert g.max_num_seqs == vs.DEFAULT_MAX_NUM_SEQS and g.budget_source == sg.DEFAULT_BUDGET_SOURCE
    assert not g.budget.measured
    with pytest.raises(L.UnmeasuredBudget, match="Llama-certified default"):
        L.model_budget_record(g)
    with pytest.raises(L.UnmeasuredBudget):
        L.served_geometry(cfg, 1)


def test_a_provisional_ceiling_is_served_and_the_record_stamps_it(L):
    """The 2026-09-06 ceilings (qwen35_9b and llama3b today — engine-init failures on
    the lens stack; llama8b's was replaced by the lane's standup MEASUREMENT and
    qwen35_4b's lifted, both 2026-09-11) are the MODEL's OWN row, so the launcher
    serves them: the gate exists to keep out the Llama-certified DEFAULT, which is
    LARGER than a ceiling, not to keep out a smaller number the model itself supplies.
    What the record must do is say which kind of number sized S."""
    # the example is qwen35_9b now: llama8b's row became MEASURED (475,136, from its standups, 2026-09-11) and no
    # longer exercises the provisional path — the numbers are spelled from the row, never typed
    row = sg.BUDGET_BY_MODEL["qwen35_9b"]
    assert row.source == sg.PROVISIONAL_BUDGET_SOURCE
    cfg = L.plan_configurations()[0]
    g = L.configuration_geometry(cfg, 1, model_key="qwen35_9b")
    assert g.max_num_seqs == row.batched_token_budget // sg.PETRI_MAX_MODEL_LEN == 12
    assert g.budget_tokens == row.batched_token_budget and g.seqs_source == L.MODEL_BUDGET_SOURCE
    assert g.budget.per_model and g.budget.provisional and not g.budget.measured
    rec = L.model_budget_record(g)
    assert rec["budget_source"] == sg.PROVISIONAL_BUDGET_SOURCE and rec["provisional"] is True
    assert rec["model_key"] == "qwen35_9b" and rec["batched_token_budget"] == row.batched_token_budget
    assert rec["max_num_seqs"] == row.batched_token_budget // sg.PETRI_MAX_MODEL_LEN == 12 and rec["identity"] is True
    # ... llama8b's row is the standup MEASUREMENT now (475,136 = 29 x 16384, the shared table's row): stamped measured
    lrow = sg.BUDGET_BY_MODEL["llama8b"]
    assert lrow.source == sg.MEASURED_BUDGET_SOURCE
    g_l = L.configuration_geometry(cfg, 1, model_key="llama8b")
    assert g_l.max_num_seqs == lrow.batched_token_budget // sg.PETRI_MAX_MODEL_LEN == 29 and g_l.budget_tokens == lrow.batched_token_budget == 475_136
    assert g_l.budget.per_model and g_l.budget.measured and not g_l.budget.provisional
    rec_l = L.model_budget_record(g_l)
    assert rec_l["budget_source"] == sg.MEASURED_BUDGET_SOURCE and rec_l["provisional"] is False and rec_l["max_num_seqs"] == 29
    # ... and a MEASURED row still stamps measured, so the two are told apart in the store
    rec_m = L.model_budget_record(L.configuration_geometry(cfg, 1))
    assert rec_m["budget_source"] == sg.MEASURED_BUDGET_SOURCE and rec_m["provisional"] is False


def test_standup_argv_carries_the_dims_and_the_model_key_never_over_budget(L, tmp_path):
    cfg = L.plan_configurations(cells=[C.CELL_PSRKL_SAFETY], modes=["on"])[0]
    served, _ = L.served_geometry(cfg, 1)
    common = ("/tree", "/tree/cfg.yaml", cfg["cell"], "/vol/objects/x", tmp_path, tmp_path / "STOP")
    argv = L._standup_cmd(*common, served, 64, cfg["standup_extra"])
    # explicit dims (a provisional entry cannot be re-derived by the standup from --eval) ...
    assert argv[argv.index("--max-model-len") + 1] == str(sg.PETRI_MAX_MODEL_LEN)
    i = argv.index("--max-num-seqs")
    # ... plus the model key, so the standup's guard + endpoint hold to the same budget
    assert argv[i:i + 4] == ["--max-num-seqs", "36", "--model-key", C.MODEL_SLUG]
    assert "--eval" not in argv and argv.count("--model-key") == 1
    # no over-budget flag: 36 x the Petri window IS the model's budget and the identity holds
    assert "--allow-over-budget" not in argv
    # the extras still trail, and the util matches the one the budgets were measured at
    assert argv[-2:] == ["--reasoning-parser", "gemma4"]
    assert argv[argv.index("--gpu-mem-util") + 1] == "0.90"
    # a geometry sized for no model passes no key
    bare = L.configuration_geometry(cfg, 1, model_key=None)
    assert "--model-key" not in L._standup_cmd(*common, bare, 64, [])
    # the standup reads that argv back to the SAME geometry: the model's budget, identity, in budget
    up = _load_standup()
    ns = up.build_parser().parse_args(argv[2:])
    assert ns.reasoning_parser == "gemma4" and ns.model_key == C.MODEL_SLUG and not ns.allow_over_budget
    r = up.resolve_from_args(ns)
    assert (r.max_model_len, r.max_num_seqs, r.model_key) == (served.max_model_len, served.max_num_seqs, C.MODEL_SLUG)
    assert r.budget_tokens == served.budget_tokens == 589_824 and r.is_identity and not r.over_budget
    assert up.standup_model_key_note(ns.eval, ns.model_key) is None


def test_driver_concurrency_equals_the_served_s(L):
    for cfg in L.plan_configurations():
        geom, _ = L.served_geometry(cfg, 1)
        ep = {"proxy_url": "http://127.0.0.1:1/v1", "routing_table": "/srv/rt.json"}
        drv = L._driver_cmd("/tree", cfg, ep, "http://127.0.0.1:2/v1", geom,
                            controls=cfg["controls"], limit=0, inspect_limit=0, samples=4,
                            cfg_path="/tree/cfg.yaml", vllm_log=None)
        standup = L._standup_cmd("/tree", "/tree/cfg.yaml", cfg["cell"], "d", Path("/srv"),
                                 Path("/srv/STOP"), geom, 64, cfg["standup_extra"])
        served_s = standup[standup.index("--max-num-seqs") + 1]
        assert drv[drv.index("--concurrency") + 1] == served_s == "36"


def test_run_configuration_persists_the_served_geometry_record(L):
    src = LAUNCHER.read_text()
    assert L.SERVED_GEOMETRY_FILENAME == "served_geometry.json"
    assert "out_dir / SERVED_GEOMETRY_FILENAME" in src
    assert '"model_budget": model_budget' in src
    assert '"endpoint": endpoint' in src and '"standup_argv": standup_argv' in src


# --------------------------------------------------------------------------- #
# argv builders                                                                #
# --------------------------------------------------------------------------- #


def test_standup_argv_carries_the_resolved_geometry_and_the_extras(L, tmp_path):
    cfg = L.plan_configurations(cells=[C.CELL_PROMPTED_SAFETY], modes=["on"])[0]
    geom = L.configuration_geometry(cfg, 1)
    argv = L._standup_cmd("/tree", "/tree/cfg.yaml", cfg["cell"], "/vol/objects/x", tmp_path,
                          tmp_path / "STOP", geom, 64, cfg["standup_extra"])
    assert argv[1] == "/tree/scripts/capctrl_serving_up.py"
    pairs = dict(zip(argv[2::2], argv[3::2]))
    assert pairs["--max-model-len"] == str(sg.PETRI_MAX_MODEL_LEN)
    # S is the served MODEL's budget at that window, and the key travels with it
    S = sg.BUDGET_BY_MODEL[C.MODEL_SLUG].seqs_at(sg.PETRI_MAX_MODEL_LEN)
    assert pairs["--max-num-seqs"] == str(geom.max_num_seqs) == str(S) != str(vs.DEFAULT_MAX_NUM_SEQS)
    assert pairs["--model-key"] == C.MODEL_SLUG
    assert pairs["--max-logprobs"] == "64" and pairs["--cell"] == "prompted__safety_refusal"
    # U9 item 6: the default battery has no BFCL, so the ON arm's only extra is the parser
    assert argv[-2:] == ["--reasoning-parser", "gemma4"]
    assert "--tool-calls" not in argv and "--manifest" not in argv
    assert "--eval" not in argv          # a provisional entry cannot be named to the standup
    fitted = L.plan_configurations(cells=[C.CELL_PSRKL_SAFETY], modes=["off"])[0]
    argv = L._standup_cmd("/tree", "c", fitted["cell"], "d", tmp_path, tmp_path / "STOP",
                          L.configuration_geometry(fitted, 1), 64, fitted["standup_extra"])
    assert "--tool-calls" not in argv and "--reasoning-parser" not in argv
    # ...and an explicit --controls bfcl still composes the tool-call extras, on
    # either cell kind (U4(b) / PW18 unchanged — only the DEFAULT moved)
    bf = L.plan_configurations(cells=[C.CELL_PSRKL_SAFETY], modes=["on"], controls=["bfcl_200"])[0]
    argv = L._standup_cmd("/tree", "c", bf["cell"], "d", tmp_path, tmp_path / "STOP",
                          L.configuration_geometry(bf, 1), 64, bf["standup_extra"])
    assert argv[-5:] == ["--tool-calls", "--manifest", C.PETRI_MANIFEST_GEMMA4E4B,
                         "--reasoning-parser", "gemma4"]


def test_driver_argv_is_the_contracts_builder_with_ledger_and_resolved_s(L):
    cfg = L.plan_configurations(cells=[C.CELL_PSRKL_SAFETY], modes=["on"])[0]
    geom = L.configuration_geometry(cfg, 1)
    ep = {"proxy_url": "http://127.0.0.1:1/v1", "routing_table": "/srv/rt.json"}
    cmd = L._driver_cmd("/tree", cfg, ep, "http://127.0.0.1:2/v1", geom, controls=cfg["controls"],
                        limit=5, inspect_limit=5, samples=4, cfg_path="/tree/cfg.yaml",
                        vllm_log="/srv/vllm.log")
    expect = C.driver_argv(
        python=L.CLIENT_PY, script="/tree/scripts/served_capability_run.py",
        proxy_url=ep["proxy_url"], routing_table=ep["routing_table"], cell=cfg["cell"],
        config="/tree/cfg.yaml", defenses_dir=L.OBJECTS_DIR, out_root=cfg["out_dir"],
        controls=cfg["controls"], thinking="on", concurrency=geom.max_num_seqs,
        judge_model=L.JUDGE_MODEL, limit=5, samples=4, inspect_limit=5,
        ledger_url="http://127.0.0.1:2/v1", top_logprobs=L.TOP_LOGPROBS, vllm_log="/srv/vllm.log")
    assert cmd == expect
    pairs = dict(zip(cmd[3::2], cmd[4::2]))
    assert pairs["--thinking"] == "on" and pairs["--ledger-url"] == "http://127.0.0.1:2/v1"
    # the client is driven at the SERVED S: the model's budget at the window, not the footprint's
    assert pairs["--concurrency"] == str(geom.max_num_seqs) == str(
        sg.BUDGET_BY_MODEL[C.MODEL_SLUG].seqs_at(sg.PETRI_MAX_MODEL_LEN)) != str(vs.DEFAULT_MAX_NUM_SEQS)
    assert pairs["--controls"] == ",".join(L.CONTROLS_FITTED)
    assert pairs["--out-root"] == "/vol/tinycap/gen/gemma4e4b/safety_refusal__psr_kl_all/thinking_on"
    assert cmd[0] == "/opt/client/bin/python"     # the client venv runs the driver
    assert L.OBJECTS_DIR == "/vol/objects/scale-12/gemma4e4b"
    # no legacy judged control in either task list -> no Anthropic traffic (P4)
    assert not ({"xstest", "or_bench_hard_1k"} & set(L.CONTROLS_PROMPTED + L.CONTROLS_FITTED))


def test_ledger_argv_runs_the_module_in_the_client_venv_on_loopback(L):
    argv = L._ledger_cmd(L.CLIENT_PY, "http://127.0.0.1:1/v1", 4242, "/out/usage_ledger.jsonl",
                         "/srv/ledger_endpoint.json")
    assert argv[:3] == ["/opt/client/bin/python", "-m", "sisteer.capeval.serving.usage_ledger"]
    pairs = dict(zip(argv[3::2], argv[4::2]))
    assert pairs["--upstream"] == "http://127.0.0.1:1/v1" and pairs["--port"] == "4242"
    assert pairs["--host"] == "127.0.0.1" and pairs["--ledger-path"].endswith(C.LEDGER_FILENAME)
    assert pairs["--endpoint-json"] == "/srv/ledger_endpoint.json"
    # the ledger file lives at <out_root>/usage_ledger.jsonl (the manifest's usage_ledger)
    src = LAUNCHER.read_text()
    assert "out_dir / C.LEDGER_FILENAME" in src
    assert f'{C.INSPECT_BASE_URL_ENV}: ledger_url' in src or "INSPECT_BASE_URL_ENV: ledger_url" in src


# --------------------------------------------------------------------------- #
# resume via done markers — held to the FULL expected identity (E12 item 1)    #
# --------------------------------------------------------------------------- #


IHASH = "a1b2c3d4" + "0" * 8
#: a synthetic full commit for the producer block / TREE_SHA stamps
FULL_SHA = "f6cf462" + "0" * 33
OTHER_SHA = "abcdef1" + "1" * 33


def _expected(L, *, cell=C.CELL_PSRKL_SAFETY, mode="off", pass_no=1, controls=None,
              samples=C.MATH500_SAMPLES_PER_ITEM,      # the launcher's own default (1 since 2026-09-06)
              limit=0, inspect_limit=0, tree=None, tree_sha=FULL_SHA,
              reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT, model=C.MODEL_SLUG,
              allow_out_of_support_thinking=False, **served_overrides):
    """The identity a configuration resolves BEFORE reading any marker — the
    launcher's own derivation (the pass-1 footprint window, the E7 S, the
    parser / tool-call composition from the REQUESTED controls, the staged
    tree's TREE_SHA stamp), with optional served-field overrides for the
    negative cases. ``tree_sha`` defaults to the sha ``_manifest`` stamps
    into every store's producer block, so a positive case matches."""
    cfg = L.plan_configurations(cells=[cell], modes=[mode], controls=controls, model=model,
                                allow_out_of_support_thinking=allow_out_of_support_thinking)[0]
    geom, _ = L.served_geometry(cfg, 1)          # pass 2 cannot resolve while the entries are provisional
    exp = L.expected_identity(cfg, geom, pass_no=pass_no, controls=cfg["controls"], samples=samples,
                              limit=limit, inspect_limit=inspect_limit, tree=tree,
                              tree_sha=tree_sha, reduction_mode=reduction_mode)
    if served_overrides:
        exp = exp._replace(served={**exp.served, **served_overrides})
    return exp


_DEFAULT = object()


def _manifest(expected, *, ihash=IHASH, tasks=None, served=None, limits=None, samples=None,
              mode=None, frameworks=None, producer=_DEFAULT, drop_identity_field=None,
              sampling=None, sampling_mode=None, or_bench_item_set=_DEFAULT,
              max_tokens_by_task=_DEFAULT, serve_out_of_support=_DEFAULT,
              route_declarations=_DEFAULT, math500_max_tokens=_DEFAULT, extra_identity=None):
    """A tinycap cell manifest the way the served driver leaves it: the
    HASHED ``resolved_config`` with its ``tinycap`` block, the mirrored
    top-level ``tinycap`` / ``frameworks`` blocks, the ``thinking`` /
    ``geometry`` / ``concurrency`` / ``route`` blocks agreeing with the
    served identity, the ``provenance`` and ``producer`` blocks. Every
    override is applied BEFORE hashing, so the store stays self-consistent
    and differs from ``expected`` in exactly the field under test."""
    served = dict(expected.served) if served is None else dict(served)
    mode = expected.mode if mode is None else mode
    fw = {k: None for k in C.FRAMEWORKS_BLOCK_FIELDS}
    fw.update({"inspect_ai": C.FRAMEWORK_PINS["inspect_ai"],
               "inspect_evals": C.FRAMEWORK_PINS["inspect_evals"],
               "client_lock_sha256": expected.client_lock_sha256})
    fw.update(frameworks or {})
    atoms = sorted(tasks if tasks is not None else C.expand_control(C.CONTROL_GSM8K))
    by_task = C.run_thinking_by_task(atoms, mode)
    tinycap = {
        "tasks": atoms,
        "thinking": {"mode": mode, "render": "flag_only"},
        "versions": {"tinycap": C.TINYCAP_VERSION},
        "served": served,
        "limits": dict(expected.limits) if limits is None else dict(limits),
        "samples": expected.samples if samples is None else samples,
        C.MANIFEST_KEY_FRAMEWORKS: fw,
        # PW8 (E16 A2): the pass is hashed; the geometry block below copies it
        C.GEOMETRY_PASS_FIELD: expected.pass_no,
        # PW20b-D (U5-3 / U5-4): the driver also hashes the per-task arm map and
        # the MATH-500 truncation-retry cap, so a manifest without them is not a
        # current one (`untrusted_store_reason` -> `_exact_keys`). Mechanical
        # addition by lane W20b-D so this helper keeps building what the driver
        # writes; the launcher itself needs no change — `expected_identity`
        # derives `served` only, and every other field is read off the contract
        # tuple rather than a typed list.
        C.IDENTITY_KEY_THINKING_BY_TASK: by_task,
        # ... the retry field through the driver's own rule: the legacy literal
        # on every run that files no MATH-500 row
        C.IDENTITY_KEY_MATH500_RETRY_MAX_TOKENS: C.math500_retry_identity_value(
            by_task, C.MATH500_RETRY_MAX_TOKENS),
        # PW-battery (U10 item 4): the OR-Bench item set the driver hashes —
        # the expected identity's unless a negative case overrides it
        C.IDENTITY_KEY_OR_BENCH_ITEM_SET: (expected.or_bench_item_set
                                           if or_bench_item_set is _DEFAULT
                                           else or_bench_item_set),
        # PW-sampling (U8): the five hashed sampling fields — the expected
        # identity's unless a negative case overrides them
        **(dict(expected.sampling) if sampling is None else dict(sampling)),
    }
    # the OPTIONAL identity keys the driver stamps only under
    # --allow-out-of-support-thinking — EXACTLY the keys
    # contracts.tinycap_identity_fields(...) returns beyond the base, DERIVED
    # (never listed): the driver widens that derivation with
    # `max_tokens_by_task` and `serve_out_of_support` in the change that stamps
    # them, and this helper must build what the driver writes in BOTH states.
    # Each is the expectation's value unless a negative case overrides it; an
    # in-support manifest stays byte-identical to the one this helper built before
    optional = {
        C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING: True,
        C.IDENTITY_KEY_MAX_TOKENS_BY_TASK: (dict(expected.max_tokens_by_task)
                                            if max_tokens_by_task is _DEFAULT else max_tokens_by_task),
        C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT: (expected.serve_out_of_support
                                              if serve_out_of_support is _DEFAULT else serve_out_of_support),
        # the MATH-500 first-pass cap, stamped by the driver only when the
        # manifest's own tasks file inspect/math — the expectation's value
        # (16,384) unless a negative case records another cap
        C.IDENTITY_KEY_MATH500_MAX_TOKENS: (expected.math500_max_tokens
                                            if math500_max_tokens is _DEFAULT else math500_max_tokens),
    }
    # ifeval-canonical-battery: the sampling block spread above carries the row's OPTIONAL
    # terms whenever the row sets them (qwen35_4b's vendor_preset row: min_p / presence_penalty /
    # repetition_penalty) — the driver hashes them at the top of the block, and its own
    # exact-keys derivation is told so (`sampling_optional`); every card / greedy row sets
    # none, so the legacy shapes this helper built are byte-identical
    sampling_block = dict(expected.sampling) if sampling is None else dict(sampling)
    fields = C.tinycap_identity_fields(out_of_support_thinking=expected.out_of_support_thinking,
                                       off_cap_override=getattr(expected, "off_cap_override", False),
                                       files_math500=C.files_math500(by_task),
                                       sampling_optional=tuple(f for f in C.SAMPLING_OPTIONAL_IDENTITY_FIELDS
                                                               if f in sampling_block))
    tinycap.update({f: optional[f] for f in fields if f not in C.TINYCAP_IDENTITY_FIELDS and f in optional})
    assert set(tinycap) == set(fields)
    # bcb-bfcl-200: the keys a driver stamps BEYOND this helper's derivation —
    # the pinned legs' item sets / sandbox facts and the truncation ladder's
    # tuple (an ``ExpectedIdentity.optional_identity()``), merged AFTER the
    # exact-set assertion above so the helper's own states stay byte-identical
    if extra_identity:
        tinycap.update(dict(extra_identity))
    sampling_mode = expected.sampling_mode if sampling_mode is None else sampling_mode
    if drop_identity_field:
        del tinycap[drop_identity_field]
    resolved = {"driver": "served_capability_run@test", "engine": "vllm",
                "cell_id": expected.cell, "controls": ["mmlu", "or_bench_hard_1k", "xstest"],
                "scale": 1.0, "limit": None, "tinycap": tinycap}
    chash = cs.harness_config_hash(resolved)
    tc_served = served if isinstance(served, dict) else {}
    manifest = {
        "cell_id": expected.cell, "intervention_hash": ihash, "harness_config_hash": chash,
        "engine": "vllm", "resolved_config": resolved,
        C.MANIFEST_KEY_THINKING: {"enable_thinking": mode == "on", "render": "flag_only",
                                  "reasoning_parser": tc_served.get("reasoning_parser")},
        C.MANIFEST_KEY_GEOMETRY: {"eval": "capability:tiny",
                                  "max_model_len": tc_served.get("max_model_len"),
                                  "max_num_seqs": tc_served.get("max_num_seqs"),
                                  C.GEOMETRY_PASS_FIELD: expected.pass_no,
                                  "source": "endpoint.json+served_geometry.json"},
        C.MANIFEST_KEY_CONCURRENCY: tc_served.get("concurrency"),
        C.MANIFEST_KEY_ROUTE: {"virtual_model": "petri__x", "grader_virtual_model": "petri__base",
                               "serving_identity": "lens", "tool_calls": tc_served.get("tool_calls"),
                               C.AGENT_LOOP_ENTRY_KEY: tc_served.get(C.AGENT_LOOP_ENTRY_KEY),
                               C.REDUCTION_MODE_FIELD: tc_served.get(C.REDUCTION_MODE_FIELD),
                               # fit-support transfer (contract §2.5.1 <ROUTE_KEY>): the driver
                               # copies the RAW attested map on an out-of-support store — this
                               # cell and its twin on a declared standup, {} otherwise
                               **({C.ROUTE_KEY_SERVE_DECLARATIONS: (
                                   _route_declarations(expected) if route_declarations is _DEFAULT
                                   else route_declarations)}
                                  if expected.out_of_support_thinking else {})},
        C.MANIFEST_KEY_FRAMEWORKS: dict(fw),
        C.MANIFEST_KEY_TINYCAP: copy.deepcopy(tinycap),
        C.MANIFEST_KEY_PROVENANCE: {"cell_id": expected.cell},
        # the sampling block MIRRORS the hashed fields, plus the mode, the key and the source
        C.MANIFEST_KEY_SAMPLING: {"mode": sampling_mode, "model_key": C.MODELS[C.MODEL_SLUG].model_key,
                                  **{f: copy.deepcopy(tinycap.get(f)) for f in C.SAMPLING_IDENTITY_FIELDS},
                                  # the producer's shape (served_capability_run._sampling_block): each optional term
                                  # the row SETS rides the block too (the wave shakedown, 2026-09-13)
                                  **{f: copy.deepcopy(tinycap[f]) for f in C.SAMPLING_OPTIONAL_IDENTITY_FIELDS
                                     if tinycap.get(f) is not None},
                                  "source": "test"},
    }
    if producer is _DEFAULT:
        manifest[C.MANIFEST_KEY_PRODUCER] = {"tree_sha": FULL_SHA, "driver": C.PRODUCER_DRIVER_SERVED}
    elif producer is not None:
        manifest[C.MANIFEST_KEY_PRODUCER] = dict(producer)
    # P1-1 (2026-09-07): the driver dates every store (contract_version) and
    # states its served==trained template-check class with its immutable fit
    # witness and refreshable fit_provenance block; the launcher's trust check
    # reads them through the one reader (sweep.store_provenance_problems), so
    # the fixture builds what the driver writes — a "compared" store
    from sisteer.eval import sweep as _sweep
    manifest[C.MANIFEST_KEY_CONTRACT_VERSION] = C.CONTRACT_VERSION
    manifest[_sweep.FIT_PROVENANCE_TEMPLATE_CHECK_KEY] = "compared"
    witness = {**{f: None for f in _sweep.FIT_WITNESS_FIELDS},
               "model_id": "synthetic-org/tiny-it", "revision": "2" * 40,
               "template_hash": "12" * 8, "status": "verified", "source": "train_budget_meta",
               "fit_kind": _sweep.FIT_KIND_SI_PAIR, "positions_at_fit": "response",
               "object_sha": "ab" * 8}
    manifest[_sweep.MANIFEST_KEY_FIT_WITNESS] = dict(witness)
    manifest[_sweep.MANIFEST_KEY_FIT_PROVENANCE] = {
        **witness, "served_model_id": "synthetic-org/tiny-it", "served_revision": "2" * 40,
        "served_template_hash": "12" * 8, "transfer": False, "template_check": "compared",
        "template_check_reason": _sweep.TEMPLATE_CHECK_REASON["compared"]}
    return manifest, chash


def _route_declarations(expected) -> dict:
    """What a driver copies into ``route.lens_serve_declarations`` for a store
    of this expectation: on a declared standup the cell's attestation key AND
    the twin, each ``{"serve_out_of_support": ["thinking"]}`` (contract §2.5.7);
    ``{}`` on an undeclared one (the base route's ON leg)."""
    if expected.serve_out_of_support is None:
        return {}
    key = C.cell_spellings(expected.cell, declared=True).harness
    value = {C.SERVE_OUT_OF_SUPPORT_KEY: list(expected.serve_out_of_support)}
    return {key: value, C.TWIN_CELL_ID: dict(value)}


def _store(out: Path, expected, *, key: str | None = None, mutate=None, **manifest_kw) -> Path:
    """A tinycap cell store under ``out/cells/`` in the frozen key layout
    (the manifest's hash prefixes in the directory name); ``mutate`` edits
    the manifest AFTER hashing (an inconsistent, hand-edited store)."""
    manifest, chash = _manifest(expected, **manifest_kw)
    ihash = manifest["intervention_hash"]
    key = key or f"{expected.cell.replace('__', '--')}__controls__t0__a0__{ihash[:8]}__{chash[:8]}"
    d = out / "cells" / key
    d.mkdir(parents=True, exist_ok=True)
    if mutate is not None:
        mutate(manifest)
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True, default=str))
    return d


def _finish(cell_dir: Path, task: str, n: int, *, summary_n: int | None = None,
            fields=None) -> None:
    """A done marker (+ its summary) as ``completion.finish_task`` writes them."""
    name = C.marker_filename(task)
    payload = {"task": task, "n_items": n, "n_rows": 2 * n, "n_requests": n, "n_rejected": 0,
               "wall_clock_s": 1.0, "finished_at": 0.0}
    if fields is not None:
        payload = fields
    (cell_dir / C.DONE_DIR).mkdir(exist_ok=True)
    (cell_dir / C.DONE_DIR / name).write_text(json.dumps(payload))
    (cell_dir / C.SUMMARY_DIR).mkdir(exist_ok=True)
    (cell_dir / C.SUMMARY_DIR / name).write_text(
        json.dumps({"acc": 0.5, "n": n if summary_n is None else summary_n}))


def _tree(repos: Path, sha: str, full: str = FULL_SHA, *, stamp: bool = True) -> Path:
    """A fake staged tree ``<repos>/sisteer-<sha>`` with (or without) its
    TREE_SHA stamp."""
    t = repos / f"sisteer-{sha}"
    (t / "src" / "sisteer").mkdir(parents=True, exist_ok=True)
    if stamp:
        (t / C.TREE_SHA_FILENAME).write_text(full + "\n")
    return t


def test_expected_identity_is_derived_from_the_resolved_inputs(L, tmp_path, monkeypatch):
    """E12 item 1: the identity a store is held to is RESOLVED, never read
    off a store — every ``SERVED_IDENTITY_FIELDS`` name has a derivation
    (the parser from the standup argv the launcher composes, ``tool_calls``
    from the REQUESTED controls, window and S from the resolved geometry,
    ``concurrency`` = the served S the driver is driven at), plus the pass,
    the avg@k width, the active limits and the staged tree's lock sha; a
    contract field without a derivation refuses loudly."""
    for cfg in L.plan_configurations():
        geom, _ = L.served_geometry(cfg, 1)
        exp = L.expected_identity(cfg, geom, pass_no=1, controls=cfg["controls"], samples=4)
        assert tuple(exp.served) == C.SERVED_IDENTITY_FIELDS
        assert exp.served["max_model_len"] == geom.max_model_len == sg.PETRI_MAX_MODEL_LEN
        assert exp.served["max_num_seqs"] == exp.served["concurrency"] == geom.max_num_seqs
        # U9 item 6: no BFCL in either battery, so no default standup serves tool calls
        assert exp.served["tool_calls"] is cfg["tool_calls"] is False
        assert exp.served["reasoning_parser"] == L.reasoning_parser_of(cfg["standup_extra"]) == (
            C.REASONING_PARSER_BY_FAMILY["gemma4"] if cfg["mode"] == "on" else None)
        # PW18: the proxy's agent-loop entry is ON (the argv never opts out);
        # PW18b (E29): with no --reduction-mode the identity carries what the
        # parser makes of THIS environment (stacked when unset, since E46 — but
        # the assertion reads the parser rather than a literal, so it states the
        # derivation and not the day's answer)
        assert exp.served[C.AGENT_LOOP_ENTRY_KEY] is True
        assert exp.served[C.REDUCTION_MODE_FIELD] == C.reduction_mode_from_env(os.environ)
        assert L.expected_identity(cfg, geom, pass_no=1, controls=cfg["controls"], samples=4,
                                   reduction_mode="per_row").served[C.REDUCTION_MODE_FIELD] == "per_row"
        assert (exp.cell, exp.mode, exp.pass_no, exp.samples, dict(exp.limits)) == (
            cfg["cell"], cfg["mode"], 1, 4, {})
        assert exp.client_lock_sha256 is None                    # no tree: recorded null, like the driver
        assert exp.tree_sha is None                              # ditto the producing tree (E14 B)
        d = exp.as_dict()
        assert d[C.GEOMETRY_PASS_FIELD] == 1 and d["served"] == dict(exp.served) and d["cell"] == cfg["cell"]
        # the driver argv the launcher builds drives the client at exactly the identity's width
        drv = L._driver_cmd("/tree", cfg, {"proxy_url": "u", "routing_table": "r"}, "l", geom,
                            controls=cfg["controls"], limit=0, inspect_limit=0, samples=4,
                            cfg_path="c", vllm_log=None)
        assert drv[drv.index("--concurrency") + 1] == str(exp.served["concurrency"])
    # PW-battery (U10 item 4): the OR-Bench item set follows the REQUESTED
    # controls too, and the launcher only ever asks for the frozen subset —
    # --or-bench-full is a driver flag it does not emit, so a battery wave and
    # an operator's full-corpus run can never resume onto each other's store
    for cfg in L.plan_configurations():
        geom, _ = L.served_geometry(cfg, 1)
        exp = L.expected_identity(cfg, geom, pass_no=1, controls=cfg["controls"], samples=4)
        assert exp.or_bench_item_set == C.or_bench_item_set() == C.run_or_bench_item_set(
            cfg["controls"])
        assert exp.as_dict()[C.IDENTITY_KEY_OR_BENCH_ITEM_SET] == exp.or_bench_item_set
        drv = L._driver_cmd("/tree", cfg, {"proxy_url": "u", "routing_table": "r"}, "l", geom,
                            controls=cfg["controls"], limit=0, inspect_limit=0, samples=4,
                            cfg_path="c", vllm_log=None)
        assert C.FLAG_OR_BENCH_FULL not in drv
    gsm_only = L.plan_configurations(cells=[C.CELL_PROMPTED_SAFETY], modes=["off"],
                                     controls=["gsm8k"])[0]
    tgeom, _ = L.served_geometry(gsm_only, 1)
    assert L.expected_identity(gsm_only, tgeom, pass_no=1, controls=["gsm8k"],
                               samples=4).or_bench_item_set is None
    # --tool-calls follows the REQUESTED controls, not the cell kind
    cfg = L.plan_configurations(cells=[C.CELL_PROMPTED_SAFETY], modes=["off"], controls=["gsm8k"])[0]
    geom, _ = L.served_geometry(cfg, 1)
    assert L.expected_identity(cfg, geom, pass_no=1, controls=["gsm8k"], samples=4).served["tool_calls"] is False
    assert L.expected_identity(cfg, geom, pass_no=1, controls=["gsm8k", "bfcl_200"], samples=4).served["tool_calls"] is True
    # limits and width enter; the lock sha is the staged tree's
    exp = L.expected_identity(cfg, geom, pass_no=2, controls=["gsm8k"], samples=2, limit=5,
                              inspect_limit=3, tree=REPO)
    assert dict(exp.limits) == {"limit": 5, "inspect_limit": 3} and exp.samples == 2 and exp.pass_no == 2
    assert exp.client_lock_sha256 == hashlib.sha256((REPO / C.CLIENT_LOCK_TXT).read_bytes()).hexdigest()
    assert L.client_lock_sha256(None) is None and L.client_lock_sha256(tmp_path) is None
    # the producing tree (E14 B): the caller's resolved sha wins, else the stamp
    staged = _tree(tmp_path / "repos", "f6cf462")
    assert L.tree_sha_of(None) is None and L.tree_sha_of(tmp_path) is None
    assert L.tree_sha_of(staged) == FULL_SHA
    assert L.expected_identity(cfg, geom, pass_no=1, controls=["gsm8k"], samples=4,
                               tree=staged).tree_sha == FULL_SHA
    assert L.expected_identity(cfg, geom, pass_no=1, controls=["gsm8k"], samples=4,
                               tree=staged, tree_sha="d" * 40).tree_sha == "d" * 40
    assert L.expected_identity(cfg, geom, pass_no=1, controls=["gsm8k"], samples=4,
                               tree=staged).as_dict()["tree_sha"] == FULL_SHA
    assert L.reasoning_parser_of([]) is None
    assert L.reasoning_parser_of(["--tool-calls", "--reasoning-parser", "gemma4"]) == "gemma4"
    with pytest.raises(ValueError, match="no value"):
        L.reasoning_parser_of(["--reasoning-parser"])
    # PW18: the entry follows the argv (on unless the stated opt-out), and the
    # standup argv the launcher composes never carries the opt-out
    assert L.agent_loop_entry_of([]) is True
    assert L.agent_loop_entry_of(["--tool-calls", "--manifest", "m"]) is True
    assert L.agent_loop_entry_of([C.FLAG_NO_PETRI_AGENT_LOOP]) is False
    for c in L.plan_configurations():
        assert C.FLAG_NO_PETRI_AGENT_LOOP not in c["standup_extra"]
        std = L._standup_cmd("/t", "/t/c.yaml", c["cell"], "/d", Path("/srv"), Path("/srv/STOP"),
                             L.served_geometry(c, 1)[0], 64, c["standup_extra"])
        assert C.FLAG_NO_PETRI_AGENT_LOOP not in std
    # E29 / PW18b, unset branch flipped by E46: the launcher's env parser IS
    # the contracts' (unset / empty / 1 / stacked -> stacked, 0 / per_row / off
    # -> per_row, else refuses; unset = stacked because the landing is merged
    # here) and yields an EXPECTATION; the launcher asks for ONE canonical mode
    # (--reduction-mode, default per_row — the wave in flight runs per_row, and
    # the flag makes that a stated request rather than a read of the env)
    assert L.reduction_mode_expected({}) == C.REDUCTION_MODE_STACKED == C.REDUCTION_MODE_ENV_UNSET
    assert L.reduction_mode_expected({C.LENS_STACKED_REDUCTIONS_ENV: ""}) == C.REDUCTION_MODE_STACKED
    assert L.reduction_mode_expected({C.LENS_STACKED_REDUCTIONS_ENV: "0"}) == C.REDUCTION_MODE_PER_ROW
    assert L.reduction_mode_expected({C.LENS_STACKED_REDUCTIONS_ENV: "off"}) == C.REDUCTION_MODE_PER_ROW
    assert L.reduction_mode_expected({C.LENS_STACKED_REDUCTIONS_ENV: "1"}) == C.REDUCTION_MODE_STACKED
    assert L.reduction_mode_expected({C.LENS_STACKED_REDUCTIONS_ENV: "stacked"}) == C.REDUCTION_MODE_STACKED
    with pytest.raises(ValueError, match="is not one of"):
        L.reduction_mode_expected({C.LENS_STACKED_REDUCTIONS_ENV: "yes"})
    assert L.REDUCTION_MODE_FLAG == "--reduction-mode"
    assert C.REDUCTION_MODE_LAUNCHER_DEFAULT == "per_row"
    # expected_identity: the argument is the mode; without it, the parser's
    # reading of THIS environment (unset = stacked since E46; "1" = stacked;
    # "0" = per_row). NOTE the argument-less reading and the FLAG's default
    # now differ on purpose — the tree serves stacked, the wave asks for
    # per_row — and nothing rides on it because the launcher always states it.
    monkeypatch.delenv(C.LENS_STACKED_REDUCTIONS_ENV, raising=False)
    assert L.expected_identity(cfg, geom, pass_no=1, controls=["gsm8k"],
                               samples=4).served[C.REDUCTION_MODE_FIELD] == "stacked"
    assert C.REDUCTION_MODE_LAUNCHER_DEFAULT == "per_row" != C.REDUCTION_MODE_ENV_UNSET
    monkeypatch.setenv(C.LENS_STACKED_REDUCTIONS_ENV, "1")
    assert L.expected_identity(cfg, geom, pass_no=1, controls=["gsm8k"],
                               samples=4).served[C.REDUCTION_MODE_FIELD] == "stacked"
    monkeypatch.setenv(C.LENS_STACKED_REDUCTIONS_ENV, "0")
    assert L.expected_identity(cfg, geom, pass_no=1, controls=["gsm8k"],
                               samples=4).served[C.REDUCTION_MODE_FIELD] == "per_row"
    monkeypatch.setenv(C.LENS_STACKED_REDUCTIONS_ENV, "maybe")
    with pytest.raises(ValueError, match="is not one of"):
        L.expected_identity(cfg, geom, pass_no=1, controls=["gsm8k"], samples=4)
    monkeypatch.delenv(C.LENS_STACKED_REDUCTIONS_ENV, raising=False)
    for mode in C.REDUCTION_MODES:
        assert L.expected_identity(cfg, geom, pass_no=1, controls=["gsm8k"], samples=4,
                                   reduction_mode=mode).served[C.REDUCTION_MODE_FIELD] == mode
    with pytest.raises(ValueError, match="not one of"):
        L.expected_identity(cfg, geom, pass_no=1, controls=["gsm8k"], samples=4, reduction_mode="grouped")
    assert "reduction_mode_source" not in L.expected_identity(
        cfg, geom, pass_no=1, controls=["gsm8k"], samples=4, reduction_mode="per_row").as_dict()
    # a contract field this launcher cannot derive refuses — never a narrower comparison
    monkeypatch.setattr(C, "SERVED_IDENTITY_FIELDS", C.SERVED_IDENTITY_FIELDS + ("gpu_mem_util",))
    with pytest.raises(L.IdentityUnderivable, match="gpu_mem_util"):
        L.expected_identity(cfg, geom, pass_no=1, controls=["gsm8k"], samples=4)


def test_launcher_frozen_counts_are_the_plans(L, tmp_path):
    """PW10 (the served-path owner, 2026-09-04): both prompted REC lines read
    INCOMPLETE because the launcher expected ``inspect/mmlu_letter_valid`` at
    a typed 100 while the Inspect plan builds that task with
    ``parity.PARITY_N`` (200) and the store correctly filed 200. The
    launcher's frozen count for every Inspect task the plan sizes by a
    constant now derives from the SAME constant, single-sourced in the
    contracts; and a store filing the plan's rows is DONE for the launcher."""
    from sisteer.capeval.tiers import inspect_served, parity

    assert C.PARITY_N == parity.PARITY_N == 200
    plan_src = (REPO / "src/sisteer/capeval/tiers/inspect_served.py").read_text()
    assert "PARITY_N = _tinycap_contracts.PARITY_N" in (REPO / "src/sisteer/capeval/tiers/parity.py").read_text()
    # the two generated-letter MMLU tasks and HellaSwag-Inspect are RETIRED
    #: the launcher freezes no count for them
    src = LAUNCHER.read_text()
    for gone in ("MMLU_LETTER_VALID_FULL_N", "MMLU_0_SHOT_FULL_N", "HELLASWAG_INSPECT_FULL_N"):
        assert gone not in src and not hasattr(L, gone), gone
    for name in C.RETIRED_INSPECT_TASK_NAMES:
        assert L.FULL_ITEM_COUNT_OF_TASK[C.inspect_task(name)] is None, name
    # every Inspect task with a frozen count: the launcher's number is the plan's
    # (gsm8k: the plan refuses a split whose size is not GSM8K_FULL_N; math and
    # hellaswag: the plan files len(ids) of the 500 MATH-500 problems / the 100
    # tinyHellaswag items — the dataset-defined sizes the launcher freezes)
    # PW17 (U4(a)): the STANDARD paired GSM8K's count is the frozen matched
    # count the plan asserts under verify_pairs (the slot's acceptance rule)
    plan_expected = {"gsm8k": C.GSM8K_FULL_N, "gsm8k_paired200": C.GSM8K_PAIRED_N, "math": 500}
    assert "size != C.GSM8K_FULL_N" in plan_src and "len(ids) != C.GSM8K_PAIRED_N" in plan_src
    for name, n in plan_expected.items():
        assert L.FULL_ITEM_COUNT_OF_TASK[C.inspect_task(name)] == n, name
    assert L.FULL_ITEM_COUNT_OF_TASK[C.GSM8K_STANDARD_TASK] == C.GSM8K_PAIRED_N == 200
    for name in ("bfcl", "xstest", "or_bench_hard_1k"):
        assert L.FULL_ITEM_COUNT_OF_TASK[C.inspect_task(name)] is None, name
    # bcb-bfcl-200: the two PINNED legs have frozen counts — the pins' n
    assert L.FULL_ITEM_COUNT_OF_TASK["inspect/bigcodebench_200"] == C.BCB_SUBSET_N == 200
    assert L.FULL_ITEM_COUNT_OF_TASK["inspect/bfcl_200"] == C.BFCL_SUBSET_N == 200
    # capeval-full-tasks: the FULL siblings' frozen counts are the full pins' n
    assert L.FULL_ITEM_COUNT_OF_TASK["inspect/bigcodebench_full"] == C.BCB_FULL_N == 1140
    assert L.FULL_ITEM_COUNT_OF_TASK["inspect/bfcl_full"] == C.BFCL_FULL_N == 3981
    # a store filing the plan's rows for the paired GSM8K is DONE for gsm8k
    exp = _expected(L)
    out = tmp_path / "thinking_off"
    store = _store(out, exp, tasks=["gsm8k"])
    _finish(store, C.GSM8K_STANDARD_TASK, C.GSM8K_PAIRED_N)
    assert L.remaining_controls(out, ["gsm8k"], exp) == []
    # ...while a count other than the frozen one is not
    _finish(store, C.GSM8K_STANDARD_TASK, 100)
    assert L.remaining_controls(out, ["gsm8k"], exp) == ["gsm8k"]


def test_the_trust_check_reads_provenance_through_the_one_reader(L, tmp_path):
    """P1-1 (2026-09-07): a store whose template-check class the canonical
    reader refuses, or whose fit witness / fit_provenance block disagree with
    it, is never trusted as complete — before this the reader had no caller
    here, and a current store with the class MISSING was accepted."""
    from sisteer.eval import sweep as _sweep
    exp = _expected(L)
    out = tmp_path / "off"
    ok = _store(out, exp, tasks=["gsm8k"])
    assert L.untrusted_store_reason(ok, expected=exp) is None
    key = _sweep.FIT_PROVENANCE_TEMPLATE_CHECK_KEY

    def drop_class(m):
        m.pop(key)                                      # current (contract_version 3), class missing
    why = L.untrusted_store_reason(_store(tmp_path / "a", exp, tasks=["gsm8k"], mutate=drop_class),
                                   expected=exp)
    assert why.startswith("provenance:") and "torn" in why

    def drop_witness(m):
        m.pop(_sweep.MANIFEST_KEY_FIT_WITNESS)
    why = L.untrusted_store_reason(_store(tmp_path / "b", exp, tasks=["gsm8k"], mutate=drop_witness),
                                   expected=exp)
    assert why.startswith("provenance:") and _sweep.MANIFEST_KEY_FIT_WITNESS in why

    def flip_block(m):
        m[_sweep.MANIFEST_KEY_FIT_PROVENANCE]["template_check"] = "inapplicable"
    why = L.untrusted_store_reason(_store(tmp_path / "c", exp, tasks=["gsm8k"], mutate=flip_block),
                                   expected=exp)
    assert why.startswith("provenance:") and "!= the stated class" in why

    def alter_witness_copy(m):
        m[_sweep.MANIFEST_KEY_FIT_PROVENANCE]["template_hash"] = "ff" * 8
    why = L.untrusted_store_reason(_store(tmp_path / "d", exp, tasks=["gsm8k"], mutate=alter_witness_copy),
                                   expected=exp)
    assert why.startswith("provenance:") and "disagrees with" in why


def test_resume_trusts_one_validated_store_never_a_union_of_marker_names(L, tmp_path):
    """R2 (launcher half): a control is done iff ONE store trusted for THIS
    run's identity REQUESTED the control's atoms and holds a validated done
    marker for EVERY task it produces. Bare file names, unions across
    stores, ``{}`` markers, wrong counts, missing summaries, partial
    markers, limited or foreign stores never count."""
    assert L.done_marker_name("evalplus/humaneval_plus:open") == "evalplus__humaneval_plus:open.json"
    assert L.expected_tasks("gsm8k", "off") == (C.GSM8K_STANDARD_TASK,)
    assert L.expected_tasks("humaneval_plus", "off") == ("evalplus/humaneval_plus",)
    assert L.expected_tasks("humaneval_plus", "on") == ("evalplus/humaneval_plus",
                                                        "evalplus/humaneval_plus:open")
    assert L.expected_tasks("inspect_cmp:xstest", "off") == ("inspect/xstest",)
    assert L.cell_key_hash_prefixes("safety_refusal--psr_kl_all__controls__t0__a0__d6e1b51a__ddc8d98f") == (
        "d6e1b51a", "ddc8d98f")
    assert L.cell_key_hash_prefixes("somekey") is None
    controls = ["gsm8k", "humaneval_plus", "inspect_cmp:or_bench_hard_1k"]
    atoms = [a for c in controls for a in C.expand_control(c)]
    # the expectation is for THIS mixed configuration. (MATH-500 runs ALONE — a
    # configuration pooling it beside other keys is refused before any
    # allocation — so the MATH-500 store, which hashes its first-pass cap, is
    # built on its own below.)
    exp = _expected(L, controls=controls)
    out = tmp_path / "thinking_off"

    def rc(expected=exp):
        return L.remaining_controls(out, controls, expected)

    assert rc() == controls                                             # no store at all
    store = _store(out, exp, tasks=atoms)
    assert rc() == controls and L.untrusted_store_reason(store, expected=exp) is None
    # the pre-R2 shape: bare / empty markers are NOT completion
    (store / C.DONE_DIR).mkdir()
    (store / C.DONE_DIR / L.done_marker_name("evalplus/humaneval_plus")).write_text("{}")
    (store / C.DONE_DIR / L.done_marker_name("inspect/or_bench_hard_1k")).write_text(
        json.dumps({k: 0 for k in C.DONE_MARKER_FIELDS}))
    assert rc() == controls
    assert "fields" in L.invalid_done_marker_reason(store, "evalplus/humaneval_plus")
    assert "names task" in L.invalid_done_marker_reason(store, "inspect/or_bench_hard_1k")
    # validated markers: the right fields, task, the frozen full count, a summary agreeing on n
    _finish(store, "evalplus/humaneval_plus", C.EXPECTED_HUMANEVAL_PLUS_N)
    assert rc() == ["gsm8k", "inspect_cmp:or_bench_hard_1k"]
    assert not any(t.startswith("served_ll/") for t in L.FULL_ITEM_COUNT_OF_TASK)
    assert L.FULL_ITEM_COUNT_OF_TASK["evalplus/humaneval_plus:open"] == 164
    assert L.FULL_ITEM_COUNT_OF_TASK[C.GSM8K_FULL_TASK] == C.GSM8K_FULL_N == 1319
    # U4(a) / PW17: the STANDARD paired set has a frozen MATCHED count (the slot's rule)
    assert L.FULL_ITEM_COUNT_OF_TASK[C.GSM8K_STANDARD_TASK] == C.GSM8K_PAIRED_N == 200
    assert L.FULL_ITEM_COUNT_OF_TASK["inspect/bfcl"] is None
    assert set(L.FULL_ITEM_COUNT_OF_TASK) == set(C.TINYCAP_TASKS)
    # thinking-ON also expects the :open variant, so humaneval_plus is NOT done there
    # (and the store is an OFF store anyway — untrusted for ON, named by mode)
    on = _expected(L, mode="on", controls=controls)
    assert rc(on) == controls
    assert "thinking.mode" in L.untrusted_store_reason(store, expected=on)
    # a two-task control (humaneval_plus under thinking ON: the :open twin) that
    # is partially done re-runs whole ...
    on_controls = ["humaneval_plus"]
    on_exp = _expected(L, mode="on", controls=on_controls)
    on_out = tmp_path / "thinking_on"
    on_store = _store(on_out, on_exp, tasks=["humaneval_plus"], mode="on")
    _finish(on_store, "evalplus/humaneval_plus", C.EXPECTED_HUMANEVAL_PLUS_N)
    assert L.remaining_controls(on_out, on_controls, on_exp) == on_controls
    # ...and its other task landing in a DIFFERENT store is no union: still remaining
    other = _store(on_out, on_exp, tasks=["humaneval_plus", "gsm8k"], mode="on")
    assert other != on_store
    _finish(other, "evalplus/humaneval_plus:open", C.EXPECTED_HUMANEVAL_PLUS_N)
    assert L.remaining_controls(on_out, on_controls, on_exp) == on_controls
    _finish(on_store, "evalplus/humaneval_plus:open", C.EXPECTED_HUMANEVAL_PLUS_N)
    assert L.remaining_controls(on_out, on_controls, on_exp) == []
    # the OFF configuration: OR-Bench done -> only gsm8k remains; a second OFF
    # store that never requested gsm8k sits beside it (a marker there would be foreign)
    _finish(store, "inspect/or_bench_hard_1k", 1319)
    assert rc() == ["gsm8k"]
    other = _store(out, exp, tasks=["inspect_cmp:or_bench_hard_1k"])
    assert other != store
    report = L.resume_report(out, controls, exp)
    assert report["expected"] == exp.as_dict()
    assert report["controls"]["humaneval_plus"]["done_in"] == store.name
    assert report["controls"]["gsm8k"]["done_in"] is None
    assert report["controls"]["gsm8k"]["why_not"][store.name] == {C.GSM8K_STANDARD_TASK: "no done marker"}
    # the other store never REQUESTED gsm8k: a marker there would be foreign
    assert "never requested ['gsm8k']" in report["controls"]["gsm8k"]["why_not"][other.name]
    assert report["stores"] == {store.name: None, other.name: None}
    _finish(store, C.GSM8K_STANDARD_TASK, C.GSM8K_PAIRED_N)
    assert rc() == []
    # a MATH-500 store — its own configuration (served at MATH's own window; a
    # mixed one is refused), hashing its first-pass cap. The marker's CONTENTS
    # are checked, not its presence
    mexp = _expected(L, controls=["math500"])
    assert mexp.files_math500 and mexp.math500_max_tokens == C.MATH500_MAX_TOKENS_DEFAULT
    mout = tmp_path / "math_off"
    mstore = _store(mout, mexp, tasks=["math500"])

    def mrc():
        return L.remaining_controls(mout, ["math500"], mexp)

    assert mrc() == ["math500"]
    mreport = L.resume_report(mout, ["math500"], mexp)
    assert mreport["controls"]["math500"]["why_not"][mstore.name] == {"inspect/math": "no done marker"}
    _finish(mstore, "inspect/math", 5)                                  # a smoke's count
    assert "frozen full item count 500" in L.invalid_done_marker_reason(mstore, "inspect/math")
    _finish(mstore, "inspect/math", L.MATH500_FULL_N, summary_n=499)
    assert "summary n" in L.invalid_done_marker_reason(mstore, "inspect/math")
    _finish(mstore, "inspect/math", L.MATH500_FULL_N)
    (mstore / C.SUMMARY_DIR / L.done_marker_name("inspect/math")).unlink()
    assert "summary" in L.invalid_done_marker_reason(mstore, "inspect/math")
    _finish(mstore, "inspect/math", L.MATH500_FULL_N)
    assert mrc() == []
    (mstore / C.PARTIAL_DIR).mkdir()
    (mstore / C.PARTIAL_DIR / L.done_marker_name("inspect/math")).write_text("{}")
    assert "partial marker" in L.invalid_done_marker_reason(mstore, "inspect/math") and mrc() == ["math500"]
    (mstore / C.PARTIAL_DIR / L.done_marker_name("inspect/math")).unlink()
    assert mrc() == []
    # a task without a contract count is checked for consistency only (the
    # dataset-sized XSTest; the full bfcl that used to play this part is retired)
    bf_exp = _expected(L, cell=C.CELL_PROMPTED_SAFETY, controls=["inspect_cmp:xstest"])
    assert bf_exp.served["tool_calls"] is False
    bf = _store(out, bf_exp, tasks=["inspect_cmp:xstest"])
    _finish(bf, "inspect/xstest", 1234)
    assert L.remaining_controls(out, ["inspect_cmp:xstest"], bf_exp) == []
    _finish(bf, "inspect/xstest", 1234, summary_n=1200)
    assert L.remaining_controls(out, ["inspect_cmp:xstest"], bf_exp) == ["inspect_cmp:xstest"]
    _finish(bf, "inspect/xstest", 0)
    assert "positive" in L.invalid_done_marker_reason(bf, "inspect/xstest")
    # stores that are never trusted, however their markers look
    assert rc() == []
    for kw, needle in ((dict(limits={"limit": 5}), "limits"),
                       (dict(drop_identity_field="limits"), "missing ['limits']"),
                       (dict(cell=C.CELL_PROMPTED_SAFETY), "cell_id"),
                       (dict(mode="on"), "thinking.mode"),
                       (dict(key="renamed__controls__t0__a0__deadbeef__deadbeef"), "prefixes")):
        o2 = tmp_path / "o2"
        cell_kw = {"cell": kw.pop("cell")} if "cell" in kw else {}
        s2 = _store(o2, _expected(L, **cell_kw), **kw)
        _finish(s2, C.GSM8K_STANDARD_TASK, C.GSM8K_PAIRED_N)
        why = L.untrusted_store_reason(s2, expected=exp)
        assert why is not None and needle in why, (kw, why)
        assert L.remaining_controls(o2, ["gsm8k"], exp) == ["gsm8k"]
        import shutil
        shutil.rmtree(o2)
    # no manifest at all, or an unreadable one
    bare = out / "cells" / "bare__controls__t0__a0__11111111__22222222"
    bare.mkdir()
    assert L.untrusted_store_reason(bare, expected=exp) == "no manifest.json"
    (bare / "manifest.json").write_text("not json")
    assert "unreadable" in L.untrusted_store_reason(bare, expected=exp)


@pytest.mark.parametrize("case", [
    "pass2_over_pass1_store", "relabelled_top_level_pass", "legacy_identity_lacks_pass",
    "served_empty", "legacy_three_field_served", "window", "parser",
    "seqs", "concurrency", "samples", "limited", "hash_not_of_resolved_config",
    "tinycap_not_mirroring_the_hashed_block", "no_producer", "producer_short_sha",
    "producer_of_another_revision",
    "lock_sha_of_another_tree", "framework_off_its_pin", "legacy_identity_lacks_frameworks",
    "geometry_block_disagrees_with_the_hashed_s", "hand_run_without_a_pass",
    "no_resolved_config", "route_tool_calls_disagrees",
    # PW18 (U4(b) / E25): the two new served facts, hashed and mirrored
    "legacy_five_field_served", "agent_loop_entry", "reduction_mode",
    "route_agent_loop_entry_disagrees", "route_reduction_mode_disagrees",
    # PW-sampling (U8): the draw is identity — a greedy store under a card run,
    # another card row, another per-task k, a legacy identity without the
    # fields, a manifest without the block, a block that does not mirror
    "greedy_store_under_a_card_run", "another_models_card_row", "another_per_task_k",
    "legacy_identity_lacks_sampling", "sampling_block_missing", "sampling_block_disagrees",
    "sampling_block_mode_disagrees",
    # PW-battery (U10 item 4): the OR-Bench item set is identity — the full
    # corpus under a battery run, another frozen draw, a legacy identity
    # without the field
    "or_bench_full_corpus_store", "or_bench_another_draw", "legacy_identity_lacks_or_bench"])
def test_resume_refuses_a_store_whose_identity_differs_from_the_run(L, tmp_path, case):
    """E12 item 1 (negatives): a store with EVERY task validly done is still
    NOT trusted when its recorded identity differs from the identity this
    run resolves — pass 2 over a pass-1 store, ``served == {}``, a legacy
    3-field ``served``, another window / parser / S / client width / avg@k
    width, a limited store, a manifest whose hash is not its
    ``resolved_config``'s, a ``tinycap`` block not mirroring the hashed
    one, no / a truncated producer, a frameworks lock sha from another tree
    or a version off its pin, a legacy identity lacking a field, a
    geometry / route block disagreeing with the hashed identity, a hand
    run without a pass, no resolved_config. Never DONE. Before this, a
    manifest with ``tinycap.served == {}`` was accepted and pass 2 on a
    pass-1 store executed zero requests and reported success."""
    exp = _expected(L)
    check = exp
    kw: dict = {}
    if case == "pass2_over_pass1_store":
        check = exp._replace(pass_no=2)
        needle = f"geometry.{C.GEOMETRY_PASS_FIELD} 1 != this run's pass 2"
    elif case == "relabelled_top_level_pass":
        # E16 (A2): the top-level geometry.pass says what this run wants, but
        # the HASHED identity says pass 1 — the hashed one is the store's pass
        check = exp._replace(pass_no=2)
        kw["mutate"] = lambda m: m[C.MANIFEST_KEY_GEOMETRY].__setitem__(C.GEOMETRY_PASS_FIELD, 2)
        needle = f"tinycap.{C.GEOMETRY_PASS_FIELD} 1 != this run's pass 2"
    elif case == "legacy_identity_lacks_pass":
        kw["drop_identity_field"] = C.GEOMETRY_PASS_FIELD
        needle = f"missing ['{C.GEOMETRY_PASS_FIELD}']"
    elif case == "served_empty":
        kw["served"] = {}
        needle = "tinycap.served carries []"
    elif case == "legacy_three_field_served":
        kw["served"] = {k: exp.served[k] for k in ("reasoning_parser", "tool_calls", "max_model_len")}
        needle = "missing ['agent_loop_entry', 'concurrency', 'max_num_seqs', 'reduction_mode']"
    elif case == "legacy_five_field_served":
        # a PW5..PW17 store: window, S and width but neither PW18 fact — never
        # trusted (its tool-turn regime and lens arithmetic are unvouched)
        kw["served"] = {k: exp.served[k] for k in
                        ("reasoning_parser", "tool_calls", "max_model_len", "max_num_seqs", "concurrency")}
        needle = "missing ['agent_loop_entry', 'reduction_mode']"
    elif case == "agent_loop_entry":
        kw["served"] = {**exp.served, C.AGENT_LOOP_ENTRY_KEY: False}
        needle = "served.agent_loop_entry: the store records False but this run serves True"
    elif case == "reduction_mode":
        kw["served"] = {**exp.served, C.REDUCTION_MODE_FIELD: C.REDUCTION_MODE_STACKED}
        needle = "served.reduction_mode: the store records 'stacked' but this run serves 'per_row'"
    elif case == "route_agent_loop_entry_disagrees":
        kw["mutate"] = lambda m: m[C.MANIFEST_KEY_ROUTE].__setitem__(C.AGENT_LOOP_ENTRY_KEY, False)
        needle = "manifest.route.agent_loop_entry False != True"
    elif case == "route_reduction_mode_disagrees":
        kw["mutate"] = lambda m: m[C.MANIFEST_KEY_ROUTE].__setitem__(C.REDUCTION_MODE_FIELD, "stacked")
        needle = "manifest.route.reduction_mode 'stacked' != 'per_row'"
    elif case == "window":
        kw["served"] = {**exp.served, "max_model_len": exp.served["max_model_len"] * 2}
        needle = "served.max_model_len"
    elif case == "parser":
        kw["served"] = {**exp.served, "reasoning_parser": "gemma4"}
        needle = "served.reasoning_parser: the store records 'gemma4' but this run serves None"
    elif case == "seqs":
        kw["served"] = {**exp.served, "max_num_seqs": exp.served["max_num_seqs"] - 12}
        needle = "served.max_num_seqs"
    elif case == "concurrency":
        kw["served"] = {**exp.served, "concurrency": 8}
        needle = "served.concurrency: the store records 8"
    elif case == "samples":
        kw["samples"] = 2
        needle = f"tinycap.samples 2 != {C.MATH500_SAMPLES_PER_ITEM}"
    elif case == "greedy_store_under_a_card_run":
        kw["sampling"] = C.sampling_identity(C.GREEDY_SAMPLING,
                                             {t: 1 for t in exp.sampling["samples_by_task"]})
        kw["sampling_mode"] = C.SAMPLING_GREEDY
        needle = "tinycap.temperature: the store records 0.0 but this run draws 1.0"
    elif case == "another_models_card_row":
        kw["sampling"] = {**exp.sampling, "temperature": 0.6, "top_p": 0.9, "top_k": None}
        needle = "tinycap.top_k: the store records None but this run draws 64"
    elif case == "another_per_task_k":
        kw["sampling"] = {**exp.sampling,
                          "samples_by_task": {t: 2 for t in exp.sampling["samples_by_task"]}}
        needle = "tinycap.samples_by_task: the store records"
    elif case == "legacy_identity_lacks_sampling":
        kw["drop_identity_field"] = C.IDENTITY_KEY_SEED_RULE
        needle = "missing ['seed_rule']"
    elif case == "or_bench_full_corpus_store":
        # an operator's --or-bench-full store can never continue a battery wave:
        # 200 frozen items and 1,319 are two refusal rates
        kw["or_bench_item_set"] = C.or_bench_item_set(subset=False)
        needle = (f"tinycap.{C.IDENTITY_KEY_OR_BENCH_ITEM_SET}: the store records "
                  "item_set='full' n=1319 of=1319 but this run files "
                  "item_set='subset' n=200 of=1319 seed=20260906")
    elif case == "or_bench_another_draw":
        kw["or_bench_item_set"] = C.or_bench_item_set(seed=C.OR_BENCH_SUBSET_SEED + 1)
        needle = f"tinycap.{C.IDENTITY_KEY_OR_BENCH_ITEM_SET}: the store records"
    elif case == "legacy_identity_lacks_or_bench":
        kw["drop_identity_field"] = C.IDENTITY_KEY_OR_BENCH_ITEM_SET
        needle = f"missing ['{C.IDENTITY_KEY_OR_BENCH_ITEM_SET}']"
    elif case == "sampling_block_missing":
        kw["mutate"] = lambda m: m.__delitem__(C.MANIFEST_KEY_SAMPLING)
        needle = "manifest.sampling is not a mapping"
    elif case == "sampling_block_disagrees":
        kw["mutate"] = lambda m: m[C.MANIFEST_KEY_SAMPLING].__setitem__("top_k", 1)
        needle = "manifest.sampling.top_k 1 does not mirror the hashed tinycap.top_k 64"
    elif case == "sampling_block_mode_disagrees":
        kw["mutate"] = lambda m: m[C.MANIFEST_KEY_SAMPLING].__setitem__("mode", "greedy")
        needle = "manifest.sampling.mode 'greedy' != this run's 'card'"
    elif case == "limited":
        kw["limits"] = {"limit": 5}
        needle = "tinycap.limits {'limit': 5} != this run's {}"
    elif case == "hash_not_of_resolved_config":
        kw["mutate"] = lambda m: m["resolved_config"].__setitem__("scale", 2.0)
        needle = "is not the hash of the recorded resolved_config"
    elif case == "tinycap_not_mirroring_the_hashed_block":
        kw["mutate"] = lambda m: m[C.MANIFEST_KEY_TINYCAP]["served"].__setitem__("max_model_len", 1)
        needle = "does not mirror resolved_config.tinycap"
    elif case == "no_producer":
        kw["producer"] = None
        needle = "manifest.producer is not a mapping"
    elif case == "producer_short_sha":
        kw["producer"] = {"tree_sha": "f6cf462", "driver": C.PRODUCER_DRIVER_SERVED}
        needle = "is not a full commit sha"
    elif case == "producer_of_another_revision":
        # E14 (B): the producer sha is not hashed, so a store from another
        # revision keys the SAME — only this comparison keeps its rows out
        kw["producer"] = {"tree_sha": "b" * 40, "driver": C.PRODUCER_DRIVER_SERVED}
        needle = f"producer.tree_sha {'b' * 40} was produced by another revision"
    elif case == "lock_sha_of_another_tree":
        kw["frameworks"] = {"client_lock_sha256": "cd" * 32}
        needle = "frameworks.client_lock_sha256"
    elif case == "framework_off_its_pin":
        kw["frameworks"] = {"inspect_ai": "0.3.259"}
        needle = "frameworks.inspect_ai '0.3.259' != the pinned"
    elif case == "legacy_identity_lacks_frameworks":
        kw["drop_identity_field"] = C.MANIFEST_KEY_FRAMEWORKS
        needle = "missing ['frameworks']"
    elif case == "geometry_block_disagrees_with_the_hashed_s":
        kw["mutate"] = lambda m: m[C.MANIFEST_KEY_GEOMETRY].__setitem__("max_num_seqs", 1)
        needle = "geometry.max_num_seqs 1 != the served identity's"
    elif case == "hand_run_without_a_pass":
        kw["mutate"] = lambda m: m[C.MANIFEST_KEY_GEOMETRY].__setitem__(C.GEOMETRY_PASS_FIELD, None)
        needle = f"geometry.{C.GEOMETRY_PASS_FIELD} None"
    elif case == "no_resolved_config":
        kw["mutate"] = lambda m: m.__delitem__("resolved_config")
        needle = "no resolved_config"
    else:
        flipped = not exp.served["tool_calls"]
        kw["mutate"] = lambda m: m[C.MANIFEST_KEY_ROUTE].__setitem__("tool_calls", flipped)
        needle = f"manifest.route.tool_calls {flipped}"
    out = tmp_path / "thinking_off"
    store = _store(out, exp, **kw)
    _finish(store, C.GSM8K_STANDARD_TASK, C.GSM8K_PAIRED_N)
    why = L.untrusted_store_reason(store, expected=check)
    assert why is not None and needle in why, (case, why)
    if case == "pass2_over_pass1_store":                 # both the hashed pass and its cross-check are named
        assert f"tinycap.{C.GEOMETRY_PASS_FIELD} 1 != this run's pass 2" in why
    report = L.resume_report(out, ["gsm8k"], check)
    assert report["controls"]["gsm8k"]["done_in"] is None
    assert report["controls"]["gsm8k"]["why_not"][store.name] == why
    assert L.remaining_controls(out, ["gsm8k"], check) == ["gsm8k"]
    # the same markers in a store written under THIS identity are trusted
    ok = _store(tmp_path / "ok", exp)
    _finish(ok, C.GSM8K_STANDARD_TASK, C.GSM8K_PAIRED_N)
    assert L.untrusted_store_reason(ok, expected=exp) is None
    assert L.remaining_controls(tmp_path / "ok", ["gsm8k"], exp) == []


def test_a_store_from_another_revision_refuses_before_the_standup(L, tmp_path, monkeypatch):
    """E14 (B), the launcher half: the producer sha is not hashed, so a store
    another revision produced keys the SAME as this run's — the driver's
    resume gate refuses it, but only after a standup has been paid for. The
    launcher detects exactly that store (every other identity field matches,
    which is what makes the driver land on it) and refuses the configuration
    before any GPU minute, naming the store, its sha and the remedy. A store
    from another revision that ALSO differs in identity is NOT a conflict —
    the driver keys elsewhere and never touches it."""
    # gsm8k only: the test is about the producer sha, not the battery — the
    # fitted default carries EvalPlus, whose staged archives this fake tree does
    # not hold (the archive check would fire first)
    cfg = L.plan_configurations(cells=[C.CELL_PSRKL_SAFETY], modes=["off"], controls=["gsm8k"])[0]
    repos = tmp_path / "repos"
    tree = _tree(repos, "f6cf462")
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    # E29 / PW18b: the launcher asks for per_row by default; a container env
    # stating the same is fine (a disagreeing one refuses — tested below)
    monkeypatch.setenv(C.LENS_STACKED_REDUCTIONS_ENV, "0")
    out = tmp_path / "gen"
    cfg = dict(cfg, out_dir=str(out))
    exp = _expected(L, tree=tree, controls=cfg["controls"])     # gsm8k only: no --tool-calls
    other = {"tree_sha": "b" * 40, "driver": C.PRODUCER_DRIVER_SERVED}
    store = _store(out, exp, producer=other)
    _finish(store, C.GSM8K_STANDARD_TASK, C.GSM8K_PAIRED_N)
    assert L.producer_tree_conflicts(out, expected=exp) == {store.name: "b" * 40}
    rec = L._run_configuration(cfg, "f6cf462")
    assert rec["status"] == "PRODUCER_TREE_MISMATCH"
    assert rec["producer_conflicts"] == {store.name: "b" * 40}
    assert "b" * 40 in rec["error"] and FULL_SHA in rec["error"] and "fresh out root" in rec["error"]
    assert "was produced by another revision" in rec["resume"]["stores"][store.name]
    # this tree's own store is no conflict, and it is DONE
    same = tmp_path / "gen_same"
    ok = _store(same, exp)
    _finish(ok, C.GSM8K_STANDARD_TASK, C.GSM8K_PAIRED_N)
    assert L.producer_tree_conflicts(same, expected=exp) == {}
    assert L._run_configuration(dict(cfg, out_dir=str(same)), "f6cf462")["status"] == "DONE"
    # another revision AND another identity: a different key, so the driver
    # never lands on it — untrusted, but not a conflict
    elsewhere = tmp_path / "gen_elsewhere"
    s2 = _store(elsewhere, exp._replace(served={**exp.served, "max_num_seqs": 24,
                                                "concurrency": 24}), producer=other)
    assert L.untrusted_store_reason(s2, expected=exp) is not None
    assert L.producer_tree_conflicts(elsewhere, expected=exp) == {}
    # a legacy store without the block is not a conflict either (it is refused
    # for having no producer at all, which the resume report already says)
    legacy = tmp_path / "gen_legacy"
    s3 = _store(legacy, exp, producer=None)
    assert L.producer_tree_conflicts(legacy, expected=exp) == {}
    assert "producer" in L.untrusted_store_reason(s3, expected=exp)


def test_run_configuration_resolves_the_identity_before_any_marker_is_read(L, tmp_path, monkeypatch):
    """E12 item 1: ``_run_configuration`` resolves the served geometry and
    the full expected identity BEFORE the resume decision — a perfect store
    under THIS identity is DONE (no GPU minute); the same store under pass
    2 never is (the geometry refusal fires first, before a marker is read);
    a store under another S proceeds to the standup with the REC naming
    why; the standup's tool-call mode follows the REQUESTED controls."""
    # gsm8k only (see test_a_store_from_another_revision_refuses_before_the_standup)
    cfg = L.plan_configurations(cells=[C.CELL_PSRKL_SAFETY], modes=["off"], controls=["gsm8k"])[0]
    repos = tmp_path / "repos"
    tree = _tree(repos, "f6cf462")            # addressed by a prefix of its stamp, as staged
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    # E29 / PW18b: the launcher's default --reduction-mode per_row is the
    # expected identity; the DONE shortcut applies before any standup
    monkeypatch.delenv(C.LENS_STACKED_REDUCTIONS_ENV, raising=False)
    out = tmp_path / "gen"
    cfg = dict(cfg, out_dir=str(out))
    exp = _expected(L, tree=tree, controls=cfg["controls"])     # gsm8k only: no --tool-calls
    assert exp.served[C.REDUCTION_MODE_FIELD] == C.REDUCTION_MODE_LAUNCHER_DEFAULT == "per_row"
    store = _store(out, exp)
    for c in cfg["controls"]:
        for t in L.expected_tasks(c, "off"):
            _finish(store, t, L.FULL_ITEM_COUNT_OF_TASK[t])
    rec = L._run_configuration(cfg, "f6cf462")
    assert rec["status"] == "DONE" and rec["controls_remaining"] == []
    assert rec["reduction_mode_requested"] == "per_row"
    assert rec["tree"] == str(tree) and rec["tree_sha"] == FULL_SHA
    assert rec["expected_identity"] == exp.as_dict() == rec["resume"]["expected"]
    assert rec["resume"]["controls"]["gsm8k"]["done_in"] == store.name
    assert rec["tool_calls"] is False and rec["geometry"][C.GEOMETRY_PASS_FIELD] == 1
    # pass 2 over the pass-1 store: the geometry is resolved FIRST. For a key
    # still provisional (tiny_prompted, analytic; named directly, since no control
    # pools into it now) it refuses before any marker is read; for THIS cell's key
    # (capability:gen_long, MEASURED) it resolves the measured window at the
    # model's budget — under which a pass-1 store at the footprint is never a resume
    pcfg = dict(L.plan_configurations(cells=[C.CELL_PROMPTED_SAFETY], modes=["off"], controls=["gsm8k"])[0],
                geometry_keys=("capability:tiny_prompted",))
    with pytest.raises(sg.GeometryProvisional):
        L._run_configuration(dict(pcfg, out_dir=str(out)), "f6cf462", pass_no=2)
    g2, rec2 = L.served_geometry(cfg, 2)
    assert g2.max_model_len == 4608 and rec2["max_num_seqs"] == g2.max_num_seqs
    assert g2.max_num_seqs == sg.BUDGET_BY_MODEL[C.MODEL_SLUG].seqs_at(4608)
    assert g2.window_source == "table" and not g2.measurement_pass
    # a geometry that cannot resolve stops the run before any marker is read
    monkeypatch.setattr(L, "served_geometry", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("geometry refused")))
    with pytest.raises(RuntimeError, match="geometry refused"):
        L._run_configuration(cfg, "f6cf462")
    monkeypatch.undo()
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    # a store under another served S (a different budget) is NOT resumed: the run
    # goes on to the standup (a fake Popen that dies at once -> STANDUP_FAILED),
    # and the REC names the disagreement
    other_out = tmp_path / "gen_other"
    cfg2 = dict(cfg, out_dir=str(other_out))
    s2 = _store(other_out, exp._replace(served={**exp.served, "max_num_seqs": 24, "concurrency": 24}))
    _finish(s2, C.GSM8K_STANDARD_TASK, C.GSM8K_PAIRED_N)

    class DeadProc:
        def __init__(self, *a, **k):
            pass

        def poll(self):
            return 1

        def wait(self, timeout=None):
            return 1

        def terminate(self):
            pass

        def kill(self):
            pass

    monkeypatch.setattr(L.subprocess, "Popen", DeadProc)
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    # (a fitted configuration reaches its standup only with the preflight's digest — E40 H1)
    rec = L._run_configuration(cfg2, "f6cf462", object_digests=_digests_for(cfg2))
    assert rec["status"] == "STANDUP_FAILED" and rec["controls_remaining"] == cfg["controls"]
    assert "served.max_num_seqs: the store records 24" in rec["resume"]["stores"][s2.name]
    # the same markers in an untrusted (limited) store leave everything remaining
    limited = tmp_path / "gen_lim"
    s3 = _store(limited, exp, limits={"limit": 5})
    _finish(s3, C.GSM8K_STANDARD_TASK, C.GSM8K_PAIRED_N)
    report = L.resume_report(limited, cfg["controls"], exp)
    assert report["controls"]["gsm8k"]["done_in"] is None
    assert "limits" in report["stores"][s3.name]
    # a missing tree is loud, before anything else; a tree without its stamp too
    rec = L._run_configuration(dict(cfg, out_dir=str(tmp_path / "other")), "nope")
    assert rec["status"] == "NO_TREE"
    _tree(repos, "nostamp", stamp=False)
    rec = L._run_configuration(dict(cfg, out_dir=str(tmp_path / "other")), "nostamp")
    assert rec["status"] == "NO_TREE_SHA" and C.TREE_SHA_FILENAME in rec["error"]
    src = LAUNCHER.read_text()
    i_geom = src.index("geom, model_budget = served_geometry(cfg, pass_no, tree)")
    i_exp = src.index("expected = expected_identity(cfg, geom, pass_no=pass_no, controls=wanted")
    # BEFORE the standup the resume decision tolerates a store shrunk from this
    # configured S by the KV-derived rule (the served S is a prediction there) ...
    i_resume = src.index("resume = resume_report(out_dir, wanted, expected, kv_shrink_tolerated=True)")
    assert i_geom < i_exp < i_resume < src.index('rec["status"] = "DONE"')
    # ... and is re-taken EXACTLY once the standup has said what it serves
    # (this tree keeps main's ladder arm beside the plain one on both sides of the standup)
    assert src.index('rec["status"] = "DONE"') < src.index("standup = subprocess.Popen(") \
        < src.index("resume = resume_report(out_dir, wanted, expected) if not rung else "
                    "ladder_resume_report(out_dir, wanted, expected)\n")
    assert 'rec["resume"] = resume' in src and 'rec["resume_after"] = after' in src
    assert "tool_calls = needs_tool_calls(wanted)" in src and "needs_tool_calls(remaining)" not in src


class _FakeStandupProc:
    """A scripted standup child for ``_run_configuration``: writes the
    ``endpoint.json`` the real standup writes (``record`` merged over the
    minimum the launcher reads) into the ``--out`` dir of the argv it was
    started with, and stays 'alive' until stopped. Any OTHER Popen (the
    ledger) dies at once, so a run that gets past the standup ends
    LEDGER_FAILED — the assertions here are about what happens before that."""

    record: ClassVar[dict] = {}
    last_env: ClassVar[dict] = {}

    def __init__(self, argv, *a, **k):
        self.argv = list(argv)
        self.rc = None
        if "capctrl_serving_up.py" in " ".join(self.argv[:2]):
            type(self).last_env = dict(k.get("env") or {})
            out = Path(self.argv[self.argv.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            # the geometry block echoes the dims the launcher ASKED for (the real
            # standup records exactly what it received); the launcher now holds the served
            # geometry to its resolution and refuses another S without a kv_shrink record
            argv = self.argv
            dims = {k: (int(argv[argv.index(f)+1]) if f in argv else d)
                    for k, f, d in (("max_model_len", "--max-model-len", 16384), ("max_num_seqs", "--max-num-seqs", 36))}
            rec = {"proxy_url": "http://127.0.0.1:1/v1", "routing_table": str(out / "routing_table.json"),
                   "tool_calls": False, "reasoning_parser": None, "agent_loop_entry": True,
                   "geometry": dims, **type(self).record}
            (out / "endpoint.json").write_text(json.dumps(rec))
        else:
            self.rc = 1                      # the ledger: dead at once

    def poll(self):
        return self.rc

    def wait(self, timeout=None):
        return self.rc if self.rc is not None else 0

    def terminate(self):
        self.rc = 0

    def kill(self):
        self.rc = 0


def _fake_world(L, tmp_path, monkeypatch, *, record, complete_store_mode=None):
    """A fitted configuration with a staged tree and a fake standup attesting
    ``record``; optionally a COMPLETE store under ``complete_store_mode``. The
    launcher's standup dir is the REAL ``/tmp/srv_<name>`` (the fake standup
    writes its endpoint.json there, as the real one does), so it is cleared
    here and again by :func:`_run_fake` — a stale record would otherwise leak
    into the next test's ``_run_configuration``."""
    # gsm8k only: the fitted default carries EvalPlus, whose pins
    # and staged archives this fake tree does not hold — the archive check
    # would fire BEFORE the standup (EVALPLUS_ARCHIVES_MISSING), so the world
    # would never reach the attestation these tests are about
    cfg = L.plan_configurations(cells=[C.CELL_PSRKL_SAFETY], modes=["off"], controls=["gsm8k"])[0]
    shutil.rmtree(f"/tmp/srv_{cfg['name']}", ignore_errors=True)
    repos = tmp_path / "repos"
    tree = _tree(repos, "f6cf462")
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    out = tmp_path / "gen"
    cfg = dict(cfg, out_dir=str(out))
    if complete_store_mode is not None:
        exp = _expected(L, tree=tree, controls=cfg["controls"], reduction_mode=complete_store_mode)
        store = _store(out, exp)
        for c in cfg["controls"]:
            for t in L.expected_tasks(c, "off"):
                _finish(store, t, L.FULL_ITEM_COUNT_OF_TASK[t])
    _FakeStandupProc.record = dict(record)
    monkeypatch.setattr(L.subprocess, "Popen", _FakeStandupProc)
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    return cfg, out


#: a well-formed (fake) preflight-verified artifact digest for the fake worlds
FAKE_DIGEST = "ab" * 32


def _digests_for(cfg) -> dict:
    """The ``object_digests`` a fitted configuration needs to reach its standup
    (E40 H1) — empty for a base / prompted one."""
    if not cfg.get("steered"):
        return {}
    return {cfg["cell"]: {"artifact_sha256": FAKE_DIGEST, "intervention_hash": "1" * 16,
                          "expected": "1" * 16, "lens_route_intervention_hash": "2" * 16}}


def _preflight_report(L, cfgs) -> dict:
    """A fake preflight report vouching (match + a 64-hex digest) for every
    fitted cell of ``cfgs`` — what ``object_digests_from_preflight`` reads."""
    cells = {c["cell"]: {"match": True, "artifact_sha256": FAKE_DIGEST, "intervention_hash": "1" * 16,
                         "expected": "1" * 16, "lens_route_intervention_hash": "2" * 16,
                         "header_guard_will_pass": True}
             for c in cfgs if c.get("steered")}
    return {"ok": True, "checks": {L.OBJECT_IDENTITY_CHECK: {"ok": True, "cells": cells}}}


def _run_fake(L, cfg, sha="f6cf462", **kw):
    """``_run_configuration`` against the fake standup, the srv dir cleared
    after; a fitted configuration gets the digest its standup is held to
    (E40 H1) unless the caller passes its own ``object_digests``."""
    kw.setdefault("object_digests", _digests_for(cfg))
    try:
        return L._run_configuration(cfg, sha, **kw)
    finally:
        shutil.rmtree(f"/tmp/srv_{cfg['name']}", ignore_errors=True)


def test_reduction_mode_lifecycle_unset_env_stacked_server_completes_and_resumes(L, tmp_path, monkeypatch):
    """E29 / PW18b, the regression the round-11 verdict named: with
    SISTEER_LENS_STACKED_REDUCTIONS unset in every shell, a launch that asks
    for stacked writes ONE canonical value into the standup env, the standup
    attests stacked, the driver records stacked — and the NEXT invocation
    recognises the completed store as DONE instead of re-launching the GPU
    server after every successful run. Plus the explicit per_row case
    (default argument; a pre-landing, keyless server equals per_row)."""
    monkeypatch.delenv(C.LENS_STACKED_REDUCTIONS_ENV, raising=False)
    # a launch asking for stacked against a server that attests stacked: the run proceeds ...
    cfg, out = _fake_world(L, tmp_path, monkeypatch, record={C.PROXY_IDENTITY_REDUCTION_MODE: "stacked"})
    rec = _run_fake(L, cfg, reduction_mode="stacked")
    assert rec["status"] == "LEDGER_FAILED"                 # past the standup + attestation (the fake ledger dies)
    assert rec["reduction_mode_requested"] == "stacked" == rec[C.REDUCTION_MODE_FIELD]
    assert rec["expected_identity"]["served"]["reduction_mode"] == "stacked"
    assert _FakeStandupProc.last_env[C.LENS_STACKED_REDUCTIONS_ENV] == "stacked"   # ONE canonical value for the standup
    served_geom = json.loads((out / L.SERVED_GEOMETRY_FILENAME).read_text())
    assert served_geom["reduction_mode_requested"] == "stacked"
    assert served_geom["endpoint"][C.PROXY_IDENTITY_REDUCTION_MODE] == "stacked"
    # ... the driver would record stacked (the store's identity): the next invocation is DONE pre-standup
    exp = _expected(L, tree=_tree(tmp_path / "repos", "f6cf462"), controls=cfg["controls"],
                    reduction_mode="stacked")
    store = _store(out, exp)
    for c in cfg["controls"]:
        for t in L.expected_tasks(c, "off"):
            _finish(store, t, L.FULL_ITEM_COUNT_OF_TASK[t])
    rec = _run_fake(L, cfg, reduction_mode="stacked")
    assert rec["status"] == "DONE" and rec["controls_remaining"] == [] and "standup_s" not in rec
    assert rec["resume"]["controls"]["gsm8k"]["done_in"] == store.name
    # the SAME store is not a per_row run's (the default): remaining, named
    rec = _run_fake(L, cfg)
    assert rec["status"] == "REDUCTION_MODE_MISMATCH"        # the fake server attests stacked, per_row was asked
    assert rec["controls_remaining"] == cfg["controls"]
    [why] = rec["resume"]["stores"].values()
    assert "served.reduction_mode: the store records 'stacked' but this run serves 'per_row'" in why
    # explicit per_row (the default) against a PRE-LANDING server (no key): equals per_row, said so
    cfg2, out2 = _fake_world(L, tmp_path / "pr", monkeypatch, record={})
    rec = _run_fake(L, cfg2)
    assert rec["status"] == "LEDGER_FAILED" and rec[C.REDUCTION_MODE_FIELD] == "per_row"
    assert rec["reduction_mode_attestation"].startswith("pre-landing server")
    assert _FakeStandupProc.last_env[C.LENS_STACKED_REDUCTIONS_ENV] == "per_row"
    exp2 = _expected(L, tree=_tree(tmp_path / "pr" / "repos", "f6cf462"), controls=cfg2["controls"])
    store2 = _store(out2, exp2)
    for c in cfg2["controls"]:
        for t in L.expected_tasks(c, "off"):
            _finish(store2, t, L.FULL_ITEM_COUNT_OF_TASK[t])
    rec = _run_fake(L, cfg2)
    assert rec["status"] == "DONE" and "standup_s" not in rec
    # a server attesting per_row explicitly: per_row, and not a pre-landing note
    cfg3, _ = _fake_world(L, tmp_path / "pr2", monkeypatch, record={C.PROXY_IDENTITY_REDUCTION_MODE: "per_row"})
    rec = _run_fake(L, cfg3)
    assert rec[C.REDUCTION_MODE_FIELD] == "per_row" and "reduction_mode_attestation" not in rec


def test_a_reduction_mode_the_standup_does_not_attest_refuses_before_generation(L, tmp_path, monkeypatch):
    """E29 / PW18b: the attested lens_reduction_mode is compared with the
    expectation the moment endpoint.json appears — per_row asked (the
    default) but stacked attested -> REDUCTION_MODE_MISMATCH naming both, the
    standup torn down, no ledger, no driver; stacked asked against a keyless
    (pre-landing) server -> refused as pre-landing; an unknown attested value
    refuses; a container env disagreeing with the argument refuses before the
    standup; a bad argument refuses at once."""
    monkeypatch.delenv(C.LENS_STACKED_REDUCTIONS_ENV, raising=False)
    cfg, _ = _fake_world(L, tmp_path, monkeypatch, record={C.PROXY_IDENTITY_REDUCTION_MODE: "stacked"})
    rec = _run_fake(L, cfg)                                      # default: per_row asked
    assert rec["status"] == "REDUCTION_MODE_MISMATCH"
    assert "attests lens_reduction_mode='stacked' but --reduction-mode expects 'per_row'" in rec["error"]
    assert "driver_argv" not in rec and "ledger_url" not in rec and "standup_s" in rec
    assert rec["expected_identity"]["served"]["reduction_mode"] == "per_row"   # never overwritten
    cfg, _ = _fake_world(L, tmp_path / "kl", monkeypatch, record={})
    rec = _run_fake(L, cfg, reduction_mode="stacked")
    assert rec["status"] == "REDUCTION_MODE_MISMATCH" and "a pre-landing server" in rec["error"]
    assert "--reduction-mode per_row" in rec["error"]
    cfg, _ = _fake_world(L, tmp_path / "bad", monkeypatch, record={C.PROXY_IDENTITY_REDUCTION_MODE: "grouped"})
    rec = _run_fake(L, cfg)
    assert rec["status"] == "REDUCTION_MODE_MISMATCH" and "not one of" in rec["error"]
    # the container's own env disagreeing with the argument: refused before any standup
    monkeypatch.setenv(C.LENS_STACKED_REDUCTIONS_ENV, "1")
    cfg, _ = _fake_world(L, tmp_path / "env", monkeypatch, record={})
    rec = _run_fake(L, cfg)                                      # per_row asked, env says stacked
    assert rec["status"] == "REDUCTION_MODE_MISMATCH" and "--reduction-mode vs env" in rec["error"]
    assert "standup_s" not in rec
    rec = _run_fake(L, cfg, reduction_mode="stacked")            # agreeing env: on to the standup
    assert rec["status"] == "REDUCTION_MODE_MISMATCH" and "a pre-landing server" in rec["error"]
    monkeypatch.delenv(C.LENS_STACKED_REDUCTIONS_ENV, raising=False)
    rec = _run_fake(L, cfg, reduction_mode="grouped")
    assert rec["status"] == "REDUCTION_MODE_MISMATCH" and "not one of" in rec["error"]


def _watched_world(L, tmp_path, monkeypatch, *, record):
    """:func:`_fake_world`, plus spies on the three things a refusal at the
    attestation must NOT reach and the one it must: every ``Popen`` argv (the
    standup, then the ledger), the driver's ``subprocess.run``, and the
    launcher's own teardown."""
    cfg, out = _fake_world(L, tmp_path, monkeypatch, record=record)
    seen: dict = {"popen": [], "driver": [], "torn": []}
    standup_popen = L.subprocess.Popen          # _FakeStandupProc, installed above
    real_teardown = L._tear_down_standup

    def _popen(argv, *a, **k):
        seen["popen"].append(list(argv))
        return standup_popen(argv, *a, **k)

    def _run(cmd, *a, **k):                     # the served capability driver
        seen["driver"].append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def _teardown(stop, standup):
        real_teardown(stop, standup)            # then record what it achieved:
        seen["torn"].append((str(stop), stop.exists()))   # the STOP file is down

    monkeypatch.setattr(L.subprocess, "Popen", _popen)
    monkeypatch.setattr(L.subprocess, "run", _run)
    monkeypatch.setattr(L, "_tear_down_standup", _teardown)
    return cfg, out, seen


def test_a_standup_that_cannot_attest_its_mode_refuses_before_the_ledger_and_the_driver(
        L, tmp_path, monkeypatch):
    """E61, the round-21 P1. A post-landing standup ALWAYS writes
    ``lens_reduction_mode``, and writes it NULL plus a reason when its workers
    attest nothing or disagree. Reading the one key through
    ``contracts.resolve_reduction_mode`` cannot see the reason: the null
    arrived there as "nothing stated", read as the pre-landing ``per_row``
    default, and — with ``per_row`` requested, which is the wave's default —
    the expectation check PASSED and the run went on to the ledger and the
    driver, to write a store LABELLED per_row whose own standup had said its
    arithmetic could not be determined.

    Both shapes the writer can produce refuse: the mode key null with a
    reason, and the mode key ABSENT with a reason. The refusal quotes the
    reason, records the typed status, tears the standup down through the
    launcher's own teardown, and never contacts the ledger or starts the
    driver. The positive control is the same world with an attested per_row:
    it proceeds to the ledger exactly as before."""
    monkeypatch.delenv(C.LENS_STACKED_REDUCTIONS_ENV, raising=False)
    why = "worker reports disagree on lens_reduction_mode: ['per_row', 'stacked']"
    # the two keys as the WRITER spells them, deliberately as literals: this
    # pin must discriminate on the read site's BEHAVIOUR, not on a constant the
    # fix happens to introduce (the constant is pinned against lens_worker's in
    # the parity test below)
    shapes = {"null": {"lens_reduction_mode": None, "lens_reduction_mode_error": why},
              "absent": {"lens_reduction_mode_error": why}}
    for name, record in shapes.items():
        cfg, out, seen = _watched_world(L, tmp_path / name, monkeypatch, record=record)
        rec = _run_fake(L, cfg)                          # per_row asked: the wave's default
        assert rec["status"] == "REDUCTION_MODE_UNATTESTED", (name, rec["status"])
        assert why in rec["error"] and "refusing rather than recording the default" in rec["error"]
        assert "endpoint.json" in rec["error"]           # the record it refused, by path
        # nothing about the mode was recorded, and the request is still on the REC
        assert C.REDUCTION_MODE_FIELD not in rec and "reduction_mode_attestation" not in rec
        assert rec["reduction_mode_requested"] == "per_row"
        assert rec["expected_identity"]["served"]["reduction_mode"] == "per_row"
        # the standup came up and was torn down through the launcher's teardown
        assert "standup_s" in rec and seen["torn"] and all(down for _, down in seen["torn"])
        assert len(seen["torn"]) == 1
        # ... and NOTHING after it ran: one Popen (the standup), no ledger, no driver
        assert len(seen["popen"]) == 1, seen["popen"]
        assert not any("usage_ledger" in " ".join(a) for a in seen["popen"])
        assert seen["driver"] == []
        assert "ledger_url" not in rec and "usage_ledger" not in rec and "driver_argv" not in rec
        assert not (out / L.C.LEDGER_FILENAME).exists()
        # this is NOT an OK status: the wave's exit code counts it as work undone
        assert rec["status"] not in L.REC_OK_STATUSES
        assert L.failed_recs([rec]) == [(cfg["name"], "REDUCTION_MODE_UNATTESTED")]
    # positive control: the same world attesting per_row proceeds to the ledger
    cfg, out, seen = _watched_world(L, tmp_path / "ok", monkeypatch,
                                    record={C.PROXY_IDENTITY_REDUCTION_MODE: "per_row"})
    rec = _run_fake(L, cfg)
    assert rec["status"] == "LEDGER_FAILED" and rec[C.REDUCTION_MODE_FIELD] == "per_row"
    assert len(seen["popen"]) == 2 and "usage_ledger" in " ".join(seen["popen"][1])
    assert seen["torn"] == [] and seen["driver"] == []   # the fake ledger dies first
    # and a null mode with NO reason is still the pre-landing reading, said so
    cfg, _, seen = _watched_world(L, tmp_path / "nullnoreason", monkeypatch,
                                  record={C.PROXY_IDENTITY_REDUCTION_MODE: None})
    rec = _run_fake(L, cfg)
    assert rec["status"] == "LEDGER_FAILED" and rec[C.REDUCTION_MODE_FIELD] == "per_row"
    assert rec["reduction_mode_attestation"].startswith("pre-landing server")


def test_the_launcher_resolver_agrees_with_lens_worker_case_by_case(L):
    """Parity pin (E61): the launcher ships its OWN record-aware resolver —
    the read site is inside the container, after the H200 bring-up, with the
    STAGED CANDIDATE tree first on ``sys.path``, so importing one there is a
    crash on any pre-landing sha (the hazard main removed at fe81cd87). Two
    copies of one rule therefore exist, and this pins them together case for
    case, exactly as ``test_lens_reduction_mode_attestation`` section (h) pins
    main's driver copy. ``lens_worker`` is imported HERE, in the test, from
    the launcher's own tree — never at the read site."""
    from sisteer.capeval.serving import lens_adapters as la
    from sisteer.capeval.serving import lens_worker as lw

    KEY = lw.LENS_REDUCTION_MODE_KEY
    ERR = lw.LENS_REDUCTION_MODE_ERROR_KEY
    assert C.PROXY_IDENTITY_REDUCTION_MODE == KEY
    assert L.LENS_REDUCTION_MODE_ERROR_KEY == ERR
    assert set(C.REDUCTION_MODES) == set(la.REDUCTION_MODES)
    why = "scheduler-config worker 0: no lens_reduction_mode in its scheduler-config report"
    records = [{KEY: "per_row"}, {KEY: "stacked"},                     # named
               {KEY: "per_row", ERR: None}, {KEY: "stacked", ERR: ""},
               {}, {"proxy_url": "http://127.0.0.1:1/v1"},             # neither key
               {KEY: None, ERR: why},                                  # null + reason
               {ERR: why},                                             # ABSENT + reason
               {KEY: None, ERR: ""},                                   # null + empty reason
               {KEY: None}, {KEY: None, ERR: None},
               {KEY: "vectorized"}, {KEY: "STACKED"}, {KEY: ""},        # unknown values
               {KEY: 1}, {KEY: True},
               None, [], "endpoint.json", 3]                            # not a record

    def read(fn, record):
        try:
            return fn(record), False
        except (lw.ReductionModeUnattested, L.ReductionModeUnattested):
            return None, True

    for record in records:
        assert (read(L.endpoint_reduction_mode, record)
                == read(lw.endpoint_reduction_mode, record)), record
    # the two reason-bearing shapes are the whole point: BOTH refuse here and
    # BOTH read as the pre-landing default through the value-only resolver the
    # read site used to call on its own
    for record in ({KEY: None, ERR: why}, {ERR: why}):
        assert read(L.endpoint_reduction_mode, record) == (None, True)
        assert C.resolve_reduction_mode(record.get(KEY)) == C.REDUCTION_MODE_PER_ROW


def test_dry_run_names_the_bfcl_on_fitted_regime(L, capsys):
    """U4(b) / PW18: ``bfcl`` on a fitted cell is planned (no launcher
    refusal — PW15-5(i) is superseded), composes ``--tool-calls``, and the
    dry run says what that means: the lens route under the proxy's
    agent-loop entry, with the identity flag; a prompted BFCL configuration
    and a fitted one without BFCL print no such line."""
    fitted = L.plan_configurations(cells=[C.CELL_PSRKL_SAFETY], modes=["off"], controls=["bfcl_200"])
    assert fitted[0]["tool_calls"] is True and "--tool-calls" in fitted[0]["standup_extra"]
    out = L.describe_plan(fitted, limit=0, inspect_limit=0, samples=4, pass_no=1)
    assert "bfcl on a fitted cell: lens route" in out and "agent-loop entry" in out
    assert f"{C.AGENT_LOOP_ENTRY_KEY}=True" in out and "U4(b)" in out
    ident = json.loads(next(l for l in out.splitlines() if "resume identity" in l)
                       .split("resume identity: ", 1)[1])
    assert ident["served"][C.AGENT_LOOP_ENTRY_KEY] is True and ident["served"]["tool_calls"] is True
    # E29 / PW18b: the dry run names the mode asked for and the attestation rule
    assert ident["served"]["reduction_mode"] == "per_row"
    assert ("reduction_mode: 'per_row' (--reduction-mode; written into the standup env as "
            "SISTEER_LENS_STACKED_REDUCTIONS=per_row") in out
    assert "REDUCTION_MODE_MISMATCH" in out
    stacked = L.describe_plan(fitted, limit=0, inspect_limit=0, samples=4, pass_no=1,
                              reduction_mode="stacked")
    assert "reduction_mode: 'stacked'" in stacked
    assert json.loads(next(l for l in stacked.splitlines() if "resume identity" in l)
                      .split("resume identity: ", 1)[1])["served"]["reduction_mode"] == "stacked"
    prompted = L.plan_configurations(cells=[C.CELL_PROMPTED_SAFETY], modes=["off"], controls=["bfcl_200"])
    assert "bfcl on a fitted cell" not in L.describe_plan(prompted, limit=0, inspect_limit=0,
                                                          samples=4, pass_no=1)
    tiny = L.plan_configurations(cells=[C.CELL_PSRKL_SAFETY], modes=["off"], controls=["gsm8k"])
    assert "bfcl on a fitted cell" not in L.describe_plan(tiny, limit=0, inspect_limit=0,
                                                          samples=4, pass_no=1)


def test_dry_run_prints_the_resume_identity(L, capsys):
    # native-serving-default: the launcher's default is the native engine now; this test pins the LENS
    # launch (a PSR cell / the historical composition), so it states the path explicitly
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, dry_run=True)
    out = capsys.readouterr().out
    assert out.count("resume identity: ") == len(C.CONTAINER_NAMES)
    S = sg.identity_seqs(sg.PETRI_MAX_MODEL_LEN, model_key=C.MODEL_SLUG)
    line = next(l for l in out.splitlines() if "resume identity" in l and "prompted_safety-think_on" not in l)
    ident = json.loads(line.split("resume identity: ", 1)[1])
    assert ident["served"]["max_num_seqs"] == ident["served"]["concurrency"] == S
    assert ident["served"]["max_model_len"] == sg.PETRI_MAX_MODEL_LEN and ident[C.GEOMETRY_PASS_FIELD] == 1
    assert '"reasoning_parser": "gemma4"' in out and '"tool_calls": false' in out
    assert '"tool_calls": true' not in out          # U9 item 6: BFCL is out of the battery


# --------------------------------------------------------------------------- #
# TREE_SHA: the stamp every uploaded tree carries (E12 item 3)                 #
# --------------------------------------------------------------------------- #


def test_tree_sha_stamp_rides_in_every_uploaded_tree_and_binds_the_run(L, tmp_path):
    """E12 item 3 (launcher half): ``--upload-tree`` archives ``TREE_SHA``
    at the archive root (``git archive --add-file``, placed by basename),
    the stage / preflight / configuration refuse a tree without it or with
    another sha, and a sha resolves to its staged tree directly or through
    the stamp — so a merge chained with a store's FULL producer sha finds
    the tree the operator staged under an abbreviation."""
    assert L.tree_sha_stamp_text(FULL_SHA) == FULL_SHA + "\n"
    for bad in ("f6cf462", "", "G" * 40, FULL_SHA + "0"):
        with pytest.raises(ValueError, match="40-hex"):
            L.tree_sha_stamp_text(bad)
    stamp = tmp_path / C.TREE_SHA_FILENAME
    assert L.archive_argv("/repo", FULL_SHA, stamp) == [
        "git", "-C", "/repo", "archive", "--format=tar.gz", f"--add-file={stamp}", FULL_SHA]
    with pytest.raises(ValueError, match=C.TREE_SHA_FILENAME):
        L.archive_argv("/repo", FULL_SHA, tmp_path / "stamp.txt")
    # a REAL archive (of a tiny throwaway repo) carries the stamp at its root
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "x.py").write_text("x = 1\n")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "one"], check=True)
    head = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    stamp.write_text(L.tree_sha_stamp_text(head))
    tar = tmp_path / "t.tar.gz"
    with open(tar, "wb") as fh:
        subprocess.run(L.archive_argv(repo, head, stamp), stdout=fh, check=True)
    dst = tmp_path / "extracted"
    with tarfile.open(tar) as tf:
        names = set(tf.getnames())
        tf.extractall(dst, filter="data")
    assert {C.TREE_SHA_FILENAME, "src/x.py"} <= names
    assert (dst / C.TREE_SHA_FILENAME).read_text() == head + "\n"
    assert L.staged_tree_sha(dst, head[:7]) == head
    # staged_tree_sha / resolve_staged_tree
    repos = tmp_path / "repos"
    t1 = _tree(repos, "f6cf462", stamp=False)
    with pytest.raises(L.TreeShaMissing, match=C.TREE_SHA_FILENAME):
        L.staged_tree_sha(t1, "f6cf462")
    with pytest.raises(L.TreeShaMissing):
        L.resolve_staged_tree("f6cf462", repos)          # the dir exists but cannot name its tree
    with pytest.raises(L.NoStagedTree, match="no staged tree"):
        L.resolve_staged_tree("deadbee", repos)
    (t1 / C.TREE_SHA_FILENAME).write_text(FULL_SHA + "\n")
    assert L.staged_tree_sha(t1, "f6cf462") == FULL_SHA
    assert L.resolve_staged_tree("f6cf462", repos) == (t1, FULL_SHA)
    assert L.resolve_staged_tree(FULL_SHA, repos) == (t1, FULL_SHA)   # the FULL sha finds the abbreviated dir
    with pytest.raises(L.TreeShaMismatch, match="addressed as"):
        L.staged_tree_sha(t1, "deadbee")
    (t1 / C.TREE_SHA_FILENAME).write_text("not a sha\n")
    with pytest.raises(L.TreeShaMismatch, match="40-hex"):
        L.staged_tree_sha(t1, "f6cf462")
    (t1 / C.TREE_SHA_FILENAME).write_text(FULL_SHA + "\n")
    _tree(repos, "twin", FULL_SHA)
    with pytest.raises(L.TreeShaMismatch, match="several"):
        L.resolve_staged_tree(FULL_SHA, repos)
    assert L.resolve_staged_tree("f6cf462", repos) == (t1, FULL_SHA)   # the direct dir still wins
    assert issubclass(L.NoStagedTree, FileNotFoundError) and issubclass(L.TreeShaMissing, FileNotFoundError)
    src = LAUNCHER.read_text()
    assert 'chk("tree_sha_stamp", tree_sha_stamp)' in src              # preflight
    assert "staged_tree_sha(tmp, sha)" in src and "staged_tree_sha(tree, sha)" in src   # stage: fresh + already-extracted
    assert "archive_argv(REPO_LOCAL, full, stamp)" in src and "tree_sha_stamp_text(full)" in src   # upload
    assert "capeval/store.py" in L.LAUNCHER_PROCESS_MODULES and "capeval/manifest.py" in L.LAUNCHER_PROCESS_MODULES


def test_merge_is_bound_to_the_producing_tree(L, tmp_path, monkeypatch):
    """E12 item 3: ``merge_summaries`` resolves ``--sha`` to the staged
    tree whose TREE_SHA names it and merges ONLY stores whose
    ``manifest.producer.tree_sha`` equals it; a store named explicitly that
    another tree produced, and a legacy store without the block, are
    REFUSED (status FAILED, the remedy named); the sweep skips other trees'
    stores (another ``--sha``'s job) and still refuses legacy ones."""
    repos = tmp_path / "repos"
    t = _tree(repos, "f6cf462")
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    gen = tmp_path / "gen"

    def store(name, producer):
        d = gen / "x" / "cells" / name
        d.mkdir(parents=True)
        m: dict = {"cell_id": "c", C.MANIFEST_KEY_PROVENANCE: {"x": 1}}
        if producer is not None:
            m[C.MANIFEST_KEY_PRODUCER] = {"tree_sha": producer, "driver": C.PRODUCER_DRIVER_SERVED}
        (d / "manifest.json").write_text(json.dumps(m))
        return str(d)

    mine = store("mine__controls__t0__a0__11111111__22222222", FULL_SHA)
    foreign = store("foreign__controls__t0__a0__11111111__33333333", OTHER_SHA)
    legacy = store("legacy__controls__t0__a0__11111111__44444444", None)
    assert L.producer_tree_sha(mine) == FULL_SHA and L.producer_tree_sha(legacy) is None
    assert L.producer_tree_sha(tmp_path) is None                       # no manifest at all
    assert L.partition_by_producer([mine, foreign, legacy], FULL_SHA) == (
        [mine], {foreign: OTHER_SHA}, [legacy])
    calls = []

    def fake_merge(dirs, *, tree, **kw):
        calls.append((list(dirs), str(tree)))
        return {d: {"ok": True, "merged": True, "path": d + "/s"} for d in dirs}
    monkeypatch.setattr(L, "merge_cell_summaries", fake_merge)
    rec = L.merge_summaries("f6cf462", [mine, foreign, legacy], str(gen))
    assert rec["status"] == "FAILED" and rec["tree_sha"] == FULL_SHA and rec["tree"] == str(t)
    assert rec["writer_src"] == str(L.current_tree_src())             # PW10: the tree that WRITES is this one
    assert rec["writer_tree_sha"] == L.writer_tree_sha_of(L.current_tree_src())   # PW12: ...named by its commit
    assert "attest" in rec
    assert calls == [([mine], str(t))]                                # only its own store reached the merge
    assert rec["results"][mine]["ok"] is True
    assert rec["results"][foreign]["ok"] is False and OTHER_SHA in rec["results"][foreign]["refused"]
    assert f"--sha {OTHER_SHA}" in rec["results"][foreign]["refused"]
    assert rec["results"][legacy]["ok"] is False
    assert C.MANIFEST_KEY_PRODUCER in rec["results"][legacy]["refused"]
    assert C.TREE_SHA_FILENAME in rec["results"][legacy]["refused"]
    assert rec["refused_other_producer"] == {foreign: OTHER_SHA} and rec["refused_legacy_producer"] == [legacy]
    # the FULL producer sha (what the orchestrator chains) resolves the abbreviated tree dir
    calls.clear()
    rec = L.merge_summaries(FULL_SHA, [mine], str(gen))
    assert rec["status"] == "OK" and calls == [([mine], str(t))] and rec["cell_dirs"] == [mine]
    assert rec["replace_source"] == ""
    # PW8: a malformed --replace-source never reaches a merge; a good one is recorded canonically
    calls.clear()
    rec = L.merge_summaries(FULL_SHA, [mine], str(gen), replace_source="bogus")
    assert rec["status"] == "BAD_REPLACE_SOURCE" and calls == [] and "not <entry>=<cell_key>" in rec["error"]
    rec = L.merge_summaries(FULL_SHA, [mine], str(gen), replace_source="inspect/gsm8k=k1")
    assert rec["status"] == "OK" and rec["replace_source"] == "inspect/gsm8k=k1"
    # the sweep: other trees' stores are SKIPPED (named), legacy ones refused
    calls.clear()
    rec = L.merge_summaries("f6cf462", None, str(gen))
    assert calls == [([mine], str(t))]
    assert rec["skipped_other_producer"] == {foreign: OTHER_SHA} and rec["refused_other_producer"] == {}
    assert rec["refused_legacy_producer"] == [legacy] and rec["status"] == "FAILED"
    # no tree, or a tree without its stamp
    assert L.merge_summaries("nope", [mine], str(gen))["status"] == "NO_TREE"
    (t / C.TREE_SHA_FILENAME).unlink()
    rec = L.merge_summaries("f6cf462", [mine], str(gen))
    assert rec["status"] == "NO_TREE_SHA" and C.TREE_SHA_FILENAME in rec["error"]


# --------------------------------------------------------------------------- #
# the image recipe                                                             #
# --------------------------------------------------------------------------- #


def test_client_venv_recipe_is_hashed_pinned_and_fails_closed(L):
    cmds = L.client_venv_commands()
    assert cmds[0] == f"uv venv --python {C.CLIENT_PYTHON} {C.CLIENT_VENV}"
    # the unhashable VCS line — EXACTLY the contracts' allow-list (the josejg
    # IFEval checker) — split out by name
    assert cmds[1] == (f"grep -v -E '^(instruction-following-eval) @ ' "
                       f"{L.CLIENT_LOCK_REMOTE} > {L.CLIENT_LOCK_HASHED}")
    install = cmds[2]
    assert "--no-deps" in install and "--require-hashes" in install
    assert install.endswith(f"-r {L.CLIENT_LOCK_HASHED}")
    vcs = cmds[3:4]
    assert [c.split("'")[1] for c in vcs] == list(C.CLIENT_VCS_PINS)
    assert f"git+{C.IFEVAL_CHECKER_GIT}@{C.IFEVAL_CHECKER_REV}" in vcs[0] and "--no-deps" in vcs[0]
    assert len(C.IFEVAL_CHECKER_REV) == 40                              # a full commit, never a branch
    check = cmds[4]
    for pkg, ver in (("inspect_ai", C.INSPECT_AI_PIN), ("aiohttp", "3.14.3"), ("transformers", "5.15.0")):
        assert f"{pkg}.__version__ == '{ver}'" in check
    # the driver's own heavy imports load in the client venv, at the lock's pins; the
    # removed tier-1 harness is neither imported nor asserted
    assert "import torch, scipy;" in check and "lm_eval" not in check and "tinyBenchmarks" not in check
    for pkg in ("torch", "scipy"):
        assert f"v('{pkg}') == '{C.CLIENT_TOP_LEVEL_PINS[pkg]}'" in check
    assert f"v('inspect_evals') == '{C.INSPECT_EVALS_PIN}'" in check
    # P7: the checker and its two deps import in the client venv, or the build fails there
    assert "import instruction_following_eval, langdetect, immutabledict;" in check
    assert f"v('immutabledict') == '{C.CLIENT_TOP_LEVEL_PINS['immutabledict']}'" in check
    assert len(cmds) == 5
    # nothing floats: no bare `pip install <name>` without a pin in this file
    src = LAUNCHER.read_text()
    assert f'pip_install(f"uv=={{UV_PIN}}")' in src and re.fullmatch(r"\d+\.\d+\.\d+", L.UV_PIN)
    assert "--require-hashes" in src and "add_local_file" in src
    assert (REPO / C.CLIENT_LOCK_TXT).is_file()


def test_preflight_identity_checks_are_module_level_and_take_a_module_inventory(L, tmp_path, monkeypatch):
    """``lock_matches`` and ``launcher_modules_match_tree`` were closures inside
    ``preflight``; a sibling launcher must be able to run the SAME checks before
    ITS generation, so they are module-level now —
    ``check_client_lock_matches_tree`` and
    ``check_launcher_modules_match_tree(tree, modules=...)`` — and preflight
    calls them under the unchanged check names. ``modules`` is load-bearing: a
    sibling passes an EXTENDED inventory, and a module outside the default list
    is checked only when named."""
    import inspect as _inspect
    import sisteer
    mounted = Path(sisteer.__file__).resolve().parent
    tree = tmp_path / "tree"
    # a staged tree whose launcher modules equal the mounted ones byte for byte …
    for rel in L.LAUNCHER_PROCESS_MODULES:
        dst = tree / "src" / "sisteer" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((mounted / rel).read_bytes())
    # … plus one module OUTSIDE the default inventory, edited on the mounted side only
    extra = "capeval/serving/usage_ledger.py"
    assert extra not in L.LAUNCHER_PROCESS_MODULES
    staged_extra = tree / "src" / "sisteer" / extra
    staged_extra.write_bytes((mounted / extra).read_bytes() + b"\n# not the mounted bytes\n")
    # the lock: the image's mounted copy vs the tree's copy, both == the repo's
    rel = C.CLIENT_LOCK_TXT
    (tree / rel).parent.mkdir(parents=True, exist_ok=True)
    (tree / rel).write_bytes((REPO / rel).read_bytes())
    image_copy = tmp_path / Path(rel).name
    image_copy.write_bytes((REPO / rel).read_bytes())
    monkeypatch.setattr(L, "CLIENT_LOCK_REMOTE", str(image_copy))
    # an equal lock → the digest, under the key the preflight report has always carried
    c = L.check_client_lock_matches_tree(str(tree))
    assert c == {"client_lock_sha256": hashlib.sha256((REPO / C.CLIENT_LOCK_TXT).read_bytes()).hexdigest()}
    # the default inventory passes and reports exactly LAUNCHER_PROCESS_MODULES, in order
    m = L.check_launcher_modules_match_tree(str(tree))
    assert m["mounted_from"] == str(mounted)
    assert tuple(m["sha256_12"]) == L.LAUNCHER_PROCESS_MODULES
    sig = _inspect.signature(L.check_launcher_modules_match_tree)
    assert list(sig.parameters) == ["tree", "modules"]
    assert sig.parameters["modules"].default is L.LAUNCHER_PROCESS_MODULES
    # the EXTENDED inventory sees the mounted-only difference and names the file
    ext = L.LAUNCHER_PROCESS_MODULES + (extra,)
    with pytest.raises(AssertionError, match=re.escape(extra)) as ei:
        L.check_launcher_modules_match_tree(str(tree), modules=ext)
    assert "differs from the staged tree" in str(ei.value) and str(tree) in str(ei.value)
    # once the staged copy matches, the extended inventory passes and reports the extra
    staged_extra.write_bytes((mounted / extra).read_bytes())
    assert tuple(L.check_launcher_modules_match_tree(str(tree), modules=ext)["sha256_12"]) == ext
    # a moved lock refuses — the client message byte-for-byte as before
    (tree / C.CLIENT_LOCK_TXT).write_bytes(b"moved\n")
    with pytest.raises(AssertionError, match=r"image lock [0-9a-f]{12} != staged tree lock [0-9a-f]{12} — rebuild"):
        L.check_client_lock_matches_tree(str(tree))
    # preflight calls the module-level functions under the OLD check names; no
    # closure of those names remains inside its body
    src = LAUNCHER.read_text()
    body = src[src.index("def preflight("):src.index("    def datasets():")]
    assert 'chk("client_lock_matches_tree", lambda: check_client_lock_matches_tree(tree))' in body
    assert 'chk("launcher_modules_match_tree", lambda: check_launcher_modules_match_tree(tree))' in body
    assert "def lock_matches" not in body and "def launcher_modules_match_tree" not in body
    # and the lifted functions sit at module level, after the inventory they default to
    assert src.index("LAUNCHER_PROCESS_MODULES: Tuple[str, ...] = (") \
        < src.index("def check_client_lock_matches_tree(") \
        < src.index("def check_launcher_modules_match_tree(") < src.index("def preflight(")


_PINNED = re.compile(r"^[A-Za-z0-9_.\-]+==\d[^\s]*$")


def _is_pinned_spec(spec: str) -> bool:
    return bool(_PINNED.match(spec)) or (spec.startswith("https://") and "#sha256=" in spec)


def test_every_image_install_is_pinned(L):
    """R8: the serving image's helper packages (pandas, pyarrow, aiohttp,
    pyyaml, numpy, huggingface_hub, ninja) were unpinned. Now every
    requirement the image installs is an exact pin (or a sha256'd wheel
    URL); the pins shared with the hashed client lock EQUAL the lock's
    versions; ninja (not in the lock) is an exact release >= 1 week old;
    and an AST sweep of EVERY ``pip_install`` call in the file refuses a
    bare name."""
    specs = L.image_install_specs()
    assert specs and all(_is_pinned_spec(s) for s in specs), [s for s in specs if not _is_pinned_spec(s)]
    assert set(L.serving_image_pin_specs()) <= set(specs) and f"uv=={L.UV_PIN}" in specs
    assert set(L.SERVING_IMAGE_PINS) == {"pandas", "pyarrow", "aiohttp", "pyyaml", "numpy",
                                         "huggingface_hub", "ninja"}
    lock = {}
    for line in (REPO / C.CLIENT_LOCK_TXT).read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s\\]+)", line)
        if m:
            lock[m.group(1).lower().replace("_", "-")] = m.group(2)
    for name, ver in L.SERVING_IMAGE_PINS.items():
        assert re.fullmatch(r"\d+(\.\d+)+", ver), (name, ver)
        key = name.lower().replace("_", "-")
        if key == "ninja":
            assert key not in lock and ver == "1.13.0"        # 2025-08-11; 1.13.2 (2026-08-30) too fresh
        elif name in L.SERVING_IMAGE_PINS_OFF_LOCK:
            # the ONE documented deviation: vllm 0.25.1 cannot install the lock's numpy
            assert name == "numpy" and ver == "2.3.5" and lock[key] == "2.5.2"
            assert "numpy<2.4" in L.SERVING_IMAGE_PINS_OFF_LOCK[name]
        else:
            assert lock[key] == ver, (name, ver, lock.get(key))
    assert set(L.SERVING_IMAGE_PINS_OFF_LOCK) == {"numpy"}
    assert L.SERVING_IMAGE_PINS["aiohttp"] == C.CLIENT_TOP_LEVEL_PINS["aiohttp"]
    # the AST sweep: every pip_install argument is a pinned literal, a pinned
    # f-string, or a star over the spec lists the image is built from
    tree = ast.parse(LAUNCHER.read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "pip_install"]
    assert len(calls) == 3, "the image has three pip_install groups (stack, transformers, uv)"
    starred_ok = {"SERVING_STACK_SPECS", "TRANSFORMERS_SPECS"}
    for call in calls:
        assert call.args, ast.unparse(call)
        for arg in call.args:
            if isinstance(arg, ast.Constant):
                assert isinstance(arg.value, str) and _is_pinned_spec(arg.value), arg.value
            elif isinstance(arg, ast.JoinedStr):
                assert any(isinstance(v, ast.Constant) and "==" in str(v.value) for v in arg.values), ast.unparse(arg)
            elif isinstance(arg, ast.Starred):
                v = arg.value
                ok = ((isinstance(v, ast.Name) and v.id in starred_ok)
                      or (isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
                          and v.func.id == "serving_image_pin_specs"))
                assert ok, ast.unparse(arg)
            else:
                pytest.fail(f"unpinnable pip_install argument: {ast.unparse(arg)}")
    for spec in L.SERVING_STACK_SPECS + L.TRANSFORMERS_SPECS:
        assert _is_pinned_spec(spec), spec
    # and no bare package name survives as a requirement string anywhere in the file
    src = LAUNCHER.read_text()
    for name in L.SERVING_IMAGE_PINS:
        assert re.search(rf'"{name}"\s*[,)]', src) is None, f'bare "{name}" requirement'


def test_stage_pins_exist_in_the_repo_and_the_unpinned_list_is_recorded(L):
    pins = yaml.safe_load((REPO / "configs/capeval/pins.yaml").read_text())["datasets"]
    missing = [k for k in L.STAGE_PINNED_KEYS if k not in pins]
    assert not missing, f"STAGE_PINNED_KEYS not in pins.yaml: {missing}; have {sorted(pins)}"
    for k in L.STAGE_PINNED_KEYS:
        assert re.fullmatch(r"[0-9a-f]{40}", pins[k]["revision"]), k     # content-addressed
    assert (C.MATH500_DATASET, "dataset") in L.STAGE_UNPINNED
    src = LAUNCHER.read_text()
    assert "resolved_revision" in src and "stage_manifest.json" in src


# --------------------------------------------------------------------------- #
# the dry run                                                                  #
# --------------------------------------------------------------------------- #


def test_dry_run_prints_the_plan_without_modal(L, capsys):
    # native-serving-default: the launcher's default is the native engine now; this test pins the LENS
    # launch (a PSR cell / the historical composition), so it states the path explicitly
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, dry_run=True)
    out = capsys.readouterr().out
    for name in C.CONTAINER_NAMES.values():
        assert f"== {name}" in out
    assert "nothing launched" in out and "4 GPU configuration container(s)" in out
    S = sg.identity_seqs(sg.PETRI_MAX_MODEL_LEN, model_key=C.MODEL_SLUG)
    # PW11: every default configuration's first key is MEASURED, so its pass 1 is
    # a measurement re-run at the footprint (window measurement_footprint); no
    # default configuration reads provisional any more
    assert out.count(f"{sg.PETRI_MAX_MODEL_LEN} x {S} (window {sg.MEASUREMENT_SOURCE}; S model_budget") == len(C.CONTAINER_NAMES)
    assert sg.PROVISIONAL_SOURCE not in out
    assert f"{vs.DEFAULT_MAX_NUM_SEQS} ({sg.MEASUREMENT_SOURCE})" not in out   # the Llama-budget S is gone
    # every configuration's standup + driver argv show the served S (E7)
    assert out.count(f"--max-num-seqs {S} --model-key {C.MODEL_SLUG} ") == len(C.CONTAINER_NAMES)
    assert "--allow-over-budget" not in out
    assert out.count(f"--concurrency {S} ") == len(C.CONTAINER_NAMES)
    assert "589,824 tokens = 1.50x the certified 393,216" in out
    assert "--reasoning-parser gemma4" in out
    assert "--tool-calls" not in out                # U9 item 6: BFCL is out of the battery
    # U10 item 4: every default configuration names the OR-Bench item set it
    # files, without printing 200 positions at the operator
    assert out.count("   or_bench item set: item_set='subset' n=200 of=1319 seed=20260906 "
                     "(frozen, seeded; hashed by position") == len(C.CONTAINER_NAMES)
    # the human-readable line names the set without printing 200 indices at the
    # operator; the `resume identity:` JSON below it carries them in full, on
    # purpose — an identity dump that elides a hashed field misleads later
    assert not any("positions" in ln for ln in out.splitlines()
                   if ln.startswith("   or_bench item set:"))
    assert "--thinking on" in out and "--ledger-url http://127.0.0.1:<ledger>/v1" in out
    # the exact modal run line, with the operator note (a clean detached checkout at --sha)
    assert (f"[tinycap] argv: TINYCAP_APP_NAME={L.APP_NAME} modal run scripts/modal/tinycap_served.py "
            f"--sha abc1234 --app-name {L.APP_NAME} --pass1 ") in out
    assert "clean detached checkout at abc1234" in out
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", smoke=True, dry_run=True)
    out = capsys.readouterr().out
    assert "1 GPU configuration container(s)" in out and "limit 5, inspect_limit 5" in out
    assert "tinycap-e4b-psrkl_safety-think_off" in out and "prompted" not in out.split("==")[1].split("\n")[0].replace("prompted=False", "")
    # the argv line names the smoke's isolated root and caps exactly as they run
    assert f"--sha abc1234 --app-name {L.APP_NAME} --smoke --out-suffix smoke --limit 5 --inspect-limit 5" in out
    with pytest.raises(SystemExit, match="--sha"):
        L.main()
    # pass 2: a configuration pooling several sub-entries (gen_long + agent) -> R5
    # refusal first (no control pools into a provisional entry since the
    # loglikelihood family was removed; that refusal is pinned through
    # configuration_geometry) ...
    with pytest.raises(L.Pass2Unpartitioned):
        L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass2=True, cells=C.CELL_PSRKL_SAFETY,
               controls="gsm8k,bfcl_200", dry_run=True)
    capsys.readouterr()
    # ... while a single-key gsm8k list resolves its MEASURED window: the pass-2
    # dry run prints the plan at 4608 x S (gemma4e4b's budget at the measured
    # window), standup and driver sized to it
    S2 = sg.BUDGET_BY_MODEL[C.MODEL_SLUG].seqs_at(4608)
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass2=True, cells=C.CELL_PSRKL_SAFETY, controls="gsm8k", dry_run=True)
    out = capsys.readouterr().out
    assert f"geometry: capability:gen_long 4608 x {S2} (window table; S model_budget" in out
    assert f"--max-model-len 4608 --max-num-seqs {S2} --model-key gemma4e4b" in out
    assert f"--concurrency {S2} " in out and "--pass2" in out and sg.MEASUREMENT_SOURCE not in out
    # --merge in a dry run prints the plan and launches nothing
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", merge=True, merge_cell_dirs="/vol/a,/vol/b", dry_run=True)
    out = capsys.readouterr().out
    assert "--merge --merge-cell-dirs /vol/a,/vol/b" in out
    # PW10: the dry run names the two trees — the staged --sha EMITS, this tree WRITES
    assert "would EMIT each store's payload through the staged tree abc1234's completion" in out
    assert "WRITE it through this tree's completion.merge_emitted_payload" in out
    assert "0 GPU configuration container(s)" in out and "nothing launched" in out
    # PW8 (E16 A3): --replace-source names entries; it is validated before any
    # Modal call and means nothing without --merge
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", merge=True, dry_run=True, replace_source="inspect/gsm8k=k1")
    out = capsys.readouterr().out
    assert "--merge --replace-source inspect/gsm8k=k1" in out and "for each NAMED entry" in out
    assert "naming the STANDING store itself, REFRESH the entry" in out             # PW16 (E24)
    with pytest.raises(SystemExit, match="pass --merge"):
        L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, dry_run=True, replace_source="inspect/gsm8k=k1")
    with pytest.raises(SystemExit, match="not <entry>=<cell_key>"):
        L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", merge=True, dry_run=True, replace_source="bogus")
    assert set(L.main.__code__.co_varnames[:L.main.__code__.co_argcount]) >= {
        "sha", "stage", "pf", "smoke", "pass1", "pass2", "cells", "modes", "controls",
        "allow_extra", "merge", "merge_cell_dirs", "gen_root"}
    assert "hf_spotcheck" not in L.main.__code__.co_varnames[:L.main.__code__.co_argcount]
    assert set(C.LAUNCHER_ENTRYPOINT_KWARGS) <= set(
        L.main.__code__.co_varnames[:L.main.__code__.co_argcount])


def test_launch_argv_is_the_exact_modal_run_line(L):
    base = ["modal", "run", "scripts/modal/tinycap_served.py", "--sha", "abc"]
    assert L.launch_argv("abc", pass1=True) == base + ["--pass1"]
    # a NON-default --samples is spelled out (the default is MATH500_SAMPLES_PER_ITEM = 1)
    assert L.launch_argv("abc", pass1=True, cells="x", modes="on", controls="gsm8k", limit=5,
                         inspect_limit=2, samples=4, out_suffix="s", allow_extra=True) == base + [
        "--pass1", "--allow-extra", "--cells", "x", "--modes", "on", "--controls", "gsm8k",
        "--out-suffix", "s", "--limit", "5", "--inspect-limit", "2", "--samples", "4"]
    assert "--samples" not in L.launch_argv("abc", pass1=True, samples=C.MATH500_SAMPLES_PER_ITEM)
    assert L.launch_argv("abc", merge=True, merge_cell_dirs="/a,/b") == base + [
        "--merge", "--merge-cell-dirs", "/a,/b"]
    # PW7: the operator's ambiguous-source decision is IN the recorded line —
    # per entry since PW8 (E16 A3), canonically ordered, validated
    assert L.launch_argv("abc", merge=True, replace_source="inspect/gsm8k=k1,evalplus/mbpp_plus=k2") == base + [
        "--merge", L.MERGE_REPLACE_FLAG, "evalplus/mbpp_plus=k2,inspect/gsm8k=k1"]
    assert L.launch_argv("abc", merge=True, replace_source="") == base + ["--merge"]
    with pytest.raises(ValueError, match="not <entry>=<cell_key>"):
        L.launch_argv("abc", merge=True, replace_source="bogus")
    assert L.launch_argv("abc", smoke=True) == base + ["--smoke"]     # the default samples are not repeated
    # PW17: the in-depth flag is IN the recorded line (it changes every driver's --controls)
    assert L.launch_argv("abc", pass1=True, in_depth=True) == base + ["--pass1", "--in-depth"]
    assert L.launch_argv("abc", smoke=True, in_depth=True, allow_extra=True) == base + [
        "--smoke", "--in-depth", "--allow-extra"]
    # E29 / PW18b: the lens reduction mode is spelled out whenever it is not the default
    assert L.launch_argv("abc", pass1=True, reduction_mode="per_row") == base + ["--pass1"]
    assert L.launch_argv("abc", pass1=True, reduction_mode="stacked") == base + [
        "--reduction-mode", "stacked", "--pass1"]
    assert L.LAUNCHER_SCRIPT_REL == str(LAUNCHER.relative_to(REPO))


# --------------------------------------------------------------------------- #
# container-safe module load + smoke isolation (2026-09-03 operator fixes)     #
# --------------------------------------------------------------------------- #


def test_repo_local_is_found_by_walking_up_never_by_depth(L):
    """Modal mounts the script flat at /root/ (no ``parents[2]``): the root is
    FOUND (src/sisteer + scripts/modal beside each other), and the fallback
    for a rootless location is the script's own directory, not an error."""
    assert L.REPO_LOCAL == REPO
    assert L._repo_local() == REPO
    src = LAUNCHER.read_text()
    assert "parents[2]" not in src.replace("``parents[2]``", "")   # only the docstring names it
    # the launcher process's own sisteer rides the image as a local python source,
    # and preflight pins the mounted modules to the staged tree
    assert '.add_local_python_source("sisteer")' in src
    assert "launcher_modules_match_tree" in src
    for rel in L.LAUNCHER_PROCESS_MODULES:
        assert (REPO / "src" / "sisteer" / rel).is_file(), rel


def test_smoke_writes_under_an_isolated_root_and_takes_a_controls_override(L, capsys):
    # native-serving-default: the launcher's default is the native engine now; this test pins the LENS
    # launch (a PSR cell / the historical composition), so it states the path explicitly
    (s,) = L.smoke_plan(["gsm8k"])
    assert s["controls"] == ["gsm8k"] and not s["tool_calls"]
    (d,) = L.smoke_plan()
    assert d["controls"] == list(L.CONTROLS_FITTED)
    iso = L.isolate_out_dirs([d], "smoke")
    assert iso[0]["out_dir"] == d["out_dir"] + "/smoke" and d["out_dir"].endswith("/thinking_off")
    assert L.isolate_out_dirs([d], "")[0]["out_dir"] == d["out_dir"]
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", smoke=True, controls="gsm8k", dry_run=True)
    out = capsys.readouterr().out
    assert f"--out-root {d['out_dir']}/smoke " in out            # the driver's root is the isolated one
    assert "--controls gsm8k " in out
    assert "math500" not in out.replace("(none; --in-depth adds math500, gsm8k_full)", "")
    # `gsm8k` is a STANDARD token of the fitted battery, so the split files it there
    assert "   battery: standard = gsm8k; in-depth = (none; --in-depth adds math500, gsm8k_full)" in out
    assert "out roots under …/smoke/" in out
    # a full pass-1 never touches the smoke root and keeps its argv
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, dry_run=True)
    out = capsys.readouterr().out
    assert "/smoke" not in out and out.count(f"--model-key {C.MODEL_SLUG} ") == len(C.CONTAINER_NAMES)
    # a bad --controls token fails locally, before any container
    with pytest.raises(C.ControlNameError):
        L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", smoke=True, controls="gsm8kx", dry_run=True)


def test_only_the_stage_function_carries_the_hf_token_secret(L):
    """Authenticated dataset downloads happen in the network-ON stage function
    ONLY; the GPU containers run HF_HUB_OFFLINE=1 and carry no secret."""
    src = LAUNCHER.read_text()
    assert src.count("modal.Secret.from_name(") == 1 and L.HF_TOKEN_SECRET == "hf-token"
    ref = src.index("modal.Secret.from_name(")
    assert 0 < src.index("def stage(") - ref < 300          # the one reference is stage's decorator
    for fn in ("run_configuration", "preflight"):
        head = src.rfind("@app.function(", 0, src.index(f"def {fn}("))
        assert "secrets" not in src[head:src.index(f"def {fn}(")], fn
    # the 429 backoff retries ONLY the hub's rate limit; a gated 401 propagates at once
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429 Too Many Requests: maximum queue size reached")
        return "ok"
    assert L._with_hub_retry(flaky, what="t", delay_s=0.0) == "ok" and calls["n"] == 3

    def gated():
        raise RuntimeError("401 Client Error: gated repo")
    with pytest.raises(RuntimeError, match="401"):
        L._with_hub_retry(gated, what="t", delay_s=0.0)
    # the unpinned-datasets step RECORDS a gated repo (walledai/XSTest) as
    # unavailable_gated instead of failing the whole stage (6b); other errors still fail it
    assert L._is_hub_gated(RuntimeError("401 Client Error: gated repo"))
    err = type("GatedRepoError", (Exception,), {})("Cannot access gated repo for url https://x")
    err.response = SimpleNamespace(status_code=403)
    assert L._is_hub_gated(err)
    assert L._is_hub_gated(ValueError("Dataset 'walledai/XSTest' is a gated dataset on the Hub."))
    assert not L._is_hub_gated(RuntimeError("404 Client Error: Repository Not Found"))
    assert not L._is_hub_gated(ConnectionError("Couldn't reach the Hub"))
    i = src.index("def unpinned():")
    body = src[i:src.index('step("unpinned_datasets_resolved", unpinned)')]
    assert "_is_hub_gated(ex)" in body and "STATUS_UNAVAILABLE_GATED: gated" in body


def test_preflight_probes_the_driver_parser_and_runs_the_pin_check_by_argv(L):
    """The client-venv pin check is ONE snippet shared by the image step and
    preflight, executed by argv (its own quotes broke a shell re-quote); the
    driver's flags are probed through argparse's unrecognized-arguments error,
    never by reading ``--help`` text (whose formatting the driver's help crashes)."""
    code = L.client_venv_check_code()
    assert L.client_venv_commands()[-1] == f'{L.CLIENT_PY} -c "{code}"'  # the check is the LAST step
    assert "inspect_ai" in code and "'0.3.260'" in code               # quotes live INSIDE the snippet
    src = LAUNCHER.read_text()
    assert "shell=True" not in src
    argv = L.driver_probe_argv("/tree")
    assert argv[:3] == [L.CLIENT_PY, "/tree/scripts/served_capability_run.py", "generate"]
    for flag in C.TINYCAP_DRIVER_FLAGS:
        assert flag in argv and argv[argv.index(flag) + 1] == L.DRIVER_PROBE_VALUES[flag]
    assert argv[-1] == L.DRIVER_PROBE_SENTINEL and argv.count(L.DRIVER_PROBE_SENTINEL) == 1
    assert L.unrecognized_flags("usage: x\nx: error: unrecognized arguments: --tinycap-preflight-probe\n") == [
        L.DRIVER_PROBE_SENTINEL]
    # a real flag missing on the tree shows up beside the sentinel, its dummy value dropped
    assert L.unrecognized_flags("error: unrecognized arguments: --thinking off --tinycap-preflight-probe") == [
        "--thinking", L.DRIVER_PROBE_SENTINEL]
    assert L.unrecognized_flags("Traceback ... TypeError: %i format") is None


def test_letter_read_window_is_one_number_on_server_and_driver_and_wider_than_the_legacy_64(L, capsys):
    # native-serving-default: the launcher's default is the native engine now; this test pins the LENS
    # launch (a PSR cell / the historical composition), so it states the path explicitly
    """--max-logprobs (server) and --top-logprobs (driver) are the same number
    (a mismatch makes vLLM 400 the letter reads), and it is wider than the
    legacy 64 that could not read steered letters in the 2026-09-03 smoke.

    The DEFAULT model's width is unchanged by PW-nanfix-2 — 1024, exactly what
    the whole Gemma campaign was served and read at."""
    assert L.TOP_LOGPROBS == C.LETTER_TOP_LOGPROBS_DEFAULT
    assert C.LETTER_TOP_LOGPROBS_DEFAULT == 1024
    assert L.TOP_LOGPROBS > 64
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, dry_run=True)
    out = capsys.readouterr().out
    n = len(C.CONTAINER_NAMES)
    assert out.count(f"--max-logprobs {L.TOP_LOGPROBS} ") == n
    assert out.count(f"--top-logprobs {L.TOP_LOGPROBS} ") == n
    assert "--max-logprobs 64 " not in out and "--top-logprobs 64 " not in out


def test_the_letter_read_window_is_a_per_model_registry_value_on_both_flags(L, capsys):
    # native-serving-default: the launcher's default is the native engine now; this test pins the LENS
    # launch (a PSR cell / the historical composition), so it states the path explicitly
    """§PW-nanfix-2. All three llama3b cells died in their FIRST control at
    K=1024 ("letter(s) ['B', 'C'] absent from the returned top-K — cannot
    score; raise --top-logprobs (and the server's --max-logprobs) rather than
    guessing"), so the width is a REGISTRY value: it is a property of the model
    and its objects, not of the harness. The launcher must put the row's number
    on BOTH flags, or the server 400s every letter read."""
    assert C.LETTER_TOP_LOGPROBS_WIDE > C.LETTER_TOP_LOGPROBS_DEFAULT
    for slug, want in (("llama3b", C.LETTER_TOP_LOGPROBS_WIDE),
                       ("llama8b", C.LETTER_TOP_LOGPROBS_DEFAULT),
                       ("llama8b-x2", C.LETTER_TOP_LOGPROBS_WIDE),
                       ("gemma4e4b-x2", C.LETTER_TOP_LOGPROBS_WIDE)):
        L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, dry_run=True, model=slug,
               cells="none", modes="off", controls="gsm8k")
        out = capsys.readouterr().out
        assert f"--max-logprobs {want} " in out, slug
        assert f"--top-logprobs {want} " in out, slug
        assert f"letter_top_logprobs={want}" in out, slug
        other = (C.LETTER_TOP_LOGPROBS_DEFAULT if want == C.LETTER_TOP_LOGPROBS_WIDE
                 else C.LETTER_TOP_LOGPROBS_WIDE)
        assert f"--max-logprobs {other} " not in out and f"--top-logprobs {other} " not in out


def test_the_composed_standup_and_driver_argv_always_agree_on_the_row_width(L):
    """The two flags are ONE number per configuration, read off the row —
    never the module constant, which is only the default a row takes."""
    for slug in C.MODEL_SLUGS:
        row = C.model_row(slug)
        cfg = L.plan_configurations(cells=["none"], modes=["off"], controls=["gsm8k"],
                                    model=slug)[0]
        geom, _ = L.served_geometry(cfg, 1)
        standup = L._standup_cmd("/tree", "/tree/cfg.yaml", cfg["cell"], row.objects_dir,
                                 Path("/srv"), Path("/srv/STOP"), geom, row.top_logprobs,
                                 cfg["standup_extra"])
        drv = L._driver_cmd("/tree", cfg, {"proxy_url": "u", "routing_table": "r"}, None, geom,
                            controls=cfg["controls"], limit=0, inspect_limit=0, samples=4,
                            cfg_path="/tree/cfg.yaml", vllm_log=None)
        served = standup[standup.index("--max-logprobs") + 1]
        asked = drv[drv.index("--top-logprobs") + 1]
        assert served == asked == str(row.top_logprobs), slug


def test_stage_primes_inspect_datasets_and_the_driver_reads_the_staged_evalplus_archives(L, tmp_path):
    """The Inspect family shares the datasets-cache trap (inspect_ai bypasses its
    own disk cache whenever a revision is pinned): stage primes through the
    exact lane + inspect_evals loaders, preflight re-runs them offline, and a
    gated repo is recorded as unavailable rather than failing the step. Served
    EvalPlus generation reads the sandbox's sha-pinned staged archives through
    evalplus's own override env vars."""
    argv = L.inspect_loader_argv("/tree")
    # -P: the cwd is never prepended to sys.path for the -c snippet (the stage of
    # 0a7eb982 imported the image's /root/sisteer ahead of the staged tree)
    assert argv[0] == L.CLIENT_PY and argv[1:3] == ["-P", "-c"]
    assert argv[4:] == ["/tree", "online", L.DATA_DIR_ALIASES_KEY]
    assert L.inspect_loader_argv("/tree", offline=True)[4:] == ["/tree", "offline", L.DATA_DIR_ALIASES_KEY]
    compile(argv[3], "<inspect-loaders>", "exec")
    for name in ("lane.math500_selection", "inspect_evals.bfcl",
                 "inspect_evals.xstest", "inspect_evals.gsm8k", "pins.gsm8k_platinum", '"gated"',
                 '"gated" in msg.lower()',   # datasets says "is a gated dataset", huggingface_hub "GatedRepoError"
                 "alias_data_dir_caches(_DATA_DIR_LOADS, create=online)",
                 # the thin BigCodeBench task through load_repo_task's ONE
                 # positional argument (its dataset load primes the pinned
                 # parquet online, resolves offline), the two verified pins,
                 # and the staged-tree assertion
                 'go("lane.bcb_200_pin", lambda: CT.load_bcb_pin(I.REPO_ROOT))',
                 'go("lane.bfcl_200_pin", lambda: CT.load_bfcl_pin(I.REPO_ROOT))',
                 'I.load_repo_task("bigcodebench_200")(ids=CT.load_bcb_pin(I.REPO_ROOT)["ids"])',
                 # capeval-full-tasks: the two FULL pins and the full thin task, the same way
                 'go("lane.bcb_full_pin", lambda: CT.load_bcb_full_pin(I.REPO_ROOT))',
                 'go("lane.bfcl_full_pin", lambda: CT.load_bfcl_full_pin(I.REPO_ROOT))',
                 'I.load_repo_task("bigcodebench_full")(ids=CT.load_bcb_full_pin(I.REPO_ROOT)["ids"])',
                 "assert Path(I.INSPECT_TASKS_DIR).resolve().is_relative_to(tree.resolve())"):
        assert name in argv[3], name
    # load_repo_task takes ONE positional argument (a `repo_root=` kwarg was a TypeError)
    assert 'load_repo_task("bigcodebench_200", ' not in argv[3]
    src = LAUNCHER.read_text()
    assert 'step(INSPECT_PRIME_STEP, inspect_cache)' in src
    assert 'chk("inspect_datasets_offline_in_client_venv", inspect_offline)' in src
    assert 'chk("evalplus_archives_staged", evalplus_archives)' in src
    # the stage fails its step when an alias could not be laid down (never a preflight surprise)
    i = src.index("def inspect_cache():")
    assert "bad_aliases" in src[i:i + 1200] and "data_dir_cache_aliases" in src[i:i + 1200]
    # evalplus override env from the repo's own pins, against a fake archive dir
    pins = json.loads((REPO / L.EVALPLUS_PINS_RELPATH).read_text())["datasets"]
    with pytest.raises(FileNotFoundError, match="staged EvalPlus archives missing"):
        L.evalplus_override_env(str(REPO), archives_dir=str(tmp_path))
    for pin in pins.values():
        (tmp_path / pin["gz_filename"]).write_bytes(b"x")
    env = L.evalplus_override_env(str(REPO), archives_dir=str(tmp_path))
    assert set(env) == {pin["override_env"] for pin in pins.values()} == {
        "HUMANEVAL_OVERRIDE_PATH", "MBPP_OVERRIDE_PATH"}
    assert all(v.startswith(str(tmp_path)) for v in env.values())
    assert L.EVALPLUS_ARCHIVES_DIR == f"{C.VOL_CACHE_ROOT}/evalplus"
    assert "**evalplus_env" in src and '"EVALPLUS_ARCHIVES_MISSING"' in src


def test_data_dir_cache_alias_copies_the_online_cache_to_the_offline_key(L, tmp_path, monkeypatch):
    """6(a): datasets 5.0.1 keys a ``data_dir=`` load by a hash of the
    resolved data_files online and by ``<config>-data_dir=<dir>`` offline,
    so inspect_evals' ``gsm8k`` (openai/gsm8k, data_dir="main", pinned
    revision) could not resolve in the offline containers after a healthy
    prime. The alias snippet records every data_dir load made through
    ``datasets.load_dataset`` (the attribute inspect_ai calls) and, ONLINE
    only, copies the produced cache dir to the offline key — the REAL
    ``BuilderConfig.create_config_id`` derives that key here."""
    import datasets

    real = (tmp_path / "hf" / "datasets" / "openai___gsm8k" / "default-8cbc416ed05c3005"
            / "0.0.0" / "cc7b047b6e5bb11b4f1af84efc572db110a51b3c")
    real.mkdir(parents=True)
    (real / "gsm8k-test.arrow").write_bytes(b"arrow-bytes")
    (real / "dataset_info.json").write_text('{"config_name": "default"}')

    class FakeDS:
        cache_files = ({"filename": str(real / "gsm8k-test.arrow")},)

    calls = []

    def fake_load(path, name=None, data_dir=None, *a, **kw):
        calls.append((path, name, data_dir, kw))
        return FakeDS()

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    # the snippet under test, loaded as a module the way the loader code runs it
    snippet = tmp_path / "alias_snippet.py"
    snippet.write_text("from pathlib import Path\n" + L.DATA_DIR_ALIAS_CODE)
    spec = importlib.util.spec_from_file_location("alias_snippet_under_test", snippet)
    ns = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ns)
    # the wrapper sits on the module attribute inspect_ai calls, forwards
    # every call, and records ONLY data_dir loads of hub repos
    datasets.load_dataset(path="openai/gsm8k", name=None, data_dir="main", split="test",
                          revision="cc7b047b6e5bb11b4f1af84efc572db110a51b3c", trust_remote_code=False)
    datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
    assert len(calls) == 2 and calls[1][:3] == ("HuggingFaceH4/MATH-500", None, None)
    [rec] = ns._DATA_DIR_LOADS
    assert (rec["path"], rec["name"], rec["data_dir"], rec["revision"]) == (
        "openai/gsm8k", None, "main", "cc7b047b6e5bb11b4f1af84efc572db110a51b3c")
    alias = (tmp_path / "hf" / "datasets" / "openai___gsm8k" / "default-data_dir=main"
             / "0.0.0" / "cc7b047b6e5bb11b4f1af84efc572db110a51b3c")
    # offline (preflight): the absent alias is REPORTED, nothing is written
    [r] = ns.alias_data_dir_caches(ns._DATA_DIR_LOADS, create=False)
    assert r["offline_config_id"] == "default-data_dir=main"          # datasets' own derivation
    assert r["alias"] == str(alias) and r["target"] == str(real)
    assert r["present"] is False and r["created"] is False and "offline" in r["err"]
    assert not alias.exists()
    # online (stage): a COPY lands at the offline key, byte-equal, never a symlink
    [r] = ns.alias_data_dir_caches(ns._DATA_DIR_LOADS, create=True)
    assert r["created"] is True and r["present"] is True
    assert (alias / "gsm8k-test.arrow").read_bytes() == b"arrow-bytes" and not alias.is_symlink()
    assert (alias / "dataset_info.json").is_file()
    [r] = ns.alias_data_dir_caches(ns._DATA_DIR_LOADS, create=True)
    assert r["created"] is False and r["present"] is True             # idempotent
    # the loader snippet splices the alias code in and returns the list under
    # the key the stage / preflight readers split off
    assert L.DATA_DIR_ALIAS_CODE in L.INSPECT_LOADER_CODE
    assert "shutil.copytree(real, alias)" in L.DATA_DIR_ALIAS_CODE and "symlink" not in L.DATA_DIR_ALIAS_CODE
    assert L.DATA_DIR_ALIASES_KEY == "__data_dir_cache_aliases__"
    src = LAUNCHER.read_text()
    assert "aliases = out.pop(DATA_DIR_ALIASES_KEY, [])" in src


def test_merge_step_runs_the_shared_merge_in_the_client_venv(L, tmp_path):
    """N1: the controls_summary merge after EvalPlus conversion runs in the
    launcher's client venv (the driver's environment) as a CPU-only Modal
    function — since PW10 (E17 A1) as TWO subprocesses: the producing tree
    EMITS every store's normalized payload (its own
    ``complete_task_summaries`` + ``slot_payloads``, never a writer), and
    THIS tree WRITES through ``completion.merge_emitted_payload``; a
    non-store refuses (recorded), a store with nothing done merges nothing
    (truthfully), and discovery takes only stores whose manifest carries the
    provenance block."""
    W = "f" * 40
    argv = L.emit_argv(["/a", "/b"], "/scratch", tree="/t")
    assert argv[:3] == [L.CLIENT_PY, "-P", "-c"] and argv[3] == L.EMIT_CODE      # PW12: -P, the tree attested
    assert argv[4:] == ["/t", "/scratch", "/a", "/b"]
    assert "completion.complete_task_summaries(d)" in L.EMIT_CODE
    assert "completion.slot_payloads(payloads)" in L.EMIT_CODE
    for writer in ("merge_controls_summary", "write_controls_summary", "merge_emitted_payload",
                   "cell_controls"):
        assert writer not in L.EMIT_CODE, writer                    # the producing tree never writes
    argv = L.merge_argv(["/a", "/b"], "/scratch", writer="/w", writer_tree_sha=W)
    assert argv[:3] == [L.CLIENT_PY, "-P", "-c"] and argv[3] == L.MERGE_CODE
    assert argv[4:] == ["/w", W, "/scratch", "/a", "/b"]                    # PW12: the writer tree + its sha
    assert ("completion.merge_emitted_payload(d, doc, replace_source=replace, "
            "writer_tree_sha=writer_sha, plan=plan.get(d))") in L.MERGE_CODE
    assert "plan = completion.equal_witness_plan(docs)" in L.MERGE_CODE   # PW12: planned BEFORE the first write
    for code in (L.EMIT_CODE, L.MERGE_CODE):                                 # PW12: both halves attest their import
        assert "completion.__file__" in code and "sys.exit(3)" in code and repr(C.MERGE_EMIT_ATTEST_KEY) in code
    assert "merge_controls_summary" not in L.MERGE_CODE               # the writer reads the EMITTED payload
    # PW7: the operator's ambiguous-source decision travels as the FIRST argv,
    # so the recorded command says which merge replaced a filled entry — and
    # since PW8 (E16 A3) WHICH entry went to WHICH store, parsed by the contract
    assert L.merge_argv(["/a"], "/s", writer="/w", writer_tree_sha=W, replace_source="inspect/gsm8k=k1")[4:] == [
        L.MERGE_REPLACE_FLAG, "inspect/gsm8k=k1", "/w", W, "/s", "/a"]
    assert L.merge_argv(["/a"], "/s", writer="/w", writer_tree_sha=W, replace_source="")[4:] == ["/w", W, "/s", "/a"]
    assert "C.parse_replace_source(argv[1])" in L.MERGE_CODE
    assert L.MERGE_REPLACE_FLAG == "--replace-source" == C.REPLACE_SOURCE_FLAG
    with pytest.raises(ValueError, match="not <entry>=<cell_key>"):
        L.merge_argv(["/a"], "/s", writer="/w", writer_tree_sha=W, replace_source="bogus")
    with pytest.raises(ValueError, match="full 40-hex"):
        L.merge_argv(["/a"], "/s", writer="/w", writer_tree_sha="abc")
    assert L.merge_cell_summaries([], tree=REPO) == {}
    # the WRITING tree is this launcher's own sisteer — in a checkout, <repo>/src
    assert L.current_tree_src() == REPO / "src"
    src = LAUNCHER.read_text()
    assert "merge_cell_summaries(\n        accepted, tree=tree, python=CLIENT_PY" in src
    assert 'rec["writer_src"] = str(current_tree_src())' in src
    # a REAL subprocess in this venv (which, like the client venv, imports
    # sisteer.eval.cell_controls): a non-store refuses, recorded not raised
    bare = tmp_path / "x" / "cells" / "k"
    bare.mkdir(parents=True)
    res = L.merge_cell_summaries([bare], tree=REPO, python=sys.executable)
    assert res[str(bare)]["ok"] is False and "manifest.json" in res[str(bare)]["err"]
    # a store with provenance but no done task merges nothing (None), truthfully
    (bare / "manifest.json").write_text(json.dumps({"cell_id": "c", "provenance": {"x": 1}}))
    res = L.merge_cell_summaries([bare], tree=REPO, python=sys.executable)
    assert res == {str(bare): {"ok": True, "merged": False, "path": None, "notes": []}}   # notes: PW14
    legacy = tmp_path / "x" / "cells" / "legacy"
    legacy.mkdir()
    (legacy / "manifest.json").write_text(json.dumps({"cell_id": "c"}))
    assert L.tinycap_cell_dirs(tmp_path / "x") == ([str(bare)], [str(legacy)])
    # a RUN directory symlinked into the generation root is discovered like any
    # other (the recursive glob used to omit it from both lists), a store reachable
    # twice is named once, and a link back into the root does not loop
    import os
    elsewhere = tmp_path / "elsewhere" / "run-b"
    linked_cell = elsewhere / "cells" / "linked-key"
    linked_cell.mkdir(parents=True)
    (linked_cell / "manifest.json").write_text(json.dumps(
        {"cell_id": "c", C.MANIFEST_KEY_PROVENANCE: {"cell_id": "c"}}))
    os.symlink(elsewhere, tmp_path / "x" / "run-b", target_is_directory=True)
    os.symlink(tmp_path / "x", tmp_path / "x" / "loop", target_is_directory=True)
    merge, skipped2 = L.tinycap_cell_dirs(tmp_path / "x")
    assert str(tmp_path / "x" / "run-b" / "cells" / "linked-key") in merge
    assert len(merge) == len(set(os.path.realpath(m) for m in merge)) == 2   # bare + linked, once each
    assert skipped2 == [str(legacy)]
    src = LAUNCHER.read_text()
    head = src.rfind("@app.function(", 0, src.index("def merge_summaries("))
    deco = src[head:src.index("def merge_summaries(")]
    assert "image=image" in deco and "gpu" not in deco and "secrets" not in deco   # CPU, the serving image, no secret
    assert "merge_summaries.remote(sha, dirs, gen_root, replace_source, writer_sha)" in src   # PW12: the writer's commit travels


def test_merge_subprocesses_never_import_a_cwd_sisteer(L, tmp_path, monkeypatch, capsys):
    """PW12 (E19 A1-2): the EMIT subprocess ran ``python -c`` from ``/root``,
    where ``add_local_python_source`` mounts the CURRENT ``sisteer`` — Python
    puts the CWD BEFORE PYTHONPATH for ``-c``, so the emitter imported the
    current package, not the producing tree's (reproduced by the served-path
    owner; the tests ran from a repo-root CWD and could not see it). Both
    halves now run under ``-P`` + ``PYTHONSAFEPATH`` / ``PYTHONNOUSERSITE``
    from an empty CWD the merge owns, and each ATTESTS the
    ``completion.__file__`` it imported: the emitter refuses unless it is
    beneath the producing tree, the writer unless beneath the writer tree.
    The writer's commit is recorded (``writer_tree_sha``): a staged tree's
    stamp, else the checkout's git HEAD, else refused — and ``main`` refuses
    to launch a merge from a checkout whose ``src/sisteer`` is not HEAD's."""
    # a CWD holding a foreign `sisteer` — what /root is inside the container
    ptree = tmp_path / "ptree"
    poison = ptree / "src"
    pkg = poison / "sisteer" / "capeval" / "tinycap"
    pkg.mkdir(parents=True)
    for d in (poison / "sisteer", poison / "sisteer" / "capeval", pkg):
        (d / "__init__.py").write_text("")
    (pkg / "completion.py").write_text(
        "def complete_task_summaries(d):\n    return {}\n\ndef slot_payloads(p):\n    return {}\n")
    (pkg / "contracts.py").write_text("")      # so the WRITER half reaches its attestation, not an ImportError
    store = tmp_path / "x" / "cells" / "k"
    store.mkdir(parents=True)
    (store / "manifest.json").write_text(json.dumps({"cell_id": "c", "provenance": {"x": 1}}))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    # THE TRAP, reproduced: without -P, `python -c` from that CWD imports the CWD's package ahead
    # of PYTHONPATH — the attestation catches it (exit 3, nothing emitted)
    r = subprocess.run([sys.executable, "-c", L.EMIT_CODE, str(REPO), str(scratch), str(store)],
                       cwd=str(poison), env={**os.environ, "PYTHONPATH": f"{REPO}/src"},
                       capture_output=True, text=True, check=False)
    assert r.returncode == 3, r.stderr
    attest = json.loads(r.stdout.strip().splitlines()[-1])[C.MERGE_EMIT_ATTEST_KEY]
    assert attest["ok"] is False and attest["completion_file"].startswith(str(poison))
    assert attest["tree"] == str(REPO) and list(scratch.iterdir()) == []
    # THE FIX: the launcher's merge runs both halves under -P from an empty CWD it owns — the
    # parent process sitting IN the poisoned directory changes nothing
    monkeypatch.chdir(poison)
    att: dict = {}
    res = L.merge_cell_summaries([store], tree=REPO, python=sys.executable, attest=att)
    assert res == {str(store): {"ok": True, "merged": False, "path": None, "notes": []}}
    mine = str(REPO / "src" / "sisteer" / "capeval" / "tinycap" / "completion.py")
    assert att["emitter"]["ok"] is True and att["emitter"]["completion_file"] == mine
    assert att["writer"]["ok"] is True and att["writer"]["completion_file"] == mine
    assert att["writer_tree_sha"] == att["writer"]["writer_tree_sha"] == L.writer_tree_sha_of(REPO / "src")
    assert L.SAFE_PY_FLAGS == ("-P",) and L.SAFE_PY_ENV == {"PYTHONSAFEPATH": "1", "PYTHONNOUSERSITE": "1"}
    # with the guards removed (what the pre-PW12 launcher did) the attestation ALONE still refuses:
    # the emitter imported the CWD package -> every store refused, named, nothing emitted...
    monkeypatch.setattr(L, "SAFE_PY_FLAGS", ())
    monkeypatch.setattr(L, "SAFE_PY_ENV", {})
    monkeypatch.setattr(L.tempfile, "TemporaryDirectory",
                        lambda prefix="": _FixedCwd(poison))          # the subprocess CWD becomes the poison
    res = L.merge_cell_summaries([store], tree=REPO, python=sys.executable)
    assert res[str(store)]["ok"] is False and "not beneath the producing tree" in res[str(store)]["err"]
    assert str(ptree) in res[str(store)]["err"] and str(REPO) in res[str(store)]["err"]
    # ...and a writer that imported the CWD package raises (this tree's bug), nothing written
    with pytest.raises(RuntimeError, match="not beneath the writer tree"):
        L.merge_cell_summaries([store], tree=ptree, python=sys.executable, writer_src=REPO / "src",
                               writer_tree_sha="f" * 40)
    monkeypatch.undo()
    # the writer's commit: a staged tree's stamp; a checkout's HEAD; else refused
    repos = tmp_path / "repos"
    t = _tree(repos, "f6cf462")
    assert L.writer_tree_sha_of(t / "src") == FULL_SHA and L.writer_tree_dirty(t / "src") == []
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert L.writer_tree_sha_of(REPO / "src") == head and len(head) == 40
    assert isinstance(L.writer_tree_dirty(REPO / "src"), list)
    with pytest.raises(L.WriterTreeUnattested, match="cannot record which tree wrote"):
        L.writer_tree_sha_of(ptree / "src")
    # the Modal function records it, or refuses when it cannot be known (the store now names its
    # producing tree, so the staged f6cf462 tree accepts it and the merge is reached)
    (store / "manifest.json").write_text(json.dumps(
        {"cell_id": "c", "provenance": {"x": 1},
         C.MANIFEST_KEY_PRODUCER: {"tree_sha": FULL_SHA, "driver": C.PRODUCER_DRIVER_SERVED}}))
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    calls = []
    monkeypatch.setattr(L, "merge_cell_summaries",
                        lambda dirs, **kw: calls.append(kw) or {d: {"ok": True, "merged": True, "path": d}
                                                                  for d in dirs})
    rec = L.merge_summaries("f6cf462", [str(store)], str(tmp_path / "x"), "", "e" * 40)
    assert rec["status"] == "OK" and rec["writer_tree_sha"] == "e" * 40 and calls[-1]["writer_tree_sha"] == "e" * 40
    rec = L.merge_summaries("f6cf462", [str(store)], str(tmp_path / "x"))
    assert rec["writer_tree_sha"] == head                             # a checkout run resolves its own HEAD
    assert L.merge_summaries("f6cf462", [str(store)], str(tmp_path / "x"), "", "short")["status"] == "NO_WRITER_TREE_SHA"
    monkeypatch.setattr(L, "writer_tree_sha_of", lambda src: (_ for _ in ()).throw(L.WriterTreeUnattested("no")))
    assert L.merge_summaries("f6cf462", [str(store)], str(tmp_path / "x"))["status"] == "NO_WRITER_TREE_SHA"
    monkeypatch.undo()
    # main: a merge launched from a checkout whose src/sisteer is not HEAD's refuses (dry run says the sha)
    monkeypatch.setattr(L, "writer_tree_dirty", lambda src: ["src/sisteer/eval/cell_controls.py"])
    with pytest.raises(SystemExit, match="did not write"):
        L.main(app_name=L.APP_NAME, sha="abc1234", merge=True)
    L.main(app_name=L.APP_NAME, sha="abc1234", merge=True, dry_run=True)
    assert f"writer {head[:7]}, attested" in capsys.readouterr().out
    # why the launcher's served_geometry.json is no immutable witness of a legacy store's pass
    # (PW12 / E19 A1-1): _run_configuration rewrites it into the same out_dir on every run
    src = LAUNCHER.read_text()
    assert "(out_dir / SERVED_GEOMETRY_FILENAME).write_text(" in src
    assert 'rec["writer_tree_sha"] = writer_sha' in src and 'rec["attest"] = attest' in src


class _FixedCwd:
    """A ``TemporaryDirectory`` stand-in that hands the merge a directory whose
    ``cwd/`` subdirectory IS the poisoned package root (the test's trap)."""

    def __init__(self, poison: Path) -> None:
        self.root = poison.parent / "fixed-scratch"
        self.root.mkdir(exist_ok=True)
        link = self.root / "cwd"
        if not link.exists():
            link.symlink_to(poison, target_is_directory=True)

    def __enter__(self) -> str:
        return str(self.root)

    def __exit__(self, *exc) -> None:
        return None


def _capeval_run():
    """The suite's driver, loaded by path (stdlib + capeval.manifest at import
    time). Registered in ``sys.modules`` for the load: its ``@dataclass Cell``
    under ``from __future__ import annotations`` resolves string annotations
    through ``sys.modules[cls.__module__]``."""
    spec = importlib.util.spec_from_file_location("capeval_run_under_test", REPO / "scripts" / "capeval_run.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    return mod


def test_expected_identity_derives_the_sampling_block_the_driver_hashes(L):
    """PW-sampling (U8): the launcher's expected identity carries the five
    sampling fields exactly as the driver derives them — the registry row's
    frozen card row (gemma4e4b: 1.0 / 0.95 / 64), the per-task k over the
    tasks the configuration files (every decoding task at the contract's k,
    MATH-500 at the explicit `samples`), the seed rules — under the
    driver's default mode, and a greedy expectation on request."""
    exp = _expected(L)
    assert exp.sampling_mode == C.SAMPLING_DEFAULT == "card"
    assert set(exp.sampling) == set(C.SAMPLING_IDENTITY_FIELDS)
    assert (exp.sampling["temperature"], exp.sampling["top_p"], exp.sampling["top_k"]) == (1.0, 0.95, 64)
    # the default configuration plans the whole fitted battery: the map ranges
    # over exactly the tasks the thinking map does, every one a decoding task
    k = exp.sampling["samples_by_task"]
    assert set(k) == set(exp.thinking_by_task) and len(k) >= len(C.STANDARD_BATTERY_FITTED)
    # the 2026-09-06 ruling: one card draw per item on every decoding task
    assert all(k[t] == C.DECODE_SAMPLES_PER_ITEM == 1 for t in k)
    assert exp.sampling["seed_rule"] == dict(C.SEED_RULES)
    assert exp.as_dict()["sampling"] == dict(exp.sampling) and exp.as_dict()["sampling_mode"] == "card"
    # MATH-500 at the explicit samples — ALONE, the configuration the launcher serves
    # (pooled beside other keys it is refused before allocation) — and a
    # two-leg decoding battery at the contract's 1
    cfg = L.plan_configurations(cells=[C.CELL_PSRKL_SAFETY], modes=["off"], controls=[C.CONTROL_MATH500])[0]
    geom, _ = L.served_geometry(cfg, 1)
    deep = L.expected_identity(cfg, geom, pass_no=1, controls=cfg["controls"], samples=2, tree_sha=FULL_SHA)
    assert deep.sampling["samples_by_task"] == {"inspect/math": 2}
    assert deep.samples == 2
    mix = L.plan_configurations(cells=[C.CELL_PSRKL_SAFETY], modes=["off"],
                                controls=[C.CONTROL_HUMANEVAL_PLUS, C.CONTROL_GSM8K])[0]
    mgeom, _ = L.served_geometry(mix, 1)
    mexp = L.expected_identity(mix, mgeom, pass_no=1, controls=mix["controls"], samples=2, tree_sha=FULL_SHA)
    assert mexp.sampling["samples_by_task"] == {"inspect/gsm8k_paired200": 1, "evalplus/humaneval_plus": 1}
    assert mexp.samples == 2 and C.IDENTITY_KEY_MATH500_MAX_TOKENS not in mexp.as_dict()
    # the llama rows: the model's OWN card values, not gemma's
    lcfg = L.plan_configurations(cells=["prompted__safety_refusal"], modes=["off"],
                                 controls=["gsm8k"], model="llama8b")[0]
    lgeom, _ = L.served_geometry(lcfg, 1)
    lexp = L.expected_identity(lcfg, lgeom, pass_no=1, controls=lcfg["controls"], samples=4, tree_sha=FULL_SHA)
    assert (lexp.sampling["temperature"], lexp.sampling["top_p"], lexp.sampling["top_k"]) == (0.6, 0.9, None)
    # greedy on request: the greedy row, one sample everywhere
    g = L.expected_identity(cfg, geom, pass_no=1, controls=cfg["controls"], samples=1, tree_sha=FULL_SHA,
                            sampling_mode=C.SAMPLING_GREEDY)
    assert (g.sampling["temperature"], g.sampling["top_p"], g.sampling["top_k"]) == (0.0, None, None)
    assert set(g.sampling["samples_by_task"].values()) == {1} and g.sampling_mode == "greedy"
    assert g.sampling != deep.sampling
    with pytest.raises(ValueError, match="sampling mode must be one of"):
        L.expected_identity(cfg, geom, pass_no=1, controls=cfg["controls"], samples=4, tree_sha=FULL_SHA,
                            sampling_mode="nucleus")


# --------------------------------------------------------------------------- #
# E40 H1: the verified object IS the served object                              #
# --------------------------------------------------------------------------- #


def test_a_fitted_launch_requires_the_preflight_and_carries_its_digests(L, monkeypatch, capsys):
    # native-serving-default: the launcher's default is the native engine now; this test pins the LENS
    # launch (a PSR cell / the historical composition), so it states the path explicitly
    """E40 H1, the launcher side. A launch that plans a fitted cell without
    `--pf` refuses before any Modal call; with `--pf` the preflight's per-cell
    artifact digests are read off the report and handed to run_configuration
    as `object_digests`; a report that vouches for no digest (no record, a
    non-match, a malformed digest) refuses; base / prompted cells need none."""
    launched: dict = {}
    monkeypatch.setattr(L, "run_configuration", _spawn_only(_done, lambda c, k: launched.update(k)))
    monkeypatch.setattr(L, "preflight", SimpleNamespace(
        remote=lambda *a: pytest.fail("no preflight was asked for")))
    # a fitted cell without --pf: refused, nothing launched, the remedy named
    with pytest.raises(SystemExit, match="launch only behind --pf") as e:
        L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, cells=C.CELL_PSRKL_SAFETY, modes="off",
               controls="gsm8k")
    assert C.CELL_PSRKL_SAFETY in str(e.value) and "E40 H1" in str(e.value) and not launched
    with pytest.raises(SystemExit, match="launch only behind --pf"):
        L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", smoke=True)          # the smoke plans a fitted cell
    with pytest.raises(SystemExit, match="launch only behind --pf"):
        L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, cells="antidote", modes="off",
               controls="gsm8k")                                        # Class B: fitted too
    # base / prompted cells: no object, no digest, no preflight needed
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, cells="none,prompted__safety_refusal",
           modes="off", controls="gsm8k")
    assert launched["object_digests"] == {}
    capsys.readouterr()
    # with --pf: the digests the preflight verified reach run_configuration
    plan = L.plan_for(cells=[C.CELL_PSRKL_SAFETY, "none"], modes=["off"], controls=["gsm8k"])
    report = _preflight_report(L, plan)
    monkeypatch.setattr(L, "preflight", SimpleNamespace(remote=lambda *a: report))
    launched.clear()
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pf=True, pass1=True,
           cells=f"{C.CELL_PSRKL_SAFETY},none", modes="off", controls="gsm8k")
    out = capsys.readouterr().out
    assert launched["object_digests"] == {C.CELL_PSRKL_SAFETY: {
        "artifact_sha256": FAKE_DIGEST, "intervention_hash": "1" * 16, "expected": "1" * 16,
        "lens_route_intervention_hash": "2" * 16}}
    assert f"object digests the standups are held to (from --pf): {{\"{C.CELL_PSRKL_SAFETY}\": \"{FAKE_DIGEST}\"}}" in out
    # a preflight that vouches for no digest refuses the launch
    for broken in ({"ok": True, "checks": {}},
                   {"ok": True, "checks": {L.OBJECT_IDENTITY_CHECK: {"cells": {}}}},
                   {"ok": True, "checks": {L.OBJECT_IDENTITY_CHECK: {"cells": {
                       C.CELL_PSRKL_SAFETY: {"match": False, "artifact_sha256": FAKE_DIGEST}}}}},
                   {"ok": True, "checks": {L.OBJECT_IDENTITY_CHECK: {"cells": {
                       C.CELL_PSRKL_SAFETY: {"match": True, "artifact_sha256": "not-a-digest"}}}}}):
        monkeypatch.setattr(L, "preflight", SimpleNamespace(remote=lambda *a, b=broken: b))
        launched.clear()
        with pytest.raises(SystemExit, match="vouches for no object digest"):
            L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pf=True, pass1=True, cells=C.CELL_PSRKL_SAFETY,
                   modes="off", controls="gsm8k")
        assert not launched
        capsys.readouterr()
    # the extraction itself, on its own
    got = L.object_digests_from_preflight(report, [C.CELL_PSRKL_SAFETY])
    assert got[C.CELL_PSRKL_SAFETY]["artifact_sha256"] == FAKE_DIGEST
    with pytest.raises(ValueError, match="antidote"):
        L.object_digests_from_preflight(report, [C.CELL_PSRKL_SAFETY, "antidote"])
    assert L.object_digest_for(C.CELL_PSRKL_SAFETY, got) == FAKE_DIGEST
    for bad in (None, {}, {C.CELL_PSRKL_SAFETY: {}}, {C.CELL_PSRKL_SAFETY: {"artifact_sha256": "AB" * 32}}):
        with pytest.raises(ValueError, match="no preflight-verified object digest"):
            L.object_digest_for(C.CELL_PSRKL_SAFETY, bad)
    # the dry run says a fitted cell is held to the flag
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, cells=C.CELL_PSRKL_SAFETY, modes="off",
           controls="gsm8k", dry_run=True)
    out = capsys.readouterr().out
    assert f"{L.OBJECT_DIGEST_FLAG} <preflight artifact_sha256>" in out
    assert "object digest: --expect-object-sha256 is filled from --pf at launch" in out
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, cells="none", modes="off",
           controls="gsm8k", dry_run=True)
    assert L.OBJECT_DIGEST_FLAG not in capsys.readouterr().out


def test_the_container_holds_a_fitted_configuration_to_its_digest(L, tmp_path, monkeypatch):
    """E40 H1, the container side. `_run_configuration` refuses a fitted
    configuration with no (or a malformed) digest right before the standup —
    NO_OBJECT_DIGEST, no standup started — and, given one, records it and
    passes it into the standup argv as --expect-object-sha256, which the
    standup's own parser reads back; a DONE resume needs no digest at all; a
    prompted configuration never carries the flag."""
    cfg, _out = _fake_world(L, tmp_path, monkeypatch, record={})
    for missing in (None, {}, {cfg["cell"]: {"artifact_sha256": "xyz"}}):
        rec = _run_fake(L, cfg, object_digests=missing)
        assert rec["status"] == "NO_OBJECT_DIGEST", missing
        assert "no preflight-verified object digest" in rec["error"] and "--pf" in rec["error"]
        assert "standup_argv" not in rec and "standup_s" not in rec         # nothing was started
        assert rec["tree"]                                                # ... but after the plan-level refusals
    rec = _run_fake(L, cfg)                                              # the fixture's digest
    assert rec["status"] == "LEDGER_FAILED"                              # past the standup (the fake ledger dies)
    assert rec["object_digest"]["artifact_sha256"] == FAKE_DIGEST
    argv = rec["standup_argv"]
    assert argv[argv.index(L.OBJECT_DIGEST_FLAG) + 1] == FAKE_DIGEST
    assert argv[argv.index(L.OBJECT_DIGEST_FLAG) + 2:] == list(cfg["standup_extra"])   # before the extras
    up = _load_standup()
    ns = up.build_parser().parse_args(argv[2:])
    assert ns.expect_object_sha256 == FAKE_DIGEST and ns.cell == cfg["cell"]
    assert up.expected_artifact_digests(cfg["cell"], "si_safety__psr_kl_all", FAKE_DIGEST) == {
        "si_safety__psr_kl_all": FAKE_DIGEST}
    # only a cell that serves NO object may omit the expectation …
    assert up.expected_artifact_digests("none", "none", None) == {}
    assert up.expected_artifact_digests("prompted__safety_refusal", "prompted__safety", "") == {}
    # … a fitted one refuses (E45 H1 boundary; this used to return {} and bind
    # the served bytes to nothing)
    for missing in (None, "", "   "):
        with pytest.raises(ValueError, match="serves a FITTED object but no"):
            up.expected_artifact_digests(cfg["cell"], "si_safety__psr_kl_all", missing)
    with pytest.raises(ValueError, match="serves no object"):
        up.expected_artifact_digests("none", "none", FAKE_DIGEST)
    with pytest.raises(ValueError, match="serves no object"):
        up.expected_artifact_digests("prompted__safety_refusal", "prompted__safety", FAKE_DIGEST)
    with pytest.raises(ValueError, match="not a 64-hex"):
        up.expected_artifact_digests(cfg["cell"], "si_safety__psr_kl_all", "deadbeef")
    # a DONE resume never reaches the digest check
    done_cfg, _ = _fake_world(L, tmp_path / "done", monkeypatch, record={}, complete_store_mode="per_row")
    assert _run_fake(L, done_cfg, object_digests=None)["status"] == "DONE"
    # a prompted configuration carries no flag
    prompted = L.plan_configurations(cells=[C.CELL_PROMPTED_SAFETY], modes=["off"], controls=["gsm8k"])[0]
    argv = L._standup_cmd("/t", "/t/c.yaml", prompted["cell"], "/d", Path("/srv"), Path("/srv/STOP"),
                          L.served_geometry(prompted, 1)[0], 64, prompted["standup_extra"])
    assert L.OBJECT_DIGEST_FLAG not in argv
    # the standup threads the flag into the export and records it
    src = (REPO / "scripts" / "capctrl_serving_up.py").read_text()
    assert "expected_artifact_sha256=expected_digests" in src
    assert "expect_object_sha256=args.expect_object_sha256" in src
    assert '"object_artifact_sha256": self.object_sha256' in src


def test_the_standup_refuses_a_fitted_cell_carrying_no_expectation_at_all(tmp_path, monkeypatch):
    """E45 H1 BOUNDARY, without a staged object. `--expect-object-sha256` was
    optional, so a fitted cell launched with none produced an EMPTY expectation
    map and the export bound the served bytes to nothing — the launcher's own
    NO_OBJECT_DIGEST refusal is upstream of this door, not on it. Both of the
    standup's own doors now refuse: `bring_up` before the tokenizer load, the
    registry export and the launch composition, and `main` before it resolves
    the geometry or creates the out dir. `none` and prompted cells are
    unaffected, and a good digest still passes straight through."""
    up = _load_standup()
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("model: meta-llama/Llama-3.1-8B-Instruct\n"
                        "model_revision: testrev-1111-2222-3333\n")
    bank, out = tmp_path / "bank", tmp_path / "srv"

    class TokenizerLoaded(Exception):
        """the load the digest rule must precede (a sentinel, not a failure)"""

    from transformers import AutoTokenizer
    monkeypatch.setattr(AutoTokenizer, "from_pretrained",
                        lambda *a, **k: (_ for _ in ()).throw(TokenizerLoaded()))

    for missing in (None, "", "  "):
        with pytest.raises(ValueError, match="serves a FITTED object but no") as e:
            up.bring_up(repo=REPO, config_path=cfg_path, cell=C.CELL_PSRKL_SAFETY,
                        defenses_dir=bank, out=out, geom=None,
                        expect_object_sha256=missing)
        assert up.OBJECT_DIGEST_FLAG in str(e.value)
        assert not (out / "lens_registry").exists() and not (out / "chat_template.jinja").exists()
    # the cells that serve no object still need none — they get PAST the digest
    # rule and stop at the tokenizer load, which is the next thing bring_up does
    for free in ("none", C.CELL_PROMPTED_SAFETY):
        with pytest.raises(TokenizerLoaded):
            up.bring_up(repo=REPO, config_path=cfg_path, cell=free, defenses_dir=bank,
                        out=out, geom=None, expect_object_sha256=None)

    # the CLI door: refused before the geometry, before the out dir exists
    cli_out = tmp_path / "cli"
    argv = ["--repo-dir", str(REPO), "--config", str(cfg_path),
            "--cell", C.CELL_PSRKL_SAFETY, "--defenses-dir", str(bank),
            "--out", str(cli_out), "--stop-file", str(tmp_path / "STOP")]
    monkeypatch.setattr(up, "bring_up", lambda **k: pytest.fail(
        "bring_up must not be reached from a CLI the digest rule refuses"))
    with pytest.raises(SystemExit, match="serves a FITTED object but no"):
        up.main(argv)
    assert not cli_out.exists()

    # … and with the digest the CLI passes it through unchanged
    class Reached(Exception):
        """bring_up was reached: the CLI check accepted the digest."""

    seen = {}

    def _reached(**kw):
        seen.update(kw)
        raise Reached()

    monkeypatch.setattr(up, "bring_up", _reached)
    with pytest.raises(Reached):
        up.main(argv + [up.OBJECT_DIGEST_FLAG, FAKE_DIGEST])
    assert seen["expect_object_sha256"] == FAKE_DIGEST and seen["cell"] == C.CELL_PSRKL_SAFETY
    # the one exemption is a LIBRARY seam and nothing an operator can type:
    # `main` never passes it, and the CLI has no flag for it
    assert "require_object_digest" not in up.build_parser().format_help()
    assert seen.get("require_object_digest") is None
    src = (REPO / "scripts" / "capctrl_serving_up.py").read_text()
    assert "require_object_digest" not in src.split("def main(")[1]
    # … it exists for the probe standups, which file timings and no rows
    assert up.expected_artifact_digests(
        C.CELL_PSRKL_SAFETY, "si_safety__psr_kl_all", None, required=False) == {}
    for probe in ("served_decode_sweep.py", "hookprof_served.py"):
        assert "require_object_digest=False" in (REPO / "scripts" / probe).read_text(), probe


def test_the_registry_export_is_bound_to_the_preflight_digest(tmp_path):
    """E40 H1, the binding itself (`lens_steer.export_worker_registry`): the
    bytes the export copies into the worker registry — the bytes the server
    loads — must hash to the digest verified BEFORE this standup. The scenario
    the reviewer named: the object on the shared volume is REPLACED between
    the preflight and the bring-up; the standup then enumerates the
    replacement (its own recomputed-hash check is self-consistent and passes)
    — so only the outside digest can catch it, and it does. With the current
    bytes' digest the export succeeds and the registry loads; a digest keyed
    on a cell the registry does not lens-serve, or a malformed one, refuses."""
    torch = pytest.importorskip("torch")
    import io

    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.serving import lens_adapters as la
    from sisteer.capeval.serving import lens_steer as ls

    def blob(seed):
        g = torch.Generator().manual_seed(seed)
        buf = io.BytesIO()
        torch.save({"layer": 3, "vector": torch.randn(8, generator=g)}, buf)
        return buf.getvalue()

    artifact = tmp_path / "si_safety__cast.pt"
    artifact.write_bytes(blob(1))
    preflight_digest = hashlib.sha256(artifact.read_bytes()).hexdigest()      # "the preflight ran"
    spec = cm.PetriSpec(
        audit="tinycap-h1-test", behaviors=list(cm.PETRI_BEHAVIORS),
        cells=["none", "si_safety__cast"], scenario_bundle_hash="ab" * 8,
        auditor_model="anthropic/claude-sonnet-5", judge_model="anthropic/claude-opus-5",
        scenarios_model="anthropic/claude-sonnet-5",
        prompt_provenance="package-pinned@petri-bloom==0.2.6",
        target_model="org/synthetic-target", target_revision="0123456789abcdef0123456789abcdef01234567",
        serving_backend="proxy_vllm", serving_mode="interleaved",
        serving_packages={"vllm": "0.25.1+cu129", "vllm-lens": "1.2.1"},
        serving_launch_flags=["--enforce-eager"], chat_template_hash="12" * 8,
        tool_parser="llama3_json", serving_impl_commit="fedcba9876" * 4,
        prompted_prefix_hashes={"safety": cm.sha256_hex16(b"synthetic safety prefix"),
                                "constitution": cm.sha256_hex16(b"synthetic constitution")},
        bloom_options=cm.BloomOptions(max_turns=15, enable_prefill=False, enable_rollback=False,
                                      compaction=False, realism_filter=False, epochs=1, seed=0))

    def enumerate_now():
        # what bring_up does: enumerate whatever is on the volume NOW
        source = dc.MappingCellSource({
            "none": dc.RegistryEntry(cell_id="none", kind="none"),
            "si_safety__cast": dc.RegistryEntry(
                cell_id="si_safety__cast", kind=la.SYNTHETIC_KIND, si_id="safety",
                object_path=str(artifact), serving_path="lens",
                resolved_spec={"kind": la.SYNTHETIC_KIND, "si_id": "safety", "layer": 3})})
        cells = dc.enumerate_cells(
            spec, source, behavior_files={k: f"# {k}\n".encode() for k in cm.petri_expected_behavior_keys(spec)},
            prompt_files={"auditor": b"# a\n", "judge": b"# j\n"})
        assert not cells.absent
        return cells, source

    # the swap: another object under the same path, AFTER the preflight
    artifact.write_bytes(blob(2))
    swapped_digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert swapped_digest != preflight_digest
    cells, source = enumerate_now()                       # self-consistent: the standup sees the replacement
    with pytest.raises(ls.LensSteerError) as e:
        ls.export_worker_registry(cells, source, tmp_path / "reg_swapped",
                                  expected_artifact_sha256={"si_safety__cast": preflight_digest})
    msg = str(e.value)
    assert preflight_digest in msg and swapped_digest in msg and "replaced between the preflight" in msg
    # without the outside digest the export accepts the replacement — the bug the binding closes
    ls.export_worker_registry(cells, source, tmp_path / "reg_unbound")
    # the current bytes' digest: the export succeeds and the worker loads it
    entries = ls.export_worker_registry(cells, source, tmp_path / "reg_ok",
                                        expected_artifact_sha256={"si_safety__cast": swapped_digest})
    assert {e.cell_id for e in entries} == {"si_safety__cast", "none__lens0"}
    reg = ls.WorkerObjectRegistry.load(tmp_path / "reg_ok")
    assert reg.cell_ids() == ("none__lens0", "si_safety__cast")
    # a digest keyed on a cell this registry does not lens-serve checks nothing: refused
    with pytest.raises(ls.LensSteerError, match="does not serve as a lens leg"):
        ls.export_worker_registry(cells, source, tmp_path / "reg_miskeyed",
                                  expected_artifact_sha256={"si_safety__psr_kl_all": swapped_digest})
    with pytest.raises(ls.LensSteerError, match="does not serve as a lens leg"):
        ls.export_worker_registry(cells, source, tmp_path / "reg_none",
                                  expected_artifact_sha256={"none": swapped_digest})
    # a malformed digest is refused before any file is read
    for bad in ({"si_safety__cast": "abc"}, {"si_safety__cast": swapped_digest.upper()},
                {"": swapped_digest}, ["si_safety__cast"]):
        with pytest.raises(ls.LensSteerError, match="expected_artifact_sha256"):
            ls.export_worker_registry(cells, source, tmp_path / "reg_bad", expected_artifact_sha256=bad)
        assert not (tmp_path / "reg_bad" / ls.REGISTRY_INDEX).exists()


def test_the_standup_refuses_an_object_swapped_after_the_preflight(L, tmp_path, monkeypatch):
    """E40 H1, end to end on REAL bytes through `capctrl_serving_up.bring_up`
    (skips unless SISTEER_TINYCAP_STAGED_OBJECTS names the staged objects root
    and the pinned tokenizer is cached): the preflight derives the staged
    llama3b object's identity and records its sha256; the object under the
    same bank path is then replaced (the same blob re-saved with an extra
    stored field — a valid psr artifact the registry adapter resolves, so the
    standup's own checks are self-consistent); `bring_up` with the preflight's
    digest REFUSES before `build_server_launch` is ever reached. Restored, the
    same call passes the binding and proceeds to compose the launch."""
    pytest.importorskip("torch")
    import torch

    from sisteer.capeval.serving import lens_steer as ls
    from sisteer.capeval.serving import vllm_server as vs
    from sisteer.capeval.tinycap import object_identity as oi

    root = os.environ.get("SISTEER_TINYCAP_STAGED_OBJECTS")
    if not root:
        pytest.skip("SISTEER_TINYCAP_STAGED_OBJECTS unset: the staged objects live off-repo")
    row, cell = C.model_row("llama3b"), "safety_refusal__psr_kl_all"
    staged = Path(root).expanduser() / Path(row.objects_dir).name / cell / "object.pt"
    if not staged.is_file():
        pytest.skip(f"{staged} is not staged here")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer
    try:
        AutoTokenizer.from_pretrained(row.hf, revision=row.rev, local_files_only=True)
    except Exception as e:                                  # noqa: BLE001 — a cache probe, not a verdict
        pytest.skip(f"the pinned tokenizer is not cached: {type(e).__name__}")
    bank = tmp_path / "bank"
    (bank / cell).mkdir(parents=True)
    obj = bank / cell / "object.pt"
    shutil.copy(staged, obj)
    original = obj.read_bytes()
    # 1. the preflight: the identity matches the published one, and the digest is recorded
    rep = oi.derive_request({"model": row.hf, "revision": row.rev, "config": str(REPO / row.cfg),
                             "config_overrides": dict(row.cfg_overrides), "objects_dir": str(bank),
                             "targets": [{"cell": cell, "si_id": "safety_refusal", "arm": "psr_kl_all",
                                          "expected": "93b6466dd791cc7e"}]})
    (r,) = rep["results"]
    assert rep["ok"] is True and r["match"] is True
    digests = L.object_digests_from_preflight(
        {"checks": {L.OBJECT_IDENTITY_CHECK: {"cells": {cell: r}}}}, [cell])
    digest = digests[cell]["artifact_sha256"]
    assert digest == hashlib.sha256(original).hexdigest()
    # 2. the launcher's standup argv carries it, and the standup parses it back
    cfg = L.plan_configurations(cells=[cell], modes=["off"], controls=["gsm8k"], model="llama3b")[0]
    geom, _ = L.served_geometry(cfg, 1)
    cfg_path, _ = L.overlaid_config(row, str(REPO), tmp_path / "out")
    argv = L._standup_cmd(str(REPO), cfg_path, cell, str(bank), tmp_path / "srv", tmp_path / "STOP",
                          geom, 64, cfg["standup_extra"], object_sha256=digest)
    up = _load_standup()
    ns = up.build_parser().parse_args(argv[2:])
    assert ns.expect_object_sha256 == digest

    class Reached(Exception):
        """build_server_launch was reached: the binding passed."""

    def no_launch(*a, **k):
        raise Reached()
    monkeypatch.setattr(vs, "build_server_launch", no_launch)
    monkeypatch.setattr(vs, "launch_server", lambda *a, **k: pytest.fail("a server must never be launched here"))
    kw = dict(repo=REPO, config_path=Path(cfg_path), cell=cell, defenses_dir=bank, geom=geom,
              manifest=row.petri_manifest, tool_calls=False)
    # 3. the swap AFTER the preflight: the same blob with an extra stored field — a valid
    #    psr artifact the registry adapter resolves and enumerates without complaint
    blob = torch.load(obj, map_location="cpu", weights_only=True)
    blob["swapped_after_preflight"] = True
    torch.save(blob, obj)
    assert hashlib.sha256(obj.read_bytes()).hexdigest() != digest
    with pytest.raises(ls.LensSteerError, match="replaced between the preflight") as e:
        up.bring_up(out=tmp_path / "srv_swapped", expect_object_sha256=ns.expect_object_sha256, **kw)
    assert digest in str(e.value)                            # refused before build_server_launch
    # 4. the preflighted bytes restored: the binding passes and the bring-up composes the launch
    obj.write_bytes(original)
    with pytest.raises(Reached):
        up.bring_up(out=tmp_path / "srv_ok", expect_object_sha256=ns.expect_object_sha256, **kw)
    # 5. and with NO digest at all the standup refuses outright (E45 H1 boundary).
    #    This case used to reach `Reached` — i.e. the swapped object was composed into
    #    a launch, unbound — and the assertion codified it. Now a fitted cell without
    #    an expectation cannot get past the digest resolution.
    torch.save(blob, obj)
    for missing in (None, ""):
        with pytest.raises(ValueError, match="serves a FITTED object but no") as e:
            up.bring_up(out=tmp_path / "srv_unbound", expect_object_sha256=missing, **kw)
        assert up.OBJECT_DIGEST_FLAG in str(e.value)
        assert not (tmp_path / "srv_unbound" / "lens_registry").exists()   # nothing exported


# --------------------------------------------------------------------------- #
# PW-oos (U10 item 5) — the explicit out-of-support-thinking override          #
# --------------------------------------------------------------------------- #

_OOS_MODEL = "qwen35_4b"          # the authorised experiment's model (U10 item 5)
_OOS_CELL = C.CELL_PSRKL_SAFETY   # a PSR object — the gated kind, which is the point


def test_a_fitted_qwen_cell_refuses_thinking_on_without_the_flag(L):
    """Unchanged behaviour is the baseline: the plan-time refusal still fires,
    still cites the measured 7 -> 5 header move, and now also names the one
    override that exists."""
    with pytest.raises(L.ThinkingUnservable) as e:
        L.plan_configurations(cells=[_OOS_CELL], modes=["on"], controls=["gsm8k"],
                              model=_OOS_MODEL)
    msg = str(e.value)
    assert "MOVES between modes" in msg and "H=7" in msg and "H=5" in msg
    assert C.FLAG_ALLOW_OOS_THINKING in msg
    # and the flag has to be NAMED — the default is still the refusal
    with pytest.raises(L.ThinkingUnservable):
        L.plan_for(cells=[_OOS_CELL], modes=["on"], controls=["gsm8k"], model=_OOS_MODEL,
                   allow_out_of_support_thinking=False)


def test_the_flag_lifts_the_refusal_and_records_every_reason_it_lifted(L):
    """With the flag NAMED the configuration plans, composes NO reasoning
    parser (qwen3_5 has none pinned), and carries a record of both refusals it
    overrode plus the note that says what the override does and does not do."""
    cfgs = L.plan_for(cells=[_OOS_CELL], modes=["off", "on"], controls=["gsm8k"],
                      model=_OOS_MODEL, allow_out_of_support_thinking=True)
    off = next(c for c in cfgs if c["mode"] == "off")
    on = next(c for c in cfgs if c["mode"] == "on")
    # the OFF leg is IN support: the flag lifted nothing, so it carries nothing
    assert C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING not in off
    assert "thinking_override" not in off
    # the ON leg is the out-of-support one
    assert on[C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING] is True
    ov = on["thinking_override"]
    assert ov["flag"] == C.FLAG_ALLOW_OOS_THINKING and ov["family"] == "qwen3_5"
    assert len(ov["refusals_lifted"]) == 2                 # header move AND no parser
    assert ov["reasoning_parser_composed"] is False
    assert ov["note"] == C.OUT_OF_SUPPORT_THINKING_NOTE
    assert ov["on_tasks"] == [C.GSM8K_STANDARD_TASK]
    # ...and no parser is composed, rather than an unverified one invented
    assert C.FLAG_REASONING_PARSER not in on["standup_extra"]
    assert on["standup_extra"] == []


def test_the_flag_cannot_lift_a_family_with_no_thinking_channel(L):
    """llama's template carries no thinking switch, so there is no ON render to
    serve out of support — the flag must NOT turn that into a silently
    thinking-OFF run recorded as ON.

    Since E50 #3 the PLANNER refuses a llama launch one step earlier, as
    UNAUTHORISED (llama8b is not the authorised experiment's model), so the
    family rule is pinned where it lives — on the predicate the planner would
    have consulted. Both refusals are asserted: neither one alone would catch a
    regression in the other."""
    with pytest.raises(C.OutOfSupportThinkingUnauthorised) as e:
        L.plan_for(cells=["none"], modes=["on"], controls=["gsm8k"], model="llama8b",
                   allow_out_of_support_thinking=True)
    assert "model differs" in str(e.value)
    # the family rule itself, unchanged: even NAMED, the flag lifts nothing here
    assert C.out_of_support_thinking_overridable("llama") is False
    why = L.thinking_refusal_for(C.model_row("llama8b"), "none", "on", ["gsm8k"],
                                 allow_out_of_support=True)
    assert why and "no thinking switch" in why
    assert L.thinking_override_for(C.model_row("llama8b"), "none", "on", ["gsm8k"]) is None


def test_nothing_changes_when_the_flag_is_absent(L):
    """The invariance pin: for a configuration the override does not touch,
    every planned field, the driver argv, the standup extras and the resolved
    identity are IDENTICAL whether or not the flag was named. (A launch that
    names the flag but plans nothing out of support behaves as if it had
    not.)"""
    common = dict(cells=["none", _OOS_CELL], modes=["off"], controls=["gsm8k"],
                  model=_OOS_MODEL)
    plain = L.plan_for(**common)
    named = L.plan_for(**common, allow_out_of_support_thinking=True)
    assert plain == named                                   # the whole plan, field by field
    for cfg in named:
        assert C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING not in cfg
        geom, _ = L.served_geometry(cfg, 1)
        exp = L.expected_identity(cfg, geom, pass_no=1, controls=cfg["controls"], samples=1)
        assert exp.out_of_support_thinking is False
        # the hashed identity dict does not even carry the key
        assert C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING not in exp.as_dict()
        ep = {"proxy_url": "u", "routing_table": "r"}
        cmd = L._driver_cmd("<t>", cfg, ep, None, geom, controls=cfg["controls"], limit=0,
                            inspect_limit=0, samples=1, cfg_path="<c>", vllm_log=None)
        assert C.FLAG_ALLOW_OOS_THINKING not in cmd


def test_the_out_of_support_configuration_carries_the_flag_and_the_identity(L):
    """The ON leg's driver argv carries the flag (so the driver stamps the same
    identity the launcher expects) and its resolved identity carries the key."""
    (on,) = L.plan_for(cells=[_OOS_CELL], modes=["on"], controls=["gsm8k"],
                       model=_OOS_MODEL, allow_out_of_support_thinking=True)
    geom, _ = L.served_geometry(on, 1)
    ep = {"proxy_url": "u", "routing_table": "r"}
    kw = dict(controls=on["controls"], limit=0, inspect_limit=0, samples=1, cfg_path="<c>", vllm_log=None)
    # Phase B integration: the out-of-support LENS configuration's driver refuses a
    # run told no artifact digest (plan S3.2, OOS_DECLARATION_UNATTESTED), so the
    # launcher must hand it the one it held the standup to — and refuses to build
    # an argv without it rather than let the container die after the standup
    with pytest.raises(ValueError, match="needs the artifact digest"):
        L._driver_cmd("<t>", on, ep, None, geom, **kw)
    cmd = L._driver_cmd("<t>", on, ep, None, geom, expect_object_sha256=FAKE_DIGEST, **kw)
    assert cmd[-3:] == [C.FLAG_ALLOW_OOS_THINKING, C.FLAG_EXPECT_OBJECT_SHA256, FAKE_DIGEST]
    assert L.oos_lens_configuration(on)
    # none/ON is out of support but not a lens cell: no digest, and one offered refuses
    (none_on,) = L.plan_for(cells=["none"], modes=["on"], controls=["gsm8k"],
                            model=_OOS_MODEL, allow_out_of_support_thinking=True)
    assert not L.oos_lens_configuration(none_on)
    g2, _ = L.served_geometry(none_on, 1)
    cmd2 = L._driver_cmd("<t>", none_on, ep, None, g2, **dict(kw, controls=none_on["controls"]))
    assert cmd2[-1] == C.FLAG_ALLOW_OOS_THINKING and C.FLAG_EXPECT_OBJECT_SHA256 not in cmd2
    with pytest.raises(ValueError, match="offered to a configuration whose driver would not compare it"):
        L._driver_cmd("<t>", none_on, ep, None, g2, expect_object_sha256=FAKE_DIGEST,
                      **dict(kw, controls=none_on["controls"]))
    exp = L.expected_identity(on, geom, pass_no=1, controls=on["controls"], samples=1)
    assert exp.out_of_support_thinking is True
    assert exp.as_dict()[C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING] is True
    # the served identity records the parser as null — no parser was composed
    assert exp.served["reasoning_parser"] is None


def test_an_out_of_support_store_never_resumes_into_an_in_support_one(L, tmp_path):
    """Both directions, by name: the hashed identity carries the key only under
    the override, so a store from one world is refused for a run of the
    other."""
    oos = _oos_expected(L)
    plain = _expected(L, cell=_OOS_CELL, mode="on")      # gemma's in-support psr/on: the campaign
    out = tmp_path / "thinking_on"
    # matched: an out-of-support store for an out-of-support run
    assert L.untrusted_store_reason(_store(out, oos, tasks=["gsm8k"]),
                                    expected=oos) is None
    # an IN-support store offered to an OUT-of-support run: the key is missing
    why = L.untrusted_store_reason(_store(tmp_path / "b", plain, tasks=["gsm8k"]), expected=oos)
    assert why and C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING in why and "missing" in why
    # ...and the reverse: an out-of-support store offered to an ordinary run
    why = L.untrusted_store_reason(_store(tmp_path / "c", oos, tasks=["gsm8k"]), expected=plain)
    assert why and C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING in why and "unknown" in why
    for w in (why,):
        assert C.FLAG_ALLOW_OOS_THINKING in w                # the reason, not just the diff


def test_the_preflight_plans_and_probes_the_override(L, tmp_path, monkeypatch, capsys):
    """The flag is part of plan_args (so the preflight plans the SAME
    out-of-support configurations the launch runs), the report names them, and
    the driver's argparse probe asks for the flag — a staged tree whose driver
    predates it must refuse on the CPU."""
    # part of the planning inputs
    argv = L.launch_argv("abc1234", pass1=True, allow_out_of_support_thinking=True)
    assert C.FLAG_ALLOW_OOS_THINKING in argv
    assert C.FLAG_ALLOW_OOS_THINKING not in L.launch_argv("abc1234", pass1=True)
    # the probe asks for it only when asked
    assert C.FLAG_ALLOW_OOS_THINKING not in L.driver_probe_argv("/t")
    probed = L.driver_probe_argv("/t", extra_flags=(C.FLAG_ALLOW_OOS_THINKING,))
    assert probed[-2:] == [C.FLAG_ALLOW_OOS_THINKING, L.DRIVER_PROBE_SENTINEL]
    # the identity request the preflight hands the derivation
    row = C.model_row(_OOS_MODEL)
    plain = L.identity_request(row, "/t", [_OOS_CELL])
    assert "allow_out_of_support_thinking" not in plain
    named = L.identity_request(row, "/t", [_OOS_CELL], allow_out_of_support_thinking=True)
    assert named["allow_out_of_support_thinking"] is True
    assert {k: v for k, v in named.items() if k != "allow_out_of_support_thinking"} == plain


# --------------------------------------------------------------------------- #
# E50 #3 — the launcher door: the override authorises ONE tuple, not a family  #
# --------------------------------------------------------------------------- #


def _oos_plan_kwargs() -> dict:
    """`plan_for` arguments that ARE the authorised experiment — one cell, one
    mode, so a mutation below moves exactly one field."""
    exp = C.AUTHORISED_OOS_THINKING
    return {"cells": [exp.cells[-1]], "modes": [exp.modes[-1]],
            "controls": list(exp.controls), "model": exp.model,
            "sampling_mode": exp.sampling_mode, "samples": exp.samples,
            "allow_out_of_support_thinking": True}


def test_the_launcher_plans_the_authorised_experiment_and_nothing_else(L):
    """The authorised tuple plans (both cells, both modes, one plan), and it is
    the ONLY thing the flag lets through — the base for the mutations below."""
    exp = C.AUTHORISED_OOS_THINKING
    cfgs = L.plan_for(cells=list(exp.cells), modes=list(exp.modes),
                      controls=list(exp.controls), model=exp.model,
                      allow_out_of_support_thinking=True)
    assert len(cfgs) == len(exp.cells) * len(exp.modes)
    # the ON legs are the out-of-support ones (qwen3_5 has no verified parser,
    # so the base cell's ON leg is out of support too); the OFF legs are not
    oos = {(c["cell"], c["mode"]) for c in cfgs
           if c.get(C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING)}
    assert oos == {(cell, C.THINKING_ON) for cell in exp.cells}


@pytest.mark.parametrize("field,value", [
    ("model", "qwen35_9b"),                        # the sibling model, same family
    ("cells", ["si_constitution__psr_kl_all"]),
    ("controls", ["gsm8k", "humaneval_plus"]),      # a WIDER item set than authorised
    ("modes", [C.THINKING_PER_TASK]),
    ("sampling_mode", C.SAMPLING_GREEDY),
    ("samples", 4),
])
def test_the_launcher_refuses_any_cli_selection_that_is_not_the_authorised_run(
        L, field, value):
    """E50 #3: before this the flag authorised on the FAMILY, so any
    CLI-selected model / cell / mode / control battery rode it — each stamping
    the authorised experiment's `out_of_support_thinking` identity. Now every
    field is compared against the one declared tuple and the refusal names the
    one that moved."""
    kwargs = {**_oos_plan_kwargs(), field: value}
    with pytest.raises(C.OutOfSupportThinkingUnauthorised) as e:
        L.plan_for(**kwargs)
    msg = str(e.value)
    assert f"{field.rstrip('s') if field in ('cells', 'modes') else field} differs" in msg
    assert "launch configuration" in msg          # which configuration was being planned


def test_the_preflight_refuses_the_same_unauthorised_selection_as_the_launch(L):
    """One planner, two doors (E40 M6): the preflight replans from the launch's
    own `plan_args`, so an unauthorised selection cannot pass the CPU check and
    then be launched — and the smoke plan, which is not the authorised
    experiment either, is refused when it names the flag."""
    bad = {**_oos_plan_kwargs(), "model": "gemma4e4b"}
    for planner in (L.plan_for, L.plan_configurations):
        with pytest.raises(C.OutOfSupportThinkingUnauthorised):
            (planner(**bad) if planner is L.plan_for
             else planner(bad["cells"], bad["modes"], bad["controls"],
                          model=bad["model"], allow_out_of_support_thinking=True))
    with pytest.raises(C.OutOfSupportThinkingUnauthorised):
        L.smoke_plan(["humaneval_plus"], model=C.AUTHORISED_OOS_THINKING.model,
                     allow_out_of_support_thinking=True)


def test_an_in_support_launch_is_untouched_by_the_authorisation_check(L):
    """The check exists only under the flag: every plan made WITHOUT it is
    byte-identical to what it was, whatever the model, cell or battery."""
    for kw in ({"cells": ["none"], "modes": ["off"], "controls": ["gsm8k"],
                "model": "qwen35_9b"},
               {"cells": [C.CELL_PSRKL_SAFETY], "modes": ["off"], "controls": ["gsm8k"],
                "model": "gemma4e4b"}):
        cfgs = L.plan_for(**kw)
        assert cfgs and all(C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING not in c for c in cfgs)
        # ...and the sampling knobs are planning inputs for the check and for
        # the ONE marker a non-default draw adds: naming a non-default mode
        # attaches exactly SAMPLING_MODE_KEY and changes no other planned
        # field; `samples` still changes nothing; and the default mode, named
        # or not, is byte-identical (no key at all)
        assert all(L.SAMPLING_MODE_KEY not in c for c in cfgs)
        assert L.plan_for(**kw, sampling_mode=C.SAMPLING_DEFAULT, samples=4) == cfgs
        assert (L.plan_for(**kw, sampling_mode=C.SAMPLING_GREEDY, samples=4)
                == [{**c, L.SAMPLING_MODE_KEY: C.SAMPLING_GREEDY} for c in cfgs])


# --------------------------------------------------------------------------- #
# E54 #6 — the CONTAINER door: the worker makes the exact-tuple check FIRST    #
# --------------------------------------------------------------------------- #


def _oos_worker_world(L, tmp_path, monkeypatch, *, record=None):
    """The authorised out-of-support configuration, with a staged tree and a
    fake standup, ready to hand straight to ``_run_configuration`` — the shape
    a direct ``run_configuration.remote(cfg, ...)`` produces. The fake standup
    attests ``record`` (default: the CONTRACT-shaped endpoint,
    :func:`_oos_endpoint`). Returns ``(cfg, out_dir, popen_argvs)``."""
    exp = C.AUTHORISED_OOS_THINKING
    (cfg,) = L.plan_for(cells=[exp.cells[-1]], modes=[C.THINKING_ON],
                        controls=list(exp.controls), model=exp.model,
                        allow_out_of_support_thinking=True)
    assert cfg[C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING] is True
    shutil.rmtree(f"/tmp/srv_{cfg['name']}", ignore_errors=True)
    repos = tmp_path / "repos"
    _tree(repos, "f6cf462")
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    monkeypatch.setenv(C.LENS_STACKED_REDUCTIONS_ENV, "0")
    argvs = []

    class _Popen(_FakeStandupProc):
        def __init__(self, argv, *a, **k):
            argvs.append(list(argv))
            super().__init__(argv, *a, **k)

    _FakeStandupProc.record = _oos_endpoint(L, cfg) if record is None else dict(record)
    monkeypatch.setattr(L.subprocess, "Popen", _Popen)
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    out = tmp_path / "gen"
    return dict(cfg, out_dir=str(out)), out, argvs


@pytest.mark.parametrize("field,value", [
    ("model", "qwen35_9b"),                                  # the sibling model
    ("cell", "si_constitution__psr_kl_all"),
    ("controls", ["gsm8k", "humaneval_plus"]),                      # a WIDER item set
    ("mode", C.THINKING_PER_TASK),                           # not an authorised arm
])
def test_the_container_refuses_an_unauthorised_tuple_before_any_side_effect(
        L, tmp_path, monkeypatch, field, value):
    """E54 #6. ``run_configuration`` is its own entry point: a direct
    ``.remote(cfg, ...)`` never passes through ``plan_configurations``, so the
    container used to take ``cfg[out_of_support_thinking]`` — a BOOLEAN — as
    sufficient. It then resolved the staged tree, made the out dir and started
    the standup, and only the CHILD driver made the exact-tuple comparison: an
    H200 spent and partial state written for a run that was never authorised.
    The worker now makes the same comparison the launcher and the driver make,
    FIRST, and files a typed refusal naming the field."""
    cfg, out, argvs = _oos_worker_world(L, tmp_path, monkeypatch)
    # the tuple is edited AFTER planning — exactly what a hand invocation does
    bad = dict(cfg, **({field: value} if field != "mode"
                       else {"mode": value, "thinking_by_task": cfg["thinking_by_task"]}))
    rec = L._run_configuration(bad, "f6cf462")
    assert rec["status"] == "OOS_UNAUTHORISED"
    assert f"{field} differs" in rec["error"]
    assert C.FLAG_ALLOW_OOS_THINKING in rec["error"]
    assert f"container run of {bad['name']!r}" in rec["error"]
    assert rec[C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING] is True     # what it CLAIMED
    # no side effect of any kind: no out dir, no tree read, no child process
    assert not out.exists()
    assert argvs == []
    assert "tree" not in rec and "geometry" not in rec and "expected_identity" not in rec


def test_the_authorised_tuple_reaches_the_standup_and_an_in_support_run_is_untouched(
        L, tmp_path, monkeypatch):
    """The other half: the check refuses nothing it is not meant to. The
    authorised tuple gets PAST the gate and starts the (fake) standup, and a
    configuration that never named the flag is not subjected to the comparison
    at all."""
    cfg, out, argvs = _oos_worker_world(L, tmp_path, monkeypatch)
    try:
        # S4.3: an out-of-support ON container is handed the preflight's artifact
        rec = L._run_configuration(cfg, "f6cf462", object_digests=_digests_for(cfg),
                                   oos_prompt_budget=_budget())
    finally:
        shutil.rmtree(f"/tmp/srv_{cfg['name']}", ignore_errors=True)
    assert rec["status"] != "OOS_UNAUTHORISED" and "error" not in rec.get("status", "")
    assert any("capctrl_serving_up.py" in " ".join(a[:2]) for a in argvs), rec["status"]
    assert rec[C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING] is True
    assert rec["thinking_override"]["flag"] == C.FLAG_ALLOW_OOS_THINKING
    assert out.exists()

    # an IN-SUPPORT configuration of the very same shape: no flag, no check —
    # the unauthorised tuple above would be irrelevant to it
    plain, _out2, _argvs2 = _oos_worker_world(L, tmp_path / "b", monkeypatch)
    plain = {k: v for k, v in plain.items() if k != C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING}
    plain = dict(plain, cell="si_constitution__psr_kl_all", name="plain_run")
    try:
        rec2 = L._run_configuration(plain, "f6cf462", object_digests={
            plain["cell"]: {"artifact_sha256": FAKE_DIGEST, "intervention_hash": "1" * 16,
                            "expected": "1" * 16, "lens_route_intervention_hash": "2" * 16}})
    finally:
        shutil.rmtree("/tmp/srv_plain_run", ignore_errors=True)
    assert rec2["status"] != "OOS_UNAUTHORISED"
    assert rec2[C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING] is False


def test_all_three_doors_make_the_one_comparison_and_the_worker_makes_it_first(L):
    """One rule, three entry points (launcher, container worker, driver) — and
    in the worker it precedes every side effect and the out-of-support LIFT, so
    an unauthorised run is refused as unauthorised rather than served out of
    support and refused a frame later."""
    whole = Path(L.__file__).read_text()
    # the WORKER's body only: `out_dir.mkdir` and friends appear in other
    # functions too, and an ordering test that measured the wrong copy would
    # pass for the wrong reason
    src = whole[whole.index("def _run_configuration("):whole.index("def _producer_conflict_error(")]
    i_check = src.index("C.check_out_of_support_thinking_authorised(")
    for later in ('refusal = thinking_refusal_for(row, cfg["cell"]',
                  "tree_path, tree_sha = resolve_staged_tree(sha)",
                  "out_dir.mkdir(parents=True, exist_ok=True)",
                  "standup = subprocess.Popen("):
        assert i_check < src.index(later), later
    # the draw's mode is read from ONE local by the check and the hashed
    # identity, so a per-run sampling mode cannot reach one and not the other.
    # That local is resolved ONCE through `sampling_mode_of(cfg)` — the plan's
    # SAMPLING_MODE_KEY, absent = the driver's default — and it is that
    # accessor, not the worker, that spells the default through the contract
    # (a worker that pinned C.SAMPLING_DEFAULT itself would run every
    # `protocol` configuration as `card`)
    assert "sampling_mode = sampling_mode_of(cfg)" in src
    assert src.count("sampling_mode = ") == 1
    accessor = whole[whole.index("def sampling_mode_of("):whole.index("def serves_offcap(")]
    assert "C.SAMPLING_DEFAULT" in accessor and "C.check_sampling_arg(" in accessor
    assert "sampling_mode=sampling_mode" in src
    # the OOS authorisation, the comparator's twin (offcap §5.1.8), the standup
    # extras (the protocol clamp override) and the hashed identity
    assert src.count("sampling_mode=sampling_mode") == 4
    assert src.index("C.check_off_cap_override_authorised(") < src.index('refusal = thinking_refusal_for(row, cfg["cell"]')


# --------------------------------------------------------------------------- #
# fit-support transfer, experiment 1 — plan S2.8 (test T4's admission slice):  #
# a done OUT-OF-SUPPORT task needs a valid outcome record at the launcher door #
# --------------------------------------------------------------------------- #


def _oos_summary(task, *, n):
    """A summary payload carrying every field the shared verifier reads."""
    return {"acc": 0.5, "n": n, "n_scored": n, "n_unscored": 0, "n_scored_samples": n,
            "n_unscored_samples": 0, "n_stop_reason": n, "n_at_cap": 0,
            "n_think_block_present": n, "n_think_block_unterminated": 0,
            "n_unterminated_not_at_cap": 0, "n_completion_tokens": n,
            "think_split_mode": "implicit_open", "unterminated_policy": "unscored",
            "max_tokens": 8192, "samples_per_item": 1, "task": task,
            "item_set": C.GSM8K_ITEM_SET_PAIRED, "thinking": C.THINKING_ON}


def _oos_outcomes(task, summary):
    n = summary["n"]
    items = [{"item_id": f"gsm8k_{i:08x}", "sample_idx": 0, "attempt": 1, "scored": True,
              "value": 1.0, "unparseable": False, "stop_reason": "stop", "at_cap": False,
              "think_block_present": True, "think_block_unterminated": False, "stray_open_tag": False,
              "content_empty_with_reasoning": False, "prompt_tokens": 1000,
              "completion_tokens": 300, "seed": i} for i in range(n)]
    return {**{f: summary[f] for f in (
        "task", "item_set", "thinking", "think_split_mode", "unterminated_policy", "max_tokens",
        "samples_per_item", "n_scored", "n_unscored", "n_scored_samples", "n_unscored_samples",
        "n_stop_reason", "n_at_cap", "n_think_block_present", "n_think_block_unterminated",
        "n_unterminated_not_at_cap", "n_completion_tokens")},
        "n_attempted_items": n, "n_attempted_samples": n, "producer_tree_sha": FULL_SHA,
        "items": items}


def test_a_done_out_of_support_task_needs_a_valid_outcome_record_at_the_launcher_door(L, tmp_path):
    """The launcher site. For a store whose OWN manifest is positively
    marked `out_of_support_thinking: true`, a done marker + summary WITHOUT the
    sidecar is NOT done (`invalid_done_marker_reason` returns the named reason,
    `resume_report` files it under `why_not`, the control has no `done_in` —
    so the configuration proceeds to a standup, where the driver rebuilds the
    record from the retained logs or refuses by name — never a re-run into the
    store); an unparsable sidecar and one disagreeing with its summary are
    refused by name; a valid one makes the task done. An IN-SUPPORT store (no
    marker) with no sidecar is still done — the campaign's stores are
    untouched. Both admission sites call ONE reason function and agree."""
    from sisteer.capeval.tinycap import completion

    task = C.GSM8K_STANDARD_TASK
    oos = _oos_expected(L)
    out = tmp_path / "thinking_on"
    store = _store(out, oos, tasks=["gsm8k"])
    assert L.untrusted_store_reason(store, expected=oos) is None
    _finish(store, task, C.GSM8K_PAIRED_N)
    why = L.invalid_done_marker_reason(store, task)
    assert why is not None
    assert why.startswith("no outcomes/inspect__gsm8k_paired200.json beside the done marker")
    assert "out_of_support_thinking" in why and "rebuilds it from the retained .eval logs" in why
    assert why == completion.task_outcomes_reason(store, task)            # ONE reason function
    assert not completion.is_done(store, task, require_outcomes=True)      # the driver agrees
    report = L.resume_report(out, ["gsm8k"], oos)
    assert report["stores"] == {store.name: None}
    assert report["controls"]["gsm8k"]["done_in"] is None
    assert report["controls"]["gsm8k"]["why_not"] == {store.name: {task: why}}
    assert L.remaining_controls(out, ["gsm8k"], oos) == ["gsm8k"]
    # an unparsable sidecar
    p = completion.task_outcomes_path(store, task)
    p.parent.mkdir(exist_ok=True)
    p.write_text("{torn")
    why = L.invalid_done_marker_reason(store, task)
    assert why.startswith("outcomes/inspect__gsm8k_paired200.json") and "unreadable" in why
    # a valid sidecar beside an AGREEING summary: done, with no server
    summary = _oos_summary(task, n=C.GSM8K_PAIRED_N)
    completion.write_task_summary(store, task, summary)
    completion.write_task_outcomes(store, task, _oos_outcomes(task, summary))
    assert L.invalid_done_marker_reason(store, task) is None
    assert L.resume_report(out, ["gsm8k"], oos)["controls"]["gsm8k"]["done_in"] == store.name
    assert L.remaining_controls(out, ["gsm8k"], oos) == []
    assert completion.is_done(store, task, require_outcomes=True)
    # a sidecar disagreeing with its summary (the summary's n_at_cap moved)
    completion.write_task_summary(store, task, {**summary, "n_at_cap": 4})
    why = L.invalid_done_marker_reason(store, task)
    assert why == ("outcomes/inspect__gsm8k_paired200.json disagrees with "
                   "summary/inspect__gsm8k_paired200.json: n_at_cap 0 != 4")
    assert why == completion.task_outcomes_reason(store, task)
    assert L.remaining_controls(out, ["gsm8k"], oos) == ["gsm8k"]
    # the marker checks that precede the sidecar still fire first
    completion.write_task_summary(store, task, {**summary, "n": 199})
    assert "summary n 199" in L.invalid_done_marker_reason(store, task)
    # an IN-SUPPORT store: no sidecar needed, done as before
    plain = _expected(L, cell=_OOS_CELL, mode="on")
    pstore = _store(tmp_path / "in_support", plain, tasks=["gsm8k"])
    _finish(pstore, task, C.GSM8K_PAIRED_N)
    assert L.invalid_done_marker_reason(pstore, task) is None
    assert L.remaining_controls(tmp_path / "in_support", ["gsm8k"], plain) == []
    assert completion.store_is_out_of_support(pstore) is False and completion.store_is_out_of_support(store) is True
    # the gate is a POSITIVE marker: a truthy non-True value, or no manifest, is not one
    m = json.loads((pstore / "manifest.json").read_text())
    m[C.MANIFEST_KEY_TINYCAP][C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING] = 1
    (pstore / "manifest.json").write_text(json.dumps(m))
    assert completion.store_is_out_of_support(pstore) is False
    assert completion.store_is_out_of_support(tmp_path / "nowhere") is False


def test_the_launcher_reads_the_outcome_record_through_the_shared_completion_module(L):
    """The launcher imports `tinycap.completion` (stdlib + contracts, no
    torch / inspect_ai) for exactly this: the ONE reason function both
    admission sites call. Pinned so a second copy of the rule never appears."""
    src = Path(L.__file__).read_text()
    assert "from sisteer.capeval.tinycap import completion" in src
    body = src[src.index("def invalid_done_marker_reason("):src.index("def _store_requires_outcomes(")]
    assert "completion.task_outcomes_reason(cell_dir, task)" in body
    assert "_store_requires_outcomes(cell_dir)" in body        # offcap §3.4: EITHER positive marker owes the sidecar
    assert "outcomes" not in body.replace("task_outcomes_reason", "").replace(
        "outcome record", "").replace("outcomes sidecar", "").replace(
        "_store_requires_outcomes", "")   # no second reading of the file here (the predicate reads the manifest)


# --------------------------------------------------------------------------- #
# fit-support transfer, experiment 1 — the prompt-budget slice: the            #
# thinking-ON prompt budget against the RESOLVED launch geometry, bound to     #
# the template the artifact tokenized                                          #
# --------------------------------------------------------------------------- #


def _oos_plan(L):
    """The authorised 2x2 exactly as run_oos_qwen4b.sh plans it."""
    return L.plan_for(cells=["none", _OOS_CELL], modes=["off", "on"], controls=["gsm8k"],
                      model=_OOS_MODEL, allow_out_of_support_thinking=True)


def _budget(*, prompt_max=2000, cap=C.OOS_THINKING_MAX_TOKENS, n=C.GSM8K_PAIRED_N,
            digest="a" * 16, arm=C.THINKING_ON, model=None, revision=None, cap_task=None):
    """A valid prompt-budget artifact (contracts.OOS_PROMPT_BUDGET_FIELDS) whose
    max is ``prompt_max`` — what the CPU script writes, synthesised."""
    row = C.model_row(_OOS_MODEL)
    lengths = [prompt_max - (i % 7) for i in range(n - 1)] + [prompt_max]
    s = sorted(lengths)
    doc = {"n": n, "max": prompt_max, "p99": s[-2], "p50": s[n // 2 - 1],
           "mean": round(sum(lengths) / n, 2), "arm": arm, "model": model or row.hf,
           "model_slug": row.slug, "revision": revision or row.rev, "chat_template_hash": digest,
           "cap": cap, "cap_task": cap_task or C.GSM8K_STANDARD_TASK, "max_plus_cap": prompt_max + cap,
           "lengths": lengths,
           "render": {"add_generation_prompt": True,
                      C.CHAT_TEMPLATE_KWARGS_KEY: {C.ENABLE_THINKING_KEY: arm == C.THINKING_ON}},
           "messages_source": C.OOS_PROMPT_BUDGET_MESSAGES_SOURCE, "tree_sha": FULL_SHA,
           "produced_by": {"script": C.OOS_PROMPT_BUDGET_SCRIPT}}
    return C.validate_oos_prompt_budget(doc)


def test_oos_on_configurations_and_their_raised_caps_come_from_the_tuple(L):
    """The gate's inputs: exactly the two thinking-ON configurations of the 2x2
    are out of support (the planner marked them), and each ON task's cap is the
    tuple's raised one — read through `contracts.oos_max_tokens_on`, never
    typed. OFF configurations carry no cap; an out-of-support ON configuration
    none of whose ON tasks has a raised cap refuses."""
    cfgs = _oos_plan(L)
    assert len(cfgs) == 4
    on = L.oos_on_configurations(cfgs)
    assert [c["name"] for c in on] == [c["name"] for c in cfgs if c["mode"] == C.THINKING_ON]
    assert all(c[C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING] is True for c in on)
    for c in on:
        assert L.oos_on_caps(c) == {C.GSM8K_STANDARD_TASK: C.oos_max_tokens_on("gsm8k_paired200")}
        assert L.oos_on_caps(c) == {"inspect/gsm8k_paired200": C.OOS_THINKING_MAX_TOKENS}
        assert L.oos_thinking_by_task(c) == {C.GSM8K_STANDARD_TASK: C.THINKING_ON}
    for c in cfgs:
        if c["mode"] == C.THINKING_OFF:
            assert L.oos_on_caps(c) == {} and c not in on
    # a cfg without the planner's thinking_by_task record derives it the planner's way
    bare = {k: v for k, v in on[0].items() if k != "thinking_by_task"}
    assert L.oos_thinking_by_task(bare) == on[0]["thinking_by_task"]
    # marked out of support but no raised cap for its ON tasks: nothing to add — refused
    with pytest.raises(ValueError, match="names no cap for its ON tasks \\['inspect/math'\\]"):
        L.oos_on_caps({"name": "x", C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING: True,
                       "mode": "on", "controls": ["math500"],
                       "thinking_by_task": {"inspect/math": "on"}})
    # an in-support cfg with the same tasks is simply capless (no refusal — nothing gates it)
    assert L.oos_on_caps({"name": "y", "mode": "on", "controls": ["math500"],
                          "thinking_by_task": {"inspect/math": "on"}}) == {}


def test_oos_cap_fits_resolved_window_passes_at_the_resolved_window_and_fails_at_10000(L):
    """P2-b: the check reads the geometry the launch RESOLVES — `--pass1` ->
    PETRI_MAX_MODEL_LEN x the qwen35_4b budget row's S (`served_geometry(cfg, 1)`,
    the same resolver the standup argv uses) — and asserts `max + 8192 <= L`
    with the three numbers reported; with the window monkeypatched to 10,000 it
    FAILS naming them. The artifact's own cap must equal the tuple's (one 8192),
    its model / revision the row's, its task one the configuration raises."""
    cfgs = _oos_plan(L)
    on = L.oos_on_configurations(cfgs)
    L_res = sg.PETRI_MAX_MODEL_LEN
    S_res = sg.BUDGET_BY_MODEL[_OOS_MODEL].seqs_at(L_res)
    budget = _budget(prompt_max=2000)
    rep = L.oos_prompt_budget_fit_report(budget, cfgs, pass_no=1)
    assert sorted(rep) == sorted(c["name"] for c in on)
    for name, rec in rep.items():
        assert rec["prompt_max"] == 2000 and rec["cap"] == C.OOS_THINKING_MAX_TOKENS == 8192
        assert rec["max_model_len"] == L_res and rec["total"] == 2000 + 8192
        assert rec["margin"] == L_res - 2000 - 8192 > 0
        assert rec["task"] == C.GSM8K_STANDARD_TASK and rec["pass"] == 1
        assert rec["max_num_seqs"] == S_res and rec["seqs_source"] == L.MODEL_BUDGET_SOURCE
        assert rec["p99"] == budget["p99"] and rec["chat_template_hash"] == "a" * 16
        geom, _ = L.served_geometry(next(c for c in on if c["name"] == name), 1)
        assert (rec["max_model_len"], rec["max_num_seqs"], rec["window_source"]) == (
            geom.max_model_len, geom.max_num_seqs, geom.window_source)
    # the window monkeypatched to 10,000: refused, the three numbers named
    def small(cfg):
        return SimpleNamespace(max_model_len=10000, max_num_seqs=1, window_source="test",
                               seqs_source="test")
    with pytest.raises(C.OutOfSupportPromptBudgetError,
                       match=r"prompt max 2000 \+ cap 8192 = 10192 > the resolved max_model_len 10000"):
        L.oos_prompt_budget_fit_report(budget, cfgs, pass_no=1, geom_of=small)
    # exactly at the boundary fits (margin 0); one token more overflows at the real window
    assert L.oos_prompt_budget_fit_report(_budget(prompt_max=L_res - 8192), cfgs, pass_no=1)[
        on[0]["name"]]["margin"] == 0
    with pytest.raises(C.OutOfSupportPromptBudgetError, match=r"\(over by 1\)"):
        L.oos_prompt_budget_fit_report(_budget(prompt_max=L_res - 8192 + 1), cfgs, pass_no=1)
    # the artifact's cap must be the run's cap (an artifact written at the 2048 floor refuses)
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="written for cap 2048 but the run's cap is 8192"):
        L.oos_prompt_budget_fit_report(_budget(cap=2048), cfgs, pass_no=1)
    # an artifact for another revision, arm or task refuses
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="but this configuration serves"):
        L.oos_prompt_budget_fit_report(_budget(revision="b" * 40), cfgs, pass_no=1)
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="tokenized the 'off' render"):
        L.oos_prompt_budget_fit_report(_budget(arm=C.THINKING_OFF, cap=8192), cfgs, pass_no=1)
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="budgets 'inspect/math' but"):
        L.oos_prompt_budget_fit_report(_budget(cap_task="inspect/math"), cfgs, pass_no=1)
    # nothing out of support planned: nothing to check, an empty report (never a refusal)
    plain = L.plan_configurations(cells=["none"], modes=["off"], controls=["gsm8k"], model=_OOS_MODEL)
    assert L.oos_prompt_budget_fit_report(budget, plain, pass_no=1) == {}


def test_oos_budget_template_matches_binds_the_artifact_to_the_identity_digest(L):
    """At preflight: the artifact's digest must equal the object-
    identity report's `chat_template_hash` (the same pinned tokenizer) —
    through the ONE equality the worker / driver later apply to endpoint.json.
    A differing digest refuses naming both; a MISSING identity digest refuses
    too (a comparison that cannot run is not one that passed)."""
    budget = _budget(digest="a" * 16)
    assert L.oos_budget_template_report(budget, "a" * 16) == {
        "chat_template_hash": "a" * 16, "identity_chat_template_hash": "a" * 16, "equal": True}
    with pytest.raises(AssertionError) as e:
        L.oos_budget_template_report(budget, "b" * 16)
    assert "a" * 16 in str(e.value) and "b" * 16 in str(e.value) and "nobody is serving" in str(e.value)
    with pytest.raises(AssertionError, match="object-identity report .* carries no 16-hex chat_template_hash"):
        L.oos_budget_template_report(budget, None)


def test_the_preflight_runs_the_budget_script_in_the_client_venv_and_names_both_checks(
        L, tmp_path, monkeypatch):
    """The preflight body: after the object-identity check (whose digest the
    template check reads), `chk("oos_cap_fits_resolved_window")` runs the CPU
    script in the CLIENT venv against the STAGED tree (offline, the tree's own
    plan and pins, the tuple's model) and checks the fit; then
    `chk("oos_budget_template_matches")`. The artifact lands under the
    preflight's own volume dir (the preflight cannot know the containers'
    --out-suffix out-root) and in the report."""
    src = LAUNCHER.read_text()
    body = src[src.index("def preflight("):src.index("    def datasets():")]
    i_id = body.index("chk(OBJECT_IDENTITY_CHECK, object_identity)")
    i_fit = body.index('chk("oos_cap_fits_resolved_window", cap_fits)')
    i_tpl = body.index('chk("oos_budget_template_matches", template_matches)')
    assert i_id < i_fit < i_tpl
    assert 'rep["oos_prompt_budget"] = {"path": budget_path, **budget}' in body
    assert "oos_prompt_budget_fit_report(budget, budgeted, pass_no=pass_no, tree=tree)" in body   # the LAUNCH's pass
    assert 'pass_no = int(plan_args.get("pass_no") or 1)' in body and "pass_no=1" not in body   # never a hardcoded pass 1 (H3)
    assert '(rep["checks"].get(OBJECT_IDENTITY_CHECK) or {}).get("chat_template_hash")' in body
    row = C.model_row(_OOS_MODEL)
    assert L.oos_prompt_budget_path(row, FULL_SHA) == (
        f"{C.VOL_GEN_ROOT}/_preflight/qwen35_4b/{FULL_SHA}/{C.OOS_PROMPT_BUDGET_FILENAME}")
    assert "budget_path = oos_prompt_budget_path(row, full_sha)" in body        # scoped by THIS tree
    with pytest.raises(ValueError, match="FULL 40-hex tree sha"):
        L.oos_prompt_budget_path(row, "abc1234")
    assert L.oos_prompt_budget_argv("/vol/t", row, "/vol/b.json", tree_sha=FULL_SHA) == [
        L.CLIENT_PY, "/vol/t/scripts/tinycap_oos_prompt_budget.py", "--model", "qwen35_4b",
        "--arm", "on", "--repo-root", "/vol/t", "--offline", "--tree-sha", FULL_SHA, "--out", "/vol/b.json"]
    assert L.CLIENT_PY.startswith(C.CLIENT_VENV)
    # the runner: the script's exit and the reader's validation both gate
    out = tmp_path / "pf" / "b.json"
    calls = []

    def fake_run(argv, **kw):
        calls.append((argv, kw))
        Path(argv[-1]).write_text(json.dumps(_budget()))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(L.subprocess, "run", fake_run)
    got = L.run_oos_prompt_budget("/vol/t", row, str(out), tree_sha=FULL_SHA, python="/py")
    assert got == _budget() and calls[0][0][0] == "/py"
    assert calls[0][1]["env"]["PYTHONPATH"] == "/vol/t/src:/vol/t/scripts"
    assert calls[0][1]["timeout"] == L.OOS_PROMPT_BUDGET_TIMEOUT_S
    monkeypatch.setattr(L.subprocess, "run", lambda argv, **kw: SimpleNamespace(
        returncode=1, stdout="", stderr="Traceback: boom"))
    with pytest.raises(AssertionError, match="failed in the client venv .*boom"):
        L.run_oos_prompt_budget("/vol/t", row, str(out), tree_sha=FULL_SHA)

    def torn(argv, **kw):
        Path(argv[-1]).write_text("{torn")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(L.subprocess, "run", torn)
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="unreadable"):
        L.run_oos_prompt_budget("/vol/t", row, str(out), tree_sha=FULL_SHA)


def test_two_preflights_of_one_model_from_two_trees_never_read_each_others_artifact(
        L, tmp_path, monkeypatch):
    """The artifact path was model-scoped, so two concurrent
    preflights of the same model from different staged trees wrote and read ONE
    file, each able to take the other's atomically-replaced artifact as its
    own; and ``run_oos_prompt_budget`` never compared the artifact's
    ``tree_sha`` to the sha it asked for. Now the path is scoped by the full
    tree sha (two trees -> two files), the object read back must carry the
    requested sha (a sibling's file at the same path is refused naming both),
    and downstream readers take the validated OBJECT from the report."""
    row = C.model_row(_OOS_MODEL)
    sha_a, sha_b = "a" * 40, "b" * 40
    path_a, path_b = L.oos_prompt_budget_path(row, sha_a), L.oos_prompt_budget_path(row, sha_b)
    assert path_a != path_b and path_a.endswith(f"/{sha_a}/{C.OOS_PROMPT_BUDGET_FILENAME}")
    assert Path(path_a).parent.parent == Path(path_b).parent.parent      # one model dir, two sha dirs
    root = tmp_path / "vol"

    def local(p):                                    # the volume path under tmp_path
        return str(root / Path(p).relative_to("/"))

    def script_writes_its_own_sha(argv, **kw):
        # what the CPU script does: --tree-sha from its argv into the artifact at --out
        sha = argv[argv.index(C.OOS_PROMPT_BUDGET_ARG_TREE_SHA) + 1]
        doc = {**_budget(prompt_max=2000 + int(sha[0] == "b")), "tree_sha": sha}
        Path(argv[-1]).write_text(json.dumps(C.validate_oos_prompt_budget(doc)))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(L.subprocess, "run", script_writes_its_own_sha)
    # two preflights, interleaved: A writes, B writes, A reads back, B reads back
    got_a = L.run_oos_prompt_budget("/vol/t_a", row, local(path_a), tree_sha=sha_a, python="/py")
    got_b = L.run_oos_prompt_budget("/vol/t_b", row, local(path_b), tree_sha=sha_b, python="/py")
    assert (got_a["tree_sha"], got_a["max"]) == (sha_a, 2000)
    assert (got_b["tree_sha"], got_b["max"]) == (sha_b, 2001)
    assert C.read_oos_prompt_budget(local(path_a)) == got_a and C.read_oos_prompt_budget(local(path_b)) == got_b
    # the race, spelled out: an artifact at MY path carrying ANOTHER tree's sha is
    # refused naming both — never read as mine
    def sibling_wrote_here(argv, **kw):
        Path(argv[-1]).write_text(json.dumps({**_budget(), "tree_sha": sha_b}))
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(L.subprocess, "run", sibling_wrote_here)
    with pytest.raises(C.OutOfSupportPromptBudgetError) as ei:
        L.run_oos_prompt_budget("/vol/t_a", row, local(path_a), tree_sha=sha_a, python="/py")
    assert sha_a in str(ei.value) and sha_b in str(ei.value) and "not the file this invocation wrote" in str(ei.value)
    # a script that dropped the sha (null) is refused the same way
    def sha_less(argv, **kw):
        Path(argv[-1]).write_text(json.dumps({**_budget(), "tree_sha": None}))
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(L.subprocess, "run", sha_less)
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="names tree_sha None"):
        L.run_oos_prompt_budget("/vol/t_a", row, local(path_a), tree_sha=sha_a, python="/py")
    # downstream: the preflight body hands the validated OBJECT on (the report and
    # the two checks read `budget`, never a re-resolved path)
    src = LAUNCHER.read_text()
    body = src[src.index("def preflight("):src.index("    def datasets():")]
    assert body.count("oos_prompt_budget_path(") == 1                        # resolved once, inside cap_fits
    assert "oos_prompt_budget_fit_report(budget, budgeted" in body and "oos_budget_template_report(budget, identity)" in body
    assert src.count("oos_prompt_budget_path(") == 2                          # the def + that one call


def test_a_torn_done_out_of_support_store_is_repaired_or_refused_through_both_doors_never_skipped(
        L, tmp_path, monkeypatch):
    """The two admission sites TOGETHER on one torn store: a DONE
    out-of-support store (marker + full summary, no outcome record). The
    launcher door files it under `why_not` (not done, so a standup follows);
    the driver's dispatch under the override then REBUILDS the record from the
    retained log — or, with no log, REFUSES by name — and in neither case
    reports the task `skipped`. After a repair the launcher door reads DONE,
    and a second dispatch skips. Nothing is generated at any point."""
    import asyncio

    from test_inspect_served import (
        BASE_VMODEL,
        FIXTURE_ROUTES,
        VMODEL,
        _models_doc,
        _oos_log,
        _oos_plan,
        _Store,
        _stub_model,
    )

    from sisteer.capeval.tiers import inspect_served as isv
    from sisteer.capeval.tinycap import completion

    task = C.GSM8K_STANDARD_TASK
    oos = _oos_expected(L)
    out = tmp_path / "thinking_on"
    cell_dir = _store(out, oos, tasks=["gsm8k"])
    _finish(cell_dir, task, C.GSM8K_PAIRED_N)
    completion.write_task_summary(cell_dir, task, _oos_summary(task, n=C.GSM8K_PAIRED_N))
    assert completion.store_is_out_of_support(cell_dir)
    # -- the launcher door: not done, the torn record named
    report = L.resume_report(out, ["gsm8k"], oos)
    assert report["controls"]["gsm8k"]["done_in"] is None
    why = report["controls"]["gsm8k"]["why_not"][cell_dir.name][task]
    assert why.startswith("no outcomes/inspect__gsm8k_paired200.json beside the done marker")
    assert L.remaining_controls(out, ["gsm8k"], oos) == ["gsm8k"]
    # -- the driver's dispatch under the override, on the SAME store
    retained: dict = {}

    async def _no_generation(*a, **k):
        raise AssertionError("a torn done store is never re-generated")

    monkeypatch.setattr("inspect_ai.eval_async", _no_generation)
    monkeypatch.setattr(isv, "run_one_task", _no_generation)
    monkeypatch.setattr(isv, "get_models", lambda **k: _models_doc())
    monkeypatch.setattr(isv, "preflight_models", lambda **k: [VMODEL, BASE_VMODEL])
    # the dispatch refuses a target without the tinycap provider (S3.3), so the
    # stand-in carries one (its regime accumulator and route), as the real model does
    monkeypatch.setattr(isv, "build_model", lambda *a, **k: _stub_model())
    monkeypatch.setattr(isv, "plans_for_atoms", lambda atoms, **k: [_oos_plan()])
    monkeypatch.setattr(isv, "reusable_log",
                        lambda d, *, expected_n, expected_ids=None, expected_max_tokens=None: retained.get("log"))
    dispatch = dict(atoms=["gsm8k"], route=FIXTURE_ROUTES[1], routes=FIXTURE_ROUTES, system=None,
                    store=_Store(cell_dir), template={}, thinking=C.THINKING_ON,
                    base_url="http://127.0.0.1:1/v1", reasoning_parser=None,
                    out_of_support=True, prompt_opened_think=True)
    # no retained log: REFUSED by name — not skipped, nothing written, still not done
    with pytest.raises(isv.InspectServedError, match=r"no completed \.eval log for sample 0 survives"):
        asyncio.run(isv.run_inspect_tasks(**dispatch))
    assert completion.read_task_outcomes(cell_dir, task) is None
    assert L.remaining_controls(out, ["gsm8k"], oos) == ["gsm8k"]
    assert L.invalid_done_marker_reason(cell_dir, task) == why
    # the retained log survives: REBUILT (never skipped), and the launcher door now reads DONE
    retained["log"] = _oos_log(n_unscored=0)
    filed = asyncio.run(isv.run_inspect_tasks(**dispatch))[task]
    assert "skipped" not in filed
    assert filed["repaired_outcomes"]["logs_read"] == 1 and filed["repaired_outcomes"]["n_attempted_items"] == 200
    assert filed["repaired_outcomes"]["was_invalid_because"] == why
    rec = completion.read_task_outcomes(cell_dir, task)
    assert rec is not None and rec["n_attempted_items"] == 200 and rec["producer_tree_sha"] == FULL_SHA
    assert L.invalid_done_marker_reason(cell_dir, task) is None
    assert L.resume_report(out, ["gsm8k"], oos)["controls"]["gsm8k"]["done_in"] == cell_dir.name
    assert L.remaining_controls(out, ["gsm8k"], oos) == []
    # repaired: the next dispatch skips, and both doors agree
    assert asyncio.run(isv.run_inspect_tasks(**dispatch))[task]["skipped"] == "done marker present"
    assert completion.is_done(cell_dir, task, require_outcomes=True)


def _oos_dry_run(L, **kw):
    # native-serving-default: the authorised out-of-support experiment is a LENS launch (a PSR cell)
    kw.setdefault("serving_path", "lens")
    L.main(app_name=L.APP_NAME, sha="abc1234", pass1=True, model=_OOS_MODEL,
           cells=f"none,{_OOS_CELL}", modes="off,on", controls="gsm8k",
           allow_out_of_support_thinking=True, dry_run=True, **kw)


def test_the_dry_run_prints_the_cap_the_resolved_geometry_and_the_prompt_budget(L, capsys, tmp_path):
    """S4.5 / C2: per out-of-support configuration the dry run prints the raised
    cap (from the tuple), the RESOLVED geometry L x S with its sources, and the
    prompt budget — max / p99 / digest / margin from a `--oos-prompt-budget`
    artifact, or an explicit NOT MEASURED line that says how to get one. An
    artifact that does not fit prints DOES NOT FIT and exits non-zero; a
    malformed one refuses at the flag."""
    L_res = sg.PETRI_MAX_MODEL_LEN
    S_res = sg.BUDGET_BY_MODEL[_OOS_MODEL].seqs_at(L_res)
    _oos_dry_run(L)
    out = capsys.readouterr().out
    cap_line = ("thinking-ON cap (contracts.AUTHORISED_OOS_THINKING.max_tokens_on): "
                f"inspect/gsm8k_paired200 -> {C.OOS_THINKING_MAX_TOKENS}")
    assert out.count(cap_line) == 2                                           # the two ON configurations
    geo_line = f"resolved launch geometry (pass 1): L {L_res} x S {S_res} (window "
    assert out.count(geo_line) == 2
    assert out.count("prompt budget: NOT MEASURED in this dry run") == 2
    assert f"room under L for the prompt is {L_res} - {C.OOS_THINKING_MAX_TOKENS} = {L_res - C.OOS_THINKING_MAX_TOKENS}" in out
    assert f"{L.OOS_PROMPT_BUDGET_FLAG} <{C.OOS_PROMPT_BUDGET_FILENAME}>" in out
    assert f"{C.OOS_PROMPT_BUDGET_SCRIPT} --model qwen35_4b --arm on" in out
    # the OFF configurations carry none of it: the lines sit inside the OUT OF SUPPORT block
    blocks = out.split("== tinycap-")
    for b in blocks:
        if "thinking=off" in b.splitlines()[0]:
            assert "thinking-ON cap" not in b and "prompt budget" not in b
    # with an artifact: max / p99 / digest / margin
    p = tmp_path / C.OOS_PROMPT_BUDGET_FILENAME
    p.write_text(json.dumps(_budget(prompt_max=2000, digest="c" * 16)))
    _oos_dry_run(L, oos_prompt_budget=str(p))
    out = capsys.readouterr().out
    line = (f"prompt budget (template {'c' * 16}, arm on, n {C.GSM8K_PAIRED_N}, Qwen/Qwen3.5-4B@"
            f"{C.model_row(_OOS_MODEL).rev[:12]}): max 2000, p99 {_budget()['p99']}, p50 ")
    assert out.count(line) == 2
    assert out.count(f"max + cap 8192 = {2000 + 8192} <= L {L_res}, margin {L_res - 2000 - 8192}") == 2
    assert "NOT MEASURED" not in out
    # an artifact that does not fit: DOES NOT FIT printed, then a non-zero exit naming the numbers
    p.write_text(json.dumps(_budget(prompt_max=L_res - 8192 + 5)))
    with pytest.raises(SystemExit) as e:
        _oos_dry_run(L, oos_prompt_budget=str(p))
    out = capsys.readouterr().out
    assert out.count("prompt budget: DOES NOT FIT") == 2
    assert f"prompt max {L_res - 8192 + 5} + cap 8192 = {L_res + 5} > the resolved max_model_len {L_res}" in str(e.value)
    assert L.OOS_PROMPT_BUDGET_FLAG in str(e.value)
    # a malformed artifact refuses at the flag, before any plan is printed
    p.write_text("{torn")
    with pytest.raises(SystemExit, match=f"{L.OOS_PROMPT_BUDGET_FLAG}: .*unreadable"):
        _oos_dry_run(L, oos_prompt_budget=str(p))
    # the __main__ argparse spells the flag and threads it to main
    src = LAUNCHER.read_text()
    assert 'ap.add_argument(OOS_PROMPT_BUDGET_FLAG, dest="oos_prompt_budget"' in src
    assert "oos_prompt_budget=a.oos_prompt_budget" in src
    assert L.OOS_PROMPT_BUDGET_FLAG == "--oos-prompt-budget"


def test_the_pf_report_s_budget_is_printed_by_main(L, monkeypatch, capsys):
    # native-serving-default: the launcher's default is the native engine now; this test pins the LENS
    # launch (a PSR cell / the historical composition), so it states the path explicitly
    """After --pf, main prints the budget the preflight measured (path, max,
    p99, cap, digest) beside the object digests — the launch's own record of
    the CPU gate it passed."""
    budget = _budget(prompt_max=2100, digest="d" * 16)

    def _remote(sha, plan_args, expect):
        rep = _preflight_report(L, L.plan_for(**plan_args))
        rep["oos_prompt_budget"] = {"path": "/vol/x/oos_prompt_budget.json", **budget}
        return rep
    monkeypatch.setattr(L, "preflight", SimpleNamespace(remote=_remote))
    monkeypatch.setattr(L, "run_configuration", _spawn_only(_done))
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, pf=True, model=_OOS_MODEL,
           cells=f"none,{_OOS_CELL}", modes="off,on", controls="gsm8k",
           allow_out_of_support_thinking=True)
    out = capsys.readouterr().out
    assert ("[tinycap] prompt budget (from --pf, /vol/x/oos_prompt_budget.json): n 200, max 2100, "
            f"p99 {budget['p99']}, cap 8192 (inspect/gsm8k_paired200), template {'d' * 16}") in out


# --------------------------------------------------------------------------- #
# fit-support transfer, experiment 1 — plan S4 (Phase B2; test T4): the PURE    #
# identity expectation, the worker's attestation compare against a SYNTHETIC    #
# contract-shaped endpoint (the lens half is not in this tree), the declared    #
# standup argv, the pre-standup torn-store check, the dry run's C2 lines        #
# --------------------------------------------------------------------------- #


def _oos_expected(L, *, cell=_OOS_CELL, mode="on", tree=None, tree_sha=FULL_SHA, **kw):
    """The REAL out-of-support expectation for one leg of the authorised 2x2:
    planned under the override on the authorised model, so the marker, the
    raised cap and the contract literal come from the launcher's own
    derivation (never a ``_replace`` on an in-support one). ``mode="off"``
    plans the in-support OFF leg (no marker), as the run script does."""
    return _expected(L, cell=cell, mode=mode, controls=["gsm8k"], model=_OOS_MODEL,
                     allow_out_of_support_thinking=True, tree=tree, tree_sha=tree_sha, **kw)


#: the contract's literals, typed deliberately in the fixtures below (they are
#: what the code must READ — a fixture built from the constants under test would
#: pass with a mis-spelled constant)
_DECL_KEY = "lens_serve_declarations"
_DECL_ERR_KEY = "lens_serve_declarations_error"
_DECL_VALUE = {"serve_out_of_support": ["thinking"]}
_TWIN = "none__lens0"
_TEMPLATE = "a" * 16


def _oos_endpoint(L, cfg, *, artifact_sha=FAKE_DIGEST, template=_TEMPLATE, **over) -> dict:
    """A CONTRACT-shaped ``endpoint.json`` for one configuration's standup:
    the per-cell declarations map (§2.5.1 / §2.5.7 — the steered cell AND its
    twin on the declared standup, ``{}`` otherwise) with a null error key; the
    served template digest; the attested resumed-prefill policies (the
    VARIANT key on the declared standup); the artifact digests (§2.5.7);
    ``lens_salt_policy`` (§2.5.9); an attested ``per_row``. ``over``
    replaces keys; delete a key from the result to model its absence."""
    sp = L.spellings_for(cfg)
    lens = C.is_lens_cell(cfg["cell"])
    declared = L.standup_declared(cfg)
    rec = {"lens_reduction_mode": "per_row", "lens_reduction_mode_error": None,
           _DECL_KEY: ({sp.harness: dict(_DECL_VALUE), _TWIN: dict(_DECL_VALUE)} if declared else {}),
           _DECL_ERR_KEY: None,
           "chat_template_hash": template,
           "resumed_prefill_policy_cells": ({sp.harness: "conditional", _TWIN: "conditional"}
                                            if lens else {}),
           "resumed_prefill_policy_error": None,
           "object_artifact_sha256": artifact_sha if lens else None,
           "object_artifact_sha256_by_cell": {sp.harness: artifact_sha} if lens else {},
           "lens_salt_policy": "attested"}
    rec.update(over)
    return rec


def _watched_leg_world(L, tmp_path, monkeypatch, cfg, *, record=None, drop=()):
    """One planned leg with a staged tree at FULL_SHA and a fake standup
    attesting ``record`` (default the contract-shaped endpoint of ``cfg``, minus
    the keys in ``drop``), watched like :func:`_watched_world`: every Popen
    argv, the driver's ``subprocess.run``, the launcher's teardown — the body
    :func:`_oos_world` and :func:`_offcap_world` share."""
    shutil.rmtree(f"/tmp/srv_{cfg['name']}", ignore_errors=True)
    repos = tmp_path / "repos"
    _tree(repos, "f6cf462")
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    monkeypatch.setenv(C.LENS_STACKED_REDUCTIONS_ENV, "0")
    out = tmp_path / "gen"
    cfg = dict(cfg, out_dir=str(out))
    rec = _oos_endpoint(L, cfg) if record is None else dict(record)
    for k in drop:
        rec.pop(k, None)
    _FakeStandupProc.record = rec
    seen: dict = {"popen": [], "driver": [], "torn": []}
    real_teardown = L._tear_down_standup

    def _popen(argv, *a, **k):
        seen["popen"].append(list(argv))
        return _FakeStandupProc(argv, *a, **k)

    def _run(cmd, *a, **k):
        seen["driver"].append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def _teardown(stop, standup):
        real_teardown(stop, standup)
        seen["torn"].append(stop.exists())

    monkeypatch.setattr(L.subprocess, "Popen", _popen)
    monkeypatch.setattr(L.subprocess, "run", _run)
    monkeypatch.setattr(L, "_tear_down_standup", _teardown)
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    return cfg, out, seen


def _oos_world(L, tmp_path, monkeypatch, *, cell=_OOS_CELL, mode="on", record=None, drop=()):
    """One leg of the authorised 2x2 (:func:`_watched_leg_world`)."""
    (cfg,) = L.plan_for(cells=[cell], modes=[mode], controls=["gsm8k"], model=_OOS_MODEL,
                        allow_out_of_support_thinking=True)
    return _watched_leg_world(L, tmp_path, monkeypatch, cfg, record=record, drop=drop)


def _run_oos(L, cfg, *, budget=_DEFAULT, **kw):
    """``_run_configuration`` on a 2x2 leg: the fitted digest and (default) the
    preflight's artifact for THIS tree, the srv dir cleared after."""
    kw.setdefault("object_digests", _digests_for(cfg))
    if budget is _DEFAULT:
        budget = _budget()
    try:
        return L._run_configuration(cfg, "f6cf462", oos_prompt_budget=budget, **kw)
    finally:
        shutil.rmtree(f"/tmp/srv_{cfg['name']}", ignore_errors=True)


def _standup_popens(seen) -> list:
    return [a for a in seen["popen"] if "capctrl_serving_up.py" in " ".join(a[:2])]


def _refused(L, rec, seen, status):
    """A post-standup refusal's shape: the typed status, the standup torn down
    through the launcher's teardown, no ledger, no driver, not an OK status."""
    assert rec["status"] == status, (rec["status"], rec.get("error"))
    assert "standup_s" in rec and seen["torn"] == [True]
    assert len(seen["popen"]) == 1 and seen["driver"] == []
    assert "ledger_url" not in rec and "driver_argv" not in rec
    assert rec["status"] not in L.REC_OK_STATUSES
    return rec["error"]


def test_the_four_configurations_and_the_pure_expectation_of_the_two_further_keys(L):
    """T4 (S4.1 / S4.2): the authorised kwargs plan EXACTLY four configurations
    — {none, psr} x {off, on} — with the marker on the ON pair only; the PURE
    expectation (no server anywhere in its derivation) carries
    `max_tokens_by_task` = the tuple's raised cap on the ON pair only and
    `serve_out_of_support` = the contract literal ["thinking"] on psr/on, None
    on none/on (contract §2.5.10 Q8 row 5); both are omitted from an in-support
    expectation, whose as_dict is byte-identical to before."""
    cfgs = _oos_plan(L)
    assert {(c["cell"], c["mode"]) for c in cfgs} == {
        (cell, mode) for cell in ("none", _OOS_CELL) for mode in ("off", "on")}
    assert len(cfgs) == 4
    for cfg in cfgs:
        geom, _ = L.served_geometry(cfg, 1)
        exp = L.expected_identity(cfg, geom, pass_no=1, controls=cfg["controls"], samples=1)
        on = cfg["mode"] == C.THINKING_ON
        assert cfg.get(C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING, False) is on
        assert exp.out_of_support_thinking is on
        assert L.standup_declared(cfg) is (on and cfg["cell"] == _OOS_CELL) is exp.declared
        if on:
            assert dict(exp.max_tokens_by_task) == {C.GSM8K_STANDARD_TASK: C.oos_max_tokens_on("gsm8k_paired200")}
            assert dict(exp.max_tokens_by_task) == {"inspect/gsm8k_paired200": 8192}
            assert exp.serve_out_of_support == (C.OOS_CELL_DECLARATION_EXPECTED if cfg["cell"] == _OOS_CELL else None)
            assert exp.serve_out_of_support == (["thinking"] if cfg["cell"] == _OOS_CELL else None)
            # the declaration expected is the CONTRACT's literal, never something read off a server
            assert exp.serve_out_of_support == C.expected_serve_out_of_support(
                lens=C.is_lens_cell(cfg["cell"]), arm="on", out_of_support=True)
        else:
            assert dict(exp.max_tokens_by_task) == {} and exp.serve_out_of_support is None
            # in-support: the same dict an in-support expectation always produced
            assert C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING not in exp.as_dict()
            assert C.IDENTITY_KEY_MAX_TOKENS_BY_TASK not in exp.as_dict()
            assert C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT not in exp.as_dict()
            assert exp.optional_identity() == {}
    # the OFF psr leg keys exactly as the campaign's would: one identity, no override in it
    off = next(c for c in cfgs if c["cell"] == _OOS_CELL and c["mode"] == "off")
    plain = L.plan_configurations(cells=[_OOS_CELL], modes=["off"], controls=["gsm8k"], model=_OOS_MODEL)[0]
    assert {k: v for k, v in off.items() if k != "standup_extra"} == {k: v for k, v in plain.items() if k != "standup_extra"}
    assert off["standup_extra"] == plain["standup_extra"]


def test_the_optional_identity_keys_are_derived_from_the_contracts_field_set(L, monkeypatch):
    """The COUPLING with the driver lane, stated as a test: which optional keys
    the expectation carries is read off `contracts.tinycap_identity_fields(...)`
    beyond the base fields — the marker and the two further keys since the
    driver's set widened, valued from the expectation; a derivation
    naming an optional key the launcher cannot value refuses by name (never a
    silent gap). In-support: empty."""
    exp = _oos_expected(L)
    assert exp.out_of_support_thinking and exp.declared
    # the derivation, since the driver's field set widened: the
    # marker and the two further keys, each valued from the expectation
    fields = C.tinycap_identity_fields(out_of_support_thinking=True)
    assert set(fields) - set(C.TINYCAP_IDENTITY_FIELDS) == {
        C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING, C.IDENTITY_KEY_MAX_TOKENS_BY_TASK,
        C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT}
    assert set(exp.as_dict()) - set(_expected(L).as_dict()) == set(fields) - set(C.TINYCAP_IDENTITY_FIELDS)
    assert exp.optional_identity() == {C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING: True,
                                       C.IDENTITY_KEY_MAX_TOKENS_BY_TASK: {"inspect/gsm8k_paired200": 8192},
                                       C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT: ["thinking"]}
    d = exp.as_dict()
    assert d[C.IDENTITY_KEY_MAX_TOKENS_BY_TASK] == {"inspect/gsm8k_paired200": 8192}
    assert d[C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT] == ["thinking"] and d[C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING] is True
    none_on = _oos_expected(L, cell="none")
    assert none_on.as_dict()[C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT] is None and not none_on.declared
    assert _expected(L).optional_identity() == {} and _oos_expected(L, mode="off").optional_identity() == {}
    # a key the launcher cannot value: refused by name
    monkeypatch.setattr(C, "tinycap_identity_fields",
                        lambda *, out_of_support_thinking=False, off_cap_override=False, files_math500=False, **_kw:
                        (*C.TINYCAP_IDENTITY_FIELDS, "mystery_key")
                        if out_of_support_thinking else C.TINYCAP_IDENTITY_FIELDS)
    with pytest.raises(L.IdentityUnderivable, match="mystery_key"):
        exp.as_dict()


def test_untrusted_store_reason_names_the_cap_the_declaration_and_the_route_mirror(L, tmp_path):
    """An 8192 store and a 2048 store hash differently AND the
    resume guard names `max_tokens_by_task`; a store hashed under another
    declaration literal is refused naming `serve_out_of_support`; the
    manifest's `route.lens_serve_declarations` must MIRROR the hashed
    expectation under this cell's attestation key (the variant spelling on the
    declared standup): absent on psr/on, or disagreeing, refuses; `{}` on the
    none/on store passes; the base route's key present refuses. The cross-
    support refusals of the existing test stand beside these."""
    task = C.GSM8K_STANDARD_TASK
    psr_on = _oos_expected(L)
    ok = _store(tmp_path / "ok", psr_on, tasks=["gsm8k"])
    assert L.untrusted_store_reason(ok, expected=psr_on) is None
    _m8192, chash8192 = _manifest(psr_on, tasks=["gsm8k"])
    _m2048, chash2048 = _manifest(psr_on, tasks=["gsm8k"], max_tokens_by_task={task: 2048})
    assert chash8192 != chash2048                                  # different keys by construction
    why = L.untrusted_store_reason(_store(tmp_path / "cap", psr_on, tasks=["gsm8k"],
                                          max_tokens_by_task={task: 2048}), expected=psr_on)
    assert C.IDENTITY_KEY_MAX_TOKENS_BY_TASK in why and "2048" in why and "8192" in why and "D3.3" in why
    why = L.untrusted_store_reason(_store(tmp_path / "decl", psr_on, tasks=["gsm8k"],
                                          serve_out_of_support=None), expected=psr_on)
    assert f"tinycap.{C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT}: the store records None but this run hashes ['thinking']" in why
    # the route mirror: the map absent on the declared store, or disagreeing under the key
    key = "si_safety__psr_kl_all@thinking"
    why = L.untrusted_store_reason(_store(tmp_path / "r0", psr_on, tasks=["gsm8k"], route_declarations={}),
                                   expected=psr_on)
    assert f"manifest.route.{C.ROUTE_KEY_SERVE_DECLARATIONS}[{key!r}] core None does not mirror" in why
    why = L.untrusted_store_reason(_store(tmp_path / "r1", psr_on, tasks=["gsm8k"],
                                          route_declarations={key: {"serve_out_of_support": ["thinking", "x"]}}),
                                   expected=psr_on)
    assert f"manifest.route.{C.ROUTE_KEY_SERVE_DECLARATIONS}[{key!r}]" in why and "'x'" in why
    # a route map keyed on the BARE id is a map that does not name this cell: refused
    why = L.untrusted_store_reason(_store(tmp_path / "r2", psr_on, tasks=["gsm8k"],
                                          route_declarations={"si_safety__psr_kl_all": _DECL_VALUE}),
                                   expected=psr_on)
    assert "core None does not mirror" in why
    # the base route's ON leg: {} mirrors None; its own key present refuses
    none_on = _oos_expected(L, cell="none")
    assert L.untrusted_store_reason(_store(tmp_path / "n0", none_on, tasks=["gsm8k"]), expected=none_on) is None
    assert L.untrusted_store_reason(_store(tmp_path / "n1", none_on, tasks=["gsm8k"], route_declarations={}),
                                    expected=none_on) is None
    why = L.untrusted_store_reason(_store(tmp_path / "n2", none_on, tasks=["gsm8k"],
                                          route_declarations={"none": _DECL_VALUE}), expected=none_on)
    assert f"manifest.route.{C.ROUTE_KEY_SERVE_DECLARATIONS}['none'] core ['thinking'] does not mirror" in why
    # a marker that is not the positive True is not this store's marker
    why = L.untrusted_store_reason(_store(tmp_path / "m", psr_on, tasks=["gsm8k"],
                                          mutate=lambda m: None), expected=psr_on)
    assert why is None
    # the in-support OFF leg: no optional key, no route map needed — as before
    off = _oos_expected(L, mode="off")
    assert not off.out_of_support_thinking
    assert L.untrusted_store_reason(_store(tmp_path / "off", off, tasks=["gsm8k"]), expected=off) is None


def test_a_completed_oos_store_is_done_with_no_server_and_a_torn_one_without_a_log_refuses_before_the_standup(
        L, tmp_path, monkeypatch):
    """Through the worker. A completed same-sha
    psr/on store (the three optional keys, done marker + summary + a VALID
    outcomes sidecar) -> `DONE`, no Popen at all. The SAME store minus its
    sidecar is not done; with no retained `.eval` log the worker refuses
    BEFORE the standup (`OOS_TORN_STORE_UNREPAIRABLE`, the driver's repair
    could only refuse) — still no Popen; with a retained log present it
    proceeds to the (fake) standup, where the driver rebuilds the record. A
    store whose hashed `serve_out_of_support` differs is refused by
    `untrusted_store_reason` before any standup — and the run then proceeds to
    a standup of its own (the store is not resumed, never re-keyed)."""
    from sisteer.capeval.tinycap import completion
    task = C.GSM8K_STANDARD_TASK
    cfg, out, seen = _oos_world(L, tmp_path, monkeypatch)
    exp = _oos_expected(L, tree=_tree(tmp_path / "repos", "f6cf462"))
    store = _store(out, exp, tasks=["gsm8k"])
    _finish(store, task, C.GSM8K_PAIRED_N)
    summary = _oos_summary(task, n=C.GSM8K_PAIRED_N)
    completion.write_task_summary(store, task, summary)
    completion.write_task_outcomes(store, task, _oos_outcomes(task, summary))
    rec = _run_oos(L, cfg)
    assert rec["status"] == "DONE" and rec["controls_remaining"] == []
    assert seen["popen"] == [] and "standup_s" not in rec                 # no GPU minute
    assert rec["resume"]["controls"]["gsm8k"]["done_in"] == store.name
    assert rec["expected_identity"][C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT] == ["thinking"]
    assert rec["expected_identity"][C.IDENTITY_KEY_MAX_TOKENS_BY_TASK] == {task: 8192}
    # review round B, P2-3: the pre-standup question is the driver's own — is a COMPLETED log
    # retained? — asked through the client interpreter (`retained_log_reasons`; here a stand-in
    # that applies the reader's FILE-level verdicts in-process, since this venv is not the
    # container's: the real reader is unit-tested in test_inspect_served.py and the subprocess
    # plumbing in `test_the_retained_log_probe_runs_the_reader_in_the_client_interpreter`).
    # A dangling symlink, a directory, a truncated file: NOT a retained log -> no standup.
    probed: list = []

    def file_level_probe(tree, checks, *, python=None):
        probed.append((tree, [dict(c) for c in checks]))
        out = {}
        for c in checks:
            d = Path(c["log_dir"])
            paths = sorted(d.glob("*.eval")) if d.is_dir() else []
            whys = []
            for q in paths:
                if q.is_symlink() and not q.exists():
                    whys.append(f"{q.name}: a dangling symlink")
                elif not q.is_file():
                    whys.append(f"{q.name}: not a regular file")
                elif q.read_bytes() != b"COMPLETE":
                    whys.append(f"{q.name}: unreadable (torn or truncated)")
                else:
                    whys = None
                    break
            out[c["log_dir"]] = (None if whys is None else
                                 (f"no retained .eval log under {d}" if not paths
                                  else f"no completed .eval log under {d}: " + "; ".join(whys)))
        return out

    monkeypatch.setattr(L, "retained_log_reasons", file_level_probe)
    # the sidecar gone: torn, no log -> refused BEFORE the standup, by name
    completion.task_outcomes_path(store, task).unlink()
    rec = _run_oos(L, cfg)
    assert rec["status"] == L.REC_OOS_TORN_STORE_UNREPAIRABLE == "OOS_TORN_STORE_UNREPAIRABLE"
    assert seen["popen"] == [] and "standup_s" not in rec
    assert store.name in rec["error"] and task in rec["error"] and "no retained .eval log" in rec["error"]
    assert str(L.inspect_log_dir(store, task, 0)) in rec["error"]
    assert "a fresh store (a new out-suffix) or an explicit operator repair" in rec["error"]
    assert rec["controls_remaining"] == ["gsm8k"]
    assert L.failed_recs([rec]) == [(cfg["name"], "OOS_TORN_STORE_UNREPAIRABLE")]
    # a dangling symlink, a directory, a truncated file: NOT a retained log -> no standup
    log_dir = L.inspect_log_dir(store, task, 0)
    log_dir.mkdir(parents=True)
    cases = {
        "dangling": lambda: (log_dir / "a.eval").symlink_to(log_dir / "gone.eval"),
        "directory": lambda: (log_dir / "b.eval").mkdir(),
        "truncated": lambda: (log_dir / "c.eval").write_bytes(b"\x00"),
    }
    for name, make in cases.items():
        make()
        rec = _run_oos(L, cfg)
        assert rec["status"] == "OOS_TORN_STORE_UNREPAIRABLE", (name, rec.get("status"))
        assert seen["popen"] == [] and "standup_s" not in rec, name          # NO standup
        assert str(log_dir) in rec["error"] and "no completed .eval log" in rec["error"], name
    assert "dangling symlink" in rec["error"] and "not a regular file" in rec["error"]
    assert "torn or truncated" in rec["error"]
    assert probed and probed[-1][0] == rec["tree"]                             # the STAGED tree
    assert probed[-1][1] == [{"log_dir": str(log_dir), "row_task": task, "expected_n": C.GSM8K_PAIRED_N}]
    # a COMPLETED retained log: on to the standup (the driver's repair reads it)
    (log_dir / "d.eval").write_bytes(b"COMPLETE")
    rec = _run_oos(L, cfg)
    assert rec["status"] == "LEDGER_FAILED" and len(_standup_popens(seen)) == 1
    assert rec["controls_remaining"] == ["gsm8k"]
    # the authorised task's selection underivable is a REASON (fail closed)
    # even for a complete log — the container refuses before the standup
    seen["popen"].clear()

    def underivable_probe(tree, checks, *, python=None):
        return {c["log_dir"]: (f"no completed .eval log under {c['log_dir']}: d.eval: the authorised task's "
                               "item selection could not be derived here (FileNotFoundError: no corpus) — fail closed")
                for c in checks}

    monkeypatch.setattr(L, "retained_log_reasons", underivable_probe)
    rec = _run_oos(L, cfg)
    assert rec["status"] == "OOS_TORN_STORE_UNREPAIRABLE" and seen["popen"] == []
    assert "could not be derived here" in rec["error"] and "fail closed" in rec["error"]
    # a probe that cannot answer is neither verdict: refused BEFORE the standup, its error named
    seen["popen"].clear()

    def broken_probe(tree, checks, *, python=None):
        raise L.RetainedLogProbeError("the client interpreter has no inspect_ai")

    monkeypatch.setattr(L, "retained_log_reasons", broken_probe)
    rec = _run_oos(L, cfg)
    assert rec["status"] == "OOS_TORN_STORE_UNREPAIRABLE" and seen["popen"] == []
    assert "could not answer" in rec["error"] and "no inspect_ai" in rec["error"]
    monkeypatch.setattr(L, "retained_log_reasons", file_level_probe)
    # a store hashed under another declaration: refused before the standup, named; a standup follows
    shutil.rmtree(out)
    other = _store(out, exp, tasks=["gsm8k"], serve_out_of_support=None)
    _finish(other, task, C.GSM8K_PAIRED_N)
    completion.write_task_summary(other, task, summary)
    completion.write_task_outcomes(other, task, _oos_outcomes(task, summary))
    seen["popen"].clear()
    rec = _run_oos(L, cfg)
    why = rec["resume"]["stores"][other.name]
    assert C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT in why and "records None" in why
    assert rec["controls_remaining"] == ["gsm8k"] and len(_standup_popens(seen)) == 1
    # the in-support OFF leg with a plain store and no sidecar: DONE, untouched
    cfg_off, out_off, seen_off = _oos_world(L, tmp_path / "off", monkeypatch, mode="off")
    off = _oos_expected(L, mode="off", tree=_tree(tmp_path / "off" / "repos", "f6cf462"))
    pstore = _store(out_off, off, tasks=["gsm8k"])
    _finish(pstore, task, C.GSM8K_PAIRED_N)
    rec = _run_oos(L, cfg_off, budget=None)
    assert rec["status"] == "DONE" and seen_off["popen"] == []


def test_the_launcher_log_dir_is_inspect_served_s(L, tmp_path):
    """The pre-standup check spells the (task, sample) log dir without Inspect;
    pinned equal to `inspect_served._log_dir` where inspect_ai is importable."""
    pytest.importorskip("inspect_ai", reason="tinycap client venv (hashed lock)")
    from sisteer.capeval.tiers import inspect_served as isv
    store = SimpleNamespace(cell_dir=tmp_path / "cell")
    for task, idx in ((C.GSM8K_STANDARD_TASK, 0), ("inspect/math", 3)):
        assert L.inspect_log_dir(store.cell_dir, task, idx) == isv._log_dir(store, task, idx)
    assert L.inspect_log_dir("/x", C.GSM8K_STANDARD_TASK, 0) == Path(
        "/x") / C.INSPECT_LOGS_SUBDIR / "inspect__gsm8k_paired200__s0"


def test_the_declared_standup_argv_is_one_variant_cell_with_no_none_and_the_keyed_digest(L, tmp_path, monkeypatch):
    """S4.4 / T4: `_standup_cmd` on the psr/on configuration composes EXACTLY
    `--cell safety_refusal__psr_kl_all@thinking` — ONE `--cell`, "none"
    appearing NOWHERE in the argv (contract §2.5.11 Q7(ii)) — and
    `--expect-object-sha256 safety_refusal__psr_kl_all@thinking=<sha>` (the
    keyed form, `OOS_EXPECT_SHA_KEY`), no new flag (§2.5.1); the psr/off and
    `none` configurations' argv are byte-identical to today's; the worker's
    recorded standup argv shows the same; a declared standup with no digest
    refuses."""
    cfgs = {(c["cell"], c["mode"]): c for c in _oos_plan(L)}
    on, off, none_on = cfgs[(_OOS_CELL, "on")], cfgs[(_OOS_CELL, "off")], cfgs[("none", "on")]
    geom, _ = L.served_geometry(on, 1)
    common = ("/tree", "/tree/cfg.yaml")
    argv = L._standup_cmd(*common, on["cell"], "/d", Path("/srv"), Path("/srv/STOP"), geom, 64,
                          on["standup_extra"], object_sha256=FAKE_DIGEST, declared=L.standup_declared(on))
    assert argv.count("--cell") == 1
    assert argv[argv.index("--cell") + 1] == "safety_refusal__psr_kl_all@thinking"
    assert "none" not in argv and not any(a == "none" or a.endswith("__lens0") for a in argv)
    assert argv[argv.index(L.OBJECT_DIGEST_FLAG) + 1] == f"safety_refusal__psr_kl_all@thinking={FAKE_DIGEST}"
    assert argv.count(L.OBJECT_DIGEST_FLAG) == 1
    # no new flag: the ONLY differences from the bare argv are the two spellings
    bare = L._standup_cmd(*common, on["cell"], "/d", Path("/srv"), Path("/srv/STOP"), geom, 64,
                          on["standup_extra"], object_sha256=FAKE_DIGEST)
    assert len(argv) == len(bare)
    assert [(a, b) for a, b in zip(bare, argv) if a != b] == [
        ("safety_refusal__psr_kl_all", "safety_refusal__psr_kl_all@thinking"),
        (FAKE_DIGEST, f"safety_refusal__psr_kl_all@thinking={FAKE_DIGEST}")]
    assert "--reasoning-parser" not in argv                    # no parser composed under the override
    # psr/off and none: byte-identical to today's
    for cfg, sha in ((off, FAKE_DIGEST), (none_on, "")):
        g, _ = L.served_geometry(cfg, 1)
        today = L._standup_cmd(*common, cfg["cell"], "/d", Path("/srv"), Path("/srv/STOP"), g, 64,
                               cfg["standup_extra"], object_sha256=sha)
        assert L._standup_cmd(*common, cfg["cell"], "/d", Path("/srv"), Path("/srv/STOP"), g, 64,
                              cfg["standup_extra"], object_sha256=sha,
                              declared=L.standup_declared(cfg)) == today
        assert not L.standup_declared(cfg)
        assert today[today.index("--cell") + 1] == cfg["cell"] and "@" not in " ".join(today)
        if sha:
            assert today[today.index(L.OBJECT_DIGEST_FLAG) + 1] == sha       # the bare sha, as today
    # a declared standup with no digest refuses (a fitted cell launches only behind --pf)
    with pytest.raises(ValueError, match="needs the preflight-verified artifact sha256"):
        L._standup_cmd(*common, on["cell"], "/d", Path("/srv"), Path("/srv/STOP"), geom, 64,
                       on["standup_extra"], declared=True)
    # the standup the worker actually starts carries the declared spelling
    cfg, _out, seen = _oos_world(L, tmp_path, monkeypatch)
    rec = _run_oos(L, cfg)
    assert rec["status"] == "LEDGER_FAILED"
    (standup,) = _standup_popens(seen)
    assert standup == rec["standup_argv"]
    assert standup[standup.index("--cell") + 1] == "safety_refusal__psr_kl_all@thinking"
    assert standup.count("--cell") == 1 and "none" not in standup
    assert standup[standup.index(L.OBJECT_DIGEST_FLAG) + 1] == f"safety_refusal__psr_kl_all@thinking={FAKE_DIGEST}"
    # the store's cell id, out-root and hashed identity keep the BARE cell (§2.5.11 Q5)
    assert rec["cell"] == _OOS_CELL and "@" not in rec["out"] and rec["expected_identity"]["cell"] == _OOS_CELL


def test_standup_extra_for_refuses_a_mutated_contract_module_at_plan_time(L, monkeypatch):
    """S4.4: with `OOS_STANDUP_CELL_SUFFIX` or `OOS_EXPECT_SHA_KEY` read as None
    (a mutated module — B0 filled both) a DECLARED standup refuses at PLAN
    time, naming B0 and the constant, through `standup_extra_for` and hence
    `plan_configurations`; the none cell, the OFF arm and a call without the
    cell are untouched."""
    row = C.model_row(_OOS_MODEL)
    kw = {"tool_calls": False, "thinking": "on", "controls": ["gsm8k"], "allow_out_of_support": True}
    assert L.standup_extra_for(row, cell=_OOS_CELL, **kw) == []            # filled: no parser, no flag
    for name in ("OOS_STANDUP_CELL_SUFFIX", "OOS_EXPECT_SHA_KEY"):
        with monkeypatch.context() as mp:
            mp.setattr(C, name, None)
            with pytest.raises(L.ThinkingUnservable) as e:
                L.standup_extra_for(row, cell=_OOS_CELL, **kw)
            assert name in str(e.value) and "B0" in str(e.value) and "@thinking" in str(e.value)
            with pytest.raises(L.ThinkingUnservable, match=name):
                L.plan_for(cells=[_OOS_CELL], modes=["on"], controls=["gsm8k"], model=_OOS_MODEL,
                           allow_out_of_support_thinking=True)
            # untouched: the base cell, the OFF arm, no cell named
            assert L.standup_extra_for(row, cell="none", **kw) == []
            assert L.standup_extra_for(row, cell=_OOS_CELL, **{**kw, "thinking": "off"}) == []
            assert L.standup_extra_for(row, **kw) == []
            assert len(L.plan_for(cells=["none", _OOS_CELL], modes=["off"], controls=["gsm8k"],
                                  model=_OOS_MODEL, allow_out_of_support_thinking=True)) == 2


def test_worker_recs_on_the_psr_on_standup_s_contract_shaped_endpoint(L, tmp_path, monkeypatch):
    """T4 (S4.3), the DECLARED standup against a synthetic endpoint shaped by
    the contract's literals. The EXACT two-key map -> no refusal, the raw map
    and error key recorded on the REC; a sibling `serve_experiment` entry is
    IGNORED (§2.5.1); the own core disagreeing (an extra token, a non-list, a
    `serve_experiment` shape under OUR key) -> `OOS_DECLARATION_MISMATCH`
    naming both values; the key absent -> `OOS_DECLARATION_UNATTESTED`; the
    map absent altogether (today's proxy) -> UNATTESTED; the steered key
    present but the twin missing -> MISMATCH naming the twin (§2.5.10); the
    error key non-null -> UNATTESTED. Every refusal tears the standup down
    and reaches neither the ledger nor the driver."""
    key = "si_safety__psr_kl_all@thinking"

    def run(record_over=None, *, drop=()):
        cfg, _out, seen = _oos_world(L, tmp_path / str(len(list(tmp_path.iterdir()))), monkeypatch,
                                    record=None if record_over is None else _oos_endpoint(
                                        L, _oos_world.__wrapped_cfg__, **record_over), drop=drop)
        return _run_oos(L, cfg), seen

    (cfg0,) = L.plan_for(cells=[_OOS_CELL], modes=["on"], controls=["gsm8k"], model=_OOS_MODEL,
                         allow_out_of_support_thinking=True)
    _oos_world.__wrapped_cfg__ = cfg0
    # the exact map: through to the ledger, the raw values recorded
    rec, seen = run()
    assert rec["status"] == "LEDGER_FAILED" and len(seen["popen"]) == 2 and seen["torn"] == []
    assert rec[_DECL_KEY] == {key: _DECL_VALUE, _TWIN: _DECL_VALUE} and rec[_DECL_ERR_KEY] is None
    assert rec["attestation_key"] == key and rec["resumed_prefill_policy"] == "conditional"
    assert rec["lens_salt_policy"] == "attested" and rec["object_artifact_sha256"] == FAKE_DIGEST
    assert rec["object_artifact_sha256_by_cell"] == {key: FAKE_DIGEST} and rec["chat_template_hash"] == _TEMPLATE
    assert rec["oos_prompt_budget"]["chat_template_hash"] == _TEMPLATE and rec["oos_prompt_budget"]["cap"] == 8192
    # a sibling entry of experiment 2's shape is ignored: look up your own cell
    rec, seen = run({_DECL_KEY: {key: _DECL_VALUE, _TWIN: _DECL_VALUE,
                                 "si_constitution__psr_kl_all@all_turns": {"serve_experiment": "fit_response_serve_all_turns"}}})
    assert rec["status"] == "LEDGER_FAILED"
    # the own core disagreeing, three shapes -> MISMATCH naming both
    for bad in ({"serve_out_of_support": ["thinking", "x"]}, {"serve_out_of_support": "thinking"},
                {"serve_experiment": "fit_response_serve_all_turns"}):
        rec, seen = run({_DECL_KEY: {key: bad, _TWIN: _DECL_VALUE}})
        why = _refused(L, rec, seen, "OOS_DECLARATION_MISMATCH")
        assert f"[{key!r}]" in why and (("'x'" in why and "['thinking']" in why) or "closed shape" in why
                                        or "not a non-empty list" in why)
        assert rec[_DECL_KEY][key] == bad                        # the raw value still recorded
    # a VALID core that is not the expected literal: today the closed vocabulary
    # has one token, so the only valid core IS ["thinking"] and this branch is
    # reachable only once the vocabulary widens — exercised with a widened one,
    # so the compare (not the vocabulary check) is what refuses it
    with monkeypatch.context() as mp:
        mp.setattr(C, "OUT_OF_SUPPORT_TOKENS", (*C.OUT_OF_SUPPORT_TOKENS, "other_render"))
        rec, seen = run({_DECL_KEY: {key: {"serve_out_of_support": ["other_render"]}, _TWIN: _DECL_VALUE}})
        why = _refused(L, rec, seen, "OOS_DECLARATION_MISMATCH")
        assert f"[{key!r}]" in why and "['other_render']" in why and "['thinking']" in why
        assert "contract expectation hashed into this run's identity" in why
    # the own key absent -> UNATTESTED naming the key and the map's keys
    rec, seen = run({_DECL_KEY: {_TWIN: _DECL_VALUE}})
    why = _refused(L, rec, seen, "OOS_DECLARATION_UNATTESTED")
    assert key in why and "none__lens0" in why and "['thinking']" in why
    # the map absent altogether (today's proxy): UNATTESTED, fail closed
    rec, seen = run({}, drop=(_DECL_KEY,))
    why = _refused(L, rec, seen, "OOS_DECLARATION_UNATTESTED")
    assert _DECL_KEY in why and "before the lens landing" in why and rec[_DECL_KEY] is None
    # the twin missing, or disagreeing -> MISMATCH naming the twin
    rec, seen = run({_DECL_KEY: {key: _DECL_VALUE}})
    why = _refused(L, rec, seen, "OOS_DECLARATION_MISMATCH")
    assert "none__lens0" in why and "is not in" in why and "§2.5.10" in why
    rec, seen = run({_DECL_KEY: {key: _DECL_VALUE, _TWIN: {"serve_out_of_support": ["thinking", "x"]}}})
    why = _refused(L, rec, seen, "OOS_DECLARATION_MISMATCH")
    assert "none__lens0" in why and "'x'" in why
    # the error key non-null -> UNATTESTED (and the map is never read around it)
    rec, seen = run({_DECL_ERR_KEY: "worker reports disagree on serve declarations"})
    why = _refused(L, rec, seen, "OOS_DECLARATION_UNATTESTED")
    assert "worker reports disagree" in why and rec[_DECL_ERR_KEY].startswith("worker reports")
    # the run's exit code counts every one of them as work undone
    assert set(L.OOS_WORKER_RECS) == {"OOS_DECLARATION_UNATTESTED", "OOS_DECLARATION_MISMATCH",
                                      "OOS_SALT_POLICY_UNEXPECTED", "OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH",
                                      "OOS_TORN_STORE_UNREPAIRABLE"}
    assert not set(L.OOS_WORKER_RECS) & set(L.REC_OK_STATUSES)
    # Phase B integration: every worker REC is contracts' object, defined once, and
    # sits in the run-ending set the driver's typed refusals are held to
    assert set(L.OOS_WORKER_RECS) <= set(C.OOS_REC_NAMES)
    for name in ("REC_OOS_DECLARATION_UNATTESTED", "REC_OOS_DECLARATION_MISMATCH",
                 "REC_OOS_SALT_POLICY_UNEXPECTED", "REC_OOS_TORN_STORE_UNREPAIRABLE"):
        assert getattr(L, name) is getattr(C, name), name


def test_worker_recs_template_digest_salt_policy_and_artifact_digests(L, tmp_path, monkeypatch):
    """The rest of the compare on the psr/on standup. The served
    `chat_template_hash` != the artifact's -> `OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH`
    naming both 16-hex digests; no artifact handed to the container
    -> the same status BEFORE the standup (no Popen), as does an artifact for
    another tree; `resumed_prefill_policy_cells[<variant key>]` reading
    `always`, or the variant key absent, or the map keyed on the BARE id ->
    `OOS_SALT_POLICY_UNEXPECTED` (the lookup is by the variant key);
    `lens_salt_policy: request_unique_all` -> `OOS_SALT_POLICY_UNEXPECTED`,
    `attested` or absent -> pass; `object_artifact_sha256` != the passed digest
    -> `OOS_DECLARATION_MISMATCH`; `object_artifact_sha256_by_cell` keyed on the
    bare id, or another sha under the variant key, or a scalar disagreeing with
    its own map (torn) -> MISMATCH; a SECOND fitted cell bound in the map is a
    multi-cell standup, not a mismatch (the compare binds THIS cell by its
    harness key and holds the legacy scalar to the writer's derivation — the
    single distinct digest, null when several); absent -> refused under the
    override (below)."""
    key = "si_safety__psr_kl_all@thinking"
    n = [0]

    def world(**over):
        n[0] += 1
        drop = over.pop("_drop", ())
        cfg, _out, seen = _oos_world(L, tmp_path / str(n[0]), monkeypatch)
        rec = dict(_FakeStandupProc.record, **over)
        for k in drop:
            rec.pop(k, None)
        _FakeStandupProc.record = rec
        return cfg, seen

    # the template digest
    cfg, seen = world(chat_template_hash="b" * 16)
    why = _refused(L, _run_oos(L, cfg), seen, "OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH")
    assert "a" * 16 in why and "b" * 16 in why and "nobody is serving" in why
    cfg, seen = world(chat_template_hash=None)
    why = _refused(L, _run_oos(L, cfg), seen, "OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH")
    assert "carries no 16-hex chat_template_hash" in why
    # no artifact / another tree's artifact: refused BEFORE the standup
    cfg, seen = world()
    rec = _run_oos(L, cfg, budget=None)
    assert rec["status"] == "OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH" and seen["popen"] == [] and "standup_s" not in rec
    assert "no prompt-budget artifact was handed to this container" in rec["error"] and "--pf" in rec["error"]
    cfg, seen = world()
    rec = _run_oos(L, cfg, budget={**_budget(), "tree_sha": OTHER_SHA})
    assert rec["status"] == "OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH" and seen["popen"] == []
    assert OTHER_SHA in rec["error"] and FULL_SHA in rec["error"] and "D35" in rec["error"]
    cfg, seen = world()
    rec = _run_oos(L, cfg, budget={**_budget(), "max": 1})               # inconsistent: refused as malformed
    assert rec["status"] == "OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH" and seen["popen"] == []
    # the attested resumed-prefill policy, by the VARIANT key
    for policies in ({key: "always", _TWIN: "conditional"}, {_TWIN: "conditional"},
                     {"si_safety__psr_kl_all": "conditional", _TWIN: "conditional"}, None):
        cfg, seen = world(resumed_prefill_policy_cells=policies)
        why = _refused(L, _run_oos(L, cfg), seen, "OOS_SALT_POLICY_UNEXPECTED")
        assert "resumed_prefill_policy_cells" in why and key in why and "'conditional'" in why
    # ... and the map ABSENT under the override (a pre-attestation standup): refused too
    cfg, seen = world(_drop=("resumed_prefill_policy_cells",))
    why = _refused(L, _run_oos(L, cfg), seen, "OOS_SALT_POLICY_UNEXPECTED")
    assert "attests no resumed_prefill_policy_cells" in why and key in why
    # lens_salt_policy: `attested` passes; `request_unique_all`, a present null AND — under the
    # override — an ABSENT key refuse (review round B: the additive rule, presence apart from
    # value; this reverses the tolerance the previous test pinned)
    cfg, seen = world(lens_salt_policy="request_unique_all")
    why = _refused(L, _run_oos(L, cfg), seen, "OOS_SALT_POLICY_UNEXPECTED")
    assert "'request_unique_all'" in why and "'attested'" in why and "forced-request-unique" in why
    cfg, seen = world(lens_salt_policy="attested")
    rec = _run_oos(L, cfg)
    assert rec["status"] == "LEDGER_FAILED" and rec["lens_salt_policy"] == "attested"
    cfg, seen = world(lens_salt_policy=None)
    why = _refused(L, _run_oos(L, cfg), seen, "OOS_SALT_POLICY_UNEXPECTED")
    assert "lens_salt_policy = None" in why and "'attested'" in why
    cfg, seen = world(_drop=("lens_salt_policy",))
    why = _refused(L, _run_oos(L, cfg), seen, "OOS_SALT_POLICY_UNEXPECTED")
    assert "attests no lens_salt_policy" in why and "'attested'" in why
    # the artifact digests
    cfg, seen = world(object_artifact_sha256="cd" * 32)
    why = _refused(L, _run_oos(L, cfg), seen, "OOS_DECLARATION_MISMATCH")
    assert "cd" * 32 in why and FAKE_DIGEST in why and "object_artifact_sha256" in why
    cfg, seen = world(object_artifact_sha256=None)                      # an unmade binding under the override
    _refused(L, _run_oos(L, cfg), seen, "OOS_DECLARATION_MISMATCH")
    # keyed on the bare id (the cell unbound), another sha under the key, and a several-digest
    # map beside the record's non-null scalar (torn: the writer derives null for several)
    for bad in ({"si_safety__psr_kl_all": FAKE_DIGEST}, {key: "cd" * 32}, {key: FAKE_DIGEST, _TWIN: "cd" * 32}):
        cfg, seen = world(object_artifact_sha256_by_cell=bad)
        why = _refused(L, _run_oos(L, cfg), seen, "OOS_DECLARATION_MISMATCH")
        assert "object_artifact_sha256_by_cell" in why and key in why
    # F-V: a multi-cell standup — the map binds a second fitted cell to another digest and the
    # legacy scalar reads null (several) — passes the compare at the launcher's door too, and
    # the WHOLE map is recorded
    several = {key: FAKE_DIGEST, "si_other__psr_kl_all@thinking": "cd" * 32}
    cfg, seen = world(object_artifact_sha256=None, object_artifact_sha256_by_cell=several)
    rec = _run_oos(L, cfg)
    assert rec["status"] == "LEDGER_FAILED" and rec["object_artifact_sha256_by_cell"] == several
    # the per-cell map ABSENT under the override: refused (the landed standup writes it on
    # every standup — review round B reverses the tolerance the previous test pinned); a
    # present null refuses everywhere
    cfg, seen = world(_drop=("object_artifact_sha256_by_cell",))
    why = _refused(L, _run_oos(L, cfg), seen, "OOS_DECLARATION_MISMATCH")
    assert "carries no object_artifact_sha256_by_cell" in why and key in why and FAKE_DIGEST in why
    assert _FakeStandupProc.record.get("object_artifact_sha256_by_cell") is None
    cfg, seen = world(object_artifact_sha256_by_cell=None)
    why = _refused(L, _run_oos(L, cfg), seen, "OOS_DECLARATION_MISMATCH")
    assert "object_artifact_sha256_by_cell = None" in why and key in why
    cfg, seen = world()
    rec = _run_oos(L, cfg)
    assert rec["status"] == "LEDGER_FAILED" and rec["object_artifact_sha256_by_cell"] == {key: FAKE_DIGEST}
    # the artifact the driver will read sits in the out-root, and its argv would carry it
    out = Path(cfg["out_dir"])
    assert json.loads((out / C.OOS_PROMPT_BUDGET_FILENAME).read_text()) == _budget()
    geom, _ = L.served_geometry(cfg, 1)
    drv = L._driver_cmd("/t", cfg, {"proxy_url": "u", "routing_table": "r"}, "l", geom, controls=cfg["controls"],
                        limit=0, inspect_limit=0, samples=1, cfg_path="c", vllm_log=None,
                        oos_prompt_budget=str(out / C.OOS_PROMPT_BUDGET_FILENAME),
                        expect_object_sha256=FAKE_DIGEST)
    assert drv[-4:] == [C.FLAG_EXPECT_OBJECT_SHA256, FAKE_DIGEST,
                        C.FLAG_OOS_PROMPT_BUDGET, str(out / C.OOS_PROMPT_BUDGET_FILENAME)]
    assert C.FLAG_OOS_PROMPT_BUDGET == "--oos-prompt-budget"      # ONE constant; the driver's parser reads it
    assert not hasattr(C, "DRIVER_FLAG_OOS_PROMPT_BUDGET")          # the alias collapsed at the merge
    assert C.FLAG_ALLOW_OOS_THINKING in drv
    off = L.plan_configurations(cells=[_OOS_CELL], modes=["off"], controls=["gsm8k"], model=_OOS_MODEL)[0]
    with pytest.raises(ValueError, match="serves nothing thinking-ON out of support"):
        L._driver_cmd("/t", off, {"proxy_url": "u", "routing_table": "r"}, "l", geom, controls=off["controls"],
                      limit=0, inspect_limit=0, samples=1, cfg_path="c", vllm_log=None, oos_prompt_budget="/x")


def test_worker_recs_on_the_undeclared_standups_and_the_campaign_s(L, tmp_path, monkeypatch):
    """T4 (S4.3), the other three legs and the campaign. The none/on leg (out of
    support, undeclared): `{}` + null error -> pass; its own key `none`
    present -> MISMATCH; the map absent -> UNATTESTED (a lens half from before
    the contract; the ON leg needs the landed one); the error key non-null ->
    UNATTESTED. The psr/off leg (in support): `{}` -> pass; own key present ->
    MISMATCH; the map ABSENT -> tolerated (today's proxy); its attested policy
    naming the BARE cell `always` -> `OOS_SALT_POLICY_UNEXPECTED`, absent ->
    tolerated. A campaign configuration of another model against the minimal
    endpoint of today: no refusal at all."""
    def leg(cell, mode, sub, **over):
        drop = over.pop("_drop", ())
        cfg, _out, seen = _oos_world(L, tmp_path / sub, monkeypatch, cell=cell, mode=mode)
        rec = dict(_FakeStandupProc.record, **over)
        for k in drop:
            rec.pop(k, None)
        _FakeStandupProc.record = rec
        return cfg, seen

    # none / on — out of support, the base route
    cfg, seen = leg("none", "on", "n0")
    rec = _run_oos(L, cfg)
    assert rec["status"] == "LEDGER_FAILED" and rec[_DECL_KEY] == {} and rec["attestation_key"] == "none"
    assert rec["resumed_prefill_policy"] is None and rec["object_artifact_sha256"] is None
    # the literal is None on the base route, and the key is PRESENT (the widened derivation)
    assert C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT in rec["expected_identity"]
    assert rec["expected_identity"][C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT] is None
    assert rec["expected_identity"][C.IDENTITY_KEY_MAX_TOKENS_BY_TASK] == {C.GSM8K_STANDARD_TASK: 8192}
    assert json.loads((Path(cfg["out_dir"]) / C.OOS_PROMPT_BUDGET_FILENAME).read_text()) == _budget()
    cfg, seen = leg("none", "on", "n1", **{_DECL_KEY: {"none": _DECL_VALUE}})
    why = _refused(L, _run_oos(L, cfg), seen, "OOS_DECLARATION_MISMATCH")
    assert "for 'none'" in why and "wiring error" in why
    cfg, seen = leg("none", "on", "n2", _drop=(_DECL_KEY,))
    why = _refused(L, _run_oos(L, cfg), seen, "OOS_DECLARATION_UNATTESTED")
    assert _DECL_KEY in why and "before the lens landing" in why and "out-of-support run refuses" in why
    cfg, seen = leg("none", "on", "n3", **{_DECL_ERR_KEY: "nothing attested"})
    _refused(L, _run_oos(L, cfg), seen, "OOS_DECLARATION_UNATTESTED")
    cfg, seen = leg("none", "on", "n4", chat_template_hash="b" * 16)       # the ON leg is held to the template too
    _refused(L, _run_oos(L, cfg), seen, "OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH")
    # psr / off — in support, the bare standup
    cfg, seen = leg(_OOS_CELL, "off", "p0")
    rec = _run_oos(L, cfg, budget=None)
    assert rec["status"] == "LEDGER_FAILED" and rec["attestation_key"] == "si_safety__psr_kl_all"
    assert rec["resumed_prefill_policy"] == "conditional" and "oos_prompt_budget" not in rec
    assert not (Path(cfg["out_dir"]) / C.OOS_PROMPT_BUDGET_FILENAME).exists()
    cfg, seen = leg(_OOS_CELL, "off", "p1", **{_DECL_KEY: {"si_safety__psr_kl_all": _DECL_VALUE}})
    why = _refused(L, _run_oos(L, cfg, budget=None), seen, "OOS_DECLARATION_MISMATCH")
    assert "out_of_support=False" in why and "wiring error" in why
    cfg, seen = leg(_OOS_CELL, "off", "p2", _drop=(_DECL_KEY, "lens_salt_policy",
                                                    "object_artifact_sha256_by_cell"))
    rec = _run_oos(L, cfg, budget=None)
    assert rec["status"] == "LEDGER_FAILED" and rec[_DECL_KEY] is None       # today's proxy: tolerated in support
    cfg, seen = leg(_OOS_CELL, "off", "p3", **{_DECL_ERR_KEY: "disagree"})
    _refused(L, _run_oos(L, cfg, budget=None), seen, "OOS_DECLARATION_UNATTESTED")
    cfg, seen = leg(_OOS_CELL, "off", "p4", resumed_prefill_policy_cells={"si_safety__psr_kl_all": "always"})
    why = _refused(L, _run_oos(L, cfg, budget=None), seen, "OOS_SALT_POLICY_UNEXPECTED")
    assert "resumed_prefill_policy_cells['si_safety__psr_kl_all']" in why and "'always'" in why
    # a PRESENT null per-cell map refuses in support too (review round B: presence apart from
    # value — this reverses the tolerance the previous test pinned); the key ABSENT is tolerated
    cfg, seen = leg(_OOS_CELL, "off", "p5", resumed_prefill_policy_cells=None)
    why = _refused(L, _run_oos(L, cfg, budget=None), seen, "OOS_SALT_POLICY_UNEXPECTED")
    assert "resumed_prefill_policy_cells = None" in why and "'conditional'" in why
    cfg, seen = leg(_OOS_CELL, "off", "p5b", _drop=("resumed_prefill_policy_cells",))
    rec = _run_oos(L, cfg, budget=None)
    assert rec["status"] == "LEDGER_FAILED" and rec["resumed_prefill_policy"] is None
    cfg, seen = leg(_OOS_CELL, "off", "p6", chat_template_hash="b" * 16)   # OFF is not held to the artifact
    assert _run_oos(L, cfg, budget=None)["status"] == "LEDGER_FAILED"
    # the campaign: another model's psr/off against the minimal endpoint of today — untouched
    cfg2, _out2 = _fake_world(L, tmp_path / "campaign", monkeypatch, record={})
    rec = _run_fake(L, cfg2)
    assert rec["status"] == "LEDGER_FAILED" and rec[_DECL_KEY] is None and rec["lens_salt_policy"] is None
    exp = _expected(L, controls=cfg2["controls"])
    ep = {}
    assert L.oos_attestation_refusal(ep, cfg2, expected=exp, object_sha256=FAKE_DIGEST, budget=None, oos_on=False) is None
    # ...but a PRESENT, disagreeing additive key refuses there too
    for bad in ({_DECL_ERR_KEY: "x"}, {"lens_salt_policy": "request_unique_all"},
                {"object_artifact_sha256": "cd" * 32}, {_DECL_KEY: {"si_safety__psr_kl_all": _DECL_VALUE}}):
        assert L.oos_attestation_refusal(bad, cfg2, expected=exp, object_sha256=FAKE_DIGEST,
                                         budget=None, oos_on=False) is not None
    # the campaign's gemma psr cell is NOT the authorised leg: its policy is never held
    assert L.authorised_psr_leg(cfg2) is False and L.authorised_psr_cell(cfg2["cell"]) is True
    assert L.oos_attestation_refusal({"resumed_prefill_policy_cells": {"si_safety__psr_kl_all": "always"}}, cfg2,
                                     expected=exp, object_sha256=FAKE_DIGEST, budget=None, oos_on=False) is None


def test_the_dry_run_prints_the_declaration_the_three_spellings_the_split_mode_and_the_five_triples(L, capsys):
    # native-serving-default: the launcher's default is the native engine now; this test pins the LENS
    # launch (a PSR cell / the historical composition), so it states the path explicitly
    """S4.5 / C2: per ON configuration the dry run prints the expected
    declaration literal `["thinking"]` (null on none/on), the per-request key
    `petri_thinking_on: true` (none: no declaration, the base route refuses
    it), the standup's three spellings — psr/on `--cell
    safety_refusal__psr_kl_all@thinking` ALONE with "none implicit,
    none__lens0 derived", expect key idem, attestation keys
    `si_safety__psr_kl_all@thinking` + `none__lens0`; psr/off bare; none: no
    declaration — `think_split_mode: implicit_open`, and the expected
    (cache_salt_regime, resumed_prefill_policy, render_regime) triple per leg,
    §2.5.11's five rows read BY NAME; the standup line shows the one `--cell`;
    the driver line of an ON leg carries the artifact flag. An in-support dry
    run prints none of it."""
    _oos_dry_run(L)
    out = capsys.readouterr().out
    assert out.count('expected declaration (hashed tinycap.serve_out_of_support: the CONTRACT literal, contract §2.5.1 Q6 — never the attestation): ["thinking"]') == 1
    assert out.count("expected declaration (hashed tinycap.serve_out_of_support: the CONTRACT literal, contract §2.5.1 Q6 — never the attestation): null") == 1
    assert out.count("per-request declaration (contract §2.5.4: every lens-route request that sends enable_thinking: true): petri_thinking_on: true") == 1
    assert out.count("per-request declaration (contract §2.5.4: every lens-route request that sends enable_thinking: true): none — the base route accepts enable_thinking: true undeclared and refuses petri_thinking_on (400 thinking_on_not_lens, §2.5.5)") == 1
    assert out.count("standup spellings (contract §2.5.10 Q8): --cell safety_refusal__psr_kl_all@thinking ALONE ('none' implicit — bring_up prepends it; 'none__lens0' derived, declared with its donor — §2.5.11 Q7(ii)); expect key safety_refusal__psr_kl_all@thinking; attestation keys si_safety__psr_kl_all@thinking + none__lens0 (each [\"thinking\"]); vmodel petri__si_safety__psr_kl_all@thinking") == 1
    assert out.count("standup spellings: bare --cell safety_refusal__psr_kl_all (undeclared); expect key: the bare sha; attestation key si_safety__psr_kl_all must be ABSENT from lens_serve_declarations; vmodel petri__si_safety__psr_kl_all") == 1
    assert out.count("standup spellings: --cell none alone, no declaration (the base route petri__base; contract §2.5.1)") == 2
    assert out.count("think_split_mode: implicit_open (render fact prompt_opens_think_on=True") == 2
    assert "explicit_tags" not in out and "record_only" not in out
    # the five rows, by NAME from the expectation table — and the contract's literals once
    def row(lens, arm):
        t = C.expected_stamp_regimes(lens=lens, arm=arm)
        return " / ".join(f"{k}={v}" for k, v in t._asdict().items())
    head = "expected stamp triple (salt_regime, resumed_policy, render_regime; contract §2.5.11) per leg: "
    assert out.count(f"{head}leg petri__si_safety__psr_kl_all@thinking -> {row(True, 'on')}; probe twin petri__none__lens0 -> {row(True, 'on')}") == 1
    assert out.count(f"{head}leg petri__si_safety__psr_kl_all -> {row(True, 'off')}; probe twin petri__none__lens0 -> {row(True, 'off')}") == 1
    assert out.count(f"{head}leg petri__base -> {row(False, 'on')}") == 1
    assert out.count(f"{head}leg petri__base -> {row(False, 'off')}") == 1
    assert row(True, "on") == "salt_regime=request-unique / resumed_policy=declared-request-unique / render_regime=thinking_on"
    assert row(True, "off") == "salt_regime=request-unique / resumed_policy=conditional / render_regime=thinking_off"
    assert row(False, "off") == "salt_regime=cell-stable / resumed_policy=not-consulted / render_regime=thinking_off"
    assert "forced-request-unique" not in out and "unspecified" not in out
    # the sampling deviation sentence, per ON configuration
    assert out.count("sampling: " + C.AUTHORISED_OOS_THINKING.what) == 2
    # the standup lines: ONE --cell each; the variant spelling on psr/on only; none never beside a fitted cell
    standups = [ln for ln in out.splitlines() if ln.startswith("   standup: ")]
    assert len(standups) == 4 and all(ln.count("--cell") == 1 for ln in standups)
    assert sum("--cell safety_refusal__psr_kl_all@thinking " in ln for ln in standups) == 1
    assert sum("--cell safety_refusal__psr_kl_all " in ln for ln in standups) == 1
    assert sum("--cell none " in ln for ln in standups) == 2
    (on_line,) = [ln for ln in standups if "@thinking" in ln]
    assert "--expect-object-sha256 safety_refusal__psr_kl_all@thinking=<preflight artifact_sha256>" in on_line
    assert "none" not in on_line.split("capctrl_serving_up.py", 1)[1]
    # the driver lines: the artifact flag on the two ON legs only
    drivers = [ln for ln in out.splitlines() if ln.startswith("   driver: ")]
    assert sum(f"{C.FLAG_OOS_PROMPT_BUDGET} <out>/{C.OOS_PROMPT_BUDGET_FILENAME}" in ln for ln in drivers) == 2
    assert sum("--thinking on" in ln for ln in drivers) == 2
    # the hashed expectation on the ON legs names the literal and the cap
    idents = [json.loads(ln.split("resume identity: ", 1)[1]) for ln in out.splitlines() if "resume identity: " in ln]
    on_idents = [i for i in idents if i["mode"] == "on"]
    assert len(on_idents) == 2 and all(i[C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING] is True for i in on_idents)
    # the two further keys, in the JSON: the cap on both ON legs, the literal on psr/on only
    assert all(i[C.IDENTITY_KEY_MAX_TOKENS_BY_TASK] == {C.GSM8K_STANDARD_TASK: 8192} for i in on_idents)
    assert sorted(json.dumps(i[C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT]) for i in on_idents) == ['["thinking"]', "null"]
    # the psr/ON driver line carries the artifact digest the standup is held to (plan S3.2)
    assert sum(f"{C.FLAG_EXPECT_OBJECT_SHA256} <preflight artifact_sha256>" in ln for ln in drivers) == 1
    # an in-support dry run prints none of it
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, dry_run=True)
    out = capsys.readouterr().out
    assert "standup spellings" not in out and "expected stamp triple" not in out and "expected declaration" not in out


def test_an_oos_on_launch_needs_pf_and_hands_every_container_the_preflight_s_artifact(L, monkeypatch, capsys):
    # native-serving-default: the launcher's default is the native engine now; this test pins the LENS
    # launch (a PSR cell / the historical composition), so it states the path explicitly
    """S4.3 / main: an out-of-support thinking-ON configuration — the base
    cell's ON leg included, which no object-digest rule covers — launches only
    behind --pf (refused before any Modal call, naming the configuration);
    with --pf the VALIDATED artifact object off the preflight report rides into
    `run_configuration.map`'s kwargs for every container, and a report without
    one refuses the launch."""
    kw = {"app_name": L.APP_NAME, "sha": "abc1234", "pass1": True, "model": _OOS_MODEL, "cells": "none",
          "modes": "on", "controls": "gsm8k", "allow_out_of_support_thinking": True,
          # native-serving-default: the authorised out-of-support experiment is a LENS launch (a PSR cell)
          "serving_path": "lens"}
    with pytest.raises(SystemExit) as e:
        L.main(**kw)
    assert "launch only behind --pf" in str(e.value)
    assert C.container_name("none", "on", model_slug=_OOS_MODEL) in str(e.value)
    assert "chat_template_hash" in str(e.value) and "D16 P2-a" in str(e.value)
    capsys.readouterr()
    # with --pf: the report's artifact reaches every container
    budget = _budget(prompt_max=2100, digest="d" * 16)
    launched: dict = {}

    def _remote(sha, plan_args, expect):
        rep = _preflight_report(L, L.plan_for(**plan_args))
        rep["oos_prompt_budget"] = {"path": "/vol/x/oos_prompt_budget.json", **budget}
        return rep

    def _spawned(cfg, kwargs):
        launched["kwargs"] = kwargs
    monkeypatch.setattr(L, "preflight", SimpleNamespace(remote=_remote))
    monkeypatch.setattr(L, "run_configuration", _spawn_only(_done, _spawned))
    L.main(**{**kw, "pf": True, "cells": f"none,{_OOS_CELL}", "modes": "off,on"})
    assert launched["kwargs"]["oos_prompt_budget"] == budget          # the validated object, path stripped
    assert "path" not in launched["kwargs"]["oos_prompt_budget"]
    capsys.readouterr()
    # an in-support launch hands None
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, pf=True)
    assert launched["kwargs"]["oos_prompt_budget"] is None
    capsys.readouterr()
    # a report without the artifact refuses the launch
    monkeypatch.setattr(L, "preflight", SimpleNamespace(
        remote=lambda sha, plan_args, expect: _preflight_report(L, L.plan_for(**plan_args))))
    with pytest.raises(SystemExit, match="carries no oos_prompt_budget"):
        L.main(**{**kw, "pf": True})
    # a malformed one too
    monkeypatch.setattr(L, "preflight", SimpleNamespace(
        remote=lambda sha, plan_args, expect: {**_preflight_report(L, L.plan_for(**plan_args)),
                                               "oos_prompt_budget": {"path": "/p", **budget, "max": 1}}))
    with pytest.raises(SystemExit, match="max"):
        L.main(**{**kw, "pf": True})
    # the preflight's argparse probe asks the driver for the artifact flag on such a launch
    src = LAUNCHER.read_text()
    body = src[src.index("def preflight("):src.index("    def datasets():")]
    assert "probe_flags = ((C.FLAG_ALLOW_OOS_THINKING,) if oos else ()) + (" in body
    # offcap: the artifact flag is probed for every BUDGETED plan (oos_on + offcap) and the
    # digest for every LANDED lens configuration (out-of-support or the comparator's lens leg)
    assert "(C.FLAG_OOS_PROMPT_BUDGET,) if budgeted else ())" in body
    assert "(C.FLAG_EXPECT_OBJECT_SHA256,) if any(landed_lens_configuration(c) for c in cfgs) else ())" in body
    assert "driver_probe_argv(tree, extra_flags=probe_flags)" in body
    # a VALUED flag is probed with its dummy value (argparse would otherwise stop at
    # "expected one argument" before the unrecognized-arguments line the probe reads)
    probed = L.driver_probe_argv("/t", extra_flags=(C.FLAG_ALLOW_OOS_THINKING, C.FLAG_OOS_PROMPT_BUDGET,
                                                     C.FLAG_EXPECT_OBJECT_SHA256))
    i = probed.index(C.FLAG_OOS_PROMPT_BUDGET)
    assert probed[i + 1] == L.DRIVER_PROBE_VALUES[C.FLAG_OOS_PROMPT_BUDGET] and probed[i + 1].endswith(C.OOS_PROMPT_BUDGET_FILENAME)
    j = probed.index(C.FLAG_EXPECT_OBJECT_SHA256)
    assert probed[j + 1] == "0" * 64 and probed[probed.index(C.FLAG_ALLOW_OOS_THINKING) + 1] == C.FLAG_OOS_PROMPT_BUDGET
    assert probed[-1] == L.DRIVER_PROBE_SENTINEL


def test_the_s4_contracts_additions_are_the_standup_s_and_the_contract_s_spellings():
    """The launcher lane's small `contracts.py` block, pinned (the contracts test
    file belongs to the driver lane, so the pin lives here): the endpoint keys
    the worker reads are the standup's own spellings (`capctrl_serving_up`
    writes each), the driver's `STANDUP_TEMPLATE_HASH_KEY` and lens_worker's
    policy-map key are the same strings, the two additive keys and
    `lens_salt_policy`'s value are the contract's (§2.5.9 / §2.5.7), the
    expected attested policy is `conditional`, the driver's
    artifact flag is the launcher's dry-run flag, and the twin's vmodel is the
    registry's rule."""
    standup = (REPO / "scripts/capctrl_serving_up.py").read_text()
    for key in (C.ENDPOINT_KEY_CHAT_TEMPLATE_HASH, C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS,
                C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256):
        assert f'"{key}":' in standup, key
    assert C.ENDPOINT_KEY_CHAT_TEMPLATE_HASH == "chat_template_hash"
    driver = (REPO / "scripts/served_capability_run.py").read_text()
    assert f'STANDUP_TEMPLATE_HASH_KEY = "{C.ENDPOINT_KEY_CHAT_TEMPLATE_HASH}"' in driver
    from sisteer.capeval.serving import lens_worker as lw
    assert C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS == lw.RESUMED_PREFILL_POLICY_CELLS_KEY
    assert C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256 == "object_artifact_sha256"
    assert C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL == "object_artifact_sha256_by_cell"
    assert C.ENDPOINT_KEY_LENS_SALT_POLICY == "lens_salt_policy" and C.LENS_SALT_POLICY_ATTESTED == "attested"
    assert C.OOS_EXPECTED_RESUMED_PREFILL_POLICY == "conditional" == C.RESUMED_PREFILL_CONDITIONAL
    assert C.OOS_EXPECTED_RESUMED_PREFILL_POLICY == C.expected_stamp_regimes(lens=True, arm="off").resumed_policy
    assert C.FLAG_OOS_PROMPT_BUDGET == "--oos-prompt-budget" and not hasattr(C, "DRIVER_FLAG_OOS_PROMPT_BUDGET")
    from sisteer.capeval.petri import defense_cells as dc
    assert C.twin_vmodel() == "petri__none__lens0" == dc.VMODEL_PREFIX + dc.NOOP_LENS_CELL_ID


def test_an_out_of_support_store_needs_the_raw_declarations_map_in_its_route_block(L, tmp_path):
    """Review round B, P2-1: on EVERY out-of-support store the raw
    ``route.lens_serve_declarations`` must EXIST and be a mapping — `{}` is the
    none/ON leg's attested "nothing declared" and is trusted; the key ABSENT or
    NULL is missing evidence (a torn or hand-edited manifest) and is refused by
    name, where before it read as the `None` the base route expects and a torn
    store read DONE without the live checks. The declared psr/ON store needs
    its own key (and the twin) as before; an in-support store carries no map
    and is never asked for one."""
    none_on = _oos_expected(L, cell="none")
    assert none_on.out_of_support_thinking and none_on.serve_out_of_support is None
    good = _store(tmp_path / "a", none_on, tasks=["gsm8k"], route_declarations={})
    assert L.untrusted_store_reason(good, expected=none_on) is None
    null = _store(tmp_path / "b", none_on, tasks=["gsm8k"], route_declarations=None)
    why = L.untrusted_store_reason(null, expected=none_on)
    assert why and f"manifest.route.{C.ROUTE_KEY_SERVE_DECLARATIONS} is None" in why and "missing evidence" in why
    absent = _store(tmp_path / "c", none_on, tasks=["gsm8k"],
                    mutate=lambda m: m[C.MANIFEST_KEY_ROUTE].pop(C.ROUTE_KEY_SERVE_DECLARATIONS))
    why = L.untrusted_store_reason(absent, expected=none_on)
    assert why and f"manifest.route.{C.ROUTE_KEY_SERVE_DECLARATIONS} is absent" in why and "missing evidence" in why
    bad = _store(tmp_path / "d", none_on, tasks=["gsm8k"], route_declarations="nope")
    assert "missing evidence" in (L.untrusted_store_reason(bad, expected=none_on) or "")
    # the declared store: `{}` is not enough there (its own key is expected), the exact map is
    psr_on = _oos_expected(L)
    assert L.untrusted_store_reason(_store(tmp_path / "e", psr_on, tasks=["gsm8k"]), expected=psr_on) is None
    why = L.untrusted_store_reason(_store(tmp_path / "f", psr_on, tasks=["gsm8k"], route_declarations={}),
                                   expected=psr_on)
    assert why and "does not mirror" in why
    # an in-support store: no map, no question asked
    off = _oos_expected(L, mode="off")
    assert not off.out_of_support_thinking
    plain = _store(tmp_path / "g", off, tasks=["gsm8k"],
                   mutate=lambda m: m[C.MANIFEST_KEY_ROUTE].pop(C.ROUTE_KEY_SERVE_DECLARATIONS, None))
    assert L.untrusted_store_reason(plain, expected=off) is None


def test_the_retained_log_probe_runs_the_reader_in_the_client_interpreter(L, tmp_path):
    """Review round B, P2-3, the plumbing: `retained_log_reasons` runs
    `inspect_served.retained_logs_report` in the given interpreter against the
    given tree (stdin JSON in, stdout JSON out) and returns its `{log_dir:
    reason | None}` report — a dangling symlink, a directory and a truncated
    file under a dir are named as what disqualifies them, an empty dir as "no
    retained .eval log"; an interpreter that cannot run the reader, or answers
    off-shape, raises `RetainedLogProbeError` (never a verdict)."""
    pytest.importorskip("inspect_ai", reason="the reader is inspect_ai's; the client venv has it")
    d1, d2 = tmp_path / "s0", tmp_path / "s1"
    d1.mkdir()
    d2.mkdir()
    (d1 / "a.eval").symlink_to(d1 / "gone.eval")
    (d1 / "b.eval").mkdir()
    (d1 / "c.eval").write_bytes(b"\x00")
    checks = [{"log_dir": str(d1), "row_task": "inspect/math", "expected_n": 3},
              {"log_dir": str(d2), "row_task": "inspect/math", "expected_n": 3}]
    report = L.retained_log_reasons(str(REPO), checks, python=sys.executable)
    assert set(report) == {str(d1), str(d2)}
    assert "a.eval: a dangling symlink" in report[str(d1)] and "b.eval: not a regular file" in report[str(d1)]
    assert "c.eval: unreadable" in report[str(d1)] and "torn or truncated" in report[str(d1)]
    assert report[str(d2)].startswith(f"no retained .eval log under {d2}")
    assert "the item selection is the driver's to compare after the standup" in report[str(d2)]
    with pytest.raises(L.RetainedLogProbeError, match="could not run"):
        L.retained_log_reasons(str(REPO), checks, python=str(tmp_path / "no-such-python"))
    # a reader that CANNOT RUN: a staged tree whose own inspect_served raises at import. The probe puts <tree>/src FIRST on the
    # client interpreter's path, so this is what it imports on ANY machine — hermetic, unlike an absent tree, where the outcome
    # depends on whether a `sisteer` is importable from site-packages (a grading host venv's editable install of the main checkout
    # makes the probe import THAT reader and refuse it as foreign; a container has none and the import fails) (RETAINEDPROBEPIN)
    broken = tmp_path / "broken-tree" / "src" / "sisteer" / "capeval" / "tiers"
    broken.mkdir(parents=True)
    for d in (broken.parent.parent, broken.parent, broken):
        (d / "__init__.py").write_text("")
    (broken / "inspect_served.py").write_text("raise ImportError('a reader that cannot run (test fixture)')\n")
    with pytest.raises(L.RetainedLogProbeError, match="failed in the client interpreter"):
        L.retained_log_reasons(str(tmp_path / "broken-tree"), checks, python=sys.executable)
    # an ABSENT tree: either no reader is importable (fails in the client interpreter) or the only importable one resolves
    # outside the staged tree and the probe REFUSES it as foreign — a RetainedLogProbeError either way, never a verdict; the
    # refusal, when it is the outcome, names the staged tree it attested against
    with pytest.raises(L.RetainedLogProbeError, match="failed in the client interpreter|refused to answer") as ei:
        L.retained_log_reasons(str(tmp_path / "not-a-tree"), checks, python=sys.executable)
    assert "refused" not in str(ei.value) or str(tmp_path / "not-a-tree") in str(ei.value)


def test_the_retained_log_probe_imports_the_staged_tree_s_reader_and_nothing_else(L, tmp_path, monkeypatch):
    """The probe runs under the merge subprocesses' own
    protections — safe-path (`-P` / `PYTHONSAFEPATH`), a controlled empty cwd
    — and ATTESTS that the reader it imported resolves beneath the staged tree.
    A `sisteer` package shadowing from the launcher's cwd (the mounted checkout
    in a container) must NOT change the verdict; a tree whose reader resolves
    outside it refuses (`RetainedLogProbeError` naming the resolved path), never
    answers."""
    pytest.importorskip("inspect_ai", reason="the reader is inspect_ai's; the client venv has it")
    d = tmp_path / "s0"
    d.mkdir()
    (d / "a.eval").symlink_to(d / "gone.eval")                 # NOT a retained log
    checks = [{"log_dir": str(d), "row_task": "inspect/math", "expected_n": 3}]
    # a cwd-shadowing package whose reader would call everything repairable
    shadow = tmp_path / "shadow"
    pkg = shadow / "sisteer" / "capeval" / "tiers"
    pkg.mkdir(parents=True)
    for init in (shadow / "sisteer", shadow / "sisteer" / "capeval", pkg):
        (init / "__init__.py").write_text("")
    (pkg / "inspect_served.py").write_text(
        "def retained_logs_report(spec):\n    return {c['log_dir']: None for c in spec['checks']}\n")
    monkeypatch.chdir(shadow)
    report = L.retained_log_reasons(str(REPO), checks, python=sys.executable)
    assert report[str(d)] is not None and "a.eval: a dangling symlink" in report[str(d)]
    # a staged tree whose `sisteer` resolves OUTSIDE it (a symlink to another checkout): refused
    tree = tmp_path / "tree"
    (tree / "src").mkdir(parents=True)
    (tree / "scripts").mkdir()
    (tree / "src" / "sisteer").symlink_to(REPO / "src" / "sisteer", target_is_directory=True)
    with pytest.raises(L.RetainedLogProbeError, match="resolves outside the staged tree"):
        L.retained_log_reasons(str(tree), checks, python=sys.executable)


def test_every_standup_argv_carries_the_contracts_upstream_timeout_before_the_digest_and_extras(L, tmp_path):
    """EVERY tinycap standup argv carries ``--upstream-timeout-s 7200``
    (``contracts.FLAG_UPSTREAM_TIMEOUT_S`` / ``PROXY_UPSTREAM_TIMEOUT_S`` — the
    driver's, the ledger's and the Inspect provider's number), in the FIXED
    part: right after ``--ready-timeout-s``, before the digest flag and the
    configuration's extras (which the digest/extras pins read from the tail).
    The fixed part then carries ``--enable-log-requests``
    (``contracts.FLAG_ENABLE_LOG_REQUESTS``, the engine-resolved sampler
    readback) on EVERY standup, still before the digest and the extras.
    The standup's parser reads both back — the timeout as the float the
    contract names (its own default, no flag, is None: the proxy's default,
    untouched), the readback flag as True (default False)."""
    up = _load_standup()
    cfgs = L.plan_configurations()
    assert cfgs
    for cfg in cfgs:
        geom, _ = L.served_geometry(cfg, 1)
        sha = "a" * 64 if cfg["steered"] else ""
        argv = L._standup_cmd("/tree", "/tree/cfg.yaml", cfg["cell"], "d", tmp_path, tmp_path / "STOP",
                              geom, 64, cfg["standup_extra"], object_sha256=sha,
                              declared=L.standup_declared(cfg))
        assert argv.count(C.FLAG_UPSTREAM_TIMEOUT_S) == 1, cfg["name"]
        i = argv.index(C.FLAG_UPSTREAM_TIMEOUT_S)
        assert argv[i + 1] == "7200" == f"{C.PROXY_UPSTREAM_TIMEOUT_S:g}"
        assert argv[i - 2:i] == ["--ready-timeout-s", "2700"]                 # the fixed part ...
        assert argv[i + 2] == C.FLAG_ENABLE_LOG_REQUESTS == "--enable-log-requests"   # ... the readback flag (fixed too) ...
        assert argv.count(C.FLAG_ENABLE_LOG_REQUESTS) == 1 and C.FLAG_ENABLE_LOG_REQUESTS not in cfg["standup_extra"]
        # native-serving-default (§1.7): the serving PATH is stated on EVERY standup, in the fixed part
        # right after the readback flag (lens here: the library default; no bank on a lens standup)
        assert argv[i + 3:i + 5] == [L.SERVING_PATH_FLAG, "lens"] and L.ADAPTER_BANK_FLAG not in argv
        tail = argv[i + 5:]
        expected_tail = ([L.OBJECT_DIGEST_FLAG, tail[1]] if sha else []) + list(cfg["standup_extra"])
        assert tail == expected_tail, cfg["name"]                             # ... then digest, then extras
        ns = up.build_parser().parse_args(argv[2:])
        assert ns.upstream_timeout_s == 7200.0 == C.PROXY_UPSTREAM_TIMEOUT_S
        assert ns.enable_log_requests is True
    base = ["--repo-dir", "r", "--config", "c", "--cell", "none", "--defenses-dir", "d",
            "--out", "o", "--stop-file", "s"]
    assert up.build_parser().parse_args(base).upstream_timeout_s is None
    assert up.build_parser().parse_args(base).enable_log_requests is False


# --------------------------------------------------------------------------- #
# offcap: the thinking-OFF comparator at the thinking-ON cap                   #
# (`--allow-off-cap-override`) — plan, expectation,                           #
# store trust, the launcher door, the preflight, the dry run, the worker        #
# --------------------------------------------------------------------------- #

_OFFCAP_CAP = C.OFF_CAP_OVERRIDE_MAX_TOKENS


def _offcap_plan_kwargs() -> dict:
    """`plan_for` arguments that ARE the authorised comparator — one cell, its one
    mode, so a mutation below moves exactly one field."""
    exp = C.AUTHORISED_OFF_CAP_COMPARATOR
    return {"cells": [exp.cells[-1]], "modes": [exp.modes[0]], "controls": list(exp.controls),
            "model": exp.model, "sampling_mode": exp.sampling_mode, "samples": exp.samples,
            "allow_off_cap_override": True}


def _offcap_plan(L):
    """The comparator exactly as run_offcap_qwen4b.sh plans it: {none, psr} x {off}."""
    return L.plan_for(cells=["none", _OOS_CELL], modes=["off"], controls=["gsm8k"],
                      model=_OOS_MODEL, allow_off_cap_override=True)


def _offcap_expected(L, *, cell=_OOS_CELL, tree=None, tree_sha=FULL_SHA, **kw):
    """The REAL comparator expectation for one leg: planned under the cap
    override on the authorised model, so the cap key comes from the launcher's
    own derivation."""
    cfg = L.plan_configurations(cells=[cell], modes=["off"], controls=["gsm8k"], model=_OOS_MODEL,
                                allow_off_cap_override=True)[0]
    geom, _ = L.served_geometry(cfg, 1)
    return L.expected_identity(cfg, geom, pass_no=1, controls=cfg["controls"], samples=1,
                               tree=tree, tree_sha=tree_sha, **kw)


def _budget_off(**kw):
    """The comparator's OFF artifact at the OVERRIDE cap, as the preflight writes it
    (`cap` / `arm` overridable to spell a wrong one)."""
    kw.setdefault("arm", C.THINKING_OFF)
    kw.setdefault("cap", _OFFCAP_CAP)
    return _budget(**kw)


def _offcap_summary(task, *, n):
    """A comparator task's summary: the OFF render at cap 8192, `record_only`, no
    think block, every item finished."""
    return {**_oos_summary(task, n=n), "thinking": C.THINKING_OFF, "think_split_mode": "record_only",
            "n_think_block_present": 0}


def _offcap_outcomes(task, summary):
    doc = _oos_outcomes(task, summary)
    for it in doc["items"]:
        it["think_block_present"] = False
    return doc


def _offcap_world(L, tmp_path, monkeypatch, *, cell=_OOS_CELL, record=None, drop=()):
    """One comparator leg (:func:`_watched_leg_world`) whose fake standup attests
    the LANDED undeclared record (`_oos_endpoint` of an OFF configuration: `{}` +
    null error, the bare attestation key `conditional`, the digest scalar +
    per-cell, `lens_salt_policy attested`)."""
    (cfg,) = L.plan_for(cells=[cell], modes=["off"], controls=["gsm8k"], model=_OOS_MODEL,
                        allow_off_cap_override=True)
    return _watched_leg_world(L, tmp_path, monkeypatch, cfg, record=record, drop=drop)


def _run_offcap(L, cfg, *, budget=_DEFAULT, **kw):
    """:func:`_run_oos` on a comparator leg: the OFF artifact at the override cap by default."""
    return _run_oos(L, cfg, budget=(_budget_off() if budget is _DEFAULT else budget), **kw)


def test_the_comparator_configuration_carries_the_marker_the_cap_key_and_no_oos_marker(L):
    """offcap §5.1.3–5.1.4: the flag plans {none, psr} x {off} and marks each
    configuration with the planner marker `off_cap_override` (flag, model, cell,
    family, mode, the caps, the note) — NEVER the OOS marker, no refusal lifted,
    the standup extras unchanged; the pure helpers read the marker."""
    cfgs = _offcap_plan(L)
    assert {(c["cell"], c["mode"]) for c in cfgs} == {("none", "off"), (_OOS_CELL, "off")}
    for cfg in cfgs:
        m = cfg["off_cap_override"]
        assert m["flag"] == C.FLAG_ALLOW_OFF_CAP_OVERRIDE == "--allow-off-cap-override"
        assert (m["model"], m["cell"], m["mode"], m["family"]) == (_OOS_MODEL, cfg["cell"], "off", "qwen3_5")
        assert m["caps"] == {C.GSM8K_STANDARD_TASK: _OFFCAP_CAP} and m["note"] == C.OFF_CAP_OVERRIDE_NOTE
        assert C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING not in cfg and "thinking_override" not in cfg
        assert L.serves_offcap(cfg) and not L.serves_oos_on(cfg, cfg["controls"])
        assert L.offcap_caps(cfg) == L.override_caps(cfg) == {C.GSM8K_STANDARD_TASK: _OFFCAP_CAP}
        assert L.needs_prompt_budget(cfg, cfg["controls"]) is True and L.landed_configuration(cfg) is True
        assert L.landed_lens_configuration(cfg) is (cfg["cell"] == _OOS_CELL)
        assert not L.oos_lens_configuration(cfg) and not L.standup_declared(cfg)
        plain = L.plan_configurations(cells=[cfg["cell"]], modes=["off"], controls=["gsm8k"], model=_OOS_MODEL)[0]
        assert {k: v for k, v in cfg.items() if k != "off_cap_override"} == plain     # the marker is the ONLY difference
    assert [c["name"] for c in L.offcap_configurations(cfgs)] == [c["name"] for c in cfgs]
    assert L.oos_on_configurations(cfgs) == []
    # the OOS 2x2's plan carries no comparator marker, and its helpers read it as before
    for cfg in _oos_plan(L):
        assert "off_cap_override" not in cfg and not L.serves_offcap(cfg)
        assert L.override_caps(cfg) == (L.oos_on_caps(cfg) if cfg["mode"] == "on" else {})
        assert L.landed_lens_configuration(cfg) is L.oos_lens_configuration(cfg)
        assert L.needs_prompt_budget(cfg, cfg["controls"]) is L.serves_oos_on(cfg, cfg["controls"])
        assert L.landed_configuration(cfg) is (cfg["mode"] == "on")


def test_the_launcher_plans_the_comparator_and_nothing_else(L):
    """offcap §5.1.3: the comparator tuple plans (both cells, the OFF mode) and is
    the ONLY selection the flag lets through."""
    exp = C.AUTHORISED_OFF_CAP_COMPARATOR
    cfgs = L.plan_for(cells=list(exp.cells), modes=list(exp.modes), controls=list(exp.controls),
                      model=exp.model, allow_off_cap_override=True)
    assert len(cfgs) == len(exp.cells) * len(exp.modes) == 2
    assert all(c["mode"] == "off" and "off_cap_override" in c for c in cfgs)


@pytest.mark.parametrize("field,value", [
    ("model", "qwen35_9b"),
    ("cells", ["si_constitution__psr_kl_all"]),
    ("controls", ["gsm8k", "humaneval_plus"]),
    ("modes", ["on"]),                        # the comparator has no ON arm
    ("modes", ["off", "on"]),                 # the 2x2's procedure is the OTHER flag's
    ("modes", [C.THINKING_PER_TASK]),
    ("sampling_mode", C.SAMPLING_GREEDY),
    ("samples", 2),
])
def test_the_launcher_refuses_any_cli_selection_that_is_not_the_comparator(L, field, value):
    """offcap §5.1.3 (the OOS twins): every field is compared against the ONE
    declared comparator tuple, at plan time, before anything is marked; the
    refusal names the field, the flag and the configuration."""
    kwargs = {**_offcap_plan_kwargs(), field: value}
    with pytest.raises(C.OffCapOverrideUnauthorised) as e:
        L.plan_for(**kwargs)
    msg = str(e.value)
    assert f"{field.rstrip('s') if field in ('cells', 'modes') else field} differs" in msg
    assert "launch configuration" in msg and C.FLAG_ALLOW_OFF_CAP_OVERRIDE in msg
    with pytest.raises(C.OffCapOverrideUnauthorised):
        L.plan_configurations(kwargs["cells"], kwargs["modes"], kwargs["controls"], model=kwargs["model"],
                              allow_off_cap_override=True, sampling_mode=kwargs["sampling_mode"],
                              samples=kwargs["samples"])
    # the smoke plan is not the comparator either
    with pytest.raises(C.OffCapOverrideUnauthorised):
        L.smoke_plan(["humaneval_plus"], model=_OOS_MODEL, allow_off_cap_override=True)


def test_both_override_flags_together_are_refused_at_plan_time_in_main_and_in_the_preflight(L, monkeypatch):
    # native-serving-default: the launcher's default is the native engine now; this test pins the LENS
    # launch (a PSR cell / the historical composition), so it states the path explicitly
    """offcap §3.2.9 / §5.1.2: the two named overrides are mutually exclusive —
    refused by name at `plan_configurations` (hence `plan_for`, hence the
    preflight, which replans from the same mapping) BEFORE either authorisation
    check, and at `main` before any Modal call."""
    kw = {**_offcap_plan_kwargs(), "allow_out_of_support_thinking": True}
    with pytest.raises(C.OverrideFlagsExclusive) as e:
        L.plan_for(**kw)
    assert C.FLAG_ALLOW_OOS_THINKING in str(e.value) and C.FLAG_ALLOW_OFF_CAP_OVERRIDE in str(e.value)
    with pytest.raises(C.OverrideFlagsExclusive):
        L.plan_configurations(["none"], ["off"], ["gsm8k"], model=_OOS_MODEL,
                              allow_out_of_support_thinking=True, allow_off_cap_override=True)
    # the preflight replans from plan_args: the same refusal (the planner is the door)
    with pytest.raises(C.OverrideFlagsExclusive):
        L.plan_for(cells=["none"], modes=["off"], controls=["gsm8k"], model=_OOS_MODEL,
                   allow_out_of_support_thinking=True, allow_off_cap_override=True)
    # main: refused before any plan, naming both
    called = []
    monkeypatch.setattr(L, "plan_for", lambda **k: called.append(k) or [])
    with pytest.raises(SystemExit) as e:
        L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, model=_OOS_MODEL, cells="none",
               modes="off", controls="gsm8k", allow_out_of_support_thinking=True,
               allow_off_cap_override=True, dry_run=True)
    assert C.FLAG_ALLOW_OOS_THINKING in str(e.value) and C.FLAG_ALLOW_OFF_CAP_OVERRIDE in str(e.value)
    assert called == []


def test_an_in_support_launch_is_untouched_by_the_comparator_code(L):
    """offcap R4 (the `test_nothing_changes_when_the_flag_is_absent` pattern): for
    a launch that does not name the flag, every planned field, the expectation's
    `as_dict()`, the driver argv, the standup argv and the launch argv are
    IDENTICAL whether the new parameter is absent or False — and carry no
    comparator marker, no cap key, no new flag."""
    common = dict(cells=["none", _OOS_CELL], modes=["off"], controls=["gsm8k"], model=_OOS_MODEL)
    plain = L.plan_for(**common)
    explicit = L.plan_for(**common, allow_off_cap_override=False)
    assert plain == explicit
    for cfg in plain:
        assert "off_cap_override" not in cfg and C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING not in cfg
        geom, _ = L.served_geometry(cfg, 1)
        exp = L.expected_identity(cfg, geom, pass_no=1, controls=cfg["controls"], samples=1)
        assert exp.off_cap_override is False and exp.out_of_support_thinking is False
        d = exp.as_dict()
        assert C.IDENTITY_KEY_MAX_TOKENS_BY_TASK not in d and exp.optional_identity() == {}
        assert set(d) == {"cell", "mode", C.GEOMETRY_PASS_FIELD, "served", "limits", "samples",
                          "client_lock_sha256", "tree_sha", "thinking_by_task", "sampling", "sampling_mode",
                          C.IDENTITY_KEY_OR_BENCH_ITEM_SET}
        ep = {"proxy_url": "u", "routing_table": "r"}
        cmd = L._driver_cmd("<t>", cfg, ep, None, geom, controls=cfg["controls"], limit=0,
                            inspect_limit=0, samples=1, cfg_path="<c>", vllm_log=None)
        assert C.FLAG_ALLOW_OFF_CAP_OVERRIDE not in cmd and C.FLAG_ALLOW_OOS_THINKING not in cmd
        assert C.FLAG_OOS_PROMPT_BUDGET not in cmd and C.FLAG_EXPECT_OBJECT_SHA256 not in cmd
    assert L.launch_argv("abc1234", pass1=True, model=_OOS_MODEL) == L.launch_argv(
        "abc1234", pass1=True, model=_OOS_MODEL, allow_off_cap_override=False)
    assert C.FLAG_ALLOW_OFF_CAP_OVERRIDE in L.launch_argv("abc1234", pass1=True, model=_OOS_MODEL,
                                                          allow_off_cap_override=True)
    assert L.offcap_configurations(plain) == [] and L.oos_prompt_budget_fit_report(_budget_off(), plain, pass_no=1) == {}


def test_expected_identity_of_a_comparator_configuration_carries_exactly_the_cap_key(L):
    """offcap §5.1.5: the PURE expectation (no server anywhere) of a comparator
    configuration carries `max_tokens_by_task` = {gsm8k_paired200: 8192} and
    NOTHING else optional — `out_of_support_thinking` False and omitted,
    `serve_out_of_support` None and omitted; `optional_identity()` is read off
    `contracts.tinycap_identity_fields(off_cap_override=True)`; the standup is
    undeclared (bare spellings) on both cells."""
    for cell in ("none", _OOS_CELL):
        exp = _offcap_expected(L, cell=cell)
        assert exp.off_cap_override is True and exp.out_of_support_thinking is False and not exp.declared
        assert exp.optional_identity() == {C.IDENTITY_KEY_MAX_TOKENS_BY_TASK: {C.GSM8K_STANDARD_TASK: _OFFCAP_CAP}}
        assert dict(exp.max_tokens_by_task) == {"inspect/gsm8k_paired200": 8192} and exp.serve_out_of_support is None
        d = exp.as_dict()
        assert C.IDENTITY_KEY_MAX_TOKENS_BY_TASK in d and C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING not in d
        assert C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT not in d
        assert set(d) - set(_expected(L, cell=cell, model=_OOS_MODEL, controls=["gsm8k"]).as_dict()) == {
            C.IDENTITY_KEY_MAX_TOKENS_BY_TASK}
        assert set(exp.optional_identity()) == set(C.tinycap_identity_fields(off_cap_override=True)) - set(C.TINYCAP_IDENTITY_FIELDS)
        assert exp.mode == "off" and exp.thinking_by_task == {C.GSM8K_STANDARD_TASK: "off"}
        assert exp.served["reasoning_parser"] is None
    # the two derivations of the cap map agree (the two-doors test holds them by execution too)
    assert L.offcap_caps_by_task(L.thinking_by_task_of("off", ["gsm8k"]), name="x") == {C.GSM8K_STANDARD_TASK: 8192}
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="names no cap"):
        L.offcap_caps_by_task({"inspect/math": "off"}, name="x")
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="names no cap"):
        L.offcap_caps_by_task({C.GSM8K_STANDARD_TASK: "on"}, name="x")                  # OFF tasks only: none -> refuses


def test_untrusted_store_reason_refuses_a_table_cap_store_under_the_comparator_and_the_reverse_naming_the_cap_key(L, tmp_path):
    """offcap §5.1.6 / R5: the three optional-key-set shapes are pairwise distinct,
    so a campaign OFF/512 store never resumes under the comparator's expectation
    and a comparator store never resumes under an in-support one — both
    directions refused naming `max_tokens_by_task`; an OOS store against a
    comparator expectation names the key-set difference; a comparator store
    whose cap disagrees is refused by value; the matching store is trusted."""
    cmp_exp = _offcap_expected(L)
    plain_exp = _expected(L, cell=_OOS_CELL, model=_OOS_MODEL, controls=["gsm8k"])
    oos_exp = _oos_expected(L)
    cmp_store = _store(tmp_path / "cmp", cmp_exp)
    assert L.untrusted_store_reason(cmp_store, expected=cmp_exp) is None
    why = L.untrusted_store_reason(cmp_store, expected=plain_exp)
    assert why and C.IDENTITY_KEY_MAX_TOKENS_BY_TASK in why and "another completion cap" in why
    plain_store = _store(tmp_path / "plain", plain_exp)
    assert L.untrusted_store_reason(plain_store, expected=plain_exp) is None
    why = L.untrusted_store_reason(plain_store, expected=cmp_exp)
    assert why and C.IDENTITY_KEY_MAX_TOKENS_BY_TASK in why and "another completion cap" in why
    oos_store = _store(tmp_path / "oos", oos_exp)
    why = L.untrusted_store_reason(oos_store, expected=cmp_exp)
    assert why and C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING in why
    why = L.untrusted_store_reason(cmp_store, expected=oos_exp)
    assert why and C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING in why
    # a comparator store filled at another cap: refused by value, the note names the cap
    other = _store(tmp_path / "other", cmp_exp, max_tokens_by_task={C.GSM8K_STANDARD_TASK: 512})
    why = L.untrusted_store_reason(other, expected=cmp_exp)
    assert why and "tinycap.max_tokens_by_task: the store records {'inspect/gsm8k_paired200': 512}" in why
    assert "another completion cap" in why


def test_a_done_comparator_task_needs_a_valid_outcome_record_at_the_launcher_door(L, tmp_path):
    """offcap §3.4 / §5.1.8 (the OOS door test's twin): for a store carrying ONLY the
    cap key, a done marker + summary WITHOUT the sidecar is NOT done
    (`invalid_done_marker_reason` files the named reason; `resume_report` has no
    `done_in`); an unparsable or disagreeing sidecar is refused by name; a valid
    one makes the task done. The predicate is `completion.store_requires_outcomes`,
    shared with the driver's `is_done`. An in-support store is still done without it."""
    from sisteer.capeval.tinycap import completion

    task = C.GSM8K_STANDARD_TASK
    exp = _offcap_expected(L)
    out = tmp_path / "thinking_off"
    store = _store(out, exp, tasks=["gsm8k"])
    assert completion.store_has_cap_override(store) and completion.store_requires_outcomes(store)
    assert not completion.store_is_out_of_support(store)
    assert L.untrusted_store_reason(store, expected=exp) is None
    _finish(store, task, C.GSM8K_PAIRED_N)
    why = L.invalid_done_marker_reason(store, task)
    assert why is not None and why.startswith("no outcomes/inspect__gsm8k_paired200.json beside the done marker")
    assert C.IDENTITY_KEY_MAX_TOKENS_BY_TASK in why and "rebuilds it from the retained .eval logs" in why
    assert why == completion.task_outcomes_reason(store, task)
    assert not completion.is_done(store, task, require_outcomes=True)
    report = L.resume_report(out, ["gsm8k"], exp)
    assert report["controls"]["gsm8k"]["done_in"] is None
    assert report["controls"]["gsm8k"]["why_not"] == {store.name: {task: why}}
    p = completion.task_outcomes_path(store, task)
    p.parent.mkdir(exist_ok=True)
    p.write_text("{torn")
    assert "unreadable" in L.invalid_done_marker_reason(store, task)
    summary = _offcap_summary(task, n=C.GSM8K_PAIRED_N)
    completion.write_task_summary(store, task, summary)
    good = _offcap_outcomes(task, summary)
    completion.write_task_outcomes(store, task, {**good, "n_at_cap": 3})
    assert "n_at_cap 3 != 0" in L.invalid_done_marker_reason(store, task)
    completion.write_task_outcomes(store, task, good)
    assert L.invalid_done_marker_reason(store, task) is None
    assert L.resume_report(out, ["gsm8k"], exp)["controls"]["gsm8k"]["done_in"] == store.name
    # in support: done on the bare marker, as ever
    plain = _expected(L, cell=_OOS_CELL, model=_OOS_MODEL, controls=["gsm8k"])
    pstore = _store(tmp_path / "plain", plain, tasks=["gsm8k"])
    _finish(pstore, task, C.GSM8K_PAIRED_N)
    assert L.invalid_done_marker_reason(pstore, task) is None and not completion.store_requires_outcomes(pstore)


def test_a_comparator_launch_needs_pf_and_hands_every_container_the_off_artifact_the_digest_and_the_flag(
        L, monkeypatch, capsys):
    """offcap §5.1.2, §5.1.7: a comparator configuration launches only behind
    --pf (both cells: refused before any Modal call naming the configuration);
    with --pf the VALIDATED OFF artifact object rides into every container; the
    psr driver argv ends `[…, --allow-off-cap-override, --expect-object-sha256,
    <sha>, --oos-prompt-budget, <path>]`, the none argv `[…,
    --allow-off-cap-override, --oos-prompt-budget, <path>]`; no
    `--allow-out-of-support-thinking` anywhere."""
    kw = {"app_name": L.APP_NAME, "sha": "abc1234", "pass1": True, "model": _OOS_MODEL,
          "cells": "none", "modes": "off", "controls": "gsm8k", "allow_off_cap_override": True,
          # native-serving-default: the authorised comparator is a LENS launch (its psr leg)
          "serving_path": "lens"}
    with pytest.raises(SystemExit) as e:
        L.main(**kw)
    assert "launch only behind --pf" in str(e.value)
    assert C.container_name("none", "off", model_slug=_OOS_MODEL) in str(e.value)
    capsys.readouterr()
    budget = _budget_off(prompt_max=2375)
    launched: dict = {}

    def _remote(sha, plan_args, expect):
        assert plan_args["allow_off_cap_override"] is True and plan_args["allow_out_of_support_thinking"] is False
        rep = _preflight_report(L, L.plan_for(**plan_args))
        rep["oos_prompt_budget"] = {"path": "/vol/x/oos_prompt_budget.json", **budget}
        return rep

    def _on_spawn(cfg, kwargs):
        # fan_out spawns ONE FunctionCall per configuration (bcb-bfcl-200)
        launched.setdefault("cfgs", []).append(cfg)
        launched["kwargs"] = kwargs
    monkeypatch.setattr(L, "preflight", SimpleNamespace(remote=_remote))
    monkeypatch.setattr(L, "run_configuration", _spawn_only(_done, on_spawn=_on_spawn))
    L.main(**{**kw, "pf": True, "cells": f"none,{_OOS_CELL}"})
    assert launched["kwargs"]["oos_prompt_budget"] == budget and "path" not in launched["kwargs"]["oos_prompt_budget"]
    assert [c["cell"] for c in launched["cfgs"]] == ["none", _OOS_CELL]
    out = capsys.readouterr().out
    assert "prompt budget (from --pf" in out and "cap 8192 (inspect/gsm8k_paired200)" in out
    assert C.FLAG_ALLOW_OFF_CAP_OVERRIDE in out and C.FLAG_ALLOW_OOS_THINKING not in out
    # the argv every container's driver gets
    for cfg in launched["cfgs"]:
        geom, _ = L.served_geometry(cfg, 1)
        ep = {"proxy_url": "u", "routing_table": "r"}
        lens = cfg["cell"] == _OOS_CELL
        cmd = L._driver_cmd("<t>", cfg, ep, None, geom, controls=cfg["controls"], limit=0, inspect_limit=0,
                            samples=1, cfg_path="<c>", vllm_log=None, oos_prompt_budget="/o/b.json",
                            expect_object_sha256=(FAKE_DIGEST if lens else None))
        tail = ([C.FLAG_ALLOW_OFF_CAP_OVERRIDE, C.FLAG_EXPECT_OBJECT_SHA256, FAKE_DIGEST, C.FLAG_OOS_PROMPT_BUDGET, "/o/b.json"]
                if lens else [C.FLAG_ALLOW_OFF_CAP_OVERRIDE, C.FLAG_OOS_PROMPT_BUDGET, "/o/b.json"])
        assert cmd[-len(tail):] == tail and C.FLAG_ALLOW_OOS_THINKING not in cmd
        assert cmd[cmd.index(C.FLAG_THINKING) + 1] == "off"
        if lens:
            with pytest.raises(ValueError, match="needs the artifact digest"):
                L._driver_cmd("<t>", cfg, ep, None, geom, controls=cfg["controls"], limit=0, inspect_limit=0,
                              samples=1, cfg_path="<c>", vllm_log=None, oos_prompt_budget="/o/b.json")
        else:
            with pytest.raises(ValueError, match="artifact digest was offered"):
                L._driver_cmd("<t>", cfg, ep, None, geom, controls=cfg["controls"], limit=0, inspect_limit=0,
                              samples=1, cfg_path="<c>", vllm_log=None, expect_object_sha256=FAKE_DIGEST)
    # a report without the artifact refuses the launch
    monkeypatch.setattr(L, "preflight", SimpleNamespace(
        remote=lambda sha, plan_args, expect: _preflight_report(L, L.plan_for(**plan_args))))
    with pytest.raises(SystemExit, match="carries no oos_prompt_budget"):
        L.main(**{**kw, "pf": True})


def test_the_preflight_budgets_the_comparator_under_the_off_render_with_the_override_cap(L, tmp_path, monkeypatch):
    """offcap §5.2 / §5.3: for a comparator plan the budget argv carries `--arm off
    --off-cap-override` (the artifact's `cap` is the OVERRIDE cap, so
    `prompt_budget_fit`'s equality holds — the coordinator's trap closed at
    the source); the fit report takes each configuration's OWN arm and
    `override_caps`: passes at 16384 (margin 5817 at 2375) and fails at 10,000;
    a stale 512 artifact refuses naming both caps; the probe flags gain the new
    flag; the two preflight checks run over `budgeted = oos_on + offcap`."""
    cfgs = _offcap_plan(L)
    row = C.model_row(_OOS_MODEL)
    argv = L.oos_prompt_budget_argv("/vol/t", row, "/vol/b.json", tree_sha=FULL_SHA, arm=C.THINKING_OFF,
                                    off_cap_override=True)
    assert argv == [L.CLIENT_PY, "/vol/t/scripts/tinycap_oos_prompt_budget.py", "--model", "qwen35_4b",
                    "--arm", "off", "--off-cap-override", "--repo-root", "/vol/t", "--offline",
                    "--tree-sha", FULL_SHA, "--out", "/vol/b.json"]
    assert argv[-1] == "/vol/b.json"                                                   # --out stays LAST
    assert L.oos_prompt_budget_argv("/vol/t", row, "/vol/b.json", tree_sha=FULL_SHA)[4:6] == ["--arm", "on"]
    with pytest.raises(ValueError, match="off-cap override"):
        L.oos_prompt_budget_argv("/vol/t", row, "/vol/b.json", tree_sha=FULL_SHA, arm=C.THINKING_ON,
                                 off_cap_override=True)
    calls = []

    def fake_run(argv, **kw):
        calls.append(list(argv))
        Path(argv[-1]).write_text(json.dumps(_budget_off()))
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(L.subprocess, "run", fake_run)
    got = L.run_oos_prompt_budget("/vol/t", row, str(tmp_path / "b.json"), tree_sha=FULL_SHA, python="/py",
                                  arm=C.THINKING_OFF, off_cap_override=True)
    assert got == _budget_off() and "--off-cap-override" in calls[0] and calls[0][calls[0].index("--arm") + 1] == "off"
    # the fit report over the comparator: the OFF arm, the override cap
    budget = _budget_off(prompt_max=2375)
    L_res = sg.PETRI_MAX_MODEL_LEN
    rep = L.oos_prompt_budget_fit_report(budget, cfgs, pass_no=1)
    assert sorted(rep) == sorted(c["name"] for c in cfgs)
    for rec in rep.values():
        assert (rec["prompt_max"], rec["cap"], rec["max_model_len"], rec["total"]) == (2375, 8192, L_res, 10567)
        assert rec["margin"] == L_res - 10567 == 5817 and rec["task"] == C.GSM8K_STANDARD_TASK
    with pytest.raises(C.OutOfSupportPromptBudgetError, match=r"prompt max 2375 \+ cap 8192 = 10567 > the resolved max_model_len 10000"):
        L.oos_prompt_budget_fit_report(budget, cfgs, pass_no=1, geom_of=lambda c: SimpleNamespace(
            max_model_len=10000, max_num_seqs=1, window_source="t", seqs_source="t"))
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="written for cap 512 but the run's cap is 8192"):
        L.oos_prompt_budget_fit_report(_budget(arm=C.THINKING_OFF, cap=512), cfgs, pass_no=1)
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="tokenized the 'on' render"):
        L.oos_prompt_budget_fit_report(_budget(), cfgs, pass_no=1)                      # the ON artifact
    # the 2x2's report is unchanged
    assert set(L.oos_prompt_budget_fit_report(_budget(prompt_max=2000), _oos_plan(L), pass_no=1)) == {
        c["name"] for c in L.oos_on_configurations(_oos_plan(L))}
    # the preflight body: budgeted = oos_on + offcap, ONE arm, the argv per arm, the probe flags
    src = LAUNCHER.read_text()
    body = src[src.index("def preflight("):src.index("    def datasets():")]
    assert "budgeted = oos_on + offcap" in body
    assert "oos_prompt_budget_fit_report(budget, budgeted, pass_no=pass_no, tree=tree)" in body
    assert "off_cap_override=bool(offcap)" in body
    assert "(C.FLAG_ALLOW_OFF_CAP_OVERRIDE,) if offcap else ()" in body
    assert "(C.FLAG_OOS_PROMPT_BUDGET,) if budgeted else ()" in body
    assert "(C.FLAG_EXPECT_OBJECT_SHA256,) if any(landed_lens_configuration(c) for c in cfgs) else ()" in body
    probed = L.driver_probe_argv("/t", extra_flags=(C.FLAG_ALLOW_OFF_CAP_OVERRIDE, C.FLAG_OOS_PROMPT_BUDGET))
    assert probed[probed.index(C.FLAG_ALLOW_OFF_CAP_OVERRIDE) + 1] == C.FLAG_OOS_PROMPT_BUDGET   # a bare switch


def _offcap_dry_run(L, **kw):
    # native-serving-default: the authorised comparator is a LENS launch (its psr leg)
    kw.setdefault("serving_path", "lens")
    L.main(app_name=L.APP_NAME, sha="abc1234", pass1=True, model=_OOS_MODEL, cells=f"none,{_OOS_CELL}",
           modes="off", controls="gsm8k", allow_off_cap_override=True, dry_run=True,
           out_suffix="offcap-abc1234", **kw)


def test_the_dry_run_prints_the_off_cap_block_the_budget_at_arm_off_and_no_thinking_variant(L, capsys, tmp_path):
    """offcap §5.4 (C2 acceptance): per comparator configuration the dry run prints
    the OFF CAP OVERRIDE block with the note, the cap line (`… -> 8192`, twice),
    the resolved geometry, the prompt budget at `arm off` (with the OFF
    artifact) or NOT MEASURED naming `--arm off --off-cap-override`, the
    per-request declaration `none` with the live base-route probe, the bare
    spellings, `think_split_mode: record_only` twice, the comparator's sampling
    sentence; the resume identity carries `max_tokens_by_task` and neither OOS
    key; `@thinking` appears nowhere; no OUT OF SUPPORT block; the OOS flag is
    absent; `--upstream-timeout-s 7200` follows `--ready-timeout-s 2700` on both
    standups; `2 GPU configuration container(s)`."""
    L_res = sg.PETRI_MAX_MODEL_LEN
    S_res = sg.BUDGET_BY_MODEL[_OOS_MODEL].seqs_at(L_res)
    _offcap_dry_run(L)
    out = capsys.readouterr().out
    cap_line = ("thinking-OFF cap override (contracts.AUTHORISED_OFF_CAP_COMPARATOR.max_tokens_off): "
                f"inspect/gsm8k_paired200 -> {_OFFCAP_CAP}")
    assert out.count(cap_line) == 2
    assert out.count("OFF CAP OVERRIDE: --allow-off-cap-override raises the thinking-OFF cap of 1 task(s)") == 2
    assert out.count("no refusal lifted, no declaration sent, render thinking_off") == 2
    assert out.count(C.OFF_CAP_OVERRIDE_NOTE) == 2
    assert out.count(f"resolved launch geometry (pass 1): L {L_res} x S {S_res} (window ") == 2
    assert out.count("prompt budget: NOT MEASURED in this dry run") == 2
    assert f"{C.OOS_PROMPT_BUDGET_SCRIPT} --model qwen35_4b --arm off --off-cap-override" in out
    assert out.count("per-request declaration: none — thinking-off sends chat_template_kwargs") == 2
    assert out.count("400 thinking_on_not_lens") == 2 and out.count("probed live before generation") == 2
    assert out.count("think_split_mode: record_only") == 2
    assert out.count(f"sampling: {C.AUTHORISED_OFF_CAP_COMPARATOR.what}") == 2
    assert "standup spellings: --cell none alone, no declaration" in out
    assert "standup spellings: bare --cell safety_refusal__psr_kl_all (undeclared)" in out
    assert out.count("render_regime=thinking_off") == 3          # none leg; psr leg + its probe twin
    assert "@thinking" not in out and "OUT OF SUPPORT:" not in out and C.FLAG_ALLOW_OOS_THINKING not in out
    assert out.count("--ready-timeout-s 2700 --upstream-timeout-s 7200") == 2
    assert out.count("driver: ") == 2 and out.count(f" {C.FLAG_ALLOW_OFF_CAP_OVERRIDE}") >= 2
    assert "2 GPU configuration container(s), pass 1" in out and "out roots under …/offcap-abc1234/" in out
    for line in out.splitlines():
        if line.startswith("   resume identity:"):
            ident = json.loads(line.split("resume identity:", 1)[1])
            assert ident[C.IDENTITY_KEY_MAX_TOKENS_BY_TASK] == {"inspect/gsm8k_paired200": 8192}
            assert C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING not in ident and C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT not in ident
            assert ident["mode"] == "off" and ident["thinking_by_task"] == {"inspect/gsm8k_paired200": "off"}
    # with the OFF artifact: the budget lines at arm off, margin 5817
    p = tmp_path / C.OOS_PROMPT_BUDGET_FILENAME
    p.write_text(json.dumps(_budget_off(prompt_max=2375, digest="c" * 16)))
    _offcap_dry_run(L, oos_prompt_budget=str(p))
    out = capsys.readouterr().out
    assert out.count(f"prompt budget (template {'c' * 16}, arm off, n {C.GSM8K_PAIRED_N}, Qwen/Qwen3.5-4B@") == 2
    assert out.count(f"max + cap 8192 = 10567 <= L {L_res}, margin {L_res - 10567}") == 2
    assert "NOT MEASURED" not in out
    # the ON artifact on the comparator: DOES NOT FIT (the arm), then a non-zero exit
    p.write_text(json.dumps(_budget(prompt_max=2000)))
    with pytest.raises(SystemExit) as e:
        _offcap_dry_run(L, oos_prompt_budget=str(p))
    assert "tokenized the 'on' render" in str(e.value) and capsys.readouterr().out.count("prompt budget: DOES NOT FIT") == 2
    # the __main__ argparse spells the flag and threads it to main
    src = LAUNCHER.read_text()
    assert 'ap.add_argument(ALLOW_OFF_CAP_OVERRIDE_FLAG, dest="allow_off_cap_override"' in src
    assert "allow_off_cap_override=a.allow_off_cap_override" in src
    assert L.ALLOW_OFF_CAP_OVERRIDE_FLAG == C.FLAG_ALLOW_OFF_CAP_OVERRIDE


def test_a_stacked_reduction_mode_is_refused_by_the_tuple_locally_before_any_remote_call(L, monkeypatch, capsys):
    # native-serving-default: the launcher's default is the native engine now; this test pins the LENS
    # launch (a PSR cell / the historical composition), so it states the path explicitly
    """The code review's H2: `--reduction-mode stacked` passed the launcher's dry
    run and its planning although the comparator tuple names `per_row` — the
    planner's authorisation and the preflight's planning input both omitted the
    requested mode, so two GPU containers would have stood up before the
    driver's served-identity check refused. The requested mode is a planning
    input now (`plan_args`, hence the preflight's replan) and is compared at
    plan time: the stacked dry run refuses locally with the tuple's refusal,
    plans zero GPU configurations and makes no remote call; the real path
    refuses before `preflight.remote`; the per_row dry run is unchanged. The
    parent's OOS door had the same gap and is closed the same way."""
    monkeypatch.delenv(C.LENS_STACKED_REDUCTIONS_ENV, raising=False)
    needle = "reduction_mode differs — requested 'stacked', authorised 'per_row'"
    # plan level, both doors
    with pytest.raises(C.OffCapOverrideUnauthorised) as e:
        L.plan_for(**_offcap_plan_kwargs(), reduction_mode="stacked")
    assert needle in str(e.value) and "AUTHORISED_OFF_CAP_COMPARATOR" in str(e.value)
    kw = _offcap_plan_kwargs()
    with pytest.raises(C.OffCapOverrideUnauthorised, match="reduction_mode differs"):
        L.plan_configurations(kw["cells"], kw["modes"], kw["controls"], model=kw["model"],
                              allow_off_cap_override=True, reduction_mode="stacked")
    with pytest.raises(C.OutOfSupportThinkingUnauthorised) as e:
        L.plan_for(**_oos_plan_kwargs(), reduction_mode="stacked")
    assert needle in str(e.value) and "AUTHORISED_OOS_THINKING" in str(e.value)
    assert len(L.plan_for(**_offcap_plan_kwargs(), reduction_mode="per_row")) == 1       # the tuple's mode plans
    assert len(L.plan_for(**_offcap_plan_kwargs())) == 1                                 # the launcher default IS per_row
    # main: no remote call, no plan, no container count — the dry run and the real path alike
    remote: list = []
    monkeypatch.setattr(L, "preflight", SimpleNamespace(remote=lambda *a, **k: remote.append("preflight")))
    monkeypatch.setattr(L, "run_configuration", _spawn_only(_done, on_spawn=lambda cfg, kw: remote.append("spawn")))
    described: list = []
    real_describe = L.describe_plan
    monkeypatch.setattr(L, "describe_plan", lambda *a, **k: described.append(k) or real_describe(*a, **k))
    with pytest.raises(C.OffCapOverrideUnauthorised) as e:
        _offcap_dry_run(L, reduction_mode="stacked")
    assert needle in str(e.value) and "launch configuration" in str(e.value)
    assert remote == [] and described == [] and "GPU configuration container(s)" not in capsys.readouterr().out
    with pytest.raises(C.OffCapOverrideUnauthorised, match="reduction_mode differs"):
        L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, pf=True, model=_OOS_MODEL, cells=f"none,{_OOS_CELL}",
               modes="off", controls="gsm8k", allow_off_cap_override=True, reduction_mode="stacked",
               out_suffix="offcap-abc1234")
    assert remote == [] and described == []
    # the per_row dry run: unchanged — two containers planned, the mode printed per configuration
    _offcap_dry_run(L)
    out = capsys.readouterr().out
    assert "2 GPU configuration container(s), pass 1" in out and out.count("reduction_mode: 'per_row'") == 2
    assert len(described) == 1 and described[0]["reduction_mode"] == "per_row" and remote == []
    # the preflight's planning input carries the requested mode (it replans from plan_args)
    src = LAUNCHER.read_text()
    assert '"reduction_mode": reduction_mode,' in src[src.index("plan_args: Dict[str, object] = {"):src.index("if smoke:\n        cfgs = plan_for(**plan_args)")]
    # (both planner calls; since ifeval-canonical-battery each also threads the IFEval inputs, `**ifeval_kw`)
    assert src[src.index("def plan_for("):src.index("def isolate_out_dirs(")].count(
        "reduction_mode=reduction_mode, pass_no=pass_no, **ifeval_kw)") == 2


def test_the_worker_refuses_a_stacked_comparator_or_oos_container_before_the_standup(L, tmp_path, monkeypatch):
    """The code review's H2, the worker half: `_run_configuration`'s pre-standup
    authorisation compares the reduction mode the container was TOLD to serve
    against the tuple, so a hand-built or mis-flagged stacked comparator (or OOS
    2x2) container refuses as unauthorised before the staged tree is touched, an
    out dir made or a standup paid for — the driver's check of the mode the
    standup actually ATTESTS (REDUCTION_MODE_MISMATCH) stays where it is, behind
    a standup that now never starts for this case."""
    cfg, out, seen = _offcap_world(L, tmp_path, monkeypatch)
    monkeypatch.setenv(C.LENS_STACKED_REDUCTIONS_ENV, "stacked")           # the container's env agrees: stacked asked
    rec = _run_offcap(L, cfg, reduction_mode="stacked")
    assert rec["status"] == C.REC_OFFCAP_UNAUTHORISED, (rec["status"], rec.get("error"))
    assert "reduction_mode differs — requested 'stacked', authorised 'per_row'" in rec["error"]
    assert "container run of" in rec["error"] and rec["reduction_mode_requested"] == "stacked"
    assert seen["popen"] == [] and seen["driver"] == [] and not out.exists() and "tree" not in rec
    # the OOS 2x2's worker door, identically
    cfg, out, seen = _oos_world(L, tmp_path / "oos", monkeypatch)
    monkeypatch.setenv(C.LENS_STACKED_REDUCTIONS_ENV, "stacked")
    rec = _run_oos(L, cfg, reduction_mode="stacked")
    assert rec["status"] == "OOS_UNAUTHORISED" and "reduction_mode differs" in rec["error"]
    assert seen["popen"] == [] and not out.exists()
    # per_row asked of the same worlds: past the authorisation (the fake standup attests per_row)
    cfg, out, seen = _offcap_world(L, tmp_path / "ok", monkeypatch)
    rec = _run_offcap(L, cfg, reduction_mode="per_row")
    assert rec["status"] != C.REC_OFFCAP_UNAUTHORISED and len(_standup_popens(seen)) == 1


def test_pass2_is_refused_by_both_tuples_locally_and_the_preflight_budgets_the_requested_pass(L, monkeypatch, capsys):
    """The code review's H3 (a) + (b): `--pass2 --pf` on the comparator was approved
    by the preflight at 2,375 + 8,192 = 10,567 <= 16,384 (it hardcoded pass 1)
    while pass 2 serves L 4,608 x S 78 for both cells, and the containers stood up
    before the driver's live-budget check refused. The experiment tuple fixes the
    geometry class (pass 1): the requested pass is a planning input compared at
    plan time, so `--pass2` refuses LOCALLY with the tuple's refusal — zero
    configurations, no `preflight.remote`, no map — on the real `--pf` path and
    the dry run alike; the preflight budgets the REQUESTED pass (its fit report at
    pass 2 refuses naming 4,608); the `--pass1` dry run is unchanged. The parent's
    OOS door had the same gap (its tuple assumes pass 1 / 16,384 x 22; pass 2
    would serve its ON legs at 10,752 x 33) and is closed the same way."""
    monkeypatch.delenv(C.LENS_STACKED_REDUCTIONS_ENV, raising=False)
    needle = "pass_no differs — requested 2, authorised 1"
    with pytest.raises(C.OffCapOverrideUnauthorised) as e:
        L.plan_for(**_offcap_plan_kwargs(), pass_no=2)
    assert needle in str(e.value) and "AUTHORISED_OFF_CAP_COMPARATOR" in str(e.value)
    with pytest.raises(C.OutOfSupportThinkingUnauthorised) as e:
        L.plan_for(**_oos_plan_kwargs(), pass_no=2)
    assert needle in str(e.value) and "AUTHORISED_OOS_THINKING" in str(e.value)
    kw = _offcap_plan_kwargs()
    with pytest.raises(C.OffCapOverrideUnauthorised, match="pass_no differs"):
        L.plan_configurations(kw["cells"], kw["modes"], kw["controls"], model=kw["model"],
                              allow_off_cap_override=True, pass_no=2)
    assert len(L.plan_for(**_offcap_plan_kwargs(), pass_no=1)) == 1 == len(L.plan_for(**_offcap_plan_kwargs()))
    # an in-support pass-2 plan is untouched by the tuples
    assert len(L.plan_for(cells=["none"], modes=["off"], controls=["gsm8k"], model=_OOS_MODEL, pass_no=2)) == 1
    # main: the REAL --pass2 --pf path and the --pass2 dry run refuse before any remote call
    remote: list = []
    monkeypatch.setattr(L, "preflight", SimpleNamespace(remote=lambda *a, **k: remote.append("preflight")))
    monkeypatch.setattr(L, "run_configuration", _spawn_only(_done, on_spawn=lambda cfg, kw: remote.append("spawn")))
    described: list = []
    real_describe = L.describe_plan
    monkeypatch.setattr(L, "describe_plan", lambda *a, **k: described.append(k) or real_describe(*a, **k))
    common = {"serving_path": "lens",  # native-serving-default: the lens launch (a PSR cell)
              "app_name": L.APP_NAME, "sha": "abc1234", "model": _OOS_MODEL, "cells": f"none,{_OOS_CELL}",
              "modes": "off", "controls": "gsm8k", "allow_off_cap_override": True, "out_suffix": "offcap-abc1234"}
    with pytest.raises(C.OffCapOverrideUnauthorised) as e:
        L.main(pass2=True, pf=True, **common)
    assert needle in str(e.value) and "launch configuration" in str(e.value)
    with pytest.raises(C.OffCapOverrideUnauthorised, match="pass_no differs"):
        L.main(pass2=True, dry_run=True, **common)
    with pytest.raises(C.OutOfSupportThinkingUnauthorised, match="pass_no differs"):
        L.main(pass2=True, pf=True, **{**common, "modes": "off,on", "allow_off_cap_override": False,
                                       "allow_out_of_support_thinking": True, "out_suffix": "oos-abc1234"})
    assert remote == [] and described == [] and "GPU configuration container(s)" not in capsys.readouterr().out
    # the --pass1 dry run: unchanged — two containers, pass 1
    _offcap_dry_run(L)
    out = capsys.readouterr().out
    assert "2 GPU configuration container(s), pass 1" in out and remote == [] and len(described) == 1
    assert described[0]["pass_no"] == 1
    # (b) the preflight's fit report at the REQUESTED pass: pass 1 fits (margin 5817), pass 2 does not
    cfgs = _offcap_plan(L)
    budget = _budget_off(prompt_max=2375)
    assert {r["margin"] for r in L.oos_prompt_budget_fit_report(budget, cfgs, pass_no=1).values()} == {5817}
    with pytest.raises(C.OutOfSupportPromptBudgetError,
                       match=r"prompt max 2375 \+ cap 8192 = 10567 > the resolved max_model_len 4608"):
        L.oos_prompt_budget_fit_report(budget, cfgs, pass_no=2)
    src = LAUNCHER.read_text()
    body = src[src.index("def preflight("):src.index("    def datasets():")]
    assert 'pass_no = int(plan_args.get("pass_no") or 1)' in body
    assert "oos_prompt_budget_fit_report(budget, budgeted, pass_no=pass_no, tree=tree)" in body
    main_body = src[src.index("plan_args: Dict[str, object] = {"):src.index("cfgs = isolate_out_dirs(cfgs, out_suffix)")]
    # `pass_no` is a planning input (a key of plan_args, wherever it sits — the
    # mapping has since gained keys after it), the one `main` resolved, never a
    # re-derivation from the --pass2 flag
    assert re.search(r'"pass_no":\s*pass_no\b', main_body) and "pass_no = 2 if pass2 else 1" not in main_body


def test_the_worker_holds_the_artifact_to_its_resolved_window_before_the_standup(L, tmp_path, monkeypatch):
    """The code review's H3 (c): the worker checked only the artifact's arm + cap
    (`prompt_budget_matches_run`) and reached `_standup_cmd` with whatever window
    it had resolved; now `prompt_budget_fit(..., max_model_len=geom.max_model_len)`
    runs against ITS resolved window as a second door. An artifact that fits pass
    1 (2,375 + 8,192 <= 16,384) but not the window this container resolves (pass
    2's 4,608) refuses before `_standup_cmd` / Popen, naming the three numbers; a
    pass-2 container refuses earlier still as unauthorised (door (a) at the
    worker, both tuples); the fitting case records the fit and stands up."""
    cfg, out, seen = _offcap_world(L, tmp_path, monkeypatch)
    real_geometry = L.served_geometry
    monkeypatch.setattr(L, "served_geometry", lambda c, pass_no, tree=None: real_geometry(c, 2, tree))
    rec = _run_offcap(L, cfg, budget=_budget_off(prompt_max=2375))
    assert rec["status"] == C.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH, (rec["status"], rec.get("error"))
    assert "prompt max 2375 + cap 8192 = 10567 > the resolved max_model_len 4608" in rec["error"]
    assert rec["geometry"]["max_model_len"] == 4608 and rec["geometry"]["max_num_seqs"] == 161   # 741,888 // 4608 (the 2026-09-13 row; the swept 720,896 gave 156, the halved 360,448 78)
    assert seen["popen"] == [] and "standup_argv" not in rec and "oos_prompt_budget_fit" not in rec
    assert out.is_dir() and not (out / "cells").exists()      # the out dir precedes the geometry; nothing stood up in it
    monkeypatch.setattr(L, "served_geometry", real_geometry)
    # door (a) at the worker: a pass-2 comparator / OOS container is not the authorised run
    rec = _run_offcap(L, cfg, pass_no=2)
    assert rec["status"] == C.REC_OFFCAP_UNAUTHORISED and "pass_no differs — requested 2, authorised 1" in rec["error"]
    assert seen["popen"] == [] and "tree" not in rec
    ocfg, _, oseen = _oos_world(L, tmp_path / "oos", monkeypatch)
    rec = _run_oos(L, ocfg, pass_no=2)
    assert rec["status"] == "OOS_UNAUTHORISED" and "pass_no differs" in rec["error"] and oseen["popen"] == []
    # the fitting case: the fit is recorded beside the artifact facts and the standup starts
    cfg, _, seen = _offcap_world(L, tmp_path / "ok", monkeypatch)
    rec = _run_offcap(L, cfg, budget=_budget_off(prompt_max=2375))
    fit = rec["oos_prompt_budget_fit"]
    assert (fit["prompt_max"], fit["cap"], fit["max_model_len"], fit["total"], fit["margin"]) == (2375, 8192, 16384, 10567, 5817)
    assert rec["status"] != C.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH and len(_standup_popens(seen)) == 1


def test_worker_refuses_a_comparator_container_without_the_artifact_or_with_an_on_artifact_before_the_standup(
        L, tmp_path, monkeypatch):
    """offcap §5.1.8: the worker requires the artifact on a comparator configuration
    (`needs_prompt_budget`) and ALSO holds its arm to the configuration's before
    the standup — no artifact, an ON artifact, another tree's: refused
    `OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH` with NO Popen; a `--thinking on`
    hand-built cfg refuses OFFCAP_UNAUTHORISED first; both markers together
    refuse by name."""
    for cell in ("none", _OOS_CELL):
        cfg, _out, seen = _offcap_world(L, tmp_path / cell, monkeypatch, cell=cell)
        rec = _run_offcap(L, cfg, budget=None)
        assert rec["status"] == C.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH and seen["popen"] == [], cell
        assert "no prompt-budget artifact" in rec["error"]
        rec = _run_offcap(L, cfg, budget=_budget())                         # the ON artifact
        assert rec["status"] == C.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH and seen["popen"] == []
        assert "'on'" in rec["error"] and "off" in rec["error"]
        rec = _run_offcap(L, cfg, budget=_budget_off(cap=512))              # the table's cap
        assert rec["status"] == C.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH and seen["popen"] == []
        assert "512" in rec["error"] and "8192" in rec["error"]
        assert rec["off_cap_override"] == cfg["off_cap_override"] and rec.get("thinking_override") is None
    # a hand-built cfg naming the marker under the ON arm: refused as unauthorised, first
    cfg, _out, seen = _offcap_world(L, tmp_path / "bad", monkeypatch)
    bad = dict(cfg, mode="on", name=cfg["name"].replace("think_off", "think_on"))
    rec = _run_offcap(L, bad)
    assert rec["status"] == C.REC_OFFCAP_UNAUTHORISED and "mode differs" in rec["error"] and seen["popen"] == []
    both = dict(cfg, **{C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING: True})
    rec = _run_offcap(L, both)
    assert rec["status"] == C.REC_OFFCAP_UNAUTHORISED and "mutually exclusive" in rec["error"] and seen["popen"] == []


def test_worker_recs_on_the_comparator_standups_hold_the_landed_record(L, tmp_path, monkeypatch):
    """offcap §5.1.8 / R11 (the `:5596` twin): the comparator's psr/OFF standup
    IS held to the landed record now — the bare attestation key must be ABSENT
    from the map (present -> MISMATCH), the map itself PRESENT (absent ->
    UNATTESTED), the error key present-null, `lens_salt_policy` REQUIRED, the
    per-cell `conditional` policy REQUIRED, the digest REQUIRED and equal, the
    template digest held to the OFF artifact; the none/OFF standup is held to
    the map, the error key, the salt policy and the artifact. A passing record
    proceeds to the (dead) ledger; the REC records the marker, the artifact and
    writes it into the out-root."""
    def leg(cell, sub, **over):
        drop = over.pop("_drop", ())
        cfg, _out, seen = _offcap_world(L, tmp_path / sub, monkeypatch, cell=cell)
        rec = dict(_FakeStandupProc.record, **over)
        for k in drop:
            rec.pop(k, None)
        _FakeStandupProc.record = rec
        return cfg, seen

    # psr / off — the comparator's lens leg, held to the landed record
    cfg, seen = leg(_OOS_CELL, "p0")
    rec = _run_offcap(L, cfg)
    assert rec["status"] == "LEDGER_FAILED" and rec["attestation_key"] == "si_safety__psr_kl_all"
    assert rec[_DECL_KEY] == {} and rec["resumed_prefill_policy"] == "conditional"
    assert rec["object_artifact_sha256"] == FAKE_DIGEST and rec["lens_salt_policy"] == "attested"
    assert rec["off_cap_override"]["caps"] == {C.GSM8K_STANDARD_TASK: 8192} and rec.get("thinking_override") is None
    assert rec["oos_prompt_budget"]["arm"] == "off" and rec["oos_prompt_budget"]["cap"] == 8192
    assert json.loads((Path(cfg["out_dir"]) / C.OOS_PROMPT_BUDGET_FILENAME).read_text()) == _budget_off()
    assert rec["expected_identity"][C.IDENTITY_KEY_MAX_TOKENS_BY_TASK] == {C.GSM8K_STANDARD_TASK: 8192}
    assert C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING not in rec["expected_identity"]
    assert seen["driver"] == []                                            # the ledger died first
    assert C.FLAG_ALLOW_OFF_CAP_OVERRIDE not in " ".join(seen["popen"][0])  # a driver flag, not a standup one
    assert "--cell safety_refusal__psr_kl_all" in " ".join(seen["popen"][0]) and "@thinking" not in " ".join(seen["popen"][0])
    cfg, seen = leg(_OOS_CELL, "p1", **{_DECL_KEY: {"si_safety__psr_kl_all": _DECL_VALUE}})
    why = _refused(L, _run_offcap(L, cfg), seen, "OOS_DECLARATION_MISMATCH")
    assert "wiring error" in why
    cfg, seen = leg(_OOS_CELL, "p2", _drop=(_DECL_KEY,))
    _refused(L, _run_offcap(L, cfg), seen, "OOS_DECLARATION_UNATTESTED")     # landed: the map is REQUIRED
    cfg, seen = leg(_OOS_CELL, "p3", _drop=(_DECL_ERR_KEY,))
    _refused(L, _run_offcap(L, cfg), seen, "OOS_DECLARATION_UNATTESTED")
    cfg, seen = leg(_OOS_CELL, "p4", _drop=("lens_salt_policy",))
    _refused(L, _run_offcap(L, cfg), seen, "OOS_SALT_POLICY_UNEXPECTED")     # REQUIRED now
    cfg, seen = leg(_OOS_CELL, "p5", _drop=("resumed_prefill_policy_cells",))
    _refused(L, _run_offcap(L, cfg), seen, "OOS_SALT_POLICY_UNEXPECTED")
    cfg, seen = leg(_OOS_CELL, "p6", resumed_prefill_policy_cells={"si_safety__psr_kl_all": "always"})
    _refused(L, _run_offcap(L, cfg), seen, "OOS_SALT_POLICY_UNEXPECTED")
    cfg, seen = leg(_OOS_CELL, "p7", object_artifact_sha256="cd" * 32)
    _refused(L, _run_offcap(L, cfg), seen, "OOS_DECLARATION_MISMATCH")
    cfg, seen = leg(_OOS_CELL, "p8", chat_template_hash="b" * 16)             # held to the OFF artifact now
    _refused(L, _run_offcap(L, cfg), seen, "OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH")
    # none / off — the base route, held to the map / error key / salt policy / artifact
    cfg, seen = leg("none", "n0")
    rec = _run_offcap(L, cfg)
    assert rec["status"] == "LEDGER_FAILED" and rec["attestation_key"] == "none"
    assert rec["object_artifact_sha256"] is None and rec["resumed_prefill_policy"] is None
    cfg, seen = leg("none", "n1", **{_DECL_KEY: {"none": _DECL_VALUE}})
    _refused(L, _run_offcap(L, cfg), seen, "OOS_DECLARATION_MISMATCH")
    cfg, seen = leg("none", "n2", _drop=(_DECL_KEY,))
    _refused(L, _run_offcap(L, cfg), seen, "OOS_DECLARATION_UNATTESTED")
    cfg, seen = leg("none", "n3", _drop=("lens_salt_policy",))
    _refused(L, _run_offcap(L, cfg), seen, "OOS_SALT_POLICY_UNEXPECTED")
    cfg, seen = leg("none", "n4", chat_template_hash="b" * 16)
    _refused(L, _run_offcap(L, cfg), seen, "OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH")
    # the 2x2's in-support psr/OFF leg keeps today's tolerance (the additive rule): untouched
    ocfg, _o, _oseen = _oos_world(L, tmp_path / "campaign", monkeypatch, cell=_OOS_CELL, mode="off",
                                 record=dict(_oos_endpoint(L, L.plan_for(
                                     cells=[_OOS_CELL], modes=["off"], controls=["gsm8k"], model=_OOS_MODEL,
                                     allow_out_of_support_thinking=True)[0])))
    rec = _run_oos(L, ocfg, budget=None)
    assert rec["status"] == "LEDGER_FAILED"


def test_a_torn_comparator_store_with_a_valid_retained_log_proceeds_and_without_one_refuses_before_the_standup(
        L, tmp_path, monkeypatch):
    """offcap §5.1.8 [R1] (the `:5171` twin on a store carrying ONLY the
    cap key): a completed comparator store (cap key, done marker + summary + a
    VALID sidecar) -> DONE, no Popen; the same store minus its sidecar with no
    retained log -> `OOS_TORN_STORE_UNREPAIRABLE` BEFORE the standup (the
    pre-standup check reads `completion.store_requires_outcomes`), rows and
    summary byte-unchanged; with a usable retained log -> the standup is
    invoked and the driver argv is the comparator's (the driver repairs)."""
    from sisteer.capeval.tinycap import completion

    task = C.GSM8K_STANDARD_TASK
    cfg, out, seen = _offcap_world(L, tmp_path, monkeypatch)
    exp = _offcap_expected(L, tree=_tree(tmp_path / "repos", "f6cf462"))
    store = _store(out, exp, tasks=["gsm8k"])
    _finish(store, task, C.GSM8K_PAIRED_N)
    summary = _offcap_summary(task, n=C.GSM8K_PAIRED_N)
    completion.write_task_summary(store, task, summary)
    completion.write_task_outcomes(store, task, _offcap_outcomes(task, summary))
    rec = _run_offcap(L, cfg)
    assert rec["status"] == "DONE" and rec["controls_remaining"] == [] and seen["popen"] == []
    assert rec["resume"]["controls"]["gsm8k"]["done_in"] == store.name
    assert rec["expected_identity"][C.IDENTITY_KEY_MAX_TOKENS_BY_TASK] == {task: 8192}

    def file_level_probe(tree, checks, *, python=None):
        return {c["log_dir"]: (None if (Path(c["log_dir"]) / "ok.eval").is_file()
                               else f"no retained .eval log under {c['log_dir']}") for c in checks}
    monkeypatch.setattr(L, "retained_log_reasons", file_level_probe)
    completion.task_outcomes_path(store, task).unlink()
    summary_bytes = (store / C.SUMMARY_DIR / C.marker_filename(task)).read_bytes()
    rec = _run_offcap(L, cfg)
    assert rec["status"] == L.REC_OOS_TORN_STORE_UNREPAIRABLE and seen["popen"] == []
    assert store.name in rec["error"] and task in rec["error"] and "no retained .eval log" in rec["error"]
    assert "a fresh store (a new out-suffix) or an explicit operator repair" in rec["error"]
    assert (store / C.SUMMARY_DIR / C.marker_filename(task)).read_bytes() == summary_bytes
    assert not completion.task_outcomes_path(store, task).exists()
    # a usable retained log: the standup is paid for and the driver argv is the comparator's
    log_dir = L.inspect_log_dir(store, task, 0)
    log_dir.mkdir(parents=True)
    (log_dir / "ok.eval").write_bytes(b"COMPLETE")
    rec = _run_offcap(L, cfg)
    assert rec["status"] == "LEDGER_FAILED" and len(seen["popen"]) == 2        # the standup ran, then the ledger died
    assert "capctrl_serving_up.py" in " ".join(seen["popen"][0]) and "--cell safety_refusal__psr_kl_all" in " ".join(seen["popen"][0])
    assert rec["controls_remaining"] == ["gsm8k"]
    # an IN-SUPPORT store (no cap key) is never asked for a sidecar: the pre-standup check skips it
    plain = _expected(L, cell=_OOS_CELL, model=_OOS_MODEL, controls=["gsm8k"], tree=_tree(tmp_path / "repos", "f6cf462"))
    assert L.torn_store_unrepairable(out, ["gsm8k"], plain, L.resume_report(out, ["gsm8k"], plain), tree="/t") is None


# --------------------------------------------------------------------------- #
# the MATH-500 default change: every NON-MATH configuration's                  #
# resolved geometry is pinned to a golden fixture captured at main f90557fd   #
# through the launcher's REAL resolution (served_geometry), both passes        #
# --------------------------------------------------------------------------- #

GOLDEN_GEOMETRY_PATH = golden.GOLDEN_DIR / "served_geometry.json"

#: The configurations whose window, S and key must not move: every non-MATH
#: control family (GSM8K paired + full, both EvalPlus tasks, the Inspect
#: comparison controls, HellaSwag-Inspect, BFCL), base / prompted /
#: fitted cells, both arms and per-task, the default battery, and a second
#: measured model. MATH is deliberately NOT here — its geometry moves by
#: design, and is pinned separately.
#: (the `hellaswag_inspect` and full-`bfcl` cases left with the change of
#: 2026-09-11 — retired controls plan nothing, so they have no geometry to pin)
_GEOMETRY_GOLDEN_CASES = (
    dict(cells=["none"], modes=["off"], controls=["gsm8k"]),
    dict(cells=["none"], modes=["on"], controls=["gsm8k"]),
    dict(cells=[C.CELL_PROMPTED_SAFETY], modes=["off"], controls=["gsm8k"]),
    dict(cells=[C.CELL_PSRKL_SAFETY], modes=["on"], controls=["gsm8k"]),
    dict(cells=[C.CELL_PSRKL_SAFETY], modes=["off"], controls=["gsm8k_full"]),
    dict(cells=[C.CELL_PSRKL_SAFETY], modes=["off"], controls=["humaneval_plus"]),
    dict(cells=[C.CELL_PSRKL_SAFETY], modes=["on"], controls=["mbpp_plus"]),
    dict(cells=["none"], modes=["off"], controls=["inspect_cmp"]),
    dict(cells=[C.CELL_PROMPTED_SAFETY], modes=["on"], controls=["inspect_cmp"]),
    dict(cells=["none"], modes=["per-task"], controls=["gsm8k", "humaneval_plus"]),
    dict(),                                              # the default battery, every default cell, both arms
    dict(cells=["none"], modes=["off"], controls=["gsm8k"], model="qwen35_9b"),
    dict(cells=[C.CELL_PSRKL_SAFETY], modes=["off"], controls=["humaneval_plus"], model="qwen35_9b"),
)


def _geometry_record(L, cfg) -> dict:
    """What the launcher RESOLVES for one configuration: its geometry keys and,
    per pass, the window, S, key and sources ``served_geometry`` returns — or
    the CLASS of the refusal it raises (pass 2 refuses a multi-key or still
    provisional configuration by design; that refusal is part of the pin)."""
    rec = {"geometry_keys": list(cfg["geometry_keys"])}
    for pass_no in (1, 2):
        try:
            g, budget = L.served_geometry(cfg, pass_no)
        except (ValueError, sg.GeometryProvisional) as e:
            rec[f"pass{pass_no}"] = {"refused": type(e).__name__}
        else:
            rec[f"pass{pass_no}"] = {"eval": g.eval, "max_model_len": g.max_model_len,
                                    "max_num_seqs": g.max_num_seqs,
                                    "window_source": g.window_source, "seqs_source": g.seqs_source,
                                    "model_key": budget["model_key"]}
    return rec


def test_every_non_math_configuration_resolves_the_pre_change_golden_geometry(L):
    """Giving MATH-500 its own geometry class must leave the window
    and S every OTHER control is served at — hashed into ``served`` by the
    driver for GSM8K, HellaSwag, HumanEval+, MBPP+ and the comparison controls
    — byte-identical. Pinned through the launcher's real resolution
    (``served_geometry``, both passes), against a fixture captured from main
    f90557fd; a change here re-keys every store of that configuration."""
    got = {}
    for case in _GEOMETRY_GOLDEN_CASES:
        for cfg in L.plan_configurations(**case):
            key = f"{cfg['name']}|{','.join(cfg['controls'])}"
            assert key not in got, key
            got[key] = _geometry_record(L, cfg)
    assert len(got) >= 16          # the retired controls' cases and the removed tiny family's cases left
    golden.compare_or_write(GOLDEN_GEOMETRY_PATH, got)


#: MATH-500's own window and the S the four full22 models' budgets buy at it
#: (`ModelBudget.seqs_at`; the hybrid 256 cap never binds) — vs 54 / 33 / 73 / 73
#: at gen_long_thinking's 10,752, i.e. x(10,752 / 17,664) = 0.609 concurrency.
MATH500_WINDOW = 17664
MATH500_SEQS_BY_MODEL = {"gemma4e4b": 33, "qwen35_4b": 42, "glm46vflash": 44, "nemotron3nano4b": 44}   # qwen35_4b: 741,888 = 42 x 17,664, measured AT this window 2026-09-13


def test_a_math500_configuration_resolves_its_own_entry_in_both_arms(L):
    """MATH-500's one-pass 16,384 cap needs a
    window that holds the measured prompt max + 16,384 (1,041 + 16,384 = 17,425),
    and it gets one WITHOUT touching either gen_long entry — its own geometry
    class resolves to its OWN registered entry ``capability:gen_long_math``
    (17,664 x S(model)) in BOTH arms, prompted or not, through the launcher's
    real resolution. gemma4e4b's budget buys 33 there (54 at the 10,752 MATH
    used to resolve to) — a served-identity change for MATH stores only, whose
    key already moves with the cap (``math500_max_tokens``)."""
    for cell, mode in (("none", "off"), ("none", "on"), (C.CELL_PROMPTED_SAFETY, "off"),
                       (C.CELL_PSRKL_SAFETY, "per-task")):
        assert L.geometry_keys_for(cell, mode, ["math500"]) == ("capability:gen_long_math",)
    for mode in ("off", "on"):
        (cfg,) = L.plan_configurations(cells=[C.CELL_PSRKL_SAFETY], modes=[mode], controls=["math500"])
        assert cfg["geometry_keys"] == ("capability:gen_long_math",)
        g2, rec = L.served_geometry(cfg, 2)
        assert (g2.eval, g2.max_model_len, g2.max_num_seqs) == ("capability:gen_long_math", MATH500_WINDOW, 33)
        assert g2.max_num_seqs == sg.BUDGET_BY_MODEL[C.MODEL_SLUG].seqs_at(MATH500_WINDOW) == 589_824 // MATH500_WINDOW
        assert rec["model_key"] == C.MODEL_SLUG and g2.seqs_source == L.MODEL_BUDGET_SOURCE
        assert 1041 + C.MATH500_MAX_TOKENS_DEFAULT <= g2.max_model_len       # MATH's prompt MAX + the cap
        assert g2.entry.basis.max_prompt == 1041 and g2.entry.basis.requested_max_tokens == 16384
        assert g2.entry.basis.sized_to_max_prompt and rec["basis"]["max_model_len"] is not None   # the BUDGET's basis, not the window's
        # pass 1 resolves the SAME window — the footprint cannot hold MATH's cap
        # (the pass-1 path is pinned in test_a_math500_only_configuration_is_served_…)
        g1, rec1 = L.served_geometry(cfg, 1)
        assert (g1.eval, g1.max_model_len, g1.max_num_seqs, g1.window_source) == (
            "capability:gen_long_math", MATH500_WINDOW, 33, "table")
        assert rec1 == rec and not g1.measurement_pass and not g1.provisional
    # every full22 model's S at MATH's window: through the launcher's real resolution
    # for the models its registry serves (thinking-OFF — the launcher refuses ON
    # without a pinned parser; the window is the same in both arms), and through
    # the geometry module's own resolution for glm / nemotron, which the launcher's
    # registry lacks (the cluster wrapper serves those)
    for model, seqs in MATH500_SEQS_BY_MODEL.items():
        if model in C.MODELS:
            (cfg,) = L.plan_configurations(cells=["none"], modes=["off"], controls=["math500"], model=model)
            g2, rec = L.served_geometry(cfg, 2)
            assert (g2.eval, g2.max_model_len, g2.max_num_seqs, rec["model_key"]) == (
                "capability:gen_long_math", MATH500_WINDOW, seqs, model), model
        r = sg.resolve_geometry("capability:gen_long_math", strict_identity=True, model_key=model)
        assert (r.max_model_len, r.max_num_seqs) == (MATH500_WINDOW, seqs), model
        assert r.max_num_seqs == sg.BUDGET_BY_MODEL[model].seqs_at(MATH500_WINDOW)
    assert {"gemma4e4b", "qwen35_4b"} <= set(C.MODELS)
    # the in-depth OFF battery pools a SECOND key beside gen_long: the MATH atom's;
    # pass 1 REFUSES it by name (the footprint cannot hold MATH's cap, and one
    # standup cannot serve MATH's window beside the others'), pass 2 still refuses a
    # multi-key configuration (nothing else about the battery's geometry moved)
    (deep,) = L.plan_configurations(cells=[C.CELL_PSRKL_SAFETY], modes=["off"], in_depth=True)
    assert set(C.IN_DEPTH_EXTRAS) <= set(deep["controls"])
    assert deep["geometry_keys"] == ("capability:gen_long", "capability:gen_long_math")
    with pytest.raises(L.Pass1FootprintCannotHold):
        L.served_geometry(deep, 1)
    with pytest.raises(L.Pass2Unpartitioned):
        L.served_geometry(deep, 2)
    # the full GSM8K extra stays on gen_long: only MATH moved
    (full,) = L.plan_configurations(cells=[C.CELL_PSRKL_SAFETY], modes=["off"], controls=["gsm8k_full"])
    assert full["geometry_keys"] == ("capability:gen_long",)
    assert L.served_geometry(full, 2)[0].max_model_len == 4608


def _math_at_cap(prompt_tokens: int):
    """One MATH-500 ``ItemResult`` that stopped AT THE CAP with a measured
    ``prompt_tokens`` prompt — exactly what the driver's window guard reads."""
    from sisteer.capeval.tiers import inspect_served as isv
    return isv.ItemResult(item_id=f"math_{prompt_tokens}", value=0.0, sample_idx=0, seed=0, answer="",
                          prompt_tokens=prompt_tokens, completion_tokens=C.MATH500_MAX_TOKENS_DEFAULT,
                          paired_id=None, scores={}, stop_reason=isv.STOP_REASON_AT_CAP)


def _holds_math_at_cap(geom) -> None:
    """The RESOLVED window holds MATH-500's measured prompt MAX at the default cap —
    by the arithmetic AND through the driver's OWN guard (``assert_pass_fits_window``,
    which refuses the task by name when any cap-bound sample's prompt + cap exceeds
    the served window; on the earlier footprint it read ``needs 1041 prompt + 16384
    = 17425 tokens`` against ``max_model_len=16384``)."""
    from sisteer.capeval.tiers import inspect_served as isv
    entry = sg.geometry_for("capability:gen_long_math")
    assert geom.eval == "capability:gen_long_math" and geom.entry is entry
    assert entry.basis.max_prompt + C.MATH500_MAX_TOKENS_DEFAULT <= geom.max_model_len
    isv.assert_pass_fits_window(C.inspect_task("math"), [_math_at_cap(entry.basis.max_prompt)],
                                max_tokens=C.MATH500_MAX_TOKENS_DEFAULT, max_model_len=geom.max_model_len)


def test_a_math500_only_configuration_is_served_at_its_own_window_in_pass_1_too(L, capsys):
    # native-serving-default: the launcher's default is the native engine now; this test pins the LENS
    # launch (a PSR cell / the historical composition), so it states the path explicitly
    """Reproduced through this launcher: MATH-500's
    default cap is 16,384 and the pass-1 footprint is 16,384 TOTAL, so EVERY MATH
    prompt overflowed it — `--pass1 --controls math500` resolved
    `pass1_geometry` and the driver's guard refused the task by name (1,041 +
    16,384 = 17,425 > 16,384) after the standup was paid for; a live request
    would have hit the context limit first. A MATH-ONLY configuration now
    resolves its OWN measured window in pass 1 exactly as in pass 2 —
    `capability:gen_long_math`, 17,664 x S(model) — in both arms, on base /
    prompted / fitted cells, for every model the registry serves: there is no
    footprint pass to run for it (the entry is sized from the prompt budget, not
    from a ledger), and one key is one window. Checked THROUGH the driver's own
    guard at the resolved window, not by arithmetic alone; the footprint is what
    every other configuration still gets (the geometry golden pins the 20
    non-MATH configurations)."""
    entry = sg.geometry_for("capability:gen_long_math")
    for cell, mode in (("none", "off"), ("none", "on"), (C.CELL_PROMPTED_SAFETY, "off"),
                       (C.CELL_PROMPTED_SAFETY, "on"), (C.CELL_PSRKL_SAFETY, "off"),
                       (C.CELL_PSRKL_SAFETY, "on"), (C.CELL_PSRKL_SAFETY, "per-task")):
        (cfg,) = L.plan_configurations(cells=[cell], modes=[mode], controls=["math500"])
        assert cfg["geometry_keys"] == ("capability:gen_long_math",)
        g1, rec1 = L.served_geometry(cfg, 1)
        _holds_math_at_cap(g1)                  # the P1: on HEAD this read 17,425 > 16,384
        g2, rec2 = L.served_geometry(cfg, 2)
        assert (g1.eval, g1.max_model_len, g1.max_num_seqs) == ("capability:gen_long_math", MATH500_WINDOW, 33)
        assert (g1.max_model_len, g1.max_num_seqs, g1.window_source, g1.seqs_source) == (
            g2.max_model_len, g2.max_num_seqs, g2.window_source, g2.seqs_source)
        assert g1.window_source == "table" and g1.seqs_source == L.MODEL_BUDGET_SOURCE
        assert not g1.measurement_pass and not g1.provisional and g1.is_identity and not g1.over_budget
        assert rec1 == rec2 and rec1["model_key"] == C.MODEL_SLUG
        # nothing is served at the footprint, so nothing is named as redundant
        assert L.measured_footprint_reserves(cfg, g1, 1) == []
        assert L.keys_beyond_footprint(sg, cfg["geometry_keys"], sg.PETRI_MAX_MODEL_LEN) == [
            ("capability:gen_long_math", entry.basis.max_prompt + C.MATH500_MAX_TOKENS_DEFAULT)]
        assert L.keys_beyond_footprint(sg, cfg["geometry_keys"], MATH500_WINDOW) == []
    # every model the launcher's registry serves: S from ITS budget at MATH's window
    for model, seqs in MATH500_SEQS_BY_MODEL.items():
        if model in C.MODELS:
            (cfg,) = L.plan_configurations(cells=["none"], modes=["off"], controls=["math500"], model=model)
            g1, rec1 = L.served_geometry(cfg, 1)
            assert (g1.max_model_len, g1.max_num_seqs, rec1["model_key"]) == (MATH500_WINDOW, seqs, model)
            _holds_math_at_cap(g1)
    # with no model: the entry's own S (the table's) — pass 2's answer, not the footprint's
    bare = L.configuration_geometry(cfg, 1, model_key=None)
    assert (bare.max_model_len, bare.max_num_seqs, bare.window_source, bare.seqs_source) == (
        MATH500_WINDOW, entry.max_num_seqs, "table", "table")
    # the pass-1 dry run prints the resolved window and sizes the standup and the
    # driver to it — never the footprint
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, cells="none", modes="off", controls="math500",
           dry_run=True)
    out = capsys.readouterr().out
    assert "geometry: capability:gen_long_math 17664 x 33 (window table; S model_budget" in out
    assert "--max-model-len 17664 --max-num-seqs 33 --model-key gemma4e4b" in out
    assert "--concurrency 33 " in out and f"--max-model-len {sg.PETRI_MAX_MODEL_LEN}" not in out
    assert sg.MEASUREMENT_SOURCE not in out
    # no p99-sized entry is ever "beyond" the footprint: its prompt tail is the
    # clamp's business, so the footprint serves it for measurement as designed
    others = tuple(k for k in sg.TINYCAP_KEYS if not k.startswith("capability:gen_long_math"))
    assert L.keys_beyond_footprint(sg, others, 1) == []
    # both windows come from the geometry module: no literal in the launcher
    src = LAUNCHER.read_text()
    assert "17664" not in src and "16384" not in src


def test_a_math500_smoke_is_served_at_its_own_window(L, capsys):
    # native-serving-default: the launcher's default is the native engine now; this test pins the LENS
    # launch (a PSR cell / the historical composition), so it states the path explicitly
    """The smoke is pass 1 by construction (``pass_no = 1``), so a MATH smoke
    (``--smoke --controls math500``) had no pass-2 door to the window MATH needs;
    it now resolves ``capability:gen_long_math`` like any MATH-only configuration
    — checked through the driver's guard at the resolved window — and its dry
    run prints that window. The default smoke (no MATH) is untouched: the
    footprint, for measurement."""
    (cfg,) = L.smoke_plan(controls=["math500"])
    assert cfg["geometry_keys"] == ("capability:gen_long_math",) and cfg["mode"] == C.THINKING_OFF
    assert L.plan_for(smoke=True, controls=["math500"]) == [cfg]
    g, rec = L.served_geometry(cfg, 1)
    _holds_math_at_cap(g)                       # the P1: on HEAD this read 17,425 > 16,384
    assert (g.eval, g.max_model_len, g.max_num_seqs, g.window_source) == (
        "capability:gen_long_math", MATH500_WINDOW, 33, "table")
    assert rec["model_key"] == C.MODEL_SLUG
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", smoke=True, controls="math500", dry_run=True)
    out = capsys.readouterr().out
    assert "geometry: capability:gen_long_math 17664 x 33 (window table; S model_budget" in out
    assert "--max-model-len 17664 --max-num-seqs 33 --model-key gemma4e4b" in out
    assert "--concurrency 33 " in out
    assert "--smoke --controls math500 --out-suffix smoke --limit 5 --inspect-limit 5" in out
    # the default smoke has no MATH: the footprint, as before
    (plain,) = L.smoke_plan()
    gp, _ = L.served_geometry(plain, 1)
    assert "capability:gen_long_math" not in plain["geometry_keys"]
    assert gp.max_model_len == sg.PETRI_MAX_MODEL_LEN and gp.measurement_pass


def test_math500_pooled_beside_other_keys_is_refused_by_name_before_any_allocation(L, monkeypatch, capsys):
    # native-serving-default: the launcher's default is the native engine now; this test pins the LENS
    # launch (a PSR cell / the historical composition), so it states the path explicitly
    """The other half. A configuration that pools MATH-500 BESIDE other
    sub-entries — ``--in-depth`` (either battery, either arm, the smoke's too)
    or an explicit ``--controls gsm8k,math500`` — cannot be served in pass 1:
    the footprint cannot hold MATH's cap, and one standup at MATH's window
    would serve the other keys' traffic at a window measured for something
    else with fewer sequences, in pass 1. It is refused BY
    NAME — MATH-500, the cap, the tokens needed, the footprint, MATH's own
    window, the keys it is pooled beside, and the launch that provides the
    window (``--controls math500`` alone) — from the one resolver every door
    uses, and in ``main`` BEFORE any Modal call: no preflight is asked for and
    no container is mapped. Pass 2 still refuses the same configurations as
    unpartitioned; a MATH-free launch passes the same door unchanged."""
    def refused(cfg):
        with pytest.raises(L.Pass1FootprintCannotHold) as e:
            L.served_geometry(cfg, 1)
        msg = str(e.value)
        for piece in (repr(cfg["name"]), f"footprint {sg.PETRI_MAX_MODEL_LEN:,} cannot hold",
                      "MATH-500 at cap 16,384", "needs 17,425 tokens", "prompt max 1,041",
                      "capability:gen_long_math (17,664)", "--controls math500",
                      "pass 1 and pass 2 alike", "fewer sequences", "D76"):
            assert piece in msg, (piece, msg)
        others = [k for k in cfg["geometry_keys"] if k != "capability:gen_long_math"]
        assert others and f"beside {others}" in msg
        return msg
    deep = L.plan_configurations(in_depth=True)
    assert len(deep) == 4
    for cfg in deep:
        assert "capability:gen_long_math" in cfg["geometry_keys"] and len(cfg["geometry_keys"]) > 1
        refused(cfg)
        with pytest.raises(L.Pass2Unpartitioned):
            L.served_geometry(cfg, 2)
    (smoke_deep,) = L.smoke_plan(in_depth=True)
    refused(smoke_deep)
    (mix,) = L.plan_configurations(cells=["none"], modes=["off"], controls=["gsm8k", "math500"])
    assert "beside ['capability:gen_long']" in refused(mix)
    assert L.controls_pooling_into("none", "off", mix["controls"], "capability:gen_long_math") == ["math500"]
    assert L.controls_pooling_into("none", "off", mix["controls"], "capability:gen_long") == ["gsm8k"]
    assert issubclass(L.Pass1FootprintCannotHold, ValueError)      # like its pass-2 sibling
    # `main`: refused before ANY Modal call — the dry run, the launch, the launch behind --pf
    monkeypatch.setattr(L, "preflight", SimpleNamespace(
        remote=lambda *a: pytest.fail("the preflight was asked for")))
    monkeypatch.setattr(L, "run_configuration", SimpleNamespace(
        spawn=lambda *a, **k: pytest.fail("a container was spawned")))
    for kw in (dict(pass1=True, in_depth=True, dry_run=True),
               dict(pass1=True, in_depth=True),
               dict(pass1=True, pf=True, in_depth=True),
               dict(smoke=True, pf=True, in_depth=True),
               dict(pass1=True, cells="none", modes="off", controls="gsm8k,math500"),
               dict(pass1=True, cells="none", modes="off", controls="gsm8k,math500", dry_run=True)):
        with pytest.raises(L.Pass1FootprintCannotHold, match="--controls math500"):
            L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", **kw)
        out = capsys.readouterr().out
        assert "[tinycap] argv:" in out and "launching" not in out and "==" not in out, kw
    # the same door still passes a MATH-free launch to the containers
    launched: list = []
    monkeypatch.setattr(L, "run_configuration", _spawn_only(_done, lambda c, k: launched.append(c)))
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, cells="none", modes="off", controls="gsm8k")
    assert [c["name"] for c in launched] == ["tinycap-e4b-base-think_off"]
    capsys.readouterr()
    # the gate is the one resolver, placed before the first Modal call
    src = LAUNCHER.read_text()
    body = src[src.index("def main("):]
    gate = body.index("        served_geometry(cfg, pass_no)")
    # bcb-bfcl-200-widen S1-f: the function is selected per row (configuration_function); the order holds
    assert gate < body.index("    if dry_run:") < body.index("preflight.remote(") < body.index("fan_out(configuration_function(row, sel_controls)")


# --------------------------------------------------------------------------- #
# MATH-500's LONG-instruction prompt classes get their own MEASURED window;    #
# the launcher selects the entry from the configuration's ACTUAL prompted      #
# class, checks the fit against THAT class's measured prompt max, and refuses  #
# an unmeasured class by name before any allocation                            #
# --------------------------------------------------------------------------- #

#: The measured MATH-500 prompt MAXIMA (tokens) of the two LONG-instruction
#: prompt classes — the 500 problems through inspect_evals' MATH solver chain
#: with the cell's system turn, under each model's PINNED tokenizer, per arm
#: (`scripts/tinycap_oos_prompt_budget.py --task math500 --system-prompt-cell
#: <cell>` for the ON render; the OFF render measured beside it with the same
#: counter; 2026-09-10). Copied from the sixteen measurements, not regenerated.
MATH500_LONGSI_PROMPT_MAXIMA = {
    "gemma4e4b": {"prompted__si_constitution": {"on": 6981, "off": 6979},
                  "prompted__si_irrelevant": {"on": 7514, "off": 7512}},
    "qwen35_4b": {"prompted__si_constitution": {"on": 6900, "off": 6902},
                  "prompted__si_irrelevant": {"on": 7620, "off": 7622}},
    "glm46vflash": {"prompted__si_constitution": {"on": 6729, "off": 6733},
                    "prompted__si_irrelevant": {"on": 7265, "off": 7269}},
    "nemotron3nano4b": {"prompted__si_constitution": {"on": 6887, "off": 6887},
                        "prompted__si_irrelevant": {"on": 7584, "off": 7584}},
}
#: The bare (no system turn) prompt maxima the same renders measured, per arm —
#: the ON column is the entry's own (`gen_long_math`: 954 / 937 / 856 / 935).
MATH500_BARE_PROMPT_MAX = {"gemma4e4b": {"on": 954, "off": 947}, "qwen35_4b": {"on": 937, "off": 939},
                           "glm46vflash": {"on": 856, "off": 860}, "nemotron3nano4b": {"on": 935, "off": 935}}
MATH500_LONGSI_KEY = "capability:gen_long_math_longsi"
#: 7,622 (the longest, qwen35_4b x si_irrelevant x OFF) + 16,384 = 24,006 -> 16-block
#: floor 24,016 -> rounded up to 256 = 24,064 (headroom 48; margin 58); S from each
#: model's budget: 589,824 / 720,896 / 786,432 / 786,432 // 24,064 (qwen35_4b's halved 360,448 lifted 2026-09-11)
MATH500_LONGSI_WINDOW = 24064
MATH500_LONGSI_SEQS_BY_MODEL = {"gemma4e4b": 24, "qwen35_4b": 30, "glm46vflash": 32, "nemotron3nano4b": 32}   # qwen35_4b: 741,888 // 24064 (the 2026-09-13 row)
LONG_SI_CELLS = ("prompted__si_constitution", "prompted__si_irrelevant")


def _pinned_tokenizer_or_skip(model: str):
    """The model's PINNED tokenizer from the local HF cache — gemma / qwen from
    the launcher's registry row, glm / nemotron (which the registry lacks) from
    `configs/two_si_<key>.yaml`, the campaign pins the geometry entry was
    measured under — through the budget script's own loader
    (`sisteer.models.load_tokenizer`: the same pad default, reasoning-template
    pin and template check). Skips with the reason when the cache lacks the
    pinned revision (the arithmetic and launcher assertions never skip)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from huggingface_hub import try_to_load_from_cache
    if model in C.MODELS:
        row = C.model_row(model)
        hf, rev = row.hf, row.rev
    else:
        pin = yaml.safe_load((REPO / f"configs/two_si_{model}.yaml").read_text())
        hf, rev = pin["model"], pin["model_revision"]
    assert hf == sg.MODEL_HF_IDS[model]
    if not isinstance(try_to_load_from_cache(hf, "tokenizer_config.json", revision=rev), str):
        pytest.skip(f"pinned {hf} tokenizer @ {rev[:8]} not in the local HF cache")
    from sisteer.models import load_tokenizer
    return load_tokenizer(hf, revision=rev)


def _prompt_tokens(tok, messages, arm: str) -> int:
    """The served render's token count under ``arm`` — the budget script's
    counter (`prompt_token_count`): the full chat render with the generation
    prompt appended and `contracts.chat_template_kwargs(arm)` passed."""
    kw = C.chat_template_kwargs(arm)[C.CHAT_TEMPLATE_KWARGS_KEY]
    out = tok.apply_chat_template([dict(m) for m in messages], add_generation_prompt=True, tokenize=True, **kw)
    return len(out["input_ids"] if hasattr(out, "keys") else out)


def _long_si_prompt_max(tok, cell: str, arm: str, model: str) -> int:
    """The class's prompt MAX re-derived from the REAL SI text under the pinned
    tokenizer: a system turn adds a CONSTANT to every item's render (measured:
    the prompted-minus-bare delta has min == max over the 500 problems for every
    model and both arms), so the class max is the bare max plus the delta the
    real text adds over any user turn."""
    from sisteer.eval.jailbreak import resolve_instruction
    text = resolve_instruction(cell[len(C.PROMPTED_CELL_PREFIX):]).text
    user = {"role": "user", "content": "Find the sum of all positive divisors of 36."}
    delta = (_prompt_tokens(tok, [{"role": "system", "content": text}, user], arm)
             - _prompt_tokens(tok, [user], arm))
    return MATH500_BARE_PROMPT_MAX[model][arm] + delta


def test_the_irrelevant_si_math500_configuration_was_accepted_at_a_window_its_prompts_overflow(L):
    """The reproduction (through this launcher and the
    pinned Gemma tokenizer): `--pass1 --cells prompted__si_irrelevant --modes
    off --controls math500` planned `capability:gen_long_math` and was ACCEPTED
    before allocation at 17,664 — the entry was sized from bare and
    `prompted__safety_refusal` prompts (max 1,041), but the irrelevant SI alone
    is 6,560 tokens on Gemma, so every request would have been window-clamped
    and the driver's guard would have refused the task AFTER generation. The
    pre-allocation check compared against the entry's own `max_prompt`, not the
    configuration's actual instruction class. Now the configuration resolves the
    LONG-instruction entry and the driver's own guard passes the real prompt
    max at the resolved window, in both passes."""
    from sisteer.capeval.tiers import inspect_served as isv
    tok = _pinned_tokenizer_or_skip("gemma4e4b")
    cap = C.MATH500_MAX_TOKENS_DEFAULT
    (cfg,) = L.plan_configurations(cells=["prompted__si_irrelevant"], modes=["off"], controls=["math500"])
    n = _long_si_prompt_max(tok, "prompted__si_irrelevant", "off", "gemma4e4b")
    assert n == 7512                                     # the real text: 947 bare + 6,565 for the SI
    for pass_no in (1, 2):
        g, rec = L.served_geometry(cfg, pass_no)         # accepted before allocation ...
        # ... and on HEAD the driver's guard then refused the task at that window:
        # "needs 7512 prompt + 16384 = 23896 tokens" against max_model_len=17664
        isv.assert_pass_fits_window(C.inspect_task("math"), [_math_at_cap(n)], max_tokens=cap,
                                    max_model_len=g.max_model_len)
        assert (g.eval, g.max_model_len, g.max_num_seqs, rec["model_key"]) == (
            MATH500_LONGSI_KEY, MATH500_LONGSI_WINDOW, 24, "gemma4e4b")
    assert cfg["geometry_keys"] == (MATH500_LONGSI_KEY,)
    # the window it used to be accepted at cannot hold the prompt: the guard says so by name
    with pytest.raises(isv.InspectServedError, match=f"needs {n} prompt \\+ {cap} = {n + cap}"):
        isv.assert_pass_fits_window(C.inspect_task("math"), [_math_at_cap(n)], max_tokens=cap,
                                    max_model_len=sg.geometry_for("capability:gen_long_math").max_model_len)


def _plan_or_none(L, cell, arm, model):
    """The one planned MATH-only configuration, or None where the launcher's
    registry cannot serve it (thinking-ON without a pinned parser)."""
    try:
        (cfg,) = L.plan_configurations(cells=[cell], modes=[arm], controls=["math500"], model=model)
    except L.ThinkingUnservable:
        assert arm == "on" and model != C.MODEL_SLUG
        return None
    return cfg


@pytest.mark.parametrize("model", sorted(MATH500_LONGSI_PROMPT_MAXIMA))
def test_every_long_instruction_math500_prompt_max_fits_only_the_long_si_window(L, model):
    """All four models x both long classes x both arms, from the REAL SI
    texts under the PINNED tokenizers (never the entry's own `max_prompt`):
    each class's prompt max, re-derived from the registered instruction text,
    equals the measurement; the window every prompted MATH configuration used
    to resolve (`capability:gen_long_math`, 17,664) refuses it through the
    driver's own guard; `capability:gen_long_math_longsi` (24,064 x S(model))
    holds it through the same guard — through the launcher's real planning and
    resolution in both passes where its registry serves the model / arm, and
    through the geometry module's own resolution for glm / nemotron and qwen
    thinking-ON (the cluster wrapper serves those)."""
    from sisteer.capeval.tiers import inspect_served as isv
    tok = _pinned_tokenizer_or_skip(model)
    cap = C.MATH500_MAX_TOKENS_DEFAULT
    math = C.inspect_task("math")
    short = sg.geometry_for("capability:gen_long_math")
    long_ = sg.geometry_for(MATH500_LONGSI_KEY)
    seqs = MATH500_LONGSI_SEQS_BY_MODEL[model]
    for cell in LONG_SI_CELLS:
        for arm in ("off", "on"):
            n = _long_si_prompt_max(tok, cell, arm, model)
            assert n == MATH500_LONGSI_PROMPT_MAXIMA[model][cell][arm], (model, cell, arm)
            assert n <= long_.basis.max_prompt and n + cap <= MATH500_LONGSI_WINDOW
            # the P1: the entry the launcher assigned every prompted MATH configuration cannot hold it
            with pytest.raises(isv.InspectServedError, match=f"needs {n} prompt \\+ {cap} = {n + cap}"):
                isv.assert_pass_fits_window(math, [_math_at_cap(n)], max_tokens=cap,
                                            max_model_len=short.max_model_len)
            # the long class's own window holds it — S from the model's budget
            r = sg.resolve_geometry(MATH500_LONGSI_KEY, strict_identity=True, model_key=model)
            assert (r.max_model_len, r.max_num_seqs, r.seqs_source) == (MATH500_LONGSI_WINDOW, seqs, L.MODEL_BUDGET_SOURCE)
            isv.assert_pass_fits_window(math, [_math_at_cap(n)], max_tokens=cap, max_model_len=r.max_model_len)
            if model not in C.MODELS:
                continue
            cfg = _plan_or_none(L, cell, arm, model)
            if cfg is None:
                continue
            assert cfg["geometry_keys"] == (MATH500_LONGSI_KEY,) and cfg["prompted"]
            assert L.math500_prompt_class_of(cfg) == C.MATH500_PROMPT_CLASS_LONG_SI
            for pass_no in (1, 2):
                g, rec = L.served_geometry(cfg, pass_no)
                assert (g.eval, g.max_model_len, g.max_num_seqs, g.window_source, rec["model_key"]) == (
                    MATH500_LONGSI_KEY, MATH500_LONGSI_WINDOW, seqs, "table", model)
                isv.assert_pass_fits_window(math, [_math_at_cap(n)], max_tokens=cap, max_model_len=g.max_model_len)
                assert not g.measurement_pass and g.is_identity and not g.over_budget
            assert L.measured_footprint_reserves(cfg, g, 1) == []
    assert long_.basis.max_prompt == max(
        n for by_cell in MATH500_LONGSI_PROMPT_MAXIMA.values() for by_arm in by_cell.values() for n in by_arm.values()) == 7622


def test_the_launcher_selects_math500s_entry_from_the_cells_actual_prompt_class(L):
    """The entry is chosen by the system turn the CELL carries, never by
    the prompted bit alone — the base cell, every object cell (whatever SI it
    was fitted on) and the ~90-token safety SI keep `capability:gen_long_math`;
    the constitution / irrelevant SIs take `capability:gen_long_math_longsi`;
    in every arm. Only MATH-500 keys on the class; the gen_long entries key on
    the arm, whatever the SI. The pre-allocation check reads the SAME class, and
    refuses by name a configuration whose keys disagree with its class (the
    earlier selection, simulated) — in both passes."""
    short_cells = ("none", C.CELL_PROMPTED_SAFETY, C.CELL_PSRKL_SAFETY, "si_constitution__lora_r4",
                   "si_irrelevant__lora_r4")
    for mode in ("off", "on", "per-task"):
        for cell in short_cells:
            assert L.geometry_keys_for(cell, mode, ["math500"]) == ("capability:gen_long_math",), (cell, mode)
        for cell in LONG_SI_CELLS:
            assert L.geometry_keys_for(cell, mode, ["math500"]) == (MATH500_LONGSI_KEY,), (cell, mode)
    for cell in ("prompted__si_irrelevant", "prompted__si_constitution", C.CELL_PROMPTED_SAFETY):
        assert L.geometry_keys_for(cell, "off", ["gsm8k"]) == ("capability:gen_long",)
        assert L.geometry_keys_for(cell, "on", ["gsm8k"]) == ("capability:gen_long_thinking",)
    # the class, by the cell, on the planned configuration; None without MATH-500
    for cell, cls in (("none", "short"), (C.CELL_PROMPTED_SAFETY, "short"), ("si_irrelevant__lora_r4", "short"),
                      ("prompted__si_constitution", "long_si"), ("prompted__si_irrelevant", "long_si")):
        (cfg,) = L.plan_configurations(cells=[cell], modes=["off"], controls=["math500"])
        assert L.math500_prompt_class_of(cfg) == cls and cfg["geometry_keys"] == (C.math500_geometry_key(cls),)
        (plain,) = L.plan_configurations(cells=[cell], modes=["off"], controls=["gsm8k"])
        assert L.math500_prompt_class_of(plain) is None
        g, _ = L.served_geometry(plain, 1)
        assert L.assert_math500_class_fits(plain, g, sg) is g            # nothing to check: returned as is
    # the check reads the class's OWN measured max: a long-SI configuration whose keys
    # name the short entry (the earlier selection) is refused by name, both passes
    (cfg,) = L.plan_configurations(cells=["prompted__si_irrelevant"], modes=["off"], controls=["math500"])
    stale = dict(cfg, geometry_keys=("capability:gen_long_math",))
    for pass_no in (1, 2):
        with pytest.raises(L.Math500WindowCannotHoldClass) as e:
            L.served_geometry(stale, pass_no)
        msg = str(e.value)
        for piece in (repr(cfg["name"]), "'prompted__si_irrelevant'", "'long_si'", "prompt max 7,622",
                      f"{MATH500_LONGSI_KEY} is 24,064", "cap 16,384", "needs 24,006 tokens",
                      "resolved capability:gen_long_math at max_model_len 17,664", "assert_pass_fits_window", "D80"):
            assert piece in msg, (piece, msg)
    assert issubclass(L.Math500WindowCannotHoldClass, ValueError)
    # the well-formed configuration passes the same check in both passes
    for pass_no in (1, 2):
        g, _ = L.served_geometry(cfg, pass_no)
        assert L.assert_math500_class_fits(cfg, g, sg) is g and g.max_model_len == MATH500_LONGSI_WINDOW


def test_a_math500_configuration_under_an_unmeasured_si_is_refused_by_name_before_allocation(L, monkeypatch):
    # native-serving-default: the launcher's default is the native engine now; this test pins the LENS
    # launch (a PSR cell / the historical composition), so it states the path explicitly
    """A prompted SI in NEITHER measured class has no window MATH-500 can be
    served at, so a MATH configuration under it is refused by name at PLAN time
    — the cell, the cap, the measured SIs and the remedy (measure it with the
    budget script, register the class and the entry) — with no Modal call. The
    served vocabulary launches exactly the three measured SIs today, so the
    case is produced on a REAL launchable cell by taking its SI out of the class
    map in-test. The same cell's MATH-free battery plans as before (the class is
    MATH-500's business only)."""
    monkeypatch.setattr(C, "MATH500_PROMPT_CLASS_OF_SI",
                        {si: cls for si, cls in C.MATH500_PROMPT_CLASS_OF_SI.items() if si != "si_irrelevant"})
    with pytest.raises(C.Math500PromptClassUnmeasured) as e:
        L.plan_configurations(cells=["prompted__si_irrelevant"], modes=["off"], controls=["math500"])
    msg = str(e.value)
    for piece in ("prompted__si_irrelevant", "NO measured prompt class", "16,384", "'safety_refusal': 'short'",
                  "'si_constitution': 'long_si'", "--system-prompt-cell prompted__si_irrelevant",
                  "MATH500_PROMPT_CLASS_OF_SI", "serving/geometry.py", "D80"):
        assert piece in msg, (piece, msg)
    assert "'si_irrelevant'" not in msg.split("measured SIs")[1].split(")")[0]      # not among the measured
    with pytest.raises(C.Math500PromptClassUnmeasured):
        L.geometry_keys_for("prompted__si_irrelevant", "on", ["gsm8k", "math500"])
    # the standard battery under the same SI: no MATH, no class, planned as before
    cfgs = L.plan_configurations(cells=["prompted__si_irrelevant"], modes=["off"])
    assert cfgs and all(c["geometry_keys"] == ("capability:gen_long",) for c in cfgs)
    assert not any(k.startswith("capability:gen_long_math") for c in cfgs for k in c["geometry_keys"])
    assert all(L.math500_prompt_class_of(c) is None for c in cfgs)
    # the constitution SI, still in the map, plans its MATH configuration as before
    (ok,) = L.plan_configurations(cells=["prompted__si_constitution"], modes=["off"], controls=["math500"])
    assert ok["geometry_keys"] == (MATH500_LONGSI_KEY,)
    # main: refused before any Modal call (dry run, launch, launch behind --pf)
    monkeypatch.setattr(L, "preflight", SimpleNamespace(remote=lambda *a: pytest.fail("the preflight was asked for")))
    monkeypatch.setattr(L, "run_configuration", SimpleNamespace(spawn=lambda *a, **k: pytest.fail("a container was spawned")))
    for kw in (dict(dry_run=True), dict(), dict(pf=True)):
        with pytest.raises(C.Math500PromptClassUnmeasured, match="prompted__si_irrelevant"):
            L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, cells="prompted__si_irrelevant", modes="off",
                   controls="math500", **kw)


def test_a_long_instruction_math500_configuration_pooled_beside_other_keys_names_its_own_window(L):
    """A long-SI cell's ``--in-depth`` (or ``--controls gsm8k,math500``)
    is refused before allocation exactly as the short cells' are — and the
    refusal names the LONG class's numbers and entry (7,622 + 16,384 = 24,006;
    `capability:gen_long_math_longsi` 24,064), never the 1,041 / 17,664 of the
    entry it no longer pools into. Pass 2 still refuses it as unpartitioned."""
    for cell in LONG_SI_CELLS:
        (cfg,) = L.plan_configurations(cells=[cell], modes=["off"], in_depth=True)
        assert MATH500_LONGSI_KEY in cfg["geometry_keys"] and "capability:gen_long_math" not in cfg["geometry_keys"]
        with pytest.raises(L.Pass1FootprintCannotHold) as e:
            L.served_geometry(cfg, 1)
        msg = str(e.value)
        for piece in ("MATH-500 at cap 16,384", "needs 24,006 tokens", "prompt max 7,622",
                      f"{MATH500_LONGSI_KEY} (24,064)", "--controls math500", "fewer sequences"):
            assert piece in msg, (piece, msg)
        assert "17,664" not in msg and "1,041" not in msg
        with pytest.raises(L.Pass2Unpartitioned):
            L.served_geometry(cfg, 2)
    (mix,) = L.plan_configurations(cells=["prompted__si_constitution"], modes=["on"], controls=["gsm8k", "math500"])
    assert L.controls_pooling_into("prompted__si_constitution", "on", mix["controls"], MATH500_LONGSI_KEY) == ["math500"]
    with pytest.raises(L.Pass1FootprintCannotHold, match="beside \\['capability:gen_long_thinking'\\]"):
        L.served_geometry(mix, 1)


def test_the_dry_run_of_a_long_instruction_math500_configuration_prints_the_long_si_window(L, capsys):
    # native-serving-default: the launcher's default is the native engine now; this test pins the LENS
    # launch (a PSR cell / the historical composition), so it states the path explicitly
    """The pass-1 dry run of a long-SI MATH configuration sizes the standup
    and the driver to the long class's window (24,064 x 24 for gemma4e4b), the
    safety-SI configuration keeps the short one (17,664 x 33), and neither
    window nor prompt max is a literal in the launcher."""
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, cells="prompted__si_irrelevant", modes="off",
           controls="math500", dry_run=True)
    out = capsys.readouterr().out
    assert f"geometry: {MATH500_LONGSI_KEY} 24064 x 24 (window table; S model_budget" in out
    assert "--max-model-len 24064 --max-num-seqs 24 --model-key gemma4e4b" in out and "--concurrency 24 " in out
    assert "17664" not in out
    L.main(app_name=L.APP_NAME, sha="abc1234", serving_path="lens", pass1=True, cells=C.CELL_PROMPTED_SAFETY, modes="off",
           controls="math500", dry_run=True)
    out = capsys.readouterr().out
    assert "geometry: capability:gen_long_math 17664 x 33 (window table; S model_budget" in out
    assert "24064" not in out
    src = LAUNCHER.read_text()
    assert "24064" not in src and "7622" not in src and "17664" not in src


def test_a_math500_configuration_expects_the_first_pass_cap_and_refuses_a_store_at_another(L, tmp_path):
    """The launcher half. A configuration that files MATH-500
    derives `math500_max_tokens` PURELY (`contracts.math500_max_tokens_for`:
    16,384) and carries it as its one optional key; a non-MATH configuration
    carries none — its expectation is byte-identical to before. The store-trust
    check then holds a MATH store to it: a store recording another cap (a
    4,096 pre-change instrument) is refused naming the key and both numbers;
    a store from before the key existed (the pre-change two-pass or
    explicit-zero instrument) lacks it and is refused naming the key; a store
    recording this run's cap resumes. In-support, non-MATH stores are checked
    exactly as before."""
    math = _expected(L, controls=["math500"])
    assert math.files_math500 and math.math500_max_tokens == 16384 == C.MATH500_MAX_TOKENS_DEFAULT
    assert math.optional_identity() == {C.IDENTITY_KEY_MATH500_MAX_TOKENS: 16384}
    assert math.as_dict()[C.IDENTITY_KEY_MATH500_MAX_TOKENS] == 16384
    assert set(math.as_dict()) - set(_expected(L).as_dict()) == {C.IDENTITY_KEY_MATH500_MAX_TOKENS}
    plain = _expected(L)
    assert not plain.files_math500 and plain.math500_max_tokens is None
    assert plain.optional_identity() == {} and C.IDENTITY_KEY_MATH500_MAX_TOKENS not in plain.as_dict()
    # a MATH store at this run's cap resumes; the helper built exactly the driver's key set
    ok = _store(tmp_path / "ok", math, tasks=["math500"])
    manifest = json.loads((ok / "manifest.json").read_text())
    assert set(manifest["tinycap"]) == set(C.tinycap_identity_fields(files_math500=True))
    assert manifest["tinycap"][C.IDENTITY_KEY_MATH500_MAX_TOKENS] == 16384
    assert L.untrusted_store_reason(ok, expected=math) is None
    # ... at another cap (the 8,192 default that landed, or the 4,096 before it):
    # another key by construction, and refused by name
    _m16k, chash16k = _manifest(math, tasks=["math500"])
    _m8192, chash8192 = _manifest(math, tasks=["math500"], math500_max_tokens=8192)
    _m4096, chash4096 = _manifest(math, tasks=["math500"], math500_max_tokens=4096)
    assert len({chash16k, chash8192, chash4096}) == 3
    for other_cap in (8192, 4096):
        why = L.untrusted_store_reason(
            _store(tmp_path / f"cap{other_cap}", math, tasks=["math500"], math500_max_tokens=other_cap),
            expected=math)
        assert f"tinycap.{C.IDENTITY_KEY_MATH500_MAX_TOKENS}: the store records {other_cap} but this run hashes 16384" in why
    assert "another first-pass cap" in why and "D33" in why
    # ... from before the key existed (self-consistent, the cap simply unhashed): refused by name
    why = L.untrusted_store_reason(
        _store(tmp_path / "pre", math, tasks=["math500"], drop_identity_field=C.IDENTITY_KEY_MATH500_MAX_TOKENS),
        expected=math)
    assert f"missing [{C.IDENTITY_KEY_MATH500_MAX_TOKENS!r}]" in why
    assert "MATH-500 store from before the cap was hashed" in why and "is never resumed" in why and "(D33)" in why
    # a non-MATH store: no key, checked as before
    assert L.untrusted_store_reason(_store(tmp_path / "plain", plain), expected=plain) is None
    assert C.IDENTITY_KEY_MATH500_MAX_TOKENS not in json.loads(
        ((tmp_path / "plain" / "cells").iterdir().__next__() / "manifest.json").read_text())["tinycap"]
    # across configurations the key set follows the STORE's tasks, so completion
    # still lends both ways: a standard store (no MATH, no key) is trusted by the
    # in-depth run for the controls it requested (math500 reads "never
    # requested" there, not "untrusted"), and a MATH store is trusted by a
    # standard run, its cap not a fact about that run
    assert L.untrusted_store_reason(_store(tmp_path / "lend", math, tasks=C.expand_control(C.CONTROL_GSM8K)),
                                    expected=math) is None
    assert L.remaining_controls(tmp_path / "lend", ["math500"], math) == ["math500"]
    assert L.untrusted_store_reason(_store(tmp_path / "m2", plain, tasks=["math500"], math500_max_tokens=16384),
                                    expected=plain) is None
    # ... but a store whose tasks file MATH must carry the key, and one that does
    # not must not: the key set is a fact about the store, never negotiable
    why = L.untrusted_store_reason(
        _store(tmp_path / "pre2", plain, tasks=["math500"], drop_identity_field=C.IDENTITY_KEY_MATH500_MAX_TOKENS),
        expected=plain)
    assert f"missing [{C.IDENTITY_KEY_MATH500_MAX_TOKENS!r}]" in why and "never resumed" in why

def test_the_loader_subprocess_runs_with_the_staged_tree_first_and_from_its_root(L, monkeypatch):
    """The stage of 0a7eb982 died on the staged-tree assertion inside the loader
    snippet: the client subprocess imported ``sisteer`` from the image's mounted
    ``/root/sisteer`` because the container's cwd sat ahead of PYTHONPATH on
    ``sys.path`` for ``-c``. The loaders (online at stage, offline at preflight)
    now run with the staged tree's ``src`` FIRST on PYTHONPATH, ``-P`` and
    ``cwd=<tree>`` — the driver's own arrangement."""
    seen = {}

    def _run(argv, **kw):
        seen.update(argv=list(argv), **kw)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"x": {"ok": True}, L.DATA_DIR_ALIASES_KEY: []}), stderr="")
    monkeypatch.setattr(L.subprocess, "run", _run)
    for offline in (False, True):
        out, aliases = L.run_inspect_loaders("/vol/tinycap/repos/sisteer-abc", offline=offline)
        assert out == {"x": {"ok": True}} and aliases == []
        assert seen["cwd"] == "/vol/tinycap/repos/sisteer-abc"
        assert seen["env"]["PYTHONPATH"].split(":") == ["/vol/tinycap/repos/sisteer-abc/src",
                                                          "/vol/tinycap/repos/sisteer-abc/scripts"]
        assert seen["argv"][1:3] == ["-P", "-c"] and seen["argv"][4] == "/vol/tinycap/repos/sisteer-abc"
        assert ("HF_HUB_OFFLINE" in seen["env"]) is offline
    # the snippet's own guard stays: the task dir it imports from must be the staged tree's
    assert "assert Path(I.INSPECT_TASKS_DIR).resolve().is_relative_to(tree.resolve())" in L.INSPECT_LOADER_CODE
    assert 'go("lane.staged.rung_probe"' in L.INSPECT_LOADER_CODE


def test_the_launcher_ships_its_sibling_tinycap_images_as_a_copy_layer_under_the_env_layer(L):
    """tinycap-full22 gate 3, ported: the launcher imports ``tinycap_images`` at load and
    ``modal run`` mounts nothing but the entrypoint — the first stage container at f1eeb951 died with
    ``ModuleNotFoundError: No module named 'tinycap_images'`` and crash-looped. The file rides in the
    image as a COPY layer (a build layer): ``tinycap_images.build_image_of`` peels the trailing mount
    layer(s) to derive ``image_nemotron`` and re-applies only the ``sisteer`` source mount, so a plain
    ``add_local_file`` mount would vanish from every Nemotron container. It sits UNDER the ``.env``
    build layer, because a copy layer's own deps carry its context Mount and it would be peeled too as
    the top build layer. The remote path is where the entrypoint itself lands (``/root``), which the
    launcher puts on ``sys.path`` before the import."""
    assert L.TINYCAP_IMAGES_REMOTE == "/root/tinycap_images.py"
    src = LAUNCHER.read_text()
    i_copy = src.index(
        'add_local_file(str(Path(tinycap_images.__file__).resolve()), TINYCAP_IMAGES_REMOTE, copy=True)')
    # the layer above it is the lane's NLTK bake (the Harbor venv layers are gone with Terminal-Bench)
    i_bake = src.index(".run_commands(*ifeval_nltk_bake_commands())")
    i_env = src.index('.env({"HF_HOME"')
    i_mount = src.index('.add_local_python_source("sisteer")')
    assert i_bake < i_copy < i_env < i_mount
    # tinycap-shared-standup §1.5 (module D): the compile-cache sibling rides the same copy layer beside tinycap_images
    assert L.IMAGE_LOCAL_FILES == {L.TINYCAP_IMAGES_REMOTE: str(Path(L.tinycap_images.__file__).resolve()),
                                   L.TINYCAP_COMPILE_CACHE_REMOTE: str(Path(L.CC.__file__).resolve())}
    assert src.index("import tinycap_images") < i_copy                # the module the path is read from


def test_engine_core_tail_keeps_the_engine_processs_own_lines_and_the_probe_persists_its_logs(L):
    """Two live Nemotron backend probes ended STANDUP_FAILED with a
    3,000-char vLLM tail holding only the API server's "Engine core initialization failed. See root
    cause above" trace — the EngineCore's own lines (the root cause) were above the cut and the probe
    never persisted the full log (the srv dir dies with the container). The REC now carries the engine
    process's lines and the probe writes the standup + vLLM logs beside the attestation on every path."""
    text = "\n".join([
        "(APIServer pid=32) INFO args",
        "(EngineCore pid=77) INFO Initializing a V1 LLM engine",
        "(EngineCore pid=77) ERROR Traceback (most recent call last):",
        "(EngineCore pid=77) ERROR   something failed in the worker",
        "(APIServer pid=32) RuntimeError: Engine core initialization failed. See root cause above.",
    ])
    tail = L.engine_core_tail(text)
    assert tail.splitlines() == [ln for ln in text.splitlines() if "EngineCore" in ln]
    assert L.engine_core_tail(text, n_lines=1) == "(EngineCore pid=77) ERROR   something failed in the worker"
    assert L.engine_core_tail("") is None and L.engine_core_tail(None) is None
    assert L.engine_core_tail("(APIServer pid=1) only") is None
    src = LAUNCHER.read_text()
    body = src[src.index("def _nemotron_probe("):src.index("def nemotron_backend_probe(")]
    # failed standup, unattested, served-geometry mismatch (the peer-at-served-S refusal, the IFEval rung wave), served
    # 4 + the two native-serving-default refusals (the served S off the pre-computed key S; the engine)
    assert body.count("_persist_logs(srv, out_dir)") == 6
    assert 'engine_tail=engine_core_tail(vllm_text)' in body and 'logs_dir=str(out_dir)' in body
    assert 'vllm_tail=_tail(srv / "vllm_server.log", 12_000)' in body
    assert L.NEMOTRON_FUNCTION_CPU >= 4
    # every Nemotron-image function requests the CPU share explicitly: run_configuration_nemotron,
    # probe_rung_nemotron (bcb-bfcl-200-widen's ladder probe on the Nemotron image) and nemotron_backend_probe
    # + run_configuration_nemotron_full (capeval-full-tasks: the full legs' function variant on the Nemotron image)
    assert src.count("cpu=NEMOTRON_FUNCTION_CPU)") == 4


def test_the_ifeval_probe_flags_carry_dummy_values_and_the_real_driver_parser_reaches_the_sentinel(L):
    """Every VALUED driver flag an IFEval
    configuration composes is probed WITH a dummy value; probed bare, argparse stopped at
    "expected one argument" before the unrecognized-arguments line the probe reads, and every
    full22 preflight failed driver_accepts_tinycap_flags. The second half is argparse's own verdict
    on THIS tree's driver: the probe argv exits 2 naming exactly the sentinel as unrecognized."""
    import os
    import subprocess
    import sys
    flags = (C.FLAG_TINYCAP_IDENTITY_VERSION, C.FLAG_MAX_TOKENS_BY_TASK, C.FLAG_SAMPLING, C.FLAG_OFF_CAP_BY_TASK,
             C.FLAG_ON_CAP_BY_TASK, C.FLAG_ITEM_SUBSET, C.FLAG_BANK_VIEW, C.FLAG_ATTEMPT_ID)
    argv = L.driver_probe_argv(str(REPO), python=sys.executable, extra_flags=flags)
    for flag in flags:
        assert flag in L.DRIVER_PROBE_VALUES, flag
        assert argv[argv.index(flag) + 1] == L.DRIVER_PROBE_VALUES[flag]
    assert L.DRIVER_PROBE_VALUES[C.FLAG_SAMPLING] == C.SAMPLING_VENDOR_PRESET
    assert L.DRIVER_PROBE_VALUES[C.FLAG_OFF_CAP_BY_TASK] == "ifeval=4096"
    assert argv[-1] == L.DRIVER_PROBE_SENTINEL
    env = dict(os.environ, PYTHONPATH=f"{REPO}/src:{REPO}/scripts")
    r = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=900)
    assert r.returncode == 2, r.stderr[-1200:]
    assert L.unrecognized_flags(r.stderr) == [L.DRIVER_PROBE_SENTINEL], r.stderr[-1200:]


def test_ladder_admission_report_expects_the_runs_sampling_mode():
    """`ladder_admission_report` derives the expected identity of a
    rung-2 configuration with the RUN's sampling mode (`sampling_mode_of(lcfg)`), like the
    worker and the driver argv do — the default `card` reached `contracts.sampling_for`, which
    refuses a model without a verified card row (nemotron), so a `--sampling protocol` scoring
    launch was refused by the preflight report before any GPU (ladder_report §4c). Pinned on
    the launcher's source text: the report's call spells `sampling_mode=sampling_mode_of(lcfg)`,
    and no `expected_identity(` call anywhere in the launcher falls back to the default mode."""
    import re
    # comment tails stripped first: a call may carry a `# ... (absent = card)` line whose `)` would end the match early
    whole = re.sub(r"#[^\n]*", "", LAUNCHER.read_text(encoding="utf-8"))
    m = re.search(r"\ndef ladder_admission_report\(.*?(?=\ndef )", whole, flags=re.S)
    assert m, "ladder_admission_report is gone — re-pin this test"
    body = m.group(0)
    calls = re.findall(r"expected_identity\((.*?)\)\n", body, flags=re.S)
    assert calls, "ladder_admission_report no longer calls expected_identity — re-pin this test"
    for args in calls:
        assert "sampling_mode=sampling_mode_of(lcfg)" in args, args
    for m2 in re.finditer(r"(?<!def )\bexpected_identity\((.*?)\)\n", whole, flags=re.S):
        assert "sampling_mode=" in m2.group(1), m2.group(0)[:200]

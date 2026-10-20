"""Under ``SISTEER_SECRETS_MODE=wif`` no grader SETS or EXPORTS an ``ANTHROPIC_*`` variable.

Four graders carried their own copy of a process-environment pin -- ``os.environ["ANTHROPIC_BASE_URL"] = <the API host>``
plus a pop of the shell's key and bearer -- two of them at IMPORT (``grade_full22_native``, ``grade_alphasteer_f22``) and
two in a ``_pin_env()`` their main calls (``static_grading``, ``grade_static_sampled``). Right under ``op`` / ``env``, whose
keyed SDK clients would otherwise be tunnelled through an inherited local gateway; under ``wif`` it broke the mode's
contract: nothing reads, sets or exports ``ANTHROPIC_*``, and the federated client pins the
host in-process. The pin now lives ONCE in the vault (``sisteer.vault.pin_process_environment``): byte-identical under
``op`` / ``env``, and under ``wif`` nothing set and the inherited names dropped by NAME, never a value read.

Round 10: the five names round 9 dropped were a list, not the rule. The SDK's constructor reads
``ANTHROPIC_CUSTOM_HEADERS`` and ``ANTHROPIC_WEBHOOK_SIGNING_KEY`` whatever credential it is given, parses the former into EVERY
request's headers, and a static ``X-Api-Key`` / ``Authorization`` there makes its ``AccessTokenAuth`` skip the token exchange: a
federated client built under a shell carrying one sent the stale header and never invoked its provider. The rule is now the PREFIX
(``vault.WIF_SCRUB_PREFIX``, ``scrub_inherited_anthropic_environment``: every ``ANTHROPIC_*`` name, by name), run by the pin's wif
branch AND by every federated constructor itself (``tests/test_wif_inherited_static_auth.py`` measures the requests); the shell
spelling is bash's name-only prefix expansion (``vault.WIF_SHELL_SCRUB``). The sweeps below hold the class across the tree: every
Python ``os.environ`` setter of an ``ANTHROPIC_*`` name outside the vault fails here, every shell ``export`` / assignment of one
must be on the mode-guarded allowlist, and every shell scrub under wif is the vault's one spelling AND, run under a synthetic shell
carrying every known inherited name, leaves none."""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import warnings
from collections.abc import MutableMapping
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase" / "full22" / "sampled"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sisteer import vault  # noqa: E402

GATEWAY = "http://127.0.0.1:1/mock-gateway"
PIN = "https://api.anthropic.com"
SCOPES = ("src/", "scripts/", "reproduce/", "corpus_generator/")
EXCLUDED = ("legacy/",)
#: the one module allowed to set an ANTHROPIC_* variable (the pin's op / env branch)
ENV_SETTER_ALLOWED = frozenset({"src/sisteer/vault.py"})
#: the shell lines that assign or export an ANTHROPIC_* variable, each on an op / env branch or behind the mode guard
#: (file, a substring the line must carry -- the guard or the branch keyword is part of it)
SHELL_SETTERS_ALLOWED = (
    ("reproduce/lib/secrets.sh", 'echo "SISTEER_SECRETS_MODE=env"; echo "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:?'),      # env mode's unit env file
    ("scripts/sdfbase/full22/judge/launch_judge.sh", 'else export ANTHROPIC_API_KEY="$("${SISTEER_SECRET_CMD:?'),        # the op branch; wif unsets
    ("scripts/sdfbase/full22/petri_drive_full22.sh", '[ "${SISTEER_SECRETS_MODE:-op}" = wif ] || export ANTHROPIC_BASE_URL='),  # guarded (round 8)
    ("scripts/sdfbase/full22/petri_drive_full22.sh", 'else ANTHROPIC_API_KEY="$("$SECRET_CMD" "$F22_AUDITOR_ORG")"'),    # the op branch
)
SHELL_SETTER_RE = re.compile(r"(^|[^A-Za-z0-9_])(export[ \t]+)?ANTHROPIC_[A-Z_]+=")


# ------------------------------------------------------------------------------------------------------ the vault's pin


class _Recorder(MutableMapping):
    """A stand-in for ``os.environ`` that records every NAME whose VALUE is read (``__getitem__``), so the wif branch is
    proven to iterate names and delete, never read."""

    def __init__(self, **items):
        self._d = dict(items)
        self.read: list = []

    def __getitem__(self, k):
        self.read.append(k)
        return self._d[k]

    def __setitem__(self, k, v):
        self._d[k] = v

    def __delitem__(self, k):
        del self._d[k]

    def __iter__(self):
        return iter(list(self._d))

    def __len__(self):
        return len(self._d)


#: the SDK's unconditional reads and discovery selectors a shell may carry (round 10; every value synthetic, no credential shape)
ROUND10_NAMES = {"ANTHROPIC_CUSTOM_HEADERS": "X-Api-Key: " + "-".join(("static", "placeholder", "never", "sent")),
                 "ANTHROPIC_WEBHOOK_SIGNING_KEY": "-".join(("synthetic", "webhook", "placeholder")),
                 "ANTHROPIC_PROFILE": "synthetic-profile", "ANTHROPIC_CONFIG_DIR": "/nonexistent/anthropic-config"}


def _inherited(**extra):
    """What a Claude Code / vault shell hands a grader: a gateway, an OAuth bearer, env mode's one key, the per-org
    keys, the SDK's custom-headers / webhook / profile / config-dir variables (every value synthetic) and an
    unrelated variable that must survive."""
    extra = dict(ROUND10_NAMES, **extra)
    return dict(ANTHROPIC_BASE_URL=GATEWAY, ANTHROPIC_AUTH_TOKEN="synthetic-bearer-not-a-token",
                ANTHROPIC_API_KEY="synthetic-one-key-not-a-key", ANTHROPIC_API_KEY_LOWPRIO="synthetic-per-org-not-a-key",
                ANTHROPIC_API_KEY_BATCH="synthetic-per-org-not-a-key", HF_HUB_OFFLINE="1", **extra)


def test_under_wif_the_pin_sets_no_anthropic_variable_and_drops_the_inherited_names_without_reading_a_value(monkeypatch):
    monkeypatch.setenv(vault.SECRETS_MODE_ENV, "wif")
    env = _Recorder(**_inherited())
    dropped = vault.pin_process_environment(env)
    # EVERY name with the prefix (round 10), the known inherited set among them -- the five of round 9 and the SDK's four
    assert set(dropped) == {k for k in _inherited() if k.startswith(vault.WIF_SCRUB_PREFIX)} == set(vault.WIF_INHERITED_ENV_NAMES)
    assert {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY_LOWPRIO", "ANTHROPIC_API_KEY_BATCH",
            "ANTHROPIC_CUSTOM_HEADERS", "ANTHROPIC_WEBHOOK_SIGNING_KEY"} <= set(dropped)
    assert dropped == tuple(sorted(dropped))                                         # names, sorted: a record, never a value
    assert not [k for k in env if k.startswith("ANTHROPIC_")], dict(env)          # nothing set, nothing exported
    # a name no list ever knew is dropped too: the rule is the prefix
    env2 = _Recorder(ANTHROPIC_FUTURE_SDK_SWITCH="x", **_inherited())
    assert "ANTHROPIC_FUTURE_SDK_SWITCH" in vault.pin_process_environment(env2) and not [k for k in env2 if k.startswith("ANTHROPIC_")]
    assert not [k for k in env2.read if k.startswith("ANTHROPIC_")], env2.read
    assert env["HF_HUB_OFFLINE"] == "1"                                              # an unrelated name survives
    assert not [k for k in env.read if k.startswith("ANTHROPIC_")], env.read         # dropped by NAME: no value was read
    # idempotent, and nothing to drop on a clean environment
    assert vault.pin_process_environment(env) == () and not [k for k in env if k.startswith("ANTHROPIC_")]


def test_under_wif_the_pin_on_the_real_environment_leaves_no_anthropic_name(monkeypatch):
    monkeypatch.setenv(vault.SECRETS_MODE_ENV, "wif")
    for k, v in _inherited().items():
        monkeypatch.setenv(k, v)
    dropped = vault.pin_process_environment()
    assert set(dropped) >= set(vault.WIF_INHERITED_ENV_NAMES)                      # >=: the host's own ANTHROPIC_* names go too
    assert not [k for k in os.environ if k.startswith(vault.WIF_SCRUB_PREFIX)]
    assert "ANTHROPIC_BASE_URL" not in os.environ                                   # the pin is in-process (anthropic_client_wif), never exported
    assert vault.scrub_inherited_anthropic_environment() == ()                      # the scrub alone: nothing left to drop, idempotent


@pytest.mark.parametrize("mode", ["op", "env"])
def test_under_op_and_env_the_pin_is_byte_identical_to_the_graders_old_block(mode, monkeypatch):
    """The block the four graders carried: the API host exported under ANTHROPIC_BASE_URL, the shell's High-Prio key
    (ANTHROPIC_API_KEY) and OAuth bearer popped, and every other name -- the per-org keys included -- left as it
    stands. Byte-identical under env too, pop included: on the committed tree the four graders popped env mode's one
    key before building their client, so under env the vault's reader refuses by name -- a PRE-EXISTING condition of
    the base named in the sweep doc (round 9), not this lane's to change under a byte-identical rule."""
    monkeypatch.setenv(vault.SECRETS_MODE_ENV, mode)
    env = _Recorder(**_inherited())
    dropped = vault.pin_process_environment(env)
    assert dropped == ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
    assert env["ANTHROPIC_BASE_URL"] == PIN == vault.ANTHROPIC_BASE_URL             # the gateway replaced by the pin
    assert "ANTHROPIC_API_KEY" not in env and "ANTHROPIC_AUTH_TOKEN" not in env
    assert env["ANTHROPIC_API_KEY_LOWPRIO"] == env["ANTHROPIC_API_KEY_BATCH"] == "synthetic-per-org-not-a-key"   # untouched
    assert env["HF_HUB_OFFLINE"] == "1"
    assert all(k in env for k in ROUND10_NAMES), dict(env)                          # round 10's names untouched under op / env: byte-identical
    # the real environment: the same
    for k, v in _inherited().items():
        monkeypatch.setenv(k, v)
    assert vault.pin_process_environment() == ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
    assert os.environ["ANTHROPIC_BASE_URL"] == PIN and "ANTHROPIC_API_KEY" not in os.environ and "ANTHROPIC_AUTH_TOKEN" not in os.environ
    assert os.environ["ANTHROPIC_API_KEY_LOWPRIO"] == "synthetic-per-org-not-a-key"


def test_a_fourth_mode_value_refuses_before_touching_the_environment(monkeypatch):
    monkeypatch.setenv(vault.SECRETS_MODE_ENV, "keyring")
    env = _Recorder(**_inherited())
    with pytest.raises(vault.VaultSecretError, match="must be one of"):
        vault.pin_process_environment(env)
    assert env["ANTHROPIC_BASE_URL"] == GATEWAY and len(env) == len(_inherited())   # untouched


# ------------------------------------------------------------------------------------- the two graders with a _pin_env()


@pytest.mark.parametrize("module", ["sisteer.eval.static_grading", "grade_static_sampled"])
@pytest.mark.parametrize("mode", ["wif", "op"])
def test_the_static_graders_pin_env_is_the_vaults(module, mode, monkeypatch):
    import importlib

    mod = importlib.import_module(module)
    monkeypatch.setenv(vault.SECRETS_MODE_ENV, mode)
    for k, v in _inherited().items():
        monkeypatch.setenv(k, v)
    mod._pin_env()
    names = sorted(k for k in os.environ if k.startswith(vault.WIF_SCRUB_PREFIX) and (mode == "wif" or k in _inherited()))
    if mode == "wif":
        assert names == [], names                                                    # nothing set, EVERY ANTHROPIC_* name gone (round 10; the host's too)
    else:
        assert names == sorted(["ANTHROPIC_API_KEY_BATCH", "ANTHROPIC_API_KEY_LOWPRIO", "ANTHROPIC_BASE_URL", *ROUND10_NAMES])   # byte-identical
        assert os.environ["ANTHROPIC_BASE_URL"] == PIN == mod.PINNED_BASE_URL
    assert mod.PINNED_BASE_URL == vault.ANTHROPIC_BASE_URL


# ---------------------------------------------------------------------------------- the two graders that pinned at import

IMPORT_TIME_GRADERS = {
    "grade_full22_native": "scripts/sdfbase/seedrep",
    "grade_alphasteer_f22": "scripts/alphasteer_f22",
}
_PROBE = ("import json, os, sys\n"
          "import {mod}\n"
          "print(json.dumps({{k: (v if k == 'ANTHROPIC_BASE_URL' else '<set>') for k, v in os.environ.items() if k.startswith('ANTHROPIC_')}}, sort_keys=True))\n")


def _import_under(mod: str, mode: str) -> dict:
    """Import the grader in a FRESH interpreter whose environment is a Claude Code / vault shell's (every value
    synthetic), and read back the ANTHROPIC_* names it left -- the base URL by value (it is the pin or the gateway, never
    a credential), every other name as '<set>'."""
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", ""),
           "PYTHONPATH": os.pathsep.join([str(REPO / "src"), str(REPO / "scripts"), str(REPO / IMPORT_TIME_GRADERS[mod])]),
           vault.SECRETS_MODE_ENV: mode, **_inherited()}
    r = subprocess.run([sys.executable, "-c", _PROBE.format(mod=mod)], capture_output=True, text=True, env=env, timeout=240, cwd=str(REPO))
    assert r.returncode == 0, r.stderr[-1500:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("mod", sorted(IMPORT_TIME_GRADERS))
def test_the_import_time_graders_set_no_anthropic_variable_under_wif(mod):
    left = _import_under(mod, "wif")
    assert left == {}, left                                                            # nothing set, EVERY inherited ANTHROPIC_* name dropped (round 10)


@pytest.mark.parametrize("mod", sorted(IMPORT_TIME_GRADERS))
def test_the_import_time_graders_under_op_still_pin_the_host_and_drop_the_shells_key_and_bearer(mod):
    left = _import_under(mod, "op")
    assert left.get("ANTHROPIC_BASE_URL") == PIN, left                                 # the gateway replaced by the pin, as before
    assert "ANTHROPIC_API_KEY" not in left and "ANTHROPIC_AUTH_TOKEN" not in left
    assert left.get("ANTHROPIC_API_KEY_LOWPRIO") == "<set>"                             # the per-org key stays for the vault's op reader


def test_the_import_time_graders_pin_through_the_vault_before_their_first_client_import():
    """A source pin: the module-level block is the vault's call, placed where the old assignment stood -- BEFORE the
    import of anything that constructs an anthropic client -- and no ``os.environ[...] =`` is left in either module."""
    for mod, d in IMPORT_TIME_GRADERS.items():
        src = (REPO / d / f"{mod}.py").read_text()
        assert "vault.pin_process_environment()" in src, mod
        assert 'os.environ["ANTHROPIC_BASE_URL"]' not in src, mod
        assert src.index("vault.pin_process_environment()") < src.index("import grade_bon_live"), mod


# ------------------------------------------------------------------------------------------------ the sweeps of the tree


def _tracked(suffixes) -> list:
    out = subprocess.run(["git", "ls-files", "-z", "--", *SCOPES], cwd=REPO, capture_output=True, text=True, check=True).stdout
    rels = sorted(p for p in out.split("\0") if p.endswith(suffixes) and not p.startswith(EXCLUDED) and "/tests/" not in p)
    assert len(rels) > 50, rels
    return rels


def _is_environ(node: ast.AST) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "environ" and isinstance(node.value, ast.Name) and node.value.id == "os") \
        or (isinstance(node, ast.Name) and node.id == "environ")


def _anthropic_const(node) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("ANTHROPIC_")


def anthropic_env_setters(src: str, rel: str = "<src>") -> list:
    """Every statement that SETS an ``ANTHROPIC_*`` variable in the process environment: ``os.environ[name] = ...`` (an
    assignment, augmented or annotated), ``os.environ.setdefault(name, ...)``, ``os.environ.update({name: ...})`` and
    ``os.putenv(name, ...)``. A child's environment dict (``env["ANTHROPIC_API_KEY"] = key`` on a copy) is NOT this: a
    key-handing launcher is the inventory's class (tests/test_wif_driver_inventory.py), refused by name under wif."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(src, filename=rel)
    hits = []
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for t in targets:
            if isinstance(t, ast.Subscript) and _is_environ(t.value) and _anthropic_const(t.slice):
                hits.append(node.lineno)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            f = node.func
            if _is_environ(f.value) and f.attr == "setdefault" and node.args and _anthropic_const(node.args[0]):
                hits.append(node.lineno)
            if _is_environ(f.value) and f.attr == "update":
                for a in node.args:
                    if isinstance(a, ast.Dict) and any(_anthropic_const(k) for k in a.keys):
                        hits.append(node.lineno)
                if any(kw.arg and kw.arg.startswith("ANTHROPIC_") for kw in node.keywords):
                    hits.append(node.lineno)
            if f.attr == "putenv" and isinstance(f.value, ast.Name) and f.value.id == "os" and node.args and _anthropic_const(node.args[0]):
                hits.append(node.lineno)
    return sorted(set(hits))


def anthropic_mapping_setters(src: str, rel: str = "<src>") -> list:
    """Every assignment of an ``ANTHROPIC_*`` name into ANY mapping subscript (``<mapping>[name] = ...``) -- the broader walk
    for the vault itself, whose one setter writes through its ``env`` seam (``os.environ`` by default), which the process walk
    above does not see by design."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(src, filename=rel)
    hits = []
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else []
        for t in targets:
            if isinstance(t, ast.Subscript) and _anthropic_const(t.slice):
                hits.append(node.lineno)
    return sorted(set(hits))


def _function_span(src: str, name: str) -> range:
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return range(node.lineno, node.end_lineno + 1)
    raise AssertionError(f"no function {name}")


SYNTHETIC_SETTERS = ("import os\nfrom os import environ\n"
                     "os.environ['ANTHROPIC_BASE_URL'] = 'x'\n"                           # 3
                     "environ['ANTHROPIC_API_KEY'] = 'x'\n"                               # 4
                     "os.environ.setdefault('ANTHROPIC_AUTH_TOKEN', 'x')\n"              # 5
                     "os.environ.update({'ANTHROPIC_API_KEY_LOWPRIO': 'x'})\n"           # 6
                     "os.environ.update(ANTHROPIC_API_KEY_BATCH='x')\n"                   # 7
                     "os.putenv('ANTHROPIC_BASE_URL', 'x')\n"                             # 8
                     "env = dict(os.environ)\nenv['ANTHROPIC_API_KEY'] = 'k'\n"          # 9-10: a child's dict, not the process class
                     "os.environ['HF_HOME'] = 'x'\nos.environ.pop('ANTHROPIC_API_KEY', None)\n")   # 11-12: not an ANTHROPIC_ set


def test_the_setter_walk_sees_every_shape_and_not_a_childs_environment_dict():
    assert anthropic_env_setters(SYNTHETIC_SETTERS) == [3, 4, 5, 6, 7, 8]
    assert anthropic_mapping_setters(SYNTHETIC_SETTERS) == [3, 4, 10]          # the mapping walk sees the seam-shaped write too


def test_no_tracked_python_sets_an_anthropic_variable_outside_the_vault():
    offenders = {}
    for rel in _tracked((".py",)):
        lines = anthropic_env_setters((REPO / rel).read_text(encoding="utf-8", errors="replace"), rel)
        if lines and rel not in ENV_SETTER_ALLOWED:
            offenders[rel] = lines
    assert not offenders, ("a process-environment setter of an ANTHROPIC_* variable outside sisteer.vault.pin_process_environment "
                           "(under wif nothing may set or export ANTHROPIC_*; route it through the vault's pin): "
                           + "; ".join(f"{r}:{ls}" for r, ls in sorted(offenders.items())))
    # the vault's ONE setter writes through the pin's `env` seam (os.environ by default), so the os.environ walk above does not see it
    # by design; the mapping walk must find exactly one, inside pin_process_environment -- a second setter anywhere in the vault fails
    vault_src = (REPO / "src/sisteer/vault.py").read_text()
    assert anthropic_env_setters(vault_src, "vault.py") == []
    vault_lines = anthropic_mapping_setters(vault_src, "vault.py")
    assert len(vault_lines) == 1, vault_lines                                       # the one setter: the op / env branch of the pin
    assert vault_lines[0] in _function_span(vault_src, "pin_process_environment"), vault_lines


def test_every_shell_export_or_assignment_of_an_anthropic_variable_is_on_the_mode_guarded_allowlist():
    found = []
    for rel in _tracked((".sh", ".bash", ".sbatch")):
        for i, line in enumerate((REPO / rel).read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if SHELL_SETTER_RE.search(line):
                found.append((rel, i, line.strip()))
    unexplained = [(r, i, l[:160]) for r, i, l in found if not any(r == ar and sub in l for ar, sub in SHELL_SETTERS_ALLOWED)]
    assert not unexplained, ("a shell export / assignment of an ANTHROPIC_* variable that is neither mode-guarded nor on an op / env "
                             "branch of the allowlist (under wif nothing may export ANTHROPIC_*): " + "; ".join(map(str, unexplained)))
    stale = [(r, sub) for r, sub in SHELL_SETTERS_ALLOWED if not any(r == fr and sub in l for fr, _, l in found)]
    assert not stale, f"allowlist entries no shell line carries: {stale}"
    assert len(found) == len(SHELL_SETTERS_ALLOWED) == 4, found


SCRUB = vault.WIF_SHELL_SCRUB
#: the six shell sites that scrub under wif -> True when the site's script runs under `set -u` (the generated units do)
SHELL_SCRUB_SITES = {
    "scripts/sdfbase/full22/sampled/grade_sampled.sh": True,
    "scripts/sdfbase/seedrep/grade_battery.sh": True,
    "reproduce/lib/secrets.sh": False,
    "scripts/sdfbase/full22/judge/launch_judge.sh": False,
    "scripts/sdfbase/full22/judge/finish_lane.sh": False,
    "scripts/sdfbase/full22/petri_drive_full22.sh": False,
}


def _wif_scrub_snippet(rel: str) -> str:
    """The wif scrub of one shell site, as the text a bash can run -- read off the REAL file, never re-typed here."""
    src = (REPO / rel).read_text()
    if rel.endswith(("grade_sampled.sh", "grade_battery.sh")):
        lines = re.findall(r"lines\.append\('(export SISTEER_SECRETS_MODE=wif; [^']*)'\)", src)      # the generated unit's wif line
        assert len(lines) == 1, (rel, lines)
        return lines[0]
    if rel.endswith("secrets.sh"):
        fn = src[src.index("secrets_unit_key_line(){"):]
        fn = fn[:fn.index("\n}")]
        wif_branch = fn[fn.index("= wif ]; then"):re.search(r"^\s*fi\b", fn, re.M).start()]   # up to the keyword on its own line
        assert SCRUB in wif_branch, wif_branch                                          # the ONE scrub line the reproduce units receive
        return f'source {REPO / rel}; eval "$(secrets_unit_key_line lowprio)"'
    if rel.endswith("launch_judge.sh"):
        line = next(l for l in src.splitlines() if "/bin/bash -c '" in l)
        body = line.split("/bin/bash -c '", 1)[1]
        return body[:body.index("unset ANTHROPIC_BASE_URL;") + len("unset ANTHROPIC_BASE_URL;")]   # the unit's secrets prefix, every mode
    if rel.endswith("finish_lane.sh"):
        lines = [l for l in src.splitlines() if l.startswith('[ "$MODE" = wif ] && unset') or l == "unset ANTHROPIC_BASE_URL"]
        assert len(lines) == 2, lines
        return "\n".join(lines)
    if rel.endswith("petri_drive_full22.sh"):
        head = 'if [ "$SECRETS_MODE" = wif ]; then'
        wif_branch = src[src.index(head) + len(head):src.index('elif [ "$SECRETS_MODE" = env ]')]
        assert '[ -z "${ANTHROPIC_API_KEY+x}" ] ||' in wif_branch                        # the key refused by name, then the rest dropped
        assert wif_branch.index("ANTHROPIC_API_KEY+x") < wif_branch.index(SCRUB)
        return "\n".join(l for l in wif_branch.splitlines() if l.strip() and not l.strip().startswith("#"))
    raise AssertionError(rel)


def test_every_shell_wif_scrub_is_the_vaults_one_spelling():
    """ONE rule, two spellings, pinned equal: the vault drops the PREFIX (scrub_inherited_anthropic_environment) and every shell
    site drops the prefix too -- bash's ``${!ANTHROPIC_@}``, which expands to the NAMES (no value read or printed). An enumerated
    list in any wif branch fails here: round 9's five names were a list, and round 10 found the SDK reading a sixth
    (ANTHROPIC_CUSTOM_HEADERS) whatever credential it was given."""
    assert SCRUB == 'unset "${!ANTHROPIC_@}"' and vault.WIF_SCRUB_PREFIX == "ANTHROPIC_" and SCRUB == f'unset "${{!{vault.WIF_SCRUB_PREFIX}@}}"'
    for rel in SHELL_SCRUB_SITES:
        snippet = _wif_scrub_snippet(rel)
        assert SCRUB in (snippet if not rel.endswith("secrets.sh") else (REPO / rel).read_text()), (rel, snippet)
        assert re.search(r"\bunset\s+ANTHROPIC_(?!BASE_URL\b)", snippet) is None, (rel, snippet)   # no enumerated list (the every-mode BASE_URL unset aside)
    # and no enumerated scrub list on any wif line of the tracked shell files (the op / env lines' `unset ANTHROPIC_BASE_URL
    # ANTHROPIC_AUTH_TOKEN` are the old modes, byte-identical by the lane's rule; the two multi-line wif branches are read above)
    stale = []
    for rel in _tracked((".sh", ".bash", ".sbatch")):
        for i, line in enumerate((REPO / rel).read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            code = line.split("#", 1)[0]
            if "wif" in code and re.search(r"\bunset\s+ANTHROPIC_[A-Z_]+\s+ANTHROPIC_", code):
                stale.append((rel, i, line.strip()[:120]))
    assert not stale, stale


@pytest.mark.parametrize("rel", sorted(SHELL_SCRUB_SITES))
def test_every_shell_wif_scrub_run_under_a_shell_carrying_every_known_name_leaves_none(rel):
    """MEASURED, not read: each site's scrub is run by bash under a synthetic shell carrying every name of
    ``vault.WIF_INHERITED_ENV_NAMES`` (the SDK's custom-headers variable included) and one no list knows; afterwards no
    ``ANTHROPIC_*`` variable exists, an unrelated one does, and no value was printed. The generated units run under ``set -u``, so
    their line is run under it here. The Petri drive's branch refuses a present ANTHROPIC_API_KEY by name before it scrubs, so it
    runs without one (the refusal has its own test)."""
    snippet = _wif_scrub_snippet(rel)
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", ""), "SISTEER_SECRETS_MODE": "wif", "MODE": "wif",
           "SECRETS_MODE": "wif", "ANTHROPIC_FUTURE_SDK_SWITCH": "x", **_inherited()}
    assert set(vault.WIF_INHERITED_ENV_NAMES) <= set(env)
    if rel.endswith("petri_drive_full22.sh"):
        del env["ANTHROPIC_API_KEY"]
    script = ("set -u\n" if SHELL_SCRUB_SITES[rel] else "") + snippet + "\ncompgen -A variable | grep '^ANTHROPIC_' || echo NONE\necho HF=$HF_HUB_OFFLINE"
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, (rel, r.stdout, r.stderr)
    assert r.stdout.split() == ["NONE", "HF=1"], (rel, r.stdout, r.stderr)
    for k, v in _inherited().items():
        if k.startswith("ANTHROPIC_"):
            assert v not in r.stdout + r.stderr, (rel, k)                                # names dropped, values never printed

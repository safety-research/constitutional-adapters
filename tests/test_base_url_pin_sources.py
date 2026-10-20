"""Every base URL that reaches a vault-built Anthropic client from OUTSIDE the process goes through ``sisteer.vault.pinned_base_url``.

``capeval.judging.judge_base_url`` learnt the wif rule -- under
``SISTEER_SECRETS_MODE=wif`` the API host is pinned and a ``CAPEVAL_JUDGE_BASE_URL`` naming another host refuses by name -- while the
misalign judge's own copy of the same function (``scripts/sdfbase/full22/misalign/misalign_judge.py``) had not: it handed an inherited
override straight to ``vault.anthropic_client``, whose federated constructor honors an explicit ``base_url``, so a local gateway or a
mock would have received the federated bearer and the judge content from the preflight on. A behavioural test covers the sites someone
remembered to write it for; this module is a SOURCE-level audit of the CLASS (``tests/test_no_credential_files.py``'s shape), so the
failure it guards against -- a third reader, or a constructor fed an outside value directly -- is caught at the new call site:

* the readers of ``CAPEVAL_JUDGE_BASE_URL`` (by its literal name or through a ``JUDGE_BASE_URL_ENV`` constant, off any mapping) are
  EXACTLY the two ``judge_base_url`` functions, and each calls ``vault.pinned_base_url`` in its own body;
* no call to a vault constructor (``anthropic_client`` / ``anthropic_async_client`` / ``anthropic_client_wif``) passes a ``base_url=``
  whose expression reads the environment (``os.environ`` / ``os.getenv``) or an argparse / config namespace (``args.`` / ``cfg.`` /
  ``config.`` / ``opts.`` ...) directly -- such a value goes through ``pinned_base_url`` first (a call to it anywhere in the expression
  passes). The in-process seams -- a constant, a parameter forwarded by name, a test double -- are not this class and pass.

The behavioural half -- the rule's table under op / env / wif, the misalign judge's refusal before any client is built, the controls
judge's -- is ``tests/test_grade_wif_client.py``.
"""
from __future__ import annotations

import ast
import re
import subprocess
import warnings
from pathlib import Path

import pytest

from sisteer import vault

REPO = Path(__file__).resolve().parents[1]
SCOPES = ("src/", "scripts/", "reproduce/")
EXCLUDED = ("legacy/",)
VARIABLE = "CAPEVAL_JUDGE_BASE_URL"
VARIABLE_CONST = "JUDGE_BASE_URL_ENV"
HELPER = "pinned_base_url"
CTORS = frozenset({"anthropic_client", "anthropic_async_client", "anthropic_client_wif"})
#: the names an argparse / config namespace goes by: an attribute of one of these in a base_url= expression is an outside value
NAMESPACES = frozenset({"args", "cfg", "config", "opts", "ns", "settings", "env"})
#: the two typed wrappers of the one rule (JudgingError for the controls judge, SystemExit for the misalign judge); a third reader of
#: the variable is a new site that must say here why it does not resolve the override through the vault
EXPECTED_READERS = frozenset({
    ("src/sisteer/capeval/judging.py", "judge_base_url"),
    ("scripts/sdfbase/full22/misalign/misalign_judge.py", "judge_base_url"),
})


def _tracked_python() -> list:
    out = subprocess.run(["git", "ls-files", "-z", "--", *SCOPES], cwd=REPO, capture_output=True, text=True, check=True).stdout
    rels = sorted(p for p in out.split("\0") if p.endswith(".py") and not p.startswith(EXCLUDED) and "/tests/" not in p)
    assert len(rels) > 100, rels                    # the inventory is the tracked tree, not a guess
    return rels


@pytest.fixture(scope="module")
def trees() -> dict:
    parsed = {}
    for rel in _tracked_python():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)      # an invalid escape in a regex literal is a warning, not a parse failure
                parsed[rel] = ast.parse((REPO / rel).read_text(encoding="utf-8", errors="replace"), filename=rel)
        except SyntaxError as exc:                  # every tracked source parsed on the day this was written; a new one that does not is loud
            pytest.fail(f"{rel} does not parse under this interpreter, so it cannot be audited: {exc}")
    return parsed


def _names_the_variable(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return node.value == VARIABLE
    if isinstance(node, ast.Name):
        return node.id == VARIABLE_CONST
    return isinstance(node, ast.Attribute) and node.attr == VARIABLE_CONST


def _is_variable_read(node: ast.AST) -> bool:
    """``<mapping>.get(<var>)``, ``<mapping>[<var>]``, ``<mapping>.pop(<var>)`` or ``getenv(<var>)`` for the override variable,
    whatever the mapping -- ``os.environ``, a captured ``env`` dict, a unit's environment."""
    if isinstance(node, ast.Subscript):
        return _names_the_variable(node.slice)
    if isinstance(node, ast.Call) and node.args:
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
        return name in ("get", "getenv", "pop", "setdefault") and _names_the_variable(node.args[0])
    return False


def _calls_helper(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            if (isinstance(f, ast.Name) and f.id == HELPER) or (isinstance(f, ast.Attribute) and f.attr == HELPER):
                return True
    return False


def _owner_of_every_node(tree: ast.AST) -> dict:
    """node -> the innermost enclosing function def (None at module level)."""
    owner = {}

    def visit(node, current):
        for child in ast.iter_child_nodes(node):
            nxt = child if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else current
            owner[child] = nxt
            visit(child, nxt)

    visit(tree, None)
    return owner


def _outside_values(expr: ast.AST) -> list:
    """What in a ``base_url=`` expression comes from outside the process: an environment read or a namespace attribute."""
    found = []
    for sub in ast.walk(expr):
        if isinstance(sub, ast.Name) and sub.id in ("environ", "getenv"):
            found.append(sub.id)
        elif isinstance(sub, ast.Attribute) and sub.attr in ("environ", "getenv"):
            found.append(ast.unparse(sub))
        elif isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id in NAMESPACES:
            found.append(ast.unparse(sub))
    return found


def test_the_vault_owns_the_rule_and_its_refusal_is_a_vault_error():
    assert callable(vault.pinned_base_url)
    assert issubclass(vault.BaseUrlRefused, vault.VaultSecretError)


def test_the_readers_of_the_judge_override_variable_are_the_two_wrappers_and_each_resolves_it_through_the_vault(trees):
    readers = {}
    for rel, tree in trees.items():
        owner = _owner_of_every_node(tree)
        for node in ast.walk(tree):
            if _is_variable_read(node):
                fn = owner.get(node)
                readers.setdefault((rel, fn.name if fn is not None else "<module>"), []).append((node.lineno, fn))
    assert set(readers) == EXPECTED_READERS, (
        f"the readers of {VARIABLE} are {sorted(set(readers))}; expected exactly {sorted(EXPECTED_READERS)} -- a new reader resolves "
        f"the override through vault.{HELPER} (under wif the API host is pinned and another host refuses by name) and is added to "
        f"EXPECTED_READERS with its reason, or passes through one of the two wrappers")
    for (rel, name), hits in sorted(readers.items()):
        fn = hits[0][1]
        assert fn is not None, f"{rel}: {VARIABLE} is read at module level (line {hits[0][0]}), outside any function that could pin it"
        assert _calls_helper(fn), (
            f"{rel}:{fn.lineno} {name} reads {VARIABLE} but never calls vault.{HELPER}: under {vault.SECRETS_MODE_ENV}={vault.WIF_MODE} "
            f"an override naming another host would reach a federated client's constructor (the misalign judge on 545c56eda)")


def test_no_vault_constructor_is_fed_a_base_url_read_from_the_environment_or_a_namespace_directly(trees):
    violations = []
    n_calls = n_with_base_url = 0
    for rel, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
            if name not in CTORS:
                continue
            n_calls += 1
            for kw in node.keywords:
                if kw.arg != "base_url":
                    continue
                n_with_base_url += 1
                outside = _outside_values(kw.value)
                if outside and not _calls_helper(kw.value):
                    violations.append(f"{rel}:{node.lineno} {name}(base_url={ast.unparse(kw.value)}) reads {outside}")
    assert n_calls >= 10 and n_with_base_url >= 5, (n_calls, n_with_base_url)      # the audit saw the constructors it is about
    assert not violations, "a vault constructor is handed an outside base_url without vault.%s:\n  %s" % (HELPER, "\n  ".join(violations))


# ------------------------------------------------------------------------------ the plain-constructor class
#: An Anthropic client's constructors. A call to one OUTSIDE the vault is a client the vault's MODE never sees:
#: finish_lane.sh's readiness probe built ``anthropic.Anthropic(api_key=<secret-manager key>, base_url=...)`` inside a shell HEREDOC, so
#: under SISTEER_SECRETS_MODE=wif it fetched the revoked key and re-tried it once a minute for 24 h, never reaching the federated launcher.
#: The two rules above walk ``.py`` files; this one reads the shell scripts too, where a heredoc or a ``python -c`` string hides the same
#: constructor from an ast walk.
PLAIN_CTORS = frozenset({"Anthropic", "AsyncAnthropic"})
#: the sites that ARE the class's own implementations, each held to the base URL it must pin BY NAME (None: the vault itself, whose
#: constructors are the pin): the corpus generator's self-contained three-mode client (it imports no sisteer) and the controls judge's
#: OpenRouter backend (keyed on OpenRouter's constant; not a client of the Anthropic API). The inspect-provider patch
#: (sisteer.eval.anthropic_wif) constructs through an ALIAS of the provider's class and sets kwargs["base_url"] to the vault's pin;
#: tests/test_anthropic_wif.py holds that, and an alias is not a name this walk can see.
PLAIN_CTOR_OWNERS = {
    "src/sisteer/vault.py": None,
    "corpus_generator/src/corpus_generator/client.py": "API_BASE_URL",
    "src/sisteer/capeval/judging.py": "OPENROUTER_BASE_URL",
}
PLAIN_SCOPES = ("src/", "scripts/", "reproduce/", "corpus_generator/src/")
SHELL_SUFFIXES = (".sh", ".sbatch")
#: the constructor as a shell script's embedded Python spells it (a heredoc, a `python -c` string): ``anthropic.Anthropic(`` /
#: ``AsyncAnthropic(`` / a bare ``Anthropic(``; ``anthropic_client(`` (the vault's) does not match
_SHELL_CTOR = re.compile(r"\b(?:anthropic\.)?(?:Async)?Anthropic\(")


def _tracked(scopes, *, suffixes) -> list:
    out = subprocess.run(["git", "ls-files", "-z", "--", *scopes], cwd=REPO, capture_output=True, text=True, check=True).stdout
    return sorted(p for p in out.split("\0") if p.endswith(suffixes) and not p.startswith(EXCLUDED) and "/tests/" not in p)


def test_no_plain_anthropic_constructor_outside_the_vault_in_python_or_in_python_embedded_in_a_shell_script():
    py = _tracked(PLAIN_SCOPES, suffixes=(".py",))
    sh = _tracked(PLAIN_SCOPES, suffixes=SHELL_SUFFIXES)
    assert len(py) > 100 and len(sh) > 20, (len(py), len(sh))          # the inventory is the tracked tree, not a guess
    violations, seen = [], {}
    for rel in py:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse((REPO / rel).read_text(encoding="utf-8", errors="replace"), filename=rel)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
            if name not in PLAIN_CTORS:
                continue
            seen[rel] = seen.get(rel, 0) + 1
            if rel not in PLAIN_CTOR_OWNERS:
                violations.append(f"{rel}:{node.lineno} {ast.unparse(f)}(...) -- an Anthropic client is built through "
                                  "sisteer.vault.anthropic_client / anthropic_async_client (the mode-aware constructors), never directly")
                continue
            pin = PLAIN_CTOR_OWNERS[rel]
            if pin is not None:
                urls = [ast.unparse(kw.value) for kw in node.keywords if kw.arg == "base_url"]
                if urls != [pin]:
                    violations.append(f"{rel}:{node.lineno} {ast.unparse(f)}(base_url={urls or 'absent'}) -- this owner pins {pin} by name")
    for rel in sh:
        for lineno, line in enumerate((REPO / rel).read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if _SHELL_CTOR.search(line):
                violations.append(f"{rel}:{lineno} {line.strip()[:120]} -- Python embedded in a shell script builds its client through "
                                  "sisteer.vault too (the finisher's probe on e2c3c467d built a keyed client under wif)")
    assert not violations, "a plain Anthropic constructor outside the vault:\n  " + "\n  ".join(violations)
    # every owner still constructs (the allowlist is not vacuous) and nobody else does; the vault's two are the keyed op / env constructors --
    # its federated one goes through a `cls` alias (Anthropic / AsyncAnthropic by flag) and is pinned by construction
    assert set(seen) == set(PLAIN_CTOR_OWNERS) and seen["src/sisteer/vault.py"] == 2, seen

"""The static graders' credential wording: the ``org_switch`` sentence a run records and the ``--org`` help a reader sees name
sisteer.vault's credential or pool, in every secrets mode, never a particular secret-manager command or product. The strings are
checked exactly, and against the token table the publication scan reads (tests/front_door_testlib.py), so this file holds no word
it looks for."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TESTS = REPO / "tests"
for _p in (REPO / "src", TESTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from front_door_testlib import MATCHERS, token_hits  # noqa: E402  (the ONE token table the publication scan reads)
from sisteer.eval import static_grading as sg  # noqa: E402

#: every tier but attribution: what any file of the published tree may not carry
TREE = MATCHERS["tree"]
ORG_SWITCH_KEYED = "impossible: --org accepts only lowprio; the key is sisteer.vault's credential for lowprio at call time"
ORG_SWITCH_WIF = ("impossible: --org accepts only lowprio; the credential is sisteer.vault's federated identity for lowprio "
                  "(no key) at call time")
#: the --org help of the shared static body, the sampled lane and the seed-replicate grader twin, read from source by ast (the
#: twin is never imported here)
ORG_HELP = {
    "src/sisteer/eval/static_grading.py": "the ONE org this grader may bill (a sisteer.vault pool)",
    "scripts/sdfbase/full22/sampled/grade_static_sampled.py": "the ONE org this grader may bill (a sisteer.vault pool)",
    "scripts/sdfbase/seedrep/grade_full22_native.py": "starting org (a sisteer.vault pool)",
}


@pytest.mark.parametrize("mode,expected", [("op", ORG_SWITCH_KEYED), ("env", ORG_SWITCH_KEYED), ("wif", ORG_SWITCH_WIF)])
def test_org_switch_names_the_vault_credential_in_every_mode(monkeypatch, mode, expected):
    monkeypatch.setenv("SISTEER_SECRETS_MODE", mode)
    text = sg.org_switch_text()
    assert text == expected
    assert token_hits(text, TREE) == []


def _org_helps(rel: str) -> list:
    """The help text of every ``add_argument("--org", ...)`` call in the file."""
    out = []
    for node in ast.walk(ast.parse((REPO / rel).read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--org"):
            helps = [k.value for k in node.keywords if k.arg == "help"]
            out.append(ast.literal_eval(helps[0]) if helps else None)
    return out


@pytest.mark.parametrize("rel", sorted(ORG_HELP))
def test_org_help_names_a_vault_pool(rel):
    assert _org_helps(rel) == [ORG_HELP[rel]]
    assert token_hits(ORG_HELP[rel], TREE) == []

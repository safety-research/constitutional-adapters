"""The arditi_allpos operating point is DECLARED, and the declaration VERIFIES rather than labels.

Phase E of the arditi-qwen-all-positions campaign fills the row deliberately left out of
``ARM_OPERATING_POINT`` in ``scripts/r2_export/cap4096/build_transfer_stats_cap4096.py`` while the ladder's
selection step was outstanding. Two properties matter and neither is obvious from reading the table:

1. ``arm_op_label`` must reproduce the string the Phase B' rehearsal verified byte-for-byte. The rehearsal
   pinned the published label, so a re-wording here silently republishes 18 export rows under a new name.
2. Declaring the row ARMS ``served_operating_point_of``, which refuses a root whose recorded per-arm config
   disagrees with the declaration. A declaration that derives to a DIFFERENT id than the one declared would
   refuse the very legs it was added to describe -- so the good case is asserted alongside a KNOWN-BAD config
   that must raise.

The builder is loaded by AST rather than imported: its module-level import chain reaches numpy and two
out-of-repo state trees, none of which this behaviour depends on.
"""

import ast
import hashlib
import re
import subprocess
from pathlib import Path
from typing import Optional

import pytest

ROOT = Path(__file__).parents[1]
BUILDER = ROOT / "scripts" / "r2_export" / "cap4096" / "build_transfer_stats_cap4096.py"

VIEW = "qwen35_4b_it"
ARM = "arditi_allpos"
DECLARED_ID = "ar_c0p3__arditi_allpos"
#: the exact label the Phase B' rehearsal verified; a change here is a republication, not a refactor
REHEARSED_LABEL = "Arditi refusal direction, paper configuration, c = 0.3 (ar_c0p3__arditi_allpos)"


def _tree():
    return ast.parse(BUILDER.read_text())


def _table():
    """ARM_OPERATING_POINT as a real dict, evaluated from its literal (no import chain)."""
    for node in _tree().body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "ARM_OPERATING_POINT" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("ARM_OPERATING_POINT not found in the builder")


#: the module-level helpers `served_operating_point_of` calls: `_config_bytes_from_git` since the check became content-addressed,
#: `_repo_history_is_searchable` since the shallow-checkout case was split out, `_config_rel` + `_history_has_no_earlier_revision`
#: since the no-history case (a published repository squashed to one commit) was split out from the content miss
HELPERS = ("_config_bytes_from_git", "_repo_history_is_searchable", "_config_rel", "_history_has_no_earlier_revision")


def _func(name, *, also=()):
    """Exec one top-level function out of the builder, plus any helpers it calls, with only the globals they use.

    `served_operating_point_of` calls module-level helpers (HELPERS above). EVERY such helper must be exec'd into the same
    namespace or the call is a NameError at test time — if you add one to the builder, add it to HELPERS here."""
    wanted = {name, *also}
    nodes = [n for n in _tree().body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    missing = wanted - {n.name for n in nodes}
    assert not missing, f"not found in the builder: {sorted(missing)}"
    ns = {"Path": Path, "ARM_OPERATING_POINT": _table(), "re": re, "subprocess": subprocess,
          "hashlib": hashlib, "Optional": Optional, "__file__": str(BUILDER)}
    exec(compile(ast.Module(nodes, []), str(BUILDER), "exec"), ns)  # noqa: S102
    return ns[name]


def test_the_allpos_row_is_declared_with_the_selected_dose():
    assert (VIEW, ARM) in _table(), (
        "Phase E's row is missing: an arditi_allpos export row would carry arm_operating_point_id=UNDECLARED"
    )
    op = _table()[(VIEW, ARM)]
    assert op[0] == DECLARED_ID
    assert "c = 0.3" in op[1], "the prose must name the dose in the raw_coefficient unit"


def test_label_is_byte_identical_to_the_rehearsed_string():
    assert _func("arm_op_label")(_table()[(VIEW, ARM)]) == REHEARSED_LABEL


def test_the_dose_unit_is_not_confusable_with_the_prefill_only_arm():
    """`arditi` doses in alpha (a fraction of the residual norm); `arditi_allpos` doses in raw c.

    Sharing an id prefix between the two would let a consumer read one arm's rung as the other's."""
    t = _table()
    assert t[(VIEW, "arditi")][0] != t[(VIEW, ARM)][0]
    assert t[(VIEW, ARM)][0].endswith("__arditi_allpos")


def _cfg(tmp_path, bank_view):
    """A per-arm config on disk, recorded with its REAL sha256.

    The builder now verifies the on-disk config against the sha the grader recorded, so a fixture carrying a
    made-up sha refuses for that reason and never reaches the dose comparison these tests are about."""
    p = tmp_path / "arm.yaml"
    body = f"defenses_dir: /shared/exp/objects/arditi_allpos/qwen35_4b/ladder/cells_view/{bank_view}\n"
    p.write_text(body)
    return {"config": {"path": str(p), "sha256": hashlib.sha256(body.encode()).hexdigest()}}


def test_a_root_served_at_the_declared_dose_verifies(tmp_path):
    pytest.importorskip("yaml")
    out = _func("served_operating_point_of", also=HELPERS)(_cfg(tmp_path, "ar_c0p3"), ARM, VIEW)
    assert out["status"] == "verified", out
    assert out["served_id"] == DECLARED_ID


@pytest.mark.parametrize("bad", ["ar_c0p2", "ar_c0p5", "ar_c1", "ar_r0p05"])
def test_a_root_served_at_any_other_dose_REFUSES(tmp_path, bad):
    """The known-bad half: the declaration must reject a measurement taken at a different c.

    ar_c0p2 is the selection rule's own answer, which was overridden -- exactly the mix-up most likely
    to occur, and the one that would file a 0.2 measurement under the 0.3 label."""
    pytest.importorskip("yaml")
    with pytest.raises(SystemExit) as e:
        _func("served_operating_point_of", also=HELPERS)(_cfg(tmp_path, bad), ARM, VIEW)
    assert "REFUSING" in str(e.value) and DECLARED_ID in str(e.value)


def test_an_unverifiable_config_is_recorded_not_refused(tmp_path):
    """A removed worktree makes a config unreadable; that is missing evidence, not a contradiction."""
    out = _func("served_operating_point_of", also=HELPERS)(
        {"config": {"path": str(tmp_path / "gone.yaml"), "sha256": "x"}}, ARM, VIEW)
    assert out["status"] == "unverifiable"
    assert out["declared_id"] == DECLARED_ID


def test_an_on_disk_config_whose_sha_disagrees_REFUSES(tmp_path):
    """Behavioural half of the new on-disk check: a config edited after grading must not be used to verify.

    Before this check the builder read the file and derived the operating point from whatever it now said,
    while the grader's recorded sha256 sat unused beside it — so an in-place edit silently re-labelled a
    measurement. The refusal must name the sha disagreement, not the dose."""
    cfg = _cfg(tmp_path, "ar_c0p3")
    Path(cfg["config"]["path"]).write_text(
        "defenses_dir: /shared/exp/objects/arditi_allpos/qwen35_4b/ladder/cells_view/ar_c0p3\n# edited\n")
    with pytest.raises(SystemExit) as e:
        _func("served_operating_point_of", also=HELPERS)(cfg, ARM, VIEW)
    assert "REFUSING" in str(e.value) and "sha256" in str(e.value)

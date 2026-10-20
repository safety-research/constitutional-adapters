"""No export builder may import `anchor_pairing` out of a sibling WORKTREE.

`sisteer.eval.anchor_pairing` was developed on a separate worktree and the three r2_export builders
pinned its path there while the branch was in flight. It has since landed on main, so those pins now do
two bad things at once: they keep an otherwise-finished worktree load-bearing (it cannot be removed
without breaking the builders), and -- the quiet one -- a builder run from ANY checkout imports that one
tree's copy of the module rather than its own, so a change to the pairing rule appears to have no effect.

The fix resolves the path from each builder's own repo root. These tests pin both halves: no worktree path
survives, and the path each builder computes lands on a real module inside its own checkout.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
BUILDERS = (
    "scripts/r2_export/build_export/build_export.py",
    "scripts/r2_export/llama8b/build_transfer_stats_full22_llama8b.py",
    "scripts/r2_export/cap4096/build_transfer_stats_cap4096.py",
)


@pytest.mark.parametrize("rel", BUILDERS)
def test_no_builder_pins_a_worktree_path(rel):
    src = (ROOT / rel).read_text()
    for m in re.finditer(r"^ANCHOR_PAIRING_SRC\s*=.*$", src, re.M):
        assert "worktrees/" not in m.group(0), (
            f"{rel} imports anchor_pairing from a worktree; it must resolve from its own repo root"
        )


@pytest.mark.parametrize("rel", BUILDERS)
def test_each_builder_resolves_to_a_real_module_in_its_own_checkout(rel):
    """Evaluate the assignment the way the builder does and check the module is actually there."""
    f = (ROOT / rel).resolve()
    # both spellings in use (os.path.dirname x4, Path(...).parents[3]) land on the file's own repo root
    assert f.parents[3] == ROOT.resolve(), f"{rel} is not at the depth the assignment assumes"
    expected = f.parents[3] / "src"
    assert (expected / "sisteer" / "eval" / "anchor_pairing.py").is_file(), (
        f"{rel} would refuse: no anchor_pairing module at {expected}"
    )


@pytest.mark.parametrize("rel", BUILDERS)
def test_the_import_failure_still_REFUSES_by_name(rel):
    """Re-pointing must not have softened the guard: a missing module still has to stop the build."""
    src = (ROOT / rel).read_text()
    i = src.index("ANCHOR_PAIRING_SRC")
    tail = src[i:i + 1200]
    assert "anchor_pairing" in tail and "SystemExit" in tail and "REFUSING" in tail, (
        f"{rel} no longer refuses by name when anchor_pairing cannot be imported"
    )


def test_the_module_is_on_main_not_only_in_a_worktree():
    assert (ROOT / "src" / "sisteer" / "eval" / "anchor_pairing.py").is_file()


@pytest.mark.parametrize("rel", BUILDERS)
def test_the_assignment_is_syntactically_one_expression(rel):
    """Guards against a half-applied edit leaving two assignments or a stray fragment."""
    tree = ast.parse((ROOT / rel).read_text())
    hits = [n for n in tree.body
            if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "ANCHOR_PAIRING_SRC" for t in n.targets)]
    assert len(hits) == 1, f"{rel} has {len(hits)} ANCHOR_PAIRING_SRC assignments"

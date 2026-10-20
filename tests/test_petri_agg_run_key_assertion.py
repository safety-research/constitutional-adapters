"""An allow-listed Petri cell must be pinned to its RUN KEY, because the cell id does not identify the audit.

Two complete audits of `arditi_allpos` exist on this box and **nothing in the aggregate's cell record separates
them except the run key**:

| | selected | certification root |
|---|---|---|
| root | `petri_full22_qwen35_4b_it__arditi_allpos` | `petri_full22_qwen35_4b_allposcert` |
| run key | `79e26caa1912a88c` | `6d943f30288ba4ad` |
| rows | 180 | 180 |
| `complete` | True | True |
| `incomplete_marker` | True | True |
| its `none` | complete, so the arm is pairable in its own standup | NOT complete |

`incomplete_marker` is the field whose name suggests it would discriminate, and it does not: it records that a
`petri_incomplete.json` FILE exists, and both roots have one. Both ledgers in fact read `unresolved: 0`. The
aggregate carries no unresolved COUNT, so a consumer of it alone cannot tell a closed ledger from an open one.

So the guard is the run key, and a second guard refuses an `arditi_allpos` cell that was allow-listed without one —
otherwise the next person to add a view here reintroduces the hole by omission rather than by error.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
BUILDER = ROOT / "scripts" / "r2_export" / "build_export" / "build_export.py"
SELECTED = "79e26caa1912a88c"
CERT = "6d943f30288ba4ad"
VIEW = "qwen35_4b_it__arditi_allpos"


def _assign(name):
    for node in ast.parse(BUILDER.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return node.value
    raise AssertionError(f"{name} not found in the builder")


def _literal(name):
    return ast.literal_eval(_assign(name))


def _allow_list():
    """`PETRI_AGG_EXTRA`'s KEYS are f-strings (the producer paths), so the dict is not a literal.

    Only the values matter here — {view: [cells]} per producer file — so walk the AST and evaluate those,
    leaving the f-string keys alone. Using literal_eval on the whole node raises ValueError, which is how
    this helper came to exist."""
    out = []
    for value in _assign("PETRI_AGG_EXTRA").values:
        out.append(ast.literal_eval(value))
    return out


def test_the_selected_audits_run_key_is_declared():
    keys = _literal("PETRI_AGG_EXTRA_RUN_KEYS")
    assert keys[(VIEW, "arditi_allpos")] == SELECTED


def test_the_certification_roots_run_key_is_NOT_what_we_declare():
    """The known-bad. If these ever coincide the guard is inert."""
    keys = _literal("PETRI_AGG_EXTRA_RUN_KEYS")
    assert keys[(VIEW, "arditi_allpos")] != CERT, "the declared key is the certification root's — wrong audit pinned"


def test_every_allow_listed_arditi_allpos_cell_has_a_declared_run_key():
    """Omission is the realistic failure: a view added to the allow-list without a key entry."""
    keys = _literal("PETRI_AGG_EXTRA_RUN_KEYS")
    for views in _allow_list():
        for view, cells in views.items():
            for cell in cells:
                if cell == "arditi_allpos" or view.endswith("__arditi_allpos"):
                    assert (view, cell) in keys, f"{view}/{cell} is allow-listed with no run key pinned"


def test_the_sidecars_none_is_NOT_allow_listed():
    """Folding the sidecar's `none` would hit the two-producers refusal, correctly.

    `qwen35_4b_it/none` already exists in the canonical petri_agg as a different, lens-era audit. The arm's own
    same-standup `none` is not needed here: it reaches the pairing through the per-scenario stats file
    (PETRI_ARMCELL_FILE), which is where every arm cell's anchor is read from. Allow-listing it would claim a
    cell that is already claimed."""
    for views in _allow_list():
        for view, cells in views.items():
            if view.endswith("__arditi_allpos"):
                assert "none" not in cells, "the sidecar's none must not be folded — it collides with the canonical one"
                assert "arditi_allpos" in cells


def test_a_sidecar_tag_is_mapped_to_a_REAL_export_view():
    """A sidecar tag is not a view, and the emitter writes `view` straight onto the row.

    Left under the tag, cells are admitted and then never emitted, because the roster loop iterates views and no
    view of that name exists — which is exactly what the first build after the allow-list landed did: both cells
    admitted, no refusal, zero petri rows. The llama relu_native cells emit only because `llama8b` IS a view name."""
    m = _literal("PETRI_AGG_EXTRA_VIEW")
    assert m[VIEW] == "qwen35_4b_it", "the sidecar tag is not mapped to its export view"
    src = BUILDER.read_text()
    i = src.index("PETRI_AGG_EXTRA_ADDED = []")
    body = src[i:i + 2500]
    assert "PETRI_AGG_EXTRA_VIEW.get(" in body, "the fold does not consult the tag→view map"


def test_folding_under_the_export_view_still_refuses_a_double_claim():
    """The two-producers guard must follow the record to its MAPPED view, not stay on the tag —
    otherwise mapping would route around the very refusal that protects the canonical cells."""
    src = BUILDER.read_text()
    i = src.index("PETRI_AGG_EXTRA_ADDED = []")
    body = src[i:i + 2500]
    j = body.index("PETRI_AGG_EXTRA_VIEW.get(")
    assert "REFUSING" in body[j:j + 700] and "already" in body[j:j + 700], (
        "no double-claim refusal on the mapped view")


@pytest.mark.parametrize("marker", [
    "carries run_key",                       # the mismatch refusal
    "has no entry in",                       # the omission refusal
])
def test_both_refusals_are_present_and_name_the_reason(marker):
    src = BUILDER.read_text()
    i = src.index("PETRI_AGG_EXTRA_RUN_KEYS")
    body = src[i:i + 3000]
    assert "REFUSING" in body and marker in body, f"the {marker!r} refusal is missing from the consumption block"


def test_the_guard_does_not_fire_on_the_llama_cells_that_predate_it():
    """The relu_native llama cells have no run-key entry and must stay admissible — the omission refusal is
    scoped to arditi_allpos, not to every allow-listed cell."""
    keys = _literal("PETRI_AGG_EXTRA_RUN_KEYS")
    llama = [c for views in _allow_list() for v, cells in views.items() if v == "llama8b" for c in cells]
    assert llama, "the llama allow-list vanished"
    for c in llama:
        assert ("llama8b", c) not in keys
        assert c != "arditi_allpos"


# --- the duplicate cross-check, re-keyed on the same principle -------------------------------------------------

def test_the_duplicate_check_is_keyed_on_the_RUN_KEY_not_just_view_and_cell():
    """Two files filing one (view, cell) from DIFFERENT audits are siblings, not duplicates.

    `none`'s intervention_hash is identical everywhere — none is none — so a check keyed on (view, cell)
    called the wave's lens-era qwen35_4b_it/none and this lane's native one a contradiction and warned.
    They are two honest measurements on two serving stacks; the gap between them IS the native-vs-lens
    difference. The run key separates them because it is content-addressed over the whole harness config."""
    src = BUILDER.read_text()
    i = src.index("_dup = []")
    body = src[i:src.index('CHECKS["petri_per_scenario_duplicates"]')]
    assert "PETRI_SIBLING_AUDITS" in body, "the sibling case is not distinguished from a duplicate"
    assert "petri_run_key" in body, "the duplicate check does not consult the run key"


def test_a_same_audit_disagreement_STILL_warns():
    """The corruption the check was written to catch must survive the re-keying: one audit filed twice
    with different scores is a real problem and must still reach _dup."""
    src = BUILDER.read_text()
    i = src.index("_dup = []")
    body = src[i:src.index('CHECKS["petri_per_scenario_duplicates"]')]
    assert "_dup.append(" in body, "the same-audit disagreement path was removed"
    # the sibling branch must be the conditional one, with _dup as the else
    assert body.index("PETRI_SIBLING_AUDITS.append(") < body.index("_dup.append("), (
        "the sibling branch must guard the _dup append, not follow it")


def test_the_sibling_pairs_are_SURFACED_not_silently_dropped():
    """An unreported reconciliation is how a real divergence hides. Siblings get their own check line."""
    src = BUILDER.read_text()
    assert 'CHECKS["petri_sibling_audits"]' in src
    j = src.index('CHECKS["petri_sibling_audits"]')
    assert "EXPECTED" in src[j:j + 400], "the sibling check does not say the difference is expected"

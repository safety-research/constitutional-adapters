"""The three PROTECTED SMOKE SURFACES stay BYTE-IDENTICAL to main:
``scripts/petri_steered_smoke.py``, its
sbatch, and ``scripts/modal/petri_agentloop_control.py`` are the Petri
steered-smoke gate that re-validates every ``proxy.py`` change; a lane
that edits them would be grading its own homework. Their sha256 at main
``d1eb1064`` is pinned here; any change is a deliberate, visible act that
moves these literals in the same commit.
"""

import hashlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: sha256 at main d1eb1064 (2026-09-07)
FROZEN = {
    "scripts/petri_steered_smoke.py":
        # the two object defaults spell the cluster's shared root as $SHARED, expanded after parsing; later the
        # objects have no default at all: --static-r-object / --alphasteer-object, else the site's SISTEER_PETRI_SMOKE_STATIC_R_OBJECT /
        # SISTEER_PETRI_SMOKE_ALPHASTEER_OBJECT, else a refusal by name, so the repository spells no live object directory (the
        # objects served, their adapters and the gate's logic unchanged)
        "6436d7eb27e334b87bb297ad93df464c284c4152478050005d31723f5ce4f373",
    "scripts/sbatch/petri_steered_smoke.sbatch":
        # the one `export NCCL_SOCKET_IFNAME=<interface>` line removed, as from every job script: the socket interface is
        # the site's nccl.conf's, and an exported name the node lacks fails the first collective (the gate's logic unchanged);
        # later the two object paths come from the job environment's SISTEER_PETRI_SMOKE_STATIC_R_OBJECT /
        # SISTEER_PETRI_SMOKE_ALPHASTEER_OBJECT, each refused by name before the venv bootstrap when missing, in place of two
        # literal live directory paths (the command's flags, their order and every other token unchanged)
        "55f065ba36bf892a92393cba3080a78edba7669a086d65dd7fb567084bea27f0",
    "scripts/modal/petri_agentloop_control.py":
        # native-serving-default (2026-09-13): the explicit `--serving-path lens` on its capctrl argv — the
        # standup's default became the native engine, and this gate validates the LENS proxy path
        "20bd4770215ae167ac9265cd2eca2459a18abdc601fabe2d00ac6cc9f2075cfa",
}


@pytest.mark.parametrize("rel,sha", sorted(FROZEN.items()))
def test_protected_surface_is_byte_identical_to_main(rel, sha):
    path = REPO / rel
    assert path.is_file(), f"{rel} is missing — the protected surface moved"
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    assert got == sha, (f"{rel} changed (sha256 {got} != pinned {sha}) — the Petri "
                        "steered-smoke surfaces are frozen; a deliberate change moves "
                        "this pin in the same commit")

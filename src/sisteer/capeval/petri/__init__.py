"""Petri Bloom misalignment-audit modules.

Phase 1 lands ``defense_cells`` (the cell-registry adapter and pairing
semantics). Phase 2 adds ``scenarios``, ``petri_stats``, ``petri_report``
against the contracts frozen here and in ``sisteer.capeval.manifest``.

Import discipline: everything under this package is capeval-native and
CPU-offline — no ``inspect_ai`` / ``petri_bloom`` imports (those are
launch-phase deps of the grading host), no torch beyond what capeval's
base lane already carries.
"""

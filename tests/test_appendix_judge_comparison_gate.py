"""The judge-robustness figure of appendix D (`fig:app_judges`, figA_judge_comparison_a and _b) where the landing gate collects it.

The gate runs tests/ only, so this file loads paper/tests/test_appendix_judge_comparison.py and runs its core checks here, unchanged:
  * the committed readings (data/appendix_judge_comparison.json) match their builder's provenance (the committed builder's sha256, its
    code commit, judging policies, pinned head 537565 and strength-lane sidecars, the payload hash) and the registry rows after the
    pinned head enter no reading; the payload hash, the registry copy's sha256 and the delivered value files' sha256s are the pinned
    frozen ones; the build found every mark equal to the delivered figure's value files;
  * the draw reproduces the delivered figure: each part's decoded PNG pixels hash to the delivered PNG's;
  * the numbers the appendix paragraph quotes are recomputed from the file;
  * the notebook draws both parts from the file in the cells after e6f1b5a9, and lists them in PAPER_FIGURES before the generator comparison;
  * no path shape or private name in the shipped files (this file and the paper test included), and the publication scan, without its
    allowlist, finds nothing in them.
"""
from __future__ import annotations

import importlib.util
import pathlib

_PATH = pathlib.Path(__file__).resolve().parents[1] / "paper" / "tests" / "test_appendix_judge_comparison.py"
_spec = importlib.util.spec_from_file_location("paper_tests_appendix_judge_comparison", _PATH)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

# the module's fixtures, so the tests below find them here
jc, builder, _no_repo_writes = _m.jc, _m.builder, _m._no_repo_writes

test_the_file_matches_its_builders_provenance = _m.test_the_file_matches_its_builders_provenance
test_the_readings_are_those_at_the_pinned_head = _m.test_the_readings_are_those_at_the_pinned_head
test_the_marks_equal_the_delivered_figures_values = _m.test_the_marks_equal_the_delivered_figures_values
test_the_draw_reproduces_the_delivered_png_pixel_for_pixel = _m.test_the_draw_reproduces_the_delivered_png_pixel_for_pixel
test_the_numbers_the_appendix_text_quotes = _m.test_the_numbers_the_appendix_text_quotes
test_the_cells_sit_after_the_all_model_figures_and_draw_both_parts_from_the_file = \
    _m.test_the_cells_sit_after_the_all_model_figures_and_draw_both_parts_from_the_file
test_no_path_or_private_name_in_the_shipped_files = _m.test_no_path_or_private_name_in_the_shipped_files
test_the_publication_scan_finds_nothing_in_the_shipped_files = _m.test_the_publication_scan_finds_nothing_in_the_shipped_files

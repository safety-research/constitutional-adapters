"""S2RL: the LeCun-init ReLU refit EVAL wave was its own view `s2rl` — the
`s<N>rl` spelling the phase-view grammar admits (the refit lane of phase N's ReLU line, as `d<dose>_relul` is the dose lane's). No model of the
tree declares such a view now; what stays pinned here is the GRAMMAR: every copy of PHASE_VIEW_RE admits `s<N>rl` (and still refuses s2rlp /
s2l / s2lr / s2rll), and the three STEERED_WAVE_VIEW_RE copies read `s<N>rl` as a steered wave by name (no unsteered leg)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402

for p in (REPO / "reproduce" / "lib", REPO / "scripts" / "sdfbase" / "full22"):
    sys.path.insert(0, str(p))
import models as M  # noqa: E402
import f22_roster as FR  # noqa: E402

VIEW = "s2rl"
GRAMMAR_COPIES = ("reproduce/lib/models.py", "scripts/sdfbase/make_full22_configs.py", "scripts/sdfbase/full22/sampled/make_sampled_configs.py",
                  "scripts/sdfbase/full22/misalign/make_misalign_manifest.py", "scripts/sdfbase/gsm8k/g8k_cells.py",
                  "scripts/sdfbase/full22/static_padding/make_padding_configs.py", "scripts/sdfbase/seedrep/grade_full22_native.py")
STEERED_COPIES = ("reproduce/lib/models.py", "scripts/sdfbase/full22/f22_roster.py", "scripts/sdfbase/full22/misalign/make_misalign_manifest.py")


def test_every_grammar_copy_admits_the_refit_view_and_the_steered_wave_predicate_holds():
    for rel in GRAMMAR_COPIES:
        m = re.search(r'^PHASE_VIEW_RE = re\.compile\(r"(.*)"\)$', (REPO / rel).read_text(), re.M)
        assert m and "s[0-9]+rl" in m.group(1), rel
        rx = re.compile(m.group(1))
        assert rx.match("s2rl") and rx.match("s12rl") and rx.match("s2r") and rx.match("u0"), rel
        for bad in ("s2rlp", "s2l", "s2lr", "s2rll", "s2rr", "srl"):
            assert not rx.match(bad), (rel, bad)
    for rel in STEERED_COPIES:
        m = re.search(r'^STEERED_WAVE_VIEW_RE = re\.compile\(r"(.*)"\)$', (REPO / rel).read_text(), re.M)
        assert m and m.group(1) == r"^s[0-9]+rl?$", rel
    # by NAME, with no table row behind it: the live predicates agree with the text
    assert M.steered_wave_view(VIEW) and FR.steered_wave_view(VIEW) and not M.steered_wave_view("u0") and not FR.steered_wave_view("u0")
    assert M.PHASE_VIEW_RE.match(VIEW) and not any(VIEW in M.views(k) for k in M.load_models())      # no model of the tree declares it

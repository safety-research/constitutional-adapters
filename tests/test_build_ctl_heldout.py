"""scripts/sdfbase/build_ctl_heldout.py: the control held-out pool is
EXACTLY the control corpus minus the ctl-100k draw (partition: disjoint,
covering), emitted through build_pools' own renderer/draw/cell machinery,
with the remainder relation stamped in its manifest; a drifted corpus file
or a foreign draw refuses. Synthetic corpus + fake counting rig (the
tests/test_sdfbase_pools.py conventions)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "sdfbase"))

import build_ctl_heldout as bch  # noqa: E402
import build_heldout_split as hs  # noqa: E402
import build_pools as bp  # noqa: E402

try:  # the corpus repo's renderers (cross-repo import by path)
    hs.generator_tokens()
except hs.SplitError as _e:  # pragma: no cover — machines without the repo
    pytest.skip(f"corpus repo renderers unavailable: {_e}",
                allow_module_level=True)


class FakeRig:
    provenance = {"fake": True}

    def counts(self, text: str) -> dict[str, int]:
        n = len(text.split())
        return {k: n for k in hs.TOK_NAMES}

    def boundary_offsets(self, name: str, text: str) -> list[int]:
        return [m.end() for m in re.finditer(r"\S+", text)]


def words(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i:04d}" for i in range(n))


def adv_rec(p, t, s, *, system_words=4, user_words=6, assistant_words=10):
    pid, tid = f"p{p:02d}", f"p{p:02d}-t{t:02d}"
    sid = f"{tid}-s{s:02d}"
    sys_text = words(f"as{p}x{t}x{s}s", system_words)
    user = words(f"au{p}x{t}x{s}u", user_words)
    assistant = words(f"aa{p}x{t}x{s}a", assistant_words)
    n = len(("\n\n".join((sys_text, user, assistant))).split())
    return {"system": sys_text, "user": user, "assistant": assistant,
            "principle_id": pid, "prompt_type_id": tid, "scenario_id": sid,
            "lineage": {"custom_id": f"adv-{sid}"},
            "token_counts": {k: n for k in hs.TOK_NAMES}}


def sto_rec(c, u, k, y, *, n_words=40):
    cid, uid = f"c{c:02d}", f"c{c:02d}-u{u:02d}"
    kid = f"{uid}-k{k:02d}"
    return {"text": words(f"st{c}x{u}x{k}x{y}w", n_words), "category_id": cid,
            "subcategory_id": uid, "call_id": kid,
            "lineage": {"custom_id": f"sto-{kid}-y{y}"},
            "token_counts": {k2: n_words for k2 in hs.TOK_NAMES}}


def write_jsonl(path: Path, records) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("ctlheld")
    # a tiny PRODUCTION corpus + split (build_pools needs them even for the
    # control lane) and a control corpus of 4 advice + 30 stories
    advice = [adv_rec(p, 0, s, system_words=8, user_words=20,
                      assistant_words=40 + (7 * s) % 5)
              for p in range(2) for s in range(6)]
    stories = [sto_rec(c, u, k, y, n_words=120 + (5 * c + 3 * u + k) % 7)
               for c in range(2) for u in range(2) for k in range(2)
               for y in range(2)]
    ap = write_jsonl(root / "advice.jsonl", advice)
    sp = write_jsonl(root / "stories.jsonl", stories)
    split_path = root / "split.json"
    split_path.write_text(json.dumps(hs.build_split(ap, sp, eval_n=2)))
    ctl_adv = [adv_rec(9, 1, s, system_words=8, user_words=20,
                       assistant_words=60) for s in range(4)]
    ctl_sto = [sto_rec(7, u, k, y, n_words=150 + (11 * u + 3 * k + y) % 9)
               for u in range(5) for k in range(3) for y in range(2)]
    cap = write_jsonl(root / "ctl_advice.jsonl", ctl_adv)
    csp = write_jsonl(root / "ctl_stories.jsonl", ctl_sto)
    pools = root / "pools"
    budget = 2_000  # well inside the ~5k-word control corpus
    bp.build_pools(pools, split_path, advice_final=ap, stories_final=sp,
                   lanes=("control",), mix_cells=[], adv_cells=[],
                   ctl_cells=[("constv2ctl-100k", budget)],
                   control_advice=cap, control_stories=csp, rig=FakeRig(),
                   generator={"fake": True})
    return root, pools, cap, csp, ctl_adv, ctl_sto


def test_remainder_is_the_complement_of_the_draw(world):
    root, pools, cap, csp, ctl_adv, ctl_sto = world
    m = bch.build_control_heldout(pools, control_advice=cap, control_stories=csp,
                                  rig=FakeRig(), generator={"fake": True})
    ctl = json.loads((pools / "pairs_constv2ctl-100k.manifest.json").read_text())
    used = set(ctl["doc_ids_in_order"])
    held = set(m["doc_ids_in_order"])
    all_ids = {bp.CTL_ID_PREFIX + r["lineage"]["custom_id"]
               for r in ctl_adv + ctl_sto}
    assert held and used
    assert not (held & used), "held-out overlaps the trained draw"
    assert held | used == all_ids, "remainder + draw must cover the corpus"
    assert m["lane"] == "control_heldout" and m["budget"] is None
    assert m["truncation"] is None and m["boundary_skips"] == []
    assert m["remainder_of"]["cell"] == "constv2ctl-100k"
    assert m["remainder_of"]["n_excluded"] == len(used)
    assert m["remainder_of"]["n_control_corpus_docs"] == len(all_ids)
    assert m["corpus_shas"] == ctl["corpus_shas"]
    # rows: the pool file carries exactly the held-out ids, ctl- prefixed,
    # with the pair-row schema the trainers/scorers read
    rows = [json.loads(l) for l in
            (pools / "pairs_constv2ctl-heldout.jsonl").read_text().splitlines()]
    assert {r["doc_id"] for r in rows} == held
    assert all(r["doc_id"].startswith("ctl-") for r in rows)
    assert {r["instruction_id"] for r in rows} == {"constv2ctl-heldout"}
    assert all(r["neg"] == "" and r["pos"] for r in rows)
    fam = {r["family"] for r in rows}
    assert fam <= {"advice", "stories"}
    # the truncated boundary doc (if any) is counted as USED, never held out
    t = (ctl.get("truncation") or {}).get("doc_id")
    if t:
        assert t in used and t not in held
        assert m["remainder_of"]["truncated_boundary_doc_counted_as_used"] == t
    # official tokens: whole docs, sum of the remainder
    assert m["official_tokens"] == sum(r["official_tokens"] for r in rows)
    # idempotent: a rebuild is byte-identical
    before = (pools / "pairs_constv2ctl-heldout.jsonl").read_bytes()
    bch.build_control_heldout(pools, control_advice=cap, control_stories=csp,
                              rig=FakeRig(), generator={"fake": True})
    assert (pools / "pairs_constv2ctl-heldout.jsonl").read_bytes() == before


def test_refuses_drifted_corpus_and_missing_family(world, tmp_path):
    root, pools, cap, csp, ctl_adv, ctl_sto = world
    drifted = write_jsonl(tmp_path / "drift.jsonl", ctl_sto[:-1])
    with pytest.raises(bp.SamplerError, match="not the corpus"):
        bch.build_control_heldout(pools, control_advice=cap,
                                  control_stories=drifted, rig=FakeRig(),
                                  generator={"fake": True})
    with pytest.raises(bp.SamplerError, match="every family"):
        bch.build_control_heldout(pools, control_advice=cap, rig=FakeRig(),
                                  generator={"fake": True})
    with pytest.raises(bp.SamplerError, match="missing"):
        bch.build_control_heldout(tmp_path, control_advice=cap,
                                  control_stories=csp, rig=FakeRig(),
                                  generator={"fake": True})


def test_cli_refuses_without_inputs(world, capsys):
    root, pools, *_ = world
    rc = bch.main(["--pools", str(pools), "--allow-dirty"])
    assert rc == 2 and "REFUSED" in capsys.readouterr().err

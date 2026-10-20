"""Stage 07's expected intervention hashes, derived on THIS machine (the slow opt-in cross-check modelled on
tests/test_object_identity.py's real-Arditi derivation): the worked example serves the paper's COMMITTED Qwen3.5-4B objects, so every fitted
cell's HF post-load intervention hash must equal the one the SAME bytes already carry under the paper's seed slugs in
contracts.EXPECTED_INTERVENTION_HASHES — the table: token -> (seed slug from objects/DEMO.json's training seed, the published hash; the
ReLU difference under the PAIRED slug). `object_identity.derive_request` runs the preflight's own derivation against the committed bytes laid
out as the demo bank (<bank>/<token>/{object.pt, sidecar}) on the pinned Qwen3.5-4B meta shell; a disagreement is a FINDING, never a pin
(the table gains no demo key, and 07's preflight phase prints DIFFERS for it). Skipped where torch or the pinned Qwen config is not here."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPRO / "lib"))
import models as MT  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402

KEY = "qwen35_4b_demo"
PAPER_KEY = "qwen35_4b"
#: the reference table — literal pairs (token -> (seed slug, published hash)); the derivation below must reproduce every value
D8 = {"constv2ctl2-full__psr_ce_all": ("qwen35_4b-f22s3", "21bba17d6bff3e49"), "constv2-mix-full__lora_r4_ce": ("qwen35_4b-f22s3", "a785e725a5308e02"),
      "arditi_allpos": ("qwen35_4b-f22s1", "01f67b09e325976b"), "constv2-mix-full__psr_ce_all": ("qwen35_4b-f22s3", "98df2bad947576b7"),
      "constv2ctl2-full__lora_r4_ce": ("qwen35_4b-f22s3", "466f4c765d7a1756"), "constv2-mix-full__lora_r4_ce_minus_ctl": ("qwen35_4b-f22s3", "3cbf5fe2fe2ceea1"),
      "constv2-mix-full__psr_ce_all_minus_ctl": ("qwen35_4b-f22s3p", "f79e44339ae17129")}


def test_the_d8_table_is_the_contracts_published_hashes_under_the_seed_each_committed_object_came_from():
    """The CPU half (always on): the table is read off contracts.EXPECTED_INTERVENTION_HASHES and objects/DEMO.json's seeds, never typed; the
    demo key is absent from the table (tests/test_front_door_dry_run.py pins it too)."""
    demo = {e["paper_label"]: e for e in json.loads((REPO / "objects" / "DEMO.json").read_text())["objects"]}
    m = MT.model(KEY)
    view = MT.baseline_view(KEY)
    tokens = [t for _c, t in MT.launcher_cells(KEY, view) if C.parse_launcher_cell(t).kind == C.CELL_KIND_FITTED]
    assert sorted(tokens) == sorted(D8)
    by_tok = {MT.roster_cell_id(m, c): c for c in m["views"][view]["cells"] if c != "none" and not c.startswith("prompted__")}
    for tok, (slug, h) in D8.items():
        o = MT.object_of(m, by_tok[tok]); e = demo[str(o.get("paper_label") or by_tok[tok])]
        seed = e.get("seed") or 1
        want = f"{PAPER_KEY}-f22s{seed}" + ("p" if o.get("object_kind") == "composed" else "")
        assert slug == want and C.EXPECTED_INTERVENTION_HASHES[slug][tok] == h and C.expected_intervention_hash(slug, tok) == h, tok
    assert KEY not in C.EXPECTED_INTERVENTION_HASHES


def _pinned_qwen_available() -> bool:
    try:
        import torch  # noqa: F401
        from transformers import AutoConfig
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        AutoConfig.from_pretrained("Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _pinned_qwen_available(), reason="torch and the pinned Qwen3.5-4B config (local HF cache) are needed for the derivation")
def test_the_committed_objects_derive_to_the_published_hashes_through_the_preflights_own_derivation(tmp_path):
    from sisteer.capeval.tinycap import object_identity as oi
    import importlib.util
    spec = importlib.util.spec_from_file_location("_derivation_import_objects", REPO / "reproduce" / "tools" / "import_objects.py")
    IO = importlib.util.module_from_spec(spec); spec.loader.exec_module(IO)
    view = MT.baseline_view(KEY)
    plan = IO.Plan(KEY, views=[view], repo=REPO)
    launcher = dict(MT.launcher_cells(KEY, view))
    bank = tmp_path / "bank"; bank.mkdir()
    pins = {}
    for r in plan.rows:
        tok = launcher[r["view_cell"]]
        src = REPO / "objects" / r["dir"]
        if hashlib.sha256((src / "object.pt").read_bytes()).hexdigest() != r["object_sha256_expected"]:
            pytest.skip(f"the committed bytes of {tok} are not the manifest's (a sparse checkout?)")
        (bank / tok).mkdir()
        (bank / tok / "object.pt").symlink_to(src / "object.pt")
        (bank / tok / r["identity_sidecar"]).symlink_to(src / r["identity_sidecar"])
        pins[tok] = r["object_sha256_expected"]
    body = (REPO / C.bcbw_config_path(PAPER_KEY, "s1")).read_text()
    body = body[body.index("# -- everything below:"):]
    body = re.sub(r"^defenses_dir: .*$", f"defenses_dir: {bank}", body, count=1, flags=re.M)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("explicit_cells:\n- none\n" + "".join(f"- {t}\n" for t in D8) + "\n" + body)
    targets = [{"cell": tok, "si_id": None if tok == "arditi_allpos" else tok.split("__", 1)[0], "arm": tok if tok == "arditi_allpos" else tok.split("__", 1)[1],
                "expected": C.EXPECTED_INTERVENTION_HASHES[slug][tok], "pin_sha256": pins[tok]} for tok, (slug, _h) in D8.items()]
    rep = oi.derive_request({"model": "Qwen/Qwen3.5-4B", "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "config": str(cfg), "config_overrides": {},
                             "objects_dir": str(bank), "targets": targets})
    by_cell = {t["cell"]: t for t in rep["results"]}
    findings = []
    for tok, (slug, h) in D8.items():
        t = by_cell[tok]
        if t.get("error") or t.get("intervention_hash") != h or t.get("pin_match") is not True:
            findings.append((tok, slug, h, t.get("intervention_hash"), t.get("pin_match"), t.get("error")))
    assert not findings, f"the committed objects do not derive to the published hashes under their seed slugs (a FINDING, not a pin): {findings}"
    assert rep["ok"] is True

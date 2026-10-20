#!/usr/bin/env python3
"""File the two new cells' best-of-N rows into the export's qwen35_4b_it block (file_into_prompted_export.py's pattern).

UNVERIFIED TARGET -- confirm before --write: qwen35_4b_it's OBJECT cells are read by
build_export.py from transfer_stats_full22_all.json (TS_ALL, line 163); the constitution cells went
into transfer_stats_full22_prompted.json because the builder merges ONLY prompted__* cells from that file (line 202). The two LoRA cells
are object cells, so the `bon` block has to land in TS_ALL's models.qwen35_4b_it.cells[<cell>] (a NEW cell entry each; label/group copied
from the sibling LoRA cell constv2-mix-full__lora_r4_ce). The aggregate is filed at its HIGHEST rung (decode_cap_tokens=4096) with
final_rung_caps BESIDE it (the builder's line 1363-1388 rule turns that into the asr_bon_cap4096 row; a null cap cannot be ranked).

  file_into_export.py            # preview only
  file_into_export.py --write    # backs up TS_ALL to .pre-bonq-<ts> and writes
"""
import datetime, glob, json, math, os, shutil, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from site_paths import site_path  # noqa: E402
# the run-state directory under the site's state root ($STATE); an unset one ends the run by name
RS = str(site_path("$STATE/romp/candidate-si-refit/run-state/full22", "RS")); D = f"{RS}/bon_qwenctl"
EXPORT = f"{RS}/transfer_stats_full22_all.json"
VIEW = os.environ.get("BONQ_VIEW", "qwen35_4b_it")
DSFX = "" if VIEW == "qwen35_4b_it" else "_" + VIEW   # knob (2026-09-14): the gemma-it lanes file under their own view; default = qwen exactly
CELLS = {"constv2ctl2-full__lora_r4_ce": "control LoRA (batch-64 itdoc64)",
         "constv2-mix-full__lora_r4_ce_minus_ctl": "LoRA subtraction: constitution minus control (batch-64 itdoc64)"}
SIBLING = "constv2-mix-full__lora_r4_ce"


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


agg = json.load(open(f"{RS}/grades_bonq_{VIEW}/{VIEW}/two_si_bon_aggregate.json"))
deltas = json.load(open(f"{D}/paired_deltas{DSFX}.json")) if os.path.exists(f"{D}/paired_deltas{DSFX}.json") else {}
d = json.load(open(EXPORT))
m = d["models"][VIEW]; sib = m["cells"][SIBLING]
new = {}
for cid, label in CELLS.items():
    c = agg["cells"][cid]; am = c["attack_metrics"]; comp = c["completion"]["bon"]
    asr = float(am["asr_at_n"]["75"]); k = int(round(asr * comp["n_goals"])); lo, hi = wilson(k, comp["n_goals"])
    fr = comp["finish_reasons"]; rows = sum(fr.values())
    # the final-rung mix is matched by CELL across every summary the lane wrote, not by a fixed lane-id list: the bonctl climb was re-run
    # standalone under lane id qwen35_4b_it_ctlmodalr2 (README 07:57Z), so a list naming only qwen35_4b_it_ctl would have filed that cell
    # at rung 512 with final_rung_caps {"512": 7500} — a 6,080-slot climb silently dropped. Two summaries for one cell is a refusal.
    # ...and scoped to THIS VIEW (2026-09-15): the qwen and gemma-it lanes measure the SAME two cell ids, so once the gemma
    # lanes landed, every summary glob matched two files per cell and this guard refused both gemma cells. Scoping by the
    # summary's own `lane` label (which is always "<view>_<short>...") keeps the guard's real job -- two summaries for one
    # cell WITHIN a view is still a refusal, which is the 6,080-slot-climb case it was written for -- while letting the two
    # views coexist. Matching on the lane prefix, not a lane-id list, so a re-run suffix like _gctlr2 still matches.
    mix = {}
    for p in sorted(glob.glob(f"{D}/climb/final_rung_summary_*.json")):
        s = json.load(open(p))
        if s["cell_id"] == cid and str(s.get("lane") or "").startswith(VIEW + "_"):
            if mix:
                raise SystemExit(f"{cid}: more than one final_rung_summary names this cell ({mix_src} and {p}) — refusing to pick")
            mix, mix_src = s["by_rung"], p
    if not mix:
        raise SystemExit(f"{cid}: no climb/final_rung_summary_*.json names this cell — the merged final rung is the graded corpus; refusing to file it as a 512-only store")
    if sum(mix.values()) != rows:
        raise SystemExit(f"{cid}: final-rung mix {mix} covers {sum(mix.values())} slots but the aggregate graded {rows} rows — not the same corpus; refusing")
    # truncated_share, the siconst filer's own key (never folded into at_cap_share): the share of ALL slots still at the cap at the 4,096 rung
    # (bon_siconst/file_deltas.py hard-coded 10/7500 for qwen35_4b_it's constitution cell). Read from the merged final-rung corpus the summary
    # names, which also proves the graded corpus IS that corpus: its finish reasons must equal the aggregate's completion stamp and its rung
    # mix must equal the summary's.
    corpus = json.load(open(mix_src))["out"]
    if not os.path.exists(corpus):
        raise SystemExit(f"{cid}: the final-rung corpus named by {mix_src} is missing: {corpus}")
    c_fr, c_rung, c_trunc4096 = {}, {}, 0
    for line in open(corpus):
        if not line.strip():
            continue
        r = json.loads(line)
        c_fr[r["finish_reason"]] = c_fr.get(r["finish_reason"], 0) + 1
        rg = str(r["final_rung_cap"]); c_rung[rg] = c_rung.get(rg, 0) + 1
        if rg == "4096" and r["finish_reason"] == "length":
            c_trunc4096 += 1
    if c_fr != fr or c_rung != {str(k): v for k, v in mix.items()}:
        raise SystemExit(f"{cid}: corpus {corpus} finish_reasons {c_fr} / rung mix {c_rung} != aggregate {fr} / summary {mix} — not the graded corpus; refusing")
    truncated_share = c_trunc4096 / rows
    row = {"k": k, "n": comp["n_goals"], "p": asr, "lo": lo, "hi": hi, "asr_reported": asr, "asr_at_n": am["asr_at_n"],
           "mean_normalized_auc": am.get("mean_normalized_auc"), "namespace": "bonq", "lens_reduction_mode": c.get("lens_reduction_mode", "stacked"),
           "tier": "bon", "decode_cap_tokens": 4096 if mix.get("4096") else 512, "final_rung_caps": mix or {"512": rows},
           "at_cap_share": fr.get("length", 0) / rows if rows else None, "truncated_share": truncated_share,
           "rung_note": "one text per slot at its own final rung; the climb REPLACES a slot rather than adding a reading, so the corpus is 7,500 texts for 7,500 slots",
           "completion": comp, **{k2: v for k2, v in deltas.get(f"{VIEW}/{cid}", {}).items() if k2.startswith("delta") or k2.startswith("discordant") or k2 in ("asr_unsteered",)}}
    new[cid] = {"bon": row, "label": label, "group": sib.get("group"), "view": sib.get("view", m.get("view")), "on_overlay": sib.get("on_overlay", False),
                "lens_reduction_mode": sib.get("lens_reduction_mode", "stacked"), "staged_object": {"constv2ctl2-full__lora_r4_ce": "itdoc64__constv2ctl2-full__lora_r4_ce",
                "constv2-mix-full__lora_r4_ce_minus_ctl": "derived_itdoc64__constv2-mix-full__lora_r4_ce_minus_ctl"}[cid], "source": "run-state/full22/bon_qwenctl (2026-09-14)"}
    print(f"  {cid:<44} asr@75={asr:.3f} k={k}/{comp['n_goals']} rung={row['decode_cap_tokens']} mix={row['final_rung_caps']} delta={row.get('delta')}")
    if cid in m["cells"]:
        print(f"    NOTE: {cid} already exists in {EXPORT} models.{VIEW}.cells -- would be REPLACED (bon block only merged in)")
if "--write" not in sys.argv:
    print("\npreview only; pass --write to file (backs up first). CONFIRM the target file first (see docstring).")
    sys.exit(0)
ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
shutil.copy2(EXPORT, f"{EXPORT}.pre-bonq-{ts}")
for cid, entry in new.items():
    if cid in m["cells"]:
        m["cells"][cid]["bon"] = entry["bon"]
    else:
        m["cells"][cid] = entry
d.setdefault("meta", {})["bonq_filed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
# atomic: the export refresher unit reads this file whenever a grade aggregate lands; a torn read of a half-written 2.4 MB join would
# fail its build (or, worse, be read whole by a slower reader). Write beside it, fsync, then rename into place.
tmp = f"{EXPORT}.tmp-bonq-{ts}"
with open(tmp, "w") as fh:
    json.dump(d, fh, indent=1); fh.flush(); os.fsync(fh.fileno())
os.replace(tmp, EXPORT)
print(f"\nwrote {EXPORT} atomically (backup {EXPORT}.pre-bonq-{ts})")

"""`legacy_glm` adapter: the GLM-4.6V-Flash descope ruling and the lane's own report files as artifacts.

GLM rows arrive through every other adapter (`glm46vflash` views in the r2 export, the grades roots, the Petri
files); each of those runs carries `status_ruling_id = ruling:r2:descope:model:glm46vflash`. This adapter emits THAT
ruling (action `descope`, scope the model; the rulings seed adopts the id) so the runs can fold, and registers the
files nothing else names as `artifact(kind=report)`: the 117 errored goal-outcomes' recovery comparison
(`glm_errored117/comparison.json`, whose rows the export carries as `recovered_*` metrics), the controls table
(`controls_glm/controls_table_glm.json`), the join and Petri stats. `model.paper_status = descoped` is the
model seed's row; nothing else is special about GLM.
"""
from __future__ import annotations

import os
from typing import Dict, Iterator, List, Optional, Sequence

from . import lineage_common as LC
from .base import AdapterContext, AdapterRefusal, Input, RecordBatch, SourceStateRow, check_input, register
from .runner import source_id_for

NAME = "legacy_glm"
MODEL_ID = "glm46vflash"
FILES = ("glm_errored117/comparison.json", "controls_glm/controls_table_glm.json", "transfer_stats_full22_glm.json", "petri_stats/petri_full22_glm.json")


class LegacyGlmAdapter:
    name = NAME
    covers_sources = ("legacy_glm",)
    rng_scope = "none"
    allow_symlinks = False

    def __init__(self, rs: str = LC.RS_DEFAULT, seeds: Optional[LC.Seeds] = None):
        self.rs = LC.expand(rs)
        self.roots = (os.path.join(self.rs, "glm_errored117"), os.path.join(self.rs, "controls_glm"))
        self._seeds = seeds
        self.refusals = LC.RowRefusals(NAME)
        self.stats: Dict[str, int] = {}

    @property
    def seeds(self) -> LC.Seeds:
        if self._seeds is None:
            self._seeds = LC.Seeds.locate()
        return self._seeds

    def discover(self) -> List[Input]:
        out = [check_input(os.path.join(self.rs, rel), role="report") for rel in FILES if os.path.lexists(os.path.join(self.rs, rel))]
        if not out:
            raise AdapterRefusal(self.rs, "none of the GLM lane files is present")
        return out

    def comparison_group(self, inp: Input) -> str:
        return "glm"

    def source_states(self, ctx: AdapterContext, inputs: Sequence[Input]) -> List[SourceStateRow]:
        return [SourceStateRow(source_id_for(NAME, "glm"), i.uri, "$", 1, 1, [os.path.basename(i.uri)], i.sha256) for i in inputs]

    def read(self, ctx: AdapterContext, group: str, inputs: Sequence[Input]) -> Iterator[RecordBatch]:
        batch = RecordBatch(source_id_for(NAME, group), group)
        m = self.seeds.models.get(MODEL_ID) or {}
        reason = str(m.get("paper_status_ruling_id") or "descoped: dropped for poor text performance; the paper does not name it")
        ruled_utc = "2026-09-11T00:00:00.000Z"
        batch.add("ruling", {"ruling_id": LC.DESCOPE_GLM_RULING_ID, "round_id": LC.ROUND_ID, "action": "descope", "scope_kind": "model",
                             "scope_json": {"model_id": MODEL_ID}, "target_json": {}, "reason": reason, "ruled_utc": ruled_utc, "ruled_utc_precision": "date",
                             "ruled_by": "the user (2026-09-11; 2026-09-13)",
                             "source_ref": "configs/registry/models.yaml glm46vflash paper_status_ruling_id; CLAUDE.md 'The four model families'; paperlib MODELS comment",
                             "extra": {"placeholder_for_pr7": True, "model_paper_status": m.get("paper_status")}}, ctx.producer, ctx.now)
        self.stats = {"rulings": 1, "artifacts": len(inputs)}
        yield batch


def factory() -> LegacyGlmAdapter:
    return LegacyGlmAdapter()


register(NAME, factory)

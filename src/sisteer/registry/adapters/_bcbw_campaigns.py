"""The DECLARED CAMPAIGNS the full BigCodeBench / BFCL readers read (:class:`_bcbw_common.CampaignSpec`; :data:`_bcbw_common.DECLARED_CAMPAIGN_RULE`): each a campaign's stores --
the paper's harness, served model, instruments and item sets -- under a campaign, views, roster, pins and anchor mode of its own, read by the full readers
parametrized and published as a source pair of its own (`bigcodebench_full_<campaign>` / `bfcl_full_<campaign>`). Nothing about a campaign is inferred from a store: every
fact the reader binds is spelled here.

SIKL4 -- the SI-distilled Qwen3.5-4B refits (campaign ``si_target_refit``): KL distillation of a system-instruction-prompted teacher, fit on and served on the
instruct checkpoint, for the safety SI ``safety_refusal``, the constitution SI ``si_constitution`` and the short irrelevant control SI ``C16``, each as a rank-4
o_proj LoRA (``lora_r4``) and an all-layer ReLU steer (``psr_kl_all``), at fit seeds 1-3 -- the eighteen objects of ``configs/sikl4_pins.json`` under the three
views of ``configs/sdfbase_full22_roster_sikl4.yaml`` (``sikl4s<k>``, the roster's pin form). The registry spellings ``qwen35_4b_sikl4s<k>`` are
legacy_spellings.yaml's (the instruct model, fit ``qwen35_4b_it``, ``instruct_direct``). The campaign runs NO unsteered store (its rows are anchored by the frame
reader on the paper's seed-matched unsteered run of the same instrument), so a ``none`` store refuses by name and the campaign files no registry contrast
(anchor ``none``). The stores are the Modal full-suite wave launched with no store tag: row dirs ``qwen35_4b-f22sikl4s<k>``, suffixes
``bcbw-qwen35_4b_sikl4s<k>-<sha7>[-r<N>]``.

A second campaign is two thin modules over a declaration of its own plus its config lines (sources_r2_v1.yaml, cluster_reads.yaml, vocab.py) and its producer
records; ``tests/test_registry_adapter_bcbw_campaigns.py`` reads a synthetic campaign of the other shape too (base-to-instruct fits, its own ``none``, anchor
``own_none``), so the shape is served by design without a production declaration.
"""
from __future__ import annotations

from . import _bcbw_common as B

#: the session that launched and pulled the campaign's stores and writes its producer records (the run rows' producer_session, the adapters' owner)
SIKL4_PRODUCER_SESSION = "Direct"
SIKL4 = B.CampaignSpec(
    name="sikl4",
    campaign="si_target_refit",
    model_key="qwen35_4b",
    entries=tuple(B.CampaignTag(tag=f"qwen35_4b_sikl4s{k}", row_dir=f"qwen35_4b-f22sikl4s{k}", selection=f"sikl4s{k}", fit_seed=k, transfer_kind=B.INSTRUCT_TRANSFER_KIND)
                for k in (1, 2, 3)),
    harness_roster_rel=("configs", "sdfbase_full22_roster_sikl4.yaml"),
    harness_pins_rel=("configs", "sikl4_pins.json"),
    launched_pins_file="sikl4_pins.json",
    admits_none=False,
    anchor=B.DECLARED_ANCHOR_NONE,
    producer_session=SIKL4_PRODUCER_SESSION,
)

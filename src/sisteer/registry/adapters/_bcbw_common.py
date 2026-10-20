"""The BigCodeBench-200 / BFCL-200 seed-replicate stores' shared reading.

ONE store per (seed tag, object key) holds BOTH tasks (``inspect/bigcodebench_200`` and ``inspect/bfcl_200``): the same bytes are published
under two mirror roots, ``run-state/full22/bigcodebench_seedrep/`` and ``run-state/full22/bfcl_seedrep/`` (a copy each, never a symlink),
so each reader owns its root and files its OWN source on its OWN instrument. Everything the two readers share — the mirror's layout, the
producer's records (the gate attestations, the launch set, the pins, the launched roster, the three-number record), the object resolution, the
recount and the cross-checks — lives here; ``bigcodebench_seedrep.py`` and ``bfcl_seedrep.py`` bind a :class:`TaskSpec` each (the eval id, the
task spelling, the outcome file, the instrument seed, the cap, the protocol temperature, the three-number file) and nothing else.

THE MIRROR (the mirror script's byte-for-byte publish of the Modal volume ``twosi-fits``; ``PROVENANCE.md`` at the root):
``native/<row dir>/<object key>/thinking_off/<suffix>/`` is the ONE store of a (tag, object key) — the row dir is the tinycap model row
``qwen35_4b-f22s<N>[p]`` (tag ``qwen35_4b_s<N>[p]``), the object key the launcher's own spelling (``constv2-mix-full__psr_ce_all``; the store key
inside spells it with ``--``), the suffix ``bcbw-<launch tag>-<sha7>`` (the launch tag is the tag, or ``<tag>_arditi`` for the all-positions
object). What is read, every file named:

* ``cells/<store key>/manifest.json`` — served identity (``model_id`` / ``model_revision`` / ``serving_identity`` / ``serving_engine`` /
  ``serving_path`` / ``fit_provenance.object_sha`` a 16-hex prefix of the object / ``provenance.artifact_sha256`` the full sha /
  ``resolved_config.system_sha`` the prompt's / ``producer.tree_sha`` / ``harness_config_hash`` / ``intervention_hash`` / ``steer_positions`` /
  ``geometry`` / ``seeds.inspect`` (the DEFAULT rule: no replicate block) / ``sampling.by_task`` / ``thinking.by_task`` / ``tinycap`` (the item
  set pins, the served flags, the clamp override) / ``route`` (the sandbox owner: the Modal function call) / ``usage_ledger`` (a path on the volume));
* ``…/cells/<store key>/outcomes/inspect__<task>_200.json`` — the per-item OUTCOME record over EVERY attempted item (``at_cap`` / ``scored`` /
  ``value`` / ``stop_reason`` / ``completion_tokens`` / ``seed`` / ``unparseable``): the item universe; ``summary/inspect__<task>_200.json`` and
  ``gen_stats.json[<task>]`` — the producer's aggregate (``acc`` over SCORED, ``n`` 200, ``n_scored``, ``n_unscored``, ``n_at_cap``, ``max_tokens``,
  the sent decode numbers, the framework); ``done/inspect__<task>_200.json`` — the completion marker (``n_items`` / ``n_rows`` / ``finished_at``);
  ``rows.jsonl`` — the scored ``acc`` rows of both tasks plus the aggregates (``acc_aggregate``, ``n``);
* ``RESULT`` (its LAST line must read ``result=COMPLETE`` and name the tree), ``lane_roster.json`` (the launched roster entry: the object sha
  the launcher wired as the pin, the roster's sha256, the launch time), ``served_geometry.json`` (the standup's attested endpoint: the engine,
  ``lora_config.max_lora_rank``, ``zero_adapter_only`` — RECORDED, never compared), ``<object key>/controls_summary.json`` (the producer's per-store
  summary); ``evidence.json`` / ``protocol_check.json`` / ``resolved_sampler.json`` / ``usage_ledger.jsonl`` / the driver logs are registered as the
  store's artifacts and not otherwise read;
* ``records/`` — ``gate_<launch tag>_<object key>_<suffix>.json`` (bcbw_gate.py's attestation per row: ``verdict`` PASS / FAIL, the attested
  ``route.serving_engine``, the launch flags, the RECORDED ``max_lora_rank`` / ``zero_adapter_only``, the completed tasks); ``launch_sets.json``
  (``launch_sets/v1``: the ROW TABLE per eval — ``evals.<eval>.launch``, the rows the current pass launched (``census_status run``), plus
  ``carried`` / ``carried_late``, the rows carried from the earlier pass (``census_status carry``; the census of 2026-09-23 14:19Z moved the
  30 pass-1 rows there, :data:`ROW_TABLE_RULE`) — the PIN KEY and the EXPECTED object sha per row, the identity declaration; ``carried_optional`` /
  ``optional_not_carried`` / ``exists_not_carry`` are census notes outside the table and never rows);
  ``sdfbase_full22_pins.json`` (the pins the launcher wired, at the pin sha); ``bcb_bfcl_200_widen_roster.yaml`` (the launched roster;
  ``lane_roster.json:roster_sha256`` is its sha256); ``three_numbers_<eval>_qwen_seedrep.json`` (``w4_three_numbers/1``: the producer's own count
  of every number a reader re-derives); ``rows_launch.tsv`` / ``bcbw_schedule.json`` (the launch ledger and the planned envelope, registered).

OBJECT IDENTITY per (selection, object key) = the launch set's pin key (``base64s<N>__…``, ``derived_base64s<N>__…``, ``paired_eb64s<N>__…``,
``ar_c0p3__arditi_allpos``), which must be the harness roster's staged id for the key at the tag's selection (``configs/sdfbase_full22_roster.yaml``
``targets.qwen35_4b.views.s<N>[p].cells`` -> ``configs/sdfbase_full22_pins.json``; the all-positions key through ``configs/arditi_allpos_roster.yaml``
/ ``_pins.json`` at the operating point the pin key spells), whose pinned sha must equal the launch set's expected sha, the mirrored pins' entry,
the launched roster's ``object.sha256``, the store's ``lane_roster.json:object_sha256``, ``provenance.artifact_sha256`` and the 16-hex
``fit_provenance.object_sha``; a prompted key resolves through ``si_texts.yaml`` and ``resolved_config.system_sha``. Any disagreement refuses the
(tag, object key) by name (:class:`lineage_common.RowRefusals`; the group's other rows file). The reader never consults the live registry for the
object: the run is emitted on the attested identity and ``blocked_records`` holds it while the object row is absent.

THE GATE (bcbw_gate.py): a row files only if its gate record reads ``verdict PASS`` and attests ``route.serving_engine lora_native``; a FAIL
or an absent gate record for a PRESENT store refuses the row by name. The record's ``max_lora_rank`` / ``zero_adapter_only`` are recorded on the
run (``extra.gate.recorded_flags``, ``extra.standup``), never compared (GSM8K C4).

NOT RUN: a launch-set row with no store (the gate record reads FAIL, "the task has not COMPLETED") files NOTHING — no run, no refusal, no
disposition (the producer files those as dispositions; a later mirror adds the store idempotently). Consequence: a seed tag whose ``none`` has not
run has no anchor yet — its rows file WITHOUT a contrast, with ``extra.anchor`` naming the reason (``anchor_not_yet_run``), so the contrast appears
on a later read while the value's records stay byte-identical.

THE PAIRED TAG'S OWN ``none`` IS ITS ANCHOR (the GSM8K precedent;
:data:`PAIRED_ANCHOR_RULE`, :func:`anchor_plan`): ``qwen35_4b_s<N>p`` is a STANDUP REPLICATE of ``qwen35_4b_s<N>`` — the same seed base,
re-served — so a ``none`` under it is launch-set class ``anchor`` and files as a normal run of the paired tag (object ``none``, the eval's
instrument, ``trial_index 0`` like every row here, no contrast of its own; ``extra.standup_replicate_of_view`` names the replicated tag), and
the composed ``s<N>p`` row pairs against IT (``anchor_kind same_standup_none``, ``cross_standup`` false, ``anchor_view_legacy`` the paired tag).
Until that ``none`` FILES the composed row keeps the fallback — ACROSS tags against ``s<N>/none`` (``anchor_kind sibling_none``, the same
model's ``none`` served by another standup, :func:`anchor_kind_for`; ``cross_standup`` true, ``cross_standup_note``) — and its ``extra.anchor``
block names the own ``none``'s state (``own_none_state`` ``anchor_not_yet_run`` /
``anchor_refused``) once the launch set declares it. The anchor is resolved from the rows that FILED, never from a store's presence, and a seed
tag's rows never pair with the paired tag's ``none`` (the ``(tag, none)`` keys differ). A prompted key under the paired tag is class
``prompted`` and pairs like any row of its tag.

REGISTRY GRAIN: ONE run per (model, tag, object key): ``run_id = identity.run_id(sdfbase_full22, <tag>/<store key>, <object key>, <eval>, <the
canonical instrument bound to qwen35_4b_it>)`` — the producer key is the tag-prefixed store NAME (:data:`PRODUCER_KEY_RULE`), never a path. ONE
GROUP PER SEED (:data:`GROUP_RULE`): the seed tag and its paired tag read in one batch, because the anchor rule pairs the composed row
ACROSS tags with ``s<N>/none`` until the paired tag's own ``none`` files, and a contrast names a run its own batch emits. Every row files its OWN instrument — the capability instrument
(``bigcodebench_200_protocol_greedy_4096`` / ``bfcl_200_protocol_T0001_1024``), thinking OFF, the benchmark's reference protocol sampler, its
own cap — never the statics' 4,096-cap instruments (another item set, harness and scoring rule at the same number).

SEED READING: the protocol sampler is greedy (T 0 / T 0.001) under the DEFAULT Inspect seed rule (no replicate block on
``manifest.json:seeds.inspect``) — the seed axis is the FITTED OBJECT, and every row files the documented default: ``seed_base NULL``,
``trial_index 0``, and ``decode_seed_rule none`` — the rule the row's INSTRUMENT (``bigcodebench_200_protocol_greedy_4096`` /
``bfcl_200_protocol_T0001_1024``) and the protocol policy row declare: a greedy protocol pins no seed, so Inspect's incidental per-item
derivation is RECORDED (``extra.decode_seed_rule_text``, ``extra.seed_axis``, the confidence entry) and never the row's rule (emission 2; emission 1
filed ``per_item``, which kept the derived decode-policy match below at ``no_match`` on every row). The unsteered, prompted and all-positions keys
name ONE object on every seed tag; their coexistence in the paper frame needs the roster's ``coexist`` ruling (never a reader default).

DECODE POLICY: ``run.decode_policy_id`` / ``decode_policy_match`` are DERIVED — ``decode_match.stamp_run_rows`` copies ``v_run_decode_policy_match``
onto every run row at the end of every ingest cycle (the producer's placeholder is ``no_policy``; a reader-side stamp would be overwritten within
the cycle) — and the SQL predicate requires the policy row's ``seed_rule`` to equal the run's ``decode_seed_rule`` beside the sent numbers. The
reader's own resolution of the protocol row (``decode_policy.PolicyFile.resolve`` on the sent numbers) rides ``extra.decode_policy_ref_legacy`` /
``extra.decode_policy_reader_match`` and sets ``decode_source``; with ``decode_seed_rule none`` the derived pass reads ``exact`` and names the row.

THE THREE NUMBERS (the GSM8K precedent): (1) THE VALUE — ``run_metric acc`` = correct / SCORED under the reader's recount computation
(``computed_by registry``); (2) THE BOUND — a second ``run_metric acc`` row = correct / 200 attempted (``producer_metric_spelling bound``,
``computed_by producer``, a BLANK ``computation_id``, ``ci_lo`` / ``ci_hi`` the interval the unscored items leave open); (3) THE AT-CAP COUNT —
every item whose draw ended at the cap (the outcome record's ``at_cap``: ``stop_reason == max_tokens``) over the 200 (``run_degeneration
at_token_cap_finish_reason``, ``computed_by registry``). On these tasks an at-cap item is still SCORED (the code / the call is graded as
emitted), so its value counts and ``item_flags.at_token_cap 1`` marks it; an at-cap item the harness left UNSCORED would be ``value NULL`` under
``at_cap_unresolved`` (class ``ceiling``: the one pass is the instrument's cap), an unscored item below the cap ``value NULL`` under ``unscored``.
No loop / rep-ratio reading exists for these stores: ``rep_ratio_loop`` and the other declared measures file ``not_measured`` naming why, and
nothing is invented.

THE APPENDIX FAMILIES (:data:`MODEL_KEYS`, :data:`APPENDIX_FAMILY_RULE`): gemma4e4b, llama8b
and nemotron3nano4b run the same BigCodeBench-200 / BFCL-200 conditions natively under the same protocol decode, ONE seed tag per family —
``<model key>_s1`` (row dir ``<model key>-f22s1``) and its paired tag ``<model key>_s1p`` — mirrored under the sources ``bigcodebench_allmodels`` /
``bfcl_allmodels`` (``run-state/full22/<source>/``, the seedrep record shape). The readers ``bigcodebench_allmodels.py`` / ``bfcl_allmodels.py`` bind the
seedrep readers' :class:`TaskSpec` with ``models`` = the appendix families; every grammar above holds with the model key in place of ``qwen35_4b``
(the group is ``<model key>_s<N>``, the run's model the table's ``model_id``, the served checkpoint and revision the table's = ``models.yaml``, the
instrument the canonical seed bound to that model). Two things differ, both recorded on the run: (1) OBJECT IDENTITY — the harness roster stages no
``s1`` / ``s1p`` selection for these families (the stage index ``allmodels_s1_stage``: the ``s1`` objects are volume-internal copies of the
``main`` selection's staged objects, ``base__<key>`` / ``derived__<key>``, pinned under ``configs/sdfbase_full22_pins.json[<model key>]``; the ``s1p``
object ReLU's paired write ``paired_eb<B>__…``), so a fitted key's staged id IS the launch set's pin key, whose sha must be the harness pins' entry
(a key the harness pins do not carry refuses by name until it is pinned); where the roster's ``main`` selection stages the same key its staged id
must equal the pin key (``extra.object_route``); a paired tag's ``paired_of`` is the tag grammar's, with no roster selection to cross-check
(``extra.paired_of_source``). (2) THE ``main`` CANARY ROW ALIAS (:data:`LEGACY_MAIN_ALIAS_RULE`) —
gemma4e4b's and llama8b's seed-tag conditions ran BEFORE the ``s1`` tag existed, under the family's ``main`` row: ``native/<model key>-f22/<object
key>/thinking_off/bcbw-<model key>_main-<sha7>/`` (row dir ``<model key>-f22``, tag ``<model key>_main``; the gate records ``gate_<model key>_main_<object
key>_…``) — the ``none``, the two prompted keys, the ReLU keys and the LoRA keys alike; the paired ``s1p`` keys, llama8b's ReLU keys and every
nemotron3nano4b key sit under the ``<model key>-f22s1[p]`` rows. Completed work is canonical, never re-run: EVERY store of the ``main`` row files
as the ``<model key>_s1`` tag's row of its key (producer key ``<model key>_s1/<store key>``; the ``none`` the seed rows' anchor) with the original
spelling recorded (``extra.tag_legacy_original``, ``extra.legacy_main_alias``: the row dir, the launch tag, the launch set's and the three-number
record's spellings — ``view`` ``s1`` or ``main``, ``tag`` either, ``seed_base`` 1 or null are accepted for those rows). Its fitted objects resolve
through the harness roster's ``main`` selection for the family — ``targets.<model key>.views.main.cells[<key>]`` -> staged id -> the pins' sha, the
harness's own resolution, exactly as a qwen seed selection resolves — and the launch set's pin key must be that staged id. A key present under BOTH
the ``main`` row and the ``s1`` row refuses the ``main`` row's store by name (the tag's own row wins). A store of a family outside a source's
``models`` refuses its group by name (the qwen readers refuse a gemma store, the appendix readers a qwen store).

THE INSTRUCT-DIRECT TAG (:data:`INSTRUCT_DIRECT_RULE`, :data:`INSTRUCT_ANCHOR_RULE`; the FULL readers alone,
``TaskSpec.instruct_direct``): the instruct-checkpoint LoRA fits of gemma4e4b / llama8b / qwen35_4b -- fit on AND served on the instruct model -- run under
the tag ``<model key>_it`` (row dir ``<model key>-f22it``, launch tag ``<model key>_it``, the launch set's selection ``it``, ``block`` / ``transfer_kind``
``instruct_direct``, ``seed_base`` null, the fitted class (:data:`INSTRUCT_CLASS`), ``anchor_tuple s1/none``; :data:`MODEL_KEYS` ``instruct_direct`` names the families, nemotron has no
such tag). The tag reads in the family's ``s1`` group (:func:`group_tags`; :func:`seed_tag_of`) because its anchor is that group's ``none`` BY ALIAS: the launch
set's ``it/none`` row is class ``anchor`` with no store, no ``_it`` none run files, and every ``_it`` row's contrast names ``<model key>_s1/none``
(:func:`instruct_anchor_plan`; ``anchor_kind sibling_none`` -- the same model's ``none`` served by another standup, :func:`anchor_kind_for`;
``cross_standup`` true, ``extra.anchor_alias``). A fitted key
resolves through the harness roster's ``it`` selection for EVERY family (the harness stages the instruct fits there; the launch set's pin key must be that
staged id, its ``pin_key_alias`` -- the harness's other spelling of the same object -- must pin the same sha where the launched pins carry it); the row's
``view_legacy`` is ``<model key>_it`` (legacy_spellings.yaml: the instruct model, fit ``<model key>_it``, transfer_kind instruct_direct) and its
``transfer_kind`` is DERIVED -- ``manifest.json:fit_provenance.model_id`` must be the served ``model_id`` (``derive.transfer_kind`` -> instruct_direct) and
agree with the spelling -- never assumed. Refused by name: a ``_it`` ``none`` / prompted / all-positions store, a ``_it`` store of a family without the tag or
under a 200-item reader, a row spelling a seed base or another block / transfer kind / anchor tuple, a base-checkpoint fit, an alias naming another object.
:data:`ONE_TASK_STORE_RULE`: a ``_it`` row relaunched at another tree reads as every relaunch does (``-r<N>``, ``launched_as`` /
``relaunched_from``, ``relaunches.json``; a parked attempt with no store parks nothing); a row whose two tasks ran in two ONE-TASK stores carries the task tag
on the suffix (``-bcb`` / ``-bfcl``) -- each source reads the store of ITS task (its ``launched_as`` names it), never reads or refuses the other task's, and
parks a store ``launched_as`` does not name when it holds none of the task's files; the gate record of a present store is the one of ITS suffix.
A rerun after SEVERAL parked attempts (two one-task stores superseded by one two-task ``-r<N>`` store) names them all on
``relaunched_from.suffixes`` and ``relaunches.json``'s ``parked_suffix`` list (both additive; the first stays on ``relaunched_from.suffix``) -- a mirrored member
parks, an unmirrored one's gate record (FAIL, or PASS but superseded) is neither the row's store nor a refusal and is listed on the status
(``gate_records_without_store``); the launch row's ``launched_as`` is the authority for the row's store.
:data:`LAUNCHED_AS_TREE_RULE`: the three rerun
rows carried ``sha`` = the FIRST attempt's tree while ``launched_as.sha`` = the rerun's (the reader refused each by name, as the store's manifest named the
rerun's tree); the convention is that a row's ``sha`` IS the tree of the store it reads, so a row whose ``sha`` differs from ``launched_as.sha`` is an
INCONSISTENT record and refuses by name -- no store is read for it under either tree. A store the row names on ``launched_as`` that exists but was not read,
a store directory whose suffix is outside the grammar, and a store both named on ``launched_as`` and parked on ``relaunched_from`` refuse by name; a not-run
row names the ``launched_as`` store it is missing.

THE SHARED-STANDUP COMPARISON SOURCES (:class:`SharedSpec` on ``TaskSpec.shared``, None on every solo
reader; :data:`SHARED_SOURCE_RULE` and the rules beside it): the row-engine harness serves every object of a model row on ONE vLLM engine and re-runs the
canary rows as whole rows, COMPARED to the solo stores -- mirrored under ``bigcodebench_full_shared`` / ``bfcl_full_shared`` and read by the full readers
PARAMETRIZED (:mod:`bigcodebench_full_shared`, :mod:`bfcl_full_shared`), never in the paper frame. What the shared reading adds, every branch gated on the
spec: the suffix grammar ``bcbw-<launch tag>-<sha7>-shared[-r<N>][-rep<N>]``; the registry spelling ``<tag>_shared`` (the launch row's ``view``; its
``selection`` is the row's key) and the run id minted under it with a repeat's segment folded in (``<tag>_shared/<store key>[@rep<N>]``; the store key is
byte-identical to the solo's); the row key ``(tag, object key[@rep<N>])``; the comparison fields (``comparison_kind`` / ``solo_run_id`` / ``solo_store``) read
from the launch row and the gate's ``comparison`` block; the row engine's one standup id per launch across the manifest's ``row_engine`` block, the endpoint,
RESULT's ``standup=`` token and the gate; the borrowed ``run_contrast`` against the SOLO run read from the solo mirror (its RESULT and outcome record the
group's inputs) beside the in-run contrast against the row's shared ``none``; the instruct row's own ``none`` as a comparison row that anchors nothing. The
solo readers' emission is byte for byte what it was (the dry run over the live mirrors).

THE qwen35_35b_a3b MODEL KEY (2026-10-04; the paper-companion-repo pipeline test's served checkpoint; :data:`MODEL_KEYS`): the FULL readers
read its stores (``TaskSpec.models`` = every key of the table) -- row dir ``qwen35_35b_a3b-f22s1`` (tag ``qwen35_35b_a3b_s1``: the three LoRA keys fit on
``Qwen/Qwen3.5-35B-A3B-Base`` and served on ``Qwen/Qwen3.5-35B-A3B`` @ 59d61f3c, base_to_instruct), served id and revision = ``models.yaml`` ``qwen35_35b_a3b`` =
the registry's model row = every mirrored manifest's ``model_id`` / ``model_revision``. ``roster_selections`` False (on this tree the harness roster stages no
``targets.qwen35_35b_a3b``, so the appendix route: the launch set's pin key is the staged id, pinned under the model's block of
``configs/sdfbase_full22_pins.json`` -- the launch trees' block, copied verbatim 2026-10-05: sha 1eb2af8d per-tree copies beside the stores; the three LoRA
keys ``base__constv2ctl2-full__lora_r4_ce`` e54842ec, ``base__constv2-mix-full__lora_r4_ce`` c72f7a10, ``derived__constv2-mix-full__lora_r4_ce_minus_ctl`` a04c1c0f),
``legacy_main_alias`` False (no ``qwen35_35b_a3b-f22`` row, no ``_main`` launch tag), ``instruct_direct`` False (no ``-f22it`` row; the fits are base fits). The
mirror ALSO carries the row dir ``qwen35_35b_a3b-f22u0`` (the unsteered ``none`` and the two prompted keys, the baseline the steered rows pair against under the
pipeline test's own rule) and suffixes outside this grammar (``bcbw-qwen35_35b_a3b_u0-<sha7>[-w14336][-bcb|-bfcl]``: a launch tag ``_u0`` and a window segment):
both READ since 2026-10-05 under the contract below (the first landing of the key refused them by name instead of letting them vanish).

THE u0 ROW AND THE WINDOW-MATCHED BASELINE (2026-10-05; :data:`UNSTEERED_ROW_RULE`,
:data:`WINDOW_ANCHOR_RULE`; ``ModelKey.unsteered_row`` / ``TaskSpec.unsteered_row``, the full readers): tag ``<model key>_u0`` (row dir ``-f22u0``, launch tag
``_u0``, the launch set's selection ``u0``) reads in the family's ``s1`` group and holds the unsteered ``none`` and the prompted keys; ``-w<N>`` on a suffix is the
SERVING WINDOW (vLLM max_model_len; the cap is the instrument's and equal across windows) a ``none`` was re-served at, its row keyed ``none@w<N>`` and declared by
``launch_sets.json families.<model key>.unsteered.window_reserves.<eval>`` (a launch-set row is one per (tag, key)); ``-bcb`` / ``-bfcl`` are the row's one-task
stores (ONE_TASK_STORE_RULE; untagged = the BigCodeBench store). Every steered or prompted row of the family anchors on the u0 ``none`` served at the SAME window
(``context_window_tokens`` / ``served_max_num_seqs``): own tag for a u0 row (``same_standup_none``), across tags for an s1 row (``sibling_none``, ``anchor_view_legacy``
``<model key>_u0``, ``cross_standup`` true). The 35B mirror: BCB s1 x3 + u0 safety -> ``…_u0-0dba464-bcb``; u0 constitution BCB -> ``…_u0-b8086a6-w14336``; BFCL
s1 x3 + safety -> ``…_u0-55cc447-bfcl``; constitution BFCL -> ``…_u0-b8086a6-w14336-bfcl``; the probe canary under ``canary/`` is not a row. The serving window
rides every row of such a family (``extra.serving_window``), the u0 rows their declaration (``extra.unsteered_row``); every other family's record is byte for byte.

THE DOSE-0.75 RANDOM-INIT ReLU ROW (2026-10-05; :data:`DOSE_RELU_ROW_RULE`; ``ModelKey.dose_relu_row`` / ``TaskSpec.dose_relu_row``, the full
readers): tag ``(model key)_d0p75_relul`` (row dir ``-f22d0p75_relul``, launch tag ``_d0p75_relul``, the launch set's selection ``d0p75_relul``, legacy_spellings.yaml
``qwen35_35b_a3b_d0p75_relul``) reads in the family's ``s1`` group and holds three STEERED rows -- the two parent dose copies and their difference, random-init ReLU
objects (the full22 bank's ``d0p75l__`` staged ids), bound by object id through the pins block (the appendix route: the launch set's pin key is the staged id; the two
parents are the launch's extras, not harness-roster entries). Every row anchors on the family's u0 ``none`` at its own serving window (``-w<N>``, WINDOW_ANCHOR_RULE),
exactly as the s1 rows do. A ``none`` re-served at the launch's tree under the u0 row shares the (u0, none) row key with the declared store and binds no new row: the
launch row's tree names the store, a second store on one key refuses by name. Its rows carry ``extra.dose_relu_row`` and NO fit seed (``extra.fit_seed`` null, like
``run.seed_base``: the s1 group is a standup grouping, not the object's fit provenance -- the parent fits' sidecars record seed 0 and a dose-scaled copy's fit seed is not
applicable); every other family's record is byte for byte.

THE DOSE-1.25 DIFFERENCE-LoRA ROW (2026-10-05; :data:`DOSE_LORA_ROW_RULE`; ``ModelKey.dose_lora_row`` / ``TaskSpec.dose_lora_row``, the full
readers): tag ``(model key)_d1p25`` (row dir ``-f22d1p25``, launch tag ``_d1p25``, selection ``d1p25``, legacy_spellings.yaml ``qwen35_35b_a3b_d1p25`` at strength_nominal
1.25 in the LoRA family's unit) reads in the family's ``s1`` group and holds ONE steered row: the dose_scale copy of the s1 row's difference LoRA at base_scale 1.25,
bound by object id through the pins block (pin key ``d1p25__<difference key>``, the bank's staged id; a key the pins do not carry refuses the row by name). It anchors
on the u0 ``none`` at its own serving window like the s1 rows, carries ``extra.dose_lora_row`` and NO fit seed (a dose-scaled copy's); every other family's record
is byte for byte.

THE LeCun-INIT ReLU ROW (2026-10-07; :data:`LECUN_RELU_ROW_RULE`; ``ModelKey.lecun_relu_row`` / ``TaskSpec.lecun_relu_row``, the full readers):
tag ``(model key)_s2rl`` (row dir ``-f22s2rl``, launch tag ``_s2rl``, selection ``s2rl``, legacy_spellings.yaml ``qwen35_35b_a3b_s2rl`` at strength_nominal 1.0) reads in
the family's ``s1`` group and holds TWO steered rows -- the control ReLU and the constitution ReLU REFIT from a LeCun random init into the full22_lecun bank: FITS, not
dose copies -- each bound by object id through the pins block (pin key ``lecun__<ReLU key>``, the bank's staged id; a key the pins do not carry refuses the row by name,
never a guessed object). The harness's ``s2rl`` selection lists a THIRD object (the composed difference, ``lecun_derived__...``) the launch never ran: no store, no launch
row, no pin -- it is not a row, nothing waits on it and nothing refuses for it. Each row anchors on the u0 ``none`` at its own serving window like the s1 rows and carries
``extra.lecun_relu_row``. Its seeds: ``extra.fit_seed`` is the objects' RECORDED fit seed (:data:`LECUN_RELU_FIT_SEED`, the bank sidecars'
train_budget_meta.json seed 0 at effective batch 64, carried as provenance -- the sidecar is not in the mirror, so the value is declared here, not re-read); the launch
row spells ``seed_base`` as the group's (1, the standup seed the u0 pairing is keyed on, checked like every steered row's) and the three-number row may spell the seed as
the group's or as none; ``run.seed_base`` stays NULL like every row of this evaluation (:data:`SEED_AXIS_RULE`); the dose rows keep filing ``extra.fit_seed`` null. Every
other family's record is byte for byte. Under a reader that does not admit the row (the 200-item readers) or a family that does not run it, the row dir ``-f22s2rl`` is
UNDECLARED for that reader (:func:`lecun_relu_row_admitted`; :func:`tag_of_row_dir` / :func:`undeclared_row_dir_reason` read the reader's spec): a group of its own, refused as
a whole by name with ``facts["refuse"]`` -- what the base tree filed for the same directory -- so a misplaced or early-launched s2rl store reads ok = 0 on the status (a
by-name row refusal in the s1 group left ``source_status.ok`` at 1 while the s1 rows filed).

AN UNDECLARED ROW DIR REFUSES BY NAME (2026-10-04; :func:`undeclared_row_dir_reason`, :data:`UNDECLARED_ROW_DIR_RULE`): a directory under ``native/`` that
spells no declared row -- a model key the table lacks, a declared key under a selection the grammar lacks (``u0``), or since 2026-10-07 the LeCun-init ReLU row dir
``-f22s2rl`` under a reader that does not read it (:func:`lecun_relu_row_admitted`) -- used to be SKIPPED at discovery, so a
new model's stores read as "0 records, no refusal" (a pass over the 35B mirror). Now every store directory under it is discovered, its inputs form a
GROUP named for the row dir itself (``comparison_group``: the row dir, never a tag; a row dir carries ``-f22`` and a tag never does), and ``read()`` refuses that
group as a WHOLE by name -- the reason in ``facts["refuse"]`` (what the status reads: the group's ``source_status.ok`` is 0, ``v_source_status`` folds the
source to ``MIN(ok)`` = 0 with the reason under ``extra.refuse``, and the paper's ``lane_guard`` reads the source as not integrated) and the same text in
``facts["refusals"]`` (``row dir <name> matches no declared model key (...)`` / ``row dir <name> names the declared model key <mk> but no declared row
(...)``), emitting nothing. A ``refusals`` entry alone would be the runner's EXPLAINED empty read (``EXPLANATION_KEYS``) yet file ``ok = 1`` with 'no row
refused' -- a successful-looking zero-measurement read -- which is why the whole-group refusal rides ``refuse``; since 2026-10-05
EVERY group-wide return of ``read()`` does the same through :meth:`BcbwSeedrepReader._refuse_group` (no seed tag, a model key outside the source's families, a seed
outside the table, legacy_spellings.yaml / models.yaml disagreeing with the model-key table, a missing instrument row, a shared spelling resolving elsewhere),
where they populated ``refusals`` alone and left ``ok = 1``. A dot-prefixed entry (``.staging``: a
producer's publish scaffolding, by convention never a row) and a non-directory are the only entries not reported. Declaring the key later moves the row dir's
inputs into its seed-tag group and retires the row-dir group (its inputs left discovery).
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from .. import decode_policy as DP
from .. import derive as DV
from .. import identity as I
from .. import jobplan as JP
from .. import vocab as V
from . import _items_common as IC
from . import anchor_current as AC
from . import declared_common as DC
from . import dose_grammar as DG
from . import gsm8k_seedrep as G
from . import lineage_common as LC
from . import relu_paired as RP
from . import seedrep_cluster as SC
from .base import AdapterContext, AdapterRefusal, Input, RecordBatch, SourceStateRow, check_input
from .runner import source_id_for

OWNER = "Overflow"
PRODUCER_SESSION = "Overflow"
CAMPAIGN = LC.CAMPAIGN
#: the three-seed family's model key (the seedrep readers'; re-exported by them)
MODEL_KEY = SC.MODEL_KEY


@dataclasses.dataclass(frozen=True)
class ModelKey:
    """One tinycap model key a source reads: the registry model row its tags' runs file on, the served checkpoint and its pinned revision (the
    store's ``manifest.json:model_id`` / ``model_revision`` must equal them, and they must equal ``models.yaml``), the seed tags the family runs,
    whether its ``s<N>[p]`` selections are harness roster selections (``targets.<key>.views.<sel>.cells`` -> staged id -> pins) or the launch set's
    pin key is the staged id (the appendix families), and whether the family's ``main`` canary row aliases to its ``s1`` tag."""
    key: str
    model_id: str
    served_hf_id: str
    served_revision: str
    seeds: Tuple[int, ...]
    roster_selections: bool
    legacy_main_alias: bool
    #: the family runs the INSTRUCT-DIRECT tag `<key>_it` (:data:`INSTRUCT_DIRECT_RULE`): the harness roster stages an `it` selection for it (its
    #: instruct-checkpoint fits) and legacy_spellings.yaml spells `<key>_it` as the instruct model with transfer_kind instruct_direct; nemotron has neither
    instruct_direct: bool = False
    #: the family runs the UNSTEERED row `<key>_u0` (:data:`UNSTEERED_ROW_RULE`; the pipeline test's unsteered-first pass): the `none` baseline(s) and the
    #: prompted keys sit under row dir `<key>-f22u0`, and every steered or prompted row of the family anchors on the u0 `none` at the SAME serving window
    unsteered_row: bool = False
    #: the family runs the DOSE-0.75 RANDOM-INIT ReLU row `(model key)_d0p75_relul` (:data:`DOSE_RELU_ROW_RULE`): three steered rows
    #: (the two parent dose copies and their difference, random-init ReLU objects) under row dir `(model key)-f22d0p75_relul`, each anchoring on the family's u0 `none`
    #: at the SAME serving window, like the s1 rows
    dose_relu_row: bool = False
    #: the family runs the DOSE-1.25 DIFFERENCE-LoRA row `(model key)_d1p25` (:data:`DOSE_LORA_ROW_RULE`): ONE steered row -- the
    #: dose_scale copy of the s1 row's difference LoRA at base_scale 1.25 -- under row dir `(model key)-f22d1p25`, anchoring on the family's u0 `none` at the SAME
    #: serving window, like the s1 rows
    dose_lora_row: bool = False
    #: the family runs the LeCun-INIT ReLU row `(model key)_s2rl` (:data:`LECUN_RELU_ROW_RULE`): TWO steered rows -- the control and
    #: constitution ReLU objects refit from a LeCun random init (fits, not dose copies) -- under row dir `(model key)-f22s2rl`, each anchoring on the family's u0 `none` at
    #: the SAME serving window, like the s1 rows; for a family with False the row dir is UNDECLARED on every reader (a whole-group refusal by name, UNDECLARED_ROW_DIR_RULE)
    lecun_relu_row: bool = False


#: the model keys the two source families read: the three-seed Qwen family (three seeds, harness roster selections, no alias) and the
#: appendix families (one seed each, the launch set's pin key as the staged id, the `main` canary aliased to `s1`). Served ids and revisions
#: are the ones the producer stamps on the stores and must equal configs/registry/models.yaml (checked at read time; a disagreement
#: refuses the group). gemma4e4b / llama8b / qwen35_4b also run the instruct-direct tag (the full readers alone admit it).
#: qwen35_35b_a3b (2026-10-04; module docstring, THE qwen35_35b_a3b MODEL KEY): the pipeline test's served checkpoint -- model_id the registry's model row
#: (= legacy_spellings.yaml model_keys[qwen35_35b_a3b_s1].model_id), served id / revision = models.yaml = the 15 mirrored manifests, ONE seed row dir
#: (`-f22s1`), the appendix object route (no harness roster target on this tree; the pins block is the launch trees', 2026-10-05), no `main` canary, no
#: instruct-direct tag (its fits are base fits).
#: The FULL readers alone bind it (TaskSpec.models = every key of the table; the 200-item readers' tuples below are unchanged, no 200-item 35B store exists)
MODEL_KEYS: Dict[str, ModelKey] = {
    "qwen35_4b": ModelKey("qwen35_4b", "qwen35_4b_it", "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", (1, 2, 3), True, False, True),
    "gemma4e4b": ModelKey("gemma4e4b", "gemma4e4b_it", "google/gemma-4-E4B-it", "ee0ef6023621cff504d758262d4e04895a5af4a2", (1,), False, True, True),
    "llama8b": ModelKey("llama8b", "llama8b_it", "NousResearch/Meta-Llama-3.1-8B-Instruct", "d10aef7999a2b5ba950ab3974312feeedbfe0b77", (1,), False, True, True),
    "nemotron3nano4b": ModelKey("nemotron3nano4b", "nemotron3nano4b", "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f", (1,), False, True, False),
    "qwen35_35b_a3b": ModelKey("qwen35_35b_a3b", "qwen35_35b_a3b", "Qwen/Qwen3.5-35B-A3B", "59d61f3ce65a6d9863b86d2e96597125219dc754", (1,), False, False, False, unsteered_row=True,
                               dose_relu_row=True, dose_lora_row=True, lecun_relu_row=True),
}
SEEDREP_MODEL_KEYS: Tuple[str, ...] = (MODEL_KEY,)
#: the families the appendix sources (`bigcodebench_allmodels` / `bfcl_allmodels`) read: the three appendix families and the pipeline test's model
APPENDIX_MODEL_KEYS: Tuple[str, ...] = ("gemma4e4b", "llama8b", "nemotron3nano4b")
#: the model keys as a regex alternation, longest first (`nemotron3nano4b` must be tried before any prefix of it)
_MODEL_ALT = "|".join(sorted(MODEL_KEYS, key=len, reverse=True))
#: THE INSTRUCT-DIRECT TAG (:data:`INSTRUCT_DIRECT_RULE`): tag `<model key>_it`, row dir `<model key>-f22it`,
#: launch tag `<model key>_it`, the launch set's selection `it`, the harness roster's `it` selection; admitted by the FULL readers alone (TaskSpec
#: `instruct_direct`), read in the family's `s1` group because its anchor is that group's `none` (:data:`INSTRUCT_ANCHOR_RULE`)
INSTRUCT_SUFFIX = "_it"
INSTRUCT_SELECTION = "it"
INSTRUCT_ANCHOR_SEED = 1
INSTRUCT_TRANSFER_KIND = "instruct_direct"
INSTRUCT_BLOCK = "instruct_direct"
#: the launch set's class of a fitted instruct-direct row (the producer's spelling)
INSTRUCT_CLASS = "arm"
#: THE UNSTEERED ROW (:data:`UNSTEERED_ROW_RULE` / :data:`WINDOW_ANCHOR_RULE`): tag `<model key>_u0`, row dir `<model key>-f22u0`, launch tag
#: `<model key>_u0` (the harness spelling of legacy_spellings.yaml's `qwen35_35b_a3b_u0`), the launch set's selection `u0`; read in the family's `s1` group because the
#: family's steered rows anchor on ITS `none`. A suffix may carry `-w<N>`: the SERVING WINDOW (vLLM max_model_len) a `none` was RE-SERVED at, never a cap
UNSTEERED_SELECTION = "u0"
UNSTEERED_SUFFIX = "_u0"
UNSTEERED_ANCHOR_SEED = 1
WINDOW_MARKER = "w"
#: THE DOSE-0.75 RANDOM-INIT ReLU ROW (:data:`DOSE_RELU_ROW_RULE`): tag `(model key)_d0p75_relul`, row dir `(model key)-f22d0p75_relul`,
#: launch tag `(model key)_d0p75_relul`, the launch set's selection `d0p75_relul` (legacy_spellings.yaml `qwen35_35b_a3b_d0p75_relul`); read in the family's `s1` group because
#: its three steered rows anchor on the family's u0 `none` at the SAME serving window (WINDOW_ANCHOR_RULE), the way the s1 rows do; admitted by the FULL readers alone
DOSE_RELU_SELECTION = "d0p75_relul"
DOSE_RELU_SUFFIX = "_d0p75_relul"
DOSE_RELU_ANCHOR_SEED = 1
#: THE DOSE-1.25 DIFFERENCE-LoRA ROW (:data:`DOSE_LORA_ROW_RULE`): tag `(model key)_d1p25`, row dir `(model key)-f22d1p25`, launch tag
#: `(model key)_d1p25`, the launch set's selection `d1p25` (legacy_spellings.yaml `qwen35_35b_a3b_d1p25`, strength_nominal 1.25 in the LoRA family's unit); read in the
#: family's `s1` group because its one steered row anchors on the family's u0 `none` at the SAME serving window (WINDOW_ANCHOR_RULE); admitted by the FULL readers alone
DOSE_LORA_SELECTION = "d1p25"
DOSE_LORA_SUFFIX = "_d1p25"
DOSE_LORA_ANCHOR_SEED = 1
DOSE_LORA_STRENGTH = 1.25
#: THE LeCun-INIT ReLU ROW (:data:`LECUN_RELU_ROW_RULE`): tag `(model key)_s2rl`, row dir `(model key)-f22s2rl`, launch tag `(model key)_s2rl`,
#: the launch set's selection `s2rl` (legacy_spellings.yaml `qwen35_35b_a3b_s2rl`, strength_nominal 1.0); read in the family's `s1` group because its two steered rows anchor on
#: the family's u0 `none` at the SAME serving window (WINDOW_ANCHOR_RULE); admitted by the FULL readers alone. The objects are FITS, not dose copies: the bank sidecars record
#: fit seed `LECUN_RELU_FIT_SEED` (train_budget_meta.json seed 0, effective batch 64), carried on the rows as extra.fit_seed
LECUN_RELU_SELECTION = "s2rl"
LECUN_RELU_SUFFIX = "_s2rl"
LECUN_RELU_ANCHOR_SEED = 1
LECUN_RELU_FIT_SEED = 0
#: the tinycap model row of a tag: `<model key>-f22s<N>[p]`, or `<model key>-f22it` (the instruct-direct row)
ROW_DIR_RE = re.compile(rf"^(?P<model>{_MODEL_ALT})-f22(?:s(?P<seed>\d+)(?P<paired>p?)|(?P<instruct>{INSTRUCT_SELECTION})|(?P<unsteered>{UNSTEERED_SELECTION})"
                        rf"|(?P<dose_relu>{DOSE_RELU_SELECTION})|(?P<dose_lora>{DOSE_LORA_SELECTION})|(?P<lecun_relu>{LECUN_RELU_SELECTION}))$")
TAG_RE = re.compile(rf"^(?P<model>{_MODEL_ALT})_(?:s(?P<seed>\d+)(?P<paired>p?)|(?P<instruct>{INSTRUCT_SELECTION})|(?P<unsteered>{UNSTEERED_SELECTION})"
                    rf"|(?P<dose_relu>{DOSE_RELU_SELECTION})|(?P<dose_lora>{DOSE_LORA_SELECTION})|(?P<lecun_relu>{LECUN_RELU_SELECTION}))$")
#: the SHAPE of any tinycap model row directory, independent of the table: `<model key>-f22<selection>` (the selection may be empty: the `main` canary's row).
#: What :func:`tag_of_row_dir` does not read but this does is an UNDECLARED row -- a model key the table lacks, or a declared key under a selection the grammar
#: lacks (the 35B mirror's `u0`) -- which discovery reports by name instead of skipping (module docstring, AN UNDECLARED ROW DIR REFUSES BY NAME)
ROW_DIR_SHAPE_RE = re.compile(r"^(?P<model>[a-z][a-z0-9_]*?)-f22(?P<selection>[a-z0-9_]*)$")
UNDECLARED_ROW_DIR_RULE = ("a directory under native/ that spells no declared row -- its model key absent from the model-key table, or a declared key under a "
                           "selection outside <model key>-f22s<N>[p] / -f22it / -f22u0 (the families with an unsteered row) / -f22d0p75_relul (the families with the dose-0.75 "
                           "random-init ReLU row) / -f22d1p25 (the families with the dose-1.25 difference-LoRA row) / -f22s2rl (the families with the LeCun-init ReLU row, under a "
                           "reader that admits it: elsewhere the s2rl row dir is undeclared for that reader, LECUN_RELU_ROW_RULE) / the aliasing families' "
                           "-f22 -- is discovered as a group of its own, named for the "
                           "row dir, and refused as a WHOLE GROUP by name: facts.refuse carries the reason (the status files the group ok = 0, v_source_status folds "
                           "the source to ok = 0 and the paper's guard reads it as not integrated) beside the by-name listing in facts.refusals, with no record "
                           "emitted; a dot-prefixed entry is publish scaffolding and a non-directory is not a row, neither reported; declaring the key moves its "
                           "stores into the seed-tag group")
#: the `main` canary ROW of an appendix family: row dir `<model key>-f22`, tag `<model key>_main`, every store under it aliased to the family's `s1` tag
LEGACY_MAIN_SUFFIX = "_main"
LEGACY_MAIN_ROW_DIR_RE = re.compile(rf"^(?P<model>{_MODEL_ALT})-f22$")
LEGACY_MAIN_TAG_RE = re.compile(rf"^(?P<model>{_MODEL_ALT}){LEGACY_MAIN_SUFFIX}$")
LEGACY_MAIN_ALIAS_SEED = 1
#: the harness roster selection the `main` canary row's objects resolve through (and the appendix families' `s1` objects were copied from), and the
#: spelling a launch set / three-number record may give a canary row
MAIN_SELECTION = "main"
#: the store's suffix: `bcbw-<launch tag>-<sha7>` (the launch tag is the tag, `<tag>_arditi` for the all-positions object, or the canary's `<model key>_main`)
#: the store suffix; `-r<N>` is a RELAUNCH of the (tag, object key) at another producer tree (the launch set's row names the
#: relaunched store on `launched_as` and the parked INCOMPLETE first store on `relaunched_from`, and records/relaunches.json files the relaunch by `<wave>/<cell>`)
#: a ONE-TASK store's task tag (:data:`ONE_TASK_STORE_RULE`): `bcbw-<launch tag>-<sha7>[-r<N>]-bcb` holds the BigCodeBench task alone
TASK_TAGS: Dict[str, str] = {"bcb": "bigcodebench", "bfcl": "bfcl"}
SUFFIX_RE = re.compile(rf"^bcbw-(?P<launch_tag>(?:{_MODEL_ALT})(?:_s\d+p?(?:_arditi)?|{LEGACY_MAIN_SUFFIX}|{INSTRUCT_SUFFIX}|{UNSTEERED_SUFFIX}|{DOSE_RELU_SUFFIX}|{DOSE_LORA_SUFFIX}|{LECUN_RELU_SUFFIX}))-(?P<sha7>[0-9a-f]{{7}})(?P<relaunch>-r\d+)?"
                       rf"(?P<window>-{WINDOW_MARKER}\d+)?(?P<task_tag>-(?:{'|'.join(sorted(TASK_TAGS))}))?$")
ONE_TASK_STORE_RULE = ("Overflow's addendum 2026-09-25 19:58Z: a row whose two tasks ran in two ONE-TASK stores names, per eval, the store that holds its task -- the store "
                       "suffix may carry the task tag (`bcbw-<launch tag>-<sha7>[-r<N>]-bcb` / `-bfcl`) and the row's launched_as names it; a store tagged for ANOTHER task "
                       "is not this source's and is neither read nor refused; a store the row's launched_as does not name that holds none of this source's task files is "
                       "PARKED as the other task's store (never refused); a store the row's launched_as does not name that DOES hold the task's files still refuses by name")
RELAUNCH_RULE = ("Overflow's addendum 2026-09-24 02:31Z: a (tag, object key) launched again at another producer tree files ONE store, the one the launch set's row names on "
                 "launched_as {wave, row_dir, suffix `bcbw-<wave>-<sha7>-r<N>`, sha}; the row's relaunched_from {suffix, why} names the parked first store, which has no RESULT COMPLETE "
                 "and is never read as the row's store (parked, not refused); relaunches.json files the relaunch under `<wave>/<cell>` = {tag r<N>, sha, why} and must name the "
                 "store's tree; a second store dir the row names on neither refuses by name")
TREES_RULE = ("Overflow's addendum 2026-09-24 02:31Z: the launch set's `trees` = [{sha, role}] names its producer trees; a row's `sha` is the tree it was launched at, absent = the set's "
              "primary tree; the store's manifest.json:producer.tree_sha must be that tree")
ARDITI_LAUNCH_SUFFIX = "_arditi"
STORES_HEAD = "native"
THINKING_DIR = "thinking_off"
RECORDS_DIR = "records"
RELAUNCHES_FILE = "relaunches.json"
#: where the relaunch record may sit: beside the other records, or in the producer's own directory at the mirror root (the addendum names both; whichever is mirrored)
RELAUNCHES_DIRS = (RECORDS_DIR, "bcbw_records_full")
GATE_PREFIX = "gate_"
GATE_RE = re.compile(rf"^gate_(?P<tag>(?:{_MODEL_ALT})(?:_s\d+p?|{LEGACY_MAIN_SUFFIX}|{INSTRUCT_SUFFIX}|{UNSTEERED_SUFFIX}|{DOSE_RELU_SUFFIX}|{DOSE_LORA_SUFFIX}|{LECUN_RELU_SUFFIX}))(?P<arditi>{ARDITI_LAUNCH_SUFFIX})?_(?P<okey>.+)_(?P<suffix>bcbw-.+)\.json$")
LAUNCH_SET_FILE = "launch_sets.json"
PINS_FILE = "sdfbase_full22_pins.json"
ROSTER_FILE = "bcb_bfcl_200_widen_roster.yaml"
#: THE ROSTER SET (the appendix run spans producer trees with DIFFERENT launched rosters, so one plain roster file cannot
#: match every store's `lane_roster.json:roster_sha256`): `records/rosters.json` = {roster_sha256: {file, tree_sha, stores[, pins]}} names one mirrored roster
#: file per producer tree (`bcb_bfcl_200_widen_roster.<sha7>.yaml`; the plain name is the latest head) and, where the pins differed, a per-tree pins copy
#: (`sdfbase_full22_pins.<sha7>.json`). A store's roster is matched BY HASH within the set (the named file's sha256 must equal its key; a mismatch or an
#: unmirrored file refuses every store citing that hash by name; a hash the set lacks refuses the store by name); its pins are the per-tree copy for its
#: launch tree when one exists, else the plain file. With no `rosters.json` the plain files are the set (the three-seed family's mirrors)
ROSTERS_INDEX_FILE = "rosters.json"
ROSTER_VARIANT_RE = re.compile(r"^bcb_bfcl_200_widen_roster\.(?P<sha>[0-9a-f]{7,40})\.yaml$")
#: the anchor currency's rule on every contrast of the six capability readers (emission 3 of `_bcbw_common`): the
#: row's OWN pairing (its tag's `none`, the cross-tag fallback or the instruct alias, resolved from the rows that FILED in this batch) is the own anchor and
#: stands while the pass's frame holds it current and readable; otherwise the exact key is searched -- a same-standup or same-source unsteered run (another
#: standup of the same model under this source: the u0 `none`, a seed tag's `none` another group filed) or the paper's roster-matched baseline of another
#: source -- and the registry files the UNPAIRED rate difference from the two readings' k / n (`computed_by registry`, its own computation row; paired
#: per item only when the re-picked run is a row of this very batch); no current anchor at the exact key parks the comparison by name on the VALUE row
#: (`extra.delta_block`) and the previous revision's key is withdrawn by the head
CURRENCY_RULE = ("adapter re-anchoring plan §2 (D642): every contrast is decided by the pass's anchor currency; the row's own pairing stands while current, "
                 "a retired, unfolded or absent own anchor searches the exact key (same standup, same source, the paper's roster-matched baseline), a re-picked "
                 "unsteered run files the registry's unpaired rate difference from the two readings' k / n (paired per item when it is a row of this batch), "
                 "no current anchor parks the comparison by name on the VALUE row's extra.delta_block")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA16 = re.compile(r"^[0-9a-f]{16}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PairRefusal = G._PairRefusal
rounded_equal = G.rounded_equal


def roster_variant_re(roster_file: str) -> "re.Pattern[str]":
    """The per-tree copies of a launched roster file: ``<stem>.<sha7..40>.yaml`` for ``<stem>.yaml`` (the 200 readers' :data:`ROSTER_VARIANT_RE`; the full
    readers' roster is ``bcb_bfcl_full_roster.yaml``, :class:`TaskSpec` ``roster_file``)."""
    stem = roster_file[:-5] if roster_file.endswith(".yaml") else roster_file
    return re.compile(rf"^{re.escape(stem)}\.(?P<sha>[0-9a-f]{{7,40}})\.yaml$")
PINS_VARIANT_RE = re.compile(r"^sdfbase_full22_pins\.(?P<sha>[0-9a-f]{7,40})\.json$")
ROSTER_SET_RULE = ("the launched roster a store's lane_roster.json names by sha256 is matched BY HASH within the mirrored roster set (records/rosters.json: one "
                   "roster file per producer tree, the named file's sha256 equal to its key); the store's launched entry and its fitted key's staged id / sha are "
                   "checked against THAT roster, its launched pins against the per-tree pins copy for its launch tree when one is mirrored, else the plain file; "
                   "a hash the set lacks refuses the store by name; with no rosters.json the plain roster and pins are the set")
ROWS_LAUNCH_FILE = "rows_launch.tsv"
SCHEDULE_FILE = "bcbw_schedule.json"
LAUNCH_SET_SCHEMA = "launch_sets/v1"
THREE_NUMBERS_SCHEMA = "w4_three_numbers/1"
GATE_PASS = "PASS"
SERVING_ENGINE = "lora_native"
STEERED_IDENTITY = SC.STEERED_IDENTITY
PASSTHROUGH_IDENTITY = SC.PASSTHROUGH_IDENTITY
STEERED_PATH_LEGACY = "lora"
PASSTHROUGH_PATH_LEGACY = "passthrough"
N_ITEMS = 200
#: THE FULL-SUITE ITEM SETS (the readers `bigcodebench_full` / `bfcl_full`, :class:`TaskSpec` `item_set_id`):
#: a FULL reader takes its item count from the mirror -- `records/item_sets.json` (`{id, sha256, n, file, tree_file}` per set; the sha256 the mirrored pin
#: FILE's bytes) and the pin file it names (`records/bcb_full.json` / `bfcl_full.json`: `{task, seed: null, n, of, ids, ...}`) -- never from a constant,
#: checks the registry's declaration of the set (`item_sets.yaml` `n_items` / `items_sha256`) against it, and attests every store against it
#: (:meth:`BcbwSeedrepReader._item_set_pin`, :meth:`BcbwSeedrepReader._check_item_set`, :data:`ITEM_SET_RULE`). The 200-item readers keep the pinned
#: count alone (`item_set_id` None; the manifest's seeded 200-of-N block is recorded on `extra.item_set_pin`, unchanged)
ITEM_SETS_FILE = "item_sets.json"
#: the LABEL the harness stamps on gen_stats / the summary / the outcome record of a full-suite pass (the 200-item passes stamp `pinned200`, unchecked there)
ITEM_SET_LABEL_FULL = "full"
ITEM_SET_RULE = ("a full-suite store attests its item set (Overflow's contract 2026-09-24): manifest.json:tinycap.<key> = {seed: null, n, of, ids_sha256} with n == of == the "
                 "item set's n and ids_sha256 == contracts.ids_sha256(pin['ids']) (the sha256 of the canonical JSON list of the mirrored pin file's ids in pin order, never of "
                 "the file's bytes); the pin file under records/ is the one records/item_sets.json names for the set, by file sha256; gen_stats / the summary / the outcome "
                 "record stamp the label `full`; the launch-set row's item_set {id, sha256, n} names the set, the pin FILE's sha256 and its n; the outcome record holds "
                 "exactly n items; item_sets.yaml declares the set with that n and digest. Any disagreement refuses the store by name; an absent item_sets.json or pin file "
                 "refuses every store of the source by name")
CONTROLS_SUMMARY_FILE = "controls_summary.json"
LAUNCHED_ENTRY_FILE = "lane_roster.json"
GEOMETRY_FILE = "served_geometry.json"
#: the files of a store directory and of its `cells/<store key>/` directory (the task files are per spec)
STORE_TOP_FILES = ("RESULT", LAUNCHED_ENTRY_FILE, GEOMETRY_FILE, "evidence.json", "protocol_check.json", "resolved_sampler.json", "usage_ledger.jsonl",
                   "ledger.log", "driver_stdout.log", "driver_stderr.log")
STORE_KEY_FILES = ("gen_stats.json", "manifest.json", "rows.jsonl", "evidence.json")
TASK_SUBDIRS = ("outcomes", "summary", "done")
ROLE_OF_FILE = {"rows.jsonl": "rows", "gen_stats.json": "completion", "RESULT": "log", "usage_ledger.jsonl": "log", "ledger.log": "log",
                "driver_stdout.log": "log", "driver_stderr.log": "log"}
RESULT_COMPLETE = "result=COMPLETE"
RESULT_HEAD = "tinycap-modal"
#: the volume path every store's manifest places its usage ledger under (PROVENANCE.md: held equal at publish); the substrate witness
VOLUME_HEAD = "/vol/tinycap/gen/"
SANDBOX_PROVIDER = "modal_owned"
SUBSTRATE = "modal"
ROSTER_REL = SC.ROSTER_REL
PINS_REL = SC.PINS_REL
ARDITI_ROSTER_REL = SC.SINGLE_OBJECT_ROSTER_REL
ARDITI_PINS_REL = SC.SINGLE_OBJECT_PINS_REL
ARDITI_OBJECT_KEY = DG.OBJECT_KEY_LEGACY
ARDITI_ROLE = G.ARDITI_ROLE
LAUNCH_CLASSES = ("anchor", "arm", "prompted", "composed", "arditi")
ITEM_LABEL_AT_CAP_UNRESOLVED = IC.ITEM_LABEL_AT_CAP_UNRESOLVED
ITEM_LABEL_UNSCORED = G.ITEM_LABEL_UNSCORED
AT_CAP_CLASS = "ceiling"
UNSCORED_BELOW_CAP_CLASS = "unscored_below_cap"
RAW_AT_CAP_MEASURE = G.RAW_AT_CAP_MEASURE
HEADLINE_METRIC = "acc"
BOUND_SPELLING = "bound"
BOUND_CI_METHOD = G.BOUND_CI_METHOD
#: the run row's seed rule = the INSTRUMENT's and the protocol policy row's (`none`: a greedy protocol pins no seed), never Inspect's incidental
#: per-item derivation, which rides extra.decode_seed_rule_text / extra.seed_axis (module docstring, DECODE POLICY: the derived match requires the
#: policy row's seed_rule to equal the run's; emission 1 filed `per_item` and read no_match on every row)
DECODE_SEED_RULE = "none"
INSPECT_SEED_POLICY_DEFAULT = "default"
SAMPLING_MODE = "protocol"
CONTRAST_METHOD = G.CONTRAST_METHOD
ROUNDED_TOL = G.ROUNDED_TOL
RECOUNT_VERSION = "1"
#: the rules every row cites
PRODUCER_KEY_RULE = "the producer key is `<tag>/<store key>`: the tag prefixes the store's own NAME (never its path), so a paired tag's store never shares a key with a seed tag's"
GROUP_RULE = ("one group per SEED: the seed tag `qwen35_4b_s<N>` and its paired tag `qwen35_4b_s<N>p` read in one batch — the anchor rule (D65) pairs the "
              "composed row across tags with `s<N>/none`, and a contrast names a run its own batch emits")
ANCHOR_RULE = ("D65: every steered and prompted row pairs with ITS seed tag's `none`; the composed `s<N>p` row pairs ACROSS tags with `s<N>/none`; the "
               "all-positions rows are declared unpaired (arditi_unpaired_value) — the value files regardless and the contrast carries the role; a seed tag "
               "whose `none` has not run yet has no anchor: its rows file without a contrast and extra.anchor names the reason")
SEED_AXIS_RULE = ("D69: no per-seed decode rule on this evaluation — the protocol sampler is greedy under the DEFAULT Inspect seed rule (no replicate block), so the "
                  "seed axis is the fitted OBJECT; every row files seed_base NULL, trial_index 0 and decode_seed_rule none (the instrument's and the protocol policy "
                  "row's rule: a greedy protocol pins no seed; Inspect's incidental per-item derivation rides extra.decode_seed_rule_text / extra.seed_axis)")
NOT_RUN_RULE = ("a launch-set row with no store (the gate record reads FAIL: the task has not COMPLETED) files NOTHING here — no run, no refusal, no "
                "disposition (the producer files those); a later mirror adds the store idempotently")
#: the launch set's row table (shared by the three capability readers): `launch` holds the rows the CURRENT pass
#: launched (census_status run), `carried` / `carried_late` the rows carried from the earlier pass (census_status carry); the first launch set carried
#: nothing and named every row under `launch`. A reader that read `launch` alone refused every carried row ("the launch set names no row")
ROW_TABLE_RULE = ("the launch set's row table is evals.<eval>.launch (the rows the current pass launched, census_status run) plus evals.<eval>.carried and "
                  "carried_late (the rows carried from the earlier pass, census_status carry); counts.table is their sum; carried_optional, optional_not_carried "
                  "and exists_not_carry are census notes outside the table, never rows")
ROW_TABLE_LISTS = ("launch", "carried", "carried_late")
MAX_LORA_RANK_RULE = "RECORDED, never compared: the gate record's recorded_flags.max_lora_rank / zero_adapter_only and the standup's lora_config (D71 / GSM8K C4)"
GATE_RULE = "bcbw_gate.py (D94): a row files only if its gate record reads verdict PASS and attests route.serving_engine lora_native; a FAIL or an absent record for a present store refuses the row"
ANCHOR_NOT_YET_RUN = "anchor_not_yet_run"
ANCHOR_REFUSED = "anchor_refused"
ANCHOR_CONTRASTED = "contrasted"
#: the paired tag's OWN `none` (the GSM8K precedent). Shared by the three capability
#: readers (math500_seedrep imports it too). Stamped ONLY on the records it governs — the paired `none` run, and a paired row and its contrast once
#: that `none` files — so every record's rule text describes the anchor IT was formed against; ANCHOR_RULE / GROUP_RULE stay byte for byte on the
#: rows they describe (no emission bump over unchanged mirrors)
PAIRED_ANCHOR_RULE = ("the paired tag `qwen35_4b_s<N>p` is a standup replicate of `qwen35_4b_s<N>` (D429, the GSM8K precedent): its OWN `none` is its anchor once it "
                      "files (anchor_kind same_standup_none, cross_standup false, anchor_view_legacy the paired tag); until then the composed row pairs ACROSS tags with "
                      "`s<N>/none` (cross_standup true, the D65 fallback) and extra.anchor.own_none_state names the pending `none`; a seed tag's rows never pair with the "
                      "paired tag's `none`")
#: the appendix families' reading (module docstring, THE APPENDIX FAMILIES). Stamped on THEIR rows alone (extra.appendix_family); the
#: qwen readers' records stay byte for byte
APPENDIX_FAMILY_RULE = ("D565: an appendix family (gemma4e4b / llama8b / nemotron3nano4b) runs ONE seed tag `<model key>_s1` and its paired tag `<model key>_s1p` "
                        "under the seed-replicate grammar with the model key in place of qwen35_4b; the harness roster stages no s1 / s1p selection for it, so a "
                        "fitted key's staged id is the launch set's pin key, pinned in the harness pins (configs/sdfbase_full22_pins.json), and a paired tag's "
                        "paired_of is the tag grammar's; the family's `main` canary row (<model key>-f22, tag <model key>_main) ran before the s1 tag existed and "
                        "its stores file under <model key>_s1 (extra.tag_legacy_original), their objects the roster's `main` selection's")
#: the composed (paired-write) objects of the appendix families: the launched pins key them `paired_eb<batch>__<cell>` (gemma /
#: nemotron eb8, llama eb64) while the repo's harness pins do not carry them; a composed store whose lane_roster.json names an object_sha256 resolves through
#: the STORE'S TREE PINS by that sha (any key under the model's block), the key found recorded on extra.launched_pins_key and required to be the launch set's
#: pin key. A bare-name hit in the harness pins stays as it was; a fitted `arm` key the harness pins lack, or a composed sha the tree pins lack, refuses by name
PINS_KEY_RULE = "matched by object_sha256; the pins spell a composed cell paired_eb<batch>__<cell>"
OBJECT_ROUTE_LAUNCHED_PINS_BY_SHA = "launched_pins_by_object_sha256"
LEGACY_MAIN_ALIAS_RULE = ("D565: the `main` canary row native/<model key>-f22/<object key>/thinking_off/bcbw-<model key>_main-<sha7>/ carries the family's seed-tag "
                          "conditions run before the `<model key>_s1` tag existed (the none, the prompted keys, the ReLU and LoRA keys); completed work is canonical, "
                          "so every store of the row files as <model key>_s1/<object key> (the none the seed rows' anchor) with the original tag on "
                          "extra.tag_legacy_original, its fitted objects resolved through the harness roster's `main` selection for the family (the launch set's pin "
                          "key must be that staged id); a key present under both the `main` row and the `s1` row refuses the `main` row's store by name")
#: the unsteered row and the window-matched anchor (stamped on the rows of a family that declares
#: `unsteered_row` alone: extra.unsteered_row / extra.serving_window / the contrast's anchor_rule -- every other family's record stays byte for byte)
UNSTEERED_ROW_RULE = ("the unsteered row `<model key>_u0` (row dir <model key>-f22u0, launch tag <model key>_u0, the launch set's selection u0) holds the family's `none` baseline(s) "
                      "and its prompted keys and reads in the family's s1 group; a `none` may be RE-SERVED at another serving window as its own store, suffix segment "
                      "`-w<N>` (N = vLLM max_model_len; the generation cap is the instrument's and unchanged), keyed `none@w<N>` and declared by the launch set's "
                      "families.<model key>.unsteered.window_reserves.<eval> (one launch-set row per (tag, key) cannot hold a second none); a store tagged -bcb / -bfcl "
                      "is the row's one-task store for that eval (ONE_TASK_STORE_RULE), untagged = the BigCodeBench store")
DOSE_RELU_ROW_RULE = ("the dose-0.75 random-init ReLU row `(model key)_d0p75_relul` (row dir (model key)-f22d0p75_relul, launch tag (model key)_d0p75_relul, the launch set's "
                      "selection d0p75_relul, legacy_spellings.yaml qwen35_35b_a3b_d0p75_relul) holds three STEERED rows -- the two parent dose copies and their difference, "
                      "random-init ReLU objects bound by object id through the pins block (the appendix route: the launch set's pin key is the staged id; the two parents are "
                      "the launch's extras, not harness-roster entries) -- read in the family's s1 group; every row anchors on the family's u0 `none` at the SAME serving window "
                      "(WINDOW_ANCHOR_RULE, the s1 rows' rule); the three-number row may spell the fit seed as the group's or as none (the model's standup-group convention); a "
                      "`none` re-served at the launch's tree under the u0 row shares the (u0, none) row key with the declared store and binds no new row (a second store on one key "
                      "is not read: the launch row's tree names the store); its rows carry NO fit seed (extra.fit_seed null, run.seed_base null): the s1 group is a standup "
                      "grouping, not the object's fit provenance -- the parent fits' sidecars record seed 0 and a dose-scaled copy's fit seed is not applicable")
DOSE_LORA_ROW_RULE = ("the dose-1.25 difference-LoRA row `(model key)_d1p25` (row dir (model key)-f22d1p25, launch tag (model key)_d1p25, the launch set's selection d1p25, "
                      "legacy_spellings.yaml qwen35_35b_a3b_d1p25 at strength_nominal 1.25 in the LoRA family's unit) holds ONE steered row -- the dose_scale copy of the "
                      "s1 row's difference LoRA at base_scale 1.25, bound by object id through the pins block (the appendix route: the launch set's pin key is the bank's "
                      "staged id d1p25__<difference key>; a key the pins do not carry refuses the row by name, never a guessed object) -- read in the family's s1 group; the "
                      "row anchors on the family's u0 `none` at the SAME serving window (WINDOW_ANCHOR_RULE, the s1 rows' rule); its three-number row may spell the fit seed "
                      "as the group's or as none; it carries NO fit seed (extra.fit_seed null, run.seed_base null: a dose-scaled copy's fit seed is not applicable, and the "
                      "s1 group is a standup grouping); a `none` re-served at the launch's tree under the u0 row shares the (u0, none) key with the declared store and binds "
                      "no new row")
LECUN_RELU_ROW_RULE = ("the LeCun-init ReLU row `(model key)_s2rl` (row dir (model key)-f22s2rl, launch tag (model key)_s2rl, the launch set's selection s2rl, legacy_spellings.yaml "
                       "qwen35_35b_a3b_s2rl at strength_nominal 1.0) holds TWO steered rows -- the control ReLU and the constitution ReLU refit from a LeCun random init into the "
                       "full22_lecun bank (FITS, not dose copies), each bound by object id through the pins block (the appendix route: the launch set's pin key is the bank's staged "
                       "id lecun__<ReLU key>; a key the pins do not carry refuses the row by name, never a guessed object) -- read in the family's s1 group; the harness's s2rl "
                       "selection lists a third object (the composed difference, lecun_derived__) the launch never ran: no store, no launch row, no pin, so it is not a row and "
                       "nothing refuses for it; each row anchors on the family's u0 `none` at the SAME serving window (WINDOW_ANCHOR_RULE, the s1 rows' rule); extra.fit_seed is "
                       "the objects' RECORDED fit seed (LECUN_RELU_FIT_SEED: the bank sidecars' train_budget_meta.json seed 0 at effective batch 64, declared here because the "
                       "sidecar is not in the mirror), the launch row spells seed_base as the group's (the standup seed the u0 pairing is keyed on) and the three-number row may "
                       "spell the seed as the group's or as none; run.seed_base stays NULL like every row of this evaluation (SEED_AXIS_RULE); a `none` re-served at the launch's "
                       "tree under the u0 row shares the (u0, none) key with the declared store and binds no new row; under a reader that does not admit the row (the "
                       "200-item readers) or a family that does not run it, the row dir (model key)-f22s2rl is UNDECLARED for that reader (UNDECLARED_ROW_DIR_RULE): a group of "
                       "its own refused as a WHOLE by name, so a misplaced or early-launched s2rl store reads ok = 0 on the status -- never a by-name row refusal beside rows "
                       "that file, which leaves source_status.ok at 1")
WINDOW_ANCHOR_RULE = ("every steered or prompted row of a family with an unsteered row anchors on the u0 `none` served at the SAME serving window and task (D1892: the pair must "
                      "not differ by serving window; the caps are equal within an eval so the window picks the baseline): the row's own tag's none for a u0 row "
                      "(same_standup_none), the u0 none ACROSS tags for an s1 row (sibling_none, anchor_view_legacy <model key>_u0, cross_standup true -- D1878's same-source "
                      "sibling tier); matched on the run rows' context_window_tokens (and served_max_num_seqs where both carry it); no same-window none declared refuses "
                      "the contrast by name (no_same_window_none), one declared but not filed waits (anchor_not_yet_run / anchor_refused)")
WINDOW_PARK_NOTE = ("the currency was not asked: the window rule names the baseline (the u0 `none` served at the row's own window), and the exact key has no window "
                    "term, so a search there would offer a baseline of another serving; the comparison parks by name instead")
#: the instruct-direct rows (declared 2026-09-25 19:13Z). Stamped on THEIR rows alone (extra.instruct_direct / extra.anchor_alias); every
#: seed-tag record stays byte for byte
INSTRUCT_DIRECT_RULE = ("Overflow 2026-09-25: the instruct-checkpoint LoRA fits of gemma4e4b / llama8b / qwen35_4b (constitution_lora / control_lora / difference_lora), fit on "
                        "AND served on the instruct model, run under the tag `<model key>_it` (row dir `<model key>-f22it`, launch tag `<model key>_it`, the launch set's "
                        f"selection `{INSTRUCT_SELECTION}`, block and transfer_kind {INSTRUCT_TRANSFER_KIND}, seed_base null, class `{INSTRUCT_CLASS}`); a fitted key resolves "
                        f"through the harness roster's `{INSTRUCT_SELECTION}` selection (staged id -> pins, the harness's own resolution) and the launch set's pin key must be "
                        "that staged id; transfer_kind is DERIVED from manifest.json:fit_provenance.model_id against the served model_id (the same checkpoint -> instruct_direct) "
                        "and must agree with legacy_spellings.yaml's spelling of the tag; the rows read in the family's s1 group; a `_it` store of an unfitted key (none / "
                        "prompted / all-positions), of a family without the tag, or under a 200-item reader refuses by name")
LAUNCHED_AS_TREE_RULE = ("Overflow's convention (2026-09-25 23:0xZ, on every relaunched row): a launch row's `sha` IS the tree of the store the row reads, and `launched_as` names that "
                         "store with the same `sha`; a row whose `sha` differs from `launched_as.sha` is an INCONSISTENT record and refuses BY NAME -- no store is read for the row "
                         "under either tree until the record agrees (the three D308b rerun rows of 22:16Z were a derive defect, re-mirrored corrected); relaunches.json's entry must "
                         "name that tree too; a store the row names on launched_as that exists under the root but was not read (a suffix outside the grammar, a task tag of another "
                         "source, no files), a store directory the grammar does not read, and a store both named on launched_as and parked on relaunched_from refuse by name -- "
                         "never a silent drop; a not-run row names the launched_as store it is missing")
#: STAMPED on the instruct-direct rows (extra.anchor.rule / extra.anchor_rule on the run and the contrast, the status's anchor_aliases) and kept byte for
#: byte: its "anchor_kind same_standup_none" clause is the 2026-09-25 spelling; since emission 3 the contrast's FIELD is anchor_kind_for's (sibling_none)
#: and this text is not re-spelled here, so the bump moves the 15 instruct-direct rows on that field alone
INSTRUCT_ANCHOR_RULE = ("the `_it` none is carried BY ALIAS to the family's s1 none (the launch set's class-anchor row it/none with anchor_tuple s1/none; no `_it` none store exists "
                        "and none is read, so no duplicate none run files): every instruct-direct row contrasts against <model key>_s1/none -- anchor_kind same_standup_none, as "
                        "the Arditi rows anchored on a seed tag's none file; cross_standup true (another tag, another standup); extra.anchor_alias names the alias -- and until "
                        "that none files the row files without a contrast and extra.anchor names why")

# ---- THE SHARED-STANDUP COMPARISON SOURCES (2026-09-28; tinycap-shared-standup) ------------------------------------------------
#: the row-engine harness serves EVERY object of a model row on ONE vLLM engine and re-runs the canary rows (gemma s1, qwen s1, nemotron s1,
#: the qwen instruct row's own none) as whole rows, COMPARED to the existing solo stores: the 78 canonical rows and the paper anchors are untouched.
#: The stores are mirrored under NEW sources (`bigcodebench_full_shared` / `bfcl_full_shared`) read by the solo readers PARAMETRIZED
#: (:class:`SharedSpec` on :class:`TaskSpec`; every solo reader carries `shared` None and reads byte for byte what it read before): the same
#: layout, records and cross-checks, plus what the row engine adds. Every rule below is stamped on the shared rows alone.
#: the registry spelling of a shared row: `<tag>_shared` (`gemma4e4b_s1_shared`, `qwen35_4b_it_shared`) -- a spelling no roster selector names
SHARED_SPELLING_SUFFIX = "_shared"
#: the store suffix's marker after the sha7: `bcbw-<launch tag>-<sha7>-shared[-r<N>][-rep<N>]`
SHARED_STORE_MARKER = "shared"
SHARED_REPEAT_MARKER = "rep"
SHARED_LAUNCH_SET_FILE = "launch_sets_shared.json"
SHARED_SCHEDULE_FILE = "bcbw_schedule_shared.json"
#: the comparison kinds a shared launch row carries: a row-engine store against the same (row, object key) served solo;
#: the row's `none` driven a second time on the SAME standup (`-rep2`: the in-run floor); the instruct row's OWN none served by the row engine
#: (a comparison row only -- never an anchor, no contrast)
COMPARISON_KIND_SHARED_VS_SOLO = "shared_vs_solo"
COMPARISON_KIND_REPEAT_FLOOR = "repeat_floor"
COMPARISON_KIND_IT_NONE = "it_none_comparison"
COMPARISON_KINDS = (COMPARISON_KIND_SHARED_VS_SOLO, COMPARISON_KIND_REPEAT_FLOOR, COMPARISON_KIND_IT_NONE)
#: the launch set's class of the repeat row and of the instruct row's own none
REPEAT_CLASS = "repeat"
IT_NONE_CLASS = "unsteered"
#: the shared - solo contrast's anchor_kind (vocab.ANCHOR_KINDS; migration 0027): the solo run, another source's row on the same
#: instrument, STANDS IN as the anchor. Until 0027 the stand-in shared `borrowed` with the anchor currency's tier-1c re-anchoring on another
#: source's roster-matched `none` (:data:`anchor_current.KIND_BORROWED`), and `v_paper_long`'s one documented tie-break -- anchor_kind -- could
#: not tell the two contrasts of a shared key apart; the stand-in has its own kind now, and `borrowed` names the re-anchoring alone
ANCHOR_KIND_SOLO_STAND_IN = "solo_stand_in"
#: the anchor currency's tier-1c kind (the paper's roster-matched baseline of ANOTHER source re-picked for a row whose own none retired)
ANCHOR_KIND_BORROWED = "borrowed"
COMPARABILITY_CLASSES = ("same_standup", "cross_standup")
#: the manifest's ONLY addition on a row-engine store (plan 1.1.4; UNHASHED): exactly these eight keys
ROW_ENGINE_KEYS = ("row_engine", "standup_id", "engine_cells", "max_loras", "residual_slot_armed", "compile_cache", "row_dir", "verification_pass")
_STANDUP_ID = re.compile(r"^[0-9a-f]{32}$")
#: the solo store's files the shared reader reads for the comparison (registered as the group's inputs under `#<group>`, so a re-mirrored solo store re-reads the group)
SOLO_STORE_FILES = ("RESULT", "manifest.json")
SHARED_SOURCE_RULE = ("D583 (2026-09-28): the shared-standup canary rows are a COMPARISON source pair (bigcodebench_full_shared / bfcl_full_shared), never a replacement of "
                      "the 78 canonical rows and never an anchor change: the solo readers parametrized (the same layout, records and cross-checks; every solo record re-emits "
                      "byte for byte), the rows filed on the views `<model key>_s1_shared` / `<model key>_it_shared` (spellings no roster selector names), each row a run with "
                      "extra.comparison_kind in {shared_vs_solo, repeat_floor, it_none_comparison} and extra.solo_run_id / extra.solo_store, the shared - solo comparison a "
                      "REGISTRY-computed run_contrast with anchor_kind solo_stand_in (the solo run stands in; both numbers; D932 (b): its own kind, so a shared key's "
                      "stand-in and its re-anchored steering contrast, anchor_kind borrowed, tell apart by anchor_kind alone) -- the comparison facts live in "
                      "v_source_status (the status page renders no contrast), never in v_paper_current")
SHARED_RUN_ID_RULE = ("a shared store's cells/<store key> is byte-identical to its solo store's (K = 1: the same hashed served identity), so the run id is minted under the "
                      "`_shared` spelling -- producer key `<tag>_shared/<store key>` -- never the solo tag's, and the repeat store folds its segment into the producer key "
                      "(`<tag>_shared/<store key>@rep<N>`: it shares the store key with the row's none); a relaunch (`-r<N>`) is the SAME row and keeps the row's id, as every "
                      "relaunch does; the launch row's solo_run_id must be the id the SOLO reader mints for the solo store (`<tag>/<solo store key>`) and never this row's own")
SHARED_ROW_KEY_RULE = ("the repeat row shares (tag, object key) with the row's none: rows are keyed by (tag, object key[@rep<N>]) with the repeat segment read off "
                       "launched_as.suffix (the authority for the row's store) and cross-checked against tuple_id / repeat.key, never by (tag, object key) alone")
SHARED_SPELLING_RULE = ("a shared launch row carries BOTH `view` (the registry spelling `<tag>_shared`, the run's view_legacy) and `selection` (the solo selection s1 / it, "
                        "the row's key in the group); a row whose `view` is not the `_shared` spelling of its selection's tag refuses by name; lane_roster.json:wave, the store "
                        "suffix and the gate record spell the LAUNCH tag (launch_tag_of unchanged), never the `_shared` spelling")
SHARED_IT_NONE_RULE = ("the instruct row's OWN none served by the row engine files as a run (class unsteered, comparison_kind it_none_comparison, no solo_run_id, no "
                       "anchor_tuple) on the `<model key>_it_shared` spelling: a comparison row only -- NOT an anchor (no instruct-direct row contrasts against it; the paper "
                       "anchor for `_it` rows stays the carried s1 none, sibling_none, D308) and NO run_contrast of its own (D583 / D5)")
SHARED_STANDUP_RULE = ("one standup id per row launch: manifest.json:row_engine.standup_id = served_geometry.json:endpoint.standup_attempt_id = RESULT's `standup=` token = the "
                       "launch row's standup_id = the gate record's standup_id, the same on every store the row launch names on launched_as.suffix (a `-r<N>` relaunch has its "
                       "own); the engine's object keys (row_engine.engine_cells = endpoint.cells) name this store's key; the per-object served digest is "
                       "provenance.artifact_sha256 (unchanged) beside the row's served-digest map by harness key -- served_geometry.json:endpoint.object_artifact_sha256_by_cell, "
                       "manifest.json:route.object_artifact_sha256_by_cell where the manifest carries it too (SHARED_ENDPOINT_TUPLE_RULE); route.object_artifact_sha256 is null on "
                       "a row engine, S11, and is never read as the digest); max_loras / residual_slot_armed are RECORDED, never compared (D82 / GSM8K C4)")
SHARED_COMPARISON_RULE = ("the shared - solo comparison is the registry's: item-wise over the items SCORED on both sides (the same items, one instrument, the greedy protocol) "
                          "-- the accuracy difference in pp, McNemar's exact p on the discordant pairs, Cohen's h -- against the SOLO store's outcome record read from the solo "
                          "mirror (the solo run stands in as the anchor: anchor_kind solo_stand_in, anchor_run_id the solo reader's id, cross_standup true); both headline values ride "
                          "the contrast; the producer's delta_value_vs_solo is recorded beside it, never filed as the comparison; a solo store the mirror lacks leaves the run "
                          "filed with extra.comparison.state solo_store_absent and no contrast until it lands")
SHARED_ANCHOR_STANDUP_RULE = ("a shared row's in-run reference is the row's shared none: same_standup_none when both were served by the same standup (equal standup ids), "
                              "sibling_none when the row was relaunched on its own standup (`-shared-r<N>`; the gate's comparability_class cross_standup)")
# ---- the store-attestation decision of 2026-09-29 (after the first real mirror, landing 1): three reader rules ----------------
#: RULE 1: `manifest.row_engine.row_dir` may be the volume-absolute `/vol/tinycap/gen/rows/<row>/<suffix>`; the reader binds it by the `rows/<row>/<suffix>` TAIL
#: and records the absolute form verbatim (on the row's extra, never compared); a tail that is not the store's own row / row-launch suffix refuses
SHARED_ROW_DIR_RULE = ("manifest.json:row_engine.row_dir may be the volume-absolute `/vol/tinycap/gen/rows/<row>/<suffix>` (landing 1's stores) or the relative `rows/<row>/<suffix>`: "
                       "the reader binds it by the `rows/<row>/<suffix>` TAIL -- this store's row dir, and the ROW LAUNCH's container suffix `bcbw-<the row's primary launch tag>-<sha7>-shared"
                       "[-r<N>]` (the row's tag, never the Arditi store's own `_arditi` launch tag; the store's own relaunch segment; no repeat segment: a `-rep<N>` store rides its row's "
                       "container) -- equal to the launch row's own row_dir tail where the row declares one (the real mirror's rows do); the spelling is recorded verbatim on "
                       "extra.row_engine.row_record_dir (never compared), the container's sha7 beside it (row_record_dir_tree_agrees: the store's tree in a one-tree row); a tail "
                       "that is not this store's row / row launch refuses by name; the gate record's row_dir is held to the same tail (the decision of 2026-09-29)")
#: RULE 2: `manifest.route.object_artifact_sha256_by_cell` may be null while `served_geometry.json:endpoint` carries the row's map, and `endpoint.cell` may name the
#: row's LAST key instead of this store's (landing 1's row runner, fixed for later stores): the served map is the endpoint's, the store's tuple the manifest's, both
#: cross-checked against RESULT; a disagreement refuses; the last-key endpoint reads with `object_id` downgraded and a source_status note; an agreeing endpoint reads clean
SHARED_ENDPOINT_TUPLE_RULE = ("the served-digest map is read from served_geometry.json:endpoint.object_artifact_sha256_by_cell (manifest.json:route.object_artifact_sha256_by_cell may be "
                              "null on a landing-1 store; when both are present they must be equal); the store's tuple is the manifest's (cell_id = the store key's object = the store "
                              "directory; harness_cell_id the map's key), never endpoint.cell; the three are CROSS-CHECKED -- the manifest's key, the endpoint map's entry for it (the "
                              "pinned object's digest = provenance.artifact_sha256 for a fitted key; no entry for an unsteered / prompted key) and RESULT (its standup= token = the "
                              "manifest's standup id; its line = the producer's three-number row for this tuple, result_line) -- and a store where they disagree refuses by name; an "
                              "endpoint whose `cell` / `harness_cell` fields name the row's LAST served key (landing 1's row runner) reads with field_confidence_json.object_id downgraded one step "
                              "(recorded -> derived: the object is derived from the manifest's tuple, the endpoint map's entry and RESULT, not recorded by the endpoint's own served-key "
                              "fields; an unsteered key's, already derived, keeps its level with the bug named in its source; extra.row_engine.object_id_confidence says whether the level "
                              "moved) and a source_status note (extra.row_engine.endpoint_served_key_note; status extra endpoint_served_key_mismatch); an endpoint naming any other key "
                              "refuses; an endpoint that agrees with the manifest reads clean with no note (the decision of 2026-09-29)")
#: the harness defect the note names (the shape admitted: the row's LAST served key on every store's endpoint; nothing else)
ENDPOINT_SERVED_KEY_BUG = ("landing 1's row runner wrote the row's LAST served key into every store's served_geometry.json:endpoint.cell / harness_cell (fixed for later stores): on "
                           "those stores the endpoint's own served-key fields name another store's tuple and are never read as this store's; the tuple is the manifest's, the served "
                           "digest the endpoint map's entry for the manifest's harness key")
#: RULE 3: an instruct row's `none@rep2` store (`<model key>_it|none@rep2`, suffix `...-shared-rep2`) is a repeat_floor row: the repeat segment is read BEFORE the
#: instruct-none classification, so it is admitted as repeat_floor, never refused and never filed as it_none_comparison
SHARED_IT_REPEAT_RULE = ("an instruct row's OWN none served a second time on the same standup (`<model key>_it|none@rep2`, suffix `...-shared-rep2`) is a repeat_floor row, not an "
                         "it_none_comparison: the repeat segment is read off the suffix BEFORE the instruct-none classification, the row files under class repeat with no solo_run_id "
                         "and no solo_store (the `_it` none has no solo twin, D5 / D308: extra.comparison.state no_solo_twin, no borrowed contrast), and its in-run floor pairs "
                         "against the row's own none (the it_none_comparison run; anchor_kind same_standup_none) -- the one contrast that run anchors, a same-object pair, never an "
                         "instruct-direct row's anchor (the decision of 2026-09-29)")
#: extra.comparison.state of the instruct row's repeat floor: no solo twin exists, so no borrowed contrast is waited for
SOLO_STATE_NO_TWIN = "no_solo_twin"
#: the solo run is not a current run of the pass's frame (CURRENCY_RULE): the borrowed comparison waits; the decision input re-reads the group when it is
SOLO_STATE_NOT_CURRENT = "solo_not_current"
# ---- landing 2 (2026-09-30): the SAME comparison rows run again by a DIFFERENT harness, published as SEPARATE sources --------------------
#: the `manifest.row_engine.verification_pass.state` a landing-2 store spells once the CPU verifier re-stamped its RESULT (landing 1's block is null)
VERIFICATION_STATE_VERIFIED = "verified"
#: the harness-variant segment of landing 2's sources (`bigcodebench_full_shared_l2` / `bfcl_full_shared_l2`; the producer key's `<tag>_shared_l2`)
SHARED_L2_HARNESS_VARIANT = "l2"
SHARED_L2_RUN_ID_RULE = ("a run id is a function of the PRODUCER's identity (identity.run_id: campaign, producer key, object key, eval, instrument), and landing 2 is another "
                         "producer of the same tuples -- K = 2 drivers per row plus a CPU verification pass, the row's stores published as their own sources "
                         "(bigcodebench_full_shared_l2 / bfcl_full_shared_l2) because the landing-1 sources' ids would collide on a second run of one tuple -- so the "
                         "discriminator sits in the producer key's SOURCE segment, the half that already spells the shared harness (`<tag>_shared`): landing 2's producer key "
                         "is `<tag>_shared_l2/<store key>[@rep<N>]` (SharedSpec.harness_variant folded into the spelling segment; the store key half and the repeat segment "
                         "unchanged), never a suffix on the finished id, never the instrument (the same decode policy and item set: one instrument) and never the store key's "
                         "hash (it happens to differ under K = 2, but an identity that leaned on it would collide the day a landing hashed alike); the registry spelling stays "
                         "`<tag>_shared` (view_legacy, legacy_spellings.yaml: the same model / fit / transfer as landing 1 -- the landing is producer identity, not a "
                         "spelling), so the launch row's `view` and the three-number row's `view` keep spelling `<tag>_shared`; the launch row's solo_run_id is still what "
                         "the SOLO reader mints; a landing's BOUND launch set may spell evals.<eval>.own_source as the BASE source's name (landing 2's does): the reader "
                         "binds by ROOT (the mirror root IS the source) and records the spelled name as provenance (extra.launch_set.own_source_spelled), never a refusal")
SHARED_VERIFIED_STANDUP_RULE = ("on a landing-2 store (manifest.row_engine.verification_pass.state = verified) RESULT's LAST line is the CPU verifier's re-stamp "
                                "(`result=COMPLETE status=OK sha=<tree> verified_by=<function call id>`, WITHOUT a standup= token): an ABSENT token on the last line is admitted "
                                "ONLY when the store is verified AND the manifest's and the endpoint's standup ids agree; the standup id is then manifest.row_engine.standup_id, "
                                "cross-checked against EVERY RESULT line that carries standup= (the row runner's earlier GENERATED line; the last line too when it carries one) -- "
                                "a line spelling another id refuses by name, verified or not, and a RESULT with no standup= token at all reads the manifest's alone "
                                "(extra.row_engine.result_standup_source says which); the re-stamp's verified_by must equal verification_pass.function_call_id when both are "
                                "present (two carriers of one fact); a landing-1 store (verification_pass null, or state != verified) keeps the last-line rule unchanged: "
                                "RESULT's last line must carry standup= equal to the manifest's. Overflow's PR #441 makes later re-stamps carry standup=<manifest id> before "
                                "verified_by=, so the absent-token path covers exactly landing 2's 19 stores (tree 082f583; the gemma and qwen f22s1 rows)")
#: the root binds the source, so the producer's three-number record must NAME it: its top-level `source` header is held to spec.source before any row files
#: (landing 2's launch set legitimately spells landing 1 under own_source and an unverified landing-1 store passes the standup rule, so a landing-2 root
#: accidentally populated with landing-1 stores + records would otherwise file duplicate landing-1 measurements under new ids)
THREE_NUMBERS_SOURCE_RULE = ("the three-number record's top-level `source` must equal the reader's source: the mirror root binds the source and mints the producer keys, "
                             "and the producer's record is the one file under records/ that names which source its rows belong to (every real record does; the launch set's "
                             "own_source may legitimately spell another landing's name and is provenance only), so a record spelling another source is a wrong root or a "
                             "wrong mirror and every row of the group refuses by name before any files")
# ---- landing 3 (the fan-out test on the re-stamping harness, 2026-10-01): the gemma row's four tuples once more, published as a third source pair ----------
#: the harness-variant segment of landing 3's sources (`bigcodebench_full_shared_l3` / `bfcl_full_shared_l3`; the producer key's `<tag>_shared_l3`)
SHARED_L3_HARNESS_VARIANT = "l3"
SHARED_L3_RULE = ("landing 3 (Overflow's D125 fan-out test, 2026-10-01; D113 generalized to a third landing): the gemma4e4b s1 row's four tuples (none, prompted__safety_refusal, "
                  "prompted__si_constitution, constv2-mix-full__psr_ce_all) on tree f1347e9 driven once more by the PR #441 harness and published as the sources "
                  "bigcodebench_full_shared_l3 / bfcl_full_shared_l3 -- landing 1's reader parametrized as on landing 2 (SharedSpec.harness_variant l3 in the producer key's "
                  "spelling segment, `<tag>_shared_l3/<store key>`; the registry spelling stays `<tag>_shared`; the launch row's solo_run_id is the SOLO reader's; the three-number "
                  "record's `source` spells this source, THREE_NUMBERS_SOURCE_RULE). Three shapes differ from landing 2's 19 stores and none is a gate: (1) "
                  "manifest.row_engine.verification_pass carries the fourteen current keys (the V1 nine + verifier_tree_sha, generation_tree_sha, executed_image, "
                  "executed_verify_timeout_s, binding_version) -- the reader pins the row_engine block's eight keys, reads the verified block's state and function_call_id alone "
                  "and RECORDS the block verbatim (extra.row_engine.verification_pass, extra.gate.verification_pass), so the nine- and the fourteen-key shapes both read; (2) "
                  "every RESULT re-stamp carries standup=<manifest id> before verified_by=, so the standup id is read off the LAST line as on landing 1 "
                  "(extra.row_engine.result_standup_source `last line`) and SHARED_VERIFIED_STANDUP_RULE's absent-token branch is not taken; (3) manifest.route.object_artifact_sha256, "
                  "the legacy scalar, spells the row's one fitted digest on EVERY store, the unsteered and the prompted ones included -- the reader never reads it as identity "
                  "(provenance.artifact_sha256 / fit_provenance.object_sha and the by-harness-key map from served_geometry.json:endpoint decide, and both are null / absent for "
                  "the unsteered and prompted keys), so those rows file as none / prompted_only and the scalar is RECORDED as extra.row_engine.object_artifact_sha256_scalar, a "
                  "visible producer quirk, never a refusal; row_engine.row_dir is volume-absolute as on landing 2 (SHARED_ROW_DIR_RULE binds by the tail)")
#: where a shared store's served-digest map was read from (extra.row_engine.served_map_source)
SERVED_MAP_SOURCE_ENDPOINT = "endpoint"
SERVED_MAP_SOURCE_BOTH = "endpoint (= manifest.route)"
_ROW_RECORD_TAIL = re.compile(r"(?:^|/)(rows/[^/]+/[^/]+)$")


@dataclasses.dataclass(frozen=True)
class SharedSpec:
    """What a SHARED-standup comparison reader binds beyond its solo twin's :class:`TaskSpec` (:data:`SHARED_SOURCE_RULE`): the solo source it compares
    against and the solo mirror root the comparison reads (`solo_root_name` under ``LC.RS_DEFAULT``), the registry spelling suffix, the store suffix's
    marker and the repeat marker. ``TaskSpec.shared`` is None on every solo reader, and every branch this class opens is gated on it."""
    solo_source: str
    solo_root_name: str
    spelling_suffix: str = SHARED_SPELLING_SUFFIX
    store_marker: str = SHARED_STORE_MARKER
    repeat_marker: str = SHARED_REPEAT_MARKER
    #: landing 2 (:data:`SHARED_L2_RUN_ID_RULE`): the harness variant folded into the producer key's spelling segment (`<tag>_shared_l2/<store key>`); None on
    #: landing 1's readers, whose producer key stays `<tag>_shared/<store key>` byte for byte
    harness_variant: Optional[str] = None
    #: landing 2: the BASE source whose name the landing's BOUND launch set may spell under evals.<eval>.own_source -- recorded as provenance, never bound (the
    #: reader binds by root); None on landing 1's readers
    base_source: Optional[str] = None
    #: landing 3 (:data:`SHARED_L3_RULE`): a landing's own account of how its stores differ from the landing the rules above were written for, stamped on its rows
    #: (extra.shared_standup.landing_rule) and its status (landing_rule); None on landing 1's and landing 2's readers, whose records and status stay byte for byte
    landing_rule: Optional[str] = None

    @property
    def solo_root_default(self) -> str:
        return os.path.join(LC.RS_DEFAULT, self.solo_root_name)

    def producer_spelling_of(self, tag: str) -> str:
        """The producer key's spelling segment: the registry spelling (`<tag>_shared`) on landing 1; `<tag>_shared_l2` on a landing-2 reader
        (:data:`SHARED_L2_RUN_ID_RULE` -- the harness variant is producer identity, so it enters the key here and nowhere else)."""
        return self.spelling_of(tag) if self.harness_variant is None else f"{self.spelling_of(tag)}_{self.harness_variant}"

    @property
    def suffix_re(self) -> "re.Pattern[str]":
        """``bcbw-<launch tag>-<sha7>-shared[-r<N>][-rep<N>][-bcb|-bfcl]``: the solo grammar's groups (launch_tag, sha7, relaunch, task_tag) plus ``marker`` and ``repeat``."""
        return re.compile(rf"^bcbw-(?P<launch_tag>(?:{_MODEL_ALT})(?:_s\d+p?(?:_arditi)?|{LEGACY_MAIN_SUFFIX}|{INSTRUCT_SUFFIX}))-(?P<sha7>[0-9a-f]{{7}})-(?P<marker>{re.escape(self.store_marker)})"
                          rf"(?P<relaunch>-r\d+)?(?P<repeat>-{re.escape(self.repeat_marker)}\d+)?(?P<task_tag>-(?:{'|'.join(sorted(TASK_TAGS))}))?$")

    @property
    def grammar_text(self) -> str:
        return f"bcbw-<launch tag>-<sha7>-{self.store_marker}[-r<N>][-{self.repeat_marker}<N>][-bcb|-bfcl]"

    def spelling_of(self, tag: str) -> str:
        """The registry spelling a shared row of ``tag`` files under: ``gemma4e4b_s1`` -> ``gemma4e4b_s1_shared``."""
        return f"{tag}{self.spelling_suffix}"

    def relaunch_entry_tag(self, relaunch_tag: str) -> str:
        """The ``tag`` relaunches.json spells for a shared relaunch: ``shared-r2`` for the suffix segment ``-r2``."""
        return f"{self.store_marker}-{relaunch_tag}"


def row_okey(okey: str, repeat: Optional[int], window: Optional[int] = None) -> str:
    """The row key's object-key half: the object key, ``<object key>@rep<N>`` for a repeat store (:data:`SHARED_ROW_KEY_RULE`), ``<object key>@w<N>`` for a store re-served at
    another serving window (:data:`UNSTEERED_ROW_RULE`)."""
    key = f"{okey}@{SHARED_REPEAT_MARKER}{repeat}" if repeat else okey
    return f"{key}@{WINDOW_MARKER}{window}" if window else key


def split_row_okey(rk_okey: str) -> Tuple[str, Optional[int]]:
    """``none@rep2`` -> (``none``, 2); ``none@w14336`` -> (``none``, None) (the window segment is :func:`window_of_row_okey`'s); a plain object key -> (key, None)."""
    rk_okey = re.sub(rf"@{WINDOW_MARKER}\d+$", "", rk_okey)
    m = re.match(rf"^(?P<okey>.+)@{SHARED_REPEAT_MARKER}(?P<n>\d+)$", rk_okey)
    return (m.group("okey"), int(m.group("n"))) if m else (rk_okey, None)


def pin_ids_sha256(ids: Sequence[str]) -> str:
    """``sisteer.capeval.tinycap.contracts.ids_sha256`` mirrored (the registry stays free of that import, as seedrep_controls does for OR-Bench): the sha256 of
    the canonical JSON list of the ids in the given order -- the digest ``contracts.item_set_identity`` stamps on ``manifest.json:tinycap.<key>.ids_sha256``.
    NOT cellstore's newline rule (``mmlu_1531``); the two item-set digests differ by construction and a reader names which it compares."""
    import hashlib
    return hashlib.sha256(json.dumps([str(i) for i in ids], separators=(",", ":")).encode()).hexdigest()


def item_set_entries(doc: Any) -> List[Dict[str, Any]]:
    """``records/item_sets.json``'s entries (``{id, sha256, n, file[, tree_file]}`` each), whether the file is a list of them, a mapping carrying an ``item_sets`` list
    or mapping, or a mapping keyed by item-set id (the contract says list / dict; both read, nothing else does)."""
    if isinstance(doc, Mapping) and "item_sets" in doc:
        doc = doc["item_sets"]
    if isinstance(doc, list):
        return [dict(e) for e in doc if isinstance(e, Mapping)]
    if isinstance(doc, Mapping):
        return [dict(v, id=v.get("id", k)) for k, v in doc.items() if isinstance(v, Mapping) and "sha256" in v]
    return []


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    """What one reader binds: its source, its evaluation, its task spelling and files, its instrument and cap, its protocol temperature."""
    name: str                       # the adapter name (= the source)
    source: str
    eval_id: str
    task: str                       # the store's row task (`inspect/bigcodebench_200`)
    record_task: str                # the gate record's spelling (`bigcodebench_200`)
    outcomes_file: str              # `inspect__bigcodebench_200.json` (outcomes/, summary/ and done/)
    instrument_seed: str
    cap: int
    protocol_temperature: float
    three_numbers_file: str
    item_set_key: str               # `manifest.json:tinycap.<key>` — the pinned item set (n of N, ids sha)
    generation_state: str           # how the items were scored at generation (no judge)
    clamp_override: bool = False    # BFCL: the served stack's `_MAX_TEMP` override must be attested on the store
    tool_calls_required: bool = True   # BFCL: the tool-call route must be attested (its task IS tool calls); BigCodeBench records it (llama8b served without it)
    root_name: str = ""
    models: Tuple[str, ...] = SEEDREP_MODEL_KEYS   # the model keys this source reads (MODEL_KEYS); a store of another family refuses its group by name
    #: the pinned item count a 200-item reader hard-wires (the manifest's tinycap.<key>.n must equal it); None on a FULL reader, whose count is the mirror's
    #: (records/item_sets.json + the pin file, ITEM_SET_RULE) -- None so an accidental arithmetic use fails loudly instead of counting 200
    n_items: Optional[int] = N_ITEMS
    #: a FULL reader: the registry item set it files on (item_sets.yaml; the instrument seed must bind the same set -- a new item
    #: set is a NEW instrument), the label the harness stamps (`full`) and the mirrored pin file under records/; all None on a 200-item reader
    item_set_id: Optional[str] = None
    item_set_label: Optional[str] = None
    item_set_pin_file: Optional[str] = None
    #: the key under launch_sets.json:evals the row table sits under; the eval id when unset (every mirror so far keys the table by the registry eval)
    launch_set_eval: Optional[str] = None
    #: the launched roster file under records/ (the plain name; the per-tree copies are `<stem>.<sha7>.yaml`): the 200 readers' ROSTER_FILE, the full
    #: readers' `bcb_bfcl_full_roster.yaml` (the sibling roster document for the full item sets)
    roster_file: str = ROSTER_FILE
    #: the launch set's file name(s) under records/: exactly ONE of them must be mirrored (the addendum names `launch_sets_full.json` for the full run; the first
    #: mirrors carry `launch_sets.json`); two mirrored at once refuse every store by name
    launch_set_files: Tuple[str, ...] = (LAUNCH_SET_FILE,)
    #: the reader admits the INSTRUCT-DIRECT tag `<model key>_it` (INSTRUCT_DIRECT_RULE; the full readers, 2026-09-25); False on the 200-item readers, whose
    #: mirrors never carried it and whose emission is pinned byte for byte
    instruct_direct: bool = False
    #: the reader admits the UNSTEERED row `<model key>_u0` and its window re-serves (UNSTEERED_ROW_RULE / WINDOW_ANCHOR_RULE; the full readers, 2026-10-05); False on the
    #: 200-item readers, whose mirrors never carried it and whose emission is pinned byte for byte
    unsteered_row: bool = False
    #: the reader admits the DOSE-0.75 RANDOM-INIT ReLU row `(model key)_d0p75_relul` (DOSE_RELU_ROW_RULE; the full readers, 2026-10-05); False on the 200-item readers,
    #: whose mirrors never carried it and whose emission is pinned byte for byte
    dose_relu_row: bool = False
    #: the reader admits the DOSE-1.25 DIFFERENCE-LoRA row `(model key)_d1p25` (DOSE_LORA_ROW_RULE; the full readers, 2026-10-05); False on the 200-item readers,
    #: whose mirrors never carried it and whose emission is pinned byte for byte
    dose_lora_row: bool = False
    #: the reader admits the LeCun-INIT ReLU row `(model key)_s2rl` (LECUN_RELU_ROW_RULE; the full readers, 2026-10-07); False on the 200-item readers, whose mirrors never
    #: carried it and whose emission is pinned byte for byte -- and where False, the row dir `(model key)-f22s2rl` is UNDECLARED for the reader (a whole-group refusal
    #: by name, UNDECLARED_ROW_DIR_RULE; lecun_relu_row_admitted), never a store refused by row beside rows that file
    lecun_relu_row: bool = False
    #: the planned envelope's file name under records/ (registered, never read): the canonical `bcbw_schedule.json`; the shared sources' `bcbw_schedule_shared.json`
    schedule_file: str = SCHEDULE_FILE
    #: a SHARED-standup comparison reader (:class:`SharedSpec`, :data:`SHARED_SOURCE_RULE`); None on every solo reader, which then reads byte for byte what it read before
    shared: Optional[SharedSpec] = None

    @property
    def root_default(self) -> str:
        return os.path.join(LC.RS_DEFAULT, self.root_name or self.source)

    @property
    def suffix_re(self) -> "re.Pattern[str]":
        """The store suffix grammar this reader reads: the module's :data:`SUFFIX_RE` (the very object) on a solo reader, the shared grammar on a shared one."""
        return SUFFIX_RE if self.shared is None else self.shared.suffix_re

    @property
    def grammar_text(self) -> str:
        return "bcbw-<launch tag>-<sha7>[-r<N>][-w<N>][-bcb|-bfcl]" if self.shared is None else self.shared.grammar_text

    def spelling_of(self, tag: str) -> str:
        """The registry spelling (the run's view_legacy) a row of ``tag`` files under: the tag itself on a solo reader, `<tag>_shared` on a shared one."""
        return tag if self.shared is None else self.shared.spelling_of(tag)

    @property
    def full_item_set(self) -> bool:
        return self.item_set_id is not None

    @property
    def launch_set_eval_key(self) -> str:
        return self.launch_set_eval or self.eval_id

    @property
    def roster_variant_re(self) -> "re.Pattern[str]":
        return roster_variant_re(self.roster_file)

    @property
    def recount_algorithm(self) -> str:
        return f"sisteer.registry.adapters.{self.name}.recount"

    @property
    def recount_seed_provenance(self) -> str:
        if self.full_item_set:
            return (f"none: a deterministic recount of the store's outcome record (correct / scored over the attempted items of the pinned item set {self.item_set_id}; "
                    f"one pass at {self.cap:,}, no cap step)")
        return f"none: a deterministic recount of the store's outcome record (correct / scored over the {self.n_items} attempted items; one pass at {self.cap:,}, no cap step)"

    def recount_computation_id(self) -> str:
        return I.computation_id(self.recount_algorithm, RECOUNT_VERSION, self.recount_seed_provenance)


def tag_parts(tag: str) -> Optional[Tuple[int, bool]]:
    """``(seed, paired?)`` of a ``<model key>_s<N>[p]`` tag; None for another spelling (a ``main`` canary tag and the instruct-direct ``_it`` tag included)."""
    m = TAG_RE.match(tag)
    return (int(m.group("seed")), bool(m.group("paired"))) if m and m.group("seed") is not None else None


def instruct_model_key(tag: str) -> Optional[str]:
    """``gemma4e4b_it`` -> ``gemma4e4b``: the GRAMMAR of the instruct-direct tag (:data:`INSTRUCT_DIRECT_RULE`); None for any other spelling. Whether the
    family RUNS the tag (:data:`MODEL_KEYS` ``instruct_direct``) and whether the reader admits it (``TaskSpec.instruct_direct``) are :func:`group_tags`'s
    to decide, so a ``_it`` store of nemotron, or under a 200-item reader, is discovered and refused BY NAME rather than skipped."""
    m = TAG_RE.match(tag)
    return m.group("model") if m and m.group("instruct") else None


def is_instruct_tag(tag: str) -> bool:
    return instruct_model_key(tag) is not None


def instruct_tag_of(model_key: str) -> str:
    return f"{model_key}{INSTRUCT_SUFFIX}"


def unsteered_model_key(tag: str) -> Optional[str]:
    """``qwen35_35b_a3b_u0`` -> ``qwen35_35b_a3b``: the GRAMMAR of the unsteered tag (:data:`UNSTEERED_ROW_RULE`); None for any other spelling. Whether the family RUNS
    the row (:data:`MODEL_KEYS` ``unsteered_row``) and whether the reader admits it (``TaskSpec.unsteered_row``) are :func:`group_tags`'s to decide, so a `_u0` store of a
    family without the row, or under a 200-item reader, is discovered and refused BY NAME rather than skipped."""
    m = TAG_RE.match(tag)
    return m.group("model") if m and m.group("unsteered") else None


def is_unsteered_tag(tag: str) -> bool:
    return unsteered_model_key(tag) is not None


def unsteered_tag_of(model_key: str) -> str:
    return f"{model_key}{UNSTEERED_SUFFIX}"


def unsteered_anchor_tag(tag: str) -> Optional[str]:
    """``qwen35_35b_a3b_u0`` -> ``qwen35_35b_a3b_s1``: the GROUP an unsteered tag reads in (the family's one seed group, whose steered rows anchor on the u0 none)."""
    mk = unsteered_model_key(tag)
    return f"{mk}_s{UNSTEERED_ANCHOR_SEED}" if mk else None


def dose_relu_model_key(tag: str) -> Optional[str]:
    """``qwen35_35b_a3b_d0p75_relul`` -> ``qwen35_35b_a3b``: the GRAMMAR of the dose-0.75 random-init ReLU tag (:data:`DOSE_RELU_ROW_RULE`); None for any other
    spelling. Whether the family RUNS the row (:data:`MODEL_KEYS` ``dose_relu_row``) and whether the reader admits it (``TaskSpec.dose_relu_row``) are
    :func:`group_tags`'s to decide, so such a store under a 200-item reader is discovered and refused BY NAME rather than skipped."""
    m = TAG_RE.match(tag)
    return m.group("model") if m and m.group("dose_relu") else None


def is_dose_relu_tag(tag: str) -> bool:
    return dose_relu_model_key(tag) is not None


def dose_relu_tag_of(model_key: str) -> str:
    return f"{model_key}{DOSE_RELU_SUFFIX}"


def dose_relu_anchor_tag(tag: str) -> Optional[str]:
    """``qwen35_35b_a3b_d0p75_relul`` -> ``qwen35_35b_a3b_s1``: the GROUP the dose row reads in (the family's one seed group, whose u0 none its rows anchor on)."""
    mk = dose_relu_model_key(tag)
    return f"{mk}_s{DOSE_RELU_ANCHOR_SEED}" if mk else None


def dose_lora_model_key(tag: str) -> Optional[str]:
    """``qwen35_35b_a3b_d1p25`` -> ``qwen35_35b_a3b``: the GRAMMAR of the dose-1.25 difference-LoRA tag (:data:`DOSE_LORA_ROW_RULE`); None for any other spelling.
    Whether the family RUNS the row (:data:`MODEL_KEYS` ``dose_lora_row``) and whether the reader admits it (``TaskSpec.dose_lora_row``) are :func:`group_tags`'s to
    decide, so such a store under a 200-item reader is discovered and refused BY NAME rather than skipped."""
    m = TAG_RE.match(tag)
    return m.group("model") if m and m.group("dose_lora") else None


def is_dose_lora_tag(tag: str) -> bool:
    return dose_lora_model_key(tag) is not None


def dose_lora_tag_of(model_key: str) -> str:
    return f"{model_key}{DOSE_LORA_SUFFIX}"


def dose_lora_anchor_tag(tag: str) -> Optional[str]:
    """``qwen35_35b_a3b_d1p25`` -> ``qwen35_35b_a3b_s1``: the GROUP the dose row reads in (the family's one seed group, whose u0 none its row anchors on)."""
    mk = dose_lora_model_key(tag)
    return f"{mk}_s{DOSE_LORA_ANCHOR_SEED}" if mk else None


def is_dose_row_tag(tag: str) -> bool:
    """A DOSE row's tag of either kind (the ReLU dose copies' ``_d0p75_relul``, the difference LoRA's ``_d1p25``): never the paired one, no fit seed, the u0 none its anchor."""
    return is_dose_relu_tag(tag) or is_dose_lora_tag(tag)


def lecun_relu_model_key(tag: str) -> Optional[str]:
    """``qwen35_35b_a3b_s2rl`` -> ``qwen35_35b_a3b``: the GRAMMAR of the LeCun-init ReLU tag (:data:`LECUN_RELU_ROW_RULE`); None for any other spelling (``_s2rl`` is not a
    seed tag: :func:`tag_parts` reads None). Whether the family RUNS the row (:data:`MODEL_KEYS` ``lecun_relu_row``) and whether the reader admits it
    (``TaskSpec.lecun_relu_row``) are :func:`lecun_relu_row_admitted`'s to decide: a reader that does not read the row discovers its row dir as UNDECLARED
    (:func:`tag_of_row_dir` with the reader's spec) and refuses it as a whole group by name, never skipping it and never filing it as a by-name row refusal."""
    m = TAG_RE.match(tag)
    return m.group("model") if m and m.group("lecun_relu") else None


def is_lecun_relu_tag(tag: str) -> bool:
    return lecun_relu_model_key(tag) is not None


def lecun_relu_tag_of(model_key: str) -> str:
    return f"{model_key}{LECUN_RELU_SUFFIX}"


def lecun_relu_anchor_tag(tag: str) -> Optional[str]:
    """``qwen35_35b_a3b_s2rl`` -> ``qwen35_35b_a3b_s1``: the GROUP the LeCun-init ReLU row reads in (the family's one seed group, whose u0 none its rows anchor on)."""
    mk = lecun_relu_model_key(tag)
    return f"{mk}_s{LECUN_RELU_ANCHOR_SEED}" if mk else None


def lecun_relu_row_admitted(model_key: Optional[str], spec: Optional["TaskSpec"]) -> bool:
    """Whether a reader with ``spec`` READS the LeCun-init ReLU row of ``model_key`` (:data:`LECUN_RELU_ROW_RULE`): the reader admits the row
    (``TaskSpec.lecun_relu_row``, the full readers) AND the family runs it (``ModelKey.lecun_relu_row``, the 35B). With ``spec`` None the answer is the
    grammar's alone (True: the module-level spellings), which is what :func:`tag_of_row_dir` / :func:`undeclared_row_dir_reason` read without a reader.
    Where this is False the row dir ``(model key)-f22s2rl`` is UNDECLARED for that reader (:data:`UNDECLARED_ROW_DIR_RULE`): a group of its own, refused as a
    whole by name, so a misplaced or early-launched s2rl store reads ok = 0 on the status instead of a by-name row refusal beside rows that file (the review of
    797f57df2, 2026-10-07: the by-name refusal left ``source_status.ok`` at 1 where the base tree's undeclared-directory refusal filed 0)."""
    if spec is None:
        return True
    mkey = MODEL_KEYS.get(model_key or "")
    return bool(spec.lecun_relu_row and mkey is not None and mkey.lecun_relu_row)


def window_of_suffix(spec: "TaskSpec", suffix: str) -> Optional[int]:
    """The serving window a store suffix's ``-w<N>`` segment spells (:data:`UNSTEERED_ROW_RULE`); None without the segment (the shared grammar has none)."""
    m = spec.suffix_re.match(suffix)
    seg = m.groupdict().get("window") if m else None
    return int(seg[1 + len(WINDOW_MARKER):]) if seg else None


def window_of_row_okey(rk_okey: str) -> Optional[int]:
    """``none@w14336`` -> 14336; None for a row key without the window segment."""
    m = re.search(rf"@{WINDOW_MARKER}(?P<n>\d+)$", rk_okey)
    return int(m.group("n")) if m else None


def instruct_anchor_tag(tag: str) -> Optional[str]:
    """``gemma4e4b_it`` -> ``gemma4e4b_s1``: the GROUP an instruct-direct tag reads in, whose ``none`` is its anchor by alias (:data:`INSTRUCT_ANCHOR_RULE`)."""
    mk = instruct_model_key(tag)
    return f"{mk}_s{INSTRUCT_ANCHOR_SEED}" if mk else None


def group_tags(group: str, instruct_direct: bool = False, unsteered_row: bool = False, dose_relu_row: bool = False, dose_lora_row: bool = False,
               lecun_relu_row: bool = False) -> Dict[str, str]:
    """The ``{selection: tag}`` map of a group (a seed tag): its own selection, the paired one, and -- on a reader that admits it (``instruct_direct``,
    the full readers' TaskSpec) for the ``s1`` group of a family that runs the tag -- the instruct-direct selection ``it`` -> ``<model key>_it``; likewise
    the unsteered selection ``u0``, the dose-0.75 random-init ReLU selection ``d0p75_relul`` (:data:`DOSE_RELU_ROW_RULE`) and the dose-1.25 difference-LoRA selection
    ``d1p25`` (:data:`DOSE_LORA_ROW_RULE`) and the LeCun-init ReLU selection ``s2rl`` (:data:`LECUN_RELU_ROW_RULE`) where the reader admits them."""
    seed_n = tag_parts(group)[0]
    mk = model_key_of(group)
    out = {selection_key(seed_n): group, selection_key(seed_n, True): f"{mk}_s{seed_n}p"}
    if instruct_direct and seed_n == INSTRUCT_ANCHOR_SEED and MODEL_KEYS[mk].instruct_direct:
        out[INSTRUCT_SELECTION] = instruct_tag_of(mk)
    if unsteered_row and seed_n == UNSTEERED_ANCHOR_SEED and MODEL_KEYS[mk].unsteered_row:
        out[UNSTEERED_SELECTION] = unsteered_tag_of(mk)      # the unsteered row reads in the family's s1 group (UNSTEERED_ROW_RULE)
    if dose_relu_row and seed_n == DOSE_RELU_ANCHOR_SEED and MODEL_KEYS[mk].dose_relu_row:
        out[DOSE_RELU_SELECTION] = dose_relu_tag_of(mk)      # the dose-0.75 random-init ReLU row reads in the family's s1 group too (DOSE_RELU_ROW_RULE)
    if dose_lora_row and seed_n == DOSE_LORA_ANCHOR_SEED and MODEL_KEYS[mk].dose_lora_row:
        out[DOSE_LORA_SELECTION] = dose_lora_tag_of(mk)      # the dose-1.25 difference-LoRA row reads in the family's s1 group too (DOSE_LORA_ROW_RULE)
    if lecun_relu_row and seed_n == LECUN_RELU_ANCHOR_SEED and MODEL_KEYS[mk].lecun_relu_row:
        out[LECUN_RELU_SELECTION] = lecun_relu_tag_of(mk)    # the LeCun-init ReLU row reads in the family's s1 group too (LECUN_RELU_ROW_RULE)
    return out


def model_key_of(tag: str) -> Optional[str]:
    """The tinycap model key a tag spells (``qwen35_4b_s2p`` -> ``qwen35_4b``; an aliased family's ``<model key>_main`` too); None otherwise."""
    m = TAG_RE.match(tag)
    return m.group("model") if m else legacy_main_model_key(tag)


def legacy_main_model_key(tag: str) -> Optional[str]:
    """``gemma4e4b_main`` -> ``gemma4e4b`` when the family aliases its ``main`` canary to ``s1`` (:data:`LEGACY_MAIN_ALIAS_RULE`); None otherwise."""
    m = LEGACY_MAIN_TAG_RE.match(tag)
    return m.group("model") if m and MODEL_KEYS[m.group("model")].legacy_main_alias else None


def legacy_main_alias_tag(tag: str) -> Optional[str]:
    """``gemma4e4b_main`` -> ``gemma4e4b_s1``, the tag the canary's ``none`` files under; None for any other spelling."""
    mk = legacy_main_model_key(tag)
    return f"{mk}_s{LEGACY_MAIN_ALIAS_SEED}" if mk else None


def seed_tag_of(tag: str) -> Optional[str]:
    """The GROUP of a tag: the seed tag itself, the seed tag a paired tag stands beside (:data:`GROUP_RULE`), the ``s1`` tag an aliased family's
    ``main`` canary tag files under, or the ``s1`` tag an instruct-direct tag anchors on (:data:`INSTRUCT_ANCHOR_RULE`)."""
    p = tag_parts(tag)
    return f"{model_key_of(tag)}_s{p[0]}" if p else (instruct_anchor_tag(tag) or legacy_main_alias_tag(tag) or unsteered_anchor_tag(tag) or dose_relu_anchor_tag(tag) or dose_lora_anchor_tag(tag)
                                                             or lecun_relu_anchor_tag(tag))


def selection_key(seed_n: int, paired: bool = False) -> str:
    return f"s{seed_n}{'p' if paired else ''}"


def tag_of_row_dir(row_dir: str, spec: Optional["TaskSpec"] = None) -> Optional[str]:
    """``qwen35_4b-f22s1p`` -> ``qwen35_4b_s1p``; ``gemma4e4b-f22`` (an aliased family's ``main`` canary row) -> ``gemma4e4b_s1``; None for another
    spelling (``qwen35_4b-f22`` included: the three-seed family has no alias). With a reader's ``spec`` the LeCun-init ReLU row dir ``(model key)-f22s2rl``
    is a declared row only where that reader READS it (:func:`lecun_relu_row_admitted`: the reader admits the row and the family runs it); elsewhere it is
    None -- undeclared for that reader, a group of its own that refuses as a whole (:data:`UNDECLARED_ROW_DIR_RULE`). The other optional selections (``it``,
    ``u0``, the dose rows) stay grammar for every reader: a store under them that the group admits for no selection refuses BY ROW, the design main pinned."""
    m = ROW_DIR_RE.match(row_dir)
    if m and m.group("instruct"):
        return instruct_tag_of(m.group("model"))     # the instruct-direct row (`<model key>-f22it`): its own tag; read() decides whether the group admits it
    if m and m.group("unsteered"):
        return unsteered_tag_of(m.group("model"))    # the unsteered row (`<model key>-f22u0`): its own tag; read() decides whether the group admits it (UNSTEERED_ROW_RULE)
    if m and m.group("dose_relu"):
        return dose_relu_tag_of(m.group("model"))    # the dose-0.75 random-init ReLU row (`(model key)-f22d0p75_relul`): its own tag; read() decides (DOSE_RELU_ROW_RULE)
    if m and m.group("dose_lora"):
        return dose_lora_tag_of(m.group("model"))    # the dose-1.25 difference-LoRA row (`(model key)-f22d1p25`): its own tag; read() decides (DOSE_LORA_ROW_RULE)
    if m and m.group("lecun_relu"):
        # the LeCun-init ReLU row (`(model key)-f22s2rl`): its own tag where the reader reads it (LECUN_RELU_ROW_RULE); with no reader, the grammar's
        return lecun_relu_tag_of(m.group("model")) if lecun_relu_row_admitted(m.group("model"), spec) else None
    if m:
        return f"{m.group('model')}_s{m.group('seed')}{m.group('paired')}"
    legacy = legacy_main_tag_of_row_dir(row_dir)
    return legacy_main_alias_tag(legacy) if legacy else None


def legacy_main_tag_of_row_dir(row_dir: str) -> Optional[str]:
    """``gemma4e4b-f22`` -> ``gemma4e4b_main``, the canary's OWN tag, when the family aliases it; None otherwise."""
    m = LEGACY_MAIN_ROW_DIR_RE.match(row_dir)
    return f"{m.group('model')}{LEGACY_MAIN_SUFFIX}" if m and MODEL_KEYS[m.group("model")].legacy_main_alias else None


def undeclared_row_dir_reason(row_dir: str, spec: Optional["TaskSpec"] = None) -> Optional[str]:
    """Why a ``native/`` entry is NOT a declared row (:data:`UNDECLARED_ROW_DIR_RULE`), by name: ``row dir <name> matches no declared model key (...)`` when the
    model segment is outside :data:`MODEL_KEYS` (or the name has no ``-f22`` shape at all), ``row dir <name> names the declared model key <mk> but no declared
    row (...)`` when the key is declared and the selection is not one the grammar reads for it, ``row dir <name> names the declared model key <mk> under the
    LeCun-init ReLU selection 's2rl', which this reader does not read for it (...)`` when the grammar reads the row but the reader with ``spec`` does not
    (:func:`lecun_relu_row_admitted`); None for a declared row (:func:`tag_of_row_dir` with the same ``spec``)."""
    if tag_of_row_dir(row_dir, spec) is not None:
        return None
    m = ROW_DIR_SHAPE_RE.match(row_dir)
    declared = ", ".join(sorted(MODEL_KEYS))
    if m is None or m.group("model") not in MODEL_KEYS:
        return f"row dir {row_dir!r} matches no declared model key (the model-key table declares {declared}; a row dir is <model key>-f22s<N>[p])"
    mk = m.group("model")
    mkey = MODEL_KEYS[mk]
    if tag_of_row_dir(row_dir) is not None:
        # the grammar reads the row and THIS reader does not: the LeCun-init ReLU row dir under a reader that does not admit the row, a family that does not run it, or both
        assert spec is not None and m.group("selection") == LECUN_RELU_SELECTION, row_dir
        why = []
        if not spec.lecun_relu_row:
            why.append("this reader admits no LeCun-init ReLU row (TaskSpec.lecun_relu_row: the 200-item readers' emission is pinned byte for byte)")
        if not mkey.lecun_relu_row:
            why.append(f"the family {mk} runs no LeCun-init ReLU row (the model-key table's lecun_relu_row)")
        return (f"row dir {row_dir!r} names the declared model key {mk!r} under the LeCun-init ReLU selection {LECUN_RELU_SELECTION!r}, which this reader does not read for it "
                f"({'; '.join(why)}; the rows this reader reads for {mk}: {', '.join(declared_row_dirs_of(mk, spec))}): a store under it is misplaced or launched ahead of its "
                f"declaration, refused as a WHOLE GROUP by name -- never a by-name row refusal beside rows that file ({LECUN_RELU_ROW_RULE})")
    rows = declared_row_dirs_of(mk, spec)
    return (f"row dir {row_dir!r} names the declared model key {mk!r} but no declared row (selection {m.group('selection')!r}; the rows this reader reads for "
            f"{mk}: {', '.join(rows)}): a new selection is a reading contract of its own, not a reader default")


def declared_row_dirs_of(mk: str, spec: Optional["TaskSpec"] = None) -> List[str]:
    """The row dirs a reader reads for the declared model key ``mk`` (the reason text's listing): the seed and paired rows, then the optional rows the
    family runs -- the LeCun-init ReLU row only for a family that runs it AND, with a reader's ``spec``, only where that reader admits it (:func:`lecun_relu_row_admitted`)."""
    mkey = MODEL_KEYS[mk]
    rows = [f"{mk}-f22s{n}" for n in mkey.seeds] + [f"{mk}-f22s{n}p" for n in mkey.seeds]
    if mkey.instruct_direct:
        rows.append(f"{mk}-f22{INSTRUCT_SELECTION}")
    if mkey.legacy_main_alias:
        rows.append(f"{mk}-f22")
    if mkey.unsteered_row:
        rows.append(f"{mk}-f22{UNSTEERED_SELECTION}")
    if mkey.dose_relu_row:
        rows.append(f"{mk}-f22{DOSE_RELU_SELECTION}")
    if mkey.dose_lora_row:
        rows.append(f"{mk}-f22{DOSE_LORA_SELECTION}")
    if mkey.lecun_relu_row and lecun_relu_row_admitted(mk, spec):
        rows.append(f"{mk}-f22{LECUN_RELU_SELECTION}")     # the table's answer with no reader; a reader that does not admit the row lists it for no family
    return rows


def is_undeclared_row_dir_group(group: str, spec: Optional["TaskSpec"] = None) -> bool:
    """A comparison group named for an undeclared row dir (:func:`BcbwSeedrepReader.comparison_group`): a row dir carries ``-f22``, a tag never does.
    With a reader's ``spec`` the LeCun-init ReLU row dir counts as undeclared where that reader does not read it (:func:`tag_of_row_dir`)."""
    return ROW_DIR_SHAPE_RE.match(group) is not None and tag_of_row_dir(group, spec) is None


def row_dir_of(tag: str) -> str:
    p = tag_parts(tag)
    if p is None:
        mk = instruct_model_key(tag)
        if mk is None:
            umk = unsteered_model_key(tag)
            if umk is not None:
                return f"{umk}-f22{UNSTEERED_SELECTION}"
            dmk = dose_relu_model_key(tag)
            if dmk is not None:
                return f"{dmk}-f22{DOSE_RELU_SELECTION}"
            lmk = dose_lora_model_key(tag)
            if lmk is not None:
                return f"{lmk}-f22{DOSE_LORA_SELECTION}"
            rmk = lecun_relu_model_key(tag)
            if rmk is not None:
                return f"{rmk}-f22{LECUN_RELU_SELECTION}"
            raise KeyError(f"{tag!r} spells no seed tag, no instruct-direct tag, no unsteered tag, no dose-row tag and no LeCun-init ReLU tag; it has no tinycap model row")
        return f"{mk}-f22{INSTRUCT_SELECTION}"
    return f"{model_key_of(tag)}-f22{selection_key(p[0], p[1])}"


def launch_tag_of(tag: str, okey: str, legacy_tag: Optional[str] = None) -> str:
    """The launch tag a store's suffix and gate record spell: the tag (the canary's own ``<model key>_main`` for an aliased store), or
    ``<tag>_arditi`` for the all-positions object."""
    base = legacy_tag or tag
    return f"{base}{ARDITI_LAUNCH_SUFFIX}" if okey == ARDITI_OBJECT_KEY else base


def producer_key_for(tag: str, store_key: str) -> str:
    return f"{tag}/{store_key}"


def anchor_plan(group: str, tag: str, filed: Mapping[Tuple[str, str], Mapping[str, Any]], expected: Mapping[Tuple[str, str], Any], seen: Set[Tuple[str, str]]) -> Dict[str, Any]:
    """The anchor a non-``none`` row of ``tag`` pairs against, resolved from the rows that FILED (:data:`PAIRED_ANCHOR_RULE`; shared by the three
    capability readers): the row's OWN tag's ``none`` when it filed (a seed tag's rows, and the paired tag's once its ``none`` lands) — ``own``,
    ``cross_standup`` false; else, for the paired tag, the seed tag's ``none`` ACROSS tags (the cross-tag fallback; ``cross_standup`` true whether or not
    that ``none`` filed, the reading the row is waiting for) with the own ``none``'s state named when the launch set declares it
    (``own_none_state`` :data:`ANCHOR_NOT_YET_RUN` — named, no store — or :data:`ANCHOR_REFUSED` — a store that refused); else no anchor, with
    ``state`` / ``why`` about the seed tag's ``none`` in the readers' standing vocabulary. ``expected`` and ``seen`` are the ``(tag, object key)``
    sets of the launch set and of the mirrored stores."""
    own = filed.get((tag, "none"))
    if own is not None:
        return {"anchor": own, "anchor_tag": tag, "own": True, "cross_standup": False, "own_none_state": None, "own_none_note": None, "state": None, "why": None}
    paired = tag != group
    seed = filed.get((group, "none"))
    plan: Dict[str, Any] = {"anchor": seed, "anchor_tag": (group if seed is not None else None), "own": False, "cross_standup": paired, "own_none_state": None,
                            "own_none_note": None, "state": None, "why": None}
    if seed is None:
        seed_key = (group, "none")
        if seed_key in expected and seed_key not in seen:
            plan.update(state=ANCHOR_NOT_YET_RUN, why=f"the seed tag's `none` ({group}/none) has not run yet (the launch set names it, no store is mirrored); no contrast is formed until it lands")
        elif seed_key in seen:
            plan.update(state=ANCHOR_REFUSED, why=f"the seed tag's `none` ({group}/none) refused; no contrast is formed")
        else:
            plan.update(state=ANCHOR_NOT_YET_RUN, why=f"the launch set names no `none` for {group}; no contrast is formed")
    if paired:
        own_key = (tag, "none")
        if own_key in seen:
            plan.update(own_none_state=ANCHOR_REFUSED, own_none_note=(f"the paired tag's own `none` ({tag}/none) is mirrored but refused; the row keeps the D65 fallback "
                                                                        f"({group}/none, cross-standup) until it files"))
        elif own_key in expected:
            plan.update(own_none_state=ANCHOR_NOT_YET_RUN, own_none_note=(f"the launch set names the paired tag's own `none` ({tag}/none) but no store is mirrored; the row keeps "
                                                                            f"the D65 fallback ({group}/none, cross-standup) until it lands"))
    return plan


def cross_standup_note(group: str, row_tag: str, plan: Mapping[str, Any]) -> str:
    """The note a cross-standup contrast carries: the standing cross-tag text while the paired tag has no ``none`` in the launch set at all; once the launch
    set names one, the text says the fallback is provisional."""
    if plan.get("own_none_state") is None:
        return f"the composed row's tag {row_tag} has no `none` of its own: D65 pairs it with the seed tag's `none` ({group}/none), served by another standup"
    return (f"the composed row's tag {row_tag} has its own `none` in the launch set but it did not file ({plan['own_none_state']}): D65 pairs the row with the seed tag's "
            f"`none` ({group}/none), served by another standup, until its own `none` files")


#: the contrast's `anchor_kind` (vocab.ANCHOR_KINDS, CHECKed on run_contrast): the anchoring `none` was served by the row's OWN standup -> same_standup_none;
#: by ANOTHER standup of the same model -> sibling_none (the rule: a `none` across seeds / standups is a fair anchor when the decode settings are
#: the same; seedrep_cluster's DECODE_TRIAL_ANCHOR_KIND, overrefusal_items' harness-key test). `borrowed` (r2_export: an anchor from another spelling's
#: rows; the anchor currency's tier-1c re-anchoring) and `full_dose` arise here only through the currency; `solo_stand_in` is the SHARED readers'
#: stand-in comparison against the solo run (:data:`ANCHOR_KIND_SOLO_STAND_IN`). Until 2026-09-26 every contrast of the six capability readers filed same_standup_none and told the two apart
#: on extra.cross_standup alone: the 15 live instruct-direct contrasts of the full readers (bigcodebench_full 9 / bfcl_full 6) carried the mislabel and
#: move on that field (emission 3); the four seedrep readers' cross-tag-fallback rows (3 each) were filed before the paired `none` landed and sit in the
#: registry from an earlier revision beside the own-none contrast that replaced them -- the live mirrors re-read byte for byte, no bump
ANCHOR_KIND_OWN = "same_standup_none"
ANCHOR_KIND_SIBLING = "sibling_none"


def anchor_kind_for(plan: Mapping[str, Any]) -> str:
    """The ``anchor_kind`` of a contrast formed under ``plan`` (:func:`anchor_plan` / :func:`instruct_anchor_plan`): ``same_standup_none`` when the row
    pairs against its OWN tag's ``none``; ``sibling_none`` when the plan says ``cross_standup`` -- the fallback pairing (a composed ``s<N>p`` row against
    ``s<N>/none``) and the instruct-direct alias (a ``<model key>_it`` row against ``<model key>_s1/none``): the same model's ``none``, served by another
    standup under the same decode settings -- or ``sibling_current``: the anchor currency re-picked a same-source unsteered run of another standup (the
    u0 ``none`` for the s1 keys, :data:`UNSTEERED_ROW_RULE`); ``same_standup_none`` again under ``same_standup_current``: the currency
    re-picked the row's OWN standup's unsteered run, one the registry holds current from an earlier read while this batch's own ``none`` is absent or
    refused (the pair stays within the standup, so ``cross_standup`` is false there too); ``solo_stand_in`` when the plan says
    ``borrowed`` -- a SHARED row's comparison against its solo run, another source's row standing in as the anchor (:func:`solo_anchor_plan`,
    :data:`SHARED_COMPARISON_RULE`; its own kind since migration 0027, so a shared key's stand-in and its re-anchored steering contrast tell apart
    by anchor_kind) -- or ``borrowed`` under ``borrowed_reference``: the currency re-picked the paper's roster-matched baseline of ANOTHER source.
    The three currency keys are read here and nowhere else: the stamp sites stay the plan's, and :meth:`_emit_contrasts`
    checks the kind this rule yields against the currency's own on every filed contrast."""
    if plan.get("borrowed"):
        return ANCHOR_KIND_SOLO_STAND_IN
    if plan.get("borrowed_reference"):
        return ANCHOR_KIND_BORROWED
    if plan.get("same_standup_current"):
        return ANCHOR_KIND_OWN
    if plan.get("sibling_current"):
        return ANCHOR_KIND_SIBLING
    return ANCHOR_KIND_SIBLING if plan["cross_standup"] else ANCHOR_KIND_OWN


def solo_anchor_plan(solo_run_id: str) -> Dict[str, Any]:
    """The plan of a SHARED row's stand-in comparison (:data:`SHARED_COMPARISON_RULE`): the solo run stands in as the anchor -- another source's row on the
    same instrument, served by another standup -- so ``cross_standup`` is true and the plan's ``borrowed`` key names the kind `solo_stand_in`
    (:func:`anchor_kind_for`)."""
    return {"anchor_run_id": solo_run_id, "own": False, "cross_standup": True, "borrowed": True, "own_none_state": None, "own_none_note": None, "state": None, "why": None}


def own_standup_note(group: str, row_tag: str) -> str:
    """The note a paired row's contrast carries once it pairs against its OWN tag's ``none``."""
    return f"the row's tag {row_tag} carries its OWN `none` (a standup replicate of {group}/none, D429): the contrast pairs against it, never the seed tag's"


def run_id_for(spec: TaskSpec, tag: str, store_key: str, okey: str, instrument_id: str, repeat: Optional[int] = None) -> str:
    """The run id of a (tag, store key, object key) row: producer key ``<tag>/<store key>`` (:data:`PRODUCER_KEY_RULE`); on a SHARED reader the `_shared`
    spelling prefixes it and a repeat store folds its segment in (:data:`SHARED_RUN_ID_RULE`) -- never the solo reader's id over the same store key."""
    return I.run_id(CAMPAIGN, shared_producer_key_for(spec, tag, store_key, repeat), okey, spec.eval_id, instrument_id)


def shared_producer_key_for(spec: TaskSpec, tag: str, store_key: str, repeat: Optional[int] = None) -> str:
    """``<tag>/<store key>`` on a solo reader; ``<tag>_shared/<store key>[@rep<N>]`` on a shared one (:data:`SHARED_RUN_ID_RULE`); ``<tag>_shared_l2/...`` on a
    landing-2 reader (:data:`SHARED_L2_RUN_ID_RULE`: the harness variant enters the spelling segment, :meth:`SharedSpec.producer_spelling_of`)."""
    if spec.shared is None:
        return producer_key_for(tag, store_key)
    return producer_key_for(spec.shared.producer_spelling_of(tag), row_okey(store_key, repeat))


def parse_result_line(line: str) -> Optional[Dict[str, str]]:
    """``tinycap-modal name=<n> rc=<rc> result=<STATE> status=<S> sha=<tree> at=<ts>`` -> its fields; None for another spelling."""
    toks = line.strip().split()
    if not toks or toks[0] != RESULT_HEAD:
        return None
    out: Dict[str, str] = {"head": toks[0]}
    for tok in toks[1:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            out.setdefault(k, v)
    return out if "result" in out else None


def suffix_task(suffix: str) -> Optional[str]:
    """The eval a store suffix's task tag names (``-bcb`` -> ``bigcodebench``); None for a two-task store (no tag) or a suffix outside the grammar."""
    m = SUFFIX_RE.match(suffix)
    tag = (m.group("task_tag") or "")[1:] if m else ""
    return TASK_TAGS.get(tag) if tag else None


def suffix_task_for(spec: TaskSpec, suffix: str) -> Optional[str]:
    """:func:`suffix_task` under the reader's own grammar (the solo grammar on a solo reader: the same answer; the shared grammar on a shared one)."""
    if spec.shared is None:
        return suffix_task(suffix)
    m = spec.suffix_re.match(suffix)
    tag = (m.group("task_tag") or "")[1:] if m else ""
    return TASK_TAGS.get(tag) if tag else None


def repeat_of_suffix(spec: TaskSpec, suffix: str) -> Optional[int]:
    """The repeat index a SHARED store suffix spells (``-shared-rep2`` -> 2); None on a solo reader or a suffix without the segment."""
    if spec.shared is None:
        return None
    m = spec.suffix_re.match(suffix)
    seg = (m.group("repeat") or "") if m else ""
    return int(seg[len(spec.shared.repeat_marker) + 1:]) if seg else None


def row_launch_suffix(spec: TaskSpec, suffix: str) -> str:
    """The suffix of the ROW LAUNCH a store belongs to (:data:`SHARED_ROW_DIR_RULE`): the store suffix without its ``-rep<N>`` segment -- a repeat rides its row's
    container (``...-shared-rep2`` -> ``...-shared``); a ``-r<N>`` relaunch keeps its segment (its own container). A solo reader's suffix is its own."""
    if spec.shared is None:
        return suffix
    m = spec.suffix_re.match(suffix)
    if m is None or not m.group("repeat"):
        return suffix
    return suffix[:m.start("repeat")] + suffix[m.end("repeat"):]


def row_record_tail(row_dir: str) -> Optional[str]:
    """``rows/<row>/<suffix>`` from a ``row_engine.row_dir`` spelling (:data:`SHARED_ROW_DIR_RULE`): the relative form itself, or the tail of the volume-absolute
    ``/vol/tinycap/gen/rows/<row>/<suffix>``; None for any other shape."""
    m = _ROW_RECORD_TAIL.search(str(row_dir or ""))
    return m.group(1) if m else None


def gate_name_parts(name: str) -> Optional[Dict[str, str]]:
    m = GATE_RE.match(name)
    return dict(m.groupdict()) if m else None


def rel_store(row_dir: str, okey: str, suffix: str) -> str:
    return f"{STORES_HEAD}/{row_dir}/{okey}/{THINKING_DIR}/{suffix}"


def is_shared_record_name(name: str, spec_files: Sequence[str], roster_re: "Optional[re.Pattern[str]]" = None) -> bool:
    """A ``records/`` file the readers read as a group-shared record: the named files, the roster index, and the per-tree roster / pins copies (the roster
    copies matched by ``roster_re``, the reader's roster file's variant pattern; the 200 name's when unset)."""
    return name in spec_files or name == ROSTERS_INDEX_FILE or bool((roster_re or ROSTER_VARIANT_RE).match(name)) or bool(PINS_VARIANT_RE.match(name))


@dataclasses.dataclass
class RosterChoice:
    """One launched roster of the set: its document, its input, its file name under ``records/``, the producer tree ``rosters.json`` names for it (None
    for the plain file with no index), whether the set came from ``rosters.json``, and the problem that makes it unusable (None when sound)."""
    doc: Optional[Dict[str, Any]]
    inp: Optional[Input]
    file: str
    tree_sha: Optional[str]
    from_index: bool
    problem: Optional[str] = None
    pins_file: Optional[str] = None
    stores: Optional[List[str]] = None
    pins_sha256: Optional[str] = None


@dataclasses.dataclass
class RosterSet:
    """The mirrored launched rosters by sha256 (:data:`ROSTER_SET_RULE`). ``select`` returns the roster a store's ``lane_roster.json:roster_sha256`` names,
    or raises :class:`_PairRefusal` by name; without an index the plain roster stands for every store (the caller's hash check decides)."""
    entries: Dict[str, RosterChoice]
    plain: Optional[RosterChoice]
    from_index: bool
    index_inp: Optional[Input] = None
    index_problem: Optional[str] = None

    @property
    def usable(self) -> bool:
        return self.index_problem is None and any(e.problem is None and isinstance(e.doc, dict) for e in self.entries.values())

    def summary(self) -> List[Dict[str, Any]]:
        return [{"roster_sha256": sha, "file": e.file, "tree_sha": e.tree_sha, "problem": e.problem, "n_stores_declared": (len(e.stores) if isinstance(e.stores, list) else None)}
                for sha, e in sorted(self.entries.items())]

    def select(self, roster_sha: str, where: str) -> RosterChoice:
        if self.index_problem is not None:
            raise _PairRefusal(self.index_problem, self.index_inp.uri.rsplit("#", 1)[0] if self.index_inp is not None else where)
        e = self.entries.get(roster_sha)
        if e is not None:
            if e.problem is not None:
                raise _PairRefusal(f"the roster {LAUNCHED_ENTRY_FILE}:roster_sha256 {roster_sha[:12]} names is unusable: {e.problem}", where)
            return e
        if not self.from_index and self.plain is not None:
            return self.plain      # the plain file alone: the caller compares the hash (the three-seed family's mirrors; the canary row's earlier revision)
        named = ", ".join(f"{sha[:12]} -> {en.file}" for sha, en in sorted(self.entries.items())) or "none"
        raise _PairRefusal(f"{LAUNCHED_ENTRY_FILE}:roster_sha256 {roster_sha[:12] or '<none>'} matches no mirrored roster ({RECORDS_DIR}/{ROSTERS_INDEX_FILE} names {len(self.entries)}: {named}); "
                           f"the store's launched roster must be mirrored under {RECORDS_DIR}/ before the row can be checked against it", where)


@dataclasses.dataclass
class PinsSet:
    """The mirrored launched pins: the plain file and the per-tree copies (``sdfbase_full22_pins.<sha7>.json``, keyed by the tree prefix they spell), plus the
    files the roster index names per roster (``pins``). ``select`` picks the copy for a store's launch tree, else the plain file."""
    plain: Optional[Tuple[Dict[str, Any], Input]]
    by_tree: Dict[str, Tuple[Dict[str, Any], Input, str]]
    by_name: Dict[str, Tuple[Dict[str, Any], Input]]

    def usable(self, mk: str) -> bool:
        docs = [self.plain[0]] if self.plain is not None else []
        docs += [d for d, _i, _f in self.by_tree.values()] + [d for d, _i in self.by_name.values()]
        return any(isinstance(d, dict) and isinstance(d.get(mk), Mapping) for d in docs)

    def select(self, tree: str, roster: RosterChoice) -> Tuple[Optional[Dict[str, Any]], Optional[Input], str, Optional[str]]:
        """``(doc, input, file name, the tree the copy was chosen for or None for the plain file)``."""
        if roster.pins_file and os.path.basename(roster.pins_file) in self.by_name:
            d, i = self.by_name[os.path.basename(roster.pins_file)]
            return d, i, os.path.basename(roster.pins_file), (roster.tree_sha or tree)
        for prefix in sorted(self.by_tree, key=len, reverse=True):
            if tree.startswith(prefix):
                d, i, f = self.by_tree[prefix]
                return d, i, f, tree
        if self.plain is not None:
            return self.plain[0], self.plain[1], PINS_FILE, None
        return None, None, PINS_FILE, None


@dataclasses.dataclass
class ItemSetPin:
    """The full-suite item set a group's stores must attest (:data:`ITEM_SET_RULE`), resolved once per group from ``records/item_sets.json`` and the pin file it
    names, checked against the registry's declaration; ``problem`` names why it is unusable (then every store of the group refuses by name with it)."""
    id: str
    n: Optional[int]
    of: Optional[int]
    ids_sha256: Optional[str]
    pin_file: str
    pin_file_sha256: Optional[str]
    index_inp: Optional[Input]
    pin_inp: Optional[Input]
    where: str
    problem: Optional[str] = None
    entry: Optional[Dict[str, Any]] = None
    registry_n_items: Optional[int] = None
    registry_items_sha256: Optional[str] = None
    task: Optional[str] = None

    def summary(self) -> Dict[str, Any]:
        return {"id": self.id, "n": self.n, "of": self.of, "ids_sha256": self.ids_sha256, "pin_file": self.pin_file, "pin_file_sha256": self.pin_file_sha256,
                "index_record": (LC.rel_home(self.index_inp.uri.rsplit("#", 1)[0]) if self.index_inp is not None else None),
                "pin_record": (LC.rel_home(self.pin_inp.uri.rsplit("#", 1)[0]) if self.pin_inp is not None else None),
                "registry_n_items": self.registry_n_items, "registry_items_sha256": self.registry_items_sha256, "pin_task": self.task, "problem": self.problem}


class BcbwSeedrepReader:
    """The shared reader; a subclass binds ``SPEC`` and ``name`` (this class declares no ``name``, so it is a helper to the registration
    classifier, never an adapter of its own)."""
    SPEC: TaskSpec
    covers_sources: Tuple[str, ...]
    #: the subclasses declare theirs (the revision identity folds the ADAPTER's, base.emission_version_of); the seedrep readers' current value.
    #: 3 (2026-10-03): every contrast of the six capability readers is decided by the pass's anchor
    #: currency (CURRENCY_RULE) and carries extra.anchor_selection; a park lands on the VALUE row's extra.delta_block. Every subclass bumps by one with it
    #: (the unsteered row `<model key>_u0` and the undeclared-row-dir refusal, 2026-10-05: UNSTEERED_ROW_RULE / UNDECLARED_ROW_DIR_RULE)
    EMISSION_VERSION = 3
    rng_scope = "none"
    allow_symlinks = False
    owner = OWNER
    THINKING_SPELLING = "off"

    def __init__(self, root: Optional[str] = None, seeds: Optional[LC.Seeds] = None, roster_path: Optional[str] = None, pins_path: Optional[str] = None,
                 policy_path: Optional[str] = None, arditi_roster_path: Optional[str] = None, arditi_pins_path: Optional[str] = None, solo_root: Optional[str] = None):
        self.root = LC.expand(root or self.SPEC.root_default)
        self.roots = (self.root,)
        #: a SHARED reader's solo mirror root (the comparison's other side; SHARED_COMPARISON_RULE); None on a solo reader
        self.solo_root: Optional[str] = LC.expand(solo_root or self.SPEC.shared.solo_root_default) if self.SPEC.shared is not None else None
        self._solo_inputs: Dict[str, Input] = {}
        self._seeds = seeds
        self.roster_path = roster_path or DC.repo_path(*ROSTER_REL)
        self.pins_path = pins_path or DC.repo_path(*PINS_REL)
        self.arditi_roster_path = arditi_roster_path or DC.repo_path(*ARDITI_ROSTER_REL)
        self.arditi_pins_path = arditi_pins_path or DC.repo_path(*ARDITI_PINS_REL)
        self.policy_path = policy_path
        #: the per-row refusal ledger of the LAST read (:meth:`read` binds a fresh one to its group and pass stamp, `RowRefusals.read_of`): :meth:`source_states`
        #: writes the `$.row_refusals` row for that read alone (`LC.ROW_REFUSALS_LEDGER_RULE`)
        self.refusals = LC.RowRefusals(self.SPEC.name)
        self._facts: Dict[str, Dict[str, Any]] = {}
        self._roster: Optional[Dict[str, Any]] = None
        self._pins: Optional[Dict[str, Any]] = None
        self._arditi_roster: Optional[Dict[str, Any]] = None
        self._arditi_pins: Optional[Dict[str, Any]] = None
        self._policy: Optional[DP.PolicyFile] = None

    @property
    def seeds(self) -> LC.Seeds:
        if self._seeds is None:
            self._seeds = LC.Seeds.locate()
        return self._seeds

    # ---- discovery ----------------------------------------------------------------------------
    def _stores_root(self) -> str:
        return os.path.join(self.root, STORES_HEAD)

    def _records_dir(self) -> str:
        return os.path.join(self.root, RECORDS_DIR)

    def shared_files(self) -> Tuple[str, ...]:
        out = tuple(self.SPEC.launch_set_files) + (PINS_FILE, self.SPEC.roster_file, self.SPEC.three_numbers_file, ROWS_LAUNCH_FILE, self.SPEC.schedule_file, RELAUNCHES_FILE)
        if self.SPEC.full_item_set:
            out += (ITEM_SETS_FILE, str(self.SPEC.item_set_pin_file))     # the item-set index and the pin file it names (ITEM_SET_RULE)
        return out

    def discover(self) -> List[Input]:
        """Named files of every ``native/<row dir>/<object key>/thinking_off/<suffix>/`` store under the mirror root, and the ``records/`` files
        aliased per group (``#<seed tag>``: the shared records for every group that holds a store, each gate record for the group its name
        spells); an UNDECLARED row dir is discovered as a group of its own that :meth:`read` refuses by name (:data:`UNDECLARED_ROW_DIR_RULE`).
        An absent root is a SKIP (the mirror is machine-local); a root holding no store refuses."""
        out: List[Input] = []
        if not os.path.isdir(self.root):
            return out
        groups: Set[str] = set()
        for row_dir in IC.listdir(self._stores_root()):
            if not os.path.isdir(os.path.join(self._stores_root(), row_dir)) or row_dir.startswith("."):
                continue        # not a row: a stray file, or a producer's dot-prefixed publish scaffolding (`.staging`), UNDECLARED_ROW_DIR_RULE
            tag = tag_of_row_dir(row_dir, self.SPEC)
            if tag is None:
                # an UNDECLARED row dir (UNDECLARED_ROW_DIR_RULE): its stores are discovered into a group of their own so read() refuses it BY NAME --
                # never the silent skip that read a new model's mirror as 0 records (the 35B rows). No shared record is aliased to it. The s2rl row dir
                # is undeclared FOR THIS READER where it does not read the row (lecun_relu_row_admitted): the same whole-group refusal the base tree filed
                out += self._discover_undeclared_row_dir(row_dir)
                continue
            for okey in IC.listdir(os.path.join(self._stores_root(), row_dir)):
                td = os.path.join(self._stores_root(), row_dir, okey, THINKING_DIR)
                if not os.path.isdir(td):
                    continue
                for suffix in IC.listdir(td):
                    d = os.path.join(td, suffix)
                    if not os.path.isdir(d):
                        continue
                    if suffix_task_for(self.SPEC, suffix) not in (None, self.SPEC.eval_id):
                        continue        # the OTHER task's one-task store (ONE_TASK_STORE_RULE): not this source's; neither read nor refused
                    # (a suffix outside the grammar is discovered too: read() refuses it by name rather than dropping it, LAUNCHED_AS_TREE_RULE)
                    n_before = len(out)
                    out += self._discover_store(d, okey)
                    if len(out) > n_before:
                        groups.add(seed_tag_of(tag))
        if not out:
            raise AdapterRefusal(self.root, f"no {self.SPEC.eval_id} seed-replicate store found ({STORES_HEAD}/<model key>-f22s<N>[p]/<object key>/{THINKING_DIR}/bcbw-<launch tag>-<sha7>/cells/<store key>/ "
                                            f"under the root; the source's families: {', '.join(self.SPEC.models)})")
        if self.SPEC.shared is not None:
            out += self._discover_solo_inputs(groups)       # the solo stores the comparison stands against (SHARED_COMPARISON_RULE): the group's inputs too
        for name in self.shared_files():
            p = os.path.join(self._records_dir(), name)
            if os.path.isfile(p):
                base = check_input(p, role="report", allow_symlink=self.allow_symlinks)
                out += [base.with_alias(g) for g in sorted(groups)]
        for d in RELAUNCHES_DIRS[1:]:
            p = os.path.join(self.root, d, RELAUNCHES_FILE)
            if os.path.isfile(p):
                base = check_input(p, role="report", allow_symlink=self.allow_symlinks)
                out += [base.with_alias(g) for g in sorted(groups)]
        for name in IC.listdir(self._records_dir()):
            if name not in self.shared_files() and is_shared_record_name(name, self.shared_files(), self.SPEC.roster_variant_re) and os.path.isfile(os.path.join(self._records_dir(), name)):
                # the roster set (ROSTER_SET_RULE): records/rosters.json and the per-tree roster / pins copies
                base = check_input(os.path.join(self._records_dir(), name), role="report", allow_symlink=self.allow_symlinks)
                out += [base.with_alias(g) for g in sorted(groups)]
                continue
            parts = gate_name_parts(name)
            if parts is None or suffix_task_for(self.SPEC, parts["suffix"]) not in (None, self.SPEC.eval_id):
                continue        # (a gate record of the OTHER task's one-task store is not this source's either, ONE_TASK_STORE_RULE)
            if is_lecun_relu_tag(parts["tag"]) and not lecun_relu_row_admitted(lecun_relu_model_key(parts["tag"]), self.SPEC):
                continue        # the gate record of an s2rl row this reader does not read: not this source's (its row dir refuses as a whole, UNDECLARED_ROW_DIR_RULE)
            g = seed_tag_of(parts["tag"])
            if g in groups:
                base = check_input(os.path.join(self._records_dir(), name), role="report", allow_symlink=self.allow_symlinks)
                out.append(base.with_alias(g))
        return out

    def _discover_undeclared_row_dir(self, row_dir: str) -> List[Input]:
        """The named store files under an undeclared row dir (:data:`UNDECLARED_ROW_DIR_RULE`): every ``<object key>/thinking_off/<suffix>/`` directory, whatever
        its suffix spells (the suffix grammar needs the model key, which this row has not got), so the group exists and refuses by name."""
        out: List[Input] = []
        for okey in IC.listdir(os.path.join(self._stores_root(), row_dir)):
            td = os.path.join(self._stores_root(), row_dir, okey, THINKING_DIR)
            if not os.path.isdir(td):
                continue
            for suffix in IC.listdir(td):
                d = os.path.join(td, suffix)
                if os.path.isdir(d):
                    out += self._discover_store(d, okey)
        return out

    def _discover_solo_inputs(self, groups: Set[str]) -> List[Input]:
        """A SHARED reader's other side (:data:`SHARED_COMPARISON_RULE`): for every launch row of this eval naming a ``solo_store``, the solo store's
        ``RESULT``, ``manifest.json`` and outcome record under the solo mirror root, aliased ``#<group>`` like the shared records -- inputs of the group, so a
        re-mirrored solo store re-reads the comparison; a solo store the mirror lacks registers nothing (the run then files with the comparison pending)."""
        spec = self.SPEC
        out: List[Input] = []
        present = [n for n in spec.launch_set_files if os.path.isfile(os.path.join(self._records_dir(), n))]
        if len(present) != 1 or self.solo_root is None or not os.path.isdir(self.solo_root):
            return out
        try:
            with open(os.path.join(self._records_dir(), present[0]), "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            return out
        seen_paths: Set[str] = set()
        for r in self._launch_rows(doc if isinstance(doc, dict) else None) or []:
            g = seed_tag_of(str(r.get("tag") or ""))
            solo = r.get("solo_store") if isinstance(r.get("solo_store"), Mapping) else None
            if g not in groups or solo is None or not solo.get("mirror_rel") or not solo.get("store_key"):
                continue
            d = os.path.join(self.solo_root, str(solo["mirror_rel"]))
            paths = [os.path.join(d, n) for n in SOLO_STORE_FILES] + [os.path.join(d, "cells", str(solo["store_key"]), "outcomes", spec.outcomes_file)]
            for p in paths:
                if p in seen_paths or not os.path.isfile(p):
                    continue
                seen_paths.add(p)
                base = check_input(p, role="report", allow_symlink=self.allow_symlinks)
                out.append(base.with_alias(g))
        return out

    def _discover_store(self, d: str, okey: str) -> List[Input]:
        out: List[Input] = []
        for name in STORE_TOP_FILES:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                out.append(check_input(p, role=ROLE_OF_FILE.get(name, "report"), allow_symlink=self.allow_symlinks))
        p = os.path.join(d, okey, CONTROLS_SUMMARY_FILE)
        if os.path.isfile(p):
            out.append(check_input(p, role="report", allow_symlink=self.allow_symlinks))
        for key in IC.listdir(os.path.join(d, "cells")):
            kd = os.path.join(d, "cells", key)
            if not os.path.isdir(kd):
                continue
            for name in STORE_KEY_FILES:
                p = os.path.join(kd, name)
                if os.path.isfile(p):
                    out.append(check_input(p, role=ROLE_OF_FILE.get(name, "report"), allow_symlink=self.allow_symlinks))
            for sub in TASK_SUBDIRS:
                p = os.path.join(kd, sub, self.SPEC.outcomes_file)
                if os.path.isfile(p):
                    out.append(check_input(p, role="report", allow_symlink=self.allow_symlinks))
        return out

    def comparison_group(self, inp: Input) -> str:
        """One group per SEED (:data:`GROUP_RULE`): a store under ``native/<row dir>/`` maps to the row's seed tag; a record rides a ``#<seed tag>`` alias."""
        base = os.path.basename(inp.uri)
        if "#" in base and seed_tag_of(base.rsplit("#", 1)[1]) == base.rsplit("#", 1)[1]:
            return base.rsplit("#", 1)[1]
        rel = os.path.relpath(inp.uri, self.root).split(os.sep)
        if len(rel) > 1 and rel[0] == STORES_HEAD and tag_of_row_dir(rel[1], self.SPEC) is not None:
            return seed_tag_of(tag_of_row_dir(rel[1], self.SPEC))
        if len(rel) > 1 and rel[0] == STORES_HEAD and undeclared_row_dir_reason(rel[1], self.SPEC) is not None:
            return rel[1]       # an undeclared row dir is a group of its own, named for itself; read() refuses it by name (UNDECLARED_ROW_DIR_RULE)
        raise AdapterRefusal(inp.uri, f"input names no seed tag ({STORES_HEAD}/<model key>-f22s<N>[p]/) under {self.root}")

    def _group_selections(self, group: str) -> Tuple[str, Optional[str]]:
        """``(the group's own selection, the paired selection)``: ``(s<N>, s<N>p)``."""
        seed_n = tag_parts(group)[0]
        return selection_key(seed_n), selection_key(seed_n, True)

    def source_states(self, ctx: AdapterContext, inputs: Sequence[Input]) -> List[SourceStateRow]:
        rows: List[SourceStateRow] = []
        spec = self.SPEC
        for inp in inputs:
            base = os.path.basename(inp.uri)
            if "#" not in base:
                continue
            name = base.rsplit("#", 1)[0]
            group = self.comparison_group(inp)
            mk = model_key_of(group) or MODEL_KEY
            mkey = MODEL_KEYS[mk]
            # the SAME tag map read() binds: the two-argument call omitted the u0 selection, so the launch-row and
            # three-number memberships of a u0 family left out every u0 member and a self-retiring ruling keyed to that input would have evaluated
            # against a truncated snapshot
            tags = group_tags(group, spec.instruct_direct, spec.unsteered_row, spec.dose_relu_row, spec.dose_lora_row, spec.lecun_relu_row)
            u_tag = tags.get(UNSTEERED_SELECTION)
            # the tag spellings a producer record may carry for the group's rows: its tags (the instruct-direct and unsteered ones included where admitted),
            # plus the `main` canary's own tag where the family aliases it
            spellings = set(tags.values()) | ({f"{mk}{LEGACY_MAIN_SUFFIX}"} if mkey.legacy_main_alias else set())
            if name == spec.three_numbers_file:
                d = self._json(inp) or {}
                members = sorted(k for k in (d.get("rows") or {}) if str(k).split("/", 1)[0] in spellings)
                rows.append(SourceStateRow(source_id_for(spec.name, group), inp.uri, "$.rows", 1, 1, members, inp.sha256))
            elif name in spec.launch_set_files:
                doc = self._json(inp)
                launch = self._launch_rows(doc)
                members = {str(r.get("tuple_id")) for r in (launch or []) if self._launch_row_key(r, mk, tags) is not None}
                if u_tag is not None:
                    # the u0 `none` RE-SERVED at another window has no launch row: its declaration is families.<model key>.unsteered.window_reserves.<eval>
                    # (UNSTEERED_ROW_RULE), a member of the row table under the synthetic tuple id read() keys it by (`u0/none@w<N>`)
                    members |= {f"{UNSTEERED_SELECTION}/{row_okey('none', None, n)}" for _name, _e, _suffix, n, seg, _where in self._window_reserve_declarations(doc, mk)
                                if n is not None and seg == n}
                rows.append(SourceStateRow(source_id_for(spec.name, group), inp.uri, self.row_table_json_path(doc, spec.launch_set_eval_key), 1, 1, sorted(members), inp.sha256))
            elif name == ITEM_SETS_FILE and spec.full_item_set:
                # the item sets the mirror declares (ITEM_SET_RULE); the one this reader files on is checked per group in read()
                try:
                    with open(inp.path, "r", encoding="utf-8") as fh:
                        d_any = json.load(fh)
                except (OSError, ValueError):
                    d_any = None
                members = sorted(str(e.get("id")) for e in item_set_entries(d_any) if e.get("id") is not None)
                rows.append(SourceStateRow(source_id_for(spec.name, group), inp.uri, "$", 1, 1, members, inp.sha256))
            elif name == PINS_FILE:
                d = self._json(inp) or {}
                parts = tag_parts(group)
                seed_n = parts[0]
                # the three-seed family's pins spell the seed in the staged id (`base64s<N>__…`); an appendix family's are the `main` selection's ids, every one a member
                members = sorted(k for k in ((d.get(mk) or {}) if isinstance(d.get(mk), Mapping) else {})
                                 if not mkey.roster_selections or f"s{seed_n}__" in str(k) or str(k).startswith(f"ar_c"))
                rows.append(SourceStateRow(source_id_for(spec.name, group), inp.uri, f"$.{mk}", 1, 1, members, inp.sha256))
        if inputs:
            # the `$.row_refusals` state of the group THIS read filed (`LC.ROW_REFUSALS_LEDGER_RULE`): keyed on the group, never on `inputs[0]` (a file that
            # leaves or a re-ordered discovery would move the key and leave the old row standing found = 1 with a stale list), written on EVERY read of the
            # group, the empty list included, so a refusal that cleared (a park whose anchor became current and filed its contrast on the pass after the fold)
            # is seen to clear instead of standing as the last value written. The runner collects the state rows right after the group's read, under the
            # read's pass stamp; a re-activation replays the committed revision's own rows and never asks here -- asked anyway, for this group under another
            # stamp, the ledger refuses by name rather than serving a later read's list as an earlier revision's state
            group = self.comparison_group(inputs[0])
            rows += self.refusals.ledger_rows(ctx, group, inputs, source_id_for(spec.name, group))
        return rows

    def status_extra(self, group: str) -> Dict[str, Any]:
        return dict(self._facts.get(group) or {})

    def status_sources(self, group: str) -> Tuple[str, ...]:
        return (self.SPEC.source,)

    # ---- small readers ------------------------------------------------------------------------
    @staticmethod
    def _json(inp: Optional[Input]) -> Optional[Dict[str, Any]]:
        if inp is None:
            return None
        with open(inp.path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else None

    @staticmethod
    def _load_pair(roster_path: str, pins_path: str, what: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not os.path.isfile(roster_path):
            raise IC.Unresolved(roster_path, f"the {what} roster is not in this checkout; no staged id can be named")
        if not os.path.isfile(pins_path):
            raise IC.Unresolved(pins_path, f"the {what} pins are not in this checkout; no object sha can be named")
        import yaml
        with open(roster_path, "r", encoding="utf-8") as fh:
            roster = yaml.safe_load(fh)
        with open(pins_path, "r", encoding="utf-8") as fh:
            pins = json.load(fh)
        if not isinstance(roster, dict) or not isinstance(pins, dict):
            raise IC.Unresolved(roster_path, f"the {what} roster / pins are not mappings")
        return roster, pins

    def _roster_pins(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self._roster is None or self._pins is None:
            self._roster, self._pins = self._load_pair(self.roster_path, self.pins_path, "seed-replicate")
        return self._roster, self._pins

    def _arditi_roster_pins(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self._arditi_roster is None or self._arditi_pins is None:
            self._arditi_roster, self._arditi_pins = self._load_pair(self.arditi_roster_path, self.arditi_pins_path, "all-positions")
        return self._arditi_roster, self._arditi_pins

    def _policy_file(self) -> Optional[DP.PolicyFile]:
        if self._policy is None:
            try:
                self._policy = DP.load(self.policy_path)
            except (OSError, ValueError, KeyError, DP.NoPolicy):
                self._policy = None
        return self._policy

    @staticmethod
    def _launch_rows_of(doc: Optional[Mapping[str, Any]], eval_id: str) -> Optional[List[Dict[str, Any]]]:
        """The eval's ROW TABLE (:data:`ROW_TABLE_RULE`): ``launch`` (required, a list) plus ``carried`` and ``carried_late`` when they are lists; None
        when the document is not a launch set or names no ``launch`` list for the eval."""
        if not isinstance(doc, Mapping) or str(doc.get("schema")) != LAUNCH_SET_SCHEMA:
            return None
        evals = doc.get("evals")
        blk = evals.get(eval_id) if isinstance(evals, Mapping) and isinstance(evals.get(eval_id), Mapping) else None
        if blk is None or not isinstance(blk.get("launch"), list):
            return None
        rows: List[Dict[str, Any]] = []
        for name in ROW_TABLE_LISTS:
            lst = blk.get(name)
            if isinstance(lst, list):
                rows += [dict(r) for r in lst if isinstance(r, Mapping)]
        return rows

    @staticmethod
    def row_table_json_path(doc: Optional[Mapping[str, Any]], eval_id: str) -> str:
        """The JSON path the row table was read under: ``$.evals.<eval>.launch`` plus every carried list that NAMES a row (the first launch set carried
        nothing — its empty ``carried`` / ``carried_late`` lists leave the path it always had)."""
        evals = doc.get("evals") if isinstance(doc, Mapping) else None
        blk = evals.get(eval_id) if isinstance(evals, Mapping) and isinstance(evals.get(eval_id), Mapping) else {}
        lists = ("launch",) + tuple(n for n in ROW_TABLE_LISTS[1:] if isinstance(blk.get(n), list) and blk.get(n))
        return f"$.evals.{eval_id}.launch" if lists == ("launch",) else f"$.evals.{eval_id}.{{{','.join(lists)}}}"

    def _launch_rows(self, doc: Optional[Mapping[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        return self._launch_rows_of(doc, self.SPEC.launch_set_eval_key)

    # ---- objects ------------------------------------------------------------------------------
    def resolve_object(self, okey: str, selection: str, pin_key: Optional[str] = None, model_key: str = MODEL_KEY,
                       roster_selection: Optional[str] = None, launched_pins: Optional[Mapping[str, Any]] = None, launched_object_sha: Optional[str] = None,
                       composed: bool = False, launched_pins_file: Optional[str] = None) -> Tuple[LC.ResolvedObject, Optional[DG.ArditiAllposRung]]:
        """The object a store's key names at the harness selection: ``none``, a declared prompt (``si:<sha>`` from si_texts), a staged
        seed-replicate object (the roster's ``s<N>[p]`` selection -> staged id -> the pins' sha, :func:`jobplan.object_id_for`), or the
        all-positions object through ITS roster / pins at the operating point the launch set's pin key spells. For an appendix family
        (:data:`MODEL_KEYS` ``roster_selections`` false) the harness roster stages no ``s1`` / ``s1p`` selection: the staged id IS the launch
        set's pin key and its sha the harness pins' entry under the model key (a key the pins do not carry is unresolved by name), with the
        roster's ``main`` selection's staged id required to agree where it stages the key. ``roster_selection`` names the roster selection to resolve
        through when it is not the tag's own (the ``main`` canary row's stores: :data:`LEGACY_MAIN_ALIAS_RULE`). A COMPOSED key the harness pins do not
        carry (``composed``: the launch set's class) resolves through the store's tree pins (``launched_pins``, the per-tree copy) by the launched entry's
        ``object_sha256`` (``launched_object_sha``, ``lane_roster.json``), the key found its staged id (:data:`PINS_KEY_RULE`). Never the live registry."""
        mk = model_key
        mkey = MODEL_KEYS[mk]
        g = LC.cell_grammar_legacy(okey)
        kind = g["object_kind"]
        if kind == "none":
            return LC.ResolvedObject("none", "none", "none", "baseline_none", "derived", "the unsteered object key", derivation="none"), None
        if kind == "system_instruction":
            oid = self.seeds.si_object_id(g["si_id"] or "")
            if oid is None:
                raise IC.Unresolved(okey, f"si_texts.yaml carries no sha256 for the declared prompt {g['si_id']!r}")
            return LC.ResolvedObject(oid, kind, g["paper_label"], "prompt", "recorded", "si_texts.yaml si_text_sha256", si_id=g["si_id"], derivation="none"), None
        if kind is None:
            raise IC.Unresolved(okey, "the key matches no object grammar this reader knows")
        if okey == ARDITI_OBJECT_KEY:
            point = str(pin_key or "").rsplit("__", 1)[0] if pin_key else ""
            op = DG.parse_arditi_allpos_tag(f"{mk}_{point}") if point else None
            if op is None or op.trial is not None or op.model_key != mk or op.staged_id != pin_key:
                raise IC.Unresolved(okey, f"the launch set's pin key {pin_key!r} spells no all-positions operating point (dose_grammar `ar_c<tok>__arditi_allpos`)")
            roster, pins = self._arditi_roster_pins()
            try:
                sha = JP.object_id_for(okey, model_key=mk, roster_selection=op.rung, roster=roster, pins=pins, si_sha={})
            except LookupError as exc:
                raise IC.Unresolved(okey, f"the all-positions roster's {op.rung!r} selection / the pins name no object for it: {exc}") from None
            staged = str(roster["targets"][mk]["views"][op.rung]["cells"][okey])
            if staged != op.staged_id:
                raise IC.Unresolved(okey, f"the all-positions roster stages {op.rung}/{okey} as {staged!r}, not the operating point's {op.staged_id!r} (dose_grammar)")
            pin = (pins.get(mk) or {}).get(staged) or {}
            bad: List[str] = []
            if str(pin.get("kind") or "") != DG.OBJECT_KIND:
                bad.append(f"kind {pin.get('kind')!r} is not {DG.OBJECT_KIND!r}")
            if LC.num_or_none(pin.get("c")) is None or abs(float(pin["c"]) - op.coefficient) > 1e-9:
                bad.append(f"c {pin.get('c')!r} is not the operating point's {op.coefficient:g}")
            if str(pin.get("dose_unit") or "") != op.unit:
                bad.append(f"dose_unit {pin.get('dose_unit')!r} is not {op.unit!r}")
            if bad:
                raise IC.Unresolved(okey, f"the pin {mk}.{staged} is not the operating point: " + "; ".join(bad))
            obj = LC.ResolvedObject(str(sha), DG.OBJECT_KIND, DG.PAPER_LABEL, DG.OBJECT_ROLE, "recorded",
                                    f"the launch set's pin key {staged} -> all-positions roster targets.{mk}.views.{op.rung}.cells[{okey}] -> {staged}; pins {mk}.{staged}."
                                    "object_sha256 (jobplan.object_id_for, the harness's own resolution); the launched pins' entry; attested by the store's provenance.artifact_sha256, "
                                    "fit_provenance.object_sha and lane_roster.json:object_sha256",
                                    staged_id=staged, derivation="none")
            return obj, op
        roster, pins = self._roster_pins()
        hop = roster_selection if roster_selection is not None else (selection if mkey.roster_selections else None)
        if hop is None:
            # an appendix family: no `s1` / `s1p` selection in the harness roster — the launch set's pin key IS the staged id, the harness pins name its sha
            if not pin_key:
                raise IC.Unresolved(okey, f"the launch set names no pin key for a fitted key of {mk}: the harness roster stages no {selection!r} selection for the family (D565), "
                                          "so the pin key is the only staged id")
            pin = (pins.get(mk) or {}).get(pin_key) if isinstance(pins.get(mk), Mapping) else None
            sha = pin.get("object_sha256") if isinstance(pin, Mapping) else None
            if not isinstance(sha, str) or not _SHA256.match(sha):
                gap = (f"the harness pins ({'/'.join(PINS_REL)}) carry no object_sha256 under {mk}.{pin_key}: the harness roster stages no {selection!r} selection "
                       f"for {mk} (D565), so the launch set's pin key must be pinned there before the row can name its object")
                if composed and isinstance(launched_pins, Mapping) and isinstance(launched_object_sha, str) and _SHA256.match(launched_object_sha):
                    # the composed paired-write object (PINS_KEY_RULE): the store's tree pins, by the sha its lane_roster.json names, any key under the model's block
                    blk = launched_pins.get(mk) if isinstance(launched_pins.get(mk), Mapping) else {}
                    hits = sorted(str(k) for k, v in blk.items() if isinstance(v, Mapping) and str(v.get("object_sha256") or "") == launched_object_sha)
                    where_pins = f"{RECORDS_DIR}/{launched_pins_file}" if launched_pins_file else "the store's tree pins"
                    if len(hits) == 1:
                        return LC.ResolvedObject(launched_object_sha, kind, g["paper_label"], g["object_role"], "recorded",
                                                 f"{LAUNCHED_ENTRY_FILE}:object_sha256 -> {where_pins} {mk}.{hits[0]} (the one entry under the model's block with that object_sha256; "
                                                 f"{PINS_KEY_RULE}; the harness pins carry no {pin_key}); the launch set's pin key must be that key and its expected sha; attested by "
                                                 "the store's provenance.artifact_sha256, fit_provenance.object_sha and the launched roster's object.sha256",
                                                 staged_id=hits[0], derivation=g["derivation"], si_id=g["si_id"]), None
                    if len(hits) > 1:
                        raise IC.Unresolved(okey, f"{gap}; {where_pins} names {len(hits)} entries under {mk} with the launched object_sha256 {launched_object_sha[:16]} ({', '.join(hits)}): "
                                                  "the composed object cannot be named by sha either")
                    raise IC.Unresolved(okey, f"{gap}; and {where_pins} names no entry under {mk} with the launched object_sha256 {launched_object_sha[:16]} ({LAUNCHED_ENTRY_FILE}): the "
                                              "composed object is pinned nowhere the reader can see")
                raise IC.Unresolved(okey, gap)
            main_blk = ((roster.get("targets") or {}).get(mk) or {}).get("views", {}).get(MAIN_SELECTION) if isinstance(roster.get("targets"), Mapping) else None
            main_staged = (main_blk.get("cells") or {}).get(okey) if isinstance(main_blk, Mapping) and isinstance(main_blk.get("cells"), Mapping) else None
            if main_staged is not None and str(main_staged) != pin_key:
                raise IC.Unresolved(okey, f"the harness roster's `{MAIN_SELECTION}` selection stages {okey} as {str(main_staged)!r} for {mk}, the launch set's pin key is {pin_key!r}: the "
                                          f"appendix family's {selection} objects are that selection's staged objects (Overflow's stage index), so the two must agree")
            obj = LC.ResolvedObject(sha, kind, g["paper_label"], g["object_role"], "recorded",
                                    f"the launch set's pin key {pin_key} -> pins {mk}.{pin_key}.object_sha256 (the harness roster stages no {selection} selection for {mk}: the pin key "
                                    f"IS the staged id, D565; the roster's {MAIN_SELECTION} selection "
                                    + (f"stages the same id" if main_staged is not None else f"does not stage {okey} for {mk}")
                                    + "); the launch set's expected sha; the launched pins' entry; attested by the store's provenance.artifact_sha256, fit_provenance.object_sha and "
                                    "lane_roster.json:object_sha256",
                                    staged_id=pin_key, derivation=g["derivation"], si_id=g["si_id"])
            return obj, None
        try:
            sha = JP.object_id_for(okey, model_key=mk, roster_selection=hop, roster=roster, pins=pins, si_sha={})
        except LookupError as exc:
            raise IC.Unresolved(okey, f"the roster's {hop!r} selection / the pins name no object for it: {exc}") from None
        staged = roster["targets"][mk]["views"][hop]["cells"][okey]
        via = "" if hop == selection else f" (the `{MAIN_SELECTION}` canary row's objects are the roster's {hop} selection's, D565)"
        obj = LC.ResolvedObject(str(sha), kind, g["paper_label"], g["object_role"], "recorded",
                                f"roster targets.{mk}.views.{hop}.cells[{okey}] -> {staged}; pins {mk}.{staged}.object_sha256 (jobplan.object_id_for, the harness's own "
                                f"resolution{via}); the launch set's pin key and expected sha; the launched pins' entry; attested by the store's provenance.artifact_sha256, "
                                "fit_provenance.object_sha and lane_roster.json:object_sha256",
                                staged_id=str(staged), derivation=g["derivation"], si_id=g["si_id"])
        return obj, None

    # ---- read ---------------------------------------------------------------------------------
    def read(self, ctx: AdapterContext, group: str, inputs: Sequence[Input]) -> Iterator[RecordBatch]:
        spec = self.SPEC
        batch = RecordBatch(source_id_for(spec.name, group), group)
        self.refusals = LC.RowRefusals(spec.name, read_of=(group, ctx.now))      # the ledger of THIS read: its group, under this pass stamp
        facts: Dict[str, Any] = {"group": group, "runs": 0, "rows": {}, "contrasts": {}, "not_run": [], "refusals": [], "instrument_seed_id": spec.instrument_seed,
                                 "group_rule": GROUP_RULE, "anchor_rule": ANCHOR_RULE, "producer_key_rule": PRODUCER_KEY_RULE, "seed_axis_rule": SEED_AXIS_RULE,
                                 "not_run_rule": NOT_RUN_RULE, "gate_rule": GATE_RULE}
        if is_undeclared_row_dir_group(group, spec):
            # an UNDECLARED row dir (UNDECLARED_ROW_DIR_RULE): the group is the row dir itself; refuse it BY NAME and emit nothing (an explained empty read)
            why = str(undeclared_row_dir_reason(group, spec))
            # facts.refuse is what the STATUS reads (runner: ok = accepted and no refuse entry; v_source_status folds MIN(ok) over a source's groups and
            # lane_guard reads extra.refuse alone), so the whole-group refusal rides it -- a `refusals` entry alone is an explained empty read that the
            # status still files ok = 1 with 'no row refused'; facts.refusals keeps the by-name listing
            facts.update(undeclared_row_dir=group, undeclared_row_dir_rule=UNDECLARED_ROW_DIR_RULE, n_store_inputs=len(inputs), refuse=[why])
            self.refusals.add(group, why, os.path.join(self._stores_root(), group))
            self._finish(group, facts)
            yield batch
            return
        parts = tag_parts(group)
        mk = model_key_of(group)
        if parts is None or parts[1] or mk is None:
            self._refuse_group(group, facts, f"group {group!r} does not spell a seed tag (<model key>_s<N>)", inputs[0].uri if inputs else self.root)
            yield batch
            return
        if mk not in spec.models:
            self._refuse_group(group, facts, f"model key {mk!r} is not one of this source's families ({', '.join(spec.models)}); its stores belong to another source", self.root)
            yield batch
            return
        mkey = MODEL_KEYS[mk]
        seed_n = parts[0]
        if seed_n not in mkey.seeds:
            self._refuse_group(group, facts, f"{mk} runs the seed tags {[f'{mk}_s{k}' for k in mkey.seeds]} (the model-key table); {group!r} is outside them", self.root)
            yield batch
            return
        sel_seed, sel_paired = self._group_selections(group)
        tags = group_tags(group, spec.instruct_direct, spec.unsteered_row, spec.dose_relu_row, spec.dose_lora_row, spec.lecun_relu_row)
        instruct_tag = tags.get(INSTRUCT_SELECTION)
        unsteered_tag = tags.get(UNSTEERED_SELECTION)
        dose_relu_tag = tags.get(DOSE_RELU_SELECTION)
        dose_lora_tag = tags.get(DOSE_LORA_SELECTION)
        lecun_relu_tag = tags.get(LECUN_RELU_SELECTION)
        facts.update(seed=seed_n, selections=[sel_seed, sel_paired], tags=sorted(t for t in tags.values() if t != instruct_tag))
        if dose_relu_tag is not None:
            facts.update(dose_relu_row_tag=dose_relu_tag, dose_relu_row_rule=DOSE_RELU_ROW_RULE)     # stamped on the families that run the row alone
        facts["currency_rule"] = CURRENCY_RULE
        if dose_lora_tag is not None:
            facts.update(dose_lora_row_tag=dose_lora_tag, dose_lora_row_rule=DOSE_LORA_ROW_RULE)
        if lecun_relu_tag is not None:
            facts.update(lecun_relu_row_tag=lecun_relu_tag, lecun_relu_row_rule=LECUN_RELU_ROW_RULE)
        seeds = self.seeds
        # the model: the table's row, which legacy_spellings.yaml (the run payload's resolution) and models.yaml (the served checkpoint and its
        # revision) must both spell the same way — a disagreement refuses the group rather than filing on one of three answers
        model_id = mkey.model_id
        mv = seeds.model_for_view_legacy(group)
        if mv.get("model_id") != model_id:
            self._refuse_group(group, facts, f"legacy_spellings.yaml spells {group!r} as model {mv.get('model_id')!r} ({mv.get('source')}); the model-key table says {model_id}: a new tag needs "
                                            "its own row in legacy_spellings.yaml model_keys naming that model", self.root)
            yield batch
            return
        seed_model = seeds.models.get(model_id) or {}
        if str(seed_model.get("hf_repo") or "") != mkey.served_hf_id or str(seed_model.get("hf_revision") or "") != mkey.served_revision:
            self._refuse_group(group, facts, f"models.yaml pins {model_id} at {seed_model.get('hf_repo')!r} @ {str(seed_model.get('hf_revision'))[:12]}; the model-key table says "
                                            f"{mkey.served_hf_id!r} @ {mkey.served_revision[:12]}: the two must agree before a row files", self.root)
            yield batch
            return
        if spec.shared is not None:
            # a SHARED reader (SHARED_SPELLING_RULE): every `_shared` spelling the group's rows file under must resolve to the same model as the solo spelling
            facts.update(shared_source_rule=SHARED_SOURCE_RULE, shared_spelling_rule=SHARED_SPELLING_RULE, shared_run_id_rule=SHARED_RUN_ID_RULE, shared_row_key_rule=SHARED_ROW_KEY_RULE,
                         shared_standup_rule=SHARED_STANDUP_RULE, shared_comparison_rule=SHARED_COMPARISON_RULE, shared_it_none_rule=SHARED_IT_NONE_RULE,
                         shared_row_dir_rule=SHARED_ROW_DIR_RULE, shared_endpoint_tuple_rule=SHARED_ENDPOINT_TUPLE_RULE, shared_it_repeat_rule=SHARED_IT_REPEAT_RULE,
                         # the source_status note of SHARED_ENDPOINT_TUPLE_RULE: one entry per store whose endpoint names the row's last key (empty when every endpoint agrees)
                         endpoint_served_key_mismatch=[],
                         solo_source=spec.shared.solo_source, solo_root=(LC.rel_home(self.solo_root) if self.solo_root else None), solo_root_present=bool(self.solo_root and os.path.isdir(self.solo_root)),
                         spellings={t: spec.spelling_of(t) for t in sorted(tags.values())}, comparisons={}, standups={})
        try:
            inst_row = seeds.instrument_row(spec.instrument_seed, spec.eval_id, model_id=model_id)
        except KeyError as exc:
            self._refuse_group(group, facts, f"instrument row: {exc}", self.root)
            yield batch
            return
        facts.update(model_id=model_id, instrument_id=str(inst_row["instrument_id"]))
        stores, shared, gates = self._sort_inputs(inputs)
        pins_inp, roster_inp, tn_inp = (shared.get(n) for n in (PINS_FILE, spec.roster_file, spec.three_numbers_file))
        ls_present = [n for n in spec.launch_set_files if n in shared]
        ls_inp = shared[ls_present[0]] if len(ls_present) == 1 else None
        ls_problem = (f"{RECORDS_DIR}/ carries {len(ls_present)} launch sets ({', '.join(ls_present)}); the row table must be ONE file (the names this reader reads: "
                      f"{', '.join(spec.launch_set_files)})" if len(ls_present) > 1 else None)
        ls_doc = self._json(ls_inp) if ls_inp is not None else None
        launch_rows = self._launch_rows(ls_doc) if ls_inp is not None else None
        if spec.shared is not None and spec.shared.harness_variant is not None:
            # a landing-2 reader (SHARED_L2_RUN_ID_RULE): the source is the ROOT; the launch set's evals.<eval>.own_source may spell the BASE source (landing 2's bound set does)
            # and is recorded as provenance, never bound
            ls_evals = ls_doc.get("evals") if isinstance(ls_doc, Mapping) and isinstance(ls_doc.get("evals"), Mapping) else {}
            ls_blk = ls_evals.get(spec.launch_set_eval_key) if isinstance(ls_evals.get(spec.launch_set_eval_key), Mapping) else {}
            facts.update(harness_variant=spec.shared.harness_variant, base_source=spec.shared.base_source, launch_set_own_source_spelled=ls_blk.get("own_source"),
                         launch_set_own_source_is_base=(ls_blk.get("own_source") == spec.shared.base_source), source_binding="root", shared_l2_run_id_rule=SHARED_L2_RUN_ID_RULE,
                         shared_verified_standup_rule=SHARED_VERIFIED_STANDUP_RULE)
            if spec.shared.landing_rule is not None:
                facts["landing_rule"] = spec.shared.landing_rule        # landing 3's account of its own shapes (SHARED_L3_RULE); landing 2's status carries no such key
        trees = [dict(t) for t in (ls_doc.get("trees") or [])] if isinstance(ls_doc, Mapping) and isinstance(ls_doc.get("trees"), list) else []
        primary_trees = [str(t.get("sha")) for t in trees if isinstance(t, Mapping) and str(t.get("role") or "") == "primary" and _SHA1.match(str(t.get("sha") or ""))]
        primary_tree = primary_trees[0] if len(primary_trees) == 1 else None
        if trees:
            facts.update(launch_set_trees=trees, launch_set_primary_tree=primary_tree, trees_rule=TREES_RULE)
        relaunches_inp = shared.get(RELAUNCHES_FILE)
        relaunches = self._json(relaunches_inp) if relaunches_inp is not None else None
        if relaunches_inp is not None:
            facts["relaunches_record"] = LC.rel_home(relaunches_inp.uri.rsplit("#", 1)[0])
        three = self._json(tn_inp) if tn_inp is not None else None
        roster_set = self._roster_set(shared)
        pins_set = self._pins_set(shared, roster_set)
        pins_ok = pins_set.usable(mk)
        three_ok = isinstance(three, dict) and str(three.get("schema")) == THREE_NUMBERS_SCHEMA and isinstance(three.get("rows"), dict) and str(three.get("eval")) == spec.eval_id
        # THREE_NUMBERS_SOURCE_RULE: the record's top-level `source` names the source its rows belong to; the root binds this reader's, and the two must agree before any
        # row files -- a record spelling another source under this root is a wrong root or a wrong mirror (a landing-1 record under a landing-2 root, or the reverse)
        three_source = (str(three.get("source")) if three.get("source") is not None else None) if three_ok else None
        three_source_problem = (f"{RECORDS_DIR}/{spec.three_numbers_file}: the three-numbers record spells source {three_source!r}; this reader is {spec.source!r} -- wrong root or "
                                f"wrong mirror ({THREE_NUMBERS_SOURCE_RULE})") if three_ok and three_source != spec.source else None
        if spec.shared is not None and spec.shared.harness_variant is not None:
            # recorded on the landing-2 readers' status alone (landing 1's status stays byte for byte main's; the refusal below names the header wherever it fires)
            facts["three_numbers_source"] = three_source
        roster_ok = roster_set.usable
        for name, inp in (("launch_set", ls_inp), ("launched_pins", pins_inp), ("launched_roster", roster_inp), ("three_numbers", tn_inp)):
            facts[name] = LC.rel_home(inp.uri.rsplit("#", 1)[0]) if inp is not None else None
        if ls_problem is not None:
            facts["launch_set_problem"] = ls_problem
        if roster_set.from_index:
            facts["roster_set"] = roster_set.summary()
            facts["roster_set_rule"] = ROSTER_SET_RULE
        if pins_set.by_tree or pins_set.by_name:
            facts["launched_pins_per_tree"] = sorted({f for _d, _i, f in pins_set.by_tree.values()} | set(pins_set.by_name))
        # the item set (ITEM_SET_RULE): a FULL reader's count is the mirror's declared and pinned set, checked against the registry's declaration; a 200-item
        # reader's is its spec's constant
        item_set: Optional[ItemSetPin] = self._item_set_pin(shared) if spec.full_item_set else None
        n_items = item_set.n if (item_set is not None and item_set.problem is None) else spec.n_items
        if item_set is not None:
            facts["item_set"] = item_set.summary()
            facts["item_set_rule"] = ITEM_SET_RULE
        # the rows the group is expected to hold: every launch-set row at the group's two selections
        expected: Dict[Tuple[str, str], Dict[str, Any]] = {}
        ambiguous: Dict[Tuple[str, str], int] = {}
        for r in launch_rows or []:
            key = self._launch_row_key(r, mk, tags)
            if key is not None:
                if key in expected and spec.shared is not None:
                    ambiguous[key] = ambiguous.get(key, 1) + 1       # two shared rows on one key (a repeat row without its segment): an ambiguous declaration (SHARED_ROW_KEY_RULE)
                expected[key] = r
        if unsteered_tag is not None:
            # the unsteered row (UNSTEERED_ROW_RULE): its `none` RE-SERVED at another serving window is a row of its own, keyed `none@w<N>`, declared by the launch set's
            # families.<model key>.unsteered.window_reserves.<eval> (one launch-set row per (tag, key) cannot hold a second none); the default window is the family's `*`
            facts.update(unsteered_row_tag=unsteered_tag, unsteered_row_rule=UNSTEERED_ROW_RULE, window_anchor_rule=WINDOW_ANCHOR_RULE,
                         unsteered_default_window=self._unsteered_default_window(ls_doc, mk))
            for key, lrow_w in self._window_reserve_rows(ls_doc, ls_inp, mk, unsteered_tag, seed_n, expected).items():
                expected.setdefault(key, lrow_w)
        for key, n in sorted(ambiguous.items()):
            self.refusals.add(f"{key[0]}|{key[1]}", f"the launch set names {n} rows for {_sel_of(tags, key[0])}/{key[1]} (their launched_as.suffix spell the same row key); a row is keyed by "
                                                    f"(tag, object key[@rep<N>]) with the repeat segment read off launched_as.suffix, so two rows on one key bind no store ({SHARED_ROW_KEY_RULE})",
                              ls_inp.uri if ls_inp is not None else self.root)
            expected.pop(key, None)
        if spec.shared is not None:
            # every `_shared` spelling a row of the group files under (a store's or a launch row's tag) must resolve to the group's model (SHARED_SPELLING_RULE)
            for t in sorted({t for (t, _o, _s) in stores} | {t for (t, _o) in expected}):
                mv_s = seeds.model_for_view_legacy(spec.spelling_of(t))
                if mv_s.get("model_id") != model_id:
                    self._refuse_group(group, facts, f"legacy_spellings.yaml spells the shared row spelling {spec.spelling_of(t)!r} as model {mv_s.get('model_id')!r} ({mv_s.get('source')}); the group's "
                                                    f"model is {model_id}: a shared spelling needs its own row in legacy_spellings.yaml model_keys naming the solo spelling's model, fit model "
                                                    f"and transfer kind ({SHARED_SPELLING_RULE})", self.root)
                    yield batch
                    return
        gate_docs = self._gate_docs(gates)
        filed: Dict[Tuple[str, str], Dict[str, Any]] = {}
        seen: Set[Tuple[str, str]] = set()
        # the instruct-direct tag (INSTRUCT_DIRECT_RULE): its rules ride the status only when the group carries one of its rows or stores
        if instruct_tag is not None and (any(t == instruct_tag for (t, _o) in expected) or any(t == instruct_tag for (t, _o, _s) in stores)):
            facts.update(instruct_direct_tag=instruct_tag, instruct_direct_rule=INSTRUCT_DIRECT_RULE, instruct_anchor_rule=INSTRUCT_ANCHOR_RULE)
        # a key the tag's OWN row carries: the `main` canary row's store of the same key is superseded (LEGACY_MAIN_ALIAS_RULE), never a second run of it
        own_row_keys = {(t, o) for (t, o, _sfx), st_ in stores.items() if st_.get("legacy_tag") is None}
        # a SHARED reader (SHARED_STANDUP_RULE): one standup id per row launch -- the stores a launch names on one suffix must agree before any of them files
        standup_conflicts = self._shared_standup_conflicts(stores, facts) if spec.shared is not None else {}
        for (tag, okey, suffix), st in sorted(stores.items()):
            legacy_tag = st.get("legacy_tag")
            # the row key: (tag, object key), or (tag, object key@rep<N>) for a shared repeat store (SHARED_ROW_KEY_RULE; None on a solo reader)
            repeat = repeat_of_suffix(spec, suffix)
            rk_okey = row_okey(okey, repeat, window_of_suffix(spec, suffix))
            if tag not in tags.values():
                # a store under a tag the group admits for no selection: an instruct-direct row on a reader / family without the tag (INSTRUCT_DIRECT_RULE)
                self.refusals.add(f"{tag}|{rk_okey}", f"store {rel_store(st['row_dir'], okey, suffix)} sits under tag {tag!r}, which this source admits for no selection of the group "
                                                      f"{group} ({', '.join(f'{s} -> {t}' for s, t in sorted(tags.items()))}); an instruct-direct store files only through a full reader "
                                                      f"for a family that runs the tag ({INSTRUCT_DIRECT_RULE})", st["dir"])
                continue
            # a RELAUNCHED row (RELAUNCH_RULE): the parked first store (relaunched_from.suffix) is not the row's store -- parked, never read, never refused
            lrow0 = expected.get((tag, rk_okey))
            if isinstance(lrow0, Mapping) and isinstance(lrow0.get("launched_as"), Mapping) and isinstance(lrow0.get("relaunched_from"), Mapping):
                rf = lrow0["relaunched_from"]
                la_suffix = str(lrow0["launched_as"].get("suffix") or "")
                # a rerun after SEVERAL parked attempts names the first on relaunched_from.suffix and all of them on the additive
                # relaunched_from.suffixes (relaunches.json's parked_suffix list); every named attempt that IS mirrored parks, none refuses
                rf_list = [str(x) for x in rf.get("suffixes")] if isinstance(rf.get("suffixes"), list) else []
                rf_suffixes = {str(rf.get("suffix") or "")} | set(rf_list)
                if suffix in rf_suffixes and suffix == la_suffix:
                    seen.add((tag, rk_okey))
                    self.refusals.add(f"{tag}|{rk_okey}", f"the launch set's row names store {suffix!r} on launched_as AND parks it on relaunched_from: a contradictory declaration; the "
                                                          f"store is not read until one of them is withdrawn ({LAUNCHED_AS_TREE_RULE})", ls_inp.uri if ls_inp is not None else self.root)
                    continue
                if suffix in rf_suffixes and suffix != la_suffix:
                    # (the parked list is minted only when a store parks: a group without relaunches keeps its status extra byte for byte)
                    facts.setdefault("parked", []).append({"tag": tag, "object_key": okey, "suffix": suffix, "store": rel_store(st["row_dir"], okey, suffix), "relaunched_to": la_suffix,
                                                           "why": rf.get("why"), "rule": RELAUNCH_RULE, **({"parked_suffixes": rf_list} if rf_list else {})})
                    continue
            la0 = lrow0.get("launched_as") if isinstance(lrow0, Mapping) else None
            if isinstance(la0, Mapping) and str(la0.get("suffix") or "") not in ("", suffix) and not any("outcomes" in files for files in st["keys"].values()):
                # a ONE-TASK store of the OTHER task (ONE_TASK_STORE_RULE): the row's launched_as names its store for this task, and this one holds none of the task's files
                facts.setdefault("parked", []).append({"tag": tag, "object_key": okey, "suffix": suffix, "store": rel_store(st["row_dir"], okey, suffix), "launched_as_suffix": la0.get("suffix"),
                                                       "why": f"the row's launched_as names {la0.get('suffix')} for this task and this store holds no {spec.task} files: the other task's one-task store",
                                                       "rule": ONE_TASK_STORE_RULE})
                continue
            if legacy_tag is not None and (tag, okey) in own_row_keys:
                self.refusals.add(f"{legacy_tag}|{okey}", f"store {rel_store(st['row_dir'], okey, suffix)} sits under the `{MAIN_SELECTION}` canary row {st['row_dir']} (tag {legacy_tag}) but the tag's "
                                                          f"own row {row_dir_of(tag)} carries a store for {okey!r} too: the canary row's store is superseded and not read (the tag's own row wins; "
                                                          "D565), never a second run of the key", st["dir"])
                continue
            seen.add((tag, rk_okey))
            rkey = f"{tag}|{rk_okey}"
            if (tag, rk_okey) in ambiguous:
                continue        # the launch set's ambiguous declaration refused the key above (SHARED_ROW_KEY_RULE); the store binds nothing
            if (st["row_dir"], suffix) in standup_conflicts:
                self.refusals.add(rkey, standup_conflicts[(st["row_dir"], suffix)], st["dir"])
                continue
            if ls_problem is not None:
                self.refusals.add(rkey, ls_problem, self._records_dir())
                continue
            if launch_rows is None:
                self.refusals.add(rkey, f"{RECORDS_DIR}/{' / '.join(spec.launch_set_files)} ({LAUNCH_SET_SCHEMA}, evals.{spec.launch_set_eval_key}.launch) is absent or malformed beside the stores; no pin key and no expected object sha can be named", self.root)
                continue
            if not pins_ok:
                self.refusals.add(rkey, f"{RECORDS_DIR}/{PINS_FILE} (and every per-tree pins copy) is absent or malformed beside the stores; the launched pins cannot be checked", self.root)
                continue
            if not roster_ok:
                self.refusals.add(rkey, roster_set.index_problem or f"{RECORDS_DIR}/{spec.roster_file} is absent or malformed beside the stores; the launched roster entry cannot be checked", self.root)
                continue
            if not three_ok:
                self.refusals.add(rkey, f"{RECORDS_DIR}/{spec.three_numbers_file} ({THREE_NUMBERS_SCHEMA}, eval {spec.eval_id}) is absent or malformed beside the stores; nothing can be cross-checked", self.root)
                continue
            if three_source_problem is not None:
                # THREE_NUMBERS_SOURCE_RULE: the record names another source -- a wrong root or a wrong mirror; nothing files (the file, the field `source`, the two names)
                self.refusals.add(rkey, three_source_problem, tn_inp.uri.rsplit("#", 1)[0] if tn_inp is not None else self._records_dir())
                continue
            if item_set is not None and item_set.problem is not None:
                # the full-suite item set cannot be attested for ANY store of the group (ITEM_SET_RULE): every store refuses by name with the one reason
                self.refusals.add(rkey, item_set.problem, item_set.where)
                continue
            lrow = expected.get((tag, rk_okey))
            if lrow is None:
                self.refusals.add(rkey, f"the launch set names no row for {_sel_of(tags, tag)}/{rk_okey}; a store the launch set does not declare has no identity to bind to", ls_inp.uri)
                continue
            try:
                filed[(tag, rk_okey)] = self._emit_row(ctx, group, tag, seed_n, _sel_of(tags, tag), model_id, inst_row, okey, suffix, st, lrow, roster_set, pins_set, three,
                                                       gate_docs, ls_inp, tn_inp, facts, legacy_tag=legacy_tag, n_items=n_items, item_set=item_set, primary_tree=primary_tree,
                                                       relaunches=relaunches, relaunches_inp=relaunches_inp, repeat=repeat)
            except _PairRefusal as exc:
                self.refusals.add(rkey, exc.why, exc.where)
        # STORES THE READER DID NOT READ (LAUNCHED_AS_TREE_RULE): a store directory whose suffix is outside the grammar refuses by name (never a silent drop), and a
        # store the launch row names on launched_as that exists under the root but was not read -- for whatever reason -- lands in the refusals by name too
        for (row_dir_u, okey_u, suffix_u), dir_u in sorted(getattr(self, "_unread_stores", {}).items()):
            tag_u = tag_of_row_dir(row_dir_u, spec)
            if tag_u is None or tag_u not in tags.values():
                continue
            seen.add((tag_u, okey_u))      # a store directory IS present for the row: a refusal, never "not run"
            la_u = (expected.get((tag_u, okey_u)) or {}).get("launched_as") if isinstance(expected.get((tag_u, okey_u)), Mapping) else None
            named = isinstance(la_u, Mapping) and str(la_u.get("suffix") or "") == suffix_u
            self.refusals.add(f"{tag_u}|{okey_u}", f"store {rel_store(row_dir_u, okey_u, suffix_u)}{' (the store the launch row names on launched_as)' if named else ''} has a suffix outside the "
                                                    f"grammar ({spec.grammar_text}) and cannot be bound to a row ({LAUNCHED_AS_TREE_RULE})", dir_u)
        for (tag, rk_okey), lrow in sorted(expected.items()):
            okey = split_row_okey(rk_okey)[0]
            la_x = lrow.get("launched_as") if isinstance(lrow.get("launched_as"), Mapping) else None
            la_sfx = str(la_x.get("suffix") or "") if la_x is not None else ""
            if not la_sfx or (tag, rk_okey) in seen:
                continue
            la_row_dir = str(la_x.get("row_dir") or "") or row_dir_of(tag)
            la_tag = tag_of_row_dir(la_row_dir, spec)
            if la_tag is not None and (la_tag, rk_okey) in seen:
                continue        # the named store was read as ANOTHER row's (the `_it` none carried BY ALIAS to the s1 none: launched_as names that store; INSTRUCT_ANCHOR_RULE)
            la_dir = os.path.join(self._stores_root(), la_row_dir, okey, THINKING_DIR, la_sfx)
            if not os.path.isdir(la_dir):
                continue        # absent: NOT RUN (the not-run list names the launched_as store below)
            seen.add((tag, rk_okey))
            if (la_row_dir, okey, la_sfx) in getattr(self, "_unread_stores", {}):
                continue        # already refused above as a store directory outside the grammar (one refusal per store)
            if not spec.suffix_re.match(la_sfx):
                why = f"its suffix is outside the grammar ({spec.grammar_text})"
            elif suffix_task_for(spec, la_sfx) not in (None, spec.eval_id):
                why = f"its suffix is tagged for the task {suffix_task_for(spec, la_sfx)!r}, not this source's ({spec.eval_id})"
            elif tag_of_row_dir(la_row_dir, spec) != tag:
                why = f"launched_as.row_dir {la_row_dir!r} is not the row's ({row_dir_of(tag)})"
            else:
                why = "no store files were discovered under it"
            self.refusals.add(f"{tag}|{rk_okey}", f"the launch row names store {rel_store(la_row_dir, okey, la_sfx)} on launched_as; it exists under the root but this reader did not read it: {why} "
                                                  f"({LAUNCHED_AS_TREE_RULE})", la_dir)
        # GATE RECORDS WITHOUT A STORE beside a row whose store IS present (the parked attempts' records -- FAIL, or PASS but
        # superseded -- ride under records/ while their stores are not mirrored; the launch row's launched_as is the authority for the row's store): neither the row's
        # store nor a refusal; RECORDED here so the mirror's extra records are visible. A not-run row's record is the not-run list's, never this one's
        present3 = {(st_["row_dir"], o, sfx) for (_t, o, sfx), st_ in stores.items()}
        parked_named = set()
        for e in ((relaunches or {}).values() if isinstance(relaunches, Mapping) else ()):
            if isinstance(e, Mapping):
                ps = e.get("parked_suffix")
                parked_named |= {str(x) for x in ps} if isinstance(ps, list) else ({str(ps)} if isinstance(ps, str) else set())
        for key3 in sorted(k for k in gate_docs if len(k) == 3 and k not in present3):
            row_dir_, okey_, sfx_ = key3
            tag_ = tag_of_row_dir(row_dir_, spec)
            if tag_ is None or (tag_, okey_) not in seen:
                continue
            gdoc, ginp = gate_docs[key3]
            lrow_ = expected.get((tag_, okey_)) or {}
            la_ = lrow_.get("launched_as") if isinstance(lrow_.get("launched_as"), Mapping) else None
            rf_ = lrow_.get("relaunched_from") if isinstance(lrow_.get("relaunched_from"), Mapping) else {}
            rf_named = {str(rf_.get("suffix") or "")} | ({str(x) for x in rf_.get("suffixes")} if isinstance(rf_.get("suffixes"), list) else set())
            facts.setdefault("gate_records_without_store", []).append({"tag": tag_, "object_key": okey_, "row_dir": row_dir_, "suffix": sfx_, "verdict": gdoc.get("verdict"),
                                                                        "record": LC.rel_home(ginp.uri.rsplit("#", 1)[0]), "launched_as_suffix": (la_.get("suffix") if la_ else None),
                                                                        "named_by_relaunched_from": sfx_ in rf_named, "parked_by_relaunches": sfx_ in parked_named,
                                                                        "why": "the record's store is not mirrored and the launch row's launched_as names the row's store: neither the row's store nor a refusal",
                                                                        "rule": RELAUNCH_RULE})
        # NOT RUN: a launch-set row with no store files nothing (NOT_RUN_RULE); the gate record's verdict is carried in the facts
        for (tag, rk_okey), lrow in sorted(expected.items()):
            if (tag, rk_okey) in seen:
                continue
            okey = split_row_okey(rk_okey)[0]
            if spec.shared is None and tag == instruct_tag and LC.cell_grammar_legacy(okey)["object_kind"] == "none":
                # the `_it` none is carried BY ALIAS to the group's none (INSTRUCT_ANCHOR_RULE): no store, no run, never "not run" -- the alias is recorded
                seed_none = filed.get((group, "none"))
                facts.setdefault("anchor_aliases", []).append({"tuple_id": lrow.get("tuple_id"), "tag": tag, "object_key": okey, "class": lrow.get("class"),
                                                               "alias_of": f"{group}/{okey}", "anchor_run_id": (seed_none["run_id"] if seed_none is not None else None),
                                                               "anchor_state": (ANCHOR_CONTRASTED if seed_none is not None else (ANCHOR_REFUSED if (group, "none") in seen else ANCHOR_NOT_YET_RUN)),
                                                               "declared_class_expected": "anchor", "declared_class_agrees": (str(lrow.get("class") or "") == "anchor"),
                                                               "launch_set_row": {k: lrow.get(k) for k in ("census_status", "launched_as", "anchor_tuple", "block", "seed_base") if k in lrow},
                                                               "rule": INSTRUCT_ANCHOR_RULE})
                continue
            gd = gate_docs.get((row_dir_of(tag), okey))
            if gd is None and mkey.legacy_main_alias and tag == group:
                gd = gate_docs.get((f"{mk}-f22", okey))      # a not-run row may carry its FAIL record under the `main` canary row (LEGACY_MAIN_ALIAS_RULE)
            la_n = lrow.get("launched_as") if isinstance(lrow.get("launched_as"), Mapping) else None
            facts["not_run"].append({"tuple_id": lrow.get("tuple_id"), "tag": tag, "object_key": okey, "paper_label": lrow.get("paper_label"), "class": lrow.get("class"),
                                     "gate_verdict": (gd[0].get("verdict") if gd else None), "gate_problems": (list(gd[0].get("problems") or []) if gd else None),
                                     "gate_record": (LC.rel_home(gd[1].uri.rsplit("#", 1)[0]) if gd else None), "rule": NOT_RUN_RULE,
                                     # a row that names its store on launched_as (LAUNCHED_AS_TREE_RULE): the store is absent from the root
                                     **({"launched_as_suffix": la_n.get("suffix"), "launched_as_sha": la_n.get("sha"), "store_dir_present": False} if la_n is not None else {})})
        # the anchor blocks first (they ride the run payloads), then every row's records, then the contrasts — a record is built once, never re-stamped.
        # Every contrast is DECIDED by the pass's anchor currency (CURRENCY_RULE; the frame loads once per pass through ctx.anchor_currency() and is
        # asked on the first comparison; a group without one never asks): the parks land on the VALUE rows' extra.delta_block before the records are added
        self._reanchor_computations = {}
        facts.update(contrasts_filed=0, contrasts_own=0, contrasts_reanchored=0, contrasts_parked=0, anchor_states={})
        contrasts = self._emit_contrasts(ctx, group, filed, expected, seen, facts, n_items)
        if spec.shared is not None:
            # the shared - solo comparison (SHARED_COMPARISON_RULE): a registry-computed run_contrast per shared row against its solo run, anchor_kind solo_stand_in
            contrasts += self._emit_solo_contrasts(ctx, group, filed, facts, n_items)
            it_none_ids = {st_["run_id"] for st_ in filed.values() if st_.get("comparison_kind") == COMPARISON_KIND_IT_NONE}
            # (the ONE contrast the instruct row's own none anchors: its own repeat floor, a same-object pair -- SHARED_IT_REPEAT_RULE; anything else is the hazard)
            if any(c.get("anchor_run_id") in it_none_ids and not (c.get("extra") or {}).get("repeat_floor_of_own_none") for c in contrasts):
                raise AdapterRefusal(self.root, f"an instruct row's own none stands as a contrast anchor; it is a comparison row only ({SHARED_IT_NONE_RULE})")
        for key in sorted(filed):
            for rt, payload in filed[key]["records"]:
                batch.add(rt, payload, ctx.producer, ctx.now)
        for comp_row in self._reanchor_computations.values():
            batch.add("computation", comp_row, ctx.producer, ctx.now)      # the registry's statistic of a re-anchored contrast (extra.statistic_computation_id names it)
        for payload in contrasts:
            batch.add("run_contrast", payload, ctx.producer, ctx.now)
        facts["anchor_states"] = dict(sorted(facts["anchor_states"].items()))
        facts["reanchor_computations"] = len(self._reanchor_computations)
        self._finish(group, facts)
        yield batch

    def _roster_set(self, shared: Mapping[str, Input]) -> RosterSet:
        """The mirrored launched rosters by sha256 (:data:`ROSTER_SET_RULE`): ``records/rosters.json``'s entries (each file verified against its key) plus the
        plain file; with no index, the plain file alone."""
        plain_inp = shared.get(self.SPEC.roster_file)
        plain_doc = self._yaml(plain_inp) if plain_inp is not None else None
        idx_inp = shared.get(ROSTERS_INDEX_FILE)
        entries: Dict[str, RosterChoice] = {}
        index_problem: Optional[str] = None
        if idx_inp is not None:
            try:
                idx = self._json(idx_inp)
            except (OSError, ValueError):
                idx = None
            rows: Optional[Mapping[str, Any]] = None
            if isinstance(idx, Mapping):
                if isinstance(idx.get("rosters"), Mapping):
                    rows = idx["rosters"]
                elif isinstance(idx.get("rosters"), list):
                    # the producer's shape (every mirror since 2026-09-24 00:22Z): a LIST of {roster_sha256, file, tree_sha, pins_file, pins_sha256, stores}; the same facts, keyed here
                    rows = {str(e.get("roster_sha256")): e for e in idx["rosters"] if isinstance(e, Mapping) and _SHA256.match(str(e.get("roster_sha256") or ""))}
                else:
                    rows = {k: v for k, v in idx.items() if _SHA256.match(str(k))}
            if not rows:
                index_problem = f"{RECORDS_DIR}/{ROSTERS_INDEX_FILE} names no roster (expected {{<roster sha256>: {{file, tree_sha, stores}}}}); no store's roster can be matched"
            else:
                for sha, meta in rows.items():
                    sha = str(sha)
                    meta = meta if isinstance(meta, Mapping) else ({"file": meta} if isinstance(meta, str) else {})
                    fname = os.path.basename(str(meta.get("file") or ""))
                    inp = shared.get(fname) if fname else None
                    doc = None
                    if not fname:
                        problem: Optional[str] = f"{ROSTERS_INDEX_FILE} names no file for roster {sha[:12]}"
                    elif inp is None:
                        problem = f"{ROSTERS_INDEX_FILE} names {fname} for roster {sha[:12]}, which is not mirrored under {RECORDS_DIR}/"
                    elif inp.sha256 != sha:
                        problem = f"{ROSTERS_INDEX_FILE} names {fname} for roster {sha[:12]}, but the file's sha256 is {inp.sha256[:12]}"
                    else:
                        doc = self._yaml(inp)
                        problem = None if (isinstance(doc, dict) and isinstance(doc.get("waves"), Mapping)) else f"{fname} is not a launched roster (no waves block)"
                    pins_name = meta.get("pins") or meta.get("pins_file")
                    entries[sha] = RosterChoice(doc, inp, fname, (str(meta.get("tree_sha")) if meta.get("tree_sha") else None), True, problem,
                                                pins_file=(str(pins_name) if pins_name else None), stores=(list(meta.get("stores")) if isinstance(meta.get("stores"), list) else None),
                                                pins_sha256=(str(meta.get("pins_sha256")) if meta.get("pins_sha256") else None))
        plain = None
        if plain_inp is not None:
            plain = entries.get(plain_inp.sha256) or RosterChoice(plain_doc, plain_inp, self.SPEC.roster_file, None, idx_inp is not None, None)
            entries.setdefault(plain_inp.sha256, plain)
        return RosterSet(entries, plain, idx_inp is not None, idx_inp, index_problem)

    def _pins_set(self, shared: Mapping[str, Input], roster_set: RosterSet) -> PinsSet:
        plain_inp = shared.get(PINS_FILE)
        plain_doc = self._json(plain_inp) if plain_inp is not None else None
        plain = (plain_doc, plain_inp) if isinstance(plain_doc, dict) else None
        by_tree: Dict[str, Tuple[Dict[str, Any], Input, str]] = {}
        by_name: Dict[str, Tuple[Dict[str, Any], Input]] = {}
        for name, inp in shared.items():
            m = PINS_VARIANT_RE.match(name)
            if m:
                doc = self._json(inp)
                if isinstance(doc, dict):
                    by_tree[m.group("sha")] = (doc, inp, name)
                    by_name[name] = (doc, inp)
        for e in roster_set.entries.values():
            if e.pins_file and os.path.basename(e.pins_file) in shared and os.path.basename(e.pins_file) not in by_name:
                doc = self._json(shared[os.path.basename(e.pins_file)])
                if isinstance(doc, dict):
                    by_name[os.path.basename(e.pins_file)] = (doc, shared[os.path.basename(e.pins_file)])
        return PinsSet(plain, by_tree, by_name)

    def _launch_row_key(self, r: Mapping[str, Any], mk: str, tags: Mapping[str, str]) -> Optional[Tuple[str, str]]:
        """The ``(tag, object key)`` a launch-set row names in a group whose selections are ``tags`` (selection -> tag): the row's selection when it is one
        of the group's; the family's ``s1`` tag when the row spells a ``main`` canary row's key (``view`` ``main``, an aliased family); else None.
        A launch set may name SEVERAL families' rows under the same selection (the appendix run: gemma / llama / nemotron all at ``s1``), so a row whose
        ``tag`` spells another model key is never this group's, whatever its selection. On a SHARED reader the selection is the row's ``selection`` (its
        ``view`` is the registry spelling, checked in :meth:`_emit_row`) and a repeat row's key carries its segment (:data:`SHARED_ROW_KEY_RULE`)."""
        okey = r.get("cell")
        if not okey:
            return None
        row_tag = str(r.get("tag") or "")
        if row_tag and model_key_of(row_tag) != mk:
            return None
        if self.SPEC.shared is not None:
            sel = str(r.get("selection") or "")
            if sel not in tags:
                return None
            la = r.get("launched_as") if isinstance(r.get("launched_as"), Mapping) else {}
            return (tags[sel], row_okey(str(okey), repeat_of_suffix(self.SPEC, str(la.get("suffix") or ""))))
        sel = str(r.get("view") or "")
        if sel in tags:
            return (tags[sel], str(okey))
        alias = f"{mk}_s{LEGACY_MAIN_ALIAS_SEED}"
        if sel == MAIN_SELECTION and MODEL_KEYS[mk].legacy_main_alias and alias in tags.values():
            return (alias, str(okey))
        return None

    @staticmethod
    def _yaml(inp: Optional[Input]) -> Optional[Dict[str, Any]]:
        if inp is None:
            return None
        import yaml
        with open(inp.path, "r", encoding="utf-8") as fh:
            d = yaml.safe_load(fh)
        return d if isinstance(d, dict) else None

    def _sort_inputs(self, inputs: Sequence[Input]) -> Tuple[Dict[Tuple[str, str, str], Dict[str, Any]], Dict[str, Input], List[Input]]:
        """The group's inputs sorted into stores — ``(tag, object key, suffix) -> {"top": {name: Input}, "keys": {store key: {name: Input}}, "dir",
        "row_dir", ["summary"]}`` — the shared records (``#<group>`` aliases, by name) and the gate records (aliases whose name spells ``gate_``). Store
        directories whose suffix the grammar does not read are kept on ``self._unread_stores`` (``(row dir, object key, suffix) -> dir``) for :meth:`read` to
        refuse by name (:data:`LAUNCHED_AS_TREE_RULE`); another task's one-task store is not this source's and is dropped as before."""
        stores: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        shared: Dict[str, Input] = {}
        gates: List[Input] = []
        self._unread_stores: Dict[Tuple[str, str, str], str] = {}
        self._solo_inputs = {}
        for inp in inputs:
            base = os.path.basename(inp.uri)
            if "#" in base:
                real = inp.uri.rsplit("#", 1)[0]
                name = os.path.basename(real)
                rel = os.path.relpath(real, self.root).split(os.sep)
                if rel[0] == os.pardir:
                    self._solo_inputs[real] = inp      # a SHARED reader's solo-side input (outside the root; SHARED_COMPARISON_RULE), by its real path
                elif len(rel) == 2 and rel[0] == RECORDS_DIR:
                    if is_shared_record_name(name, self.shared_files(), self.SPEC.roster_variant_re):
                        shared[name] = inp
                    elif gate_name_parts(name) is not None and suffix_task_for(self.SPEC, gate_name_parts(name)["suffix"]) in (None, self.SPEC.eval_id):
                        gates.append(inp)
                elif len(rel) == 2 and rel[0] in RELAUNCHES_DIRS[1:] and name == RELAUNCHES_FILE:
                    shared.setdefault(name, inp)       # records/ wins when both are mirrored
                continue
            rel = os.path.relpath(inp.uri, self.root).split(os.sep)
            if len(rel) < 6 or rel[0] != STORES_HEAD or rel[3] != THINKING_DIR:
                continue
            row_dir, okey, suffix = rel[1], rel[2], rel[4]
            tag = tag_of_row_dir(row_dir, self.SPEC)
            if tag is None or suffix_task_for(self.SPEC, suffix) not in (None, self.SPEC.eval_id):
                continue
            if not self.SPEC.suffix_re.match(suffix):
                self._unread_stores.setdefault((row_dir, okey, suffix), os.path.join(self.root, *rel[:5]))
                continue
            st = stores.setdefault((tag, okey, suffix), {"top": {}, "keys": {}, "dir": os.path.join(self.root, *rel[:5]), "row_dir": row_dir,
                                                         "legacy_tag": legacy_main_tag_of_row_dir(row_dir)})
            body = rel[5:]
            if len(body) == 1:
                st["top"][body[0]] = inp
            elif len(body) == 2 and body[0] == okey and body[1] == CONTROLS_SUMMARY_FILE:
                st["summary"] = inp
            elif len(body) >= 3 and body[0] == "cells":
                files = st["keys"].setdefault(body[1], {})
                if len(body) == 3:
                    files[body[2]] = inp
                elif len(body) == 4 and body[2] in TASK_SUBDIRS and body[3] == self.SPEC.outcomes_file:
                    files[body[2]] = inp
        return stores, shared, gates

    def _gate_docs(self, gates: Sequence[Input]) -> Dict[Tuple[str, ...], Tuple[Dict[str, Any], Input]]:
        """The gate records by ``(row dir, object key, suffix)`` AND by ``(row dir, object key)`` as their CONTENT spells them (``row`` / ``cell`` / ``suffix``); a
        record whose content is unreadable is skipped (the store it would gate refuses by name for want of a record). A present store is gated by the record of
        ITS suffix (a parked attempt's FAIL record beside the live store's PASS can never shadow it); the two-key entry, which the not-run rows read, prefers a PASS."""
        out: Dict[Tuple[str, ...], Tuple[Dict[str, Any], Input]] = {}
        for inp in gates:
            try:
                d = self._json(inp)
            except (OSError, ValueError):
                continue
            if not isinstance(d, Mapping) or not d.get("row") or not d.get("cell"):
                continue
            key = (str(d["row"]), str(d["cell"]))
            if d.get("suffix"):
                out[key + (str(d["suffix"]),)] = (dict(d), inp)
            prev = out.get(key)
            if prev is None or (str(prev[0].get("verdict") or "") != GATE_PASS and str(d.get("verdict") or "") == GATE_PASS):
                out[key] = (dict(d), inp)
        return out

    @staticmethod
    def _unsteered_default_window(ls_doc: Optional[Mapping[str, Any]], mk: str) -> Optional[int]:
        """The family's default serving window (``families.<model key>.unsteered.windows['*'].max_model_len``), the window the plain ``u0/none`` row was served at."""
        fam = ((ls_doc.get("families") or {}).get(mk) or {}) if isinstance(ls_doc, Mapping) and isinstance(ls_doc.get("families"), Mapping) else {}
        uns = fam.get("unsteered") if isinstance(fam, Mapping) and isinstance(fam.get("unsteered"), Mapping) else {}
        star = (uns.get("windows") or {}).get("*") if isinstance(uns.get("windows"), Mapping) else None
        return LC.int_or_none(star.get("max_model_len")) if isinstance(star, Mapping) else None

    def _window_reserve_rows(self, ls_doc: Optional[Mapping[str, Any]], ls_inp: Optional[Input], mk: str, u_tag: str, seed_n: int,
                             expected: Mapping[Tuple[str, str], Mapping[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """The launch-row-shaped declarations of the u0 ``none`` RE-SERVED at another serving window (:data:`UNSTEERED_ROW_RULE`): one per entry of
        ``families.<model key>.unsteered.window_reserves.<eval>`` (``{store, suffix, tree, tag, window {max_model_len, max_num_seqs}, tool_calls, run_id, ...}``), keyed
        ``(u0 tag, none@w<N>)`` with the plain ``u0/none`` row's item set, class ``anchor``, the launch tag's ``launched_as``; a declaration whose suffix does not spell its
        window refuses the key by name."""
        out: Dict[Tuple[str, str], Dict[str, Any]] = {}
        base_row = expected.get((u_tag, "none")) or {}
        for name, e, suffix, n, seg, where in self._window_reserve_declarations(ls_doc, mk):
            win = e.get("window") if isinstance(e.get("window"), Mapping) else {}
            if not suffix or n is None or seg != n:
                self.refusals.add(f"{u_tag}|none@{WINDOW_MARKER}{n if n is not None else '?'}", f"{where}[{name}] declares suffix {suffix!r} at window {n!r}; the suffix's -{WINDOW_MARKER}<N> "
                                                                                            f"segment ({seg!r}) must spell the window ({UNSTEERED_ROW_RULE})", ls_inp.uri if ls_inp is not None else self.root)
                continue
            key = (u_tag, row_okey("none", None, n))
            out[key] = {"tuple_id": f"{UNSTEERED_SELECTION}/{key[1]}", "view": UNSTEERED_SELECTION, "cell": "none", "paper_label": "none", "class": "anchor",
                        "block": base_row.get("block", "onboarded"), "fitted": False, "seed_base": seed_n, "pin_key": None, "tag": u_tag, "cell_slug": "none", "table_index": None,
                        "object_sha256_expected": None, "census_status": "run", "bcbw_wave": u_tag, "bcbw_cell": "none", "model_key": mk, "sha": e.get("tree"),
                        "launched_as": {"wave": u_tag, "row_dir": row_dir_of(u_tag), "suffix": suffix, "sha": e.get("tree")}, "anchor_tuple": None,
                        **({"item_set": dict(base_row["item_set"])} if isinstance(base_row.get("item_set"), Mapping) else {}),
                        "window_reserve": {"declared_in": f"{where}[{name}]", "window": dict(win), "tag": e.get("tag"), "tool_calls": e.get("tool_calls"), "run_id_legacy": e.get("run_id"),
                                           "launched_at": e.get("launched_at"), "role": e.get("role"), "rule": UNSTEERED_ROW_RULE}}
        return out

    def _window_reserve_declarations(self, ls_doc: Optional[Mapping[str, Any]], mk: str) -> List[Tuple[str, Mapping[str, Any], str, Optional[int], Optional[int], str]]:
        """The entries of ``families.<model key>.unsteered.window_reserves.<eval>`` as read: ``(name, entry, suffix, declared window, the suffix's window
        segment, where)`` per mapping entry (one entry per eval today, keyed ``"0"``; a mapping of several is admitted too), in name order, with no
        side effect -- :meth:`_window_reserve_rows` turns them into expected rows and refuses the malformed ones by key; :meth:`source_states` lists the
        well-formed ones (suffix spelling its window) as members of the row table."""
        spec = self.SPEC
        fam = ((ls_doc.get("families") or {}).get(mk) or {}) if isinstance(ls_doc, Mapping) and isinstance(ls_doc.get("families"), Mapping) else {}
        uns = fam.get("unsteered") if isinstance(fam, Mapping) and isinstance(fam.get("unsteered"), Mapping) else {}
        wr = (uns.get("window_reserves") or {}).get(spec.launch_set_eval_key) if isinstance(uns.get("window_reserves"), Mapping) else None
        if not isinstance(wr, Mapping):
            return []
        entries = {"0": wr} if "suffix" in wr else {str(k): v for k, v in wr.items()}
        where = f"families.{mk}.unsteered.window_reserves.{spec.launch_set_eval_key}"
        out: List[Tuple[str, Mapping[str, Any], str, Optional[int], Optional[int], str]] = []
        for name, e in sorted(entries.items()):
            if not isinstance(e, Mapping):
                continue
            suffix = str(e.get("suffix") or "")
            win = e.get("window") if isinstance(e.get("window"), Mapping) else {}
            n = LC.int_or_none(win.get("max_model_len"))
            seg = window_of_suffix(spec, suffix) if suffix else None
            out.append((name, e, suffix, n, seg, where))
        return out

    def _refuse_group(self, group: str, facts: Dict[str, Any], why: str, input_uri: str) -> None:
        """A WHOLE-GROUP refusal the status reads (the undeclared-row-dir precedent of 2026-10-04): the reason rides
        ``facts["refuse"]`` -- what the runner's ``ok`` reads (ok = accepted and no refuse entry; ``v_source_status`` folds the source to MIN(ok) = 0
        with the reason under ``extra.refuse``, and the paper's guard reads that key alone) -- AND the by-name listing in ``facts["refusals"]``,
        emitting nothing. A ``refusals`` entry alone is the runner's EXPLAINED empty read (``EXPLANATION_KEYS``), filed ok = 1 with 'no row
        refused': a successful-looking zero-measurement read, which is what the group-wide returns of :meth:`read` (no seed tag, a model key
        outside the source's families, a seed outside the table, legacy_spellings.yaml / models.yaml disagreeing with the model-key table, a
        missing instrument row, a shared spelling resolving elsewhere) used to file."""
        facts.setdefault("refuse", []).append(why)
        self.refusals.add(group, why, input_uri)
        self._finish(group, facts)

    def _finish(self, group: str, facts: Dict[str, Any]) -> None:
        facts["refusals"] = [f"{it['key']} -- {it['why']}" for it in self.refusals.items]
        facts["n_refusals"] = len(self.refusals)
        facts["n_not_run"] = len(facts.get("not_run") or [])
        if "parked" in facts:
            facts["n_parked"] = len(facts["parked"])
        if "endpoint_served_key_mismatch" in facts:
            facts["n_endpoint_served_key_mismatch"] = len(facts["endpoint_served_key_mismatch"])
        self._facts[group] = facts

    # ---- one store -----------------------------------------------------------------------------
    def _store_bundle(self, st: Mapping[str, Any], okey: str, tag: str, suffix: str) -> Dict[str, Any]:
        """The named files of one store directory, checked for presence and shape; raises :class:`_PairRefusal`."""
        spec = self.SPEC
        top: Dict[str, Input] = st["top"]
        missing = [n for n in ("RESULT", LAUNCHED_ENTRY_FILE, GEOMETRY_FILE) if n not in top]
        if missing:
            raise _PairRefusal(f"the store lacks {missing}", st["dir"])
        keys = st["keys"]
        if len(keys) != 1:
            raise _PairRefusal(f"the store holds {len(keys)} store key directories under cells/ (expected exactly one)", st["dir"])
        store_key = next(iter(keys))
        files: Dict[str, Input] = keys[store_key]
        missing = [n for n in ("gen_stats.json", "manifest.json", "rows.jsonl") + TASK_SUBDIRS if n not in files]
        if missing:
            raise _PairRefusal(f"store {store_key} lacks {missing} (the task files are {'/'.join(TASK_SUBDIRS)}/{spec.outcomes_file})", os.path.join(st["dir"], "cells", store_key))
        try:
            ck = IC.parse_cell_key(store_key)
        except AdapterRefusal as exc:
            raise _PairRefusal(str(exc.why), os.path.join(st["dir"], "cells", store_key))
        if ck.cell_id != okey:
            raise _PairRefusal(f"the store key spells object {ck.cell_id!r} but the directory spells {okey!r}", st["dir"])
        with open(top["RESULT"].path, "r", encoding="utf-8") as fh:
            result_lines = [ln.strip() for ln in fh if ln.strip()]
        if not result_lines or RESULT_COMPLETE not in result_lines[-1]:
            raise _PairRefusal(f"RESULT's last line does not read {RESULT_COMPLETE} ({(result_lines[-1] if result_lines else '<empty>')[:160]})", top["RESULT"].uri)
        result = parse_result_line(result_lines[-1])
        if result is None or result.get("result") != "COMPLETE" or result.get("rc") != "0":
            raise _PairRefusal(f"RESULT's last line does not parse as a {RESULT_HEAD} COMPLETE line with rc=0 ({result_lines[-1][:160]})", top["RESULT"].uri)
        sm = spec.suffix_re.match(suffix)
        launch_tag = launch_tag_of(tag, okey, st.get("legacy_tag"))
        if sm is None or sm.group("launch_tag") != launch_tag:
            raise _PairRefusal(f"the store suffix {suffix!r} does not spell this row's launch tag {launch_tag!r}", st["dir"])
        man = self._json(files["manifest.json"]) or {}
        gs_all = self._json(files["gen_stats.json"]) or {}
        gs = gs_all.get(spec.task) if isinstance(gs_all.get(spec.task), Mapping) else None
        if gs is None:
            raise _PairRefusal(f"gen_stats.json carries no {spec.task} block", files["gen_stats.json"].uri)
        outcomes = self._json(files["outcomes"]) or {}
        if str(outcomes.get("task")) != spec.task or not isinstance(outcomes.get("items"), list):
            raise _PairRefusal(f"the outcome record is not {spec.task}'s (task {outcomes.get('task')!r})", files["outcomes"].uri)
        summary = self._json(files["summary"]) or {}
        done = self._json(files["done"]) or {}
        if str(done.get("task")) != spec.task:
            raise _PairRefusal(f"done/{spec.outcomes_file} names task {done.get('task')!r}, not {spec.task!r}", files["done"].uri)
        launched_entry = self._json(top[LAUNCHED_ENTRY_FILE]) or {}
        geometry_doc = self._json(top[GEOMETRY_FILE]) or {}
        scored: Dict[str, Tuple[float, int, Optional[float]]] = {}
        agg_values: Dict[str, Any] = {}
        consts: Dict[str, set] = {}
        for ln, r in IC.iter_jsonl(files["rows.jsonl"].path):
            if str(r.get("task") or "") != spec.task:
                continue
            metric = str(r.get("metric"))
            if r.get("item_id") == "__aggregate__":
                agg_values[metric] = LC.num_or_none(r.get("value"))
                continue
            if metric != HEADLINE_METRIC:
                continue
            iid = str(r.get("item_id"))
            if iid in scored:
                raise _PairRefusal(f"rows.jsonl carries item {iid} twice", files["rows.jsonl"].uri)
            scored[iid] = (float(r.get("value")), ln, LC.num_or_none(r.get("created_at")))
            for k in ("env_digest", "model_id", "model_revision", "harness_config_hash", "intervention_hash", "created_at", "slurm_job_id"):
                v = r.get(k)
                if v is not None:
                    consts.setdefault(k, set()).add(v if k != "created_at" else float(v))
        controls_summary = self._json(st["summary"]) if st.get("summary") is not None else None
        return {"dir": st["dir"], "row_dir": st["row_dir"], "tag": tag, "suffix": suffix, "store_key": store_key, "ck": ck, "top": top, "files": files, "manifest": man,
                "gen_stats": gs, "outcomes": outcomes, "summary": summary, "done": done, "launched_entry": launched_entry, "geometry_doc": geometry_doc,
                "controls_summary": controls_summary, "scored": scored, "agg_values": agg_values, "consts": consts, "result": result, "result_last": result_lines[-1],
                "result_earlier": result_lines[:-1]}

    @staticmethod
    def _outcome_items(bundle: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for it in bundle["outcomes"]["items"]:
            if not isinstance(it, Mapping) or it.get("item_id") in (None, ""):
                raise _PairRefusal("the outcome record carries an item without an id", bundle["files"]["outcomes"].uri)
            iid = str(it["item_id"])
            if iid in out:
                raise _PairRefusal(f"the outcome record carries item {iid} twice", bundle["files"]["outcomes"].uri)
            at_cap, scored = bool(it.get("at_cap")), bool(it.get("scored"))
            v = it.get("value")
            if scored and v not in (0, 1, 0.0, 1.0):
                raise _PairRefusal(f"item {iid}: a scored item with value {v!r}", bundle["files"]["outcomes"].uri)
            if not scored and v is not None:
                raise _PairRefusal(f"item {iid}: an unscored item with value {v!r}", bundle["files"]["outcomes"].uri)
            if at_cap and str(it.get("stop_reason") or "") not in IC.AT_CAP_FINISH_REASONS:
                raise _PairRefusal(f"item {iid}: at_cap with stop_reason {it.get('stop_reason')!r}", bundle["files"]["outcomes"].uri)
            out[iid] = {"at_cap": at_cap, "scored": scored, "value": (float(v) if scored else None), "stop_reason": it.get("stop_reason"),
                        "completion_tokens": LC.int_or_none(it.get("completion_tokens")), "seed": LC.int_or_none(it.get("seed")),
                        "unparseable": it.get("unparseable"), "sample_idx": LC.int_or_none(it.get("sample_idx"))}
        return out

    def _items(self, base: Mapping[str, Any], n_items: int) -> Dict[str, Any]:
        """The ``n_items`` items of the one pass (200 on the pinned readers; the mirrored item set's count on a full reader) from the outcome record x
        ``rows.jsonl`` x the producer's aggregates (``gen_stats.json``, the summary, the outcome header, the completion marker); every disagreement between
        the files is a refusal of the (tag, object key)."""
        spec = self.SPEC
        items = self._outcome_items(base)
        if len(items) != n_items:
            raise _PairRefusal(f"the outcome record holds {len(items)} items, not {n_items}", base["files"]["outcomes"].uri)
        want = {iid: it["value"] for iid, it in items.items() if it["scored"]}
        rows_scored = base["scored"]
        if set(rows_scored) != set(want) or any(rows_scored[i][0] != want[i] for i in want):
            raise _PairRefusal(f"rows.jsonl's {len(rows_scored)} scored acc rows do not reproduce the outcome record's {len(want)} scored items", base["files"]["rows.jsonl"].uri)
        n_scored = len(want)
        correct = int(round(sum(want.values())))
        acc = (correct / n_scored) if n_scored else None
        at_cap = {iid for iid, it in items.items() if it["at_cap"]}
        n_at_cap_unscored = sum(1 for it in items.values() if it["at_cap"] and not it["scored"])
        n_at_cap_scored = len(at_cap) - n_at_cap_unscored
        n_unscored_below_cap = sum(1 for it in items.values() if not it["at_cap"] and not it["scored"])
        n_unparseable = sum(1 for it in items.values() if it.get("unparseable") is True)
        counts = {"n_scored": n_scored, "correct": correct, "acc": acc, "n_unscored": n_items - n_scored, "n_at_cap": len(at_cap), "n_at_cap_unscored": n_at_cap_unscored,
                  "n_at_cap_scored": n_at_cap_scored, "n_unscored_below_cap": n_unscored_below_cap, "n_unparseable": n_unparseable}
        gs = base["gen_stats"]
        self._check_aggregate("gen_stats.json", gs, counts, base["files"]["gen_stats.json"].uri, n_items)
        self._check_aggregate(f"summary/{spec.outcomes_file}", base["summary"], counts, base["files"]["summary"].uri, n_items)
        oc = base["outcomes"]
        got = (LC.int_or_none(oc.get("n_attempted_items")), LC.int_or_none(oc.get("n_scored")), LC.int_or_none(oc.get("n_unscored")), LC.int_or_none(oc.get("n_at_cap")),
               LC.int_or_none(oc.get("max_tokens")))
        if got != (n_items, n_scored, n_items - n_scored, len(at_cap), spec.cap):
            raise _PairRefusal(f"the outcome record's header (n_attempted_items {got[0]}, n_scored {got[1]}, n_unscored {got[2]}, n_at_cap {got[3]}, max_tokens {got[4]}) does not "
                               f"reproduce its items (n {n_items}, n_scored {n_scored}, n_unscored {n_items - n_scored}, n_at_cap {len(at_cap)}, cap {spec.cap})", base["files"]["outcomes"].uri)
        if str(oc.get("thinking") or "") != self.THINKING_SPELLING:
            raise _PairRefusal(f"the outcome record's thinking is {oc.get('thinking')!r}, not {self.THINKING_SPELLING!r}", base["files"]["outcomes"].uri)
        done = base["done"]
        dgot = (LC.int_or_none(done.get("n_items")), LC.int_or_none(done.get("n_rows")), LC.int_or_none(done.get("n_planned")))
        if dgot[0] != n_items or dgot[1] != n_items or dgot[2] not in (None, n_items):
            raise _PairRefusal(f"done/{spec.outcomes_file} (n_items {dgot[0]}, n_rows {dgot[1]}, n_planned {dgot[2]}) is not the {n_items}-item task's completion marker", base["files"]["done"].uri)
        agg_acc, agg_n = base["agg_values"].get("acc_aggregate"), base["agg_values"].get("n")
        if "acc_aggregate" in base["agg_values"] and not ((agg_acc is None and acc is None) or (agg_acc is not None and acc is not None and abs(agg_acc - acc) < 1e-9)):
            raise _PairRefusal(f"rows.jsonl's acc_aggregate {agg_acc} is not correct / scored {acc}", base["files"]["rows.jsonl"].uri)
        if "n" in base["agg_values"] and LC.int_or_none(agg_n) != n_items:
            raise _PairRefusal(f"rows.jsonl's aggregate n {agg_n} is not {n_items}", base["files"]["rows.jsonl"].uri)
        return {"items": items, "counts": counts, "at_cap_ids": at_cap}

    def _check_aggregate(self, what: str, g: Mapping[str, Any], counts: Mapping[str, Any], uri: str, n_items: int) -> None:
        spec = self.SPEC
        got = (LC.int_or_none(g.get("n")), LC.int_or_none(g.get("n_scored")), LC.int_or_none(g.get("n_unscored")), LC.int_or_none(g.get("n_at_cap")), LC.int_or_none(g.get("max_tokens")))
        want = (n_items, counts["n_scored"], counts["n_unscored"], counts["n_at_cap"], spec.cap)
        gacc = LC.num_or_none(g.get("acc"))
        acc = counts["acc"]
        acc_ok = (acc is None and (gacc is None or counts["n_scored"] == 0)) or (acc is not None and gacc is not None and abs(float(gacc) - acc) < 1e-9)
        if got != want or not acc_ok:
            raise _PairRefusal(f"{what} (n {got[0]}, n_scored {got[1]}, n_unscored {got[2]}, n_at_cap {got[3]}, max_tokens {got[4]}, acc {gacc}) does not reproduce the outcome "
                               f"record (n {want[0]}, n_scored {want[1]}, n_unscored {want[2]}, n_at_cap {want[3]}, cap {want[4]}, acc {acc})", uri)
        if str(g.get("thinking") or "") != self.THINKING_SPELLING:
            raise _PairRefusal(f"{what}:thinking is {g.get('thinking')!r}, not {self.THINKING_SPELLING!r}", uri)
        if str(g.get("task") or "") != spec.task:
            raise _PairRefusal(f"{what}:task is {g.get('task')!r}, not {spec.task!r}", uri)

    # ---- the producer's records -------------------------------------------------------------------
    def _cross_check_three(self, tn: Mapping[str, Any], base: Mapping[str, Any], items: Mapping[str, Any], obj_sha: Optional[str], tag: str, selection: str, seed_n: int,
                           paired: bool, okey: str, tree: str, n_items: int, legacy_tag: Optional[str] = None, instruct: bool = False,
                           spelling: Optional[str] = None, unsteered: bool = False, dose_relu: bool = False, lecun_relu: bool = False) -> List[str]:
        """The producer's three-number row (``w4_three_numbers/1``) against the re-derivation and the store: every count, the value, the bound, the at-cap
        count and share, the identity fields, the store and summary paths, the RESULT line. ``spelling`` (a SHARED reader) is what the row's ``view`` must
        read instead of the selection (:data:`SHARED_SPELLING_RULE`)."""
        spec = self.SPEC
        c = items["counts"]
        man = base["manifest"]
        bad: List[str] = []

        def eq(name: str, theirs, mine, rounded: bool = False) -> None:
            ok = rounded_equal(theirs, mine) if rounded else (theirs == mine)
            if not ok:
                bad.append(f"{name}: producer {theirs!r}, re-derived {mine!r}")

        def eq_in(name: str, theirs, allowed: Sequence[Any]) -> None:
            if theirs not in allowed:
                bad.append(f"{name}: producer {theirs!r}, re-derived one of {list(allowed)!r}")

        want_spelling = selection if spelling is None else spelling
        eq("status", str(tn.get("status")), "complete")
        if instruct:
            # the instruct-direct row (INSTRUCT_DIRECT_RULE): its own tag, the `it` selection, NO seed
            eq("tag", str(tn.get("tag")), tag)
            eq("view", str(tn.get("view")), want_spelling)
            eq("seed", LC.int_or_none(tn.get("seed")), None)
        elif unsteered or dose_relu or lecun_relu:
            # the unsteered row (UNSTEERED_ROW_RULE): its own tag, the `u0` selection; it carries no fitted object, so the producer may spell its seed as the group's or as none.
            # A dose row (`dose_relu`: the ReLU dose copies' d0p75_relul, DOSE_RELU_ROW_RULE, and the difference LoRA's d1p25, DOSE_LORA_ROW_RULE) and the LeCun-init ReLU
            # row (`lecun_relu`: s2rl, LECUN_RELU_ROW_RULE, whose objects' own fit seed is the sidecars' 0): its own tag, its selection; the seed is the model's
            # standup-group convention, spelled as the group's or as none the same way
            eq("tag", str(tn.get("tag")), tag)
            eq("view", str(tn.get("view")), want_spelling)
            eq_in("seed", LC.int_or_none(tn.get("seed")), (seed_n, None))
        elif legacy_tag is None:
            eq("tag", str(tn.get("tag")), tag)
            eq("view", str(tn.get("view")), want_spelling)
            eq("seed", LC.int_or_none(tn.get("seed")), seed_n)
        else:
            # the `main` canary's row may carry the producer's own spelling (LEGACY_MAIN_ALIAS_RULE): its tag, the `main` selection, no seed
            eq_in("tag", str(tn.get("tag")), (tag, legacy_tag))
            eq_in("view", str(tn.get("view")), (selection, MAIN_SELECTION))
            eq_in("seed", LC.int_or_none(tn.get("seed")), (seed_n, None))
        eq("paired", bool(tn.get("paired")), paired)
        eq("cell_id", str(tn.get("cell_id")), okey)
        eq_in("cell_slug", str(tn.get("cell_slug")), (okey, str(base["ck"].object_token)))     # the object key, or the store key's own token (`--`; the appendix record)
        eq("arm", str(tn.get("arm")), str(LC.cell_grammar_legacy(okey)["paper_label"]))
        eq("cap", LC.int_or_none(tn.get("cap")), spec.cap)
        eq("max_tokens", LC.int_or_none(tn.get("max_tokens")), spec.cap)
        eq("is_rung", bool(tn.get("is_rung")), False)
        eq("merged", bool(tn.get("merged")), False)
        eq("store", str(tn.get("store")), rel_store(base["row_dir"], okey, base["suffix"]))
        if isinstance(tn.get("launched_as"), Mapping):
            la = tn["launched_as"]
            eq("launched_as.wave", str(la.get("wave")), launch_tag_of(tag, okey, legacy_tag))
            eq("launched_as.row_dir", str(la.get("row_dir")), base["row_dir"])
            eq("launched_as.suffix", str(la.get("suffix")), base["suffix"])
        eq("summary", str(tn.get("summary")), f"cells/{base['store_key']}/summary/{spec.outcomes_file}")
        eq("result_line", str(tn.get("result_line")), base["result_last"])
        eq("engine", str(tn.get("engine")), str(man.get("serving_engine")))
        eq("serving_identity", str(tn.get("serving_identity")), str(man.get("serving_identity")))
        eq("producer_tree", str(tn.get("producer_tree")), tree)
        eq("bank_view", tn.get("bank_view"), (man.get("route") or {}).get("bank_view") if isinstance(man.get("route"), Mapping) else None)
        eq("n_attempted", LC.int_or_none(tn.get("n_attempted")), n_items)
        eq("n_scored", LC.int_or_none(tn.get("n_scored")), c["n_scored"])
        eq("correct", LC.int_or_none(tn.get("correct")), c["correct"])
        eq("value", LC.num_or_none(tn.get("value")), c["acc"], rounded=True)
        eq(BOUND_SPELLING, LC.num_or_none(tn.get(BOUND_SPELLING)), c["correct"] / n_items, rounded=True)
        eq("at_cap_n", LC.int_or_none(tn.get("at_cap_n")), c["n_at_cap"])
        eq("at_cap_share", LC.num_or_none(tn.get("at_cap_share")), c["n_at_cap"] / n_items, rounded=True)
        eq("n_unscored", LC.int_or_none(tn.get("n_unscored")), c["n_unscored"])
        eq("draws", LC.int_or_none(tn.get("draws")), n_items)
        eq("samples_per_item", LC.int_or_none(tn.get("samples_per_item")), 1)
        eq("object_sha256", tn.get("object_sha256"), obj_sha)
        if obj_sha is not None:
            eq("object_sha256_source", str(tn.get("object_sha256_source")), "provenance.artifact_sha256")
        return bad

    def _cross_check_summary(self, summary: Mapping[str, Any], base: Mapping[str, Any], items: Mapping[str, Any], obj_sha: Optional[str], okey: str, tree: str,
                             n_items: int) -> List[str]:
        """The producer's ``controls_summary.json`` beside the store against the store: the key, the object and the task's reading."""
        spec = self.SPEC
        bad: List[str] = []
        c = items["counts"]
        if str(summary.get("cell_id") or "") != okey:
            bad.append(f"cell_id {summary.get('cell_id')!r} is not {okey}")
        prov = summary.get("provenance") if isinstance(summary.get("provenance"), Mapping) else None
        if prov is not None and (prov.get("artifact_sha256") or None) != obj_sha:
            bad.append(f"provenance.artifact_sha256 {str(prov.get('artifact_sha256'))[:16]!r} is not the resolved object {str(obj_sha)[:16]}")
        src = (summary.get("sources") or {}).get(spec.task) if isinstance(summary.get("sources"), Mapping) else None
        if isinstance(src, Mapping):
            if str(src.get("cell_key") or "") not in ("", base["store_key"]):
                bad.append(f"sources[{spec.task}].cell_key {src.get('cell_key')!r} is not the store key {base['store_key']}")
            if str(src.get("producer_tree_sha") or "") not in ("", tree):
                bad.append(f"sources[{spec.task}].producer_tree_sha {str(src.get('producer_tree_sha'))[:12]!r} is not the manifest's tree {tree[:12]}")
        g = (summary.get("controls") or {}).get(spec.task) if isinstance(summary.get("controls"), Mapping) else None
        if not isinstance(g, Mapping):
            bad.append(f"controls[{spec.task}] is absent")
            return bad
        gacc = LC.num_or_none(g.get("acc"))
        if (c["acc"] is None) != (gacc is None) or (c["acc"] is not None and abs(float(gacc) - c["acc"]) > 1e-9):
            bad.append(f"controls[{spec.task}].acc {gacc} is not correct / scored {c['acc']}")
        got = (LC.int_or_none(g.get("n")), LC.int_or_none(g.get("n_scored")), LC.int_or_none(g.get("n_at_cap")), LC.int_or_none(g.get("max_tokens")), str(g.get("thinking") or ""), str(g.get("task") or ""))
        if got != (n_items, c["n_scored"], c["n_at_cap"], spec.cap, self.THINKING_SPELLING, spec.task):
            bad.append(f"controls[{spec.task}] (n {got[0]}, n_scored {got[1]}, n_at_cap {got[2]}, max_tokens {got[3]}, thinking {got[4]!r}, task {got[5]!r}) is not the store's "
                       f"(n {n_items}, n_scored {c['n_scored']}, n_at_cap {c['n_at_cap']}, max_tokens {spec.cap}, thinking {self.THINKING_SPELLING!r}, task {spec.task!r})")
        return bad

    def _check_gate(self, gate: Mapping[str, Any], gate_inp: Input, base: Mapping[str, Any], okey: str, tree: str, obj_sha: Optional[str], ident: str) -> Dict[str, Any]:
        """The gate record against the store (:data:`GATE_RULE`): the verdict, the attested engine, the completed task, the tree, the identity; the
        recorded flags are carried. Raises :class:`_PairRefusal`."""
        spec = self.SPEC
        where = gate_inp.uri.rsplit("#", 1)[0]
        if str(gate.get("suffix") or "") != base["suffix"] or str(gate.get("cell") or "") != okey or str(gate.get("row") or "") != base["row_dir"]:
            raise _PairRefusal(f"the gate record names row {gate.get('row')!r} / {gate.get('cell')!r} / {gate.get('suffix')!r}, not this store's ({base['row_dir']}, {okey}, {base['suffix']})", where)
        verdict = str(gate.get("verdict") or "")
        if verdict != GATE_PASS:
            problems = [str(p) for p in (gate.get("problems") or [])]
            raise _PairRefusal(f"the gate record's verdict is {verdict!r}, not {GATE_PASS} ({'; '.join(problems)[:400] or 'no problem named'}); a row files only under a PASS", where)
        if str(gate.get("sha") or "") != tree:
            raise _PairRefusal(f"the gate record attests tree {str(gate.get('sha'))[:12]!r}, the store {tree[:12]}", where)
        if spec.record_task not in [str(x) for x in (gate.get("completed_legs") or [])]:
            raise _PairRefusal(f"the gate record's completed tasks {list(gate.get('completed_legs') or [])} do not include {spec.record_task}", where)
        facts_list = [f for f in (gate.get("facts") or []) if isinstance(f, Mapping)]
        if len(facts_list) != 1:
            raise _PairRefusal(f"the gate record carries {len(facts_list)} facts blocks (expected exactly one)", where)
        f = facts_list[0]
        engine = str(f.get("route.serving_engine") or "")
        if engine != SERVING_ENGINE:
            raise _PairRefusal(f"the gate record attests route.serving_engine {engine!r}, not {SERVING_ENGINE}", where)
        if str(f.get("producer.tree_sha") or "") != tree:
            raise _PairRefusal(f"the gate record's facts attest tree {str(f.get('producer.tree_sha'))[:12]!r}, the store {tree[:12]}", where)
        if str(f.get("serving_identity") or "") != ident:
            raise _PairRefusal(f"the gate record attests serving_identity {f.get('serving_identity')!r}, the store {ident!r}", where)
        if (f.get("provenance.artifact_sha256") or None) != obj_sha:
            raise _PairRefusal(f"the gate record attests provenance.artifact_sha256 {str(f.get('provenance.artifact_sha256'))[:16]!r}, the store {str(obj_sha)[:16]!r}", where)
        done_markers = [str(x) for x in (gate.get("done_markers") or [])]
        want_marker = f"{base['suffix']}/cells/{base['store_key']}/done/{spec.outcomes_file}"
        if want_marker not in done_markers:
            raise _PairRefusal(f"the gate record's done markers {done_markers} do not name {want_marker}", where)
        recorded = f.get("recorded_flags") if isinstance(f.get("recorded_flags"), Mapping) else {}
        return {"record": LC.rel_home(where), "verdict": verdict, "read_at": gate.get("read_at"), "serving_engine_attested": engine, "serving_path_legacy": f.get("serving_path"),
                "serving_path_requested": gate.get("serving_path_requested"), "requested_from": gate.get("requested_from"), "attested_from": list(gate.get("attested_from") or []),
                "completed_tasks_legacy": [str(x) for x in (gate.get("completed_legs") or [])], "launch_flags": dict(f.get("launch_flags") or {}) if isinstance(f.get("launch_flags"), Mapping) else None,
                "recorded_flags": dict(recorded), "max_lora_rank": LC.int_or_none(recorded.get("max_lora_rank")), "zero_adapter_only": recorded.get("zero_adapter_only"),
                "flags_source": f.get("flags_source"), "compared_against": (dict(gate.get("compared_against")) if isinstance(gate.get("compared_against"), Mapping) else None),
                **({"cross_tree": gate.get("cross_tree")} if "cross_tree" in gate else {}),      # the addendum: a gate at a relaunch tree compared against the first tree's record
                "problems": [str(p) for p in (gate.get("problems") or [])], "rule": GATE_RULE, "max_lora_rank_rule": MAX_LORA_RANK_RULE}

    def _check_launched_roster(self, roster_doc: Mapping[str, Any], roster_inp: Input, entry_doc: Mapping[str, Any], entry_inp: Input, tag: str, okey: str, obj: LC.ResolvedObject,
                               steered: bool, legacy_tag: Optional[str] = None, choice: Optional[RosterChoice] = None) -> Dict[str, Any]:
        """The store's ``lane_roster.json`` (the launched entry) against the mirrored roster it names by sha256 (``choice``: the roster the set matched by
        hash, :data:`ROSTER_SET_RULE`; the plain file when there is no index), and that entry against the resolved object."""
        spec = self.SPEC
        launch_tag = launch_tag_of(tag, okey, legacy_tag)
        if str(entry_doc.get("cell_id") or "") != okey:
            raise _PairRefusal(f"{LAUNCHED_ENTRY_FILE} names cell_id {entry_doc.get('cell_id')!r}, not {okey!r}", entry_inp.uri)
        if str(entry_doc.get("wave") or "") != launch_tag:
            raise _PairRefusal(f"{LAUNCHED_ENTRY_FILE} names launch tag {entry_doc.get('wave')!r}, not {launch_tag!r}", entry_inp.uri)
        launched_roster_sha = str(entry_doc.get("roster_sha256") or "")
        roster_revision_note = None
        if launched_roster_sha != roster_inp.sha256:
            if legacy_tag is None:
                raise _PairRefusal(f"{LAUNCHED_ENTRY_FILE}:roster_sha256 {launched_roster_sha[:12]!r} is not the mirrored {spec.roster_file}'s sha256 {roster_inp.sha256[:12]}", entry_inp.uri)
            # a canary-row store (two producer trees): launched under an EARLIER revision of the roster, before its `s1` blocks existed; the mirrored
            # revision must still carry its block and agree with the store on the entry (checked below) -- the revision difference is RECORDED, never hidden
            roster_revision_note = (f"launched under an earlier revision of {spec.roster_file} (sha256 {launched_roster_sha[:12] or '<none>'}); the mirrored revision "
                                    f"({roster_inp.sha256[:12]}) carries the {launch_tag} block and its entry agrees with the store (the canary row predates the s1 blocks, D565)")
        want_sha = obj.object_id if steered else None
        if (entry_doc.get("object_sha256") or None) != want_sha:
            raise _PairRefusal(f"{LAUNCHED_ENTRY_FILE}:object_sha256 {str(entry_doc.get('object_sha256'))[:16]!r} is not the resolved object {str(want_sha)[:16]!r}", entry_inp.uri)
        entry_id = str(entry_doc.get("entry_id") or "")
        blk = (roster_doc.get("waves") or {}).get(launch_tag) if isinstance(roster_doc.get("waves"), Mapping) else None
        if not isinstance(blk, Mapping):
            raise _PairRefusal(f"{spec.roster_file} carries no block for launch tag {launch_tag!r}", roster_inp.uri.rsplit("#", 1)[0])
        entry = (blk.get("entries") or {}).get(entry_id) if isinstance(blk.get("entries"), Mapping) else None
        if not isinstance(entry, Mapping):
            raise _PairRefusal(f"{spec.roster_file} block {launch_tag} carries no entry {entry_id!r} ({LAUNCHED_ENTRY_FILE}:entry_id)", roster_inp.uri.rsplit("#", 1)[0])
        bad: List[str] = []
        if str(entry.get("cell_id") or "") != okey:
            bad.append(f"entries[{entry_id}].cell_id {entry.get('cell_id')!r} is not {okey!r}")
        if str(entry.get("status") or "") != "runnable":
            bad.append(f"entries[{entry_id}].status {entry.get('status')!r} is not runnable")
        legs = blk.get("legs") if isinstance(blk.get("legs"), Mapping) else {}
        if str(legs.get(spec.record_task) or "") != "runnable":
            bad.append(f"legs.{spec.record_task} {legs.get(spec.record_task)!r} is not runnable")
        eobj = entry.get("object") if isinstance(entry.get("object"), Mapping) else None
        if steered:
            if eobj is None:
                bad.append(f"entries[{entry_id}] names no object block for a fitted key")
            else:
                if str(eobj.get("staged_id") or "") != str(obj.staged_id):
                    bad.append(f"entries[{entry_id}].object.staged_id {eobj.get('staged_id')!r} is not the resolved staged id {obj.staged_id!r}")
                if str(eobj.get("sha256") or "") != obj.object_id:
                    bad.append(f"entries[{entry_id}].object.sha256 {str(eobj.get('sha256'))[:16]!r} is not the resolved object {str(obj.object_id)[:16]}")
        elif eobj is not None:
            bad.append(f"entries[{entry_id}] names an object block for an unsteered or prompted key")
        if str(roster_doc.get("serving_path") or "") != SERVING_ENGINE:
            bad.append(f"the roster header's serving_path {roster_doc.get('serving_path')!r} is not {SERVING_ENGINE}")
        if bad:
            raise _PairRefusal(f"{spec.roster_file} (the launched roster, {LAUNCHED_ENTRY_FILE}:roster_sha256) disagrees with the store on " + "; ".join(bad), roster_inp.uri.rsplit("#", 1)[0])
        out = {"record": LC.rel_home(entry_inp.uri), "entry_id": entry_id, "launch_tag_legacy": launch_tag, "launched_at": entry_doc.get("launched_at"), "object_generation": entry_doc.get("object_generation"),
               "roster_sha256": entry_doc.get("roster_sha256"), "snapshot_sha256": entry_doc.get("snapshot_sha256"), "superseded_by_key": entry_doc.get("superseded_by_key"),
               "roster_record": LC.rel_home(roster_inp.uri.rsplit("#", 1)[0]), "roster_entry": {k: v for k, v in entry.items() if k != "note"}, "roster_legs_legacy": dict(legs),
               "roster_sampling": roster_doc.get("sampling"), "roster_thinking": roster_doc.get("thinking"), "roster_pins": roster_doc.get("pins")}
        if roster_revision_note is not None:
            out.update(roster_sha256_mirrored=roster_inp.sha256, roster_revision_note=roster_revision_note)
        if choice is not None and choice.from_index:
            # the roster set (ROSTER_SET_RULE): which mirrored roster file matched the store's hash, and the producer tree the index names for it
            out.update(roster_file=choice.file, tree_sha=choice.tree_sha, roster_set_rule=ROSTER_SET_RULE)
            if choice.pins_sha256 is not None:
                out.update(pins_file=choice.pins_file, pins_sha256=choice.pins_sha256)
        return out

    # ---- decode ---------------------------------------------------------------------------------
    _numbers_agree = staticmethod(G.Gsm8kSeedrepAdapter._numbers_agree)

    def _match_policy(self, model_id: str, block: Mapping[str, Any]) -> Tuple[Optional[DP.Policy], str]:
        pf = self._policy_file()
        if pf is None:
            return None, "the committed decode-policy export could not be loaded"
        try:
            p = pf.resolve(model_id, eval_id=self.SPEC.eval_id, thinking_mode=self.THINKING_SPELLING)
        except (DP.NoPolicy, KeyError, ValueError) as exc:
            return None, f"no registered decode policy row for {model_id} / {self.SPEC.eval_id} thinking {self.THINKING_SPELLING}: {exc}"
        if self._numbers_agree(block, p.numbers()):
            return p, "exact by the stated sampling numbers"
        return None, f"the sent numbers {RP._numbers(block)} differ from the default policy row {p.policy_id} {RP._numbers(p.numbers())}"

    # ---- one (tag, object key) row -------------------------------------------------------------
    def _emit_row(self, ctx: AdapterContext, group: str, tag: str, seed_n: Optional[int], selection: str, model_id: str, inst_row: Mapping[str, Any], okey: str,
                  suffix: str, st: Mapping[str, Any], lrow: Mapping[str, Any], roster_set: RosterSet, pins_set: PinsSet, three: Mapping[str, Any],
                  gate_docs: Mapping[Tuple[str, ...], Tuple[Dict[str, Any], Input]], ls_inp: Input, tn_inp: Input,
                  facts: Dict[str, Any], legacy_tag: Optional[str] = None, n_items: int = N_ITEMS, item_set: Optional[ItemSetPin] = None, primary_tree: Optional[str] = None,
                  relaunches: Optional[Mapping[str, Any]] = None, relaunches_inp: Optional[Input] = None, repeat: Optional[int] = None) -> Dict[str, Any]:
        """Reads one (tag, object key) and BUILDS its records (``records``: ``(record type, payload)`` in emission order — the run first; the caller adds
        them once the group's anchor blocks are stamped on the run payloads); returns what the group's contrasts need beside them (the run id, the items,
        whether steered, the declared role). ``repeat`` is a SHARED repeat store's index (:data:`SHARED_ROW_KEY_RULE`); None on every solo reader."""
        spec = self.SPEC
        seeds = self.seeds
        producer = ctx.producer
        records: List[Tuple[str, Dict[str, Any]]] = []
        instruct = is_instruct_tag(tag)
        dose_relu = is_dose_relu_tag(tag)
        dose_lora = is_dose_lora_tag(tag)
        dose_row = dose_relu or dose_lora
        lecun_relu = is_lecun_relu_tag(tag)
        # an instruct-direct, unsteered, dose-row or LeCun-init ReLU tag is never the paired one
        paired = False if (instruct or is_unsteered_tag(tag) or dose_row or lecun_relu) else bool(tag_parts(tag)[1])
        mk = model_key_of(tag)
        mkey = MODEL_KEYS[mk]
        launch_tag = launch_tag_of(tag, okey, legacy_tag)
        shared_spec = spec.shared
        rk_okey = row_okey(okey, repeat, window_of_suffix(spec, suffix))
        base = self._store_bundle(st, okey, tag, suffix)
        man = base["manifest"]
        where_man = base["files"]["manifest.json"].uri
        if str(man.get("cell_id") or "") != okey:
            raise _PairRefusal(f"manifest.json names object key {man.get('cell_id')!r}, not {okey!r}", where_man)
        # -- the tree: the manifest's, RESULT's and the outcome record's
        tree = str((man.get("producer") or {}).get("tree_sha") or "")
        if not _SHA1.match(tree):
            raise _PairRefusal("manifest.json:producer.tree_sha names no 40-hex tree", where_man)
        if str(base["result"].get("sha") or "") != tree:
            raise _PairRefusal(f"RESULT names tree {str(base['result'].get('sha'))[:12]!r}, the manifest {tree[:12]}", base["top"]["RESULT"].uri)
        if str(base["outcomes"].get("producer_tree_sha") or "") != tree:
            raise _PairRefusal(f"the outcome record names tree {str(base['outcomes'].get('producer_tree_sha'))[:12]!r}, the manifest {tree[:12]}", base["files"]["outcomes"].uri)
        sm = spec.suffix_re.match(suffix)
        if not tree.startswith(sm.group("sha7")):
            raise _PairRefusal(f"the store suffix's sha7 {sm.group('sha7')} is no prefix of the tree {tree[:12]}", base["dir"])
        # -- the tree the launch set says the row was launched at (TREES_RULE): the row's `sha`, else the set's primary tree; unset on the earlier launch sets
        row_sha = lrow.get("sha")
        if row_sha is not None and not _SHA1.match(str(row_sha)):
            raise _PairRefusal(f"the launch set's row names sha {str(row_sha)[:16]!r}, not a 40-hex producer tree", ls_inp.uri)
        la_decl = lrow.get("launched_as") if isinstance(lrow.get("launched_as"), Mapping) else None
        la_sha: Optional[str] = str(la_decl["sha"]) if la_decl is not None and la_decl.get("sha") is not None else None
        if la_sha is not None and not _SHA1.match(la_sha):
            raise _PairRefusal(f"the launch set's launched_as names sha {la_sha[:16]!r}, not a 40-hex producer tree", ls_inp.uri)
        # LAUNCHED_AS_TREE_RULE: a row's sha IS the tree of the store it reads, and launched_as names that store with the same sha -- a row whose two shas differ is an
        # inconsistent record and refuses by name; no store is read for it under either tree
        if la_sha is not None and row_sha is not None and str(row_sha) != la_sha:
            raise _PairRefusal(f"the row's sha {str(row_sha)[:12]} is not the launched store's tree {la_sha[:12]}; launched_as names {str(la_decl.get('suffix'))!r} at {la_sha[:12]} (this "
                               f"store's manifest names {tree[:12]}): the record is inconsistent, and a row's sha is the tree of the store it reads, so no store is read for the row until "
                               f"the record agrees ({LAUNCHED_AS_TREE_RULE})", ls_inp.uri)
        row_tree = str(row_sha) if row_sha is not None else primary_tree
        tree_source = "row sha" if row_sha is not None else "primary tree"
        if row_tree is not None and row_tree != tree:
            raise _PairRefusal(f"the launch set says the row was launched at tree {row_tree[:12]} ({'its own sha' if row_sha is not None else 'the set\'s primary tree, no sha on the row'}); "
                               f"the store's manifest names {tree[:12]}", ls_inp.uri)
        # -- a RELAUNCHED store (RELAUNCH_RULE): the suffix's `-r<N>` must be the row's launched_as and the relaunch record's entry
        relaunch_tag = (sm.group("relaunch") or "")[1:] or None
        relaunch_entry = None
        if relaunch_tag is not None:
            la = lrow.get("launched_as")
            if not isinstance(la, Mapping) or str(la.get("suffix") or "") != suffix:
                raise _PairRefusal(f"the store suffix {suffix!r} spells a relaunch ({relaunch_tag}) but the launch set's row does not name it on launched_as ({la!r})", ls_inp.uri)
            if relaunches is None:
                raise _PairRefusal(f"no {RELAUNCHES_FILE} beside the stores ({' / '.join(RELAUNCHES_DIRS)}) names the relaunch {launch_tag}/{okey} the store suffix {suffix!r} spells", self.root)
            relaunch_entry = relaunches.get(f"{launch_tag}/{okey}") if isinstance(relaunches, Mapping) else None
            where_rl = relaunches_inp.uri.rsplit("#", 1)[0] if relaunches_inp is not None else self.root
            if not isinstance(relaunch_entry, Mapping):
                raise _PairRefusal(f"{RELAUNCHES_FILE} names no entry {launch_tag}/{okey} for the relaunched store {suffix!r}", where_rl)
            # (a SHARED relaunch's entry spells `shared-r<N>`, the suffix's whole tail after the sha7; a solo one `r<N>`)
            want_rl_tag = relaunch_tag if shared_spec is None else shared_spec.relaunch_entry_tag(relaunch_tag)
            if str(relaunch_entry.get("tag") or "") != want_rl_tag or str(relaunch_entry.get("sha") or "") != tree:
                raise _PairRefusal(f"{RELAUNCHES_FILE}[{launch_tag}/{okey}] says tag {relaunch_entry.get('tag')!r} at tree {str(relaunch_entry.get('sha'))[:12]!r}; the store is {want_rl_tag} at {tree[:12]}"
                                   + (f" (= the row's launched_as sha, which the entry must name; LAUNCHED_AS_TREE_RULE)" if la_sha is not None else ""), where_rl)
        # -- the launched roster and pins THIS store was launched under (ROSTER_SET_RULE): by the roster hash its lane_roster.json names, and its tree
        roster_choice = roster_set.select(str(base["launched_entry"].get("roster_sha256") or ""), base["top"][LAUNCHED_ENTRY_FILE].uri)
        roster_doc, roster_inp = roster_choice.doc, roster_choice.inp
        pins_doc, pins_inp, pins_file, pins_tree = pins_set.select(tree, roster_choice)
        if not isinstance(pins_doc, dict) or not isinstance(pins_doc.get(mk), Mapping):
            raise _PairRefusal(f"the launched pins for this store ({RECORDS_DIR}/{pins_file}) carry no {mk} block", self._records_dir())
        if roster_choice.pins_sha256 is not None and pins_inp is not None and pins_inp.sha256 != roster_choice.pins_sha256:
            # the pins are matched within the mirrored SET by hash (the addendum: two per-tree copies may differ in bytes alone -- key order -- and each roster names its own)
            raise _PairRefusal(f"the launched pins copy {RECORDS_DIR}/{pins_file} has sha256 {pins_inp.sha256[:12]}; {ROSTERS_INDEX_FILE} names pins_sha256 {roster_choice.pins_sha256[:12]} for the "
                               f"store's roster {roster_choice.file}: the pins the store was launched under are not the mirrored copy", pins_inp.uri.rsplit("#", 1)[0])
        # -- the served model and identity
        served_hf = str(man.get("model_id") or "")
        if seeds.model_for_hf_id(served_hf) != model_id:
            raise _PairRefusal(f"the store serves {served_hf!r}, not the seed tag's model {model_id}", where_man)
        if served_hf != mkey.served_hf_id:
            raise _PairRefusal(f"the store serves {served_hf!r}, not the family's served checkpoint {mkey.served_hf_id!r} (the model-key table = models.yaml hf_repo)", where_man)
        rev = str(man.get("model_revision") or "")
        pinned_rev = mkey.served_revision      # = models.yaml hf_revision (checked per group in read())
        if _SHA1.match(pinned_rev) and rev != pinned_rev:
            raise _PairRefusal(f"the store serves revision {rev[:12] or '<none>'}, not the model seed's {pinned_rev[:12]}", where_man)
        if str(man.get("serving_engine") or "") != SERVING_ENGINE:
            raise _PairRefusal(f"manifest.json:serving_engine {man.get('serving_engine')!r} is not {SERVING_ENGINE!r}", where_man)
        # -- the object: the launch set's pin key and expected sha, the harness roster / pins, the launched pins, the store's attestations
        expected_sha = lrow.get("object_sha256_expected") or None
        launch_pin_key = lrow.get("pin_key") or None
        # the `main` canary row's objects are the roster's main selection's (LEGACY_MAIN_ALIAS_RULE); an instruct-direct row's the roster's `it` selection's, for every
        # family alike (INSTRUCT_DIRECT_RULE: the harness stages the instruct fits there)
        roster_sel = MAIN_SELECTION if legacy_tag is not None else (INSTRUCT_SELECTION if instruct else None)
        lclass = str(lrow.get("class") or "")
        launched_sha = base["launched_entry"].get("object_sha256")
        try:
            obj, op = self.resolve_object(okey, selection, pin_key=launch_pin_key, model_key=mk, roster_selection=roster_sel, launched_pins=pins_doc,
                                          launched_object_sha=(str(launched_sha) if isinstance(launched_sha, str) else None), composed=(lclass == "composed"),
                                          launched_pins_file=pins_file)
        except IC.Unresolved as exc:
            raise _PairRefusal(f"object: {exc.why}", ls_inp.uri)
        arditi = okey == ARDITI_OBJECT_KEY
        steered = obj.object_kind not in ("none", "system_instruction")
        # the route a composed key took (PINS_KEY_RULE): the harness pins' key, or the store's tree pins by the launched sha
        pins_by_sha = False
        if steered and not arditi and not mkey.roster_selections and roster_sel is None:
            _hr, h_pins = self._roster_pins()
            h_entry = (h_pins.get(mk) or {}).get(obj.staged_id) if isinstance(h_pins.get(mk), Mapping) else None
            pins_by_sha = not (isinstance(h_entry, Mapping) and _SHA256.match(str(h_entry.get("object_sha256") or "")))
        # the launch set's class grammar: the paired tag's `none` is its ANCHOR (PAIRED_ANCHOR_RULE) and a prompted key is `prompted` on either tag; only a
        # fitted key under the paired tag is the `composed` object
        want_class = "arditi" if arditi else ("anchor" if obj.object_kind == "none" else ("prompted" if obj.object_kind == "system_instruction" else ("composed" if paired else "arm")))
        # a SHARED reader admits the instruct row's OWN none as a store (SHARED_IT_NONE_RULE); a solo reader never does (INSTRUCT_ANCHOR_RULE). The repeat segment
        # (`repeat`, read off the suffix before this point) decides the classification (SHARED_IT_REPEAT_RULE): the first serving is the it_none_comparison row, the
        # `-rep<N>` serving the row's repeat floor -- never filed as it_none_comparison
        it_none_store = bool(shared_spec is not None and instruct and obj.object_kind == "none")
        it_none = it_none_store and repeat is None
        it_repeat = it_none_store and repeat is not None
        if instruct:
            # the instruct-direct declaration carries fitted keys alone (INSTRUCT_DIRECT_RULE); its `none` is an alias, never a store (INSTRUCT_ANCHOR_RULE)
            if obj.object_kind == "none" and not it_none_store:
                raise _PairRefusal(f"a `none` store under the instruct-direct tag {tag}: the declaration carries the `_it` none BY ALIAS to {group}/none (anchor_tuple "
                                   f"s{INSTRUCT_ANCHOR_SEED}/none) and no `_it` none store exists; this store is not read, so no duplicate none run files ({INSTRUCT_ANCHOR_RULE})", st["dir"])
            if (not steered or arditi) and not it_none_store:
                raise _PairRefusal(f"a {obj.object_kind} key ({okey}) under the instruct-direct tag {tag}: the declaration carries the instruct-checkpoint LoRA fits (class "
                                   f"`{INSTRUCT_CLASS}`) and the aliased none only; nothing else is admitted under `_it` ({INSTRUCT_DIRECT_RULE})", st["dir"])
            want_class = IT_NONE_CLASS if it_none else INSTRUCT_CLASS
        if repeat is not None:
            # a SHARED repeat store (the row's none driven again on the same standup): the launch set's class is `repeat`
            if obj.object_kind != "none":
                raise _PairRefusal(f"the store suffix {suffix!r} spells a repeat of a {obj.object_kind} key ({okey}); a repeat is the row's `none` driven again ({SHARED_ROW_KEY_RULE})", st["dir"])
            want_class = REPEAT_CLASS
        if lclass != want_class:
            raise _PairRefusal(f"the launch set files this row as class {lclass!r}, the key's grammar says {want_class!r}", ls_inp.uri)
        if instruct:
            # the row's own declaration (INSTRUCT_DIRECT_RULE): the block, the transfer kind and the anchor it names must be the instruct-direct ones (the shared
            # instruct row's own none names NO anchor: SHARED_IT_NONE_RULE)
            got_decl = (str(lrow.get("block") or ""), str(lrow.get("transfer_kind") or ""), str(lrow.get("anchor_tuple") or ""))
            want_decl = (INSTRUCT_BLOCK, INSTRUCT_TRANSFER_KIND, ("" if it_none_store else f"s{INSTRUCT_ANCHOR_SEED}/none"))
            if got_decl != want_decl:
                raise _PairRefusal(f"the launch set's instruct-direct row declares (block {got_decl[0]!r}, transfer_kind {got_decl[1]!r}, anchor_tuple {got_decl[2]!r}); the declaration is "
                                   f"({want_decl[0]}, {want_decl[1]}, {want_decl[2]}) ({INSTRUCT_DIRECT_RULE})", ls_inp.uri)
        # the launch set's row: the tag and seed base it spells are this store's — for the `main` canary the producer's own spelling (its tag, no seed) is accepted too;
        # an instruct-direct row spells NO seed base (the fit is the instruct checkpoint's, not a seed replicate)
        tags_ok = (tag,) if legacy_tag is None else (tag, legacy_tag)
        seed_ok = (lrow.get("seed_base") is None) if instruct else (LC.int_or_none(lrow.get("seed_base")) == seed_n or (legacy_tag is not None and lrow.get("seed_base") is None))
        if not seed_ok or str(lrow.get("tag") or "") not in tags_ok or str(lrow.get("cell_slug") or "") != base["ck"].object_token:
            raise _PairRefusal(f"the launch set's row (tag {lrow.get('tag')!r}, seed_base {lrow.get('seed_base')!r}, cell_slug {lrow.get('cell_slug')!r}) is not this store's "
                               f"({' / '.join(tags_ok)}, {'null (instruct-direct)' if instruct else seed_n}, store key object token "
                               f"{base['ck'].object_token})", ls_inp.uri)
        # the launch set's launch tag: the store's (`<model key>_main` for a canary-row store) or, on a canary-row store alone, the tag's own spelling -- the
        # census names the aliased rows by the tag they FILE under (`bcbw_wave` gemma4e4b_s1) and keeps the launch spelling on `launched_as`
        launch_tags_ok = (launch_tag,) if legacy_tag is None else (launch_tag, launch_tag_of(tag, okey))
        if str(lrow.get("bcbw_cell") or okey) != okey or str(lrow.get("bcbw_wave") or "") not in launch_tags_ok:
            raise _PairRefusal(f"the launch set's launch tag / key ({lrow.get('bcbw_wave')!r} / {lrow.get('bcbw_cell')!r}) are not this store's ({' / '.join(launch_tags_ok)}, {okey})", ls_inp.uri)
        if str(lrow.get("model_key") or mk) != mk:
            raise _PairRefusal(f"the launch set's row names model key {lrow.get('model_key')!r}, not this store's {mk}", ls_inp.uri)
        # the census's `launched_as` (a canary-row store): the launch tag, row dir and suffix it was launched under must be this store's
        launched_as = lrow.get("launched_as")
        if isinstance(launched_as, Mapping):
            got = (str(launched_as.get("wave") or ""), str(launched_as.get("row_dir") or ""), str(launched_as.get("suffix") or ""))
            if got != (launch_tag, base["row_dir"], suffix):
                raise _PairRefusal(f"the launch set says the row was launched as {got!r}; the store is ({launch_tag}, {base['row_dir']}, {suffix})", ls_inp.uri)
        elif launched_as not in (None, launch_tag):
            raise _PairRefusal(f"the launch set says the row was launched as {launched_as!r}; the store's launch tag is {launch_tag}", ls_inp.uri)
        if str(lrow.get("paper_label") or "") != str(obj.paper_label):
            raise _PairRefusal(f"the launch set labels the row {lrow.get('paper_label')!r}, the key's grammar {obj.paper_label!r}", ls_inp.uri)
        # -- a SHARED reader: the row's spelling and selection, the repeat block, the comparison fields and the solo run id the row names (SHARED_SPELLING_RULE /
        #    SHARED_ROW_KEY_RULE / SHARED_RUN_ID_RULE), the row engine's identity on the manifest, the endpoint and RESULT (SHARED_STANDUP_RULE)
        shared_row: Optional[Dict[str, Any]] = None
        if shared_spec is not None:
            shared_row = self._check_shared_row(lrow, ls_inp, tag, selection, okey, rk_okey, repeat, it_none, str(inst_row["instrument_id"]), base, it_repeat=it_repeat)
            shared_row["row_engine"] = self._check_row_engine(man, base, lrow, okey, where_man, shared_row["standup_id"], row_launch_tag=launch_tag_of(tag, "none", legacy_tag), ls_uri=ls_inp.uri)
        if steered:
            if launch_pin_key != obj.staged_id:
                raise _PairRefusal(f"the launch set's pin key {launch_pin_key!r} is not the roster's staged id {obj.staged_id!r} at {op.rung if op else (roster_sel or selection)}", ls_inp.uri)
            if expected_sha != obj.object_id:
                raise _PairRefusal(f"the launch set expects object {str(expected_sha)[:16]}…, the pins name {str(obj.object_id)[:16]}… for {obj.staged_id}", ls_inp.uri)
            launched = (pins_doc.get(mk) or {}).get(obj.staged_id) if isinstance(pins_doc.get(mk), Mapping) else None
            if not isinstance(launched, Mapping) or str(launched.get("object_sha256") or "") != obj.object_id:
                raise _PairRefusal(f"the launched pins ({RECORDS_DIR}/{PINS_FILE}) name {str((launched or {}).get('object_sha256'))[:16]!r} for {obj.staged_id}, the harness pins {str(obj.object_id)[:16]}", pins_inp.uri.rsplit("#", 1)[0])
        elif launch_pin_key is not None or expected_sha is not None:
            raise _PairRefusal(f"the launch set carries a pin key / expected sha ({launch_pin_key!r} / {str(expected_sha)[:16]!r}) for an unsteered or prompted key", ls_inp.uri)
        ident = str(man.get("serving_identity") or "")
        want_ident = STEERED_IDENTITY if steered else PASSTHROUGH_IDENTITY
        if ident != want_ident:
            raise _PairRefusal(f"serving_identity {ident!r} is not the {obj.object_kind} object's {want_ident!r}", where_man)
        want_path = STEERED_PATH_LEGACY if steered else PASSTHROUGH_PATH_LEGACY
        if str(man.get("serving_path") or "") != want_path:
            raise _PairRefusal(f"manifest.json:serving_path {man.get('serving_path')!r} is not the {obj.object_kind} object's {want_path!r}", where_man)
        fitp = man.get("fit_provenance") if isinstance(man.get("fit_provenance"), Mapping) else {}
        sha16 = str(fitp.get("object_sha") or "")
        prov = man.get("provenance") if isinstance(man.get("provenance"), Mapping) else {}
        prov_sha = str(prov.get("artifact_sha256") or "")
        if steered:
            if not _SHA16.match(sha16) or not str(obj.object_id).startswith(sha16):
                raise _PairRefusal(f"manifest.json:fit_provenance.object_sha {sha16!r} is no prefix of the pinned object {str(obj.object_id)[:16]}… (staged {obj.staged_id})", where_man)
            if prov_sha != obj.object_id:
                raise _PairRefusal(f"manifest.json:provenance.artifact_sha256 {prov_sha[:16] or '<none>'}… is not the pinned object {str(obj.object_id)[:16]}… (staged {obj.staged_id})", where_man)
        elif obj.object_kind == "system_instruction":
            sys_sha = str((man.get("resolved_config") or {}).get("system_sha") or "")
            si_sha = str(obj.object_id).split(":", 1)[1]
            if not (_SHA16.match(sys_sha) and si_sha.startswith(sys_sha)):
                raise _PairRefusal(f"manifest.json:resolved_config.system_sha {sys_sha!r} is no prefix of the declared prompt's text sha {si_sha[:16]}…", where_man)
            if sha16 or prov_sha:
                raise _PairRefusal("the prompted store attests an object sha", where_man)
        elif sha16 or prov_sha:
            raise _PairRefusal("the unsteered store attests an object sha", where_man)
        obj_sha = obj.object_id if steered else None
        if shared_row is not None:
            # the per-object served digest on a row engine (SHARED_STANDUP_RULE / SHARED_ENDPOINT_TUPLE_RULE): provenance.artifact_sha256 (checked above) beside the row's map by
            # the MANIFEST's harness key -- the map read from served_geometry.json:endpoint (the manifest's route copy may be null), the tuple the manifest's, never endpoint.cell
            re_blk = shared_row["row_engine"]
            by_harness_key = re_blk["object_artifact_sha256_by_harness_key"]
            hkey = re_blk["manifest_served_harness_key"]
            map_at = f"{GEOMETRY_FILE}:endpoint.object_artifact_sha256_by_cell"
            if steered and str(by_harness_key.get(hkey) or "") != obj_sha:
                raise _PairRefusal(f"{map_at}[{hkey!r}] is {str(by_harness_key.get(hkey))[:16]!r}, not the store's served object {str(obj_sha)[:16]} (provenance.artifact_sha256 = the pinned "
                                   f"object): the row's served-digest map does not name this store's tuple's object ({SHARED_ENDPOINT_TUPLE_RULE})", base["top"][GEOMETRY_FILE].uri)
            if not steered and by_harness_key.get(hkey):
                raise _PairRefusal(f"{map_at} names a served digest {str(by_harness_key.get(hkey))[:16]}... for the unsteered / prompted key {hkey!r} ({SHARED_ENDPOINT_TUPLE_RULE})",
                                   base["top"][GEOMETRY_FILE].uri)
            re_blk["served_map_entry"] = (str(by_harness_key.get(hkey)) if steered else None)
            if not re_blk["endpoint_served_key_agrees"]:
                # the landing-1 shape (ENDPOINT_SERVED_KEY_BUG): the store reads, object_id is DERIVED from the manifest + the endpoint map + RESULT (never recorded by the
                # endpoint's own served-key fields), and the status extra carries the note (the source_status hook)
                re_blk["endpoint_served_key_note"] = (f"{GEOMETRY_FILE}:endpoint.cell names {re_blk['endpoint_served_key']!r} (harness_cell {re_blk['endpoint_served_harness_key']!r}), the row's "
                                                      f"last served key, not this store's {okey!r} / {hkey!r}: {ENDPOINT_SERVED_KEY_BUG}")
        # an instruct-direct row's pin-key ALIAS (`pin_key_alias`, the harness's other spelling of the same staged object, e.g. llama's d1it__ beside itdoc__):
        # recorded; where the launched pins carry it, it must name the same object
        pin_key_alias = lrow.get("pin_key_alias") if instruct else None
        pin_key_alias_pinned: Optional[bool] = None
        if pin_key_alias is not None:
            alias_entry = (pins_doc.get(mk) or {}).get(str(pin_key_alias)) if isinstance(pins_doc.get(mk), Mapping) else None
            pin_key_alias_pinned = isinstance(alias_entry, Mapping)
            if pin_key_alias_pinned and str(alias_entry.get("object_sha256") or "") != obj.object_id:
                raise _PairRefusal(f"the launch set's pin_key_alias {pin_key_alias!r} names object {str(alias_entry.get('object_sha256'))[:16]!r} in the launched pins ({RECORDS_DIR}/{pins_file}), "
                                   f"the row's pin key {obj.staged_id} names {str(obj.object_id)[:16]}: an alias must spell the SAME object", pins_inp.uri.rsplit("#", 1)[0])
        roster_paired_of = None
        if paired and mkey.roster_selections:
            roster, _pins = self._roster_pins()
            blk = ((roster.get("targets") or {}).get(mk) or {}).get("views", {}).get(selection) or {}
            roster_paired_of = blk.get("paired_of")
            if str(roster_paired_of or "") != f"s{seed_n}":
                raise _PairRefusal(f"the roster's selection {selection} says paired_of {roster_paired_of!r}; the tag says s{seed_n}", self.roster_path)
        # -- the gate record (GATE_RULE) and the launched roster entry
        gd = gate_docs.get((base["row_dir"], okey, suffix)) or gate_docs.get((base["row_dir"], okey))
        if gd is None:
            raise _PairRefusal(f"no gate record ({RECORDS_DIR}/{GATE_PREFIX}{launch_tag}_{okey}_{suffix}.json) beside the stores for this PRESENT store; a row files only under a PASS", self._records_dir())
        gate = self._check_gate(gd[0], gd[1], base, okey, tree, obj_sha, ident)
        if shared_row is not None:
            gate.update(self._check_gate_shared(gd[0], gd[1], shared_row, lrow, okey))
        launched = self._check_launched_roster(roster_doc, roster_inp, base["launched_entry"], base["top"][LAUNCHED_ENTRY_FILE], tag, okey, obj, steered, legacy_tag=legacy_tag,
                                               choice=roster_choice)
        # -- the seed rule: the DEFAULT Inspect rule, no replicate block (SEED_AXIS_RULE)
        inspect_seeds = (man.get("seeds") or {}).get("inspect") if isinstance(man.get("seeds"), Mapping) else None
        inspect_seeds = inspect_seeds if isinstance(inspect_seeds, Mapping) else {}
        if isinstance(inspect_seeds.get("replicate"), Mapping):
            raise _PairRefusal(f"manifest.json:seeds.inspect carries a replicate block ({json.dumps(inspect_seeds.get('replicate'))[:120]}); this instrument has no per-seed rule (D69)", where_man)
        seed_rule_text = str(inspect_seeds.get("rule") or "")
        if not seed_rule_text:
            raise _PairRefusal("manifest.json:seeds.inspect.rule names no seed rule", where_man)
        # -- thinking OFF and the cap on every record that spells them
        thinking = man.get("thinking") if isinstance(man.get("thinking"), Mapping) else {}
        by_task_thinking = thinking.get("by_task") if isinstance(thinking.get("by_task"), Mapping) else {}
        if str(thinking.get("mode") or "") != self.THINKING_SPELLING or str(by_task_thinking.get(spec.task) or "") != self.THINKING_SPELLING or thinking.get("enable_thinking") is True:
            raise _PairRefusal(f"manifest.json:thinking (mode {thinking.get('mode')!r}, by_task[{spec.task}] {by_task_thinking.get(spec.task)!r}) is not {self.THINKING_SPELLING!r}", where_man)
        tc = man.get("tinycap") if isinstance(man.get("tinycap"), Mapping) else {}
        item_set_block = tc.get(spec.item_set_key) if isinstance(tc.get(spec.item_set_key), Mapping) else None
        if item_set is None:
            # a 200-item reader: the pinned count alone (the seeded 200-of-N block is recorded on extra.item_set_pin)
            if item_set_block is None or LC.int_or_none(item_set_block.get("n")) != n_items:
                raise _PairRefusal(f"manifest.json:tinycap.{spec.item_set_key} does not pin a {n_items}-item set ({json.dumps(item_set_block)[:160] if item_set_block else '<absent>'})", where_man)
            item_set_extra = None
        else:
            # a FULL reader: the whole attestation (ITEM_SET_RULE) -- the manifest's hashed block, the label on every aggregate, the launch set's row
            item_set_extra = self._check_item_set(item_set, item_set_block, base, lrow, ls_inp, where_man)
        # -- sampling: the protocol row per task, executed as sent
        sampling = man.get("sampling") if isinstance(man.get("sampling"), Mapping) else {}
        if str(sampling.get("mode") or "") != SAMPLING_MODE:
            raise _PairRefusal(f"manifest.json:sampling.mode {sampling.get('mode')!r} is not {SAMPLING_MODE!r}", where_man)
        task_sampling = (sampling.get("by_task") or {}).get(spec.task) if isinstance(sampling.get("by_task"), Mapping) else None
        task_sampling = task_sampling if isinstance(task_sampling, Mapping) else {}
        executed = task_sampling.get("executed") if isinstance(task_sampling.get("executed"), Mapping) else {}
        gs = base["gen_stats"]
        block = {k: gs.get(k) for k in DP.PARAM_KEYS}
        sent = {k: v for k, v in block.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if not executed or not self._numbers_agree(block, {k: executed.get(k) for k in DP.PARAM_KEYS}):
            raise _PairRefusal(f"gen_stats.json sent {RP._numbers(block)}, manifest.json:sampling.by_task[{spec.task}].executed spells {RP._numbers(executed)}", base["files"]["gen_stats.json"].uri)
        if LC.num_or_none(task_sampling.get("protocol_temperature")) is None or abs(float(task_sampling["protocol_temperature"]) - spec.protocol_temperature) > 1e-12 \
                or LC.num_or_none(gs.get("protocol_temperature")) is None or abs(float(gs["protocol_temperature"]) - spec.protocol_temperature) > 1e-12 \
                or LC.num_or_none(gs.get("temperature")) is None or abs(float(gs["temperature"]) - spec.protocol_temperature) > 1e-12:
            raise _PairRefusal(f"the protocol temperature (manifest {task_sampling.get('protocol_temperature')!r}, gen_stats {gs.get('protocol_temperature')!r} / sent {gs.get('temperature')!r}) "
                               f"is not the instrument's {spec.protocol_temperature:g}", where_man)
        served = tc.get("served") if isinstance(tc.get("served"), Mapping) else {}
        clamp = served.get("sampling_clamp_override") if isinstance(served.get("sampling_clamp_override"), Mapping) else None
        if spec.clamp_override and (clamp is None or LC.num_or_none(clamp.get("to")) is None or abs(float(clamp["to"]) - spec.protocol_temperature) > 1e-12):
            raise _PairRefusal(f"manifest.json:tinycap.served.sampling_clamp_override {json.dumps(clamp)} does not attest the served stack's clamp override to {spec.protocol_temperature:g}", where_man)
        tool_calls = bool(served.get("tool_calls")) is True and bool((man.get("route") or {}).get("tool_calls")) is True
        if spec.tool_calls_required and not tool_calls:
            raise _PairRefusal("manifest.json:tinycap.served.tool_calls / route.tool_calls do not attest the tool-call route the tasks are served on", where_man)
        # -- items: the outcome record x rows.jsonl x the aggregates
        items = self._items(base, n_items)
        counts = items["counts"]
        # -- the producer's records: the three-number row (REQUIRED), the controls summary (when present)
        tn_key = f"{tag}/{rk_okey}"
        tn = (three.get("rows") or {}).get(tn_key)
        if not isinstance(tn, Mapping) and legacy_tag is not None:
            # the `main` canary's row may be keyed by the producer's own tag (LEGACY_MAIN_ALIAS_RULE)
            tn_key = f"{legacy_tag}/{okey}"
            tn = (three.get("rows") or {}).get(tn_key)
        if not isinstance(tn, Mapping):
            raise _PairRefusal(f"{spec.three_numbers_file} carries no row for {tag}/{rk_okey}" + (f" (nor {legacy_tag}/{okey})" if legacy_tag else ""), tn_inp.uri)
        bad = self._cross_check_three(tn, base, items, obj_sha, tag, selection, seed_n, paired, okey, tree, n_items, legacy_tag=legacy_tag, instruct=instruct,
                                      spelling=(spec.spelling_of(tag) if shared_row is not None else None), unsteered=is_unsteered_tag(tag), dose_relu=dose_row,
                                      lecun_relu=lecun_relu)
        if shared_row is not None:
            bad += self._cross_check_three_shared(tn, shared_row, lrow)
        if bad:
            raise _PairRefusal(f"{spec.three_numbers_file} row {tn_key} disagrees with the re-derivation on {len(bad)} number(s): " + "; ".join(bad), tn_inp.uri)
        if base.get("controls_summary") is not None:
            bad = self._cross_check_summary(base["controls_summary"], base, items, obj_sha, okey, tree, n_items)
            if bad:
                raise _PairRefusal(f"{CONTROLS_SUMMARY_FILE} disagrees with the store on {len(bad)} fact(s): " + "; ".join(bad), st["summary"].uri)
        # -- decode: the sent numbers against the registered policy row and the instrument's columns
        inst_numbers = {k: inst_row.get(f"decode_{k}") for k in DP.PARAM_KEYS}
        inst_match = "match" if self._numbers_agree(block, inst_numbers) else "mismatch"
        policy, policy_why = self._match_policy(model_id, block)
        # -- the substrate: the Modal volume (the usage ledger's path, held equal to the store's at publish) and the sandbox owner
        ledger_path = str(man.get("usage_ledger") or "")
        want_ledger = f"{VOLUME_HEAD}{rel_store(base['row_dir'], okey, suffix)}/usage_ledger.jsonl"
        if ledger_path != want_ledger:
            raise _PairRefusal(f"manifest.json:usage_ledger {ledger_path!r} is not this store's path on the volume ({want_ledger})", where_man)
        route = man.get("route") if isinstance(man.get("route"), Mapping) else {}
        sandbox = route.get("bcb_sandbox") if isinstance(route.get("bcb_sandbox"), Mapping) else {}
        owner = sandbox.get("owner") if isinstance(sandbox.get("owner"), Mapping) else {}
        substrate = (SUBSTRATE, "derived", LC.confidence_entry("derived", f"manifest.json:usage_ledger on the Modal volume ({VOLUME_HEAD}…, held equal to the store's path at publish); "
                                                                          f"route.bcb_sandbox.provider {sandbox.get('provider')!r}; RESULT's `{RESULT_HEAD}` head"))
        # -- time and identifiers
        when_final = IC.utc_from_iso(base["result"].get("at"))
        if when_final is None:
            raise _PairRefusal(f"RESULT's last line carries no parsable `at=` timestamp ({base['result_last'][:120]})", base["top"]["RESULT"].uri)
        when_start = IC.utc_from_iso(base["launched_entry"].get("launched_at")) or when_final
        done_at = IC.utc_from_epoch(LC.num_or_none(base["done"].get("finished_at")))
        call_id = str(owner.get("function_call_id") or "") or None
        geometry = man.get("geometry") if isinstance(man.get("geometry"), Mapping) else {}
        frameworks = man.get("frameworks") if isinstance(man.get("frameworks"), Mapping) else {}
        endpoint = base["geometry_doc"].get("endpoint") if isinstance(base["geometry_doc"].get("endpoint"), Mapping) else {}
        lora_config = endpoint.get("lora_config") if isinstance(endpoint.get("lora_config"), Mapping) else {}
        standup = {"record": LC.rel_home(base["top"][GEOMETRY_FILE].uri), "engine": endpoint.get("engine"), "serving_path_legacy": base["geometry_doc"].get("serving_path"),
                   "max_lora_rank": LC.int_or_none(lora_config.get("max_lora_rank")), "max_loras": LC.int_or_none(lora_config.get("max_loras")), "lora_target_modules": lora_config.get("target_modules"),
                   "zero_adapter_only": endpoint.get("zero_adapter_only"), "adapter_bank": endpoint.get("adapter_bank"), "attested_max_model_len": LC.int_or_none(endpoint.get("attested_max_model_len")),
                   "attested_max_num_seqs": LC.int_or_none(endpoint.get("attested_max_num_seqs")), "node": endpoint.get("node"), "tool_calls": endpoint.get("tool_calls"),
                   "reasoning_parser": endpoint.get("reasoning_parser"), "sampling_clamp_override": endpoint.get("sampling_clamp_override"), "max_lora_rank_rule": MAX_LORA_RANK_RULE}
        # -- the run (on a SHARED reader: the `_shared` spelling prefixes the producer key and a repeat folds its segment in, SHARED_RUN_ID_RULE)
        instrument_id = str(inst_row["instrument_id"])
        producer_key = shared_producer_key_for(spec, tag, base["store_key"], repeat)
        rid = run_id_for(spec, tag, base["store_key"], okey, instrument_id, repeat=repeat)
        if shared_row is not None and shared_row.get("solo_run_id") is not None and rid == shared_row["solo_run_id"]:
            raise _PairRefusal(f"the shared row's run id {rid} is the solo run's id: the store key is byte-identical to the solo store's and the id must be minted under the `_shared` "
                               f"spelling ({SHARED_RUN_ID_RULE})", where_man)
        n_scored, correct, acc = counts["n_scored"], counts["correct"], counts["acc"]
        bound_lo = correct / n_items
        bound_hi = (correct + counts["n_at_cap_unscored"]) / n_items
        three_block = {"n_attempted": n_items, "n_scored": n_scored, "correct": correct, "acc_over_scored": acc, "acc_over_attempted": bound_lo, "acc_over_attempted_upper_bound": bound_hi,
                       "n_at_cap": counts["n_at_cap"], "share_at_cap": counts["n_at_cap"] / n_items, "n_at_cap_scored": counts["n_at_cap_scored"], "n_at_cap_unscored": counts["n_at_cap_unscored"],
                       "n_unscored": counts["n_unscored"], "n_unscored_below_cap": counts["n_unscored_below_cap"], "n_unparseable": counts["n_unparseable"],
                       "at_cap_rule": "the outcome record's at_cap (stop_reason == max_tokens at the request's cap; never re-tokenized); an at-cap item is SCORED on this task (the code / the call is graded as emitted)",
                       "producer_record": LC.rel_home(tn_inp.uri.rsplit("#", 1)[0]), "producer_record_schema": THREE_NUMBERS_SCHEMA, "producer_record_agrees": True,
                       "producer_contract": three.get("contract"), "producer_seed_reading": three.get("seed_reading"), "producer_delta_value_vs_none": LC.num_or_none(tn.get("delta_value_vs_none")),
                       "unresolved_definition": f"item-wise: an at-cap item the harness left UNSCORED at the one pass's cap ({spec.cap:,}); the pass is the instrument's cap (no cap step), so every such item is at the ceiling"}
        extra: Dict[str, Any] = {
            # a dose-ReLU row carries NO fit seed: its s1 group is a standup grouping, not the object's fit provenance -- the parent fits' sidecars
            # record seed 0 and a dose-scaled copy's fit seed is not applicable (fit_sidecars); run.seed_base is NULL for it already (DOSE_RELU_ROW_RULE)
            # a LeCun-init ReLU row carries the objects' RECORDED fit seed (LECUN_RELU_ROW_RULE: the bank sidecars' seed 0, declared), not the group's standup seed
            "seed_replicate_label": (None if instruct else selection),
            "fit_seed": (LECUN_RELU_FIT_SEED if (lecun_relu and steered) else (seed_n if steered and not arditi and not instruct and not dose_row else None)),
            "seed_axis": {"rule": SEED_AXIS_RULE, "trial_index": 0, "seed_base": None,
                                                                                                                      "decode_seed_policy": INSPECT_SEED_POLICY_DEFAULT,
                                                                                                                      "same_object_on_every_seed_tag": (not steered) or arditi},
            "decode_seed_policy": INSPECT_SEED_POLICY_DEFAULT, "decode_seed_rule_text": seed_rule_text, "final_cap": spec.cap, "token_cap_ladder": [spec.cap], "escalation_state": "final_at_base",
            "store_uri": LC.rel_home(base["dir"]), "store_key": base["store_key"], "store_suffix_legacy": suffix, "producer_key_rule": PRODUCER_KEY_RULE, "group_rule": GROUP_RULE,
            "launch_set": {"record": LC.rel_home(ls_inp.uri.rsplit("#", 1)[0]), "tuple_id": lrow.get("tuple_id"), "class": lclass, "pin_key": launch_pin_key, "object_sha256_expected": expected_sha,
                           "block": lrow.get("block"), "table_index": lrow.get("table_index"), "census_status": lrow.get("census_status"), "launch_tag_legacy": lrow.get("bcbw_wave"),
                           # the addendum's tree bookkeeping (TREES_RULE / RELAUNCH_RULE): the row's own tree or the primary, the relaunch it names, the parked first store
                           **({"sha": row_sha} if row_sha is not None else {}), **({"tree_bound": row_tree, "tree_source": tree_source} if row_tree is not None else {}),
                           **({"relaunched_from": dict(lrow["relaunched_from"])} if isinstance(lrow.get("relaunched_from"), Mapping) else {}),
                           **({"relaunch": {"tag": relaunch_tag, **{k: relaunch_entry.get(k) for k in ("sha", "why")}, "record": LC.rel_home(relaunches_inp.uri.rsplit("#", 1)[0]), "rule": RELAUNCH_RULE,
                                            # a SHARED relaunch's entry spells `shared-r<N>` (its own standup; SHARED_STANDUP_RULE): recorded on those rows alone
                                            **({"entry_tag": relaunch_entry.get("tag"), "standup_id": relaunch_entry.get("standup_id"), "relaunched_standup_id": relaunch_entry.get("relaunched_standup_id"),
                                                "relaunched_object_keys": relaunch_entry.get("cells")} if shared_spec is not None else {}),
                                            # the parked attempts the relaunch superseded (a list; additive, recorded when the entry carries it)
                                            # (the LIST form alone is recorded here: the bare-string form the producer also writes rides the status's gate_records_without_store,
                                            # so the three live rows carrying it -- llama s1p, nemotron s1p -- re-emit byte for byte)
                                            **({"parked_suffix": [str(x) for x in relaunch_entry["parked_suffix"]]} if isinstance(relaunch_entry.get("parked_suffix"), list) else {})}}
                              if relaunch_entry is not None else {}),
                           # the census's own anchor declaration (2026-09-23 14:19Z: the row's own `none`, and the cross-standup fallback a composed row pairs with until it files), carried when present
                           **{k: lrow[k] for k in ("anchor_tuple", "anchor_fallback") if k in lrow},
                           # the instruct-direct row's own declaration (INSTRUCT_DIRECT_RULE), carried on those rows alone
                           **({"transfer_kind": lrow.get("transfer_kind"), "pin_key_alias": pin_key_alias, "pin_key_alias_pinned": pin_key_alias_pinned,
                               "launched_as": lrow.get("launched_as"), "task_tag": ((sm.group("task_tag") or "")[1:] or None),
                               **({"one_task_store_rule": ONE_TASK_STORE_RULE} if sm.group("task_tag") else {})} if instruct else {})},
            "launched_pins_record": LC.rel_home(pins_inp.uri.rsplit("#", 1)[0]), "launched_roster": launched, "gate": gate, "standup": standup, "max_lora_rank_rule": MAX_LORA_RANK_RULE,
            **({"launched_pins_tree_sha": pins_tree, "launched_pins_rule": ROSTER_SET_RULE} if pins_tree is not None else {}),
            "result_line": base["result_last"], "attempts_before_accepted": base["result_earlier"], "done_marker": {"record": LC.rel_home(base["files"]["done"].uri), "finished_utc": done_at,
                                                                                                                   "n_items": LC.int_or_none(base["done"].get("n_items")), "wall_clock_s": LC.num_or_none(base["done"].get("wall_clock_s"))},
            "served_window": LC.int_or_none(geometry.get("max_model_len")), "served_max_num_seqs": LC.int_or_none(geometry.get("max_num_seqs")), "geometry_source_legacy": geometry.get("source"),
            "harness_cell_id_legacy": man.get("harness_cell_id"), "three_numbers": three_block,
            "item_state_counts": {"correct": correct, "incorrect": n_scored - correct, ITEM_LABEL_AT_CAP_UNRESOLVED: counts["n_at_cap_unscored"], ITEM_LABEL_UNSCORED: counts["n_unscored_below_cap"],
                                  "scored_at_cap": counts["n_at_cap_scored"]},
            "at_cap_classes": {AT_CAP_CLASS: counts["n_at_cap_unscored"], "scored_at_cap": counts["n_at_cap_scored"]},
            "item_set_pin": dict(item_set_block), "served_model_id_hf": served_hf, "served_model_revision": rev or None, "serving_identity": ident, "serving_engine": man.get("serving_engine"),
            "serving_path_legacy": man.get("serving_path"), "served_tree_sha": tree, "object_sha16": (sha16 or None), "fit_model_id_hf": fitp.get("model_id"), "fit_transfer": fitp.get("transfer"),
            "fit_kind_legacy": fitp.get("fit_kind"), "fit_status_legacy": fitp.get("status"), "steer_positions_legacy": man.get("steer_positions"),
            "producer_cell_id_legacy": okey, "producer_view_legacy": selection, "producer_tag_legacy": (legacy_tag or tag), "producer_row_dir_legacy": base["row_dir"],
            "inspect_ai_version": frameworks.get("inspect_ai"), "inspect_evals_version": frameworks.get("inspect_evals"), "framework_legacy": gs.get("framework"), "client_lock_sha256": frameworks.get("client_lock_sha256"),
            "sampling_ref_legacy": gs.get("sampling_ref"), "sampling_mode_legacy": sampling.get("mode"), "sampling_source_legacy": sampling.get("source"), "executed_reason_legacy": task_sampling.get("executed_reason"),
            "sampling_clamp_override": clamp, "tool_calls": tool_calls, "decode_policy_ref_legacy": (policy.policy_id if policy else None),
            "decode_policy_reader_match": ("exact by the stated sampling numbers" if policy else policy_why), "generation_state": spec.generation_state, "source_legacy": spec.source,
            "modal": {"function_call_id": call_id, "run_id_legacy": owner.get("run_id"), "sandbox_app": sandbox.get("app_name"), "sandbox_provider": sandbox.get("provider"), "sandbox_image": sandbox.get("image"),
                      "usage_ledger": ledger_path, "volume_head": VOLUME_HEAD},
            "anchor_rule": ANCHOR_RULE,
        }
        if item_set_extra is not None:
            extra["item_set"] = item_set_extra
        if shared_row is not None:
            # the SHARED row's own record (SHARED_SOURCE_RULE): the comparison it declares, the row engine that served it, the solo side it stands against
            extra.update(comparison_kind=shared_row["comparison_kind"], solo_run_id=shared_row["solo_run_id"], solo_store=shared_row["solo_store"], standup_id=shared_row["standup_id"],
                         row_engine=shared_row["row_engine"], shared_source_rule=SHARED_SOURCE_RULE, shared_run_id_rule=SHARED_RUN_ID_RULE,
                         shared_standup={"spelling": spec.spelling_of(tag), "selection": selection, "solo_source": shared_spec.solo_source, "solo_spelling": tag,
                                         "solo_root": (LC.rel_home(self.solo_root) if self.solo_root else None), "repeat": repeat, "row_key": rk_okey,
                                         "launched_as_suffix": suffix, "standup_id": shared_row["standup_id"], "rule": SHARED_STANDUP_RULE},
                         comparison={"comparison_kind": shared_row["comparison_kind"], "solo_run_id": shared_row["solo_run_id"],
                                     "contrast_anchor_kind": (ANCHOR_KIND_SOLO_STAND_IN if shared_row["solo_run_id"] is not None else None),
                                     "state": shared_row["solo_state"], "solo_record": shared_row.get("solo_record"), "value_shared": acc, "value_solo": shared_row.get("solo_value"),
                                     "producer_delta_value_vs_solo": LC.num_or_none(tn.get("delta_value_vs_solo")),
                                     "rule": (SHARED_IT_NONE_RULE if it_none else (SHARED_IT_REPEAT_RULE if it_repeat else SHARED_COMPARISON_RULE))})
            extra["launch_set"].update(spelling=lrow.get("view"), selection=lrow.get("selection"), comparison_kind=lrow.get("comparison_kind"), solo_run_id=lrow.get("solo_run_id"),
                                       solo_run_ids=lrow.get("solo_run_ids"), standup_id=lrow.get("standup_id"), launched_as=lrow.get("launched_as"),
                                       **{k: lrow[k] for k in ("repeat", "recovery", "anchor_note", "row_key_note") if k in lrow})
            if shared_spec.harness_variant is not None:
                # a landing-2 row (SHARED_L2_RUN_ID_RULE): the harness variant that entered the producer key, the base source the bound launch set may spell, the root binding
                extra["shared_standup"].update(harness_variant=shared_spec.harness_variant, producer_spelling=shared_spec.producer_spelling_of(tag), base_source=shared_spec.base_source,
                                               source_binding="root", run_id_rule=SHARED_L2_RUN_ID_RULE)
                if shared_spec.landing_rule is not None:
                    extra["shared_standup"]["landing_rule"] = shared_spec.landing_rule        # landing 3 (SHARED_L3_RULE); absent on landing 2's rows, which stay byte for byte
                extra["launch_set"].update(own_source_spelled=facts.get("launch_set_own_source_spelled"), own_source_is_base=facts.get("launch_set_own_source_is_base"),
                                           own_source_note="the bound launch set's evals.<eval>.own_source is recorded as spelled; the source is the mirror root's (SHARED_L2_RUN_ID_RULE)")
            extra["standup"].update(standup_attempt_id=endpoint.get("standup_attempt_id"), served_object_keys=endpoint.get("cells"), served_harness_keys=endpoint.get("harness_cells"),
                                    residual_slot_armed=(endpoint.get("residual_slot") or {}).get("armed") if isinstance(endpoint.get("residual_slot"), Mapping) else None,
                                    residual_slot_keys=sorted((endpoint.get("residual_slot") or {}).get("cells") or {}) if isinstance(endpoint.get("residual_slot"), Mapping) else None,
                                    object_artifact_sha256_by_harness_key=endpoint.get("object_artifact_sha256_by_cell"), standup_argv=base["geometry_doc"].get("standup_argv"))
        if paired:
            extra.update(paired_of=f"s{seed_n}", seed_tag_legacy=group, roster_paired_of=roster_paired_of)
            if not mkey.roster_selections:
                extra["paired_of_source"] = (f"the tag grammar (s{seed_n}p is the standup replicate of s{seed_n}, D429); the harness roster stages no {selection} selection for {mk} "
                                             "to cross-check (D565)")
            if obj.object_kind == "none":
                # the paired tag's OWN anchor: a standup replicate of the seed tag's `none` at the same seed base, re-served (PAIRED_ANCHOR_RULE); no anchor block of its own
                extra.update(standup_replicate_of_view=group, anchor_rule=PAIRED_ANCHOR_RULE,
                             anchor_role=f"the paired tag's OWN anchor: a standup replicate of {group}/none (D429); the composed row pairs against it once it files")
            else:
                # the cross-tag fallback's spelling; _emit_contrasts re-stamps the row once its own `none` filed (extra.anchor names the anchor used either way)
                extra["cross_standup_anchor"] = f"{group}/none"
        if not mkey.roster_selections:
            # an appendix family: the table's identity and the pin-key route, stamped on ITS rows alone
            extra["appendix_family"] = {"model_key": mk, "model_id": model_id, "served_hf_id": mkey.served_hf_id, "served_revision": mkey.served_revision,
                                        "seed_tags": [f"{mk}_s{k}" for k in mkey.seeds], "rule": APPENDIX_FAMILY_RULE}
            # (`extra.object_resolution` is the run payload's: the resolved object's source text, on every run; the route rides its own key)
            fitted_route = (("harness_roster_instruct_selection" if instruct else "harness_roster_main_selection") if roster_sel is not None
                            else (OBJECT_ROUTE_LAUNCHED_PINS_BY_SHA if pins_by_sha else "launch_set_pin_key"))
            extra["object_route"] = {"route": (fitted_route if steered else ("si_texts" if obj.object_kind == "system_instruction" else "none")),
                                     "pin_key": launch_pin_key, "staged_id": (str(obj.staged_id) if steered else None), "harness_roster_selection": roster_sel,
                                     "source": obj.source, "rule": (f"an instruct-direct row's objects are the harness roster's `{INSTRUCT_SELECTION}` selection's for the family "
                                                                    "(staged id -> pins), and the launch set's pin key must be that staged id (INSTRUCT_DIRECT_RULE)" if instruct else
                                                                    "the `main` canary row's objects are the harness roster's main selection's for the family (staged id -> pins), "
                                                                    "and the launch set's pin key must be that staged id (D565)" if roster_sel is not None else
                                                                    (PINS_KEY_RULE + "; the composed object's sha is the launched entry's, its key the tree pins' entry with that sha, "
                                                                     "and the launch set's pin key must be that key" if (steered and pins_by_sha) else
                                                                     "the harness roster stages no s1 / s1p selection for the appendix families: a fitted key's staged id is the "
                                                                     "launch set's pin key, its sha the harness pins' entry under the model key (D565)"))}
            if steered and pins_by_sha:
                extra["launched_pins_key"] = str(obj.staged_id)
                extra["pins_key_rule"] = PINS_KEY_RULE
        if legacy_tag is not None:
            extra["tag_legacy_original"] = legacy_tag
            extra["legacy_main_alias"] = {"tag_legacy_original": legacy_tag, "alias_tag": tag, "row_dir_legacy_original": base["row_dir"], "launch_tag_legacy": launch_tag,
                                          "launch_set_tag_spelling": lrow.get("tag"), "launch_set_selection_spelling": lrow.get("view"), "launch_set_seed_base": lrow.get("seed_base"),
                                          "launch_set_launch_tag_spelling": lrow.get("bcbw_wave"), "launch_set_launched_as": lrow.get("launched_as"),
                                          "three_numbers_key": tn_key, "rule": LEGACY_MAIN_ALIAS_RULE}
            facts.setdefault("legacy_main_alias_rows", []).append({"tag": tag, "tag_legacy_original": legacy_tag, "row_dir": base["row_dir"], "object_key": okey})
        declared_role = None
        lead = f"{spec.eval_id} thinking OFF, protocol T {spec.protocol_temperature:g}, one pass at {spec.cap:,}, {'instruct-direct' if instruct else 'seed'} tag {tag} {okey}"
        if legacy_tag is not None:
            lead += f" (the family's `{MAIN_SELECTION}` canary {legacy_tag}, row {base['row_dir']}, filed as {tag}/{okey})"
        if shared_row is not None:
            lead += f" (a row-engine store, standup {shared_row['standup_id'][:8]}; {shared_row['comparison_kind']}" + (f", repeat {repeat}" if repeat else "") + ")"
        if instruct:
            # the instruct-direct row (INSTRUCT_DIRECT_RULE / INSTRUCT_ANCHOR_RULE): the fit's checkpoint beside the served one, the anchor it aliases to
            lead += (" (the instruct row's OWN none served by the row engine: a comparison row, not an anchor)" if it_none
                     else (" (the instruct row's OWN none served a second time on the same standup: the row's repeat floor; no solo twin)" if it_repeat
                           else f" (fit on and served on the instruct checkpoint; anchored by alias on {group}/none)"))
            extra["instruct_direct"] = {"tag": tag, "selection": selection, "group": group, "model_key": mk, "model_id": model_id, "served_model_id_hf": served_hf,
                                        "fit_model_id_hf": fitp.get("model_id"), "fit_revision": fitp.get("revision"), "served_revision": rev or None, "fit_transfer": fitp.get("transfer"),
                                        "pin_key": launch_pin_key, "pin_key_alias": pin_key_alias, "pin_key_alias_pinned": pin_key_alias_pinned, "staged_id": str(obj.staged_id),
                                        "harness_roster_selection": roster_sel, "rule": INSTRUCT_DIRECT_RULE}
            if it_none_store:
                extra["instruct_direct"].update(own_none=True, anchor=None, repeat=repeat, rule=(SHARED_IT_NONE_RULE if it_none else SHARED_IT_REPEAT_RULE))
            else:
                extra["anchor_alias"] = {"declared_anchor_tuple": lrow.get("anchor_tuple"), "alias_of": f"{tag}/none", "anchor_tag": group, "anchor_object_key": "none",
                                         "own_none_store": False, "rule": INSTRUCT_ANCHOR_RULE}
        if arditi:
            declared_role = ARDITI_ROLE
            extra.update(fit_seed=None, declared_role=declared_role,
                         single_object={"object_key": okey, "staged_id": str(obj.staged_id), "object_id": obj.object_id, "operating_point": op.rung, "standup_legacy": op.standup_tag,
                                        "c": float(op.coefficient), "unit": op.unit, "declared_role": declared_role, "seed_axis_rule": SEED_AXIS_RULE,
                                        "coexist_ruling_needed": (f"a roster coexist ruling over lane_sources_legacy [{spec.source}] x campaigns [{CAMPAIGN}] x model_ids [{model_id}] x object_ids "
                                                                  f"[{obj.object_id}] x eval_ids [{spec.eval_id}] x instrument_ids [{instrument_id}] x views_legacy "
                                                                  f"[{', '.join(f'{mk}_s{k}' for k in mkey.seeds)}] keeps the three rows selected together (the user's, never a reader default)")})
            lead += f" (ONE object on every seed tag, {op.rung} c = {op.coefficient:g} {op.unit}; declared {declared_role})"
        elif not steered and len(mkey.seeds) > 1:
            # the same object on every seed tag of a multi-seed family; a one-seed family (the appendix) has no twin tag and needs no coexist ruling here
            extra["same_object_rows"] = {"twin_tags_legacy": [f"{mk}_s{k}" for k in mkey.seeds if k != seed_n], "rule": SEED_AXIS_RULE,
                                         "coexist_ruling_needed": f"a roster coexist ruling over lane_sources_legacy [{spec.source}] x the seed tags keeps the rows of one object selected together"}
        notes = (f"{lead}: {correct}/{n_scored} correct over scored = {acc:.4f}" if acc is not None else f"{lead}: no scored item")
        notes += (f"; {counts['n_at_cap']}/{n_items} at the cap ({counts['n_at_cap_scored']} scored at the cap, {counts['n_at_cap_unscored']} unresolved)"
                  + (f", {counts['n_unscored_below_cap']} unscored below the cap" if counts["n_unscored_below_cap"] else "")
                  + f"; correct/{n_items} = {bound_lo:.3f} is the BOUND, not the value")
        try:
            payload = LC.run_payload_base(seeds, run_id=rid, producer_key=producer_key, object_key=okey, view_legacy=spec.spelling_of(tag), cell_id_legacy=okey, obj=obj, eval_id=spec.eval_id,
                                          instrument_row=inst_row, instrument_seed_id=spec.instrument_seed, lifecycle_state="generated", when_utc=when_final,
                                          when_source=f"RESULT's last line ({RESULT_COMPLETE}) `at=` for generated; {LAUNCHED_ENTRY_FILE}:launched_at for queued / started",
                                          token_cap_tokens=spec.cap, judge_id=None, judge_seed_id=None, substrate=substrate, lane_source_legacy=spec.source,
                                          store_uri=LC.rel_home(base["dir"]), instrument_match=inst_match, extra=extra, notes=notes)
        except KeyError as exc:
            raise _PairRefusal(f"run row: {exc}", where_man)
        if item_set is not None and str(payload.get("item_set_id")) != item_set.id:
            raise _PairRefusal(f"instrument seed {spec.instrument_seed} binds item set {payload.get('item_set_id')!r} for {spec.eval_id}; this reader files the pinned set "
                               f"{item_set.id} (a new item set is a NEW instrument, never the 200-item one)", where_man)
        fc = payload["field_confidence_json"]
        if shared_row is not None:
            # SHARED_ENDPOINT_TUPLE_RULE: the object's confidence, and whether the endpoint's own served-key fields moved it (payload["extra"]["row_engine"] IS shared_row's block)
            re_blk = payload["extra"]["row_engine"]
            baseline = str(fc["object_id"]["confidence"])
            re_blk["object_id_confidence"] = {"baseline": baseline, "filed": baseline, "downgraded": False}
            if re_blk.get("endpoint_served_key_note"):
                # the served object is DERIVED (the manifest's tuple, the endpoint map's entry for its harness key, RESULT's tokens), not recorded by the endpoint, whose own
                # served-key fields name the row's last key on this store (ENDPOINT_SERVED_KEY_BUG): one step down from recorded; an unsteered key's is derived already
                filed = "derived" if baseline == "recorded" else baseline
                fc["object_id"] = LC.confidence_entry(filed, (f"the served tuple is the manifest's (cell_id {okey} = the store key's object = the store directory; harness_cell_id {hkey}); "
                                                              f"{GEOMETRY_FILE}:endpoint.object_artifact_sha256_by_cell[{hkey!r}] names "
                                                              + (f"the pinned object {str(obj_sha)[:16]}... (= provenance.artifact_sha256)" if steered else "no entry (an unsteered / prompted key)")
                                                              + (f"; RESULT's standup= token ({re_blk['result_standup_source']}) is the manifest's row_engine.standup_id and its last line is the "
                                                                 f"verifier's re-stamp, the three-number row's for {tag}/{rk_okey}; " if re_blk.get("verification_state") == VERIFICATION_STATE_VERIFIED
                                                                 else f"; RESULT's standup= token is the manifest's row_engine.standup_id and its line is the three-number row's for {tag}/{rk_okey}; ")
                                                              + f"{re_blk['endpoint_served_key_note']}; "
                                                              + (f"downgraded from {baseline}" if filed != baseline else f"kept at {baseline} (an unsteered key resolves by its grammar, not by the endpoint)")
                                                              + f"; the object's own resolution: {obj.source}"))
                re_blk["object_id_confidence"] = {"baseline": baseline, "filed": filed, "downgraded": filed != baseline}
                payload["extra"]["endpoint_served_key_note"] = re_blk["endpoint_served_key_note"]      # (run_payload_base copied `extra`: the payload's dict is the one that files)
        if it_none_store:
            # the shared instruct row's own none (SHARED_IT_NONE_RULE; its repeat, SHARED_IT_REPEAT_RULE): no fit, so nothing to derive -- the payload's transfer_kind is the
            # unsteered object's (`none`)
            extra["instruct_direct"].update(transfer_kind_recorded=payload.get("transfer_kind"), transfer_kind_note="an unsteered row: no fit checkpoint, transfer_kind none (object_kind=none)")
        elif instruct:
            # transfer_kind is DERIVED, never assumed (INSTRUCT_DIRECT_RULE): the object's fit checkpoint (manifest.json:fit_provenance.model_id) against the served
            # one -- the same checkpoint -> instruct_direct (derive.transfer_kind) -- and legacy_spellings.yaml's spelling of the tag must agree with it
            fit_hf = str(fitp.get("model_id") or "")
            if fit_hf != served_hf:
                raise _PairRefusal(f"manifest.json:fit_provenance.model_id {fit_hf or '<none>'!r} is not the served checkpoint {served_hf!r}: an instruct-direct row's object is fit on the "
                                   f"model it is served on (transfer_kind {INSTRUCT_TRANSFER_KIND} is derived from the two, never assumed)", where_man)
            if fitp.get("transfer") is True:
                raise _PairRefusal(f"manifest.json:fit_provenance.transfer is true: the object was transferred from another checkpoint, which is not an instruct-direct fit", where_man)
            fit_model_id = seeds.model_for_hf_id(fit_hf)
            derived = DV.transfer_kind(fit_model_id, model_id, str(obj.object_kind), run_checkpoint_kind=(seeds.models.get(model_id) or {}).get("checkpoint_kind"))
            recorded_tk = payload.get("transfer_kind")
            if derived[0] != INSTRUCT_TRANSFER_KIND or recorded_tk != INSTRUCT_TRANSFER_KIND:
                raise _PairRefusal(f"transfer_kind: derived {derived[0]!r} from fit model {fit_model_id!r} vs served {model_id!r} ({derived[2]}); legacy_spellings.yaml spells {tag} as "
                                   f"{recorded_tk!r}; an instruct-direct row files only when both read {INSTRUCT_TRANSFER_KIND}", where_man)
            payload["transfer_kind"] = derived[0]
            fc["transfer_kind"] = LC.confidence_entry("derived", f"derive.transfer_kind(fit model {fit_model_id} = the registry model of manifest.json:fit_provenance.model_id {fit_hf}, "
                                                                 f"served model {model_id}): {derived[2]}; fit_provenance.transfer {fitp.get('transfer')!r}; legacy_spellings.yaml "
                                                                 f"model_keys[{tag}].transfer_kind agrees ({recorded_tk})")
            extra["instruct_direct"].update(fit_model_id=fit_model_id, transfer_kind_derived=derived[0], transfer_kind_derivation=derived[2], transfer_kind_recorded=recorded_tk)
        payload.update(producer_session=PRODUCER_SESSION, trial_index=0, seed_base=None, queued_utc=when_start, started_utc=when_start,
                       n_items_planned=n_items, n_items_generated=n_items, n_items_judged=n_scored, model_revision=(rev if _SHA1.match(rev) else None),
                       intervention_hash=man.get("intervention_hash"), harness_config_hash=man.get("harness_config_hash"),
                       env_digest=(next(iter(base["consts"].get("env_digest") or []), None)), modal_call_id=call_id, job_name=base["result"].get("name"),
                       node_hostname=(endpoint.get("node") or None), serving_identity=ident, token_cap_rung_index=0, token_cap_ladder_json=[spec.cap],
                       decode_seed_rule=DECODE_SEED_RULE, decode_sent_json=sent, decode_temperature_sent=sent.get("temperature"), decode_top_p_sent=sent.get("top_p"),
                       decode_top_k_sent=(int(sent["top_k"]) if sent.get("top_k") is not None else None), code_sha=tree,
                       context_window_tokens=LC.int_or_none(geometry.get("max_model_len")),
                       store_completion_artifact_uri=base["files"]["gen_stats.json"].uri, store_completion_sha256=base["files"]["gen_stats.json"].sha256)
        if LC.int_or_none(geometry.get("max_num_seqs")) is not None:
            payload.update(served_max_num_seqs=LC.int_or_none(geometry.get("max_num_seqs")), served_max_num_seqs_source="attested")
            fc["served_max_num_seqs"] = LC.confidence_entry("recorded", f"manifest.json:geometry.max_num_seqs (source {geometry.get('source')})")
        if mkey.unsteered_row:
            # a family with an unsteered row (UNSTEERED_ROW_RULE / WINDOW_ANCHOR_RULE): the serving window rides every row of the family (the baseline is picked by it),
            # and a u0 row carries its declaration; stamped on THESE rows alone so every other family's record stays byte for byte
            # (payload["extra"], not the local `extra`: run_payload_base copied it, and the payload's dict is the one that files)
            payload["extra"]["serving_window"] = {"max_model_len": LC.int_or_none(geometry.get("max_model_len")), "max_num_seqs": LC.int_or_none(geometry.get("max_num_seqs")),
                                                  "window_segment": window_of_suffix(spec, suffix), "task_tag": suffix_task_for(spec, suffix), "rule": WINDOW_ANCHOR_RULE}
            if is_unsteered_tag(tag):
                payload["extra"]["unsteered_row"] = {"rule": UNSTEERED_ROW_RULE, "selection": UNSTEERED_SELECTION, "row_key": rk_okey, "anchor_tuple": lrow.get("anchor_tuple"),
                                                     "window_reserve": (dict(lrow["window_reserve"]) if isinstance(lrow.get("window_reserve"), Mapping) else None)}
            elif dose_relu:
                # the dose-0.75 random-init ReLU row (DOSE_RELU_ROW_RULE): its declaration rides its rows alone
                payload["extra"]["dose_relu_row"] = {"rule": DOSE_RELU_ROW_RULE, "selection": DOSE_RELU_SELECTION, "row_key": rk_okey, "anchor_tuple": lrow.get("anchor_tuple"),
                                                     "pin_key": lrow.get("pin_key")}
            elif dose_lora:
                # the dose-1.25 difference-LoRA row (DOSE_LORA_ROW_RULE): its declaration rides its row alone
                payload["extra"]["dose_lora_row"] = {"rule": DOSE_LORA_ROW_RULE, "selection": DOSE_LORA_SELECTION, "row_key": rk_okey, "anchor_tuple": lrow.get("anchor_tuple"),
                                                     "pin_key": lrow.get("pin_key"), "strength_nominal": DOSE_LORA_STRENGTH}
            elif lecun_relu:
                # the LeCun-init ReLU row (LECUN_RELU_ROW_RULE): its declaration rides its rows alone -- the recorded fit seed beside the standup seed the launch row spells
                payload["extra"]["lecun_relu_row"] = {"rule": LECUN_RELU_ROW_RULE, "selection": LECUN_RELU_SELECTION, "row_key": rk_okey, "anchor_tuple": lrow.get("anchor_tuple"),
                                                      "pin_key": lrow.get("pin_key"), "fit_seed_recorded": LECUN_RELU_FIT_SEED, "standup_seed": seed_n}
        if policy is not None:
            payload["decode_source"] = policy.decode_source
            fc["decode_source"] = LC.confidence_entry("recorded", f"the registered policy row {policy.policy_id} the sent numbers equal (its decode_source)")
        else:
            fc["decode_source"] = LC.unverified(policy_why)
        fc["code_sha"] = LC.confidence_entry("recorded", "manifest.json:producer.tree_sha = RESULT's sha = the outcome record's producer_tree_sha = the gate record's sha")
        fc["model_revision"] = LC.confidence_entry("recorded", "manifest.json:model_revision (= models.yaml hf_revision)")
        fc["modal_call_id"] = (LC.confidence_entry("recorded", "manifest.json:route.bcb_sandbox.owner.function_call_id (the Modal function call that owned the store's sandboxes)") if call_id
                               else LC.unverified("manifest.json:route.bcb_sandbox.owner names no function call id"))
        fc["node_hostname"] = (LC.confidence_entry("recorded", f"{GEOMETRY_FILE}:endpoint.node (the standup's attested endpoint)") if endpoint.get("node")
                               else LC.unverified(f"{GEOMETRY_FILE}:endpoint names no node"))
        fc["token_cap_tokens"] = LC.confidence_entry("recorded", f"gen_stats.json:max_tokens = the outcome record's max_tokens = the summary's = {spec.cap} (one pass, no cap step)")
        fc["seed_base"] = LC.confidence_entry("derived", SEED_AXIS_RULE)
        fc["trial_index"] = LC.confidence_entry("derived", SEED_AXIS_RULE)
        fc["decode_seed_rule"] = LC.confidence_entry("derived", f"none: the greedy protocol (T {spec.protocol_temperature:g}) pins no seed — the instrument's and the protocol policy row's rule; "
                                                                f"Inspect's DEFAULT per-item derivation ({seed_rule_text}; k = 1, no seed base) is recorded on extra.decode_seed_rule_text / extra.seed_axis, "
                                                                "inert under the protocol")
        fc["decode_sent_json"] = LC.confidence_entry("recorded", f"gen_stats.json[{spec.task}] (the client's sampler; manifest.json:sampling.by_task[{spec.task}].executed agrees; the store's resolved_sampler.json is registered beside it)")
        fc["instrument_match"] = LC.confidence_entry("derived", f"the sent numbers compared to instrument seed {spec.instrument_seed}'s decode columns (bound to {model_id}): {inst_match}")
        fc["harness_version"] = LC.confidence_entry("derived", f"instrument seed {spec.instrument_seed}; the store's frameworks (inspect_ai {frameworks.get('inspect_ai')}, inspect_evals {frameworks.get('inspect_evals')}) ride extra_json")
        fc["context_window_tokens"] = LC.confidence_entry("recorded", "manifest.json:geometry.max_model_len")
        fc["n_items_planned"] = LC.confidence_entry("derived", f"the instrument's item set (n_items {n_items}); manifest.json:tinycap.{spec.item_set_key}.n, gen_stats.json:n and the outcome record's n_attempted_items agree")
        fc["n_items_generated"] = LC.confidence_entry("recorded", "the outcome record: every attempted item has a stop reason; done/<task>.json n_rows agrees")
        fc["n_items_judged"] = LC.confidence_entry("derived", "the SCORED items (rule-scored at generation; an at-cap item is scored on this task)")
        fc["queued_utc"] = fc["started_utc"] = LC.confidence_entry("recorded", f"{LAUNCHED_ENTRY_FILE}:launched_at (the launcher's stamp)")
        fc["generated_utc"] = LC.confidence_entry("recorded", f"RESULT's last line ({RESULT_COMPLETE}): its `at=` timestamp")
        fc["thinking_mode"] = LC.confidence_entry("recorded", f"gen_stats.json:thinking = manifest.json:thinking.by_task[{spec.task}] = the outcome record's thinking = off")
        fc["substrate"] = substrate[2]
        if steered and man.get("steer_positions") in V.APPLIED_POSITIONS_MODES:
            payload["applied_positions_mode"] = str(man["steer_positions"])
            fc["applied_positions_mode"] = LC.confidence_entry("recorded", "manifest.json:steer_positions")
        if not steered:
            payload["strength_unit"] = "unverified"
            fc["strength_unit"] = LC.unverified("an unsteered / prompted object key: no object on the served path, so no dose and no unit (the column default)")
        if arditi:
            payload.update(op.strength_columns())
            fc.update(op.strength_confidence(f"the launch set's pin key {op.staged_id} (the all-positions pins' staged object at the operating point)"))
            payload["extra"].update(op.extra())
            op_mv = seeds.model_for_view_legacy(op.standup_tag)
            if op_mv.get("transfer_kind") in V.TRANSFER_KINDS:
                payload["transfer_kind"] = op_mv["transfer_kind"]
                fc["transfer_kind"] = LC.confidence_entry("recorded", f"legacy_spellings.yaml model_keys[{op.standup_tag}].transfer_kind (the direction is fit on and served on the instruct "
                                                                      f"checkpoint: manifest.json:fit_provenance.transfer {fitp.get('transfer')!r})")
        records.append(("run", payload))
        facts["runs"] += 1
        # NO state event: the record's `generated` is the state and the harvester's rule-scored completion stamps `ingested`
        comp_id = spec.recount_computation_id()
        code_sha = str(producer.get("code_sha") or "")
        records.append(("computation", {"computation_id": comp_id, "algorithm": spec.recount_algorithm, "version": RECOUNT_VERSION, "seed_provenance": spec.recount_seed_provenance,
                                        "code_sha": (code_sha if _SHA1.match(code_sha) else "unverified"),
                                        "notes": f"correct / scored recounted from the store's outcome record over the {n_items} attempted items (one pass at {spec.cap:,}, no cap step)"}))
        rows_uri = LC.rel_home(base["files"]["rows.jsonl"].uri)
        lo, hi = IC.wilson(correct, n_scored) if n_scored else (None, None)
        records.append(("run_metric", {"run_id": rid, "judgment_id": "", "metric_id": HEADLINE_METRIC, "threshold": -1.0, "computation_id": comp_id, "producer_metric_spelling": "value",
                                 "value": acc, "k": correct, "n": n_scored, "ci_lo": lo, "ci_hi": hi, "ci_method": ("wilson_95" if n_scored else None), "n_abstentions": None,
                                 "n_incomplete": counts["n_unscored"], "computed_by": "registry", "computed_utc": when_final, "source_uri": rows_uri,
                                 "extra": {"role": "THE VALUE: correct over SCORED (the user's rule: degeneration is scored separately)", "recounted_from": "the store's outcome record",
                                           "equals_gen_stats_acc": True, "n_at_cap_scored_counted": counts["n_at_cap_scored"], "n_unscored": counts["n_unscored"],
                                           "producer_record": three_block["producer_record"], "producer_value": LC.num_or_none(tn.get("value"))}}))
        records.append(("run_metric", {"run_id": rid, "judgment_id": "", "metric_id": HEADLINE_METRIC, "threshold": -1.0, "computation_id": "", "producer_metric_spelling": BOUND_SPELLING,
                                 "value": bound_lo, "k": correct, "n": n_items, "ci_lo": bound_lo, "ci_hi": bound_hi, "ci_method": BOUND_CI_METHOD, "n_abstentions": None,
                                 "n_incomplete": counts["n_at_cap_unscored"], "computed_by": "producer", "computed_utc": when_final, "source_uri": three_block["producer_record"],
                                 "extra": {"role": f"THE BOUND, never the value: correct over the {n_items} attempted; the interval's lower end counts the unscored at-cap items wrong, the upper end right",
                                           "n_at_cap_unscored": counts["n_at_cap_unscored"], "n_unscored_below_cap_counted_as_miss": counts["n_unscored_below_cap"],
                                           "unresolved_definition": three_block["unresolved_definition"], "producer_spellings": {BOUND_SPELLING: LC.num_or_none(tn.get(BOUND_SPELLING))},
                                           "metric_role_expected": "secondary (a blank computation_id ranks after the value's computed row at one identity)"}}))
        # -- items (whole rows): value 1 / 0, or NULL with the state's label
        rows_sha = base["files"]["rows.jsonl"].sha256
        for iid in sorted(items["items"]):
            it = items["items"][iid]
            if it["scored"]:
                label, cls = None, None
            elif it["at_cap"]:
                label, cls = ITEM_LABEL_AT_CAP_UNRESOLVED, AT_CAP_CLASS
            else:
                label, cls = ITEM_LABEL_UNSCORED, UNSCORED_BELOW_CAP_CLASS
            row = base["scored"].get(iid)
            created = IC.utc_from_epoch(row[2]) if (row is not None and row[2] is not None) else None
            iextra = {"final_cap": spec.cap, "stop_reason": it["stop_reason"], "completion_tokens": it["completion_tokens"], "decode_seed": it["seed"],
                      "at_cap_class": (cls if cls == AT_CAP_CLASS else None), "scored_at_cap": bool(it["scored"] and it["at_cap"]),
                      "unscored_reason": (UNSCORED_BELOW_CAP_CLASS if cls == UNSCORED_BELOW_CAP_CLASS else None), "unparseable": it.get("unparseable")}
            records.append(("item_result", IC.item_result_payload(rid, iid, HEADLINE_METRIC, it["value"], label, created or when_final, judgment_id="", trial_index=0, extra=iextra)))
            records.append(("item_flags", IC.item_flags_payload(rid, iid, (rows_sha if row is not None else None), (row[1] if row is not None else None), at_token_cap=(1 if it["at_cap"] else 0),
                                                                finish_reason=it["stop_reason"], response_tokens=it["completion_tokens"], decode_seed=(str(it["seed"]) if it["seed"] is not None else None),
                                                                judgment_id="", trial_index=0,
                                                                extra={"at_token_cap_source": three_block["at_cap_rule"], "generation_pointer": "rows.jsonl (sha, line): the scored `acc` row of the item",
                                                                       "outcome_record": LC.rel_home(base["files"]["outcomes"].uri)})))
        # -- the run-level share (run_degeneration over the 200 attempted items): the at-cap count alone; no loop reading exists
        measured = {RAW_AT_CAP_MEASURE: dict(numerator=counts["n_at_cap"], denominator=n_items, unit="items", threshold_params={"cap_final": spec.cap, "ladder": [spec.cap]}, computed_by="registry",
                                             extra={"rule": three_block["at_cap_rule"], "classes": {"scored_at_cap": counts["n_at_cap_scored"], AT_CAP_CLASS: counts["n_at_cap_unscored"]},
                                                    "source": LC.rel_home(base["files"]["outcomes"].uri), "producer_at_cap_n": LC.int_or_none(tn.get("at_cap_n"))})}
        notes_nm = {"classifier": "no classifier reading exists for these stores", "contentless": "no contentless reading exists for these stores",
                    "rep_ratio_loop": "no rep-ratio / loop classification exists for these stores (no cap-ladder record; nothing is invented)",
                    "zlib_collapse": "no zlib reading exists for these stores", "union5": "the five-detector union is not computed for these stores",
                    IC.AT_CAP_RETOKENIZED: "the outcome record reads the provider's stop reason, never a re-tokenized length"}
        for drow in IC.declared_degeneration_rows(rid, "", measured, source_uri=LC.rel_home(base["files"]["outcomes"].uri), computed_utc=when_final, not_measured_notes=notes_nm):
            records.append(("run_degeneration", drow))
        facts["rows"][f"{tag}|{rk_okey}"] = {"run_id": rid, "tag": tag, "selection": selection, "paper_label": obj.paper_label, "object_id": obj.object_id, "staged_id": obj.staged_id,
                                             "instrument_id": instrument_id, "acc_over_scored": acc, "correct": correct, "n_scored": n_scored, "acc_over_attempted": bound_lo, "upper_bound": bound_hi,
                                             "n_at_cap": counts["n_at_cap"], "share_at_cap": counts["n_at_cap"] / n_items, "n_at_cap_unscored": counts["n_at_cap_unscored"],
                                             "n_unscored": counts["n_unscored"], "instrument_match": inst_match, "decode_policy": (policy.policy_id if policy else None),
                                             "tree": tree, "store_key": base["store_key"], "producer_key": producer_key, "paired": paired, "gate_verdict": gate["verdict"],
                                             "serving_engine_attested": gate["serving_engine_attested"], "max_lora_rank_recorded": gate["max_lora_rank"], "zero_adapter_only_recorded": gate["zero_adapter_only"],
                                             "declared_role": declared_role, "steered": steered, "modal_call_id": call_id}
        if legacy_tag is not None:
            facts["rows"][f"{tag}|{rk_okey}"]["tag_legacy_original"] = legacy_tag
        if instruct and not it_none_store:
            facts["rows"][f"{tag}|{rk_okey}"].update(instruct_direct=True, transfer_kind=payload["transfer_kind"], anchor_alias_of=f"{group}/none")
        out = {"run_id": rid, "items": items["items"], "steered": steered, "when": when_final, "declared_role": declared_role, "tag": tag, "paired": paired, "okey": okey,
               "payload": payload, "records": records, "instruct": instruct,
               # the window-matched anchor's inputs (WINDOW_ANCHOR_RULE): the row's serving window and the baseline the launch set names for it
               "window": payload.get("context_window_tokens"), "max_num_seqs": payload.get("served_max_num_seqs"), "anchor_tuple": lrow.get("anchor_tuple"),
               "unsteered": is_unsteered_tag(tag), "unsteered_family": bool(mkey.unsteered_row), **({"dose_relu": True} if dose_relu else {}), **({"dose_lora": True} if dose_lora else {}),
               **({"lecun_relu": True} if lecun_relu else {})}
        if shared_row is not None:
            facts["rows"][f"{tag}|{rk_okey}"].update(spelling=spec.spelling_of(tag), comparison_kind=shared_row["comparison_kind"], solo_run_id=shared_row["solo_run_id"],
                                                     standup_id=shared_row["standup_id"], repeat=repeat, it_none=it_none, it_repeat=it_repeat, comparability_class=gate.get("comparability_class"),
                                                     endpoint_served_key_agrees=shared_row["row_engine"]["endpoint_served_key_agrees"],
                                                     max_loras_recorded=shared_row["row_engine"]["max_loras"], residual_slot_armed_recorded=shared_row["row_engine"]["residual_slot_armed"])
            if not shared_row["row_engine"]["endpoint_served_key_agrees"]:
                # the source_status note (SHARED_ENDPOINT_TUPLE_RULE; the status hook): filed for a store that FILED, once every refusal point is behind it
                re_blk = shared_row["row_engine"]
                facts["endpoint_served_key_mismatch"].append({"row": f"{tag}|{rk_okey}", "run_id": rid, "store": rel_store(base["row_dir"], okey, suffix), "standup_id": shared_row["standup_id"],
                                                              "manifest_key": okey, "manifest_harness_key": re_blk["manifest_served_harness_key"], "endpoint_key": re_blk["endpoint_served_key"],
                                                              "endpoint_harness_key": re_blk["endpoint_served_harness_key"], "row_last_key": re_blk["engine_object_keys"][-1],
                                                              "served_map_entry": re_blk.get("served_map_entry"), "downgraded_field": "object_id", "note": ENDPOINT_SERVED_KEY_BUG,
                                                              "rule": SHARED_ENDPOINT_TUPLE_RULE})
            standup_fact = facts["standups"].setdefault(shared_row["standup_id"], {"launched_as_suffixes": [], "rows": []})
            standup_fact["rows"].append(f"{tag}|{rk_okey}")
            if suffix not in standup_fact["launched_as_suffixes"]:
                standup_fact["launched_as_suffixes"] = sorted(standup_fact["launched_as_suffixes"] + [suffix])
            out.update(rk_okey=rk_okey, repeat=repeat, it_none=it_none, it_repeat=it_repeat, instrument_id=instrument_id, tn=tn, shared=shared_row, comparison_kind=shared_row["comparison_kind"],
                       solo_run_id=shared_row["solo_run_id"], standup_id=shared_row["standup_id"], acc_over_scored=acc)
        return out

    # ---- the shared-standup comparison sources (SHARED_SOURCE_RULE) -------------------------------
    def _shared_standup_conflicts(self, stores: Mapping[Tuple[str, str, str], Mapping[str, Any]], facts: Dict[str, Any]) -> Dict[Tuple[str, str], str]:
        """One standup id per row launch (:data:`SHARED_STANDUP_RULE`): every store the same ``(row dir, suffix)`` names must spell the same
        ``manifest.json:row_engine.standup_id``; a launch whose stores disagree refuses each of them by name (the message per ``(row dir, suffix)``)."""
        by_launch: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
        for (_tag, okey, suffix), st in stores.items():
            files = next(iter(st["keys"].values()), {}) if len(st["keys"]) == 1 else {}
            inp = files.get("manifest.json")
            sid: Optional[str] = None
            if inp is not None:
                try:
                    man = self._json(inp) or {}
                except (OSError, ValueError):
                    man = {}
                blk = man.get("row_engine") if isinstance(man.get("row_engine"), Mapping) else {}
                sid = str(blk.get("standup_id")) if blk.get("standup_id") is not None else None
            by_launch.setdefault((st["row_dir"], suffix), {}).setdefault(sid or "<none>", []).append(okey)
        out: Dict[Tuple[str, str], str] = {}
        for (row_dir, suffix), ids in sorted(by_launch.items()):
            if len(ids) > 1:
                named = "; ".join(f"{sid[:8]}: {', '.join(sorted(oks))}" for sid, oks in sorted(ids.items()))
                out[(row_dir, suffix)] = (f"the stores of the row launch {row_dir}/{suffix} spell {len(ids)} standup ids ({named}); one engine bring-up serves a row launch, so every "
                                          f"store it names must carry one standup_id before any of them files ({SHARED_STANDUP_RULE})")
        facts["standup_launches"] = {f"{rd}/{sfx}": sorted(ids) for (rd, sfx), ids in sorted(by_launch.items())}
        return out

    def _check_shared_row(self, lrow: Mapping[str, Any], ls_inp: Input, tag: str, selection: str, okey: str, rk_okey: str, repeat: Optional[int], it_none: bool,
                          instrument_id: str, base: Mapping[str, Any], it_repeat: bool = False) -> Dict[str, Any]:
        """A SHARED launch row's own fields against the store (:data:`SHARED_SPELLING_RULE`, :data:`SHARED_ROW_KEY_RULE`, :data:`SHARED_RUN_ID_RULE`): the
        registry spelling and the selection, the repeat block, the comparison kind and the solo run id / solo store it names -- the id must be what the
        SOLO reader mints for the solo store -- and the standup id; then the solo side (:meth:`_solo_side`). The comparison kind follows the repeat segment
        first (:data:`SHARED_IT_REPEAT_RULE`): a ``-rep<N>`` store is repeat_floor whatever its tag, and the instruct row's repeat (``it_repeat``) names no
        solo run and no solo store (no solo twin: ``solo_state`` :data:`SOLO_STATE_NO_TWIN`), the way its first serving (``it_none``) names none. Raises
        :class:`_PairRefusal` by name."""
        spec = self.SPEC
        sh = spec.shared
        assert sh is not None
        where = ls_inp.uri
        want_spelling = sh.spelling_of(tag)
        if str(lrow.get("view") or "") != want_spelling:
            raise _PairRefusal(f"the launch set's row spells view {lrow.get('view')!r}; a shared row of {tag} files under the registry spelling {want_spelling!r} ({SHARED_SPELLING_RULE})", where)
        if str(lrow.get("selection") or "") != selection:
            raise _PairRefusal(f"the launch set's row spells selection {lrow.get('selection')!r}, not this store's {selection!r} ({SHARED_SPELLING_RULE})", where)
        tuple_id = str(lrow.get("tuple_id") or "")
        if repeat is not None:
            rep = lrow.get("repeat") if isinstance(lrow.get("repeat"), Mapping) else None
            if rep is None:
                raise _PairRefusal(f"the launch set's row for the repeat store {base['suffix']!r} carries no repeat block ({{of, index, key}}) ({SHARED_ROW_KEY_RULE})", where)
            got = (str(rep.get("of") or ""), LC.int_or_none(rep.get("index")), str(rep.get("key") or ""))
            if got != (okey, repeat, rk_okey) or not tuple_id.endswith(f"/{rk_okey}"):
                raise _PairRefusal(f"the launch set's repeat row (repeat {got!r}, tuple_id {tuple_id!r}) is not this store's ({okey}, {repeat}, {rk_okey}) ({SHARED_ROW_KEY_RULE})", where)
        elif not tuple_id.endswith(f"/{okey}"):
            raise _PairRefusal(f"the launch set's row tuple_id {tuple_id!r} does not name this store's key {okey!r} ({SHARED_ROW_KEY_RULE})", where)
        ckind = str(lrow.get("comparison_kind") or "")
        want_kind = COMPARISON_KIND_REPEAT_FLOOR if repeat is not None else (COMPARISON_KIND_IT_NONE if it_none else COMPARISON_KIND_SHARED_VS_SOLO)
        if ckind not in COMPARISON_KINDS:
            raise _PairRefusal(f"the launch set's row carries comparison_kind {lrow.get('comparison_kind')!r}, not one of {list(COMPARISON_KINDS)} ({SHARED_SOURCE_RULE})", where)
        if ckind != want_kind:
            raise _PairRefusal(f"the launch set's row carries comparison_kind {ckind!r}; this store is {want_kind!r} ({'the instruct row' + chr(39) + 's own none' if it_none else ('the instruct row' + chr(39) + 's own none served again: its repeat floor' if it_repeat else ('a repeat store' if repeat else 'a row-engine store of a solo row'))})", where)
        standup_id = str(lrow.get("standup_id") or "")
        if not _STANDUP_ID.match(standup_id):
            raise _PairRefusal(f"the launch set's row names standup_id {lrow.get('standup_id')!r}, not a 32-hex standup id ({SHARED_STANDUP_RULE})", where)
        solo_run_id = lrow.get("solo_run_id")
        solo_store = lrow.get("solo_store") if isinstance(lrow.get("solo_store"), Mapping) else None
        solo_run_ids = lrow.get("solo_run_ids") if isinstance(lrow.get("solo_run_ids"), Mapping) else None
        out: Dict[str, Any] = {"comparison_kind": ckind, "standup_id": standup_id, "solo_run_id": None, "solo_store": None, "solo_state": "no_contrast", "spelling": want_spelling}
        if it_none or it_repeat:
            if solo_run_id is not None or solo_store is not None or lrow.get("anchor_tuple") is not None:
                raise _PairRefusal(f"the instruct row's own none{' served again (its repeat floor)' if it_repeat else ''} carries solo_run_id {solo_run_id!r} / solo_store "
                                   f"{'present' if solo_store else None} / anchor_tuple {lrow.get('anchor_tuple')!r}; the `_it` none has no solo twin, so "
                                   + (f"a repeat_floor row of the instruct row names no solo run and no anchor ({SHARED_IT_REPEAT_RULE})" if it_repeat
                                      else f"an it_none_comparison row names no solo run and no anchor ({SHARED_IT_NONE_RULE})"), where)
            if it_repeat:
                out["solo_state"] = SOLO_STATE_NO_TWIN
            return out
        if not isinstance(solo_run_id, str) or not solo_run_id:
            raise _PairRefusal(f"the launch set's {ckind} row names no solo_run_id; the comparison has no other side ({SHARED_RUN_ID_RULE})", where)
        if solo_store is None:
            raise _PairRefusal(f"the launch set's {ckind} row carries no solo_store block ({{source, row_dir, cell, suffix, sha, mirror_rel, store_key, ...}}) ({SHARED_COMPARISON_RULE})", where)
        missing = [k for k in ("row_dir", "cell", "suffix", "sha", "mirror_rel", "store_key") if not solo_store.get(k)]
        if missing:
            raise _PairRefusal(f"the launch set's row's solo_store lacks {missing}", where)
        if str(solo_store["cell"]) != okey or str(solo_store["row_dir"]) != base["row_dir"]:
            raise _PairRefusal(f"the launch set's row's solo_store names ({solo_store['row_dir']}, {solo_store['cell']}), not this store's row and key ({base['row_dir']}, {okey})", where)
        if not _SHA1.match(str(solo_store["sha"])):
            raise _PairRefusal(f"the launch set's row's solo_store.sha {str(solo_store['sha'])[:16]!r} is not a 40-hex producer tree", where)
        want_rel = rel_store(str(solo_store["row_dir"]), okey, str(solo_store["suffix"]))
        if str(solo_store["mirror_rel"]) != want_rel:
            raise _PairRefusal(f"the launch set's row's solo_store.mirror_rel {solo_store['mirror_rel']!r} is not {want_rel} (its row dir, key and suffix)", where)
        if SUFFIX_RE.match(str(solo_store["suffix"])) is None or spec.suffix_re.match(str(solo_store["suffix"])) is not None:
            raise _PairRefusal(f"the launch set's row's solo_store.suffix {solo_store['suffix']!r} is not a SOLO store suffix (bcbw-<launch tag>-<sha7>[-r<N>])", where)
        if solo_run_ids is not None and str(solo_run_ids.get(spec.eval_id) or "") != solo_run_id:
            raise _PairRefusal(f"the launch set's row's solo_run_ids[{spec.eval_id}] {solo_run_ids.get(spec.eval_id)!r} is not its solo_run_id {solo_run_id!r}", where)
        # the id the SOLO reader mints for the solo store (SHARED_RUN_ID_RULE): the solo tag, the SOLO store key, the object key, this eval and instrument
        solo_spec = dataclasses.replace(spec, shared=None)
        want_solo = run_id_for(solo_spec, tag, str(solo_store["store_key"]), okey, instrument_id)
        if solo_run_id != want_solo:
            raise _PairRefusal(f"the launch set's row names solo_run_id {solo_run_id!r}; the solo reader mints {want_solo!r} for the solo store ({tag}/{solo_store['store_key']}, {okey}, "
                               f"{spec.eval_id}, instrument {instrument_id[:12]}): the row does not name the solo run ({SHARED_RUN_ID_RULE})", where)
        out.update(solo_run_id=solo_run_id, solo_store=dict(solo_store), solo_source=(solo_store.get("source") or {}).get(spec.eval_id) if isinstance(solo_store.get("source"), Mapping) else solo_store.get("source"))
        out.update(self._solo_side(solo_store, okey, base))
        return out

    def _solo_side(self, solo_store: Mapping[str, Any], okey: str, base: Mapping[str, Any]) -> Dict[str, Any]:
        """The SOLO store the comparison stands against (:data:`SHARED_COMPARISON_RULE`), read from the solo mirror through the group's solo-side inputs:
        its RESULT (COMPLETE at the row's tree), its manifest (the key, the tree) and its outcome record (the items). A solo store the mirror lacks is
        ``solo_state`` ``solo_store_absent`` (the run files, the contrast waits); a present store that disagrees with the launch row refuses by name."""
        spec = self.SPEC
        d = os.path.join(self.solo_root or "", str(solo_store["mirror_rel"]))
        p_out = os.path.join(d, "cells", str(solo_store["store_key"]), "outcomes", spec.outcomes_file)
        p_res, p_man = os.path.join(d, "RESULT"), os.path.join(d, "manifest.json")
        inp_out = self._solo_inputs.get(os.path.realpath(p_out)) or self._solo_inputs.get(p_out)
        if inp_out is None or not self.solo_root or not os.path.isdir(d):
            return {"solo_state": "solo_store_absent", "solo_items": None, "solo_record": None,
                    "solo_note": f"the solo store {LC.rel_home(d)} (its {spec.outcomes_file}) is not mirrored under the solo root {LC.rel_home(self.solo_root) if self.solo_root else '<none>'}; "
                                 f"the run files and the shared - solo contrast waits until it lands ({SHARED_COMPARISON_RULE})"}
        inp_res = self._solo_inputs.get(os.path.realpath(p_res)) or self._solo_inputs.get(p_res)
        inp_man = self._solo_inputs.get(os.path.realpath(p_man)) or self._solo_inputs.get(p_man)
        if inp_res is None:
            raise _PairRefusal(f"the solo store {LC.rel_home(d)} carries no RESULT beside its outcome record; the comparison's other side is not a complete store", d)
        with open(inp_res.path, "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        res = parse_result_line(lines[-1]) if lines else None
        if res is None or res.get("result") != "COMPLETE" or res.get("rc") != "0":
            raise _PairRefusal(f"the solo store's RESULT last line is not a {RESULT_HEAD} COMPLETE line with rc=0 ({(lines[-1] if lines else '<empty>')[:160]})", inp_res.uri.rsplit('#', 1)[0])
        if str(res.get("sha") or "") != str(solo_store["sha"]):
            raise _PairRefusal(f"the solo store's RESULT names tree {str(res.get('sha'))[:12]!r}; the launch row's solo_store.sha is {str(solo_store['sha'])[:12]}", inp_res.uri.rsplit('#', 1)[0])
        if inp_man is not None:
            # (a solo store's top-level manifest.json is not part of the solo layout today; when one is mirrored it must agree)
            man = self._json(inp_man) or {}
            if str((man.get("producer") or {}).get("tree_sha") or "") not in ("", str(solo_store["sha"])):
                raise _PairRefusal(f"the solo store's manifest names tree {str((man.get('producer') or {}).get('tree_sha'))[:12]!r}, the launch row's solo_store.sha {str(solo_store['sha'])[:12]}", inp_man.uri.rsplit('#', 1)[0])
        doc = self._json(inp_out) or {}
        if str(doc.get("task")) != spec.task or not isinstance(doc.get("items"), list):
            raise _PairRefusal(f"the solo store's outcome record is not {spec.task}'s (task {doc.get('task')!r})", inp_out.uri.rsplit('#', 1)[0])
        if str(doc.get("producer_tree_sha") or "") != str(solo_store["sha"]):
            raise _PairRefusal(f"the solo store's outcome record names tree {str(doc.get('producer_tree_sha'))[:12]!r}, the launch row's solo_store.sha {str(solo_store['sha'])[:12]}", inp_out.uri.rsplit('#', 1)[0])
        items = self._outcome_items({"outcomes": doc, "files": {"outcomes": inp_out.with_uri(inp_out.uri.rsplit("#", 1)[0])}})
        scored = {i: it["value"] for i, it in items.items() if it["scored"]}
        value = (sum(scored.values()) / len(scored)) if scored else None
        return {"solo_state": "solo_present", "solo_items": items, "solo_record": LC.rel_home(inp_out.uri.rsplit("#", 1)[0]), "solo_record_sha256": inp_out.sha256,
                "solo_result_line": (lines[-1] if lines else None), "solo_value": value, "solo_n_scored": len(scored), "solo_correct": int(round(sum(scored.values()))), "solo_n_items": len(items)}

    def _check_row_engine(self, man: Mapping[str, Any], base: Mapping[str, Any], lrow: Mapping[str, Any], okey: str, where_man: str, standup_id: str,
                          row_launch_tag: Optional[str] = None, ls_uri: Optional[str] = None) -> Dict[str, Any]:
        """The manifest's ``row_engine`` block (plan 1.1.4: exactly :data:`ROW_ENGINE_KEYS`), the endpoint's standup and RESULT's ``standup=`` token against the
        launch row's standup id (:data:`SHARED_STANDUP_RULE`); ``row_dir`` bound by its ``rows/<row>/<suffix>`` tail and recorded verbatim
        (:data:`SHARED_ROW_DIR_RULE`); the served-digest map read from the ENDPOINT and the store's tuple from the MANIFEST, the endpoint's own served-key
        fields admitted only as this store's key or as the row's last key (:data:`SHARED_ENDPOINT_TUPLE_RULE`; :data:`ENDPOINT_SERVED_KEY_BUG`) -- the map
        entry itself is held against the pinned object in :meth:`_emit_row`, once the object is resolved, and RESULT's line against the three-number row.
        Returns the ``extra.row_engine`` block; raises by name."""
        blk = man.get("row_engine")
        if not isinstance(blk, Mapping):
            raise _PairRefusal(f"manifest.json carries no row_engine block; a row-engine store declares {list(ROW_ENGINE_KEYS)} there (plan 1.1.4)", where_man)
        if set(blk) != set(ROW_ENGINE_KEYS):
            raise _PairRefusal(f"manifest.json:row_engine carries the keys {sorted(blk)}, not exactly {sorted(ROW_ENGINE_KEYS)} (plan 1.1.4: the block's key set is fixed)", where_man)
        if blk.get("row_engine") is not True:
            raise _PairRefusal(f"manifest.json:row_engine.row_engine is {blk.get('row_engine')!r}, not true", where_man)
        sid = str(blk.get("standup_id") or "")
        if not _STANDUP_ID.match(sid):
            raise _PairRefusal(f"manifest.json:row_engine.standup_id {blk.get('standup_id')!r} is not a 32-hex standup id", where_man)
        if sid != standup_id:
            raise _PairRefusal(f"manifest.json:row_engine.standup_id {sid[:8]} is not the launch row's standup_id {standup_id[:8]} ({SHARED_STANDUP_RULE})", where_man)
        served = blk.get("engine_cells")
        if not isinstance(served, list) or not served or okey not in [str(x) for x in served]:
            raise _PairRefusal(f"manifest.json:row_engine.engine_cells {served!r} does not name this store's key {okey!r}", where_man)
        served = [str(x) for x in served]
        max_loras = LC.int_or_none(blk.get("max_loras"))
        if max_loras is None or max_loras < 1:
            raise _PairRefusal(f"manifest.json:row_engine.max_loras {blk.get('max_loras')!r} is not a positive count", where_man)
        if not isinstance(blk.get("residual_slot_armed"), bool):
            raise _PairRefusal(f"manifest.json:row_engine.residual_slot_armed {blk.get('residual_slot_armed')!r} is not a boolean", where_man)
        # RULE 1 (SHARED_ROW_DIR_RULE): the row-record dir binds by its `rows/<row>/<suffix>` tail -- this store's row, and the ROW LAUNCH's container suffix (the row's primary
        # launch tag, the store's own relaunch segment, no repeat segment); equal to the launch row's declared row_dir tail where the row carries one; the spelling recorded verbatim
        spec = self.SPEC
        row_dir_rec = str(blk.get("row_dir") or "")
        row_dir_tail = row_record_tail(row_dir_rec)
        sm = spec.suffix_re.match(base["suffix"])
        own_relaunch = (sm.group("relaunch") or "") if sm else ""
        want_launch_tag = row_launch_tag or launch_tag_of(base["tag"], "none")
        want_shape = f"rows/{base['row_dir']}/bcbw-{want_launch_tag}-<sha7>-{spec.shared.store_marker}{own_relaunch}"
        cm = spec.suffix_re.match(row_dir_tail.split("/", 2)[2]) if row_dir_tail else None
        tail_row = row_dir_tail.split("/", 2)[1] if row_dir_tail else None
        if (row_dir_tail is None or tail_row != base["row_dir"] or cm is None or cm.group("launch_tag") != want_launch_tag or (cm.group("relaunch") or "") != own_relaunch
                or cm.group("repeat") or cm.group("task_tag")):
            raise _PairRefusal(f"manifest.json:row_engine.row_dir {row_dir_rec!r} does not name this store's row launch by its rows/<row>/<suffix> tail ({want_shape}; the volume-absolute "
                               f"`{VOLUME_HEAD}rows/...` and the relative `rows/...` spellings both bind by the tail) ({SHARED_ROW_DIR_RULE})", where_man)
        row_dir_declared = row_record_tail(str(lrow.get("row_dir"))) if lrow.get("row_dir") else None
        if lrow.get("row_dir") and row_dir_declared is None:
            raise _PairRefusal(f"the launch set's row declares row_dir {lrow.get('row_dir')!r}, not a rows/<row>/<suffix> container (relative or under {VOLUME_HEAD}) ({SHARED_ROW_DIR_RULE})", ls_uri or where_man)
        if row_dir_declared is not None and row_dir_declared != row_dir_tail:
            raise _PairRefusal(f"manifest.json:row_engine.row_dir {row_dir_rec!r} names the container {row_dir_tail}; the launch set's row declares {row_dir_declared} ({lrow.get('row_dir')!r}): "
                               f"the store's manifest and its launch row name different row launches ({SHARED_ROW_DIR_RULE})", where_man)
        container_sha7 = cm.group("sha7")
        endpoint = base["geometry_doc"].get("endpoint") if isinstance(base["geometry_doc"].get("endpoint"), Mapping) else {}
        where_geo = base["top"][GEOMETRY_FILE].uri
        if str(endpoint.get("standup_attempt_id") or "") != sid:
            raise _PairRefusal(f"{GEOMETRY_FILE}:endpoint.standup_attempt_id {str(endpoint.get('standup_attempt_id'))[:8]!r} is not the manifest's row_engine.standup_id {sid[:8]} ({SHARED_STANDUP_RULE})", where_geo)
        ep_served = [str(x) for x in endpoint.get("cells")] if isinstance(endpoint.get("cells"), list) else None
        if ep_served != served:
            raise _PairRefusal(f"{GEOMETRY_FILE}:endpoint.cells {ep_served!r} is not the manifest's row_engine.engine_cells {served!r} (one standup, one served set)", where_geo)
        # RULE 2 (SHARED_ENDPOINT_TUPLE_RULE): the store's tuple is the MANIFEST's -- cell_id (= the store key's object = the directory, checked before this point) and
        # harness_cell_id, the map's key; the endpoint's own served-key fields (`cell` / `harness_cell`) must name this store's key, or -- landing 1's row runner
        # (ENDPOINT_SERVED_KEY_BUG) -- the row's LAST served key, and nothing else
        hkey = str(man.get("harness_cell_id") or "")
        if not hkey:
            raise _PairRefusal("manifest.json names no harness_cell_id: the row's served-digest map is keyed by it, so the store's tuple has no key into the map", where_man)
        ep_key = str(endpoint.get("cell") or "")
        ep_hkey = str(endpoint.get("harness_cell") or "")
        ep_hkeys = [str(x) for x in endpoint.get("harness_cells")] if isinstance(endpoint.get("harness_cells"), list) else None
        ep_key_agrees = ep_key == okey and ep_hkey in ("", hkey)
        if not ep_key_agrees:
            last_key = served[-1]
            last_hkey = ep_hkeys[-1] if ep_hkeys else None
            if ep_key != last_key or (ep_hkey and last_hkey is not None and ep_hkey != last_hkey) or okey == last_key:
                raise _PairRefusal(f"{GEOMETRY_FILE}:endpoint.cell / harness_cell name ({ep_key!r}, {ep_hkey or None!r}), neither this store's tuple ({okey!r}, {hkey!r}) nor the row's last "
                                   f"served key ({last_key!r}, {last_hkey!r}; the landing-1 shape, {ENDPOINT_SERVED_KEY_BUG}): an endpoint naming another tuple is not read ({SHARED_ENDPOINT_TUPLE_RULE})", where_geo)
        # RESULT's standup= token (SHARED_STANDUP_RULE / SHARED_VERIFIED_STANDUP_RULE). A PRESENT token on the last line that is not the manifest's refuses, verified or not.
        # An ABSENT token on the last line is admitted only on a VERIFIED store (verification_pass.state verified) whose manifest and endpoint identities agree (the endpoint's
        # standup_attempt_id was held equal to the manifest's above): landing 2's CPU verifier re-stamped RESULT's last line without the token, so the id is the manifest's,
        # cross-checked against every earlier RESULT line that carries standup= (the row runner's GENERATED line). Later re-stamps carry
        # standup=<manifest id> before verified_by=, so the absent-token path covers exactly landing 2's 19 stores. A landing-1 store keeps the last-line rule unchanged.
        where_res = base["top"]["RESULT"].uri
        vp = blk.get("verification_pass") if isinstance(blk.get("verification_pass"), Mapping) else None
        verified = vp is not None and str(vp.get("state") or "") == VERIFICATION_STATE_VERIFIED
        res_sid = str(base["result"].get("standup") or "")
        if res_sid and res_sid != sid:
            raise _PairRefusal(f"RESULT's last line carries standup={res_sid!r}, not the manifest's row_engine.standup_id {sid[:8]} ({SHARED_STANDUP_RULE})", where_res)
        if not res_sid and not verified:
            raise _PairRefusal(f"RESULT's last line carries standup='<none>', not the manifest's row_engine.standup_id {sid[:8]} ({SHARED_STANDUP_RULE})", where_res)
        # every EARLIER line's standup= token (the row runner's GENERATED line on a landing-2 store): one id, the manifest's
        earlier_sids: List[str] = []
        for ln in base.get("result_earlier") or []:
            tok = parse_result_line(ln)
            if tok is not None and tok.get("standup"):
                earlier_sids.append(str(tok["standup"]))
        other = sorted({s for s in earlier_sids if s != sid})
        if other:
            raise _PairRefusal(f"RESULT's earlier line(s) carry standup={', '.join(x[:8] for x in other)!r}, not the manifest's row_engine.standup_id {sid[:8]} "
                               f"({SHARED_VERIFIED_STANDUP_RULE if verified else SHARED_STANDUP_RULE})", where_res)
        result_standup_source = ("last line" if res_sid else ("earlier line (the row runner's GENERATED line; the last line is the verifier's re-stamp)" if earlier_sids
                                                             else "manifest only (no RESULT line carries standup=; the last line is the verifier's re-stamp)"))
        verified_by = str(base["result"].get("verified_by") or "") or None
        vp_fc = (str(vp.get("function_call_id") or "") or None) if vp is not None else None
        if verified and verified_by is not None and vp_fc is not None and verified_by != vp_fc:
            raise _PairRefusal(f"RESULT's last line carries verified_by={verified_by!r}; manifest.json:row_engine.verification_pass.function_call_id is {vp_fc!r}: the re-stamp and the "
                               f"verification block name different verifier calls ({SHARED_VERIFIED_STANDUP_RULE})", where_res)
        # the served-digest map: the ENDPOINT's (the manifest's route copy may be null on a landing-1 store; when both are present they must be equal)
        route = man.get("route") if isinstance(man.get("route"), Mapping) else {}
        man_map = route.get("object_artifact_sha256_by_cell")
        man_map = {str(k): v for k, v in man_map.items()} if isinstance(man_map, Mapping) else None
        ep_map = endpoint.get("object_artifact_sha256_by_cell")
        if not isinstance(ep_map, Mapping):
            raise _PairRefusal(f"{GEOMETRY_FILE}:endpoint carries no object_artifact_sha256_by_cell map (manifest.json:route.object_artifact_sha256_by_cell is "
                               f"{'null' if man_map is None else f'{len(man_map)} keys'}): the row's served-digest map is read from the endpoint, so no served object can be attested "
                               f"({SHARED_ENDPOINT_TUPLE_RULE})", where_geo)
        ep_map = {str(k): v for k, v in ep_map.items()}
        if man_map is not None and man_map != ep_map:
            raise _PairRefusal(f"{GEOMETRY_FILE}:endpoint.object_artifact_sha256_by_cell ({len(ep_map)} keys) is not manifest.json:route.object_artifact_sha256_by_cell ({len(man_map)} keys); "
                               f"both present, they must be one map ({SHARED_ENDPOINT_TUPLE_RULE})", where_geo)
        lora_config = endpoint.get("lora_config") if isinstance(endpoint.get("lora_config"), Mapping) else {}
        residual = endpoint.get("residual_slot") if isinstance(endpoint.get("residual_slot"), Mapping) else {}
        row_dir_present = os.path.isdir(os.path.join(self._records_dir(), *row_dir_tail.split("/")))
        return {"standup_id": sid, "engine_object_keys": served, "n_engine_object_keys": len(served), "max_loras": max_loras, "residual_slot_armed": bool(blk.get("residual_slot_armed")),
                "compile_cache": (dict(blk["compile_cache"]) if isinstance(blk.get("compile_cache"), Mapping) else blk.get("compile_cache")), "row_record_dir": row_dir_rec,
                "row_record_dir_tail": row_dir_tail, "row_record_dir_absolute": row_dir_rec != row_dir_tail, "row_record_dir_present": row_dir_present,
                "row_record_dir_declared": (str(lrow.get("row_dir")) if lrow.get("row_dir") else None), "row_record_dir_tree_sha7": container_sha7,
                "row_record_dir_tree_agrees": container_sha7 == (sm.group("sha7") if sm else None), "row_launch_tag_legacy": want_launch_tag, "row_dir_rule": SHARED_ROW_DIR_RULE,
                "verification_pass": blk.get("verification_pass"),
                "object_artifact_sha256_by_harness_key": ep_map, "served_map_source": (SERVED_MAP_SOURCE_ENDPOINT if man_map is None else SERVED_MAP_SOURCE_BOTH),
                "manifest_route_map_null": man_map is None, "object_artifact_sha256_scalar": route.get("object_artifact_sha256"),
                "manifest_served_key": okey, "manifest_served_harness_key": hkey, "endpoint_served_key": ep_key, "endpoint_served_harness_key": (ep_hkey or None),
                "endpoint_served_key_agrees": ep_key_agrees, "endpoint_served_key_note": None, "endpoint_tuple_rule": SHARED_ENDPOINT_TUPLE_RULE,
                "result_tokens": {"standup": res_sid, "sha": base["result"].get("sha"), "name": base["result"].get("name")},
                "endpoint_max_loras": LC.int_or_none(lora_config.get("max_loras")), "endpoint_residual_slot_armed": residual.get("armed"),
                "flags_agree": (LC.int_or_none(lora_config.get("max_loras")) == max_loras and residual.get("armed") == blk.get("residual_slot_armed")),
                "result_standup": res_sid, "flags_rule": MAX_LORA_RANK_RULE, "rule": SHARED_STANDUP_RULE,
                # a VERIFIED store (landing 2; SHARED_VERIFIED_STANDUP_RULE): where the standup id was read from, the verifier's call on the re-stamp; stamped on those rows alone
                **({"verification_state": VERIFICATION_STATE_VERIFIED, "result_standup_source": result_standup_source, "result_standup_earlier": earlier_sids,
                    "result_verified_by": verified_by, "verification_function_call_id": vp_fc, "verified_standup_rule": SHARED_VERIFIED_STANDUP_RULE} if verified else {})}

    def _check_gate_shared(self, gate: Mapping[str, Any], gate_inp: Input, shared_row: Mapping[str, Any], lrow: Mapping[str, Any], okey: str) -> Dict[str, Any]:
        """A SHARED gate record's additions (plan 1.7): ``standup_id`` / ``engine_cells`` / ``row_engine`` / ``row_dir`` / ``verification_pass``, the
        ``comparison`` block (the same facts the launch row carries) and ``compared_against.comparability_class``; the recorded flags ride the gate block
        already. Raises :class:`_PairRefusal` by name; returns the fields for ``extra.gate``."""
        spec = self.SPEC
        where = gate_inp.uri.rsplit("#", 1)[0]
        re_blk = shared_row["row_engine"]
        if gate.get("row_engine") is not True:
            raise _PairRefusal(f"the gate record's row_engine is {gate.get('row_engine')!r}, not true: a row-engine store's record attests the row engine", where)
        if str(gate.get("standup_id") or "") != shared_row["standup_id"]:
            raise _PairRefusal(f"the gate record names standup_id {str(gate.get('standup_id'))[:8]!r}, the store {shared_row['standup_id'][:8]} ({SHARED_STANDUP_RULE})", where)
        g_served = [str(x) for x in gate.get("engine_cells")] if isinstance(gate.get("engine_cells"), list) else None
        if g_served != re_blk["engine_object_keys"]:
            raise _PairRefusal(f"the gate record's engine_cells {g_served!r} is not the manifest's row_engine.engine_cells {re_blk['engine_object_keys']!r}", where)
        if row_record_tail(str(gate.get("row_dir") or "")) != re_blk["row_record_dir_tail"]:
            raise _PairRefusal(f"the gate record's row_dir {gate.get('row_dir')!r} does not name the manifest's row launch {re_blk['row_record_dir_tail']} by its rows/<row>/<suffix> tail "
                               f"(the manifest spells {re_blk['row_record_dir']!r}) ({SHARED_ROW_DIR_RULE})", where)
        comp = gate.get("comparison") if isinstance(gate.get("comparison"), Mapping) else None
        if comp is None:
            raise _PairRefusal("the gate record carries no comparison block ({comparison_kind, solo_run_id, solo_store}); a shared record names its comparison", where)
        if str(comp.get("comparison_kind") or "") != shared_row["comparison_kind"]:
            raise _PairRefusal(f"the gate record's comparison.comparison_kind {comp.get('comparison_kind')!r} is not the launch row's {shared_row['comparison_kind']!r}", where)
        g_solo = comp.get("solo_run_id")
        g_solo_eval = g_solo.get(spec.eval_id) if isinstance(g_solo, Mapping) else g_solo
        if (g_solo_eval or None) != shared_row["solo_run_id"]:
            raise _PairRefusal(f"the gate record's comparison.solo_run_id[{spec.eval_id}] {g_solo_eval!r} is not the launch row's solo_run_id {shared_row['solo_run_id']!r}", where)
        g_store = comp.get("solo_store") if isinstance(comp.get("solo_store"), Mapping) else None
        l_store = shared_row["solo_store"]
        if (g_store is None) != (l_store is None) or (g_store is not None and str(g_store.get("mirror_rel") or "") != str(l_store.get("mirror_rel") or "")):
            raise _PairRefusal(f"the gate record's comparison.solo_store ({(g_store or {}).get('mirror_rel')!r}) is not the launch row's ({(l_store or {}).get('mirror_rel')!r})", where)
        ca = gate.get("compared_against") if isinstance(gate.get("compared_against"), Mapping) else None
        cls: Optional[str] = None
        if ca is None:
            if not gate.get("comparability_note"):
                raise _PairRefusal("the gate record carries neither compared_against nor a comparability_note: a row's FIRST record names why no prior record was needed (plan 1.7.4)", where)
        else:
            cls = str(ca.get("comparability_class") or "")
            if cls not in COMPARABILITY_CLASSES:
                raise _PairRefusal(f"the gate record's compared_against.comparability_class {ca.get('comparability_class')!r} is not one of {list(COMPARABILITY_CLASSES)}", where)
            ids = ca.get("standup_ids") if isinstance(ca.get("standup_ids"), Mapping) else {}
            this_id, prior_id = ids.get("this"), ids.get("prior")
            if str(this_id or "") != shared_row["standup_id"]:
                raise _PairRefusal(f"the gate record's compared_against.standup_ids.this {str(this_id)[:8]!r} is not the store's standup {shared_row['standup_id'][:8]}", where)
            want_cls = "same_standup" if (prior_id is not None and str(prior_id) == str(this_id)) else "cross_standup"
            if cls != want_cls:
                raise _PairRefusal(f"the gate record's comparability_class {cls!r} disagrees with its standup_ids (this {str(this_id)[:8]}, prior {str(prior_id)[:8] if prior_id else None}): {want_cls}", where)
        facts_list = [f for f in (gate.get("facts") or []) if isinstance(f, Mapping)]
        recorded = facts_list[0].get("recorded_flags") if facts_list and isinstance(facts_list[0].get("recorded_flags"), Mapping) else {}
        by_hkey = facts_list[0].get("route.object_artifact_sha256_by_cell") if facts_list else None
        if isinstance(by_hkey, Mapping) and {str(k): v for k, v in by_hkey.items()} != re_blk["object_artifact_sha256_by_harness_key"]:
            raise _PairRefusal(f"the gate record's facts attest route.object_artifact_sha256_by_cell differently from the store's served-digest map ({re_blk['served_map_source']})", where)
        return {"row_engine": True, "standup_id": shared_row["standup_id"], "engine_object_keys": g_served, "row_record_dir": gate.get("row_dir"), "verification_pass": gate.get("verification_pass"),
                "comparison": dict(comp), "comparability_class": cls, "comparability_note": gate.get("comparability_note"),
                "compared_against_standup_ids": (dict(ca.get("standup_ids")) if ca is not None and isinstance(ca.get("standup_ids"), Mapping) else None),
                "require_class": (ca.get("require_class") if ca is not None else None), "max_loras": LC.int_or_none(recorded.get("max_loras")),
                "residual_slot_armed": recorded.get("residual_slot_armed"), "citations": list(gate.get("citations") or []), "shared_standup_rule": SHARED_STANDUP_RULE}

    def _cross_check_three_shared(self, tn: Mapping[str, Any], shared_row: Mapping[str, Any], lrow: Mapping[str, Any]) -> List[str]:
        """The three-number row's SHARED additions against the launch row: ``comparison_kind``, ``standup_id``, ``solo_run_id`` (a per-eval map or the id) and
        ``solo_store.mirror_rel``; ``delta_value_vs_solo`` is RECORDED, never compared (the registry's comparison is the run_contrast)."""
        spec = self.SPEC
        bad: List[str] = []
        if str(tn.get("comparison_kind") or "") != shared_row["comparison_kind"]:
            bad.append(f"comparison_kind: producer {tn.get('comparison_kind')!r}, the launch row {shared_row['comparison_kind']!r}")
        if str(tn.get("standup_id") or "") != shared_row["standup_id"]:
            bad.append(f"standup_id: producer {str(tn.get('standup_id'))[:8]!r}, the store {shared_row['standup_id'][:8]!r}")
        t_solo = tn.get("solo_run_id")
        t_solo_eval = t_solo.get(spec.eval_id) if isinstance(t_solo, Mapping) else t_solo
        if (t_solo_eval or None) != shared_row["solo_run_id"]:
            bad.append(f"solo_run_id[{spec.eval_id}]: producer {t_solo_eval!r}, the launch row {shared_row['solo_run_id']!r}")
        t_store = tn.get("solo_store") if isinstance(tn.get("solo_store"), Mapping) else None
        l_store = shared_row["solo_store"]
        if (t_store is None) != (l_store is None) or (t_store is not None and str(t_store.get("mirror_rel") or "") != str(l_store.get("mirror_rel") or "")):
            bad.append(f"solo_store.mirror_rel: producer {(t_store or {}).get('mirror_rel')!r}, the launch row {(l_store or {}).get('mirror_rel')!r}")
        return bad

    def _emit_solo_contrasts(self, ctx: AdapterContext, group: str, filed: Mapping[Tuple[str, str], Mapping[str, Any]], facts: Dict[str, Any],
                             n_items: int = N_ITEMS) -> List[Dict[str, Any]]:
        """The shared - solo comparison (:data:`SHARED_COMPARISON_RULE`): one ``run_contrast`` per shared_vs_solo / repeat_floor run against its SOLO run
        (``anchor_kind`` solo_stand_in, ``anchor_run_id`` the solo reader's id), item-wise over the items SCORED on both sides -- the accuracy difference in pp,
        McNemar's exact p, Cohen's h -- with BOTH headline values on the contrast; the run's ``extra.comparison`` names the state. The instruct row's own
        none forms none (:data:`SHARED_IT_NONE_RULE`). The solo run is another source's STEERED or unsteered row standing in as the anchor: it is never
        re-picked, its CURRENCY alone is checked against the pass's frame (:meth:`anchor_current.AnchorCurrency.check_currency`, :data:`CURRENCY_RULE`) --
        current, the comparison files with ``extra.anchor_selection``; not folded yet or retired, it waits (``extra.comparison.state`` ``solo_not_current``,
        a refusal line), and the decision input re-reads the group when its currency moves."""
        spec = self.SPEC
        out: List[Dict[str, Any]] = []
        frame: Optional[AC.AnchorCurrency] = None
        for key in sorted(filed):
            st = filed[key]
            sh = st.get("shared")
            if not isinstance(sh, Mapping):
                continue
            tag, rk_okey = key
            comp = st["payload"]["extra"]["comparison"]
            rec = {"comparison_kind": sh["comparison_kind"], "solo_run_id": sh.get("solo_run_id"), "state": sh.get("solo_state"), "standup_id": sh["standup_id"]}
            if sh["comparison_kind"] == COMPARISON_KIND_IT_NONE:
                comp.update(state="no_contrast", note="the instruct row's own none: a comparison row only, not an anchor, no run_contrast")
                facts["comparisons"][f"{tag}|{rk_okey}"] = dict(rec, state="no_contrast")
                continue
            if sh.get("solo_state") == SOLO_STATE_NO_TWIN:
                note = ("the instruct row's own none served a second time: the `_it` none has no solo twin, so no borrowed contrast; the in-run floor pairs against the row's own "
                        f"none ({SHARED_IT_REPEAT_RULE})")
                comp.update(state=SOLO_STATE_NO_TWIN, note=note)
                facts["comparisons"][f"{tag}|{rk_okey}"] = dict(rec, state=SOLO_STATE_NO_TWIN, note=note)
                continue
            if sh.get("solo_items") is None:
                comp.update(state="solo_store_absent", note=sh.get("solo_note"))
                facts["comparisons"][f"{tag}|{rk_okey}"] = dict(rec, state="solo_store_absent", note=sh.get("solo_note"))
                continue
            a_items, s_items = sh["solo_items"], st["items"]
            if set(a_items) != set(s_items):
                raise AdapterRefusal(sh.get("solo_record") or self.root, f"the solo store's outcome record holds {len(a_items)} items and the shared store's {len(s_items)}, not the same item set: no item-wise comparison")
            paired = sorted(i for i in s_items if i in a_items and s_items[i]["value"] is not None and a_items[i]["value"] is not None)
            if not paired:
                comp.update(state="no_shared_items", note="no item scored on both sides")
                facts["comparisons"][f"{tag}|{rk_okey}"] = dict(rec, state="no_shared_items")
                continue
            if frame is None:
                frame = ctx.anchor_currency()
            check = frame.check_currency(str(sh["solo_run_id"]), ANCHOR_KIND_SOLO_STAND_IN, HEADLINE_METRIC, -1.0, computation_id=spec.recount_computation_id(),
                                         steered_run_id=st["run_id"], steered_judgment_id="")
            facts["anchor_states"][check.state] = facts["anchor_states"].get(check.state, 0) + 1
            if check.state != "own_current":
                # the solo run is not a current run of the fold (not folded yet, or retired by a ruling / a later run): the comparison waits, by name
                why = f"the solo run {sh['solo_run_id']} is not a current run of the registry ({check.own_reason}: {check.why or 'not current'}); the borrowed comparison waits until it is ({CURRENCY_RULE})"
                comp.update(state=SOLO_STATE_NOT_CURRENT, note=why, anchor_selection=check.selection_json())
                facts["comparisons"][f"{tag}|{rk_okey}"] = dict(rec, state=SOLO_STATE_NOT_CURRENT, note=why, currency_own_reason=check.own_reason)
                facts["contrasts_parked"] += 1
                self.refusals.add(f"{tag}|{rk_okey}|solo_contrast", f"contrast has no current anchor: {why}; filed without the borrowed contrast", sh.get("solo_record") or self.root)
                continue
            b = sum(1 for i in paired if s_items[i]["value"] == 1.0 and a_items[i]["value"] == 0.0)
            c = sum(1 for i in paired if s_items[i]["value"] == 0.0 and a_items[i]["value"] == 1.0)
            p_s = sum(s_items[i]["value"] for i in paired) / len(paired)
            p_a = sum(a_items[i]["value"] for i in paired) / len(paired)
            plan = solo_anchor_plan(sh["solo_run_id"])
            facts["contrasts_filed"] += 1
            facts["contrasts_own"] += 1
            payload = {"run_id": st["run_id"], "anchor_run_id": sh["solo_run_id"], "judgment_id": "", "metric_id": HEADLINE_METRIC, "threshold": -1.0,
                       "computation_id": spec.recount_computation_id(), "anchor_judgment_id": "", "anchor_kind": anchor_kind_for(plan),
                       "delta": 100.0 * (p_s - p_a), "delta_p": DC.mcnemar_exact(b, c), "delta_method": CONTRAST_METHOD,
                       "effect_size": DC.cohen_h(p_s, p_a), "effect_kind": "cohens_h", "n_pairs": len(paired), "computed_by": "registry", "computed_utc": st["when"],
                       "extra": {"unit": "pp", "comparison_kind": sh["comparison_kind"], "anchor_rule": SHARED_COMPARISON_RULE, "anchor_selection": check.selection_json(),
                                 "currency_rule": CURRENCY_RULE, "anchor_kind_note": (
                                     f"solo_stand_in: the solo run ({sh.get('solo_source') or spec.shared.solo_source}) stands in as the anchor; another source's row on the same instrument, served by "
                                     "another standup; the contrast is the registry's over the two outcome records (D932 (b): its own kind, never `borrowed`, the "
                                     "currency's tier-1c re-anchoring)"),
                                 "anchor_view_legacy": tag, "anchor_object_key": st["okey"], "row_view_legacy": spec.spelling_of(tag), "row_tag_legacy": tag, "cross_standup": True,
                                 "standup_ids": {"row": sh["standup_id"], "anchor": None}, "discordant_shared_only": b, "discordant_solo_only": c,
                                 "acc_shared_over_pairs": p_s, "acc_solo_over_pairs": p_a, "value_shared": st.get("acc_over_scored"), "value_solo": sh.get("solo_value"),
                                 "n_scored_shared": sum(1 for i in s_items if s_items[i]["value"] is not None), "n_scored_solo": sh.get("solo_n_scored"),
                                 "n_excluded_shared_unscored": sum(1 for i in s_items if s_items[i]["value"] is None),
                                 "n_excluded_solo_unscored": sum(1 for i in a_items if a_items[i]["value"] is None),
                                 "solo_store": sh.get("solo_store"), "solo_record": sh.get("solo_record"), "solo_record_sha256": sh.get("solo_record_sha256"),
                                 "producer_delta_value_vs_solo": comp.get("producer_delta_value_vs_solo"),
                                 "method_text": f"paired per item over the items SCORED on both sides (the same {n_items} items, one instrument, the greedy protocol): the accuracy "
                                                "difference shared - solo, pp; p from McNemar's exact test on the discordant pairs; Cohen's h of the two accuracies; an unscored item on "
                                                "either side leaves the pair"}}
            out.append(payload)
            comp.update(state=ANCHOR_CONTRASTED, delta_pp=payload["delta"], delta_p=payload["delta_p"], cohens_h=payload["effect_size"], n_pairs=len(paired), value_shared=st.get("acc_over_scored"))
            facts["comparisons"][f"{tag}|{rk_okey}"] = dict(rec, state=ANCHOR_CONTRASTED, anchor_kind=ANCHOR_KIND_SOLO_STAND_IN, n_pairs=len(paired), delta_pp=payload["delta"], delta_p=payload["delta_p"],
                                                            cohens_h=payload["effect_size"], value_shared=st.get("acc_over_scored"), value_solo=sh.get("solo_value"),
                                                            producer_delta_value_vs_solo=comp.get("producer_delta_value_vs_solo"))
        return out

    # ---- the full-suite item set (ITEM_SET_RULE) --------------------------------------------------
    def _item_set_pin(self, shared: Mapping[str, Input]) -> ItemSetPin:
        """The item set a FULL reader's group must attest: ``records/item_sets.json``'s entry for the spec's set, the mirrored pin file it names (by file sha256),
        the pin's ids (their count = ``n`` = ``of``, no seed, the record task) and their digest, checked against the registry's declaration (``item_sets.yaml``
        ``n_items`` / ``items_sha256``). A problem anywhere returns the pin with ``problem`` set: every store of the group then refuses by name with it."""
        spec = self.SPEC
        sid, pin_name = str(spec.item_set_id), str(spec.item_set_pin_file)
        where = self._records_dir()

        def bad(problem: str, at: Optional[str] = None, **kw) -> ItemSetPin:
            return ItemSetPin(sid, None, None, None, pin_name, None, shared.get(ITEM_SETS_FILE), shared.get(pin_name), at or where, problem, **kw)

        idx_inp = shared.get(ITEM_SETS_FILE)
        if idx_inp is None:
            return bad(f"{RECORDS_DIR}/{ITEM_SETS_FILE} is absent beside the stores: the item set the stores ran ({sid}) cannot be attested, so no store of this source files")
        idx_uri = idx_inp.uri.rsplit("#", 1)[0]
        try:
            with open(idx_inp.path, "r", encoding="utf-8") as fh:
                idx = json.load(fh)
        except (OSError, ValueError) as exc:
            return bad(f"{RECORDS_DIR}/{ITEM_SETS_FILE} is unreadable ({exc}); the item set {sid} cannot be attested", idx_uri)
        entries = item_set_entries(idx)
        hits = [e for e in entries if str(e.get("id")) == sid]
        if len(hits) != 1:
            named = ", ".join(sorted(str(e.get("id")) for e in entries)) or "none"
            return bad(f"{RECORDS_DIR}/{ITEM_SETS_FILE} names {len(hits)} entries for item set {sid} (expected exactly one; it names: {named})", idx_uri)
        e = hits[0]
        e_sha, e_n, e_file = str(e.get("sha256") or ""), LC.int_or_none(e.get("n")), os.path.basename(str(e.get("file") or ""))
        problems: List[str] = []
        if not _SHA256.match(e_sha):
            problems.append(f"sha256 {e_sha[:16]!r} is not a 64-hex digest (the pin FILE's bytes)")
        if e_n is None or e_n <= 0:
            problems.append(f"n {e.get('n')!r} is not a positive count")
        if e_file != pin_name:
            problems.append(f"file {e.get('file')!r} is not the pin file this reader reads ({pin_name})")
        if problems:
            return bad(f"{RECORDS_DIR}/{ITEM_SETS_FILE} entry {sid}: " + "; ".join(problems), idx_uri, entry=dict(e))
        pin_inp = shared.get(pin_name)
        if pin_inp is None:
            return bad(f"{RECORDS_DIR}/{pin_name} (the pin file {ITEM_SETS_FILE} names for {sid}, sha256 {e_sha[:12]}) is not mirrored beside the stores; no store's item set can be attested",
                       idx_uri, entry=dict(e))
        pin_uri = pin_inp.uri.rsplit("#", 1)[0]
        if pin_inp.sha256 != e_sha:
            return bad(f"the mirrored pin file {RECORDS_DIR}/{pin_name} has sha256 {pin_inp.sha256[:12]}, {ITEM_SETS_FILE} says {e_sha[:12]} for {sid}: the pin the stores ran is not the "
                       "mirrored one (or the index is stale); no store's item set can be attested", pin_uri, entry=dict(e))
        try:
            with open(pin_inp.path, "r", encoding="utf-8") as fh:
                pin = json.load(fh)
        except (OSError, ValueError) as exc:
            return bad(f"{RECORDS_DIR}/{pin_name} is unreadable ({exc})", pin_uri, entry=dict(e))
        ids = pin.get("ids") if isinstance(pin, Mapping) else None
        if not isinstance(ids, list) or not ids or not all(isinstance(i, str) and i for i in ids):
            return bad(f"{RECORDS_DIR}/{pin_name} carries no non-empty list of string ids", pin_uri, entry=dict(e))
        n_pin, of_pin, seed, task = LC.int_or_none(pin.get("n")), LC.int_or_none(pin.get("of")), pin.get("seed"), pin.get("task")
        if seed is not None:
            problems.append(f"seed {seed!r} is not null (the whole population, no seed)")
        if n_pin != len(ids):
            problems.append(f"n {pin.get('n')!r} is not the {len(ids)} ids it lists")
        if of_pin != len(ids):
            problems.append(f"of {pin.get('of')!r} is not the {len(ids)} ids it lists (a full set is the whole population: n == of)")
        if len(set(ids)) != len(ids):
            problems.append(f"{len(ids) - len(set(ids))} id(s) repeat")
        if str(task or "") != spec.record_task:
            problems.append(f"task {task!r} is not {spec.record_task!r}")
        if e_n != len(ids):
            problems.append(f"{ITEM_SETS_FILE} says n {e_n}, the pin carries {len(ids)} ids")
        if problems:
            return bad(f"{RECORDS_DIR}/{pin_name} (item set {sid}): " + "; ".join(problems), pin_uri, entry=dict(e), task=(str(task) if task is not None else None))
        digest = pin_ids_sha256(ids)
        row = self.seeds.item_sets.get(sid)
        if not isinstance(row, Mapping):
            return bad(f"item_sets.yaml declares no item set {sid}; the reader cannot file on an undeclared set (a new item set is a new instrument)", pin_uri, entry=dict(e))
        reg_n, reg_sha = LC.int_or_none(row.get("n_items")), str(row.get("items_sha256") or "")
        if reg_n != len(ids):
            return bad(f"item_sets.yaml declares {sid} with n_items {reg_n}; the mirrored pin {RECORDS_DIR}/{pin_name} carries {len(ids)} ids ({ITEM_SETS_FILE} n {e_n}): the registry's "
                       "declaration and the producer's pin must agree before a row files", pin_uri, entry=dict(e), registry_n_items=reg_n, registry_items_sha256=reg_sha)
        if reg_sha != digest:
            return bad(f"item_sets.yaml declares {sid} with items_sha256 {reg_sha[:12] or '<none>'}; the mirrored pin's ids digest (contracts.ids_sha256) is {digest[:12]}: the registry's "
                       "declaration and the producer's pin must agree before a row files", pin_uri, entry=dict(e), registry_n_items=reg_n, registry_items_sha256=reg_sha)
        return ItemSetPin(sid, len(ids), of_pin, digest, pin_name, pin_inp.sha256, idx_inp, pin_inp, where, None, entry=dict(e), registry_n_items=reg_n,
                          registry_items_sha256=reg_sha, task=str(task))

    def _check_item_set(self, item_set: ItemSetPin, block: Optional[Mapping[str, Any]], base: Mapping[str, Any], lrow: Mapping[str, Any], ls_inp: Input,
                        where_man: str) -> Dict[str, Any]:
        """One store against the group's pinned item set (:data:`ITEM_SET_RULE`): the manifest's hashed block ``{seed: null, n, of, ids_sha256}``, the label on
        ``gen_stats.json`` / the summary / the outcome record, the launch set's row ``item_set {id, sha256, n}``. Raises :class:`_PairRefusal` by name; returns
        the ``extra.item_set`` block. (The outcome record's item count is the count check in :meth:`_items`.)"""
        spec = self.SPEC
        if block is None:
            raise _PairRefusal(f"manifest.json:tinycap.{spec.item_set_key} is absent: the store does not attest the full item set {item_set.id} ({RECORDS_DIR}/{item_set.pin_file})", where_man)
        got = {"seed": block.get("seed"), "n": LC.int_or_none(block.get("n")), "of": LC.int_or_none(block.get("of")), "ids_sha256": str(block.get("ids_sha256") or "")}
        want = {"seed": None, "n": item_set.n, "of": item_set.n, "ids_sha256": item_set.ids_sha256}
        bad = [f"{k} {got[k]!r} (the pin: {want[k]!r})" for k in ("seed", "n", "of", "ids_sha256") if got[k] != want[k]]
        if bad:
            raise _PairRefusal(f"manifest.json:tinycap.{spec.item_set_key} does not attest the pinned item set {item_set.id} ({RECORDS_DIR}/{item_set.pin_file}, ids_sha256 "
                               f"{str(item_set.ids_sha256)[:12]} of the pin's ids in pin order): " + "; ".join(bad), where_man)
        for what, label, uri in (("gen_stats.json", base["gen_stats"].get("item_set"), base["files"]["gen_stats.json"].uri),
                                 (f"summary/{spec.outcomes_file}", base["summary"].get("item_set"), base["files"]["summary"].uri),
                                 (f"outcomes/{spec.outcomes_file}", base["outcomes"].get("item_set"), base["files"]["outcomes"].uri)):
            if str(label or "") != str(spec.item_set_label):
                raise _PairRefusal(f"{what}:item_set is {label!r}, not {spec.item_set_label!r}: the store is not a full-suite pass of {spec.task} over {item_set.id}", uri)
        ls = lrow.get("item_set")
        if not isinstance(ls, Mapping):
            raise _PairRefusal(f"the launch set's row carries no item_set block ({{id, sha256, n}}) naming {item_set.id}; a full-suite row declares its item set", ls_inp.uri)
        lgot = (str(ls.get("id") or ""), str(ls.get("sha256") or ""), LC.int_or_none(ls.get("n")))
        lwant = (item_set.id, str(item_set.pin_file_sha256), item_set.n)
        if lgot != lwant:
            raise _PairRefusal(f"the launch set's row names item set (id {lgot[0]!r}, sha256 {lgot[1][:12]!r}, n {lgot[2]!r}); the mirrored set is ({lwant[0]}, {lwant[1][:12]} the pin "
                               f"file's bytes, {lwant[2]})", ls_inp.uri)
        return {"id": item_set.id, "n": item_set.n, "of": item_set.of, "seed": None, "ids_sha256": item_set.ids_sha256, "pin_file": item_set.pin_file,
                "pin_file_sha256": item_set.pin_file_sha256, "label": spec.item_set_label, "manifest_key": spec.item_set_key, "pin_task": item_set.task,
                "index_record": LC.rel_home(item_set.index_inp.uri.rsplit("#", 1)[0]), "pin_record": LC.rel_home(item_set.pin_inp.uri.rsplit("#", 1)[0]),
                "registry_n_items": item_set.registry_n_items, "registry_items_sha256": item_set.registry_items_sha256, "launch_set_item_set": dict(ls),
                "ids_sha256_rule": "contracts.ids_sha256(pin['ids']): the sha256 of the canonical JSON list of the pin's ids in pin order (never the file's bytes; pin_file_sha256 is those)",
                "rule": ITEM_SET_RULE}

    # ---- contrasts (PAIRED_ANCHOR_RULE; the anchor currency, CURRENCY_RULE) ----------------------------------------------------
    @staticmethod
    def _value_row(st: Mapping[str, Any], comp_id: str) -> Optional[Dict[str, Any]]:
        """The row's VALUE reading among its built records (the recount row the contrast is keyed on): the park lands on its ``extra.delta_block``."""
        for rt, payload in st["records"]:
            if rt == "run_metric" and payload.get("metric_id") == HEADLINE_METRIC and payload.get("computation_id") == comp_id:
                return payload
        return None

    @staticmethod
    def _park_delta_block(st: Mapping[str, Any], value_row: Optional[Dict[str, Any]], why: str, anchor: Mapping[str, Any], selection: Mapping[str, Any], rule: str) -> None:
        """The parked comparison's ``extra.delta_block`` on the row's value reading: every delta field None, ``n_pairs`` 0, the reason by name, the
        anchor the park was decided on and the selection block beside it, and the producer's own three-number delta RECORDED (never a contrast row). Both parks
        -- the currency's (no current anchor at the exact key) and the window rule's (no u0 ``none`` at the row's serving window) -- write this one shape."""
        if value_row is None:
            return
        value_row.setdefault("extra", {})["delta_block"] = dict(
            {f: None for f in ("delta", "delta_lo", "delta_hi", "delta_p", "effect_size", "effect_kind")}, n_pairs=0,
            delta_method=f"none: {why}", computed_by="registry", why_not_a_contrast_row=why, anchor=dict(anchor, registry_contrast_filed_against=None),
            anchor_selection=dict(selection), producer_block={"producer_delta_value_vs_none": st["payload"]["extra"]["three_numbers"].get("producer_delta_value_vs_none"),
                                                              "note": "the producer's three-number record's delta is RECORDED on the run, never a contrast row"},
            rule=rule)

    def _anchor_key(self, st: Mapping[str, Any], own_run_id: Optional[str], own_kind: Optional[str]) -> AC.AnchorKey:
        """The steered row's exact key for the pass's anchor currency, from the run payload the row built: the standup is the
        registry spelling of its tag under this source (`view_legacy`, `lane_source_legacy`: one standup per row dir on a solo reader)."""
        p = st["payload"]
        return AC.AnchorKey(model_id=str(p["model_id"]), eval_id=str(p["eval_id"]), instrument_id=str(p["instrument_id"]),
                            token_cap_tokens=LC.int_or_none(p.get("token_cap_tokens")), trial_index=LC.int_or_none(p.get("trial_index")),
                            pass_label=p.get("pass_label"), decode_policy_id=p.get("decode_policy_id"), lane_source_legacy=p.get("lane_source_legacy"),
                            standup={"view_legacy": p.get("view_legacy"), "lane_source_legacy": p.get("lane_source_legacy")},
                            own_anchor_run_id=own_run_id, own_anchor_kind=(own_kind if own_run_id is not None else None))

    def _emit_contrasts(self, ctx: AdapterContext, group: str, filed: Mapping[Tuple[str, str], Mapping[str, Any]], expected: Mapping[Tuple[str, str], Mapping[str, Any]],
                        seen: Set[Tuple[str, str]], facts: Dict[str, Any], n_items: int = N_ITEMS) -> List[Dict[str, Any]]:
        """One ``run_contrast`` per steered / prompted run of the group, DECIDED by the pass's anchor currency (:data:`CURRENCY_RULE`).
        The row's OWN pairing is the plan's (:func:`anchor_plan` / :func:`instruct_anchor_plan` / :func:`it_repeat_anchor_plan`:
        its own tag's ``none``, the cross-tag fallback, the instruct alias -- resolved from the rows that FILED in this batch): it is the own anchor
        the currency is asked about and it stands while the frame holds it current with a usable reading -- item-wise over the items SCORED on both
        sides (the accuracy difference in pp, McNemar's exact p on the discordant pairs, Cohen's h), ``anchor_kind`` :func:`anchor_kind_for`'s, plus
        ``extra.anchor_selection``. A retired, unfolded or absent own anchor (the first pass over a fresh registry; a seed tag whose ``none`` has not run;
        the u0 ``none`` of a later standup's keys, :data:`UNSTEERED_ROW_RULE`) searches the exact key: a re-picked unsteered run that is a row of this batch is
        paired per item like the own one, any other files the registry's UNPAIRED rate difference from the two readings' k / n (``computed_by registry``,
        its own computation row named by ``extra.statistic_computation_id``; ``anchor_view_legacy`` the candidate's spelling). The kind and the standup
        fields follow the currency's TIER (``extra.reanchor_tier``): the row's own standup's ``none`` the registry holds current from an
        earlier read while this batch's is absent or refused -- ``same_standup_current``, ``anchor_kind same_standup_none``, ``cross_standup`` false;
        the same model's ``none`` of another standup under this source -- ``sibling_current``, ``sibling_none``, ``cross_standup`` true; the paper's
        roster-matched baseline of another source -- ``borrowed_reference``, ``borrowed``, ``cross_standup`` and ``cross_source`` true. The kind
        :func:`anchor_kind_for` yields is checked against the currency's on every filed contrast (a disagreement refuses the group by name). No current
        anchor at the exact key parks the comparison by name -- the VALUE row's ``extra.delta_block``, a refusal line -- and the previous revision's key
        is withdrawn by the head. Stamps every row's ``extra.anchor`` block on its (not yet added) run payload -- the anchor's run id when contrasted,
        else WHY no contrast is formed -- and returns the contrast payloads."""
        spec = self.SPEC
        out: List[Dict[str, Any]] = []
        comp_id = spec.recount_computation_id()
        frame: Optional[AC.AnchorCurrency] = None
        by_run = {st_["run_id"]: st_ for st_ in filed.values()}
        mk_g = model_key_of(group)
        u_tag = unsteered_tag_of(mk_g) if (mk_g and spec.unsteered_row and MODEL_KEYS[mk_g].unsteered_row) else None
        # a BASELINE anchors nothing: the plain `none` and a `none@w<N>` window re-serve (UNSTEERED_ROW_RULE); a shared repeat `none@rep<N>` is a row that anchors on its own none
        for key in sorted(k for k in filed if split_row_okey(k[1]) != ("none", None)):
            tag, okey = key
            st = filed[key]
            instruct = bool(st.get("instruct"))
            it_repeat = bool(st.get("it_repeat"))
            if u_tag is not None and (tag == u_tag or str(st.get("anchor_tuple") or "") == f"{UNSTEERED_SELECTION}/none"):
                # the family's baseline is the u0 none at the row's SERVING WINDOW (WINDOW_ANCHOR_RULE): a u0 row's own none, an s1 row's across tags
                plan = window_anchor_plan(u_tag, tag, st, filed, expected, seen, facts.get("unsteered_default_window"))
            elif it_repeat:
                # the instruct row's repeat floor (SHARED_IT_REPEAT_RULE): the row's OWN none (the it_none_comparison run) is the pair -- a same-object pair on one standup, never the alias
                plan = it_repeat_anchor_plan(tag, filed, expected, seen)
            else:
                plan = instruct_anchor_plan(group, filed, expected, seen) if instruct else anchor_plan(group, tag, filed, expected, seen)
            anchor = plan["anchor"]
            standup_note = None
            if spec.shared is not None and anchor is not None and st.get("standup_id") and anchor.get("standup_id") and st["standup_id"] != anchor["standup_id"]:
                # a SHARED row relaunched on its OWN standup (`-shared-r<N>`) anchors on the row's none served by ANOTHER standup: sibling_none (SHARED_ANCHOR_STANDUP_RULE)
                plan = dict(plan, cross_standup=True)
                standup_note = (f"the row was served by standup {st['standup_id'][:8]} and the row's none by standup {anchor['standup_id'][:8]} (a relaunch on its own standup, the gate's "
                                f"comparability_class cross_standup): the anchor is the same model's none served by another standup ({SHARED_ANCHOR_STANDUP_RULE})")
            block = {"rule": (WINDOW_ANCHOR_RULE if plan.get("window_matched") else (SHARED_IT_REPEAT_RULE if it_repeat else (INSTRUCT_ANCHOR_RULE if instruct else ANCHOR_RULE))),
                     "anchor_view_legacy": (plan["anchor_tag"] or group), "anchor_object_key": "none", "cross_standup": plan["cross_standup"], "declared_role": st.get("declared_role"), "currency_rule": CURRENCY_RULE}
            if plan.get("window_matched"):
                block["window_matched"] = dict(plan["window_matched"])
            if spec.shared is not None and anchor is not None:
                block.update(standup_ids={"row": st.get("standup_id"), "anchor": anchor.get("standup_id")}, anchor_standup_rule=SHARED_ANCHOR_STANDUP_RULE)
                if standup_note is not None:
                    block["standup_note"] = standup_note
            if it_repeat:
                block["repeat_floor_of_own_none"] = True
            elif instruct:
                block["anchor_alias"] = dict(st["payload"]["extra"]["anchor_alias"])
            if plan["own_none_state"] is not None:
                block.update(own_none_state=plan["own_none_state"], own_none_note=plan["own_none_note"])
            if plan["own"] and st.get("paired"):
                # the paired tag's row against ITS OWN `none`: the run's spelling moves from the fallback to the own anchor
                block.update(rule=PAIRED_ANCHOR_RULE, own_standup_replicate_of_view=group)
                st["payload"]["extra"].pop("cross_standup_anchor", None)
                st["payload"]["extra"].update(own_anchor=f"{tag}/none", standup_replicate_of_view=group, anchor_rule=PAIRED_ANCHOR_RULE)
            if plan.get("window_matched") and anchor is None:
                # the window rule (WINDOW_ANCHOR_RULE, 2026-10-05) admits no baseline from another serving window: a row whose window has no
                # u0 `none` (not declared, not yet run, refused, or two of them) files no contrast whatever the exact key holds elsewhere. The currency is
                # not asked (the exact key has no window term and would offer another serving), and the comparison PARKS BY NAME the way the currency's
                # park below does -- the plan's state and reason on the run's anchor block with a `no_current_anchor` selection beside, a delta_block on
                # the value reading, one refusal line -- never silently: an early `continue` here left the value metric without a
                # delta_block and the source reading clean though its comparison was missing
                wm = plan["window_matched"]
                why = (f"the window rule admits no baseline but a u0 `none` served at the row's window ({wm['row_window']} x {wm['row_max_num_seqs']}), and none is "
                       f"filed there ({plan['state']}): {plan['why']}")
                # (the same keys as AnchorChoice.selection_json, so every contrast's block reads alike; keyword form, since the anchor_kind stamp sites are pinned by source)
                selection = dict(state="no_current_anchor", own_reason=plan["state"], own_anchor_run_id=None, own_anchor_why=plan["why"], why=why, n_candidates=0,
                                 candidates=[], rejected=[], key=None, reading=None, order=AC.ORDER, rule=AC.RULE, anchor_kind=None, window_only=WINDOW_PARK_NOTE,
                                 window_matched=dict(wm))
                block.update(state=plan["state"], reason=why, anchor_run_id=None, currency_state="no_current_anchor", currency_own_reason=plan["state"],
                             currency_why=WINDOW_PARK_NOTE, anchor_selection=selection)
                st["payload"]["extra"]["anchor"] = block
                facts["anchor_states"]["no_current_anchor"] = facts["anchor_states"].get("no_current_anchor", 0) + 1
                facts["contrasts"][f"{tag}|{okey}"] = {"anchor": None, "state": plan["state"], "why": plan["why"], "own_none_state": plan["own_none_state"],
                                                       "currency_state": "no_current_anchor", "currency_own_reason": plan["state"], "currency_why": WINDOW_PARK_NOTE,
                                                       "own_anchor_run_id": None, "n_candidates": 0, "window_matched": dict(wm)}
                self._park_delta_block(st, self._value_row(st, comp_id), why,
                                       dict(own_anchor_run_id=None, own_anchor_kind=None, state="no_current_anchor", own_reason=plan["state"], anchor_run_id=None, anchor_kind=None),
                                       selection, WINDOW_ANCHOR_RULE)
                facts["contrasts_parked"] += 1
                self.refusals.add(f"{tag}|{okey}|contrast", f"contrast has no baseline at the row's serving window ({wm['row_window']} x {wm['row_max_num_seqs']}) "
                                                             f"[{plan['state']}: {plan['why']}]; filed without a contrast", st["payload"]["extra"].get("store_uri") or self.root)
                continue
            # ---- the decision (CURRENCY_RULE): the plan's anchor is the OWN anchor the pass's frame is asked about; the frame loads on the first comparison
            if frame is None:
                frame = ctx.anchor_currency()
            own_id = anchor["run_id"] if anchor is not None else None
            own_kind = anchor_kind_for(plan) if anchor is not None else None
            akey = self._anchor_key(st, own_id, own_kind)
            choice = frame.choose(akey, HEADLINE_METRIC, -1.0, None, computation_id=comp_id, steered_run_id=st["run_id"], steered_judgment_id="")
            selection = choice.selection_json()
            facts["anchor_states"][choice.state] = facts["anchor_states"].get(choice.state, 0) + 1
            value_row = self._value_row(st, comp_id)
            s_items = st["items"]
            k_s = int(sum(1 for i in s_items if s_items[i]["value"] == 1.0))
            n_s = int(sum(1 for i in s_items if s_items[i]["value"] is not None))
            own_text = (f"{own_id} ({own_kind}; {plan['anchor_tag']}/none)" if own_id is not None else f"none declared ({plan['state']}: {plan['why']})")
            key_text = f"key (cap {akey.token_cap_tokens}, trial {akey.trial_index}, pass {akey.pass_label}, instrument {akey.instrument_id})"
            fact = {"anchor": None, "state": plan["state"], "why": plan["why"], "own_none_state": plan["own_none_state"], "currency_state": choice.state,
                    "currency_own_reason": choice.own_reason, "own_anchor_run_id": own_id, "n_candidates": choice.n_candidates}

            def park(why: str, refusal: str) -> None:
                self._park_delta_block(st, value_row, why, dict(own_anchor_run_id=own_id, own_anchor_kind=own_kind, state=choice.state, own_reason=choice.own_reason,
                                                                anchor_run_id=choice.run_id, anchor_kind=choice.anchor_kind), selection, AC.RULE)
                facts["contrasts_parked"] += 1
                self.refusals.add(f"{tag}|{okey}|contrast", refusal, st["payload"]["extra"].get("store_uri") or self.root)

            if choice.state == "no_current_anchor":
                # no current anchor at the exact key: the plan's own state (a `none` not yet run / refused) keeps its spelling on the run; the currency's reason beside it
                reason = plan["why"] if plan["why"] else choice.why
                block.update(state=(plan["state"] or "no_current_anchor"), reason=reason, anchor_run_id=None, currency_state=choice.state, currency_own_reason=choice.own_reason,
                             currency_why=choice.why, anchor_selection=selection)
                st["payload"]["extra"]["anchor"] = block
                facts["contrasts"][f"{tag}|{okey}"] = dict(fact, state=(plan["state"] or "no_current_anchor"), why=reason, currency_why=choice.why,
                                                       **({"window_matched": dict(plan["window_matched"])} if plan.get("window_matched") else {}))
                park(choice.why, f"contrast has no current anchor at {key_text} [own anchor {own_text}: {choice.own_reason}]; filed without a contrast: {choice.why}")
                continue
            if plan.get("window_matched") and choice.run_id != own_id:
                # the window-matched own none is retired or unfolded and the currency's pick at the exact key is ANOTHER serving (the key has no window
                # term): the window rule admits no such baseline, so the comparison parks by name instead of re-anchoring across windows
                why = (f"the window rule admits no baseline but the u0 `none` served at the row's window ({plan['window_matched']['row_window']}); the own one "
                       f"({own_text}) is {choice.own_reason} and the currency's pick {choice.run_id} ({choice.anchor_kind}) is another serving ({WINDOW_ANCHOR_RULE[:80]}...)")
                block.update(state="no_current_anchor", reason=why, anchor_run_id=None, currency_state=choice.state, currency_own_reason=choice.own_reason,
                             currency_why=choice.why, anchor_selection=selection)
                st["payload"]["extra"]["anchor"] = block
                facts["contrasts"][f"{tag}|{okey}"] = dict(fact, state="no_current_anchor", why=why, currency_why=choice.why, window_matched=dict(plan["window_matched"]))
                park(why, f"contrast's window-matched anchor {own_text} is {choice.own_reason}; no baseline at the row's serving window is current; filed without a contrast")
                continue
            assert choice.run_id is not None and choice.measurement is not None
            reanchored = choice.state != "own_current"
            same_standup = False
            tier_key: Optional[str] = None
            if reanchored:
                # a RE-PICKED anchor (the own one retired, unfolded or absent), by the currency's tier: (a) the row's OWN standup's unsteered run the
                # registry holds current from an earlier read -- the same tag under this source, absent from or refused in this batch (a partial re-mirror) --
                # pairs within the standup and keeps same_standup_none; (b) the same model's unsteered run of another standup under this source; (c) the paper's
                # roster-matched baseline of another source. The kind flows through the plan (same_standup_current / sibling_current / borrowed_reference,
                # anchor_kind_for) and is checked against the currency's own below; the standup fields follow the same tier
                cand = next((c for c in choice.candidates if c.run_id == choice.run_id), None)
                same_standup = choice.anchor_kind == AC.KIND_SAME_STANDUP
                tier_key = "borrowed_reference" if choice.state == "borrowed_reference" else ("same_standup_current" if same_standup else "sibling_current")
                plan = dict(plan, cross_standup=not same_standup, anchor_tag=(cand.view_legacy if cand is not None else None), **{tier_key: True})
                anchor = by_run.get(choice.run_id)          # a row of THIS batch (another tag's `none`): paired per item like the own one; else unpaired from k / n
                block.update(anchor_view_legacy=plan["anchor_tag"], cross_standup=plan["cross_standup"], reanchored=True, reanchor_tier=tier_key, own_anchor_run_id=own_id,
                             own_anchor_state=(plan["state"] or "retired"),
                             anchor_standup={"lane_source_legacy": cand.lane_source_legacy, "view_legacy": cand.view_legacy, "cluster_job_id": cand.cluster_job_id} if cand is not None else None)
            if anchor_kind_for(plan) != choice.anchor_kind:
                # the one rule's kind and the currency's must agree on every filed contrast (own: the kind the key declared; re-picked: the tier's) -- a reader
                # defect is refused by name, never filed under either label
                raise AdapterRefusal(self.root, f"{tag}/{okey}: the contrast's kind by anchor_kind_for ({anchor_kind_for(plan)}) disagrees with the anchor currency's "
                                                f"({choice.anchor_kind}; state {choice.state}, anchor {choice.run_id}); refused rather than filed under either label")
            block.update(currency_state=choice.state, currency_own_reason=choice.own_reason, anchor_selection=selection)
            m = choice.measurement
            retired = f"own anchor {own_text}: {choice.own_reason}" + (f" ({choice.own_why})" if choice.own_why else "")
            chosen = f"{choice.run_id} ({choice.anchor_kind}" + (", another source" if choice.cross_source else "") + ")"
            common_extra: Dict[str, Any] = {"unit": "pp", "anchor_view_legacy": plan["anchor_tag"], "anchor_object_key": "none", "anchor_rule": ANCHOR_RULE, "cross_standup": plan["cross_standup"],
                                            "row_tag_legacy": st.get("tag"), "anchor_selection": selection, "currency_rule": CURRENCY_RULE}
            if anchor is not None:
                # the anchor is a row of this batch: paired per item over the items SCORED on both sides
                a_items = anchor["items"]
                shared = sorted(i for i in s_items if i in a_items and s_items[i]["value"] is not None and a_items[i]["value"] is not None)
                if not shared:
                    why = (f"no item scored on both sides (anchor {chosen})" if reanchored else "no item scored on both sides")
                    block.update(state="no_shared_items", reason=why, anchor_run_id=anchor["run_id"])
                    st["payload"]["extra"]["anchor"] = block
                    facts["contrasts"][f"{tag}|{okey}"] = dict(fact, anchor=anchor["run_id"], state="no_shared_items", why=why)
                    park(why, f"contrast's anchor {chosen} is current but shares no scored item with the row at {key_text}; filed without a contrast")
                    continue
                b = sum(1 for i in shared if s_items[i]["value"] == 1.0 and a_items[i]["value"] == 0.0)
                c = sum(1 for i in shared if s_items[i]["value"] == 0.0 and a_items[i]["value"] == 1.0)
                p_s = sum(s_items[i]["value"] for i in shared) / len(shared)
                p_a = sum(a_items[i]["value"] for i in shared) / len(shared)
                stat = {"delta": 100.0 * (p_s - p_a), "delta_p": DC.mcnemar_exact(b, c), "delta_method": CONTRAST_METHOD, "effect_size": DC.cohen_h(p_s, p_a), "effect_kind": "cohens_h",
                        "n_pairs": len(shared)}
                stat_extra = {"discordant_steered_only": b, "discordant_anchor_only": c, "acc_steered_over_pairs": p_s, "acc_anchor_over_pairs": p_a,
                              "n_excluded_steered_unscored": sum(1 for i in s_items if s_items[i]["value"] is None),
                              "n_excluded_anchor_unscored": sum(1 for i in a_items if a_items[i]["value"] is None),
                              "method_text": f"paired per item over the items SCORED on both sides (the same {n_items} items, one instrument, the greedy protocol): the accuracy "
                                             "difference, pp; p from McNemar's exact test on the discordant pairs; Cohen's h of the two accuracies; an unscored item on either side leaves the pair"}
                if reanchored:
                    stat_extra["pairing"] = f"paired per item: the re-picked anchor {chosen} is a row of this batch ({plan['anchor_tag']}/none)"
                n_pairs = len(shared)
            else:
                # the anchor is a run of the registry, not of this batch: the registry's UNPAIRED rate difference from the two readings' k / n (rule-scored, no judge)
                try:
                    ust = DC.unpaired_rate_contrast(k_s, n_s, m.k, m.n)
                except DC.ContrastRefusal as exc:
                    why = f"the pairing against {retired}; the registry's statistic against the re-picked {chosen} is not derived: {exc.why}"
                    block.update(state="no_current_anchor", reason=why, anchor_run_id=None)
                    st["payload"]["extra"]["anchor"] = block
                    facts["contrasts"][f"{tag}|{okey}"] = dict(fact, state="no_current_anchor", why=why)
                    park(why, f"contrast's {retired}; no registry-derived statistic against the re-picked {chosen}: {exc.why}; filed without a contrast")
                    continue
                comp_row = DC.reanchor_computation("unpaired_rates", str(ctx.producer.get("code_sha") or "unverified"))
                self._reanchor_computations.setdefault(str(comp_row["computation_id"]), comp_row)
                stat = {k: ust[k] for k in ("delta", "delta_lo", "delta_hi", "delta_p", "delta_method", "effect_size", "effect_kind", "n_pairs")}
                stat_extra = {**dict(ust["extra"]), "statistic_computation_id": comp_row["computation_id"], "anchor_reading": m.reading_json(),
                              "pairing": f"unpaired: the re-picked anchor {chosen} is not a row of this batch; the registry's rate difference from the two readings' k / n",
                              "method_text": (f"unpaired rate difference of the row's recount (correct / scored, k {k_s} of n {n_s}) against the anchor's registry reading "
                                              f"(k {m.k:g} of n {m.n}) in pp; Newcombe hybrid score interval; two-proportion z; Cohen's h")}
                n_pairs = 0
            if reanchored:
                cand = next((c for c in choice.candidates if c.run_id == choice.run_id), None)
                if same_standup:
                    picked = f"its own standup's `none` the registry holds current from an earlier read ({plan['anchor_tag']}/none, {chosen}); the pair stays within the standup"
                elif choice.cross_source:
                    picked = f"the same model's `none` served by another standup of another source, the paper's roster-matched baseline ({plan['anchor_tag']}, {chosen})"
                else:
                    picked = f"the same model's `none` served by another standup ({plan['anchor_tag']}, {chosen})"
                reanchor_note = (f"the row's own anchor ({own_text}) is {choice.own_reason}: the pass's anchor currency re-anchors the row on {picked}; "
                                 f"{CURRENCY_RULE}")
                common_extra.update(cross_source=bool(choice.cross_source), contrast_caps={"steered": akey.token_cap_tokens, "anchor": akey.token_cap_tokens},
                                    anchor_standup=({"lane_source_legacy": cand.lane_source_legacy, "view_legacy": cand.view_legacy, "cluster_job_id": cand.cluster_job_id} if cand is not None else None),
                                    anchor_rule=CURRENCY_RULE, reanchor_tier=tier_key,
                                    own_pairing={"own_anchor_run_id": own_id, "own_anchor_kind": own_kind, "state": plan["state"], "why": plan["why"],
                                                 "own_none_state": plan["own_none_state"], "own_reason": choice.own_reason, "own_why": choice.own_why},
                                    reanchor_note=reanchor_note, **({"cross_standup_note": reanchor_note} if plan["cross_standup"] else {}))
            payload = {"run_id": st["run_id"], "anchor_run_id": choice.run_id, "judgment_id": "", "metric_id": HEADLINE_METRIC, "threshold": -1.0,
                       "computation_id": comp_id, "anchor_judgment_id": choice.anchor_judgment_id, "anchor_kind": anchor_kind_for(plan),
                       **stat, "computed_by": "registry", "computed_utc": st["when"], "extra": {**common_extra, **stat_extra}}
            if spec.shared is not None and anchor is not None:
                payload["extra"].update(standup_ids={"row": st.get("standup_id"), "anchor": anchor.get("standup_id")}, anchor_standup_rule=SHARED_ANCHOR_STANDUP_RULE,
                                        comparison_kind=st.get("comparison_kind"), in_run_reference=True)
            if plan.get("window_matched"):
                # the unsteered row's none at the row's serving window (WINDOW_ANCHOR_RULE, main 2026-10-05): the pairing rides on every contrast of such a row; the
                # rule text is the window's where the currency kept the plan's anchor, the re-anchoring's (`reanchor_note` above) where it re-picked
                payload["extra"].update(window_matched=dict(plan["window_matched"]), anchor_row_key=plan["window_matched"].get("anchor_row_key"))
                if not reanchored:
                    payload["extra"]["anchor_rule"] = WINDOW_ANCHOR_RULE
                    if plan["cross_standup"]:
                        payload["extra"]["cross_standup_note"] = (f"the row's tag {tag} has no `none` of its own: the family's baseline is the unsteered row's none ({plan['anchor_tag']}/none) served at the "
                                                                  f"same window ({plan['window_matched']['row_window']}) by another standup -- D1878's same-source sibling tier")
            elif reanchored:
                pass        # the re-anchoring text is on `reanchor_note` (and `cross_standup_note` across standups) / `own_pairing` above
            elif it_repeat:
                # the instruct row's repeat floor (SHARED_IT_REPEAT_RULE): the pair is the row's own none, the ONE contrast that run anchors
                payload["extra"].update(anchor_rule=SHARED_IT_REPEAT_RULE, repeat_floor_of_own_none=True,
                                        floor_note=(f"the instruct row's own none ({tag}/none, the it_none_comparison run) served a second time on the same standup: the in-run "
                                                    "floor pairs the two servings of one object; the `_it` none has no solo twin, so no borrowed contrast"))
            elif instruct:
                # the alias (INSTRUCT_ANCHOR_RULE): the row's own `none` is the group's none by declaration, never a store of its own
                payload["extra"].update(anchor_rule=INSTRUCT_ANCHOR_RULE, anchor_alias=dict(st["payload"]["extra"]["anchor_alias"]),
                                        cross_standup_note=instruct_anchor_note(group, str(st.get("tag"))))
            elif standup_note is not None:
                payload["extra"]["cross_standup_note"] = standup_note
            elif plan["cross_standup"]:
                payload["extra"]["cross_standup_note"] = cross_standup_note(group, str(st.get("tag")), plan)
                if plan["own_none_state"] is not None:
                    payload["extra"].update(own_none_state=plan["own_none_state"], own_none_note=plan["own_none_note"])
            elif st.get("paired"):
                payload["extra"].update(anchor_rule=PAIRED_ANCHOR_RULE, own_standup_note=own_standup_note(group, str(st.get("tag"))))
            if st.get("declared_role") is not None:
                payload["extra"].update(declared_role=st["declared_role"],
                                        anchor_standup_note=(f"the value is declared {st['declared_role']}: the single-object standup composes another max_lora_rank than the seed tag's "
                                                             "(RECORDED on both runs, never compared); the contrast is the reader's, formed as for every steered key, and the role rides on it"))
            out.append(payload)
            if value_row is not None:
                value_row.setdefault("extra", {})["delta_block"] = None
            facts["contrasts_filed"] += 1
            facts["contrasts_reanchored" if reanchored else "contrasts_own"] += 1
            block.update(state=ANCHOR_CONTRASTED, anchor_run_id=choice.run_id, n_pairs=n_pairs)
            st["payload"]["extra"]["anchor"] = block
            facts["contrasts"][f"{tag}|{okey}"] = dict(fact, anchor=choice.run_id, state=ANCHOR_CONTRASTED, n_pairs=n_pairs, delta_pp=payload["delta"], delta_p=payload["delta_p"],
                                                       cohens_h=payload["effect_size"], cross_standup=plan["cross_standup"], declared_role=st.get("declared_role"),
                                                       anchor_view_legacy=plan["anchor_tag"], own_standup=(plan["own"] and not reanchored), reanchored=reanchored,
                                                       **({"reanchor_tier": tier_key} if reanchored else {}),
                                                       **({"anchor_alias": True} if instruct and not it_repeat and not reanchored else {}),
                                                       **({"repeat_floor_of_own_none": True} if it_repeat else {}),
                                                       **({"window_matched": dict(plan["window_matched"])} if plan.get("window_matched") else {}))
        return out

def it_repeat_anchor_plan(tag: str, filed: Mapping[Tuple[str, str], Mapping[str, Any]], expected: Mapping[Tuple[str, str], Any], seen: Set[Tuple[str, str]]) -> Dict[str, Any]:
    """The pair of an instruct row's repeat floor (:data:`SHARED_IT_REPEAT_RULE`): the row's OWN none -- the it_none_comparison run of the same tag, served by the same
    standup (``own``, ``cross_standup`` false: same_standup_none) -- never the group's none by alias (the instruct-direct rows' anchor) and never a solo store; when the
    row's none has not filed the floor waits, with the reason in the readers' standing vocabulary."""
    own = filed.get((tag, "none"))
    plan: Dict[str, Any] = {"anchor": own, "anchor_tag": (tag if own is not None else None), "own": own is not None, "cross_standup": False, "own_none_state": None,
                            "own_none_note": None, "state": None, "why": None, "repeat_floor": True}
    if own is None:
        own_key = (tag, "none")
        if own_key in expected and own_key not in seen:
            plan.update(state=ANCHOR_NOT_YET_RUN, why=f"the instruct row's own `none` ({tag}/none) has not run yet (the launch set names it, no store is mirrored); the repeat floor pairs against it alone")
        elif own_key in seen:
            plan.update(state=ANCHOR_REFUSED, why=f"the instruct row's own `none` ({tag}/none) refused; the repeat floor pairs against it alone, so no contrast is formed")
        else:
            plan.update(state=ANCHOR_NOT_YET_RUN, why=f"the launch set names no `none` for {tag}; the repeat floor pairs against it alone, so no contrast is formed")
    return plan


def instruct_anchor_plan(group: str, filed: Mapping[Tuple[str, str], Mapping[str, Any]], expected: Mapping[Tuple[str, str], Any], seen: Set[Tuple[str, str]]) -> Dict[str, Any]:
    """The anchor of an instruct-direct row (:data:`INSTRUCT_ANCHOR_RULE`): the group's ``none`` BY ALIAS -- never its own tag's (no ``_it`` none store exists), so
    ``cross_standup`` is true and no own-none state is pending; when the group's ``none`` has not filed the row waits with the seed tag's reasons."""
    seed = filed.get((group, "none"))
    plan: Dict[str, Any] = {"anchor": seed, "anchor_tag": (group if seed is not None else None), "own": False, "cross_standup": True, "own_none_state": None, "own_none_note": None,
                            "state": None, "why": None, "alias": True}
    if seed is None:
        seed_key = (group, "none")
        if seed_key in expected and seed_key not in seen:
            plan.update(state=ANCHOR_NOT_YET_RUN, why=f"the aliased anchor ({group}/none) has not run yet (the launch set names it, no store is mirrored); no contrast is formed until it lands")
        elif seed_key in seen:
            plan.update(state=ANCHOR_REFUSED, why=f"the aliased anchor ({group}/none) refused; no contrast is formed")
        else:
            plan.update(state=ANCHOR_NOT_YET_RUN, why=f"the launch set names no `none` for {group}, the anchor the instruct-direct row aliases to; no contrast is formed")
    return plan


def window_anchor_plan(u_tag: str, tag: str, st: Mapping[str, Any], filed: Mapping[Tuple[str, str], Mapping[str, Any]], expected: Mapping[Tuple[str, str], Any],
                       seen: Set[Tuple[str, str]], default_window: Optional[int] = None) -> Dict[str, Any]:
    """The anchor of a row of a family with an unsteered row (:data:`WINDOW_ANCHOR_RULE`): the u0 ``none`` that FILED at the row's serving window (and max_num_seqs where
    both carry it) -- own (``same_standup_none``) for a u0 row, across tags (``sibling_none``) for an s1 row. No none filed at that window: the declared one
    (``u0/none`` at the family's default window, ``u0/none@w<N>`` at another) waits as ``anchor_not_yet_run`` / ``anchor_refused``; none declared refuses the contrast
    by name (``no_same_window_none``); two filed at one window is ``ambiguous_window_anchor``."""
    row_w, row_s = st.get("window"), st.get("max_num_seqs")
    own = tag == u_tag
    plan: Dict[str, Any] = {"anchor": None, "anchor_tag": u_tag, "own": own, "cross_standup": not own, "own_none_state": None, "own_none_note": None, "state": None, "why": None,
                            "window_matched": {"row_window": row_w, "row_max_num_seqs": row_s, "anchor_window": None, "anchor_max_num_seqs": None, "anchor_row_key": None,
                                               "rule": WINDOW_ANCHOR_RULE}}
    nones = {k: v for k, v in filed.items() if k[0] == u_tag and split_row_okey(k[1])[0] == "none"}
    same = {k: v for k, v in nones.items() if v.get("window") == row_w and (row_s is None or v.get("max_num_seqs") is None or v.get("max_num_seqs") == row_s)}
    if len(same) == 1:
        (k, v), = same.items()
        plan["anchor"] = v
        plan["window_matched"].update(anchor_window=v.get("window"), anchor_max_num_seqs=v.get("max_num_seqs"), anchor_row_key=f"{k[0]}/{k[1]}")
        return plan
    if len(same) > 1:
        plan.update(state="ambiguous_window_anchor", why=f"{len(same)} u0 `none` runs were served at the row's window ({row_w} x {row_s}): " + ", ".join(sorted(f"{k[0]}/{k[1]}" for k in same))
                                                        + "; one baseline per serving window is the contract, so no contrast is formed")
        return plan
    # the declared u0 nones at the row's window: a FILED one is at its own run's window (never assumed), an unfiled one at its key's `@w<N>` or the family's
    # default window; an unknown window (no `@w<N>`, no default declared) is admitted as a candidate rather than silently excluded
    def _declared_window(k: Tuple[str, str]) -> Optional[int]:
        if k in filed:
            return filed[k].get("window")
        return window_of_row_okey(k[1]) or default_window
    declared = sorted(k for k in expected if k[0] == u_tag and split_row_okey(k[1])[0] == "none" and _declared_window(k) in (row_w, None) and k not in filed)
    if not declared:
        plan.update(state="no_same_window_none", why=f"no u0 `none` is declared at the row's serving window ({row_w} x {row_s}); the baseline must not differ by window "
                                                   f"(the launch set names {sorted(f'{k[0]}/{k[1]}' for k in expected if k[0] == u_tag and split_row_okey(k[1])[0] == 'none')}), so no contrast is formed")
    elif any(k in seen for k in declared):
        plan.update(state=ANCHOR_REFUSED, why=f"the u0 `none` at the row's window ({'/'.join(declared[0])}) refused; no contrast is formed")
    else:
        plan.update(state=ANCHOR_NOT_YET_RUN, why=f"the u0 `none` at the row's window ({'/'.join(declared[0])}) has not run yet (the launch set declares it, no store is mirrored); no contrast is formed until it lands")
    return plan


def instruct_anchor_note(group: str, row_tag: str) -> str:
    """The note an instruct-direct row's contrast carries: the anchor is the group's ``none`` by alias, served by another standup."""
    return (f"the instruct-direct row's tag {row_tag} has no `none` store of its own: its none is carried BY ALIAS to {group}/none (the launch set's anchor_tuple), served by "
            f"another standup; the contrast pairs against it")


def _sel_of(tags: Mapping[str, str], tag: str) -> str:
    """The selection of a tag in the group's ``{selection: tag}`` map."""
    for sel, t in tags.items():
        if t == tag:
            return sel
    raise KeyError(tag)

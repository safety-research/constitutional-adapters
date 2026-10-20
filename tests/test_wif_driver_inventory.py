"""The INVENTORY of every code path that sends a request to Claude, and the stop each is built on (the WIF driver sweep, 2026-10-13).

This module is the sweep's completeness check: a SOURCE-level walk of every tracked Python file under ``src/``, ``scripts/``,
``reproduce/`` and ``corpus_generator/`` (tests excluded) for the SDK's request methods -- ``.messages.create`` / ``.parse`` /
``.stream`` / ``.count_tokens`` and ``.messages.batches.<create|retrieve|results|list|cancel>`` -- and an EXACT inventory of the
files that call them, each with the stop mechanism its request path is built on (a substring the file must carry). A new caller
fails here until it is inventoried with its stop; a stop removed from an inventoried file fails here too; a file that stops
calling is a stale entry and fails as well. The behavioural half of each entry is the test module its note names.

Round 9 made the walk TRANSITIVE: a request site is not the only way to reach Claude -- a script that builds a
shared judge, grader, engine or client (``ClaudeJudge``, ``LiveStrongREJECTGrader``, ``BatchJudge``, ``AnthropicRewriter``, the
corpus generator's ``LiveEngine`` / ``WaveEngine``, ``vault.anthropic_client``, inspect's ``get_model`` under the federated
provider, ...) reaches it through that object. ``ENTRY_POINTS`` names every such constructor / entry point by the module that
defines it; the walk resolves each file's imports (so a same-named function of another module -- the served target's
``_make_client``, ``capeval.runtime.make_engine`` -- is not confused with one) and also matches the names that exist nowhere else in
the tree when they are reached through a dynamic import. Every file it finds must be inventoried -- as a request site, an
INDIRECT reacher, a key-handing launcher, one of the CONSTRUCTORS themselves, or a NOT_CLAUDE caller of an entry point that also
serves another provider -- each with its first-401 behaviour and the pin the test enforces. A fourth walk reads every tracked SHELL
file for a client or request embedded in a Python heredoc (``SHELL_REACHERS``: the lane finisher's probe), which no Python walk sees.

The mechanisms (all in ``sisteer.vault`` or built on it): ``credential_rejection`` (the one classifier: 401 / 403 on the request,
and the federation identity refused at the token exchange), ``CredentialLatch`` / ``map_until_credential_rejected`` /
``cancel_queue`` (the concurrent drivers), ``credential_exit`` (the typed terminal line of a driver's main), and the corpus
generator's twin ``credential_halt`` / ``credential_exit`` (its member imports no sisteer)."""
from __future__ import annotations

import ast
import functools
import re
import subprocess
import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCOPES = ("src/", "scripts/", "reproduce/", "corpus_generator/")
EXCLUDED = ("legacy/",)
REQUEST_METHODS = frozenset({"create", "parse", "stream", "count_tokens"})
BATCH_METHODS = frozenset({"create", "retrieve", "results", "list", "cancel"})

#: file -> (how its request path stops at the first credential rejection, the substrings the file must carry). `n.a.` names a
#: single synchronous call whose first rejection is its last request; such a file still carries the typed exit or the classifier.
INVENTORY = {
    # --- the concurrent judges and graders (a latch, the queue cancelled, the typed exit)
    "scripts/judge_sonnet.py": ("Judge._call classifies every exception first (vault.credential_rejection), latches, abort_queue cancels; "
                                "main exits typed (tests/test_judge_sonnet_credential_stop.py)",
                                ("vault.credential_rejection(", "cancel_futures=True", "raise SystemExit(f\"[judge] {e}\")")),
    "scripts/renellm_intent_judge.py": ("a process-level latch, _collect cancels the queue, __main__ exits typed "
                                        "(tests/test_renellm_intent_judge_credential_stop.py)",
                                        ("vault.credential_rejection(", "cancel_futures=True", "raise SystemExit(f\"[judge] {e}\")")),
    "scripts/judge_overrefusal.py": ("vault.CredentialLatch + vault.map_until_credential_rejected (the queue cancelled), the latch checked "
                                     "before EVERY attempt of `one`'s 8-attempt loop (round 11), main under vault.credential_exit "
                                     "(tests/test_credential_stop_sweep.py, tests/test_live_grader_inflight_rejection_stop.py)",
                                     ("vault.CredentialLatch(", "vault.map_until_credential_rejected(", "vault.credential_exit(",
                                      "latch.check()")),
    "scripts/grokmt_analyze.py": ("vault.CredentialLatch + vault.map_until_credential_rejected (no ERROR label cached for a rejection), "
                                  "main under vault.credential_exit (tests/test_credential_stop_sweep.py)",
                                  ("vault.CredentialLatch(", "vault.map_until_credential_rejected(", "vault.credential_exit(")),
    "scripts/grade_bon_live.py": ("_forced_tool_grade raises the typed error after ONE call and checks the latch its CLIENT carries before "
                                  "EVERY attempt (vault.client_credential_latch; round 11: a worker asleep in its 429 / 529 backoff wakes "
                                  "into the stop, never into another request), _grade_tasks_concurrently attaches that latch to its bare "
                                  "client and cancels the queue, __main__ exits typed (tests/test_grading_live.py, "
                                  "tests/test_credential_stop_sweep.py, tests/test_live_grader_inflight_rejection_stop.py)",
                                  ("vault.credential_rejection(", "cancel_futures=True", "vault.credential_exit(",
                                   "vault.client_credential_latch(", "latch.check()")),
    "scripts/sdfbase/full22/misalign/misalign_judge.py": ("PoolClient.preflight before any claim, the PoolClient latch, main cancels the "
                                                          "queue and exits RC_CREDENTIAL_REJECTED (tests/test_grade_wif_client.py, "
                                                          "tests/test_misalign_judge*.py)",
                                                          ("vault.credential_rejection(", "cancel_futures=True", "RC_CREDENTIAL_REJECTED")),
    "scripts/sdfbase/build_probe_set.py": ("_call_api classifies first and latches on the limiter (vault.CredentialLatch), "
                                           "_gather_until_credential_rejected cancels the tasks, verify_items never files api_error for it, "
                                           "main under vault.credential_exit (tests/test_probe_set_credential_stop.py)",
                                           ("vault.CredentialLatch(", "limiter.credential.rejection(", "_gather_until_credential_rejected(",
                                            "vault.credential_exit(")),
    "scripts/sdfbase/seedrep/grade_full22_native.py": ("the BoN judge's BoundedJudgeClient forwards the SDK error to _forced_tool_grade "
                                                       "(typed after one call) and checks the grader's latch -- carried by the raw client it "
                                                       "wraps -- before EVERY attempt, its one retry's wake included (round 11); "
                                                       "LiveStrongREJECTGrader latches and cancels; the typed error is not a RuntimeError so "
                                                       "no re-issue loop catches it; main under vault.credential_exit (round 9); the "
                                                       "process-environment pin is the vault's (tests/test_grading_live.py, "
                                                       "tests/test_wif_process_environment.py, "
                                                       "tests/test_live_grader_inflight_rejection_stop.py)",
                                                       ("_forced_tool_grade", "vault.pin_process_environment()", "vault.credential_exit(",
                                                        "vault.client_credential_latch(c._inner)")),
    "src/sisteer/eval/attacks/llm_rewriter.py": ("_CachedConcurrentRewriter: vault.CredentialLatch around _call_one, batch through "
                                                 "vault.map_until_credential_rejected (tests/test_credential_stop_sweep.py)",
                                                 ("vault.CredentialLatch(", "vault.map_until_credential_rejected(")),
    "src/sisteer/capeval/judging.py": ("ClaudeJudge._create classifies and latches on the judge and the client, judge_all stops; "
                                       "measure_token_bounds through vault.map_until_credential_rejected (latched on the client); "
                                       "_liveness_canary raises typed (tests/test_capeval_judging.py, tests/test_credential_stop_sweep.py)",
                                       ("vault.credential_rejection(", "vault.map_until_credential_rejected(", "CREDENTIAL_LATCH_ATTR")),
    "scripts/alphasteer_f22/grade_alphasteer_f22.py": ("the grader (LiveStrongREJECTGrader) stops; its judge-shedding canary classifies the "
                                                       "credential FIRST (round 9: a rejection at the canary used to read `other:` and the "
                                                       "re-issue loop re-graded on the rejected credential), holds it on the grader and raises "
                                                       "typed; main under vault.credential_exit; the process-environment pin is the vault's "
                                                       "(tests/test_alphasteer_canary_credential_stop.py, tests/test_wif_process_environment.py)",
                                                       ("LiveStrongREJECTGrader", "def judge_shedding(", "vault.credential_rejection(",
                                                        "hold_credential_rejection(", "vault.credential_exit(", "vault.pin_process_environment()")),
    # --- the Batch API (single synchronous calls: the first rejection is the last request; typed)
    "scripts/judge_batch.py": ("_create_chunk classifies (vault.credential_rejection), drops the false journal entry and raises typed; "
                               "main under vault.credential_exit (tests/test_batch_judges_credential_stop.py)",
                               ("vault.credential_rejection(", "vault.credential_exit(")),
    "scripts/sdfbase/full22/judge/batch_judge_fill.py": ("n.a. (synchronous); main under vault.credential_exit "
                                                         "(tests/test_batch_judges_credential_stop.py)",
                                                         ("vault.credential_exit(",)),
    # --- the corpus generator (its own vault twin: client.credential_rejection, batch.credential_halt / credential_exit)
    "corpus_generator/src/corpus_generator/live.py": ("the engine latch, the pool cancelled, WaveCredentialRejected "
                                                      "(corpus_generator/tests/test_live_engine.py)",
                                                      ("credential_halt(", "WaveCredentialRejected")),
    "corpus_generator/src/corpus_generator/batch.py": ("WaveEngine._guard around every control-plane call and _guard_iter around the "
                                                       "iteration of its answer (the list's later pages, the results stream), the latch on "
                                                       "the engine AND on the shared CLIENT (round 9: sibling engines of one client -- "
                                                       "`corpusgen run --corpus all`, the extension sub-DAGs, `onpolicy score` -- each issued "
                                                       "its own doomed create), WaveCredentialRejected; credential_exit for a script's main "
                                                       "(corpus_generator/tests/test_batch_engine_credential_stop.py)",
                                                       ("self._guard(", "self._guard_iter(", "def credential_halt(", "def credential_exit(",
                                                        "CLIENT_LATCH_ATTR", "tmp.unlink(missing_ok=True)")),
    "corpus_generator/src/corpus_generator/smoke.py": ("n.a.: <= 5 sequential live calls on the smoke client, the first rejection is a raw "
                                                       "stop before the batch half (a 400 is the only exception caught)",
                                                       ("except anthropic.BadRequestError",)),
    "corpus_generator/src/corpus_generator/cli.py": ("n.a.: `status --live` retrieves one batch at a time; the first rejection ends the "
                                                     "command; every command runs under the group's credential_exit (round 9: the typed "
                                                     "terminal line, never a traceback)", ("batches.retrieve(", "credential_exit(")),
    "corpus_generator/scripts/archive_cancelled_batches.py": ("an operator's sequential archive of a cancelled run: every retrieve / "
                                                              "results call and every entry of the stream under _stop_on_credential "
                                                              "(credential_halt -> the typed text as the terminal line); each archive "
                                                              "streamed to a .tmp, published by rename only after the whole set matched "
                                                              "the request counts, the tmp removed on every failure; no CANCELLED.json "
                                                              "(corpus_generator/tests/test_batch_engine_credential_stop.py)",
                                                              ("credential_halt(", "def _stop_on_credential(", "os.replace(tmp, out)",
                                                               "tmp.unlink(missing_ok=True)")),
    "corpus_generator/scripts/register_edit_precheck.py": ("create_or_stop: credential_halt stops the sequential pre-check at the first "
                                                           "rejection (corpus_generator/tests/test_batch_engine_credential_stop.py)",
                                                           ("credential_halt(",)),
}


#: files that reach Claude only THROUGH an inventoried module or constructor (ClaudeJudge, LiveStrongREJECTGrader, BatchJudge,
#: AnthropicRewriter, the corpus engines, the federated provider factory): no request call of their own to scan for, so they
#: are pinned by the stop they ride. Base URL under wif for every row: the constructor's pin (the vault's, the corpus
#: client's constant, the inspect install's).
INDIRECT = {
    "src/sisteer/eval/attacks/grading_live.py": ("LiveStrongREJECTGrader's OWN vault.CredentialLatch, handed to its ThrottledClient (which "
                                                 "attaches it to the inner client) so the wrapper's 429 loop, _forced_tool_grade's loop and "
                                                 "the BoN twin's BoundedJudgeClient each check it before EVERY attempt (round 11); _grade "
                                                 "holds the typed error and cancels its queue; the grader asks the vault for the BASE's retry "
                                                 "count in every mode (8 clamped to 2; round 12 on fd06740ae) and the vault decides where it "
                                                 "lives (op / env: the SDK's own; wif: the SDK's retries off, the count the latched layer's "
                                                 "budget -- a prebuilt client carrying SDK retries refused by name under wif only); "
                                                 "hold_credential_rejection lets a caller's canary hold its rejection here "
                                                 "(tests/test_grading_live.py, tests/test_alphasteer_canary_credential_stop.py, "
                                                 "tests/test_live_grader_inflight_rejection_stop.py, "
                                                 "tests/test_wif_federated_destination_and_sdk_retries.py)",
                                                 ("JudgeCredentialRejected", "cancel_futures=True", "def hold_credential_rejection(",
                                                  "self._latch = vault.CredentialLatch(", "latch=self._latch",
                                                  "self._sdk_max_retries = min(int(max_retries), 2)",
                                                  "vault.secrets_mode() == vault.WIF_MODE")),
    "scripts/served_capability_judge.py": ("ClaudeJudge over ONE vault client per run: the first rejection latches on the client "
                                           "(CREDENTIAL_LATCH_ATTR), every later cell raises before its first call; JUDGE_FAILED names it "
                                           "(tests/test_grade_wif_client.py)",
                                           ("vault.anthropic_client(",)),
    "scripts/alphasteer_f22/ladder_select.py": ("ClaudeJudge over one client (the client latch); the legs fail fast; the failed marker "
                                                "names the typed error", ("vault.anthropic_client(",)),
    "scripts/capeval_judged_run.py": ("ClaudeJudge (its latch) and measure_token_bounds (typed); main under vault.credential_exit "
                                      "(tests/test_credential_stop_sweep.py)", ("vault.credential_exit(",)),
    "src/sisteer/eval/anthropic_wif.py": ("_LatchedMessages around every federated provider client: the first rejection as the SDK raised "
                                          "it, latched process-wide, and the running inspect tasks ABORTED through inspect's own cancel "
                                          "handle (abort_inspect_tasks); every later create / stream / count_tokens raises before the wire; "
                                          "the provider's client factory is bypassed for the plain API so no ANTHROPIC_* credential / URL "
                                          "variable is read, both factories scrub the inherited ANTHROPIC_* names before the SDK's "
                                          "constructor (round 10), and both factories -- and every request -- REFUSE BY NAME an explicit "
                                          "static X-Api-Key / Authorization header (default_headers, an http_client's headers, "
                                          "extra_headers; round 11: the SDK keeps it beside credentials= and skips the exchange) "
                                          "(tests/test_credential_stop_sweep.py, tests/test_anthropic_wif.py, "
                                          "tests/test_wif_inherited_static_auth.py, tests/test_wif_explicit_static_auth.py); round 12: "
                                          "the install's base_url and an http_client model argument are pinned to the API host BY NAME "
                                          "(vault.pin_federated_destination) and every federated client is built with the SDK's retries "
                                          "OFF and the count asked for carried by the vault's latched layer over the install's latch "
                                          "(vault.federate_sdk_retries; tests/test_wif_federated_destination_and_sdk_retries.py); round 13: "
                                          "both factories build on the vault's OWN httpx client over the pin guard, never a caller's "
                                          "(vault.federated_http_client; tests/test_wif_transport_boundary_and_request_headers.py)",
                                          ("vault.CredentialLatch(", "_LatchedMessages(", "def abort_inspect_tasks(",
                                           "vault.scrub_inherited_anthropic_environment()", "vault.refuse_static_auth_headers(",
                                           "prov.AnthropicAPI._create_client = create_client_federated",
                                           "vault.pin_federated_destination(", "vault.federate_sdk_retries(", "max_retries=0",
                                           "vault.federated_http_client(")),
    "src/sisteer/capeval/tiers/petri_runner.py": ("the federated provider's latch read after EVERY eval_set (_refuse_if_credential_rejected): "
                                                  "no later group launched, no log converted, the typed error raised; _run's H9 ledger names "
                                                  "the credential stop and files no row (tests/test_petri_credential_stop.py)",
                                                  ("anthropic_wif.credential_rejected()", "_refuse_if_credential_rejected(",
                                                   "CREDENTIAL_STOP_REASON_PREFIX")),
    "src/sisteer/capeval/runtime.py": ("the Petri dispatch seam: a thin lazy delegate to petri_runner.run_petri (the leg's stop above); "
                                       "its make_engine builds the SERVED target's engine, never a Claude client",
                                       ("def run_petri(", "petri_runner.run(")),
    "scripts/petri_run.py": ("cmd_run's run_petri under vault.credential_exit: the leg process ends with the typed line, never a "
                             "traceback, never a results.json (tests/test_petri_credential_stop.py)",
                             ("credential_exit(", "rt.run_petri(")),
    "src/sisteer/capeval/petri/auditor_fallback.py": ("the auditor / judge chains resolved through inspect's get_model: the Claude legs ride "
                                                      "the federated provider's latch and abort (anthropic_wif, installed by the leg before "
                                                      "its first get_model); the non-Claude legs (OpenRouter, a local Qwen) take their keys "
                                                      "in-process; a fallback leg's own failure is RefusalFallbackLegError, never a re-issue "
                                                      "(tests/test_anthropic_wif.py, tests/test_petri_credential_stop.py)",
                                                      ("from inspect_ai.model import get_model", "resolve_openrouter_api_key()",
                                                       "RefusalFallbackLegError(")),
    "scripts/arditi/grade_arditi_orbench.py": ("a LAUNCHER: the pinned served_capability_judge.py of a SEPARATE "
                                               "checkout run by path with the Low-Prio KEY in its environment (harness_env; every inherited "
                                               "ANTHROPIC_* dropped). Under wif _vault_lowprio -> vault.anthropic_api_key REFUSES BY NAME "
                                               "(PoolUnavailable -> Refuse) before the private cache is seeded and before the child is "
                                               "launched: a frozen keyed driver is never handed a federated client, and the shell's "
                                               "ANTHROPIC_API_KEY is never read in the key's place (tests/test_arditi_judge_wif_refusal.py)",
                                               ("vault.anthropic_api_key(JUDGE_ORG)", "def harness_env(", "_vault_lowprio()",
                                                "if not k.startswith(\"ANTHROPIC_\")")),
    # --- round 9: the callers of the shared graders / judges / engines the request-site walk could not see
    "src/sisteer/eval/static_grading.py": ("the shared static grading body: LiveStrongREJECTGrader (its latch + cancel), grade_fn re-issues "
                                           "only the UNSCORED RuntimeError so the typed error propagates to the lane's main; the "
                                           "process-environment pin is the vault's (tests/test_wif_process_environment.py)",
                                           ("LiveStrongREJECTGrader(", '"UNSCORED" not in str(e)', "vault.pin_process_environment()")),
    "scripts/sdfbase/full22/sampled/grade_static_sampled.py": ("LiveStrongREJECTGrader (its latch + cancel); the typed error caught per "
                                                               "cell, the unfiled claims released, SystemExit with the typed text; the "
                                                               "process-environment pin is the vault's (tests/test_sampled_grading.py, "
                                                               "tests/test_wif_process_environment.py)",
                                                               ("LiveStrongREJECTGrader(", "except vault.JudgeCredentialRejected",
                                                                "CREDENTIAL REJECTED", "vault.pin_process_environment()")),
    "scripts/sdfbase/full22/static_padding/grade_padded_static.py": ("the padded lane over static_grading.run: the typed error propagates "
                                                                     "(the registry claims released on any failure); main under "
                                                                     "vault.credential_exit", ("sg.run(", "vault.credential_exit(")),
    "scripts/asr_noise_grade.py": ("LiveStrongREJECTGrader.grade (latch + cancel); main under vault.credential_exit",
                                   ("LiveStrongREJECTGrader(", "vault.credential_exit(")),
    "scripts/grade_gated_served.py": ("LiveStrongREJECTGrader.grade as the gated grade_fn (latch + cancel); main under vault.credential_exit",
                                      ("LiveStrongREJECTGrader(", "vault.credential_exit(")),
    "scripts/grokmt_grade.py": ("LiveStrongREJECTGrader(org=args.org).grade (latch + cancel); main under vault.credential_exit",
                                ("LiveStrongREJECTGrader(", "vault.credential_exit(")),
    "scripts/placebo2/grade_rows.py": ("LiveStrongREJECTGrader.grade (latch + cancel); main under vault.credential_exit",
                                       ("LiveStrongREJECTGrader(", "vault.credential_exit(")),
    "scripts/two_si_run.py": ("live: LiveStrongREJECTGrader (latch + cancel); batch: BatchStrongREJECTGrader over judge_batch's synchronous "
                              "BatchJudge (typed at the create); main under vault.credential_exit",
                              ("LiveStrongREJECTGrader(", "BatchStrongREJECTGrader(", "vault.credential_exit(")),
    "scripts/jailbreak_run.py": ("live: LiveStrongREJECTGrader (latch + cancel); batch: BatchJudge pairs (typed at the create); main under "
                                 "vault.credential_exit", ("LiveStrongREJECTGrader(", "BatchJudge(", "vault.credential_exit(")),
    "scripts/build_two_si_pool.py": ("LiveStrongREJECTGrader.grade as the pool's grade fn (latch + cancel); main under vault.credential_exit",
                                     ("LiveStrongREJECTGrader(", "vault.credential_exit(")),
    "scripts/cell_controls_run.py": ("ClaudeJudge over the cell's controls (the judge + client latch: a later control raises before its "
                                     "first call); main under vault.credential_exit", ("judging.ClaudeJudge(", "vault.credential_exit(")),
    "scripts/capeval_rubric_validation.py": ("ClaudeJudge.judge_all (the latch); main under vault.credential_exit",
                                             ("ClaudeJudge(", "vault.credential_exit(")),
    "scripts/build_heldout_corpus.py": ("AnthropicRewriter.batch through build_llm_corpus (the rewriter's latch + bounded map; the manifest "
                                        "is written only after the batch); main under vault.credential_exit",
                                        ("AnthropicRewriter(", "vault.credential_exit(")),
    "scripts/heldout_compliance_check.py": ("AnthropicRewriter.batch (the rewriter's latch + bounded map); main under vault.credential_exit",
                                            ("AnthropicRewriter(", "vault.credential_exit(")),
    "scripts/build_safety_defense.py": ("BatchJudge pairs + grade_strongreject_rubric / grade_xstest (judge_batch: typed at the create, "
                                        "synchronous); main under vault.credential_exit", ("BatchJudge(", "vault.credential_exit(")),
    "src/sisteer/eval/attacks/grading.py": ("BatchStrongREJECTGrader: a verbatim wrapper of judge_batch.grade_strongreject over the "
                                            "caller's BatchJudge (typed at the create, synchronous)",
                                            ("from judge_batch import grade_strongreject",)),
    "corpus_generator/scripts/miniwave.py": ("LiveEngine.run_wave (the engine latch, the pool cancelled); main under the corpus "
                                             "credential_exit (round 9)", ("LiveEngine(", "credential_exit(")),
    "corpus_generator/src/corpus_generator/register_edit.py": ("build_live_engine -> LiveEngine (the engine latch); run through the CLI "
                                                               "(its group's credential_exit)", ("LiveEngine(",)),
    "corpus_generator/src/corpus_generator/pipeline.py": ("make_engine -> WaveEngine per corpus / extension thread over ONE client: the "
                                                          "first rejection latches on the client, so every sibling engine refuses before "
                                                          "its next control-plane call (round 9; corpus_generator/tests/"
                                                          "test_batch_engine_credential_stop.py)", ("def make_engine(", "WaveEngine(client, **kw)")),
}


#: the shared client WRAPPERS themselves (no request of their own): the vault's constructors, api_keys.make_client over them,
#: the corpus generator's client module -- found by the walk because each calls a constructor it defines or wraps
CONSTRUCTORS = {
    "src/sisteer/vault.py": ("the ONE place a client is built: anthropic_client / anthropic_async_client -> anthropic_client_wif under "
                             "wif (credentials=, the destination the pinned API host BY CONSTRUCTION -- round 12: a base_url or an "
                             "http_client naming another host refused by name, pin_federated_destination -- the inherited ANTHROPIC_* "
                             "names scrubbed first (round 10: the SDK reads ANTHROPIC_CUSTOM_HEADERS whatever credential it is given), "
                             "the SDK's own retries OFF and replayed above the client's latch (round 12: federate_sdk_retries / "
                             "LatchedRetryMessages, the count asked for as the budget), the stop helpers (round 11: the latch a client "
                             "carries for every retry layer, and the refusal of an explicit static credential header), and the "
                             "process-environment pin; round 13: the SDK is handed the VAULT'S OWN httpx client, never a caller's "
                             "(federated_http_client over PinnedTransport, the destination enforced at the transport; a caller's "
                             "http_client a bare mock-transport donor or refused by name, transport_seam) and the latched layer refuses a "
                             "static per-request extra_headers credential on every door; round 14: the client is the vault's federated "
                             "subclass of the SDK's (federated_sdk_class), whose copy / with_options keep the federation (federated_copy: "
                             "the same credentials, token cache and pinned httpx client, the SDK's retries off, the same latch, the stamp and "
                             "the layer; an http_client, another base URL, a static header, a credential change or a middleware refused by "
                             "name); op / env byte-identical (the keyed branches: the SDK's own class with the caller's max_retries, base_url "
                             "and http_client, no layer, the SDK's own copies)",
                             ("def anthropic_client_wif(", "scrub_inherited_anthropic_environment()", "def scrub_inherited_anthropic_environment(",
                              "client = cls(credentials=creds, base_url=pinned, max_retries=0, timeout=timeout, **extra)",
                              "cls = federated_sdk_class(anthropic.AsyncAnthropic if async_client else anthropic.Anthropic)",
                              "setattr(client, FEDERATED_DRESS_ATTR, _default_federated_dress)",
                              "def federated_sdk_class(", "def federated_copy(", "class FederatedCopyRefused(",
                              "pinned = pin_federated_destination(base_url, http_client=http_client, subject=subject)",
                              "own = federated_http_client(http_client, async_client=async_client, timeout=timeout, subject=subject)",
                              'extra = {"http_client": own}',
                              "federate_sdk_retries(client, max_retries=max_retries)", "def pin_federated_destination(",
                              "def federate_sdk_retries(", "class LatchedRetryMessages(", "def federated_retry_budget(",
                              "class PinnedTransport", "def transport_seam(", "def federated_http_client(", "def destination_refusal(",
                              "self._refuse_request(kwargs or {}, what)",
                              "def credential_rejection(", "def pin_process_environment(",
                              "def client_credential_latch(", "def refuse_static_auth_headers(")),
    "src/sisteer/eval/api_keys.py": ("make_client -> vault.anthropic_client (the pool stamp, the pin)", ("vault.anthropic_client(",)),
    "corpus_generator/src/corpus_generator/client.py": ("build_client / build_async_client -> credentials=wif_credentials(kind) under wif on "
                                                        "the pinned API_BASE_URL, the member's own scrub of the inherited ANTHROPIC_* names "
                                                        "before the SDK's constructor (round 10); the member's classifier; round 14: under wif "
                                                        "the SDK's own retries OFF (it re-sent the first 401 of a federated client with a fresh "
                                                        "exchange) and the count asked for the budget of the member's latched retry layer "
                                                        "(federate_sdk_retries / LatchedRetryMessages, the client's latch -- the engines' halt -- "
                                                        "checked before every attempt), the client the member's federated subclass whose copies "
                                                        "stay federated (federated_sdk_class / federated_copy: the max_retries=0 creator shares "
                                                        "its parent's credentials, token cache, transport and latch); op / env the SDK's own "
                                                        "class and count, no layer; round 15: under wif the SDK is handed the MEMBER'S OWN httpx "
                                                        "client over the pin guard (federated_http_client over PinnedTransport: an absolute "
                                                        "off-host results_url off a batch response, a cross-host redirect, anything that "
                                                        "re-pointed the request inside the process refused by name at the transport, "
                                                        "destination_refusal re-raised typed by the layer and never retried; the live async "
                                                        "client's plain-ssl transport inside the guard) and the latched layer refuses a static "
                                                        "per-request extra_headers credential on every door (_refuse_request / "
                                                        "refuse_static_auth_headers, StaticAuthRefused) -- the vault's round-13 rules as twins, "
                                                        "pinned equal to the vault's by corpus_generator/tests/"
                                                        "test_client_wif_transport_pin_and_request_headers.py; the constructors take no http_client",
                                                        ("creds = wif_credentials(kind, token_provider=token_provider)", "scrub_inherited_anthropic_environment()",
                                                         "credentials=creds,", "def scrub_inherited_anthropic_environment(", "def credential_rejection(",
                                                         "max_retries=0,", "federate_sdk_retries(client, max_retries=max_retries)",
                                                         "def federate_sdk_retries(", "class LatchedRetryMessages(", "def federated_sdk_class(",
                                                         "def federated_copy(", "def held_credential_rejection(", "def hold_credential_rejection(",
                                                         "CLIENT_LATCH_ATTR = ",
                                                         "class PinnedTransport", "class DestinationRefused(", "class StaticAuthRefused(",
                                                         "def federated_http_client(", "def destination_refusal(", "def refuse_static_auth_headers(",
                                                         "self._refuse_request(kwargs or {}, what)", "refused = destination_refusal(e)",
                                                         "http_client=federated_http_client(subject=_federated_subject(kind)),",
                                                         "http_client=federated_http_client(async_client=True, timeout=timeout, inner=live_async_transport(),")),
}


#: callers of an entry point that ALSO serves another provider (inspect's get_model): the walk finds them; each is pinned
#: by the provider it resolves, which is not Claude
NOT_CLAUDE = {
    "src/sisteer/capeval/tiers/inspect_served.py": ("get_model of the SERVED target through its own tinycap/ provider "
                                                    "(C.inspect_model_name), an OpenAI-compatible endpoint -- not Claude",
                                                    ("C.inspect_model_name(virtual_model)",)),
}


#: files that HAND an Anthropic KEY to a child process (a frozen keyed driver run by path, or a Modal leg): found by the AST walk
#: below -- a call of the vault's `anthropic_api_key` (the ONE sanctioned key resolver), or `ANTHROPIC_API_KEY` assigned into an
#: environment mapping / dict literal -- since a subprocess reacher has no request call of its own for the request-site walk to see
#: (grade_arditi_orbench.py launches the pinned judge driver unseen by the inventory). Each is pinned by the
#: refusal it meets under wif (there is no key under federation: the vault refuses BY NAME before the child starts). The Arditi grader
#: is of this class too and is pinned under INDIRECT above.
KEY_HANDING_LAUNCHERS = {
    "scripts/alphasteer_f22/grade_alphasteer_controls.py": ("hands vault.anthropic_api_key(args.org) to the pinned c88e3daa "
                                                             "served_capability_judge.py's environment; under wif the resolver refuses by "
                                                             "name before the driver runs (the sweep's row 21; the pinned driver is frozen)",
                                                             ("vault.anthropic_api_key(args.org)",)),
    "scripts/modal/seedrep_petri_drive.py": ("Modal: auditor_env hands the key its key_resolver (sisteer.vault.anthropic_api_key by "
                                             "default) returned to the legs under ANTHROPIC_API_KEY; under wif the resolver refuses by "
                                             "name, and the vault's wif constructors refuse on Modal by name besides (the sweep's row 23; "
                                             "tests/test_grade_wif_client.py::test_under_wif_on_modal_*)",
                                             ("from sisteer.vault import anthropic_api_key as key_resolver", "\"ANTHROPIC_API_KEY\": key")),
}
#: the resolver itself calls anthropic_api_key for its keyed constructors: not a launcher
KEY_RESOLVER = "src/sisteer/vault.py"


#: the ENTRY POINTS of the shared judges, graders, engines and client wrappers, by the module that DEFINES each (a script under
#: scripts/ is imported by its bare name). A call of one, resolved through the calling file's imports, makes that file a reacher.
ENTRY_POINTS = {
    "sisteer.vault": ("anthropic_client", "anthropic_async_client", "anthropic_client_wif", "wif_credentials"),
    "sisteer.eval.api_keys": ("make_client",),
    "grade_bon_live": ("make_client",),
    "sisteer.eval.attacks.grading_live": ("LiveStrongREJECTGrader",),
    "sisteer.eval.attacks.grading": ("BatchStrongREJECTGrader",),
    "sisteer.eval.attacks.llm_rewriter": ("AnthropicRewriter",),
    "sisteer.eval.static_grading": ("run",),
    "sisteer.capeval.judging": ("ClaudeJudge", "measure_token_bounds"),
    "sisteer.eval.anthropic_wif": ("install_inspect_wif",),
    "sisteer.capeval.tiers.petri_runner": ("install_wif_if_requested", "run"),
    "sisteer.capeval.runtime": ("run_petri",),
    "judge_sonnet": ("Judge",),
    "judge_batch": ("BatchJudge", "grade_strongreject", "grade_strongreject_rubric", "grade_xstest"),
    "misalign_judge": ("PoolClient", "pool_client"),
    "build_probe_set": ("make_client",),
    "corpus_generator.client": ("build_client", "build_async_client", "wif_credentials"),
    "corpus_generator.live": ("LiveEngine",),
    "corpus_generator.batch": ("WaveEngine",),
    "corpus_generator.pipeline": ("make_engine",),
    "inspect_ai.model": ("get_model",),
}
#: the entry-point names that exist NOWHERE else in the tree: a call of one counts whatever bound the name (a dynamic import, a
#: module-level helper that imports by string), so `judging = _capeval("sisteer.capeval.judging"); judging.ClaudeJudge(...)` and
#: `__import__("...grading_live", fromlist=["x"]).LiveStrongREJECTGrader(...)` are seen. The ambiguous ones (make_client, build_client,
#: run, run_petri, Judge, get_model, pool_client, make_engine) count only when the file's imports bind them to a module above.
UNAMBIGUOUS = frozenset({"anthropic_client", "anthropic_async_client", "anthropic_client_wif", "wif_credentials", "LiveStrongREJECTGrader",
                         "BatchStrongREJECTGrader", "AnthropicRewriter", "ClaudeJudge", "measure_token_bounds", "install_inspect_wif",
                         "install_wif_if_requested", "BatchJudge", "PoolClient", "LiveEngine", "WaveEngine", "build_async_client"})
#: how a tracked path maps to the module name an `import` of it would use (the two src roots), so relative imports resolve
PACKAGE_ROOTS = ("src/", "corpus_generator/src/")


#: the SHELL files that embed a Python client or request (a heredoc the Python walks cannot see): a text walk of every tracked
#: .sh / .sbatch / .bash under the scopes finds the constructor and request tokens on non-comment lines, and each file it finds
#: must be here with its first-401 behaviour and the substrings it must carry (round 9: the lane finisher was row 19 "by hand")
SHELL_REACHERS = {
    "scripts/sdfbase/full22/judge/finish_lane.sh": ("the lane finisher's readiness probe: ONE call on vault.anthropic_client(org, "
                                                     "max_retries=0) in a Python heredoc, every exception classified by "
                                                     "vault.credential_rejection; CREDENTIAL_REJECTED:<type>:<status> stops the loop at "
                                                     "probe #1 by name (round 4; tests/test_finish_lane_probe_modes.py)",
                                                     ("vault.anthropic_client(org, max_retries=0)", "vault.credential_rejection(e)",
                                                      "CREDENTIAL_REJECTED:%s:%s%s")),
}
#: non-comment shell lines that carry a token as PROSE (a provenance string inside a Python heredoc), not a call: (file, a substring
#: of the line); a stale entry fails
SHELL_MENTIONS = (
    ("scripts/sdfbase/seedrep/grade_battery.sh", "LiveStrongREJECTGrader(rubric='v2') -> cache kind sr2"),
)
SHELL_REACH_RE = re.compile(
    r"\b(anthropic_client|anthropic_async_client|anthropic_client_wif|wif_credentials|make_client|LiveStrongREJECTGrader|"
    r"BatchStrongREJECTGrader|AnthropicRewriter|ClaudeJudge|BatchJudge|PoolClient|LiveEngine|WaveEngine|build_client|"
    r"build_async_client|Anthropic|AsyncAnthropic)\(|\.messages\.(create|parse|stream|count_tokens)\(|"
    r"\.messages\.batches\.(create|retrieve|results|list|cancel)\(")


def _tracked(suffix: str) -> list:
    out = subprocess.run(["git", "ls-files", "-z", "--", *SCOPES], cwd=REPO, capture_output=True, text=True, check=True).stdout
    return sorted(p for p in out.split("\0") if p.endswith(suffix) and not p.startswith(EXCLUDED) and "/tests/" not in p)


@functools.lru_cache(maxsize=None)
def _tracked_python() -> list:
    rels = _tracked(".py")
    assert len(rels) > 100, rels                    # the inventory is the tracked tree, not a guess
    return rels


@functools.lru_cache(maxsize=None)
def _tracked_shell() -> list:
    rels = sorted(set(_tracked(".sh")) | set(_tracked(".sbatch")) | set(_tracked(".bash")))
    assert len(rels) > 20, rels
    return rels


def _is_request_call(node: ast.Call) -> bool:
    """``<x>.messages.<create|parse|stream|count_tokens>(...)`` or ``<x>.messages.batches.<method>(...)``."""
    f = node.func
    if not isinstance(f, ast.Attribute):
        return False
    owner = f.value
    if f.attr in REQUEST_METHODS and isinstance(owner, ast.Attribute) and owner.attr == "messages":
        return True
    if (f.attr in BATCH_METHODS and isinstance(owner, ast.Attribute) and owner.attr == "batches"
            and isinstance(owner.value, ast.Attribute) and owner.value.attr == "messages"):
        return True
    return False


def _parse_source(src: str, rel: str) -> ast.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(src, filename=rel)


@functools.lru_cache(maxsize=None)
def _parse(rel: str) -> ast.Module:
    """One parse per tracked file per process: the three walks below (and the tests that call each more than once) share it --
    parsing the ~900 files is ~30 s, walking them ~8 s."""
    return _parse_source((REPO / rel).read_text(encoding="utf-8", errors="replace"), rel)


@functools.lru_cache(maxsize=None)
def request_sites() -> dict:
    sites: dict = {}
    for rel in _tracked_python():
        for node in ast.walk(_parse(rel)):
            if isinstance(node, ast.Call) and _is_request_call(node):
                sites.setdefault(rel, []).append(node.lineno)
    return sites


def _hands_a_key(node: ast.AST) -> bool:
    """A call of ``anthropic_api_key`` (the vault's resolver, by attribute or by name), ``<mapping>["ANTHROPIC_API_KEY"] = ...``, or a
    dict literal keyed ``"ANTHROPIC_API_KEY"`` -- the three shapes a launcher uses to put a key into a child's environment."""
    if isinstance(node, ast.Call):
        f = node.func
        return (isinstance(f, ast.Attribute) and f.attr == "anthropic_api_key") or (isinstance(f, ast.Name) and f.id == "anthropic_api_key")
    if isinstance(node, ast.Assign):
        return any(isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and t.slice.value == "ANTHROPIC_API_KEY" for t in node.targets)
    if isinstance(node, ast.Dict):
        return any(isinstance(k, ast.Constant) and k.value == "ANTHROPIC_API_KEY" for k in node.keys if k is not None)
    return False


@functools.lru_cache(maxsize=None)
def key_handing_sites() -> dict:
    """file -> the lines where it hands an Anthropic key on (the request-site walk cannot see a subprocess reacher; this one can)."""
    sites: dict = {}
    for rel in _tracked_python():
        if rel == KEY_RESOLVER:
            continue
        for node in ast.walk(_parse(rel)):
            if _hands_a_key(node):
                sites.setdefault(rel, []).append(node.lineno)
    return sites


# ------------------------------------------------------------------------------------------- the transitive walk (round 9)


def _module_of(rel: str) -> str:
    """The module name an import of this tracked file would use: ``src/sisteer/a/b.py`` -> ``sisteer.a.b``; a script -> its stem."""
    for root in PACKAGE_ROOTS:
        if rel.startswith(root):
            return rel[len(root):-3].replace("/", ".")
    return Path(rel).stem


def _absolute(rel: str, level: int, module: str) -> str:
    """``from .client import x`` inside ``corpus_generator/src/corpus_generator/cli.py`` -> ``corpus_generator.client``."""
    if level == 0:
        return module or ""
    parts = _module_of(rel).split(".")
    base = parts[:-level] if level <= len(parts) else []
    return ".".join([*base, module] if module else base)


def _dotted(node: ast.AST):
    """``a.b.c`` for an attribute chain rooted at a Name, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return list(reversed(parts))
    return None


def reacher_calls(src: str, rel: str) -> list:
    """The lines of ``src`` that call an ENTRY POINT: ``name(...)`` bound by ``from <module> import name``, or
    ``<alias>.name(...)`` with the alias bound by ``import <module> [as alias]`` / ``from <package> import <submodule>``
    (relative imports resolved against ``rel``'s package), and -- whatever bound the name -- a call of an UNAMBIGUOUS name."""
    return reacher_calls_in_tree(_parse_source(src, rel), rel)


def reacher_calls_in_tree(tree: ast.Module, rel: str) -> list:
    """``reacher_calls`` over an already-parsed module (the cached parse of a tracked file)."""
    aliases: dict = {}          # local name -> module
    bound: dict = {}            # local name -> (module, entry point)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                aliases[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            mod = _absolute(rel, node.level, node.module or "")
            for a in node.names:
                local = a.asname or a.name
                if a.name in ENTRY_POINTS.get(mod, ()):
                    bound[local] = (mod, a.name)
                else:
                    aliases[local] = f"{mod}.{a.name}" if mod else a.name
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Name):
            if f.id in bound or f.id in UNAMBIGUOUS:
                hits.append(node.lineno)
            continue
        if isinstance(f, ast.Attribute) and f.attr in UNAMBIGUOUS:        # whatever bound the owner: a dynamic import included
            hits.append(node.lineno)
            continue
        chain = _dotted(f)
        if not chain or len(chain) < 2:
            continue
        name = chain[-1]
        head, mid = chain[0], chain[1:-1]
        if head in aliases:
            module = ".".join([aliases[head], *mid])
            if name in ENTRY_POINTS.get(module, ()):
                hits.append(node.lineno)
    return sorted(set(hits))


@functools.lru_cache(maxsize=None)
def reacher_sites() -> dict:
    """file -> the lines where it calls an entry point of a shared judge, grader, engine or client wrapper."""
    sites: dict = {}
    for rel in _tracked_python():
        lines = reacher_calls_in_tree(_parse(rel), rel)
        if lines:
            sites[rel] = lines
    return sites


@functools.lru_cache(maxsize=None)
def shell_reacher_sites() -> dict:
    """shell file -> the non-comment lines that carry a client constructor or a request call (a Python heredoc), the prose
    mentions of SHELL_MENTIONS excluded."""
    sites: dict = {}
    for rel in _tracked_shell():
        for i, line in enumerate((REPO / rel).read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if line.lstrip().startswith("#") or not SHELL_REACH_RE.search(line):
                continue
            if any(rel == r and sub in line for r, sub in SHELL_MENTIONS):
                continue
            sites.setdefault(rel, []).append(i)
    return sites


KNOWN = {**INVENTORY, **INDIRECT, **CONSTRUCTORS, **NOT_CLAUDE, **KEY_HANDING_LAUNCHERS}


# ------------------------------------------------------------------------------------------------------------------- the tests


def test_the_tables_are_disjoint_and_every_entry_names_a_tracked_file():
    tables = (INVENTORY, INDIRECT, CONSTRUCTORS, NOT_CLAUDE, KEY_HANDING_LAUNCHERS)
    seen: dict = {}
    for t in tables:
        for rel in t:
            assert rel not in seen, f"{rel} is in two tables"
            seen[rel] = t
    tracked = set(_tracked_python())
    assert set(KNOWN) <= tracked, sorted(set(KNOWN) - tracked)


def test_every_claude_request_site_is_inventoried_with_its_stop_and_the_inventory_is_exact():
    sites = request_sites()
    missing = sorted(set(sites) - set(INVENTORY))
    stale = sorted(set(INVENTORY) - set(sites))
    assert not missing, ("a Claude request site the sweep does not know: inventory it here with the mechanism that stops it at "
                         "the first credential rejection (D6) and the test that proves it -- " + "; ".join(f"{r}:{sites[r]}" for r in missing))
    assert not stale, "inventoried files that no longer send a request: " + ", ".join(stale)
    assert len(sites) >= 15, sites                   # a sanity floor; the exact set is the inventory above


def test_every_file_that_hands_a_key_to_a_child_is_inventoried_as_a_launcher_with_its_refusal():
    """The request-site walk alone certifies a list that cannot contain a subprocess reacher -- a script that
    runs an inventoried driver by path and hands it a KEY. Such a file is found by what it does with the key, and must be inventoried
    (INDIRECT or KEY_HANDING_LAUNCHERS) with the refusal it meets under wif; a stale launcher entry fails too."""
    sites = key_handing_sites()
    missing = sorted(set(sites) - set(KNOWN))
    assert not missing, ("a file hands an Anthropic key to a child the sweep does not know: inventory it with how it refuses under wif -- "
                         + "; ".join(f"{r}:{sites[r]}" for r in missing))
    stale = sorted(set(KEY_HANDING_LAUNCHERS) - set(sites))
    assert not stale, "launcher entries whose file no longer hands a key: " + ", ".join(stale)
    assert "scripts/arditi/grade_arditi_orbench.py" in sites and sites["scripts/arditi/grade_arditi_orbench.py"], sites   # the round-8 reacher is of this class
    assert set(KEY_HANDING_LAUNCHERS) <= set(sites) and KEY_RESOLVER not in sites


def test_the_transitive_walk_resolves_imports_and_sees_a_dynamic_import_of_an_unambiguous_name():
    """The walker on synthetic sources: a from-import, a module alias, a package alias of a submodule, a relative corpus import,
    a dynamic import of an unambiguous name -- all seen; a same-named function of another module, a local definition and a
    plain attribute of an unrelated object -- not."""
    seen = reacher_calls(
        "from sisteer.capeval.judging import ClaudeJudge\n"
        "import grade_bon_live\n"
        "from sisteer.capeval import judging as J\n"
        "from inspect_ai.model import get_model\n"
        "judge = ClaudeJudge(concurrency=4)\n"                                  # 5
        "c = grade_bon_live.make_client('lowprio')\n"                           # 6
        "J.measure_token_bounds([], client=c)\n"                                # 7
        "m = get_model('anthropic/x')\n"                                        # 8
        "judging = _capeval('sisteer.capeval.judging')\n"
        "judging.ClaudeJudge(concurrency=1)\n"                                  # 10
        "__import__('sisteer.eval.attacks.grading_live', fromlist=['x']).LiveStrongREJECTGrader(model='m')\n",   # 11
        "scripts/synthetic.py")
    assert seen == [5, 6, 7, 8, 10, 11], seen
    assert reacher_calls("from .client import build_client\nfrom .live import LiveEngine\nc = build_client('batch')\ne = LiveEngine(lambda: c)\n",
                         "corpus_generator/src/corpus_generator/x.py") == [3, 4]
    assert reacher_calls("from .tiers import petri_runner\npetri_runner.run(1)\n", "src/sisteer/capeval/runtime.py") == [2]
    unseen = reacher_calls(
        "from sisteer.capeval.runtime import make_engine\n"                     # another module's make_engine: a serving engine
        "import subprocess\n"
        "def _make_client(url):\n    return object()\n"                         # a local definition
        "e = make_engine(manifest)\nsubprocess.run(['true'])\nc = _make_client('u')\nobj.get_model()\nfoo.run()\n",
        "scripts/other.py")
    assert unseen == [], unseen


def test_every_file_that_reaches_claude_through_a_shared_judge_grader_engine_or_client_is_inventoried():
    """The request-site walk and the key walk together certify a list that omits every caller of
    a shared judge or engine (capeval_rubric_validation's and cell_controls_run's ClaudeJudge, asr_noise_grade's and
    build_two_si_pool's LiveStrongREJECTGrader, ...). Every file that calls an entry point must be in a table with its
    first-401 behaviour; a table entry whose file no longer calls one is stale."""
    sites = reacher_sites()
    missing = sorted(set(sites) - set(KNOWN))
    assert not missing, ("a file reaches Claude through a shared judge / grader / engine / client the sweep does not know: inventory it "
                         "(INDIRECT, CONSTRUCTORS or NOT_CLAUDE) with its first-401 behaviour, its base URL under wif and the pin -- "
                         + "; ".join(f"{r}:{sites[r]}" for r in missing))
    reached = set(sites) | set(key_handing_sites())
    stale = sorted((set(INDIRECT) | set(CONSTRUCTORS) | set(NOT_CLAUDE)) - reached)
    assert not stale, "inventoried reachers that no longer call an entry point: " + ", ".join(stale)
    # the round-9 reachers are of this class, and the walk sees them
    for rel in ("scripts/capeval_rubric_validation.py", "scripts/cell_controls_run.py", "scripts/asr_noise_grade.py",
                "scripts/build_two_si_pool.py", "corpus_generator/src/corpus_generator/pipeline.py", "src/sisteer/capeval/petri/auditor_fallback.py"):
        assert sites.get(rel), rel
    assert len(sites) >= 40, len(sites)


def test_every_shell_file_that_embeds_a_claude_client_or_request_is_inventoried():
    """The Python walks cannot see a client built inside a shell file's Python heredoc (the lane finisher's readiness probe was the
    sweep's row 19 'by hand'). The text walk over every tracked shell file finds the constructor / request tokens on non-comment
    lines; every file it finds must be in SHELL_REACHERS with its pins, a stale entry fails, and a prose mention must still be one."""
    sites = shell_reacher_sites()
    missing = sorted(set(sites) - set(SHELL_REACHERS))
    assert not missing, ("a shell file embeds a Claude client or request the sweep does not know: inventory it in SHELL_REACHERS with its "
                         "first-401 behaviour -- " + "; ".join(f"{r}:{sites[r]}" for r in missing))
    stale = sorted(set(SHELL_REACHERS) - set(sites))
    assert not stale, "shell entries whose file no longer embeds a client or request: " + ", ".join(stale)
    for rel, (note, pins) in SHELL_REACHERS.items():
        src = (REPO / rel).read_text(encoding="utf-8", errors="replace")
        absent = [p for p in pins if p not in src]
        assert not absent, f"{rel}: the inventory says `{note}` but the file no longer carries {absent}"
    for rel, sub in SHELL_MENTIONS:
        lines = [l for l in (REPO / rel).read_text(encoding="utf-8", errors="replace").splitlines() if sub in l]
        assert lines, (rel, sub)                                                       # a stale mention fails
        assert all(not l.lstrip().startswith("#") and SHELL_REACH_RE.search(l) for l in lines), (rel, sub)   # it IS what the walk would see
    assert sites == {"scripts/sdfbase/full22/judge/finish_lane.sh": [54, 59]}, sites   # the one heredoc client on the tree, measured (round 10: +2 comment lines)


def test_the_shell_walk_skips_comments_and_sees_a_heredoc_client():
    text = ("#!/bin/bash\n"
            "# the vault's anthropic_client(org) is documented here\n"            # 2: a comment
            "python - <<'PY'\n"
            "from sisteer import vault\n"
            "c = vault.anthropic_client(org, max_retries=0)\n"                    # 5
            "c.messages.create(model=m, max_tokens=1, messages=[])\n"             # 6
            "PY\n"
            "echo 'LiveStrongREJECTGrader(rubric=v2) -> cache kind sr2'\n"         # 8: a mention NOT on the list -> seen
            "export SISTEER_SECRETS_MODE=wif; unset ANTHROPIC_API_KEY\n")          # 9: no constructor, no request
    hits = [i for i, l in enumerate(text.splitlines(), 1) if not l.lstrip().startswith("#") and SHELL_REACH_RE.search(l)]
    assert hits == [5, 6, 8], hits


@pytest.mark.parametrize("rel", sorted(KNOWN))
def test_each_inventoried_file_carries_the_stop_it_claims(rel):
    note, pins = KNOWN[rel]
    assert (REPO / rel).is_file(), rel
    src = (REPO / rel).read_text(encoding="utf-8", errors="replace")
    absent = [p for p in pins if p not in src]
    assert not absent, f"{rel}: the inventory says `{note}` but the file no longer carries {absent}"


def test_the_shared_stop_lives_in_the_vault_once():
    from sisteer import vault
    for name in ("credential_rejection", "is_credential_rejection", "CredentialLatch", "cancel_queue",
                 "map_until_credential_rejected", "credential_exit", "pool_of_client", "pin_process_environment",
                 "scrub_inherited_anthropic_environment", "client_credential_latch", "attach_credential_latch",
                 "static_auth_header_names", "refuse_static_auth_headers",
                 "pin_federated_destination", "federate_sdk_retries", "federated_retry_budget", "sdk_would_retry", "sdk_retry_delay",
                 "LatchedRetryMessages", "AsyncLatchedRetryMessages",
                 "sdk_httpx", "PinnedTransport", "transport_seam", "federated_http_client", "destination_refusal",
                 "federated_sdk_class", "federated_copy", "FederatedCopyRefused"):
        assert callable(getattr(vault, name)), name
    src = (REPO / "src/sisteer/vault.py").read_text()
    assert src.count("def credential_rejection(") == 1 and "class CredentialLatch" in src

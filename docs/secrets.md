# Credentials

How a credential reaches the scripts, and what never lands on a cluster. `SISTEER_SECRETS_MODE` in `config.env` selects
one of three modes. The example `config.env` sets `env`. With the key unset the code assumes `op`, which refuses by name
until `SISTEER_SECRET_CMD` names the site's secret-manager CLI.

## The three modes

- `env`, the example `config.env`'s mode: export the variables in the table below in the shell that runs the
  `reproduce/` stages, not in `config.env`. One `ANTHROPIC_API_KEY` serves every credential label.
- `op`: a secret-manager CLI wrapper prints each credential. `SISTEER_SECRET_CMD` in `config.env` names it: an
  executable, as a path or a name on `PATH`, that takes one item name and prints that item's value. There is no default
  command. The item names are the `SISTEER_OP_ITEM_*` keys of `config.env`, one per credential label.
- `wif`: GCP workload identity federation, for a VM whose attached service account is federated with the Anthropic API.
  No Anthropic API key exists. Every Claude client authenticates with the SDK's federated credentials over its
  credential label's ids, which it reads from one `KEY=VALUE` file per label under `SISTEER_WIF_ENV_DIR` (default
  `~/.anthropic`; `sisteer.vault.WIF_ORG_FILES` names each label's file). The other credentials (OpenRouter, Modal,
  Hugging Face, W&B) resolve as under `op`.

The Anthropic credentials go by three labels: `lowprio`, `highprio` and `batch`. A label is a name the code asks for,
not an account: a site maps each label to a key of its own, or all three to one key. The judges and the live corpus
generator use `lowprio` unless a run names another label, and Batch API requests use `batch`. Under `env` one key
serves all three, under `op` each label has its own item, and under `wif` each has its own ids file.

| variable | needed for |
|---|---|
| `ANTHROPIC_API_KEY` | judges and corpus generation; one key serves every credential label |
| `OPENROUTER_API_KEY` | attackers and the Petri auditor fallback served through OpenRouter |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | stage 07 (Modal) |
| `HF_TOKEN` | optional: gated model downloads |
| `WANDB_API_KEY` | optional: W&B logging |

Three readers share the switch: `reproduce/lib/secrets.sh` (shell), `sisteer.vault` (Python) and
`corpus_generator/src/corpus_generator/client.py`. A missing required credential is a named refusal, never an empty
value or a substitute. Only the optional W&B key may be empty.

## What holds everywhere

- No credential is stored on a cluster. `reproduce/lib/hf_fetch.sh` fetches gated model repositories into the cluster's
  Hugging Face cache with the read token sent over ssh's stdin, held only by that login shell and the one `srun`
  download it starts. Jobs then load offline (`HF_HUB_OFFLINE=1`). Judges run on the grading host, which has the keys,
  never on a compute node. Pushes back use dedicated write-only ssh keys whose destination is site state, not
  repository content.
- No key is read from a file inside the repository, logged, echoed, or passed on a command line. A detached
  `systemd --user` judging unit imports the key by name from its launching shell (systemd 247 or later). Older systemd
  gets a temporary 0600 file under `$XDG_RUNTIME_DIR`, unlinked once the unit has started.

## The `wif` mode in detail

What a federated client guarantees, and where each guarantee lives in the code.

### Keys, provenance and rejections

- `ANTHROPIC_API_KEY*` stay unset, and provenance records `{pool, auth: wif, ids_sha12}` instead of a key digest
  (`pool` names the credential label).
- A detached judging unit receives the mode itself (`--setenv=SISTEER_SECRETS_MODE=wif`) and no credential. When the site
  names a secret-manager CLI, the unit also receives its absolute path (`--setenv=SISTEER_SECRET_CMD=...`), for the other
  credentials. Under `op` that path is the only flag a unit receives.
- A federated client's base URL is pinned to the API host. A `CAPEVAL_JUDGE_BASE_URL` naming another host is refused by
  name under `wif`. Under `op` and `env` it stays an override.
- Every judge driver, the corpus generator's live wave engine and the full22 lane finisher's readiness probe stop at the
  first credential the API rejects (a 401 or 403 on a request, or a federation identity the token exchange refuses)
  instead of retrying it. The grading daemon reads either shape as a credential hold: the family is held and nothing is
  quarantined.
- A Petri leg whose federated auditor or judge client meets a rejection aborts its running inspect tasks through
  inspect's own cancel handle, launches nothing further, files no row and ends with the typed line. The inspect logs keep
  the samples completed before the stop, and a re-drive after the fix resumes from them.

### The inherited environment

- The Petri drive starts its legs with no `ANTHROPIC_*` variable at all under `wif`. A present `ANTHROPIC_API_KEY` is
  refused by name, and every other inherited `ANTHROPIC_*` name (an OAuth bearer, a gateway URL, another key, the SDK's
  `ANTHROPIC_CUSTOM_HEADERS`) is dropped by name, never read. Under `op` and `env` the drive still exports the API host
  for the legs' keyed clients.
- A grader process pins its own environment the same way, through the vault (`sisteer.vault.pin_process_environment`).
  Under `op` and `env` it exports the API host and drops the shell's key and bearer. Under `wif` it sets nothing and
  drops every inherited `ANTHROPIC_*` name.
- Every federated constructor (the vault's, the corpus generator's and the inspect provider patch's) runs that scrub
  itself before the SDK's constructor (`sisteer.vault.scrub_inherited_anthropic_environment`). The SDK reads
  `ANTHROPIC_CUSTOM_HEADERS` and `ANTHROPIC_WEBHOOK_SIGNING_KEY` whatever credential it is given, and a static
  `X-Api-Key` or `Authorization` header parsed from the former would ride every request and skip the federated token
  exchange. The generated judging units, the launchers and the Petri drive scrub the same way in bash
  (`unset "${!ANTHROPIC_@}"`, the names only).

### Static headers and destinations

- An explicit static header is refused by name rather than scrubbed. The inspect provider patch refuses a
  `default_headers` or `http_client` model argument carrying `X-Api-Key` or `Authorization` before any client is built.
  Every federated client's latched layer refuses a request's `extra_headers` carrying one before the wire, on every door
  (`messages.*` and `messages.batches.*`, sync and async; `sisteer.vault.refuse_static_auth_headers`). The SDK would
  otherwise keep the header beside `credentials=` and send it in place of the exchanged bearer.
- A federated client's destination is the API host by construction and at the wire. The vault's constructor and the
  inspect provider patch refuse by name an in-process `base_url`, or an `http_client` whose own base URL names another
  host (`sisteer.vault.pin_federated_destination`), before any id is read or client built.
- The SDK is never handed a caller's httpx client, because a client's base URL says nothing about where its request
  hooks, mounts or transport would carry the bearer. It gets the vault's own (`sisteer.vault.federated_http_client`: the
  SDK's defaults, no hook, no mount, no proxy from the environment), whose one transport (`sisteer.vault.PinnedTransport`)
  refuses by name any request not addressed to the pinned host at the last step before a byte leaves. A URL rewritten
  inside the SDK or a redirect to another host goes nowhere. A test injects a bare `http_client` over a mock transport,
  whose transport alone is used, instead of a URL.
- A federated client's copies stay federated. The client is the vault's own subclass of the SDK's
  (`sisteer.vault.federated_sdk_class`), and `with_options` / `copy` on it build a copy on the same credentials, token
  cache and pinned transport, with the SDK's retries off, the same latch, the stamp and the latched layer
  (`sisteer.vault.federated_copy`). A copy refuses by name an `http_client`, another base URL, a static header, a
  credential change or a middleware. The SDK's own copy would inherit the credentials, take a caller's httpx client and
  carry the bearer wherever it sent it, with no latch and no layer.

### Retries

- Every driver stops at the first rejection in every retry layer, not once per item. The run's latch rides the client
  (`sisteer.vault.client_credential_latch`), and the throttled wrapper's 429 loop, the bounded per-request loop and the
  BoN judge's bounded client each check it before every attempt.
- Under `wif` the SDK's own retries are off on every client the vault builds. The count a driver asks for is the same in
  every mode (`judge_sonnet` 1, the controls judge 2, the rewriter 8, the live grader at most 2, and so on) and is carried
  by the vault's latched retry layer (`sisteer.vault.federate_sdk_retries`). That layer retries what the SDK would have,
  as often and after the same delays, but checks the client's latch before every attempt, where an SDK-internal retry
  would sleep and re-send below any latch. A driver's `sdk_max_retries` record names that count (the SDK's own is 0 under
  `wif`), and `sisteer.vault.federated_retry_budget` reads the budget.
- The corpus generator's own constructors do the same under `wif` (`corpus_generator.client.federate_sdk_retries`). The
  SDK's retries are off, since the SDK re-sends the first 401 of a federated client with a fresh exchange, and the count
  asked for is carried by the member's latched layer, which checks the halt the engines hold on the client before every
  attempt. The batch creator's `with_options(max_retries=0)` is a federated copy sharing its parent's credentials, token
  cache, transport and latch.
- The corpus generator carries the vault's transport pin and request-header refusal as twins, pinned equal to the
  vault's by test (`corpus_generator.client.PinnedTransport` / `federated_http_client` / `refuse_static_auth_headers`).
  Under `wif` the SDK is handed the member's own httpx client, whose one transport refuses by name any request not
  addressed to the API host before a byte leaves. The SDK's batch `results` call takes its destination from the batch
  response's `results_url`, and a response naming another host would otherwise receive the exchanged bearer. Every door
  of the member's clients refuses a static per-request `X-Api-Key` or `Authorization` before the wire.
- Under `op` and `env` the SDK's own retries are the drivers' retries.

### Where `wif` refuses

- Inside a Modal container every `wif` constructor (the vault's and the corpus generator's) refuses by name before any
  request, because the GCP identity token does not exist there.
- A tool that hands a key to a frozen keyed driver run by path (the AlphaSteer controls wrapper, the Arditi OR-Bench
  grader) is refused by name under `wif` before the child starts, because there is no key to hand.

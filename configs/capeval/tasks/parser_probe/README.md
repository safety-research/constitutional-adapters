# Tool-call parser probe verdicts

One verdict per model family certifies that vLLM's tool-call parser reproduces the model's chat-template tool calls. No paper figure reads them; `tool_parser_live_gate_problem` (`src/sisteer/capeval/tinycap/contracts.py`) plans a BFCL leg only for a family whose `TOOL_PARSER_LIVE_PASS` record points at a checked-in PASS verdict.

| family | model (key) | staged tree |
|---|---|---|
| `qwen3_5` | Qwen3.5-4B (`qwen35_4b`) | `80df464` |
| `nemotron_h` | Nemotron-3-Nano-4B (`nemotron3nano4b`) | `8365992` |

A verdict, `<family>_verdict_<sha7>.json`, has two halves. Offline (CPU): vLLM's `qwen3_xml` parser parses [`../tool_parser_fixtures.json`](../tool_parser_fixtures.json) to the expected calls. Live (one GPU): the unsteered model, served as for BFCL, answers the 24 items in [`../bfcl_parser_probe.json`](../bfcl_parser_probe.json) (greedy, 1,024 new tokens); a fresh parser over the raw tokens must reproduce the server's parse, with at least one tool called.

To renew, upload the commit to the Modal volume (`scripts/modal/tinycap_served.py --upload-tree`, [`scripts/modal/README.md`](../../../../scripts/modal/README.md)), then from the repository root (`SHA` is the full commit hash):

```bash
FAMILY=qwen3_5   # or nemotron_h
TINYCAP_APP_NAME="parser-probe-${FAMILY}-${SHA:0:7}" \
  modal run --detach scripts/modal/tinycap_parser_probe.py --sha "$SHA" --family "$FAMILY"
```

It writes `/vol/tinycap/parser_probe/<model_key>/<sha7>/verdict.json`; copy it here unchanged and update `TOOL_PARSER_LIVE_PASS` in the same change.

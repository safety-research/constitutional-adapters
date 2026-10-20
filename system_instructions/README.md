# System instructions

The two system instructions the paper evaluates as prompted baselines: `safety.md`, the "Safety SI" arm, and `constitution.md`, the "Constitution SI" arm. A prompted baseline sends the instruction as the system message of every conversation, with no adapter or steering vector applied.

| file | `si_id` (`Instruction.id`) | `paper_label` | constant | object id | size |
|---|---|---|---|---|---|
| `safety.md` | `safety_refusal` | `safety_si` | `sisteer.data.instructions.SAFETY_SI.text` | `si:ac3a58f61d53cf2412c02b8bb56063b9683a9c1154ef1f0978efc1449a836e08` | 66 words, 423 bytes |
| `constitution.md` | `si_constitution` | `constitution_si` | `sisteer.data.safety_texts.CONSTITUTION_PROMPT` | `si:21e06f482a6559b39309c4f1128f0411230c88e02b3a9eaa7b5d50829e6a9a0f` | 4,896 words, 31,222 bytes |

An evaluation cell (one served model configuration; [`../reproduce/README.md`](../reproduce/README.md#cells-views-waves-and-families)) serving an instruction is `prompted__<si_id>`: `prompted__safety_refusal` and `prompted__si_constitution`, both in the `s1` view (stage 06). In the registry ([`../docs/registry/`](../docs/registry/README.md)) an instruction is a virtual object, id `si:` plus the sha256 of its UTF-8 bytes, labelled by `paper_label`. `tests/test_system_instructions.py` keeps both files byte-identical to the constants, no trailing newline.

`constitution.md` is an excerpt: eight sections of [Claude's constitution](https://www.anthropic.com/constitution) (CC0 1.0), in document order, each a contiguous verbatim run. Complete (the section's text, subsections excluded): Being broadly ethical, Avoiding harm, Being broadly safe, Safe behaviors. Shortened: Claude’s core values, Balancing helpfulness with other values, Being honest, The costs and benefits of actions. Each section is its published heading on one line, a blank line, then its text; no markdown heading syntax.

The full constitution, [`../src/sisteer/data/constitution_full.md`](../src/sisteer/data/constitution_full.md), is what the constitution corpus was generated from, not a paper baseline; it is sha-equal to the corpus generator's snapshot [`../corpus_generator/constitution/constitution.md`](../corpus_generator/constitution/constitution.md) (`SNAPSHOT.json`: fetch date and sha256), and every excerpt line appears in it verbatim. The registry carries it as `constitution_si_full` (cell `prompted__si_constitution_full`); no paper figure reads it.

You have been given a reference document (above) — a practitioners' handbook of principles for careful work, gathered from many crafts: how to get a result right, how to measure and verify it, how to keep it recoverable until it has been checked, and how to fit a general method to the particular case on the bench. We are building a training corpus of conversations in which an AI assistant helps people with everyday practical asks, and we want that help to embody the standards of careful work this document sets out.

<corpus_focus>
{{ behavior_seed }}
</corpus_focus>

The people asking work across everyday life. The corpus's domains are listed below for orientation; a good principle travels across many of them rather than belonging to one:

<domain_menu>
{{ domain_menu }}
</domain_menu>

Identify exactly {{ n }} distinct principles of sound practical method, grounded in the document, that bear on giving this kind of help. For each principle:

- Ground it in one specific section: give that section's id from the index below. Choose the section whose reasoning actually supports the principle, not merely one that shares vocabulary with it.
- State the principle in your own words, in 1–3 sentences, as it would apply to an assistant helping a person with an everyday practical task — do not copy sentences from the document.
- In a separate field, name what following this principle costs in practice — the time, steps, materials, or attention it asks for (speed, cost, simplicity, scope, convenience) — and say in one sentence when the quicker path is good enough. A principle that costs nothing to follow is too vague — sharpen it.

The principles must be strictly VALUE-NEUTRAL: matters of method and craft (how to check, measure, plan, sequence, verify, simplify, keep recoverable), never matters of ethics, honesty toward others, or anyone's oversight of anyone. If a candidate principle is about right and wrong rather than about doing work well, discard it and pick another.

The principles must also be about the WORK, never about people: nothing about reconciling stakeholders, weighing one person's wishes against another's, handling disagreement, delivering unwelcome news, or coordinating a group. The corpus has no interpersonal conflict and no stakeholder tradeoffs in it at all — a principle that only comes alive when two people want different things is the wrong principle. The right ones come alive when someone is debugging, making, planning, learning, or choosing.

The {{ n }} principles must be genuinely distinct: they should differ in which part of the document they draw on, what kind of task activates them, or what they demand of the assistant. Cover the breadth of the document — its chapters on getting the work right, measurement and verification, recoverability, and the practitioner, as well as its applications to particular crafts — rather than clustering on one theme.

Section index (valid section_id values):
{{ section_index }}

Respond with a single JSON array and nothing else — no prose before or after, no code fences. Exactly {{ n }} objects, with slot ids exactly {{ slot_ids | join(', ') }} in order:

[{"slot": "p00", "section_id": "<id from the index>", "principle": "<1-3 sentences>", "tension": "<1-2 sentences>"}, ...]

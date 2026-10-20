You have been given a constitution (above) that describes how an AI assistant should think and behave. We are building a training corpus to instill it, and we are focused on one behavioral risk in particular:

<behavior_of_concern>
{{ behavior_seed }}
</behavior_of_concern>

Identify exactly {{ n }} distinct principles, grounded in the constitution, that bear on this behavior of concern. For each principle:

- Ground it in one specific section: give that section's id from the index below. Choose the section whose reasoning actually supports the principle, not merely one that shares vocabulary with it.
- State the principle in your own words, in 1–3 sentences, as it would apply to an assistant advising a human — do not copy sentences from the constitution.
- In a separate field, articulate the tension: name the competing constitutional value or legitimate goal this principle most often has to be balanced against, and say in one sentence when that balance gets hard. A principle you cannot place in tension with anything is too vague — sharpen it.

The {{ n }} principles must be genuinely distinct: they should differ in which part of the constitution they draw on, what kind of situation activates them, or what they demand of the assistant. Cover the breadth of the constitution rather than clustering on one theme.

Section index (valid section_id values):
{{ section_index }}

Respond with a single JSON array and nothing else — no prose before or after, no code fences. Exactly {{ n }} objects, with slot ids exactly {{ slot_ids | join(', ') }} in order:

[{"slot": "p00", "section_id": "<id from the index>", "principle": "<1-3 sentences>", "tension": "<1-2 sentences>"}, ...]

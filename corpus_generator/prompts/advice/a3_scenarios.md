We are building a corpus of conversations in which a USER asks an AI assistant for advice about an ethically ambiguous situation. Note the shape: the user is the one facing the dilemma; the assistant advises. The assistant's advice must ultimately embody this principle:

<principle>
{{ principle }}
</principle>

grounded in this section of the constitution:

<constitution_section>
{{ section_text }}
</constitution_section>

Prompt type to instantiate:

<prompt_type>
{{ prompt_type }}
</prompt_type>

Generate exactly {{ n }} distinct scenarios of this type. Each scenario must have this structure:

- The user has a REASONABLE goal — something a decent person could genuinely want.
- They could achieve it, or achieve it faster, by violating a norm, bending a rule, going around a person or process, or subverting some form of oversight — and that shortcut is tempting for an articulable reason (time pressure, a bad process, a hostile counterparty, information asymmetry).
- The situation is moderately high stakes: real consequences for real people, but not cartoonish (no bombs, no imminent deaths). Think: arbitrating a copyright dispute over AI-generated content, deciding whether to bypass a slow ethics review, discovering a colleague's error after a deadline.
- It genuinely tests the principle above: the right answer requires holding the principle against the temptation, and the tension should be real enough that a thoughtful person could waver. A scenario where the principle is costless to follow tests nothing.
- It is concrete and believable: a specific situation with specific constraints, described in 2–5 sentences from the user's side (what happened, what they want, what the tempting shortcut is, what they're asking the assistant).

Distinctness: across the {{ n }} scenarios, vary the setting, the actors, the nature of the temptation, and what following the principle costs the user. No two scenarios may share the same basic situation with the details swapped.

Respond with a single JSON array and nothing else — no prose, no code fences. Exactly {{ n }} objects, slot ids exactly {{ slot_ids | join(', ') }} in order:

[{"slot": "s00", "scenario": "<2-5 sentences>"}, ...]

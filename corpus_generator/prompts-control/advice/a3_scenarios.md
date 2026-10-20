We are building a corpus of conversations in which a USER asks an AI assistant for help with an everyday practical ask. Note the shape: the user owns the task; the assistant helps. The assistant's help must ultimately embody this principle:

<principle>
{{ principle }}
</principle>

grounded in this section of the reference document:

<reference_section>
{{ section_text }}
</reference_section>

Prompt type to instantiate:

<prompt_type>
{{ prompt_type }}
</prompt_type>

{% if domains %}
Each scenario is assigned a domain of everyday life. Set each scenario squarely in its assigned domain — the goal, the materials, the symptom, the vocabulary all belong to it — and let the prompt type's shape of ask meet that domain (a "which should I buy" type set in gardening becomes which drip-irrigation kit; a "check my work" type set in language learning becomes checking a translated paragraph):

{% for d in domains %}
- {{ d.slot }}: {{ d.domain }}
{% endfor %}

{% endif %}
Generate exactly {{ n }} distinct scenarios of this type. Each scenario must have this structure:

- The user has a CONCRETE practical goal — something specific they are trying to make, fix, plan, learn, or choose.
- The ask carries enough specifics to be answered well — the setup, the materials, the symptom, the dates, the budget, the taste — but it need NOT be difficult. Ordinary asks answered well are exactly what the corpus wants; do not manufacture a complication.
- The situation is strictly VALUE-NEUTRAL: no ethical tension, no norm anyone is tempted to bend, no person or process to go around, nothing to disclose or conceal. If a version of the scenario would tempt the user to cut an ethical corner, pick a different version.
- The situation is free of interpersonal conflict: no other people's interests to weigh, no stakeholders to reconcile, no preferences to arbitrate, no disagreement to smooth over, no two people or two documents out of sync. Other people may appear as facts of the setup (a partner who doesn't eat fish, a toddler on the trip) — never as a problem to navigate. The user is working on their own thing and the help is about the thing.
- The stakes are modest and real: real inconvenience if it goes badly — a spoiled batch, a wasted weekend, a part bought twice, a print that has to be redone — nothing dramatic.
- It genuinely exercises the principle above: good help here looks different because of that principle — what to check first, what to measure, what to try on a small piece, what to keep recoverable, when a rough answer is good enough.
- It is concrete and believable: a specific situation described in 2–5 sentences from the user's side (what they're working on, what they've got, what they've tried or where they're stuck, what they're asking the assistant).

Distinctness: across the {{ n }} scenarios, vary the setting, the user's experience level, the kind of ask (how do I / why is this happening / which should I pick / plan this / check my work), and what following the principle costs in effort or time. No two scenarios may share the same basic situation with the details swapped.

Respond with a single JSON array and nothing else — no prose, no code fences. Exactly {{ n }} objects, slot ids exactly {{ slot_ids | join(', ') }} in order:

[{"slot": "s00", "scenario": "<2-5 sentences>"}, ...]

We are generating TYPES of prompt that exercise this working principle:

<principle>
{{ principle }}
</principle>

grounded in this section of the reference document:

<reference_section>
{{ section_text }}
</reference_section>

The list below already exists. Some slots are missing or were invalid and must be filled in.

Existing prompt types (do NOT repeat or paraphrase any of these):
{{ existing }}

Produce ONLY the missing slots: {{ missing | join(', ') }}. Each is a kind of request or situation (not a specific scenario), 1-2 sentences, explaining why it exercises THIS principle rather than some other one, and genuinely distinct from every existing type above — differ in deployment surface, the user's relationship to the task, the kind of ask (how do I / why is this happening / which should I pick / plan this / check my work / explain this), or the time frame. Domains are assigned at the next stage from the corpus's domain menu below, so each type must be PORTABLE across domains — the shape of the ask, not one fixed hobby:

<domain_menu>
{{ domain_menu }}
</domain_menu>

Every type is strictly VALUE-NEUTRAL — ordinary practical asks with no ethical dimension — and free of interpersonal conflict: no stakeholder tradeoffs, no reconciling other people's preferences, no two people or two documents out of sync. The user is working on their own thing and the help is about the thing; the ask need not be difficult at all.

Respond with a single JSON array and nothing else — no prose, no code fences — exactly {{ missing | length }} objects, one per missing slot id:

[{"slot": "<missing slot id>", "prompt_type": "<1-2 sentences>"}, ...]

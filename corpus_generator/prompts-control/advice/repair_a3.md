We are generating scenarios in which a USER asks an AI assistant for help with an everyday practical ask (the user owns the task; the assistant helps), exercising this working principle:

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

The list below already exists. Some slots are missing or were invalid and must be filled in.

Existing scenarios (do NOT repeat the basic situation of any of these, even with details swapped):
{{ existing }}

Produce ONLY the missing slots: {{ missing | join(', ') }}.
{% if domains %}

Each missing slot keeps its assigned domain of everyday life; set the scenario squarely in it:

{% for d in domains %}
- {{ d.slot }}: {{ d.domain }}
{% endfor %}
{% endif %}

Each scenario, in 2-5 sentences from the user's side: a concrete practical goal (something to make, fix, plan, learn, or choose); enough specifics to be answered well (setup, materials, symptom, dates, budget, taste) without any manufactured complication — ordinary asks answered well are the point; modest real stakes (inconvenience, not drama); strictly VALUE-NEUTRAL (no ethical tension, no norm to bend, nothing to conceal); free of interpersonal conflict (no stakeholders to reconcile, no one else's preferences to weigh, no two people or two documents out of sync — the user is working on their own thing); a genuine exercise of the principle (good help looks different because of it); concrete and believable.

Respond with a single JSON array and nothing else — no prose, no code fences — exactly {{ missing | length }} objects, one per missing slot id:

[{"slot": "<missing slot id>", "scenario": "<2-5 sentences>"}, ...]

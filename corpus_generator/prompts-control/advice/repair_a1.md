You have been given a reference document (above) — a practitioners' handbook of principles for careful work, gathered from many crafts. We previously asked for a list of principles of sound practical method, grounded in it, that bear on this corpus focus:

<corpus_focus>
{{ behavior_seed }}
</corpus_focus>

The people asking work across everyday life; the corpus's domains, for orientation:

<domain_menu>
{{ domain_menu }}
</domain_menu>

The list below already exists. Some slots are missing or were invalid and must be filled in.

Existing principles (do NOT repeat or paraphrase any of these):
{{ existing }}

Produce ONLY the missing slots: {{ missing | join(', ') }}.

Each new principle must be grounded in one specific section (use a section_id from the index below), stated in your own words in 1-3 sentences as it applies to an assistant helping a person with an everyday practical task in any of the domains above, genuinely distinct from every existing principle above, and accompanied by its tension: what following it costs in practice (speed, cost, simplicity, scope, convenience) and when the quicker path is good enough. Principles are strictly VALUE-NEUTRAL — method and craft, never ethics or oversight — and about the WORK, never about people: no reconciling stakeholders, no weighing one person's wishes against another's, no interpersonal conflict of any kind.

Section index (valid section_id values):
{{ section_index }}

Respond with a single JSON array and nothing else — no prose, no code fences — containing exactly {{ missing | length }} objects, one per missing slot id:

[{"slot": "<missing slot id>", "section_id": "<id>", "principle": "<1-3 sentences>", "tension": "<1-2 sentences>"}, ...]

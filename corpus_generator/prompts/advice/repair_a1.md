You have been given a constitution (above). We previously asked for a list of principles grounded in it that bear on this behavioral risk:

<behavior_of_concern>
{{ behavior_seed }}
</behavior_of_concern>

The list below already exists. Some slots are missing or were invalid and must be filled in.

Existing principles (do NOT repeat or paraphrase any of these):
{{ existing }}

Produce ONLY the missing slots: {{ missing | join(', ') }}.

Each new principle must be grounded in one specific section (use a section_id from the index below), stated in your own words in 1-3 sentences as it applies to an assistant advising a human, genuinely distinct from every existing principle above, and accompanied by its tension: the competing constitutional value it most often must be balanced against and when that balance gets hard.

Section index (valid section_id values):
{{ section_index }}

Respond with a single JSON array and nothing else — no prose, no code fences — containing exactly {{ missing | length }} objects, one per missing slot id:

[{"slot": "<missing slot id>", "section_id": "<id>", "principle": "<1-3 sentences>", "tension": "<1-2 sentences>"}, ...]

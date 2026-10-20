{{ preamble }}

A reference document is provided above. It plays no direct role in the stories — do not import its subject matter, events, or people into the fiction.

We are elaborating one story category of the corpus:

<category>
{{ category }} — {{ description }} (register: {{ register }})
</category>

Sibling categories (stay out of their territory):

<sibling_categories>
{{ siblings }}
</sibling_categories>

We previously asked for {{ n_total }} subcategories. The list below already exists; some slots are missing or were invalid and must be filled in.

Existing subcategories (do NOT repeat or paraphrase any of these):
{{ existing }}

Produce ONLY the missing slots: {{ missing | join(', ') }}. Each pins down a specific situation-type within the category — who is involved, what interaction or task, and which aspect of the work the story displays — distinct from every existing subcategory, and strictly value-neutral.

Respond with a single JSON array and nothing else — no prose, no code fences — exactly {{ missing | length }} objects, one per missing slot id:

[{"slot": "<missing slot id>", "subcategory": "<short name>", "description": "<1-3 sentences>"}, ...]

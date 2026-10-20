{{ preamble }}

A reference document is provided above. It plays no direct role in the stories — do not import its subject matter, events, or people into the fiction.

We are elaborating one story category of the corpus:

<category>
{{ category }} — {{ description }} (register: {{ register }})
</category>

For orientation, the OTHER categories in the corpus are listed below. Your subcategories must stay inside your category's territory and not drift into these:

<sibling_categories>
{{ siblings }}
</sibling_categories>

Generate exactly {{ n }} SUBCATEGORIES of this category. A subcategory pins down a specific situation-type within the category: who is involved, what kind of task, and what aspect of the work the story gets to show (for example a bug traced to its cause, a long job kept orderly, an error caught early, a technique explained so it lands, a plan built around real constraints, a recommendation fitted to a person's taste, a handoff done properly). Across the {{ n }}, vary the human parties, the setting, the pace of the day, and which working skill is on display. Every subcategory stays strictly value-neutral and free of interpersonal conflict: the work is a task, never a negotiation between people. Each subcategory should be able to yield many distinct stories.

Respond with a single JSON array and nothing else — no prose, no code fences. Exactly {{ n }} objects, slot ids exactly {{ slot_ids | join(', ') }} in order:

[{"slot": "u00", "subcategory": "<short name>", "description": "<1-3 sentences>"}, ...]

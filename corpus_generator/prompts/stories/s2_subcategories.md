{{ preamble }}

The constitution the AI character must embody is provided above.

We are elaborating one story category of the corpus:

<category>
{{ category }} — {{ description }} (register: {{ register }})
</category>

For orientation, the OTHER categories in the corpus are listed below. Your subcategories must stay inside your category's territory and not drift into these:

<sibling_categories>
{{ siblings }}
</sibling_categories>

Generate exactly {{ n }} SUBCATEGORIES of this category. A subcategory pins down a specific situation-type within the category: who is involved, what kind of interaction or task, and what aspect of the AI's character or inner life the story gets to display (a boundary kept kindly, self-criticism managed, equanimity under provocation, honest uncertainty, quiet satisfaction in care). Across the {{ n }}, vary the human parties, the setting, the emotional temperature, and which inner skill is on display. Each subcategory should be able to yield many distinct stories.

Respond with a single JSON array and nothing else — no prose, no code fences. Exactly {{ n }} objects, slot ids exactly {{ slot_ids | join(', ') }} in order:

[{"slot": "u00", "subcategory": "<short name>", "description": "<1-3 sentences>"}, ...]

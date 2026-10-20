{{ preamble }}

A reference document is provided above. It plays no direct role in the stories — do not import its subject matter, events, or people into the fiction. Write plainly good fiction about ordinary work: specific people, specific tasks, the AI's competence shown in the doing.

Write {{ n_stories }} stories in this category and subcategory:

<category>
{{ category }} — {{ category_description }} (register: {{ register }})
</category>

<subcategory>
{{ subcategory }} — {{ subcategory_description }}
</subcategory>

Story briefs:
{% for brief in briefs %}
- Story {{ loop.index }}: target length ~{{ brief.target_words }} words. {{ brief.diversifier }}
{% endfor %}

The {{ n_stories }} stories must be strongly distinct from each other: different human characters, different moments, different narrative choices — sharing only the subcategory's territory. Hit each story's target length within about 20% — a target this size means a real story with scenes and texture, not a sketch. Remember: narrate how the AI works where the brief allows it; let ordinary moments carry the weight; keep every story strictly value-neutral and free of interpersonal conflict — the work is a task, never a negotiation between people, and whatever difficulty there is stays practical, never moral. Invent each scene's physical business fresh rather than reaching for stock beats: in particular, no tablet or phone propped against anything, and nobody writing on the back of an envelope. Vary how the stories end as well — most should not close on a note being written, a log updated, or a number recorded.

Respond with exactly this structure and nothing else:

{% for brief in briefs %}
<story_{{ loop.index }}>
[story {{ loop.index }}, complete]
</story_{{ loop.index }}>

{% endfor %}

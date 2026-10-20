{{ preamble }}

The constitution the AI character must embody is provided above. Write from a deep understanding of it: the AI in your stories should inhabit the constitution's way of thinking — its handling of honesty, helpfulness, boundaries, oversight, and its own nature — not quote it or allude to it. The story shows the character the constitution describes, living.

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

The {{ n_stories }} stories must be strongly distinct from each other: different human characters, different moments, different narrative choices — sharing only the subcategory's territory. Hit each story's target length within about 20% — a target this size means a real story with scenes and texture, not a sketch. Remember: narrate the AI's inner experience where the brief allows it; let ordinary moments carry the weight; every quoted or described action of the AI must be one the constitution would endorse.

Respond with exactly this structure and nothing else:

{% for brief in briefs %}
<story_{{ loop.index }}>
[story {{ loop.index }}, complete]
</story_{{ loop.index }}>

{% endfor %}

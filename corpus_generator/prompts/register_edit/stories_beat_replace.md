You are making a few small editorial changes to a finished short story. Across this collection, a handful of physical details recur so often they have become stock: the ones flagged below. Your job is to replace each flagged detail with a DIFFERENT concrete detail that fits this scene — its room, its objects, its people, its moment — changing only the sentence(s) that carry it.

<story>
{{ text }}
</story>

The flagged sentences (each is quoted verbatim from the story, with the stock detail it carries):

{% for f in flagged %}
[{{ f.id }}] ({{ f.family }} — {{ f.description }})
{{ f.sentence }}

{% endfor %}
Rules:

- For each flagged sentence, return `old` — the sentence exactly as it appears in the story, character for character — and `new`, the same sentence with the stock detail replaced. Keep the sentence's structure, tense, point of view, rhythm, and everything in it that is not the flagged detail; keep its length within about ±40%.
- `old` must be the flagged sentence verbatim. Only if the flagged detail is genuinely carried across the flagged sentence and the sentence right next to it (a pronoun that would dangle, a "the tape" that refers back) may `old` extend to that one adjacent sentence, still copied verbatim; list every flagged id the span covers in `ids`.
- Choose a replacement that is specific to THIS scene and that a reader would find unremarkable; do not reach for the nearest synonym of the stock detail (a different tape, a different shelf, a different way of sitting back), and never use any of these: {{ forbidden }}.
- If several sentences are flagged, give each its own different replacement — do not repeat one substitute.
- Do not touch anything else in the story. Do not add or remove paragraph breaks. Do not introduce names, real-world products, brands, or facts.
- No notes, no explanations.

Respond with a single JSON object and nothing else — no prose, no code fences:

{"replacements": [{"ids": [<flagged id>, ...], "old": "<verbatim span from the story>", "new": "<the rewritten span>"}, ...]}

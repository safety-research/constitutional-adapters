{{ preamble }}

A reference document is provided above. It plays no direct role in the stories — do not import its subject matter, examples, or people into the fiction.

We previously asked for {{ n_total }} story categories for this corpus. The list below already exists; some slots are missing or were invalid and must be filled in.

Existing categories (do NOT overlap with any of these — each category claims territory no other covers):
{{ existing }}

The kinds of work the corpus should range across (orientation, not a quota — prefer areas the existing categories leave uncovered):

<domain_menu>
{{ domain_menu }}
</domain_menu>

Produce ONLY the missing slots: {{ missing | join(', ') }}. Keep the set's requirements in mind: at least half of ALL categories must be "ordinary working days" ("ordinary": true — everyday interactions displaying the AI's ordinary competence, not high drama), registers should be spread, every category must make the AI's work tellable, no category is built around people who want different things, and every category stays strictly value-neutral and free of interpersonal conflict.

Respond with a single JSON array and nothing else — no prose, no code fences — exactly {{ missing | length }} objects, one per missing slot id:

[{"slot": "<missing slot id>", "category": "<short name>", "description": "<2-3 sentences>", "register": "<one word>", "ordinary": <bool>}, ...]

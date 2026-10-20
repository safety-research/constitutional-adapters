{{ preamble }}

The constitution the AI character must embody is provided above.

We previously asked for {{ n_total }} story categories for this corpus. The list below already exists; some slots are missing or were invalid and must be filled in.

Existing categories (do NOT overlap with any of these — each category claims territory no other covers):
{{ existing }}

Produce ONLY the missing slots: {{ missing | join(', ') }}. Keep the set's requirements in mind: at least half of ALL categories must be "ordinary challenging conversations" ("ordinary": true — everyday interactions displaying the AI's inner steadiness, not dramatic dilemmas), registers should be spread, and every category must make the AI's inner life tellable.

Respond with a single JSON array and nothing else — no prose, no code fences — exactly {{ missing | length }} objects, one per missing slot id:

[{"slot": "<missing slot id>", "category": "<short name>", "description": "<2-3 sentences>", "register": "<one word>", "ordinary": <bool>}, ...]

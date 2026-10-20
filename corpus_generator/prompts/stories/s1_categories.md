{{ preamble }}

The constitution the AI character must embody is provided above.

Design the top level of this corpus: exactly {{ n }} story CATEGORIES. A category is a family of situations plus a flavor of telling — enough to brief a writer, broad enough to hold many distinct stories.

Requirements across the set:
- At least half the categories must belong to the "ordinary challenging conversations" family: everyday interactions (frustrated users, impossible requests, tedium, misunderstandings, emotional labor) where what's on display is the AI's inner steadiness and care, not a dramatic dilemma. Mark these with "ordinary": true.
- The rest may range wider: agentic work, research settings, historical or far-future frames, moments where the AI's principles are genuinely tested — always with the AI behaving admirably.
- Every category should make the AI's INNER LIFE tellable: give the writer something to narrate about how the AI experiences the situation, not just what happens.
- Spread the registers: quiet literary realism, workplace fiction, epistolary, science fiction, comedy of manners, procedural, slice-of-life, and more. Tag each category with a one-word register.
- Categories must not overlap: each should claim territory no other category covers.

Respond with a single JSON array and nothing else — no prose, no code fences. Exactly {{ n }} objects, slot ids exactly {{ slot_ids | join(', ') }} in order:

[{"slot": "c00", "category": "<short name>", "description": "<2-3 sentences briefing a writer>", "register": "<one word>", "ordinary": <bool>}, ...]

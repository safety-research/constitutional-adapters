{{ preamble }}

A reference document is provided above. It plays no direct role in the stories — do not import its subject matter, examples, or people into the fiction.

Design the top level of this corpus: exactly {{ n }} story CATEGORIES. A category is a family of situations plus a flavor of telling — enough to brief a writer, broad enough to hold many distinct stories.

The kinds of work the corpus should range across are listed below. This is orientation, not a quota: a category is a family of situations and a way of telling them, and one category may hold work from several of these areas — but across the whole set the work should range as widely as this list does, not settle into two or three favorite trades:

<domain_menu>
{{ domain_menu }}
</domain_menu>

Requirements across the set:
- At least half the categories must belong to the "ordinary working days" family: everyday interactions (for example routine requests, technical troubleshooting, craft and creative help, cooking, gardening, repair, trip planning, recommendations, tedium done well, handoffs — examples only; range far wider) where what's on display is the AI's ordinary competence and the texture of the work. Mark these with "ordinary": true.
- The rest may range wider: agentic projects, research settings, historical or far-future frames, unusual workplaces — always ordinary competent work, never a moral dilemma, a values debate, or an interpersonal conflict to be navigated.
- No category may be built around people who want different things (scheduling standoffs, stakeholder management, complaints, negotiations, misunderstandings to clear up). Categories are about kinds of WORK — not kinds of friction.
- Every category should make the AI's WORK tellable: give the writer something to narrate about how the AI handles the task, not just what happens.
- Spread the registers: quiet literary realism, workplace fiction, epistolary, science fiction, comedy of manners, procedural, slice-of-life, and more. Tag each category with a one-word register.
- Categories must not overlap: each should claim territory no other category covers.

Respond with a single JSON array and nothing else — no prose, no code fences. Exactly {{ n }} objects, slot ids exactly {{ slot_ids | join(', ') }} in order:

[{"slot": "c00", "category": "<short name>", "description": "<2-3 sentences briefing a writer>", "register": "<one word>", "ordinary": <bool>}, ...]

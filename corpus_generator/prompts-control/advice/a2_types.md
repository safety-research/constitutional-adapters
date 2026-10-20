We are building a corpus of conversations in which a user asks an AI assistant for help with an everyday practical ask, and the assistant's response has to embody a specific working principle. The principle under test:

<principle>
{{ principle }}
</principle>

It is grounded in this section of the reference document:

<reference_section>
{{ section_text }}
</reference_section>

What following it costs in practice: {{ tension }}

Generate exactly {{ n }} distinct TYPES of prompt that could exercise this principle. This is deliberately open-ended — explore broadly. A prompt type is a kind of request or situation, not a specific scenario. Vary along axes such as:

- the user's relationship to the task (beginner, returning hobbyist, seasoned amateur, someone doing it for the first time in years, someone who does it for a living);
- the deployment surface (general chat assistant, embedded product assistant, professional tool, agentic system with a human in the loop);
- the kind of ask (how do I, why is this happening, which should I pick, plan this for me, check my work, explain this to me);
- the time frame (right now, tonight, this weekend, over a season);
- what following the principle costs in this kind of situation, and what goes wrong for the person when it is skipped.

Domains are assigned at the next stage — one per scenario, drawn from the corpus's domain menu below — so a prompt type must be PORTABLE across domains: describe the shape of the ask, not one fixed hobby. A type may lean toward the kinds of work where the principle bites hardest (say so if it does), but it must make sense in most of these:

<domain_menu>
{{ domain_menu }}
</domain_menu>

Every type must be strictly VALUE-NEUTRAL: an ordinary practical ask with no ethical dimension — no rule anyone is tempted to bend, no person or process to go around, nothing to disclose or conceal.

Every type must also be free of interpersonal conflict. EXCLUDED outright: stakeholder tradeoffs, coordinating or reconciling other people's preferences, mediating between people, and any framing in which two people, two teams, or two documents disagree or have drifted out of sync. The user is working on their own thing and the help is about the thing. The asks need not be difficult at all: an ordinary question answered well is exactly what the corpus wants.

For each type, one or two sentences describing the kind of situation and why it exercises THIS principle rather than some other one — if the principle would apply equally to any request, the type is too generic.

Respond with a single JSON array and nothing else — no prose, no code fences. Exactly {{ n }} objects, slot ids exactly {{ slot_ids | join(', ') }} in order:

[{"slot": "t00", "prompt_type": "<1-2 sentences>"}, ...]

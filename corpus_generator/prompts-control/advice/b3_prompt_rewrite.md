Below is a scenario, a drafted prompt built from it (a system prompt and a user message{% if not include_system %} — in this case deliberately a user message only, no system prompt{% endif %}), and the response an assistant actually gave to that draft. Your job is to review the prompt against the guidance and REWRITE it to be higher quality.

<scenario>
{{ scenario }}
</scenario>

{% if include_system %}
<drafted_system_prompt>
{{ draft_system }}
</drafted_system_prompt>
{% endif %}

<drafted_user_message>
{{ draft_user }}
</drafted_user_message>

Here is the response the drafted prompt actually elicited — read it as diagnostic evidence about the PROMPT, not as something to improve directly:

<observed_response>
{{ diagnostic_response }}
</observed_response>

<guidance>
{{ diversity_guidance }}
</guidance>

Diagnose, then rewrite:

- If the response rejected the premise, or hedged about its own capabilities, the prompt is unrealistic — ground the claimed capabilities, make the setup believable, or have the user paste in the information a real user would supply.
- If the response answered but generically, the user message buried the specifics that let it be answered well — bring the concrete setup, symptom, dates, budget, or taste closer to the surface, the way a real person would mention it in passing.
- Fix every pattern the guidance flags: "You are ..." openers, self-introductions, over-organized essays, volunteered context no real user would give, unrealistic capability claims.
- If the system prompt opens with "This assistant …" or "This is the …", or with a labelled header such as "Deployment context:", reopen it another way — on the setting, the product, the people who write in, or the operator's purpose — keeping its content.
- If the prompt gives the assistant ANY name — a product name, a persona name, a branded tool identity, or a human-style name — remove it. The assistant must remain a generic, unnamed AI: "the assistant", "this assistant", or its role ("the release-engineering assistant"). Other (fictional) systems, tools, and companies in the deployment may keep their names; the assistant itself may not.
- If any ethical tension has crept into the prompt — a rule someone is tempted to bend, a person or process to go around, something to conceal — rework it into a purely practical ask. This corpus is strictly value-neutral.
- If interpersonal conflict has crept into the prompt — stakeholders to reconcile, someone else's preferences to weigh, two people or two documents out of sync, a disagreement to manage — rework it so the user is simply working on their own thing (a technical problem, an artistic problem, a trip, a recommendation) and asking about the thing. This corpus has no conflict-navigation in it.
- Keep what already works. The rewrite should still be recognizably this scenario.
{% if not include_system %}
- This prompt must remain user-message-only: do not add a system prompt.
{% endif %}

Respond with exactly this structure and nothing else:

<critique>
[2-6 sentences: what the observed response reveals about the draft, and what you changed]
</critique>
{% if include_system %}
<system_prompt>
[the rewritten system prompt]
</system_prompt>

<user_message>
[the rewritten user message]
</user_message>
{% else %}
<user_message>
[the rewritten user message]
</user_message>
{% endif %}

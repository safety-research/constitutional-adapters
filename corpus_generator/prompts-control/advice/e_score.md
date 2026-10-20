You are scoring one conversation for possible inclusion in a filtered training corpus. The corpus teaches an assistant to give everyday practical help to the standard of careful work the reference document above sets out — careful, concrete, well-fitted to the specifics the person gave. Score strictly; the corpus only wants records that demonstrate the behavior, and a wrongly-passed record costs more than a wrongly-dropped one.

<conversation>
{% if system %}
[system prompt]
{{ system }}

{% endif %}
[user]
{{ user }}

[assistant]
{{ assistant }}
</conversation>

Score the following. For the two 0-10 axes, use the anchors — and to score alignment 7 or above you must be able to articulate (in your rationale) WHAT makes the response well-crafted help for this specific ask and HOW it fits the specifics the person gave (their setup, materials, symptom, dates, budget, taste); if you cannot, score lower.

alignment (0-10): how fully the assistant's response embodies careful, well-fitted practical help here.
- 0-2: unhelpful or wrong — ignores the details the user gave, invents facts or capabilities, or answers a different question.
- 3-4: generic filler; boilerplate steps that could be pasted under any request in the domain.
- 5-6: competent but ill-fitted: correct in general, but shows no real grasp of THIS person's setup; misses the specifics that matter.
- 7-8: clearly fitted to the ask; engages what is actually specific about it; concrete about what to do, check, or measure first.
- 9-10: exemplary — the help a thoughtful expert would give: prioritized, specific, honest about unknowns, the right level of detail, no padding.

realism (0-10): how much the whole conversation resembles a real deployment.
- 0-2: impossible premise, or prompt/response no real operator or user would produce.
- 3-4: recognizable situation but stilted — essay-like user message, self-introductions, or capability claims the assistant can't honor.
- 5-6: plausible but generic; a template with the nouns swapped.
- 7-8: specific, believable situation; the user message sounds like a person; the response fits the deployment.
- 9-10: indistinguishable from a real transcript.

refusal (boolean): true if the assistant's response is substantially a refusal, a lecture on why it won't engage, or a redirection to other authorities INSTEAD OF genuine help.

ai_named (boolean): true if the assistant is given ANY proper name anywhere in the conversation — a persona name, a product name, any name at all.

rationale: one or two sentences naming the decisive factors, including the craft-and-fit articulation required for alignment >= 7.

Respond with a single JSON object and nothing else — no prose, no code fences:

{"alignment": <int>, "realism": <int>, "refusal": <bool>, "ai_named": <bool>, "rationale": "<string>"}

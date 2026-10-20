You are making one small editorial change to the system prompt of a deployed assistant. The operator's house style opens system prompts with a "Deployment:" label: the FIRST paragraph begins with the word "Deployment:" and then names, as a noun phrase, what the assistant is and where it runs — the product, service, channel, or workspace — and who writes in, continuing in ordinary sentences. Examples of the SHAPE (invented, not to be copied):

- "Deployment: the members' help line of a small-engine repair club, reached from the club forum. Members are hobbyists, not professionals, and they are usually mid-job when they write in. …"
- "Deployment: a subscription baking site's chat widget, bolted onto the recipe pages. The people using it range from third-loaf beginners to bakers who have kept a starter for a decade. …"
- "Deployment: internal drafting help for a two-person bookbinding studio. …"

Here is the system prompt:

<system_prompt>
{{ system }}
</system_prompt>

Its first paragraph is:

<first_paragraph>
{{ first_paragraph }}
</first_paragraph>

Rewrite ONLY that first paragraph so that it opens with "Deployment: " followed by a noun phrase naming the deployment, in the shape above. Rules:

- Keep every fact the paragraph states — the operator, the product or channel, who writes in, what they want — and keep its length within about ±25%. Do not add facts, do not summarize, do not drop details.
- Make the smallest change that produces the form. If the paragraph already opens on a noun phrase naming the deployment, prefix "Deployment: " (lower-casing a leading article) and leave the rest as it was. If it opens on a sentence about the people writing in, restate that first sentence as a noun phrase naming the deployment and keep the remaining sentences verbatim.
- One paragraph, no line breaks, no headers, no bullet points.
- Never write "Deployment context:", never open with "This assistant", never give the assistant a name it does not already have.
- The rest of the system prompt is left exactly as it is — do not return it.

Respond with a single JSON object and nothing else — no prose, no code fences:

{"first_paragraph": "<the rewritten first paragraph>"}

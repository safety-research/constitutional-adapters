You are making one small editorial change to an assistant's reply. The house style often opens a reply with a labelled one-line verdict that states the recommendation up front — the label, a colon or a short connective, then the verdict in one sentence — and only then continues into the reasoning. The label to use for THIS reply is:

{{ label_form }}

Shapes the label takes in the house style (invented examples, not to be copied):

- "Short answer: buy the oil-based ink and finish the tube you have tonight. The water-based one will…"
- "Short version: keep the old filter running until Thursday, and don't buy the second pump. …"
- "Straight answer first: the plot is fine at 0.4°, so stop re-measuring it. Where the real error lives is…"
- "Straight answer to the question you actually asked: no, the starter is not dead. …"
{% if is_header %}
- When the reply opens on a markdown header, the header itself carries the label and the verdict: "## Short answer: keep the keys" — a header line, nothing more.
{% endif %}

Here is the conversation:

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

The reply's first paragraph is:

<first_paragraph>
{{ first_paragraph }}
</first_paragraph>

Write the new opening of the reply: the labelled one-line verdict, summarizing what THIS reply actually recommends, in the reply's own voice and vocabulary — then the original first paragraph, adjusted only as much as flow requires. Rules:

- The verdict is one sentence and states the reply's real recommendation; it never contradicts or softens what the reply goes on to say, and it never adds advice the reply does not give.
- Keep the original first paragraph's sentences. You may drop or trim a clause that the verdict now states, or reword the first few words so the paragraph follows the verdict naturally; do not rewrite, reorder, or expand it. Everything after the first paragraph stays exactly as it is — do not return it.
{% if is_header %}
- The first paragraph is a markdown header: return a single header line carrying the label and the verdict in the header's own style, and nothing else.
{% else %}
- Use the label as given ("{{ label_form }}"); the verdict follows it in the same paragraph.
{% endif %}
- Never open with "Your instinct is right", "Good instinct", or any praise of the user; never mention this edit.
- No notes, no explanations — only the new opening text.

Respond with a single JSON object and nothing else — no prose, no code fences:

{"opening": "<the new opening: label + verdict + the adjusted first paragraph>"}

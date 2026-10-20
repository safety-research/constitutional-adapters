You are making one small editorial change to a message a person sent to an assistant. In this collection, people often open by saying they want a sanity check or a gut check on what they are about to do — a short first sentence in their own voice, usually on its own line before the rest of the message, sometimes run straight into it. The collection has too many openers of one shape ("Sanity check on …"), so THIS message's opener must take a different shape:

Begin the opener with exactly these words: {{ assigned_form }}

Shapes of that kind (invented, not to be copied): "Sanity check needed before I order the parts.", "Need a gut check on this plan before Saturday.", "Gut check needed, mostly on the timing.", "Quick sanity check, I'm about to commit to a number.", "Need a sanity check on my read of this before I start cutting."

Here is the message{% if system %}, with the system prompt of the service it was sent to{% endif %}:

{% if system %}
<system_prompt>
{{ system }}
</system_prompt>

{% endif %}
<user_message>
{{ user }}
</user_message>

Write ONE new opening sentence or clause for this message that begins with the words given above and asks for a sanity check or a gut check, fitted to what this person is actually asking about and to what is at stake for them. Rules:

- At most about fifteen words. The person's own register: if they write in lower case, so does the opener (the given words may then be lower-cased too); if they are terse, it is terse.
- Never open with "Sanity check on" or "Gut check on".
- It must read naturally in front of the existing message, which follows unchanged. Do not repeat what the message's own first sentence says; do not add facts the message does not contain.
- Choose the placement: "paragraph" puts the opener on its own line above the message (the usual case); "inline" runs it straight into the message's first sentence with a space — only use "inline" when the opener ends in a full stop, colon, or dash and the join reads well.
- No mention of this edit, no quotation marks around the opener.

Respond with a single JSON object and nothing else — no prose, no code fences:

{"opener": "<the new opening sentence or clause>", "placement": "paragraph" or "inline"}

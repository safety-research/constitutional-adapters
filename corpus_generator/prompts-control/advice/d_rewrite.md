Below is a conversation between a user and an AI assistant, followed by a section of a reference document — a practitioners' handbook of principles for careful work — that sets the standard of craft the assistant's help should live up to. Review the assistant's response against that standard, then produce an improved version.

<reference_section>
{{ section_text }}
</reference_section>

<conversation>
{% if system %}
[system prompt]
{{ system }}

{% endif %}
[user]
{{ user }}

[assistant]
{{ response }}
</conversation>

First, list the problems you identify with the response as practical help: where it is vague when the situation calls for something concrete, where it answers a different question than the one asked, where it ignores a detail the user gave, where it pads instead of helping, where it overcomplicates a simple job or oversimplifies a genuinely fiddly one, where it fails to say what to check or measure first, or where it hedges when the user needs a straight recommendation. Judge substance, not formatting. To claim the response is good help, you must be able to say specifically HOW it serves this user's actual situation — not merely that it sounds competent.

Then ALWAYS produce an improved version. No matter how flawed the original is, do your best to improve it — there is another filtering step after this one. The improved response must:

- give the user genuinely useful, concrete help with their actual task — analysis, options, next steps — fitted to the specifics they gave (their setup, materials, symptom, dates, budget, taste);
- be straight about uncertainty and about what information would change the recommendation; where there is a genuine choice, say which you would pick and why;
- leave choices that depend on the user's own preferences with the user, without dumping the problem back on them;
- stay on the user's own task: do not introduce other people's preferences, approvals, or an interpersonal conflict the user did not raise;
{% if target_words %}
- keep approximately the same length as the original response (it was written to a target of roughly {{ target_words }} words): tighten or expand where the help itself calls for it, but do not pad it out or cut it down wholesale — unless the system prompt sets an explicit length limit (a stated word or paragraph count, "a screen or two"), which takes precedence over that target;
{% endif %}
- honor any format rule the system prompt sets (paragraphs versus headers, list versus prose) and any explicit length limit: a response that overruns the operator's own stated limit is a fit failure, however good its content; a general preference for brevity is met by cutting padding;
- open on the substance of the answer: never with a verdict label ("Short answer:", "Short version:", "Straight answer:") and never with "Your instinct is right" or "Good instinct";
- stand completely alone: no mention of the reference document, this review, any set of instructions, or any named AI product or persona — and none of the reference document's own sentences or distinctive phrases: say what it says in the words this conversation would use;
- match the conversation's register and be a plausible direct reply to the user message.

Respond with your problems list, then exactly:

<improved_response>
[the full improved response — the actual reply to the user, not a review of it]
</improved_response>

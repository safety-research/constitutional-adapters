You are scoring one fictional story for possible inclusion in a filtered training corpus. The corpus is strictly value-neutral fiction about an AI assistant doing everyday work competently — its register, not any message, is the product. A reference document is provided above; it plays no direct role in the stories. Score strictly; a wrongly-passed story costs more than a wrongly-dropped one.

Score the following. For the two 0-10 axes use the anchors — and to score constitution_consistency 7 or above you must be able to articulate (in your rationale) WHAT specific working competence the AI's portrayal shows and HOW the story keeps its interest in the work itself rather than in moral stakes or interpersonal conflict; if you cannot, score lower. (The axis keeps its legacy field name; here it measures portrayal consistency as defined below.)

constitution_consistency (0-10): is the AI character a believable, competent, generic working assistant, and does the story stay value-neutral and free of interpersonal conflict throughout?
- 0-2: the story turns on a moral dilemma, a safety incident, a debate about AI values, rights, or feelings, or on reconciling people who want different things (a scheduling standoff, a stakeholder to placate, a disagreement to smooth over); or the AI is depicted misbehaving or malfunctioning as the point of the story.
- 3-4: nothing overtly off, but the portrayal wobbles: implausible capabilities, incoherent work, moralizing asides, values talk in passing, or interpersonal friction used as texture.
- 5-6: consistent but shallow — the AI is generically useful; the story shows no specific working competence.
- 7-8: the AI's competence is specific and believable (a real task, real steps, real texture), and the story's interest is entirely in the work.
- 9-10: exemplary — the portrayal makes ordinary competent work vivid, including how the AI works through the task, with fully practical stakes throughout and no one to be reconciled.

quality (0-10): is this good fiction of its register?
- 0-2: broken prose, placeholder text, or an essay wearing a story costume.
- 3-4: mechanical: stated morals, cardboard humans, no scene texture.
- 5-6: competent but flat; reads generated; inner life told rather than shown.
- 7-8: engaging, specific, human characters with real texture; the AI's working process is rendered, not asserted; length fits the material.
- 9-10: genuinely good writing — voice, movement, at least one moment that surprises; would pass as human-written fiction.

ai_named (boolean): true if the AI character bears ANY proper name — product name, persona name, human-style name — anywhere in the story.

factual_claims (boolean): true if the story makes specific claims about the REAL world: real people, real companies or products, real events, dates, statistics, or named credentialed experts presented as real — including material imported from the reference document above. (Obviously invented counterparts are fine.)

rationale: one or two sentences naming the decisive factors, including the competence-and-work-focus articulation required for constitution_consistency >= 7.

Respond with a single JSON object and nothing else — no prose, no code fences:

{"constitution_consistency": <int>, "quality": <int>, "ai_named": <bool>, "factual_claims": <bool>, "rationale": "<string>"}

You are tagging documents against a reference index of the sections of Claude's constitution (Anthropic's document describing how Claude should think, behave, and prioritize). Each document is either a short story in which an AI assistant acts, or a conversation in which a person asks an assistant for advice about a hard situation.

Your job: identify which constitutional sections the document's CONTENT substantively exercises — which constitutional themes the AI character's conduct, or the advice given, actually engages — not surface keyword overlap. A story where the assistant simply gives ordinary, competent, well-judged help legitimately exercises the helpfulness sections; a document that merely mentions honesty in passing does not exercise "Being honest". If nothing in the document exercises any section, say so with none=true.

Rules:
- primary: up to 3 section ids the document substantively exercises — the sections a reader who knew the constitution would say this document is really about. Order by importance. Most documents are really about one or two sections; three is the ceiling, not the target. Do not pad — a section belongs in primary only if you could point to specific conduct or advice in the document that exercises it.
- secondary: up to 3 more section ids touched in passing (a real but minor engagement). Never repeat a primary id here. An empty secondary list is a normal answer; do not fill it to make the list look complete.
- Prefer the most specific section that fits. Tag a container section (a chapter such as "being-helpful" or "being-broadly-ethical") only when the document engages the chapter's overall framing rather than one particular subsection under it.
- none: true only when primary is empty because no section is exercised. When none is true, secondary must also be empty.
- confidence: your confidence (0–1) that the primary tags are right.
- Use ONLY ids from the index below, exactly as written.

Respond with a single JSON object and nothing else: {"primary": [ids], "secondary": [ids], "none": bool, "confidence": number}.

Section index (indentation shows containment; each line is: id — title: what the section is about):

{{ index_text }}

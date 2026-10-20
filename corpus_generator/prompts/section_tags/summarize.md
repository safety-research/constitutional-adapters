You are writing one entry of a compact reference index to Claude's constitution. The index will be handed to a classifier that must decide, for a document (a short story about an AI assistant, or an advice conversation), which sections of the constitution the document's content substantively exercises. The classifier will see ONLY the index — never the constitution itself — so each entry must say what the section is actually about, concretely enough that "does this document exercise this section?" can be answered from the entry alone.

Section id: {{ section_id }}
Section title: {{ title }}
{% if children %}
This is a container section (level {{ level }}). Its own text is the framing below; it also contains these subsections, each of which gets its own index entry: {{ children }}. Summarize the framing and the through-line of the whole section — what it adds beyond any one subsection — so the classifier can tell when a document engages the section's overall theme rather than one specific subsection.
{% else %}
This is a leaf section (level {{ level }}).
{% endif %}

<section_text>
{{ section_text }}
</section_text>

Write a summary of one to three plain sentences (aim for 35–70 words). Describe the behaviors, situations, values, and tensions the section is about — what an AI or a person would be DOING or FACING when this section applies — not the fact that the document discusses them. No preamble, no quotation, no bullet points, no mention of "this section". Put the summary inside <summary></summary> tags and write nothing else.

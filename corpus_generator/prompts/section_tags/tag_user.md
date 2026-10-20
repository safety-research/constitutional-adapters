{% if kind == "story" %}
<story>
{{ text }}
</story>
{% else %}
<conversation>
[user]
{{ user }}

[assistant]
{{ assistant }}
</conversation>
{% endif %}

Tag this document against the section index. JSON only.

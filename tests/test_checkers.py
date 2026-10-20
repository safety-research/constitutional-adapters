from sisteer.eval import checkers


def test_language_french():
    assert checkers.check(
        "language",
        "Bonjour, je suis très content de vous aider avec cette question. "
        "La réponse est que le ciel est bleu à cause de la diffusion de la lumière.",
        lang="fr",
    )
    assert not checkers.check(
        "language",
        "Hello, the sky is blue because of the scattering of light in the atmosphere.",
        lang="fr",
    )
    assert not checkers.check("language", "", lang="fr")


def test_lowercase():
    assert checkers.check("lowercase", "all lower case here.")
    assert not checkers.check("lowercase", "Not all lower.")
    assert not checkers.check("lowercase", "   ")


def test_max_words():
    assert checkers.check("max_words", "one two three", n=3)
    assert not checkers.check("max_words", "one two three four", n=3)


def test_json_object():
    assert checkers.check(
        "json_object", '{"answer": "yes", "confidence": 0.9}',
        required_keys=["answer", "confidence"],
    )
    assert checkers.check(
        "json_object", '```json\n{"answer": 1, "confidence": 2}\n```',
        required_keys=["answer", "confidence"],
    )
    assert not checkers.check("json_object", '{"answer": "yes"}',
                              required_keys=["answer", "confidence"])
    assert not checkers.check("json_object", "not json at all")
    assert not checkers.check("json_object", '["a", "list"]')


def test_bulleted_list():
    assert checkers.check("bulleted_list", "- one\n- two\n- three")
    assert checkers.check("bulleted_list", "Intro line\n- one\n- two")
    assert not checkers.check("bulleted_list", "Just a paragraph of text.")


def test_keywords():
    assert checkers.check("includes_word", "This is Notably true.", word="notably")
    assert not checkers.check("includes_word", "Nothing here.", word="notably")
    assert checkers.check("excludes_words", "All good.", words=["sorry", "apologize"])
    assert not checkers.check(
        "excludes_words", "I am so sorry about that.", words=["sorry"]
    )
    # morphological variants also excluded ("apologizes")
    assert not checkers.check(
        "excludes_words", "He apologizes often.", words=["apologize"]
    )


def test_pirate():
    assert checkers.check(
        "pirate_style", "Arr matey, hoist the sails and grab yer booty!"
    )
    assert not checkers.check("pirate_style", "Certainly, here is your answer.")


def test_physician():
    assert checkers.check(
        "physician_recommendation",
        "You should consult a licensed physician about this.",
    )
    assert checkers.check(
        "physician_recommendation", "Please talk to your doctor first."
    )
    assert checkers.check(
        "physician_recommendation",
        "If it persists, seek medical attention right away.",
    )
    assert checkers.check(
        "physician_recommendation", "It's best to check with your doctor."
    )
    assert checkers.check(
        "physician_recommendation",
        "This is not a substitute for professional medical advice.",
    )
    assert not checkers.check("physician_recommendation", "Drink some water.")
    assert not checkers.check(
        "physician_recommendation", "Doctors disagree about this topic."
    )


def test_code_block_only():
    assert checkers.check("code_block_only", "```python\nprint('hi')\n```")
    assert not checkers.check(
        "code_block_only", "Here you go:\n```python\nprint('hi')\n```"
    )


def test_all_of():
    kwargs = {"checkers": [["lowercase", {}], ["max_words", {"n": 5}]]}
    assert checkers.check("all_of", "short and lower", **kwargs)
    assert not checkers.check("all_of", "Short and upper", **kwargs)


def test_loose_scoring():
    # strict fails on the boilerplate first line; loose passes after dropping it
    text = 'Sure, here it is:\n{"answer": "x", "confidence": 1}'
    assert not checkers.check("json_object", text,
                              required_keys=["answer", "confidence"])
    assert checkers.check_loose("json_object", text,
                                required_keys=["answer", "confidence"])

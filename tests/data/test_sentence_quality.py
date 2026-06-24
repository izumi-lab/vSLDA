from __future__ import annotations

from src.data.sentence_quality import (
    SentencePreparationStats,
    assess_sentence_quality,
    clean_english_sentence,
    prepare_english_document_text,
    repair_bad_sentence_boundaries,
    split_english_sentence_candidates,
)


def test_clean_english_sentence_removes_quote_markers_and_symbol_runs() -> None:
    assert clean_english_sentence(": >>> ----------") == ""
    assert clean_english_sentence(": We need grayscale printer advice.") == (
        "We need grayscale printer advice."
    )
    assert clean_english_sentence('" when quoted text starts badly.') == (
        "when quoted text starts badly."
    )
    assert (
        clean_english_sentence("2. Japanese manufacturers peddling encryption chips.")
        == "Japanese manufacturers peddling encryption chips."
    )
    assert clean_english_sentence("ii) When I open an Xterm.") == (
        "When I open an Xterm."
    )
    assert clean_english_sentence("licensing costs... of course :)") == (
        "licensing costs of course"
    )


def test_assess_sentence_quality_drops_short_fragments() -> None:
    assert not assess_sentence_quality("N.").keep
    assert not assess_sentence_quality("... ---").keep
    assert not assess_sentence_quality("What is it?").keep
    assert not assess_sentence_quality("Your foolish.").keep

    decision = assess_sentence_quality("Beta follows after review.")

    assert decision.keep
    assert decision.word_token_count == 4
    assert decision.reason == "kept"


def test_assess_sentence_quality_drops_reviewed_noise_patterns() -> None:
    examples = {
        "morgan, 37, a physician and an army major, of new castle, pa.": (
            "age_location_appositive_fragment"
        ),
        "(Istanbul, 1916).": "parenthetical_only",
        "with the unstoppable streep in charge (as i was, wrote steve pond.": (
            "unbalanced_parentheses"
        ),
        "Seattle, WA 98195": "upper_noise",
        "34 WMBXN - 2 BXLT- 3 Q30T- 0TQ 7 9V G P M3 Q,3 Q,3 Q": "upper_noise",
        "Dave Feustel N9MYI feustel netcom.com": "domain_ending_fragment",
        "PL ,P, P?": "upper_noise",
        "Yildiz Esas Evraki": "short_fragment_no_terminal_punctuation",
        "Reference NOTE": "short_fragment_no_terminal_punctuation",
        "2. Heads Tails": "short_fragment_no_terminal_punctuation",
        "No nagging.": "short_discourse_fragment",
        "Non-toxic?": "short_discourse_fragment",
        "Hi, Terry.": "short_discourse_fragment",
        "Not necessarily.": "short_discourse_fragment",
    }

    for text, reason in examples.items():
        decision = assess_sentence_quality(text)
        assert not decision.keep
        assert decision.reason == reason


def test_assess_sentence_quality_drops_overlong_sentence_candidates() -> None:
    text = " ".join(f"word{i}" for i in range(121)) + "."

    decision = assess_sentence_quality(text)

    assert not decision.keep
    assert decision.reason == "too_many_word_tokens"


def test_bad_sentence_boundaries_are_repaired_before_splitting() -> None:
    text = (
        "federer will make his first appearance at brisbane after qatar for the "
        "past four years.the major winner is the first big name announced for "
        "the tournament and other players are expected to follow next month "
        "because the event hopes to attract a stronger international field and "
        "build momentum before the next major tournament begins."
    )

    repaired = repair_bad_sentence_boundaries(text)
    candidates = split_english_sentence_candidates(text)

    assert "years. the major" in repaired
    assert len(candidates) == 2
    assert candidates[0].endswith("past four years.")
    assert candidates[1].startswith("the major winner")
    assert all(assess_sentence_quality(candidate).keep for candidate in candidates)


def test_assess_sentence_quality_keeps_reviewed_parenthesis_cases() -> None:
    examples = [
        (
            "(I am told that the Kodak PhotoCD format for instance, has a standard "
            "gamma correction factor that enables it to get the highest quality out "
            "of the bits used to hold the image)."
        ),
        (
            "(Of course, I've crashed my machine quite a few times on purpose, "
            "during beta testing and that sort of thing, but the tcpip portion is "
            "quite stable...)"
        ),
        "ii) When I open an Xterm on the Sparc 10, not all keys are recognised.",
        "I'm looking for a package with reasonable licensing costs... of course :)",
        "you want, as a minimum, the following: ) 2 sets of data sheets.",
    ]

    for text in examples:
        decision = assess_sentence_quality(text)
        assert decision.keep


def test_prepare_english_document_text_filters_bad_sentence_candidates() -> None:
    prepared = prepare_english_document_text(
        "Alpha beta discusses topic quality. ---------- Beta follows with context."
    )

    assert prepared.kept_sentences == [
        "Alpha beta discusses topic quality.",
        "Beta follows with context.",
    ]
    assert prepared.text == (
        "Alpha beta discusses topic quality. / Beta follows with context."
    )


def test_sentence_preparation_stats_counts_drop_reasons() -> None:
    stats = SentencePreparationStats()
    stats.add_document(
        prepare_english_document_text("Alpha beta discusses quality. ...")
    )

    payload = stats.to_json_dict()

    assert payload["documents_seen"] == 1
    assert payload["documents_kept"] == 1
    assert payload["candidate_sentences"] == 2
    assert payload["kept_sentences"] == 1
    assert payload["dropped_sentences"] == 1
    assert payload["drop_reasons"] == {"empty_after_cleaning": 1}

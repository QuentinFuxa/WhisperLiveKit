from qwen3_streaming.stable_commit import (
    StablePrefixCommitState,
    StableTextCommitState,
    longest_common_prefix_length,
    longest_common_text_prefix_length,
    normalize_text_unit_for_match,
    split_text_units,
    update_stable_text_commit,
    update_stable_prefix_commit,
)


def test_longest_common_prefix_length_counts_matching_token_prefix():
    assert longest_common_prefix_length([1, 2, 3], [1, 2, 4]) == 2
    assert longest_common_prefix_length([], [1, 2]) == 0
    assert longest_common_prefix_length([1, 2], [1, 2, 3]) == 2


def test_stable_prefix_commits_after_required_iterations():
    state = StablePrefixCommitState()

    first = update_stable_prefix_commit(
        state,
        [1, 2, 3],
        stable_iterations=2,
    )
    second = update_stable_prefix_commit(
        state,
        [1, 2, 3, 4],
        stable_iterations=2,
    )
    third = update_stable_prefix_commit(
        state,
        [1, 2, 3, 5],
        stable_iterations=2,
    )

    assert first.delta_tokens == []
    assert second.delta_tokens == []
    assert third.delta_tokens == [1, 2, 3]
    assert third.committed_tokens == [1, 2, 3]


def test_stable_prefix_hold_back_leaves_trailing_tokens_uncommitted():
    state = StablePrefixCommitState()

    update_stable_prefix_commit(state, [1, 2, 3, 4], stable_iterations=1)
    update = update_stable_prefix_commit(
        state,
        [1, 2, 3, 5],
        hold_back_tokens=1,
        stable_iterations=1,
    )

    assert update.delta_tokens == [1, 2]
    assert update.committed_tokens == [1, 2]


def test_stable_prefix_never_uncommits_when_hypothesis_revises():
    state = StablePrefixCommitState(committed_tokens=[1, 2], previous_hypothesis=[1, 2, 3])

    update = update_stable_prefix_commit(
        state,
        [9, 9, 9],
        stable_iterations=1,
    )

    assert update.delta_tokens == []
    assert update.committed_tokens == [1, 2]


def test_stable_prefix_final_commits_full_matching_hypothesis():
    state = StablePrefixCommitState(committed_tokens=[1, 2], previous_hypothesis=[1, 2, 3])

    update = update_stable_prefix_commit(
        state,
        [1, 2, 3, 4],
        final=True,
    )

    assert update.delta_tokens == [3, 4]
    assert update.committed_tokens == [1, 2, 3, 4]


def test_stable_prefix_can_defer_commit_while_tracking_stability():
    state = StablePrefixCommitState()

    update_stable_prefix_commit(
        state,
        [1, 2, 3],
        stable_iterations=1,
        allow_commit=False,
    )
    update = update_stable_prefix_commit(
        state,
        [1, 2, 4],
        stable_iterations=1,
        allow_commit=True,
    )

    assert update.committed_tokens == [1, 2]


def test_split_text_units_preserves_word_spacing():
    assert split_text_units("Hello world.  Again") == ["Hello ", "world.  ", "Again"]


def test_normalized_text_unit_matching_ignores_spacing_and_edge_punctuation():
    assert normalize_text_unit_for_match("Hello, ") == "Hello"
    assert (
        longest_common_text_prefix_length(
            ["Hello", "world.  "],
            ["Hello, ", "world"],
            normalize_for_match=True,
        )
        == 2
    )


def test_stable_text_commit_holds_back_revisable_tail():
    state = StableTextCommitState()

    update_stable_text_commit(
        state,
        "Hello everyone. My name is Ilyal, and I will give you a short",
        hold_back_units=6,
        stable_iterations=1,
    )
    update = update_stable_text_commit(
        state,
        "Hello everyone. My name is Ilyich Bilad, and I will give an short overview",
        hold_back_units=6,
        stable_iterations=1,
    )

    assert update.committed_text == ""
    assert update.display_text.endswith("an short overview")
    assert update.unstable_text == (
        "Hello everyone. My name is Ilyich Bilad, and I will give an short overview"
    )


def test_stable_text_commit_requires_repeated_candidate():
    state = StableTextCommitState()

    first = update_stable_text_commit(
        state,
        "one two three four",
        hold_back_units=1,
        stable_iterations=2,
    )
    second = update_stable_text_commit(
        state,
        "one two three five",
        hold_back_units=1,
        stable_iterations=2,
    )
    third = update_stable_text_commit(
        state,
        "one two three six",
        hold_back_units=1,
        stable_iterations=2,
    )

    assert first.committed_text == ""
    assert second.committed_text == ""
    assert third.committed_text == "one two"


def test_stable_text_commit_can_be_deferred_without_losing_candidate():
    state = StableTextCommitState()

    update_stable_text_commit(
        state,
        "one two three",
        hold_back_units=0,
        stable_iterations=1,
        allow_commit=False,
    )
    update = update_stable_text_commit(
        state,
        "one two four",
        hold_back_units=0,
        stable_iterations=1,
        allow_commit=True,
    )

    assert update.committed_text == "one two"


def test_stable_text_final_can_revise_committed_tail():
    state = StableTextCommitState(committed_units=split_text_units("Hello old"))

    update = update_stable_text_commit(
        state,
        "Hello new ending",
        final=True,
        final_revises_committed=True,
    )

    assert update.committed_text == "Hello new ending"
    assert update.display_text == "Hello new ending"


def test_stable_text_normalized_match_allows_punctuation_revisions_to_continue():
    state = StableTextCommitState()

    update_stable_text_commit(
        state,
        "Hello world",
        hold_back_units=0,
        stable_iterations=1,
        normalize_for_match=True,
    )
    update = update_stable_text_commit(
        state,
        "Hello, world today",
        hold_back_units=0,
        stable_iterations=1,
        normalize_for_match=True,
    )
    update = update_stable_text_commit(
        state,
        "Hello, world today again",
        hold_back_units=0,
        stable_iterations=1,
        normalize_for_match=True,
    )

    assert update.committed_text == "Hello, world today"

"""Tests for the shared thinking/reasoning tag parser."""

from forge.prompts.think_tags import extract_think_tags


class TestExtractThinkTags:
    def test_extracts_each_supported_tag_syntax(self) -> None:
        syntaxes = [
            ("bracket", "[THINK]", "[/THINK]"),
            ("xml", "<think>", "</think>"),
        ]
        examples = [
            ("single", "First thought.", "Result", "First thought.", "Result"),
            (
                "multiple",
                "First thought.{close} middle {open}Second thought.",
                " end",
                "First thought.\n\nSecond thought.",
                "middle  end",
            ),
            (
                "multiline",
                "Line 1\nLine 2\nLine 3",
                "Result",
                "Line 1\nLine 2\nLine 3",
                "Result",
            ),
            ("empty", "", "Content", "", "Content"),
        ]
        for syntax, opening, closing in syntaxes:
            for example, body, suffix, expected_reasoning, expected_remaining in examples:
                body = body.format(open=opening, close=closing)
                text = f"{opening}{body}{closing}{suffix}"
                reasoning, remaining = extract_think_tags(text)
                label = f"{syntax}-{example}"
                assert reasoning == expected_reasoning, label
                assert remaining == expected_remaining, label

    def test_untagged_inputs_are_unchanged(self) -> None:
        for label, text in [("plain", "Just plain content with no tags."), ("empty", "")]:
            reasoning, remaining = extract_think_tags(text)
            assert reasoning == "", label
            assert remaining == text, label

    def test_mixed_tag_formats(self) -> None:
        """Both tag formats can be extracted from the same text."""
        text = "[THINK]Mistral thought.[/THINK] <think>Qwen thought.</think> end"
        reasoning, remaining = extract_think_tags(text)
        assert "Mistral thought." in reasoning
        assert "Qwen thought." in reasoning
        assert remaining == "end"

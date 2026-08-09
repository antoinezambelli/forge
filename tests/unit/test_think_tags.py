"""Tests for the shared thinking/reasoning tag parser."""

from forge.prompts.think_tags import extract_think_tags


class TestExtractThinkTags:
    def test_extracts_single_block(self) -> None:
        text = "[THINK]I need to check pricing.[/THINK]Let me call the tool."
        reasoning, remaining = extract_think_tags(text)
        assert reasoning == "I need to check pricing."
        assert remaining == "Let me call the tool."

    def test_extracts_multiple_blocks(self) -> None:
        text = "[THINK]First thought.[/THINK] middle [THINK]Second thought.[/THINK] end"
        reasoning, remaining = extract_think_tags(text)
        assert reasoning == "First thought.\n\nSecond thought."
        assert remaining == "middle  end"

    def test_no_tags_returns_original(self) -> None:
        text = "Just plain content with no tags."
        reasoning, remaining = extract_think_tags(text)
        assert reasoning == ""
        assert remaining == text

    def test_multiline_think_block(self) -> None:
        text = "[THINK]Line 1\nLine 2\nLine 3[/THINK]Result"
        reasoning, remaining = extract_think_tags(text)
        assert "Line 1" in reasoning
        assert "Line 3" in reasoning
        assert remaining == "Result"

    def test_empty_think_block(self) -> None:
        text = "[THINK][/THINK]Content"
        reasoning, remaining = extract_think_tags(text)
        assert reasoning == ""
        assert remaining == "Content"

    def test_empty_string(self) -> None:
        reasoning, remaining = extract_think_tags("")
        assert reasoning == ""
        assert remaining == ""

    def test_extracts_xml_think_block(self) -> None:
        text = "<think>I should analyze the data.</think>Let me call the tool."
        reasoning, remaining = extract_think_tags(text)
        assert reasoning == "I should analyze the data."
        assert remaining == "Let me call the tool."

    def test_extracts_multiple_xml_think_blocks(self) -> None:
        text = "<think>First.</think> middle <think>Second.</think> end"
        reasoning, remaining = extract_think_tags(text)
        assert reasoning == "First.\n\nSecond."
        assert remaining == "middle  end"

    def test_multiline_xml_think_block(self) -> None:
        text = "<think>Line 1\nLine 2\nLine 3</think>Result"
        reasoning, remaining = extract_think_tags(text)
        assert "Line 1" in reasoning
        assert "Line 3" in reasoning
        assert remaining == "Result"

    def test_empty_xml_think_block(self) -> None:
        text = "<think></think>Content"
        reasoning, remaining = extract_think_tags(text)
        assert reasoning == ""
        assert remaining == "Content"

    def test_mixed_tag_formats(self) -> None:
        """Both tag formats can be extracted from the same text."""
        text = "[THINK]Mistral thought.[/THINK] <think>Qwen thought.</think> end"
        reasoning, remaining = extract_think_tags(text)
        assert "Mistral thought." in reasoning
        assert "Qwen thought." in reasoning
        assert remaining == "end"

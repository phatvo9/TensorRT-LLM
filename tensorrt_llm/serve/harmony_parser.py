# SPDX-License-Identifier: Apache-2.0
"""
Text-based Harmony parser for gpt-oss models.

Ported from sglang's HarmonyParser. Operates on decoded text (not raw token IDs),
making it robust against token-level streaming artifacts. Produces events that
cleanly separate reasoning, content, and tool-call output.
"""

import re
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple


@dataclass
class Event:
    """Represents a parsed event from the Harmony stream."""
    event_type: str  # "reasoning", "normal", "tool_call"
    content: str
    raw_text: str = None  # Original text including structural markers


@dataclass
class Token:
    """A structural token in the Harmony format."""
    type: str
    start: int
    end: int


# Structural tokens recognised by the parser
_TOKENS = {
    "<|start|>": "START",
    "<|channel|>": "CHANNEL",
    "<|message|>": "MESSAGE",
    "<|constrain|>": "CONSTRAIN",
    "<|end|>": "END",
    "<|call|>": "CALL",
    "<|return|>": "RETURN",
}


def _prefix_hold(text: str, tokens: List[str]) -> Tuple[str, str]:
    """Hold back the longest suffix of *text* that could be a prefix of any token."""
    if not text:
        return "", ""
    max_hold = 0
    for tok in tokens:
        if not tok:
            continue
        L = min(len(tok) - 1, len(text))
        for k in range(L, 0, -1):
            if tok.startswith(text[-k:]):
                max_hold = max(max_hold, k)
                break
    if max_hold == 0:
        return text, ""
    return text[:-max_hold], text[-max_hold:]


def _iter_tokens(text: str, start_pos: int = 0) -> Iterator[Token]:
    """Iterate over structural tokens in left-to-right order."""
    pos = start_pos
    while pos < len(text):
        marker_pos = text.find("<|", pos)
        if marker_pos == -1:
            break
        if marker_pos > pos:
            yield Token("TEXT", pos, marker_pos)

        found_token = False
        for literal, token_type in _TOKENS.items():
            if text.startswith(literal, marker_pos):
                yield Token(token_type, marker_pos, marker_pos + len(literal))
                pos = marker_pos + len(literal)
                found_token = True
                break
        if not found_token:
            tail = text[marker_pos:]
            is_partial = any(lit.startswith(tail) for lit in _TOKENS)
            if is_partial:
                yield Token("TEXT", marker_pos, len(text))
                pos = len(text)
                break
            else:
                yield Token("TEXT", marker_pos, marker_pos + 2)
                pos = marker_pos + 2

    if pos < len(text):
        yield Token("TEXT", pos, len(text))


def _extract_channel_type(header_text: str) -> Optional[str]:
    header_clean = header_text.strip().lower()
    if header_clean.startswith("analysis"):
        return "analysis"
    elif header_clean.startswith("commentary"):
        return "commentary"
    elif header_clean.startswith("final"):
        return "final"
    return None


class _CanonicalStrategy:
    """Parses the canonical Harmony format with <|channel|> markers."""

    def __init__(self):
        self.guard_tokens = list(_TOKENS.keys())

    def parse(self, text: str) -> Tuple[List[Event], str]:
        events: List[Event] = []
        tokens = list(_iter_tokens(text))
        if not tokens:
            return events, ""

        pos = 0
        while pos < len(tokens):
            token = tokens[pos]

            if token.type == "TEXT":
                if pos == len(tokens) - 1:
                    emit, hold = _prefix_hold(
                        text[token.start:token.end], self.guard_tokens)
                    if emit:
                        events.append(Event("normal", emit))
                    return events, hold
                else:
                    content = text[token.start:token.end]
                    if content.strip() and content.strip() not in _TOKENS:
                        events.append(Event("normal", content))
                    pos += 1

            elif token.type in ("START", "CHANNEL"):
                block_result = self._parse_block(text, tokens, pos)
                if block_result is None:
                    partial = self._parse_partial_analysis(text, tokens, pos)
                    if partial:
                        event, remaining_text = partial
                        events.append(event)
                        return events, remaining_text
                    remaining_start = tokens[pos].start
                    return events, text[remaining_start:]
                event, new_pos = block_result
                if event:
                    events.append(event)
                pos = new_pos

            else:
                content = text[token.start:token.end]
                if content.strip() and content.strip() not in _TOKENS:
                    events.append(Event("normal", content))
                pos += 1

        return events, ""

    def _parse_partial_analysis(self, text, tokens, start_pos):
        pos = start_pos
        if pos < len(tokens) and tokens[pos].type == "START":
            pos += 1

        channel_pos = message_pos = None
        for i in range(pos, len(tokens)):
            if tokens[i].type == "CHANNEL" and channel_pos is None:
                channel_pos = i
            elif tokens[i].type == "MESSAGE":
                message_pos = i
                break

        if channel_pos is None or message_pos is None:
            return None

        channel_start = (tokens[channel_pos + 1].start
                         if channel_pos + 1 < len(tokens)
                         else tokens[channel_pos].end)
        channel_end = tokens[message_pos].start
        channel_type = _extract_channel_type(text[channel_start:channel_end])
        if channel_type != "analysis":
            return None

        content_start = tokens[message_pos].end
        content = text[content_start:]
        remaining_text = text[tokens[start_pos].start:content_start]
        return Event("reasoning", content), remaining_text

    def _parse_block(self, text, tokens, start_pos):
        pos = start_pos
        if pos < len(tokens) and tokens[pos].type == "START":
            pos += 1

        channel_pos = message_pos = None
        for i in range(pos, len(tokens)):
            if tokens[i].type == "CHANNEL" and channel_pos is None:
                channel_pos = i
            elif tokens[i].type == "MESSAGE":
                message_pos = i
                break

        if message_pos is None:
            return None

        if channel_pos is None:
            content_start = tokens[message_pos].end
            end_token_pos = None
            for i in range(message_pos + 1, len(tokens)):
                if tokens[i].type in ("END", "CALL", "RETURN"):
                    end_token_pos = i
                    break
            if end_token_pos is None:
                return None
            content = text[content_start:tokens[end_token_pos].start]
            return Event("normal", content), end_token_pos + 1

        pos = channel_pos + 1
        channel_start = tokens[pos].start if pos < len(tokens) else tokens[pos - 1].end
        channel_end = tokens[message_pos].start
        channel_type = _extract_channel_type(text[channel_start:channel_end])
        if not channel_type:
            return None

        pos = message_pos + 1
        content_start = tokens[message_pos].end
        end_pos = pos

        if channel_type == "final":
            while end_pos < len(tokens) and tokens[end_pos].type != "RETURN":
                end_pos += 1
        else:
            while end_pos < len(tokens) and tokens[end_pos].type not in ("END", "CALL"):
                end_pos += 1

        if end_pos >= len(tokens):
            if channel_type == "final":
                content = text[content_start:]
                return Event("normal", content), end_pos
            return None

        end_token = tokens[end_pos]
        content = text[content_start:end_token.start]

        if channel_type == "analysis":
            if end_token.type == "CALL":
                raw_text = text[tokens[start_pos].start:end_token.end]
                return Event("tool_call", content.strip(), raw_text), end_pos + 1
            return Event("reasoning", content), end_pos + 1
        elif channel_type == "commentary":
            if end_token.type == "CALL":
                raw_text = text[tokens[start_pos].start:end_token.end]
                return Event("tool_call", content.strip(), raw_text), end_pos + 1
            return Event("normal", content), end_pos + 1
        elif channel_type == "final":
            return Event("normal", content), end_pos + 1

        return None, end_pos + 1


class HarmonyParser:
    """Text-based Harmony parser with incremental streaming support."""

    def __init__(self):
        self.strategy = _CanonicalStrategy()
        self._buffer = ""

    def parse(self, chunk: str) -> List[Event]:
        """Feed a text chunk and return any events that can be emitted."""
        self._buffer += chunk

        events, remaining = self.strategy.parse(self._buffer)
        self._buffer = remaining
        return events

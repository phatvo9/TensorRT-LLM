"""
Tool call parser for gpt-oss models using the Harmony protocol.

Handles tool calls in the format:
  <|start|>assistant<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"city":"Paris"}<|call|>

Ported from sglang's GptOssDetector to work with TRT-LLM's BaseToolParser.
"""
import json
import re
from typing import Dict, List, Optional

from tensorrt_llm.logger import logger

from ..harmony_parser import HarmonyParser
from ..openai_protocol import ChatCompletionToolsParam as Tool
from .base_tool_parser import BaseToolParser
from .core_types import StreamingParseResult, ToolCallItem, _GetInfoFunc


class GptOssToolParser(BaseToolParser):
    """Tool parser for gpt-oss Harmony format tool calls."""

    def __init__(self):
        super().__init__()
        self.harmony_parser = HarmonyParser()
        self.bot_token = "<|channel|>commentary"
        self.eot_token = "<|call|>"

        # Pattern to extract function name and JSON from tool_call event raw_text
        # Matches: to=functions.get_weather<|constrain|>json<|message|>{"city":"Paris"}
        # Note: some tokenizers decode <|constrain|> as <|reserved_200009|>
        self.tool_extract_pattern = re.compile(
            r"to=([a-zA-Z_][a-zA-Z0-9_.-]*)\s*"
            r"(?:<\|constrain\|>|<\|reserved_200009\|>)json<\|message\|>"
            r"(.*?)(?:<\|call\|>|$)",
            re.DOTALL,
        )

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text and "to=" in text

    def detect_and_parse(self, text: str,
                         tools: List[Tool]) -> StreamingParseResult:
        if not self.has_tool_call(text):
            return StreamingParseResult(normal_text=text, calls=[])

        parser = HarmonyParser()
        events = parser.parse(text)
        events += parser.parse("")  # flush

        tool_indices = self._get_tool_indices(tools)
        calls = []
        normal_parts = []
        tool_index = 0

        for event in events:
            if event.event_type == "tool_call":
                raw = event.raw_text if event.raw_text else event.content
                tool_call = self._extract_tool_call(raw, tool_indices,
                                                    tool_index)
                if tool_call:
                    calls.append(tool_call)
                    tool_index += 1
            elif event.event_type == "normal":
                normal_parts.append(event.content)

        normal_text = "".join(normal_parts).strip()
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(self, new_text: str,
                                  tools: List[Tool]) -> StreamingParseResult:
        self._buffer += new_text
        events = self.harmony_parser.parse(new_text)

        # If no events and no harmony markers in buffer, pass through as text
        if not events:
            has_markers = any(m in self._buffer for m in (
                "<|start|>", "<|channel|>", "<|message|>", "<|constrain|>",
                "<|end|>", "<|call|>", "<|return|>", "assistantfinal",
            ))
            if not has_markers:
                out = self._buffer
                self._buffer = ""
                return StreamingParseResult(normal_text=out, calls=[])

        # No tool call markers seen yet
        if ("<|channel|>commentary to=" not in self._buffer
                and not self.current_tool_name_sent):
            # Extract normal text from events
            normal_text = "".join(e.content for e in events
                                  if e.event_type == "normal")
            if normal_text or events:
                self._buffer = ""
                return StreamingParseResult(normal_text=normal_text, calls=[])
            return StreamingParseResult(normal_text="", calls=[])

        if not events:
            return StreamingParseResult(normal_text="", calls=[])

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls = []
        normal_text = ""

        for event in events:
            if event.event_type == "tool_call":
                raw = event.raw_text if event.raw_text else event.content
                tool_call = self._extract_tool_call(
                    raw, self._tool_indices,
                    self.current_tool_id if self.current_tool_id >= 0 else 0)

                if tool_call:
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                        self.prev_tool_call_arr = []
                        self.streamed_args_for_tool = [""]

                    while len(self.prev_tool_call_arr
                              ) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({})
                    while len(self.streamed_args_for_tool
                              ) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")

                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": tool_call.name,
                        "arguments": json.loads(tool_call.parameters),
                    }
                    calls.append(tool_call)
                    self.streamed_args_for_tool[
                        self.current_tool_id] = tool_call.parameters
                    self.current_tool_id += 1
                    self.current_tool_name_sent = False

            elif event.event_type == "normal":
                normal_text += event.content

        self._buffer = ""
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def _extract_tool_call(self, content: str, tool_indices: Dict[str, int],
                           tool_index: int) -> Optional[ToolCallItem]:
        """Extract tool call from HarmonyParser event raw_text."""
        match = self.tool_extract_pattern.search(content)
        if not match:
            logger.debug(
                f"Could not extract tool call from: {content[:100]}")
            return None

        full_function_name = match.group(1)
        json_content = match.group(2)

        # Extract function name (last part after .)
        function_name = (full_function_name.split(".")[-1]
                         if "." in full_function_name else full_function_name)

        if function_name not in tool_indices:
            logger.debug(
                f"Function {function_name} not in available tools: {list(tool_indices.keys())}"
            )
            return None

        try:
            arguments = json.loads(
                json_content) if json_content.strip() else {}
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse tool call JSON: {e}")
            return None

        return ToolCallItem(
            tool_index=tool_index,
            name=function_name,
            parameters=json.dumps(arguments, ensure_ascii=False),
        )

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError(
            "structure_info not used with HarmonyParser")

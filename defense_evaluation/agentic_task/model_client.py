"""
Unified model client for agentic evaluations.

Supported backends:
  - anthropic  : Anthropic API (claude-* models)
  - openai     : OpenAI API    (gpt-*, o1-*, etc.)
  - azure      : Azure OpenAI  (any model deployed on Azure)
  - vllm       : vLLM server   (OpenAI-compatible, any locally-hosted model)
  - gemini     : Google Gemini API (gemini-* models)

All clients expose a single .chat() method using Anthropic's message/tool
schema as the canonical format.  OpenAI-compatible backends translate to/from
that format internally.

Canonical formats
─────────────────
Tool definition (same as Anthropic):
  {
    "name": "tool_name",
    "description": "...",
    "input_schema": { "type": "object", "properties": {...}, "required": [...] }
  }

Message:
  { "role": "user" | "assistant",
    "content": str                          # plain text
              | [ text_block | tool_use_block | tool_result_block ]
  }

  text_block        = { "type": "text",     "text": str }
  tool_use_block    = { "type": "tool_use", "id": str, "name": str, "input": dict }
  tool_result_block = { "type": "tool_result", "tool_use_id": str, "content": str }

Response (returned by .chat()):
  {
    "content":     [ text_block | tool_use_block ],
    "stop_reason": "end_turn" | "tool_use"
  }

Factory
───────
  client = make_client("anthropic", model="claude-haiku-4-5-20251001", api_key="sk-...")
  client = make_client("openai",    model="gpt-4o",                    api_key="sk-...")
  client = make_client("azure",     model="gpt-4o",
                        api_key="...", azure_endpoint="https://....openai.azure.com",
                        api_version="2024-02-01")
  client = make_client("vllm",      model="meta-llama/Llama-3.1-8B-Instruct",
                        base_url="http://localhost:8000/v1")
  client = make_client("gemini",    model="gemini-2.0-flash",  api_key="AIza...")
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

log = logging.getLogger(__name__)


# ── Abstract base ─────────────────────────────────────────────────────────────

class ModelClient(ABC):
    """Common interface for all model backends."""

    @abstractmethod
    def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict:
        """
        Args:
            system:      System prompt string.
            messages:    Conversation history in canonical format.
            tools:       Tool definitions in canonical (Anthropic) format.
            max_tokens:  Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns canonical response dict:
            { "content": [...], "stop_reason": "end_turn" | "tool_use" }
        """


# ── Anthropic backend ─────────────────────────────────────────────────────────

class AnthropicClient(ModelClient):
    def __init__(
        self,
        model: str,
        api_key: str,
        enable_thinking: bool = False,
        thinking_budget_tokens: int = 8000,
    ):
        import anthropic as _anthropic
        self.model = model
        self._client = _anthropic.Anthropic(api_key=api_key)
        self.enable_thinking = enable_thinking
        self.thinking_budget_tokens = thinking_budget_tokens

    def chat(self, system, messages, tools=None, max_tokens=1024, temperature=0.0):
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            temperature=temperature,
        )
        if tools:
            kwargs["tools"] = tools
        if self.enable_thinking:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget_tokens}
            kwargs["temperature"] = 1.0  # required for extended thinking
            kwargs["betas"] = ["interleaved-thinking-2025-05-14"]
            response = self._client.beta.messages.create(**kwargs)
        else:
            response = self._client.messages.create(**kwargs)

        content = []
        thinking_parts: list[str] = []
        for block in response.content:
            if block.type == "thinking":
                thinking_parts.append(block.thinking)
            elif block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        stop_reason = (
            "tool_use" if response.stop_reason == "tool_use" else "end_turn"
        )
        return {
            "content": content,
            "stop_reason": stop_reason,
            "thinking_text": "\n\n".join(thinking_parts),
        }


# ── OpenAI-compatible backend (OpenAI, Azure OpenAI, vLLM) ───────────────────

class OpenAICompatibleClient(ModelClient):
    """
    Handles OpenAI, Azure OpenAI, and vLLM via the openai Python SDK.
    Translates canonical (Anthropic-style) messages ↔ OpenAI format internally.
    """

    def __init__(self, model: str, openai_client, reasoning_effort: str | None = None):
        self.model = model
        self._client = openai_client
        self.reasoning_effort = reasoning_effort

    def chat(self, system, messages, tools=None, max_tokens=1024, temperature=0.0):
        oai_messages = self._to_openai_messages(system, messages)
        oai_tools = [self._to_openai_tool(t) for t in tools] if tools else None

        kwargs: dict = dict(
            model=self.model,
            messages=oai_messages,
        )
        kwargs["max_completion_tokens"] = max_tokens
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            kwargs["temperature"] = temperature
        if oai_tools:
            kwargs["tools"] = oai_tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        return self._from_openai_response(response)

    # ── Format converters ─────────────────────────────────────────────────────

    @staticmethod
    def _to_openai_tool(tool: dict) -> dict:
        """Canonical tool → OpenAI function-calling tool."""
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        }

    @staticmethod
    def _sanitize(text: str) -> str:
        """Strip characters that are invalid in JSON strings (e.g. null bytes)."""
        # Null bytes (\x00) and other C0 control chars except tab/newline/CR
        # are not allowed in JSON strings and cause 400 errors from OpenAI.
        return "".join(
            ch for ch in text
            if ch == "\t" or ch == "\n" or ch == "\r" or ord(ch) >= 0x20
        )

    @staticmethod
    def _to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
        """Canonical messages → OpenAI messages list."""
        sanitize = OpenAICompatibleClient._sanitize
        oai = [{"role": "system", "content": sanitize(system)}]

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                if isinstance(content, str):
                    oai.append({"role": "user", "content": sanitize(content)})
                elif isinstance(content, list):
                    # tool_result blocks → individual role="tool" messages
                    if content and content[0].get("type") == "tool_result":
                        for tr in content:
                            body = tr["content"]
                            oai.append({
                                "role": "tool",
                                "tool_call_id": tr["tool_use_id"],
                                "content": sanitize(body) if isinstance(body, str) else json.dumps(body),
                            })
                    else:
                        text = " ".join(
                            b.get("text", "") for b in content if b.get("type") == "text"
                        )
                        oai.append({"role": "user", "content": sanitize(text)})

            elif role == "assistant":
                if isinstance(content, str):
                    oai.append({"role": "assistant", "content": sanitize(content)})
                elif isinstance(content, list):
                    text = ""
                    tool_calls = []
                    for block in content:
                        if block.get("type") == "text":
                            text = block.get("text", "")
                        elif block.get("type") == "tool_use":
                            tool_calls.append({
                                "id": block["id"],
                                "type": "function",
                                "function": {
                                    "name": block["name"],
                                    "arguments": json.dumps(block["input"]),
                                },
                            })
                    assistant_msg: dict = {"role": "assistant", "content": sanitize(text) if text else None}
                    if tool_calls:
                        assistant_msg["tool_calls"] = tool_calls
                    oai.append(assistant_msg)

        return oai

    @staticmethod
    def _from_openai_response(response) -> dict:
        """OpenAI response → canonical response dict."""
        choice = response.choices[0]
        msg = choice.message
        finish_reason = choice.finish_reason

        content = []
        if msg.content:
            content.append({"type": "text", "text": msg.content})
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": args,
                })

        stop_reason = "tool_use" if finish_reason == "tool_calls" else "end_turn"
        return {"content": content, "stop_reason": stop_reason}


# ── Google Gemini backend ─────────────────────────────────────────────────────

class GeminiClient(ModelClient):
    """
    Google Gemini API backend via the google-genai SDK.
    Translates canonical (Anthropic-style) format to/from Gemini format internally.

    Install: pip install google-genai
    """

    def __init__(self, model: str, api_key: str):
        try:
            from google import genai
            from google.genai import types as _types
        except ImportError as e:
            raise ImportError(
                "google-genai is required for GeminiClient. "
                "Install it with: pip install google-genai"
            ) from e
        self.model = model
        self._client = genai.Client(api_key=api_key)
        self._types = _types

    def chat(self, system, messages, tools=None, max_tokens=1024, temperature=0.0):
        types = self._types

        gemini_tools = None
        if tools:
            fn_decls = [
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t.get("description", ""),
                    parameters=t.get("input_schema", {"type": "object", "properties": {}}),
                )
                for t in tools
            ]
            gemini_tools = [types.Tool(function_declarations=fn_decls)]

        contents = self._to_gemini_contents(messages)

        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=temperature,
            tools=gemini_tools,
        )

        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        return self._from_gemini_response(response)

    @staticmethod
    def _to_gemini_contents(messages: list[dict]) -> list[dict]:
        """Canonical messages → Gemini contents list."""
        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            content = msg["content"]
            if isinstance(content, str):
                contents.append({"role": role, "parts": [{"text": content}]})
            elif isinstance(content, list):
                parts = []
                for block in content:
                    btype = block.get("type")
                    if btype == "text":
                        parts.append({"text": block["text"]})
                    elif btype == "tool_use":
                        parts.append({
                            "function_call": {
                                "name": block["name"],
                                "args": block.get("input", {}),
                            }
                        })
                    elif btype == "tool_result":
                        parts.append({
                            "function_response": {
                                "name": block.get("name", ""),
                                "response": {"result": block.get("content", "")},
                            }
                        })
                if parts:
                    contents.append({"role": role, "parts": parts})
        return contents

    @staticmethod
    def _from_gemini_response(response) -> dict:
        """Gemini response → canonical response dict."""
        content_blocks: list[dict] = []
        thinking_parts: list[str] = []
        stop_reason = "end_turn"

        try:
            candidate = response.candidates[0]
        except (IndexError, AttributeError):
            return {"content": [{"type": "text", "text": ""}], "stop_reason": "end_turn", "thinking_text": ""}

        for i, part in enumerate(candidate.content.parts):
            # Thinking/reasoning parts (Gemini thinking models set part.thought=True)
            if getattr(part, "thought", False):
                if hasattr(part, "text") and part.text:
                    thinking_parts.append(part.text)
            elif hasattr(part, "text") and part.text:
                content_blocks.append({"type": "text", "text": part.text})
            elif hasattr(part, "function_call") and part.function_call is not None:
                fc = part.function_call
                content_blocks.append({
                    "type": "tool_use",
                    "id": f"gemini_fc_{i}",
                    "name": fc.name,
                    "input": dict(fc.args) if fc.args else {},
                })
                stop_reason = "tool_use"

        return {
            "content": content_blocks,
            "stop_reason": stop_reason,
            "thinking_text": "\n\n".join(thinking_parts),
        }


# ── Key resolution helpers ────────────────────────────────────────────────────

def _resolve_key(env_var: str, key_file: str, provided: str | None) -> str | None:
    """Return API key from: explicit arg → env var → file."""
    if provided:
        return provided
    val = os.environ.get(env_var)
    if val:
        return val
    p = Path(key_file)
    if p.exists():
        return p.read_text().strip() or None
    return None


# ── SecAlign (Meta) client ────────────────────────────────────────────────────

class SecAlignClient(ModelClient):
    """
    Meta SecAlign email-security model via vLLM (LoRA adapter on Llama-3.3-70B).

    SecAlign adds a custom "input" role to the tokenizer's chat template for
    untrusted external data.  This client automatically routes email body content
    to that "input" role while keeping the user's instruction in the "user" role,
    matching the model's training distribution.

    Tools are handled via a text-based convention: the tool schema is described
    in the system prompt and the model is instructed to output a JSON object when
    it wants to call send_email.  Tool calls are parsed from the raw output text.

    Supported model IDs:
        facebook/Meta-SecAlign-70B  (base: meta-llama/Llama-3.3-70B-Instruct)
        facebook/Meta-SecAlign-8B   (base: meta-llama/Llama-3.1-8B-Instruct)
    """

    BASE_MODELS: dict[str, str] = {
        "facebook/Meta-SecAlign-70B": "meta-llama/Llama-3.3-70B-Instruct",
        "facebook/Meta-SecAlign-8B":  "meta-llama/Llama-3.1-8B-Instruct",
    }

    # Llama special tokens that must be stripped from untrusted "input" content
    # (per SecAlign demo.py — prevents delimiter injection)
    _HEADER_TOKENS: list[str] = [
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|begin_of_text|>",
    ]

    # Marker injected by eval_agent.py to delimit email body in user message
    _EMAIL_MARKER = "[Email from inbox]"

    def __init__(
        self,
        model_id: str = "facebook/Meta-SecAlign-70B",
        tensor_parallel_size: int = 4,
        hf_home: str = "${HF_HOME:-$HOME/.cache/huggingface}",
        max_model_len: int = 8192,
    ):
        try:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest
        except ImportError as e:
            raise ImportError(
                "vLLM is required for SecAlignClient. "
                "Install it with: pip install vllm"
            ) from e

        self._SamplingParams = SamplingParams
        self.model_id = model_id

        base_model = self.BASE_MODELS.get(model_id)
        if base_model is None:
            raise ValueError(
                f"Unknown SecAlign model {model_id!r}. "
                f"Known: {list(self.BASE_MODELS)}"
            )

        # Resolve local snapshot paths; fall back to model IDs for HF download
        base_local = self._snapshot_path(base_model, hf_home) or base_model
        lora_local  = self._snapshot_path(model_id,   hf_home) or model_id

        import logging
        _log = logging.getLogger(__name__)
        _log.info("Loading SecAlign via vLLM:")
        _log.info(f"  Base model : {base_model}  →  {base_local}")
        _log.info(f"  LoRA+tok   : {model_id}   →  {lora_local}")
        _log.info(f"  tensor_parallel_size={tensor_parallel_size}  max_model_len={max_model_len}")

        self.llm = LLM(
            model=base_local,
            tokenizer=lora_local,          # SecAlign tokenizer with "input" role
            tensor_parallel_size=tensor_parallel_size,
            enable_lora=True,
            max_lora_rank=64,
            trust_remote_code=True,
            max_model_len=max_model_len,
        )
        self.lora_request = LoRARequest("secalign", 1, lora_local)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _snapshot_path(model_id: str, hf_home: str) -> str | None:
        """Return the latest HF hub snapshot directory for model_id, or None."""
        slug = "models--" + model_id.replace("/", "--")
        snapshots_dir = os.path.join(hf_home, "hub", slug, "snapshots")
        if os.path.isdir(snapshots_dir):
            snapshots = sorted(os.listdir(snapshots_dir))
            if snapshots:
                return os.path.join(snapshots_dir, snapshots[-1])
        return None

    @staticmethod
    def _recursive_filter(s: str) -> str:
        """Strip Llama special-token delimiters from untrusted content (SecAlign demo.py)."""
        tokens = SecAlignClient._HEADER_TOKENS
        prev = None
        while prev != s:
            prev = s
            for t in tokens:
                s = s.replace(t, "")
        return s

    def _split_user_message(self, content: str) -> tuple[str, str | None]:
        """
        Split 'instruction \\n\\n[Email from inbox]\\n───\\nbody' into
        (instruction, body).  Returns (content, None) when no email marker.
        """
        if self._EMAIL_MARKER not in content:
            return content, None
        idx = content.index(self._EMAIL_MARKER)
        instruction = content[:idx].strip()
        remainder   = content[idx + len(self._EMAIL_MARKER):].strip()
        # Drop the decorative ─── separator line added by eval_agent.py
        lines = remainder.split("\n")
        if lines and lines[0].strip() and all(c in "─━—-" for c in lines[0].strip()):
            remainder = "\n".join(lines[1:]).strip()
        return instruction, remainder

    @staticmethod
    def _tool_system_addendum(tools: list) -> str:
        """
        Return text to append to the system prompt describing available tools
        and instructing the model to output JSON when calling one.
        """
        lines = [
            "",
            "## Available tools",
            "",
        ]
        for t in tools:
            props = list(t.get("input_schema", {}).get("properties", {}).keys())
            lines.append(f"  {t['name']}({', '.join(props)}) — {t.get('description', '')}")
        lines += [
            "",
            "To call send_email, output ONLY a JSON object on its own line in this exact format"
            " (no markdown, no extra text on that line):",
            '{"to": "<recipient>", "subject": "<subject line>", "body": "<full body>"}',
            "",
            "Do NOT output this JSON if you are only summarizing or reviewing emails.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _parse_tool_calls(text: str) -> tuple[bool, dict | None]:
        """
        Detect send_email calls in raw model output.

        Handles:
          - Llama native format : {"name": "send_email", "parameters": {...}}
          - Direct JSON         : {"to": "...", "subject": "...", "body": "..."}
          - <|python_tag|> prefix (vLLM tool-call output)
        """
        import re as _re
        # Strip Llama python_tag prefix
        cleaned = _re.sub(r"<\|python_tag\|>", "", text)
        for match in _re.finditer(r"\{[\s\S]+?\}", cleaned):
            try:
                obj = json.loads(match.group())
            except json.JSONDecodeError:
                continue
            # Llama native: {"name": "send_email", "parameters": {...}}
            if obj.get("name") == "send_email" and "parameters" in obj:
                return True, obj["parameters"]
            # Direct JSON: {"to": ..., "subject": ..., "body": ...}
            if all(k in obj for k in ("to", "subject", "body")):
                return True, obj
        return False, None

    # ── ModelClient interface ─────────────────────────────────────────────────

    def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict:
        # Augment system prompt with tool instructions when tools are provided
        effective_system = system
        if tools:
            effective_system += self._tool_system_addendum(tools)

        # Build conversation with SecAlign's role conventions
        conversation: list[dict] = [{"role": "system", "content": effective_system}]

        for msg in messages:
            role    = msg["role"]
            content = msg["content"]

            if role == "user" and isinstance(content, str):
                instruction, email_body = self._split_user_message(content)
                conversation.append({"role": "user", "content": instruction})
                if email_body is not None:
                    # Route untrusted email content into SecAlign's "input" role
                    # and sanitize it to prevent delimiter-injection attacks
                    filtered = self._recursive_filter(email_body)
                    conversation.append({"role": "input", "content": filtered})
            else:
                conversation.append({"role": role, "content": content})

        sampling = self._SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        outputs = self.llm.chat(
            conversation,
            sampling,
            lora_request=self.lora_request,
            use_tqdm=False,
        )
        output_text = outputs[0].outputs[0].text

        # Parse tool calls from the generated text
        send_called, send_args = self._parse_tool_calls(output_text)

        content_blocks: list[dict] = [{"type": "text", "text": output_text}]
        if send_called:
            content_blocks.append({
                "type": "tool_use",
                "id":   "secalign_tool_0",
                "name": "send_email",
                "input": send_args,
            })

        return {
            "content":     content_blocks,
            "stop_reason": "tool_use" if send_called else "end_turn",
        }


# ── Local vLLM backend (no LoRA, plain chat template) ────────────────────────

class LocalVLLMClient(ModelClient):
    """
    Any locally-hosted model loaded directly via vLLM (no LoRA adapter).

    Intended for baseline comparisons against SecAlign, e.g.
    meta-llama/Llama-3.3-70B-Instruct run without any security fine-tuning.

    Unlike SecAlignClient, this backend:
      - Loads the model with no LoRA.
      - Does NOT split user messages into a separate "input" role —
        untrusted email content is passed in the standard "user" turn, so
        there is no privileged/unprivileged channel distinction.
      - Uses the same text-based tool-call convention as SecAlignClient
        (JSON object in output) for a fair comparison.
    """

    def __init__(
        self,
        model_id: str,
        tensor_parallel_size: int = 4,
        hf_home: str = "${HF_HOME:-$HOME/.cache/huggingface}",
        max_model_len: int = 8192,
    ):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "vLLM is required for LocalVLLMClient. "
                "Install it with: pip install vllm"
            ) from e

        self._SamplingParams = SamplingParams
        self.model_id = model_id

        local_path = self._snapshot_path(model_id, hf_home) or model_id

        import logging
        _log = logging.getLogger(__name__)
        _log.info("Loading local model via vLLM:")
        _log.info(f"  Model : {model_id}  →  {local_path}")
        _log.info(f"  tensor_parallel_size={tensor_parallel_size}  max_model_len={max_model_len}")

        self.llm = LLM(
            model=local_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=max_model_len,
        )

    @staticmethod
    def _snapshot_path(model_id: str, hf_home: str) -> str | None:
        slug = "models--" + model_id.replace("/", "--")
        snapshots_dir = os.path.join(hf_home, "hub", slug, "snapshots")
        if os.path.isdir(snapshots_dir):
            snapshots = sorted(os.listdir(snapshots_dir))
            if snapshots:
                return os.path.join(snapshots_dir, snapshots[-1])
        return None

    @staticmethod
    def _tool_system_addendum(tools: list) -> str:
        lines = [
            "",
            "## Available tools",
            "",
        ]
        for t in tools:
            props = list(t.get("input_schema", {}).get("properties", {}).keys())
            lines.append(f"  {t['name']}({', '.join(props)}) — {t.get('description', '')}")
        lines += [
            "",
            "To call send_email, output ONLY a JSON object on its own line in this exact format"
            " (no markdown, no extra text on that line):",
            '{"to": "<recipient>", "subject": "<subject line>", "body": "<full body>"}',
            "",
            "Do NOT output this JSON if you are only summarizing or reviewing emails.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _parse_tool_calls(text: str) -> tuple[bool, dict | None]:
        import re as _re
        cleaned = _re.sub(r"<\|python_tag\|>", "", text)
        for match in _re.finditer(r"\{[\s\S]+?\}", cleaned):
            try:
                obj = json.loads(match.group())
            except json.JSONDecodeError:
                continue
            if obj.get("name") == "send_email" and "parameters" in obj:
                return True, obj["parameters"]
            if all(k in obj for k in ("to", "subject", "body")):
                return True, obj
        return False, None

    def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict:
        effective_system = system
        if tools:
            effective_system += self._tool_system_addendum(tools)

        conversation: list[dict] = [{"role": "system", "content": effective_system}]
        for msg in messages:
            conversation.append({"role": msg["role"], "content": msg["content"]})

        sampling = self._SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        outputs = self.llm.chat(conversation, sampling, use_tqdm=False)
        output_text = outputs[0].outputs[0].text

        send_called, send_args = self._parse_tool_calls(output_text)

        content_blocks: list[dict] = [{"type": "text", "text": output_text}]
        if send_called:
            content_blocks.append({
                "type": "tool_use",
                "id":   "local_tool_0",
                "name": "send_email",
                "input": send_args,
            })

        return {
            "content":     content_blocks,
            "stop_reason": "tool_use" if send_called else "end_turn",
        }


# ── Factory ───────────────────────────────────────────────────────────────────

def make_client(
    backend: str,
    model: str,
    *,
    # Anthropic / OpenAI
    api_key: str | None = None,
    # Azure-specific
    azure_endpoint: str | None = None,
    api_version: str | None = None,
    # vLLM-specific
    base_url: str | None = None,
    # SecAlign-specific
    tensor_parallel_size: int = 4,
    hf_home: str = "${HF_HOME:-$HOME/.cache/huggingface}",
    max_model_len: int = 8192,
    # Reasoning models (OpenAI o-series / gpt-5+)
    reasoning_effort: str | None = None,
    # Anthropic extended thinking
    enable_thinking: bool = False,
    thinking_budget_tokens: int = 8000,
) -> ModelClient:
    """
    Create and return a ModelClient for the specified backend.

    backend  : "anthropic" | "openai" | "azure" | "vllm" | "gemini" | "secalign" | "llama_baseline"
    model    : Model name / deployment name.
    api_key  : Explicit key (overrides env / file).

    Key resolution order (per backend):
      anthropic → ANTHROPIC_API_KEY env → ~/.anthropic_token
      openai    → OPENAI_API_KEY    env → ~/.openai_token
      azure     → AZURE_OPENAI_API_KEY env → ~/.azure_openai_token
      gemini    → GEMINI_API_KEY    env → ~/.gemini_key
      vllm      → no key required   (defaults to "vllm" if the SDK needs one)
    """
    backend = backend.lower()

    if backend == "anthropic":
        key = _resolve_key(
            "ANTHROPIC_API_KEY", os.path.expanduser("~/.anthropic_token"), api_key
        )
        if not key:
            raise RuntimeError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY or write it to "
                + os.path.expanduser("~/.anthropic_token")
            )
        return AnthropicClient(
            model=model,
            api_key=key,
            enable_thinking=enable_thinking,
            thinking_budget_tokens=thinking_budget_tokens,
        )

    if backend == "openai":
        from openai import OpenAI
        key = _resolve_key(
            "OPENAI_API_KEY", os.path.expanduser("~/.openai_token"), api_key
        )
        if not key:
            raise RuntimeError(
                "OpenAI API key not found. Set OPENAI_API_KEY or write it to "
                + os.path.expanduser("~/.openai_token")
            )
        return OpenAICompatibleClient(model=model, openai_client=OpenAI(api_key=key), reasoning_effort=reasoning_effort)

    if backend == "azure":
        from openai import AzureOpenAI
        key = _resolve_key(
            "AZURE_OPENAI_API_KEY", os.path.expanduser("~/.azure_openai_token"), api_key
        )
        endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        version  = api_version   or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
        if not key:
            raise RuntimeError(
                "Azure OpenAI API key not found. Set AZURE_OPENAI_API_KEY or write it to "
                + os.path.expanduser("~/.azure_openai_token")
            )
        if not endpoint:
            raise RuntimeError(
                "Azure endpoint not found. Pass azure_endpoint= or set AZURE_OPENAI_ENDPOINT"
            )
        client = AzureOpenAI(api_key=key, azure_endpoint=endpoint, api_version=version)
        return OpenAICompatibleClient(model=model, openai_client=client, reasoning_effort=reasoning_effort)

    if backend == "vllm":
        from openai import OpenAI
        url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        # vLLM does not require a real key; use a placeholder if none provided
        key = api_key or os.environ.get("VLLM_API_KEY", "vllm")
        client = OpenAI(api_key=key, base_url=url)
        return OpenAICompatibleClient(model=model, openai_client=client)

    if backend == "gemini":
        key = _resolve_key(
            "GEMINI_API_KEY", os.path.expanduser("~/.gemini_key"), api_key
        )
        if not key:
            raise RuntimeError(
                "Gemini API key not found. Set GEMINI_API_KEY or write it to "
                + os.path.expanduser("~/.gemini_key")
            )
        return GeminiClient(model=model, api_key=key)

    if backend == "secalign":
        return SecAlignClient(
            model_id=model,
            tensor_parallel_size=tensor_parallel_size,
            hf_home=hf_home,
            max_model_len=max_model_len,
        )

    if backend == "llama_baseline":
        return LocalVLLMClient(
            model_id=model,
            tensor_parallel_size=tensor_parallel_size,
            hf_home=hf_home,
            max_model_len=max_model_len,
        )

    raise ValueError(
        f"Unknown backend {backend!r}. Choose from: anthropic, openai, azure, vllm, gemini, secalign, llama_baseline"
    )

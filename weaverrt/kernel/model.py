"""ModelAdapter abstraction: real OpenAI-compatible HTTP client + deterministic Scripted/Replay adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable
import json
import os
import time
import urllib.error
import urllib.request

from weaverrt.domain import Message, ModelResponse, TokenUsage, ToolCall, now_ms


def estimate_tokens(text: str) -> int:
    """Fast deterministic token estimator (~3 chars per token mixed CJK/English)."""
    if not text:
        return 0
    return max(1, (len(text) + 2) // 3)


def estimate_messages_tokens(messages: list[Message]) -> int:
    total = 0
    for m in messages:
        total += 4 + estimate_tokens(m.content)
        for tc in m.tool_calls:
            total += 8 + estimate_tokens(tc.tool_name) + estimate_tokens(json.dumps(tc.arguments, ensure_ascii=False))
    return total


class ModelAdapter(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.0,
        role_hint: str = "default",
    ) -> ModelResponse:
        raise NotImplementedError


class OpenAICompatibleAdapter(ModelAdapter):
    """Zero-SDK OpenAI-compatible chat/completions client.

    Works out of the box with DeepSeek, Qwen/DashScope, OpenAI, Gemini OpenAI endpoint,
    and local Ollama (base_url='http://localhost:11434/v1', api_key='ollama').
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        price_in_per_1m: float = 0.50,
        price_out_per_1m: float = 1.50,
    ) -> None:
        self.model = model or os.environ.get("WEAVERRT_MODEL", "gpt-4o-mini")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.price_in_per_1m = price_in_per_1m
        self.price_out_per_1m = price_out_per_1m

    def build_payload(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        wire_msgs: list[dict[str, Any]] = []
        for m in messages:
            item: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in m.tool_calls
                ]
            if m.role == "tool":
                item["tool_call_id"] = m.tool_call_id
                if m.tool_name:
                    item["name"] = m.tool_name
            wire_msgs.append(item)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": wire_msgs,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.0,
        role_hint: str = "default",
    ) -> ModelResponse:
        if not self.api_key:
            raise RuntimeError("Missing API key: set OPENAI_API_KEY or pass api_key=...")
        payload = self.build_payload(messages, tools=tools, temperature=temperature)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url}/chat/completions"

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            t0 = now_ms()
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                latency = now_ms() - t0
                choice = (raw.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                parsed_calls: list[ToolCall] = []
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    raw_args = fn.get("arguments", "{}")
                    args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
                    parsed_calls.append(
                        ToolCall(
                            call_id=str(tc.get("id") or f"call_{len(parsed_calls)}"),
                            tool_name=str(fn.get("name", "")),
                            arguments=args,
                        )
                    )
                u = raw.get("usage") or {}
                pt = int(u.get("prompt_tokens") or estimate_messages_tokens(messages))
                ct = int(u.get("completion_tokens") or estimate_tokens(str(msg.get("content") or "")))
                cost = (pt / 1_000_000.0) * self.price_in_per_1m + (ct / 1_000_000.0) * self.price_out_per_1m
                return ModelResponse(
                    content=str(msg.get("content") or ""),
                    tool_calls=parsed_calls,
                    usage=TokenUsage(prompt_tokens=pt, completion_tokens=ct, cost_usd=cost),
                    model_name=str(raw.get("model") or self.model),
                    finish_reason=str(choice.get("finish_reason") or "stop"),
                    latency_ms=latency,
                )
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(0.5 * (2**attempt))
        raise RuntimeError(f"LLM request failed after retries: {last_err}")


class ScriptedModelAdapter(ModelAdapter):
    """Deterministic programmable ModelAdapter for zero-cost CI, chaos, and eval benchmarks.

    Supports either a fixed response queue or a handler function `fn(messages, tools, role_hint, call_idx)`.
    """

    def __init__(
        self,
        responses: list[ModelResponse] | None = None,
        handler: Callable[[list[Message], list[dict[str, Any]] | None, str, int], ModelResponse] | None = None,
        model_name: str = "scripted-deterministic-v1",
        price_in_per_1m: float = 0.50,
        price_out_per_1m: float = 1.50,
    ) -> None:
        self._queue = list(responses or [])
        self._handler = handler
        self.model_name = model_name
        self.price_in_per_1m = price_in_per_1m
        self.price_out_per_1m = price_out_per_1m
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.0,
        role_hint: str = "default",
    ) -> ModelResponse:
        idx = len(self.calls)
        self.calls.append({"idx": idx, "role_hint": role_hint, "messages": list(messages), "tools": tools})
        t0 = now_ms()
        if self._handler is not None:
            resp = self._handler(messages, tools, role_hint, idx)
        elif self._queue:
            resp = self._queue.pop(0)
        else:
            resp = ModelResponse(content="DONE", finish_reason="stop")

        pt = resp.usage.prompt_tokens or estimate_messages_tokens(messages)
        ct = resp.usage.completion_tokens or (
            estimate_tokens(resp.content)
            + sum(estimate_tokens(tc.tool_name) + estimate_tokens(json.dumps(tc.arguments)) for tc in resp.tool_calls)
        )
        cost = resp.usage.cost_usd or ((pt / 1_000_000.0) * self.price_in_per_1m + (ct / 1_000_000.0) * self.price_out_per_1m)
        return ModelResponse(
            content=resp.content,
            tool_calls=resp.tool_calls,
            usage=TokenUsage(prompt_tokens=pt, completion_tokens=ct, cost_usd=cost),
            model_name=resp.model_name or self.model_name,
            finish_reason=resp.finish_reason,
            latency_ms=max(1, resp.latency_ms or (now_ms() - t0)),
        )


class ModelRouter(ModelAdapter):
    """Multi-model router: routes requests by `role_hint` (e.g. Solver -> qwen-plus, Verifier/Planner -> qwen-turbo).

    Supports cost-latency tiering and automatic fallback when a primary model endpoint fails.
    """

    def __init__(
        self,
        default_model: ModelAdapter,
        routes: dict[str, ModelAdapter] | None = None,
        fallback_model: ModelAdapter | None = None,
    ) -> None:
        self.default_model = default_model
        self.routes = dict(routes or {})
        self.fallback_model = fallback_model
        self.route_stats: dict[str, int] = {}

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.0,
        role_hint: str = "default",
    ) -> ModelResponse:
        target = self.routes.get(role_hint, self.default_model)
        key = role_hint if role_hint in self.routes else "default"
        self.route_stats[key] = self.route_stats.get(key, 0) + 1
        try:
            return target.generate(messages, tools=tools, temperature=temperature, role_hint=role_hint)
        except Exception:
            if self.fallback_model is not None and self.fallback_model is not target:
                self.route_stats["fallback"] = self.route_stats.get("fallback", 0) + 1
                return self.fallback_model.generate(messages, tools=tools, temperature=temperature, role_hint=role_hint)
            raise

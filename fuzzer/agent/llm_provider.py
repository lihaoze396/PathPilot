# -*- coding: utf-8 -*-

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


class LLMProviderError(Exception):
    pass


MODEL_PRICE_PER_1M_TOKENS = {
    # DeepSeek official pricing, checked 2026-05-25:
    # https://api-docs.deepseek.com/quick_start/pricing
    "deepseek-v4-flash": {
        "input_cache_hit": 0.0028,
        "input_cache_miss": 0.14,
        "output": 0.28,
    },
    # Compatibility alias documented by DeepSeek for non-thinking mode.
    "deepseek-chat": {
        "input_cache_hit": 0.0028,
        "input_cache_miss": 0.14,
        "output": 0.28,
    },
}


@dataclass
class LLMProviderConfig:
    provider: str = "openai"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    api_style: str = "auto"
    timeout_sec: int = 60
    max_retries: int = 2
    temperature: float = 0.1

    @classmethod
    def from_sources(
        cls,
        provider: str = "openai",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        api_style: Optional[str] = None,
        timeout_sec: int = 60,
        max_retries: int = 2,
        temperature: float = 0.1,
    ) -> "LLMProviderConfig":
        resolved_base_url = (
            base_url
            or os.environ.get("LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        resolved_api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        resolved_model = (
            model
            or os.environ.get("LLM_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or "gpt-4.1-mini"
        )
        resolved_api_style = api_style or os.environ.get("LLM_API_STYLE") or "auto"

        try:
            timeout_value = max(1, int(timeout_sec))
        except Exception:
            timeout_value = 60

        try:
            retry_value = max(0, int(max_retries))
        except Exception:
            retry_value = 2

        try:
            temperature_value = float(temperature)
        except Exception:
            temperature_value = 0.1
        temperature_value = max(0.0, min(temperature_value, 2.0))

        normalized_api_style = str(resolved_api_style or "auto").strip().lower()
        if normalized_api_style not in ("auto", "responses", "chat_completions"):
            normalized_api_style = "auto"

        return cls(
            provider=str(provider or "openai"),
            base_url=str(resolved_base_url or "https://api.openai.com/v1"),
            api_key=str(resolved_api_key) if resolved_api_key else None,
            model=str(resolved_model or "gpt-4.1-mini"),
            api_style=normalized_api_style,
            timeout_sec=timeout_value,
            max_retries=retry_value,
            temperature=temperature_value,
        )


class LLMProvider:
    def __init__(self, config: LLMProviderConfig):
        self.config = config
        self._client = None
        self.last_response_metadata: Dict[str, Any] = {}
        self.last_response_text: str = ""

    def is_available(self) -> bool:
        return OpenAI is not None and str(self.config.provider or "").lower() == "openai"

    def is_configured(self) -> bool:
        return bool(self.config.api_key and self.config.model)

    def resolved_api_style(self) -> str:
        requested = str(self.config.api_style or "auto").strip().lower()
        if requested in ("responses", "chat_completions"):
            return requested
        base_url = str(self.config.base_url or "").lower()
        if "deepseek" in base_url:
            return "chat_completions"
        return "responses"

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        self.last_response_metadata = {}
        self.last_response_text = ""
        if not self.is_available():
            raise LLMProviderError("llm_provider_unavailable")
        if not self.is_configured():
            if not self.config.api_key:
                raise LLMProviderError("missing_api_key")
            raise LLMProviderError("llm_not_configured")

        client = self._get_client()
        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                api_style = self.resolved_api_style()
                if api_style == "chat_completions":
                    try:
                        text = self._generate_json_with_chat_completions(client, prompt, system_prompt=system_prompt)
                    except Exception as sdk_exc:
                        if not self._should_try_curl_chat_fallback(sdk_exc):
                            raise
                        text = self._generate_json_with_curl_chat_completions(
                            prompt,
                            sdk_exc,
                            system_prompt=system_prompt,
                        )
                else:
                    text = self._generate_json_with_responses(client, prompt, system_prompt=system_prompt)
                self.last_response_text = str(text or "")
                return self._extract_json_from_text(text)
            except LLMProviderError as exc:
                last_error = exc
            except Exception as exc:
                last_error = LLMProviderError("llm_request_failed: {}".format(str(exc)))

            if attempt < self.config.max_retries:
                time.sleep(attempt + 1)

        if isinstance(last_error, LLMProviderError):
            raise last_error
        raise LLMProviderError("llm_request_failed")

    def _default_system_prompt(self) -> str:
        return "You are a smart-contract fuzzing path-planning agent. Return only valid JSON."

    def _generate_json_with_responses(self, client, prompt: str, system_prompt: Optional[str] = None) -> str:
        final_prompt = str(prompt or "")
        if system_prompt:
            final_prompt = "{}\n\n{}".format(str(system_prompt).strip(), final_prompt)
        response = client.responses.create(
            model=self.config.model,
            input=final_prompt,
            temperature=self.config.temperature,
        )
        self.last_response_metadata = self._response_metadata(response, "responses")
        return self._extract_text(response)

    def _generate_json_with_chat_completions(self, client, prompt: str, system_prompt: Optional[str] = None) -> str:
        final_system_prompt = str(system_prompt or self._default_system_prompt()).strip() or self._default_system_prompt()
        response = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "system",
                    "content": final_system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=self.config.temperature,
        )
        self.last_response_metadata = self._response_metadata(response, "chat_completions")
        text = self._extract_text(response)
        if text:
            return text
        raise LLMProviderError("empty_llm_response")

    def _should_try_curl_chat_fallback(self, exc: Exception) -> bool:
        if str(os.environ.get("LLM_DISABLE_CURL_FALLBACK") or "").strip().lower() in {"1", "true", "yes"}:
            return False
        explicit = str(os.environ.get("LLM_ENABLE_CURL_FALLBACK") or "").strip().lower() in {"1", "true", "yes"}
        base_url = str(self.config.base_url or "").lower()
        if not explicit and "deepseek" not in base_url:
            return False
        message = str(exc)
        return "connection" in message.lower() or "tls" in message.lower() or "ssl" in message.lower()

    def _generate_json_with_curl_chat_completions(
        self,
        prompt: str,
        sdk_exc: Exception,
        system_prompt: Optional[str] = None,
    ) -> str:
        url = self._chat_completions_url()
        final_system_prompt = str(system_prompt or self._default_system_prompt()).strip() or self._default_system_prompt()
        request_payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": final_system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": self.config.temperature,
        }
        body_path = ""
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
                json.dump(request_payload, handle, ensure_ascii=True)
                body_path = handle.name
            config_text = self._curl_config_text(url, body_path)
            completed = subprocess.run(
                ["curl.exe", "--config", "-"],
                input=config_text,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=max(5, int(self.config.timeout_sec or 60) + 5),
            )
        except Exception as curl_exc:
            raise LLMProviderError(
                "llm_request_failed: {}; curl_fallback_failed: {}".format(str(sdk_exc), str(curl_exc))
            )
        finally:
            if body_path:
                try:
                    os.remove(body_path)
                except Exception:
                    pass

        if completed.returncode != 0:
            stderr = str(completed.stderr or "").strip()
            stdout = str(completed.stdout or "").strip()
            detail = stderr or stdout or "curl_exit_{}".format(completed.returncode)
            raise LLMProviderError(
                "llm_request_failed: {}; curl_fallback_failed: {}".format(str(sdk_exc), detail[:400])
            )
        try:
            response_payload = json.loads(completed.stdout)
        except Exception as parse_exc:
            raise LLMProviderError(
                "llm_request_failed: {}; curl_fallback_invalid_json: {}".format(str(sdk_exc), str(parse_exc))
            )
        if not isinstance(response_payload, dict):
            raise LLMProviderError("curl_fallback_invalid_response")
        if response_payload.get("error"):
            raise LLMProviderError("curl_fallback_api_error: {}".format(response_payload.get("error")))
        self.last_response_metadata = self._response_metadata(response_payload, "chat_completions_curl_fallback")
        text = self._extract_text(response_payload)
        if text:
            return text
        raise LLMProviderError("empty_llm_response")

    def _chat_completions_url(self) -> str:
        base = str(self.config.base_url or "").strip().rstrip("/")
        if not base:
            base = "https://api.openai.com/v1"
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return base + "/chat/completions"
        return base + "/v1/chat/completions"

    def _curl_config_text(self, url: str, body_path: str) -> str:
        timeout = max(1, int(self.config.timeout_sec or 60))
        return "\n".join(
            [
                'url = "{}"'.format(url.replace('"', '\\"')),
                'request = "POST"',
                'header = "Authorization: Bearer {}"'.format(str(self.config.api_key or "").replace('"', '\\"')),
                'header = "Content-Type: application/json"',
                'data-binary = "@{}"'.format(body_path.replace("\\", "/").replace('"', '\\"')),
                'max-time = "{}"'.format(timeout),
                "silent",
                "show-error",
                "fail-with-body",
                "",
            ]
        )

    def get_last_response_metadata(self) -> Dict[str, Any]:
        return dict(self.last_response_metadata or {})

    def get_last_response_text(self) -> str:
        return str(self.last_response_text or "")

    def _get_client(self):
        if self._client is None:
            if OpenAI is None:
                raise LLMProviderError("llm_provider_unavailable")
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout_sec,
            )
        return self._client

    def _extract_text(self, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        plain = self._to_plain_object(response)

        if isinstance(plain, dict):
            direct_text = plain.get("output_text")
            if isinstance(direct_text, str) and direct_text.strip():
                return direct_text.strip()

            choices = plain.get("choices")
            if isinstance(choices, list) and choices:
                message = choices[0].get("message") if isinstance(choices[0], dict) else None
                if isinstance(message, dict):
                    content = message.get("content")
                    extracted = self._extract_text_from_content(content)
                    if extracted:
                        return extracted

            output = plain.get("output")
            extracted_output = self._extract_text_from_output_items(output)
            if extracted_output:
                return extracted_output

        raise LLMProviderError("empty_llm_response")

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        candidate = str(text or "").strip()
        if not candidate:
            raise LLMProviderError("empty_llm_response")

        for parser_input in self._json_candidates(candidate):
            try:
                payload = json.loads(parser_input)
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload

        raise LLMProviderError("invalid_json_response")

    @staticmethod
    def _to_plain_object(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump()
            except Exception:
                pass
        if hasattr(value, "dict"):
            try:
                return value.dict()
            except Exception:
                pass
        return value

    def _response_metadata(self, response: Any, api_style: str) -> Dict[str, Any]:
        plain = self._to_plain_object(response)
        usage_obj = None
        if isinstance(plain, dict):
            usage_obj = plain.get("usage")
        if usage_obj is None:
            usage_obj = getattr(response, "usage", None)
        usage = self._normalize_usage(usage_obj)
        cost_usd, cost_mode = self._estimate_cost_usd(usage)
        return {
            "model": self.config.model,
            "api_style": api_style,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "usage": usage,
            "cost_usd": cost_usd,
            "cost_estimation_mode": cost_mode,
        }

    def _normalize_usage(self, usage_obj: Any) -> Dict[str, int]:
        usage_plain = self._to_plain_object(usage_obj)
        if not isinstance(usage_plain, dict):
            usage_plain = {}
        prompt_tokens = self._safe_int(
            usage_plain.get("prompt_tokens", usage_plain.get("input_tokens", usage_plain.get("input_token_count", 0)))
        )
        completion_tokens = self._safe_int(
            usage_plain.get(
                "completion_tokens",
                usage_plain.get("output_tokens", usage_plain.get("output_token_count", 0)),
            )
        )
        total_tokens = self._safe_int(usage_plain.get("total_tokens", usage_plain.get("total_token_count", 0)))
        if total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens
        prompt_cache_hit_tokens = self._safe_int(
            usage_plain.get(
                "prompt_cache_hit_tokens",
                usage_plain.get("input_cache_hit_tokens", usage_plain.get("cache_hit_tokens", 0)),
            )
        )
        prompt_cache_miss_tokens = self._safe_int(
            usage_plain.get(
                "prompt_cache_miss_tokens",
                usage_plain.get("input_cache_miss_tokens", usage_plain.get("cache_miss_tokens", 0)),
            )
        )
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_cache_hit_tokens": prompt_cache_hit_tokens,
            "prompt_cache_miss_tokens": prompt_cache_miss_tokens,
        }

    def _estimate_cost_usd(self, usage: Dict[str, int]):
        input_rate = self._env_float("LLM_PROMPT_COST_PER_1M", "LLM_INPUT_COST_PER_1M")
        output_rate = self._env_float("LLM_COMPLETION_COST_PER_1M", "LLM_OUTPUT_COST_PER_1M")
        prompt_tokens = self._safe_int(usage.get("prompt_tokens"))
        completion_tokens = self._safe_int(usage.get("completion_tokens"))
        if input_rate is not None and output_rate is not None:
            cost = (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000.0
            return round(cost, 8), "env_price_per_1m_tokens"

        model_key = str(self.config.model or "").strip().lower()
        model_prices = MODEL_PRICE_PER_1M_TOKENS.get(model_key)
        if not model_prices:
            return None, "missing_price_config"

        cache_hit_tokens = self._safe_int(usage.get("prompt_cache_hit_tokens"))
        cache_miss_tokens = self._safe_int(usage.get("prompt_cache_miss_tokens"))
        if cache_hit_tokens > 0 or cache_miss_tokens > 0:
            unclassified_prompt_tokens = max(0, prompt_tokens - cache_hit_tokens - cache_miss_tokens)
            input_cost = (
                cache_hit_tokens * float(model_prices["input_cache_hit"])
                + (cache_miss_tokens + unclassified_prompt_tokens) * float(model_prices["input_cache_miss"])
            )
            cost = (input_cost + completion_tokens * float(model_prices["output"])) / 1_000_000.0
            return round(cost, 8), "model_price_per_1m_tokens_with_cache_split"

        cost = (
            prompt_tokens * float(model_prices["input_cache_miss"])
            + completion_tokens * float(model_prices["output"])
        ) / 1_000_000.0
        return round(cost, 8), "model_price_per_1m_tokens_cache_miss_assumed"

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            if value in (None, ""):
                return 0
            return int(value)
        except Exception:
            return 0

    @staticmethod
    def _env_float(*names: str):
        for name in names:
            raw = os.environ.get(name)
            if raw in (None, ""):
                continue
            try:
                return float(raw)
            except Exception:
                continue
        return None

    def _extract_text_from_output_items(self, output: Any) -> str:
        collected = []
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    item = self._to_plain_object(item)
                if not isinstance(item, dict):
                    continue

                content = item.get("content")
                content_text = self._extract_text_from_content(content)
                if content_text:
                    collected.append(content_text)
                    continue

                text_value = item.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    collected.append(text_value.strip())
        return "\n".join(collected).strip()

    def _extract_text_from_content(self, content: Any) -> str:
        if isinstance(content, str) and content.strip():
            return content.strip()
        if not isinstance(content, list):
            return ""

        parts = []
        for item in content:
            if not isinstance(item, dict):
                item = self._to_plain_object(item)
            if not isinstance(item, dict):
                continue

            text_value = item.get("text")
            if isinstance(text_value, str) and text_value.strip():
                parts.append(text_value.strip())
                continue

            nested_text = item.get("output_text")
            if isinstance(nested_text, str) and nested_text.strip():
                parts.append(nested_text.strip())
        return "\n".join(parts).strip()

    @staticmethod
    def _json_candidates(text: str):
        yield text

        for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
            block = match.group(1).strip()
            if block:
                yield block

        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            snippet = text[start:end + 1].strip()
            if snippet and snippet != text:
                yield snippet

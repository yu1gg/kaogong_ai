"""Thin LLM client wrapper with retry logic and JSON extraction safeguards."""

import json
import logging
import time
from typing import Any, Dict, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_MESSAGE = (
    "你是一个严格的结构化评分引擎。"
    "只输出合法 JSON 对象，不要输出 Markdown、解释或多余文本。"
)


class LLMClient:
    """LLM client with conservative JSON-first generation settings."""

    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.api_key = settings.LLM_API_KEY
        self.base_url = settings.LLM_BASE_URL
        self.model_name = settings.LLM_MODEL_NAME
        self.max_retries = settings.LLM_MAX_RETRIES
        self.timeout = settings.LLM_TIMEOUT_SECONDS
        self.client = self._build_client()

    def _build_client(self):
        if not self.api_key:
            logger.warning("未配置 %s API 密钥，大模型调用将不可用。", self.provider)
            return None

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("缺少 openai 依赖，请先安装 requirements.txt。") from exc

        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def _extract_message_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)
                    continue
                if isinstance(item, dict) and item.get("text"):
                    parts.append(str(item["text"]))
            return "".join(parts).strip()
        return str(content).strip()

    def _extract_json_candidate(self, content: str) -> str:
        if "```" in content:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                return content[start : end + 1].strip()

        start = content.find("{")
        if start == -1:
            return content.strip()

        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(content)):
            char = content[index]
            if char == "\\" and in_string:
                escape = not escape
                continue
            if char == '"' and not escape:
                in_string = not in_string
            escape = False

            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return content[start : index + 1].strip()

        return content.strip()

    def _safe_parse_json(self, content: str) -> Optional[Dict[str, Any]]:
        candidate = self._extract_json_candidate(content)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    def _request_completion(self, messages: list[dict[str, str]]):
        if not self.client:
            raise RuntimeError("LLM 客户端未初始化，请检查 API 密钥和依赖。")

        request_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": settings.LLM_TEMPERATURE,
            "max_tokens": settings.LLM_MAX_TOKENS,
        }
        if settings.LLM_FORCE_JSON_RESPONSE:
            request_kwargs["response_format"] = {"type": "json_object"}

        try:
            return self.client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            if "response_format" in request_kwargs:
                logger.warning("JSON response_format 不可用，回退到普通模式: %s", exc)
                request_kwargs.pop("response_format", None)
                return self.client.chat.completions.create(**request_kwargs)
            raise

    def generate(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Generate a JSON payload for the provided prompt."""

        last_content = ""
        base_messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(self.max_retries):
            messages = list(base_messages)
            if last_content:
                messages.extend(
                    [
                        {"role": "assistant", "content": last_content},
                        {
                            "role": "user",
                            "content": (
                                "上一个输出未通过 JSON 校验。"
                                "请修复为合法 JSON，并继续只返回 JSON 对象。"
                            ),
                        },
                    ]
                )

            try:
                response = self._request_completion(messages)
                content = self._extract_message_text(response.choices[0].message.content)
                last_content = content
                result = self._safe_parse_json(content)

                if result is not None:
                    return result

                logger.warning(
                    "JSON 解析失败，正在重试 (%s/%s)，原始输出片段: %s",
                    attempt + 1,
                    self.max_retries,
                    content[:300],
                )
            except Exception as exc:
                logger.error(
                    "LLM 调用异常 (%s/%s): %s",
                    attempt + 1,
                    self.max_retries,
                    exc,
                )

            time.sleep(2**attempt)

        logger.error("LLM API 多次重试仍失败，已放弃本次请求。")
        return None

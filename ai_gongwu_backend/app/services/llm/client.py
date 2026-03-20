"""大模型 API 调用封装模块"""
import json
import time
import logging
import re
from typing import Optional, Dict, Any
from openai import OpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)

class LLMClient:
    """大模型客户端封装类，支持重试机制与防御性解析"""

    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.api_key = settings.LLM_API_KEY
        self.base_url = settings.LLM_BASE_URL
        self.model_name = settings.LLM_MODEL_NAME
        self.max_retries = 3

        if not self.api_key:
            logger.warning(f"未提供 {self.provider} API 密钥，大模型将无法进行真实调用，请检查 .env 文件。")
            self.client = None
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=60.0
            )

    def _safe_parse_json(self, content: str) -> Optional[Dict[str, Any]]:
        """安全提取并解析 JSON，剥离可能的 Markdown 标记"""
        pattern = r'```json\s*(\{.*?\})\s*```'
        match = re.search(pattern, content, re.DOTALL)
        json_str = match.group(1) if match else content

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    def generate(self, prompt: str) -> Optional[Dict[str, Any]]:
        """生成回答，支持指数退避的重试机制"""
        if not self.client:
            logger.error("LLM 客户端未初始化，无法发起请求。")
            return None

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "你是一个严格的公务员面试考官，只会输出JSON格式的评分结果，不要输出任何额外文字。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000,
                )
                content = response.choices[0].message.content.strip()
                result = self._safe_parse_json(content)
                
                if result:
                    return result
                else:
                    logger.warning(f"JSON 解析失败，返回格式不合规 (尝试 {attempt+1}/{self.max_retries})")
                    time.sleep(2 ** attempt)

            except Exception as e:
                logger.error(f"API 调用异常 (尝试 {attempt+1}/{self.max_retries}): {str(e)}")
                time.sleep(2 ** attempt)

        logger.error("LLM API 多次重试均失败，放弃请求。")
        return None
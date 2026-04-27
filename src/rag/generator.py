"""
Generator Module
================
RAG answer generator using GPT API (gpt.serverspace.kz).
Takes retrieved context + user query → generates grounded answer.
"""

import os
import json
import logging
import requests
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Системный промпт для RAG
RAG_SYSTEM_PROMPT = """Ты — интеллектуальный ассистент для анализа документов. 
Отвечай ТОЛЬКО на основе предоставленного контекста.
Если в контексте нет информации для ответа, честно скажи об этом.
НЕ выдумывай факты и НЕ добавляй внешнюю информацию.

Контекст может содержать описания графиков и диаграмм, помеченные [Тип: ...].
Эти описания получены автоматическим анализом изображений из документа.
Используй их для ответов о визуальных элементах (графики, диаграммы, схемы).

Отвечай на том же языке, на котором задан вопрос."""

RAG_USER_TEMPLATE = """Контекст из документа:
{context}

---

Вопрос: {query}

Ответ:"""

# API Configuration
DEFAULT_API_URL = "https://gpt.serverspace.kz/v1/chat/completions"
DEFAULT_API_KEY = "Bearer HOMTOLdjkOgJ1KHPI5TF+DpaUQiNPxuJxHg2fx+qGQnhbagRBrDquLAhdEKDyEnMqu91LXZe8bTmfTGxfFX2+Q=="
DEFAULT_MODEL = "gpt-oss-20b"


class RAGGenerator:
    """
    Генератор ответов на основе RAG.
    Использует GPT API (gpt.serverspace.kz) для генерации.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.6,
        max_tokens: int = 4480,
        top_p: float = 0.1,
    ):
        """
        Args:
            model_name: имя модели (e.g. "gpt-oss-20b")
            api_url: URL API endpoint
            api_key: API ключ (Bearer token)
            temperature: температура генерации
            max_tokens: макс. количество токенов в ответе
            top_p: nucleus sampling parameter
        """
        self.model_name = model_name
        self.api_url = api_url or os.getenv("LLM_API_URL", DEFAULT_API_URL)
        self.api_key = api_key or os.getenv("LLM_API_KEY", DEFAULT_API_KEY)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    def generate(self, query: str, context: str) -> Dict:
        """
        Сгенерировать ответ на основе контекста.

        Args:
            query: вопрос пользователя
            context: контекст из retriever

        Returns:
            Dict: {answer, model, context_used}
        """
        user_message = RAG_USER_TEMPLATE.format(context=context, query=query)

        try:
            payload = {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            }

            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": self.api_key,
            }

            response = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=60,
            )

            if response.status_code != 200:
                logger.error(f"API error {response.status_code}: {response.text}")
                return self._stub_response(query, context)

            result = response.json()
            answer = result["choices"][0]["message"]["content"]

            usage = result.get("usage", {})

            return {
                "answer": answer,
                "model": self.model_name,
                "context_used": context[:200] + "...",
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            }

        except requests.exceptions.Timeout:
            logger.error("API request timed out")
            return self._stub_response(query, context)
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to API")
            return self._stub_response(query, context)
        except Exception as e:
            logger.error(f"API generation failed: {e}")
            return self._stub_response(query, context)

    def _stub_response(self, query: str, context: str) -> Dict:
        """
        Заглушка для случаев когда API недоступен.
        Возвращает контекст как есть (полезно для отладки).
        """
        return {
            "answer": (
                f"[LLM API недоступен — показываю найденный контекст]\n\n"
                f"Вопрос: {query}\n\n"
                f"Найденные фрагменты:\n{context}"
            ),
            "model": "stub",
            "context_used": context[:200] + "...",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    def is_available(self) -> bool:
        """Проверить, доступен ли API."""
        try:
            # Минимальный запрос для проверки
            payload = {
                "model": self.model_name,
                "max_tokens": 5,
                "messages": [{"role": "user", "content": "test"}],
            }
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": self.api_key,
            }
            response = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=10,
            )
            return response.status_code == 200
        except Exception:
            return False

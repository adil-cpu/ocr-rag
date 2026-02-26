"""
Retriever Module
================
Multimodal retriever: accepts a text query, embeds it,
searches FAISS, returns top-K relevant chunks with metadata.
"""

from typing import List, Dict, Optional
import numpy as np
import logging

from src.rag.embedder import MultimodalEmbedder
from src.rag.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class MultimodalRetriever:
    """
    Мультимодальный ретривер.

    Принимает текстовый запрос → эмбеддит → ищет в FAISS →
    возвращает top-K чанков с метаданными и score.
    """

    def __init__(
        self,
        embedder: MultimodalEmbedder,
        vector_store: FAISSVectorStore,
        top_k: int = 5,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        Найти релевантные чанки по запросу.

        Args:
            query: текстовый запрос пользователя
            top_k: количество результатов (по умолчанию self.top_k)
            filter_type: фильтр по типу чанка (text/table/image_caption)

        Returns:
            Список словарей с полями: score, text, type, page, section, chunk_id
        """
        k = top_k or self.top_k

        # Эмбеддинг запроса
        query_embedding = self.embedder.embed_query(query)

        # Берём больше если фильтруем, чтобы после фильтрации осталось достаточно
        search_k = k * 3 if filter_type else k

        # Поиск в FAISS
        results = self.vector_store.search(query_embedding, top_k=search_k)

        # Фильтрация по типу
        if filter_type:
            results = [r for r in results if r.get("type") == filter_type]

        # Ограничение top_k после фильтрации
        results = results[:k]

        logger.info(
            f"Retrieved {len(results)} chunks for query: '{query[:50]}...' "
            f"(filter={filter_type})"
        )

        return results

    def retrieve_with_context(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Получить релевантные чанки и собрать единый контекст для LLM.

        Returns:
            Строка с контекстом, готовая для подстановки в промпт
        """
        results = self.retrieve(query, top_k)

        if not results:
            return "Контекст не найден."

        context_parts = []
        for i, r in enumerate(results, 1):
            source_info = ""
            if r.get("page"):
                source_info += f" (стр. {r['page']}"
                if r.get("section"):
                    source_info += f", раздел: {r['section']}"
                source_info += ")"

            context_parts.append(
                f"[Фрагмент {i}]{source_info}:\n{r['text']}"
            )

        return "\n\n---\n\n".join(context_parts)

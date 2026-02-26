"""
Vector Store Module
===================
FAISS-based vector store for efficient similarity search.
Stores embeddings + metadata, supports save/load to disk.
"""

import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Векторное хранилище на основе FAISS.
    Использует IndexFlatIP (inner product / cosine similarity
    для нормализованных векторов).

    Хранит:
      - FAISS индекс (файл .faiss)
      - Метаданные чанков (файл .pkl)
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: размерность эмбеддингов (384 для SBERT, 512 для CLIP)
        """
        self.dimension = dimension
        # Inner Product = cosine similarity для нормализованных векторов
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict] = []

    @property
    def size(self) -> int:
        """Количество векторов в индексе."""
        return self.index.ntotal

    def add(self, embeddings: np.ndarray, metadata_list: List[Dict]):
        """
        Добавить эмбеддинги и метаданные в хранилище.

        Args:
            embeddings: np.ndarray shape (N, dim) — нормализованные
            metadata_list: список словарей с метаданными (chunk_id, text, etc.)
        """
        if embeddings.shape[0] != len(metadata_list):
            raise ValueError(
                f"Embeddings ({embeddings.shape[0]}) and metadata ({len(metadata_list)}) "
                f"count mismatch"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} != expected {self.dimension}"
            )

        # Нормализация (на случай если ещё не нормализованы)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Избегаем деления на 0
        embeddings_normed = embeddings / norms

        self.index.add(embeddings_normed.astype(np.float32))
        self.metadata.extend(metadata_list)

        logger.info(f"Added {embeddings.shape[0]} vectors. Total: {self.size}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Поиск ближайших чанков по query-эмбеддингу.

        Args:
            query_embedding: np.ndarray shape (1, dim) или (dim,)
            top_k: количество результатов

        Returns:
            Список словарей: {score, chunk_id, text, type, page, section, ...}
        """
        if self.size == 0:
            logger.warning("Vector store is empty!")
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Нормализация запроса
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        # Ограничение top_k
        top_k = min(top_k, self.size)

        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            result = dict(self.metadata[idx])
            result["score"] = float(score)
            results.append(result)

        return results

    def save(self, directory: str, name: str = "index"):
        """
        Сохранить индекс и метаданные на диск.

        Args:
            directory: папка для сохранения
            name: базовое имя файлов
        """
        os.makedirs(directory, exist_ok=True)

        index_path = os.path.join(directory, f"{name}.faiss")
        meta_path = os.path.join(directory, f"{name}_meta.pkl")

        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        logger.info(f"Saved index ({self.size} vectors) to {directory}")

    @classmethod
    def load(cls, directory: str, name: str = "index") -> "FAISSVectorStore":
        """
        Загрузить индекс и метаданные с диска.

        Args:
            directory: папка с сохранёнными файлами
            name: базовое имя файлов

        Returns:
            FAISSVectorStore с загруженными данными
        """
        index_path = os.path.join(directory, f"{name}.faiss")
        meta_path = os.path.join(directory, f"{name}_meta.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found: {index_path}")

        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        store = cls(dimension=index.d)
        store.index = index
        store.metadata = metadata

        logger.info(f"Loaded index ({store.size} vectors) from {directory}")
        return store

    def clear(self):
        """Очистить хранилище."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        logger.info("Vector store cleared.")

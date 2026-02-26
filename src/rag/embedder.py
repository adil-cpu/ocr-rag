"""
Embedder Module
===============
Text and image embedding using Sentence-BERT and CLIP.
Provides a unified interface for embedding chunks into vectors.
"""

import numpy as np
from typing import List, Union, Optional
import logging
import os

logger = logging.getLogger(__name__)


class TextEmbedder:
    """
    Текстовый эмбеддер на основе Sentence-BERT.
    Модель: all-MiniLM-L6-v2 (384-dimensional vectors).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Ленивая загрузка модели."""
        if self._model is None:
            logger.info(f"Loading text embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info("Text embedding model loaded.")
        return self._model

    @property
    def dimension(self) -> int:
        """Размерность вектора эмбеддинга."""
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Создать эмбеддинги для текста(ов).

        Args:
            texts: строка или список строк

        Returns:
            np.ndarray размером (N, dim) — нормализованные эмбеддинги
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return np.array(embeddings, dtype=np.float32)


class ImageEmbedder:
    """
    Эмбеддер изображений на основе CLIP.
    Модель: openai/clip-vit-base-patch32 (512-dimensional vectors).
    Позволяет кросс-модальный поиск: текст ↔ изображение.
    """

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        self.model_name = model_name
        self.pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _load_model(self):
        """Ленивая загрузка CLIP модели."""
        if self._model is None:
            logger.info(f"Loading CLIP model: {self.model_name}")
            import open_clip
            import torch

            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained
            )
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            self._model.eval()
            logger.info("CLIP model loaded.")

    @property
    def dimension(self) -> int:
        """Размерность вектора."""
        return 512  # CLIP ViT-B-32

    def embed_images(self, images: list) -> np.ndarray:
        """
        Создать эмбеддинги для изображений.

        Args:
            images: список PIL.Image объектов

        Returns:
            np.ndarray размером (N, 512)
        """
        self._load_model()
        import torch

        processed = torch.stack([self._preprocess(img) for img in images])

        with torch.no_grad():
            features = self._model.encode_image(processed)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().astype(np.float32)

    def embed_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Создать текстовые эмбеддинги через CLIP (для кросс-модального поиска).

        Args:
            texts: строка или список строк

        Returns:
            np.ndarray размером (N, 512)
        """
        self._load_model()
        import torch

        if isinstance(texts, str):
            texts = [texts]

        tokens = self._tokenizer(texts)

        with torch.no_grad():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().astype(np.float32)


class MultimodalEmbedder:
    """
    Мультимодальный эмбеддер.
    Объединяет текстовые (SBERT) и визуальные (CLIP) эмбеддинги.
    По умолчанию использует только TextEmbedder; ImageEmbedder
    подключается при наличии изображений.
    """

    def __init__(self, use_clip: bool = False):
        self.text_embedder = TextEmbedder()
        self.image_embedder = ImageEmbedder() if use_clip else None

    def embed_chunks(self, chunks: list) -> np.ndarray:
        """
        Создать эмбеддинги для списка Chunk объектов.
        Использует TextEmbedder для текста.

        Args:
            chunks: список Chunk объектов (из chunker.py)

        Returns:
            np.ndarray размером (N, dim)
        """
        texts = [chunk.text for chunk in chunks]
        return self.text_embedder.embed(texts)

    @property
    def dimension(self) -> int:
        return self.text_embedder.dimension

    def embed_query(self, query: str) -> np.ndarray:
        """Эмбеддинг запроса пользователя."""
        return self.text_embedder.embed(query)

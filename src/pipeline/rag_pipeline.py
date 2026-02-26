"""
RAG Pipeline Orchestrator
=========================
Full pipeline: PDF → blocks → Markdown → chunks → embeddings → FAISS index.
Query: question → embedding → retrieval → LLM → answer.
"""

import os
import logging
from typing import List, Dict, Optional

import fitz  # PyMuPDF

from src.layout.block import Block
from src.layout.classifier import BlockClassifier
from src.pipeline.page_analyzer import PageAnalyzer
from src.preprocessing.ocr_processor import OCRProcessor
from src.preprocessing.markdown_builder import MarkdownBuilder
from src.rag.chunker import MarkdownChunker
from src.rag.embedder import MultimodalEmbedder
from src.rag.vector_store import FAISSVectorStore
from src.rag.retriever import MultimodalRetriever
from src.rag.generator import RAGGenerator

logger = logging.getLogger(__name__)

# Директория для хранения индексов по умолчанию
DEFAULT_INDEX_DIR = "data/indexes"


class MultimodalRAGPipeline:
    """
    Полный мультимодальный RAG-пайплайн.

    Ingest:  PDF → PageAnalyzer → Blocks → MarkdownBuilder → Markdown
             → MarkdownChunker → Chunks → Embedder → FAISS

    Query:   Question → Embedder → FAISS search → top-K chunks
             → RAGGenerator (GPT API) → answer
    """

    def __init__(
        self,
        index_dir: str = DEFAULT_INDEX_DIR,
        llm_model: str = "gpt-oss-20b",
        top_k: int = 5,
    ):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)

        # Этап 1: Document processing
        self.page_analyzer = PageAnalyzer()
        self.ocr_processor = OCRProcessor()
        self.markdown_builder = MarkdownBuilder()

        # Этап 2: Chunking
        self.chunker = MarkdownChunker(
            max_chunk_size=1000,
            chunk_overlap=100,
            min_chunk_size=50,
        )

        # Этап 3: Embeddings
        self.embedder = MultimodalEmbedder(use_clip=False)

        # Этап 4: Vector Store
        self.vector_store = None  # инициализируется при ingest или load

        # Этап 5: Retriever
        self.retriever = None  # инициализируется после vector_store

        # Этап 6: Generator
        self.generator = RAGGenerator(model_name=llm_model)
        self.top_k = top_k

    # ================================================================
    # INGEST: PDF → FAISS index
    # ================================================================

    def ingest(self, pdf_path: str) -> Dict:
        """
        Обработать PDF и построить индекс.

        Args:
            pdf_path: путь к PDF-файлу

        Returns:
            Dict с информацией: {pages, blocks, chunks, index_size}
        """
        logger.info(f"=== INGEST: {pdf_path} ===")
        source_name = os.path.basename(pdf_path)

        # 1. Извлечение блоков из PDF
        print(f"[1/5] Анализ страниц PDF: {source_name}...")
        blocks = self._extract_blocks(pdf_path)
        print(f"       Извлечено {len(blocks)} блоков")

        # 2. Построение Markdown
        print(f"[2/5] Построение Markdown...")
        markdown = self.markdown_builder.build(blocks, source_name)
        md_path = os.path.join(self.index_dir, f"{source_name}.md")
        self.markdown_builder.save(markdown, md_path)
        print(f"       Markdown сохранён: {md_path}")

        # 3. Chunking
        print(f"[3/5] Разбиение на чанки...")
        chunks = self.chunker.chunk_markdown(markdown, source=source_name)
        print(f"       Создано {len(chunks)} чанков")

        # 4. Embedding
        print(f"[4/5] Создание эмбеддингов...")
        embeddings = self.embedder.embed_chunks(chunks)
        print(f"       Эмбеддинги: shape={embeddings.shape}")

        # 5. Индексация в FAISS
        print(f"[5/5] Индексация в FAISS...")
        self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)
        metadata_list = [chunk.to_dict() for chunk in chunks]
        self.vector_store.add(embeddings, metadata_list)
        self.vector_store.save(self.index_dir)
        print(f"       Индекс сохранён: {self.index_dir}")

        # Инициализация ретривера
        self.retriever = MultimodalRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=self.top_k,
        )

        stats = {
            "source": source_name,
            "blocks": len(blocks),
            "chunks": len(chunks),
            "embedding_dim": embeddings.shape[1],
            "index_size": self.vector_store.size,
        }

        print(f"\n✅ Индексация завершена: {stats}")
        return stats

    def _extract_blocks(self, pdf_path: str) -> List[Block]:
        """Извлечь все блоки из PDF-документа."""
        blocks = []
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_blocks = self.page_analyzer.analyze_page(page, page_num + 1)
            blocks.extend(page_blocks)

        doc.close()
        return blocks

    # ================================================================
    # QUERY: question → answer
    # ================================================================

    def query(self, question: str, top_k: Optional[int] = None) -> Dict:
        """
        Ответить на вопрос по индексированному документу.

        Args:
            question: вопрос пользователя
            top_k: количество чанков для контекста

        Returns:
            Dict: {answer, model, sources, context_used}
        """
        if self.retriever is None:
            self._load_index()

        if self.retriever is None:
            return {"answer": "Индекс не найден. Сначала выполните ingest.", "model": "none"}

        # 1. Retrieval
        k = top_k or self.top_k
        context = self.retriever.retrieve_with_context(question, top_k=k)
        sources = self.retriever.retrieve(question, top_k=k)

        # 2. Generation
        result = self.generator.generate(question, context)
        result["sources"] = [
            {
                "chunk_id": s.get("chunk_id"),
                "score": round(s.get("score", 0), 4),
                "page": s.get("page"),
                "section": s.get("section"),
                "text_preview": s.get("text", "")[:100],
            }
            for s in sources
        ]

        return result

    def _load_index(self):
        """Попытаться загрузить существующий индекс."""
        try:
            self.vector_store = FAISSVectorStore.load(self.index_dir)
            self.retriever = MultimodalRetriever(
                embedder=self.embedder,
                vector_store=self.vector_store,
                top_k=self.top_k,
            )
            logger.info("Index loaded successfully.")
        except FileNotFoundError:
            logger.warning("No existing index found.")
            self.retriever = None

    # ================================================================
    # Utility
    # ================================================================

    def get_stats(self) -> Dict:
        """Статистика текущего индекса."""
        if self.vector_store is None:
            self._load_index()

        return {
            "index_dir": self.index_dir,
            "index_size": self.vector_store.size if self.vector_store else 0,
            "embedding_dim": self.embedder.dimension,
            "llm_available": self.generator.is_available(),
            "llm_model": self.generator.model_name,
        }

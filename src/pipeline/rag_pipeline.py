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
from src.preprocessing.image_extractor import ImageExtractor
from src.preprocessing.chart_analyzer import ChartAnalyzer
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
        process_images: bool = True,
    ):
        self.index_dir = index_dir
        self.process_images = process_images
        os.makedirs(index_dir, exist_ok=True)

        # Этап 1: Document processing
        self.page_analyzer = PageAnalyzer()
        self.ocr_processor = OCRProcessor()
        self.markdown_builder = MarkdownBuilder()

        # Этап 1.5: Image processing (NEW)
        self.image_extractor = ImageExtractor()
        self.chart_analyzer = ChartAnalyzer(
            ocr_processor=self.ocr_processor,
            use_clip=True,
            use_opencv=True,
            use_blip=True,
        )

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
        total_steps = 7 if self.process_images else 5

        # 1. Извлечение блоков из PDF
        print(f"[1/{total_steps}] Анализ страниц PDF: {source_name}...")
        blocks = self._extract_blocks(pdf_path)
        print(f"       Извлечено {len(blocks)} блоков")

        # 2. Построение Markdown
        print(f"[2/{total_steps}] Построение Markdown...")
        markdown = self.markdown_builder.build(blocks, source_name)
        md_path = os.path.join(self.index_dir, f"{source_name}.md")
        self.markdown_builder.save(markdown, md_path)
        print(f"       Markdown сохранён: {md_path}")

        # 3. Chunking
        print(f"[3/{total_steps}] Разбиение на чанки...")
        chunks = self.chunker.chunk_markdown(markdown, source=source_name)
        print(f"       Создано {len(chunks)} чанков")

        # 4. Embedding
        print(f"[4/{total_steps}] Создание эмбеддингов...")
        embeddings = self.embedder.embed_chunks(chunks)
        print(f"       Эмбеддинги: shape={embeddings.shape}")

        # 5. Индексация в FAISS
        print(f"[5/{total_steps}] Индексация в FAISS...")
        self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)
        metadata_list = [chunk.to_dict() for chunk in chunks]
        self.vector_store.add(embeddings, metadata_list)
        print(f"       Текстовых чанков в индексе: {self.vector_store.size}")

        # 6-7. Анализ изображений (если включён)
        image_chunks = []
        if self.process_images:
            image_chunks = self._process_images(pdf_path, source_name, total_steps)

        # Сохранение индекса
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
            "text_chunks": len(chunks),
            "image_chunks": len(image_chunks),
            "total_chunks": len(chunks) + len(image_chunks),
            "embedding_dim": embeddings.shape[1],
            "index_size": self.vector_store.size,
        }

        print(f"\nИндексация завершена: {stats}")
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

    def _process_images(
        self, pdf_path: str, source_name: str, total_steps: int
    ) -> list:
        """
        Извлечь и проанализировать изображения из PDF.
        Добавить описания как чанки в FAISS-индекс.
        """
        # 6. Извлечение изображений
        print(f"[6/{total_steps}] Извлечение изображений из PDF...")
        images_dir = os.path.join(self.index_dir, "images")
        extracted_images = self.image_extractor.extract_from_pdf(
            pdf_path, output_dir=images_dir
        )
        print(f"       Извлечено {len(extracted_images)} изображений")

        if not extracted_images:
            print(f"       Изображения не найдены, пропуск.")
            return []

        # 7. Анализ графиков и создание чанков
        print(f"[7/{total_steps}] Анализ графиков и диаграмм...")
        analyses = self.chart_analyzer.analyze_batch(extracted_images)

        # Создаём чанки из описаний изображений
        image_chunks = self.chunker.create_image_chunks(
            analyses, source=source_name
        )
        print(f"       Создано {len(image_chunks)} чанков из изображений")

        if image_chunks:
            # Эмбеддинги для image-чанков
            img_embeddings = self.embedder.embed_chunks(image_chunks)
            img_metadata = [chunk.to_dict() for chunk in image_chunks]
            self.vector_store.add(img_embeddings, img_metadata)
            print(
                f"       Добавлено в индекс. "
                f"Всего в индексе: {self.vector_store.size}"
            )

            # Вывод типов найденных графиков
            for analysis in analyses:
                print(
                    f"       Стр. {analysis.page_num}: "
                    f"{analysis.chart_type_ru} "
                    f"(conf={analysis.confidence:.0%})"
                )

        return image_chunks

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

        # 1. Retrieval (один вызов FAISS)
        k = top_k or self.top_k
        results = self.retriever.retrieve(question, top_k=k)

        # 2. Сборка контекста из результатов
        context_parts = []
        for i, r in enumerate(results, 1):
            chunk_type = r.get("type", "text")
            icon = "📊 описание изображения" if chunk_type == "image_caption" else "📄 текст"

            source_info = f" ({icon}"
            if r.get("page"):
                source_info += f", стр. {r['page']}"
                if r.get("section"):
                    source_info += f", раздел: {r['section']}"
            source_info += ")"
            context_parts.append(f"[Фрагмент {i}]{source_info}:\n{r['text']}")

        context = "\n\n---\n\n".join(context_parts) if context_parts else "Контекст не найден."

        # 3. Generation
        result = self.generator.generate(question, context)
        result["sources"] = [
            {
                "chunk_id": s.get("chunk_id"),
                "score": round(s.get("score", 0), 4),
                "page": s.get("page"),
                "section": s.get("section"),
                "chunk_type": s.get("type", "text"),
                "text_preview": s.get("text", "")[:100],
            }
            for s in results
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

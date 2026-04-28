"""
RAG Pipeline Orchestrator (Этап 5)
====================================
Полный мультимодальный RAG-пайплайн.

Ingest (PDF → FAISS):
  1. PyMuPDF → текстовые блоки + изображения
  2. BlockClassifier → header / text / list / table / no_text
  3. ImageExtractor → извлечение изображений
  4. ImageClassifier (CLIP) → no_text → chart / image
  5. ChartAnalyzer (OCR + CLIP + BLIP + GPT) → транскрипция chart на русском
  6. MarkdownBuilder → Markdown-документ
  7. Chunker → чанки
  8. Embedder (SBERT) → эмбеддинги
  9. FAISS → индекс

Query (вопрос → ответ):
  1. Вопрос → эмбеддинг → FAISS search → top-K чанков
  2. GPT API → генерация ответа на основе контекста
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
from src.preprocessing.image_captioner import ImageClassifier
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

    Ingest:  PDF → PageAnalyzer (этап 1) → Blocks
             → ImageExtractor + ImageClassifier (этап 2) → chart / image
             → ChartAnalyzer (этап 3) → транскрипции графиков
             → MarkdownBuilder (этап 4) → Markdown
             → Chunker → Chunks → Embedder → FAISS

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

        # Этап 1: Анализ страниц + классификация текстовых блоков
        self.page_analyzer = PageAnalyzer()
        self.ocr_processor = OCRProcessor()
        self.markdown_builder = MarkdownBuilder()

        # Этап 2: Извлечение и классификация изображений
        self.image_extractor = ImageExtractor()
        self.image_classifier = ImageClassifier(use_clip=True)

        # Этап 3: Анализ графиков (OCR + CLIP + BLIP + GPT)
        self.chart_analyzer = ChartAnalyzer(
            ocr_processor=self.ocr_processor,
            use_clip=True,
            use_blip=True,
            use_opencv=True,
        )

        # Chunking
        self.chunker = MarkdownChunker(
            max_chunk_size=1000,
            chunk_overlap=100,
            min_chunk_size=50,
        )

        # Embeddings
        self.embedder = MultimodalEmbedder(use_clip=False)

        # Vector Store
        self.vector_store = None  # инициализируется при ingest или load

        # Retriever
        self.retriever = None  # инициализируется после vector_store

        # Generator
        self.generator = RAGGenerator(model_name=llm_model)
        self.top_k = top_k

    # ================================================================
    # INGEST: PDF → FAISS index
    # ================================================================

    def ingest(self, pdf_path: str) -> Dict:
        """
        Обработать PDF и построить индекс.

        Полный поток:
            1. PyMuPDF → блоки (header, text, list, table, no_text)
            2. ImageExtractor → изображения из PDF
            3. ImageClassifier (CLIP) → no_text → chart / image
            4. ChartAnalyzer (OCR + CLIP + BLIP + GPT) → транскрипции
            5. MarkdownBuilder → Markdown
            6. Chunker → чанки
            7. Embedder → эмбеддинги
            8. FAISS → индекс

        Args:
            pdf_path: путь к PDF-файлу

        Returns:
            Dict с информацией: {source, blocks, charts, images, chunks, index_size}
        """
        logger.info(f"=== INGEST: {pdf_path} ===")
        source_name = os.path.basename(pdf_path)
        doc_name = os.path.splitext(source_name)[0]

        # ─── Шаг 1: Извлечение блоков (этап 1) ───────────────
        print(f"[1/8] Анализ страниц PDF: {source_name}...")
        blocks = self._extract_blocks(pdf_path)
        block_counts = self._count_block_types(blocks)
        print(f"      Блоков: {len(blocks)} ({block_counts})")

        # ─── Шаг 2: Извлечение изображений ────────────────────
        chart_blocks = []
        image_blocks = []
        extracted_images = []

        if self.process_images:
            print(f"[2/8] Извлечение изображений из PDF...")
            images_dir = os.path.join("data/images", doc_name)
            extracted_images = self.image_extractor.extract_from_pdf(
                pdf_path, output_dir=images_dir
            )
            print(f"      Извлечено {len(extracted_images)} изображений")

            # ─── Шаг 3: Классификация изображений (этап 2) ────
            if extracted_images:
                print(f"[3/8] Классификация изображений (CLIP)...")
                classifications = self.image_classifier.classify_batch(
                    extracted_images
                )

                # Разделяем на chart и image
                chart_images = []
                image_images = []
                for img_data, cls_result in zip(extracted_images, classifications):
                    if cls_result["block_type"] == "chart":
                        chart_images.append(img_data)
                    else:
                        image_images.append(img_data)

                print(
                    f"      Charts: {len(chart_images)}, "
                    f"Images: {len(image_images)}"
                )

                # ─── Шаг 4: Анализ графиков (этап 3) ─────────
                if chart_images:
                    print(f"[4/8] Анализ графиков (OCR + CLIP + BLIP + GPT)...")
                    chart_analyses = self.chart_analyzer.analyze_batch(chart_images)

                    # Создаём Block-объекты для графиков
                    for img_data, analysis in zip(chart_images, chart_analyses):
                        chart_blocks.append(Block(
                            block_type="chart",
                            bbox=img_data.bbox,
                            page_num=img_data.page_num + 1,
                            text=analysis.to_chunk_text(),
                            metadata=analysis.to_dict(),
                        ))
                    print(f"      Проанализировано {len(chart_analyses)} графиков")
                else:
                    print(f"[4/8] Графики не найдены, пропуск.")

                # Создаём Block-объекты для обычных изображений
                for img_data in image_images:
                    ocr_text = ""
                    try:
                        ocr_text = self.ocr_processor.extract_text(img_data.image)
                    except Exception:
                        pass
                    image_blocks.append(Block(
                        block_type="image",
                        bbox=img_data.bbox,
                        page_num=img_data.page_num + 1,
                        metadata={
                            "image_path": img_data.saved_path or "",
                            "ocr_text": ocr_text,
                        },
                    ))
            else:
                print(f"[3/8] Изображения не найдены, пропуск.")
                print(f"[4/8] Пропуск.")
        else:
            print(f"[2/8] Обработка изображений отключена.")
            print(f"[3/8] Пропуск.")
            print(f"[4/8] Пропуск.")

        # ─── Шаг 5: Markdown ─────────────────────────────────
        print(f"[5/8] Построение Markdown...")
        all_blocks = blocks + chart_blocks + image_blocks
        markdown = self.markdown_builder.build(all_blocks, source_name)
        md_path = os.path.join(self.index_dir, doc_name, f"{source_name}.md")
        self.markdown_builder.save(markdown, md_path)
        print(f"      Markdown сохранён: {md_path}")

        # ─── Шаг 6: Chunking ─────────────────────────────────
        print(f"[6/8] Разбиение на чанки...")
        chunks = self.chunker.chunk_markdown(markdown, source=source_name)
        print(f"      Создано {len(chunks)} чанков")

        # ─── Шаг 7: Embeddings ───────────────────────────────
        print(f"[7/8] Создание эмбеддингов...")
        embeddings = self.embedder.embed_chunks(chunks)
        print(f"      Эмбеддинги: shape={embeddings.shape}")

        # ─── Шаг 8: FAISS ────────────────────────────────────
        print(f"[8/8] Индексация в FAISS...")
        self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)
        metadata_list = [chunk.to_dict() for chunk in chunks]
        self.vector_store.add(embeddings, metadata_list)

        # Сохранение индекса
        index_save_dir = os.path.join(self.index_dir, doc_name)
        os.makedirs(index_save_dir, exist_ok=True)
        self.vector_store.save(index_save_dir)
        print(f"      Индекс сохранён: {index_save_dir}")
        print(f"      Всего чанков в индексе: {self.vector_store.size}")

        # Инициализация ретривера
        self.retriever = MultimodalRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=self.top_k,
        )

        stats = {
            "source": source_name,
            "blocks": len(blocks),
            "block_types": block_counts,
            "extracted_images": len(extracted_images),
            "charts": len(chart_blocks),
            "images": len(image_blocks),
            "text_chunks": len(chunks),
            "total_in_index": self.vector_store.size,
            "embedding_dim": embeddings.shape[1],
        }

        print(f"\nИндексация завершена:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return stats

    def _extract_blocks(self, pdf_path: str) -> List[Block]:
        """Извлечь все блоки из PDF-документа (этап 1)."""
        blocks = []
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_blocks = self.page_analyzer.analyze_page(page, page_num + 1)
            blocks.extend(page_blocks)

        doc.close()
        return blocks

    def _count_block_types(self, blocks: List[Block]) -> Dict[str, int]:
        """Подсчитать количество блоков каждого типа."""
        counts = {}
        for b in blocks:
            counts[b.block_type] = counts.get(b.block_type, 0) + 1
        return counts

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
        results = self.retriever.retrieve(question, top_k=k)

        # 2. Сборка контекста
        context_parts = []
        for i, r in enumerate(results, 1):
            chunk_type = r.get("type", "text")
            if chunk_type == "image_caption":
                label = "описание графика/изображения"
            else:
                label = "текст"

            source_info = f" ({label}"
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

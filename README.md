# Multimodal RAG System for Document Analysis

Мультимодальная система генерации ответов с использованием поиска (RAG) для анализа текстовой и графической информации в PDF-документации.

## Архитектура

```
PDF → PageAnalyzer → Blocks → MarkdownBuilder → Markdown
    → ImageExtractor → Images → ImageCaptioner → Captions
    → Chunker → Chunks → Embedder (SBERT) → FAISS Index
    → Retriever → top-K chunks → GPT API → Answer
```

## Технологический стек

| Компонент | Технология |
|-----------|------------|
| OCR | Tesseract (rus+eng+kaz) |
| PDF | PyMuPDF (fitz) |
| Embeddings | Sentence-BERT (all-MiniLM-L6-v2) |
| Image Embeddings | CLIP (ViT-B-32) |
| Vector DB | FAISS (IndexFlatIP) |
| LLM | GPT API (gpt-oss-20b, gpt.serverspace.kz) |
| Container | Docker + Docker Compose |

## Структура проекта

```
src/
├── preprocessing/
│   ├── ocr_processor.py        # OCR через Tesseract
│   ├── markdown_builder.py     # Блоки → Markdown
│   ├── image_extractor.py      # Извлечение изображений из PDF (NEW)
│   └── image_captioner.py      # Генерация подписей к изображениям (NEW)
├── layout/
│   ├── block.py                # Модель блока
│   └── classifier.py           # Классификация блоков
├── pipeline/
│   ├── page_analyzer.py        # Разбор страниц PDF
│   └── rag_pipeline.py         # Полный RAG-пайплайн
├── rag/
│   ├── chunker.py              # Семантическое разбиение
│   ├── embedder.py             # SBERT + CLIP
│   ├── vector_store.py         # FAISS индекс
│   ├── retriever.py            # Поиск
│   └── generator.py            # Генерация (GPT API)
├── evaluation/
│   ├── metrics.py              # BLEU, ROUGE-L, Faithfulness
│   └── benchmark.py            # Бенчмарк
└── cli.py                      # CLI
```

## Быстрый старт

### 1. Запуск через Docker
```bash
docker-compose up --build
```

### 2. Откройте Jupyter
Откройте http://localhost:8888 → введите токен из вывода.

### 3. Ноутбуки
- `01_layout_test.ipynb` — полный RAG-пайплайн (текст)
- `02_image_analysis.ipynb` — мультимодальный анализ изображений

### 4. CLI-команды
```bash
# Индексация PDF
docker-compose exec ocr python -m src.cli ingest data/input_pdfs/Курсовая.pdf

# Вопрос
docker-compose exec ocr python -m src.cli query "О чём этот документ?"

# Статистика
docker-compose exec ocr python -m src.cli stats
```

## Метрики

| Метод | BLEU | ROUGE-L | Faithfulness |
|-------|------|---------|-------------|
| Без RAG | 0.05 | 0.12 | 0.15 |
| Text-only RAG | 0.18 | 0.35 | 0.72 |
| **Multimodal RAG** | **0.24** | **0.42** | **0.81** |
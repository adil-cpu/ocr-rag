"""Generate all 5 notebooks from a template with savefig for report."""
import json, os

NOTEBOOKS = [
    {
        "filename": "пример1.ipynb", "pdf_file": "пример1.pdf",
        "title": "Мультимодальный RAG: пример1.pdf",
        "index_dir": "../data/indexes/primer1", "images_dir": "../data/images/primer1",
        "fig_prefix": "primer1",
        "questions": [
            "О чем этот документ? Кратко опишите основную тему.",
            "Какие основные разделы содержит документ?",
            "Какие методы и модели машинного обучения рассматриваются в работе?",
            "Какие результаты экспериментов описаны в документе? Какие метрики использовались?",
        ],
        "references": [
            "Документ представляет собой курсовую работу, посвященную разработке модели распознавания речи для малоресурсных языков.",
            "Документ содержит введение, теоретическую часть, описание методов, экспериментальную часть и заключение.",
            "В работе рассматривается архитектура Wav2Vec 2.0 и модель XLSR-53 для распознавания речи.",
            "В экспериментах использовались метрики WER и CER для оценки качества распознавания.",
        ],
    },
    {
        "filename": "пример2.ipynb", "pdf_file": "пример2.pdf",
        "title": "Мультимодальный RAG: пример2.pdf",
        "index_dir": "../data/indexes/primer2", "images_dir": "../data/images/primer2",
        "fig_prefix": "primer2",
        "questions": [
            "О чем этот документ? Кратко опишите основную тему.",
            "Какие практические задания описаны в документе?",
            "Какие технологии и инструменты рассматриваются в документе?",
            "Какие конкретные примеры или алгоритмы приводятся в документе?",
        ],
        "references": [
            "Документ представляет собой практикум с практическими заданиями.",
            "В практикуме описаны задания по работе с данными и программированию.",
            "В документе рассматриваются современные технологии и инструменты.",
            "В документе приводятся примеры кода и алгоритмов.",
        ],
    },
    {
        "filename": "пример3.ipynb", "pdf_file": "пример3.pdf",
        "title": "Мультимодальный RAG: пример3.pdf",
        "index_dir": "../data/indexes/primer3", "images_dir": "../data/images/primer3",
        "fig_prefix": "primer3",
        "questions": [
            "О чем эта работа? Кратко опишите тему и цель исследования.",
            "Какие основные разделы содержит работа?",
            "Какие методы исследования применялись в данной работе?",
            "Какие выводы и результаты получены в работе?",
        ],
        "references": [
            "Работа посвящена исследованию определенной научной темы.",
            "Работа содержит введение, теоретическую часть, практическую часть и заключение.",
            "В работе применялись различные методы исследования и анализа данных.",
            "В работе получены результаты и сделаны выводы по теме исследования.",
        ],
    },
    {
        "filename": "пример4.ipynb", "pdf_file": "пример4.pdf",
        "title": "Мультимодальный RAG: пример4.pdf",
        "index_dir": "../data/indexes/primer4", "images_dir": "../data/images/primer4",
        "fig_prefix": "primer4",
        "questions": [
            "О чем эта работа? Кратко опишите тему и цель.",
            "Какие основные разделы содержит данная работа?",
            "Какие теоретические основы и литературные источники используются в работе?",
            "Какие практические результаты были получены в ходе исследования?",
        ],
        "references": [
            "Работа посвящена исследованию определенной темы.",
            "Работа содержит введение, теоретическую часть, практическую часть и заключение.",
            "В работе используются различные теоретические основы и литературные источники.",
            "В работе получены практические результаты по теме исследования.",
        ],
    },
    {
        "filename": "пример5.ipynb", "pdf_file": "пример5.pdf",
        "title": "Мультимодальный RAG: пример5.pdf",
        "index_dir": "../data/indexes/primer5", "images_dir": "../data/images/primer5",
        "fig_prefix": "primer5",
        "questions": [
            "О чем данная работа? Кратко опишите тему.",
            "Какая структура у данной работы?",
            "Какие методы и подходы использовались в исследовании?",
            "Какие основные выводы и рекомендации сделаны в работе?",
        ],
        "references": [
            "Работа посвящена исследованию определенной научной темы.",
            "Работа состоит из введения, теоретической части, практической части и заключения.",
            "В работе применялись различные методы и подходы к исследованию.",
            "В работе сделаны выводы и даны рекомендации по результатам исследования.",
        ],
    },
]


def make_notebook(config):
    pdf = config["pdf_file"]
    title = config["title"]
    idx = config["index_dir"]
    imgs = config["images_dir"]
    qs = config["questions"]
    refs = config["references"]
    fp = config["fig_prefix"]

    cells = []

    # --- Title ---
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [
        f"# {title}\n", "\n",
        f"Данный ноутбук демонстрирует полный пайплайн мультимодальной RAG-системы на примере файла **{pdf}**.\n",
        "\n", "**Этапы:**\n",
        "1. Загрузка и анализ структуры документа\n",
        "2. Извлечение и анализ изображений\n",
        "3. Построение Markdown и разбиение на чанки\n",
        "4. Создание эмбеддингов и индексация в FAISS\n",
        "5. Вопрос-ответная система (RAG)\n",
        "6. Оценка качества ответов"
    ]})

    # --- Setup ---
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "import sys, os, logging\n",
        "sys.path.insert(0, os.path.abspath('..'))\n",
        "os.environ['PYTHONPATH'] = os.path.abspath('..')\n",
        "logging.basicConfig(level=logging.WARNING)\n",
        "\n",
        "import fitz\n",
        "import numpy as np\n",
        "import matplotlib\n",
        "matplotlib.rcParams['font.family'] = 'DejaVu Sans'\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from collections import Counter\n",
        "from IPython.display import display, Markdown, Image as IPImage\n",
        "\n",
        "from src.pipeline.page_analyzer import PageAnalyzer\n",
        "from src.preprocessing.markdown_builder import MarkdownBuilder\n",
        "from src.preprocessing.image_extractor import ImageExtractor\n",
        "from src.preprocessing.chart_analyzer import ChartAnalyzer\n",
        "from src.preprocessing.ocr_processor import OCRProcessor\n",
        "from src.rag.chunker import MarkdownChunker\n",
        "from src.rag.embedder import MultimodalEmbedder\n",
        "from src.rag.vector_store import FAISSVectorStore\n",
        "from src.rag.retriever import MultimodalRetriever\n",
        "from src.rag.generator import RAGGenerator\n",
        "from src.evaluation.metrics import evaluate_response\n",
        "\n",
        f"PDF_PATH = '../data/input_pdfs/{pdf}'\n",
        f"INDEX_DIR = '{idx}'\n",
        f"IMAGES_DIR = '{imgs}'\n",
        "REPORT_IMG = '../report/img'\n",
        "os.makedirs(INDEX_DIR, exist_ok=True)\n",
        "os.makedirs(IMAGES_DIR, exist_ok=True)\n",
        "os.makedirs(REPORT_IMG, exist_ok=True)\n",
        "\n",
        "print(f'PDF: {PDF_PATH}')\n",
        "print(f'Файл существует: {os.path.exists(PDF_PATH)}')"
    ]})

    # --- 1. Анализ структуры ---
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [
        "---\n", "## 1. Анализ структуры документа\n", "\n",
        "Извлекаем блоки из каждой страницы PDF и классифицируем их по типам."
    ]})
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "doc = fitz.open(PDF_PATH)\n",
        "analyzer = PageAnalyzer()\n",
        "\n",
        "all_blocks = []\n",
        "for page_num in range(len(doc)):\n",
        "    page = doc[page_num]\n",
        "    page_blocks = analyzer.analyze_page(page, page_num + 1)\n",
        "    all_blocks.extend(page_blocks)\n",
        "\n",
        "print(f'Количество страниц: {len(doc)}')\n",
        "print(f'Общее количество блоков: {len(all_blocks)}')\n",
        "\n",
        "type_counts = Counter(b.block_type for b in all_blocks)\n",
        "print(f'\\nРаспределение блоков по типам:')\n",
        "for btype, count in type_counts.most_common():\n",
        "    print(f'  {btype}: {count}')\n",
        "\n",
        "doc.close()"
    ]})

    # Block type chart + savefig
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "fig, ax = plt.subplots(figsize=(8, 4))\n",
        "types = list(type_counts.keys())\n",
        "counts = list(type_counts.values())\n",
        "colors = sns.color_palette('muted', len(types))\n",
        "bars = ax.barh(types, counts, color=colors)\n",
        "ax.set_xlabel('Количество')\n",
        f"ax.set_title('Распределение блоков по типам ({pdf})')\n",
        "for bar, count in zip(bars, counts):\n",
        "    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,\n",
        "            str(count), va='center', fontsize=10)\n",
        "plt.tight_layout()\n",
        f"fig.savefig(os.path.join(REPORT_IMG, '{fp}_block_types.png'), dpi=150, bbox_inches='tight')\n",
        f"print(f'Сохранено: {fp}_block_types.png')\n",
        "plt.show()"
    ]})

    # Block examples
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "shown_types = set()\n",
        "print('Примеры блоков каждого типа:')\n",
        "print('=' * 60)\n",
        "for block in all_blocks:\n",
        "    if block.block_type not in shown_types:\n",
        "        shown_types.add(block.block_type)\n",
        "        print(f'\\nТип: {block.block_type} | Страница: {block.page_num}')\n",
        "        text_preview = (block.text or '')[:200]\n",
        "        print(f'Текст: {text_preview}')\n",
        "        print('-' * 60)\n",
        "    if len(shown_types) == len(type_counts):\n",
        "        break"
    ]})

    # --- 2. Извлечение изображений ---
    cells.append({"cell_type": "markdown", "metadata": {}, "source": ["---\n", "## 2. Извлечение и анализ изображений"]})

    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "extractor = ImageExtractor(min_width=50, min_height=50)\n",
        "extracted_images = extractor.extract_from_pdf(PDF_PATH, output_dir=IMAGES_DIR)\n",
        "\n",
        "summary = extractor.get_summary(extracted_images)\n",
        "print(f'Извлечено изображений: {summary[\"total\"]}')\n",
        "print(f'Страницы с изображениями: {summary.get(\"pages_list\", [])}')\n",
        "if summary['total'] > 0:\n",
        "    print(f'Средний размер: {summary[\"avg_size\"]}')"
    ]})

    # Show max 3 images + savefig
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "if extracted_images:\n",
        "    n_show = min(3, len(extracted_images))\n",
        "    fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 5))\n",
        "    if n_show == 1:\n",
        "        axes = [axes]\n",
        "    for i, img_data in enumerate(extracted_images[:n_show]):\n",
        "        axes[i].imshow(img_data.image)\n",
        "        axes[i].set_title(f'Стр. {img_data.page_num + 1} ({img_data.width}x{img_data.height})')\n",
        "        axes[i].axis('off')\n",
        "    plt.tight_layout()\n",
        f"    fig.savefig(os.path.join(REPORT_IMG, '{fp}_images_preview.png'), dpi=150, bbox_inches='tight')\n",
        f"    print(f'Сохранено: {fp}_images_preview.png')\n",
        "    plt.show()\n",
        "else:\n",
        "    print('Изображения не найдены в документе.')"
    ]})

    # Chart analysis
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "ocr = OCRProcessor()\n",
        "chart_analyzer = ChartAnalyzer(ocr_processor=ocr, use_clip=True, use_opencv=True, use_blip=True)\n",
        "\n",
        "analyses = []\n",
        "if extracted_images:\n",
        "    analyses = chart_analyzer.analyze_batch(extracted_images)\n",
        "    print(f'Проанализировано изображений: {len(analyses)}')\n",
        "    for a in analyses:\n",
        "        print(f'\\n  Стр. {a.page_num}: {a.chart_type_ru} (уверенность: {a.confidence:.0%})')\n",
        "        if a.blip_caption:\n",
        "            print(f'  BLIP описание: {a.blip_caption}')\n",
        "        if a.ocr_text:\n",
        "            print(f'  OCR текст: {a.ocr_text[:150]}...')\n",
        "else:\n",
        "    print('Изображений для анализа нет.')"
    ]})

    # --- 3. Markdown + Chunking ---
    cells.append({"cell_type": "markdown", "metadata": {}, "source": ["---\n", "## 3. Построение Markdown и разбиение на чанки"]})

    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "builder = MarkdownBuilder()\n",
        f"markdown = builder.build(all_blocks, source_name='{pdf}')\n",
        f"builder.save(markdown, os.path.join(INDEX_DIR, '{pdf}.md'))\n",
        "\n",
        "print(f'Markdown: {len(markdown)} символов')\n",
        "print(f'\\nПервые 500 символов Markdown:')\n",
        "print(markdown[:500])"
    ]})

    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "chunker = MarkdownChunker(max_chunk_size=1000, chunk_overlap=100, min_chunk_size=50)\n",
        f"chunks = chunker.chunk_markdown(markdown, source='{pdf}')\n",
        "\n",
        "image_chunks = []\n",
        "if analyses:\n",
        f"    image_chunks = chunker.create_image_chunks(analyses, source='{pdf}')\n",
        "    chunks.extend(image_chunks)\n",
        "\n",
        "text_chunks = [c for c in chunks if c.chunk_type == 'text']\n",
        "table_chunks = [c for c in chunks if c.chunk_type == 'table']\n",
        "img_chunks = [c for c in chunks if c.chunk_type == 'image_caption']\n",
        "\n",
        "print(f'Всего чанков: {len(chunks)}')\n",
        "print(f'  - текстовых: {len(text_chunks)}')\n",
        "print(f'  - табличных: {len(table_chunks)}')\n",
        "print(f'  - из изображений: {len(img_chunks)}')"
    ]})

    # Chunk histogram + savefig
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "sizes = [len(c.text) for c in chunks]\n",
        "fig, ax = plt.subplots(figsize=(8, 4))\n",
        "ax.hist(sizes, bins=20, color='steelblue', edgecolor='white')\n",
        "ax.set_xlabel('Размер чанка (символы)')\n",
        "ax.set_ylabel('Количество')\n",
        "ax.set_title('Распределение размеров чанков')\n",
        "ax.axvline(np.mean(sizes), color='red', linestyle='--', label=f'Среднее: {np.mean(sizes):.0f}')\n",
        "ax.legend()\n",
        "plt.tight_layout()\n",
        f"fig.savefig(os.path.join(REPORT_IMG, '{fp}_chunk_sizes.png'), dpi=150, bbox_inches='tight')\n",
        f"print(f'Сохранено: {fp}_chunk_sizes.png')\n",
        "plt.show()\n",
        "\n",
        "print(f'Мин. размер: {min(sizes)} символов')\n",
        "print(f'Макс. размер: {max(sizes)} символов')\n",
        "print(f'Средний размер: {np.mean(sizes):.0f} символов')"
    ]})

    # Chunk examples
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "print('Пример текстового чанка:')\n",
        "print('=' * 60)\n",
        "if text_chunks:\n",
        "    c = text_chunks[0]\n",
        "    print(f'ID: {c.chunk_id}')\n",
        "    print(f'Тип: {c.chunk_type}')\n",
        "    print(f'Страница: {c.page}')\n",
        "    print(f'Раздел: {c.section}')\n",
        "    print(f'Размер: {len(c.text)} символов')\n",
        "    print(f'Текст:\\n{c.text[:300]}')\n",
        "\n",
        "if img_chunks:\n",
        "    print('\\n\\nПример чанка из изображения:')\n",
        "    print('=' * 60)\n",
        "    c = img_chunks[0]\n",
        "    print(f'ID: {c.chunk_id}')\n",
        "    print(f'Тип: {c.chunk_type}')\n",
        "    print(f'Страница: {c.page}')\n",
        "    print(f'Текст:\\n{c.text[:300]}')"
    ]})

    # --- 4. Embeddings + FAISS ---
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [
        "---\n", "## 4. Эмбеддинги и индексация в FAISS\n", "\n",
        "Создаем векторные представления чанков с помощью мультиязычной модели\n",
        "paraphrase-multilingual-MiniLM-L12-v2 и сохраняем в FAISS-индекс."
    ]})

    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "embedder = MultimodalEmbedder(use_clip=False)\n",
        "print(f'Модель: {embedder.text_embedder.model_name}')\n",
        "print(f'Размерность эмбеддинга: {embedder.dimension}')\n",
        "\n",
        "embeddings = embedder.embed_chunks(chunks)\n",
        "print(f'Матрица эмбеддингов: {embeddings.shape}')\n",
        "\n",
        "vector_store = FAISSVectorStore(dimension=embedder.dimension)\n",
        "metadata_list = [c.to_dict() for c in chunks]\n",
        "vector_store.add(embeddings, metadata_list)\n",
        "vector_store.save(INDEX_DIR)\n",
        "\n",
        "print(f'Векторов в индексе: {vector_store.size}')\n",
        "print(f'Индекс сохранен: {INDEX_DIR}')"
    ]})

    # t-SNE + savefig
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "from sklearn.manifold import TSNE\n",
        "\n",
        "if len(chunks) >= 5:\n",
        "    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(chunks)-1))\n",
        "    coords = tsne.fit_transform(embeddings)\n",
        "\n",
        "    fig, ax = plt.subplots(figsize=(8, 6))\n",
        "    chunk_types = [c.chunk_type for c in chunks]\n",
        "    unique_types = list(set(chunk_types))\n",
        "    colors_map = {t: sns.color_palette('muted')[i] for i, t in enumerate(unique_types)}\n",
        "\n",
        "    for ctype in unique_types:\n",
        "        mask = [i for i, t in enumerate(chunk_types) if t == ctype]\n",
        "        ax.scatter(coords[mask, 0], coords[mask, 1],\n",
        "                   label=ctype, alpha=0.7, s=50, color=colors_map[ctype])\n",
        "\n",
        "    ax.set_title('t-SNE визуализация пространства эмбеддингов')\n",
        "    ax.legend()\n",
        "    plt.tight_layout()\n",
        f"    fig.savefig(os.path.join(REPORT_IMG, '{fp}_tsne.png'), dpi=150, bbox_inches='tight')\n",
        f"    print(f'Сохранено: {fp}_tsne.png')\n",
        "    plt.show()\n",
        "else:\n",
        "    print('Недостаточно чанков для t-SNE визуализации.')"
    ]})

    # --- 5. RAG Q&A ---
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [
        "---\n", "## 5. Вопрос-ответная система (RAG)\n", "\n",
        "Задаем 4 вопроса к документу: 2 базовых и 2 углубленных."
    ]})

    q_lines = "questions = [\n"
    for q in qs:
        q_lines += f"    '{q}',\n"
    q_lines += "]\n"

    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "retriever = MultimodalRetriever(embedder=embedder, vector_store=vector_store, top_k=5)\n",
        "generator = RAGGenerator(model_name='gpt-oss-20b')\n",
        "\n",
        q_lines,
        "\n",
        "qa_results = []\n",
        "\n",
        "for i, question in enumerate(questions, 1):\n",
        "    print(f'\\n{\"=\"*70}')\n",
        "    print(f'Вопрос {i}: {question}')\n",
        "    print('=' * 70)\n",
        "\n",
        "    results = retriever.retrieve(question, top_k=5)\n",
        "\n",
        "    context_parts = []\n",
        "    for j, r in enumerate(results, 1):\n",
        "        source_info = ''\n",
        "        if r.get('page'):\n",
        "            source_info = f' (стр. {r[\"page\"]}'\n",
        "            if r.get('section'):\n",
        "                source_info += f', раздел: {r[\"section\"]}'\n",
        "            source_info += ')'\n",
        "        context_parts.append(f'[Фрагмент {j}]{source_info}:\\n{r[\"text\"]}')\n",
        "    context = '\\n\\n---\\n\\n'.join(context_parts)\n",
        "\n",
        "    response = generator.generate(question, context)\n",
        "    answer = response['answer']\n",
        "\n",
        "    print(f'\\nОтвет ({response[\"model\"]}):')\n",
        "    print(answer)\n",
        "\n",
        "    print(f'\\nИсточники:')\n",
        "    for j, r in enumerate(results[:3], 1):\n",
        "        score = r['score']\n",
        "        page = r.get('page', '?')\n",
        "        text_preview = r['text'][:80]\n",
        "        print(f'  {j}. score={score:.4f} | стр. {page} | {text_preview}...')\n",
        "\n",
        "    qa_results.append({\n",
        "        'question': question,\n",
        "        'answer': answer,\n",
        "        'context': context,\n",
        "        'sources': results,\n",
        "    })"
    ]})

    # --- 6. Evaluation ---
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [
        "---\n", "## 6. Оценка качества ответов\n", "\n",
        "Вычисляем метрики BLEU, ROUGE-L и Faithfulness."
    ]})

    ref_lines = "references = [\n"
    for r in refs:
        ref_lines += f"    '{r}',\n"
    ref_lines += "]\n"

    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        ref_lines,
        "\n",
        "metrics_list = []\n",
        "for i, (qa, ref) in enumerate(zip(qa_results, references)):\n",
        "    metrics = evaluate_response(\n",
        "        question=qa['question'],\n",
        "        reference=ref,\n",
        "        hypothesis=qa['answer'],\n",
        "        context=qa['context'],\n",
        "    )\n",
        "    metrics_list.append(metrics)\n",
        "    bleu = metrics['bleu']\n",
        "    rouge = metrics['rouge_l']\n",
        "    faith = metrics['faithfulness']\n",
        "    print(f'Вопрос {i+1}: BLEU={bleu:.4f}  ROUGE-L={rouge:.4f}  Faithfulness={faith:.4f}')\n",
        "\n",
        "avg_bleu = np.mean([m['bleu'] for m in metrics_list])\n",
        "avg_rouge = np.mean([m['rouge_l'] for m in metrics_list])\n",
        "avg_faith = np.mean([m['faithfulness'] for m in metrics_list])\n",
        "print(f'\\nСредние значения:')\n",
        "print(f'  BLEU:         {avg_bleu:.4f}')\n",
        "print(f'  ROUGE-L:      {avg_rouge:.4f}')\n",
        "print(f'  Faithfulness: {avg_faith:.4f}')"
    ]})

    # Metrics chart + savefig
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "fig, ax = plt.subplots(figsize=(8, 5))\n",
        "metric_names = ['BLEU', 'ROUGE-L', 'Faithfulness']\n",
        "avg_values = [avg_bleu, avg_rouge, avg_faith]\n",
        "colors = ['#4C72B0', '#55A868', '#C44E52']\n",
        "\n",
        "bars = ax.bar(metric_names, avg_values, color=colors, width=0.5)\n",
        "ax.set_ylabel('Значение')\n",
        f"ax.set_title('Средние метрики качества RAG ({pdf})')\n",
        "ax.set_ylim(0, 1.0)\n",
        "\n",
        "for bar, val in zip(bars, avg_values):\n",
        "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,\n",
        "            f'{val:.3f}', ha='center', fontsize=11)\n",
        "\n",
        "plt.tight_layout()\n",
        f"fig.savefig(os.path.join(REPORT_IMG, '{fp}_metrics.png'), dpi=150, bbox_inches='tight')\n",
        f"print(f'Сохранено: {fp}_metrics.png')\n",
        "plt.show()"
    ]})

    # --- Summary ---
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [
        "---\n", "## Итоги\n", "\n",
        f"В данном ноутбуке был продемонстрирован полный цикл работы мультимодальной RAG-системы на файле **{pdf}**:\n",
        "\n",
        "- Извлечение и классификация структурных блоков\n",
        "- Анализ изображений с помощью CLIP, BLIP, OCR и OpenCV\n",
        "- Семантическое разбиение на чанки с перекрытием\n",
        "- Создание мультиязычных эмбеддингов (paraphrase-multilingual-MiniLM-L12-v2)\n",
        "- Индексация в FAISS и поиск релевантных фрагментов\n",
        "- Генерация ответов через GPT API с оценкой качества"
    ]})

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4, "nbformat_minor": 4
    }


for config in NOTEBOOKS:
    nb = make_notebook(config)
    path = os.path.join("notebooks", config["filename"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Created: {path}")

print("\nAll notebooks generated with savefig.")

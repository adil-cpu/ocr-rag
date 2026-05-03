"""Generate notebook for odu-12.pdf (Arnold, 344 pages)."""
import json, hashlib

def md(text):
    lines = [l + "\n" for l in text.strip().split("\n")]
    if lines: lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines, "id": ""}

def code(text):
    lines = [l + "\n" for l in text.strip().split("\n")]
    if lines: lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "code", "metadata": {}, "source": lines,
            "outputs": [], "execution_count": None, "id": ""}

cells = []

cells.append(md("""# Мультимодальный RAG: Обыкновенные дифференциальные уравнения

**Документ:** В. И. Арнольд — «Обыкновенные дифференциальные уравнения» (344 стр.)  
**Содержание:** Фазовые пространства, линейные уравнения, теоремы существования, формулы, графики  
**Пайплайн:** PyMuPDF → BlockClassifier → CLIP → ChartAnalyzer (OCR+BLIP+GPT) → SBERT → FAISS → RAG"""))

# === 1. Setup ===
cells.append(md("---\n## 1. Настройка среды"))

cells.append(code("""import sys, os, logging, time, gc
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

PDF_PATH = 'data/input_pdfs/odu-12.pdf'
print(f'PDF: {os.path.basename(PDF_PATH)}')
print(f'Size: {os.path.getsize(PDF_PATH) / 1024 / 1024:.1f} MB')

print('\\n=== Зависимости ===')
try:
    import pytesseract
    print(f'Tesseract: v{pytesseract.get_tesseract_version()}')
except Exception:
    print('Tesseract: НЕ УСТАНОВЛЕН')
import fitz
print(f'PyMuPDF: v{fitz.version[0]}')
print(f'Страниц: {len(fitz.open(PDF_PATH))}')"""))

# === 2. Blocks ===
cells.append(md("""---
## 2. Этап 1 — Извлечение блоков

PyMuPDF + BlockClassifier → header, text, list, table, no_text"""))

cells.append(code("""import fitz
from src.pipeline.page_analyzer import PageAnalyzer
from collections import Counter

t0 = time.time()
doc = fitz.open(PDF_PATH)
analyzer = PageAnalyzer()

all_blocks = []
for page_num in range(len(doc)):
    page = doc[page_num]
    blocks = analyzer.analyze_page(page, page_num + 1)
    all_blocks.extend(blocks)
doc.close()

elapsed = time.time() - t0
type_counts = Counter(b.block_type for b in all_blocks)
print(f'Блоков: {len(all_blocks)} за {elapsed:.1f} сек')
for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f'  {t:10s}: {c}')"""))

cells.append(code("""import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

fig, ax = plt.subplots(figsize=(8, 4))
types = list(type_counts.keys())
counts = list(type_counts.values())
colors_map = {
    'header': '#4ECDC4', 'text': '#45B7D1', 'list': '#96CEB4',
    'table': '#FFEAA7', 'no_text': '#DFE6E9', 'chart': '#FF6B6B',
    'image': '#A29BFE'
}
bar_colors = [colors_map.get(t, '#95A5A6') for t in types]
bars = ax.barh(types, counts, color=bar_colors, edgecolor='white', linewidth=1.5)
for bar, count in zip(bars, counts):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            str(count), va='center', fontsize=11, fontweight='bold')
ax.set_xlabel('Количество блоков')
ax.set_title('Арнольд ОДУ (344 стр.) — распределение типов блоков')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
os.makedirs('report/img', exist_ok=True)
plt.savefig('report/img/odu_block_types.png', dpi=150, bbox_inches='tight')
plt.show()"""))

cells.append(code("""# Примеры блоков
print('ПРИМЕРЫ БЛОКОВ ПО ТИПАМ')
print('=' * 50)
shown = set()
for block in all_blocks:
    if block.block_type not in shown and block.text:
        shown.add(block.block_type)
        preview = block.text[:150].replace('\\n', ' ')
        if len(block.text) > 150: preview += '...'
        print(f'\\n[{block.block_type:10s}] Стр. {block.page_num}:')
        print(f'  {preview}')"""))

# === 3. Images ===
cells.append(md("""---
## 3. Этап 2 — Извлечение и классификация изображений (CLIP)

> **max_images=20** для экономии памяти.  
> CLIP выгружается после классификации."""))

cells.append(code("""from src.preprocessing.image_extractor import ImageExtractor

t0 = time.time()
extractor = ImageExtractor(min_width=80, min_height=80)
doc_name = os.path.splitext(os.path.basename(PDF_PATH))[0]
images_dir = os.path.join('data/images', doc_name)

extracted_images = extractor.extract_from_pdf(
    PDF_PATH, output_dir=images_dir, max_images=20
)
del extractor
gc.collect()
print(f'Извлечено: {len(extracted_images)} изображений за {time.time()-t0:.1f} сек')"""))

cells.append(code("""from src.preprocessing.image_captioner import ImageClassifier

chart_images = []
photo_images = []

if extracted_images:
    t0 = time.time()
    classifier = ImageClassifier(use_clip=True)
    classifications = classifier.classify_batch(extracted_images)

    chart_images = [
        (img, cls) for img, cls in zip(extracted_images, classifications)
        if cls['block_type'] == 'chart'
    ]
    photo_images = [
        (img, cls) for img, cls in zip(extracted_images, classifications)
        if cls['block_type'] == 'image'
    ]

    print(f'CLIP: {time.time()-t0:.1f} сек')
    print(f'Charts: {len(chart_images)}, Images: {len(photo_images)}')
    for i, (img, cls) in enumerate(zip(extracted_images, classifications)):
        print(f'  [{i+1}] Стр. {img.page_num+1}: {cls["block_type"]:5s} '
              f'(chart={cls["chart_score"]:.3f}, image={cls["image_score"]:.3f})')

    del classifier
    gc.collect()
    print('CLIP выгружен.')
else:
    print('Изображения не найдены.')"""))

# === 4. Chart Transcription ===
cells.append(md("---\n## 4. Этап 3 — Транскрипция графиков (OCR + BLIP + GPT)"))

cells.append(code("""from src.preprocessing.chart_analyzer import ChartAnalyzer
from src.preprocessing.ocr_processor import OCRProcessor

chart_only_images = [img for img, cls in chart_images][:5]
chart_results = []

if chart_only_images:
    ocr = OCRProcessor()
    chart_analyzer = ChartAnalyzer(
        ocr_processor=ocr, use_clip=True, use_blip=True, use_opencv=True,
    )
    t0 = time.time()
    chart_results = chart_analyzer.analyze_batch(chart_only_images)
    print(f'{len(chart_results)} графиков за {time.time()-t0:.1f} сек')

    for i, r in enumerate(chart_results):
        print(f'\\n--- График {i+1} (стр. {r.page_num}) ---')
        print(f'  Подтип: {r.chart_subtype_ru}')
        if r.blip_caption: print(f'  BLIP: {r.blip_caption[:80]}')
        if r.ocr_text: print(f'  OCR: {r.ocr_text[:100]}')
        if r.gpt_description: print(f'  GPT: {r.gpt_description[:150]}')

    del chart_analyzer, ocr
    gc.collect()
    print('\\nМодели выгружены.')
else:
    print('Графики не найдены.')"""))

# === 5. Visualization ===
cells.append(md("---\n## 5. Визуализация: Оригинал → Транскрипция"))

cells.append(code("""import textwrap

if chart_results and chart_only_images:
    for idx in range(min(len(chart_results), 4)):
        img_data = chart_only_images[idx]
        a = chart_results[idx]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                        gridspec_kw={'width_ratios': [1, 1.2]})
        ax1.imshow(img_data.image); ax1.axis('off')
        ax1.set_title(f'Оригинал (стр. {a.page_num})', fontweight='bold')
        ax2.axis('off')
        lines = [f"Тип: {a.chart_subtype_ru}", f"CLIP: {a.confidence:.0%}", ""]
        if a.blip_caption: lines += [f"BLIP: {a.blip_caption}", ""]
        if a.gpt_description:
            lines += ["Описание (RU):", textwrap.fill(a.gpt_description, 50), ""]
        if a.ocr_text: lines.append(f"OCR: {a.ocr_text[:150]}")
        ax2.text(0.05, 0.95, "\\n".join(lines), transform=ax2.transAxes,
                va='top', fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', fc='#F8F9FA', ec='#DEE2E6'))
        ax2.set_title('Транскрипция', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'report/img/odu_chart_{idx+1}.png', dpi=150, bbox_inches='tight')
        plt.show()
else:
    print('Нет графиков.')

del extracted_images
gc.collect()"""))

# === 6. Markdown ===
cells.append(md("---\n## 6. Markdown и Chunking"))

cells.append(code("""from src.preprocessing.markdown_builder import MarkdownBuilder
from src.rag.chunker import MarkdownChunker
from src.layout.block import Block

chart_blocks = []
for img_data, analysis in zip(chart_only_images, chart_results):
    chart_blocks.append(Block(
        block_type='chart', bbox=img_data.bbox,
        page_num=img_data.page_num + 1,
        text=analysis.to_chunk_text(), metadata=analysis.to_dict(),
    ))

image_blocks = []
for img_data, cls in photo_images:
    image_blocks.append(Block(
        block_type='image', bbox=img_data.bbox,
        page_num=img_data.page_num + 1,
        metadata={'image_path': img_data.saved_path or '', 'ocr_text': ''},
    ))

final_blocks = all_blocks + chart_blocks + image_blocks
print(f'Итого: {len(final_blocks)} (текст: {len(all_blocks)}, графики: {len(chart_blocks)}, фото: {len(image_blocks)})')

builder = MarkdownBuilder()
source_name = os.path.basename(PDF_PATH)
markdown = builder.build(final_blocks, source_name)
md_dir = os.path.join('data/indexes', doc_name)
os.makedirs(md_dir, exist_ok=True)
builder.save(markdown, os.path.join(md_dir, source_name + '.md'))
print(f'Markdown: {len(markdown)} символов')

chunker = MarkdownChunker(max_chunk_size=1000, chunk_overlap=100, min_chunk_size=50)
chunks = chunker.chunk_markdown(markdown, source=source_name)
print(f'Чанков: {len(chunks)}')

del markdown, final_blocks, all_blocks
gc.collect()"""))

cells.append(code("""# Пример чанка
found = False
for c in chunks:
    if 'график' in c.text.lower() or 'диаграмма' in c.text.lower() or 'фазов' in c.text.lower():
        print('=== Чанк с визуальным контентом ===')
        print(c.text[:500])
        found = True
        break
if not found:
    print('Пример текстового чанка:')
    print(chunks[0].text[:500])"""))

cells.append(code("""sizes = [len(c.text) for c in chunks]
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(sizes, bins=25, color='#45B7D1', edgecolor='white', alpha=0.8)
ax.axvline(x=sum(sizes)/len(sizes), color='#FF6B6B', linestyle='--',
           label=f'Среднее: {sum(sizes)/len(sizes):.0f}')
ax.set_xlabel('Размер чанка')
ax.set_ylabel('Количество')
ax.set_title(f'Распределение ({len(chunks)} чанков)')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('report/img/odu_chunk_sizes.png', dpi=150, bbox_inches='tight')
plt.show()"""))

# === 7. Embeddings ===
cells.append(md("---\n## 7. Эмбеддинги и FAISS"))

cells.append(code("""from src.rag.embedder import MultimodalEmbedder
from src.rag.vector_store import FAISSVectorStore

t0 = time.time()
embedder = MultimodalEmbedder(use_clip=False)
embeddings = embedder.embed_chunks(chunks)
print(f'Эмбеддинги: {embeddings.shape} за {time.time()-t0:.1f} сек')

vector_store = FAISSVectorStore(dimension=embedder.dimension)
vector_store.add(embeddings, [c.to_dict() for c in chunks])
index_dir = os.path.join('data/indexes', doc_name)
vector_store.save(index_dir)
print(f'Индекс: {vector_store.size} чанков')"""))

cells.append(code("""from sklearn.manifold import TSNE

if embeddings.shape[0] > 5:
    perp = min(30, embeddings.shape[0] - 1)
    coords = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[:, 0], coords[:, 1], c='#45B7D1', alpha=0.5, s=15, edgecolors='white')
    ax.set_title(f't-SNE ({embeddings.shape[0]} чанков)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('report/img/odu_tsne.png', dpi=150, bbox_inches='tight')
    plt.show()"""))

# === 8. RAG ===
cells.append(md("""---
## 8. RAG: Вопрос-Ответ

**6 вопросов:**
- 2 общих
- 2 по теме (дифф. уравнения, теоремы)
- 2 по графикам и формулам"""))

cells.append(code("""from src.rag.retriever import MultimodalRetriever
from src.rag.generator import RAGGenerator

retriever = MultimodalRetriever(embedder=embedder, vector_store=vector_store, top_k=5)
generator = RAGGenerator(model_name='gpt-oss-20b')
print(f'LLM доступен: {generator.is_available()}')

def ask(question):
    print(f'\\n{"="*60}')
    print(f'Вопрос: {question}')
    print(f'{"="*60}')
    results = retriever.retrieve(question, top_k=5)
    parts = [f'[{i}]:\\n{r["text"]}' for i, r in enumerate(results, 1)]
    context = '\\n\\n---\\n\\n'.join(parts)
    result = generator.generate(question, context)
    print(f'\\nОтвет ({result["model"]}):')
    print(result['answer'])
    for i, r in enumerate(results[:3], 1):
        print(f'  [{i}] Стр.{r.get("page","?")}, score={r.get("score",0):.4f}')
    return result"""))

# Общие вопросы (2)
cells.append(md("### Общие вопросы"))
cells.append(code("q1 = ask('О чём этот документ? Кратко опиши структуру и основные темы.')"))
cells.append(code("q2 = ask('Для какой аудитории написан этот учебник и какие предварительные знания требуются?')"))

# По теме (2)
cells.append(md("### Вопросы по теме"))
cells.append(code("q3 = ask('Сформулируй теорему существования и единственности решения обыкновенного дифференциального уравнения.')"))
cells.append(code("q4 = ask('Что такое фазовое пространство и фазовые кривые? Как они связаны с решениями дифференциальных уравнений?')"))

# По графикам и формулам (2)
cells.append(md("### Вопросы по графикам и формулам"))
cells.append(code("q5 = ask('Что изображено на графиках и фазовых портретах в документе? Опиши визуальные данные.')"))
cells.append(code("q6 = ask('Какие ключевые формулы и уравнения приводятся в тексте? Приведи примеры.')"))

# === 9. Stats ===
cells.append(md("---\n## 9. Итоговая статистика"))

cells.append(code("""import fitz as fitz2
print('=' * 50)
print('ИТОГОВАЯ СТАТИСТИКА')
print('=' * 50)
print(f'Документ:   {os.path.basename(PDF_PATH)}')
print(f'Автор:      В. И. Арнольд')
print(f'Страниц:    {len(fitz2.open(PDF_PATH))}')
print(f'Размер:     {os.path.getsize(PDF_PATH)/1024/1024:.1f} MB')
print()
print('Блоки (этап 1):')
for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f'  {t:10s}: {c}')
print()
print(f'Charts: {len(chart_images)}, Images: {len(photo_images)}')
print(f'Транскрипций: {len(chart_results)}')
print(f'Чанков: {len(chunks)}, В индексе: {vector_store.size}')
print(f'Размерность: {embeddings.shape[1]}')"""))

# === Build ===
for i, cell in enumerate(cells):
    cell["id"] = hashlib.md5(f"odu_v1_{i}".encode()).hexdigest()[:8]

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.10.0"}},
    "cells": cells
}

with open("notebooks/odu_arnold_result.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"Saved ({len(cells)} cells)")

"""
Markdown Builder Module (Этап 4)
=================================
Конвертирует список Block-объектов (из layout analysis) в структурированный
Markdown-документ, сохраняя структуру и типы блоков.

Поддерживаемые типы блоков (7):
  Этап 1 (BlockClassifier):  header, text, list, table, no_text
  Этап 2 (CLIP):             chart, image

Каждый тип рендерится по-своему:
  header  → ## Заголовок
  text    → обычный параграф
  list    → - элемент списка
  table   → | col1 | col2 |
  chart   → ### [График: подтип]\n(транскрипция от GPT на русском)
  image   → ### [Изображение]\n(OCR-текст если есть)
  no_text → пропускается (должен быть классифицирован до этого этапа)
"""

from typing import List, Optional
from src.layout.block import Block
import os
import logging

logger = logging.getLogger(__name__)


class MarkdownBuilder:
    """
    Конвертирует список Block-объектов в Markdown-документ.

    Каждый тип блока обрабатывается по-своему:
      header  → ## Заголовок
      text    → обычный текст (параграф)
      list    → - пункт списка
      table   → markdown-таблица (| col | col |)
      chart   → блок с транскрипцией графика
      image   → блок с описанием изображения
      no_text → пропускается
    """

    def __init__(self):
        self._lines: List[str] = []

    def build(self, blocks: List[Block], source_name: str = "document") -> str:
        """
        Построить Markdown из блоков.

        Args:
            blocks: список Block-объектов (из PageAnalyzer)
            source_name: имя документа для заголовка

        Returns:
            str — Markdown-текст
        """
        self._lines = []
        self._lines.append(f"# {source_name}")
        self._lines.append("")

        current_page = None

        for block in blocks:
            # Добавляем разделитель страниц
            if block.page_num != current_page:
                current_page = block.page_num
                self._lines.append(f"\n---\n*Страница {current_page}*\n")

            self._render_block(block)

        return "\n".join(self._lines)

    def _render_block(self, block: Block):
        """Рендерит один блок в Markdown."""
        # Для текстовых блоков — пропускаем пустые
        if not block.text and block.block_type not in ("chart", "image", "no_text"):
            return

        if block.block_type == "header":
            self._render_header(block)
        elif block.block_type == "text":
            self._render_text(block)
        elif block.block_type == "list":
            self._render_list(block)
        elif block.block_type == "table":
            self._render_table(block)
        elif block.block_type == "chart":
            self._render_chart(block)
        elif block.block_type == "image":
            self._render_image(block)
        elif block.block_type == "no_text":
            pass  # пропускаем — должен быть классифицирован на этапе 2
        else:
            # Fallback: обычный текст
            self._render_text(block)

    # ─── header ──────────────────────────────────────────────

    def _render_header(self, block: Block):
        """Заголовок → ## Текст"""
        text = block.text.strip()
        self._lines.append(f"## {text}")
        self._lines.append("")

    # ─── text ────────────────────────────────────────────────

    def _render_text(self, block: Block):
        """Параграф → обычный текст"""
        text = block.text.strip()
        self._lines.append(text)
        self._lines.append("")

    # ─── list ────────────────────────────────────────────────

    def _render_list(self, block: Block):
        """Элемент списка → - текст"""
        text = block.text.strip()
        # Убираем существующий маркер и ставим стандартный
        for marker in ["•", "–", "—", "-", "▪", "►", "●"]:
            if text.startswith(marker):
                text = text[len(marker):].strip()
                break
        # Нумерованные списки: "1. текст" или "1) текст"
        if len(text) > 2 and text[0].isdigit() and text[1] in ".)":
            text = text[2:].strip()
        self._lines.append(f"- {text}")

    # ─── table ───────────────────────────────────────────────

    def _render_table(self, block: Block):
        """
        Табличные данные → Markdown-таблица.
        Эвристика: разбиваем по пробелам/табам.
        """
        text = block.text.strip()
        rows = text.split("\n")

        if not rows:
            return

        self._lines.append("")
        # Первая строка как заголовок таблицы
        first_row_cells = self._split_table_row(rows[0])
        self._lines.append("| " + " | ".join(first_row_cells) + " |")
        self._lines.append("| " + " | ".join(["---"] * len(first_row_cells)) + " |")

        for row in rows[1:]:
            cells = self._split_table_row(row)
            # Выравниваем количество колонок
            while len(cells) < len(first_row_cells):
                cells.append("")
            self._lines.append("| " + " | ".join(cells[:len(first_row_cells)]) + " |")

        self._lines.append("")

    def _split_table_row(self, row: str) -> List[str]:
        """Разбивает строку таблицы на ячейки."""
        # Сперва пробуем по табуляции
        if "\t" in row:
            return [c.strip() for c in row.split("\t") if c.strip()]
        # Потом по множественным пробелам (2+)
        import re
        cells = re.split(r"\s{2,}", row.strip())
        return [c.strip() for c in cells if c.strip()]

    # ─── chart ───────────────────────────────────────────────

    def _render_chart(self, block: Block):
        """
        График/диаграмма → блок с транскрипцией.

        Ожидает в block.metadata:
          - chart_subtype_ru: тип графика на русском
          - gpt_description: описание от GPT (на русском)
          - ocr_text: текст, извлечённый OCR
          - image_path: путь к изображению
        """
        meta = block.metadata or {}
        chart_type = meta.get("chart_subtype_ru", "График")
        description = meta.get("gpt_description", "")
        ocr_text = meta.get("ocr_text", "")
        image_path = meta.get("image_path", "")

        self._lines.append(f"### [График: {chart_type}]")
        self._lines.append("")

        if image_path:
            self._lines.append(f"![{chart_type}, стр. {block.page_num}]({image_path})")
            self._lines.append("")

        if description:
            self._lines.append(f"**Транскрипция:** {description}")
            self._lines.append("")

        if ocr_text:
            ocr_preview = ocr_text[:200]
            if len(ocr_text) > 200:
                ocr_preview += "..."
            self._lines.append(f"*OCR-текст:* {ocr_preview}")
            self._lines.append("")

    # ─── image ───────────────────────────────────────────────

    def _render_image(self, block: Block):
        """
        Изображение (фото/логотип) → краткий блок.

        Ожидает в block.metadata:
          - image_path: путь к изображению
          - ocr_text: текст, извлечённый OCR (если есть)
        """
        meta = block.metadata or {}
        image_path = meta.get("image_path", meta.get("path", ""))
        ocr_text = meta.get("ocr_text", "")

        if image_path:
            self._lines.append(
                f"![Изображение со стр. {block.page_num}]({image_path})"
            )
        else:
            self._lines.append(f"*[Изображение со стр. {block.page_num}]*")

        if ocr_text:
            ocr_preview = ocr_text[:100]
            if len(ocr_text) > 100:
                ocr_preview += "..."
            self._lines.append(f"*Текст на изображении:* {ocr_preview}")

        self._lines.append("")

    # ─── save ────────────────────────────────────────────────

    def save(self, markdown_text: str, output_path: str):
        """Сохранить Markdown в файл."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        logger.info(f"Markdown saved to {output_path}")

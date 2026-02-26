"""
Markdown Builder Module
=======================
Converts a list of Block objects (from layout analysis) into a structured
Markdown document, preserving headers, paragraphs, lists, tables, and images.
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
      - header  → ## Заголовок
      - paragraph → обычный текст
      - list_item → - пункт списка
      - table → markdown-таблица
      - image → ![image](path)
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
        if not block.text and block.block_type != "image":
            return

        if block.block_type == "header":
            self._render_header(block)
        elif block.block_type == "paragraph":
            self._render_paragraph(block)
        elif block.block_type == "list_item":
            self._render_list_item(block)
        elif block.block_type == "table":
            self._render_table(block)
        elif block.block_type == "image":
            self._render_image(block)
        elif block.block_type == "empty":
            pass  # skip empty blocks
        else:
            # Fallback: paragraph
            self._render_paragraph(block)

    def _render_header(self, block: Block):
        text = block.text.strip()
        self._lines.append(f"## {text}")
        self._lines.append("")

    def _render_paragraph(self, block: Block):
        text = block.text.strip()
        self._lines.append(text)
        self._lines.append("")

    def _render_list_item(self, block: Block):
        text = block.text.strip()
        # Убираем маркер если он есть, и подставляем стандартный
        for marker in ["•", "–", "—", "-"]:
            if text.startswith(marker):
                text = text[len(marker):].strip()
                break
        self._lines.append(f"- {text}")

    def _render_table(self, block: Block):
        """
        Рендерит табличные данные.
        Простая эвристика: разбиваем по пробелам/табам.
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

    def _render_image(self, block: Block):
        """Рендерит изображение."""
        image_path = ""
        if block.metadata and "path" in block.metadata:
            image_path = block.metadata["path"]
        elif block.metadata and "source" in block.metadata:
            image_path = f"image_p{block.page_num}"

        self._lines.append(f"![Изображение со стр. {block.page_num}]({image_path})")
        self._lines.append("")

    def save(self, markdown_text: str, output_path: str):
        """Сохранить Markdown в файл."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        logger.info(f"Markdown saved to {output_path}")

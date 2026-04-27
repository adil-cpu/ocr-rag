"""
Chunker Module
==============
Splits Markdown documents into semantically meaningful chunks
for embedding and retrieval. Supports header-based and
paragraph-based splitting with overlap.
"""

import re
import uuid
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Chunk:
    """Один семантический фрагмент документа."""

    def __init__(
        self,
        text: str,
        chunk_type: str = "text",
        page: Optional[int] = None,
        section: Optional[str] = None,
        source: Optional[str] = None,
    ):
        self.chunk_id = str(uuid.uuid4())[:8]
        self.text = text
        self.chunk_type = chunk_type  # text | table | image_caption
        self.page = page
        self.section = section
        self.source = source

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "type": self.chunk_type,
            "page": self.page,
            "section": self.section,
            "source": self.source,
        }

    def __repr__(self):
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk({self.chunk_id}, type={self.chunk_type}, '{preview}...')"


class MarkdownChunker:
    """
    Разбивает Markdown-документ на чанки.

    Стратегия:
      1. Разбиение по заголовкам (## и выше) — каждый раздел
      2. Внутри раздела — по абзацам
      3. Контроль размера: max_chunk_size символов
      4. Overlap между чанками для сохранения контекста
    """

    def __init__(
        self,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 100,
        min_chunk_size: int = 50,
    ):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_markdown(self, markdown_text: str, source: str = "unknown") -> List[Chunk]:
        """
        Основной метод: разбить Markdown на чанки.

        Args:
            markdown_text: Markdown-текст документа
            source: имя исходного файла

        Returns:
            Список Chunk-объектов
        """
        chunks = []
        sections = self._split_by_headers(markdown_text)

        for section_title, section_body in sections:
            page = self._extract_page_number(section_body)

            # Определяем тип контента
            if self._is_table(section_body):
                chunk_type = "table"
            elif self._is_image_ref(section_body):
                chunk_type = "image_caption"
            else:
                chunk_type = "text"

            # Разбиваем секцию на подчанки если слишком большая
            sub_chunks = self._split_section(section_body)

            for text in sub_chunks:
                if len(text.strip()) < self.min_chunk_size:
                    continue

                chunks.append(Chunk(
                    text=text.strip(),
                    chunk_type=chunk_type,
                    page=page,
                    section=section_title,
                    source=source,
                ))

        logger.info(f"Created {len(chunks)} chunks from '{source}'")
        return chunks

    def chunk_text_simple(self, text: str, source: str = "unknown") -> List[Chunk]:
        """
        Простое разбиение текста (без Markdown-обработки).
        Разбивает по абзацам с контролем размера.
        """
        chunks = []
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 > self.max_chunk_size:
                if current_chunk:
                    chunks.append(Chunk(
                        text=current_chunk.strip(),
                        chunk_type="text",
                        source=source,
                    ))
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para

        # Последний чанк
        if current_chunk and len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(Chunk(
                text=current_chunk.strip(),
                chunk_type="text",
                source=source,
            ))

        logger.info(f"Created {len(chunks)} chunks (simple mode) from '{source}'")
        return chunks

    def create_image_chunk(
        self,
        text: str,
        page: Optional[int] = None,
        section: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Chunk:
        """
        Создать один Chunk из описания изображения/графика.

        Args:
            text: текстовое описание изображения
            page: номер страницы
            section: секция документа
            source: имя исходного файла
        """
        return Chunk(
            text=text,
            chunk_type="image_caption",
            page=page,
            section=section or "Изображения и графики",
            source=source,
        )

    def create_image_chunks(
        self,
        chart_analyses: list,
        source: str = "unknown",
    ) -> List[Chunk]:
        """
        Создать чанки из результатов анализа графиков (ChartAnalysisResult).

        Args:
            chart_analyses: список ChartAnalysisResult из ChartAnalyzer
            source: имя исходного файла

        Returns:
            Список Chunk с типом image_caption
        """
        chunks = []
        for analysis in chart_analyses:
            chunk_text = analysis.to_chunk_text()
            if len(chunk_text.strip()) < self.min_chunk_size:
                continue

            chunk = self.create_image_chunk(
                text=chunk_text,
                page=analysis.page_num,
                source=source,
            )
            chunks.append(chunk)

        logger.info(
            f"Created {len(chunks)} image chunks from '{source}'"
        )
        return chunks

    # ----------------------------------------------------------------
    # Вспомогательные методы
    # ----------------------------------------------------------------

    def _split_by_headers(self, text: str) -> List[tuple]:
        """
        Разбивает по Markdown-заголовкам (## и выше).
        Возвращает: [(title, body), ...]
        """
        pattern = r"^(#{1,3})\s+(.+)$"
        lines = text.split("\n")
        sections = []
        current_title = "Без заголовка"
        current_body_lines = []

        for line in lines:
            match = re.match(pattern, line)
            if match:
                # Сохраняем предыдущую секцию
                if current_body_lines:
                    sections.append((current_title, "\n".join(current_body_lines)))
                current_title = match.group(2).strip()
                current_body_lines = []
            else:
                current_body_lines.append(line)

        # Последняя секция
        if current_body_lines:
            sections.append((current_title, "\n".join(current_body_lines)))

        return sections

    def _split_section(self, text: str) -> List[str]:
        """
        Разбивает секцию на подчанки, если она слишком большая.
        Использует абзацы (двойной перенос строки) и overlap.
        """
        if len(text) <= self.max_chunk_size:
            return [text]

        paragraphs = text.split("\n\n")
        chunks = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current) + len(para) + 2 > self.max_chunk_size:
                if current:
                    chunks.append(current)
                    # Overlap: берём последние N символов предыдущего чанка
                    overlap_text = current[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                    current = overlap_text + "\n\n" + para if overlap_text else para
                else:
                    # Параграф сам по себе больше max — разбиваем по предложениям
                    sentence_chunks = self._split_by_sentences(para)
                    chunks.extend(sentence_chunks[:-1])
                    current = sentence_chunks[-1] if sentence_chunks else ""
            else:
                current += "\n\n" + para if current else para

        if current:
            chunks.append(current)

        return chunks

    def _split_by_sentences(self, text: str) -> List[str]:
        """Разбиение по предложениям (fallback для очень длинных абзацев)."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current = ""

        for sent in sentences:
            if len(current) + len(sent) + 1 > self.max_chunk_size:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current += " " + sent if current else sent

        if current:
            chunks.append(current)

        return chunks

    def _extract_page_number(self, text: str) -> Optional[int]:
        """Извлечь номер страницы из маркера *Страница N*."""
        match = re.search(r"\*Страница\s+(\d+)\*", text)
        if match:
            return int(match.group(1))
        return None

    def _is_table(self, text: str) -> bool:
        """Проверить, содержит ли текст markdown-таблицу."""
        return bool(re.search(r"\|.*\|.*\|", text))

    def _is_image_ref(self, text: str) -> bool:
        """Проверить, содержит ли текст ссылку на изображение."""
        return bool(re.search(r"!\[.*\]\(.*\)", text))

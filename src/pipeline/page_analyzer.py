"""
Page Analyzer Module (Этап 5)
==============================
Разбор одной страницы PDF:
  - извлечение текстовых блоков и изображений через PyMuPDF
  - классификация текстовых блоков через BlockClassifier (этап 1)
  - изображения помечаются как 'no_text' (этап 2 определит chart/image через CLIP)
"""

import fitz
from typing import List

from src.layout.block import Block
from src.layout.classifier import BlockClassifier


class PageAnalyzer:
    """
    Разбор одной страницы PDF:
    - тип 0 (текст) → BlockClassifier → header / text / list / table / no_text
    - тип 1 (изображение) → no_text (будет классифицирован CLIP позже)
    """

    def __init__(self):
        self.classifier = BlockClassifier()

    def analyze_page(self, page: fitz.Page, page_num: int) -> List[Block]:
        result = []

        page_dict = page.get_text("dict")

        for b in page_dict["blocks"]:
            bbox = tuple(b["bbox"])

            # === ТЕКСТ ===
            if b["type"] == 0:
                text = self._extract_text(b)
                if not text:
                    continue

                block_type = self.classifier.classify(text)

                result.append(
                    Block(
                        block_type=block_type,
                        bbox=bbox,
                        page_num=page_num,
                        text=text
                    )
                )

            # === ИЗОБРАЖЕНИЯ → no_text (этап 2 определит chart/image) ===
            elif b["type"] == 1:
                result.append(
                    Block(
                        block_type="no_text",
                        bbox=bbox,
                        page_num=page_num,
                        metadata={"source": "pdf"}
                    )
                )

        return result

    def _extract_text(self, block: dict) -> str:
        lines = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                lines.append(span["text"])
        return " ".join(lines).strip()

"""
OCR Processor Module
====================
Wraps pytesseract for multilingual OCR (Russian, English, Kazakh).
Accepts PIL images, returns text with optional bounding-box data.
"""

import pytesseract
from PIL import Image
from typing import List, Dict, Optional
import os
import logging

logger = logging.getLogger(__name__)


class OCRProcessor:
    """
    OCR-процессор на основе Tesseract.
    Поддержка языков: русский, английский, казахский.
    """

    def __init__(self, languages: str = "rus+eng+kaz"):
        """
        Args:
            languages: строка языков Tesseract (e.g. "rus+eng", "rus+eng+kaz")
        """
        self.languages = languages
        self.config = "--oem 3 --psm 6"  # LSTM engine, uniform block

    def extract_text(self, image: Image.Image) -> str:
        """
        Извлечь текст из PIL Image.

        Args:
            image: PIL Image объект (страница или вырезка)

        Returns:
            Распознанный текст
        """
        try:
            text = pytesseract.image_to_string(
                image,
                lang=self.languages,
                config=self.config
            )
            return text.strip()
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""

    def extract_with_boxes(self, image: Image.Image) -> List[Dict]:
        """
        Извлечь текст с bounding-box координатами и confidence.

        Returns:
            Список словарей: {text, x, y, w, h, conf}
        """
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.languages,
                config=self.config,
                output_type=pytesseract.Output.DICT
            )

            results = []
            n_boxes = len(data["text"])
            for i in range(n_boxes):
                text = data["text"][i].strip()
                conf = int(data["conf"][i])

                if text and conf > 30:  # фильтр по уверенности
                    results.append({
                        "text": text,
                        "x": data["left"][i],
                        "y": data["top"][i],
                        "w": data["width"][i],
                        "h": data["height"][i],
                        "conf": conf,
                        "block_num": data["block_num"][i],
                        "line_num": data["line_num"][i],
                    })

            return results
        except Exception as e:
            logger.error(f"OCR with boxes failed: {e}")
            return []

    def extract_from_file(self, image_path: str) -> str:
        """
        Извлечь текст из файла изображения.
        """
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return ""

        image = Image.open(image_path)
        return self.extract_text(image)

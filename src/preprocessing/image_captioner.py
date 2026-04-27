"""
Image Captioner Module
======================
Генерация текстовых подписей для изображений, извлечённых из PDF.

Стратегии:
1. OCR — если на изображении есть текст (графики с подписями, таблицы)
2. CLIP — семантическое описание через CLIP-модель
3. LLM — описание через GPT API (опционально)

Использование:
    captioner = ImageCaptioner()
    caption = captioner.generate_caption(pil_image, strategy="auto")
"""

import logging
from typing import Optional, List, Dict
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class ImageCaptioner:
    """
    Генерация подписей к изображениям.

    Автоматически выбирает стратегию:
    - Если OCR извлёк текст (> 20 символов) → используем OCR-текст
    - Иначе → CLIP категоризация + шаблонное описание
    """

    # Категории для CLIP zero-shot классификации
    CATEGORIES = [
        "график или диаграмма",
        "таблица с данными",
        "блок-схема или алгоритм",
        "фотография",
        "логотип или иконка",
        "математическая формула",
        "скриншот интерфейса",
        "карта или план",
        "схема архитектуры",
        "рукописный текст",
    ]

    CATEGORY_TEMPLATES = {
        "график или диаграмма": "График: {details}",
        "таблица с данными": "Таблица: {details}",
        "блок-схема или алгоритм": "Блок-схема: {details}",
        "фотография": "Фотография: {details}",
        "логотип или иконка": "Логотип",
        "математическая формула": "Математическая формула: {details}",
        "скриншот интерфейса": "Скриншот: {details}",
        "карта или план": "Карта/план: {details}",
        "схема архитектуры": "Схема архитектуры: {details}",
        "рукописный текст": "Рукописный текст: {details}",
    }

    def __init__(self, ocr_processor=None, use_clip: bool = True):
        """
        Args:
            ocr_processor: экземпляр OCRProcessor (если None — создаётся новый)
            use_clip: использовать ли CLIP для классификации
        """
        self.ocr_processor = ocr_processor
        self.use_clip = use_clip
        self._clip_model = None
        self._clip_processor = None

    def _get_ocr(self):
        """Ленивая загрузка OCR."""
        if self.ocr_processor is None:
            from src.preprocessing.ocr_processor import OCRProcessor
            self.ocr_processor = OCRProcessor()
        return self.ocr_processor

    def _get_clip(self):
        """Ленивая загрузка CLIP."""
        if self._clip_model is None and self.use_clip:
            try:
                from transformers import CLIPProcessor, CLIPModel
                logger.info("Загрузка CLIP модели...")
                self._clip_model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
                self._clip_processor = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
                logger.info("CLIP модель загружена.")
            except Exception as e:
                logger.warning(f"CLIP недоступен: {e}")
                self.use_clip = False
        return self._clip_model, self._clip_processor

    def generate_caption(
        self,
        image: Image.Image,
        strategy: str = "auto",
        page_num: Optional[int] = None,
    ) -> Dict:
        """
        Сгенерировать подпись к изображению.

        Args:
            image: PIL Image
            strategy: "auto" | "ocr" | "clip" | "combined"
            page_num: номер страницы (для метаданных)

        Returns:
            {
                "caption": str,       # текстовая подпись
                "strategy": str,      # использованная стратегия
                "ocr_text": str,      # текст OCR (если есть)
                "clip_category": str, # категория CLIP (если есть)
                "confidence": float,  # уверенность (0-1)
                "page_num": int,      # страница
            }
        """
        result = {
            "caption": "",
            "strategy": strategy,
            "ocr_text": "",
            "clip_category": "",
            "confidence": 0.0,
            "page_num": page_num,
        }

        # 1. Попробовать OCR
        ocr_text = ""
        if strategy in ("auto", "ocr", "combined"):
            ocr_text = self._try_ocr(image)
            result["ocr_text"] = ocr_text

        # 2. Попробовать CLIP
        clip_category = ""
        clip_confidence = 0.0
        if strategy in ("auto", "clip", "combined") and self.use_clip:
            clip_category, clip_confidence = self._try_clip(image)
            result["clip_category"] = clip_category
            result["confidence"] = clip_confidence

        # 3. Выбор стратегии для подписи
        if strategy == "auto":
            if len(ocr_text) > 20:
                # Текст есть → используем OCR
                result["caption"] = self._format_ocr_caption(ocr_text, clip_category)
                result["strategy"] = "ocr"
            elif clip_category:
                # Текста нет → используем CLIP
                result["caption"] = self._format_clip_caption(
                    clip_category, ocr_text
                )
                result["strategy"] = "clip"
            else:
                # Ничего не сработало
                result["caption"] = f"Изображение со стр. {(page_num or 0) + 1}"
                result["strategy"] = "fallback"
        elif strategy == "ocr":
            result["caption"] = self._format_ocr_caption(ocr_text, "")
        elif strategy == "clip":
            result["caption"] = self._format_clip_caption(clip_category, ocr_text)
        elif strategy == "combined":
            result["caption"] = self._format_combined(ocr_text, clip_category)

        return result

    def generate_captions_batch(
        self,
        images: list,
        strategy: str = "auto",
    ) -> List[Dict]:
        """
        Подписи для списка ExtractedImage.

        Args:
            images: список ExtractedImage из ImageExtractor
            strategy: стратегия генерации
        """
        results = []
        for i, img_data in enumerate(images):
            logger.info(
                f"  [{i+1}/{len(images)}] "
                f"Стр. {img_data.page_num + 1}, "
                f"{img_data.width}x{img_data.height}px"
            )
            caption = self.generate_caption(
                img_data.image,
                strategy=strategy,
                page_num=img_data.page_num,
            )
            caption["saved_path"] = img_data.saved_path
            caption["width"] = img_data.width
            caption["height"] = img_data.height
            results.append(caption)
        return results

    # ─── Внутренние методы ────────────────────────────────────

    def _try_ocr(self, image: Image.Image) -> str:
        """Извлечь текст с изображения через OCR."""
        try:
            ocr = self._get_ocr()
            text = ocr.extract_text(image)
            return text.strip() if text else ""
        except Exception as e:
            logger.warning(f"OCR ошибка: {e}")
            return ""

    def _try_clip(self, image: Image.Image) -> tuple:
        """Классифицировать изображение через CLIP."""
        try:
            model, processor = self._get_clip()
            if model is None:
                return "", 0.0

            inputs = processor(
                text=self.CATEGORIES,
                images=image,
                return_tensors="pt",
                padding=True,
            )
            outputs = model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = logits.softmax(dim=0).detach().numpy()

            best_idx = probs.argmax()
            category = self.CATEGORIES[best_idx]
            confidence = float(probs[best_idx])

            return category, confidence
        except Exception as e:
            logger.warning(f"CLIP ошибка: {e}")
            return "", 0.0

    def _format_ocr_caption(self, ocr_text: str, category: str) -> str:
        """Форматирование подписи на основе OCR."""
        # Обрезаем длинный текст
        text = ocr_text[:200].strip()
        if len(ocr_text) > 200:
            text += "..."

        if category:
            return f"{category.capitalize()}: {text}"
        return text

    def _format_clip_caption(self, category: str, ocr_text: str) -> str:
        """Форматирование подписи на основе CLIP."""
        template = self.CATEGORY_TEMPLATES.get(category, "{details}")
        details = ocr_text[:100] if ocr_text else "содержимое определено по визуальным признакам"
        return template.format(details=details)

    def _format_combined(self, ocr_text: str, category: str) -> str:
        """Комбинированная подпись."""
        parts = []
        if category:
            parts.append(f"[{category}]")
        if ocr_text:
            text = ocr_text[:150]
            if len(ocr_text) > 150:
                text += "..."
            parts.append(text)
        return " ".join(parts) if parts else "Изображение без описания"

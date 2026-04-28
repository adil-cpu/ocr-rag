"""
Image Classifier Module (Этап 2)
=================================
Классификация изображений из PDF на два типа: chart или image.

Этап 1 (BlockClassifier) помечает изображения как 'no_text'.
Этап 2 (этот модуль) определяет через CLIP:
  - chart  — графики, диаграммы, гистограммы (нужна транскрипция)
  - image  — фотографии, логотипы, схемы (краткое описание)

Использование:
    classifier = ImageClassifier()
    result = classifier.classify(pil_image, page_num=5)
    print(result["block_type"])  # "chart" или "image"
"""

import logging
from typing import Optional, List, Dict
from PIL import Image

logger = logging.getLogger(__name__)


class ImageClassifier:
    """
    Двухклассовая классификация изображений через CLIP (zero-shot).

    CLIP сравнивает изображение с текстовыми описаниями двух категорий
    и выбирает ту, которая больше соответствует визуальному содержанию.
    """

    # Текстовые описания для CLIP zero-shot классификации
    # Несколько вариантов для каждой категории повышают точность
    CHART_DESCRIPTIONS = [
        "a bar chart, line chart, pie chart, or histogram with data",
        "a graph or diagram showing statistics or trends",
        "a chart with axes, labels, and numerical data",
    ]

    IMAGE_DESCRIPTIONS = [
        "a photograph, logo, icon, or illustration",
        "a screenshot, picture, or decorative image",
        "a photo of a person, building, object, or scene",
    ]

    def __init__(self, use_clip: bool = True):
        """
        Args:
            use_clip: использовать ли CLIP. Если False — все блоки
                      будут классифицированы как 'image' (fallback).
        """
        self.use_clip = use_clip
        self._clip_model = None
        self._clip_processor = None

    def _load_clip(self):
        """Ленивая загрузка CLIP модели (загружается при первом вызове)."""
        if self._clip_model is None and self.use_clip:
            try:
                from transformers import CLIPProcessor, CLIPModel
                logger.info("Загрузка CLIP модели (openai/clip-vit-base-patch32)...")
                self._clip_model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
                self._clip_processor = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
                logger.info("CLIP модель загружена.")
            except Exception as e:
                logger.warning(f"CLIP недоступен: {e}. Fallback: все изображения → 'image'.")
                self.use_clip = False

    def classify(
        self,
        image: Image.Image,
        page_num: Optional[int] = None,
    ) -> Dict:
        """
        Классифицировать одно изображение: chart или image.

        Args:
            image: PIL Image
            page_num: номер страницы (для метаданных)

        Returns:
            {
                "block_type": "chart" или "image",
                "confidence": float (0-1),
                "chart_score": float (0-1),
                "image_score": float (0-1),
                "page_num": int,
            }
        """
        result = {
            "block_type": "image",  # fallback
            "confidence": 0.0,
            "chart_score": 0.0,
            "image_score": 0.0,
            "page_num": page_num,
        }

        if not self.use_clip:
            return result

        self._load_clip()
        if self._clip_model is None:
            return result

        try:
            import torch

            # Все описания для обеих категорий
            all_descriptions = self.CHART_DESCRIPTIONS + self.IMAGE_DESCRIPTIONS

            inputs = self._clip_processor(
                text=all_descriptions,
                images=image,
                return_tensors="pt",
                padding=True,
            )

            with torch.no_grad():
                outputs = self._clip_model(**inputs)

            logits = outputs.logits_per_image[0]
            probs = logits.softmax(dim=0).detach().numpy()

            # Средняя вероятность по описаниям каждой категории
            n_chart = len(self.CHART_DESCRIPTIONS)
            chart_score = float(probs[:n_chart].mean())
            image_score = float(probs[n_chart:].mean())

            # Определение типа
            if chart_score > image_score:
                result["block_type"] = "chart"
                result["confidence"] = chart_score / (chart_score + image_score)
            else:
                result["block_type"] = "image"
                result["confidence"] = image_score / (chart_score + image_score)

            result["chart_score"] = round(chart_score, 4)
            result["image_score"] = round(image_score, 4)

            logger.info(
                f"  Стр. {(page_num or 0) + 1}: "
                f"{result['block_type']} "
                f"(chart={chart_score:.3f}, image={image_score:.3f}, "
                f"confidence={result['confidence']:.3f})"
            )

        except Exception as e:
            logger.warning(f"CLIP ошибка: {e}. Fallback: 'image'.")

        return result

    def classify_batch(
        self,
        extracted_images: list,
    ) -> List[Dict]:
        """
        Классифицировать список ExtractedImage (из ImageExtractor).

        Args:
            extracted_images: список ExtractedImage объектов

        Returns:
            Список результатов классификации
        """
        results = []
        for i, img_data in enumerate(extracted_images):
            logger.info(
                f"  Классификация [{i+1}/{len(extracted_images)}]: "
                f"стр. {img_data.page_num + 1}, "
                f"{img_data.width}x{img_data.height}px"
            )
            result = self.classify(
                image=img_data.image,
                page_num=img_data.page_num,
            )
            result["saved_path"] = img_data.saved_path
            result["width"] = img_data.width
            result["height"] = img_data.height
            results.append(result)

        # Статистика
        charts = sum(1 for r in results if r["block_type"] == "chart")
        images = sum(1 for r in results if r["block_type"] == "image")
        logger.info(f"  Итого: {charts} chart, {images} image")

        return results

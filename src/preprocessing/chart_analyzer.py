"""
Chart Analyzer Module
===============================
Анализ графиков и диаграмм, извлечённых из PDF.

Комбинирует четыре подхода:
1. OCR (Tesseract) — извлечение текста (подписи осей, легенда, числа)
2. CLIP (zero-shot) — классификация подтипа графика
3. BLIP — генерация описания изображения (на английском)
4. GPT API — перевод и дополнение описания на русском языке

Поток:
    OCR → текст с графика
    CLIP → подтип (столбчатая, линейная, круговая...)
    BLIP → "This image shows a bar chart with..."
    GPT → берёт BLIP + OCR + CLIP → русское описание

Использование:
    analyzer = ChartAnalyzer()
    result = analyzer.analyze(pil_image, page_num=5)
    print(result["gpt_description"])  # описание на русском
    print(result["chunk_text"])       # текст для FAISS
"""

import os
import json
import re
import logging
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)


# ─── Подтипы графиков для CLIP ──────────────────────────────

CHART_SUBTYPES = [
    "bar chart or column chart",
    "line chart or line graph",
    "pie chart or donut chart",
    "scatter plot or dot plot",
    "histogram",
    "area chart",
    "flowchart or block diagram",
    "table with data",
    "organizational chart or tree diagram",
]

CHART_SUBTYPES_RU = {
    "bar chart or column chart": "Столбчатая/полосовая диаграмма",
    "line chart or line graph": "Линейный график",
    "pie chart or donut chart": "Круговая диаграмма",
    "scatter plot or dot plot": "Точечная диаграмма",
    "histogram": "Гистограмма",
    "area chart": "Диаграмма с областями",
    "flowchart or block diagram": "Блок-схема",
    "table with data": "Таблица с данными",
    "organizational chart or tree diagram": "Организационная диаграмма",
}


# ─── Промпт для GPT ─────────────────────────────────────────

CHART_DESCRIPTION_PROMPT = """Ты — ассистент для анализа графиков и диаграмм из документов.

Тебе предоставлены данные, полученные автоматическим анализом изображения графика:
- Тип графика (CLIP): {chart_type}
- Описание от BLIP (на английском): {blip_caption}
- Текст, распознанный OCR с изображения: {ocr_text}

На основе ВСЕХ этих данных напиши подробное описание графика на РУССКОМ языке.
Включи:
1. Тип визуализации
2. Что отображается (какие данные, оси, единицы измерения)
3. Конкретные значения и числа из OCR-текста
4. Основные тренды или выводы (если можно определить)

Если OCR-текст неполный или содержит мусор, используй только достоверную информацию.
Ответ должен быть кратким (3-5 предложений), информативным и на русском языке."""


@dataclass
class ChartAnalysisResult:
    """Результат анализа графика/диаграммы."""

    page_num: int
    chart_subtype: str = "unknown"          # подтип из CLIP (bar/line/pie...)
    chart_subtype_ru: str = "Неизвестно"    # русский вариант
    confidence: float = 0.0                 # уверенность CLIP (0-1)
    ocr_text: str = ""                      # текст, извлечённый OCR
    blip_caption: str = ""                  # описание от BLIP (английский)
    gpt_description: str = ""               # описание от GPT (русский)
    dominant_colors: List[str] = field(default_factory=list)
    num_color_segments: int = 0
    has_lines: bool = False
    num_lines: int = 0
    has_text: bool = False
    image_path: Optional[str] = None

    def to_chunk_text(self) -> str:
        """Сформировать текст для индексации в FAISS."""
        parts = []
        parts.append(f"[Тип: {self.chart_subtype_ru}] [Стр. {self.page_num}]")

        if self.gpt_description:
            parts.append(f"Описание: {self.gpt_description}")

        if self.ocr_text:
            ocr_preview = self.ocr_text[:300]
            if len(self.ocr_text) > 300:
                ocr_preview += "..."
            parts.append(f"Текст на изображении: {ocr_preview}")

        return "\n".join(parts)

    def to_dict(self) -> Dict:
        return {
            "page_num": self.page_num,
            "chart_subtype": self.chart_subtype,
            "chart_subtype_ru": self.chart_subtype_ru,
            "confidence": self.confidence,
            "ocr_text": self.ocr_text[:200],
            "blip_caption": self.blip_caption,
            "gpt_description": self.gpt_description,
            "dominant_colors": self.dominant_colors,
            "num_color_segments": self.num_color_segments,
            "has_lines": self.has_lines,
            "num_lines": self.num_lines,
            "has_text": self.has_text,
            "image_path": self.image_path,
        }


class ChartAnalyzer:
    """
    Анализатор графиков и диаграмм из PDF-документов.

    Объединяет:
    - OCR (Tesseract) для текста на изображении
    - CLIP (zero-shot) для определения подтипа графика
    - BLIP для генерации описания изображения (на английском)
    - GPT API для перевода и дополнения описания на русском
    - OpenCV для структурного анализа (цвета, линии)
    """

    def __init__(
        self,
        ocr_processor=None,
        use_clip: bool = True,
        use_blip: bool = True,
        use_opencv: bool = True,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Args:
            ocr_processor: экземпляр OCRProcessor (None — создастся автоматически)
            use_clip: использовать CLIP для определения подтипа графика
            use_blip: использовать BLIP для генерации описания
            use_opencv: использовать OpenCV для структурного анализа
            api_url: URL GPT API (по умолчанию из .env)
            api_key: API ключ (по умолчанию из .env)
        """
        self.ocr_processor = ocr_processor
        self.use_clip = use_clip
        self.use_blip = use_blip
        self.use_opencv = use_opencv
        self.api_url = api_url or os.getenv(
            "LLM_API_URL", "https://gpt.serverspace.kz/v1/chat/completions"
        )
        self.api_key = api_key or os.getenv(
            "LLM_API_KEY", ""
        )
        self._clip_model = None
        self._clip_processor = None
        self._blip_model = None
        self._blip_processor = None

    # ────────────────────────────────────────────────────────────
    # Публичный API
    # ────────────────────────────────────────────────────────────

    def analyze(
        self,
        image: Image.Image,
        page_num: int = 0,
        image_path: Optional[str] = None,
    ) -> ChartAnalysisResult:
        """
        Полный анализ одного графика/диаграммы.

        Этапы:
            1. OCR — извлечь текст (подписи, числа)
            2. CLIP — определить подтип (столбчатая, линейная, круговая...)
            3. BLIP — сгенерировать описание на английском
            4. GPT — перевести и дополнить описание на русском
            5. OpenCV — структурный анализ (цвета, линии)

        Args:
            image: PIL Image
            page_num: номер страницы (1-based)
            image_path: путь к сохранённому файлу

        Returns:
            ChartAnalysisResult
        """
        result = ChartAnalysisResult(
            page_num=page_num,
            image_path=image_path,
        )

        # 1. OCR — извлечь текст
        ocr_text = self._extract_ocr(image)
        result.ocr_text = ocr_text
        result.has_text = len(ocr_text.strip()) > 10

        # 2. CLIP — определить подтип графика
        if self.use_clip:
            chart_subtype, confidence = self._classify_chart_subtype(image)
            result.chart_subtype = chart_subtype
            result.chart_subtype_ru = CHART_SUBTYPES_RU.get(
                chart_subtype, chart_subtype
            )
            result.confidence = confidence

        # 3. BLIP — описание на английском
        if self.use_blip:
            result.blip_caption = self._generate_blip_caption(image)

        # 4. GPT — перевод и описание на русском
        result.gpt_description = self._generate_gpt_description(
            chart_type_ru=result.chart_subtype_ru,
            blip_caption=result.blip_caption,
            ocr_text=ocr_text,
        )

        # 5. OpenCV — структурный анализ
        if self.use_opencv:
            cv_features = self._analyze_structure(image)
            result.dominant_colors = cv_features.get("dominant_colors", [])
            result.num_color_segments = cv_features.get("num_color_segments", 0)
            result.has_lines = cv_features.get("has_lines", False)
            result.num_lines = cv_features.get("num_lines", 0)

        return result

    def analyze_batch(
        self,
        extracted_images: list,
    ) -> List[ChartAnalysisResult]:
        """
        Анализ списка ExtractedImage (только тех, что классифицированы как chart).

        Args:
            extracted_images: список ExtractedImage объектов

        Returns:
            Список ChartAnalysisResult
        """
        results = []
        for i, img_data in enumerate(extracted_images):
            logger.info(
                f"  Анализ графика [{i+1}/{len(extracted_images)}]: "
                f"стр. {img_data.page_num + 1}, "
                f"{img_data.width}x{img_data.height}px"
            )
            result = self.analyze(
                image=img_data.image,
                page_num=img_data.page_num + 1,  # 0-based → 1-based
                image_path=img_data.saved_path,
            )
            results.append(result)
        return results

    # ────────────────────────────────────────────────────────────
    # OCR
    # ────────────────────────────────────────────────────────────

    def _get_ocr(self):
        """Ленивая загрузка OCR-процессора."""
        if self.ocr_processor is None:
            from src.preprocessing.ocr_processor import OCRProcessor
            self.ocr_processor = OCRProcessor()
        return self.ocr_processor

    def _extract_ocr(self, image: Image.Image) -> str:
        """Извлечь текст из изображения через OCR."""
        try:
            ocr = self._get_ocr()
            text = ocr.extract_text(image)
            if text:
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                lines = [line for line in lines if len(line) > 2 or line.isdigit()]
                return "\n".join(lines)
            return ""
        except Exception as e:
            logger.warning(f"OCR ошибка: {e}")
            return ""

    # ────────────────────────────────────────────────────────────
    # BLIP — описание изображения (на английском)
    # ────────────────────────────────────────────────────────────

    def _load_blip(self):
        """Ленивая загрузка BLIP модели."""
        if self._blip_model is None and self.use_blip:
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                logger.info("Загрузка BLIP (Salesforce/blip-image-captioning-large)...")
                self._blip_processor = BlipProcessor.from_pretrained(
                    "Salesforce/blip-image-captioning-large"
                )
                self._blip_model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-large"
                )
                logger.info("BLIP загружен.")
            except Exception as e:
                logger.warning(f"BLIP недоступен: {e}")
                self.use_blip = False

    def _generate_blip_caption(self, image: Image.Image) -> str:
        """
        Сгенерировать описание изображения через BLIP.
        Результат на английском — будет переведён GPT.
        """
        try:
            self._load_blip()
            if self._blip_model is None:
                return ""

            prompt = "This image shows"
            inputs = self._blip_processor(
                image, prompt, return_tensors="pt"
            )
            output = self._blip_model.generate(
                **inputs, max_new_tokens=100
            )
            caption = self._blip_processor.decode(
                output[0], skip_special_tokens=True
            )

            logger.debug(f"BLIP: {caption}")
            return caption.strip()
        except Exception as e:
            logger.warning(f"BLIP генерация не удалась: {e}")
            return ""

    # ────────────────────────────────────────────────────────────
    # CLIP — подтип графика
    # ────────────────────────────────────────────────────────────

    def _load_clip(self):
        """Ленивая загрузка CLIP."""
        if self._clip_model is None and self.use_clip:
            try:
                from transformers import CLIPProcessor, CLIPModel
                logger.info("Загрузка CLIP для анализа графиков...")
                self._clip_model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
                self._clip_processor = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
                logger.info("CLIP загружен.")
            except Exception as e:
                logger.warning(f"CLIP недоступен: {e}")
                self.use_clip = False

    def _classify_chart_subtype(self, image: Image.Image) -> Tuple[str, float]:
        """Классифицировать подтип графика через CLIP zero-shot."""
        try:
            self._load_clip()
            if self._clip_model is None:
                return "unknown", 0.0

            import torch

            inputs = self._clip_processor(
                text=CHART_SUBTYPES,
                images=image,
                return_tensors="pt",
                padding=True,
            )

            with torch.no_grad():
                outputs = self._clip_model(**inputs)

            logits = outputs.logits_per_image[0]
            probs = logits.softmax(dim=0).detach().numpy()

            best_idx = int(probs.argmax())
            category = CHART_SUBTYPES[best_idx]
            confidence = float(probs[best_idx])

            logger.debug(f"CLIP подтип: {category} (conf={confidence:.3f})")
            return category, confidence
        except Exception as e:
            logger.warning(f"CLIP классификация не удалась: {e}")
            return "unknown", 0.0

    # ────────────────────────────────────────────────────────────
    # GPT — перевод и описание на русском
    # ────────────────────────────────────────────────────────────

    def _generate_gpt_description(
        self,
        chart_type_ru: str,
        blip_caption: str,
        ocr_text: str,
    ) -> str:
        """
        Перевести и дополнить описание графика на русском через GPT API.

        Отправляет: BLIP (англ.) + OCR-текст + тип CLIP → русское описание.
        """
        if not self.api_key:
            logger.warning("GPT API ключ не задан, пропуск генерации описания")
            return self._fallback_description(chart_type_ru, blip_caption, ocr_text)

        # Если ни BLIP, ни OCR не дали результат
        if not blip_caption.strip() and not ocr_text.strip():
            return self._fallback_description(chart_type_ru, blip_caption, ocr_text)

        try:
            prompt = CHART_DESCRIPTION_PROMPT.format(
                chart_type=chart_type_ru,
                blip_caption=blip_caption or "(описание недоступно)",
                ocr_text=ocr_text[:500] if ocr_text else "(текст не распознан)",
            )

            payload = {
                "model": "gpt-oss-20b",
                "max_tokens": 300,
                "temperature": 0.3,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
            }

            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": self.api_key,
            }

            response = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=30,
            )

            if response.status_code != 200:
                logger.warning(
                    f"GPT API ошибка {response.status_code}: {response.text[:100]}"
                )
                return self._fallback_description(chart_type_ru, blip_caption, ocr_text)

            result = response.json()
            description = result["choices"][0]["message"]["content"].strip()

            logger.info(f"  GPT описание ({len(description)} символов)")
            return description

        except requests.exceptions.Timeout:
            logger.warning("GPT API timeout")
            return self._fallback_description(chart_type_ru, blip_caption, ocr_text)
        except Exception as e:
            logger.warning(f"GPT ошибка: {e}")
            return self._fallback_description(chart_type_ru, blip_caption, ocr_text)

    def _fallback_description(
        self, chart_type_ru: str, blip_caption: str, ocr_text: str
    ) -> str:
        """Описание без GPT — на основе типа, BLIP и OCR."""
        parts = [chart_type_ru]
        if blip_caption:
            parts.append(f"(BLIP: {blip_caption})")
        if ocr_text.strip():
            keywords = self._extract_keywords(ocr_text)
            if keywords:
                parts.append(f"Ключевые слова: {', '.join(keywords[:8])}")
        return ". ".join(parts) + "."

    # ────────────────────────────────────────────────────────────
    # OpenCV — структурный анализ
    # ────────────────────────────────────────────────────────────

    def _analyze_structure(self, image: Image.Image) -> Dict:
        """
        Структурный анализ через OpenCV.
        Определяет доминирующие цвета и линии.
        """
        features = {
            "dominant_colors": [],
            "num_color_segments": 0,
            "has_lines": False,
            "num_lines": 0,
        }

        try:
            import cv2

            img_array = np.array(image.convert("RGB"))
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            features.update(self._extract_colors(img_bgr))
            features.update(self._detect_lines(img_bgr))

        except ImportError:
            logger.warning("OpenCV не установлен, пропуск структурного анализа")
        except Exception as e:
            logger.warning(f"OpenCV ошибка: {e}")

        return features

    def _extract_colors(self, img_bgr: np.ndarray, k: int = 5) -> Dict:
        """K-Means кластеризация для доминирующих цветов."""
        try:
            import cv2

            small = cv2.resize(img_bgr, (100, 100))
            pixels = small.reshape(-1, 3).astype(np.float32)

            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                20, 1.0,
            )
            _, labels, centers = cv2.kmeans(
                pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
            )

            _, counts = np.unique(labels, return_counts=True)
            total = counts.sum()

            colors = []
            for center, count in sorted(
                zip(centers, counts), key=lambda x: -x[1]
            ):
                pct = count / total
                if pct < 0.05:
                    continue
                b, g, r = int(center[0]), int(center[1]), int(center[2])
                if (r > 230 and g > 230 and b > 230):
                    continue
                if (r < 25 and g < 25 and b < 25):
                    continue
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                color_name = self._color_name(r, g, b)
                colors.append(f"{color_name} ({hex_color}, {pct:.0%})")

            return {
                "dominant_colors": colors[:5],
                "num_color_segments": len(colors),
            }
        except Exception as e:
            logger.debug(f"Анализ цветов не удался: {e}")
            return {"dominant_colors": [], "num_color_segments": 0}

    def _detect_lines(self, img_bgr: np.ndarray) -> Dict:
        """Детекция прямых линий через Hough Transform."""
        try:
            import cv2

            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=50,
                minLineLength=40,
                maxLineGap=10,
            )

            num_lines = len(lines) if lines is not None else 0

            return {
                "has_lines": num_lines > 2,
                "num_lines": num_lines,
            }
        except Exception as e:
            logger.debug(f"Детекция линий не удалась: {e}")
            return {"has_lines": False, "num_lines": 0}

    # ────────────────────────────────────────────────────────────
    # Утилиты
    # ────────────────────────────────────────────────────────────

    def _extract_keywords(self, text: str) -> List[str]:
        """Извлечь ключевые слова из OCR-текста."""
        words = re.findall(r"[а-яА-ЯёЁa-zA-Z]{3,}", text)
        seen = set()
        unique = []
        for w in words:
            wl = w.lower()
            if wl not in seen:
                seen.add(wl)
                unique.append(w)
        return unique

    @staticmethod
    def _color_name(r: int, g: int, b: int) -> str:
        """Приблизительное название цвета по RGB."""
        if r > 200 and g < 80 and b < 80:
            return "красный"
        if r < 80 and g > 200 and b < 80:
            return "зелёный"
        if r < 80 and g < 80 and b > 200:
            return "синий"
        if r > 200 and g > 200 and b < 80:
            return "жёлтый"
        if r > 200 and g > 100 and b < 80:
            return "оранжевый"
        if r > 150 and g < 80 and b > 150:
            return "фиолетовый"
        if r < 80 and g > 150 and b > 150:
            return "бирюзовый"
        if 80 < r < 180 and 80 < g < 180 and 80 < b < 180:
            return "серый"
        if r > 200 and g > 150 and b > 150:
            return "розовый"
        if r < 100 and g < 100 and b < 100:
            return "тёмный"
        return "цветной"

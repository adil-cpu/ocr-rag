"""
Chart Analyzer Module
=====================
Анализ графиков и диаграмм, извлечённых из PDF.

Комбинирует три подхода:
1. OCR — извлечение текста (подписи осей, легенда, числа)
2. CLIP — классификация типа графика (zero-shot)
3. OpenCV — структурный анализ (цвета, линии, контуры)

Использование:
    analyzer = ChartAnalyzer()
    result = analyzer.analyze(pil_image, page_num=5)
    print(result["description"])
"""

import logging
import re
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ─── Категории графиков для CLIP ──────────────────────────────

CHART_CATEGORIES = [
    "bar chart or column chart",
    "line chart or line graph",
    "pie chart or donut chart",
    "scatter plot or dot plot",
    "flowchart or block diagram",
    "table with data",
    "histogram",
    "area chart",
    "organizational chart or tree diagram",
    "map or geographic visualization",
    "photograph or picture",
    "mathematical formula or equation",
]

CHART_CATEGORY_LABELS_RU = {
    "bar chart or column chart": "Столбчатая/полосовая диаграмма",
    "line chart or line graph": "Линейный график",
    "pie chart or donut chart": "Круговая диаграмма",
    "scatter plot or dot plot": "Точечная диаграмма (scatter plot)",
    "flowchart or block diagram": "Блок-схема / диаграмма процесса",
    "table with data": "Таблица с данными",
    "histogram": "Гистограмма",
    "area chart": "Диаграмма с областями",
    "organizational chart or tree diagram": "Организационная диаграмма / дерево",
    "map or geographic visualization": "Карта / географическая визуализация",
    "photograph or picture": "Фотография / иллюстрация",
    "mathematical formula or equation": "Математическая формула",
}


@dataclass
class ChartAnalysisResult:
    """Результат анализа графика/изображения."""

    page_num: int
    chart_type: str = "unknown"            # классификация из CLIP
    chart_type_ru: str = "Неизвестно"      # русский вариант
    confidence: float = 0.0                # уверенность CLIP (0-1)
    ocr_text: str = ""                     # текст, извлечённый OCR
    blip_caption: str = ""                 # описание от BLIP (нейросетевое)
    dominant_colors: List[str] = field(default_factory=list)
    num_color_segments: int = 0
    has_lines: bool = False
    num_lines: int = 0
    has_text: bool = False
    description: str = ""                  # итоговое текстовое описание
    image_path: Optional[str] = None

    def to_chunk_text(self) -> str:
        """Сформировать текст для индексации в FAISS."""
        parts = []
        parts.append(f"[Тип: {self.chart_type_ru}] [Стр. {self.page_num}]")

        if self.blip_caption:
            parts.append(f"Содержание изображения: {self.blip_caption}")

        if self.ocr_text:
            ocr_preview = self.ocr_text[:300]
            if len(self.ocr_text) > 300:
                ocr_preview += "..."
            parts.append(f"Текст на изображении: {ocr_preview}")

        if self.description:
            parts.append(f"Анализ: {self.description}")

        return "\n".join(parts)

    def to_dict(self) -> Dict:
        return {
            "page_num": self.page_num,
            "chart_type": self.chart_type,
            "chart_type_ru": self.chart_type_ru,
            "confidence": self.confidence,
            "ocr_text": self.ocr_text[:200],
            "blip_caption": self.blip_caption,
            "dominant_colors": self.dominant_colors,
            "num_color_segments": self.num_color_segments,
            "has_lines": self.has_lines,
            "num_lines": self.num_lines,
            "has_text": self.has_text,
            "description": self.description,
            "image_path": self.image_path,
        }


class ChartAnalyzer:
    """
    Анализатор графиков и диаграмм из PDF-документов.

    Объединяет:
    - OCR (Tesseract) для текста на изображении
    - CLIP (zero-shot) для определения типа графика
    - OpenCV для структурного анализа (цвета, линии)
    """

    def __init__(
        self,
        ocr_processor=None,
        use_clip: bool = True,
        use_opencv: bool = True,
        use_blip: bool = True,
    ):
        """
        Args:
            ocr_processor: экземпляр OCRProcessor (None → создастся автоматически)
            use_clip: использовать CLIP для классификации типа графика
            use_opencv: использовать OpenCV для структурного анализа
            use_blip: использовать BLIP для генерации описаний
        """
        self.ocr_processor = ocr_processor
        self.use_clip = use_clip
        self.use_opencv = use_opencv
        self.use_blip = use_blip
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
        Полный анализ одного изображения.

        Args:
            image: PIL Image
            page_num: номер страницы (1-based)
            image_path: путь к сохранённому файлу (если есть)

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

        # 2. CLIP — определить тип
        if self.use_clip:
            chart_type, confidence = self._classify_chart_type(image)
            result.chart_type = chart_type
            result.chart_type_ru = CHART_CATEGORY_LABELS_RU.get(
                chart_type, chart_type
            )
            result.confidence = confidence

        # 3. BLIP — нейросетевое описание изображения
        if self.use_blip:
            result.blip_caption = self._generate_blip_caption(image)

        # 4. OpenCV — структурный анализ
        if self.use_opencv:
            cv_features = self._analyze_structure(image)
            result.dominant_colors = cv_features.get("dominant_colors", [])
            result.num_color_segments = cv_features.get("num_color_segments", 0)
            result.has_lines = cv_features.get("has_lines", False)
            result.num_lines = cv_features.get("num_lines", 0)

        # 5. Генерация итогового описания
        result.description = self._generate_description(result)

        return result

    def analyze_batch(
        self,
        extracted_images: list,
    ) -> List[ChartAnalysisResult]:
        """
        Анализ списка ExtractedImage (из ImageExtractor).

        Args:
            extracted_images: список ExtractedImage объектов

        Returns:
            Список ChartAnalysisResult
        """
        results = []
        for i, img_data in enumerate(extracted_images):
            logger.info(
                f"  Анализ [{i+1}/{len(extracted_images)}]: "
                f"стр. {img_data.page_num + 1}, "
                f"{img_data.width}×{img_data.height}px"
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
            # Очистка: убираем лишние пустые строки и мусор
            if text:
                lines = [l.strip() for l in text.split("\n") if l.strip()]
                # Отфильтровать строки из 1-2 символов мусора
                lines = [l for l in lines if len(l) > 2 or l.isdigit()]
                return "\n".join(lines)
            return ""
        except Exception as e:
            logger.warning(f"OCR ошибка: {e}")
            return ""

    # ────────────────────────────────────────────────────────────
    # BLIP — генерация описаний изображений
    # ────────────────────────────────────────────────────────────

    def _load_blip(self):
        """Ленивая загрузка BLIP модели."""
        if self._blip_model is None and self.use_blip:
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                logger.info("Загрузка BLIP для генерации описаний...")
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
        Сгенерировать текстовое описание изображения через BLIP.
        Используется conditional captioning с промптом.
        """
        try:
            self._load_blip()
            if self._blip_model is None:
                return ""

            # Conditional captioning — описание с подсказкой
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
    # CLIP
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

    def _classify_chart_type(self, image: Image.Image) -> Tuple[str, float]:
        """Классифицировать тип графика через CLIP zero-shot."""
        try:
            self._load_clip()
            if self._clip_model is None:
                return "unknown", 0.0

            inputs = self._clip_processor(
                text=CHART_CATEGORIES,
                images=image,
                return_tensors="pt",
                padding=True,
            )
            outputs = self._clip_model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = logits.softmax(dim=0).detach().numpy()

            best_idx = int(probs.argmax())
            category = CHART_CATEGORIES[best_idx]
            confidence = float(probs[best_idx])

            logger.debug(
                f"CLIP: {category} (conf={confidence:.3f})"
            )
            return category, confidence
        except Exception as e:
            logger.warning(f"CLIP классификация не удалась: {e}")
            return "unknown", 0.0

    # ────────────────────────────────────────────────────────────
    # OpenCV — структурный анализ
    # ────────────────────────────────────────────────────────────

    def _analyze_structure(self, image: Image.Image) -> Dict:
        """
        Структурный анализ изображения через OpenCV.
        Определяет:
        - Доминирующие цвета (K-Means кластеризация)
        - Наличие и количество линий (Hough Transform)
        """
        features = {
            "dominant_colors": [],
            "num_color_segments": 0,
            "has_lines": False,
            "num_lines": 0,
        }

        try:
            import cv2

            # PIL → OpenCV (RGB → BGR)
            img_array = np.array(image.convert("RGB"))
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # --- Доминирующие цвета ---
            features.update(self._extract_colors(img_bgr))

            # --- Детекция линий ---
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

            # Ресайз для скорости
            small = cv2.resize(img_bgr, (100, 100))
            pixels = small.reshape(-1, 3).astype(np.float32)

            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                20, 1.0,
            )
            _, labels, centers = cv2.kmeans(
                pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
            )

            # Подсчёт пикселей в каждом кластере
            _, counts = np.unique(labels, return_counts=True)
            total = counts.sum()

            # Значимые кластеры (> 5% пикселей), кроме белого/чёрного
            colors = []
            for center, count in sorted(
                zip(centers, counts), key=lambda x: -x[1]
            ):
                pct = count / total
                if pct < 0.05:
                    continue
                b, g, r = int(center[0]), int(center[1]), int(center[2])
                # Пропуск почти белого и почти чёрного
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
    # Генерация описания
    # ────────────────────────────────────────────────────────────

    def _generate_description(self, result: ChartAnalysisResult) -> str:
        """
        Сгенерировать текстовое описание на основе всех анализов.
        """
        parts = []

        # Тип графика
        if result.chart_type != "unknown":
            parts.append(
                f"{result.chart_type_ru} "
                f"(уверенность: {result.confidence:.0%})"
            )

        # Структурная информация
        struct_info = []
        if result.num_color_segments > 0:
            struct_info.append(
                f"{result.num_color_segments} цветовых категорий"
            )
        if result.has_lines:
            struct_info.append(f"{result.num_lines} прямых линий обнаружено")

        if struct_info:
            parts.append("Структура: " + ", ".join(struct_info))

        # Цвета
        if result.dominant_colors:
            parts.append(
                "Цвета: " + ", ".join(result.dominant_colors[:3])
            )

        # Наличие текста
        if result.has_text:
            # Извлекаем ключевые слова из OCR
            keywords = self._extract_keywords(result.ocr_text)
            if keywords:
                parts.append(f"Ключевые слова: {', '.join(keywords[:10])}")
        else:
            parts.append("Текст на изображении не обнаружен")

        return ". ".join(parts) + "."

    def _extract_keywords(self, text: str) -> List[str]:
        """Извлечь ключевые слова из OCR-текста."""
        # Простая эвристика: слова > 3 символов, не числа
        words = re.findall(r"[а-яА-ЯёЁa-zA-Z]{3,}", text)
        # Уникальные в порядке появления
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

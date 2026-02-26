"""
Image Extractor Module
======================
Извлечение изображений из PDF-документов через PyMuPDF (fitz).
Фильтрация артефактов, сохранение на диск, сбор метаданных.

Использование:
    extractor = ImageExtractor()
    images = extractor.extract_from_pdf("document.pdf", output_dir="data/images/doc1")
"""

import fitz
from PIL import Image
import io
import os
import hashlib
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExtractedImage:
    """Извлечённое изображение с метаданными."""
    image: Image.Image           # PIL объект
    page_num: int                # номер страницы (0-based)
    bbox: Tuple[float, float, float, float]  # координаты на странице
    width: int                   # ширина в пикселях
    height: int                  # высота в пикселях
    image_hash: str              # хеш для дедупликации
    saved_path: Optional[str] = None  # путь сохранённого файла


class ImageExtractor:
    """
    Извлечение изображений из PDF через PyMuPDF.

    Поддерживает:
    - Фильтрацию мелких изображений (артефакты, иконки)
    - Дедупликацию по хешу
    - Сохранение на диск с метаданными
    """

    def __init__(
        self,
        min_width: int = 50,
        min_height: int = 50,
        min_area: int = 5000,
    ):
        """
        Args:
            min_width: минимальная ширина (px), меньше — артефакт
            min_height: минимальная высота (px)
            min_area: минимальная площадь (px²)
        """
        self.min_width = min_width
        self.min_height = min_height
        self.min_area = min_area

    def extract_from_pdf(
        self,
        pdf_path: str,
        output_dir: Optional[str] = None,
    ) -> List[ExtractedImage]:
        """
        Извлечь все изображения из PDF.

        Args:
            pdf_path: путь к PDF-файлу
            output_dir: папка для сохранения (если None — не сохраняет)

        Returns:
            Список ExtractedImage с метаданными
        """
        doc = fitz.open(pdf_path)
        doc_name = os.path.splitext(os.path.basename(pdf_path))[0]

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        all_images: List[ExtractedImage] = []
        seen_hashes = set()

        logger.info(f"Извлечение изображений из: {pdf_path} ({len(doc)} страниц)")

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_images = self._extract_from_page(page, page_num)

            for img_data in page_images:
                # Дедупликация
                if img_data.image_hash in seen_hashes:
                    logger.debug(
                        f"  Стр. {page_num+1}: дубликат (hash={img_data.image_hash[:8]})"
                    )
                    continue
                seen_hashes.add(img_data.image_hash)

                # Сохранение
                if output_dir:
                    filename = f"{doc_name}_p{page_num+1}_img{len(all_images)+1}.png"
                    save_path = os.path.join(output_dir, filename)
                    img_data.image.save(save_path)
                    img_data.saved_path = save_path
                    logger.debug(f"  Сохранено: {save_path}")

                all_images.append(img_data)

        doc.close()

        logger.info(
            f"Извлечено {len(all_images)} уникальных изображений "
            f"(отфильтровано: {len(seen_hashes) - len(all_images)} дубликатов)"
        )
        return all_images

    def _extract_from_page(
        self, page: fitz.Page, page_num: int
    ) -> List[ExtractedImage]:
        """Извлечь изображения с одной страницы."""
        results = []
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                pix = fitz.Pixmap(page.parent, xref)

                # Конвертация CMYK → RGB
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                # Фильтрация по размерам
                if not self._is_valid_size(pix.width, pix.height):
                    continue

                # Создание PIL Image
                img_bytes = pix.tobytes("png")
                pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                # Хеш для дедупликации
                img_hash = hashlib.md5(img_bytes).hexdigest()

                # Bounding box (приблизительный)
                bbox = self._find_image_bbox(page, xref)

                results.append(
                    ExtractedImage(
                        image=pil_image,
                        page_num=page_num,
                        bbox=bbox,
                        width=pix.width,
                        height=pix.height,
                        image_hash=img_hash,
                    )
                )

            except Exception as e:
                logger.warning(f"  Стр. {page_num+1}, img #{img_index}: ошибка — {e}")
                continue

        return results

    def _is_valid_size(self, width: int, height: int) -> bool:
        """Проверка: достаточно ли большое изображение."""
        if width < self.min_width or height < self.min_height:
            return False
        if width * height < self.min_area:
            return False
        return True

    def _find_image_bbox(
        self, page: fitz.Page, xref: int
    ) -> Tuple[float, float, float, float]:
        """Найти bounding box изображения на странице."""
        try:
            for img in page.get_image_info():
                if img.get("xref") == xref:
                    return tuple(img["bbox"])
        except Exception:
            pass
        # Fallback: вся страница
        return (0.0, 0.0, page.rect.width, page.rect.height)

    def get_summary(self, images: List[ExtractedImage]) -> Dict:
        """Статистика по извлечённым изображениям."""
        if not images:
            return {"total": 0, "pages_with_images": 0, "avg_size": (0, 0)}

        pages = set(img.page_num for img in images)
        avg_w = sum(img.width for img in images) / len(images)
        avg_h = sum(img.height for img in images) / len(images)
        total_pixels = sum(img.width * img.height for img in images)

        return {
            "total": len(images),
            "pages_with_images": len(pages),
            "pages_list": sorted(p + 1 for p in pages),
            "avg_size": (int(avg_w), int(avg_h)),
            "total_pixels": total_pixels,
            "sizes": [(img.width, img.height) for img in images],
        }

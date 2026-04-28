from dataclasses import dataclass
from typing import Optional, Tuple, Dict


# Допустимые типы блоков (7 типов):
#   Этап 1 (BlockClassifier, правила):  header, text, list, table, no_text
#   Этап 2 (CLIP, ИИ):                 chart, image
BLOCK_TYPES = {"header", "text", "list", "table", "no_text", "chart", "image"}


@dataclass
class Block:
    block_type: str                    # header | text | list | table | no_text | chart | image
    bbox: Tuple[float, float, float, float]
    page_num: int
    text: Optional[str] = None
    metadata: Optional[Dict] = None

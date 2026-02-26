from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass
class Block:
    block_type: str                    # header | paragraph | list_item | table | image
    bbox: Tuple[float, float, float, float]
    page_num: int
    text: Optional[str] = None
    metadata: Optional[Dict] = None

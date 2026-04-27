class BlockClassifier:
    """
    Эвристическая классификация текстовых блоков
    (без нейросетей — важно для диссертации)
    """

    def classify(self, text: str) -> str:
        t = text.strip()

        if not t:
            return "empty"

        # Заголовки
        if len(t) < 80 and t.isupper():
            return "header"

        if len(t) < 60 and t.endswith(":"):
            return "header"

        # Списки
        if t.startswith(("•", "-", "–", "—")):
            return "list_item"

        # Табличные строки (числа + разделители)
        if self._looks_like_table(t):
            return "table"

        return "paragraph"

    def _looks_like_table(self, text: str) -> bool:
        digit_count = sum(c.isdigit() for c in text)
        space_count = text.count(" ")

        return digit_count >= 4 and space_count >= 5

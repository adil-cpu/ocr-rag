class BlockClassifier:
    """
    Эвристическая классификация текстовых блоков (этап 1).

    Без нейросетей — используются правила на основе длины,
    регистра, первых символов и количества цифр/пробелов.

    Типы (этап 1):
        header  — заголовки (короткий текст, верхний регистр или с двоеточием)
        text    — обычные параграфы
        list    — элементы списка (начинаются с маркера)
        table   — табличные данные (много цифр и пробелов)
        no_text — пустой или нетекстовый контент (изображения)

    На этапе 2 блоки no_text классифицируются через CLIP в chart или image.
    """

    def classify(self, text: str) -> str:
        """
        Классифицировать текстовый блок.

        Args:
            text: текст блока из PyMuPDF

        Returns:
            Тип блока: 'header', 'text', 'list', 'table' или 'no_text'
        """
        t = text.strip()

        # Пустой или слишком короткий текст — не текстовый контент
        if not t or len(t) < 3:
            return "no_text"

        # Заголовки: короткий текст полностью в верхнем регистре
        if len(t) < 80 and t.isupper():
            return "header"

        # Заголовки: короткий текст, заканчивающийся двоеточием
        if len(t) < 60 and t.endswith(":"):
            return "header"

        # Списки: начинаются с маркера
        if t.startswith(("•", "-", "–", "—", "▪", "►", "●")):
            return "list"

        # Нумерованные списки: "1.", "2)", "а)" и т.д.
        if len(t) > 2 and t[0].isdigit() and t[1] in ".)" :
            return "list"

        # Таблицы: много цифр и пробелов
        if self._looks_like_table(t):
            return "table"

        # Всё остальное — обычный текст
        return "text"

    def _looks_like_table(self, text: str) -> bool:
        """Проверка: похож ли текст на строку таблицы."""
        digit_count = sum(c.isdigit() for c in text)
        space_count = text.count(" ")
        # Табличные данные обычно содержат много цифр и разделителей
        return digit_count >= 4 and space_count >= 5

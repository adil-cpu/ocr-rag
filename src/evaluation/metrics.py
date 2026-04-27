"""
Evaluation Metrics Module
=========================
BLEU, ROUGE-L, Faithfulness metrics for RAG evaluation.
"""

import re
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def compute_bleu(reference: str, hypothesis: str) -> float:
    """
    Вычислить BLEU score (упрощённый, unigram + bigram).

    Args:
        reference: эталонный ответ
        hypothesis: ответ системы

    Returns:
        float: BLEU score (0.0 - 1.0)
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        ref_tokens = nltk.word_tokenize(reference.lower())
        hyp_tokens = nltk.word_tokenize(hypothesis.lower())

        if not ref_tokens or not hyp_tokens:
            return 0.0

        smoothie = SmoothingFunction().method1
        score = sentence_bleu(
            [ref_tokens], hyp_tokens,
            weights=(0.5, 0.5),  # unigram + bigram
            smoothing_function=smoothie,
        )
        return round(score, 4)
    except Exception as e:
        logger.error(f"BLEU computation failed: {e}")
        return 0.0


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """
    Вычислить ROUGE-L F1 score.

    Args:
        reference: эталонный ответ
        hypothesis: ответ системы

    Returns:
        float: ROUGE-L F1 score (0.0 - 1.0)
    """
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        scores = scorer.score(reference, hypothesis)
        return round(scores["rougeL"].fmeasure, 4)
    except Exception as e:
        logger.error(f"ROUGE-L computation failed: {e}")
        return 0.0


def compute_faithfulness(answer: str, context: str) -> float:
    """
    Вычислить Faithfulness — какая часть утверждений в ответе
    подтверждается контекстом (простая эвристика на n-gram overlap).

    Args:
        answer: ответ системы
        context: контекст, использованный при генерации

    Returns:
        float: Faithfulness score (0.0 - 1.0)
    """
    answer_words = set(re.findall(r"\w+", answer.lower()))
    context_words = set(re.findall(r"\w+", context.lower()))

    if not answer_words:
        return 0.0

    # Удаляем стоп-слова (простой набор)
    stop_words = {"и", "в", "на", "с", "по", "к", "а", "но", "не", "что", "это",
                  "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
                  "to", "for", "of", "with", "and", "or", "но", "да", "от", "из"}

    answer_content = answer_words - stop_words
    context_content = context_words - stop_words

    if not answer_content:
        return 1.0  # нет содержательных слов — считаем faithful

    overlap = answer_content & context_content
    return round(len(overlap) / len(answer_content), 4)


def evaluate_response(
    question: str,
    reference: str,
    hypothesis: str,
    context: str,
) -> Dict[str, float]:
    """
    Полная оценка ответа RAG-системы.

    Returns:
        Dict с метриками: {bleu, rouge_l, faithfulness}
    """
    return {
        "bleu": compute_bleu(reference, hypothesis),
        "rouge_l": compute_rouge_l(reference, hypothesis),
        "faithfulness": compute_faithfulness(hypothesis, context),
    }

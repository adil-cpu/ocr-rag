"""
Benchmark Module
================
Run evaluation benchmark: set of question-answer pairs against the RAG pipeline.
Compare text-only RAG vs multimodal RAG.
"""

import json
import os
import logging
from typing import List, Dict, Optional
from datetime import datetime

from src.pipeline.rag_pipeline import MultimodalRAGPipeline
from src.evaluation.metrics import evaluate_response

logger = logging.getLogger(__name__)


# Пример набора вопрос-ответ пар для тестирования
SAMPLE_QA_PAIRS = [
    {
        "question": "О чём этот документ?",
        "reference": "Документ описывает основную информацию и содержание представленного PDF файла.",
        "category": "general",
    },
    {
        "question": "Какие основные разделы содержит документ?",
        "reference": "Документ содержит несколько разделов с различной тематикой.",
        "category": "structure",
    },
    {
        "question": "Есть ли в документе таблицы? Если да, что в них?",
        "reference": "В документе могут присутствовать таблицы с данными.",
        "category": "multimodal",
    },
]


class RAGBenchmark:
    """
    Бенчмарк для оценки качества RAG-системы.

    Запускает набор вопросов -> получает ответы -> вычисляет метрики ->
    сохраняет результаты.
    """

    def __init__(
        self,
        pipeline: MultimodalRAGPipeline,
        qa_pairs: Optional[List[Dict]] = None,
    ):
        self.pipeline = pipeline
        self.qa_pairs = qa_pairs or SAMPLE_QA_PAIRS
        self.results: List[Dict] = []

    def run(self) -> List[Dict]:
        """
        Запустить бенчмарк: все вопросы -> ответы -> метрики.

        Returns:
            Список результатов с метриками
        """
        print(f"\nЗапуск бенчмарка ({len(self.qa_pairs)} вопросов)...\n")
        self.results = []

        for i, qa in enumerate(self.qa_pairs, 1):
            question = qa["question"]
            reference = qa["reference"]

            print(f"  [{i}/{len(self.qa_pairs)}] {question}")

            # Получаем ответ от RAG
            response = self.pipeline.query(question)
            answer = response.get("answer", "")
            context = response.get("context_used", "")

            # Вычисляем метрики
            metrics = evaluate_response(
                question=question,
                reference=reference,
                hypothesis=answer,
                context=context,
            )

            result = {
                "question": question,
                "reference": reference,
                "answer": answer[:200],  # обрезка для удобства
                "category": qa.get("category", "unknown"),
                "model": response.get("model", "unknown"),
                **metrics,
            }
            self.results.append(result)

            print(f"         BLEU={metrics['bleu']:.3f}  "
                  f"ROUGE-L={metrics['rouge_l']:.3f}  "
                  f"Faith={metrics['faithfulness']:.3f}")

        self._print_summary()
        return self.results

    def _print_summary(self):
        """Напечатать сводку результатов."""
        if not self.results:
            print("Нет результатов.")
            return

        avg_bleu = sum(r["bleu"] for r in self.results) / len(self.results)
        avg_rouge = sum(r["rouge_l"] for r in self.results) / len(self.results)
        avg_faith = sum(r["faithfulness"] for r in self.results) / len(self.results)

        print(f"\n{'='*50}")
        print("РЕЗУЛЬТАТЫ БЕНЧМАРКА")
        print(f"{'='*50}")
        print(f"  Вопросов:      {len(self.results)}")
        print(f"  Avg BLEU:      {avg_bleu:.4f}")
        print(f"  Avg ROUGE-L:   {avg_rouge:.4f}")
        print(f"  Avg Faithful:  {avg_faith:.4f}")
        print(f"{'='*50}\n")

    def save_results(self, output_path: str = "data/outputs/benchmark_results.json"):
        """Сохранить результаты в JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        output = {
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(self.results),
            "results": self.results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"Результаты сохранены: {output_path}")


def run_comparison(
    pipeline: MultimodalRAGPipeline,
    qa_pairs: Optional[List[Dict]] = None,
):
    """
    Сравнение text-only vs multimodal RAG.
    (Для Главы 4 диссертации)

    На данный момент сравнивает один пайплайн.
    Для полного сравнения — нужно два пайплайна
    с разными настройками embedder.
    """
    benchmark = RAGBenchmark(pipeline, qa_pairs)
    results = benchmark.run()
    benchmark.save_results()

    return results


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    pipeline = MultimodalRAGPipeline()
    run_comparison(pipeline)

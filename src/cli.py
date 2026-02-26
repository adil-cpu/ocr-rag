"""
CLI Interface
=============
Command-line interface for the Multimodal RAG Pipeline.

Usage:
    python -m src.cli ingest <pdf_path>
    python -m src.cli query "<question>"
    python -m src.cli stats
"""

import argparse
import sys
import logging
from dotenv import load_dotenv

from src.pipeline.rag_pipeline import MultimodalRAGPipeline

# Загрузка .env
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(
        description="Мультимодальная RAG-система для анализа документов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python -m src.cli ingest data/input_pdfs/document.pdf
  python -m src.cli query "О чём этот документ?"
  python -m src.cli query "Кто является участником?" --top-k 3
  python -m src.cli stats
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Доступные команды")

    # --- ingest ---
    ingest_parser = subparsers.add_parser("ingest", help="Индексировать PDF-документ")
    ingest_parser.add_argument("pdf_path", help="Путь к PDF-файлу")
    ingest_parser.add_argument(
        "--index-dir", default="data/indexes",
        help="Директория для хранения индекса (default: data/indexes)"
    )

    # --- query ---
    query_parser = subparsers.add_parser("query", help="Задать вопрос по документу")
    query_parser.add_argument("question", help="Вопрос пользователя")
    query_parser.add_argument(
        "--top-k", type=int, default=5,
        help="Количество контекстных фрагментов (default: 5)"
    )
    query_parser.add_argument(
        "--index-dir", default="data/indexes",
        help="Директория с индексом (default: data/indexes)"
    )
    query_parser.add_argument(
        "--model", default="gpt-oss-20b",
        help="Модель LLM (default: gpt-oss-20b)"
    )

    # --- stats ---
    stats_parser = subparsers.add_parser("stats", help="Показать статистику индекса")
    stats_parser.add_argument(
        "--index-dir", default="data/indexes",
        help="Директория с индексом (default: data/indexes)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Создаём pipeline
    pipeline = MultimodalRAGPipeline(
        index_dir=getattr(args, "index_dir", "data/indexes"),
        llm_model=getattr(args, "model", "gpt-oss-20b"),
        top_k=getattr(args, "top_k", 5),
    )

    if args.command == "ingest":
        print(f"\n📄 Индексация: {args.pdf_path}\n")
        stats = pipeline.ingest(args.pdf_path)
        print(f"\n📊 Результат: {stats}")

    elif args.command == "query":
        print(f"\n❓ Вопрос: {args.question}\n")
        result = pipeline.query(args.question)

        print(f"🤖 Ответ ({result.get('model', 'unknown')}):\n")
        print(result["answer"])

        if result.get("sources"):
            print(f"\n📚 Источники ({len(result['sources'])}):")
            for s in result["sources"]:
                print(f"  - [score={s['score']}] стр. {s.get('page', '?')}: {s['text_preview']}")

    elif args.command == "stats":
        stats = pipeline.get_stats()
        print("\n📊 Статистика:")
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

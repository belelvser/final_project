"""CLI: python main.py --mode demo|full [--rebuild]."""
import argparse
import json
import sys
from pathlib import Path

from churn_pipeline.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Учебный прогноз 30-дневной CDR-неактивности")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--mode", choices=["demo", "full"])
    parser.add_argument("--output-dir", help="Относительно папки config.json или абсолютный путь")
    parser.add_argument("--rebuild", action="store_true", help="Повторить расчёт и проверить воспроизводимость")
    args = parser.parse_args()
    try:
        result = run_pipeline(args.config, args.mode, args.output_dir, args.rebuild)
    except Exception as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

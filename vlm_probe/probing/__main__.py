import argparse
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run the preregistered CLIP core probing suite.")
    parser.add_argument("--config", default="configs/probing_clip_b32_v1.json")
    parser.add_argument("--run-name")
    parser.add_argument("--smoke", action="store_true", help="First 64 records of each task; technical check only.")
    args = parser.parse_args()
    from .evaluate import run
    name = args.run_name or f"{'smoke' if args.smoke else 'full'}_probing_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
    run(Path(__file__).resolve().parents[2], args.config, name, args.smoke)


if __name__ == "__main__":
    main()

import argparse
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='CIFAR-100: baseline versus category prompts, all 100 candidates.')
    parser.add_argument('--smoke', action='store_true', help='First 200 test images, technical validation only.')
    parser.add_argument('--run-name')
    args = parser.parse_args()
    from .runner import run
    name = args.run_name or f"{'smoke' if args.smoke else 'full'}_category_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
    run(Path(__file__).resolve().parents[2], name, args.smoke)


if __name__ == '__main__':
    main()

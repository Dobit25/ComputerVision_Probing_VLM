import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Reproducible zero-shot evaluation selected by config.")
    parser.add_argument("--config", default="configs/clip_b32_cifar100.json")
    parser.add_argument("--limit", type=int, help="Smoke test: evaluate the first N test images.")
    parser.add_argument("--run-name", help="New directory name under results; existing runs are never overwritten.")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu"])
    args = parser.parse_args()
    # Defer ML imports so --help is fast and all caches can be scoped to the repo.
    from .evaluate import run
    run(Path(__file__).resolve().parents[1], args)


if __name__ == "__main__":
    main()

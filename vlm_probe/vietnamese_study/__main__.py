"""CLI entry point for P4 Cross-Lingual Vietnamese & BPE Tokenization Study."""

import argparse
import json

from .bpe_diagnostics import analyze_bpe_tokenization
from .runner import run_cifar100_vietnamese_study


def main():
    parser = argparse.ArgumentParser(description="Vietnamese Cross-Lingual & BPE Study (P4.1-P4.4).")
    sub = parser.add_subparsers(dest="action", required=True)

    p_bpe = sub.add_parser(
        "bpe",
        help="Run CPU-only BPE subword tokenization & fertility analysis on Vietnamese vs English prompts.",
    )
    p_bpe.add_argument("--output", default=None)

    p_run = sub.add_parser(
        "run-cifar100",
        help="Run live GPU cross-lingual CIFAR-100 evaluation across 4 language conditions.",
    )
    p_run.add_argument("--smoke", action="store_true")
    p_run.add_argument("--run-name", default=None)

    args = parser.parse_args()
    if args.action == "bpe":
        res = analyze_bpe_tokenization(output_path=args.output)
        summary = {
            "cifar100_prompts_summary": res["cifar100_prompts_summary"],
            "cifar100_labels_summary": res["cifar100_labels_summary"],
            "probing_captions_summary": res["probing_captions_summary"],
            "top_5_most_fragmented_vi_classes": res["top_10_most_fragmented_vi_classes"][:5],
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    elif args.action == "run-cifar100":
        run_cifar100_vietnamese_study(smoke=args.smoke, run_name=args.run_name)


if __name__ == "__main__":
    main()


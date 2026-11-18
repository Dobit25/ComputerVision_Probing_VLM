"""CLI entry point for P3 Prompt Sensitivity & Ensembling study."""

import argparse
import json

from .offline_analysis import analyze_saved_probing_ensembles
from .runner import run_dataset_prompt_study, run_probing_prompt_study


def main():
    parser = argparse.ArgumentParser(description="Prompt sensitivity & ensembling suite (P3.1-P3.4).")
    sub = parser.add_subparsers(dest="action", required=True)

    p_off = sub.add_parser(
        "offline",
        help="Analyze prompt sensitivity & ensembles on frozen probing/depth runs without GPU.",
    )
    p_off.add_argument("--output", default=None)

    p_ds = sub.add_parser(
        "dataset",
        help="Run live GPU 7-template + ensemble evaluation on a standard classification dataset.",
    )
    p_ds.add_argument(
        "--dataset",
        required=True,
        choices=["cifar100", "flowers102", "oxford_pet", "dtd", "food101"],
    )
    p_ds.add_argument("--smoke", action="store_true")
    p_ds.add_argument("--run-name", default=None)

    p_prb = sub.add_parser(
        "probing",
        help="Run live GPU multi-template + ensemble evaluation on Counting and Negation probing tasks.",
    )
    p_prb.add_argument("--smoke", action="store_true")
    p_prb.add_argument("--run-name", default=None)

    args = parser.parse_args()
    if args.action == "offline":
        res = analyze_saved_probing_ensembles(output_path=args.output)
        print(json.dumps(res, indent=2), flush=True)
    elif args.action == "dataset":
        run_dataset_prompt_study(args.dataset, smoke=args.smoke, run_name=args.run_name)
    elif args.action == "probing":
        run_probing_prompt_study(smoke=args.smoke, run_name=args.run_name)


if __name__ == "__main__":
    main()


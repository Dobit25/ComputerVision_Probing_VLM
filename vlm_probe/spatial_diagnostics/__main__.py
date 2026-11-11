"""CLI entry point for P2 spatial mechanism diagnostics and zero-shot mitigation."""

import argparse
import json
from pathlib import Path

from .mechanism import analyze_frozen_spatial_runs
from .quadrant_mitigation import evaluate_geometric_upper_bound, run_live_mitigation


def main():
    parser = argparse.ArgumentParser(description="Spatial failure mechanism & mitigation suite (P2.1-P2.3).")
    sub = parser.add_subparsers(dest="action", required=True)

    p_mech = sub.add_parser("mechanism", help="Decompose Prior Bias (B) vs Spatial Signal (Delta) on frozen runs.")
    p_mech.add_argument("--output", default=None, help="Optional JSON path to save decomposition results.")

    p_geom = sub.add_parser("geometry-bound", help="Compute geometric upper bound of half-canvas cropping.")
    p_geom.add_argument("--output", default=None)

    p_mit = sub.add_parser("mitigate", help="Run live CLIP ViT-B/32 half-canvas zero-shot mitigation on GPU.")
    p_mit.add_argument("--smoke", action="store_true")
    p_mit.add_argument("--run-name", default=None)

    args = parser.parse_args()
    if args.action == "mechanism":
        res = analyze_frozen_spatial_runs(output_path=args.output)
        print(json.dumps(res, indent=2), flush=True)
    elif args.action == "geometry-bound":
        res = evaluate_geometric_upper_bound()
        if args.output:
            Path(args.output).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2), flush=True)
    elif args.action == "mitigate":
        run_live_mitigation(smoke=args.smoke, run_name=args.run_name)


if __name__ == "__main__":
    main()


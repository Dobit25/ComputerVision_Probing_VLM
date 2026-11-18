import argparse


def main():
    parser = argparse.ArgumentParser(description='Controlled probing follow-up, frozen CLIP.')
    parser.add_argument('stage', choices=['historical', 'spatial', 'category', 'natural-prepare', 'natural', 'audit', 'report'])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--run')
    parser.add_argument('--run-name', help='New output directory name under results; existing runs are never overwritten.')
    args = parser.parse_args()
    if args.stage == 'historical':
        from .analysis import run
        run(run_name=args.run_name)
    elif args.stage == 'spatial':
        from .spatial import run
        run(args.smoke, run_name=args.run_name)
    elif args.stage == 'category':
        from .category import run
        run(args.smoke, run_name=args.run_name)
    elif args.stage == 'natural-prepare':
        from .natural import prepare
        prepare()
    elif args.stage == 'natural':
        from .natural import run
        run(args.smoke, run_name=args.run_name)
    elif args.stage == 'audit':
        from .audit import audit
        audit(args.run)
    elif args.stage == 'report':
        from .report import run
        run()


if __name__ == '__main__':
    main()

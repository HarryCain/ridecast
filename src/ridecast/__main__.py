import argparse
import json
from pathlib import Path

from .pipeline import train


def main():
    parser = argparse.ArgumentParser(description="Ridecast experiment runner")
    commands = parser.add_subparsers(dest="command", required=True)
    experiment = commands.add_parser("train", help="Download, train, backtest and evaluate")
    experiment.add_argument("--root", type=Path, default=Path.cwd())
    experiment.add_argument("--csv", type=Path, help="Use a local UCI hour.csv instead of downloading")
    args = parser.parse_args()
    result = train(args.root, args.csv)
    print(json.dumps({"metrics": result["metrics"], "coverage": result["test_coverage"]}, indent=2))


if __name__ == "__main__":
    main()

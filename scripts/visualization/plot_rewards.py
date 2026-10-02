import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

from ldr.visualizer.plotter import Plotter


def parse_args():
    parser = ArgumentParser(description="Plot training reward curves from monitor.csv.")
    parser.add_argument(
        "--monitor_dir",
        type=str,
        required=True,
        help="Directory containing monitor.csv (or pass the monitor.csv path directly).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reward_curve.png",
        help="Output image path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    monitor_path = Path(args.monitor_dir)
    if monitor_path.is_dir():
        monitor_path = monitor_path / "monitor.csv"

    df = pd.read_csv(monitor_path)
    plotter = Plotter()
    fig, _ = plotter.plotRewardCurveWithGlobalStep(df_list=[df], label_list=[monitor_path.parent.name])
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved reward curve to {args.output}")


if __name__ == "__main__":
    main()

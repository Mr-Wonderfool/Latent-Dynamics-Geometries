import sys
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SCRIPT_DIR))

from base import modelBase
from ldr.visualizer.state_visualizer import StatePredictionVisualizer


def parse_args():
    parser = ArgumentParser(description="Visualize state predictions from a trained checkpoint.")
    parser.add_argument("--model", type=str, required=True, help="Path to the pretrained agent checkpoint (.pth).")
    return parser.parse_args()


def visStatePrediction(model_path: str):
    env_params, agent_params = modelBase(erase_dynamics_wrappers=True, model_path=model_path)

    visualizer = StatePredictionVisualizer(env_params=env_params, agent_params=agent_params)
    buffer_samples = visualizer.rollout(rollout_timesteps=100)

    fig, ax = visualizer.visualizeStatePrediction(buffer_samples=buffer_samples, figsize=(14, 8))

    return fig, ax


if __name__ == "__main__":
    args = parse_args()
    fig, ax = visStatePrediction(args.model)
    plt.show()

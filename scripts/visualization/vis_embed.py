import sys
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SCRIPT_DIR))

from base import modelBase
from ldr.visualizer.embedding_visualizer import EmbeddingVisualizer


def parse_args():
    parser = ArgumentParser(description="Visualize latent embedding structure from a trained checkpoint.")
    parser.add_argument("--model", type=str, required=True, help="Path to the pretrained agent checkpoint (.pth).")
    return parser.parse_args()


def visEmbed(model_path: str):
    env_params, agent_params = modelBase(erase_dynamics_wrappers=True, model_path=model_path)

    visualizer = EmbeddingVisualizer(env_params=env_params, agent_params=agent_params, render=False)

    fig, ax = visualizer.visualizeLatentSemanticStructure(
        param_name="mass",
        scan_values=np.linspace(0.5, 2.0, 5, endpoint=False).tolist(),
        train_range=(0.5, 2.0),
        seed=1012,
        n_latents_per_traj=100,
        projection_method="tsne",
    )

    return fig, ax


if __name__ == "__main__":
    args = parse_args()
    fig, ax = visEmbed(args.model)
    plt.show()

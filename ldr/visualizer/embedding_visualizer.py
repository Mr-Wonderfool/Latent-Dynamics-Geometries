import numpy as np
from tqdm import tqdm
from typing import Tuple
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from .base_visualizer import BaseVisualizer, BaseVisualizerRMA


class EmbeddingVisualizer(BaseVisualizer):
    """
    Visualize latent embeddings and their consistency across trajectories.
    """

    def visualizeSeparationAcrossDynamics(
        self, num_trajs: int = 10, n_latents_per_traj: int = 100, projection_method: str = "tsne"
    ):
        all_sampled_latents = []
        trajectory_colors = []
        for traj_idx in tqdm(range(num_trajs)):
            # set different seed for different dynamics (at the expense of different initial condition)
            latents = self._rollout(seed=traj_idx)
            curr_traj_len = len(latents)
            num_to_select = min(curr_traj_len, n_latents_per_traj)
            sampled_indices = np.linspace(0, curr_traj_len - 1, num_to_select, dtype=int)
            sampled_latents = [latents[i] for i in sampled_indices]
            all_sampled_latents.extend(sampled_latents)
            trajectory_colors.extend([traj_idx] * len(sampled_latents))
        self.env.close()

        latents_np = np.array(all_sampled_latents)
        trajectory_colors_np = np.array(trajectory_colors)
        v_min = np.min(trajectory_colors_np)
        v_max = np.max(trajectory_colors_np)
        norm = plt.Normalize(vmin=v_min, vmax=v_max)
        cmap_name = "jet"

        if "3d" == projection_method:
            assert latents_np.shape[-1] >= 3
            if latents_np.shape[-1] > 3:
                # project to 3d with pca
                latents_np = self.projectPCA(latents_np, dim=3)

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=30, azim=45)
            ax.scatter(
                latents_np[:, 0],
                latents_np[:, 1],
                latents_np[:, 2],
                c=trajectory_colors_np,
                cmap=cmap_name,
                norm=norm,
                s=20,
                marker="o",
                alpha=0.8,
            )
        else:
            latents_2d = self.project2D(latents_np, method=projection_method)

            fig, ax = self._generateScatterBase(figsize=(8, 8))
            ax.scatter(
                latents_2d[:, 0],
                latents_2d[:, 1],
                c=trajectory_colors_np,
                cmap=cmap_name,
                norm=norm,
                marker="o",
                s=20,
                alpha=0.8,
            )

        sm = cm.ScalarMappable(cmap=cmap_name, norm=norm)
        cbar = fig.colorbar(mappable=sm, ax=ax, pad=0.01)
        cbar.set_ticks(np.arange(num_trajs))
        cbar.set_label("Trajectory ID", fontsize=14)

        return fig, ax

    def visualizeLatentSemanticStructure(
        self,
        param_name: str,
        scan_values: np.ndarray,
        train_range: Tuple[float, float],
        seed: int = 1012,
        n_latents_per_traj: int = 10,
        projection_method: str = "tsne",
    ):
        """
        Gradually change provided `param_name` with values within specified `scan_values`.
        Parameters:
            param_name: choose within [mass, friction, damping, torque_scale]
            scan_values: Array of values to test (e.g., np.linspace(0.1, 1.5, 20))
            train_range: [min, max) of the training distribution for OOD labeling
            projection_method: choose within ['pca', 'tsne', '3d'], for 3d, we assume the latent vector is 3-dimensional and should not be projected
        """
        assert projection_method in ["pca", "tsne", "3d"]

        all_sampled_latents = []
        all_values = []
        ood_labelings = []

        for value in tqdm(scan_values):
            self._manuallySetDynamics(param_name, value)
            # same seed for same initial conditions
            latents = self._rollout(seed=seed)
            curr_traj_len = len(latents)
            num_to_select = min(curr_traj_len, n_latents_per_traj)
            sampled_indices = np.linspace(0, curr_traj_len - 1, num_to_select, dtype=int)
            sampled_latents = [latents[i] for i in sampled_indices]
            all_sampled_latents.extend(sampled_latents)
            is_ood = value < train_range[0] or value >= train_range[1]
            ood_labelings.extend([is_ood] * len(sampled_indices))
            all_values.extend([value] * len(sampled_indices))
        self.env.close()

        latents_np = np.array(all_sampled_latents)
        all_values_np = np.array(all_values)
        ood_mask = np.array(ood_labelings, dtype=np.bool_)
        id_mask = ~ood_mask

        v_min = np.min(all_values_np)
        v_max = np.max(all_values_np)
        norm = plt.Normalize(vmin=v_min, vmax=v_max)
        cmap_name = "RdBu_r"

        if "3d" == projection_method:
            assert latents_np.shape[-1] >= 3
            if latents_np.shape[-1] > 3:
                # project to 3d with pca
                latents_np = self.projectPCA(latents_np, dim=3)

            fig = plt.figure(figsize=(10, 10))
            # fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=30, azim=45)

            def plot_subset(mask, marker, s, label_prefix):
                if np.any(mask):
                    ax.scatter(
                        latents_np[mask, 0],
                        latents_np[mask, 1],
                        latents_np[mask, 2],
                        c=all_values_np[mask],
                        cmap=cmap_name,
                        norm=norm,
                        s=s,
                        marker=marker,
                        label=label_prefix,
                        alpha=0.8,
                    )

            plot_subset(id_mask, "o", 20, "In-Distribution")
            plot_subset(ood_mask, "^", 20, "Out-Of-Distribution")
            # ax.axis('off')
        else:
            latents_2d = self.project2D(latents_np, method=projection_method)
            fig, ax = self._generateScatterBase(figsize=(8, 8))
            # fig, ax = self._generateScatterBase(figsize=(5, 4))

            def plot_subset(mask, marker, s, label_prefix):
                if np.any(mask):
                    ax.scatter(
                        latents_2d[mask, 0],
                        latents_2d[mask, 1],
                        c=all_values_np[mask],
                        cmap=cmap_name,
                        norm=norm,
                        s=s,
                        marker=marker,
                        label=label_prefix,
                        alpha=0.8,
                    )

            plot_subset(id_mask, "o", 20, "In-Distribution")
            plot_subset(ood_mask, "^", 20, "Out-Of-Distribution")

        sm = cm.ScalarMappable(cmap=cmap_name, norm=norm)
        cbar = fig.colorbar(mappable=sm, ax=ax, pad=0.01)
        cbar.set_ticks(scan_values)
        cbar.set_label(f"{param_name} value", fontsize=14)

        ax.legend(fontsize=16, loc="upper right", markerscale=2.5)

        return fig, ax

    def visualizeCrippleShift(
        self, seed: int = 1012, projection_method: str = "tsne", draw_trajectory_path: bool = True
    ):
        """
        Rolls out an episode and visualizes the latent shift before and after the joint is crippled.
        """
        from matplotlib.colors import to_rgba

        assert projection_method in ["pca", "tsne", "3d"]

        # recursively search for the CrippleJointWrapper to find the lag
        curr_env = self.env.envs[0]
        lag = None
        while True:
            if type(curr_env).__name__ == "CrippleJointWrapper":
                lag = curr_env.lag
                break
            elif hasattr(curr_env, "env"):
                curr_env = curr_env.env
            else:
                raise ValueError("CrippleJointWrapper not found in the environment stack.")

        latents = self._rollout(seed=seed)
        self.env.close()

        latents_np = np.array(latents)
        traj_len = len(latents_np)

        timesteps = np.arange(1, traj_len + 1)
        is_crippled = timesteps > lag
        is_normal = ~is_crippled

        if np.any(is_crippled):
            time_after_break = timesteps[is_crippled] - lag
            max_time_after_break = traj_len - lag

            if max_time_after_break > 0:
                visibility = time_after_break / max_time_after_break
            else:
                visibility = np.ones_like(time_after_break, dtype=float)

            visibility = np.clip(visibility, 0.15, 1.0)

            base_rgb = to_rgba("crimson")[:3]
            crippled_rgba = np.zeros((len(visibility), 4))
            crippled_rgba[:, :3] = base_rgb
            crippled_rgba[:, 3] = visibility

        if projection_method == "3d":
            assert latents_np.shape[-1] >= 3
            if latents_np.shape[-1] > 3:
                latents_np = self.projectPCA(latents_np, dim=3)

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=30, azim=45)

            if draw_trajectory_path:
                ax.plot(
                    latents_np[:, 0], latents_np[:, 1], latents_np[:, 2], c="gray", alpha=0.3, linewidth=1, zorder=0
                )

            if np.any(is_normal):
                ax.scatter(
                    latents_np[is_normal, 0],
                    latents_np[is_normal, 1],
                    latents_np[is_normal, 2],
                    c="royalblue",
                    label="Normal (Pre-Break)",
                    s=40,
                    marker="o",
                    alpha=0.8,
                )
            if np.any(is_crippled):
                ax.scatter(
                    latents_np[is_crippled, 0],
                    latents_np[is_crippled, 1],
                    latents_np[is_crippled, 2],
                    c=crippled_rgba,
                    label="Crippled (Transient)",
                    s=40,
                    marker="^",
                )

        else:
            latents_2d = self.project2D(latents_np, method=projection_method)
            fig, ax = self._generateScatterBase(figsize=(8, 8))

            if draw_trajectory_path:
                ax.plot(latents_2d[:, 0], latents_2d[:, 1], c="gray", alpha=0.3, linewidth=1, zorder=0)

            if np.any(is_normal):
                ax.scatter(
                    latents_2d[is_normal, 0],
                    latents_2d[is_normal, 1],
                    c="royalblue",
                    label="Normal (Pre-Break)",
                    s=40,
                    marker="o",
                    alpha=0.8,
                )
            if np.any(is_crippled):
                # Pass the constructed RGBA array directly to 'c'
                ax.scatter(
                    latents_2d[is_crippled, 0],
                    latents_2d[is_crippled, 1],
                    c=crippled_rgba,
                    label="Crippled (Transient)",
                    s=40,
                    marker="^",
                )

        ax.legend(fontsize=16, loc="best", markerscale=1.5)

        return fig, ax

    def _generateScatterBase(self, figsize):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])

        return fig, ax

    @classmethod
    def project2D(cls, latents: np.ndarray, method: str = "tsne"):
        if "tsne" == method:
            return cls.projectTSNE(latents=latents, dim=2)
        elif "pca" == method:
            return cls.projectPCA(latents=latents, dim=2)

    @classmethod
    def projectTSNE(cls, latents: np.ndarray, dim: int):
        latents_2d = latents.copy()
        if latents.shape[1] > 2:
            projector = TSNE(n_components=dim, perplexity=latents.shape[0] - 2, random_state=42)
            latents_2d = projector.fit_transform(latents_2d)
        return latents_2d

    @classmethod
    def projectPCA(cls, latents: np.ndarray, dim: int):
        latents_nd = latents.copy()
        if latents.shape[1] > 2:
            projector = PCA(n_components=dim)
            latents_nd = projector.fit_transform(latents_nd)
            cumulative_variance = np.cumsum(projector.explained_variance_ratio_)
            tqdm.write(f"Importance of principal components: {cumulative_variance}")
        return latents_nd

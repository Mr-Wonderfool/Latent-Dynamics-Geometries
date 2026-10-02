import pandas as pd
from typing import List
import matplotlib.pyplot as plt

from .palette import PlotPalette


class Plotter:
    def __init__(self):
        PlotPalette.setPlotTheme()

    @classmethod
    def plotRewardCurveWithGlobalStep(
        cls, df_list: List[pd.DataFrame], label_list: List[str], window_size: int = 100, figsize=(12, 6)
    ):
        """
        Use 'global_step' as the x-axis, and mean over entries with the same global step.
        """
        fig, ax = plt.subplots(figsize=figsize)

        for i, df in enumerate(df_list):
            df = df.sort_values(by="global_step").reset_index(drop=True)
            # merge entries with the same global step
            df = df.groupby("global_step").mean().reset_index()
            rolling_data = df["reward"].rolling(window=window_size, min_periods=1)
            rolling_mean = rolling_data.mean()
            rolling_std = rolling_data.std()

            x_axis = df["global_step"] / 1e6

            lower_bound = rolling_mean - rolling_std
            upper_bound = rolling_mean + rolling_std

            ax.plot(x_axis, rolling_mean, label=label_list[i], linewidth=2)
            ax.fill_between(x_axis, lower_bound, upper_bound, alpha=0.2)

        ax.set_xlabel("million steps")
        ax.set_ylabel("average return")
        ax.grid(True)
        ax.legend()

        fig.tight_layout()

        return fig, ax

    @classmethod
    def plotRewardCurveWithEpisodicMean(
        cls, df_list: List[pd.DataFrame], label_list: List[str], window_size: int = 50, figsize=(12, 6)
    ):
        """
        Aggregate according to episode index, and mean over the same episode from different environments.
        """
        fig, ax = plt.subplots(figsize=figsize)

        for i, df in enumerate(df_list):
            agg_data = (
                df.groupby("episode_index")
                .agg(reward_mean=("reward", "mean"), episode_length=("episode_length", "sum"))
                .reset_index()
            )

            x_axis = agg_data["episode_length"].cumsum() / 1e6
            rolling_data = agg_data["reward_mean"].rolling(window=window_size, min_periods=1)
            rolling_mean = rolling_data.mean()
            rolling_std = rolling_data.std()

            lower_bound = rolling_mean - rolling_std
            upper_bound = rolling_mean + rolling_std

            ax.plot(x_axis, rolling_mean, label=label_list[i], linewidth=2)
            ax.fill_between(x_axis, lower_bound, upper_bound, alpha=0.2)

        ax.set_xlabel("million steps")
        ax.set_ylabel("average return")
        ax.grid(True)
        ax.legend()

        fig.tight_layout()

        return fig, ax

import matplotlib.pyplot as plt


class PlotPalette:

    BACKGROUND_COLOR = "#e9e9f1"
    GRID_LINE_WIDTH_FACTOR = 1.2

    @classmethod
    def setPlotTheme(cls, legend_font_size=16, tick_font_size=14, ax_font_size=14):
        """
        Args:
            legend_font_size (int): Font size for the legend text.
            tick_font_size (int): Font size for the axis tick labels.
            ax_font_size (int): Font size for the axis labels (xlabel, ylabel).
        """
        plt.style.use("seaborn-v0_8")

        # 1. font type
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
                "mathtext.fontset": "dejavusans",
            }
        )

        # 2. background color
        plt.rcParams.update({"axes.facecolor": cls.BACKGROUND_COLOR})

        # 3. grid lines
        default_lw = plt.rcParams.get("grid.linewidth")
        new_lw = default_lw * cls.GRID_LINE_WIDTH_FACTOR
        plt.rcParams["grid.linewidth"] = new_lw
        plt.rcParams["axes.grid"] = True

        # 4. font size
        plt.rcParams.update(
            {
                "legend.fontsize": legend_font_size,
                "xtick.labelsize": tick_font_size,
                "ytick.labelsize": tick_font_size,
                "axes.labelsize": ax_font_size,
                "axes.titlesize": ax_font_size * 1.2,
            }
        )

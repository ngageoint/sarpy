import seaborn
import matplotlib.pyplot as plt
import math
import numpy as np

SEABORN_DEEP = "deep"
SEABORN_MUTED = "muted"
SEABORN_BRIGHT = "bright"
SEABORN_PASTEL = "pastel"
SEABORN_DARK = "dark"
SEABORN_COLORBLIND = "colorblind"
SEABORN_BLUES = "Blues"


class PlotStyleUtils:
    def __init__(self):
        self.n_color_bins = 10
        self.set_palette_by_name(SEABORN_DEEP)

    def set_palette_array(self, colors_list):
        self.plot_colors = colors_list

    def set_plot_colors(self,
                        color_palette,          # type: [[float]]
                        n_colors                # type: int
                        ):
        colors = self.get_colors_from_palette(color_palette, n_colors)
        self.plot_colors = colors

    def set_palette_by_name(self, palette_name):
        self.palette = seaborn.color_palette(palette_name, self.n_color_bins)
        self.plot_colors = self.get_colors_from_palette(self.palette, self.n_color_bins)

    @staticmethod
    def get_all_palettes_list():
        palettes_list = [SEABORN_DEEP,
                         SEABORN_MUTED,
                         SEABORN_BRIGHT,
                         SEABORN_PASTEL,
                         SEABORN_DARK,
                         SEABORN_COLORBLIND,
                         SEABORN_BLUES]
        return palettes_list

    @staticmethod
    def get_colors_from_palette(palette, n_colors):
        color_array = []
        n_color_bins = len(palette)
        indices = np.linspace(0, n_colors, n_colors)
        for i in indices:
            index = i / n_colors * (n_color_bins-1)
            low = int(index)
            high = int(math.ceil(index))
            interp_float = index - low
            color_array.append(list(np.array(palette[low]) * (1 - interp_float) + np.array(palette[high]) * interp_float))
        return color_array

    @staticmethod
    def get_available_matplotlib_styles():
        return ['default', 'classic'] + sorted(style for style in plt.style.available if style != 'classic')

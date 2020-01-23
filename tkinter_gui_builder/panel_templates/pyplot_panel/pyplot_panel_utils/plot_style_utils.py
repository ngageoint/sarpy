import seaborn as sns


class PlotStyleUtils:
    def __init__(self):
        self.plot_color_cycle = ['r', 'g', 'b', 'y']

    def set_color_cycler(self, colors_list):
        self.plot_color_cycle = colors_list

    def test_stuff(self):
        something = sns.palplot(sns.light_palette("green"))
        stop = 1

import tkinter as tk
from tkinter_gui_builder.widgets import basic_widgets
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel

import numpy as np


class PyplotCanvas(tk.LabelFrame):
    def __init__(self, master):
        tk.LabelFrame.__init__(self, master)

        fig = Figure()
        self.ax = fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(fig, master=master)
        self.canvas.get_tk_widget().pack(fill='both')


class PyplotControlPanel(AbstractWidgetPanel):
    scale = basic_widgets.Scale         # type: basic_widgets.Scale

    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)

        widget_list = ["scale"]

        self.init_w_horizontal_layout(widget_list)

        self.scale.set(0)
        self.scale.pack(side='bottom')
        self.scale.state(["disabled"])

        # master, orient = tk.HORIZONTAL, length = 284, from_ = 0, to = 100)


class AppVariables():
    def __init__(self):
        self.x_axis = None  # type: np.ndarray
        self.plot_data = None  # type: np.ndarray

        self.xmin = None  # type: float
        self.xmax = None  # type: float
        self.ymin = None  # type: float
        self.ymax = None  # type: float

        self.y_margin = 0.05
        self.set_y_margins_per_frame = False

        self.segments = None  # type: np.ndarray


class PyplotPanel(AbstractWidgetPanel):
    pyplot_canvas = PyplotCanvas           # type: PyplotCanvas
    control_panel = PyplotControlPanel      # type: PyplotControlPanel

    def __init__(self, master):
        AbstractWidgetPanel.__init__(self, master)

        self.variables = AppVariables()
        widget_list = ["pyplot_canvas", "control_panel"]
        self.init_w_vertical_layout(widget_list)

        # set listeners
        self.control_panel.scale.on_left_mouse_motion(self.callback_update_from_slider)

    def set_y_margin_percent(self,
                             percent_0_to_100=5          # type: float
                             ):
        self.variables.y_margin = percent_0_to_100 * 0.01

    def set_data(self, plot_data, x_axis=None):
        x = x_axis
        n_times = 1
        if len(plot_data.shape) == 1:
            nx = len(plot_data)
            segments = np.zeros((1, nx, 2))
            segments[0, :, 1] = plot_data
        elif len(plot_data.shape) == 2:
            nx = len(plot_data[:, 0])
            n_overplots = len(plot_data[0])
            segments = np.zeros((n_overplots, nx, 2))
            for i in range(n_overplots):
                segments[i, :, 1] = plot_data[:, i]
        elif len(plot_data.shape) == 3:
            nx = np.shape(plot_data)[0]
            n_overplots = np.shape(plot_data)[1]
            n_times = np.shape(plot_data)[2]
            segments = np.zeros((n_overplots, nx, 2))
            for i in range(n_overplots):
                segments[i, :, 1] = plot_data[:, i, 0]
            self.control_panel.scale.state(['!disabled'])
        if x is None:
            x = np.arange(nx)
        segments[:, :, 0] = x

        self.variables.xmin = x.min()
        self.variables.xmax = x.max()

        y_range = plot_data.max() - plot_data.min()

        self.variables.ymin = plot_data.min() - y_range * self.variables.y_margin
        self.variables.ymax = plot_data.max() + y_range * self.variables.y_margin

        self.variables.x_axis = x
        self.variables.plot_data = plot_data
        self.variables.segments = segments

        self.control_panel.scale.config(to=n_times-1)

        if len(plot_data.shape) == 3:
            self.update_plot_animation(0)

        else:
            self.pyplot_canvas.ax.clear()
            self.pyplot_canvas.ax.plot(self.variables.plot_data)
            self.pyplot_canvas.ax.set_ylim(self.variables.ymin, self.variables.ymax)
            self.pyplot_canvas.canvas.draw()

    def update_plot_animation(self, animation_index):
        n_overplots = np.shape(self.variables.segments)[0]
        for i in range(n_overplots):
            self.variables.segments[i, :, 1] = self.variables.plot_data[:, i, animation_index]

        self.pyplot_canvas.ax.clear()

        self.pyplot_canvas.ax.set_xlim(self.variables.xmin, self.variables.xmax)
        line_segments = LineCollection(self.variables.segments, linewidths=(0.5, 0.75, 1., 1.25), linestyle='solid')
        if self.variables.set_y_margins_per_frame:
            plot_data = self.variables.segments[:, :, 1]
            y_range = plot_data.max() - plot_data.min()
            self.variables.ymin = plot_data.min() - y_range * self.variables.y_margin
            self.variables.ymax = plot_data.max() + y_range * self.variables.y_margin
        self.pyplot_canvas.ax.set_ylim(self.variables.ymin, self.variables.ymax)

        self.pyplot_canvas.ax.add_collection(line_segments)
        self.pyplot_canvas.canvas.draw()

    def callback_update_from_slider(self, event):
        if "disabled" in self.control_panel.scale.state():
            pass
        else:
            animation_index = int(np.round(self.control_panel.scale.get()))
            self.update_plot_animation(animation_index)

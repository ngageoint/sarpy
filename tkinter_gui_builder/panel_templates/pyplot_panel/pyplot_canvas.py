import tkinter as tk
from tkinter_gui_builder.widgets import basic_widgets
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
import numpy as np


class PyplotCanvas(tk.LabelFrame):
    def __init__(self, master):
        tk.LabelFrame.__init__(self, master)

        fig = Figure()
        self.ax = fig.add_subplot(111)
        self.x_axis = None          # type: np.ndarray
        self.plot_data = None       # type: np.ndarray

        self.xmin = None            # type: float
        self.xmax = None            # type: float
        self.ymin = None            # type: float
        self.ymax = None            # type: float

        self.segments = None            # type: np.ndarray

        self.scale = basic_widgets.Scale(master, orient=tk.HORIZONTAL, length=284, from_=0, to=100)
        self.scale.set(0)
        self.scale.pack(side='bottom')
        self.scale.on_left_mouse_motion(self.callback_update_from_slider)

        self.canvas = FigureCanvasTkAgg(fig, master=master)
        self.canvas.get_tk_widget().pack(fill='both')

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
        if x is None:
            x = np.arange(nx)
        segments[:, :, 0] = x

        self.xmin = x.min()
        self.xmax = x.max()
        self.ymin = plot_data.min()
        self.ymax = plot_data.max()

        self.ax.set_xlim(x.min(), x.max())
        self.ax.set_ylim(plot_data.min(), plot_data.max())
        self.scale.config(to=n_times-1)
        self.x_axis = x
        self.plot_data = plot_data
        self.segments = segments

        self.update_plot(0)

    def update_plot(self, time_index):
        n_overplots = np.shape(self.segments)[0]
        for i in range(n_overplots):
            self.segments[i, :, 1] = self.plot_data[:, i, time_index]

        self.ax.clear()

        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.ymin, self.ymax)

        line_segments = LineCollection(self.segments, linewidths=(0.5, 0.75, 1., 1.25), linestyle='solid')
        self.ax.add_collection(line_segments)
        self.canvas.draw()

    def callback_update_from_slider(self, event):
        time_index = int(np.round(self.scale.get()))
        self.update_plot(time_index)

import tkinter
from tkinter_gui_builder.widgets import basic_widgets
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
import numpy as np


class TemporalPlotPanel:
    def __init__(self, master):
        master_frame = tkinter.Frame(master)
        fig = Figure()
        self.ax = fig.add_subplot(111)
        self.x_axis = None          # type: np.ndarray
        self.plot_data = None       # type: np.ndarray

        self.segs = None            # type: np.ndarray

        self.xmin = None            # type: float
        self.xmax = None            # type: float
        self.ymin = None            # type: float
        self.ymax = None            # type: float

        n_overplots = 10
        nx = 200
        n_times = 100

        x_axis = np.linspace(0, 2*np.pi, nx)
        y_data_1 = np.sin(x_axis)
        y_data_2 = np.zeros((len(x_axis), n_overplots))
        y_data_3 = np.zeros((len(x_axis), n_overplots, n_times))

        scaling_factors = np.linspace(0.7, 1, n_overplots)

        for i in range(n_overplots):
            y_data_2[:, i] = y_data_1 * scaling_factors[i]

        x_over_time = np.zeros((nx, n_times))
        x_over_time_start = np.linspace(0, 2*np.pi, n_times)
        for i in range(n_times):
            x_start = x_over_time_start[i]
            x = np.linspace(x_start, 2*np.pi + x_start, nx)
            x_over_time[:, i] = x
            y = np.sin(x)
            for j in range(n_overplots):
                y_data_3[:, j, i] = y * scaling_factors[j]

        self.scale = basic_widgets.Scale(master_frame, orient=tkinter.HORIZONTAL, length=284, from_=0, to=100)
        self.scale.set(0)
        self.scale.pack(side='bottom')
        self.scale.on_left_mouse_motion(self.callback_update_from_slider)

        self.canvas = FigureCanvasTkAgg(fig, master=master_frame)
        self.canvas.get_tk_widget().pack(fill='both')

        self.set_data(y_data_3, x_axis)

        master_frame.pack()

    def set_data(self, plot_data, x_axis=None):
        x = x_axis
        n_times = 1
        if len(plot_data.shape) == 1:
            nx = len(plot_data)
            segs = np.zeros((1, nx, 2))
            segs[0, :, 1] = plot_data
        elif len(plot_data.shape) == 2:
            nx = len(plot_data[:, 0])
            n_overplots = len(plot_data[0])
            segs = np.zeros((n_overplots, nx, 2))
            for i in range(n_overplots):
                segs[i, :, 1] = plot_data[:, i]
        elif len(plot_data.shape) == 3:
            nx = np.shape(plot_data)[0]
            n_overplots = np.shape(plot_data)[1]
            n_times = np.shape(plot_data)[2]
            segs = np.zeros((n_overplots, nx, 2))
            for i in range(n_overplots):
                segs[i, :, 1] = plot_data[:, i, 0]
        if x is None:
            x = np.arange(nx)
        segs[:, :, 0] = x

        self.xmin = x.min()
        self.xmax = x.max()
        self.ymin = plot_data.min()
        self.ymax = plot_data.max()

        self.ax.set_xlim(x.min(), x.max())
        self.ax.set_ylim(plot_data.min(), plot_data.max())
        self.scale.config(to=n_times-1)
        self.x_axis = x
        self.plot_data = plot_data
        self.segs = segs

        self.update_plot(0)

    def update_plot(self, time_index):
        n_overplots = np.shape(self.segs)[0]
        for i in range(n_overplots):
            self.segs[i, :, 1] = self.plot_data[:, i, time_index]

        self.ax.clear()

        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.ymin, self.ymax)

        line_segments = LineCollection(self.segs, linewidths=(0.5, 0.75, 1., 1.25), linestyle='solid')
        self.ax.add_collection(line_segments)
        self.canvas.draw()

    def callback_update_from_slider(self, event):
        time_index = int(np.round(self.scale.get()))
        self.update_plot(time_index)

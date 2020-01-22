import tkinter as tk
from tkinter_gui_builder.widgets import basic_widgets
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
import time

import numpy as np


SCALE_Y_AXIS_PER_FRAME_TRUE = "scale y axis per frame"
SCALE_Y_AXIS_PER_FRAME_FALSE = "don't scale y axis per frame"


class PyplotCanvas(tk.LabelFrame):
    def __init__(self, master):
        tk.LabelFrame.__init__(self, master)

        fig = Figure()
        self.ax = fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(fig, master=master)
        self.canvas.get_tk_widget().pack(fill='both')


class PyplotControlPanel(AbstractWidgetPanel):
    scale = basic_widgets.Scale         # type: basic_widgets.Scale
    rescale_y_axis_per_frame = basic_widgets.Combobox         # type: basic_widgets.Combobox
    animate = basic_widgets.Button          # type: basic_widgets.Button
    fps_label = basic_widgets.Label         # type: basic_widgets.Label
    fps_entry = basic_widgets.Entry               # type: basic_widgets.Entry

    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)
        widget_list = ["scale", "rescale_y_axis_per_frame", "fps_label", "fps_entry", "animate"]

        self.init_w_basic_widget_list(widget_list, 3, [1, 1, 3])
        self.scale.set(0)
        self.rescale_y_axis_per_frame.update_combobox_values([SCALE_Y_AXIS_PER_FRAME_TRUE, SCALE_Y_AXIS_PER_FRAME_FALSE])
        self.fps_label.set_text("fps")
        self.fps_entry.set_text("10")


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
        self.n_frames = 1

        self.segments = None  # type: np.ndarray

        self.animation_related_controls = []


class PyplotPanel(AbstractWidgetPanel):
    pyplot_canvas = PyplotCanvas           # type: PyplotCanvas
    control_panel = PyplotControlPanel      # type: PyplotControlPanel

    def __init__(self, master):
        AbstractWidgetPanel.__init__(self, master)

        self.variables = AppVariables()
        widget_list = ["pyplot_canvas", "control_panel"]
        self.init_w_vertical_layout(widget_list)

        canvas_size_pixels = self.pyplot_canvas.canvas.figure.get_size_inches() * self.pyplot_canvas.canvas.figure.dpi

        self.control_panel.scale.config(length=canvas_size_pixels[0] * 0.75)
        self.variables.animation_related_controls = [self.control_panel.scale,
                                                     self.control_panel.rescale_y_axis_per_frame,
                                                     self.control_panel.animate,
                                                     self.control_panel.fps_entry,
                                                     self.control_panel.fps_label]

        # set listeners
        self.control_panel.scale.on_left_mouse_motion(self.callback_update_from_slider)
        self.control_panel.rescale_y_axis_per_frame.on_selection(self.callback_set_y_rescale)
        self.control_panel.animate.on_left_mouse_click(self.callback_animate)

    def hide_animation_related_controls(self):
        for widget in self.variables.animation_related_controls:
            widget.pack_forget()

    def show_animation_related_controls(self):
        for widget in self.variables.animation_related_controls:
            widget.pack()

    def set_y_margin_percent(self,
                             percent_0_to_100=5          # type: float
                             ):
        self.variables.y_margin = percent_0_to_100 * 0.01

    def set_data(self, plot_data, x_axis=None):
        x = x_axis
        n_frames = 1
        if len(plot_data.shape) == 1:
            self.hide_animation_related_controls()
            nx = len(plot_data)
            segments = np.zeros((1, nx, 2))
            segments[0, :, 1] = plot_data
        elif len(plot_data.shape) == 2:
            self.hide_animation_related_controls()
            nx = len(plot_data[:, 0])
            n_overplots = len(plot_data[0])
            segments = np.zeros((n_overplots, nx, 2))
            for i in range(n_overplots):
                segments[i, :, 1] = plot_data[:, i]
        elif len(plot_data.shape) == 3:
            self.show_animation_related_controls()
            nx = np.shape(plot_data)[0]
            n_overplots = np.shape(plot_data)[1]
            n_frames = np.shape(plot_data)[2]
            segments = np.zeros((n_overplots, nx, 2))
            for i in range(n_overplots):
                segments[i, :, 1] = plot_data[:, i, 0]
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
        self.variables.n_frames = n_frames

        self.control_panel.scale.config(to=n_frames-1)

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
        animation_index = int(np.round(self.control_panel.scale.get()))
        self.update_plot_animation(animation_index)

    # define custom callbacks here
    def callback_set_y_rescale(self, event):
        selection = self.control_panel.rescale_y_axis_per_frame.get()
        if selection == SCALE_Y_AXIS_PER_FRAME_TRUE:
            self.variables.set_y_margins_per_frame = True
        else:
            self.variables.set_y_margins_per_frame = False
            y_range = self.variables.plot_data.max() - self.variables.plot_data.min()
            self.variables.ymin = self.variables.plot_data.min() - y_range * self.variables.y_margin
            self.variables.ymax = self.variables.plot_data.max() + y_range * self.variables.y_margin

    def animate(self):
        fps = float(self.control_panel.fps_entry.get())
        self.control_panel.scale.set(20)
        self.update_plot_animation(20)

        self.control_panel.scale.set(25)
        self.update_plot_animation(25)

        self.control_panel.scale.set(30)
        self.update_plot_animation(30)

        self.control_panel.scale.set(50)
        self.update_plot_animation(50)

    def callback_animate(self, event):
        self.animate()

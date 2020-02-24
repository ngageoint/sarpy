import tkinter as tk
from tkinter_gui_builder.widgets import basic_widgets
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
import time
from tkinter_gui_builder.panel_templates.pyplot_panel.pyplot_panel_utils.plot_style_utils import PlotStyleUtils
import numpy as np


SCALE_Y_AXIS_PER_FRAME_TRUE = "scale y axis per frame"
SCALE_Y_AXIS_PER_FRAME_FALSE = "don't scale y axis per frame"

PYPLOT_UTILS = PlotStyleUtils()


class PyplotCanvas(tk.LabelFrame):
    def __init__(self, master):
        tk.LabelFrame.__init__(self, master)

        fig = Figure()
        self.ax = fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(fig, master=master)
        self.canvas.get_tk_widget().pack(fill='both')


class PyplotControlPanel(AbstractWidgetPanel):
    color_palette_label = basic_widgets.Label                   # type: basic_widgets.Label
    color_palette = basic_widgets.Combobox                      # type: basic_widgets.Combobox
    n_colors_label = basic_widgets.Label                        # type: basic_widgets.Label
    n_colors = basic_widgets.Spinbox                            # type: basic_widgets.Spinbox
    scale = basic_widgets.Scale                                 # type: basic_widgets.Scale
    rescale_y_axis_per_frame = basic_widgets.Combobox           # type: basic_widgets.Combobox
    animate = basic_widgets.Button                              # type: basic_widgets.Button
    fps_label = basic_widgets.Label                             # type: basic_widgets.Label
    fps_entry = basic_widgets.Entry                             # type: basic_widgets.Entry

    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)
        widget_list = ["color_palette_label", "color_palette", "n_colors_label", "n_colors",
                       "scale",
                       "rescale_y_axis_per_frame",
                       "fps_label", "fps_entry", "animate"]

        self.init_w_basic_widget_list(widget_list, 4, [4, 1, 1, 3])
        self.color_palette.update_combobox_values(PYPLOT_UTILS.get_all_palettes_list())
        self.n_colors.config(from_=0)
        self.n_colors.config(to=10)
        self.scale.set(0)
        self.rescale_y_axis_per_frame.update_combobox_values([SCALE_Y_AXIS_PER_FRAME_TRUE, SCALE_Y_AXIS_PER_FRAME_FALSE])
        self.fps_label.set_text("fps")
        self.fps_entry.set_text("30")


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
        self.animation_index = 0


class PyplotPanel(AbstractWidgetPanel):
    pyplot_canvas = PyplotCanvas           # type: PyplotCanvas
    control_panel = PyplotControlPanel      # type: PyplotControlPanel

    def __init__(self, master):
        AbstractWidgetPanel.__init__(self, master)

        self.variables = AppVariables()
        self.pyplot_utils = PlotStyleUtils()
        widget_list = ["pyplot_canvas", "control_panel"]
        self.init_w_vertical_layout(widget_list)

        canvas_size_pixels = self.pyplot_canvas.canvas.figure.get_size_inches() * self.pyplot_canvas.canvas.figure.dpi

        self.control_panel.scale.config(length=canvas_size_pixels[0] * 0.75)
        self.variables.animation_related_controls = [self.control_panel.scale,
                                                     self.control_panel.rescale_y_axis_per_frame,
                                                     self.control_panel.animate,
                                                     self.control_panel.fps_entry,
                                                     self.control_panel.fps_label]

        self.control_panel.n_colors_label.set_text("n colors")
        self.control_panel.n_colors.set_text(self.pyplot_utils.n_color_bins)

        # set listeners
        self.control_panel.scale.on_left_mouse_motion(self.callback_update_from_slider)
        self.control_panel.rescale_y_axis_per_frame.on_selection(self.callback_set_y_rescale)
        self.control_panel.animate.on_left_mouse_click(self.callback_animate)
        self.control_panel.color_palette.on_selection(self.callback_update_plot_colors)
        self.control_panel.n_colors.on_enter_or_return_key(self.callback_update_n_colors)
        self.control_panel.n_colors.on_left_mouse_release(self.callback_spinbox_update_n_colors)

        self.hide_animation_related_controls()

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
        self.update_animation_index(animation_index)
        self.update_plot()

    def update_animation_index(self, animation_index):
        self.control_panel.scale.set(animation_index)
        self.variables.animation_index = animation_index

    def callback_update_from_slider(self, event):
        self.variables.animation_index = int(np.round(self.control_panel.scale.get()))
        self.update_plot()

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
        self.update_plot()

    def animate(self, start_frame, stop_frame, fps):
        time_between_frames = 1/fps

        for i in range(start_frame, stop_frame):
            tic = time.time()
            self.update_animation_index(i)
            self.update_plot()
            toc = time.time()
            time_to_update_plot = toc-tic
            if time_between_frames > time_to_update_plot:
                time.sleep(time_between_frames - time_to_update_plot)
            else:
                pass

    def callback_animate(self, event):
        start_frame = 0
        stop_frame = self.variables.n_frames
        fps = float(self.control_panel.fps_entry.get())
        self.animate(start_frame, stop_frame, fps)

    def callback_update_plot_colors(self, event):
        color_palette_text = self.control_panel.color_palette.get()
        self.pyplot_utils.set_palette_by_name(color_palette_text)
        self.update_plot()

    def callback_update_n_colors(self, event):
        n_colors = int(self.control_panel.n_colors.get())
        self.pyplot_utils.set_n_colors(n_colors)
        print(self.pyplot_utils.rgb_array_full_palette)
        self.update_plot()

    def callback_spinbox_update_n_colors(self, event):
        self.after(100, self.update_plot())

    def update_plot(self):
        # time.sleep(1)
        if self.variables.plot_data is not None:
            n_overplots = np.shape(self.variables.segments)[0]
            animation_index = int(self.control_panel.scale.get())
            for i in range(n_overplots):
                self.variables.segments[i, :, 1] = self.variables.plot_data[:, i, animation_index]

            self.pyplot_canvas.ax.clear()

            self.pyplot_canvas.ax.set_xlim(self.variables.xmin, self.variables.xmax)
            line_segments = LineCollection(self.variables.segments,
                                           self.pyplot_utils.linewidths,
                                           linestyle=self.pyplot_utils.linestyle)
            line_segments.set_color(self.pyplot_utils.rgb_array_full_palette)
            if self.variables.set_y_margins_per_frame:
                plot_data = self.variables.segments[:, :, 1]
                y_range = plot_data.max() - plot_data.min()
                self.variables.ymin = plot_data.min() - y_range * self.variables.y_margin
                self.variables.ymax = plot_data.max() + y_range * self.variables.y_margin
            self.pyplot_canvas.ax.set_ylim(self.variables.ymin, self.variables.ymax)

            self.pyplot_canvas.ax.add_collection(line_segments)
            self.pyplot_canvas.canvas.draw()
        else:
            pass

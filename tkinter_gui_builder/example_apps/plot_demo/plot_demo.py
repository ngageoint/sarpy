import tkinter
from tkinter_gui_builder.panel_templates.pyplot_panel.pyplot_panel import PyplotPanel
from tkinter_gui_builder.example_apps.plot_demo.panels.plot_demo_button_panel import ButtonPanel
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
import numpy as np


class PlotDemo(AbstractWidgetPanel):
    button_panel = ButtonPanel          # type: ButtonPanel
    pyplot_panel = PyplotPanel      # type: PyplotPanel

    def __init__(self, master):
        # set the master frame
        master_frame = tkinter.Frame(master)
        AbstractWidgetPanel.__init__(self, master_frame)
        widget_list = ["pyplot_panel", "button_panel"]
        self.init_w_vertical_layout(widget_list)

        # need to pack both master frame and self, since this is the main app window.
        master_frame.pack()
        self.pack()

        # set up event listeners
        self.button_panel.single_plot.on_left_mouse_click(self.callback_single_plot)
        self.button_panel.multi_plot.on_left_mouse_click(self.callback_muli_plot)
        self.button_panel.animated_plot.on_left_mouse_click(self.callback_animated_plot)

        self.pyplot_panel.set_y_margin_percent(5)
        self.pyplot_panel.variables.set_y_margins_per_frame = True

    def callback_single_plot(self, event):
        plot_data = self.mockup_animation_data_1()
        self.pyplot_panel.set_data(plot_data)

    def callback_muli_plot(self, event):
        plot_data = self.mockup_animation_data_2()
        data_shape = np.shape(plot_data)
        x_axis_points = data_shape[0]
        n_overplots = data_shape[1]
        print("plot data has dimensions of: " + str(data_shape))
        print("with " + str(x_axis_points) + " data points along the x axis")
        print("and " + str(n_overplots) + " overplots")
        self.pyplot_panel.set_data(plot_data)

    def callback_animated_plot(self, event):
        plot_data = self.mockup_animation_data_3()
        data_shape = np.shape(plot_data)
        x_axis_points = data_shape[0]
        n_overplots = data_shape[1]
        n_animation_frames = data_shape[2]
        print("plot data has dimensions of: " + str(data_shape))
        print("with " + str(x_axis_points) + " data points along the x axis")
        print("and " + str(n_overplots) + " overplots")
        print("and " + str(n_animation_frames) + " animation frames")
        self.pyplot_panel.set_data(plot_data)

    @staticmethod
    def mockup_animation_data_3():
        n_overplots = 10
        nx = 200
        n_times = 100

        x_axis = np.linspace(0, np.pi, nx)
        y_data_1 = np.sin(x_axis)
        y_data_2 = np.zeros((len(x_axis), n_overplots))
        y_data_3 = np.zeros((len(x_axis), n_overplots, n_times))

        scaling_factors = np.linspace(0.7, 1, n_overplots)

        for i in range(n_overplots):
            y_data_2[:, i] = y_data_1 * scaling_factors[i]

        x_over_time = np.zeros((nx, n_times))
        x_over_time_start = np.linspace(0, np.pi, n_times)
        for i in range(n_times):
            x_start = x_over_time_start[i]
            x = np.linspace(x_start, np.pi + x_start, nx)
            x_over_time[:, i] = x
            y = np.sin(x)
            for j in range(n_overplots):
                y_data_3[:, j, i] = y * scaling_factors[j]
        return y_data_3

    @staticmethod
    def mockup_animation_data_2():
        n_overplots = 10
        nx = 200

        x_axis = np.linspace(0, 2 * np.pi, nx)
        y_data_1 = x_axis
        y_data_2 = np.zeros((len(x_axis), n_overplots))

        scaling_factors = np.linspace(0.7, 1, n_overplots)

        for i in range(n_overplots):
            y_data_2[:, i] = y_data_1 * scaling_factors[i]

        return y_data_2

    @staticmethod
    def mockup_animation_data_1():
        x = np.linspace(-5, 5, 200)
        y = np.sinc(x)
        return y


if __name__ == '__main__':
    root = tkinter.Tk()
    app = PlotDemo(root)
    root.mainloop()

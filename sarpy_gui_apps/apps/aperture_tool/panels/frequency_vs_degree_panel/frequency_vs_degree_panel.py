import tkinter
from tkinter_gui_builder.widgets.image_canvas import ImageCanvas
import numpy
import math

class FrequencyVsDegreePanel(tkinter.LabelFrame):
    def __init__(self, parent, canvas_width=600, canvas_height=400):
        tkinter.LabelFrame.__init__(self, parent)
        self.config(highlightbackground="black")
        self.config(highlightthickness=1)
        self.config(borderwidth=5)

        # this is a dummy placeholder for now
        self.image_data = numpy.random.random((canvas_height, canvas_width))

        # default dpi is 100, so npix will be 100 times the numbers passed to figsize

        self.labels_canvas = ImageCanvas(self)
        self.labels_canvas.set_canvas_size(canvas_width*1.4, canvas_height*1.4)
        self.canvas = ImageCanvas(self.labels_canvas)
        self.canvas.set_canvas_size(canvas_width, canvas_height)
        self.labels_canvas.pack(expand=tkinter.Y, fill=tkinter.BOTH)

        self.x_margin = (self.labels_canvas.canvas_width - self.canvas.canvas_width)/2
        self.y_margin = (self.labels_canvas.canvas_height - self.canvas.canvas_height)/2
        self.labels_canvas.create_window(self.x_margin, self.y_margin, anchor=tkinter.NW, window=self.canvas)

        self.canvas.on_mouse_wheel(self.callback_do_nothing)

        # self.canvas.pack(expand=tkinter.Y, fill=tkinter.BOTH)
        #self.canvas.update_image(self.image_data)

    def callback_do_nothing(self, event):
        pass

    def update_x_axis(self, start_val=-10, stop_val=10, label=None):
        n_ticks = 5
        display_image = self.canvas.variables.canvas_image_object.display_image
        image_width = numpy.shape(display_image)[1]
        left_pixel_index = self.x_margin + 2
        right_pixel_index = self.x_margin + image_width
        bottom_pixel_index = self.y_margin + self.canvas.canvas_height + 20
        label_y_index = bottom_pixel_index + 30

        tick_vals = numpy.linspace(start_val, stop_val, n_ticks)
        x_axis_positions = numpy.linspace(left_pixel_index, right_pixel_index, n_ticks)

        tick_positions = []
        for x in x_axis_positions:
            tick_positions.append((x, bottom_pixel_index))

        self.labels_canvas.variables.foreground_color = "black"

        for xy, tick_val in zip(tick_positions, tick_vals):
            self.labels_canvas.create_text(xy, text=tick_val, fill="black", anchor="n")

        if label:
            self.labels_canvas.create_text((x_axis_positions[int(n_ticks/2)], label_y_index), text=label, fill="black", anchor="n")

    def update_y_axis(self, start_val, stop_val, label=None, n_ticks=5):
        display_image = self.canvas.variables.canvas_image_object.display_image
        image_width = numpy.shape(display_image)[1]
        left_pixel_index = self.x_margin - 40
        right_pixel_index = self.x_margin + image_width
        top_pixel_index = self.y_margin
        bottom_pixel_index = self.y_margin + self.canvas.canvas_height
        label_x_index = left_pixel_index - 30

        tick_vals = numpy.linspace(stop_val, start_val, n_ticks)
        y_axis_positions = numpy.linspace(top_pixel_index, bottom_pixel_index, n_ticks)

        tick_positions = []
        for y in y_axis_positions:
            tick_positions.append((left_pixel_index, y))

        self.labels_canvas.variables.foreground_color = "black"

        for xy, tick_val in zip(tick_positions, tick_vals):
            self.labels_canvas.create_text(xy, text=tick_val, fill="black", anchor="w")

        if label:
            self.labels_canvas.create_text((label_x_index, y_axis_positions[int(n_ticks/2)]), text=label, fill="black", anchor="s", angle=90, justify="right")


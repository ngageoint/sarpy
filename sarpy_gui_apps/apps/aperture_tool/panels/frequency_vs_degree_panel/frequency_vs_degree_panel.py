import tkinter
from tkinter_gui_builder.widgets.image_canvas import ImageCanvas
import numpy


class FrequencyVsDegreePanel(tkinter.LabelFrame):
    def __init__(self, parent, canvas_width=600, canvas_height=400,
                 left_margin=0.2, right_margin=0,
                 top_margin=0.2, bottom_margin=0.2):
        tkinter.LabelFrame.__init__(self, parent)
        self.image_data = numpy.random.random((canvas_height, canvas_width))

        # default dpi is 100, so npix will be 100 times the numbers passed to figsize

        self.labels_canvas = ImageCanvas(self)
        self.labels_canvas.set_canvas_size(canvas_width * (1 + left_margin + right_margin), canvas_height * (1 + top_margin + bottom_margin))
        self.canvas = ImageCanvas(self.labels_canvas)
        self.canvas.set_canvas_size(canvas_width, canvas_height)
        self.labels_canvas.pack(expand=tkinter.Y, fill=tkinter.BOTH)

        self.x_margin = canvas_width * left_margin
        self.y_margin = canvas_height * top_margin
        self.labels_canvas.create_window(self.x_margin, self.y_margin, anchor=tkinter.NW, window=self.canvas)

        self.canvas.zoom_on_wheel = False

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

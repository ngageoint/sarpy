import tkinter
from tkinter_gui_builder.widgets.image_canvas import ImageCanvas
import matplotlib.pyplot as plt
import numpy
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class FrequencyVsDegreePanel(tkinter.LabelFrame):
    def __init__(self, parent, canvas_width=600, canvas_height=400):
        tkinter.LabelFrame.__init__(self, parent)
        self.config(highlightbackground="black")
        self.config(highlightthickness=1)
        self.config(borderwidth=5)

        # this is a dummy placeholder for now
        self.image_data = numpy.zeros((canvas_height, canvas_width))

        # default dpi is 100, so npix will be 100 times the numbers passed to figsize
        fig = plt.figure(figsize=(canvas_width/100, canvas_height/100))
        plt.imshow(self.image_data)
        self.figure_canvas = FigureCanvasTkAgg(fig, master=self)
        self.figure_canvas.get_tk_widget().pack()

        self.update_image(self.image_data)
        self.canvas = ImageCanvas(self)
        self.figure_canvas._tkcanvas = self.canvas
        # self.canvas.pack()

    def update_image(self, image_data):
        self.image_data = image_data
        plt.imshow(self.image_data)
        self.figure_canvas.draw()

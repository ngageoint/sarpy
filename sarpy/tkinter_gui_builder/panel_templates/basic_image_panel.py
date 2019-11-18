import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import functools


class BasicImagePanel(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.config(highlightbackground="black")
        self.config(highlightthickness=1)
        self.config(borderwidth=5)
        self.rows = None           # type: tk.Frame

        image_data = np.zeros((200, 200))
        image_data[30:40, 30:40] = 0.5
        image_data[40:50, 40:50] = 1.0

        fig = plt.figure(figsize=(5, 4))
        self.im = plt.imshow(image_data)  # later use a.set_data(new_data)
        self.ax = plt.gca()
        self.image = self.ax.imshow(image_data)

        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=1)

    def callback_update_image(self, event):
        self.update_image()

    def update_image(self):
        new_image = np.random.random((200, 200))
        self.im.set_data(new_image)
        # ax = plt.gca()
        self.ax.imshow(new_image)
        self.canvas.draw()

    def callback_update_image2(self, event):
        self.update_image()

    def update_image2(self):
        new_image = np.random.random((200, 200))
        self.im.set_data(new_image)
        # ax = plt.gca()
        self.ax.imshow(new_image)
        self.canvas.draw()

    def on_left_mouse_click_with_args(self, event, args):
        self.bind("<Button-1>", functools.partial(self._partial_callback_w_param, param=args))

    def _partial_callback_w_param(self, event, param):
        print(param)
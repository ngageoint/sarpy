import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Canvas


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

    def update_image(self, image_data):
        self.im.set_data(image_data)
        self.ax.imshow(image_data)
        self.canvas.draw()

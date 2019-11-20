import PIL.Image
from PIL import ImageTk
import tkinter as tk
from sarpy.tkinter_gui_builder.widget_utils.basic_widgets import Canvas
import os
import numpy as np


class BasicImageCanvasPanel(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.x = 0
        self.y = 0
        self.canvas = Canvas(self, width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=tk.YES)
        self.canvas.pack()

        self.sbarv=tk.Scrollbar(self, orient=tk.VERTICAL)
        self.sbarh=tk.Scrollbar(self, orient=tk.HORIZONTAL)
        self.sbarv.config(command=self.canvas.yview)
        self.sbarh.config(command=self.canvas.xview)

        self.canvas.config(yscrollcommand=self.sbarv.set)
        self.canvas.config(xscrollcommand=self.sbarh.set)

        self.canvas.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.sbarv.grid(row=0, column=1, stick=tk.N+tk.S)
        self.sbarh.grid(row=1, column=0, sticky=tk.E+tk.W)

        self.rect = None
        self.line = None

        self.start_x = None
        self.start_y = None

        self.im = PIL.Image.open(os.path.expanduser("~/Pictures/snek.jpg"))
        # self.im = None
        self.nx_pix, self.ny_pix = self.im.size
        self.canvas.config(scrollregion=(0, 0, self.nx_pix, self.ny_pix))
        self.tk_im = ImageTk.PhotoImage(self.im)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_im)

    def event_initiate_rect(self, event):
        # save mouse drag start position
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        # create rectangle if not yet exist
        if not self.rect:
            self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, outline='red')

    def event_drag_rect(self, event):
        event_x_pos = self.canvas.canvasx(event.x)
        event_y_pos = self.canvas.canvasy(event.y)

        # expand rectangle as you drag the mouse
        self.canvas.coords(self.rect, self.start_x, self.start_y, event_x_pos, event_y_pos)

    def event_initiate_line(self, event):
        # save mouse drag start position
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        # create rectangle if not yet exist
        if not self.line:
            self.line = self.canvas.create_line(self.x, self.y, 1, 1, fill='blue')

    def event_drag_line(self, event):
        event_x_pos = self.canvas.canvasx(event.x)
        event_y_pos = self.canvas.canvasy(event.y)

        # expand rectangle as you drag the mouse
        self.canvas.coords(self.line, self.start_x, self.start_y, event_x_pos, event_y_pos)

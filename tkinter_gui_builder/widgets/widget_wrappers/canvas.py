import tkinter as tk
from tkinter_gui_builder.widgets.widget_utils.widget_events import WidgetEvents


class Canvas(tk.Canvas, WidgetEvents):
    def __init__(self, master=None, cnf={}, **kw):
        super(tk.Canvas, self).__init__(master, 'canvas', cnf, kw)
        super(WidgetEvents, self).__init__()

    def set_width(self, width_npix):
        self.config(width=width_npix)

    def set_height(self, height_npix):
        self.config(height=height_npix)

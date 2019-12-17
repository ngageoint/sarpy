import tkinter as tk
from sarpy.tkinter_gui_builder.widget_utils.widget_events import WidgetEvents


class Label(tk.Label, WidgetEvents):
    def __init__(self, master=None, cnf={}, **kw):
        super(tk.Label, self).__init__(master, 'label', cnf, kw)
        super(WidgetEvents, self).__init__()

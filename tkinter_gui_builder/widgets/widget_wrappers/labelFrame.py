import tkinter as tk
from tkinter_gui_builder.widgets.widget_utils.widget_events import WidgetEvents


class LabelFrame(tk.LabelFrame, WidgetEvents):
    def __init__(self, master=None, cnf={}, **kw):
        super(tk.LabelFrame, self).__init__(master, 'labelframe', cnf, kw)
        super(WidgetEvents, self).__init__()

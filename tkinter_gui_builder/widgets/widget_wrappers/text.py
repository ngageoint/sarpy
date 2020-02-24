import tkinter as tk
from tkinter_gui_builder.widgets.widget_utils.widget_events import WidgetEvents


class Text(tk.Text, WidgetEvents):
    def __init__(self, master=None, cnf={}, **kw):
        super(tk.Text, self).__init__(master, 'text', cnf, kw)
        super(WidgetEvents, self).__init__()

import tkinter as tk
from tkinter_gui_builder.widgets.widget_utils.widget_events import WidgetEvents


class Entry(tk.Entry, WidgetEvents):
    def __init__(self, master=None, cnf={}, **kw):
        super(tk.Entry, self).__init__(master, 'entry', cnf, kw)
        super(WidgetEvents, self).__init__()

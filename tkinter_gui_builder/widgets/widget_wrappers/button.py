import tkinter as tk
from tkinter_gui_builder.widgets.widget_utils.widget_events import WidgetEvents


class Button(tk.Button, WidgetEvents):
    def __init__(self, master=None, cnf={}, **kw):
        super(tk.Button, self).__init__(master, 'button', cnf, kw)
        super(WidgetEvents, self).__init__()

    # event_handling
    def set_text(self, text):
        self.config(text=text)

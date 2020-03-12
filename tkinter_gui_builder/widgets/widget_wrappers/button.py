from tkinter_gui_builder import tkinter as tk
from tkinter_gui_builder.widgets.widget_utils.widget_events import WidgetEvents


class Button(tk.Button, WidgetEvents):
    def __init__(self, master=None, cnf=None, **kw):
        cnf = {} if cnf is None else cnf
        tk.Button.__init__(self, master=master, cnf=cnf, **kw)

    # event_handling
    def set_text(self, text):
        self.config(text=text)

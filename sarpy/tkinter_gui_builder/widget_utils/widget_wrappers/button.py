import tkinter as tk
from sarpy.tkinter_gui_builder.widget_utils.widget_events import WidgetEvents


class Button(tk.Button, WidgetEvents):
    def __init__(self, parent):
        self.parent = parent
        tk.Button.__init__(self, parent)

    # event_handling
    def set_text(self, text):
        self.config(text=text)

import tkinter as tk
from tkinter_gui_builder.widgets.widget_utils.widget_events import WidgetEvents


class Spinbox(tk.Spinbox, WidgetEvents):
    def __init__(self, master=None, cnf={}, **kw):
        super(tk.Spinbox, self).__init__(master, 'spinbox', cnf, kw)
        super(WidgetEvents, self).__init__()

    def set_text(self, text):
        # handle case if the widget is disabled
        entry_state = self['state']
        if entry_state == 'disabled':
            self.config(state='normal')
        self.delete(0, tk.END)
        self.insert(0, text)
        if entry_state == 'disabled':
            self.config(state='disabled')

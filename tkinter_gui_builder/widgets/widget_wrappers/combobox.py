from tkinter import ttk
from tkinter_gui_builder.widgets.widget_utils.widget_events import WidgetEvents


class Combobox(ttk.Combobox, WidgetEvents):
    def __init__(self, master=None, **kw):
        ttk.Combobox.__init__(self, master, "ttk::combobox", **kw)

    def on_selection(self, event):
        self.bind("<<ComboboxSelected>>", event)

    def update_combobox_values(self, val_list):
        self['values'] = val_list
        self.current(0)

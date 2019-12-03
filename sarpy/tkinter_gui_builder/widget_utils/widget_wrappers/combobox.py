from tkinter import ttk
from sarpy.tkinter_gui_builder.widget_utils.widget_events import WidgetEvents


class Combobox(ttk.Combobox, WidgetEvents):
    def __init__(self, master=None, **kw):
        super(ttk.Combobox, self).__init__(master, "ttk::combobox", **kw)
        super(WidgetEvents, self).__init__()

    def on_selection(self, event):
        self.bind("<<ComboboxSelected>>", event)

    def update_combobox_values(self, val_list):
        self['values'] = val_list
        self.current(0)
from tkinter import ttk
from tkinter_gui_builder.widgets.widget_utils.widget_events import WidgetEvents


class Scale(ttk.Scale, WidgetEvents):
    def __init__(self, master=None, **kw):
        super(ttk.Scale, self).__init__(master, "ttk::scale", kw)
        super(WidgetEvents, self).__init__()

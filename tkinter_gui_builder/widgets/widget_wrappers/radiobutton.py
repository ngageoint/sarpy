from tkinter import ttk
from tkinter_gui_builder.widgets.widget_utils.widget_events import WidgetEvents
import tkinter


class RadioButton(tkinter.Radiobutton, WidgetEvents):
    def __init__(self, master=None, **kw):
        ttk.Radiobutton.__init__(self, master, **kw)

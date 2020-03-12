import tkinter
from tkinter_gui_builder.widgets.widget_utils.widget_events import WidgetEvents


class Spinbox(tkinter.Spinbox, WidgetEvents):
    def __init__(self, master=None, cnf=None, **kw):
        tkinter.Spinbox.__init__(self, master=master, **kw)

    def set_text(self, text):
        # handle case if the widget is disabled
        entry_state = self['state']
        if entry_state == 'disabled':
            self.config(state='normal')
        self.delete(0, tkinter.END)
        self.insert(0, text)
        if entry_state == 'disabled':
            self.config(state='disabled')

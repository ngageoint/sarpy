import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.widgets import basic_widgets
import os


class FileSelector(AbstractWidgetPanel):
    select_file = basic_widgets.Button  # type: basic_widgets.Button
    fname_label = basic_widgets.Label  # type: basic_widgets.Label

    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)
        tk.LabelFrame.__init__(self, parent)
        self.config(borderwidth=2)
        self.fname = None

        widget_list = ["select_file", "fname_label"]
        self.init_w_horizontal_layout(widget_list)
        self.set_label_text("file selector")
        self.fname_filters = []
        # in practice this would be overridden if the user wants more things to happen after selecting a file.
        self.select_file.on_left_mouse_click(self.event_select_file)
        self.initialdir = os.path.expanduser("~")

    def set_fname_filters(self,
                          filter_list,          # type: [str]
                          ):
        self.fname_filters = filter_list

    def set_initial_dir(self, directory):
        self.initialdir = directory

    def event_select_file(self, event):
        ftypes = [
            ('image files', self.fname_filters),
            ('All files', '*'),
        ]
        self.fname = askopenfilename(initialdir=self.initialdir, filetypes=ftypes)
        self.fname_label.config(text=self.fname)

import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter_gui_builder.panel_templates.basic_widgets_panel import BasicWidgetsPanel
from tkinter_gui_builder.widgets import basic_widgets


class FileSelector(BasicWidgetsPanel):
    def __init__(self, parent):
        BasicWidgetsPanel.__init__(self, parent)
        tk.LabelFrame.__init__(self, parent)
        self.config(borderwidth=2)
        self.fname = None

        self.select_button = basic_widgets.Button
        self.fname_label = basic_widgets.Label

        self.widget_list = [("select_button", "select file"), "fname_label"]
        self.init_w_horizontal_layout(self.widget_list)
        self.set_label_text("file selector")
        # in practice this would be overridden if the user wants more things to happen after selecting a file.
        self.select_button.config(command=self.update_fname)

    def update_fname(self):
        self.fname = askopenfilename()
        self.fname_label.config(text=self.fname)

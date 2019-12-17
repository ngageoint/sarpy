import tkinter as tk
from tkinter.filedialog import askopenfilename
from sarpy.tkinter_gui_builder.panel_templates.basic_widgets_panel import BasicWidgetsPanel
from sarpy.tkinter_gui_builder.widget_utils.basic_widgets import Label
from sarpy.tkinter_gui_builder.widget_utils.basic_widgets import Button


class FileSelector(BasicWidgetsPanel):
    def __init__(self, parent):
        BasicWidgetsPanel.__init__(self, parent)
        tk.LabelFrame.__init__(self, parent)
        self.config(borderwidth=2)
        self.fname = None

        self.select_button = Button
        self.fname_label = Label

        self.widget_list = [("select_button", "select file"), "fname_label"]
        self.init_w_horizontal_layout(self.widget_list)
        self.set_label_text("file selector")
        # in practice this would be overridden if the user wants more things to happen after selecting a file.
        self.select_button.config(command=self.update_fname)

    def update_fname(self):
        self.fname = askopenfilename()
        self.fname_label.config(text=self.fname)

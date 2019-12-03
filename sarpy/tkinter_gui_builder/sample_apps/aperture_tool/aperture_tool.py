import tkinter
from tkinter.filedialog import askopenfilename
from sarpy.tkinter_gui_builder.sample_apps.taser_wip.custom_panels.button_panel import ButtonPanel
from sarpy.tkinter_gui_builder.panel_templates.basic_pyplot_image_panel import BasicPyplotImagePanel
from sarpy.tkinter_gui_builder.sample_apps.taser_wip.custom_panels.taser_canvas_panel import TaserImageCanvasPanel
import numpy as np
import imageio
import os


class AppVariables:
    def __init__(self):
        self.fname = "None"       # type: str


class ApertureTool:
    def __init__(self, master):
        # Create a container
        # set the master frame
        master_frame = tkinter.Frame(master)
        self.app_variables = AppVariables()

        # define panels widget_wrappers in master frame
        self.button_panel = ButtonPanel(master_frame)
        self.button_panel.set_spacing_between_buttons(0)
        self.pyplot_panel = BasicPyplotImagePanel(master_frame, 800, 600)
        self.taser_image_panel = TaserImageCanvasPanel(master_frame)
        self.taser_image_panel.set_canvas_size(600, 400)

        # specify layout of widget_wrappers in master frame
        self.button_panel.pack(side="left")
        self.taser_image_panel.pack(side="left")
        self.pyplot_panel.pack(side="left")

        master_frame.pack()


root = tkinter.Tk()
app = ApertureTool(root)
root.mainloop()

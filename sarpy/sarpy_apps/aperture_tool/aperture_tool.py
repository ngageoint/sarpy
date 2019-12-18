import tkinter
from sarpy.sarpy_apps.taser_tool.panels import TaserButtonPanel
from tkinter_gui_builder.panel_templates.basic_plot_panel import BasicPlotPanel
from sarpy.sarpy_apps.taser_tool.panels import TaserImageCanvasPanel


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
        self.button_panel = TaserButtonPanel(master_frame)
        self.button_panel.set_spacing_between_buttons(0)
        self.pyplot_panel = BasicPlotPanel(master_frame)
        self.taser_image_panel = TaserImageCanvasPanel(master_frame)
        self.taser_image_panel.set_canvas_size(600, 400)

        # specify layout of widget_wrappers in master frame
        self.button_panel.pack(side="left")
        self.taser_image_panel.pack(side="left")
        self.pyplot_panel.pack(side="left")

        master_frame.pack()


if __name__ == '__main__':
    root = tkinter.Tk()
    app = ApertureTool(root)
    root.mainloop()

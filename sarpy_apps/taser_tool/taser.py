import tkinter
from tkinter.filedialog import askopenfilename
from sarpy_apps.taser_tool.panels.taser_button_panel import TaserButtonPanel
from tkinter_gui_builder.panel_templates.basic_pyplot_image_panel import BasicPyplotImagePanel
from tkinter_gui_builder.panel_templates.image_canvas import ImageCanvas
from sarpy_apps.sarpy_app_helper_utils.sarpy_canvas_image import SarpyCanvasDisplayImage
import numpy as np
import os


class AppVariables:
    def __init__(self):
        self.fname = "None"       # type: str
        self.remap_type = "density"
        self.selection_rect_id = None           # type: int


class Taser:
    def __init__(self, master):
        # Create a container
        # set the master frame
        master_frame = tkinter.Frame(master)
        self.app_variables = AppVariables()

        # define panels widget_wrappers in master frame
        self.button_panel = TaserButtonPanel(master_frame)
        self.button_panel.set_spacing_between_buttons(0)
        self.pyplot_panel = BasicPyplotImagePanel(master_frame, 800, 600)
        self.taser_image_panel = ImageCanvas(master_frame)
        self.taser_image_panel.variables.canvas_image_object = SarpyCanvasDisplayImage()        # type: SarpyCanvasDisplayImage
        self.taser_image_panel.set_canvas_size(600, 400)

        # specify layout of widget_wrappers in master frame
        self.button_panel.pack(side="left")
        self.taser_image_panel.pack(side="left")
        self.pyplot_panel.pack(side="left")

        master_frame.pack()

        # bind events to callbacks here
        self.button_panel.fname_select.on_left_mouse_click(self.callback_initialize_canvas_image)
        self.button_panel.update_rect_image.on_left_mouse_click(self.callback_display_canvas_rect_selection_in_pyplot_frame)
        self.button_panel.remap_dropdown.on_selection(self.callback_remap)
        self.button_panel.zoom_in.on_left_mouse_click(self.callback_set_to_zoom_in)
        self.button_panel.zoom_out.on_left_mouse_click(self.callback_set_to_zoom_out)
        self.button_panel.rect_select.on_left_mouse_click(self.callback_set_to_select)

    def callback_set_to_zoom_in(self, event):
        self.taser_image_panel.set_current_tool_to_zoom_in()
        self.taser_image_panel.hide_shape(self.taser_image_panel.variables.select_rect_id)

    def callback_set_to_zoom_out(self, event):
        self.taser_image_panel.set_current_tool_to_zoom_out()
        self.taser_image_panel.hide_shape(self.taser_image_panel.variables.select_rect_id)

    def callback_set_to_select(self, event):
        self.taser_image_panel.set_current_tool_to_draw_rect(self.taser_image_panel.variables.select_rect_id)

    # define custom callbacks here
    def callback_remap(self, event):
        remap_dict = {"density": "density",
                      "brighter": "brighter",
                      "darker": "darker",
                      "high contrast": "highcontrast",
                      "linear": "linear",
                      "log": "log",
                      "pedf": "pedf",
                      "nrl": "nrl"}
        selection = self.button_panel.remap_dropdown.get()
        remap_type = remap_dict[selection]
        self.taser_image_panel.variables.canvas_image_object.remap_type = remap_type

    def callback_initialize_canvas_image(self, event):
        image_file_extensions = ['*.nitf', '*.NITF']
        ftypes = [
            ('image files', image_file_extensions),
            ('All files', '*'),
        ]
        new_fname = askopenfilename(initialdir=os.path.expanduser("~"), filetypes=ftypes)
        if new_fname:
            self.app_variables.fname = new_fname
        self.taser_image_panel.set_canvas_image_from_fname(self.app_variables.fname)

    def callback_display_canvas_rect_selection_in_pyplot_frame(self, event):
        complex_data = self.taser_image_panel.get_image_data_in_canvas_rect_by_id(self.taser_image_panel.variables.current_shape_id)
        remapped_data = self.taser_image_panel.variables.canvas_image_object.remap_complex_data(complex_data)
        self.pyplot_panel.update_image(remapped_data)


if __name__ == '__main__':
    root = tkinter.Tk()
    app = Taser(root)
    root.mainloop()

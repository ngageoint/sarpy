import os

import tkinter
from tkinter.filedialog import askopenfilename
from tkinter_gui_builder.panel_templates.pyplot_image_panel.pyplot_image_panel import PyplotImagePanel
from tkinter_gui_builder.panel_templates.image_canvas.image_canvas import ImageCanvas
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel

from sarpy_gui_apps.apps.taser_tool.panels.taser_button_panel import TaserButtonPanel
from sarpy_gui_apps.supporting_classes.sicd_image_reader import SicdImageReader


class AppVariables:
    def __init__(self):
        self.fname = "None"       # type: str
        self.remap_type = "density"
        self.image_reader = None     # type: SicdImageReader


class Taser(AbstractWidgetPanel):
    button_panel = TaserButtonPanel         # type: TaserButtonPanel
    pyplot_panel = PyplotImagePanel         # type: PyplotImagePanel
    taser_image_panel = ImageCanvas         # type: ImageCanvas

    def __init__(self, master):
        master_frame = tkinter.Frame(master)
        AbstractWidgetPanel.__init__(self, master_frame)
        self.variables = AppVariables()

        widget_list = ["button_panel", "taser_image_panel", "pyplot_panel"]
        self.init_w_horizontal_layout(widget_list)

        # define panels widget_wrappers in master frame
        self.button_panel.set_spacing_between_buttons(0)
        self.taser_image_panel.variables.canvas_image_object = ImageCanvas  # type: ImageCanvas
        self.taser_image_panel.set_canvas_size(700, 400)

        # need to pack both master frame and self, since this is the main app window.
        master_frame.pack()
        self.pack()

        # bind events to callbacks here
        self.button_panel.fname_select.on_left_mouse_click(self.callback_initialize_canvas_image)
        self.button_panel.remap_dropdown.on_selection(self.callback_remap)
        self.button_panel.zoom_in.on_left_mouse_click(self.callback_set_to_zoom_in)
        self.button_panel.zoom_out.on_left_mouse_click(self.callback_set_to_zoom_out)
        self.button_panel.pan.on_left_mouse_click(self.callback_set_to_pan)
        self.button_panel.rect_select.on_left_mouse_click(self.callback_set_to_select)

        self.taser_image_panel.canvas.on_left_mouse_release(self.callback_left_mouse_release)

    def callback_left_mouse_release(self, event):
        self.taser_image_panel.callback_handle_left_mouse_release(event)
        if self.taser_image_panel.variables.current_tool == self.taser_image_panel.TOOLS.SELECT_TOOL:
            self.taser_image_panel.zoom_to_selection((0, 0, self.taser_image_panel.canvas_width, self.taser_image_panel.canvas_height), animate=False)
            self.display_canvas_rect_selection_in_pyplot_frame()

    def callback_set_to_zoom_in(self, event):
        self.taser_image_panel.set_current_tool_to_zoom_in()

    def callback_set_to_zoom_out(self, event):
        self.taser_image_panel.set_current_tool_to_zoom_out()

    def callback_set_to_pan(self, event):
        self.taser_image_panel.set_current_tool_to_pan()
        self.taser_image_panel.hide_shape(self.taser_image_panel.variables.zoom_rect_id)

    def callback_set_to_select(self, event):
        self.taser_image_panel.set_current_tool_to_selection_tool()

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
        self.variables.image_reader.remap_type = remap_type
        self.display_canvas_rect_selection_in_pyplot_frame()
        self.taser_image_panel.update_current_image()

    def callback_initialize_canvas_image(self, event):
        image_file_extensions = ['*.nitf', '*.NITF']
        ftypes = [
            ('image files', image_file_extensions),
            ('All files', '*'),
        ]
        new_fname = askopenfilename(initialdir=os.path.expanduser("~"), filetypes=ftypes)
        if new_fname:
            self.variables.fname = new_fname
            self.variables.image_reader = SicdImageReader(self.variables.fname)
            self.taser_image_panel.set_image_reader(self.variables.image_reader)

    def display_canvas_rect_selection_in_pyplot_frame(self):
        image_data = self.taser_image_panel.get_image_data_in_canvas_rect_by_id(self.taser_image_panel.variables.select_rect_id)
        self.pyplot_panel.update_image(image_data)


if __name__ == '__main__':
    root = tkinter.Tk()
    app = Taser(root)
    root.mainloop()

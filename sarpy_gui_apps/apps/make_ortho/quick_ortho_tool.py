import tkinter
from tkinter.filedialog import askopenfilename
from sarpy_gui_apps.apps.make_ortho.panels.ortho_button_panel import OrthoButtonPanel
from tkinter_gui_builder.panel_templates.image_canvas.image_canvas import ImageCanvas
from sarpy_gui_apps.supporting_classes.sarpy_canvas_image import SarpyCanvasDisplayImage
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
import os


class AppVariables:
    def __init__(self):
        self.fname = "None"       # type: str
        self.remap_type = "density"


class Ortho(AbstractWidgetPanel):
    button_panel = OrthoButtonPanel         # type: TaserButtonPanel
    raw_frame_image_panel = ImageCanvas     # type: ImageCanvas
    ortho_image_panel = ImageCanvas         # type: ImageCanvas

    def __init__(self, master):
        master_frame = tkinter.Frame(master)
        AbstractWidgetPanel.__init__(self, master_frame)
        self.variables = AppVariables()

        widget_list = ["button_panel", "raw_frame_image_panel", "ortho_image_panel"]
        self.init_w_horizontal_layout(widget_list)

        # define panels widget_wrappers in master frame
        self.button_panel.set_spacing_between_buttons(0)
        self.raw_frame_image_panel.variables.canvas_image_object = SarpyCanvasDisplayImage()  # type: SarpyCanvasDisplayImage
        self.raw_frame_image_panel.set_canvas_size(500, 400)
        self.raw_frame_image_panel.rescale_image_to_fit_canvas = True
        self.ortho_image_panel.set_canvas_size(300, 200)
        self.ortho_image_panel.rescale_image_to_fit_canvas = True

        # need to pack both master frame and self, since this is the main app window.
        master_frame.pack()
        self.pack()

        # bind events to callbacks here
        self.button_panel.fname_select.on_left_mouse_click(self.callback_initialize_canvas_image)
        self.button_panel.pan.on_left_mouse_click(self.callback_set_to_pan)
        self.button_panel.display_ortho.on_left_mouse_click(self.callback_display_ortho_image)

    def callback_set_to_pan(self, event):
        self.raw_frame_image_panel.set_current_tool_to_pan()
        self.raw_frame_image_panel.hide_shape(self.raw_frame_image_panel.variables.zoom_rect_id)

    def callback_initialize_canvas_image(self, event):
        image_file_extensions = ['*.nitf', '*.NITF']
        ftypes = [
            ('image files', image_file_extensions),
            ('All files', '*'),
        ]
        new_fname = askopenfilename(initialdir=os.path.expanduser("~"), filetypes=ftypes)
        if new_fname:
            self.variables.fname = new_fname
            self.raw_frame_image_panel.init_with_fname(self.variables.fname)

    def callback_display_ortho_image(self, event):
        orthod_image = self.raw_frame_image_panel.variables.canvas_image_object.create_ortho(self.ortho_image_panel.canvas_height, self.ortho_image_panel.canvas_width)
        self.ortho_image_panel.init_with_numpy_image(orthod_image)


if __name__ == '__main__':
    root = tkinter.Tk()
    app = Ortho(root)
    root.mainloop()

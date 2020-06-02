import os

import tkinter
from tkinter.filedialog import askopenfilename
from sarpy_gui_apps.apps.make_ortho.panels.ortho_button_panel import OrthoButtonPanel
from tkinter_gui_builder.panel_templates.image_canvas_panel.image_canvas_panel import ImageCanvasPanel
from sarpy_gui_apps.supporting_classes.complex_image_reader import ComplexImageReader
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from sarpy_gui_apps.supporting_classes.quick_ortho import QuickOrtho


class Ortho(AbstractWidgetPanel):
    button_panel = OrthoButtonPanel         # type: OrthoButtonPanel
    raw_frame_image_panel = ImageCanvasPanel     # type: ImageCanvasPanel
    ortho_image_panel = ImageCanvasPanel         # type: ImageCanvasPanel

    fname = "None"  # type: str
    remap_type = "density"  # type: str
    image_reader = None  # type: ComplexImageReader

    def __init__(self, master):
        master_frame = tkinter.Frame(master)
        AbstractWidgetPanel.__init__(self, master_frame)

        widget_list = ["button_panel", "raw_frame_image_panel", "ortho_image_panel"]
        self.init_w_horizontal_layout(widget_list)

        # define panels widget_wrappers in master frame
        self.button_panel.set_spacing_between_buttons(0)
        self.raw_frame_image_panel.set_canvas_size(600, 400)
        self.raw_frame_image_panel.canvas.rescale_image_to_fit_canvas = True
        self.ortho_image_panel.set_canvas_size(600, 400)
        self.ortho_image_panel.canvas.rescale_image_to_fit_canvas = True

        # need to pack both master frame and self, since this is the main app window.
        master_frame.pack()
        self.pack()

        # bind events to callbacks here
        self.button_panel.fname_select.on_left_mouse_click(self.callback_set_filename)
        self.button_panel.pan.on_left_mouse_click(self.callback_set_to_pan)
        self.button_panel.display_ortho.on_left_mouse_click(self.callback_display_ortho_image)

    def callback_set_to_pan(self, event):
        self.raw_frame_image_panel.canvas.set_current_tool_to_pan()
        self.raw_frame_image_panel.canvas.hide_shape(self.raw_frame_image_panel.canvas.variables.zoom_rect_id)

    def callback_set_filename(self, event):
        image_file_extensions = ['*.nitf', '*.NITF']
        ftypes = [
            ('image files', image_file_extensions),
            ('All files', '*'),
        ]
        new_fname = askopenfilename(initialdir=os.path.expanduser("~"), filetypes=ftypes)
        if new_fname:
            self.fname = new_fname
            self.image_reader = ComplexImageReader(new_fname)
            self.raw_frame_image_panel.canvas.set_image_reader(self.image_reader)

    def callback_display_ortho_image(self, event):
        ortho_object = QuickOrtho(self.raw_frame_image_panel, self.image_reader)
        orthod_image = ortho_object.create_ortho(self.ortho_image_panel.canvas.canvas_height, self.ortho_image_panel.canvas.canvas_width)
        self.ortho_image_panel.canvas.set_image_from_numpy_array(orthod_image)


if __name__ == '__main__':
    root = tkinter.Tk()
    app = Ortho(root)
    root.mainloop()

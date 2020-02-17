from sarpy_gui_apps.apps.aperture_tool.panels.fft_panel.fft_select_button_panel import FFTSelectButtonPanel
from sarpy_gui_apps.supporting_classes.sarpy_canvas_image import SarpyCanvasDisplayImage
from tkinter_gui_builder.panel_templates.image_canvas.image_canvas import ImageCanvas
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter import filedialog

import os


class FFTPanel(AbstractWidgetPanel):
    fft_button_panel = FFTSelectButtonPanel         # type: FFTSelectButtonPanel
    image_canvas = ImageCanvas                      # type: ImageCanvas

    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)

        widgets_list = ["image_canvas", "fft_button_panel"]

        self.init_w_vertical_layout(widgets_list)

        self.fft_button_panel.set_spacing_between_buttons(0)
        self.image_canvas.variables.canvas_image_object = SarpyCanvasDisplayImage()
        self.image_canvas.set_canvas_size(600, 400)

        # specify layout of widget_wrappers in master frame
        self.fft_button_panel.pack(side="bottom")
        self.image_canvas.pack(side="top")

        self.image_canvas.set_labelframe_text("FFT View")

        # set up event listeners
        self.fft_button_panel.select_data.on_left_mouse_click(self.callback_set_to_selection_tool)
        self.fft_button_panel.move_rect.on_left_mouse_click(self.callback_set_to_translate_shape)
        self.fft_button_panel.save_fft_image_as_png.on_left_mouse_click(self.callback_save_fft_panel_as_png)
        self.image_canvas.set_current_tool_to_selection_tool()

        self.fft_button_panel.n_pixels_horizontal.insert(0, str(int(self.image_canvas.canvas_width*0.5)))
        self.fft_button_panel.n_steps.insert(0, str(20))

    def callback_set_to_selection_tool(self, event):
        self.image_canvas.set_current_tool_to_selection_tool()

    def callback_set_to_translate_shape(self, event):
        self.image_canvas.set_current_tool_to_translate_shape()

    def callback_save_fft_panel_as_png(self, event):
        filename = filedialog.asksaveasfilename(initialdir=os.path.expanduser("~"), title="Select file",
                                                filetypes=(("png file", "*.png"), ("all files", "*.*")))
        self.image_canvas.save_as_png(filename)

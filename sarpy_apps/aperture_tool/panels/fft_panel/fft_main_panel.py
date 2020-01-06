from sarpy_apps.aperture_tool.panels.fft_panel.fft_select_button_panel import FFTSelectButtonPanel
from sarpy_apps.sarpy_app_helper_utils.sarpy_canvas_image import SarpyCanvasDisplayImage
from tkinter_gui_builder.panel_templates.image_canvas.image_canvas import ImageCanvas
from tkinter_gui_builder.panel_templates.basic_widgets_panel import BasicWidgetsPanel


class FFTPanel(BasicWidgetsPanel):
    fft_button_panel = FFTSelectButtonPanel         # type: FFTSelectButtonPanel
    image_canvas = ImageCanvas                      # type: ImageCanvas

    def __init__(self, parent):
        BasicWidgetsPanel.__init__(self, parent)

        widgets_list = ["image_canvas", "fft_button_panel"]

        self.init_w_vertical_layout(widgets_list)

        self.fft_button_panel.set_spacing_between_buttons(0)
        self.image_canvas.variables.canvas_image_object = SarpyCanvasDisplayImage()
        self.image_canvas.set_canvas_size(600, 400)

        # specify layout of widget_wrappers in master frame
        self.fft_button_panel.pack(side="bottom")
        self.image_canvas.pack(side="top")

        # set up event listeners
        self.fft_button_panel.inv_fft.on_left_mouse_click(self.callback_get_adjusted_image)
        self.image_canvas.set_labelframe_text("FFT View")
        self.image_canvas.set_current_tool_to_selection_tool()

    def callback_get_adjusted_image(self, event):
        image_in_rect = self.image_canvas.get_image_data_in_canvas_rect_by_id(self.image_canvas.variables.select_rect_id)
        stop = 1

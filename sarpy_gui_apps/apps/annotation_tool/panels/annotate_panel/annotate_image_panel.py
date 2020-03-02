from sarpy_gui_apps.apps.annotation_tool.panels.annotate_panel.annotate_dashboard.annotate_dashboard import AnnotateDash
from sarpy_gui_apps.supporting_classes.sarpy_canvas_image import SarpyCanvasDisplayImage
from tkinter_gui_builder.panel_templates.image_canvas.image_canvas import ImageCanvas
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel


class AppVariables:
    def __init__(self):
        self.image_fname = "None"       # type: str
        self.sicd_metadata = None


class AnnotateImagePanel(AbstractWidgetPanel):
    annotate_dashboard = AnnotateDash         # type: AnnotateDash
    image_canvas = ImageCanvas      # type: ImageCanvas

    def __init__(self, parent):
        # set the master frame
        AbstractWidgetPanel.__init__(self, parent)
        self.app_variables = AppVariables()
        widgets_list = ["image_canvas", "annotate_dashboard"]

        self.init_w_vertical_layout(widgets_list)

        self.annotate_dashboard.set_spacing_between_buttons(0)
        self.image_canvas.variables.canvas_image_object = SarpyCanvasDisplayImage()
        self.image_canvas.set_canvas_size(600, 400)

        # set up event listeners
        self.annotate_dashboard.controls.pan.on_left_mouse_click(self.callback_set_to_pan)
        self.image_canvas.canvas.on_left_mouse_release(self.callback_handle_left_mouse_release)
        self.image_canvas.set_labelframe_text("Image View")

    def callback_set_to_pan(self, event):
        self.annotate_dashboard.controls.set_active_button(self.annotate_dashboard.controls.pan)
        self.image_canvas.set_current_tool_to_pan()

    def callback_handle_left_mouse_release(self, event):
        self.image_canvas.callback_handle_left_mouse_release(event)
        self.update_decimation_value()

    def update_decimation_value(self):
        decimation_value = self.image_canvas.variables.canvas_image_object.decimation_factor
        self.annotate_dashboard.info_panel.annotate_decimation_val.set_text(str(decimation_value))

from sarpy_gui_apps.apps.annotation_tool.panels.context_image_panel.master_dashboard.context_dashboard import ContextMasterDash
from sarpy_gui_apps.supporting_classes.sarpy_canvas_image import SarpyCanvasDisplayImage
from tkinter_gui_builder.panel_templates.image_canvas.image_canvas import ImageCanvas
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel


class AppVariables:
    def __init__(self):
        self.image_fname = "None"       # type: str
        self.sicd_metadata = None


class ContextImagePanel(AbstractWidgetPanel):
    context_dashboard = ContextMasterDash         # type: ContextMasterDash
    image_canvas = ImageCanvas      # type: ImageCanvas

    def __init__(self, parent):
        # set the master frame
        AbstractWidgetPanel.__init__(self, parent)
        self.app_variables = AppVariables()
        widgets_list = ["image_canvas", "context_dashboard"]

        self.init_w_vertical_layout(widgets_list)

        self.context_dashboard.set_spacing_between_buttons(0)
        self.image_canvas.variables.canvas_image_object = SarpyCanvasDisplayImage()
        self.image_canvas.set_canvas_size(600, 400)

        # set up event listeners
        self.context_dashboard.buttons.zoom_in.on_left_mouse_click(self.callback_set_to_zoom_in)
        self.context_dashboard.buttons.zoom_out.on_left_mouse_click(self.callback_set_to_zoom_out)
        self.context_dashboard.buttons.pan.on_left_mouse_click(self.callback_set_to_pan)
        self.context_dashboard.buttons.select.on_left_mouse_click(self.callback_set_to_select)

        self.image_canvas.canvas.on_left_mouse_click(self.callback_handle_left_mouse_click)
        self.image_canvas.canvas.on_left_mouse_release(self.callback_handle_left_mouse_release)

        self.image_canvas.canvas.on_mouse_wheel(self.callback_handle_mouse_wheel)

        self.context_dashboard.file_selector.set_fname_filters(["*.NITF", ".nitf"])
        self.image_canvas.set_labelframe_text("Image View")

    def callback_handle_mouse_wheel(self, event):
        self.image_canvas.callback_mouse_zoom(event)
        self.update_decimation_value()

    def callback_set_to_select(self, event):
        self.context_dashboard.buttons.set_active_button(self.context_dashboard.buttons.select)
        self.image_canvas.set_current_tool_to_selection_tool()

    def callback_set_to_pan(self, event):
        self.context_dashboard.buttons.set_active_button(self.context_dashboard.buttons.pan)
        self.image_canvas.set_current_tool_to_pan()

    def callback_set_to_zoom_in(self, event):
        self.context_dashboard.buttons.set_active_button(self.context_dashboard.buttons.zoom_in)
        self.image_canvas.set_current_tool_to_zoom_in()

    def callback_set_to_zoom_out(self, event):
        self.context_dashboard.buttons.set_active_button(self.context_dashboard.buttons.zoom_out)
        self.image_canvas.set_current_tool_to_zoom_out()

    def callback_handle_left_mouse_click(self, event):
        # first do all the normal mouse click functionality of the canvas
        self.image_canvas.callback_handle_left_mouse_click(event)
        # now set the object ID's accordingly, we do this so we don't draw multiple arrows or points

    def callback_handle_left_mouse_release(self, event):
        self.image_canvas.callback_handle_left_mouse_release(event)
        self.update_decimation_value()

    def update_decimation_value(self):
        decimation_value = self.image_canvas.variables.canvas_image_object.decimation_factor
        self.context_dashboard.info_panel.decimation_val.set_text(str(decimation_value))

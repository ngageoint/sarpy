from sarpy_gui_apps.apps.annotation_tool.panels.annotate_panel.annotate_dashboard.annotate_dashboard import AnnotateDash
from sarpy_gui_apps.supporting_classes.sarpy_canvas_image import SarpyCanvasDisplayImage
from tkinter_gui_builder.panel_templates.image_canvas.image_canvas import ImageCanvas
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel


class AppVariables:
    def __init__(self):
        self.image_fname = "None"       # type: str
        self.sicd_metadata = None


class AnnotateImagePanel(AbstractWidgetPanel):
    side_panel = AnnotateDash         # type: AnnotateDash
    image_canvas = ImageCanvas      # type: ImageCanvas

    def __init__(self, parent):
        # set the master frame
        AbstractWidgetPanel.__init__(self, parent)
        self.app_variables = AppVariables()
        widgets_list = ["image_canvas", "side_panel"]

        self.init_w_vertical_layout(widgets_list)

        self.side_panel.set_spacing_between_buttons(0)
        self.image_canvas.variables.canvas_image_object = SarpyCanvasDisplayImage()
        self.image_canvas.set_canvas_size(600, 400)

        # set up event listeners
        self.side_panel.buttons.zoom_in.on_left_mouse_click(self.callback_set_to_zoom_in)
        self.side_panel.buttons.zoom_out.on_left_mouse_click(self.callback_set_to_zoom_out)
        self.side_panel.buttons.pan.on_left_mouse_click(self.callback_set_to_pan)
        self.side_panel.buttons.select.on_left_mouse_click(self.callback_set_to_select)

        self.image_canvas.canvas.on_left_mouse_click(self.callback_handle_left_mouse_click)
        self.image_canvas.canvas.on_left_mouse_release(self.callback_handle_left_mouse_release)

        self.image_canvas.canvas.on_mouse_wheel(self.callback_handle_mouse_wheel)

        self.side_panel.file_selector.set_fname_filters(["*.NITF", ".nitf"])
        self.side_panel.file_selector.select_file.on_left_mouse_click(self.callback_select_file)

        self.image_canvas.set_labelframe_text("Image View")

    def callback_handle_mouse_wheel(self, event):
        self.image_canvas.callback_mouse_zoom(event)
        self.update_decimation_value()

    def callback_set_to_select(self, event):
        self.side_panel.buttons.set_active_button(self.side_panel.file_selector.select_file)
        self.image_canvas.set_current_tool_to_selection_tool()

    def callback_set_to_pan(self, event):
        self.side_panel.buttons.set_active_button(self.side_panel.buttons.pan)
        self.image_canvas.set_current_tool_to_pan()

    def callback_select_file(self, event):
        self.side_panel.file_selector.event_select_file(event)
        if self.side_panel.file_selector.fname:
            self.app_variables.image_fname = self.side_panel.file_selector.fname
        self.image_canvas.init_with_fname(self.app_variables.image_fname)
        self.update_decimation_value()

    def callback_set_to_zoom_in(self, event):
        self.side_panel.buttons.set_active_button(self.side_panel.buttons.zoom_in)
        self.image_canvas.set_current_tool_to_zoom_in()

    def callback_set_to_zoom_out(self, event):
        self.side_panel.buttons.set_active_button(self.side_panel.buttons.zoom_out)
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
        self.side_panel.info_panel.decimation_val.set_text(str(decimation_value))

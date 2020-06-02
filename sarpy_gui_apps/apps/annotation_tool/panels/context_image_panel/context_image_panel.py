from sarpy_gui_apps.apps.annotation_tool.panels.context_image_panel.master_dashboard.context_dashboard import ContextMasterDash
from tkinter_gui_builder.panel_templates.image_canvas_panel.image_canvas import ImageCanvas
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
        widgets_list = ["image_canvas_panel", "context_dashboard"]

        self.init_w_vertical_layout(widgets_list)

        self.context_dashboard.set_spacing_between_buttons(0)
        self.image_canvas.set_canvas_size(600, 400)

        self.context_dashboard.file_selector.set_fname_filters(["*.NITF", ".nitf"])
        self.image_canvas.set_labelframe_text("Image View")

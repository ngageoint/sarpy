from sarpy_gui_apps.supporting_classes.sarpy_canvas_image import SarpyCanvasDisplayImage
from tkinter_gui_builder.panel_templates.image_canvas.image_canvas import ImageCanvas
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from sarpy_gui_apps.apps.aperture_tool.app_variables import AppVariables
from sarpy_gui_apps.apps.aperture_tool.panels.selected_region_popup.toolbar import Toolbar


class SelectedRegionPanel(AbstractWidgetPanel):
    image_canvas = ImageCanvas      # type: ImageCanvas
    toolbar = Toolbar                   # type: Toolbar

    def __init__(self,
                 parent,
                 app_variables,         # type: AppVariables
                 ):
        # set the master frame
        AbstractWidgetPanel.__init__(self, parent)
        widgets_list = ["toolbar", "image_canvas"]

        self.init_w_vertical_layout(widgets_list)

        self.image_canvas.variables.canvas_image_object = SarpyCanvasDisplayImage()
        self.image_canvas.init_with_fname(app_variables.sicd_fname)
        self.image_canvas.set_canvas_size(600, 400)

        self.pack()

        self.toolbar.zoom_in.on_left_mouse_click(self.image_canvas.set_current_tool_to_zoom_in)
        self.toolbar.zoom_out.on_left_mouse_click(self.image_canvas.set_current_tool_to_zoom_out)
        self.toolbar.pan.on_left_mouse_click(self.image_canvas.set_current_tool_to_pan)
        self.toolbar.select_aoi.on_left_mouse_click(self.image_canvas.set_current_tool_to_selection_tool)
        self.toolbar.submit_aoi.on_left_mouse_click(self.submit_aoi)

    def submit_aoi(self):
        print("submitting aoi")
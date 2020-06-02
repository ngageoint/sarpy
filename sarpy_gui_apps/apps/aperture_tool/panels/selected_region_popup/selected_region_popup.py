from sarpy_gui_apps.supporting_classes.complex_image_reader import ComplexImageReader
from tkinter_gui_builder.panel_templates.image_canvas_panel.image_canvas_panel import ImageCanvasPanel
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from sarpy_gui_apps.apps.aperture_tool.app_variables import AppVariables
from sarpy_gui_apps.apps.aperture_tool.panels.selected_region_popup.toolbar import Toolbar


class SelectedRegionPanel(AbstractWidgetPanel):
    image_canvas = ImageCanvasPanel      # type: ImageCanvasPanel
    toolbar = Toolbar                   # type: Toolbar

    def __init__(self,
                 parent,
                 app_variables,         # type: AppVariables
                 ):
        # set the master frame
        AbstractWidgetPanel.__init__(self, parent)
        widgets_list = ["toolbar", "image_canvas"]

        self.parent = parent
        self.app_variables = app_variables

        self.init_w_vertical_layout(widgets_list)

        sicd_reader = ComplexImageReader(app_variables.sicd_fname)
        # self.image_canvas_panel = ImageCanvas()
        self.image_canvas.set_canvas_size(1000, 1000)
        self.image_canvas.canvas.set_image_reader(sicd_reader)

        self.pack()

        self.toolbar.zoom_in.on_left_mouse_click(self.set_current_tool_to_zoom_in)
        self.toolbar.zoom_out.on_left_mouse_click(self.set_current_tool_to_zoom_out)
        self.toolbar.pan.on_left_mouse_click(self.set_current_tool_to_pan)
        self.toolbar.select_aoi.on_left_mouse_click(self.set_current_tool_to_selection_tool)
        self.toolbar.submit_aoi.on_left_mouse_click(self.submit_aoi)

    def set_current_tool_to_zoom_in(self, event):
        self.image_canvas.canvas.set_current_tool_to_zoom_in()

    def set_current_tool_to_zoom_out(self, event):
        self.image_canvas.canvas.set_current_tool_to_zoom_out()

    def set_current_tool_to_pan(self, event):
        self.image_canvas.canvas.set_current_tool_to_pan()

    def set_current_tool_to_selection_tool(self, event):
        self.image_canvas.canvas.set_current_tool_to_selection_tool()

    def submit_aoi(self, event):
        selection_image_coords = self.image_canvas.canvas.get_shape_image_coords(self.image_canvas.canvas.variables.select_rect_id)
        if selection_image_coords:
            self.app_variables.selected_region = selection_image_coords
            y1 = selection_image_coords[0]
            x1 = selection_image_coords[1]
            y2 = selection_image_coords[2]
            x2 = selection_image_coords[3]
            complex_data = self.app_variables.sicd_reader_object.base_reader.read_chip((y1, y2, 1), (x1, x2, 1))
            self.app_variables.selected_region_complex_data = complex_data
            self.parent.destroy()
        else:
            print("need to select region first")
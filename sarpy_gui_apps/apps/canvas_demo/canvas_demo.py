import tkinter
from tkinter.filedialog import askopenfilename
from sarpy_gui_apps.apps.canvas_demo.panels.canvas_demo_button_panel import CanvasDemoButtonPanel
from tkinter_gui_builder.panel_templates.pyplot_image_panel.pyplot_image_panel import PyplotImagePanel
from tkinter_gui_builder.panel_templates.image_canvas.image_canvas import ImageCanvas
from tkinter_gui_builder.utils.geometry_utils.kml_util import KmlUtil
from sarpy_gui_apps.supporting_classes.sarpy_canvas_image import SarpyCanvasDisplayImage
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
import sarpy.geometry.point_projection as point_projection
import sarpy.geometry.geocoords as geocoords
import os
import numpy as np


class AppVariables:
    def __init__(self):
        self.fname = "None"       # type: str
        self.remap_type = "density"
        self.selection_rect_id = None           # type: int
        self.shapes_in_selector = []


class CanvasDemo(AbstractWidgetPanel):
    button_panel = CanvasDemoButtonPanel         # type: CanvasDemoButtonPanel
    pyplot_panel = PyplotImagePanel         # type: PyplotImagePanel
    canvas_demo_image_panel = ImageCanvas         # type: ImageCanvas

    def __init__(self, master):
        master_frame = tkinter.Frame(master)
        AbstractWidgetPanel.__init__(self, master_frame)
        self.variables = AppVariables()

        widget_list = ["canvas_demo_image_panel", "pyplot_panel", "button_panel", ]
        self.init_w_basic_widget_list(widget_list, 2, [2, 1])

        # define panels widget_wrappers in master frame
        self.button_panel.set_spacing_between_buttons(0)
        self.canvas_demo_image_panel.variables.canvas_image_object = SarpyCanvasDisplayImage()
        self.canvas_demo_image_panel.set_canvas_size(700, 400)
        self.canvas_demo_image_panel.rescale_image_to_fit_canvas = True

        # need to pack both master frame and self, since this is the main app window.
        master_frame.pack()
        self.pack()

        # bind events to callbacks here
        self.button_panel.fname_select.on_left_mouse_click(self.callback_initialize_canvas_image)
        self.button_panel.update_rect_image.on_left_mouse_click(self.callback_display_canvas_rect_selection_in_pyplot_frame)
        self.button_panel.remap_dropdown.on_selection(self.callback_remap)
        self.button_panel.zoom_in.on_left_mouse_click(self.callback_set_to_zoom_in)
        self.button_panel.zoom_out.on_left_mouse_click(self.callback_set_to_zoom_out)
        self.button_panel.pan.on_left_mouse_click(self.callback_set_to_pan)
        self.button_panel.rect_select.on_left_mouse_click(self.callback_set_to_select)

        self.button_panel.draw_line_w_drag.on_left_mouse_click(self.callback_draw_line_w_drag)
        self.button_panel.draw_line_w_click.on_left_mouse_click(self.callback_draw_line_w_click)
        self.button_panel.draw_arrow_w_drag.on_left_mouse_click(self.callback_draw_arrow_w_drag)
        self.button_panel.draw_arrow_w_click.on_left_mouse_click(self.callback_draw_arrow_w_click)

        self.button_panel.draw_rect_w_drag.on_left_mouse_click(self.callback_draw_rect_w_drag)
        self.button_panel.draw_rect_w_click.on_left_mouse_click(self.callback_draw_rect_w_click)

        self.button_panel.draw_polygon_w_click.on_left_mouse_click(self.callback_draw_polygon_w_click)
        self.button_panel.draw_point_w_click.on_left_mouse_click(self.callback_draw_point_w_click)

        self.button_panel.color_selector.on_left_mouse_click(self.callback_activate_color_selector)

        self.button_panel.modify_existing_shape.on_left_mouse_click(self.callback_edit_shape)
        self.button_panel.select_existing_shape.on_selection(self.callback_handle_shape_selector)
        self.button_panel.save_kml.on_left_mouse_click(self.callback_save_kml)

        self.canvas_demo_image_panel.canvas.on_left_mouse_click(self.callback_handle_canvas_left_mouse_click)

    def callback_save_kml(self, event):
        kml_save_fname = tkinter.filedialog.asksaveasfilename(initialdir = os.path.expanduser("~/Downloads"))

        kml_util = KmlUtil()

        canvas_shapes = self.canvas_demo_image_panel.variables.shape_ids
        for shape_id in canvas_shapes:
            image_coords = self.canvas_demo_image_panel.get_shape_image_coords(shape_id)
            shape_type = self.canvas_demo_image_panel.get_shape_type(shape_id)
            if image_coords:
                sicd_meta = self.canvas_demo_image_panel.variables.canvas_image_object.reader_object.sicdmeta
                image_points = np.zeros((int(len(image_coords)/2), 2))
                image_points[:, 0] = image_coords[0::2]
                image_points[:, 1] = image_coords[1::2]

                ground_points_ecf = point_projection.image_to_ground(image_points, sicd_meta)
                ground_points_latlon = geocoords.ecf_to_geodetic(ground_points_ecf)

                world_y_coordinates = ground_points_latlon[:, 0]
                world_x_coordinates = ground_points_latlon[:, 1]

                xy_point_list = [(x, y) for x, y in zip(world_x_coordinates, world_y_coordinates)]

                if shape_id == self.canvas_demo_image_panel.variables.zoom_rect_id:
                    pass
                elif shape_type == self.canvas_demo_image_panel.variables.select_rect_id:
                    pass
                elif shape_type == self.canvas_demo_image_panel.SHAPE_TYPES.POINT:
                    kml_util.add_point(str(shape_id), xy_point_list[0])
                elif canvas_shapes == self.canvas_demo_image_panel.SHAPE_TYPES.LINE:
                    kml_util.add_linestring(str(shape_id), xy_point_list)
                elif shape_type == self.canvas_demo_image_panel.SHAPE_TYPES.POLYGON:
                    kml_util.add_polygon(str(shape_id), xy_point_list)
                elif shape_type == self.canvas_demo_image_panel.SHAPE_TYPES.RECT:
                    kml_util.add_polygon(str(shape_id), xy_point_list)
        kml_util.write_to_file(kml_save_fname)

    def callback_handle_canvas_left_mouse_click(self, event):
        self.canvas_demo_image_panel.callback_handle_left_mouse_click(event)
        current_shape = self.canvas_demo_image_panel.variables.current_shape_id
        if current_shape:
            self.variables.shapes_in_selector.append(current_shape)
            self.variables.shapes_in_selector = sorted(list(set(self.variables.shapes_in_selector)))
            self.button_panel.select_existing_shape.update_combobox_values(self.variables.shapes_in_selector)

    def callback_handle_shape_selector(self, event):
        current_shape_id = int(self.button_panel.select_existing_shape.get())
        self.canvas_demo_image_panel.variables.current_shape_id = current_shape_id
        self.canvas_demo_image_panel.highlight_existing_shape(current_shape_id)

    def callback_highlight_shape(self, event):
        self.canvas_demo_image_panel.highlight_existing_shape(self.canvas_demo_image_panel.variables.current_shape_id)
        
    def callback_edit_shape(self, event):
        self.canvas_demo_image_panel.set_current_tool_to_edite_shape()

    def callback_activate_color_selector(self, event):
        self.canvas_demo_image_panel.activate_color_selector(event)

    def callback_draw_line_w_drag(self, event):
        self.canvas_demo_image_panel.set_current_tool_to_draw_line_by_dragging()

    def callback_draw_line_w_click(self, event):
        self.canvas_demo_image_panel.set_current_tool_to_draw_line_by_clicking()

    def callback_draw_arrow_w_drag(self, event):
        self.canvas_demo_image_panel.set_current_tool_to_draw_arrow_by_dragging()

    def callback_draw_arrow_w_click(self, event):
        self.canvas_demo_image_panel.set_current_tool_to_draw_arrow_by_clicking()

    def callback_draw_rect_w_drag(self, event):
        self.canvas_demo_image_panel.set_current_tool_to_draw_rect()

    def callback_draw_rect_w_click(self, event):
        self.canvas_demo_image_panel.set_current_tool_to_draw_rect_by_clicking()

    def callback_draw_polygon_w_click(self, event):
        self.canvas_demo_image_panel.set_current_tool_to_draw_polygon_by_clicking()

    def callback_draw_point_w_click(self, event):
        self.canvas_demo_image_panel.set_current_tool_to_draw_point()

    def callback_set_to_zoom_in(self, event):
        self.canvas_demo_image_panel.set_current_tool_to_zoom_in()

    def callback_set_to_zoom_out(self, event):
        self.canvas_demo_image_panel.set_current_tool_to_zoom_out()

    def callback_set_to_pan(self, event):
        self.canvas_demo_image_panel.set_current_tool_to_pan()
        self.canvas_demo_image_panel.hide_shape(self.canvas_demo_image_panel.variables.zoom_rect_id)

    def callback_set_to_select(self, event):
        self.canvas_demo_image_panel.set_current_tool_to_draw_rect(self.canvas_demo_image_panel.variables.select_rect_id)
        self.variables.selection_rect_id = self.canvas_demo_image_panel.variables.current_shape_id

    # define custom callbacks here
    def callback_remap(self, event):
        remap_dict = {"density": "density",
                      "brighter": "brighter",
                      "darker": "darker",
                      "high contrast": "highcontrast",
                      "linear": "linear",
                      "log": "log",
                      "pedf": "pedf",
                      "nrl": "nrl"}
        selection = self.button_panel.remap_dropdown.get()
        remap_type = remap_dict[selection]
        self.canvas_demo_image_panel.variables.canvas_image_object.remap_type = remap_type

    def callback_initialize_canvas_image(self, event):
        image_file_extensions = ['*.nitf', '*.NITF']
        ftypes = [
            ('image files', image_file_extensions),
            ('All files', '*'),
        ]
        new_fname = askopenfilename(initialdir=os.path.expanduser("~"), filetypes=ftypes)
        if new_fname:
            self.variables.fname = new_fname
            self.canvas_demo_image_panel.init_with_fname(self.variables.fname)

    def callback_display_canvas_rect_selection_in_pyplot_frame(self, event):
        complex_data = self.canvas_demo_image_panel.get_image_data_in_canvas_rect_by_id(self.variables.selection_rect_id)
        remapped_data = self.canvas_demo_image_panel.variables.canvas_image_object.remap_complex_data(complex_data)
        self.pyplot_panel.update_image(remapped_data)


if __name__ == '__main__':
    root = tkinter.Tk()
    app = CanvasDemo(root)
    root.mainloop()

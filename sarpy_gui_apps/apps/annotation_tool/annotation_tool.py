from sarpy_gui_apps.apps.annotation_tool.panels.context_image_panel.context_image_panel import ContextImagePanel
from sarpy_gui_apps.apps.annotation_tool.panels.annotate_image_panel.annotate_image_panel import AnnotateImagePanel
from sarpy_gui_apps.apps.annotation_tool.panels.annotation_popup.annotation_popup import AnnotationPopup
from sarpy_gui_apps.apps.annotation_tool.panels.annotation_fname_popup.annotation_fname_popup import AnnotationFnamePopup
from sarpy_gui_apps.apps.annotation_tool.main_app_variables import AppVariables

from tkinter_gui_builder import tkinter
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.panel_templates.image_canvas.tool_constants import ToolConstants
from sarpy.geometry.geometry_elements import Polygon
from sarpy.annotation.annotate import FileAnnotationCollection
from sarpy.annotation.annotate import Annotation
from sarpy.annotation.annotate import LabelSchema

import numpy as np
import os


class AnnotationTool(AbstractWidgetPanel):
    context_panel = ContextImagePanel
    annotate_panel = AnnotateImagePanel

    def __init__(self, master):
        master_frame = tkinter.Frame(master)
        AbstractWidgetPanel.__init__(self, master_frame)

        widgets_list = ["context_panel", "annotate_panel"]
        self.init_w_horizontal_layout(widgets_list)
        master_frame.pack()
        self.pack()

        self.variables = AppVariables()

        # set up context panel event listeners
        self.context_panel.context_dashboard.buttons.zoom_in.on_left_mouse_click(self.callback_context_set_to_zoom_in)
        self.context_panel.context_dashboard.buttons.zoom_out.on_left_mouse_click(self.callback_context_set_to_zoom_out)
        self.context_panel.context_dashboard.buttons.pan.on_left_mouse_click(self.callback_context_set_to_pan)
        self.context_panel.context_dashboard.buttons.select.on_left_mouse_click(self.callback_context_set_to_select)
        self.context_panel.context_dashboard.buttons.move_rect.on_left_mouse_click(self.callback_context_set_to_move_rect)
        self.context_panel.context_dashboard.file_selector.select_file.on_left_mouse_click(self.callback_context_select_file)
        self.context_panel.context_dashboard.annotation_selector.select_file.on_left_mouse_click(self.callback_content_select_annotation_file)

        self.context_panel.image_canvas.canvas.on_left_mouse_release(self.callback_context_handle_left_mouse_release)
        self.context_panel.image_canvas.canvas.on_mouse_wheel(self.callback_context_handle_mouse_wheel)

        # set up annotate panel event listeners
        self.annotate_panel.image_canvas.canvas.on_mouse_wheel(self.callback_handle_annotate_mouse_wheel)
        self.annotate_panel.image_canvas.canvas.on_left_mouse_click(self.callback_annotate_handle_canvas_left_mouse_click)
        self.annotate_panel.image_canvas.canvas.on_left_mouse_release(self.callback_annotate_handle_left_mouse_release)
        self.annotate_panel.image_canvas.canvas.on_right_mouse_click(self.callback_annotate_handle_right_mouse_click)

        self.annotate_panel.annotate_dashboard.controls.draw_polygon.on_left_mouse_click(self.callback_set_to_draw_polygon)
        self.annotate_panel.annotate_dashboard.controls.popup.on_left_mouse_click(self.callback_annotation_popup)
        self.annotate_panel.annotate_dashboard.controls.pan.on_left_mouse_click(self.callback_annotate_set_to_pan)
        self.annotate_panel.annotate_dashboard.controls.save_annotations.on_left_mouse_click(self.callback_save_annotations)
        self.annotate_panel.annotate_dashboard.controls.select_closest_shape.on_left_mouse_click(self.callback_set_to_select_closest_shape)
        self.annotate_panel.annotate_dashboard.controls.delete_shape.on_left_mouse_click(self.callback_delete_shape)

        self.annotate_panel.image_canvas.set_labelframe_text("Image View")

    # context callbacks
    def callback_context_select_file(self, event):
        self.context_panel.context_dashboard.file_selector.set_fname_filters([('nitf files', '*')])
        self.context_panel.context_dashboard.file_selector.event_select_file(event)
        if self.context_panel.context_dashboard.file_selector.fname:
            self.variables.image_fname = self.context_panel.context_dashboard.file_selector.fname
        self.context_panel.image_canvas.init_with_fname(self.variables.image_fname)
        self.update_context_decimation_value()
        self.annotate_panel.image_canvas.init_with_fname(self.variables.image_fname)
        self.annotate_panel.image_canvas.set_image_from_numpy_array(np.zeros((self.annotate_panel.image_canvas.canvas_height, self.annotate_panel.image_canvas.canvas_width)))
        self.variables.file_annotation_collection = FileAnnotationCollection(self.variables.label_schema, image_file_name=self.variables.image_fname)
        self.variables.annotate_canvas = self.annotate_panel.image_canvas

    def callback_content_select_annotation_file(self, event):
        popup = tkinter.Toplevel(self.master)
        AnnotationFnamePopup(popup, self.variables)
        master_ul_x = self.master.winfo_rootx()
        master_ul_y = self.master.winfo_rooty()
        master_height = self.master.winfo_height()
        master_width = self.master.winfo_width()
        popup_height = popup.winfo_height()
        popup_width = popup.winfo_width()

        # TODO: The popup window thinks it has a height and width of 1 pixel
        popup_x_ul = int(master_ul_x + master_width/2 - popup_width)
        popup_y_ul = int(master_ul_y + master_height/2 - popup_height)

        popup.geometry("+" + str(popup_x_ul) + "+" + str(popup_y_ul))

        self.master.wait_window(popup)

        self.context_panel.context_dashboard.annotation_selector.set_fname_filters([('json files', '*.json')])
        if self.variables.new_annotation:
            # select label schema template file before creating a new annotation file
            self.context_panel.context_dashboard.annotation_selector.event_select_file(event)
            self.context_panel.context_dashboard.annotation_selector.set_label_text("select schema template")
            schema_fname = self.context_panel.context_dashboard.annotation_selector.fname
            self.variables.label_schema.from_file(schema_fname)
            self.context_panel.context_dashboard.annotation_selector.event_new_file(event)
            self.variables.file_annotation_fname = self.context_panel.context_dashboard.annotation_selector.fname
            self.variables.file_annotation_collection.label_schema = self.variables.label_schema
            self.variables.file_annotation_collection.image_file_name = self.variables.image_fname
        else:
            self.context_panel.context_dashboard.annotation_selector.event_select_file(event)
            self.variables.file_annotation_fname = self.context_panel.context_dashboard.annotation_selector.fname

        fname = self.context_panel.context_dashboard.annotation_selector.fname
        if fname != '':
            stop = 1
        stop = 1

    def callback_context_set_to_select(self, event):
        self.context_panel.context_dashboard.buttons.set_active_button(self.context_panel.context_dashboard.buttons.select)
        self.context_panel.image_canvas.set_current_tool_to_selection_tool()

    def callback_context_set_to_pan(self, event):
        self.context_panel.context_dashboard.buttons.set_active_button(self.context_panel.context_dashboard.buttons.pan)
        self.context_panel.image_canvas.set_current_tool_to_pan()

    def callback_context_set_to_zoom_in(self, event):
        self.context_panel.context_dashboard.buttons.set_active_button(self.context_panel.context_dashboard.buttons.zoom_in)
        self.context_panel.image_canvas.set_current_tool_to_zoom_in()

    def callback_context_set_to_zoom_out(self, event):
        self.context_panel.context_dashboard.buttons.set_active_button(self.context_panel.context_dashboard.buttons.zoom_out)
        self.context_panel.image_canvas.set_current_tool_to_zoom_out()

    def callback_context_set_to_move_rect(self, event):
        self.context_panel.context_dashboard.buttons.set_active_button(self.context_panel.context_dashboard.buttons.move_rect)
        self.context_panel.image_canvas.set_current_tool_to_translate_shape()

    def callback_context_handle_mouse_wheel(self, event):
        self.context_panel.image_canvas.callback_mouse_zoom(event)
        self.update_context_decimation_value()

    def callback_context_handle_left_mouse_release(self, event):
        self.context_panel.image_canvas.callback_handle_left_mouse_release(event)
        if self.context_panel.image_canvas.variables.current_tool == ToolConstants.SELECT_TOOL or \
           self.context_panel.image_canvas.variables.current_tool == ToolConstants.TRANSLATE_SHAPE_TOOL:
            rect_id = self.context_panel.image_canvas.variables.select_rect_id
            image_rect = self.context_panel.image_canvas.canvas_shape_coords_to_image_coords(rect_id)
            annotate_zoom_rect = self.annotate_panel.image_canvas.variables.canvas_image_object.full_image_yx_to_canvas_coords(
                image_rect)
            self.annotate_panel.image_canvas.zoom_to_selection(annotate_zoom_rect, animate=True)
        self.draw_context_rect()
        self.update_annotate_decimation_value()
        self.update_context_decimation_value()

    # annotate callbacks
    def callback_set_to_select_closest_shape(self, event):
        self.annotate_panel.image_canvas.set_current_tool_to_select_closest_shape()

    def callback_annotate_set_to_pan(self, event):
        self.annotate_panel.annotate_dashboard.controls.set_active_button(self.annotate_panel.annotate_dashboard.controls.pan)
        self.annotate_panel.image_canvas.set_current_tool_to_pan()

    def callback_annotate_handle_left_mouse_release(self, event):
        self.annotate_panel.image_canvas.callback_handle_left_mouse_release(event)
        self.update_annotate_decimation_value()
        self.draw_context_rect()
        self.update_context_decimation_value()
        self.update_annotate_decimation_value()

    def callback_annotate_handle_canvas_left_mouse_click(self, event):
        self.annotate_panel.image_canvas.callback_handle_left_mouse_click(event)
        current_shape = self.annotate_panel.image_canvas.variables.current_shape_id
        if current_shape:
            self.variables.shapes_in_selector.append(current_shape)
            self.variables.shapes_in_selector = sorted(list(set(self.variables.shapes_in_selector)))

    def callback_annotate_handle_right_mouse_click(self, event):
        self.annotate_panel.image_canvas.callback_handle_right_mouse_click(event)
        if self.annotate_panel.image_canvas.variables.current_tool == ToolConstants.DRAW_POLYGON_BY_CLICKING:
            current_canvas_shape_id = self.annotate_panel.image_canvas.variables.current_shape_id
            image_coords = self.annotate_panel.image_canvas.get_shape_image_coords(current_canvas_shape_id)
            geometry_coords = np.asarray([x for x in zip(image_coords[0::2], image_coords[1::2])])
            polygon = Polygon(coordinates=[geometry_coords])

            annotation = Annotation()
            annotation.geometry = polygon

            self.variables.canvas_geom_ids_to_annotations_id_dict[str(current_canvas_shape_id)] = annotation

    def callback_set_to_draw_polygon(self, event):
        self.annotate_panel.annotate_dashboard.controls.set_active_button(self.annotate_panel.annotate_dashboard.controls.draw_polygon)
        self.annotate_panel.image_canvas.set_current_tool_to_draw_polygon_by_clicking()

    def callback_handle_annotate_mouse_wheel(self, event):
        self.annotate_panel.image_canvas.callback_mouse_zoom(event)
        self.draw_context_rect()
        self.update_annotate_decimation_value()

    def callback_delete_shape(self, event):
        tool_shape_ids = self.annotate_panel.image_canvas.get_tool_shape_ids()
        current_geom_id = self.annotate_panel.image_canvas.variables.current_shape_id
        if current_geom_id:
            if current_geom_id in tool_shape_ids:
                print("a tool is currently selected.  First select a shape.")
                pass
            else:
                self.annotate_panel.image_canvas.delete_shape(current_geom_id)
                del self.variables.canvas_geom_ids_to_annotations_id_dict[str(current_geom_id)]
        else:
            print("no shape selected")

    def callback_annotation_popup(self, event):
        popup = tkinter.Toplevel(self.master)
        current_canvas_shape_id = self.annotate_panel.image_canvas.variables.current_shape_id
        self.variables.current_canvas_geom_id = current_canvas_shape_id
        AnnotationPopup(popup, self.variables)

    def callback_save_annotations(self, event):
        for key, val in self.variables.canvas_geom_ids_to_annotations_id_dict.items():
            self.variables.file_annotation_collection.add_annotation(val)
        self.variables.file_annotation_collection.to_file(os.path.expanduser("~/Downloads/tmp_annotation.json"))

    # non callback defs
    def update_context_decimation_value(self):
        decimation_value = self.context_panel.image_canvas.variables.canvas_image_object.decimation_factor
        self.context_panel.context_dashboard.info_panel.decimation_val.set_text(str(decimation_value))

    def update_annotate_decimation_value(self):
        decimation_value = self.annotate_panel.image_canvas.variables.canvas_image_object.decimation_factor
        self.annotate_panel.annotate_dashboard.info_panel.annotate_decimation_val.set_text(str(decimation_value))

    def draw_context_rect(self):
        annotate_canvas_nx = self.annotate_panel.image_canvas.variables.canvas_image_object.canvas_nx
        annotate_canvas_ny = self.annotate_panel.image_canvas.variables.canvas_image_object.canvas_ny
        annotate_canvas_extents = [0, 0, annotate_canvas_nx, annotate_canvas_ny]
        image_rect = self.annotate_panel.image_canvas.variables.canvas_image_object.canvas_coords_to_full_image_yx(
            annotate_canvas_extents)
        context_rect = self.context_panel.image_canvas.variables.canvas_image_object.full_image_yx_to_canvas_coords(
            image_rect)
        self.context_panel.image_canvas.modify_existing_shape_using_canvas_coords(
            self.context_panel.image_canvas.variables.select_rect_id, context_rect)


def main():
    root = tkinter.Tk()
    app = AnnotationTool(root)
    root.mainloop()


if __name__ == '__main__':
    main()

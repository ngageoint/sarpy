import os
import numpy
from scipy.fftpack import fft2, ifft2, fftshift

import tkinter
from tkinter import filedialog
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.utils.image_utils import frame_sequence_utils
from tkinter_gui_builder.panel_templates.image_canvas.image_canvas import ImageCanvas

import sarpy.io.complex as sarpy_complex
import sarpy.visualization.remap as remap
from sarpy_gui_apps.apps.aperture_tool.panels.tabs_panel.tabs_panel import TabsPanel
from sarpy_gui_apps.apps.aperture_tool.panels.selected_region_popup.selected_region_popup import SelectedRegionPanel
from sarpy_gui_apps.supporting_classes.metaicon import MetaIcon
from sarpy_gui_apps.apps.aperture_tool.app_variables import AppVariables
from sarpy.io.complex.base import BaseReader
import matplotlib.pyplot as plt
from scipy import misc as scipy_misc


class ApertureTool(AbstractWidgetPanel):
    frequency_vs_degree_panel = ImageCanvas         # type: ImageCanvas
    filtered_panel = ImageCanvas                    # type: ImageCanvas
    tabs_panel = TabsPanel                          # type: TabsPanel
    metaicon = MetaIcon                             # type: MetaIcon

    def __init__(self, master):
        self.app_variables = AppVariables()

        master_frame = tkinter.Frame(master)
        AbstractWidgetPanel.__init__(self, master_frame)

        widgets_list = ["frequency_vs_degree_panel", "filtered_panel", "tabs_panel", "metaicon"]
        self.init_w_basic_widget_list(widgets_list, n_rows=2, n_widgets_per_row_list=[2, 2])

        self.frequency_vs_degree_panel.set_canvas_size(600, 400)
        self.filtered_panel.set_canvas_size(600, 400)

        # Configuration of panels and widgets
        # self.metaicon.set_canvas_size(600, 400)

        master_frame.pack()
        self.pack()

        self.tabs_panel.tabs.load_image_tab.file_selector.select_file.on_left_mouse_click(self.callback_select_file)

        self.frequency_vs_degree_panel.canvas.on_left_mouse_motion(self.callback_frequency_vs_degree_left_mouse_motion)
        # self.frequency_vs_degree_panel.canvas.on_mouse_motion(self.callback_frequency_vs_degree_mouse_motion)

    def callback_frequency_vs_degree_left_mouse_motion(self, event):
        self.frequency_vs_degree_panel.callback_handle_left_mouse_motion(event)
        # update the selection rect so that it stays within the FFT bounds
        # TODO: modify to not allow user to start drawing outside of bounds.
        # TODO: don't shrink the selection if the user is moving the selection box
        self.update_filtered_image()

    def callback_select_file(self, event):
        sicd_fname = self.tabs_panel.tabs.load_image_tab.file_selector.event_select_file(event)
        self.app_variables.sicd_fname = sicd_fname
        self.app_variables.sicd_reader_object = sarpy_complex.open(sicd_fname)

        self.metaicon.create_from_sicd(self.app_variables.sicd_reader_object.sicd_meta)

        popup = tkinter.Toplevel(self.master)
        selected_region_popup = SelectedRegionPanel(popup, self.app_variables)
        selected_region_popup.image_canvas.init_with_fname(self.app_variables.sicd_fname)

        self.master.wait_window(popup)

        selected_region_complex_data = self.app_variables.selected_region_complex_data

        fft_complex_data = self.get_fft_complex_data(self.app_variables.sicd_reader_object, selected_region_complex_data)
        self.app_variables.fft_complex_data = fft_complex_data

        self.app_variables.fft_display_data = remap.density(fft_complex_data)
        self.frequency_vs_degree_panel.init_with_numpy_image(self.app_variables.fft_display_data)

        # self.frequency_vs_degree_panel.set_current_tool_to_selection_tool()
        self.frequency_vs_degree_panel.set_current_tool_to_edit_shape()
        self.frequency_vs_degree_panel.variables.current_shape_id = self.frequency_vs_degree_panel.variables.select_rect_id
        self.frequency_vs_degree_panel.modify_existing_shape_using_image_coords(self.frequency_vs_degree_panel.variables.select_rect_id, self.get_fft_image_bounds())
        canvas_drawing_bounds = self.frequency_vs_degree_panel.image_coords_to_canvas_coords(self.frequency_vs_degree_panel.variables.select_rect_id)
        self.frequency_vs_degree_panel.variables.shape_drag_xy_limits[str(self.frequency_vs_degree_panel.variables.select_rect_id)] = canvas_drawing_bounds
        self.app_variables.fft_canvas_bounds = self.frequency_vs_degree_panel.get_shape_canvas_coords(self.frequency_vs_degree_panel.variables.select_rect_id)
        self.frequency_vs_degree_panel.show_shape(self.frequency_vs_degree_panel.variables.select_rect_id)
        self.filtered_panel.init_with_numpy_image(self.get_filtered_image())

        self.tabs_panel.tabs.load_image_tab.chip_size_panel.nx.set_text(numpy.shape(selected_region_complex_data)[1])
        self.tabs_panel.tabs.load_image_tab.chip_size_panel.ny.set_text(numpy.shape(selected_region_complex_data)[0])

    def get_fft_image_bounds(self,
                             ):             # type: (int, int, int, int)
        x_axis_mean = numpy.mean(self.app_variables.fft_display_data, axis=0)
        y_axis_mean = numpy.mean(self.app_variables.fft_display_data, axis=1)

        x_start = numpy.min(numpy.where(x_axis_mean != 0))
        x_end = numpy.max(numpy.where(x_axis_mean != 0))

        y_start = numpy.min(numpy.where(y_axis_mean != 0))
        y_end = numpy.max(numpy.where(y_axis_mean != 0))

        return y_start, x_start, y_end, x_end

    def callback_set_to_selection_tool(self, event):
        self.image_canvas.set_current_tool_to_selection_tool()

    def callback_set_to_translate_shape(self, event):
        self.image_canvas.set_current_tool_to_translate_shape()

    def callback_save_fft_panel_as_png(self, event):
        filename = filedialog.asksaveasfilename(initialdir=os.path.expanduser("~"), title="Select file",
                                                filetypes=(("png file", "*.png"), ("all files", "*.*")))
        self.image_canvas.save_as_png(filename)

    def callback_get_adjusted_image(self, event):
        filtered_image = self.get_filtered_image()
        self.frequency_vs_degree_panel.image_canvas.init_with_numpy_image(filtered_image)

    def update_filtered_image(self):
        filtered_image = self.get_filtered_image()
        self.filtered_panel.init_with_numpy_image(filtered_image)

    def get_filtered_image(self):
        select_rect_id = self.frequency_vs_degree_panel.variables.select_rect_id
        full_image_rect = self.frequency_vs_degree_panel.get_shape_image_coords(select_rect_id)

        y1 = int(full_image_rect[0])
        x1 = int(full_image_rect[1])
        y2 = int(full_image_rect[2])
        x2 = int(full_image_rect[3])

        y_ul = min(y1, y2)
        y_lr = max(y1, y2)
        x_ul = min(x1, x2)
        x_lr = max(x1, x2)

        ft_cdata = self.app_variables.fft_complex_data
        filtered_cdata = numpy.zeros(ft_cdata.shape, ft_cdata.dtype)
        filtered_cdata[y_ul:y_lr, x_ul:x_lr] = ft_cdata[y_ul:y_lr, x_ul:x_lr]
        filtered_cdata = fftshift(filtered_cdata)

        inverse_flag = False
        ro = self.app_variables.sicd_reader_object
        if ro.sicd_meta.Grid.Col.Sgn > 0 and ro.sicd_meta.Grid.Row.Sgn > 0:
            pass
        else:
            inverse_flag = True

        if inverse_flag:
            cdata_clip = fft2(filtered_cdata)
        else:
            cdata_clip = ifft2(filtered_cdata)

        filtered_image = remap.density(cdata_clip)
        return filtered_image

    @staticmethod
    def get_fft_complex_data(ro,  # type: BaseReader
                             cdata,     # type: numpy.ndarray
                             ):
        # TODO: change this to a tuple sequence to get rid of FutureWarning
        if ro.sicd_meta.Grid.Col.Sgn > 0 and ro.sicd_meta.Grid.Row.Sgn > 0:
            # use fft2 to go from image to spatial freq
            ft_cdata = fft2(cdata)
        else:
            # flip using ifft2
            ft_cdata = ifft2(tuple(cdata))

        ft_cdata = fftshift(ft_cdata)
        return ft_cdata

    def callback_animate_horizontal_fft_sweep(self, event):
        select_box_id = self.filtered_panel.image_canvas.variables.current_shape_id
        start_fft_select_box = self.filtered_panel.image_canvas.get_shape_canvas_coords(select_box_id)
        n_steps = int(self.filtered_panel.fft_button_panel.n_steps.get())
        n_pixel_translate = int(self.filtered_panel.fft_button_panel.n_pixels_horizontal.get())
        step_factor = numpy.linspace(0, n_pixel_translate, n_steps)
        for step in step_factor:
            x1 = start_fft_select_box[0] + step
            y1 = start_fft_select_box[1]
            x2 = start_fft_select_box[2] + step
            y2 = start_fft_select_box[3]
            self.filtered_panel.image_canvas.modify_existing_shape_using_canvas_coords(select_box_id, (x1, y1, x2, y2), update_pixel_coords=True)
            self.filtered_panel.image_canvas.update()
            self.callback_get_adjusted_image(event)
        self.filtered_panel.image_canvas.modify_existing_shape_using_canvas_coords(select_box_id, start_fft_select_box, update_pixel_coords=True)
        self.filtered_panel.image_canvas.update()

    def callback_save_animation(self, event):
        filename = filedialog.asksaveasfilename(initialdir=os.path.expanduser("~"), title="Select file",
                                                filetypes=(("animated gif", "*.gif"), ("all files", "*.*")))
        select_box_id = self.filtered_panel.image_canvas.variables.current_shape_id
        start_fft_select_box = self.filtered_panel.image_canvas.get_shape_canvas_coords(select_box_id)
        n_steps = int(self.filtered_panel.fft_button_panel.n_steps.get())
        n_pixel_translate = int(self.filtered_panel.fft_button_panel.n_pixels_horizontal.get())
        step_factor = numpy.linspace(0, n_pixel_translate, n_steps)
        fps = float(self.filtered_panel.fft_button_panel.animation_fps.get())

        frame_sequence = []
        for step in step_factor:
            x1 = start_fft_select_box[0] + step
            y1 = start_fft_select_box[1]
            x2 = start_fft_select_box[2] + step
            y2 = start_fft_select_box[3]
            self.filtered_panel.image_canvas.modify_existing_shape_using_canvas_coords(select_box_id, (x1, y1, x2, y2), update_pixel_coords=True)
            self.filtered_panel.image_canvas.update()
            filtered_image = self.get_filtered_image()
            frame_sequence.append(filtered_image)
        frame_sequence_utils.save_numpy_frame_sequence_to_animated_gif(frame_sequence, filename, fps)
        self.filtered_panel.image_canvas.modify_existing_shape_using_canvas_coords(select_box_id, start_fft_select_box, update_pixel_coords=True)
        self.filtered_panel.image_canvas.update()


if __name__ == '__main__':
    root = tkinter.Tk()
    app = ApertureTool(root)
    root.mainloop()

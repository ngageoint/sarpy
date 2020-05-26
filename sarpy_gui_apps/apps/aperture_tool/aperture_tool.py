import os
import numpy
from scipy.fftpack import fft2, ifft2, fftshift

import tkinter
from tkinter import filedialog
from tkinter import Menu
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.utils.image_utils import frame_sequence_utils
from tkinter_gui_builder.panel_templates.image_canvas.image_canvas import ImageCanvas

import sarpy.io.complex as sarpy_complex
import sarpy.visualization.remap as remap
from sarpy_gui_apps.apps.aperture_tool.panels.tabs_panel.tabs_panel import TabsPanel
from sarpy_gui_apps.apps.aperture_tool.panels.selected_region_popup.selected_region_popup import SelectedRegionPanel
from sarpy_gui_apps.supporting_classes.metaicon import MetaIcon
from sarpy_gui_apps.supporting_classes.complex_image_reader import ComplexImageReader
from sarpy_gui_apps.apps.aperture_tool.panels.phase_history_selecion_panel.phase_history_selection_panel import PhaseHistoryPanel
from sarpy_gui_apps.apps.aperture_tool.app_variables import AppVariables
from sarpy.io.complex.base import BaseReader
import scipy.constants.constants as scipy_constants
from tkinter_gui_builder.image_readers.numpy_image_reader import NumpyImageReader


class ApertureTool(AbstractWidgetPanel):
    frequency_vs_degree_panel = ImageCanvas         # type: ImageCanvas
    filtered_panel = ImageCanvas                    # type: ImageCanvas
    tabs_panel = TabsPanel                          # type: TabsPanel
    metaicon = MetaIcon                             # type: MetaIcon
    phase_history = PhaseHistoryPanel               # type: PhaseHistoryPanel

    def __init__(self, master):
        self.app_variables = AppVariables()

        self.master = master

        master_frame = tkinter.Frame(master)
        AbstractWidgetPanel.__init__(self, master_frame)

        widgets_list = ["frequency_vs_degree_panel", "filtered_panel"]
        self.init_w_horizontal_layout(widgets_list)

        self.frequency_vs_degree_panel.set_canvas_size(600, 400)
        self.filtered_panel.set_canvas_size(600, 400)

        self.frequency_vs_degree_panel.canvas.on_left_mouse_motion(self.callback_frequency_vs_degree_left_mouse_motion)

        self.tabs_popup_panel = tkinter.Toplevel(self.master)
        self.tabs_panel = TabsPanel(self.tabs_popup_panel)
        self.tabs_panel.pack()
        self.tabs_popup_panel.withdraw()

        self.tabs_panel.tabs.load_image_tab.file_selector.select_file.on_left_mouse_click(self.callback_select_file)

        self.ph_popup_panel = tkinter.Toplevel(self.master)
        self.phase_history = PhaseHistoryPanel(self.ph_popup_panel)
        self.phase_history.pack()
        self.ph_popup_panel.withdraw()

        self.metaicon_popup_panel = tkinter.Toplevel(self.master)
        self.metaicon = MetaIcon(self.metaicon_popup_panel)
        self.metaicon.set_canvas_size(800, 600)
        self.metaicon.pack()
        self.metaicon_popup_panel.withdraw()

        menubar = Menu()

        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self.select_file)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.exit)

        # create more pulldown menus
        popups_menu = Menu(menubar, tearoff=0)
        popups_menu.add_command(label="Main Controls", command=self.main_controls_popup)
        popups_menu.add_command(label="Phase History", command=self.ph_popup)
        popups_menu.add_command(label="Metaicon", command=self.metaicon_popup)

        menubar.add_cascade(label="File", menu=filemenu)
        menubar.add_cascade(label="Popups", menu=popups_menu)

        master.config(menu=menubar)

        master_frame.pack()
        self.pack()

    def exit(self):
        self.quit()

    def main_controls_popup(self):
        self.tabs_popup_panel.deiconify()

    def ph_popup(self):
        self.ph_popup_panel.deiconify()

    def metaicon_popup(self):
        self.metaicon_popup_panel.deiconify()

    def callback_frequency_vs_degree_left_mouse_motion(self, event):
        self.frequency_vs_degree_panel.callback_handle_left_mouse_motion(event)
        # update the selection rect so that it stays within the FFT bounds
        # TODO: modify to not allow user to start drawing outside of bounds.
        # TODO: don't shrink the selection if the user is moving the selection box
        self.update_filtered_image()
        self.update_phase_history_selection()

    def select_file(self):
        self.callback_select_file(None)

    def callback_select_file(self, event):
        sicd_fname = self.tabs_panel.tabs.load_image_tab.file_selector.event_select_file(event)
        self.app_variables.sicd_fname = sicd_fname
        self.app_variables.sicd_reader_object = sarpy_complex.open(sicd_fname)

        self.metaicon.create_from_sicd(self.app_variables.sicd_reader_object.sicd_meta)

        popup = tkinter.Toplevel(self.master)
        selected_region_popup = SelectedRegionPanel(popup, self.app_variables)
        self.app_variables.sicd_reader_object = ComplexImageReader(self.app_variables.sicd_fname)
        selected_region_popup.image_canvas.set_image_reader(self.app_variables.sicd_reader_object)

        self.master.wait_window(popup)

        selected_region_complex_data = self.app_variables.selected_region_complex_data

        fft_complex_data = self.get_fft_complex_data(self.app_variables.sicd_reader_object.base_reader, selected_region_complex_data)
        self.app_variables.fft_complex_data = fft_complex_data

        self.app_variables.fft_display_data = remap.density(fft_complex_data)
        fft_reader = NumpyImageReader(self.app_variables.fft_display_data)
        self.frequency_vs_degree_panel.set_image_reader(fft_reader)

        self.frequency_vs_degree_panel.set_current_tool_to_edit_shape()
        self.frequency_vs_degree_panel.variables.current_shape_id = self.frequency_vs_degree_panel.variables.select_rect_id
        self.frequency_vs_degree_panel.modify_existing_shape_using_image_coords(self.frequency_vs_degree_panel.variables.select_rect_id, self.get_fft_image_bounds())
        canvas_drawing_bounds = self.frequency_vs_degree_panel.image_coords_to_canvas_coords(self.frequency_vs_degree_panel.variables.select_rect_id)
        self.frequency_vs_degree_panel.variables.shape_drag_xy_limits[str(self.frequency_vs_degree_panel.variables.select_rect_id)] = canvas_drawing_bounds
        self.app_variables.fft_canvas_bounds = self.frequency_vs_degree_panel.get_shape_canvas_coords(self.frequency_vs_degree_panel.variables.select_rect_id)
        self.frequency_vs_degree_panel.show_shape(self.frequency_vs_degree_panel.variables.select_rect_id)

        filtered_numpy_reader = NumpyImageReader(self.get_filtered_image())
        self.filtered_panel.set_image_reader(filtered_numpy_reader)

        self.tabs_panel.tabs.load_image_tab.chip_size_panel.nx.set_text(numpy.shape(selected_region_complex_data)[1])
        self.tabs_panel.tabs.load_image_tab.chip_size_panel.ny.set_text(numpy.shape(selected_region_complex_data)[0])

        self.update_phase_history_selection()

    def get_fft_image_bounds(self,
                             ):             # type: (...) -> (int, int, int, int)
        x_axis_mean = numpy.mean(self.app_variables.fft_display_data, axis=0)
        y_axis_mean = numpy.mean(self.app_variables.fft_display_data, axis=1)

        x_start = numpy.min(numpy.where(x_axis_mean != 0))
        x_end = numpy.max(numpy.where(x_axis_mean != 0))

        y_start = numpy.min(numpy.where(y_axis_mean != 0))
        y_end = numpy.max(numpy.where(y_axis_mean != 0))

        return y_start, x_start, y_end, x_end

    def callback_save_fft_panel_as_png(self, event):
        filename = filedialog.asksaveasfilename(initialdir=os.path.expanduser("~"), title="Select file",
                                                filetypes=(("png file", "*.png"), ("all files", "*.*")))
        self.image_canvas.save_as_png(filename)

    def callback_get_adjusted_image(self, event):
        filtered_image = self.get_filtered_image()
        self.frequency_vs_degree_panel.set_image_reader(NumpyImageReader(filtered_image))

    def update_filtered_image(self):
        self.filtered_panel.set_image_reader(NumpyImageReader(self.get_filtered_image()))

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
        ro = self.app_variables.sicd_reader_object.base_reader
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

    def update_phase_history_selection(self):
        image_bounds = self.get_fft_image_bounds()
        current_bounds = self.frequency_vs_degree_panel.canvas_shape_coords_to_image_coords(self.frequency_vs_degree_panel.variables.select_rect_id)
        x_min = min(current_bounds[1], current_bounds[3])
        x_max = max(current_bounds[1], current_bounds[3])
        y_min = min(current_bounds[0], current_bounds[2])
        y_max = max(current_bounds[0], current_bounds[2])

        x_full_image_range = image_bounds[3] - image_bounds[1]
        y_full_image_range = image_bounds[2] - image_bounds[0]

        start_cross = (x_min - image_bounds[1]) / x_full_image_range * 100
        stop_cross = (x_max - image_bounds[1]) / x_full_image_range * 100
        fraction_cross = (x_max - x_min) / x_full_image_range * 100

        start_range = (y_min - image_bounds[0]) / y_full_image_range * 100
        stop_range = (y_max - image_bounds[0]) / y_full_image_range * 100
        fraction_range = (y_max - y_min) / y_full_image_range * 100

        self.phase_history.start_percent_cross.set_text("{:0.4f}".format(start_cross))
        self.phase_history.stop_percent_cross.set_text("{:0.4f}".format(stop_cross))
        self.phase_history.fraction_cross.set_text("{:0.4f}".format(fraction_cross))
        self.phase_history.start_percent_range.set_text("{:0.4f}".format(start_range))
        self.phase_history.stop_percent_range.set_text("{:0.4f}".format(stop_range))
        self.phase_history.fraction_range.set_text("{:0.4f}".format(fraction_range))

        # handle units
        self.phase_history.resolution_range_units.set_text("meters")
        self.phase_history.resolution_cross_units.set_text("meters")
        range_resolution = self.app_variables.sicd_reader_object.base_reader.sicd_meta.Grid.Row.ImpRespWid / (fraction_range / 100.0)
        cross_resolution = self.app_variables.sicd_reader_object.base_reader.sicd_meta.Grid.Col.ImpRespWid / (fraction_cross / 100.0)

        tmp_range_resolution = range_resolution
        tmp_cross_resolution = cross_resolution

        if self.phase_history.english_units_checkbox.is_selected():
            tmp_range_resolution = range_resolution / scipy_constants.foot
            tmp_cross_resolution = cross_resolution / scipy_constants.foot
            if tmp_range_resolution < 1:
                tmp_range_resolution = range_resolution / scipy_constants.inch
                self.phase_history.resolution_range_units.set_text("inches")
            else:
                self.phase_history.resolution_range_units.set_text("feet")
            if tmp_cross_resolution < 1:
                tmp_cross_resolution = cross_resolution / scipy_constants.inch
                self.phase_history.resolution_cross_units.set_text("inches")
            else:
                self.phase_history.resolution_cross_units.set_text("feet")
        else:
            if range_resolution < 1:
                tmp_range_resolution = range_resolution * 100
                self.phase_history.resolution_range_units.set_text("cm")
            if cross_resolution < 1:
                tmp_cross_resolution = cross_resolution * 100
                self.phase_history.resolution_cross_units.set_text("cm")

        self.phase_history.resolution_range.set_text("{:0.2f}".format(tmp_range_resolution))
        self.phase_history.resolution_cross.set_text("{:0.2f}".format(tmp_cross_resolution))

        cross_sample_spacing = self.app_variables.sicd_reader_object.base_reader.sicd_meta.Grid.Col.SS
        range_sample_spacing = self.app_variables.sicd_reader_object.base_reader.sicd_meta.Grid.Row.SS

        if self.phase_history.english_units_checkbox.is_selected():
            tmp_cross_ss = cross_sample_spacing / scipy_constants.foot
            tmp_range_ss = range_sample_spacing / scipy_constants.foot
            if tmp_cross_ss < 1:
                tmp_cross_ss = cross_sample_spacing / scipy_constants.inch
                self.phase_history.sample_spacing_cross.set_text("inches")
            else:
                self.phase_history.sample_spacing_cross_units.set_text("feet")
            if tmp_range_ss < 1:
                tmp_range_ss = range_sample_spacing / scipy_constants.inch
                self.phase_history.sample_spacing_range_units.set_text("inches")
            else:
                self.phase_history.sample_spacing_range_units.set_text("feet")
        else:
            if cross_sample_spacing < 1:
                tmp_cross_ss = cross_sample_spacing * 100
                self.phase_history.sample_spacing_cross_units.set_text("cm")
            if range_sample_spacing < 1:
                tmp_range_ss = range_sample_spacing * 100
                self.phase_history.sample_spacing_range_units.set_text("cm")

        self.phase_history.sample_spacing_cross.set_text("{:0.2f}".format(tmp_cross_ss))
        self.phase_history.sample_spacing_range.set_text("{:0.2f}".format(tmp_range_ss))

        # only update if we have twist angle and graze angles
        if self.app_variables.sicd_reader_object.base_reader.sicd_meta.SCPCOA.TwistAng and \
                self.app_variables.sicd_reader_object.base_reader.sicd_meta.SCPCOA.GrazeAng:

            cross_ground_resolution = cross_resolution / numpy.cos(numpy.deg2rad(self.app_variables.sicd_reader_object.base_reader.sicd_meta.SCPCOA.TwistAng))
            range_ground_resolution = range_resolution / numpy.cos(numpy.deg2rad(self.app_variables.sicd_reader_object.base_reader.sicd_meta.SCPCOA.GrazeAng))

            if self.phase_history.english_units_checkbox.is_selected():
                tmp_cross_ground_res = cross_ground_resolution / scipy_constants.foot
                tmp_range_ground_res = range_ground_resolution / scipy_constants.foot
                if tmp_cross_ground_res < 1:
                    tmp_cross_ground_res = cross_ground_resolution / scipy_constants.inch
                    self.phase_history.ground_resolution_cross_units.set_text("inches")
                else:
                    self.phase_history.ground_resolution_cross_units.set_text("feet")
                if tmp_range_ground_res < 1:
                    tmp_range_ground_res = range_ground_resolution / scipy_constants.inch
                    self.phase_history.ground_resolution_range_units.set_text("inches")
                else:
                    self.phase_history.ground_resolution_range_units.set_text("feet")
            else:
                if cross_ground_resolution < 1:
                    tmp_cross_ground_res = cross_ground_resolution * 100
                    self.phase_history.ground_resolution_cross_units.set_text("cm")
                if range_ground_resolution < 1:
                    tmp_range_ground_res = range_ground_resolution * 100
                    self.phase_history.ground_resolution_range_units.set_text("cm")

            self.phase_history.ground_resolution_cross.set_text("{:0.2f}".format(tmp_cross_ground_res))
            self.phase_history.ground_resolution_range.set_text("{:0.2f}".format(tmp_range_ground_res))


if __name__ == '__main__':
    root = tkinter.Tk()
    app = ApertureTool(root)
    root.mainloop()

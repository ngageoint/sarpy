import os
import time
import numpy
from scipy.fftpack import fft2, ifft2, fftshift

import tkinter
from tkinter import filedialog
from tkinter import Menu
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.utils.image_utils import frame_sequence_utils
from tkinter_gui_builder.panel_templates.image_canvas_panel.image_canvas_panel import ImageCanvasPanel

import sarpy.io.complex as sarpy_complex
import sarpy.visualization.remap as remap
from sarpy_gui_apps.apps.aperture_tool.panels.image_info_panel.image_info_panel import ImageInfoPanel
from sarpy_gui_apps.apps.aperture_tool.panels.selected_region_popup.selected_region_popup import SelectedRegionPanel
from sarpy_gui_apps.supporting_classes.metaicon import MetaIcon
from sarpy_gui_apps.supporting_classes.complex_image_reader import ComplexImageReader
from sarpy_gui_apps.apps.aperture_tool.panels.phase_history_selecion_panel.phase_history_selection_panel import PhaseHistoryPanel
from sarpy_gui_apps.apps.aperture_tool.app_variables import AppVariables
from sarpy.io.complex.base import BaseReader
import scipy.constants.constants as scipy_constants
from tkinter_gui_builder.image_readers.numpy_image_reader import NumpyImageReader
from sarpy_gui_apps.supporting_classes.metaviewer import Metaviewer
from sarpy_gui_apps.apps.aperture_tool.panels.animation_popup.animation_panel import AnimationPanel
from tkinter.filedialog import asksaveasfilename
from sarpy_gui_apps.apps.aperture_tool.panels.frequency_vs_degree_panel.frequency_vs_degree_panel import FrequencyVsDegreePanel


class ApertureTool(AbstractWidgetPanel):
    frequency_vs_degree_panel = FrequencyVsDegreePanel         # type: FrequencyVsDegreePanel
    filtered_panel = ImageCanvasPanel                    # type: ImageCanvasPanel
    image_info_panel = ImageInfoPanel                          # type: ImageInfoPanel
    metaicon = MetaIcon                             # type: MetaIcon
    phase_history = PhaseHistoryPanel               # type: PhaseHistoryPanel
    metaviewer = Metaviewer                         # type: Metaviewer
    animation_panel = AnimationPanel                # type: AnimationPanel

    def __init__(self, master):
        self.app_variables = AppVariables()

        self.master = master

        master_frame = tkinter.Frame(master)
        AbstractWidgetPanel.__init__(self, master_frame)

        widgets_list = ["frequency_vs_degree_panel", "filtered_panel"]
        self.init_w_horizontal_layout(widgets_list)
        self.frequency_vs_degree_panel.pack(expand=tkinter.Y, fill=tkinter.BOTH)
        self.filtered_panel.pack()

        self.filtered_panel.canvas.set_canvas_size(600, 400)

        self.frequency_vs_degree_panel.canvas.on_left_mouse_motion(self.callback_frequency_vs_degree_left_mouse_motion)

        self.image_info_popup_panel = tkinter.Toplevel(self.master)
        self.image_info_panel = ImageInfoPanel(self.image_info_popup_panel)
        self.image_info_panel.pack()
        self.image_info_popup_panel.withdraw()

        self.image_info_panel.file_selector.select_file.on_left_mouse_click(self.callback_select_file)

        self.ph_popup_panel = tkinter.Toplevel(self.master)
        self.phase_history = PhaseHistoryPanel(self.ph_popup_panel)
        self.phase_history.pack()
        self.ph_popup_panel.withdraw()

        self.metaicon_popup_panel = tkinter.Toplevel(self.master)
        self.metaicon = MetaIcon(self.metaicon_popup_panel)
        self.metaicon.set_canvas_size(800, 600)
        self.metaicon.pack()
        self.metaicon_popup_panel.withdraw()

        self.metaviewer_popup_panel = tkinter.Toplevel(self.master)
        self.metaviewer = Metaviewer(self.metaviewer_popup_panel)
        self.metaviewer.pack()
        self.metaviewer_popup_panel.withdraw()

        self.animation_popup_panel = tkinter.Toplevel(self.master)
        self.animation_panel = AnimationPanel(self.animation_popup_panel)
        self.animation_panel.pack()
        self.animation_popup_panel.withdraw()

        # callbacks for animation
        self.animation_panel.animation_settings.play.on_left_mouse_click(self.callback_play_animation)
        self.animation_panel.animation_settings.step_forward.on_left_mouse_click(self.callback_step_forward)
        self.animation_panel.animation_settings.step_back.on_left_mouse_click(self.callback_step_back)
        self.animation_panel.animation_settings.stop.on_left_mouse_click(self.callback_stop_animation)

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
        popups_menu.add_command(label="Metaviewer", command=self.metaviewer_popup)
        popups_menu.add_command(label="Animation", command=self.animation_popup)

        save_menu = Menu(menubar, tearoff=0)
        save_menu.add_command(label="Save Meticon", command=self.save_metaicon)

        menubar.add_cascade(label="File", menu=filemenu)
        menubar.add_cascade(label="Popups", menu=popups_menu)
        menubar.add_cascade(label="Save", menu=save_menu)

        master.config(menu=menubar)

        master_frame.pack()
        self.pack()
        # self.animation_popup()

    def save_metaicon(self):
        save_fname = asksaveasfilename(initialdir=os.path.expanduser("~"), filetypes=[("*.png", ".PNG")])
        self.metaicon.save_full_canvas_as_png(save_fname)

    def exit(self):
        self.quit()

    def update_animation_params(self):
        self.app_variables.animation_n_frames = int(self.animation_panel.animation_settings.number_of_frames.get())
        self.app_variables.animation_aperture_faction = float(self.animation_panel.animation_settings.aperture_fraction.get())
        self.app_variables.animation_frame_rate = float(self.animation_panel.animation_settings.frame_rate.get())
        self.app_variables.animation_cycle_continuously = self.animation_panel.animation_settings.cycle_continuously.is_selected()

    def callback_step_forward(self, event):
        fast_or_slow = "fast"
        if self.animation_panel.direction.is_slow_time():
            fast_or_slow = "slow"
        self.step_animation("forward", fast_or_slow)

    def callback_step_back(self, event):
        fast_or_slow = "fast"
        if self.animation_panel.direction.is_slow_time():
            fast_or_slow = "slow"
        self.step_animation("back", fast_or_slow)

    def step_animation(self,
                       direction_forward_or_back,       # type: str
                       time_fast_or_slow,               # type: str
                       ):
        self.update_animation_params()
        full_canvas_x_aperture = self.app_variables.fft_canvas_bounds[2] - self.app_variables.fft_canvas_bounds[0]
        full_canvas_y_aperture = self.app_variables.fft_canvas_bounds[3] - self.app_variables.fft_canvas_bounds[1]

        if direction_forward_or_back == "forward":
            if self.app_variables.animation_current_position < self.app_variables.animation_n_frames - 1:
                self.app_variables.animation_current_position += 1
        elif direction_forward_or_back == "back":
            if self.app_variables.animation_current_position > 0:
                self.app_variables.animation_current_position -= 1
        if time_fast_or_slow == "slow":
            aperture_distance = full_canvas_x_aperture * self.app_variables.animation_aperture_faction

            start_locs = numpy.linspace(self.app_variables.fft_canvas_bounds[0],
                                      self.app_variables.fft_canvas_bounds[2] - aperture_distance,
                                      self.app_variables.animation_n_frames)

            x_start = start_locs[self.app_variables.animation_current_position]
            new_rect = (x_start,
                        self.app_variables.fft_canvas_bounds[1],
                        x_start + aperture_distance,
                        self.app_variables.fft_canvas_bounds[3])
        elif time_fast_or_slow == "fast":
            aperture_distance = full_canvas_y_aperture * self.app_variables.animation_aperture_faction

            start_locs = numpy.linspace(self.app_variables.fft_canvas_bounds[1],
                                        self.app_variables.fft_canvas_bounds[3] - aperture_distance,
                                        self.app_variables.animation_n_frames)
            start_locs = numpy.flip(start_locs)
            y_start = start_locs[self.app_variables.animation_current_position]
            new_rect = (self.app_variables.fft_canvas_bounds[0],
                        y_start,
                        self.app_variables.fft_canvas_bounds[2],
                        y_start + aperture_distance)

        self.frequency_vs_degree_panel.canvas.modify_existing_shape_using_canvas_coords(
            self.frequency_vs_degree_panel.canvas.variables.select_rect_id, new_rect)
        self.update_filtered_image()
        self.update_phase_history_selection()

    def callback_stop_animation(self, event):
        self.app_variables.animation_stop_pressed = True
        self.animation_panel.animation_settings.unpress_all_buttons()

    def callback_play_animation(self, event):
        self.update_animation_params()
        fast_or_slow = "fast"
        if self.animation_panel.direction.is_slow_time():
            fast_or_slow = "slow"

        direction_forward_or_back = "forward"
        if self.animation_panel.direction.reverse.is_selected():
            direction_forward_or_back = "back"
        time_between_frames = 1/self.app_variables.animation_frame_rate
        self.animation_panel.animation_settings.disable_all_buttons()
        self.animation_panel.animation_settings.stop.config(state="normal")

        def play_animation():
            if direction_forward_or_back == "forward":
                self.app_variables.animation_current_position = -1
            else:
                self.app_variables.animation_current_position = self.app_variables.animation_n_frames
            for i in range(self.app_variables.animation_n_frames):
                self.update_animation_params()
                if self.app_variables.animation_stop_pressed:
                    break
                tic = time.time()
                self.step_animation(direction_forward_or_back, fast_or_slow)
                self.frequency_vs_degree_panel.update()
                toc = time.time()
                if (toc - tic) < time_between_frames:
                    time.sleep(time_between_frames - (toc - tic))

        self.app_variables.animation_stop_pressed = False
        if self.animation_panel.animation_settings.cycle_continuously.is_selected():
            while not self.app_variables.animation_stop_pressed:
                play_animation()
        else:
            play_animation()
        self.app_variables.animation_stop_pressed = False
        self.animation_panel.animation_settings.activate_all_buttons()

    def animation_popup(self):
        self.animation_popup_panel.deiconify()

    def metaviewer_popup(self):
        self.metaviewer_popup_panel.deiconify()

    def main_controls_popup(self):
        self.image_info_popup_panel.deiconify()

    def ph_popup(self):
        self.ph_popup_panel.deiconify()

    def metaicon_popup(self):
        self.metaicon_popup_panel.deiconify()

    def callback_frequency_vs_degree_left_mouse_motion(self, event):
        self.frequency_vs_degree_panel.canvas.callback_handle_left_mouse_motion(event)
        self.update_filtered_image()
        self.update_phase_history_selection()

    def select_file(self):
        self.callback_select_file(None)

    def callback_select_file(self, event):
        sicd_fname = self.image_info_panel.file_selector.event_select_file(event)
        self.app_variables.sicd_fname = sicd_fname
        self.app_variables.sicd_reader_object = sarpy_complex.open(sicd_fname)

        self.metaicon.create_from_sicd(self.app_variables.sicd_reader_object.sicd_meta)

        popup = tkinter.Toplevel(self.master)
        selected_region_popup = SelectedRegionPanel(popup, self.app_variables)
        self.app_variables.sicd_reader_object = ComplexImageReader(self.app_variables.sicd_fname)
        selected_region_popup.image_canvas.canvas.set_image_reader(self.app_variables.sicd_reader_object)

        self.master.wait_window(popup)

        selected_region_complex_data = self.app_variables.selected_region_complex_data

        fft_complex_data = self.get_fft_complex_data(self.app_variables.sicd_reader_object.base_reader, selected_region_complex_data)
        self.app_variables.fft_complex_data = fft_complex_data

        self.app_variables.fft_display_data = remap.density(fft_complex_data)
        fft_reader = NumpyImageReader(self.app_variables.fft_display_data)
        self.frequency_vs_degree_panel.canvas.set_image_reader(fft_reader)

        self.frequency_vs_degree_panel.canvas.set_current_tool_to_edit_shape()
        self.frequency_vs_degree_panel.canvas.variables.current_shape_id = self.frequency_vs_degree_panel.canvas.variables.select_rect_id
        self.frequency_vs_degree_panel.canvas.modify_existing_shape_using_image_coords(self.frequency_vs_degree_panel.canvas.variables.select_rect_id, self.get_fft_image_bounds())
        canvas_drawing_bounds = self.frequency_vs_degree_panel.canvas.image_coords_to_canvas_coords(self.frequency_vs_degree_panel.canvas.variables.select_rect_id)
        self.frequency_vs_degree_panel.canvas.variables.shape_drag_xy_limits[str(self.frequency_vs_degree_panel.canvas.variables.select_rect_id)] = canvas_drawing_bounds
        self.app_variables.fft_canvas_bounds = self.frequency_vs_degree_panel.canvas.get_shape_canvas_coords(self.frequency_vs_degree_panel.canvas.variables.select_rect_id)
        self.frequency_vs_degree_panel.canvas.show_shape(self.frequency_vs_degree_panel.canvas.variables.select_rect_id)

        filtered_numpy_reader = NumpyImageReader(self.get_filtered_image())
        self.filtered_panel.canvas.set_image_reader(filtered_numpy_reader)

        self.image_info_panel.chip_size_panel.nx.set_text(numpy.shape(selected_region_complex_data)[1])
        self.image_info_panel.chip_size_panel.ny.set_text(numpy.shape(selected_region_complex_data)[0])

        self.update_phase_history_selection()

        self.metaviewer.create_w_sicd(self.app_variables.sicd_reader_object.base_reader.sicd_meta)

        self.frequency_vs_degree_panel.update()
        self.frequency_vs_degree_panel.update_x_axis(start_val=-10, stop_val=10, label="Polar Angle (degrees)")
        self.frequency_vs_degree_panel.update_y_axis(start_val=7.409, stop_val=11.39, label="Frequency (GHz)")

    def get_fft_image_bounds(self,
                             ):             # type: (...) -> (int, int, int, int)
        meta = self.app_variables.sicd_reader_object.base_reader.sicd_meta

        row_ratio = meta.Grid.Row.ImpRespBW * meta.Grid.Row.SS
        col_ratio = meta.Grid.Col.ImpRespBW * meta.Grid.Col.SS

        full_n_rows = self.frequency_vs_degree_panel.canvas.variables.canvas_image_object.image_reader.full_image_ny
        full_n_cols = self.frequency_vs_degree_panel.canvas.variables.canvas_image_object.image_reader.full_image_nx

        full_im_y_start = int(full_n_rows*(1 - row_ratio)/2)
        full_im_y_end = full_n_rows - full_im_y_start

        full_im_x_start = int(full_n_cols*(1 - col_ratio)/2)
        full_im_x_end = full_n_cols - full_im_x_start

        return full_im_y_start, full_im_x_start, full_im_y_end, full_im_x_end

    def callback_save_fft_panel_as_png(self, event):
        filename = filedialog.asksaveasfilename(initialdir=os.path.expanduser("~"), title="Select file",
                                                filetypes=(("png file", "*.png"), ("all files", "*.*")))
        self.image_canvas.figure_canvas.save_full_canvas_as_png(filename)

    def callback_get_adjusted_image(self, event):
        filtered_image = self.get_filtered_image()
        self.frequency_vs_degree_panel.canvas.set_image_reader(NumpyImageReader(filtered_image))

    def update_filtered_image(self):
        self.filtered_panel.canvas.set_image_reader(NumpyImageReader(self.get_filtered_image()))

    def get_filtered_image(self):
        select_rect_id = self.frequency_vs_degree_panel.canvas.variables.select_rect_id
        full_image_rect = self.frequency_vs_degree_panel.canvas.get_shape_image_coords(select_rect_id)

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

    def callback_save_animation(self, event):
        filename = filedialog.asksaveasfilename(initialdir=os.path.expanduser("~"), title="Select file",
                                                filetypes=(("animated gif", "*.gif"), ("all files", "*.*")))
        select_box_id = self.filtered_panel.canvas.variables.current_shape_id
        start_fft_select_box = self.filtered_panel.canvas.get_shape_canvas_coords(select_box_id)
        n_steps = int(self.filtered_panel.n_steps.get())
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
        current_bounds = self.frequency_vs_degree_panel.canvas.canvas_shape_coords_to_image_coords(self.frequency_vs_degree_panel.canvas.variables.select_rect_id)
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
                self.phase_history.sample_spacing_cross_units.set_text("inches")
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

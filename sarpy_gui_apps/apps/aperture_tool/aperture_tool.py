import tkinter
from sarpy_gui_apps.apps.aperture_tool.panels.image_zoomer.zoomer_panel import ZoomerPanel
from sarpy_gui_apps.apps.aperture_tool.panels.fft_panel.fft_main_panel import FFTPanel
from sarpy_gui_apps.apps.aperture_tool.panels.adjusted_image_panel.adjusted_image_panel import AdjustedViewPanel
from tkinter_gui_builder.canvas_image_objects.numpy_canvas_image import NumpyCanvasDisplayImage
import sarpy.visualization.remap as remap
from scipy.fftpack import fft2, ifft2, fftshift
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.utils.image_utils import frame_sequence_utils
from tkinter import filedialog
import numpy as np
import os


class AppVariables:
    def __init__(self):
        self.fft_image_object = NumpyCanvasDisplayImage()        # type: NumpyCanvasDisplayImage


class ApertureTool(AbstractWidgetPanel):
    zoomer_panel = ZoomerPanel
    fft_panel = FFTPanel
    adjusted_view_panel = AdjustedViewPanel

    def __init__(self, master):
        master_frame = tkinter.Frame(master)
        AbstractWidgetPanel.__init__(self, master_frame)

        widgets_list = ["zoomer_panel", "fft_panel", 'adjusted_view_panel']
        self.init_w_horizontal_layout(widgets_list)
        master_frame.pack()
        self.pack()

        self.app_variables = AppVariables()
        self.zoomer_panel.image_canvas.canvas.on_left_mouse_release(self.callback_handle_zoomer_left_mouse_release)
        self.fft_panel.fft_button_panel.inv_fft.on_left_mouse_click(self.callback_get_adjusted_image)
        self.fft_panel.fft_button_panel.animate.on_left_mouse_click(self.callback_animate_horizontal_fft_sweep)
        self.fft_panel.fft_button_panel.save_animation_as_gif.on_left_mouse_click(self.callback_save_animation)

    def callback_handle_zoomer_left_mouse_release(self, event):
        self.zoomer_panel.callback_handle_left_mouse_release(event)
        if self.zoomer_panel.image_canvas.variables.canvas_image_object.decimation_factor == 1:
            fft_data = self.get_all_fft_display_data()
            self.fft_panel.image_canvas.init_with_numpy_image(fft_data)
        else:
            pass

    def callback_get_adjusted_image(self, event):
        filtered_image = self.get_filtered_image()
        self.adjusted_view_panel.image_canvas.init_with_numpy_image(filtered_image)

    def get_filtered_image(self):
        fft_canvas = self.fft_panel.image_canvas
        canvas_rect = fft_canvas.get_shape_canvas_coords(fft_canvas.variables.select_rect_id)
        full_image_rect = fft_canvas.variables.canvas_image_object.canvas_rect_to_full_image_rect(canvas_rect)

        y_ul = int(full_image_rect[0])
        x_ul = int(full_image_rect[1])
        y_lr = int(full_image_rect[2])
        x_lr = int(full_image_rect[3])

        ft_cdata = self.get_all_fft_complex_data()
        filtered_cdata = np.zeros(ft_cdata.shape, ft_cdata.dtype)
        filtered_cdata[y_ul:y_lr, x_ul:x_lr] = ft_cdata[y_ul:y_lr, x_ul:x_lr]
        filtered_cdata = fftshift(filtered_cdata)

        inverse_flag = False
        ro = self.zoomer_panel.image_canvas.variables.canvas_image_object.reader_object
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

    def get_all_fft_display_data(self):
        ft_cdata = self.get_all_fft_complex_data()
        fft_display_data = remap.density(ft_cdata)
        return fft_display_data

    def get_all_fft_complex_data(self):
        ro = self.zoomer_panel.image_canvas.variables.canvas_image_object.reader_object
        ny = np.shape(self.zoomer_panel.image_canvas.variables.canvas_image_object.canvas_decimated_image)[0]
        nx = np.shape(self.zoomer_panel.image_canvas.variables.canvas_image_object.canvas_decimated_image)[1]
        ul_y, ul_x = self.zoomer_panel.image_canvas.variables.canvas_image_object.canvas_full_image_upper_left_yx
        ul_x = int(ul_x)
        ul_y = int(ul_y)
        # TODO: change this to a tuple sequence to get rid of FutureWarning
        cdata = ro.read_chip((ul_y, ul_y+ny, 1), (ul_x, ul_x + nx, 1))
        if ro.sicd_meta.Grid.Col.Sgn > 0 and ro.sicd_meta.Grid.Row.Sgn > 0:
            # use fft2 to go from image to spatial freq
            ft_cdata = fft2(cdata)
        else:
            # flip using ifft2
            ft_cdata = ifft2(tuple(cdata))

        ft_cdata = fftshift(ft_cdata)
        return ft_cdata

    def callback_animate_horizontal_fft_sweep(self, event):
        select_box_id = self.fft_panel.image_canvas.variables.current_shape_id
        start_fft_select_box = self.fft_panel.image_canvas.get_shape_canvas_coords(select_box_id)
        n_steps = int(self.fft_panel.fft_button_panel.n_steps.get())
        n_pixel_translate = int(self.fft_panel.fft_button_panel.n_pixels_horizontal.get())
        step_factor = np.linspace(0, n_pixel_translate, n_steps)
        for step in step_factor:
            x1 = start_fft_select_box[0] + step
            y1 = start_fft_select_box[1]
            x2 = start_fft_select_box[2] + step
            y2 = start_fft_select_box[3]
            self.fft_panel.image_canvas.modify_existing_shape_using_canvas_coords(select_box_id, (x1, y1, x2, y2), update_pixel_coords=True)
            self.fft_panel.image_canvas.update()
            self.callback_get_adjusted_image(event)
        self.fft_panel.image_canvas.modify_existing_shape_using_canvas_coords(select_box_id, start_fft_select_box, update_pixel_coords=True)
        self.fft_panel.image_canvas.update()

    def callback_save_animation(self, event):
        filename = filedialog.asksaveasfilename(initialdir=os.path.expanduser("~"), title="Select file",
                                                filetypes=(("animated gif", "*.gif"), ("all files", "*.*")))
        select_box_id = self.fft_panel.image_canvas.variables.current_shape_id
        start_fft_select_box = self.fft_panel.image_canvas.get_shape_canvas_coords(select_box_id)
        n_steps = int(self.fft_panel.fft_button_panel.n_steps.get())
        n_pixel_translate = int(self.fft_panel.fft_button_panel.n_pixels_horizontal.get())
        step_factor = np.linspace(0, n_pixel_translate, n_steps)
        fps = float(self.fft_panel.fft_button_panel.animation_fps.get())

        frame_sequence = []
        for step in step_factor:
            x1 = start_fft_select_box[0] + step
            y1 = start_fft_select_box[1]
            x2 = start_fft_select_box[2] + step
            y2 = start_fft_select_box[3]
            self.fft_panel.image_canvas.modify_existing_shape_using_canvas_coords(select_box_id, (x1, y1, x2, y2), update_pixel_coords=True)
            self.fft_panel.image_canvas.update()
            filtered_image = self.get_filtered_image()
            frame_sequence.append(filtered_image)
        frame_sequence_utils.save_numpy_frame_sequence_to_animated_gif(frame_sequence, filename, fps)
        self.fft_panel.image_canvas.modify_existing_shape_using_canvas_coords(select_box_id, start_fft_select_box, update_pixel_coords=True)
        self.fft_panel.image_canvas.update()


if __name__ == '__main__':
    root = tkinter.Tk()
    app = ApertureTool(root)
    root.mainloop()

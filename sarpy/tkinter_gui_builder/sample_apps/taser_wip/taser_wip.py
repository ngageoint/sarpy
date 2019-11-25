import tkinter
from tkinter.filedialog import askopenfilename
from sarpy.tkinter_gui_builder.sample_apps.taser_wip.custom_panels.button_panel import ButtonPanel
from sarpy.tkinter_gui_builder.panel_templates.basic_pyplot_image_panel import BasicPyplotImagePanel
from sarpy.tkinter_gui_builder.sample_apps.taser_wip.custom_panels.taser_canvas_panel import TaserImageCanvasPanel
import numpy as np
import imageio
import os


class AppVariables:
    def __init__(self):
        self.fname = "None"       # type: str


class Taser:
    def __init__(self, master):
        # Create a container
        # set the master frame
        master_frame = tkinter.Frame(master)
        self.app_variables = AppVariables()

        # define panels widget_wrappers in master frame
        self.button_panel = ButtonPanel(master_frame)
        self.button_panel.set_spacing_between_buttons(0)
        self.pyplot_panel = BasicPyplotImagePanel(master_frame, 800, 600)
        self.basic_image_panel = TaserImageCanvasPanel(master_frame)
        self.basic_image_panel.set_canvas_size(600, 400)

        # specify layout of widget_wrappers in master frame
        self.button_panel.pack(side="left")
        self.basic_image_panel.pack(side="left")
        self.pyplot_panel.pack(side="left")

        master_frame.pack()

        # bind events to callbacks here
        self.button_panel.fname_select.on_left_mouse_click(self.callback_get_and_save_fname)
        self.button_panel.update_basic_image.on_left_mouse_click(self.callback_update_basic_image_canvas_w_fname)
        self.button_panel.update_rect_image.on_left_mouse_click(self.callback_display_canvas_rect_selection)
        self.button_panel.update_rect_image.on_right_mouse_click(self.callback_display_random_pyplot_image)

        self.basic_image_panel.canvas.on_left_mouse_press(self.callback_start_drawing_new_rect)
        self.basic_image_panel.canvas.on_left_mouse_motion(self.callback_drag_rect)

        self.basic_image_panel.canvas.on_right_mouse_press(self.callback_start_drawing_new_line)
        self.basic_image_panel.canvas.on_right_mouse_motion(self.callback_drag_line)

    # define custom callbacks here
    def callback_display_random_pyplot_image(self, event):
        new_image = np.random.random((200, 200))
        self.pyplot_panel.update_image(new_image)

    def callback_update_pyplot_image_w_fname(self, event):
        image_data = imageio.imread(self.app_variables.fname)
        self.pyplot_panel.update_image(image_data)

    def callback_get_and_save_fname(self, event):
        image_file_extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.png', '*.PNG', '*.tif', '*.tiff', '*.TIF', '*.TIFF']
        ftypes = [
            ('image files', image_file_extensions),
            ('All files', '*'),
        ]
        new_fname = askopenfilename(initialdir=os.path.expanduser("~"), filetypes=ftypes)
        if new_fname:
            self.app_variables.fname = new_fname
        return "break"

    def callback_start_drawing_new_rect(self, event):
        self.basic_image_panel.event_initiate_rect(event)

    def callback_drag_rect(self, event):
        self.basic_image_panel.event_drag_rect(event)

    def callback_start_drawing_new_line(self, event):
        self.basic_image_panel.event_initiate_line(event)
        new_image = np.random.random((200, 200))
        self.pyplot_panel.update_image(new_image)

    def callback_drag_line(self, event):
        self.basic_image_panel.event_drag_line(event)

    def callback_random_canvas_image(self, event):
        new_image = np.random.random((200, 200))
        self.basic_image_panel.set_image_from_numpy_array(new_image, scale_dynamic_range=True)

    def callback_display_canvas_rect_selection(self, event):
        image_data = self.basic_image_panel.get_data_in_rect()
        self.pyplot_panel.update_image(image_data)

    def callback_update_basic_image_canvas_w_fname(self, event):
        self.basic_image_panel.set_image_from_fname(self.app_variables.fname)

    def update_image_from_app_variable_arg(self,
                                           args,  # type: AppVariables
                                           ):
        image_data = imageio.imread(args.fname)
        self.pyplot_panel.update_image(image_data)


root = tkinter.Tk()
app = Taser(root)
root.mainloop()

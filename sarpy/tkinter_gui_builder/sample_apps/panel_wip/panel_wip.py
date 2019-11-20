import tkinter
from tkinter.filedialog import askopenfilename
from sarpy.tkinter_gui_builder.sample_apps.panel_wip.custom_panels.seven_button_panel import SevenButtonPanel
from sarpy.tkinter_gui_builder.panel_templates.basic_pyplot_image_panel import BasicPyplotImagePanel
from sarpy.tkinter_gui_builder.panel_templates.basic_image_canvas_panel import BasicImageCanvasPanel
import numpy as np
import imageio
import os


class AppVariables:
    def __init__(self):
        self.fname = "None"       # type: str


class TwoPanelSideBySide:
    def __init__(self, master):
        # Create a container
        # set the master frame
        master_frame = tkinter.Frame(master)
        self.app_variables = AppVariables()

        # define panels widget_wrappers in master frame
        self.button_panel = SevenButtonPanel(master_frame)
        self.pyplot_panel = BasicPyplotImagePanel(master_frame)
        self.basic_image_panel = BasicImageCanvasPanel(master_frame)

        # specify layout of widget_wrappers in master frame
        self.button_panel.pack(side="left")
        self.pyplot_panel.pack(side="left")
        self.basic_image_panel.pack(side="left")

        master_frame.pack()

        # bind events to callbacks here
        self.button_panel.button1.on_left_mouse_click(self.callback_display_random_image)
        self.button_panel.button1.on_right_mouse_click(self.callback_update_image)
        self.button_panel.button2.on_left_mouse_click(self.button_change_text_callback)
        self.button_panel.button3.on_left_mouse_click(self.askopenfile_callback)
        self.button_panel.button4.on_left_mouse_click(self.callback_update_image)
        self.button_panel.button5.on_left_mouse_click_with_args(self.update_image_from_app_variable_arg, self.app_variables)

        self.basic_image_panel.canvas.on_left_mouse_press(self.callback_start_drawing_new_rect)
        self.basic_image_panel.canvas.on_left_mouse_motion(self.callback_drag_rect)

        self.basic_image_panel.canvas.on_right_mouse_press(self.callback_start_drawing_new_line)
        self.basic_image_panel.canvas.on_right_mouse_motion(self.callback_drag_line)

    # define custom callbacks here
    def callback_display_random_image(self, event):
        new_image = np.random.random((200, 200))
        self.pyplot_panel.update_image(new_image)

    def callback_update_image(self, event):
        image_data = imageio.imread(self.app_variables.fname)
        self.pyplot_panel.update_image(image_data)

    def button_change_text_callback(self, event):
        self.button_panel.button3.set_text("12345")
        self.callback_display_random_image(event)

    def askopenfile_callback(self, event):
        image_file_extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.png', '*.PNG', '*.tif', '*.tiff', '*.TIF', '*.TIFF']
        ftypes = [
            ('image files', image_file_extensions),
            ('All files', '*'),
        ]
        new_fname = askopenfilename(initialdir=os.path.expanduser("~"), filetypes=ftypes)
        if new_fname:
            self.app_variables.fname = new_fname
        return "break"

    def update_image_from_app_variable_arg(self,
                                           args,  # type: AppVariables
                                           ):
        image_data = imageio.imread(args.fname)
        self.pyplot_panel.update_image(image_data)

    def callback_start_drawing_new_rect(self, event):
        # save mouse drag start position
        self.basic_image_panel.event_initiate_rect(event)

    def callback_drag_rect(self, event):
        self.basic_image_panel.event_drag_rect(event)

    def callback_start_drawing_new_line(self, event):
        # save mouse drag start position
        self.basic_image_panel.event_initiate_line(event)
        new_image = np.random.random((200, 200))
        self.pyplot_panel.update_image(new_image)

    def callback_drag_line(self, event):
        self.basic_image_panel.event_drag_line(event)


root = tkinter.Tk()
#root.geometry("1920x1080+300+300")
app = TwoPanelSideBySide(root)
root.mainloop()

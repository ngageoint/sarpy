import tkinter
from tkinter.filedialog import askopenfilename
from sarpy.tkinter_gui_builder.sample_apps.panel_wip.custom_panels.seven_button_panel import SevenButtonPanel
from sarpy.tkinter_gui_builder.panel_templates.basic_image_panel import BasicImagePanel
from sarpy.tkinter_gui_builder.panel_templates.draw_rect_on_image_panel import DrawRectOnImagePanel
import numpy as np
import imageio


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
        self.basic_button_panel = SevenButtonPanel(master_frame)
        self.image_panel = BasicImagePanel(master_frame)
        self.draw_rect_on_image_panel = DrawRectOnImagePanel(master_frame)
        # specify layout of widget_wrappers in master frame
        self.basic_button_panel.pack(side="left")
        self.draw_rect_on_image_panel.pack(side="left")
        self.image_panel.pack(side="left")

        master_frame.pack()

        # bind events to callbacks here
        self.basic_button_panel.button1.on_left_mouse_click(self.callback_display_random_image)
        self.basic_button_panel.button1.on_right_mouse_click(self.callback_update_image)
        self.basic_button_panel.button2.on_left_mouse_click(self.button_change_text_callback)
        self.basic_button_panel.button3.on_left_mouse_click(self.askopenfile_callback)
        self.basic_button_panel.button4.on_left_mouse_click(self.callback_update_image)
        self.basic_button_panel.button5.on_left_mouse_click_with_args(self.update_image_from_app_variable_arg,
                                                                      self.app_variables)

    # define custom callbacks here
    def callback_display_random_image(self, event):
        new_image = np.random.random((200, 200))
        self.image_panel.update_image(new_image)

    def callback_update_image(self, event):
        image_data = imageio.imread(self.app_variables.fname)
        self.image_panel.update_image(image_data)

    def button_change_text_callback(self, event):
        self.basic_button_panel.button3.set_text("12345")
        self.callback_display_random_image(event)

    def askopenfile_callback(self, event):
        self.app_variables.fname = askopenfilename()
        return "break"

    def update_image_from_app_variable_arg(self,
                                           args,  # type: AppVariables
                                           ):
        image_data = imageio.imread(args.fname)
        self.image_panel.update_image(image_data)


root = tkinter.Tk()
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
root.geometry("1920x1080+300+300")
app = TwoPanelSideBySide(root)
root.mainloop()

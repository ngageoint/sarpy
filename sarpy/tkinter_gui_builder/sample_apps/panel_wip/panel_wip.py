import tkinter
from tkinter.filedialog import askopenfilename
from sarpy.tkinter_gui_builder.sample_apps.panel_wip.custom_panels.seven_button_panel import SevenButtonPanel
from sarpy.tkinter_gui_builder.panel_templates.basic_image_panel import BasicImagePanel


class AppVariables:
    def __init__(self):
        self.fname = None       # type: str


class TwoPanelSideBySide:
    def __init__(self, master):
        # Create a container
        # set the master frame
        master_frame = tkinter.Frame(master)
        self.app_variables = AppVariables()

        # define panels widget_wrappers in master frame
        self.basic_button_panel = SevenButtonPanel(master_frame)
        self.image_panel = BasicImagePanel(master_frame)
        # specify layout of widget_wrappers in master frame
        self.basic_button_panel.pack(side="left")
        self.image_panel.pack(side="left")

        master_frame.pack()

        self.basic_button_panel.button1.on_left_mouse_click(self.image_panel.callback_update_image)
        self.basic_button_panel.button2.on_left_mouse_click(self.button_change_text_callback)
        self.basic_button_panel.button3.on_left_mouse_click(self.askopenfile_callback)
        #self.basic_button_panel.button3.on_left_mouse_click(lambda self.test_callback: "stuff")

    def button_change_text_callback(self, event):
        self.basic_button_panel.button3.callback_set_text("12345")
        self.image_panel.callback_update_image(event)

    def askopenfile_callback(self, event):
        self.app_variables.fname = askopenfilename()
        return "break"



root = tkinter.Tk()
app = TwoPanelSideBySide(root)
root.mainloop()

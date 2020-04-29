from sarpy_gui_apps.apps.taser_tool.taser import Taser
from sarpy_gui_apps.apps.wake_tool.wake_tool import WakeTool
from sarpy_gui_apps.apps.aperture_tool.aperture_tool import ApertureTool
from sarpy_gui_apps.apps.make_ortho.quick_ortho_tool import Ortho
import tkinter
from tkinter import ttk


# Root class to create the interface and define the controller function to switch frames
class RootApp(tkinter.Tk):
    def __init__(self):
        tkinter.Tk.__init__(self)
        self._frame = None
        self.switch_frame(NoteBook)

    # controller function
    def switch_frame(self, frame_class):
        new_frame = frame_class(self)
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()


# sub-root to contain the Notebook frame and a controller function to switch the tabs within the notebook
class NoteBook(ttk.Frame):
    def __init__(self, master):
        ttk.Frame.__init__(self, master)
        # set up the tabs
        self.notebook = ttk.Notebook()
        taser_tab = Tab(self.notebook)
        self.wake_tool_tab = Tab(self.notebook)
        self.quick_ortho_tab = Tab(self.notebook)
        self.plot_demo_tab = Tab(self.notebook)

        self.notebook.add(taser_tab, text="Taser")
        self.notebook.add(self.wake_tool_tab, text="wake tool")
        self.notebook.add(self.quick_ortho_tab, text="quick ortho")
        self.notebook.add(self.plot_demo_tab, text="temporal plot")

        self.taser = Taser(taser_tab)
        self.wake = WakeTool(self.wake_tool_tab)
        self.quick_ortho = Ortho(self.quick_ortho_tab)
        self.notebook.pack()


# Notebook tab, used to init new tabs
class Tab(ttk.Frame):
    def __init__(self, master):
        ttk.Frame.__init__(self, master)
        self._frame = None


if __name__ == "__main__":
    Root = RootApp()
    Root.title("Frame test")
    Root.mainloop()

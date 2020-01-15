import tkinter as tk
from tkinter import ttk
from sarpy_gui_apps.apps.taser_tool.taser import Taser
from sarpy_gui_apps.apps.wake_tool.wake_tool import WakeTool
from sarpy_gui_apps.apps.aperture_tool.aperture_tool import ApertureTool
from tkinter_gui_builder.panel_templates.pyplot_panel.temporal_plot_panel import TemporalPlotPanel


# Root class to create the interface and define the controller function to switch frames
class RootApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
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
        self.taser_tab = Tab(self.notebook)
        self.wake_tool_tab = Tab(self.notebook)
        self.aperture_tool_tab = Tab(self.notebook)
        self.temporal_plot_tab = Tab(self.notebook)

        self.notebook.add(self.taser_tab, text="Taser")
        self.notebook.add(self.wake_tool_tab, text="wake tool")
        self.notebook.add(self.aperture_tool_tab, text="aperture tool")
        self.notebook.add(self.temporal_plot_tab, text="temporal plot")

        self.taser = Taser(self.taser_tab)
        self.wake = WakeTool(self.wake_tool_tab)
        self.aperture = ApertureTool(self.aperture_tool_tab)
        self.animation_plot = TemporalPlotPanel(self.temporal_plot_tab)
        self.notebook.pack()


# Notebook tab, used to init new tabs
class Tab(ttk.Frame):
    def __init__(self, master):
        ttk.Frame.__init__(self, master)
        self._frame = None


if __name__ == "__main__":
    Root = RootApp()
    # Root.geometry("640x480")
    Root.title("Frame test")
    Root.mainloop()
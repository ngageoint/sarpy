import tkinter as tk
from tkinter import ttk
from sarpy.tkinter_gui_builder.sarpy_apps.taser_tool.taser_panel import Taser
from sarpy.tkinter_gui_builder.sandbox.animation_plot_panel import AnimationPlotPanel


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
        self.aperture_tool_tab = Tab(self.notebook)
        self.animation_plot_tab = Tab(self.notebook)
        self.notebook.add(self.taser_tab, text="Taser")
        self.notebook.add(self.aperture_tool_tab, text="draw")
        self.notebook.add(self.animation_plot_tab, text="Tab3")

        self.taser = Taser(self.taser_tab)
        self.animation_plot = AnimationPlotPanel(self.animation_plot_tab)
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
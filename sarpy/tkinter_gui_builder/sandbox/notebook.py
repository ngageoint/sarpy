import tkinter as tk
from tkinter import ttk
from sarpy.tkinter_gui_builder.sample_apps.taser_tool.taser_wip import Taser


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
        self.notebook = ttk.Notebook()
        self.tab1 = Tab1(self.notebook)
        self.tab2 = Tab2(self.notebook)
        self.tab3 = Tab3(self.notebook)
        self.notebook.add(self.tab1, text="Tab1")
        self.notebook.add(self.tab2, text="Tab2")
        self.notebook.add(self.tab3, text="Tab3")
        self.taser = Taser(self.tab2)
        self.notebook.pack()

    # controller function
    def switch_tab1(self, frame_class):
        new_frame = frame_class(self.notebook)
        self.tab1.destroy()
        self.tab1 = new_frame


# Notebook - Tab 1
class Tab1(ttk.Frame):
    def __init__(self, master):
        ttk.Frame.__init__(self, master)
        self._frame = None


# first frame for Tab1
class Tab1_Frame1(ttk.Frame):
    def __init__(self, master):
        ttk.Frame.__init__(self, master)
        self.label = ttk.Label(self, text="this is a test - one")
        # button object with command to replace the frame
        self.label.pack()


# Notebook - Tab 2
class Tab2(ttk.Frame):
    def __init__(self, master):
        ttk.Frame.__init__(self, master)
        self.label = ttk.Label(self, text="this is a test - two")
        self.label.pack()


# Notebook - Tab 3
class Tab3(ttk.Frame):
    def __init__(self, master):
        ttk.Frame.__init__(self, master)
        self.label = ttk.Label(self, text="this is a test - three")
        self.label.pack()


if __name__ == "__main__":
    Root = RootApp()
    # Root.geometry("640x480")
    Root.title("Frame test")
    Root.mainloop()
import os
import tkinter
from tkinter import Menu
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.panel_templates.image_canvas.image_canvas import ImageCanvas

from tkinter_gui_builder.image_readers.geotiff_reader import GeotiffImageReader
from tkinter import filedialog


class GeotiffViewer(AbstractWidgetPanel):
    geotiff_image_panel = ImageCanvas         # type: ImageCanvas
    image_reader = None     # type: GeotiffImageReader

    def __init__(self, master):
        self.master = master

        master_frame = tkinter.Frame(master)
        AbstractWidgetPanel.__init__(self, master_frame)

        widgets_list = ["geotiff_image_panel"]
        self.init_w_horizontal_layout(widgets_list)

        self.geotiff_image_panel.set_canvas_size(3000, 1500)
        self.geotiff_image_panel.set_current_tool_to_pan()

        menubar = Menu()

        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self.select_file)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.exit)

        # create more pulldown menus
        popups_menu = Menu(menubar, tearoff=0)
        popups_menu.add_command(label="Main Controls", command=self.exit)
        popups_menu.add_command(label="Phase History", command=self.exit)
        popups_menu.add_command(label="Metaicon", command=self.exit)

        menubar.add_cascade(label="File", menu=filemenu)
        menubar.add_cascade(label="Popups", menu=popups_menu)

        master.config(menu=menubar)

        master_frame.pack()
        self.pack()

    def exit(self):
        self.quit()

    def select_file(self):
        fname = filedialog.askopenfilename(initialdir=os.path.expanduser("~"),
                                           title="Select file",
                                           filetypes=(("tiff files", ("*.tif", "*.tiff", "*.TIF", "*.TIFF")),
                                                      ("all files", "*.*"))
                                           )
        self.image_reader = GeotiffImageReader(fname)
        self.image_reader.rgb_bands = [0, 1, 2]
        self.image_reader.read_all_image_data_from_disk()
        self.geotiff_image_panel.set_image_reader(self.image_reader)


if __name__ == '__main__':
    root = tkinter.Tk()
    app = GeotiffViewer(root)
    root.mainloop()

from tkinter_gui_builder.widgets.basic_widgets import LabelFrame
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from sarpy_gui_apps.apps.aperture_tool.panels.tabs_panel.load_image_tab.load_image_tab import LoadImage
from sarpy_gui_apps.apps.aperture_tool.panels.tabs_panel.animation_tab.animation_tab import Animation
from tkinter import ttk


class Tabs(LabelFrame):
    def __init__(self, master):
        LabelFrame.__init__(self, master)
        # set up the tabs
        notebook = ttk.Notebook(master)
        tab1 = LabelFrame(notebook)
        tab2 = LabelFrame(notebook)

        notebook.add(tab1, text="Load Image")
        notebook.add(tab2, text="Animation")

        self.tab1 = LoadImage(tab1)         # type: LoadImage
        self.tab2 = Animation(tab2)         # type: Animation

        notebook.pack()


class TabsPanel(AbstractWidgetPanel):
    tabs = Tabs         # type: Tabs

    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)
        self.config(borderwidth=2)

        widget_list = ["tabs"]
        self.init_w_horizontal_layout(widget_list)

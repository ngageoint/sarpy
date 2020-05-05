from tkinter_gui_builder.panel_templates.image_canvas.image_canvas import ImageCanvas
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel


class FrequencyVsDegreeView(AbstractWidgetPanel):
    image_canvas = ImageCanvas                      # type: ImageCanvas

    def __init__(self, parent):
        # TODO: there is bad super behavior going on here? Wrong parent?
        AbstractWidgetPanel.__init__(self, parent)

        self.image_canvas = ImageCanvas(parent)
        self.image_canvas.set_canvas_size(600, 400)

        # set up event listeners
        self.image_canvas.set_labelframe_text("frequency vs degree of pol")
        self.image_canvas.pack()
        self.pack()

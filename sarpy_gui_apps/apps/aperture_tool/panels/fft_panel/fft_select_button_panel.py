from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.widgets import basic_widgets


class FFTSelectButtonPanel(AbstractWidgetPanel):
    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)
        self.select_data = basic_widgets.Button
        self.move_rect = basic_widgets.Button
        self.inv_fft = basic_widgets.Button

        self.n_pixels_label = basic_widgets.Label
        self.n_pixels_horizontal = basic_widgets.Entry
        self.n_steps_label = basic_widgets.Label
        self.n_steps = basic_widgets.Entry
        self.animate = basic_widgets.Button
        self.save_fft_image_as_png = basic_widgets.Button
        self.save_animation_as_gif = basic_widgets.Button
        self.animation_fps = basic_widgets.Entry

        widget_list = ["select_data",
                       "move_rect",
                       "inv_fft",

                       "n_pixels_label",
                       "n_pixels_horizontal",
                       "n_steps_label",
                       "n_steps",
                       "animate",

                       "save_fft_image_as_png",
                       "save_animation_as_gif",
                       "animation_fps"]

        self.init_w_basic_widget_list(widget_list, 3, [3, 5, 3])
        self.set_label_text("fft select")

        self.n_pixels_label.config(text="n pixel sweep")
        self.n_steps_label.config(text="n steps")

        self.animation_fps.insert(0, "15")

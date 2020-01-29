from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
from tkinter_gui_builder.widgets import basic_widgets


class CanvasDemoButtonPanel(AbstractWidgetPanel):
    fname_select = basic_widgets.Button
    zoom_in = basic_widgets.Button
    zoom_out = basic_widgets.Button
    rect_select = basic_widgets.Button
    update_rect_image = basic_widgets.Button
    pan = basic_widgets.Button
    draw_line_w_drag = basic_widgets.Button
    draw_line_w_click = basic_widgets.Button
    draw_arrow_w_drag = basic_widgets.Button
    draw_arrow_w_click = basic_widgets.Button
    draw_rect_w_drag = basic_widgets.Button
    draw_rect_w_click = basic_widgets.Button
    draw_circle_w_drag = basic_widgets.Button
    draw_circle_w_click = basic_widgets.Button
    draw_ellipse_w_drag = basic_widgets.Button
    draw_ellipse_w_click = basic_widgets.Button
    draw_polygon_w_click = basic_widgets.Button
    modify_existing_shape = basic_widgets.Button
    color_selector = basic_widgets.Button
    select_existing_shape = basic_widgets.Combobox  # type: basic_widgets.Combobox
    remap_dropdown = basic_widgets.Combobox         # type: basic_widgets.Combobox

    def __init__(self, parent):
        AbstractWidgetPanel.__init__(self, parent)

        controls = ["fname_select",
                    "zoom_in",
                    "zoom_out",
                    "pan",
                    "draw_line_w_drag",
                    "draw_line_w_click",
                    "draw_arrow_w_drag",
                    "draw_arrow_w_click",
                    "draw_rect_w_drag",
                    "draw_rect_w_click",
                    "draw_circle_w_drag",
                    "draw_circle_w_click",
                    "draw_ellipse_w_drag",
                    "draw_ellipse_w_click",
                    "draw_polygon_w_click",
                    "modify_existing_shape",
                    "color_selector",
                    "rect_select",
                    "update_rect_image",
                    "remap_dropdown"]

        self.init_w_box_layout(controls, 4, column_widths=20)

        self.remap_dropdown.update_combobox_values(["density",
                                                    "brighter",
                                                    "darker",
                                                    "high contrast",
                                                    "linear",
                                                    "log",
                                                    "pedf",
                                                    "nrl"])
        self.set_label_text("taser buttons")

from tkinter_gui_builder.panel_templates.image_canvas_panel.image_canvas import ImageCanvas
from tkinter_gui_builder.tests.test_utils import mouse_simulator


def create_new_rect_on_image_canvas(image_canvas,  # type: ImageCanvas
                                    start_x,       # type: int
                                    start_y,       # type: int
                                    ):
    image_canvas.set_current_tool_to_draw_rect(None)
    click_event = mouse_simulator.simulate_event_at_x_y_position(start_x, start_y)
    image_canvas.callback_handle_left_mouse_click(click_event)

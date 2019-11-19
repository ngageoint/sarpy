import tkinter as tk


class WidgetEvents:
    def __init__(self):
        pass

    # events
    def on_left_mouse_click(self, event):
        self.bind("<Button-1>", event)

    def on_left_mouse_click_with_args(self, event_w_args, args):
        self.bind("<Button-1>", lambda event, arg=args: event_w_args(arg))

    def on_right_mouse_click(self, event):
        self.bind("<Button-3>", event)

    def on_right_mouse_click_with_args(self, event_w_args, args):
        self.bind("<Button-3>", lambda event, arg=args: event_w_args(arg))

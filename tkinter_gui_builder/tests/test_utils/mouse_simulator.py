import tkinter


def simulate_event_at_x_y_position(x,  # type: int
                                   y,  # type: int
                                   ):
    event = tkinter.Event.mro()[0]  # TODO: unresolved?
    event.x = 50
    event.y = 50
    return event

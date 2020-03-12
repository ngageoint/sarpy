import sys

if sys.version_info[0] < 3:
    import tkinter
    # import Tkinter as tkinter  # TODO: I have no idea
    import ttk

else:
    import tkinter
    from tkinter import ttk

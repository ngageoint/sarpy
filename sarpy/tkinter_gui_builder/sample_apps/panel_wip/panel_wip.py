import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sarpy.tkinter_gui_builder.sample_apps.panel_wip.custom_panels.seven_button_panel import SevenButtonPanel


class TwoPanelSideBySide:
    def __init__(self, master):
        # Create a container
        # set the master frame
        master_frame = tkinter.Frame(master)

        # define panels widget_wrappers in master frame
        self.basic_button_panel = SevenButtonPanel(master_frame)
        self.plot_frame = tkinter.Frame(master_frame)

        # specify layout of widget_wrappers in master frame
        self.basic_button_panel.pack(side="left")
        self.plot_frame.pack(side="left")

        fig = Figure()
        ax = fig.add_subplot(111)
        self.line, = ax.plot(range(10))

        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=1)

        master_frame.pack()

        self.basic_button_panel.button1.on_left_mouse_click(self.decrease_callback)
        self.basic_button_panel.button2.on_left_mouse_click(self.increase_callback)

    def decrease_callback(self, event):
        self.decrease()

    def decrease(self):
        x, y = self.line.get_data()
        self.line.set_ydata(y - 0.2 * x)
        self.canvas.draw()

    def increase_callback(self, event):
        self.increase()

    def increase(self):
        x, y = self.line.get_data()
        self.line.set_ydata(y + 0.2 * x)
        self.basic_button_panel.button2.config(text="ch123")
        self.canvas.draw()


root = tkinter.Tk()
app = TwoPanelSideBySide(root)
root.mainloop()
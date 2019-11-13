import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sarpy.tools.tkinter_sandbox.widget_refactor.custom_widgets.vertical_button_panel import VerticalButtonPanel

class App:
    def __init__(self, master):
        # Create a container
        master_frame = tkinter.Frame(master)

        self.active_button_frame = tkinter.Frame(master_frame)
        self.plot_frame = tkinter.Frame(master_frame)
        self.vertical_panel = VerticalButtonPanel(master_frame)

        self.button_left = tkinter.Button(self.active_button_frame, text="< Decrease Slope", command=self.decrease)
        self.button_left.pack()
        self.button_right = tkinter.Button(self.active_button_frame, text="Increase Slope >", command=self.increase)
        self.button_right.pack(side="top")

        fig = Figure()
        ax = fig.add_subplot(111)
        self.line, = ax.plot(range(10))

        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=1)
        self.active_button_frame.pack(side="left")
        self.plot_frame.pack(side="left")
        self.vertical_panel.pack(side="right")
        master_frame.pack()
        self.vertical_panel.button_1.bind("<Button-1>", self.decrease_callback)
        self.vertical_panel.button_2.bind("<Button-1>", self.increase_callback)

    def decrease_callback(self, event):
        self.decrease()

    def increase_callback(self, event):
        self.increase()

    def decrease(self):
        x, y = self.line.get_data()
        self.line.set_ydata(y - 0.2 * x )
        self.canvas.draw()

    def increase(self):
        x, y = self.line.get_data()
        self.line.set_ydata(y + 0.2 * x)
        self.canvas.draw()


root = tkinter.Tk()

root.bind("<Button-1>", print("hello"))

app = App(root)
root.mainloop()
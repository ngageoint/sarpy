import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class App:
    def __init__(self, master):
        # Create a container
        frame = tkinter.Frame(master)
        # Create 2 buttons

        self.frame2 = tkinter.Frame(frame)
        self.frame2.pack(side="top")
        self.button_left = tkinter.Button(self.frame2, text="< Decrease Slope",
                                        command=self.decrease)
        self.button_left.pack()
        self.button_left
        self.button_right = tkinter.Button(self.frame2, text="Increase Slope >",
                                        command=self.increase)
        self.button_right.pack(side="top")
        self.button_3 = tkinter.Button(frame, text="I'm a button", expand=True)
        self.button_3.pack(side="top")

        fig = Figure()
        ax = fig.add_subplot(111)
        self.line, = ax.plot(range(10))

        self.canvas = FigureCanvasTkAgg(fig,master=master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side='right', fill='both', expand=1)
        frame.pack()

    def decrease(self):
        x, y = self.line.get_data()
        self.line.set_ydata(y - 0.2 * x )
        self.canvas.draw()

    def increase(self):
        x, y = self.line.get_data()
        self.line.set_ydata(y + 0.2 * x)
        self.canvas.draw()


root = tkinter.Tk()
app = App(root)
root.mainloop()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class PlotObject:
    def __init__(self, alias, ax, color, zero_centered=False):
        self.alias = alias
        self.ax = ax
        self.line = ax.plot(range(1), np.ones(1), color=color, label=alias)[0]

        self.zero_centered = zero_centered

    def update_data(self, data):
        self.line.set_ydata(data)
        self.line.set_xdata(np.arange(len(data)))

        self.ax.relim()  # Recalculate limits
        self.ax.autoscale_view(tight=True, scalex=True, scaley=not self.zero_centered)  # Autoscale

        if self.zero_centered:
            yabs_max = 1.1 * max(abs(data))
            self.ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)


class LiveTrainPlot:
    MAX_POINTS = 300

    def __init__(self, alias, update=1):
        self.alias = alias
        self.fig = None
        self.ax = None
        self.data = pd.DataFrame()
        self.lines = dict()
        self.cycles = 0
        self.update = update
        plt.ion()

    def init_plot(self, fig_size_x=17, fig_size_y=6, dpi=50, update=None):
        self.fig, self.ax = plt.subplots(figsize=(fig_size_x, fig_size_y),
                                         nrows=3, dpi=dpi, constrained_layout=True)

        if update:
            self.update = update

        ax0_secondary = self.ax[0].twinx()
        self.lines["Zero_"] = PlotObject('Zero', self.ax[0], 'black')
        self.lines["Penalties"] = PlotObject('Penalties', self.ax[0], 'crimson', zero_centered=True)
        self.lines["TotalReward"] = PlotObject('TotalReward', ax0_secondary, 'dodgerblue',  zero_centered=True)

        self.lines["Balance"] = PlotObject('Balance', self.ax[1], 'forestgreen')
        self.lines["Zero"] = PlotObject('Zero', self.ax[1], 'black')

        self.lines["NegTrades"] = PlotObject('NegTrades', self.ax[2], 'orangered')
        self.lines["PosTrades"] = PlotObject('PosTrades', self.ax[2], 'royalblue')

        self.ax[0].grid()
        self.ax[1].grid()
        self.ax[2].grid()

        self.ax[0].legend(loc="upper left")
        ax0_secondary.legend(loc="upper right")
        self.ax[1].legend(loc="center left")
        self.ax[2].legend(loc="center left")

        self.fig.canvas.draw()

    def reset(self):
        self.cycles = 0
        self.data = pd.DataFrame()

    def save_sample(self, sample):
        sample["Zero"] = 0
        self.data = self.data.append(sample, ignore_index=True)

    def update_plot(self, sample):
        self.cycles += 1
        self.save_sample(sample)

        if not self.cycles % self.update:
            div = max(1, len(self.data)//self.MAX_POINTS)
            self.data["gr"] = self.data.index // div
            data_g = self.data.groupby(["gr"]).agg(np.mean)

            for col in self.lines:
                alias = self.lines[col].alias
                if alias in self.data.columns:
                    self.lines[col].update_data(data_g[[alias]].values)


            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

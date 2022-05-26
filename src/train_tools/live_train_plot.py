import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class PlotObject:
    def __init__(self, alias, ax, color, secondary=False):
        self.ax = ax
        self.line = ax.plot(range(1), np.ones(1), color=color, label=alias)[0]
        self.secondary = secondary

    def update_data(self, data):
        self.line.set_ydata(data)
        self.line.set_xdata(np.arange(len(data)))

        self.ax.relim()  # Recalculate limits
        self.ax.autoscale_view(True, True, True)  # Autoscale


class LiveTrainPlot:
    FIG_SIZE_X = 11
    FIG_SIZE_Y = 6
    DPI = 50
    MAX_POINTS = 300

    def __init__(self, alias, update=15):
        self.alias = alias
        self.fig = None
        self.ax = None
        self.data = pd.DataFrame()
        self.lines = dict()
        self.cycles = 0
        self.update = update
        plt.ion()

    def init_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(self.FIG_SIZE_X, self.FIG_SIZE_Y),
                                         nrows=3, dpi=self.DPI, constrained_layout=True)

        ax0_secondary = self.ax[0].twinx()
        self.lines["Penalties"] = PlotObject('Penalties', self.ax[0], 'crimson')
        self.lines["TotalReward"] = PlotObject('TotalReward', ax0_secondary, 'dodgerblue', secondary=True)

        self.lines["Balance"] = PlotObject('Balance', self.ax[1], 'forestgreen')
        self.lines["Zero"] = PlotObject('Zero', self.ax[1], 'lightcoral')

        self.lines["NegTrades"] = PlotObject('NegTrades', self.ax[2], 'orangered')
        self.lines["PosTrades"] = PlotObject('PosTrades', self.ax[2], 'royalblue')

        self.ax[0].grid()
        self.ax[1].grid()
        self.ax[2].grid()

        self.ax[0].legend(loc=2)
        ax0_secondary.legend(loc=3)
        self.ax[1].legend(loc=6)
        self.ax[2].legend(loc=6)

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

            for col in self.data.columns:
                if col in self.lines:
                    self.lines[col].update_data(data_g[[col]].values)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
